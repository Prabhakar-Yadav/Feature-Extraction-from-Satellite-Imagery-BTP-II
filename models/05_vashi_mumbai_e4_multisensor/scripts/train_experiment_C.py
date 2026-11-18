"""
train_experiment_C.py
---------------------
Experiment C: Sensor-Adaptive Dual-Stem Architecture with Native Vashi SPOT NIR
- Input:
  * Vashi SPOT: 4-band native satellite data (Red, Green, Blue, NIR) from _img.npy.
    Normalized with verified SPOT channel statistics:
    Mean: [467.57, 444.84, 355.58, 580.22]
    Std:  [80.10, 96.96, 105.51, 171.67]
  * Mumbai Pléiades: 3-band native RGB data.
    Normalized with ImageNet statistics:
    Mean: [0.485, 0.456, 0.406]
    Std:  [0.229, 0.224, 0.225]
- Architecture: Sensor-Adaptive Dual-Stem ResNet-18 + scSE U-Net.
  * Stem Vashi: Conv2d(4, 64, 7, stride=2, padding=3). RGB channels initialized from
    pretrained ResNet-18; 4th channel (NIR) initialized from RGB filter mean.
  * Stem Mumbai: Conv2d(3, 64, 7, stride=2, padding=3) initialized from pretrained ResNet-18.
  * Shared ResNet-18 backbone (enc1..enc4) and scSE decoder (dec4..dec0).
- Loss: Masked Source-Aware Conditional Softmax CE + Masked Soft Dice.
- Effective Batch Size: 4 (Batch size 2, gradient accumulation steps = 2).
"""

import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.models as models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    torch.backends.cudnn.benchmark = True

BASE_DIR = Path("VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI")
EXP2_DIR = BASE_DIR / "research_corrected_v4" / "experiment_2"
CKPT_DIR = EXP2_DIR / "checkpoints"
AUDIT_DIR = EXP2_DIR / "audits"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "scripts"))
sys.path.append(str(EXP2_DIR / "scripts"))
from train_and_evaluate_5class import DecoderBlock
from train_experiment_B import MaskedConditionalSoftmaxLoss

CLASS_NAMES = ["Slum", "Non-slum", "Road", "Vegetation", "Open Land"]

# Vashi 4-band statistics
VASHI_MEANS = np.array([467.57, 444.84, 355.58, 580.22], dtype=np.float32)[:, None, None]
VASHI_STDS  = np.array([80.10, 96.96, 105.51, 171.67], dtype=np.float32)[:, None, None]

# Mumbai 3-band statistics
MUMBAI_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MUMBAI_STDS  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class SensorAdaptiveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = row["dataset_source"]
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask.shape[:2] != (512, 512):
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        if src == "vashi_spot":
            # Load 4-band SPOT
            npy_p = row["img_input_path_4band"]
            data = np.load(npy_p).astype(np.float32) # [4, 512, 512]
            # Replace NoData 65535 with band median
            data = np.where(data == 65535, VASHI_MEANS, data)

            if self.is_train:
                if random.random() > 0.5:
                    data = np.ascontiguousarray(data[:, :, ::-1])
                    mask = np.fliplr(mask).copy()
                if random.random() > 0.5:
                    data = np.ascontiguousarray(data[:, ::-1, :])
                    mask = np.flipud(mask).copy()
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    data = np.ascontiguousarray(np.rot90(data, k, axes=(1, 2)))
                    mask = np.rot90(mask, k).copy()

            norm = (data - VASHI_MEANS) / VASHI_STDS
            t_img = torch.from_numpy(norm).float()

        elif src == "mumbai_shapefile":
            # Load 3-band RGB
            bgr = cv2.imread(row["img_input_path_rgb"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (512, 512):
                rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

            if self.is_train:
                if random.random() > 0.5:
                    rgb = np.fliplr(rgb).copy()
                    mask = np.fliplr(mask).copy()
                if random.random() > 0.5:
                    rgb = np.flipud(rgb).copy()
                    mask = np.flipud(mask).copy()
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    rgb = np.rot90(rgb, k).copy()
                    mask = np.rot90(mask, k).copy()

            norm = (rgb.astype(np.float32) / 255.0 - MUMBAI_MEANS) / MUMBAI_STDS
            t_img = torch.from_numpy(norm.transpose(2, 0, 1)).float()

        t_mask = torch.from_numpy(mask).long()
        return t_img, t_mask, row["patch_id"], src

def adaptive_collate_fn(batch):
    imgs = [item[0] for item in batch]
    masks = torch.stack([item[1] for item in batch], dim=0)
    pids = [item[2] for item in batch]
    dsets = [item[3] for item in batch]
    return imgs, masks, pids, dsets

class SensorAdaptiveUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # 1. Vashi 4-band stem (Red, Green, Blue, NIR)
        self.stem_vashi = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu
        )
        with torch.no_grad():
            self.stem_vashi[0].weight[:, :3] = backbone.conv1.weight
            self.stem_vashi[0].weight[:, 3] = backbone.conv1.weight.mean(dim=1)

        # 2. Mumbai 3-band stem (Red, Green, Blue)
        self.stem_mumbai = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        # Shared encoder
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Shared scSE Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 0, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, imgs_list, sources):
        x0_list = []
        for img, src in zip(imgs_list, sources):
            img_in = img.unsqueeze(0) if img.dim() == 3 else img
            if src == "vashi_spot":
                x0 = self.stem_vashi(img_in)
            else:
                x0 = self.stem_mumbai(img_in)
            x0_list.append(x0)

        x0 = torch.cat(x0_list, dim=0)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1, None)
        return self.final_conv(d0)

def evaluate_val_adaptive(model, val_loader):
    model.eval()
    cm_v = np.zeros((5, 5), dtype=np.int64)
    m_tp, m_fn = 0, 0
    with torch.no_grad():
        for imgs_list, masks, pids, dsets in val_loader:
            imgs_list = [img.to(device) for img in imgs_list]
            logits = model(imgs_list, dsets)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.numpy()
            for p, g, d in zip(preds, gts, dsets):
                valid = (g != 255)
                if not valid.any(): continue
                if d == "vashi_spot":
                    for tc in range(1, 5):
                        mt = (g == tc) & valid
                        if mt.any():
                            for pc in range(5):
                                cm_v[tc, pc] += int((p[mt] == pc).sum())
                elif d == "mumbai_shapefile":
                    is_slum_gt = (g == 0) & valid
                    m_tp += int(((p == 0) & is_slum_gt).sum())
                    m_fn += int(((p != 0) & is_slum_gt).sum())

    ious = []
    for c in range(1, 5):
        tp = cm_v[c, c]
        denom = cm_v[c, :].sum() + cm_v[:, c].sum() - tp
        if denom > 0:
            ious.append(tp / denom)
    v_miou = float(np.mean(ious) * 100) if ious else 0.0
    v_acc = float(np.trace(cm_v[1:5, 1:5]) / max(1, cm_v[1:5, :].sum()) * 100)
    m_det = float(m_tp / max(1, m_tp + m_fn) * 100)
    comp_score = 2 * (v_miou * m_det) / max(1e-5, (v_miou + m_det))
    return v_miou, v_acc, m_det, comp_score, int(cm_v[1:5, 0].sum())

def main():
    print("\n" + "="*60)
    print("   STARTING EXPERIMENT C: SENSOR-ADAPTIVE DUAL-STEM WITH NATIVE VASHI NIR")
    print("="*60, flush=True)

    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    train_df = df[df["current_stage_in_v4"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    train_ds = SensorAdaptiveDataset(train_df, is_train=True)
    val_ds   = SensorAdaptiveDataset(val_df, is_train=False)

    from train_experiment_A import BalancedBatchSampler
    sampler = BalancedBatchSampler(train_df, num_batches=40)

    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=adaptive_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=adaptive_collate_fn, num_workers=0)

    model = SensorAdaptiveUNet(num_classes=5).to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()
            m.requires_grad_(True)

    criterion = MaskedConditionalSoftmaxLoss(smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    epochs = 25
    accum_steps = 2
    best_score = -1.0
    best_ckpt = CKPT_DIR / "best_expC_model.pth"
    history = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_b = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, (imgs_list, masks, pids, dsets) in enumerate(train_loader):
            imgs_list = [img.to(device) for img in imgs_list]
            masks = masks.to(device)

            logits = model(imgs_list, dsets)
            loss = criterion(logits, masks, dsets) / accum_steps
            loss.backward()

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum_steps
            n_b += 1

        scheduler.step()
        ep_loss = running_loss / max(1, n_b)
        v_miou, v_acc, m_det, comp_score, f_slum = evaluate_val_adaptive(model, val_loader)

        is_best = comp_score > best_score
        if is_best:
            best_score = comp_score
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "comp_score": comp_score,
                "vashi_val_miou": v_miou,
                "vashi_val_acc": v_acc,
                "mumbai_val_det_rate": m_det,
                "false_slums": f_slum,
                "config": "Experiment_C_Sensor_Adaptive_NIR"
            }, str(best_ckpt))

        print(f"Epoch {ep:02d}/25 | Loss: {ep_loss:.4f} | Vashi Val mIoU: {v_miou:.2f}% | Vashi Acc: {v_acc:.2f}% | Mumbai Det: {m_det:.2f}% | Score: {comp_score:.2f} {'[SAVED]' if is_best else ''}", flush=True)

        history.append({
            "epoch": ep,
            "train_loss": round(ep_loss, 4),
            "vashi_val_miou": round(v_miou, 2),
            "vashi_val_acc": round(v_acc, 2),
            "mumbai_val_det_rate": round(m_det, 2),
            "comp_score": round(comp_score, 2),
            "false_slums": f_slum
        })

    elapsed = time.time() - t0
    print(f"\n[DONE] Experiment C completed in {elapsed/60:.1f} mins. Best Score: {best_score:.2f}. Checkpoint: {best_ckpt}", flush=True)

    with open(AUDIT_DIR / "expC_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

