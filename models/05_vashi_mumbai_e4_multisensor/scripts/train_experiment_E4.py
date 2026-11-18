"""
train_experiment_E4.py
-----------------------
Experiment E4: NDVI Physics Feature + Calibrated Focal-Dice Loss + Balanced Selection
- Base Architecture: HighResSensorAdaptiveUNet (native dec0 skip)
- Input:
  * Vashi SPOT: 5 channels [Red, Green, Blue, NIR, NDVI]
    NDVI = (NIR - Red) / (NIR + Red + 1e-5) in [-1.0, 1.0]
    Provides unambiguous physics-based vegetation separation.
  * Mumbai Pléiades: 3 channels [Red, Green, Blue] (strictly no fabricated NIR)
- Loss:
  * Masked Focal Cross-Entropy (gamma=1.5, weight=0.50)
  * Masked Soft Dice (weight=0.50)
  * Calibrated class weights:
    Vashi: [1.25 (Non-slum), 0.65 (Road), 1.10 (Veg), 1.05 (Open Land)]
    Mumbai: [1.30 (Slum), 1.10 (Veg)]
- Training:
  * BalancedBatchSampler (1:1 Vashi:Mumbai)
  * Optimizer: AdamW(lr=3e-4, weight_decay=1e-4), CosineAnnealingLR(T_max=25, eta_min=1e-6)
  * Batch size 2, gradient accumulation 2 (effective batch size 4), 25 epochs
  * Selection score: 0.60 * Vashi_mIoU + 0.40 * Mumbai_Det (avoids harmonic crash)
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
from torch.utils.data import Dataset, DataLoader
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
EXPE1_DIR = BASE_DIR / "research_corrected_v4" / "experiment_E1"
EXPE4_DIR = BASE_DIR / "research_corrected_v4" / "experiment_E4"

CKPT_DIR = EXPE4_DIR / "checkpoints"
AUDIT_DIR = EXPE4_DIR / "audits"
PRED_DIR = EXPE4_DIR / "predictions"
VIS_DIR = EXPE4_DIR / "visualizations"
SCRIPTS_DIR = EXPE4_DIR / "scripts"

for d in [CKPT_DIR, AUDIT_DIR, PRED_DIR, VIS_DIR, SCRIPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "scripts"))
sys.path.append(str(EXP2_DIR / "scripts"))
sys.path.append(str(EXPE1_DIR / "scripts"))

from train_and_evaluate_5class import DecoderBlock, SCSEBlock
from train_experiment_A import BalancedBatchSampler
from train_experiment_C import VASHI_MEANS, VASHI_STDS, MUMBAI_MEANS, MUMBAI_STDS

CLASS_NAMES = ["Slum", "Non-slum", "Road", "Vegetation", "Open Land"]

class SensorAdaptiveNDVIDataset(Dataset):
    """
    Dataset loader for Experiment E4:
    - Vashi: Loads 4-band SPOT (R, G, B, NIR) + Computes normalized NDVI -> 5 channels
    - Mumbai: Loads 3-band RGB -> 3 channels
    """
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
            data = np.load(row["img_input_path_4band"]).astype(np.float32) # [4, 512, 512]
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

            # Normalized 4-band
            norm4 = (data - VASHI_MEANS) / VASHI_STDS # [4, 512, 512]
            
            # Physics-based NDVI calculation: (NIR - Red) / (NIR + Red + eps)
            # Band 0: Red, Band 3: NIR
            red = data[0]
            nir = data[3]
            ndvi = (nir - red) / (nir + red + 1e-5) # in [-1.0, +1.0]
            ndvi = np.clip(ndvi, -1.0, 1.0)[None, :, :] # [1, 512, 512]
            
            # Stack into 5 channels: [R, G, B, NIR, NDVI]
            data5 = np.concatenate([norm4, ndvi], axis=0) # [5, 512, 512]
            t_img = torch.from_numpy(data5).float()

        elif src == "mumbai_shapefile":
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

            norm3 = (rgb.astype(np.float32) / 255.0 - MUMBAI_MEANS) / MUMBAI_STDS
            t_img = torch.from_numpy(norm3.transpose(2, 0, 1)).float()

        t_mask = torch.from_numpy(mask).long()
        return t_img, t_mask, row["patch_id"], src

def ndvi_collate_fn(batch):
    imgs = [item[0] for item in batch]
    masks = torch.stack([item[1] for item in batch], dim=0)
    pids = [item[2] for item in batch]
    dsets = [item[3] for item in batch]
    return imgs, masks, pids, dsets

class HighResNDVIAdaptiveUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # 1. Stems
        # Vashi: 5 channels (R, G, B, NIR, NDVI) -> 64
        self.stem_vashi = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu
        )
        with torch.no_grad():
            self.stem_vashi[0].weight[:, :3] = backbone.conv1.weight
            self.stem_vashi[0].weight[:, 3] = backbone.conv1.weight.mean(dim=1)
            self.stem_vashi[0].weight[:, 4] = 0.0 # zero init for NDVI

        # Mumbai: 3 channels (R, G, B) -> 64
        self.stem_mumbai = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        # 2. Native-Resolution Shallow Skip (512x512)
        self.shallow_vashi = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.shallow_mumbai = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3. Backbone
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # 4. Decoders
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 32, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, imgs_list, sources):
        x0_list = []
        shallow_list = []

        for img, src in zip(imgs_list, sources):
            img_in = img.unsqueeze(0) if img.dim() == 3 else img
            if src == "vashi_spot":
                x0 = self.stem_vashi(img_in)
                sh = self.shallow_vashi(img_in)
            else:
                x0 = self.stem_mumbai(img_in)
                sh = self.shallow_mumbai(img_in)
            x0_list.append(x0)
            shallow_list.append(sh)

        x0 = torch.cat(x0_list, dim=0)
        sh_feat = torch.cat(shallow_list, dim=0)

        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1, sh_feat)
        return self.final_conv(d0)

class CalibratedFocalDiceLoss(nn.Module):
    """
    Calibrated Focal Cross-Entropy + Masked Soft Dice Loss:
    - Focal loss downweights easy background pixels, focusing gradients on difficult boundaries.
    - Calibrated weights: Non-slum=1.25, Road=0.65, Veg=1.10, Open Land=1.05
    """
    def __init__(self, gamma=1.5, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.weights_vashi = torch.tensor([1.25, 0.65, 1.10, 1.05], dtype=torch.float32).to(device)
        self.weights_mumbai = torch.tensor([1.30, 1.10], dtype=torch.float32).to(device)

    def forward(self, logits, targets, sources):
        B, C, H, W = logits.shape
        total_loss = 0.0

        for b in range(B):
            src = sources[b]
            y = targets[b]
            valid = (y != 255)
            if not valid.any(): continue
            l_b = logits[b]

            if src == "vashi_spot":
                vashi_valid = valid & (y >= 1) & (y <= 4)
                if not vashi_valid.any(): continue
                sub_logits = l_b[1:5]
                sub_targets = (y - 1).clamp(0, 3)

                # Focal CE Loss
                ce = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_vashi,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                pt = torch.exp(-ce)
                focal_ce = (((1.0 - pt) ** self.gamma) * ce)[vashi_valid].mean()

                # Masked Soft Dice
                probs = F.softmax(sub_logits, dim=0)
                dice_loss = 0.0
                for c_idx in range(4):
                    t_bin = ((sub_targets == c_idx) & vashi_valid).float()
                    p_c = probs[c_idx] * vashi_valid.float()
                    inter = (p_c * t_bin).sum()
                    card = (p_c + t_bin).sum()
                    dice = (2.0 * inter + self.smooth) / (card + self.smooth)
                    dice_loss += (1.0 - dice)
                dice_loss = dice_loss / 4.0

                total_loss += 0.50 * focal_ce + 0.50 * dice_loss

            elif src == "mumbai_shapefile":
                mumbai_valid = valid & ((y == 0) | (y == 3))
                if not mumbai_valid.any(): continue
                sub_logits = torch.stack([l_b[0], l_b[3]], dim=0)
                sub_targets = torch.where(y == 0, 0, 1)

                ce = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_mumbai,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                pt = torch.exp(-ce)
                focal_ce = (((1.0 - pt) ** self.gamma) * ce)[mumbai_valid].mean()

                probs = F.softmax(sub_logits, dim=0)
                dice_loss = 0.0
                for c_idx in range(2):
                    t_bin = ((sub_targets == c_idx) & mumbai_valid).float()
                    p_c = probs[c_idx] * mumbai_valid.float()
                    inter = (p_c * t_bin).sum()
                    card = (p_c + t_bin).sum()
                    dice = (2.0 * inter + self.smooth) / (card + self.smooth)
                    dice_loss += (1.0 - dice)
                dice_loss = dice_loss / 2.0

                total_loss += 0.50 * focal_ce + 0.50 * dice_loss

        return total_loss / max(1, B)

def evaluate_val_e4(model, val_loader):
    model.eval()
    cm_v = np.zeros((5, 5), dtype=np.int64)
    m_tp, m_fn = 0, 0
    with torch.no_grad():
        for imgs_list, masks, pids, dsets in val_loader:
            imgs_list = [img.to(device) for img in imgs_list]
            logits = model(imgs_list, dsets)
            gts = masks.numpy()
            
            for b, (l, g, d) in enumerate(zip(logits, gts, dsets)):
                valid = (g != 255)
                if not valid.any(): continue
                
                if d == "vashi_spot":
                    l_v = l.clone()
                    l_v[0] = -1e9
                    p = torch.argmax(l_v, dim=0).cpu().numpy()
                    for tc in range(1, 5):
                        mt = (g == tc) & valid
                        if mt.any():
                            for pc in range(5):
                                cm_v[tc, pc] += int((p[mt] == pc).sum())
                                
                elif d == "mumbai_shapefile":
                    p = torch.argmax(l, dim=0).cpu().numpy()
                    is_slum = (g == 0) & valid
                    m_tp += int(((p == 0) & is_slum).sum())
                    m_fn += int(((p != 0) & is_slum).sum())

    ious = []
    for c in range(1, 5):
        tp = cm_v[c, c]
        denom = cm_v[c, :].sum() + cm_v[:, c].sum() - tp
        if denom > 0:
            ious.append(tp / denom)
    v_miou = float(np.mean(ious) * 100) if ious else 0.0
    v_acc = float(np.trace(cm_v[1:5, 1:5]) / max(1, cm_v[1:5, :].sum()) * 100)
    m_det = float(m_tp / max(1, m_tp + m_fn) * 100)
    
    # Balanced non-harmonic selection score: 0.60 * Vashi_mIoU + 0.40 * Mumbai_Det
    balanced_score = 0.60 * v_miou + 0.40 * m_det
    return v_miou, v_acc, m_det, balanced_score, int(cm_v[1:5, 0].sum())

def main():
    print("\n" + "="*75)
    print("   STARTING EXPERIMENT E4: NDVI FEATURE + CALIBRATED FOCAL-DICE LOSS")
    print("   Vashi Input: 5 channels (RGB + NIR + NDVI)")
    print("   Mumbai Input: 3 channels (RGB)")
    print("   Loss: 0.50 * Masked Focal CE (gamma=1.5) + 0.50 * Masked Soft Dice")
    print("   Weights: Non-slum=1.25, Road=0.65, Veg=1.10, Open Land=1.05")
    print("="*75, flush=True)

    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    train_df = df[df["current_stage_in_v4"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    train_ds = SensorAdaptiveNDVIDataset(train_df, is_train=True)
    val_ds   = SensorAdaptiveNDVIDataset(val_df, is_train=False)

    sampler = BalancedBatchSampler(train_df, num_batches=40)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=ndvi_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=ndvi_collate_fn, num_workers=0)

    model = HighResNDVIAdaptiveUNet(num_classes=5).to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()
            m.requires_grad_(True)

    criterion = CalibratedFocalDiceLoss(gamma=1.5, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    epochs = 25
    accum_steps = 2
    best_score = -1.0
    best_ckpt = CKPT_DIR / "best_expE4_model.pth"
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
        v_miou, v_acc, m_det, b_score, f_slum = evaluate_val_e4(model, val_loader)

        is_best = b_score > best_score
        if is_best:
            best_score = b_score
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "balanced_score": b_score,
                "vashi_val_miou": v_miou,
                "vashi_val_acc": v_acc,
                "mumbai_val_det_rate": m_det,
                "false_slums": f_slum,
                "config": "Experiment_E4_NDVI_FocalDice"
            }, str(best_ckpt))

        print(f"Epoch {ep:02d}/25 | Loss: {ep_loss:.4f} | Vashi Val mIoU: {v_miou:.2f}% | Vashi Acc: {v_acc:.2f}% | Mumbai Det: {m_det:.2f}% | Score: {b_score:.2f} {'[SAVED BEST]' if is_best else ''}", flush=True)

        history.append({
            "epoch": ep,
            "loss": ep_loss,
            "vashi_val_miou": v_miou,
            "vashi_val_acc": v_acc,
            "mumbai_val_det_rate": m_det,
            "balanced_score": b_score,
            "false_slums": f_slum,
            "is_best": is_best
        })

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} min). Best Score: {best_score:.2f}")

    with open(AUDIT_DIR / "expE4_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

