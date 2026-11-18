"""
train_experiment_D.py
---------------------
Experiment D: Experiment C + High-Resolution Native Decoder Branch + Road Topology Loss
- Architecture:
  * Sensor-Adaptive Dual-Stem (4-band Vashi SPOT + 3-band Mumbai Pléiades).
  * Native-Resolution Shallow Skip Branch:
    Processes native 512x512 input through a shallow conv block (32 channels)
    and feeds it directly into dec0, recovering sharp building corners and crisp road margins.
- Road Topology-Aware Loss:
  * SoftCLDiceLoss applied specifically to Road (Class 2) on Vashi patches to enforce
    continuous road network connectivity and prevent fragmentation.
- Loss Formulation:
  0.45 * Masked Conditional Softmax CE + 0.45 * Masked Soft Dice + 0.10 * SoftCLDice (Road).
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
from torch.utils.data import DataLoader
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
sys.path.append(str(BASE_DIR / "research_corrected_v3" / "scripts"))
from train_and_evaluate_5class import DecoderBlock, SCSEBlock
from train_experiment_C import SensorAdaptiveDataset, adaptive_collate_fn, evaluate_val_adaptive
from train_experiment_A import BalancedBatchSampler
from cldice_loss import SoftCLDiceLoss

class HighResSensorAdaptiveUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        # 1. Stems (stride 2)
        self.stem_vashi = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu
        )
        with torch.no_grad():
            self.stem_vashi[0].weight[:, :3] = backbone.conv1.weight
            self.stem_vashi[0].weight[:, 3] = backbone.conv1.weight.mean(dim=1)

        self.stem_mumbai = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        # 2. Native-Resolution Shallow Branch (stride 1 at 512x512)
        self.shallow_vashi = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
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

        # Shared Backbone
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Decoder with Native Resolution Skip at dec0
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        # dec0 connects upsampled d1 (64) + native shallow_feat (32) -> 32 channels
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

class TopologyAwarePartialLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.weights_vashi = torch.tensor([1.15, 0.75, 1.10, 1.00], dtype=torch.float32).to(device)
        self.weights_mumbai = torch.tensor([1.30, 1.10], dtype=torch.float32).to(device)
        self.cldice = SoftCLDiceLoss(iters=3)

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

                ce_loss = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_vashi,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                ce_loss = ce_loss[vashi_valid].mean()

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

                # Road Topology clDice loss (Road is index 1 in sub_logits, class 2 in original)
                road_prob = probs[1:2].unsqueeze(0) # [1, 1, H, W]
                road_gt = ((sub_targets == 1) & vashi_valid).unsqueeze(0).unsqueeze(0).float()
                if road_gt.sum() > 50:
                    topo_loss = self.cldice(road_prob, road_gt)
                else:
                    topo_loss = torch.tensor(0.0, device=logits.device)

                total_loss += 0.45 * ce_loss + 0.45 * dice_loss + 0.10 * topo_loss

            elif src == "mumbai_shapefile":
                mumbai_valid = valid & ((y == 0) | (y == 3))
                if not mumbai_valid.any(): continue
                sub_logits = torch.stack([l_b[0], l_b[3]], dim=0)
                sub_targets = torch.where(y == 0, 0, 1)

                ce_loss = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_mumbai,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                ce_loss = ce_loss[mumbai_valid].mean()

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

                total_loss += 0.5 * ce_loss + 0.5 * dice_loss

        return total_loss / max(1, B)

def main():
    print("\n" + "="*60)
    print("   STARTING EXPERIMENT D: HIGH-RES NATIVE SKIP + ROAD TOPOLOGY LOSS")
    print("="*60, flush=True)

    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    train_df = df[df["current_stage_in_v4"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    train_ds = SensorAdaptiveDataset(train_df, is_train=True)
    val_ds   = SensorAdaptiveDataset(val_df, is_train=False)

    sampler = BalancedBatchSampler(train_df, num_batches=40)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=adaptive_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=adaptive_collate_fn, num_workers=0)

    model = HighResSensorAdaptiveUNet(num_classes=5).to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()
            m.requires_grad_(True)

    criterion = TopologyAwarePartialLoss(smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    epochs = 25
    accum_steps = 2
    best_score = -1.0
    best_ckpt = CKPT_DIR / "best_expD_model.pth"
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
                "config": "Experiment_D_HighRes_Topology"
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
    print(f"\n[DONE] Experiment D completed in {elapsed/60:.1f} mins. Best Score: {best_score:.2f}. Checkpoint: {best_ckpt}", flush=True)

    with open(AUDIT_DIR / "expD_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

