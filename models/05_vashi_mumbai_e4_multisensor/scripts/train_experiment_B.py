"""
train_experiment_B.py
---------------------
Experiment B: Corrected Partial-Label Supervision
- Loss: Masked Source-Aware Conditional Softmax CE + Masked Soft Dice
  * Vashi patches: Supervised over {1, 2, 3, 4} (Non-slum, Road, Veg, Open Land).
    Gradient on Slum (c=0) is EXACTLY 0.0.
    Calibrated Class Weights: [Non-slum=1.15, Road=0.75, Veg=1.10, Open Land=1.00].
  * Mumbai patches: Supervised over {0, 3} (Slum, Veg).
    Gradient on Non-slum, Road, Open Land is EXACTLY 0.0.
    Calibrated Class Weights: [Slum=1.30, Veg=1.10].
- Effective Batch Size: 4 (Batch size 2, gradient accumulation steps = 2).
- Input: 3-band RGB.
- Checkpoint Selection: Internal validation score (harmonic mean of Vashi mIoU and Mumbai Slum detection).
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
from train_and_evaluate_5class import UnifiedUNet5Class
from train_experiment_A import RGBDataset, BalancedBatchSampler, evaluate_val

class MaskedConditionalSoftmaxLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        # Calibrated class weights
        self.weights_vashi = torch.tensor([1.15, 0.75, 1.10, 1.00], dtype=torch.float32).to(device) # for classes 1..4
        self.weights_mumbai = torch.tensor([1.30, 1.10], dtype=torch.float32).to(device) # for classes 0 and 3

    def forward(self, logits, targets, sources):
        B, C, H, W = logits.shape
        total_loss = 0.0

        for b in range(B):
            src = sources[b]
            y = targets[b] # [H, W]
            valid = (y != 255)
            if not valid.any():
                continue

            l_b = logits[b] # [5, H, W]

            if src == "vashi_spot":
                # Supervised classes: 1, 2, 3, 4. Shift targets: 1->0, 2->1, 3->2, 4->3
                vashi_valid = valid & (y >= 1) & (y <= 4)
                if not vashi_valid.any():
                    continue
                sub_logits = l_b[1:5] # [4, H, W]
                sub_targets = (y - 1).clamp(0, 3)

                ce_loss = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_vashi,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                ce_loss = ce_loss[vashi_valid].mean()

                # Masked Soft Dice over classes 1..4
                probs = F.softmax(sub_logits, dim=0) # [4, H, W]
                dice_loss = 0.0
                for c_idx in range(4):
                    t_bin = ((sub_targets == c_idx) & vashi_valid).float()
                    p_c = probs[c_idx] * vashi_valid.float()
                    inter = (p_c * t_bin).sum()
                    card = (p_c + t_bin).sum()
                    dice = (2.0 * inter + self.smooth) / (card + self.smooth)
                    dice_loss += (1.0 - dice)
                dice_loss = dice_loss / 4.0

                total_loss += 0.5 * ce_loss + 0.5 * dice_loss

            elif src == "mumbai_shapefile":
                # Supervised classes: 0 and 3. Map: 0->0, 3->1
                mumbai_valid = valid & ((y == 0) | (y == 3))
                if not mumbai_valid.any():
                    continue
                sub_logits = torch.stack([l_b[0], l_b[3]], dim=0) # [2, H, W]
                sub_targets = torch.where(y == 0, 0, 1)

                ce_loss = F.cross_entropy(
                    sub_logits.unsqueeze(0),
                    sub_targets.unsqueeze(0),
                    weight=self.weights_mumbai,
                    ignore_index=-1,
                    reduction="none"
                )[0]
                ce_loss = ce_loss[mumbai_valid].mean()

                probs = F.softmax(sub_logits, dim=0) # [2, H, W]
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
    print("   STARTING EXPERIMENT B: CORRECTED PARTIAL-LABEL SUPERVISION")
    print("="*60, flush=True)

    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    train_df = df[df["current_stage_in_v4"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    train_ds = RGBDataset(train_df, is_train=True)
    val_ds   = RGBDataset(val_df, is_train=False)

    sampler = BalancedBatchSampler(train_df, num_batches=40)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = UnifiedUNet5Class(num_classes=5).to(device)
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
    best_ckpt = CKPT_DIR / "best_expB_model.pth"
    history = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_b = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, (imgs, masks, pids, dsets) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
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
        v_miou, v_acc, m_det, comp_score, f_slum = evaluate_val(model, val_loader)

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
                "config": "Experiment_B_Masked_Conditional_Softmax"
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
    print(f"\n[DONE] Experiment B completed in {elapsed/60:.1f} mins. Best Score: {best_score:.2f}. Checkpoint: {best_ckpt}", flush=True)

    with open(AUDIT_DIR / "expB_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

