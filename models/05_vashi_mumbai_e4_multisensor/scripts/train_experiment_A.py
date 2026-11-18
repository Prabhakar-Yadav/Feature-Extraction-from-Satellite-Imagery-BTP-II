"""
train_experiment_A.py
---------------------
Experiment A: Exact Reproduced Original Baseline
- Training Pool: Exactly 43 Vashi + 13 Mumbai training patches (70% zone).
- Validation Pool: Exactly 10 Vashi + 4 Mumbai internal validation patches (70% zone).
- Held-Out 30% Pool: Strictly excluded.
- Architecture: UnifiedUNet5Class (ResNet-18 + scSE Decoder Attention + Trainable BatchNorm).
- Input: 3-Band RGB for both datasets.
- Loss: Original Composite Loss (CrossEntropyLoss with calibrated class weights + Soft Dice).
  Weights: [Slum=1.30, Non-slum=1.15, Road=0.75, Vegetation=1.10, Open Land=1.00].
- Effective Batch Size: 4 (Batch size 2, gradient accumulation steps = 2).
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4.
- Scheduler: CosineAnnealingLR(25 epochs, eta_min=1e-6).
- Checkpoint Selection: Strictly based on validation mIoU on the 14 internal validation patches.
- Seed: 42.
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
from train_and_evaluate_5class import UnifiedUNet5Class, CompositeLoss5Class

CLASS_NAMES = ["Slum", "Non-slum", "Road", "Vegetation", "Open Land"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class RGBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row["img_input_path_rgb"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)

        if rgb.shape[:2] != (512, 512):
            rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

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

        norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).float()
        t_mask = torch.from_numpy(mask).long()
        return t_img, t_mask, row["patch_id"], row["dataset_source"]

class BalancedBatchSampler(Sampler):
    def __init__(self, df: pd.DataFrame, num_batches: int = 40):
        self.vashi_idx = df[df["dataset_source"] == "vashi_spot"].index.to_numpy()
        self.mumbai_idx = df[df["dataset_source"] == "mumbai_shapefile"].index.to_numpy()
        self.num_batches = num_batches

    def __iter__(self):
        v_shuf = np.random.permutation(self.vashi_idx)
        m_shuf = np.random.permutation(self.mumbai_idx)
        batches = []
        for i in range(self.num_batches):
            v_c = v_shuf[i % len(v_shuf)]
            m_c = m_shuf[i % len(m_shuf)]
            batch = [v_c, m_c] if random.random() > 0.5 else [m_c, v_c]
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return self.num_batches

def evaluate_val(model, val_loader):
    model.eval()
    cm_v = np.zeros((5, 5), dtype=np.int64)
    m_tp, m_fn = 0, 0
    with torch.no_grad():
        for imgs, masks, pids, dsets in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
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
    print("   STARTING EXPERIMENT A: REPRODUCED ORIGINAL BASELINE")
    print("="*60, flush=True)

    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)

    train_df = df[df["current_stage_in_v4"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    print(f"Data: Train={len(train_df)} ({len(train_df[train_df['dataset_source']=='vashi_spot'])} Vashi + {len(train_df[train_df['dataset_source']=='mumbai_shapefile'])} Mumbai)")
    print(f"      Val={len(val_df)} ({len(val_df[val_df['dataset_source']=='vashi_spot'])} Vashi + {len(val_df[val_df['dataset_source']=='mumbai_shapefile'])} Mumbai)")

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

    # Original Calibrated Class Weights: [Slum=1.30, Non-slum=1.15, Road=0.75, Veg=1.10, Open Land=1.00]
    weights = torch.tensor([1.30, 1.15, 0.75, 1.10, 1.00], dtype=torch.float32).to(device)
    criterion = CompositeLoss5Class(class_weights=weights, ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    epochs = 25
    accum_steps = 2
    best_score = -1.0
    best_ckpt = CKPT_DIR / "best_expA_model.pth"
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
            loss = criterion(logits, masks) / accum_steps
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
                "config": "Experiment_A_Reproduced_Baseline"
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
    print(f"\n[DONE] Experiment A completed in {elapsed/60:.1f} mins. Best Score: {best_score:.2f}. Checkpoint: {best_ckpt}", flush=True)

    with open(AUDIT_DIR / "expA_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()

