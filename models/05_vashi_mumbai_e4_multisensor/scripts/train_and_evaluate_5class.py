"""
train_and_evaluate_5class.py
----------------------------
Retrains the Unified 5-Class Urban Segmentation Model with:
1. Clean Ground Truth (Water -> 255 Ignore, Barren -> 4 Open Land)
2. Source-Balanced Batch Sampling (50% Vashi, 50% Mumbai per batch)
3. Calibrated Class Loss Weights (Road dampened to 0.75, Non-slum boosted to 1.15)
4. Checkpoint Selection based on Validation mIoU on an internal validation set within the 70% zone
5. Strictly Untouched 30% Held-Out Evaluation (24 Vashi + 5 Mumbai patches)
6. Comprehensive Comparison Metrics & 6-Panel Visualization Generation
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.models as models

# VRAM configuration for RTX 3050 (4GB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    torch.backends.cudnn.benchmark = True
    print(f"[GPU SETUP] Active device: {torch.cuda.get_device_name(0)} (50% VRAM cap).")

BASE_DIR = Path("VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI")
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PRED_DIR = BASE_DIR / "predictions"
VIS_DIR = BASE_DIR / "visualizations"

IMPROVED_PRED_VASHI = PRED_DIR / "improved_predictions" / "vashi_heldout"
IMPROVED_PRED_MUMBAI = PRED_DIR / "improved_predictions" / "mumbai_heldout"
BASELINE_PRED_VASHI = PRED_DIR / "baseline_predictions" / "vashi_heldout"
BASELINE_PRED_MUMBAI = PRED_DIR / "baseline_predictions" / "mumbai_heldout"

for d in [CHECKPOINT_DIR, IMPROVED_PRED_VASHI, IMPROVED_PRED_MUMBAI, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 5-Class Taxonomy
CLASS_NAMES = ["Slum", "Non-slum", "Road", "Vegetation", "Open Land"]
CLASS_COLORS = {
    0: (255, 120, 0),     # Slum: Amber
    1: (220, 80, 70),     # Non-slum: Terracotta / Coral
    2: (255, 225, 20),    # Road: Yellow
    3: (34, 150, 40),     # Vegetation: Forest Green
    4: (65, 140, 200),    # Open Land: Slate Blue
    255: (20, 20, 20)     # Ignore: Dark Grey
}

def colorize_mask(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in CLASS_COLORS.items():
        rgb[mask == cid] = col
    return rgb

# -----------------------------------------------------------------------------
# 1. Dataset & Normalization
# -----------------------------------------------------------------------------
class UnifiedFiveClassDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, is_train: bool = True, target_size: int = 512):
        self.df = manifest_df.reset_index(drop=True)
        self.is_train = is_train
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = Path(row["img_rgb_path"])
        mask_p = Path(row["mask_5class_path"])

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            raise FileNotFoundError(f"Missing image: {img_p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {mask_p}")

        if img_rgb.shape[0] != self.target_size or img_rgb.shape[1] != self.target_size:
            img_rgb = cv2.resize(img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.is_train:
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img_rgb = np.flipud(img_rgb).copy()
                mask = np.flipud(mask).copy()
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                img_rgb = np.rot90(img_rgb, k).copy()
                mask = np.rot90(mask, k).copy()

        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, row["patch_id"], row["dataset"]

# -----------------------------------------------------------------------------
# 2. Source-Balanced Batch Sampler (50% Vashi, 50% Mumbai per batch)
# -----------------------------------------------------------------------------
class SourceBalancedBatchSampler(Sampler):
    """
    Ensures every batch of size 2 contains exactly 1 Vashi patch and 1 Mumbai patch.
    This guarantees 50/50 gradient balance between datasets on every single optimization step.
    """
    def __init__(self, df: pd.DataFrame, num_batches: int = 45):
        self.vashi_indices = df[df["dataset"] == "vashi"].index.to_numpy()
        self.mumbai_indices = df[df["dataset"] == "mumbai"].index.to_numpy()
        self.num_batches = num_batches

    def __iter__(self):
        v_idx = np.random.permutation(self.vashi_indices)
        m_idx = np.random.permutation(self.mumbai_indices)
        
        batches = []
        for i in range(self.num_batches):
            v_chosen = v_idx[i % len(v_idx)]
            m_chosen = m_idx[i % len(m_idx)]
            # Randomize order in batch
            if np.random.rand() > 0.5:
                batch = [v_chosen, m_chosen]
            else:
                batch = [m_chosen, v_chosen]
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return self.num_batches

# -----------------------------------------------------------------------------
# 3. Model Architecture: UnifiedUNet5Class
# -----------------------------------------------------------------------------
class SCSEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.scse(x)

class UnifiedUNet5Class(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 0, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)
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

# -----------------------------------------------------------------------------
# 4. Composite Loss with Calibrated Weights
# -----------------------------------------------------------------------------
class CompositeLoss5Class(nn.Module):
    def __init__(self, class_weights=None, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # Soft Dice Loss
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)
        dice_loss = 0.0
        n_classes = logits.shape[1]

        for c in range(n_classes):
            target_c = (targets == c).float() * valid.float()
            prob_c = probs[:, c, :, :] * valid.float()
            intersection = (prob_c * target_c).sum(dim=(1, 2))
            cardinality = (prob_c + target_c).sum(dim=(1, 2))
            dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
            dice_loss += (1.0 - dice).mean()

        dice_loss = dice_loss / n_classes
        return 0.5 * ce_loss + 0.5 * dice_loss

# -----------------------------------------------------------------------------
# 5. Validation Function for Internal Checkpoint Selection
# -----------------------------------------------------------------------------
def evaluate_val_set(model, val_loader):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks, pids, dsets in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.numpy()
            
            for p, g, d in zip(preds, gts, dsets):
                valid = (g != 255)
                if not valid.any():
                    continue
                vp = p[valid]
                vg = g[valid]
                for t_c in range(5):
                    mask_t = (vg == t_c)
                    if mask_t.any():
                        for p_c in range(5):
                            cm[t_c, p_c] += (vp[mask_t] == p_c).sum()

    ious = []
    for c in range(5):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    mIoU = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return mIoU, cm

# -----------------------------------------------------------------------------
# 6. Training Pipeline
# -----------------------------------------------------------------------------
def train_model():
    print("\n--- Retraining Unified 5-Class Segmentation Model ---", flush=True)
    torch.cuda.empty_cache()
    
    # Load manifest
    manifest_p = DATA_DIR / "manifests" / "unified_train_manifest.csv"
    df_all = pd.read_csv(manifest_p)
    
    train_df = df_all[df_all["stage"] == "train"].copy().reset_index(drop=True)
    val_df = df_all[df_all["stage"] == "val"].copy().reset_index(drop=True)
    
    print(f"Dataset Split (within 70% zone):")
    print(f"  Training: {len(train_df)} patches ({(train_df['dataset']=='vashi').sum()} Vashi + {(train_df['dataset']=='mumbai').sum()} Mumbai)")
    print(f"  Internal Validation: {len(val_df)} patches ({(val_df['dataset']=='vashi').sum()} Vashi + {(val_df['dataset']=='mumbai').sum()} Mumbai)")

    train_ds = UnifiedFiveClassDataset(train_df, is_train=True, target_size=512)
    val_ds = UnifiedFiveClassDataset(val_df, is_train=False, target_size=512)

    # 50/50 Source-Balanced Sampler
    batch_sampler = SourceBalancedBatchSampler(train_df, num_batches=45)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    # Calibrated Class Weights:
    # 0: Slum (1.30) - high emphasis for informal settlements
    # 1: Non-slum (1.15) - reinforced building footprints
    # 2: Road (0.75) - dampened to stop overprediction and false dilation
    # 3: Vegetation (1.10) - good balance
    # 4: Open Land (1.00) - clean barren land (water excluded)
    weights = np.array([1.30, 1.15, 0.75, 1.10, 1.00], dtype=np.float32)
    weights_tensor = torch.from_numpy(weights).to(device)
    print("Calibrated Class Loss Weights [Slum, Non-slum, Road, Veg, Open Land]:", weights.tolist(), flush=True)

    model = UnifiedUNet5Class(num_classes=5).to(device)
    criterion = CompositeLoss5Class(class_weights=weights_tensor, ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    epochs = 25
    best_val_miou = -1.0
    accum_steps = 2
    best_ckpt_p = CHECKPOINT_DIR / "improved_5class_unified_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        t0 = time.time()

        for b_idx, (imgs, masks, pids, dsets) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks) / accum_steps
            loss.backward()

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, n_batches)

        # Evaluate on Internal Validation Set
        val_miou, _ = evaluate_val_set(model, val_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val mIoU: {val_miou*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s", flush=True)

        # Checkpoint selection strictly based on validation mIoU!
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "train_loss": epoch_loss,
                "class_names": CLASS_NAMES,
                "taxonomy": "5-Class Unified: Slum, Non-slum, Road, Veg, Open Land"
            }, best_ckpt_p)
            print(f"  >>> New best checkpoint saved (Val mIoU: {val_miou*100:.2f}%)", flush=True)

        torch.cuda.empty_cache()

    print(f"\nTraining completed! Best checkpoint saved to: {best_ckpt_p} with Val mIoU {best_val_miou*100:.2f}%", flush=True)
    return best_ckpt_p

# -----------------------------------------------------------------------------
# 7. Held-Out 30% Evaluation & Improved Prediction Generation
# -----------------------------------------------------------------------------
def evaluate_heldout(ckpt_p):
    print("\n--- Evaluating Improved Model on Strictly Held-Out 30% Region ---")
    heldout_manifest = DATA_DIR / "manifests" / "heldout_evaluation_manifest.csv"
    heldout_df = pd.read_csv(heldout_manifest)
    heldout_ds = UnifiedFiveClassDataset(heldout_df, is_train=False, target_size=512)
    heldout_loader = DataLoader(heldout_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UnifiedUNet5Class(num_classes=5).to(device)
    ckpt = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cm_vashi = np.zeros((5, 5), dtype=np.int64)
    cm_mumbai = np.zeros((5, 5), dtype=np.int64)

    total_valid_vashi = 0
    total_correct_vashi = 0
    total_valid_mumbai = 0
    total_correct_mumbai = 0

    patch_results = []

    with torch.no_grad():
        for imgs, masks, pids, dsets in heldout_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            gt = masks.numpy()[0]
            pid = pids[0]
            dset = dsets[0]

            out_subfolder = IMPROVED_PRED_VASHI if dset == "vashi" else IMPROVED_PRED_MUMBAI
            pred_mask_p = out_subfolder / f"{pid}_pred_mask.png"
            cv2.imwrite(str(pred_mask_p), preds.astype(np.uint8))

            pred_color_p = out_subfolder / f"{pid}_pred_color.png"
            cv2.imwrite(str(pred_color_p), cv2.cvtColor(colorize_mask(preds), cv2.COLOR_RGB2BGR))

            row = heldout_df[heldout_df["patch_id"] == pid].iloc[0]
            raw_rgb = cv2.imread(row["img_rgb_path"])
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
            if raw_rgb.shape[:2] != (512, 512):
                raw_rgb = cv2.resize(raw_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

            pred_overlay = (raw_rgb * 0.45 + colorize_mask(preds) * 0.55).astype(np.uint8)
            pred_overlay_p = out_subfolder / f"{pid}_pred_overlay.png"
            cv2.imwrite(str(pred_overlay_p), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

            # Error map (Green: Correct, Red: Error, Black: Ignore)
            error_map = np.zeros((512, 512, 3), dtype=np.uint8)
            valid = (gt != 255)
            correct = valid & (preds == gt)
            incorrect = valid & (preds != gt)
            error_map[correct] = (30, 220, 50)   # Green
            error_map[incorrect] = (240, 40, 40) # Red
            error_map_p = out_subfolder / f"{pid}_error_map.png"
            cv2.imwrite(str(error_map_p), cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))

            if valid.any():
                v_preds = preds[valid]
                v_gt = gt[valid]
                n_corr = int((v_preds == v_gt).sum())
                n_val = int(valid.sum())

                if dset == "vashi":
                    total_valid_vashi += n_val
                    total_correct_vashi += n_corr
                    for t_c in range(5):
                        mask_t = (v_gt == t_c)
                        if mask_t.any():
                            for p_c in range(5):
                                cm_vashi[t_c, p_c] += (v_preds[mask_t] == p_c).sum()
                else:
                    total_valid_mumbai += n_val
                    total_correct_mumbai += n_corr
                    for t_c in range(5):
                        mask_t = (v_gt == t_c)
                        if mask_t.any():
                            for p_c in range(5):
                                cm_mumbai[t_c, p_c] += (v_preds[mask_t] == p_c).sum()

            patch_results.append({
                "patch_id": pid,
                "dataset": dset,
                "pred_mask_path": str(pred_mask_p),
                "pred_color_path": str(pred_color_p),
                "pred_overlay_path": str(pred_overlay_p),
                "error_map_path": str(error_map_p)
            })

    def cm_to_metrics(cm, dset_name):
        res = {}
        ious = []
        for c in range(5):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            prec_denom = tp + fp
            rec_denom = tp + fn

            if rec_denom == 0:
                res[CLASS_NAMES[c]] = {
                    "IoU": "N/A (No GT in heldout)",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "F1": "N/A",
                    "GT_pixels": 0
                }
            else:
                iou = float(tp / denom) if denom > 0 else 0.0
                prec = float(tp / prec_denom) if prec_denom > 0 else 0.0
                rec = float(tp / rec_denom) if rec_denom > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                res[CLASS_NAMES[c]] = {
                    "IoU": round(iou * 100.0, 2),
                    "Precision": round(prec * 100.0, 2),
                    "Recall": round(rec * 100.0, 2),
                    "F1": round(f1 * 100.0, 2),
                    "GT_pixels": int(rec_denom)
                }
                ious.append(iou)

        res["mean_IoU"] = round(float(np.mean(ious) * 100.0), 2) if len(ious) > 0 else 0.0
        return res

    vashi_metrics = cm_to_metrics(cm_vashi, "Vashi")
    vashi_metrics["pixel_accuracy"] = round(total_correct_vashi / max(1, total_valid_vashi) * 100.0, 2)

    mumbai_metrics = cm_to_metrics(cm_mumbai, "Mumbai")
    mumbai_metrics["pixel_accuracy"] = round(total_correct_mumbai / max(1, total_valid_mumbai) * 100.0, 2)

    all_metrics = {
        "vashi_heldout_30pct": vashi_metrics,
        "mumbai_heldout_30pct": mumbai_metrics,
        "taxonomy": CLASS_NAMES,
        "patches_evaluated": len(patch_results),
        "confusion_matrix_vashi": cm_vashi.tolist(),
        "confusion_matrix_mumbai": cm_mumbai.tolist()
    }

    metrics_file = BASE_DIR / "evaluation_metrics_improved_5class.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=======================================================")
    print("   IMPROVED HELD-OUT 30% EVALUATION METRICS SUMMARY    ")
    print("=======================================================")
    print(f"\nVASHI HELD-OUT (Sectors 2 & 4, 24 Patches):")
    print(f"  Overall Pixel Accuracy: {vashi_metrics['pixel_accuracy']}%")
    print(f"  Mean IoU: {vashi_metrics['mean_IoU']}%")
    for name in ["Non-slum", "Road", "Vegetation", "Open Land"]:
        m = vashi_metrics[name]
        print(f"  {name:12s} | IoU: {m['IoU']}% | Prec: {m['Precision']}% | Rec: {m['Recall']}% | F1: {m['F1']}%")

    print(f"\nMUMBAI HELD-OUT (Lat >= 19.158°N, 5 Patches):")
    print(f"  Slum IoU: {mumbai_metrics['Slum']['IoU']}% | Prec: {mumbai_metrics['Slum']['Precision']}% | Rec: {mumbai_metrics['Slum']['Recall']}% | F1: {mumbai_metrics['Slum']['F1']}%")
    print("=======================================================\n")
    return all_metrics

# -----------------------------------------------------------------------------
# 8. Side-by-Side 6-Panel Visual Comparisons (Baseline vs Improved)
# -----------------------------------------------------------------------------
def generate_comparison_visuals(patch_ids_vashi, patch_ids_mumbai):
    print("\n--- Generating High-Resolution 6-Panel Visual Comparisons ---")
    
    heldout_manifest = DATA_DIR / "manifests" / "heldout_evaluation_manifest.csv"
    heldout_df = pd.read_csv(heldout_manifest)

    # 6 Panels:
    # 1. Satellite Image (RGB)
    # 2. Ground Truth Mask (Colorized)
    # 3. Previous Baseline Model Prediction
    # 4. Retrained Improved Model Prediction
    # 5. Prediction Overlap on Satellite Image
    # 6. Error Map (Green: Correct, Red: Error, Black: Ignore)

    def render_6panel_patch(pid, dset, title_suffix=""):
        row = heldout_df[heldout_df["patch_id"] == pid].iloc[0]
        
        # 1. Satellite Image
        img_bgr = cv2.imread(row["img_rgb_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (512, 512):
            img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 2. Ground Truth
        gt_mask = cv2.imread(row["mask_5class_path"], cv2.IMREAD_UNCHANGED)
        gt_color = colorize_mask(gt_mask)

        # 3. Baseline Prediction
        base_dir = BASELINE_PRED_VASHI if dset == "vashi" else BASELINE_PRED_MUMBAI
        base_color_p = base_dir / f"{pid}_pred_color.png"
        if base_color_p.exists():
            base_color = cv2.imread(str(base_color_p))
            base_color = cv2.cvtColor(base_color, cv2.COLOR_BGR2RGB)
        else:
            base_color = np.zeros((512, 512, 3), dtype=np.uint8)

        # 4. Improved Prediction
        imp_dir = IMPROVED_PRED_VASHI if dset == "vashi" else IMPROVED_PRED_MUMBAI
        imp_color_p = imp_dir / f"{pid}_pred_color.png"
        imp_color = cv2.imread(str(imp_color_p))
        imp_color = cv2.cvtColor(imp_color, cv2.COLOR_BGR2RGB)

        # 5. Improved Overlay
        imp_overlay_p = imp_dir / f"{pid}_pred_overlay.png"
        imp_overlay = cv2.imread(str(imp_overlay_p))
        imp_overlay = cv2.cvtColor(imp_overlay, cv2.COLOR_BGR2RGB)

        # 6. Error Map
        err_p = imp_dir / f"{pid}_error_map.png"
        err_map = cv2.imread(str(err_p))
        err_map = cv2.cvtColor(err_map, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), dpi=200)
        
        panels = [
            (img_rgb, "1. Satellite Image\n(RGB)"),
            (gt_color, "2. Ground Truth\n(5-Class)"),
            (base_color, "3. Baseline Model\nPrediction"),
            (imp_color, "4. Retrained Model\nPrediction (Improved)"),
            (imp_overlay, "5. Improved Prediction\nOverlay on Real Image"),
            (err_map, "6. Error Map\n(Green: OK, Red: Err)")
        ]

        for ax, (im, ptitle) in zip(axes, panels):
            ax.imshow(im)
            ax.set_title(ptitle, fontsize=11, fontweight="bold", pad=8)
            ax.axis("off")

        # Add Legend
        patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[0])/255.0, label="Slum"),
            mpatches.Patch(color=np.array(CLASS_COLORS[1])/255.0, label="Non-slum"),
            mpatches.Patch(color=np.array(CLASS_COLORS[2])/255.0, label="Road"),
            mpatches.Patch(color=np.array(CLASS_COLORS[3])/255.0, label="Vegetation"),
            mpatches.Patch(color=np.array(CLASS_COLORS[4])/255.0, label="Open Land"),
            mpatches.Patch(color=np.array(CLASS_COLORS[255])/255.0, label="Ignore (255)")
        ]
        fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.05), frameon=True)
        
        plt.suptitle(f"Held-Out Evaluation Patch: {pid} ({title_suffix})", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        out_png = VIS_DIR / f"comparison_6panel_{pid}.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 6-panel comparison: {out_png}")
        return out_png

    vashi_imgs = []
    for pid in patch_ids_vashi:
        p_img = render_6panel_patch(pid, "vashi", "Vashi Sector 2/4 Held-Out")
        vashi_imgs.append(p_img)

    mumbai_imgs = []
    for pid in patch_ids_mumbai:
        p_img = render_6panel_patch(pid, "mumbai", "Mumbai Lat >= 19.158°N Held-Out")
        mumbai_imgs.append(p_img)

    return vashi_imgs, mumbai_imgs

if __name__ == "__main__":
    ckpt = train_model()
    metrics = evaluate_heldout(ckpt)
    
    # Generate 6-panel comparisons for 3 Vashi and 3 Mumbai patches
    vashi_samples = [
        "vashi_0064_Sector_2_r1536_c2458",
        "vashi_0052_Sector_2_r1280_c2048",
        "vashi_0023_Sector_4_r0512_c2304"
    ]
    mumbai_samples = [
        "mumbai_013_malad_appapada",
        "mumbai_014_malad_pathan",
        "mumbai_015_kandivali_damu"
    ]
    generate_comparison_visuals(vashi_samples, mumbai_samples)
    print("\nAll pipeline tasks executed successfully!")
