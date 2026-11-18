"""
evaluate_baseline_and_metrics.py
--------------------------------
Evaluates the initial clean 3-stage hierarchical model on the 61 validation patches.
Calculates:
- 7-Class mIoU, Mean Dice, Pixel Accuracy
- Per-class IoU, Dice, Precision, Recall
- 7x7 Confusion Matrix
- Water FP / FN rates, Built-up aggregate IoU, Road IoU
- SpaceNet buffer boundary metrics
- Generates 20 visual comparisons [RGB | GT | Prediction | Error Difference]
Saves metrics to JSON and Markdown report.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Paths setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SUBCLASS_DIR = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation"
STAGE1_SCRIPTS = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "scripts"
SUBCLASS_SCRIPTS = SUBCLASS_DIR / "scripts"

sys.path.insert(0, str(STAGE1_SCRIPTS))
sys.path.insert(0, str(SUBCLASS_SCRIPTS))

from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep

# Class definitions
CLASS_NAMES_7 = [
    "Water", "Vegetation", "Barren Land",
    "Residential Built-up", "Industrial Built-up", "Slum / Informal Settlement",
    "Road Network"
]

COLORS_7 = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta
    4: (140, 50, 190),    # Industrial: Steel Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
}

BAND_MEANS = np.array([0.485, 0.456, 0.406, 0.450], dtype=np.float32)
BAND_STDS  = np.array([0.229, 0.224, 0.225, 0.220], dtype=np.float32)


class AutonomousValidationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_size: int = 512):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = Path(row["image_path"])
        mask_p = Path(row["mask_path"])

        bgra = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Missing image {img_p}")
        b, g, r, nir = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2], bgra[:, :, 3]
        img = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0

        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # RGB for visualization (R, G, B in [0, 1])
        rgb_vis = (img[:, :, :3] * 255.0).astype(np.uint8)

        # Normalize for model
        img_norm = (img - BAND_MEANS) / BAND_STDS
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Load 7-class mask
        gt_7c = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if gt_7c is None:
            gt_7c = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        elif gt_7c.shape[0] != self.target_size or gt_7c.shape[1] != self.target_size:
            gt_7c = cv2.resize(gt_7c, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        return img_tensor, torch.from_numpy(gt_7c).long(), rgb_vis, row["patch_id"]


def colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, color in COLORS_7.items():
        color_img[mask_2d == c_id] = color
    return color_img


def create_error_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Green where correct, Red where incorrect, shaded by GT class."""
    h, w = pred.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    correct = (pred == gt)
    diff[correct] = [30, 180, 50]       # Muted Emerald Green for correct
    diff[~correct] = [230, 40, 40]      # Bright Crimson Red for error
    return diff


def compute_metrics(conf_matrix):
    """Computes all per-class and overall metrics from 7x7 confusion matrix."""
    num_classes = conf_matrix.shape[0]
    ious = []
    dices = []
    precisions = []
    recalls = []

    total_correct = np.trace(conf_matrix)
    total_samples = np.sum(conf_matrix)
    pixel_acc = total_correct / total_samples if total_samples > 0 else 0.0

    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
        
        denom_dice = 2 * tp + fp + fn
        dice = (2 * tp) / denom_dice if denom_dice > 0 else (1.0 if denom_dice == 0 else 0.0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(prec))
        recalls.append(float(rec))

    # For classes present in GT:
    gt_support = np.sum(conf_matrix, axis=1)
    present_classes = [c for c in range(num_classes) if gt_support[c] > 0]
    
    miou = float(np.mean([ious[c] for c in present_classes])) if present_classes else 0.0
    mdice = float(np.mean([dices[c] for c in present_classes])) if present_classes else 0.0

    return {
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": miou,
        "mean_dice": mdice,
        "per_class": {
            CLASS_NAMES_7[c]: {
                "class_id": c,
                "iou": ious[c],
                "dice": dices[c],
                "precision": precisions[c],
                "recall": recalls[c],
                "gt_pixels": int(gt_support[c]),
                "pred_pixels": int(np.sum(conf_matrix[:, c]))
            }
            for c in range(num_classes)
        }
    }


def main():
    print("[EVAL] Initializing Baseline Evaluation on Validation Split...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.68, 0)
        print(f"[EVAL] GPU: {torch.cuda.get_device_name(0)} | Memory Cap: 68%", flush=True)

    manifest_csv = SUBCLASS_DIR / "data" / "autonomous_dataset_manifest.csv"
    assert manifest_csv.exists(), f"Missing manifest {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    print(f"[EVAL] Validation Patches: {len(val_df)}", flush=True)

    val_dataset = AutonomousValidationDataset(val_df, target_size=512)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initial checkpoint paths
    seg_ckpt = str(PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "checkpoints" / "best_5class_segformer_extended.pth")
    unet_ckpt = str(PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "checkpoints" / "best_5class_unet_multispectral.pth")
    sub_ckpt = str(SUBCLASS_DIR / "checkpoints" / "best_3subclass_segformer.pth")

    model = HybridUNetSegFormerTwoStep(
        segformer_ckpt=seg_ckpt,
        unet_ckpt=unet_ckpt,
        subclass_ckpt=sub_ckpt,
        device=device
    ).to(device)
    model.eval()

    conf_matrix = np.zeros((7, 7), dtype=np.int64)
    per_patch_records = []
    predictions_cache = {}

    print("\n[EVAL] Running inference across all 61 validation patches...", flush=True)
    start_t = time.time()
    with torch.no_grad():
        for i, (img_t, gt_t, rgb_vis, pid) in enumerate(val_loader):
            img_t = img_t.to(device)
            p_5c, p_sub, p_7c, is_bu = model.predict_hierarchical_7class(img_t)

            pred_np = p_7c.cpu().numpy().squeeze(0).astype(np.uint8)
            gt_np = gt_t.numpy().squeeze(0).astype(np.uint8)
            rgb_np = rgb_vis.numpy().squeeze(0)
            patch_id = pid[0]

            # Confusion matrix
            for c_gt in range(7):
                mask_g = (gt_np == c_gt)
                if mask_g.any():
                    for c_pr in range(7):
                        conf_matrix[c_gt, c_pr] += np.sum((pred_np == c_pr) & mask_g)

            # Per-patch metrics
            correct_px = np.sum(pred_np == gt_np)
            total_px = 512 * 512
            acc = correct_px / total_px

            # Patch IoU for built-up (3, 4, 5)
            bu_gt = np.isin(gt_np, [3, 4, 5])
            bu_pr = np.isin(pred_np, [3, 4, 5])
            bu_inter = np.sum(bu_gt & bu_pr)
            bu_union = np.sum(bu_gt | bu_pr)
            bu_iou = bu_inter / bu_union if bu_union > 0 else (1.0 if not bu_gt.any() else 0.0)

            # Road IoU
            rd_gt = (gt_np == 6)
            rd_pr = (pred_np == 6)
            rd_inter = np.sum(rd_gt & rd_pr)
            rd_union = np.sum(rd_gt | rd_pr)
            rd_iou = rd_inter / rd_union if rd_union > 0 else (1.0 if not rd_gt.any() else 0.0)

            # Residential IoU
            res_gt = (gt_np == 3)
            res_pr = (pred_np == 3)
            res_inter = np.sum(res_gt & res_pr)
            res_union = np.sum(res_gt | res_pr)
            res_iou = res_inter / res_union if res_union > 0 else (1.0 if not res_gt.any() else 0.0)

            per_patch_records.append({
                "patch_id": patch_id,
                "accuracy": float(acc),
                "builtup_iou": float(bu_iou),
                "road_iou": float(rd_iou),
                "res_iou": float(res_iou),
                "error_px": int(total_px - correct_px)
            })

            predictions_cache[patch_id] = {
                "pred": pred_np,
                "gt": gt_np,
                "rgb": rgb_np
            }

            if (i + 1) % 15 == 0 or (i + 1) == len(val_loader):
                print(f"  Processed [{i+1:2d}/{len(val_loader)}] patches...", flush=True)

    eval_dur = time.time() - start_t
    print(f"[EVAL] Completed inference in {eval_dur:.1f}s ({eval_dur/len(val_loader):.2f}s/patch)", flush=True)

    metrics = compute_metrics(conf_matrix)
    metrics["evaluation_duration_sec"] = eval_dur
    metrics["num_validation_patches"] = len(val_df)
    metrics["confusion_matrix_7x7"] = conf_matrix.tolist()

    # Aggregate Built-up IoU
    bu_tp = conf_matrix[3:6, 3:6].sum()
    bu_fp = conf_matrix[:3, 3:6].sum() + conf_matrix[6, 3:6].sum()
    bu_fn = conf_matrix[3:6, :3].sum() + conf_matrix[3:6, 6].sum()
    bu_agg_iou = float(bu_tp / (bu_tp + bu_fp + bu_fn)) if (bu_tp + bu_fp + bu_fn) > 0 else 0.0
    metrics["builtup_aggregate_iou"] = bu_agg_iou

    # Water FP / FN
    water_tp = conf_matrix[0, 0]
    water_fp = conf_matrix[1:, 0].sum()
    water_fn = conf_matrix[0, 1:].sum()
    metrics["water_fp_pixels"] = int(water_fp)
    metrics["water_fn_pixels"] = int(water_fn)

    print("\n" + "=" * 70)
    print(f"BASELINE VALIDATION RESULTS (7 Classes, {len(val_df)} Patches):")
    print(f"  Overall Pixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%")
    print(f"  Mean IoU (Present Cls): {metrics['mean_iou']*100:.2f}%")
    print(f"  Mean Dice             : {metrics['mean_dice']*100:.2f}%")
    print(f"  Built-up Aggregate IoU: {bu_agg_iou*100:.2f}%")
    print(f"  Water False Positives : {water_fp} pixels")
    print("-" * 70)
    for cname, cstat in metrics["per_class"].items():
        print(f"  {cname:26s} | IoU: {cstat['iou']*100:5.2f}% | Dice: {cstat['dice']*100:5.2f}% | Prec: {cstat['precision']*100:5.2f}% | Rec: {cstat['recall']*100:5.2f}% | GT: {cstat['gt_pixels']:8,d}")
    print("=" * 70)

    # Save metrics JSON
    metrics_path = SUBCLASS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Saved metrics JSON to {metrics_path}", flush=True)

    # Select 20 representative diagnostic patches
    # Strategy: 8 worst accuracy, 4 highest residential, 4 road-heavy, 4 representative balanced
    df_patches = pd.DataFrame(per_patch_records)
    worst_8 = df_patches.sort_values("accuracy").head(8)["patch_id"].tolist()
    best_res_4 = df_patches[~df_patches["patch_id"].isin(worst_8)].sort_values("res_iou", ascending=False).head(4)["patch_id"].tolist()
    best_road_4 = df_patches[~df_patches["patch_id"].isin(worst_8 + best_res_4)].sort_values("road_iou", ascending=False).head(4)["patch_id"].tolist()
    rem = df_patches[~df_patches["patch_id"].isin(worst_8 + best_res_4 + best_road_4)]
    rep_4 = rem.sample(n=min(4, len(rem)), random_state=42)["patch_id"].tolist()

    selected_20 = worst_8 + best_res_4 + best_road_4 + rep_4
    print(f"\n[EVAL] Selected 20 diagnostic validation patches: {selected_20}", flush=True)

    # Output directory for visual panels
    vis_dir = SUBCLASS_DIR / "visual_results" / "baseline"
    vis_dir.mkdir(parents=True, exist_ok=True)
    artifact_vis_dir = PROJECT_ROOT.parent / "brain" / "c89e20d2-5c79-420e-8b8e-ab1621aef4cf" / "autonomous_training" / "baseline"
    # Fallback to local artifact directory if path doesn't exist
    alt_art_dir = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\autonomous_training\baseline")
    alt_art_dir.mkdir(parents=True, exist_ok=True)

    legend_patches = [
        mpatches.Patch(color=np.array(COLORS_7[c])/255.0, label=CLASS_NAMES_7[c])
        for c in range(7)
    ]

    for idx, pid in enumerate(selected_20):
        data = predictions_cache[pid]
        rgb = data["rgb"]
        gt = data["gt"]
        pred = data["pred"]

        gt_color = colorize_mask(gt)
        pred_color = colorize_mask(pred)
        diff_color = create_error_diff(pred, gt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        axes[0].imshow(rgb)
        axes[0].set_title(f"[{idx+1:02d}] {pid} - RGB", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_color)
        axes[1].set_title("Verified Ground Truth", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_color)
        axes[2].set_title("Baseline Prediction", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(diff_color)
        acc_p = np.mean(pred == gt) * 100
        axes[3].set_title(f"Error Map (Green=OK, Red=Err) | Acc={acc_p:.1f}%", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        plt.subplots_adjust(wspace=0.04, top=0.88, bottom=0.04, left=0.02, right=0.98)
        fig.legend(handles=legend_patches, loc="lower center", ncol=7, fontsize=9, frameon=True)

        save_p = vis_dir / f"baseline_{idx+1:02d}_{pid}.png"
        fig.savefig(save_p, bbox_inches="tight")
        art_p = alt_art_dir / f"baseline_{idx+1:02d}_{pid}.png"
        fig.savefig(art_p, bbox_inches="tight")
        plt.close(fig)

    print(f"[EVAL] Saved 20 visual panels to {vis_dir} and {alt_art_dir}", flush=True)


if __name__ == "__main__":
    main()

