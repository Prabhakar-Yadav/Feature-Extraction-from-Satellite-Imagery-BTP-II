import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SUBCLASS_DIR = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation"
STAGE1_SCRIPTS = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "scripts"
SUBCLASS_SCRIPTS = SUBCLASS_DIR / "scripts"

sys.path.insert(0, str(STAGE1_SCRIPTS))
sys.path.insert(0, str(SUBCLASS_SCRIPTS))
sys.path.insert(0, str(SCRIPT_DIR))

from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep, regularize_subclass_blocks
from autonomous_overnight_trainer import AutonomousHierarchicalDataset

CLASS_NAMES_7 = [
    "Water", "Vegetation", "Barren Land",
    "Residential Built-up", "Industrial Built-up", "Slum / Informal Settlement",
    "Road Network"
]

CLASS_NAMES_5 = ["Water", "Vegetation", "Barren Land", "Built-up (Aggregate)", "Road Network"]
CLASS_NAMES_SUB = ["Residential", "Industrial", "Slum / Informal"]

def main():
    print("=" * 80)
    print("RUNNING HIGH-PRECISION DEEP DIAGNOSTIC EVALUATION (GPU ACCELERATED)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest_csv = SUBCLASS_DIR / "data" / "autonomous_dataset_manifest.csv"
    manifest_df = pd.read_csv(manifest_csv)
    val_df = manifest_df[manifest_df["split"] == "val"].copy().reset_index(drop=True)
    train_df = manifest_df[manifest_df["split"] == "train"].copy().reset_index(drop=True)

    print(f"Validation patches: {len(val_df)} | Training patches: {len(train_df)}")

    # 1. Dataset class distributions
    def get_class_counts(df):
        counts_7c = np.zeros(7, dtype=np.int64)
        for _, row in df.iterrows():
            m = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                for c in range(7):
                    counts_7c[c] += np.sum(m == c)
        return counts_7c

    print("\n[DIAGNOSTIC 1] Calculating True Ground-Truth Class Distributions...")
    val_counts = get_class_counts(val_df)
    train_counts = get_class_counts(train_df)

    total_val_px = val_counts.sum()
    total_train_px = train_counts.sum()

    print("\n--- CLASS PIXEL DISTRIBUTION ---")
    print(f"{'Class Name':30s} | {'Train Pixels':12s} ({'Train %':7s}) | {'Val Pixels':12s} ({'Val %':7s})")
    print("-" * 75)
    for c in range(7):
        tr_pct = (train_counts[c] / total_train_px) * 100
        val_pct = (val_counts[c] / total_val_px) * 100
        print(f"{CLASS_NAMES_7[c]:30s} | {train_counts[c]:12,d} ({tr_pct:6.2f}%) | {val_counts[c]:12,d} ({val_pct:6.2f}%)")

    # 2. Load Model Checkpoint
    ckpt_path = SUBCLASS_DIR / "checkpoints" / "best_autonomous_model.pth"
    print(f"\n[DIAGNOSTIC 2] Loading Checkpoint: {ckpt_path.name}...")
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model = HybridUNetSegFormerTwoStep(
        segformer_ckpt=str(PROJECT_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training/checkpoints/best_5class_segformer_extended.pth"),
        unet_ckpt=str(PROJECT_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training/checkpoints/best_5class_unet_multispectral.pth"),
        subclass_ckpt=str(SUBCLASS_DIR / "checkpoints/best_3subclass_segformer.pth"),
        device=device
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = AutonomousHierarchicalDataset(val_df, is_train=False, target_size=512)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Confusion matrices
    conf_7c = np.zeros((7, 7), dtype=np.int64)
    conf_5c = np.zeros((5, 5), dtype=np.int64)
    conf_sub_oracle = np.zeros((3, 3), dtype=np.int64)
    conf_sub_cascaded = np.zeros((3, 3), dtype=np.int64)

    print(f"\n[DIAGNOSTIC 3] Running GPU-Accelerated Inference on 61 Validation Patches...")

    with torch.no_grad():
        for idx, (img, target_5c, target_sub, orig_7c, rgb_vis, pid) in enumerate(val_loader):
            img = img.to(device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits_5c, logits_seg, logits_unet, prob_builtup, pred_builtup_mask = model.forward_step1(img)
                probs_5c = F.softmax(logits_5c, dim=1)
                pred_5c = torch.argmax(probs_5c, dim=1)

                l_seg = logits_seg.detach()
                l_unet = logits_unet.detach()
                p_bu = prob_builtup.detach()
                logits_sub = model.forward_step2(img, l_seg, l_unet, p_bu)
                pred_sub = torch.argmax(logits_sub, dim=1)

                p_5c_full, p_sub_full, p_7c_full, is_bu = model.predict_hierarchical_7class(img)

            gt_7c_np = orig_7c.numpy().squeeze(0).astype(np.uint8)
            pred_7c_np = p_7c_full.cpu().numpy().squeeze(0).astype(np.uint8)
            gt_5c_np = target_5c.numpy().squeeze(0).astype(np.uint8)
            pred_5c_np = pred_5c.cpu().numpy().squeeze(0).astype(np.uint8)

            gt_sub_np = target_sub.numpy().squeeze(0) # 0, 1, 2, or 255
            pred_sub_np = pred_sub.cpu().numpy().squeeze(0).astype(np.uint8)

            # 7-Class confusion matrix
            for g in range(7):
                mask_g = (gt_7c_np == g)
                if mask_g.any():
                    for p_i in range(7):
                        conf_7c[g, p_i] += np.sum((pred_7c_np == p_i) & mask_g)

            # Stage 1 (5-Class) confusion matrix
            for g in range(5):
                mask_g = (gt_5c_np == g)
                if mask_g.any():
                    for p_i in range(5):
                        conf_5c[g, p_i] += np.sum((pred_5c_np == p_i) & mask_g)

            # Stage 2 Oracle (Inside Ground-Truth Built-up)
            valid_gt_bu = (gt_sub_np != 255) & np.isin(gt_sub_np, [0, 1, 2])
            if valid_gt_bu.any():
                for g in range(3):
                    mask_g = (gt_sub_np == g) & valid_gt_bu
                    if mask_g.any():
                        for p_i in range(3):
                            conf_sub_oracle[g, p_i] += np.sum((pred_sub_np == p_i) & mask_g)

            # Stage 2 Cascaded (Inside Predicted Built-up)
            valid_pred_bu = is_bu.cpu().numpy().squeeze(0) & valid_gt_bu
            if valid_pred_bu.any():
                for g in range(3):
                    mask_g = (gt_sub_np == g) & valid_pred_bu
                    if mask_g.any():
                        for p_i in range(3):
                            conf_sub_cascaded[g, p_i] += np.sum((pred_sub_np == p_i) & mask_g)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(val_loader):
                print(f"  Completed [{idx+1:2d}/{len(val_loader)}] validation patches...", flush=True)

    def print_matrix_and_metrics(conf, class_names, title):
        print(f"\n{'='*30} {title} {'='*30}")
        n = len(class_names)
        print(f"{'Class':28s} | {'IoU (%)':8s} | {'Dice (%)':8s} | {'Prec (%)':8s} | {'Rec (%)':8s} | {'GT Px':10s} | {'Pred Px':10s}")
        print("-" * 88)
        ious, dices, precs, recs = [], [], [], []
        for c in range(n):
            tp = conf[c, c]
            fp = np.sum(conf[:, c]) - tp
            fn = np.sum(conf[c, :]) - tp
            gt_s = np.sum(conf[c, :])
            pr_s = np.sum(conf[:, c])

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if gt_s == 0 and pr_s == 0 else 0.0)
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else (1.0 if (2 * tp + fp + fn) == 0 else 0.0)
            prec = tp / pr_s if pr_s > 0 else 0.0
            rec = tp / gt_s if gt_s > 0 else 0.0

            ious.append(iou)
            dices.append(dice)
            precs.append(prec)
            recs.append(rec)
            print(f"{class_names[c]:28s} | {iou*100:8.2f} | {dice*100:8.2f} | {prec*100:8.2f} | {rec*100:8.2f} | {gt_s:10,d} | {pr_s:10,d}")

        present_classes = [c for c in range(n) if np.sum(conf[c, :]) > 0]
        miou = np.mean([ious[c] for c in present_classes]) if present_classes else 0.0
        mdice = np.mean([dices[c] for c in present_classes]) if present_classes else 0.0
        acc = np.trace(conf) / np.sum(conf) if np.sum(conf) > 0 else 0.0
        print("-" * 88)
        print(f"OVERALL: Pixel Accuracy = {acc*100:.2f}% | Mean IoU = {miou*100:.2f}% | Mean Dice = {mdice*100:.2f}%")

        print(f"\nRAW CONFUSION MATRIX (Rows: Ground Truth, Cols: Prediction):")
        header = f"{'GT \\ Pred':25s}" + "".join([f"{name[:8]:>10s}" for name in class_names])
        print(header)
        for r in range(n):
            row_str = f"{class_names[r]:25s}" + "".join([f"{conf[r, c]:10,d}" for c in range(n)])
            print(row_str)

        print(f"\nROW-NORMALIZED CONFUSION MATRIX (Recall % per class):")
        print(header)
        for r in range(n):
            row_tot = np.sum(conf[r, :])
            if row_tot > 0:
                row_str = f"{class_names[r]:25s}" + "".join([f"{conf[r, c]/row_tot*100:9.1f}%" for c in range(n)])
            else:
                row_str = f"{class_names[r]:25s}" + "".join([f"{'0.0%':>10s}" for _ in range(n)])
            print(row_str)

    print_matrix_and_metrics(conf_7c, CLASS_NAMES_7, "FULL 7-CLASS HIERARCHICAL SEGMENTATION")
    print_matrix_and_metrics(conf_5c, CLASS_NAMES_5, "STAGE 1: 5-CLASS MACRO SEGMENTATION (INDEPENDENT)")
    print_matrix_and_metrics(conf_sub_oracle, CLASS_NAMES_SUB, "STAGE 2: ORACLE SUBCLASS (Inside Ground-Truth Built-up)")
    print_matrix_and_metrics(conf_sub_cascaded, CLASS_NAMES_SUB, "STAGE 2: CASCADED SUBCLASS (Inside Stage 1 Predicted Built-up)")

if __name__ == "__main__":
    main()
