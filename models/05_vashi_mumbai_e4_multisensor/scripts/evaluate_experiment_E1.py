"""
evaluate_experiment_E1.py
--------------------------
Comprehensive evaluation and comparison of:
1. Experiment B (Masked Partial-Label)
2. Experiment C (Sensor-Adaptive NIR)
3. Experiment D (High-Res + clDice)
4. Experiment E1 (High-Res, No clDice)

Tasks:
1. Evaluate all 4 models on Internal Validation Set (14 patches: 10 Vashi + 4 Mumbai).
   - Pixel Accuracy & Global 4-class mIoU
   - Per-class IoU, Precision, Recall
   - 5x5 Confusion Matrix
   - Predicted class pixel percentages
   - Boundary F1
2. Specific failure analysis on target patches:
   - vashi_0041_Sector_4_r1024_c1792 (held-out diagnostic target)
   - vashi_0020_Sector_5_r0512_c1536 (training diagnostic patch)
   - vashi_0003_Sector_5_r0000_c1536 (validation patch)
3. Generate 7-panel visual comparisons for each of the 3 patches:
   [RGB, Ground Truth, Exp B, Exp C, Exp D, Exp E1, Error Maps]
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

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path("VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI")
EXP2_DIR = BASE_DIR / "research_corrected_v4" / "experiment_2"
EXPE1_DIR = BASE_DIR / "research_corrected_v4" / "experiment_E1"

CKPT_DIR_E1 = EXPE1_DIR / "checkpoints"
AUDIT_DIR_E1 = EXPE1_DIR / "audits"
PRED_DIR_E1 = EXPE1_DIR / "predictions"
VIS_DIR_E1 = EXPE1_DIR / "visualizations"

for d in [CKPT_DIR_E1, AUDIT_DIR_E1, PRED_DIR_E1, VIS_DIR_E1]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "scripts"))
sys.path.append(str(EXP2_DIR / "scripts"))
sys.path.append(str(EXPE1_DIR / "scripts"))

from train_and_evaluate_5class import UnifiedUNet5Class
from train_experiment_C import SensorAdaptiveUNet, VASHI_MEANS, VASHI_STDS, MUMBAI_MEANS, MUMBAI_STDS
from train_experiment_D import HighResSensorAdaptiveUNet

CLASS_NAMES = ["Slum", "Non-slum", "Road", "Vegetation", "Open Land"]
CLASS_COLORS = {
    0: (255, 120, 0),     # Slum: Amber
    1: (220, 80, 70),     # Non-slum: Terracotta
    2: (255, 225, 20),    # Road: Yellow
    3: (34, 150, 40),     # Vegetation: Forest Green
    4: (65, 140, 200),    # Open Land: Slate Blue
    255: (40, 40, 40)     # Ignore: Dark Charcoal
}

MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_RGB  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in CLASS_COLORS.items():
        rgb[mask == cid] = col
    return rgb

def compute_boundary_f1(pred, gt, theta=2):
    valid = (gt != 255)
    f1_list = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for c in range(1, 5):
        gt_c = ((gt == c) & valid).astype(np.uint8)
        pr_c = ((pred == c) & valid).astype(np.uint8)
        if gt_c.sum() == 0 and pr_c.sum() == 0:
            continue
        if gt_c.sum() == 0 or pr_c.sum() == 0:
            f1_list.append(0.0)
            continue
        gt_b = cv2.dilate(gt_c, kernel) - gt_c
        pr_b = cv2.dilate(pr_c, kernel) - pr_c

        tol_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*theta+1, 2*theta+1))
        gt_b_dil = cv2.dilate(gt_b, tol_kernel)
        pr_b_dil = cv2.dilate(pr_b, tol_kernel)

        prec = (pr_b * gt_b_dil).sum() / max(1, pr_b.sum())
        rec  = (gt_b * pr_b_dil).sum() / max(1, gt_b.sum())
        bf1 = 2 * prec * rec / max(1e-5, (prec + rec))
        f1_list.append(float(bf1))
    return float(np.mean(f1_list) * 100) if f1_list else 0.0

def load_input_for_model(row, model_type):
    src = row["dataset_source"]
    bgr = cv2.imread(row["img_input_path_rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (512, 512):
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

    if model_type in ["baseline", "expA", "expB"]:
        norm = (rgb.astype(np.float32) / 255.0 - MEAN_RGB) / STD_RGB
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return t_img, rgb
    else:
        if src == "vashi_spot":
            npy_p = row["img_input_path_4band"]
            data = np.load(npy_p).astype(np.float32)
            data = np.where(data == 65535, VASHI_MEANS, data)
            norm = (data - VASHI_MEANS) / VASHI_STDS
            t_img = torch.from_numpy(norm).float()
        else:
            norm = (rgb.astype(np.float32) / 255.0 - MUMBAI_MEANS) / MUMBAI_STDS
            t_img = torch.from_numpy(norm.transpose(2, 0, 1)).float()
        return t_img, rgb

def predict_patch(model, model_type, row):
    src = row["dataset_source"]
    t_img, rgb = load_input_for_model(row, model_type)
    t_img = t_img.to(device)

    with torch.no_grad():
        if model_type in ["baseline", "expA", "expB"]:
            logits = model(t_img)[0]
        else:
            logits = model([t_img], [src])[0]

        # Domain restriction for Vashi
        if src == "vashi_spot":
            logits_eval = logits.clone()
            logits_eval[0] = -1e9
            pred = torch.argmax(logits_eval, dim=0).cpu().numpy()
        else:
            pred = torch.argmax(logits, dim=0).cpu().numpy()
            
    return pred, logits.cpu().numpy()

def evaluate_on_val_set(model, model_type, val_df):
    model.eval()
    cm_v = np.zeros((5, 5), dtype=np.int64)
    m_tp, m_fn = 0, 0
    bf1_scores = []
    patch_records = []

    for _, row in val_df.iterrows():
        pid = row["patch_id"]
        src = row["dataset_source"]
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask.shape[:2] != (512, 512):
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        pred, _ = predict_patch(model, model_type, row)
        valid = (mask != 255)
        if not valid.any(): continue

        if src == "vashi_spot":
            for tc in range(1, 5):
                mt = (mask == tc) & valid
                if mt.any():
                    for pc in range(5):
                        cm_v[tc, pc] += int((pred[mt] == pc).sum())

            bf1 = compute_boundary_f1(pred, mask)
            bf1_scores.append(bf1)

            p_acc = float((pred[valid] == mask[valid]).sum() / valid.sum() * 100)
            p_ious = []
            for c in range(1, 5):
                tp = int(((pred == c) & (mask == c) & valid).sum())
                denom = int(((pred == c) | (mask == c) & valid).sum())
                if denom > 0:
                    p_ious.append(tp / denom)
            p_miou = float(np.mean(p_ious) * 100) if p_ious else 0.0

            patch_records.append({
                "patch_id": pid,
                "accuracy": round(p_acc, 2),
                "miou": round(p_miou, 2),
                "bf1": round(bf1, 2)
            })

        elif src == "mumbai_shapefile":
            is_slum_gt = (mask == 0) & valid
            tp_cnt = int(((pred == 0) & is_slum_gt).sum())
            fn_cnt = int(((pred != 0) & is_slum_gt).sum())
            m_tp += tp_cnt
            m_fn += fn_cnt

    # Compile global metrics on Vashi val (classes 1..4)
    per_class = {}
    ious = []
    total_valid = int(cm_v[1:5, :].sum())
    total_corr = int(sum(cm_v[c, c] for c in range(1, 5)))
    v_acc = total_corr / max(1, total_valid) * 100.0

    pred_class_counts = {c: int(cm_v[1:5, c].sum()) for c in range(5)}
    pred_class_pcts = {c: float(pred_class_counts[c] / max(1, total_valid) * 100.0) for c in range(5)}

    for c in range(1, 5):
        c_name = CLASS_NAMES[c]
        tp = int(cm_v[c, c])
        fp = int(cm_v[:, c].sum() - tp)
        fn = int(cm_v[c, :].sum() - tp)
        denom = tp + fp + fn
        iou = tp / max(1, denom) * 100.0
        prec = tp / max(1, tp + fp) * 100.0
        rec = tp / max(1, tp + fn) * 100.0
        ious.append(iou)
        per_class[c_name] = {
            "iou": round(iou, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "gt_count": int(cm_v[c, :].sum()),
            "pred_count": int(cm_v[:, c].sum())
        }

    global_miou = float(np.mean(ious))
    mean_bf1 = float(np.mean(bf1_scores)) if bf1_scores else 0.0
    m_det = float(m_tp / max(1, m_tp + m_fn) * 100.0)

    return {
        "pixel_accuracy": round(v_acc, 2),
        "global_miou": round(global_miou, 2),
        "boundary_f1": round(mean_bf1, 2),
        "mumbai_slum_det_rate": round(m_det, 2),
        "per_class": per_class,
        "confusion_matrix": cm_v.tolist(),
        "pred_class_pcts": pred_class_pcts,
        "patch_records": patch_records
    }

def main():
    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)
    val_df = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    print("="*80)
    print("LOADING MODELS: EXPERIMENT B, EXPERIMENT C, EXPERIMENT D, EXPERIMENT E1")
    print("="*80)

    # 1. Exp B
    m_b = UnifiedUNet5Class(5).to(device)
    ckpt_b = torch.load(EXP2_DIR / "checkpoints" / "best_expB_model.pth", map_location=device, weights_only=False)
    m_b.load_state_dict(ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b)
    m_b.eval()

    # 2. Exp C
    m_c = SensorAdaptiveUNet(5).to(device)
    ckpt_c = torch.load(EXP2_DIR / "checkpoints" / "best_expC_model.pth", map_location=device, weights_only=False)
    m_c.load_state_dict(ckpt_c["model_state_dict"] if "model_state_dict" in ckpt_c else ckpt_c)
    m_c.eval()

    # 3. Exp D
    m_d = HighResSensorAdaptiveUNet(5).to(device)
    ckpt_d = torch.load(EXP2_DIR / "checkpoints" / "best_expD_model.pth", map_location=device, weights_only=False)
    m_d.load_state_dict(ckpt_d["model_state_dict"] if "model_state_dict" in ckpt_d else ckpt_d)
    m_d.eval()

    # 4. Exp E1
    m_e1 = HighResSensorAdaptiveUNet(5).to(device)
    ckpt_e1 = torch.load(CKPT_DIR_E1 / "best_expE1_model.pth", map_location=device, weights_only=False)
    m_e1.load_state_dict(ckpt_e1["model_state_dict"] if "model_state_dict" in ckpt_e1 else ckpt_e1)
    m_e1.eval()

    models = [
        ("Experiment_B", "expB", m_b),
        ("Experiment_C", "expC", m_c),
        ("Experiment_D", "expD", m_d),
        ("Experiment_E1", "expD", m_e1) # E1 has same architecture & inputs as D
    ]

    # Evaluate on Internal Validation Set
    val_results = {}
    for name, mtype, model in models:
        res = evaluate_on_val_set(model, mtype, val_df)
        val_results[name] = res

    # Save summary json
    with open(AUDIT_DIR_E1 / "val_evaluation_comparison_B_C_D_E1.json", "w") as f:
        json.dump(val_results, f, indent=2)

    print("\n" + "="*80)
    print("INTERNAL VALIDATION COMPARISON (EXPERIMENT B, C, D, E1)")
    print("="*80)
    summary_rows = []
    for name in ["Experiment_B", "Experiment_C", "Experiment_D", "Experiment_E1"]:
        r = val_results[name]
        summary_rows.append({
            "Model": name,
            "Accuracy": r["pixel_accuracy"],
            "mIoU": r["global_miou"],
            "BF1": r["boundary_f1"],
            "Non-slum IoU": r["per_class"]["Non-slum"]["iou"],
            "Road IoU": r["per_class"]["Road"]["iou"],
            "Veg IoU": r["per_class"]["Vegetation"]["iou"],
            "OpenLand IoU": r["per_class"]["Open Land"]["iou"],
            "Road Pred %": round(r["pred_class_pcts"][2], 2)
        })
    df_val_summary = pd.DataFrame(summary_rows)
    print(df_val_summary.to_string(index=False))

    # Evaluate specifically on the 3 focus patches:
    # 1. vashi_0041_Sector_4_r1024_c1792 (Held-out diagnostic patch)
    # 2. vashi_0020_Sector_5_r0512_c1536 (Training patch)
    # 3. vashi_0003_Sector_5_r0000_c1536 (Validation patch)
    target_pids = [
        ("vashi_0041_Sector_4_r1024_c1792", "Held-out Diagnostic Target"),
        ("vashi_0020_Sector_5_r0512_c1536", "Sector 5 Training Patch"),
        ("vashi_0003_Sector_5_r0000_c1536", "Sector 5 Validation Patch")
    ]

    patch_eval_results = {}
    for pid, desc in target_pids:
        row = df[df["patch_id"] == pid].iloc[0]
        gt = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        valid = (gt != 255)
        total_valid = int(valid.sum())

        bgr = cv2.imread(row["img_input_path_rgb"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        preds = {}
        for name, mtype, model in models:
            p, _ = predict_patch(model, mtype, row)
            preds[name] = p
            # Save prediction PNG
            out_pred_p = PRED_DIR_E1 / f"{name}_{pid}_pred5class.png"
            cv2.imwrite(str(out_pred_p), p.astype(np.uint8))

        # Metrics on this patch
        p_stats = {}
        for name in ["Experiment_B", "Experiment_C", "Experiment_D", "Experiment_E1"]:
            p = preds[name]
            acc = float(((p == gt) & valid).sum() / total_valid * 100.0)
            ious = {}
            precs = {}
            recs = {}
            pred_cnts = {}
            for c in range(1, 5):
                cname = CLASS_NAMES[c]
                tp = int(((p == c) & (gt == c) & valid).sum())
                fp = int(((p == c) & (gt != c) & valid).sum())
                fn = int(((p != c) & (gt == c) & valid).sum())
                ious[cname] = round(tp / max(1, tp + fp + fn) * 100.0, 2)
                precs[cname] = round(tp / max(1, tp + fp) * 100.0, 2)
                recs[cname] = round(tp / max(1, tp + fn) * 100.0, 2)
                pred_cnts[cname] = int(((p == c) & valid).sum())

            # Specific confusion to road
            ns_to_road = int(((gt == 1) & (p == 2) & valid).sum())
            ol_to_road = int(((gt == 4) & (p == 2) & valid).sum())

            p_stats[name] = {
                "acc": round(acc, 2),
                "miou": round(float(np.mean(list(ious.values()))), 2),
                "ious": ious,
                "precisions": precs,
                "recalls": recs,
                "pred_road_pct": round(pred_cnts["Road"] / total_valid * 100.0, 2),
                "pred_road_cnt": pred_cnts["Road"],
                "ns_to_road": ns_to_road,
                "ol_to_road": ol_to_road
            }
        patch_eval_results[pid] = p_stats

        # Generate 7-Panel Figure for this patch
        # Panels: 1. RGB, 2. GT, 3. Exp B, 4. Exp C, 5. Exp D, 6. Exp E1, 7. Error Maps (Exp D vs Exp E1)
        fig = plt.figure(figsize=(24, 7))
        gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1])

        # 1. RGB
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(rgb)
        ax1.set_title(f"1. Satellite RGB\n{pid[:10]} ({desc})", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # 2. GT
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(colorize(gt))
        gt_road_pct = ((gt == 2) & valid).sum() / total_valid * 100.0
        ax2.set_title(f"2. Ground Truth\n(Road: {gt_road_pct:.1f}%)", fontsize=10, fontweight="bold")
        ax2.axis("off")

        # 3. Exp B
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(colorize(preds["Experiment_B"]))
        ax3.set_title(f"3. Exp B (No NIR, No clDice)\nAcc: {p_stats['Experiment_B']['acc']}% | Rd: {p_stats['Experiment_B']['pred_road_pct']}%", fontsize=10, fontweight="bold")
        ax3.axis("off")

        # 4. Exp C
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(colorize(preds["Experiment_C"]))
        ax4.set_title(f"4. Exp C (+ NIR, No clDice)\nAcc: {p_stats['Experiment_C']['acc']}% | Rd: {p_stats['Experiment_C']['pred_road_pct']}%", fontsize=10, fontweight="bold")
        ax4.axis("off")

        # 5. Exp D
        ax5 = fig.add_subplot(gs[4])
        ax5.imshow(colorize(preds["Experiment_D"]))
        ax5.set_title(f"5. Exp D (+ clDice + Skip)\nAcc: {p_stats['Experiment_D']['acc']}% | Rd: {p_stats['Experiment_D']['pred_road_pct']}%", fontsize=10, fontweight="bold", color="darkred")
        ax5.axis("off")

        # 6. Exp E1
        ax6 = fig.add_subplot(gs[5])
        ax6.imshow(colorize(preds["Experiment_E1"]))
        ax6.set_title(f"6. Exp E1 (Remove clDice)\nAcc: {p_stats['Experiment_E1']['acc']}% | Rd: {p_stats['Experiment_E1']['pred_road_pct']}%", fontsize=10, fontweight="bold", color="navy")
        ax6.axis("off")

        # 7. Comparison Error Map: Exp D errors vs Exp E1 errors
        # Color coding: Green = Both Correct, Blue = Exp E1 Correct (D Failed), Orange = Exp D Correct (E1 Failed), Red = Both Wrong
        ax7 = fig.add_subplot(gs[6])
        comp_map = np.zeros((512, 512, 3), dtype=np.uint8)
        d_corr = (preds["Experiment_D"] == gt) & valid
        e1_corr = (preds["Experiment_E1"] == gt) & valid
        
        comp_map[d_corr & e1_corr] = (60, 180, 60)         # Green: Both correct
        comp_map[~d_corr & ~e1_corr & valid] = (180, 40, 40) # Red: Both wrong
        comp_map[~d_corr & e1_corr] = (30, 140, 240)       # Blue: E1 ONLY correct (D regressed)
        comp_map[d_corr & ~e1_corr] = (240, 140, 20)       # Orange: D ONLY correct
        comp_map[~valid] = (40, 40, 40)
        
        e1_only_cnt = int((~d_corr & e1_corr).sum())
        d_only_cnt = int((d_corr & ~e1_corr).sum())
        net_delta = e1_only_cnt - d_only_cnt

        ax7.imshow(comp_map)
        ax7.set_title(f"7. E1 vs D Difference\nBlue: E1 Only (+{e1_only_cnt:,})\nOrange: D Only (+{d_only_cnt:,})\nNet: {net_delta:+,} px", fontsize=10, fontweight="bold")
        ax7.axis("off")

        # Legend
        legend_patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[1])/255., label="Non-slum"),
            mpatches.Patch(color=np.array(CLASS_COLORS[2])/255., label="Road"),
            mpatches.Patch(color=np.array(CLASS_COLORS[3])/255., label="Vegetation"),
            mpatches.Patch(color=np.array(CLASS_COLORS[4])/255., label="Open Land")
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=11)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig_path = VIS_DIR_E1 / f"7panel_{pid[:10]}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Generated 7-panel visualization: {fig_path}")

    with open(AUDIT_DIR_E1 / "target_patches_detailed_metrics.json", "w") as f:
        json.dump(patch_eval_results, f, indent=2)

    print("\n" + "="*80)
    print("TARGET PATCHES COMPARISON SUMMARY:")
    print("="*80)
    for pid, desc in target_pids:
        print(f"\n--- {pid} ({desc}) ---")
        p_stats = patch_eval_results[pid]
        header = f"{'Model':<15} | {'Acc (%)':<8} | {'mIoU (%)':<9} | {'Road %':<8} | {'NS IoU':<8} | {'Road IoU':<9} | {'Veg IoU':<8} | {'OL IoU':<8} | {'NS->Road':<9} | {'OL->Road':<9}"
        print(header)
        print("-" * len(header))
        for m in ["Experiment_B", "Experiment_C", "Experiment_D", "Experiment_E1"]:
            s = p_stats[m]
            print(f"{m:<15} | {s['acc']:<8.2f} | {s['miou']:<9.2f} | {s['pred_road_pct']:<7.2f}% | {s['ious']['Non-slum']:<8.2f} | {s['ious']['Road']:<9.2f} | {s['ious']['Vegetation']:<8.2f} | {s['ious']['Open Land']:<8.2f} | {s['ns_to_road']:<9,d} | {s['ol_to_road']:<9,d}")

if __name__ == "__main__":
    main()

