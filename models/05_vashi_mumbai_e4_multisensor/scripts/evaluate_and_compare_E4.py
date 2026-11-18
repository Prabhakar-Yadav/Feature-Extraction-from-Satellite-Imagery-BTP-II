"""
evaluate_and_compare_E4.py
--------------------------
Evaluates Experiment E4 (NDVI + Calibrated Focal-Dice Loss) on:
1. Internal Validation Set (14 patches)
2. Compares directly against Reference Model E1 and Model E3
3. Generates 8-Panel Visual Comparison Figures for:
   - vashi_0003_Sector_5_r0000_c1536 (Validation patch: Dense Residential Buildings)
   - vashi_0020_Sector_5_r0512_c1536 (Training patch: Dense Urban Grid)
   - vashi_0041_Sector_4_r1024_c1792 (Diagnostic target: High Road/Building Conflict)

8 Panels:
1. Satellite RGB
2. Ground Truth Mask
3. Experiment E1 Prediction
4. Experiment E4 Prediction
5. E1 Boundaries over RGB
6. E4 Boundaries over RGB
7. E4 Error Map (Green=Correct, Red=Incorrect)
8. Difference Map E4 vs E1 (Blue=E4 Only OK, Orange=E1 Only OK, Green=Both OK, Red=Both Wrong)
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
EXPE3_DIR = BASE_DIR / "research_corrected_v4" / "experiment_E3"
EXPE4_DIR = BASE_DIR / "research_corrected_v4" / "experiment_E4"

CKPT_DIR_E4 = EXPE4_DIR / "checkpoints"
AUDIT_DIR_E4 = EXPE4_DIR / "audits"
PRED_DIR_E4 = EXPE4_DIR / "predictions"
VIS_DIR_E4 = EXPE4_DIR / "visualizations"

for d in [CKPT_DIR_E4, AUDIT_DIR_E4, PRED_DIR_E4, VIS_DIR_E4]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "scripts"))
sys.path.append(str(EXP2_DIR / "scripts"))
sys.path.append(str(EXPE1_DIR / "scripts"))
sys.path.append(str(EXPE3_DIR / "scripts"))
sys.path.append(str(EXPE4_DIR / "scripts"))

from train_experiment_D import HighResSensorAdaptiveUNet
from train_experiment_E4 import HighResNDVIAdaptiveUNet
from evaluate_experiment_E1 import compute_boundary_f1, colorize, CLASS_NAMES, CLASS_COLORS, MEAN_RGB, STD_RGB, VASHI_MEANS, VASHI_STDS, MUMBAI_MEANS, MUMBAI_STDS

def load_input_e4(row):
    src = row["dataset_source"]
    bgr = cv2.imread(row["img_input_path_rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (512, 512):
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

    if src == "vashi_spot":
        data = np.load(row["img_input_path_4band"]).astype(np.float32)
        data = np.where(data == 65535, VASHI_MEANS, data)
        norm4 = (data - VASHI_MEANS) / VASHI_STDS
        red = data[0]
        nir = data[3]
        ndvi = (nir - red) / (nir + red + 1e-5)
        ndvi = np.clip(ndvi, -1.0, 1.0)[None, :, :]
        data5 = np.concatenate([norm4, ndvi], axis=0)
        t_img = torch.from_numpy(data5).float()
    else:
        norm3 = (rgb.astype(np.float32) / 255.0 - MUMBAI_MEANS) / MUMBAI_STDS
        t_img = torch.from_numpy(norm3.transpose(2, 0, 1)).float()
    return t_img, rgb

def predict_e4(model, row):
    src = row["dataset_source"]
    t_img, rgb = load_input_e4(row)
    t_img = t_img.to(device)

    with torch.no_grad():
        logits = model([t_img], [src])[0]
        if src == "vashi_spot":
            l_v = logits.clone()
            l_v[0] = -1e9
            pred = torch.argmax(l_v, dim=0).cpu().numpy()
        else:
            pred = torch.argmax(logits, dim=0).cpu().numpy()
    return pred, logits.cpu().numpy()

def evaluate_full_val_e4(model, val_df):
    model.eval()
    cm_v = np.zeros((5, 5), dtype=np.int64)
    m_tp, m_fn = 0, 0
    bf1_scores = []

    for _, row in val_df.iterrows():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        if mask.shape[:2] != (512, 512):
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        pred, _ = predict_e4(model, row)
        valid = (mask != 255)
        if not valid.any(): continue

        src = row["dataset_source"]
        if src == "vashi_spot":
            for tc in range(1, 5):
                mt = (mask == tc) & valid
                if mt.any():
                    for pc in range(5):
                        cm_v[tc, pc] += int((pred[mt] == pc).sum())
            bf1_scores.append(compute_boundary_f1(pred, mask))

        elif src == "mumbai_shapefile":
            is_slum = (mask == 0) & valid
            m_tp += int(((pred == 0) & is_slum).sum())
            m_fn += int(((pred != 0) & is_slum).sum())

    per_class = {}
    ious = []
    tot_valid = int(cm_v[1:5, :].sum())
    tot_corr = int(sum(cm_v[c, c] for c in range(1, 5)))
    acc = tot_corr / max(1, tot_valid) * 100.0

    for c in range(1, 5):
        cname = CLASS_NAMES[c]
        tp = int(cm_v[c, c])
        fp = int(cm_v[:, c].sum() - tp)
        fn = int(cm_v[c, :].sum() - tp)
        denom = tp + fp + fn
        iou = tp / max(1, denom) * 100.0
        prec = tp / max(1, tp + fp) * 100.0
        rec = tp / max(1, tp + fn) * 100.0
        ious.append(iou)
        per_class[cname] = {
            "iou": round(iou, 2), "precision": round(prec, 2), "recall": round(rec, 2)
        }

    return {
        "accuracy": round(acc, 2),
        "miou": round(float(np.mean(ious)), 2),
        "bf1": round(float(np.mean(bf1_scores)), 2),
        "mumbai_det": round(float(m_tp / max(1, m_tp + m_fn) * 100.0), 2),
        "per_class": per_class,
        "confusion_matrix": cm_v.tolist(),
        "road_pred_pct": round(float(cm_v[1:5, 2].sum() / max(1, tot_valid) * 100.0), 2)
    }

def get_boundaries(mask):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = np.zeros_like(mask, dtype=np.uint8)
    for c in range(1, 5):
        c_bin = (mask == c).astype(np.uint8)
        if c_bin.sum() == 0: continue
        dil = cv2.dilate(c_bin, k)
        b = dil - c_bin
        boundary[b > 0] = c
    return boundary

def overlay_boundaries_on_rgb(rgb_img, boundary_mask):
    overlay = rgb_img.copy()
    for cid, col in CLASS_COLORS.items():
        if cid in [0, 255]: continue
        overlay[boundary_mask == cid] = col
    return overlay

def main():
    manifest_p = BASE_DIR / "research_corrected_v4" / "audits" / "v4_dataset_manifest.csv"
    df = pd.read_csv(manifest_p)
    val_df = df[df["current_stage_in_v4"] == "val"].copy().reset_index(drop=True)

    print("="*80)
    print("EVALUATING EXPERIMENT E4 (NDVI + CALIBRATED FOCAL-DICE LOSS)")
    print("="*80)

    # 1. Load E1
    from evaluate_and_compare_E3 import predict_single
    m_e1 = HighResSensorAdaptiveUNet(5).to(device)
    ckpt_e1 = torch.load(EXPE1_DIR / "checkpoints" / "best_expE1_model.pth", map_location=device, weights_only=False)
    m_e1.load_state_dict(ckpt_e1["model_state_dict"])
    m_e1.eval()

    # 2. Load E4
    m_e4 = HighResNDVIAdaptiveUNet(5).to(device)
    ckpt_e4 = torch.load(CKPT_DIR_E4 / "best_expE4_model.pth", map_location=device, weights_only=False)
    m_e4.load_state_dict(ckpt_e4["model_state_dict"])
    m_e4.eval()

    from evaluate_and_compare_E3 import evaluate_full_val
    res_e1 = evaluate_full_val(m_e1, val_df)
    res_e4 = evaluate_full_val_e4(m_e4, val_df)

    # Load E3 results
    with open(EXPE3_DIR / "audits" / "val_evaluation_E1_E2_E3.json") as f:
        e3_data = json.load(f)
        res_e3 = e3_data["Experiment_E3"]

    print("\n" + "="*80)
    print("INTERNAL VALIDATION COMPARISON: E1 vs E3 vs E4")
    print("="*80)
    summary_rows = [
        {"Model": "Experiment_E1 (Reference)", "Accuracy": res_e1["accuracy"], "mIoU": res_e1["miou"], "BF1": res_e1["bf1"], "Non-slum IoU": res_e1["per_class"]["Non-slum"]["iou"], "Road IoU": res_e1["per_class"]["Road"]["iou"], "Veg IoU": res_e1["per_class"]["Vegetation"]["iou"], "OpenLand IoU": res_e1["per_class"]["Open Land"]["iou"], "Road Pred %": res_e1["road_pred_pct"], "Mumbai Det %": res_e1["mumbai_det"]},
        {"Model": "Experiment_E3 (Boundary Gate)", "Accuracy": res_e3["accuracy"], "mIoU": res_e3["miou"], "BF1": res_e3["bf1"], "Non-slum IoU": res_e3["per_class"]["Non-slum"]["iou"], "Road IoU": res_e3["per_class"]["Road"]["iou"], "Veg IoU": res_e3["per_class"]["Vegetation"]["iou"], "OpenLand IoU": res_e3["per_class"]["Open Land"]["iou"], "Road Pred %": res_e3["road_pred_pct"], "Mumbai Det %": res_e3["mumbai_det"]},
        {"Model": "Experiment_E4 (NDVI + FocalDice)", "Accuracy": res_e4["accuracy"], "mIoU": res_e4["miou"], "BF1": res_e4["bf1"], "Non-slum IoU": res_e4["per_class"]["Non-slum"]["iou"], "Road IoU": res_e4["per_class"]["Road"]["iou"], "Veg IoU": res_e4["per_class"]["Vegetation"]["iou"], "OpenLand IoU": res_e4["per_class"]["Open Land"]["iou"], "Road Pred %": res_e4["road_pred_pct"], "Mumbai Det %": res_e4["mumbai_det"]}
    ]
    df_eval = pd.DataFrame(summary_rows)
    print(df_eval.to_string(index=False))

    with open(AUDIT_DIR_E4 / "val_evaluation_E1_E3_E4.json", "w") as f:
        json.dump({"Experiment_E1": res_e1, "Experiment_E3": res_e3, "Experiment_E4": res_e4}, f, indent=2)

    # Generate 8-Panel Comparisons for Representative Patches
    target_patches = [
        ("vashi_0003_Sector_5_r0000_c1536", "Validation: Dense Residential Buildings"),
        ("vashi_0020_Sector_5_r0512_c1536", "Training: Complex Building-Road Grid"),
        ("vashi_0041_Sector_4_r1024_c1792", "Diagnostic: High Road/Building Conflict")
    ]

    for pid, desc in target_patches:
        row = df[df["patch_id"] == pid].iloc[0]
        gt = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        valid = (gt != 255)
        tot_valid = max(1, int(valid.sum()))

        bgr = cv2.imread(row["img_input_path_rgb"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        p_e1, _ = predict_single(m_e1, row)
        p_e4, _ = predict_e4(m_e4, row)

        # Save predictions
        cv2.imwrite(str(PRED_DIR_E4 / f"Experiment_E1_{pid}_pred5class.png"), p_e1.astype(np.uint8))
        cv2.imwrite(str(PRED_DIR_E4 / f"Experiment_E4_{pid}_pred5class.png"), p_e4.astype(np.uint8))

        acc_e1 = float(((p_e1 == gt) & valid).sum() / tot_valid * 100.0)
        acc_e4 = float(((p_e4 == gt) & valid).sum() / tot_valid * 100.0)
        rd_e1 = float(((p_e1 == 2) & valid).sum() / tot_valid * 100.0)
        rd_e4 = float(((p_e4 == 2) & valid).sum() / tot_valid * 100.0)

        b_e1 = get_boundaries(p_e1)
        b_e4 = get_boundaries(p_e4)
        ov_e1 = overlay_boundaries_on_rgb(rgb, b_e1)
        ov_e4 = overlay_boundaries_on_rgb(rgb, b_e4)

        err_e4 = np.zeros((512, 512, 3), dtype=np.uint8)
        err_e4[valid & (p_e4 == gt)] = (50, 180, 50)
        err_e4[valid & (p_e4 != gt)] = (220, 50, 50)
        err_e4[~valid] = (40, 40, 40)

        diff_map = np.zeros((512, 512, 3), dtype=np.uint8)
        e1_ok = (p_e1 == gt) & valid
        e4_ok = (p_e4 == gt) & valid
        diff_map[e1_ok & e4_ok] = (50, 180, 50)
        diff_map[~e1_ok & ~e4_ok & valid] = (160, 40, 40)
        diff_map[~e1_ok & e4_ok] = (30, 140, 240)       # Blue: E4 Only Correct
        diff_map[e1_ok & ~e4_ok] = (240, 140, 20)       # Orange: E1 Only Correct
        diff_map[~valid] = (40, 40, 40)

        e4_only_cnt = int((~e1_ok & e4_ok).sum())
        e1_only_cnt = int((e1_ok & ~e4_ok).sum())
        net_gain = e4_only_cnt - e1_only_cnt

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.suptitle(f"Phase 7 Visual Quality Evaluation: Experiment E4 vs Experiment E1\n{pid} ({desc})", fontsize=15, fontweight="bold")

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("1. Satellite RGB", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(colorize(gt))
        gt_rd = float(((gt == 2) & valid).sum() / tot_valid * 100.0)
        axes[0, 1].set_title(f"2. Ground Truth Mask\n(True Road: {gt_rd:.1f}%)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(colorize(p_e1))
        axes[0, 2].set_title(f"3. Exp E1 Prediction (Reference)\nAcc: {acc_e1:.2f}% | Road: {rd_e1:.1f}%", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(colorize(p_e4))
        axes[0, 3].set_title(f"4. Exp E4 Prediction (NDVI + FocalDice)\nAcc: {acc_e4:.2f}% | Road: {rd_e4:.1f}%", fontsize=11, fontweight="bold", color="darkgreen")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(ov_e1)
        axes[1, 0].set_title("5. E1 Boundaries over Satellite RGB", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(ov_e4)
        axes[1, 1].set_title("6. E4 Boundaries over Satellite RGB\n(Sharp Roof Corridors)", fontsize=11, fontweight="bold", color="darkgreen")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(err_e4)
        axes[1, 2].set_title(f"7. E4 Error Map\nGreen=OK, Red=Error ({tot_valid-int(e4_ok.sum()):,} px)", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(diff_map)
        axes[1, 3].set_title(f"8. Difference Map (E4 vs E1)\nBlue: E4 Only (+{e4_only_cnt:,})\nOrange: E1 Only (+{e1_only_cnt:,})\nNet Delta: {net_gain:+,} px", fontsize=11, fontweight="bold")
        axes[1, 3].axis("off")

        legend_patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[1])/255., label="Non-slum"),
            mpatches.Patch(color=np.array(CLASS_COLORS[2])/255., label="Road"),
            mpatches.Patch(color=np.array(CLASS_COLORS[3])/255., label="Vegetation"),
            mpatches.Patch(color=np.array(CLASS_COLORS[4])/255., label="Open Land")
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=11)

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        out_fig = VIS_DIR_E4 / f"8panel_comparison_{pid[:10]}.png"
        plt.savefig(out_fig, dpi=180)
        plt.close()
        print(f"Generated 8-panel comparison: {out_fig}")

if __name__ == "__main__":
    main()

