"""evaluate.py
-----------
Computes comprehensive evaluation metrics (mIoU, per-class IoU, F1 score, Accuracy)
and generates visual prediction maps comparing Ground Truth vs. SegFormer Predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("Evaluator")

CLASS_NAMES = ["Water", "Vegetation", "Barren", "Built-up", "Roads"]
CLASS_COLORS = {
    0: (30, 120, 220),   # Water: Blue
    1: (34, 139, 34),    # Vegetation: Green
    2: (210, 180, 140),  # Barren: Tan
    3: (220, 40, 40),    # Built-up: Red
    4: (255, 220, 0),    # Roads: Yellow
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 5) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_visuals: bool = True,
    visuals_dir: Path | None = None,
    max_visuals: int = 6,
) -> dict[str, float]:
    model.eval()
    num_classes = 5
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    if save_visuals and visuals_dir is not None:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals_saved = 0
    visual_candidates = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            patch_ids = batch["patch_id"]

            logits = model(pixel_values)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            imgs_np = pixel_values.cpu().numpy()

            for b in range(len(preds_np)):
                cm = compute_confusion_matrix(preds_np[b], labels_np[b], num_classes)
                total_conf_matrix += cm

                if save_visuals and visuals_dir is not None:
                    p_id = patch_ids[b]
                    road_px = int(np.count_nonzero(labels_np[b] == 4))
                    builtup_px = int(np.count_nonzero(labels_np[b] == 3))
                    
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    raw_img = (imgs_np[b] * std + mean).transpose(1, 2, 0)
                    raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)

                    visual_candidates.append({
                        "patch_id": p_id,
                        "raw_img": raw_img,
                        "gt_mask": labels_np[b],
                        "pred_mask": preds_np[b],
                        "road_px": road_px,
                        "builtup_px": builtup_px,
                    })

        # Render and save the most representative visual samples
        if save_visuals and visuals_dir is not None and len(visual_candidates) > 0:
            # Prioritize patches with verified road networks so inspection is informative
            with_roads = [c for c in visual_candidates if c["road_px"] >= 1500]
            with_roads.sort(key=lambda x: x["road_px"], reverse=True)
            others = [c for c in visual_candidates if c not in with_roads]
            
            # Select up to 4 dense road patches + 2 mixed urban patches
            selected = (with_roads[:4] + others[:2])[:max_visuals]
            if len(selected) < max_visuals:
                selected = visual_candidates[:max_visuals]

            for item in selected:
                p_id = item["patch_id"]
                raw_img = item["raw_img"]
                gt_rgb = mask_to_rgb(item["gt_mask"])
                pred_rgb = mask_to_rgb(item["pred_mask"])
                overlay = cv2.addWeighted(raw_img, 0.55, pred_rgb, 0.45, 0)

                # Compute patch road IoU
                gt_r = (item["gt_mask"] == 4)
                pr_r = (item["pred_mask"] == 4)
                p_inter = np.logical_and(gt_r, pr_r).sum()
                p_union = np.logical_or(gt_r, pr_r).sum()
                p_road_iou = (p_inter / (p_union + 1e-8)) * 100

                fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=140)
                axes[0].imshow(raw_img[:, :, 0], cmap="gray")
                axes[0].set_title(f"Satellite ({p_id})", fontsize=11, fontweight="bold")
                axes[0].axis("off")

                axes[1].imshow(gt_rgb)
                axes[1].set_title(f"Ground Truth (Road={item['road_px']}px)", fontsize=11, fontweight="bold")
                axes[1].axis("off")

                axes[2].imshow(pred_rgb)
                axes[2].set_title(f"SegFormer (Road IoU: {p_road_iou:.1f}%)", fontsize=11, fontweight="bold")
                axes[2].axis("off")

                axes[3].imshow(overlay)
                axes[3].set_title("Prediction Overlay", fontsize=11, fontweight="bold")
                axes[3].axis("off")

                handles = [
                    plt.Line2D([0], [0], marker="s", color="w", label=name,
                               markerfacecolor=np.array(col)/255.0, markersize=12)
                    for cid, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS.values()))
                ]
                fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True, fontsize=10)
                plt.tight_layout(rect=[0, 0.08, 1, 1])

                out_fig = visuals_dir / f"eval_pred_{p_id}.png"
                plt.savefig(out_fig, bbox_inches="tight")
                plt.close()

    # Compute metrics from confusion matrix
    ious = []
    f1s = []
    metrics = {}

    for c in range(num_classes):
        tp = total_conf_matrix[c, c]
        fp = np.sum(total_conf_matrix[:, c]) - tp
        fn = np.sum(total_conf_matrix[c, :]) - tp

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        ious.append(iou)
        f1s.append(f1)
        cname = CLASS_NAMES[c].lower().replace(" ", "_")
        metrics[f"iou_{cname}"] = float(iou)
        metrics[f"f1_{cname}"] = float(f1)

    metrics["mIoU"] = float(np.mean(ious))
    metrics["mF1"] = float(np.mean(f1s))
    metrics["overall_accuracy"] = float(np.trace(total_conf_matrix) / (np.sum(total_conf_matrix) + 1e-8))

    return metrics
