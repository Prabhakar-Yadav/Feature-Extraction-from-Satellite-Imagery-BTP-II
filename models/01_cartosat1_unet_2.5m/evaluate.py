"""evaluate.py
-----------
Computes comprehensive evaluation metrics (mIoU, per-class IoU, F1 score, Accuracy)
and generates visual prediction maps comparing Ground Truth vs. RoadResUNet Predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("UNetEvaluator")

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
    road_threshold: float = 0.50,
    compute_spacenet: bool = False,
) -> dict[str, float]:
    model.eval()
    num_classes = 5
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    if save_visuals and visuals_dir is not None:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    visuals_saved = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            patch_ids = batch["patch_id"]

            logits = model(pixel_values)
            if isinstance(logits, tuple):
                logits = logits[0]

            if road_threshold is not None and road_threshold > 0:
                probs = torch.softmax(logits, dim=1)
                road_prob = probs[:, 4]
                non_road_best = torch.argmax(probs[:, :4], dim=1)
                preds = torch.where(road_prob >= road_threshold, 4, non_road_best)
            else:
                preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            imgs_np = pixel_values.cpu().numpy()

            for b in range(len(preds_np)):
                pred_single = preds_np[b]
                # Filter inland tiny false water shadows (< 500 px)
                water_mask = (pred_single == 0).astype(np.uint8)
                if water_mask.sum() > 0:
                    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(water_mask, connectivity=8)
                    for lab in range(1, num_labels):
                        if stats[lab, cv2.CC_STAT_AREA] < 500:
                            pred_single[labels_im == lab] = 3  # revert shadow puddle to built-up

                cm = compute_confusion_matrix(pred_single, labels_np[b], num_classes)
                total_conf_matrix += cm

                p_id = patch_ids[b]
                should_save = save_visuals and visuals_dir is not None and (visuals_saved < max_visuals or p_id in ["patch_00055", "patch_00295"])
                if should_save:
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    raw_img = (imgs_np[b] * std + mean)[0]  # Take band 0 (panchromatic)
                    raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)

                    gt_rgb = mask_to_rgb(labels_np[b])
                    pred_rgb = mask_to_rgb(pred_single)

                    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
                    overlay = cv2.addWeighted(raw_rgb, 0.55, pred_rgb, 0.45, 0)

                    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=140)
                    axes[0].imshow(raw_img, cmap="gray")
                    axes[0].set_title(f"Satellite Panchromatic ({p_id})", fontsize=11, fontweight="bold")
                    axes[0].axis("off")

                    axes[1].imshow(gt_rgb)
                    axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
                    axes[1].axis("off")

                    axes[2].imshow(pred_rgb)
                    axes[2].set_title("UltraRoadUNet Prediction", fontsize=11, fontweight="bold")
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
                    visuals_saved += 1

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
        metrics[f"precision_{cname}"] = float(precision)
        metrics[f"recall_{cname}"] = float(recall)

    metrics["mIoU"] = float(np.mean(ious))
    metrics["mean_f1"] = float(np.mean(f1s))
    metrics["overall_accuracy"] = float(np.trace(total_conf_matrix) / (np.sum(total_conf_matrix) + 1e-8))

    # SpaceNet buffer-tolerant road evaluation across test set (optional for speed)
    if compute_spacenet:
        spacenet_metrics = compute_spacenet_road_metrics(model, dataloader, device, road_threshold=road_threshold)
        metrics.update(spacenet_metrics)

    return metrics


def compute_spacenet_road_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    road_threshold: float = 0.50,
) -> dict[str, float]:
    """Computes standard remote-sensing buffer-tolerant road metrics (SpaceNet / IEEE TGRS)."""
    model.eval()
    radii = [2, 3, 4]  # 2px (5.0m), 3px (7.5m), 4px (10.0m)
    kernels = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) for r in radii}

    stats = {r: {"tp_prec": 0, "tp_rec": 0, "total_pred": 0, "total_gt": 0} for r in radii}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values)
            if isinstance(logits, tuple):
                logits = logits[0]

            if road_threshold is not None and road_threshold > 0:
                probs = torch.softmax(logits, dim=1)
                road_prob = probs[:, 4]
                non_road = torch.argmax(probs[:, :4], dim=1)
                preds = torch.where(road_prob >= road_threshold, 4, non_road)
            else:
                preds = torch.argmax(logits, dim=1)

            preds_np = (preds == 4).cpu().numpy().astype(np.uint8)
            labels_np = (labels == 4).cpu().numpy().astype(np.uint8)

            for b in range(len(preds_np)):
                p = preds_np[b]
                g = labels_np[b]
                pred_cnt = int(p.sum())
                gt_cnt = int(g.sum())

                if pred_cnt == 0 and gt_cnt == 0:
                    continue

                for r in radii:
                    k = kernels[r]
                    g_dil = cv2.dilate(g, k)
                    p_dil = cv2.dilate(p, k)

                    tp_p = int((p & (g_dil > 0)).sum())
                    tp_r = int((g & (p_dil > 0)).sum())

                    stats[r]["tp_prec"] += tp_p
                    stats[r]["tp_rec"] += tp_r
                    stats[r]["total_pred"] += pred_cnt
                    stats[r]["total_gt"] += gt_cnt

    out = {}
    for r in radii:
        m_dist = r * 2.5
        tp_p = stats[r]["tp_prec"]
        tp_r = stats[r]["tp_rec"]
        tot_p = stats[r]["total_pred"]
        tot_g = stats[r]["total_gt"]

        prec = float(tp_p / (tot_p + 1e-8))
        rec = float(tp_r / (tot_g + 1e-8))
        f1 = float(2 * prec * rec / (prec + rec + 1e-8))
        iou = float(f1 / (2.0 - f1 + 1e-8)) if f1 > 0 else 0.0

        tag = f"spacenet_road_{m_dist:.1f}m"
        out[f"{tag}_precision"] = prec
        out[f"{tag}_recall"] = rec
        out[f"{tag}_f1"] = f1
        out[f"{tag}_iou"] = iou

    return out
