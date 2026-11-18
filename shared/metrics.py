"""
shared/metrics.py
Standard remote sensing segmentation evaluation metrics:
1. Strict 0m IoU (pixel-level)
2. SpaceNet 5m buffer tolerance IoU
3. Volodymyr Mnih relaxed precision, recall, and F1
"""

import numpy as np
import cv2

def compute_strict_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes exact pixel-level Intersection-over-Union."""
    intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def compute_spacenet_buffer_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, buffer_pixels: int = 8) -> float:
    """
    Computes buffer tolerance IoU according to SpaceNet protocol.
    At 0.60m GSD, a 5-meter buffer corresponds to an 8-pixel dilation radius.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_pixels + 1, 2 * buffer_pixels + 1))
    buffered_gt = cv2.dilate((gt_mask > 0).astype(np.uint8), kernel)
    buffered_pred = cv2.dilate((pred_mask > 0).astype(np.uint8), kernel)
    
    tp_pred = np.logical_and(pred_mask > 0, buffered_gt > 0).sum()
    tp_gt = np.logical_and(gt_mask > 0, buffered_pred > 0).sum()
    tp = 0.5 * (tp_pred + tp_gt)
    
    fn = np.logical_and(gt_mask > 0, buffered_pred == 0).sum()
    fp = np.logical_and(pred_mask > 0, buffered_gt == 0).sum()
    
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return float(tp) / float(denom)

def compute_mnih_relaxed_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, slack_pixels: int = 5):
    """
    Computes Volodymyr Mnih's relaxed precision, recall, and F1 score
    with a slack buffer of slack_pixels (default 5 pixels ~ 3m at 0.6m GSD).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack_pixels + 1, 2 * slack_pixels + 1))
    buffered_gt = cv2.dilate((gt_mask > 0).astype(np.uint8), kernel)
    buffered_pred = cv2.dilate((pred_mask > 0).astype(np.uint8), kernel)
    
    pred_pos = (pred_mask > 0).sum()
    gt_pos = (gt_mask > 0).sum()
    
    tp_prec = np.logical_and(pred_mask > 0, buffered_gt > 0).sum()
    tp_rec = np.logical_and(gt_mask > 0, buffered_pred > 0).sum()
    
    prec = float(tp_prec) / float(pred_pos) if pred_pos > 0 else 1.0
    rec = float(tp_rec) / float(gt_pos) if gt_pos > 0 else 1.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {"precision": prec, "recall": rec, "f1": f1}
