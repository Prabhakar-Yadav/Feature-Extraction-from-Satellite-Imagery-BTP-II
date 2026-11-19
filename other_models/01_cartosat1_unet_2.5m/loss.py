"""loss.py
-------
Multi-Task Road-Disentangled Compound Loss formulation:
1. Lovász-Softmax Loss (Direct continuous surrogate for Jaccard / IoU index)
2. Class-Weighted Focal Loss (gamma=2.0, w_road=4.0 to stop built-up confusion)
3. Multi-Class Dice Loss (optimizing thin boundary overlap)
4. Auxiliary Road Edge Sharpness Loss (BCE on road boundary gradients)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: list[int], weights: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-class Lovász-Softmax loss on flattened probabilities (N, C) and labels (N)."""
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []
    class_to_weight = {}
    if weights is not None:
        for c in range(C):
            class_to_weight[c] = weights[c]

    for c in classes:
        target_c = (labels == c).float()
        if target_c.sum() == 0 and (probas[:, c] < 0.1).all():
            continue  # Class not present in this batch, skip to save compute
        errors = (target_c - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = target_c[perm]
        grad = lovasz_grad(gt_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        w = class_to_weight.get(c, 1.0) if weights is not None else 1.0
        losses.append(loss_c * w)

    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax Loss directly maximizing multi-class Intersection-over-Union (IoU) at native resolution."""

    def __init__(self, num_classes: int = 5, weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        if H > 256 or W > 256:
            probas = F.softmax(logits, dim=1)
            probas = F.interpolate(probas, scale_factor=0.5, mode="area")
            targets_down = F.interpolate(targets.unsqueeze(1).float(), scale_factor=0.5, mode="nearest").squeeze(1).long()
        else:
            probas = F.softmax(logits, dim=1)
            targets_down = targets

        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets_down.view(-1)
        classes = list(range(C))
        return lovasz_softmax_flat(probas_flat, targets_flat, classes, self.weights)


class RoadPrecisionTverskyLoss(nn.Module):
    """Tversky Loss on road class with alpha > beta to aggressively penalize False Positives.
    Forces predicted road corridors to be thin, crisp, and tightly bounded, eliminating spread and bloat."""

    def __init__(self, alpha: float = 0.75, beta: float = 0.25, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha  # Heavy penalty for False Positives (over-prediction)
        self.beta = beta    # Penalty for False Negatives
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)[:, 4]  # Road class probability (B, H, W)
        target_road = (targets == 4).float()    # Ground truth road (B, H, W)

        tp = (probs * target_road).sum(dim=(1, 2))
        fp = (probs * (1.0 - target_road)).sum(dim=(1, 2))
        fn = ((1.0 - probs) * target_road).sum(dim=(1, 2))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 5, smooth: float = 1e-6, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.class_weights is not None:
            weights = self.class_weights.to(dice_loss.device)
            return torch.sum(dice_loss * weights) / torch.sum(weights)

        return torch.mean(dice_loss)


class RoadEdgeExtractor(nn.Module):
    """Computes Sobel gradient edges from ground-truth road masks on GPU."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, road_mask_binary: torch.Tensor) -> torch.Tensor:
        # road_mask_binary: (B, 1, H, W) float
        gx = F.conv2d(road_mask_binary, self.sobel_x.to(road_mask_binary.device), padding=1)
        gy = F.conv2d(road_mask_binary, self.sobel_y.to(road_mask_binary.device), padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        # Normalize to binary edge [0, 1]
        return (mag > 0.5).float()


class CompoundRoadPrioritizedLoss(nn.Module):
    """Multi-Task Compound Loss combining Focal, Dice, Lovász-Softmax, Precision Tversky,
    and Weighted Auxiliary Edge Boundary Supervision."""

    def __init__(
        self,
        num_classes: int = 5,
        road_weight: float = 1.8,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        tversky_weight: float = 1.0,
        edge_weight: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Balanced class weights: [0: Water, 1: Veg, 2: Barren, 3: Built-up, 4: Roads]
        weights = torch.tensor([1.0, 1.0, 1.3, 1.2, road_weight], dtype=torch.float32)
        self.register_buffer("class_weights", weights)

        self.focal = FocalLoss(alpha=weights, gamma=gamma)
        self.dice = MultiClassDiceLoss(num_classes=num_classes, class_weights=weights)
        self.lovasz = LovaszSoftmaxLoss(num_classes=num_classes, weights=weights)
        self.road_tversky = RoadPrecisionTverskyLoss(alpha=0.75, beta=0.25)
        self.edge_extractor = RoadEdgeExtractor()

        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.tversky_weight = tversky_weight
        self.edge_weight = edge_weight

    def forward(
        self,
        seg_logits: torch.Tensor,
        targets: torch.Tensor,
        edge_logits: torch.Tensor | None = None,
        aux1: torch.Tensor | None = None,
        aux2: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_focal = self.focal(seg_logits, targets)
        loss_dice = self.dice(seg_logits, targets)
        loss_lovasz = self.lovasz(seg_logits, targets)
        loss_tversky = self.road_tversky(seg_logits, targets)

        total_loss = (
            loss_focal
            + self.dice_weight * loss_dice
            + self.lovasz_weight * loss_lovasz
            + self.tversky_weight * loss_tversky
        )

        if edge_logits is not None:
            road_gt = (targets == 4).float().unsqueeze(1)
            edge_gt = self.edge_extractor(road_gt)
            pos_weight = torch.tensor([10.0], device=edge_logits.device)
            loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_gt, pos_weight=pos_weight)
            total_loss = total_loss + self.edge_weight * loss_edge

        if aux1 is not None:
            loss_aux1 = self.focal(aux1, targets) + self.dice(aux1, targets)
            total_loss = total_loss + 0.3 * loss_aux1

        if aux2 is not None:
            loss_aux2 = self.focal(aux2, targets) + self.dice(aux2, targets)
            total_loss = total_loss + 0.2 * loss_aux2

        if pixel_values is not None:
            # Shadow-Water Disentanglement: penalize water predictions on high-gradient land pixels
            sobel_mag = pixel_values[:, 2]  # (B, H, W)
            water_prob = F.softmax(seg_logits, dim=1)[:, 0]  # Water class prob
            is_land = (targets > 0).float()
            high_gradient_land = (sobel_mag > 0.15).float() * is_land
            if high_gradient_land.sum() > 0:
                loss_shadow_water = (water_prob * high_gradient_land).sum() / (high_gradient_land.sum() + 1e-6)
                total_loss = total_loss + 1.0 * loss_shadow_water

        return total_loss

