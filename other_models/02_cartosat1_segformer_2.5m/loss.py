"""loss.py
-------
Compound Road-Prioritized Multi-Task Loss formulation for SegFormer:
1. Lovász-Softmax Loss (Direct continuous surrogate for Jaccard / IoU index)
2. Class-Weighted Focal Loss (gamma=2.0, w_road=3.5 to stop built-up confusion)
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
            continue
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
    """Lovász-Softmax Loss directly maximizing multi-class Intersection-over-Union (IoU)."""

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
        gx = F.conv2d(road_mask_binary, self.sobel_x.to(road_mask_binary.device), padding=1)
        gy = F.conv2d(road_mask_binary, self.sobel_y.to(road_mask_binary.device), padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return (mag > 0.5).float()


class CompoundRoadPrioritizedLoss(nn.Module):
    """Multi-Task Compound Loss combining Focal, Dice, Lovász-Softmax, and Auxiliary Edge Supervision."""

    def __init__(
        self,
        num_classes: int = 5,
        road_weight: float = 3.5,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        lovasz_weight: float = 0.75,
        edge_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Class weights: [0: Water, 1: Veg, 2: Barren, 3: Built-up, 4: Roads]
        weights = torch.tensor([1.0, 1.0, 1.4, 1.0, road_weight], dtype=torch.float32)
        self.register_buffer("class_weights", weights)

        self.focal = FocalLoss(alpha=weights, gamma=gamma)
        self.dice = MultiClassDiceLoss(num_classes=num_classes, class_weights=weights)
        self.lovasz = LovaszSoftmaxLoss(num_classes=num_classes, weights=weights)
        self.edge_extractor = RoadEdgeExtractor()
        self.edge_criterion = nn.BCEWithLogitsLoss()

        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.edge_weight = edge_weight

    def forward(
        self,
        seg_logits: torch.Tensor,
        targets: torch.Tensor,
        edge_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_focal = self.focal(seg_logits, targets)
        loss_dice = self.dice(seg_logits, targets)
        loss_lovasz = self.lovasz(seg_logits, targets)

        total_loss = loss_focal + self.dice_weight * loss_dice + self.lovasz_weight * loss_lovasz

        if edge_logits is not None:
            road_gt = (targets == 4).float().unsqueeze(1)
            edge_gt = self.edge_extractor(road_gt)
            loss_edge = self.edge_criterion(edge_logits, edge_gt)
            total_loss = total_loss + self.edge_weight * loss_edge

        return total_loss
