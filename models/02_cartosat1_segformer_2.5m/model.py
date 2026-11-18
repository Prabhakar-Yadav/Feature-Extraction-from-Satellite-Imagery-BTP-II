"""model.py
--------
Pretrained SegFormer-B2 (Vision Transformer) for 5-Class Urban Feature Extraction.
Enhanced with:
1. Multi-Scale Hierarchical Transformer Backbone (MiT-B2, Cityscapes pretrained).
2. High-Resolution Spatial and Channel Attention (scSE) Edge Refinement Stem.
3. Native 512x512 Linear Boundary Delineator to prevent thin road corridors from blurring.
4. Auxiliary Road Boundary Supervision Head.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger("SegFormerModel")


class SCSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


class RoadEnhancedSegFormer(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_name = pretrained_model_name

        logger.info(f"Loading pretrained SegFormer backbone from: {pretrained_model_name}")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # High-resolution edge refinement stem (processes native 512x512 gradients to sharpen narrow road borders)
        self.edge_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SCSEBlock(32),
        )

        # Fusion & Refinement Head with scSE Attention
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes + 32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SCSEBlock(48),
            nn.Conv2d(48, num_classes, kernel_size=1),
        )

        # Auxiliary Road Edge Head
        self.edge_head = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        b, c, h, w = pixel_values.shape
        outputs = self.segformer(pixel_values=pixel_values)
        logits_low = outputs.logits  # shape (B, num_classes, H/4, W/4)

        # Upsample coarse Transformer logits to native resolution (H, W)
        logits_up = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)

        # High-resolution edge feature extraction
        edge_feats = self.edge_stem(pixel_values)

        # Concatenate and pass through attention refinement
        cat_feats = torch.cat([logits_up, edge_feats], dim=1)
        feat = self.fusion[0:4](cat_feats) # up through scSE
        residual = self.fusion[4](feat)    # final 1x1 conv -> num_classes

        refined_logits = logits_up + residual
        edge_logits = self.edge_head(feat)

        if self.training:
            return refined_logits, edge_logits
        return refined_logits


def build_model(num_classes: int = 5) -> RoadEnhancedSegFormer:
    return RoadEnhancedSegFormer(num_classes=num_classes)
