"""
models_subclass.py
------------------
Vision Transformer and Attention U-Net Architectures for 3-Subclass Built-up Segmentation.
Classes:
0: Residential Built-up
1: Industrial Built-up
2: Slum / Informal Settlement
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger("SubclassModels")


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


class MultispectralSegFormerSubclass(nn.Module):
    """
    Hierarchical Vision Transformer adapted for 4-channel multispectral input (RGB + NIR)
    and 3 built-up urban subclasses.
    """
    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_classes: int = 3,
        in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        if in_channels != 3:
            old_conv = self.segformer.segformer.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            self.segformer.segformer.encoder.patch_embeddings[0].proj = new_conv

        self.edge_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SCSEBlock(32),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes + 32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SCSEBlock(48),
            nn.Conv2d(48, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        outputs = self.segformer(x)
        logits_coarse = outputs.logits
        logits_up = F.interpolate(logits_coarse, size=(h, w), mode="bilinear", align_corners=False)
        edge_feats = self.edge_stem(x)
        fused = torch.cat([logits_up, edge_feats], dim=1)
        refined_logits = self.fusion(fused) + logits_up
        return refined_logits
