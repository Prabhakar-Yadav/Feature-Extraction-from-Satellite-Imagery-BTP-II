"""
models.py
---------
Architectures for 5-Class Multispectral (4-Band: R, G, B, NIR) Urban Semantic Segmentation.
1. MultispectralSegFormer: Hierarchical Vision Transformer with native 4-band patch projection
   and high-resolution scSE spatial/channel attention edge refinement stem.
2. MultispectralUltraRoadUNet: ResNet-34 Attention U-Net with Dense ASPP bottleneck.
3. EnsembleSegFormerUNet: Probability blending and Test-Time Augmentation (TTA) module.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger("MultispectralModels5Class")


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


class MultispectralSegFormer5Class(nn.Module):
    """
    Hierarchical Vision Transformer adapted for 4-channel multispectral input (RGB + NIR)
    and 5 urban land-cover classes.
    """
    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        num_classes: int = 5,
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


class ASPPBottleneck(nn.Module):
    """Dense Atrous Spatial Pyramid Pooling Bottleneck."""
    def __init__(self, in_channels: int = 512, out_channels: int = 512) -> None:
        super().__init__()
        mid = out_channels // 4
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(8, mid),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SCSEBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        f0 = self.b0(x)
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        fp = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([f0, f1, f2, f3, fp], dim=1))


class DecoderBlock(nn.Module):
    """U-Net Decoder block with skip connection and scSE attention."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SCSEBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultispectralUltraRoadUNet5Class(nn.Module):
    """
    SOTA Road and Boundary-Preserving Attention U-Net with ResNet-34 backbone
    adapted for 5-Class 4-channel multispectral satellite data (RGB + NIR).
    """
    def __init__(self, num_classes: int = 5, in_channels: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SCSEBlock(32),
        )

        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        old_conv1 = base.conv1
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = old_conv1.weight
            new_conv1.weight[:, 3:4, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)
        self.enc0 = nn.Sequential(new_conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.aspp = ASPPBottleneck(512, 512)

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 32, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        e0 = self.enc0(x)
        e1 = self.enc1(self.maxpool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        bottleneck = self.aspp(e4)

        d4 = self.dec4(bottleneck, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1, s0)

        logits = self.head(d0)
        return logits


class EnsembleSegFormerUNet5Class:
    """Ensemble Blending Module for 5 Classes with TTA."""
    def __init__(self, segformer_model: nn.Module, unet_model: nn.Module, alpha: float = 0.55):
        self.segformer = segformer_model.eval()
        self.unet = unet_model.eval()
        self.alpha = alpha

    @torch.inference_mode()
    def predict_single(self, x: torch.Tensor) -> torch.Tensor:
        p_seg = F.softmax(self.segformer(x), dim=1)
        p_unet = F.softmax(self.unet(x), dim=1)
        return self.alpha * p_seg + (1.0 - self.alpha) * p_unet

    @torch.inference_mode()
    def predict_tta(self, x: torch.Tensor) -> torch.Tensor:
        transforms = [
            (lambda t: t, lambda t: t),
            (lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-1])),
            (lambda t: torch.flip(t, [-2]), lambda t: torch.flip(t, [-2])),
            (lambda t: torch.rot90(t, 1, [-2, -1]), lambda t: torch.rot90(t, -1, [-2, -1])),
            (lambda t: torch.rot90(t, 2, [-2, -1]), lambda t: torch.rot90(t, -2, [-2, -1])),
            (lambda t: torch.rot90(t, 3, [-2, -1]), lambda t: torch.rot90(t, -3, [-2, -1])),
            (lambda t: torch.rot90(torch.flip(t, [-1]), 1, [-2, -1]), lambda t: torch.flip(torch.rot90(t, -1, [-2, -1]), [-1])),
            (lambda t: torch.rot90(torch.flip(t, [-2]), 1, [-2, -1]), lambda t: torch.flip(torch.rot90(t, -1, [-2, -1]), [-2])),
        ]
        accum = torch.zeros(x.shape[0], 5, x.shape[2], x.shape[3], device=x.device)
        for f_tf, inv_tf in transforms:
            aug_x = f_tf(x)
            p_aug = self.predict_single(aug_x)
            accum += inv_tf(p_aug)
        return accum / len(transforms)
