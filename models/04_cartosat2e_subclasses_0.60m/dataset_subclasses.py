"""
dataset_subclasses.py
---------------------
PyTorch Dataset and Subclass Pseudo-Label Extraction Pipeline for Built-up Land-Cover Categories.

Subclass Schema (Target for Stage 2):
0: Residential Built-up (formal residential, concrete/terracotta roofs, mapped from original class 3)
1: Industrial Built-up (massive logistics warehouses, factories, metallic roofs, mapped from original class 4)
2: Slum / Informal Settlement (ultra-dense shacks, corrugated metal/asbestos, narrow alleys, mapped from original class 5)
255: Ignore Index (Non-built-up pixels: Water, Vegetation, Barren Land, Road Network)
"""

from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 7-Class Schema
ORIG_CLASS_NAMES = [
    "Water",
    "Vegetation",
    "Barren Land",
    "Residential Built-up",
    "Industrial Built-up",
    "Slum / Informal Settlement",
    "Road Network"
]

ORIG_CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta / Coral
    4: (140, 50, 190),    # Industrial: Purple / Steel Violet
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
}

# 3 Built-Up Subclass Schema
SUBCLASS_NAMES = [
    "Residential Built-up",
    "Industrial Built-up",
    "Slum / Informal Settlement"
]

SUBCLASS_COLORS = {
    0: (230, 90, 70),     # Residential: Soft Terracotta / Coral
    1: (140, 50, 190),    # Industrial: Purple / Steel Violet
    2: (255, 120, 0),     # Slum: Bright Amber / Orange
    255: (35, 40, 50),    # Background / Non-built-up: Dark Slate Gray
}

# Multispectral normalization parameters (R, G, B, NIR)
BAND_MEANS = np.array([0.485, 0.456, 0.406, 0.450], dtype=np.float32)
BAND_STDS  = np.array([0.229, 0.224, 0.225, 0.220], dtype=np.float32)


def remap_7class_to_3subclasses(mask_7c: np.ndarray) -> np.ndarray:
    """
    Extracts and remaps the 3 built-up subclasses from the original 7-class pseudo-labels.
    Class 3 (Residential) -> 0
    Class 4 (Industrial)  -> 1
    Class 5 (Slum)        -> 2
    All other pixels      -> 255 (ignore_index for loss & training)
    """
    sub_mask = np.full_like(mask_7c, 255, dtype=np.uint8)
    sub_mask[mask_7c == 3] = 0
    sub_mask[mask_7c == 4] = 1
    sub_mask[mask_7c == 5] = 2
    return sub_mask


def subclass_mask_to_rgb(sub_mask: np.ndarray) -> np.ndarray:
    """Renders a (H, W) 3-subclass mask with ignore_index into vibrant RGB."""
    rgb = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)
    for c, col in SUBCLASS_COLORS.items():
        rgb[sub_mask == c] = col
    return rgb


def orig_mask_to_rgb(mask_7c: np.ndarray) -> np.ndarray:
    """Renders a (H, W) 7-class mask into vibrant RGB."""
    rgb = np.zeros((*mask_7c.shape, 3), dtype=np.uint8)
    for c, col in ORIG_CLASS_COLORS.items():
        rgb[mask_7c == c] = col
    return rgb


def bands_to_true_color(img_4c: np.ndarray) -> np.ndarray:
    """Extracts True-Color RGB [Red, Green, Blue] from normalized [0, 1] 4-band array."""
    if img_4c.shape[0] == 4:
        r, g, b = img_4c[0], img_4c[1], img_4c[2]
    else:
        r, g, b = img_4c[..., 0], img_4c[..., 1], img_4c[..., 2]
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def bands_to_false_color(img_4c: np.ndarray) -> np.ndarray:
    """Extracts False-Color Composite [NIR, Red, Green] highlighting vegetation and moisture."""
    if img_4c.shape[0] == 4:
        nir, r, g = img_4c[3], img_4c[0], img_4c[1]
    else:
        nir, r, g = img_4c[..., 3], img_4c[..., 0], img_4c[..., 1]
    fc = np.stack([nir, r, g], axis=-1)
    return np.clip(fc * 255.0, 0, 255).astype(np.uint8)


class SubclassMultispectralDataset(Dataset):
    """
    High-Performance PyTorch Dataset specialized for Built-Up Subclass Segmentation.
    Loads 4-band multispectral imagery and extracts target pseudo-labels with 3 subclasses.
    """
    def __init__(
        self,
        index_df: pd.DataFrame,
        is_train: bool = True,
        target_size: int = 512,
        filter_builtup: bool = False,
        min_builtup_pct: float = 0.05
    ):
        df = index_df.copy()
        if filter_builtup:
            builtup_pct = df["residential_pct"] + df["industrial_pct"] + df["slum_pct"]
            df = df[builtup_pct >= min_builtup_pct]
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        # 1. Load 4-band image (BGRA PNG -> R, G, B, NIR)
        bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Missing image file: {img_path}")
        b, g, r, nir = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2], bgra[:, :, 3]
        img = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0

        # Always load original mask for alignment and return tensor
        orig_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if orig_mask is not None and (orig_mask.shape[0] != self.target_size or orig_mask.shape[1] != self.target_size):
            orig_mask = cv2.resize(orig_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        if orig_mask is None:
            orig_mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        # 2. Load clean subclass mask (0: unmarked, 1: res, 2: ind, 3: slum)
        pid = Path(row["image_path"]).stem
        sub_path = Path(__file__).resolve().parent.parent / "data" / "subclass_masks" / f"{pid}.png"
        
        if sub_path.exists():
            raw_sub = cv2.imread(str(sub_path), cv2.IMREAD_GRAYSCALE)
            if raw_sub.shape[0] != self.target_size or raw_sub.shape[1] != self.target_size:
                raw_sub = cv2.resize(raw_sub, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
            # Remap: 0 (unmarked) -> 255 (ignore), 1 (res) -> 0, 2 (ind) -> 1, 3 (slum) -> 2
            sub_mask = np.full_like(raw_sub, 255, dtype=np.uint8)
            sub_mask[raw_sub == 1] = 0
            sub_mask[raw_sub == 2] = 1
            sub_mask[raw_sub == 3] = 2
        else:
            sub_mask = remap_7class_to_3subclasses(orig_mask)

        # Ensure spatial resolution of image is 512x512
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # 4. Data Augmentations (Train mode only)
        if self.is_train:
            # Horizontal Flip
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                sub_mask = np.fliplr(sub_mask).copy()
                orig_mask = np.fliplr(orig_mask).copy()
            # Vertical Flip
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                sub_mask = np.flipud(sub_mask).copy()
                orig_mask = np.flipud(orig_mask).copy()
            # 90-degree Rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k).copy()
                sub_mask = np.rot90(sub_mask, k).copy()
                orig_mask = np.rot90(orig_mask, k).copy()
            # Subtle spectral brightness jitter
            if random.random() > 0.5:
                factor = random.uniform(0.92, 1.08)
                img = np.clip(img * factor, 0.0, 1.0)

        # 5. Normalization
        img_norm = (img - BAND_MEANS) / BAND_STDS
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()    # [4, H, W]
        sub_mask_tensor = torch.from_numpy(sub_mask.astype(np.int64)).long()  # [H, W] (0, 1, 2, 255)
        orig_mask_tensor = torch.from_numpy(orig_mask.astype(np.int64)).long()# [H, W] (0..6)

        return img_tensor, sub_mask_tensor, orig_mask_tensor, row.get("patch_id", f"patch_{idx:05d}")
