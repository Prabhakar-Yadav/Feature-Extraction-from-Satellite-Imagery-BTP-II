"""dataset.py
----------
PyTorch Dataset for 5-Class Cartosat-1 (2.5m) Urban Feature Extraction on SegFormer.
Applies Tri-Band Panchromatic Feature Synthesis with In-Memory Caching:
  - Channel 0: Percentile-normalized Panchromatic intensity
  - Channel 1: CLAHE (Contrast-Limited Adaptive Histogram Equalization) for asphalt vs concrete contrast
  - Channel 2: Sobel Edge Gradient Magnitude for physical curbs, road borders, and divider medians
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CartosatUrbanDataset(Dataset):
    def __init__(
        self,
        patches_root: str | Path,
        split: str = "train",
        augment: bool = True,
        cache_in_memory: bool = True,
    ) -> None:
        self.patches_root = Path(patches_root)
        self.split = split
        self.augment = augment and (split == "train")
        self.cache_in_memory = cache_in_memory
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        csv_path = self.patches_root / "dataset_index.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset index: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # ImageNet normalization parameters for 3-channel input
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _synthesize_bands(self, img_gray: np.ndarray) -> np.ndarray:
        """Synthesizes (H, W, 3) rich feature tensor from 1-band panchromatic."""
        b0 = img_gray.astype(np.float32) / 255.0
        b1_clahe = self.clahe.apply(img_gray).astype(np.float32) / 255.0
        gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag_norm = np.clip(mag / 180.0, 0.0, 1.0)
        return np.stack([b0, b1_clahe, mag_norm], axis=-1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        patch_id = row["patch_id"]

        if self.cache_in_memory and patch_id in self._cache:
            img_3c, mask = self._cache[patch_id]
            img_3c = img_3c.copy()
            mask = mask.copy()
        else:
            img_path = self.patches_root / row["image_path"]
            mask_path = self.patches_root / row["mask_path"]

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                raise RuntimeError(f"Failed to load patch {patch_id}: {img_path}")

            img_3c = self._synthesize_bands(img)
            if self.cache_in_memory:
                self._cache[patch_id] = (img_3c.copy(), mask.copy())

        # Augmentation
        if self.augment:
            img_3c, mask = self._apply_augmentation(img_3c, mask)

        # Normalize
        img_norm = (img_3c - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()

        return {
            "pixel_values": img_tensor,
            "labels": mask_tensor,
            "patch_id": patch_id,
        }

    def _apply_augmentation(self, img_3c: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Horizontal Flip
        if random.random() > 0.5:
            img_3c = np.ascontiguousarray(img_3c[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Vertical Flip
        if random.random() > 0.5:
            img_3c = np.ascontiguousarray(img_3c[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # Random 90-degree Rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img_3c = np.ascontiguousarray(np.rot90(img_3c, k, (0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, (0, 1)))

        return img_3c, mask
