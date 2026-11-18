"""
generate_autonomous_dataset_manifest.py
----------------------------------------
Generates the official dataset manifest for autonomous training.
- 401 approved training patches partitioned into:
  - train (341 patches, ~85% stratified by city)
  - val (60 patches, ~15% stratified by city)
- 186 held-out test patches (strictly held-out test set, NEVER used in training/tuning)
Records: patch_id, city, split, image_path, mask_path, annotation_version,
pixel counts for all 7 classes, and class percentages.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation"
DATA_DIR = BASE_DIR / "data"
INDEX_CSV = DATA_DIR / "dataset_index.csv"

assert INDEX_CSV.exists(), f"Missing {INDEX_CSV}"
df_orig = pd.read_csv(INDEX_CSV)
print(f"Loaded {len(df_orig)} patches from {INDEX_CSV}")

train_df = df_orig[df_orig["split"] == "train"].copy().reset_index(drop=True)
test_df = df_orig[df_orig["split"] == "test"].copy().reset_index(drop=True)

# Set random seed for 100% reproducible split
np.random.seed(42)

# Stratify 401 training patches by city (191 Hinjewadi, 210 Chennai)
val_indices = []
for city in train_df["city"].unique():
    city_idx = train_df[train_df["city"] == city].index.values
    np.random.shuffle(city_idx)
    n_val = int(round(len(city_idx) * 0.15))
    val_indices.extend(city_idx[:n_val])

train_df["split"] = "train"
train_df.loc[val_indices, "split"] = "val"

print("Approved patches partition:")
print(train_df["split"].value_counts())
print("\nBy city:")
print(pd.crosstab(train_df["city"], train_df["split"]))

test_df["split"] = "held_out_test"

combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df["annotation_version"] = "v3_verified_water_purged"

# Re-verify and compute exact pixel counts for each sample
class_cols = ["px_water", "px_veg", "px_barren", "px_res", "px_ind", "px_slum", "px_road"]
for c in class_cols:
    combined_df[c] = 0

print("\nVerifying mask arrays and calculating class pixel statistics...")
for idx, row in combined_df.iterrows():
    mp = row["mask_path"]
    if os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        unq, counts = np.unique(m, return_counts=True)
        cdict = dict(zip(unq, counts))
        combined_df.at[idx, "px_water"] = cdict.get(0, 0)
        combined_df.at[idx, "px_veg"] = cdict.get(1, 0)
        combined_df.at[idx, "px_barren"] = cdict.get(2, 0)
        combined_df.at[idx, "px_res"] = cdict.get(3, 0)
        combined_df.at[idx, "px_ind"] = cdict.get(4, 0)
        combined_df.at[idx, "px_slum"] = cdict.get(5, 0)
        combined_df.at[idx, "px_road"] = cdict.get(6, 0)

manifest_path = DATA_DIR / "autonomous_dataset_manifest.csv"
combined_df.to_csv(manifest_path, index=False)
print(f"\nSaved official manifest to: {manifest_path} ({len(combined_df)} rows)")

# Summary stats per split
for s in ["train", "val", "held_out_test"]:
    sub = combined_df[combined_df["split"] == s]
    total_px = len(sub) * 512 * 512
    print(f"\n--- Split: {s} ({len(sub)} patches) ---")
    for ci, cname in enumerate(["Water", "Veg", "Barren", "Residential", "Industrial", "Slum", "Road"]):
        col = class_cols[ci]
        sum_px = sub[col].sum()
        pct = (sum_px / total_px) * 100
        print(f"  {cname:12s}: {sum_px:10,d} px ({pct:5.2f}%)")

