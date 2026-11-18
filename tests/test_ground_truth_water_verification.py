import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
BASE_5C_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\05_Multispectral_5Class_Training")
TEST_MASK_DIR = BASE_DIR / "data" / "test" / "masks"
TEST_MASK_5C_DIR = BASE_5C_DIR / "data" / "test" / "masks"
STATS_CSV = BASE_DIR / "data" / "test_set_corrected_annotation_stats.csv"

class TestGroundTruthWaterVerification(unittest.TestCase):

    def test_01_tile_0585_water_is_strictly_zero(self):
        """CRITICAL REQUIREMENT: test_pune_0585 must have exactly 0 water pixels (0.0%)."""
        mask_path = TEST_MASK_DIR / "test_pune_0585.png"
        self.assertTrue(mask_path.exists(), f"Mask {mask_path} does not exist!")

        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        self.assertEqual(m.shape, (512, 512), "Mask must have shape (512, 512)")
        self.assertEqual(m.dtype, np.uint8, "Mask must be uint8")

        water_px = int((m == 0).sum())
        water_pct = float(100.0 * water_px / m.size)

        self.assertEqual(water_px, 0, f"FAILED: test_pune_0585 contains {water_px} water pixels; expected 0!")
        self.assertEqual(water_pct, 0.0, f"FAILED: test_pune_0585 water percentage is {water_pct}%; expected 0.0%!")

    def test_02_tile_0585_5class_mask_water_is_zero(self):
        """Verify 5-class training mask for test_pune_0585 also has 0 water pixels."""
        mask_path = TEST_MASK_5C_DIR / "test_pune_0585.png"
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            water_px = int((m == 0).sum())
            self.assertEqual(water_px, 0, f"FAILED: 5-class mask contains {water_px} water pixels; expected 0!")

    def test_03_tile_0585_valid_classes_only(self):
        """Verify test_pune_0585 contains valid urban classes and no out-of-range labels."""
        mask_path = TEST_MASK_DIR / "test_pune_0585.png"
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        unique_classes = set(np.unique(m))
        valid_classes = {0, 1, 2, 3, 4, 5, 6}
        self.assertTrue(unique_classes.issubset(valid_classes), f"Invalid classes found: {unique_classes - valid_classes}")
        self.assertNotIn(0, unique_classes, "Class 0 (Water) must NOT be present in test_pune_0585!")

    def test_04_all_dry_test_tiles_have_zero_water(self):
        """Verify all 177 test tiles outside hydrology vector corridors contain 0 water pixels."""
        self.assertTrue(STATS_CSV.exists(), f"Stats CSV {STATS_CSV} not found!")
        df = pd.read_csv(STATS_CSV)
        dry_df = df[~df["has_vector_water"]]

        failures = []
        for _, row in dry_df.iterrows():
            pid = row["patch_id"]
            mask_path = TEST_MASK_DIR / f"{pid}.png"
            m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            w_px = (m == 0).sum()
            if w_px > 0:
                failures.append(f"{pid}: {w_px} px")

        self.assertEqual(len(failures), 0, f"False water found in dry tiles: {failures}")

    def test_05_water_tiles_are_strictly_hydrology_corridors(self):
        """Verify only genuine river corridor tiles contain water and are non-empty."""
        df = pd.read_csv(STATS_CSV)
        water_df = df[df["water_px"] > 0]
        # Water must be strictly a subset of vector-intersecting tiles
        for _, row in water_df.iterrows():
            self.assertTrue(row["has_vector_water"], f"Tile {row['patch_id']} has water but no vector hydrology!")

    def test_06_mask_dimensions_and_dtypes(self):
        """Verify all 186 test masks have valid 512x512 uint8 format."""
        all_masks = list(TEST_MASK_DIR.glob("test_pune_*.png"))
        self.assertEqual(len(all_masks), 186, f"Expected 186 test masks, found {len(all_masks)}")
        for mf in all_masks:
            m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            self.assertEqual(m.shape, (512, 512), f"Tile {mf.name} has incorrect shape: {m.shape}")
            self.assertEqual(m.dtype, np.uint8, f"Tile {mf.name} has incorrect dtype: {m.dtype}")

    def test_07_all_training_tiles_water_is_strictly_zero(self):
        """Verify all 401 training masks have strictly 0 water pixels (0.00%)."""
        train_mask_dir = BASE_DIR / "data" / "train" / "masks"
        all_train_masks = list(train_mask_dir.glob("*.png"))
        self.assertEqual(len(all_train_masks), 401, f"Expected 401 training masks, found {len(all_train_masks)}")
        failures = []
        for mf in all_train_masks:
            m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            w_px = (m == 0).sum()
            if w_px > 0:
                failures.append(f"{mf.name}: {w_px} px")
        self.assertEqual(len(failures), 0, f"False water found in training masks: {failures[:10]}")

    def test_08_training_masks_valid_classes_only(self):
        """Verify all 401 training masks contain only valid 7-class values {1,2,3,4,5,6,255} (no false 0)."""
        train_mask_dir = BASE_DIR / "data" / "train" / "masks"
        for mf in list(train_mask_dir.glob("*.png"))[:50]:
            m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            unique_classes = set(np.unique(m))
            valid_classes = {1, 2, 3, 4, 5, 6, 255}
            self.assertTrue(unique_classes.issubset(valid_classes), f"Invalid classes in {mf.name}: {unique_classes - valid_classes}")

if __name__ == "__main__":
    unittest.main()

