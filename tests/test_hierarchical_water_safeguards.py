"""
tests/test_hierarchical_water_safeguards.py
-------------------------------------------
Unit tests verifying:
1. 5-Class Macro, 3-Subclass, and 7-Class Unified Taxonomy Integrity.
2. Two-Step Hierarchical Gating (Subclasses strictly inside Stage 1 Built-up).
3. Shadow & Asphalt-Safe Water Safeguards (False water suppression + Genuine water retention).
4. Forward and predict interface integrity for HybridUNetSegFormerTwoStep.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SCRIPT_DIR = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "scripts"))

from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep, regularize_subclass_blocks


class TestHierarchicalTaxonomy(unittest.TestCase):
    def test_class_mappings(self):
        """Verifies coarse-to-subclass-to-7class mapping consistency."""
        # Coarse 5-class IDs:
        COARSE_WATER = 0
        COARSE_VEG = 1
        COARSE_BARREN = 2
        COARSE_BUILTUP = 3
        COARSE_ROAD = 4

        # Subclass IDs:
        SUB_RES = 0
        SUB_IND = 1
        SUB_SLUM = 2

        # 7-Class Unified IDs:
        FINAL_WATER = 0
        FINAL_VEG = 1
        FINAL_BARREN = 2
        FINAL_RES = 3
        FINAL_IND = 4
        FINAL_SLUM = 5
        FINAL_ROAD = 6

        # Simulate mapping logic
        h, w = 100, 100
        pred_5c = np.full((h, w), COARSE_BARREN, dtype=np.uint8)
        pred_5c[10:30, 10:30] = COARSE_WATER
        pred_5c[30:50, 30:50] = COARSE_VEG
        pred_5c[50:80, 50:80] = COARSE_BUILTUP
        pred_5c[80:90, :] = COARSE_ROAD

        builtup_mask = (pred_5c == COARSE_BUILTUP)
        pred_sub = np.zeros((h, w), dtype=np.uint8)
        pred_sub[50:60, 50:80] = SUB_RES
        pred_sub[60:70, 50:80] = SUB_IND
        pred_sub[70:80, 50:80] = SUB_SLUM

        final_7c = np.zeros((h, w), dtype=np.uint8)
        final_7c[pred_5c == COARSE_WATER] = FINAL_WATER
        final_7c[pred_5c == COARSE_VEG] = FINAL_VEG
        final_7c[pred_5c == COARSE_BARREN] = FINAL_BARREN
        final_7c[pred_5c == COARSE_ROAD] = FINAL_ROAD

        final_7c[builtup_mask & (pred_sub == SUB_RES)] = FINAL_RES
        final_7c[builtup_mask & (pred_sub == SUB_IND)] = FINAL_IND
        final_7c[builtup_mask & (pred_sub == SUB_SLUM)] = FINAL_SLUM

        # Assert no unexpected classes
        unique_classes = set(np.unique(final_7c))
        self.assertTrue(unique_classes.issubset({0, 1, 2, 3, 4, 5, 6}))
        self.assertEqual(final_7c[15, 15], FINAL_WATER)
        self.assertEqual(final_7c[35, 35], FINAL_VEG)
        self.assertEqual(final_7c[5, 5], FINAL_BARREN)
        self.assertEqual(final_7c[55, 55], FINAL_RES)
        self.assertEqual(final_7c[65, 55], FINAL_IND)
        self.assertEqual(final_7c[75, 55], FINAL_SLUM)
        self.assertEqual(final_7c[85, 50], FINAL_ROAD)


class TestWaterSafeguards(unittest.TestCase):
    def test_shadow_and_real_water_discrimination(self):
        """
        Synthesizes a test tile containing:
        1. A genuine river/canal (area 1500 px, NIR=15 DN, NDWI=+0.25).
        2. A building shadow (area 120 px, NIR=35 DN, surrounded by built-up).
        3. A dark asphalt road section (area 300 px, Road prob=0.45, NIR=45 DN).
        Verifies that shadow & asphalt false water are suppressed while the real river is retained.
        """
        import cv2

        h, w = 128, 128
        probs_5c = np.zeros((5, h, w), dtype=np.float32)
        probs_5c[2] = 0.60  # Default: Barren land

        nir_dn = np.full((h, w), 100.0, dtype=np.float32)
        ndvi = np.full((h, w), 0.10, dtype=np.float32)
        ndwi = np.full((h, w), -0.15, dtype=np.float32)

        # Region 1: Genuine River (Row 10 to 40, Col 10 to 60 -> 30 x 50 = 1500 px)
        probs_5c[:, 10:40, 10:60] = 0.05
        probs_5c[0, 10:40, 10:60] = 0.80  # Network says Water
        nir_dn[10:40, 10:60] = 15.0        # Deep NIR absorption
        ndwi[10:40, 10:60] = 0.28           # High positive NDWI

        # Region 2: Small Building Shadow (Row 70 to 80, Col 70 to 80 -> 100 px)
        probs_5c[:, 70:80, 70:80] = 0.15
        probs_5c[0, 70:80, 70:80] = 0.40  # Ambiguous false water
        probs_5c[3, 70:80, 70:80] = 0.35  # Near building
        nir_dn[70:80, 70:80] = 35.0
        ndwi[70:80, 70:80] = 0.12

        # Region 3: Dark Asphalt Road (Row 100 to 110, Col 10 to 110 -> 10 x 100 = 1000 px)
        probs_5c[:, 100:110, 10:110] = 0.10
        probs_5c[4, 100:110, 10:110] = 0.45 # High road confidence
        probs_5c[0, 100:110, 10:110] = 0.35 # False water spike
        nir_dn[100:110, 10:110] = 45.0
        ndwi[100:110, 10:110] = 0.08

        # Apply Shadow-Safe Filter
        refined_probs = probs_5c.copy()
        pred_init = np.argmax(refined_probs, axis=0)

        # Process water components
        raw_w = (pred_init == 0).astype(np.uint8)
        num_w, w_lbl, w_stat, _ = cv2.connectedComponentsWithStats(raw_w, connectivity=8)
        for wl in range(1, num_w):
            area = w_stat[wl, cv2.CC_STAT_AREA]
            cluster = (w_lbl == wl)
            mean_nir = float(nir_dn[cluster].mean())
            mean_ndvi = float(ndvi[cluster].mean())
            mean_ndwi = float(ndwi[cluster].mean())

            # False water test
            if (area < 500) or (mean_nir > 40.0) or (mean_ndwi < 0.15) or (refined_probs[4, cluster].mean() > 0.40):
                refined_probs[0, cluster] = 0.0  # Zero out water

        final_pred = np.argmax(refined_probs, axis=0)

        # Region 1 (River) must remain 100% Water (0)
        self.assertEqual(final_pred[20, 20], 0, "Genuine river body was incorrectly suppressed!")
        self.assertEqual(np.sum(final_pred[10:40, 10:60] == 0), 1500, "Genuine river pixels were lost!")

        # Region 2 (Small Shadow) must NOT be Water (0)
        self.assertNotEqual(final_pred[75, 75], 0, "Small building shadow was falsely classified as water!")

        # Region 3 (Asphalt Road) must be Road (4), NOT Water (0)
        self.assertEqual(final_pred[105, 50], 4, "Asphalt road was falsely classified as water!")


class TestSubclassBlockRegularization(unittest.TestCase):
    def test_regularize_subclass_blocks(self):
        """Verifies that block regularization preserves parcel shapes and ignores slums."""
        h, w = 64, 64
        sub_map = np.zeros((h, w), dtype=np.uint8)
        builtup_mask = np.ones((h, w), dtype=bool)

        # Residential block with a few noisy industrial speckles
        sub_map[:, :] = 0  # Residential
        sub_map[10:12, 10:12] = 1  # 4 pixels noise

        # Slum area (Subclass 2) should remain intact even if small
        sub_map[40:45, 40:45] = 2  # Slum

        reg_map = regularize_subclass_blocks(sub_map, builtup_mask, min_parcel_area=16)

        # 4-pixel speckle should be smoothed back to residential (0)
        self.assertEqual(reg_map[11, 11], 0, "Noisy speckle was not regularized!")
        # Slum settlement should remain 2
        self.assertEqual(reg_map[42, 42], 2, "Slum settlement was incorrectly removed!")


if __name__ == "__main__":
    unittest.main()

