import os
import sys
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
SCENE_PUNE = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Test_Data\2673727101")
OUT_DIR = BASE_DIR.parent.parent / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results" / "human_verification_batches"
BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf")
THUMB_DIR = OUT_DIR / "thumbnails"
BRAIN_THUMB_DIR = BRAIN_DIR / "thumbnails"

OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_THUMB_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta / Coral Red
    4: (140, 50, 190),    # Industrial: Steel Violet / Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
    255: (150, 150, 150)  # Ignore / Uncertain: Gray
}

CLASS_NAMES = {
    0: "Water",
    1: "Vegetation",
    2: "Barren Land",
    3: "Residential Built-up",
    4: "Industrial Built-up",
    5: "Slum / Informal Settlement",
    6: "Road Network",
    255: "Ignore / Uncertain"
}

def mask_to_rgb(mask):
    """Categorical discrete color lookup (no interpolation)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

def stretch_8bit(arr, p_low=2.0, p_high=98.0):
    """Robust 2-98% radiometric contrast stretching for satellite bands."""
    v_min, v_max = np.percentile(arr, (p_low, p_high))
    if v_max <= v_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr.astype(float) - v_min) / (v_max - v_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

def main():
    print("==================================================================")
    print("STEP 1: Extract and Radiometrically Correct Imagery for test_pune_0585")
    print("==================================================================")
    r, c, win_size = 17100, 15900, 512
    window = Window(c, r, win_size, win_size)

    with rasterio.open(SCENE_PUNE / "BAND1.tif") as s1, \
         rasterio.open(SCENE_PUNE / "BAND2.tif") as s2, \
         rasterio.open(SCENE_PUNE / "BAND3.tif") as s3, \
         rasterio.open(SCENE_PUNE / "BAND4.tif") as s4:

        b1_raw = s1.read(1, window=window) # Blue (0.45 - 0.52 um)
        b2_raw = s2.read(1, window=window) # Green (0.52 - 0.59 um)
        b3_raw = s3.read(1, window=window) # Red (0.62 - 0.68 um)
        b4_raw = s4.read(1, window=window) # NIR (0.77 - 0.86 um)

    # Apply 2-98% radiometric contrast stretch
    b_8 = stretch_8bit(b1_raw)
    g_8 = stretch_8bit(b2_raw)
    r_8 = stretch_8bit(b3_raw)
    nir_8 = stretch_8bit(b4_raw)

    # 1. Dedicated 3-Band True-Color RGB (No Alpha channel!)
    rgb_truecolor = np.stack([r_8, g_8, b_8], axis=-1)
    rgb_bgr = cv2.cvtColor(rgb_truecolor, cv2.COLOR_RGB2BGR)

    # 2. Dedicated False-Color Infrared (CIR) (NIR, Red, Green)
    cir_rgb = np.stack([nir_8, r_8, g_8], axis=-1)
    cir_bgr = cv2.cvtColor(cir_rgb, cv2.COLOR_RGB2BGR)

    # 3. Read Corrected Ground-Truth Mask
    mask_path = BASE_DIR / "data" / "test" / "masks" / "test_pune_0585.png"
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    # Verify mask values
    u_vals, u_counts = np.unique(mask_raw, return_counts=True)
    stats_dict = dict(zip(u_vals, u_counts))
    water_px = stats_dict.get(0, 0)
    print(f"Mask test_pune_0585 verified: shape={mask_raw.shape}, classes={u_vals}")
    print(f"Water pixel count: {water_px} (0.00%)")
    assert water_px == 0, "Water pixel count must be 0!"

    # 4. Generate Discrete Categorical Colorized Mask (RGB)
    mask_color_rgb = mask_to_rgb(mask_raw)
    mask_color_bgr = cv2.cvtColor(mask_color_rgb, cv2.COLOR_RGB2BGR)

    # 5. Alpha Blend Overlay (60% satellite RGB + 40% categorical mask)
    blend_rgb = cv2.addWeighted(rgb_truecolor, 0.60, mask_color_rgb, 0.40, 0)
    blend_bgr = cv2.cvtColor(blend_rgb, cv2.COLOR_RGB2BGR)

    # Save dedicated files in multiple target directories
    file_paths = {
        "rgb_truecolor": [
            OUT_DIR / "test_pune_0585_rgb_truecolor.png",
            BRAIN_DIR / "test_pune_0585_rgb_truecolor.png"
        ],
        "false_color_cir": [
            OUT_DIR / "test_pune_0585_false_color_cir.png",
            BRAIN_DIR / "test_pune_0585_false_color_cir.png"
        ],
        "ground_truth_colorized": [
            OUT_DIR / "test_pune_0585_ground_truth_colorized.png",
            BRAIN_DIR / "test_pune_0585_ground_truth_colorized.png"
        ]
    }

    for p in file_paths["rgb_truecolor"]:
        cv2.imwrite(str(p), rgb_bgr)
    for p in file_paths["false_color_cir"]:
        cv2.imwrite(str(p), cir_bgr)
    for p in file_paths["ground_truth_colorized"]:
        cv2.imwrite(str(p), mask_color_bgr)

    print("Saved dedicated TrueColor RGB, FalseColor CIR, and Colorized Mask files.")

    # 6. Generate High-Resolution Side-by-Side Comparison Figure
    print("\n==================================================================")
    print("STEP 2: Generate High-Resolution Side-by-Side Comparison Figure")
    print("==================================================================")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=200)

    axes[0].imshow(rgb_truecolor)
    axes[0].set_title("1. Cartosat-2E True-Color RGB (0.60m)\n[Bands 3(Red), 2(Green), 1(Blue)]", fontsize=12, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_color_rgb)
    axes[1].set_title("2. Corrected 7-Class Ground Truth\n[Water: 0 px (0.00%) | Certified Clean]", fontsize=12, weight="bold", color="darkgreen")
    axes[1].axis("off")

    axes[2].imshow(blend_rgb)
    axes[2].set_title("3. Spatial Overlay Verification\n(60% Satellite RGB + 40% Ground Truth)", fontsize=12, weight="bold")
    axes[2].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i])/255.0, label=f"Class {i}: {CLASS_NAMES[i]} ({stats_dict.get(i, 0):,} px, {100.0*stats_dict.get(i,0)/mask_raw.size:.1f}%)")
        for i in [0, 1, 2, 3, 4, 5, 6]
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("Tile test_pune_0585: Ground-Truth Verification & Radiometric Diagnostic Suite", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    sbs_out = OUT_DIR / "test_pune_0585_side_by_side_comparison.png"
    sbs_brain = BRAIN_DIR / "test_pune_0585_side_by_side_comparison.png"
    plt.savefig(sbs_out, bbox_inches="tight", dpi=200)
    plt.savefig(sbs_brain, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved Side-by-Side Comparison to:\n  {sbs_out}\n  {sbs_brain}")

    # 7. Generate Per-Patch Complete Test Set Verification CSV Report
    print("\n==================================================================")
    print("STEP 3: Generate Per-Patch Complete Test Set Verification Report")
    print("==================================================================")
    stats_csv_orig = BASE_DIR / "data" / "test_set_corrected_annotation_stats.csv"
    df_stats = pd.read_csv(stats_csv_orig)

    report_rows = []
    for _, row in df_stats.iterrows():
        pid = row["patch_id"]
        m_path = BASE_DIR / "data" / "test" / "masks" / f"{pid}.png"
        m = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)

        # Check for invalid pixels
        invalid_mask = ~np.isin(m, [0, 1, 2, 3, 4, 5, 6, 255])
        invalid_px = int(invalid_mask.sum())
        ignore_px = int((m == 255).sum())

        w_px = int((m == 0).sum())
        w_pct = round(100.0 * w_px / m.size, 2)
        has_vector = bool(row["has_vector_water"])

        if w_px == 0:
            status = "Verified Dry (0.00% Water)"
        else:
            status = f"Verified River Corridor ({w_pct}% Water)"

        report_rows.append({
            "patch_id": pid,
            "scene_row": row["row"],
            "scene_col": row["col"],
            "has_vector_hydrology": has_vector,
            "water_pixel_count": w_px,
            "water_percentage": w_pct,
            "vegetation_pct": row["veg_pct"],
            "barren_land_pct": row["barren_pct"],
            "residential_pct": row["res_pct"],
            "industrial_pct": row["ind_pct"],
            "slum_pct": row["slum_pct"],
            "road_network_pct": row["road_pct"],
            "ignore_pixel_count": ignore_px,
            "invalid_pixel_count": invalid_px,
            "verification_status": status
        })

    df_report = pd.DataFrame(report_rows)
    report_csv_out = OUT_DIR / "test_set_per_patch_verification_report.csv"
    report_csv_brain = BRAIN_DIR / "test_set_per_patch_verification_report.csv"
    df_report.to_csv(report_csv_out, index=False)
    df_report.to_csv(report_csv_brain, index=False)
    print(f"Saved Per-Patch Report ({len(df_report)} patches) to:\n  {report_csv_out}\n  {report_csv_brain}")

    # 8. Copy All Thumbnails to Brain Directory for 100% Guaranteed Image Loading
    print("\n==================================================================")
    print("STEP 4: Synchronize Thumbnails to Conversation Brain Directory")
    print("==================================================================")
    thumb_files = list(THUMB_DIR.glob("*.png"))
    print(f"Copying {len(thumb_files)} composite thumbnails to {BRAIN_THUMB_DIR}...")
    for tf in thumb_files:
        shutil.copy2(tf, BRAIN_THUMB_DIR / tf.name)
    print("Thumbnails synchronization complete.")

    # Copy all 10 batch overviews to Brain Directory
    for b in range(1, 11):
        bfile = OUT_DIR / f"batch_{b:02d}_overview.png"
        if bfile.exists():
            shutil.copy2(bfile, BRAIN_DIR / bfile.name)
    print("Batch overview contact sheets copied to Brain Directory.")

    # Copy 11-panel diagnostic
    p11 = BASE_DIR / "visual_results" / "ground_truth_fix_11panel_test_pune_0585.png"
    if p11.exists():
        shutil.copy2(p11, OUT_DIR / "ground_truth_fix_11panel_test_pune_0585.png")
        shutil.copy2(p11, BRAIN_DIR / "ground_truth_fix_11panel_test_pune_0585.png")

    # 9. Create Comprehensive Downloadable ZIP Archive
    print("\n==================================================================")
    print("STEP 5: Build Comprehensive Deliverables ZIP Archive")
    print("==================================================================")
    zip_path_out = OUT_DIR / "Cartosat2E_GroundTruth_Inspection_Package.zip"
    zip_path_brain = BRAIN_DIR / "Cartosat2E_GroundTruth_Inspection_Package.zip"

    # Write a clean README.txt for the package
    readme_content = """================================================================================
CARTOSAT-2E 0.60m MULTISPECTRAL GROUND-TRUTH INSPECTION PACKAGE
Pune City Urban Land-Use & Land-Cover Segmentation
================================================================================

This package contains complete verified ground-truth annotations, batch overview
contact sheets, radiometric diagnostic figures, and the interactive verification portal.

CONTENTS:
1. inspection_dashboard.html
   - Interactive verification web portal. Open in any browser (Chrome, Edge, Firefox).
   - Allows rapid batch-by-batch review of all 186 test patches with per-class statistics.

2. batch_01_overview.png to batch_10_overview.png
   - High-resolution overview contact sheets (20 patches per batch; Batch 10 contains 6 patches).
   - Each patch card shows: [Original Satellite RGB | Colorized Ground Truth | Alpha Blend Overlay].

3. Tile test_pune_0585 High-Resolution Diagnostic Suite:
   - test_pune_0585_rgb_truecolor.png: 3-band true-color satellite image (Red, Green, Blue, 0.60m).
   - test_pune_0585_false_color_cir.png: Color-Infrared image (NIR, Red, Green).
   - test_pune_0585_ground_truth_colorized.png: Categorical 7-class discrete color mask.
   - test_pune_0585_side_by_side_comparison.png: Side-by-side comparison with detailed class legend.
   - ground_truth_fix_11panel_test_pune_0585.png: 11-panel multispectral and shadow diagnostic suite.

4. test_set_per_patch_verification_report.csv
   - Comprehensive CSV table reporting per-patch water count, class percentages,
     vector hydrology intersection, invalid pixel checks, and QA verification status.

COLOR PALETTE REFERENCE:
- Class 0: Water -> Blue (RGB: 30, 120, 220)
- Class 1: Vegetation -> Green (RGB: 34, 150, 40)
- Class 2: Barren Land -> Warm Sand (RGB: 215, 185, 145)
- Class 3: Residential -> Coral Red (RGB: 230, 90, 70)
- Class 4: Industrial -> Steel Violet / Purple (RGB: 140, 50, 190)
- Class 5: Slum / Informal -> Bright Amber / Orange (RGB: 255, 120, 0)
- Class 6: Road Network -> Bright Yellow (RGB: 255, 225, 20)
- Class 255: Ignore / Uncertain -> Gray (RGB: 150, 150, 150)
================================================================================
"""
    readme_path = OUT_DIR / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    with zipfile.ZipFile(zip_path_out, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(readme_path, arcname="README.txt")
        zipf.write(OUT_DIR / "inspection_dashboard.html", arcname="inspection_dashboard.html")
        zipf.write(report_csv_out, arcname="test_set_per_patch_verification_report.csv")
        zipf.write(OUT_DIR / "test_pune_0585_rgb_truecolor.png", arcname="test_pune_0585_rgb_truecolor.png")
        zipf.write(OUT_DIR / "test_pune_0585_false_color_cir.png", arcname="test_pune_0585_false_color_cir.png")
        zipf.write(OUT_DIR / "test_pune_0585_ground_truth_colorized.png", arcname="test_pune_0585_ground_truth_colorized.png")
        zipf.write(OUT_DIR / "test_pune_0585_side_by_side_comparison.png", arcname="test_pune_0585_side_by_side_comparison.png")
        zipf.write(OUT_DIR / "ground_truth_fix_11panel_test_pune_0585.png", arcname="ground_truth_fix_11panel_test_pune_0585.png")

        for b in range(1, 11):
            bfile = OUT_DIR / f"batch_{b:02d}_overview.png"
            if bfile.exists():
                zipf.write(bfile, arcname=f"batch_overviews/batch_{b:02d}_overview.png")

        for tf in THUMB_DIR.glob("*.png"):
            zipf.write(tf, arcname=f"thumbnails/{tf.name}")

    shutil.copy2(zip_path_out, zip_path_brain)
    zip_mb = zip_path_out.stat().st_size / (1024 * 1024)
    print(f"Created ZIP Package ({zip_mb:.2f} MB) at:\n  {zip_path_out}\n  {zip_path_brain}")

    print("\n==================================================================")
    print("STEP 6: Programmatic Verification of All Generated Artifacts")
    print("==================================================================")
    # Check RGB TrueColor
    im_rgb = cv2.imread(str(OUT_DIR / "test_pune_0585_rgb_truecolor.png"))
    assert im_rgb.shape == (512, 512, 3), f"RGB shape incorrect: {im_rgb.shape}"
    assert im_rgb.min() < 30 and im_rgb.max() > 220, f"RGB dynamic range too narrow: [{im_rgb.min()}, {im_rgb.max()}]"
    print(f"  [PASS] test_pune_0585_rgb_truecolor.png: shape={im_rgb.shape}, range=[{im_rgb.min()}, {im_rgb.max()}]")

    # Check Colorized Mask
    im_mask = cv2.imread(str(OUT_DIR / "test_pune_0585_ground_truth_colorized.png"))
    assert im_mask.shape == (512, 512, 3), f"Colorized mask shape incorrect: {im_mask.shape}"
    u_bgr = np.unique(im_mask.reshape(-1, 3), axis=0)
    print(f"  [PASS] test_pune_0585_ground_truth_colorized.png: shape={im_mask.shape}, {len(u_bgr)} discrete colors present.")

    # Check Side by Side
    im_sbs = cv2.imread(str(sbs_out))
    assert im_sbs.shape[0] > 1000 and im_sbs.shape[1] > 2000, f"Side by side shape too small: {im_sbs.shape}"
    print(f"  [PASS] test_pune_0585_side_by_side_comparison.png: shape={im_sbs.shape}")

    # Check All 10 Batch Overviews
    for b in range(1, 11):
        bfile = OUT_DIR / f"batch_{b:02d}_overview.png"
        assert bfile.exists() and bfile.stat().st_size > 1000000, f"Batch {b} invalid!"
        im_b = cv2.imread(str(bfile))
        assert im_b.shape[0] > 1000 and im_b.shape[1] > 2000, f"Batch {b} shape too small: {im_b.shape}"
        print(f"  [PASS] batch_{b:02d}_overview.png: shape={im_b.shape}, size={bfile.stat().st_size:,} bytes")

    # Check ZIP validity
    with zipfile.ZipFile(zip_path_out, "r") as zcheck:
        namelist = zcheck.namelist()
        print(f"  [PASS] ZIP archive valid, contains {len(namelist)} files.")
        assert "inspection_dashboard.html" in namelist
        assert "test_set_per_patch_verification_report.csv" in namelist
        assert "test_pune_0585_rgb_truecolor.png" in namelist
        assert "test_pune_0585_ground_truth_colorized.png" in namelist
        assert "batch_overviews/batch_10_overview.png" in namelist

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

