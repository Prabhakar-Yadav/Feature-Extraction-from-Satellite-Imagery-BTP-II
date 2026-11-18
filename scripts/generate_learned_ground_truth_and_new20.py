import os
import sys
import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
IMG_DIR = BASE_DIR / "data" / "test" / "images"
MASK_DIR = BASE_DIR / "data" / "test" / "masks"

FINAL_SUBMISSION = BASE_DIR.parent.parent / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results"
REVIEW_DIR = FINAL_SUBMISSION / "human_verification_batches" / "builtup_reannotation_review"
DATA_DIR = REVIEW_DIR / "data"

NEW20_PANELS_DIR = REVIEW_DIR / "new_20_random_patch_comparisons"
NEW20_PANELS_DIR.mkdir(parents=True, exist_ok=True)

V4_DIR = DATA_DIR / "masks_v4_learned_candidate"
V4_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review")
BRAIN_NEW20_PANELS = BRAIN_DIR / "new_20_random_patch_comparisons"
BRAIN_NEW20_PANELS.mkdir(parents=True, exist_ok=True)
BRAIN_V4 = BRAIN_DIR / "data" / "masks_v4_learned_candidate"
BRAIN_V4.mkdir(parents=True, exist_ok=True)

# 7-Class Color Palette
CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren Land: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta / Coral Red
    4: (140, 50, 190),    # Industrial: Steel Violet / Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
    255: (150, 150, 150)  # Uncertain / Ignore: Neutral Gray
}

CLASS_NAMES = {
    0: "Water",
    1: "Vegetation",
    2: "Barren Land",
    3: "Residential Built-up",
    4: "Industrial Built-up",
    5: "Slum / Informal Settlement",
    6: "Road Network",
    255: "Uncertain / Ignore"
}

NEW_20_RANDOM_PATCHES = [
    "test_pune_0407", "test_pune_0408", "test_pune_0409", "test_pune_0423",
    "test_pune_0424", "test_pune_0427", "test_pune_0429", "test_pune_0436",
    "test_pune_0456", "test_pune_0458", "test_pune_0460", "test_pune_0463",
    "test_pune_0471", "test_pune_0516", "test_pune_0550", "test_pune_0560",
    "test_pune_0564", "test_pune_0572", "test_pune_0575", "test_pune_0584"
]

def stretch_8bit(arr, p_low=2.0, p_high=98.0):
    v_min, v_max = np.percentile(arr, (p_low, p_high))
    if v_max <= v_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr.astype(float) - v_min) / (v_max - v_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

def compute_mbi(gray):
    mbi = np.zeros_like(gray, dtype=float)
    scales = [7, 13, 21, 29]
    for L in scales:
        se_0 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 3))
        se_90 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, L))
        se_45 = np.eye(L, dtype=np.uint8)
        se_135 = np.fliplr(np.eye(L, dtype=np.uint8))
        for se in [se_0, se_90, se_45, se_135]:
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
            bothat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
            mbi += tophat.astype(float) + 0.5 * bothat.astype(float)
    return mbi / (len(scales) * 4.0)

def extract_learned_builtup_and_subclasses(im, m_orig, patch_id):
    """
    Learned ground-truth generation pipeline incorporating:
    1. Residential neighborhood context and block continuity (learned from test_pune_0505).
    2. Large consolidated industrial compound rules (preventing false industrial labeling of large residential roofs).
    3. Small irregular cluster morphology for slums.
    4. Building detection near trees (recovering roofs under light canopy overlap).
    5. Preservation of smooth barren land and existing water/road vectors.
    """
    b = im[:, :, 0].astype(float)
    g = im[:, :, 1].astype(float)
    r = im[:, :, 2].astype(float)
    nir = im[:, :, 3].astype(float)

    r_str = stretch_8bit(r)
    g_str = stretch_8bit(g)
    b_str = stretch_8bit(b)
    nir_str = stretch_8bit(nir)

    rgb = np.stack([r_str, g_str, b_str], axis=-1)
    cir = np.stack([nir_str, r_str, g_str], axis=-1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Spectral Indices
    denom_ndvi = nir + r + 1e-6
    ndvi = (nir - r) / denom_ndvi
    
    # Dense vegetation canopy vs light overhang
    dense_veg = ndvi > 0.32
    moderate_veg = (ndvi > 0.18) & (~dense_veg)

    # Road Network preservation (Class 6)
    road_mask = (m_orig == 6)

    # Water Body preservation (Class 0)
    water_mask = (m_orig == 0)

    # 2. Structural & Edge Indices
    mbi = compute_mbi(gray)
    
    # Canny edges & Morphological Gradient
    edges = cv2.Canny(gray, 35, 110)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k3)
    dilated_edges = cv2.dilate(edges, k3)

    # Texture roughness
    mean_sq = cv2.boxFilter(gray.astype(float)**2, -1, (13, 13))
    sq_mean = cv2.boxFilter(gray.astype(float), -1, (13, 13))**2
    local_std = np.sqrt(np.maximum(0, mean_sq - sq_mean))

    # Shadow adjacency
    shadow_mask = (gray < 48) & (nir < 62)
    dilated_shadow = cv2.dilate(shadow_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))) > 0

    # 3. Tree-Roof Overlap Diagnostic (Section 4)
    # Roofs partially occluded by trees have moderate NDVI (0.18-0.32) BUT have strong structural gradients or high MBI
    roof_near_trees = moderate_veg & ((mbi > 6.0) | (grad > 30)) & (~dense_veg)

    # Available ground for built-up: non-dense vegetation, non-road, non-water
    avail_mask = (~dense_veg) & (~road_mask) & (~water_mask)

    # 4. Building Core Detection:
    # High MBI OR bright roof OR textured roof with adjacent shadow OR recovered roof near trees
    bright_roof = (gray > 115) & (local_std > 5.0)
    struct_roof = (mbi > 6.0) | ((mbi > 3.5) & ((local_std > 9.0) | dilated_shadow))
    core_bldg = (bright_roof | struct_roof | roof_near_trees) & avail_mask

    # Decouple contiguous roof clusters using edge boundaries
    bldg_separated = core_bldg & (dilated_edges == 0)
    
    # Morphological closing to solidify building blocks
    bldg_solid = cv2.morphologyEx(bldg_separated.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0

    # 5. Connected Component Analysis for Subclass Typing
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bldg_solid.astype(np.uint8))

    res_mask = np.zeros((512, 512), dtype=bool)
    ind_mask = np.zeros((512, 512), dtype=bool)
    slum_mask = np.zeros((512, 512), dtype=bool)
    uncertain_mask = np.zeros((512, 512), dtype=bool)

    # Compute neighborhood context (15x15 density of buildings)
    bldg_density = cv2.boxFilter(bldg_solid.astype(float), -1, (25, 25))

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        comp = (labels == lbl)
        comp_gray = gray[comp]
        comp_std = comp_gray.std()
        comp_local_std = local_std[comp].mean()
        comp_density = bldg_density[comp].mean()

        if area < 20:
            uncertain_mask[comp] = True
            continue

        # Rule B: Industrial Built-up (Class 4) - Consolidated complexes only!
        # Must be large (>1400 px ~ 504 m2) AND very uniform (comp_std < 24.0) OR massive (>2500 px)
        if (area > 1400 and comp_std < 24.0) or (area > 2800):
            ind_mask[comp] = True
        # Rule C: Slum / Informal Settlement (Class 5)
        # Small structures (<160 px), high local roughness (>18.0), high building density (>0.50)
        elif area < 160 and comp_local_std > 18.0 and comp_density > 0.45:
            slum_mask[comp] = True
        # Rule A: Residential Built-up (Class 3)
        # Standard residential buildings, housing blocks, and distributed neighborhood homes
        else:
            res_mask[comp] = True

    # 6. Residential Block Consolidation (User's Style from 0505):
    # Where dense residential structures exist, include their immediate urban courtyards/compounds
    k_res_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    res_closed = cv2.morphologyEx(res_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_res_close) > 0
    res_mask = res_closed & avail_mask & (~ind_mask) & (~slum_mask)

    # 7. Barren Land: Remaining smooth unbuilt soil/dirt clearings
    all_bldg = res_mask | ind_mask | slum_mask
    barren_mask = avail_mask & (~all_bldg) & (~uncertain_mask)

    # 8. True Vegetation
    veg_final = (dense_veg | (moderate_veg & (~all_bldg))) & (~road_mask) & (~water_mask)

    # 9. Assemble 7-Class Candidate Mask
    mask_learned = np.full((512, 512), 2, dtype=np.uint8) # default barren
    mask_learned[veg_final] = 1
    mask_learned[barren_mask] = 2
    mask_learned[res_mask] = 3
    mask_learned[ind_mask] = 4
    mask_learned[slum_mask] = 5
    mask_learned[road_mask] = 6
    mask_learned[water_mask] = 0
    mask_learned[uncertain_mask] = 255

    return {
        "rgb": rgb,
        "cir": cir,
        "gray": gray,
        "ndvi": ndvi,
        "mbi": mbi,
        "local_std": local_std,
        "veg_mask": veg_final,
        "road_mask": road_mask,
        "water_mask": water_mask,
        "barren_mask": barren_mask,
        "all_bldg": all_bldg,
        "res_mask": res_mask,
        "ind_mask": ind_mask,
        "slum_mask": slum_mask,
        "uncertain_mask": uncertain_mask,
        "roof_near_trees": roof_near_trees,
        "mask_learned": mask_learned
    }

def render_10panel_comparison_new20(patch_id, data, orig_mask):
    """
    Renders high-resolution 10-panel diagnostic figure for one new patch:
    1. Satellite RGB (0.60m)
    2. False-Color CIR (NIR-R-G)
    3. Existing Ground Truth Mask (flawed baseline)
    4. Newly Generated Candidate Annotation (learned rules)
    5. Residential Mask (Class 3)
    6. Industrial Mask (Class 4)
    7. Slum / Informal Settlement Mask (Class 5)
    8. Vegetation & Building-Overlap Diagnostic (recovered roofs near trees)
    9. Road & Structure Boundaries
    10. Final Candidate Overlay (60% RGB + 40% Mask)
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 10.5), facecolor="#181a1b")
    fig.suptitle(f"Cartosat-2E Evaluation Suite (Learned Methodology) — Patch: {patch_id}",
                 fontsize=18, fontweight='bold', color="#ffffff", y=0.98)

    rgb = data["rgb"]
    cir = data["cir"]
    orig_rgb = mask_to_rgb(orig_mask)
    learned_rgb = mask_to_rgb(data["mask_learned"])
    overlay = (0.55 * rgb.astype(float) + 0.45 * learned_rgb.astype(float)).astype(np.uint8)

    tot_px = 512 * 512
    orig_barren = (orig_mask == 2).sum() / tot_px * 100
    orig_built = ((orig_mask == 3) | (orig_mask == 4) | (orig_mask == 5)).sum() / tot_px * 100
    
    new_barren = (data["mask_learned"] == 2).sum() / tot_px * 100
    new_res = (data["mask_learned"] == 3).sum() / tot_px * 100
    new_ind = (data["mask_learned"] == 4).sum() / tot_px * 100
    new_slum = (data["mask_learned"] == 5).sum() / tot_px * 100
    new_built = new_res + new_ind + new_slum

    # Overlap diagnostic: vegetation in green, recovered roof near trees in magenta
    overlap_diag = np.zeros((512, 512, 3), dtype=np.uint8)
    overlap_diag[data["veg_mask"]] = [34, 150, 40]
    overlap_diag[data["roof_near_trees"]] = [255, 0, 180] # Magenta

    # Road & structural boundaries
    bound_diag = np.zeros((512, 512, 3), dtype=np.uint8)
    bound_diag[data["all_bldg"]] = [200, 200, 200]
    bound_diag[data["road_mask"]] = [255, 225, 20]

    panels = [
        # Row 1
        (axes[0, 0], rgb, "1. Satellite RGB (0.60m True-Color)", "2%-98% Radiometric Stretch"),
        (axes[0, 1], cir, "2. False-Color Infrared (CIR: NIR-R-G)", "Biomass vs Impervious Roofs"),
        (axes[0, 2], orig_rgb, "3. Existing Ground-Truth Mask", f"Flawed GT: Barren={orig_barren:.1f}%, Built={orig_built:.1f}%"),
        (axes[0, 3], learned_rgb, "4. Newly Generated Candidate GT", f"Learned GT: Barren={new_barren:.1f}%, Built={new_built:.1f}%"),
        (axes[0, 4], np.stack([data["res_mask"]*230, data["res_mask"]*90, data["res_mask"]*70], axis=-1),
         "5. Residential Built-up (Class 3)", f"Area: {new_res:.1f}% ({data['res_mask'].sum()} px)"),
        # Row 2
        (axes[1, 0], np.stack([data["ind_mask"]*140, data["ind_mask"]*50, data["ind_mask"]*190], axis=-1),
         "6. Industrial Built-up (Class 4)", f"Consolidated Complexes: {new_ind:.1f}% ({data['ind_mask'].sum()} px)"),
        (axes[1, 1], np.stack([data["slum_mask"]*255, data["slum_mask"]*120, np.zeros_like(data['slum_mask'])], axis=-1),
         "7. Slum / Informal (Class 5)", f"Irregular Clusters: {new_slum:.1f}% ({data['slum_mask'].sum()} px)"),
        (axes[1, 2], overlap_diag, "8. Tree-Roof Overlap Diagnostic", "Green: Canopy | Magenta: Recovered Roofs"),
        (axes[1, 3], bound_diag, "9. Road & Parcel Boundaries", "Yellow: Roads | White: Structural Footprints"),
        (axes[1, 4], overlay, "10. Candidate Spatial Overlay", "60% Satellite RGB + 40% Mask")
    ]

    for ax, img, title, subtitle in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', color="#ffffff", pad=6)
        ax.set_xlabel(subtitle, fontsize=9, color="#9cdcfe", labelpad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#3c3c3c')
            spine.set_linewidth(1.5)

    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c])/255.0, label=CLASS_NAMES[c])
        for c in [0, 1, 2, 3, 4, 5, 6, 255]
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=8, framealpha=0.8,
               facecolor='#252526', edgecolor='#555555', labelcolor='#ffffff', fontsize=10)

    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.96])

    fname = f"{patch_id}_10panel_comparison.png"
    p1 = NEW20_PANELS_DIR / fname
    p2 = BRAIN_NEW20_PANELS / fname

    plt.savefig(p1, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(p2, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname

def build_new20_master_contact_sheet(records):
    """
    Creates master composite overview contact sheet for the 20 new randomly selected evaluation patches.
    """
    num_patches = len(records)
    fig, axes = plt.subplots(num_patches, 4, figsize=(18, 4.2 * num_patches), facecolor="#181a1b")
    fig.suptitle("EVALUATION SUITE: 20 NEW RANDOMLY SELECTED CARTOSAT-2E PATCHES (LEARNED METHODOLOGY)",
                 fontsize=17, fontweight='bold', color="#ffffff", y=0.995)

    col_titles = [
        "1. Satellite RGB (0.60m Stretched)",
        "2. Existing Ground Truth (Flawed Baseline)",
        "3. Newly Generated Candidate GT (Learned)",
        "4. Spatial Overlay (Boundary Alignment)"
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight='bold', color="#9cdcfe", pad=8)

    for row_idx, rec in enumerate(records):
        pid = rec["patch_id"]
        rgb = rec["rgb"]
        orig_rgb = mask_to_rgb(rec["orig_mask"])
        learned_rgb = mask_to_rgb(rec["mask_learned"])
        overlay = (0.55 * rgb.astype(float) + 0.45 * learned_rgb.astype(float)).astype(np.uint8)

        imgs = [rgb, orig_rgb, learned_rgb, overlay]
        subtitles = [
            f"{pid} | 0.60m Res",
            f"Barren={rec['orig_barren']:.1f}% | Built={rec['orig_built']:.1f}%",
            f"Barren={rec['new_barren']:.1f}% | Built={rec['new_built']:.1f}%",
            f"Res={rec['new_res']:.1f}% | Ind={rec['new_ind']:.1f}% | Slum={rec['new_slum']:.1f}%"
        ]

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx])
            ax.set_xlabel(subtitles[col_idx], fontsize=9.5, color="#e0e0e0", labelpad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(pid, fontsize=11, fontweight='bold', color="#ffffff", rotation=0, labelpad=70, va='center')
            for spine in ax.spines.values():
                spine.set_color('#3c3c3c')
                spine.set_linewidth(1.2)

    plt.tight_layout(rect=[0.03, 0.01, 0.99, 0.99])

    fname = "new_20_random_patches_overview.png"
    p1 = REVIEW_DIR / fname
    p2 = BRAIN_DIR / fname

    plt.savefig(p1, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(p2, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved master contact sheet: {fname}")
    return fname

def main():
    print("==================================================================")
    print("REGENERATING GROUND TRUTH USING LEARNED RULES (ALL 186 PATCHES)")
    print("==================================================================")

    all_mask_files = sorted(list(MASK_DIR.glob("*.png")))
    tot_px = 512 * 512
    records_new20 = []
    qc_stats = []

    for idx, m_file in enumerate(all_mask_files, 1):
        pid = m_file.stem
        im_file = IMG_DIR / f"{pid}.png"
        if not im_file.exists():
            continue

        im = cv2.imread(str(im_file), cv2.IMREAD_UNCHANGED)
        m_orig = cv2.imread(str(m_file), cv2.IMREAD_UNCHANGED)

        # Apply learned feature extraction
        data = extract_learned_builtup_and_subclasses(im, m_orig, pid)
        m_learned = data["mask_learned"]

        # Save to masks_v4_learned_candidate
        cv2.imwrite(str(V4_DIR / f"{pid}.png"), m_learned)
        cv2.imwrite(str(BRAIN_V4 / f"{pid}.png"), m_learned)
        
        color_bgr = cv2.cvtColor(mask_to_rgb(m_learned), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(V4_DIR / f"{pid}_color.png"), color_bgr)
        cv2.imwrite(str(BRAIN_V4 / f"{pid}_color.png"), color_bgr)

        # Calculate statistics
        orig_barren = (m_orig == 2).sum() / tot_px * 100
        orig_built = ((m_orig == 3) | (m_orig == 4) | (m_orig == 5)).sum() / tot_px * 100
        new_barren = (m_learned == 2).sum() / tot_px * 100
        new_res = (m_learned == 3).sum() / tot_px * 100
        new_ind = (m_learned == 4).sum() / tot_px * 100
        new_slum = (m_learned == 5).sum() / tot_px * 100
        new_built = new_res + new_ind + new_slum
        new_veg = (m_learned == 1).sum() / tot_px * 100
        new_road = (m_learned == 6).sum() / tot_px * 100
        new_water = (m_learned == 0).sum() / tot_px * 100
        new_unc = (m_learned == 255).sum() / tot_px * 100

        qc_stats.append({
            "patch_id": pid,
            "orig_barren_pct": round(orig_barren, 2),
            "orig_built_pct": round(orig_built, 2),
            "learned_barren_pct": round(new_barren, 2),
            "learned_residential_pct": round(new_res, 2),
            "learned_industrial_pct": round(new_ind, 2),
            "learned_slum_pct": round(new_slum, 2),
            "learned_total_built_pct": round(new_built, 2),
            "learned_vegetation_pct": round(new_veg, 2),
            "learned_road_pct": round(new_road, 2),
            "learned_water_pct": round(new_water, 2),
            "learned_uncertain_pct": round(new_unc, 2)
        })

        if pid in NEW_20_RANDOM_PATCHES:
            print(f"Generating 10-panel suite for new patch: {pid}...")
            render_10panel_comparison_new20(pid, data, m_orig)
            records_new20.append({
                "patch_id": pid,
                "rgb": data["rgb"],
                "orig_mask": m_orig,
                "mask_learned": m_learned,
                "orig_barren": orig_barren,
                "orig_built": orig_built,
                "new_barren": new_barren,
                "new_res": new_res,
                "new_ind": new_ind,
                "new_slum": new_slum,
                "new_built": new_built
            })

    # Save master overview contact sheet for new 20 patches
    print("\nBuilding master contact sheet for 20 new randomly selected evaluation patches...")
    build_new20_master_contact_sheet(records_new20)

    # Save QC statistics CSV
    df_qc = pd.DataFrame(qc_stats)
    qc_csv1 = REVIEW_DIR / "all_186_patches_learned_reannotation_stats.csv"
    qc_csv2 = BRAIN_DIR / "all_186_patches_learned_reannotation_stats.csv"
    df_qc.to_csv(qc_csv1, index=False)
    df_qc.to_csv(qc_csv2, index=False)
    print(f"Saved complete 186-patch QC stats to {qc_csv1.name}")

    print("\n==================================================================")
    print("ALL 186 PATCHES REGENERATED & NEW 20-PATCH EVALUATION SUITE READY!")
    print("==================================================================")

if __name__ == "__main__":
    main()

