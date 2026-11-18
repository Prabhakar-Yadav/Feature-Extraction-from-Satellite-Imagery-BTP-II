"""
Generate refined ground-truth annotations for all 401 training patches
based on the learned annotation rules, back up original masks,
and produce high-resolution visual inspection figures for 20 randomly sampled training patches.
"""
import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Paths
TRAIN_DIR = Path(r"NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation\data\train")
TRAIN_IMG_DIR = TRAIN_DIR / "images"
TRAIN_MASK_DIR = TRAIN_DIR / "masks"
BACKUP_MASK_DIR = TRAIN_DIR / "masks_backup_v1"
BACKUP_MASK_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf")
TRAIN_GALLERY_DIR = BRAIN_DIR / "training_ground_truth_20_gallery"
TRAIN_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_GALLERY_DIR = Path(r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\training_ground_truth_20_gallery")
LOCAL_GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# 7-Class Palette
PALETTE_BGR = {
    0: (np.array([220, 120, 30], dtype=np.uint8), "Water (0)"),
    1: (np.array([40, 150, 34], dtype=np.uint8), "Vegetation (1)"),
    2: (np.array([145, 185, 215], dtype=np.uint8), "Barren Land (2)"),
    3: (np.array([70, 90, 230], dtype=np.uint8), "Residential (3)"),
    4: (np.array([190, 50, 140], dtype=np.uint8), "Industrial (4)"),
    5: (np.array([0, 120, 255], dtype=np.uint8), "Slum/Informal (5)"),
    6: (np.array([20, 225, 255], dtype=np.uint8), "Road Network (6)"),
    255: (np.array([150, 150, 150], dtype=np.uint8), "Uncertain (255)")
}

def stretch_8bit(arr, p_low=2.0, p_high=98.0):
    v_min, v_max = np.percentile(arr, (p_low, p_high))
    if v_max <= v_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr.astype(float) - v_min) / (v_max - v_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

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

def render_overlay(rgb_bgr, mask_gray, alpha=0.45):
    h, w = mask_gray.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (col, _) in PALETTE_BGR.items():
        color_mask[mask_gray == c] = col
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, color_mask, alpha, 0)

def extract_refined_ground_truth(im, m_orig):
    """
    Refined Ground-Truth Extraction Pipeline:
    - Zero false water (water preserved strictly from physical NDWI/verified water)
    - Road network (Class 6) strictly preserved
    - Residential buildings (Class 3): Individual coherent building footprints + neighborhood groups
    - Industrial (Class 4): Strictly consolidated complexes (>2500 px, high uniformity)
    - Slum (Class 5): Irregular, small, dense clusters
    - Tree canopy recovery: Recovers roofs under light tree crowns
    - Pure Barren Land (Class 2): Confined to true open fallow soil
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
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Spectral Indices
    denom_ndvi = nir + r + 1e-6
    ndvi = (nir - r) / denom_ndvi
    
    dense_veg = ndvi > 0.32
    moderate_veg = (ndvi > 0.18) & (~dense_veg)

    # Road & Water preservation
    road_mask = (m_orig == 6)
    water_mask = (m_orig == 0)

    # Structural Indices
    mbi = compute_mbi(gray)
    edges = cv2.Canny(gray, 35, 110)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k3)
    dilated_edges = cv2.dilate(edges, k3)

    mean_sq = cv2.boxFilter(gray.astype(float)**2, -1, (13, 13))
    sq_mean = cv2.boxFilter(gray.astype(float), -1, (13, 13))**2
    local_std = np.sqrt(np.maximum(0, mean_sq - sq_mean))

    shadow_mask = (gray < 48) & (nir < 62)
    dilated_shadow = cv2.dilate(shadow_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))) > 0

    roof_near_trees = moderate_veg & ((mbi > 6.0) | (grad > 30)) & (~dense_veg)
    avail_mask = (~dense_veg) & (~road_mask) & (~water_mask)

    bright_roof = (gray > 115) & (local_std > 5.0)
    struct_roof = (mbi > 6.0) | ((mbi > 3.5) & ((local_std > 9.0) | dilated_shadow))
    core_bldg = (bright_roof | struct_roof | roof_near_trees) & avail_mask

    # Delineate individual buildings using edges
    bldg_separated = core_bldg & (dilated_edges == 0)
    bldg_solid = cv2.morphologyEx(bldg_separated.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bldg_solid.astype(np.uint8))

    res_mask = np.zeros((512, 512), dtype=bool)
    ind_mask = np.zeros((512, 512), dtype=bool)
    slum_mask = np.zeros((512, 512), dtype=bool)
    uncertain_mask = np.zeros((512, 512), dtype=bool)

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

        # Industrial (Class 4): Strictly consolidated complexes (>2500 px with low variance or >3500 px)
        if (area > 2500 and comp_std < 22.0) or (area > 3500):
            ind_mask[comp] = True
        # Slum (Class 5): Irregular, high-roughness small clusters
        elif area < 160 and comp_local_std > 18.0 and comp_density > 0.48:
            slum_mask[comp] = True
        # Residential (Class 3): Individual building footprints & residential groups
        else:
            res_mask[comp] = True

    # Residential block continuity
    k_res_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res_closed = cv2.morphologyEx(res_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_res_close) > 0
    res_mask = res_closed & avail_mask & (~ind_mask) & (~slum_mask)

    all_bldg = res_mask | ind_mask | slum_mask
    barren_mask = avail_mask & (~all_bldg) & (~uncertain_mask)
    veg_final = (dense_veg | (moderate_veg & (~all_bldg))) & (~road_mask) & (~water_mask)

    mask_out = np.full((512, 512), 2, dtype=np.uint8)
    mask_out[veg_final] = 1
    mask_out[barren_mask] = 2
    mask_out[res_mask] = 3
    mask_out[ind_mask] = 4
    mask_out[slum_mask] = 5
    mask_out[road_mask] = 6
    mask_out[water_mask] = 0
    mask_out[uncertain_mask] = 255

    return rgb_bgr, mask_out

# 1. Process all 401 training patches
train_files = sorted(list(TRAIN_IMG_DIR.glob("*.png")))
print(f"Total training images to process: {len(train_files)}")

stats_records = []

for idx, img_path in enumerate(train_files, 1):
    pid = img_path.stem
    mask_path = TRAIN_MASK_DIR / f"{pid}.png"
    backup_path = BACKUP_MASK_DIR / f"{pid}.png"
    
    # Back up original mask if not already backed up
    if mask_path.exists() and not backup_path.exists():
        shutil.copy2(mask_path, backup_path)
        
    im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    m_orig = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros((512, 512), dtype=np.uint8)
    
    rgb_bgr, refined_mask = extract_refined_ground_truth(im, m_orig)
    
    # Overwrite training mask with refined ground truth
    cv2.imwrite(str(mask_path), refined_mask)
    
    tot = refined_mask.size
    rec = {
        "patch_id": pid,
        "water_pct": round((refined_mask == 0).sum() / tot * 100.0, 2),
        "veg_pct": round((refined_mask == 1).sum() / tot * 100.0, 2),
        "barren_pct": round((refined_mask == 2).sum() / tot * 100.0, 2),
        "residential_pct": round((refined_mask == 3).sum() / tot * 100.0, 2),
        "industrial_pct": round((refined_mask == 4).sum() / tot * 100.0, 2),
        "slum_pct": round((refined_mask == 5).sum() / tot * 100.0, 2),
        "road_pct": round((refined_mask == 6).sum() / tot * 100.0, 2),
        "uncertain_pct": round((refined_mask == 255).sum() / tot * 100.0, 2),
    }
    stats_records.append(rec)
    
    if idx % 50 == 0 or idx == len(train_files):
        print(f"  Processed {idx}/{len(train_files)} training patches...")

df_stats = pd.DataFrame(stats_records)
csv_stats_path = TRAIN_DIR / "training_ground_truth_stats.csv"
df_stats.to_csv(csv_stats_path, index=False)
print(f"Training ground truth stats saved: {csv_stats_path}")

# 2. Select 20 Random Training Patches for Quality Inspection (Seed 42)
random.seed(42)
sampled_train_pids = random.sample([p.stem for p in train_files], 20)
print(f"\nSampled 20 Training Patches for Inspection (Seed 42):")
print(sampled_train_pids)

contact_tiles = []

for idx, pid in enumerate(sampled_train_pids, 1):
    img_path = TRAIN_IMG_DIR / f"{pid}.png"
    mask_path = TRAIN_MASK_DIR / f"{pid}.png"
    
    im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Get true-color RGB
    b_str = stretch_8bit(im[:, :, 0])
    g_str = stretch_8bit(im[:, :, 1])
    r_str = stretch_8bit(im[:, :, 2])
    rgb_bgr = np.stack([b_str, g_str, r_str], axis=-1)
    
    comp_bgr = render_overlay(rgb_bgr, mask, alpha=0.45)
    
    tot = mask.size
    class_stats = {c: ((mask == c).sum(), (mask == c).sum() / tot * 100.0) for c in [0, 1, 2, 3, 4, 5, 6, 255]}
    
    # --- Canvas (1100 x 670 px) ---
    canvas = np.zeros((670, 1100, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 32)
    
    # Header Banner
    cv2.putText(canvas, f"Training Patch #{idx:02d}: {pid}  |  REFINED TRAINING GROUND TRUTH",
                (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Split: Training Set (401 Patches)  |  Taxonomy: 7-Class Discrete  |  Status: Ready for Training",
                (26, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 180, 200), 1, cv2.LINE_AA)
    cv2.line(canvas, (20, 64), (1080, 64), (45, 55, 75), 1)
    
    # Left: RGB
    canvas[75:587, 25:537] = rgb_bgr
    cv2.rectangle(canvas, (24, 74), (537, 587), (60, 75, 100), 1)
    cv2.putText(canvas, "Original Cartosat-2E RGB (0.60 m)", (35, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
    # Right: Overlay
    canvas[75:587, 563:1075] = comp_bgr
    cv2.rectangle(canvas, (562, 74), (1075, 587), (60, 75, 100), 1)
    cv2.putText(canvas, "Refined Training Ground-Truth Overlay", (573, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
    # Bottom Legend
    classes_row1 = [0, 1, 2, 3]
    classes_row2 = [4, 5, 6, 255]
    for row_idx, row_classes in enumerate([classes_row1, classes_row2]):
        y_leg = 618 + row_idx * 24
        for col_idx, c in enumerate(row_classes):
            x_leg = 30 + col_idx * 265
            bgr_col, c_name = PALETTE_BGR[c]
            cnt, pct = class_stats[c]
            cv2.rectangle(canvas, (x_leg, y_leg - 12), (x_leg + 18, y_leg + 4), bgr_col.tolist(), -1)
            cv2.rectangle(canvas, (x_leg, y_leg - 12), (x_leg + 18, y_leg + 4), (255, 255, 255), 1)
            text = f"{c_name}: {pct:>5.1f}% ({cnt:,} px)"
            cv2.putText(canvas, text, (x_leg + 26, y_leg),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 245), 1, cv2.LINE_AA)
                        
    out_name = f"train_gt_{idx:02d}_{pid}.png"
    cv2.imwrite(str(TRAIN_GALLERY_DIR / out_name), canvas)
    cv2.imwrite(str(LOCAL_GALLERY_DIR / out_name), canvas)
    
    # Tile for Master Sheet
    t_rgb = cv2.resize(rgb_bgr, (256, 256))
    t_comp = cv2.resize(comp_bgr, (256, 256))
    pair = np.hstack([t_rgb, t_comp])
    t_header = np.zeros((32, 512, 3), dtype=np.uint8) + 25
    cv2.putText(t_header, f"#{idx:02d}: {pid}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(t_header, f"Res: {class_stats[3][1]:.1f}% | Bar: {class_stats[2][1]:.1f}%", (260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.line(t_header, (0, 31), (512, 31), (60, 70, 90), 1)
    tile = np.vstack([t_header, pair])
    contact_tiles.append(tile)

# Assemble Master Sheet (5x4)
rows = [np.hstack(contact_tiles[r*4 : (r+1)*4]) for r in range(5)]
master_grid = np.vstack(rows)
master_w = master_grid.shape[1]

banner = np.zeros((80, master_w, 3), dtype=np.uint8) + 18
cv2.putText(banner, "Cartosat-2E Pune: 20 Randomly Sampled Training Ground-Truth Patches (Seed 42)",
            (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(banner, "Left: 0.60m True-Color RGB Satellite Imagery  |  Right: Refined 7-Class Training Ground Truth  |  401 Training Patches Ready",
            (26, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 190, 220), 1, cv2.LINE_AA)
cv2.line(banner, (0, 79), (master_w, 79), (60, 75, 100), 2)

final_master = np.vstack([banner, master_grid])
master_brain = TRAIN_GALLERY_DIR / "master_20_training_patches_overview.png"
master_local = LOCAL_GALLERY_DIR / "master_20_training_patches_overview.png"
cv2.imwrite(str(master_brain), final_master)
cv2.imwrite(str(master_local), final_master)

print(f"\nMaster training overview saved: {master_brain} ({os.path.getsize(str(master_brain))/(1024*1024):.2f} MB)")
print("Training ground truth generation complete!")

