import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
SCENE_PUNE = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Test_Data\2673727101")
IMG_DIR = BASE_DIR / "data" / "test" / "images"
MASK_DIR = BASE_DIR / "data" / "test" / "masks"
OUT_REVIEWS_DIR = BASE_DIR.parent.parent / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results" / "builtup_reannotation_review"
OUT_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta / Coral Red
    4: (140, 50, 190),    # Industrial: Steel Violet / Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
    255: (150, 150, 150)  # Uncertain / Ignore: Neutral Gray
}

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

def compute_mbi(gray):
    """
    Morphological Building Index (MBI) for high-resolution optical imagery.
    Computes multi-directional and multi-scale morphological top-hat transformations.
    """
    mbi = np.zeros_like(gray, dtype=float)
    scales = [7, 13, 21, 29]
    for L in scales:
        # 0 degree (horizontal)
        se_0 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 3))
        # 90 degree (vertical)
        se_90 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, L))
        # 45 degree (diagonal)
        se_45 = np.eye(L, dtype=np.uint8)
        # 135 degree (anti-diagonal)
        se_135 = np.fliplr(np.eye(L, dtype=np.uint8))
        
        for se in [se_0, se_90, se_45, se_135]:
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
            bothat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
            mbi += tophat.astype(float) + 0.6 * bothat.astype(float)
            
    return mbi / (len(scales) * 4.0)

def extract_patch_features(im, m):
    """
    Extracts multispectral and spatial features, identifies candidate building footprints,
    and categorizes them into Residential (3), Industrial (4), Slum (5), Barren (2), and Uncertain (255).
    """
    b, g, r, nir = im[:,:,0].astype(float), im[:,:,1].astype(float), im[:,:,2].astype(float), im[:,:,3].astype(float)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Spectral Indices
    denom_ndvi = nir + r + 1e-6
    ndvi = (nir - r) / denom_ndvi
    
    # True vegetation
    veg_mask = ndvi > 0.22
    
    # Existing road corridors (from ground truth mask)
    road_mask = (m == 6)
    
    # Candidate area for built-up classification: non-vegetation, non-road
    avail_mask = (~veg_mask) & (~road_mask)
    
    # 2. Texture & Morphological Building Index
    mbi = compute_mbi(gray)
    
    # Local standard deviation (texture roughness)
    mean_sq = cv2.boxFilter(gray.astype(float)**2, -1, (13, 13))
    sq_mean = cv2.boxFilter(gray.astype(float), -1, (13, 13))**2
    local_std = np.sqrt(np.maximum(0, mean_sq - sq_mean))
    
    # 3. Building Cast Shadows (adjacent to building walls)
    shadow_mask = (gray < 45) & (nir < 60)
    kernel_shadow = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated_shadow = cv2.dilate(shadow_mask.astype(np.uint8), kernel_shadow) > 0
    
    # 4. Building candidate logic:
    # Roofs have high MBI OR (moderate MBI + elevated local texture + adjacent shadow)
    # Bright metal/concrete roofs: (gray > 130) & (local_std > 8)
    bright_roof = (gray > 130) & (local_std > 6.0) & avail_mask
    textured_bldg = (mbi > 10.0) | ((mbi > 5.0) & ((local_std > 12.0) | dilated_shadow))
    textured_bldg = textured_bldg & avail_mask
    
    raw_bldg = bright_roof | textured_bldg
    
    # Morphological filtering: remove tiny 1-2 pixel isolated salt-noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_bldg = cv2.morphologyEx(raw_bldg.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    cleaned_bldg = cv2.morphologyEx(cleaned_bldg, cv2.MORPH_CLOSE, kernel_open) > 0
    
    # 5. Connected Component Analysis for Subclass Typing
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_bldg.astype(np.uint8))
    
    # Target masks
    res_mask = np.zeros((512, 512), dtype=bool)
    ind_mask = np.zeros((512, 512), dtype=bool)
    slum_mask = np.zeros((512, 512), dtype=bool)
    uncertain_mask = np.zeros((512, 512), dtype=bool)
    
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        comp = (labels == lbl)
        comp_gray = gray[comp]
        comp_std = comp_gray.std()
        comp_mean = comp_gray.mean()
        
        # Tiny components (< 35 pixels ~ 12 m2) -> boundary uncertainty / terrain transition
        if area < 35:
            uncertain_mask[comp] = True
            continue
            
        # Industrial Built-up (Class 4):
        # Large footprint (>1100 px ~ 400 m2), homogeneous roof (low internal std), elongated or warehouse shape
        if area > 1100 and comp_std < 26.0:
            ind_mask[comp] = True
        # Slum / Informal Settlement (Class 5):
        # Small footprint, irregular, high texture roughness in neighborhood (>18), dense clustering
        elif area < 180 and local_std[comp].mean() > 18.5 and comp_std > 20.0:
            slum_mask[comp] = True
        # Residential Built-up (Class 3):
        # Standard residential buildings, apartments, colony houses
        else:
            res_mask[comp] = True
            
    # 6. Barren Land: Remaining available ground with smooth texture (soil, fallow clearings)
    barren_mask = avail_mask & (~res_mask) & (~ind_mask) & (~slum_mask) & (~uncertain_mask)
    
    # 7. Construct Proposed 7-Class Mask
    # Preserve existing 0 (Water), 1 (Vegetation), 6 (Road Network)
    mask_prop = np.zeros((512, 512), dtype=np.uint8)
    mask_prop[:] = 2 # Barren Land base
    
    mask_prop[veg_mask] = 1
    mask_prop[barren_mask] = 2
    mask_prop[res_mask] = 3
    mask_prop[ind_mask] = 4
    mask_prop[slum_mask] = 5
    mask_prop[road_mask] = 6
    mask_prop[uncertain_mask] = 255 # Uncertain / Ignore
    
    # If original mask had water (e.g. river corridor), preserve it!
    mask_prop[m == 0] = 0
    
    return {
        "rgb": rgb,
        "gray": gray,
        "ndvi": ndvi,
        "mbi": mbi,
        "local_std": local_std,
        "veg_mask": veg_mask,
        "road_mask": road_mask,
        "barren_mask": barren_mask,
        "res_mask": res_mask,
        "ind_mask": ind_mask,
        "slum_mask": slum_mask,
        "uncertain_mask": uncertain_mask,
        "bldg_candidate": cleaned_bldg,
        "mask_prop": mask_prop
    }

if __name__ == "__main__":
    print("Module loaded.")

