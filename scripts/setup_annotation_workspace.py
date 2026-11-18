import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
IMG_DIR = BASE_DIR / "data" / "test" / "images"
MASK_DIR = BASE_DIR / "data" / "test" / "masks"

FINAL_SUBMISSION = BASE_DIR.parent.parent / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results"
REVIEW_DIR = FINAL_SUBMISSION / "human_verification_batches" / "builtup_reannotation_review"
DATA_DIR = REVIEW_DIR / "data"

RGB_DIR = DATA_DIR / "images_rgb"
CIR_DIR = DATA_DIR / "images_cir"
V1_DIR = DATA_DIR / "masks_v1_original"
V2_DIR = DATA_DIR / "masks_v2_suggested"
V3_DIR = DATA_DIR / "masks_v3_human_verified"

for d in [RGB_DIR, CIR_DIR, V1_DIR, V2_DIR, V3_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Also create in brain artifact directory
BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review")
BRAIN_DATA = BRAIN_DIR / "data"
for sub in ["images_rgb", "images_cir", "masks_v1_original", "masks_v2_suggested", "masks_v3_human_verified"]:
    (BRAIN_DATA / sub).mkdir(parents=True, exist_ok=True)

# Import feature extractor from scripts
sys.path.append(str(Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\scripts")))
from generate_20_priority_10panel_comparisons import (
    PRIORITY_PATCHES, CLASS_COLORS, CLASS_NAMES, stretch_8bit, mask_to_rgb, extract_builtup_and_subclasses
)

def main():
    print("==================================================================")
    print("SETTING UP PERSISTENT 3-VERSION ANNOTATION WORKSPACE (20 PATCHES)")
    print("==================================================================")
    
    patches_meta = []
    tot_px = 512 * 512
    
    for idx, pid in enumerate(PRIORITY_PATCHES, 1):
        print(f"[{idx:02d}/20] Preparing assets for {pid}...")
        im_path = IMG_DIR / f"{pid}.png"
        mask_path = MASK_DIR / f"{pid}.png"
        
        im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
        m_v1 = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        # 1. Optical Bands Processing
        b, g, r, nir = im[:,:,0].astype(float), im[:,:,1].astype(float), im[:,:,2].astype(float), im[:,:,3].astype(float)
        r_str = stretch_8bit(r)
        g_str = stretch_8bit(g)
        b_str = stretch_8bit(b)
        nir_str = stretch_8bit(nir)
        
        rgb = np.stack([r_str, g_str, b_str], axis=-1)
        cir = np.stack([nir_str, r_str, g_str], axis=-1)
        
        # Save RGB & CIR (BGR order for cv2.imwrite)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cir_bgr = cv2.cvtColor(cir, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(RGB_DIR / f"{pid}_rgb.png"), rgb_bgr)
        cv2.imwrite(str(CIR_DIR / f"{pid}_cir.png"), cir_bgr)
        cv2.imwrite(str(BRAIN_DATA / "images_rgb" / f"{pid}_rgb.png"), rgb_bgr)
        cv2.imwrite(str(BRAIN_DATA / "images_cir" / f"{pid}_cir.png"), cir_bgr)
        
        # 2. Version 1: Original Ground Truth Mask
        # Save raw 8-bit mask and colorized version
        v1_color = mask_to_rgb(m_v1)
        v1_color_bgr = cv2.cvtColor(v1_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(V1_DIR / f"{pid}.png"), m_v1)
        cv2.imwrite(str(V1_DIR / f"{pid}_color.png"), v1_color_bgr)
        cv2.imwrite(str(BRAIN_DATA / "masks_v1_original" / f"{pid}.png"), m_v1)
        cv2.imwrite(str(BRAIN_DATA / "masks_v1_original" / f"{pid}_color.png"), v1_color_bgr)
        
        # 3. Version 2: Suggested Automated Mask
        data = extract_builtup_and_subclasses(im, m_v1)
        m_v2 = data["mask_prop"]
        v2_color = mask_to_rgb(m_v2)
        v2_color_bgr = cv2.cvtColor(v2_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(V2_DIR / f"{pid}.png"), m_v2)
        cv2.imwrite(str(V2_DIR / f"{pid}_color.png"), v2_color_bgr)
        cv2.imwrite(str(BRAIN_DATA / "masks_v2_suggested" / f"{pid}.png"), m_v2)
        cv2.imwrite(str(BRAIN_DATA / "masks_v2_suggested" / f"{pid}_color.png"), v2_color_bgr)
        
        # 4. Version 3: Manually Corrected & Approved Mask (Initialized with v2 as a starting draft)
        v3_path = V3_DIR / f"{pid}.png"
        if not v3_path.exists():
            cv2.imwrite(str(v3_path), m_v2)
            cv2.imwrite(str(V3_DIR / f"{pid}_color.png"), v2_color_bgr)
            cv2.imwrite(str(BRAIN_DATA / "masks_v3_human_verified" / f"{pid}.png"), m_v2)
            cv2.imwrite(str(BRAIN_DATA / "masks_v3_human_verified" / f"{pid}_color.png"), v2_color_bgr)
            
        # Class distribution calculations
        v1_dist = {int(c): round(float((m_v1 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v1_dist[255] = round(float((m_v1 == 255).sum() / tot_px * 100), 2)
        
        v2_dist = {int(c): round(float((m_v2 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v2_dist[255] = round(float((m_v2 == 255).sum() / tot_px * 100), 2)
        
        patches_meta.append({
            "patch_id": pid,
            "index": idx,
            "rgb_img": f"data/images_rgb/{pid}_rgb.png",
            "cir_img": f"data/images_cir/{pid}_cir.png",
            "v1_raw_mask": f"data/masks_v1_original/{pid}.png",
            "v1_color_mask": f"data/masks_v1_original/{pid}_color.png",
            "v2_raw_mask": f"data/masks_v2_suggested/{pid}.png",
            "v2_color_mask": f"data/masks_v2_suggested/{pid}_color.png",
            "v3_raw_mask": f"data/masks_v3_human_verified/{pid}.png",
            "v3_color_mask": f"data/masks_v3_human_verified/{pid}_color.png",
            "v1_dist": v1_dist,
            "v2_dist": v2_dist,
            "status": "Not reviewed",  # Options: Not reviewed, Currently editing, Corrections saved, Approved, Uncertain
            "last_modified": None,
            "notes": ""
        })
        
    meta_json_path1 = DATA_DIR / "patches_meta.json"
    meta_json_path2 = BRAIN_DATA / "patches_meta.json"
    
    with open(meta_json_path1, "w", encoding="utf-8") as f:
        json.dump(patches_meta, f, indent=2)
    with open(meta_json_path2, "w", encoding="utf-8") as f:
        json.dump(patches_meta, f, indent=2)
        
    print(f"\nMetadata saved to {meta_json_path1}")
    print("All 3-version assets successfully generated and linked.")

if __name__ == "__main__":
    main()

