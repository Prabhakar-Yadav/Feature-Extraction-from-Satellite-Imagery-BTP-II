import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import datetime

# Paths
BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches")
REVIEW_DIR = BASE_DIR / "builtup_reannotation_review"
DATA_DIR = REVIEW_DIR / "data"
V3_DIR = DATA_DIR / "masks_v3_human_verified"
V2_DIR = DATA_DIR / "masks_v2_suggested"
V1_DIR = DATA_DIR / "masks_v1_original"
META_PATH = DATA_DIR / "patches_meta.json"
CSV_PATH = REVIEW_DIR / "priority_20_patches_reannotation_stats.csv"

BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review")
BRAIN_V3 = BRAIN_DIR / "data" / "masks_v3_human_verified"
BRAIN_META = BRAIN_DIR / "data" / "patches_meta.json"
BRAIN_CSV = BRAIN_DIR / "priority_20_patches_reannotation_stats.csv"

DOWNLOADS_DIR = Path(os.path.expanduser("~")) / "Downloads"

CLASS_COLORS = {
    0: (30, 120, 220),    # Water
    1: (34, 150, 40),     # Vegetation
    2: (215, 185, 145),   # Barren Land
    3: (230, 90, 70),     # Residential
    4: (140, 50, 190),    # Industrial
    5: (255, 120, 0),     # Slum / Informal
    6: (255, 225, 20),    # Road
    255: (150, 150, 150)  # Uncertain / Ignore
}

COLOR_TO_CLASS = {
    (30, 120, 220): 0,
    (34, 150, 40): 1,
    (215, 185, 145): 2,
    (230, 90, 70): 3,
    (140, 50, 190): 4,
    (255, 120, 0): 5,
    (255, 225, 20): 6,
    (150, 150, 150): 255
}

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

def color_to_categorical(img_bgr):
    rgb = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    mask = np.full((512, 512), 255, dtype=np.uint8)
    for col, cid in COLOR_TO_CLASS.items():
        match = (rgb[:, :, 0] == col[0]) & (rgb[:, :, 1] == col[1]) & (rgb[:, :, 2] == col[2])
        mask[match] = cid
    return mask

def main():
    print("==================================================================")
    print("INTEGRATING USER'S MANUAL EDITS FROM DOWNLOADS INTO PROJECT")
    print("==================================================================")
    
    # 1. Check for exported files in Downloads
    user_files = list(DOWNLOADS_DIR.glob("*test_pune*.png"))
    print(f"Found {len(user_files)} exported masks in Downloads:")
    for f in user_files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")
        
    for f in user_files:
        # e.g. test_pune_0505_human_verified.png
        pid = f.stem.replace("_human_verified", "").replace("_color", "")
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
            
        print(f"\nProcessing manual annotation for {pid}...")
        if len(img.shape) == 3:
            mask_raw = color_to_categorical(img)
        else:
            mask_raw = img
            
        tot_px = 512 * 512
        dist = {int(c): round(float((mask_raw == c).sum() / tot_px * 100), 2) for c in range(7)}
        dist[255] = round(float((mask_raw == 255).sum() / tot_px * 100), 2)
        
        print(f"  Class breakdown: Res={dist[3]}%, Veg={dist[1]}%, Barren={dist[2]}%, Ind={dist[4]}%, Slum={dist[5]}%, Unc={dist[255]}%")
        
        # Save raw 8-bit mask and colorized mask
        cv2.imwrite(str(V3_DIR / f"{pid}.png"), mask_raw)
        cv2.imwrite(str(BRAIN_V3 / f"{pid}.png"), mask_raw)
        
        color_bgr = cv2.cvtColor(mask_to_rgb(mask_raw), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(V3_DIR / f"{pid}_color.png"), color_bgr)
        cv2.imwrite(str(BRAIN_V3 / f"{pid}_color.png"), color_bgr)
        
        # Update patches_meta.json
        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as meta_f:
                meta = json.load(meta_f)
            for item in meta:
                if item["patch_id"] == pid:
                    item["status"] = "Approved"
                    item["last_modified"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    item["notes"] = "Manually verified & exported by user"
                    item["v3_dist"] = dist
            with open(META_PATH, "w", encoding="utf-8") as meta_f:
                json.dump(meta, meta_f, indent=2)
            with open(BRAIN_META, "w", encoding="utf-8") as meta_f:
                json.dump(meta, meta_f, indent=2)
                
        # Update priority stats CSV
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            idx = df[df["patch_id"] == pid].index
            if len(idx) > 0:
                df.loc[idx[0], "review_status"] = "Approved"
                df.loc[idx[0], "proposed_barren_pct"] = dist[2]
                df.loc[idx[0], "proposed_residential_pct"] = dist[3]
                df.loc[idx[0], "proposed_industrial_pct"] = dist[4]
                df.loc[idx[0], "proposed_slum_pct"] = dist[5]
                df.loc[idx[0], "proposed_total_builtup_pct"] = round(dist[3] + dist[4] + dist[5], 2)
                df.loc[idx[0], "proposed_uncertain_pct"] = dist[255]
                df.to_csv(CSV_PATH, index=False)
                df.to_csv(BRAIN_CSV, index=False)
                
        print(f"  Successfully integrated user's manual ground truth for {pid}!")

if __name__ == "__main__":
    main()

