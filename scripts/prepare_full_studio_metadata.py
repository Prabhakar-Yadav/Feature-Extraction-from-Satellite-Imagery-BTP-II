import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

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
V4_DIR = DATA_DIR / "masks_v4_learned_candidate"

BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review")
BRAIN_DATA = BRAIN_DIR / "data"

sys.path.append(str(Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\scripts")))
from generate_learned_ground_truth_and_new20 import (
    stretch_8bit, mask_to_rgb, NEW_20_RANDOM_PATCHES
)
from generate_20_priority_10panel_comparisons import PRIORITY_PATCHES

def main():
    print("==================================================================")
    print("PREPARING FULL 40-PATCH STUDIO METADATA & ASSETS")
    print("==================================================================")

    tot_px = 512 * 512
    all_meta = []

    # 1. Process Priority 20 Patches
    for idx, pid in enumerate(PRIORITY_PATCHES, 1):
        m_v1 = cv2.imread(str(V1_DIR / f"{pid}.png"), cv2.IMREAD_UNCHANGED)
        m_v2 = cv2.imread(str(V2_DIR / f"{pid}.png"), cv2.IMREAD_UNCHANGED)
        m_v3 = cv2.imread(str(V3_DIR / f"{pid}.png"), cv2.IMREAD_UNCHANGED)
        m_v4 = cv2.imread(str(V4_DIR / f"{pid}.png"), cv2.IMREAD_UNCHANGED)

        v1_dist = {int(c): round(float((m_v1 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v1_dist[255] = round(float((m_v1 == 255).sum() / tot_px * 100), 2)

        v4_dist = {int(c): round(float((m_v4 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v4_dist[255] = round(float((m_v4 == 255).sum() / tot_px * 100), 2)

        status = "Approved" if pid == "test_pune_0505" else "Not reviewed"

        all_meta.append({
            "patch_id": pid,
            "group": "Priority 20 Patches (Pune SW Void)",
            "index": idx,
            "rgb_img": f"data/images_rgb/{pid}_rgb.png",
            "cir_img": f"data/images_cir/{pid}_cir.png",
            "v1_raw_mask": f"data/masks_v1_original/{pid}.png",
            "v1_color_mask": f"data/masks_v1_original/{pid}_color.png",
            "v2_raw_mask": f"data/masks_v2_suggested/{pid}.png",
            "v2_color_mask": f"data/masks_v2_suggested/{pid}_color.png",
            "v3_raw_mask": f"data/masks_v3_human_verified/{pid}.png",
            "v3_color_mask": f"data/masks_v3_human_verified/{pid}_color.png",
            "v4_raw_mask": f"data/masks_v4_learned_candidate/{pid}.png",
            "v4_color_mask": f"data/masks_v4_learned_candidate/{pid}_color.png",
            "v1_dist": v1_dist,
            "v4_dist": v4_dist,
            "status": status,
            "notes": "User manual edits integrated" if pid == "test_pune_0505" else ""
        })

    # 2. Process New 20 Random Evaluation Patches
    for idx, pid in enumerate(NEW_20_RANDOM_PATCHES, 1):
        im_path = IMG_DIR / f"{pid}.png"
        mask_path = MASK_DIR / f"{pid}.png"

        im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
        m_v1 = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        m_v4 = cv2.imread(str(V4_DIR / f"{pid}.png"), cv2.IMREAD_UNCHANGED)

        # Generate RGB & CIR if not exist
        rgb_out = RGB_DIR / f"{pid}_rgb.png"
        if not rgb_out.exists():
            b, g, r, nir = im[:,:,0].astype(float), im[:,:,1].astype(float), im[:,:,2].astype(float), im[:,:,3].astype(float)
            r_str = stretch_8bit(r)
            g_str = stretch_8bit(g)
            b_str = stretch_8bit(b)
            nir_str = stretch_8bit(nir)
            rgb = np.stack([r_str, g_str, b_str], axis=-1)
            cir = np.stack([nir_str, r_str, g_str], axis=-1)
            cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(CIR_DIR / f"{pid}_cir.png"), cv2.cvtColor(cir, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(BRAIN_DATA / "images_rgb" / f"{pid}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(BRAIN_DATA / "images_cir" / f"{pid}_cir.png"), cv2.cvtColor(cir, cv2.COLOR_RGB2BGR))

        # Generate v1 color if not exist
        v1_color_out = V1_DIR / f"{pid}_color.png"
        if not v1_color_out.exists():
            cv2.imwrite(str(V1_DIR / f"{pid}.png"), m_v1)
            cv2.imwrite(str(v1_color_out), cv2.cvtColor(mask_to_rgb(m_v1), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(BRAIN_DATA / "masks_v1_original" / f"{pid}.png"), m_v1)
            cv2.imwrite(str(BRAIN_DATA / "masks_v1_original" / f"{pid}_color.png"), cv2.cvtColor(mask_to_rgb(m_v1), cv2.COLOR_RGB2BGR))

        # Initialize v3 as copy of v4
        v3_raw_out = V3_DIR / f"{pid}.png"
        if not v3_raw_out.exists():
            cv2.imwrite(str(v3_raw_out), m_v4)
            cv2.imwrite(str(V3_DIR / f"{pid}_color.png"), cv2.cvtColor(mask_to_rgb(m_v4), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(BRAIN_DATA / "masks_v3_human_verified" / f"{pid}.png"), m_v4)
            cv2.imwrite(str(BRAIN_DATA / "masks_v3_human_verified" / f"{pid}_color.png"), cv2.cvtColor(mask_to_rgb(m_v4), cv2.COLOR_RGB2BGR))

        v1_dist = {int(c): round(float((m_v1 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v1_dist[255] = round(float((m_v1 == 255).sum() / tot_px * 100), 2)

        v4_dist = {int(c): round(float((m_v4 == c).sum() / tot_px * 100), 2) for c in range(7)}
        v4_dist[255] = round(float((m_v4 == 255).sum() / tot_px * 100), 2)

        all_meta.append({
            "patch_id": pid,
            "group": "New 20 Evaluation Patches (Generalization Test)",
            "index": 20 + idx,
            "rgb_img": f"data/images_rgb/{pid}_rgb.png",
            "cir_img": f"data/images_cir/{pid}_cir.png",
            "v1_raw_mask": f"data/masks_v1_original/{pid}.png",
            "v1_color_mask": f"data/masks_v1_original/{pid}_color.png",
            "v2_raw_mask": f"data/masks_v4_learned_candidate/{pid}.png",
            "v2_color_mask": f"data/masks_v4_learned_candidate/{pid}_color.png",
            "v3_raw_mask": f"data/masks_v3_human_verified/{pid}.png",
            "v3_color_mask": f"data/masks_v3_human_verified/{pid}_color.png",
            "v4_raw_mask": f"data/masks_v4_learned_candidate/{pid}.png",
            "v4_color_mask": f"data/masks_v4_learned_candidate/{pid}_color.png",
            "v1_dist": v1_dist,
            "v4_dist": v4_dist,
            "status": "Not reviewed",
            "notes": "Learned candidate generation"
        })

    with open(DATA_DIR / "patches_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    with open(BRAIN_DATA / "patches_meta.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    print(f"Saved {len(all_meta)} patches metadata to patches_meta.json (20 Priority + 20 New Evaluation).")

if __name__ == "__main__":
    main()

