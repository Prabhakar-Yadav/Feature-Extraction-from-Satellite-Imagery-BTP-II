import os
import cv2
import numpy as np

up1_p = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg"
up1 = cv2.imread(up1_p)
h, w = up1.shape[:2]

# In up1:
# Roads are yellow: (B < 80, G > 140, R > 170)
b, g, r = cv2.split(up1)
road_up1 = (b < 100) & (g > 140) & (r > 170)
# Purple is industrial: (B > 100, R > 80, G < 60)
ind_up1 = (b > 100) & (r > 80) & (g < 60)

# Crop out black margin on right if any
valid_cols = np.where(np.sum(up1, axis=(0, 2)) > 5)[0]
c_min, c_max = valid_cols[0], valid_cols[-1]
road_up1_crop = cv2.resize(road_up1[:, c_min:c_max+1].astype(np.uint8), (512, 512))
ind_up1_crop = cv2.resize(ind_up1[:, c_min:c_max+1].astype(np.uint8), (512, 512))

print(f"up1 road pixels: {road_up1_crop.sum()}, ind pixels: {ind_up1_crop.sum()}")

base_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data"
candidates = [
    "test_pune_0436", "test_pune_0564", "test_pune_0572", "test_pune_0575",
    "test_pune_0456", "test_pune_0460", "test_pune_0407", "test_pune_0516",
    "test_pune_0550", "test_pune_0424", "test_pune_0509", "test_pune_0471",
    "test_pune_0584"
]

print("\n--- Comparing candidate masks with up1 ---")
for pid in candidates:
    for ver in ["masks_v4_learned_candidate", "masks_v1_original"]:
        mf = os.path.join(base_dir, ver, f"{pid}.png")
        if not os.path.exists(mf):
            continue
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        road_m = (m == 6).astype(np.uint8)
        ind_m = (m == 4).astype(np.uint8)
        
        # Calculate road IoU
        road_inter = (road_up1_crop & road_m).sum()
        road_union = (road_up1_crop | road_m).sum()
        road_iou = road_inter / (road_union + 1e-6)
        
        # Calculate ind IoU
        ind_inter = (ind_up1_crop & ind_m).sum()
        ind_union = (ind_up1_crop | ind_m).sum()
        ind_iou = ind_inter / (ind_union + 1e-6)
        
        combined_score = road_iou * 0.5 + ind_iou * 0.5
        if combined_score > 0.05 or road_iou > 0.15 or ind_iou > 0.15:
            print(f"  {pid:<16} | {ver:<26} | Road IoU: {road_iou:.4f} | Ind IoU: {ind_iou:.4f} | Comb: {combined_score:.4f}")

