import os
import cv2
import numpy as np
import glob

sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # Image 1
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # Image 2
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # Image 3

# Notice that in sc1, the right portion is black!
# Let's see why: let's inspect the actual values across columns in sc1
col_means_1 = np.mean(sc1, axis=(0, 2))
# Find where col_means_1 drops to near 0:
active_cols_1 = np.where(col_means_1 > 5)[0]
print(f"sc1 active cols: min {active_cols_1[0]}, max {active_cols_1[-1]} out of {sc1.shape[1]}")
crop1 = sc1[:, active_cols_1[0]:active_cols_1[-1]+1]

col_means_2 = np.mean(sc2, axis=(0, 2))
active_cols_2 = np.where(col_means_2 > 5)[0]
print(f"sc2 active cols: min {active_cols_2[0]}, max {active_cols_2[-1]} out of {sc2.shape[1]}")
crop2 = sc2[:, active_cols_2[0]:active_cols_2[-1]+1]

col_means_3 = np.mean(sc3, axis=(0, 2))
active_cols_3 = np.where(col_means_3 > 5)[0]
print(f"sc3 active cols: min {active_cols_3[0]}, max {active_cols_3[-1]} out of {sc3.shape[1]}")
crop3 = sc3[:, active_cols_3[0]:active_cols_3[-1]+1]

# Resize each cropped screenshot to 512x512
c1_512 = cv2.resize(crop1, (512, 512))
c2_512 = cv2.resize(crop2, (512, 512))
c3_512 = cv2.resize(crop3, (512, 512))

# Save resized crops so we can inspect them
out_dir = r"scripts\crop_test"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "crop1.png"), c1_512)
cv2.imwrite(os.path.join(out_dir, "crop2.png"), c2_512)
cv2.imwrite(os.path.join(out_dir, "crop3.png"), c3_512)

# Now let's compare with all 40 studio patches:
# In each patch, we have images_rgb, masks_v4_learned_candidate, masks_v1_original, masks_v3_human_verified
base_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data"
rgb_files = sorted(glob.glob(os.path.join(base_dir, "images_rgb", "*_rgb.png")))

PALETTE_BGR = {
    0: np.array([220, 120, 30], dtype=np.uint8),
    1: np.array([40, 150, 34], dtype=np.uint8),
    2: np.array([145, 185, 215], dtype=np.uint8),
    3: np.array([70, 90, 230], dtype=np.uint8),
    4: np.array([190, 50, 140], dtype=np.uint8),
    5: np.array([0, 120, 255], dtype=np.uint8),
    6: np.array([20, 225, 255], dtype=np.uint8),
    255: np.array([150, 150, 150], dtype=np.uint8)
}

def render_comp(rgb, mask, alpha=0.45):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in PALETTE_BGR.items():
        color_mask[mask == c] = col
    return cv2.addWeighted(rgb, 1.0 - alpha, color_mask, alpha, 0)

for q_idx, (q_img, q_title) in enumerate([(c1_512, "Screenshot 1"), (c2_512, "Screenshot 2"), (c3_512, "Screenshot 3")]):
    print(f"\nMatching {q_title} across all 40 patches:")
    results = []
    for f in rgb_files:
        pid = os.path.basename(f).replace("_rgb.png", "")
        rgb = cv2.imread(f)
        for ver in ["masks_v4_learned_candidate", "masks_v3_human_verified", "masks_v1_original"]:
            mf = os.path.join(base_dir, ver, f"{pid}.png")
            if not os.path.exists(mf):
                continue
            mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            comp = render_comp(rgb, mask)
            # Normalized correlation
            norm_q = (q_img.astype(np.float32) - np.mean(q_img)) / (np.std(q_img) + 1e-6)
            norm_c = (comp.astype(np.float32) - np.mean(comp)) / (np.std(comp) + 1e-6)
            corr = np.mean(norm_q * norm_c)
            results.append((corr, pid, ver))
    results.sort(reverse=True)
    for corr, pid, ver in results[:5]:
        print(f"  {pid:<16} | {ver:<28} | Pearson r: {corr:.4f}")

