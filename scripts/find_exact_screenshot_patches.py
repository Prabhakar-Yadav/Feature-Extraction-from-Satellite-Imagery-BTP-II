import os
import cv2
import numpy as np
import glob

# Load screenshots
sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # ~00:02
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # ~00:03
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # ~00:08

# Also check up1, up2
up1 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg")
up2 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793990007163_3831517d.jpg")

# Notice up1 is a crop of sc1, or sc1 is a crop of canvas
# Let's see: in sc1, there is a black margin on the right:
# Let's crop out the canvas area:
# In sc1: shape is (486, 546). Notice 546 is width. There is black on the right from ~486 to 546!
# 486 x 486 is a square! Exactly 1:1 aspect ratio!
print("sc1 shape:", sc1.shape)
print("sc2 shape:", sc2.shape)
print("sc3 shape:", sc3.shape)

# Let's inspect where the square canvas is in sc1, sc2, sc3:
# In sc1, let's take the first 486 columns:
canvas_sc1 = cv2.resize(sc1[:, :486], (512, 512))
# In sc2, shape is (746, 900). Let's see where the non-black pixels are:
gray2 = cv2.cvtColor(sc2, cv2.COLOR_BGR2GRAY)
coords2 = cv2.findNonZero(gray2)
x2, y2, w2, h2 = cv2.boundingRect(coords2)
print("sc2 content rect:", x2, y2, w2, h2)
canvas_sc2 = cv2.resize(sc2[y2:y2+h2, x2:x2+w2], (512, 512))

# In sc3:
gray3 = cv2.cvtColor(sc3, cv2.COLOR_BGR2GRAY)
coords3 = cv2.findNonZero(gray3)
x3, y3, w3, h3 = cv2.boundingRect(coords3)
print("sc3 content rect:", x3, y3, w3, h3)
canvas_sc3 = cv2.resize(sc3[y3:y3+h3, x3:x3+w3], (512, 512))

# Palette:
# 0: Water [220, 120, 30] (BGR)
# 1: Veg [40, 150, 34]
# 2: Barren [145, 185, 215]
# 3: Res [70, 90, 230]
# 4: Ind [190, 50, 140]
# 5: Slum [0, 120, 255]
# 6: Road [20, 225, 255]
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

def render_composite(rgb, mask, alpha=0.45):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in PALETTE_BGR.items():
        color_mask[mask == c] = col
    comp = cv2.addWeighted(rgb, 1.0 - alpha, color_mask, alpha, 0)
    return comp

base_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data"
rgb_files = glob.glob(os.path.join(base_dir, "images_rgb", "*_rgb.png"))

print(f"\nSearching across {len(rgb_files)} patches...")

def match_query(canvas_query, q_name):
    best_score = float('inf')
    best_pid = None
    best_ver = None
    
    for f in rgb_files:
        pid = os.path.basename(f).replace("_rgb.png", "")
        rgb = cv2.imread(f)
        
        # Test v1, v4, v3
        for ver in ["masks_v4_learned_candidate", "masks_v1_original", "masks_v3_human_verified"]:
            mf = os.path.join(base_dir, ver, f"{pid}.png")
            if not os.path.exists(mf):
                continue
            mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            comp = render_composite(rgb, mask)
            
            # Compute MSE
            diff = cv2.absdiff(canvas_query, comp)
            mse = np.mean(diff ** 2)
            if mse < best_score:
                best_score = mse
                best_pid = pid
                best_ver = ver
    print(f"[{q_name}] EXACT MATCH: {best_pid} | Version: {best_ver} | MSE: {best_score:.2f}")

match_query(canvas_sc1, "Screenshot 1 / Image 1")
match_query(canvas_sc2, "Screenshot 2 / Image 2")
match_query(canvas_sc3, "Screenshot 3 / Image 3")

