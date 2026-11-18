import os
import cv2
import numpy as np
import glob

# Load snippets
sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # Snippet 1
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # Snippet 2
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # Snippet 3

# Remove black padding if any on right of sc1:
# In sc1, let's inspect width:
col_means = np.mean(sc1, axis=(0, 2))
valid_cols = np.where(col_means > 10)[0]
sc1_clean = sc1[:, valid_cols[0]:valid_cols[-1]+1]

# Same for sc2, sc3
col_means = np.mean(sc2, axis=(0, 2))
valid_cols = np.where(col_means > 10)[0]
sc2_clean = sc2[:, valid_cols[0]:valid_cols[-1]+1]

col_means = np.mean(sc3, axis=(0, 2))
valid_cols = np.where(col_means > 10)[0]
sc3_clean = sc3[:, valid_cols[0]:valid_cols[-1]+1]

print("Clean shapes:", sc1_clean.shape, sc2_clean.shape, sc3_clean.shape)

# Convert snippets to grayscale
g1 = cv2.cvtColor(sc1_clean, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(sc2_clean, cv2.COLOR_BGR2GRAY)
g3 = cv2.cvtColor(sc3_clean, cv2.COLOR_BGR2GRAY)

base_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data"
rgb_files = sorted(glob.glob(os.path.join(base_dir, "images_rgb", "*_rgb.png")))

# Let's also check color masks
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

# Multi-scale template matching:
# Since snippet might be taken at 100%, 125%, 150% display scaling (Windows default is 125% or 150%!),
# the canvas in browser might be displayed at e.g. 512px, 640px, or 768px!
def find_snippet(snippet, snip_name):
    print(f"\n================ Finding {snip_name} ================")
    h_s, w_s = snippet.shape[:2]
    best_val = -1
    best_info = None
    
    # Try different scales of snippet: 0.5 to 1.5
    for f in rgb_files:
        pid = os.path.basename(f).replace("_rgb.png", "")
        rgb = cv2.imread(f)
        
        # Test candidate mask v4 and v1
        for ver in ["masks_v4_learned_candidate", "masks_v1_original"]:
            mf = os.path.join(base_dir, ver, f"{pid}.png")
            if not os.path.exists(mf):
                continue
            mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            comp = render_comp(rgb, mask)
            comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
            
            # Match at scales
            for scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.33, 1.4, 1.5]:
                new_w = int(w_s * scale)
                new_h = int(h_s * scale)
                if new_w >= 512 or new_h >= 512 or new_w <= 30 or new_h <= 30:
                    continue
                resized_s = cv2.resize(cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY), (new_w, new_h))
                res = cv2.matchTemplate(comp_gray, resized_s, cv2.TM_CCOEFF_NORMED)
                max_v = np.max(res)
                if max_v > best_val:
                    best_val = max_v
                    best_info = (pid, ver, scale, max_v)
                    print(f"  New best: {pid} ({ver}) scale={scale:.2f} score={max_v:.4f}")
                    if max_v > 0.85:
                        break
            if best_val > 0.85:
                break
        if best_val > 0.85:
            break
    print(f"FINAL RESULT FOR {snip_name}: {best_info}")

find_snippet(sc1_clean, "Screenshot 1 / Image 1")
find_snippet(sc2_clean, "Screenshot 2 / Image 2")
find_snippet(sc3_clean, "Screenshot 3 / Image 3")

