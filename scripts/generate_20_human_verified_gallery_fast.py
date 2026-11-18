"""
Ultra-fast, headless generator for 20 human-verified ground-truth patch visualizations
and a master 20-patch contact sheet using pure OpenCV and NumPy.
"""
import os
import glob
import datetime
import cv2
import numpy as np

BASE_DIR = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review"
RGB_DIR = os.path.join(BASE_DIR, "data", "images_rgb")
V3_DIR = os.path.join(BASE_DIR, "data", "masks_v3_human_verified")

BRAIN_DIR = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf"
GALLERY_DIR = os.path.join(BRAIN_DIR, "human_verified_20_gallery")
LOCAL_GALLERY_DIR = os.path.join(BASE_DIR, "human_verified_20_gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(LOCAL_GALLERY_DIR, exist_ok=True)

PATCH_LIST = [
    "test_pune_0407", "test_pune_0408", "test_pune_0409", "test_pune_0423",
    "test_pune_0429", "test_pune_0456", "test_pune_0458", "test_pune_0460",
    "test_pune_0471", "test_pune_0505", "test_pune_0506", "test_pune_0507",
    "test_pune_0516", "test_pune_0538", "test_pune_0550", "test_pune_0560",
    "test_pune_0564", "test_pune_0572", "test_pune_0575", "test_pune_0584"
]

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

def render_overlay(rgb_bgr, mask_gray, alpha=0.45):
    h, w = mask_gray.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, (col, _) in PALETTE_BGR.items():
        color_mask[mask_gray == c] = col
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, color_mask, alpha, 0)

print("Starting generation of 20 patch visualizations...")
contact_tiles = []

for idx, pid in enumerate(PATCH_LIST, 1):
    rgb_file = os.path.join(RGB_DIR, f"{pid}_rgb.png")
    mask_file = os.path.join(V3_DIR, f"{pid}.png")
    
    if not os.path.exists(mask_file):
        print(f"Skipping {pid}, mask not found")
        continue
        
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(mask_file))
    rgb_bgr = cv2.imread(rgb_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    comp_bgr = render_overlay(rgb_bgr, mask, alpha=0.45)
    
    # Calculate stats
    tot = mask.size
    class_stats = {}
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        cnt = (mask == c).sum()
        pct = cnt / tot * 100.0
        class_stats[c] = (cnt, pct)
        
    # --- 1. Build Individual Visualization Canvas (1100 x 670 px) ---
    canvas = np.zeros((670, 1100, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 32) # Dark background
    
    # Header Banner (y: 0 to 65)
    cv2.putText(canvas, f"Patch #{idx:02d}: {pid}  |  HUMAN-CORRECTED GROUND TRUTH (Version 3)",
                (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')}  |  Status: Verified & Approved  |  Training: Halted",
                (26, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 180, 200), 1, cv2.LINE_AA)
    cv2.line(canvas, (20, 64), (1080, 64), (45, 55, 75), 1)
    
    # Left: RGB image (x: 25 to 537, y: 75 to 587)
    canvas[75:587, 25:537] = rgb_bgr
    cv2.rectangle(canvas, (24, 74), (537, 587), (60, 75, 100), 1)
    cv2.putText(canvas, "Original Cartosat-2E RGB (0.60 m)", (35, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
    # Right: Ground-Truth Overlay (x: 563 to 1075, y: 75 to 587)
    canvas[75:587, 563:1075] = comp_bgr
    cv2.rectangle(canvas, (562, 74), (1075, 587), (60, 75, 100), 1)
    cv2.putText(canvas, "Human-Corrected Ground-Truth Overlay", (573, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
    # Bottom Legend Bar (y: 595 to 665)
    # 2 rows of 4 legend items
    classes_row1 = [0, 1, 2, 3]
    classes_row2 = [4, 5, 6, 255]
    
    for row_idx, row_classes in enumerate([classes_row1, classes_row2]):
        y_leg = 618 + row_idx * 24
        for col_idx, c in enumerate(row_classes):
            x_leg = 30 + col_idx * 265
            bgr_col, c_name = PALETTE_BGR[c]
            cnt, pct = class_stats[c]
            
            # Swatch box
            cv2.rectangle(canvas, (x_leg, y_leg - 12), (x_leg + 18, y_leg + 4), bgr_col.tolist(), -1)
            cv2.rectangle(canvas, (x_leg, y_leg - 12), (x_leg + 18, y_leg + 4), (255, 255, 255), 1)
            
            # Label
            text = f"{c_name}: {pct:>5.1f}% ({cnt:,} px)"
            cv2.putText(canvas, text, (x_leg + 26, y_leg),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 245), 1, cv2.LINE_AA)
                        
    out_name = f"gt_{idx:02d}_{pid}.png"
    cv2.imwrite(os.path.join(GALLERY_DIR, out_name), canvas)
    cv2.imwrite(os.path.join(LOCAL_GALLERY_DIR, out_name), canvas)
    
    # --- 2. Build Tile for Master Contact Sheet ---
    # Downscale pair to 256x256 each -> 512x256 + 32px header
    t_rgb = cv2.resize(rgb_bgr, (256, 256))
    t_comp = cv2.resize(comp_bgr, (256, 256))
    pair = np.hstack([t_rgb, t_comp])
    
    t_header = np.zeros((32, 512, 3), dtype=np.uint8) + 25
    cv2.putText(t_header, f"#{idx:02d}: {pid}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    # Add small stats text
    res_pct = class_stats[3][1]
    bar_pct = class_stats[2][1]
    cv2.putText(t_header, f"Res: {res_pct:.1f}% | Bar: {bar_pct:.1f}%", (260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.line(t_header, (0, 31), (512, 31), (60, 70, 90), 1)
    
    tile = np.vstack([t_header, pair])
    contact_tiles.append(tile)
    print(f"  [{idx:02d}/20] Generated {out_name}")

# Build Master Contact Sheet (5 rows of 4 tiles = 2048 x 1440 px + 80px master banner)
print("Assembling master 20-patch contact sheet...")
rows = []
for r in range(5):
    row_tiles = contact_tiles[r*4 : (r+1)*4]
    rows.append(np.hstack(row_tiles))
master_grid = np.vstack(rows)

master_w = master_grid.shape[1]
banner = np.zeros((80, master_w, 3), dtype=np.uint8) + 18
cv2.putText(banner, "Cartosat-2E Pune: 20 Human-Verified Ground-Truth Patches (Version 3)",
            (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(banner, "Left: 0.60m True-Color RGB Satellite Imagery  |  Right: Human-Approved 7-Class Annotation Overlay  |  Training: Strictly Halted",
            (26, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 190, 220), 1, cv2.LINE_AA)
cv2.line(banner, (0, 79), (master_w, 79), (60, 75, 100), 2)

final_sheet = np.vstack([banner, master_grid])
master_brain = os.path.join(GALLERY_DIR, "master_20_human_verified_overview.png")
master_local = os.path.join(LOCAL_GALLERY_DIR, "master_20_human_verified_overview.png")
cv2.imwrite(master_brain, final_sheet)
cv2.imwrite(master_local, final_sheet)

print(f"Master overview saved successfully: {master_brain} ({os.path.getsize(master_brain)/(1024*1024):.2f} MB)")

