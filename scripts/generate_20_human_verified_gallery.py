"""
Generate high-resolution visual comparisons for all 20 human-verified patches
in masks_v3_human_verified, plus a master 20-patch contact sheet.
Save directly to artifact directory for in-IDE rendering.
"""
import os
import glob
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review"
RGB_DIR = os.path.join(BASE_DIR, "data", "images_rgb")
V3_DIR = os.path.join(BASE_DIR, "data", "masks_v3_human_verified")

BRAIN_DIR = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf"
GALLERY_DIR = os.path.join(BRAIN_DIR, "human_verified_20_gallery")
LOCAL_GALLERY_DIR = os.path.join(BASE_DIR, "human_verified_20_gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(LOCAL_GALLERY_DIR, exist_ok=True)

# 20 human-verified patches
PATCH_LIST = [
    "test_pune_0407", "test_pune_0408", "test_pune_0409", "test_pune_0423",
    "test_pune_0429", "test_pune_0456", "test_pune_0458", "test_pune_0460",
    "test_pune_0471", "test_pune_0505", "test_pune_0506", "test_pune_0507",
    "test_pune_0516", "test_pune_0538", "test_pune_0550", "test_pune_0560",
    "test_pune_0564", "test_pune_0572", "test_pune_0575", "test_pune_0584"
]

PALETTE_HEX = {
    0: ("#1E78DC", "Water (0)"),
    1: ("#229628", "Vegetation (1)"),
    2: ("#D7B991", "Barren Land (2)"),
    3: ("#E65A46", "Residential (3)"),
    4: ("#8C32BE", "Industrial (4)"),
    5: ("#FF7800", "Slum / Informal (5)"),
    6: ("#FFE114", "Road Network (6)"),
    255: ("#969696", "Uncertain / Ignore (255)")
}

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

def render_overlay(rgb_bgr, mask_gray, alpha=0.45):
    h, w = mask_gray.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in PALETTE_BGR.items():
        color_mask[mask_gray == c] = col
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, color_mask, alpha, 0)

print(f"Generating individual 20 high-res visualizations...")

contact_tiles = []

for idx, pid in enumerate(PATCH_LIST, 1):
    rgb_path = os.path.join(RGB_DIR, f"{pid}_rgb.png")
    mask_path = os.path.join(V3_DIR, f"{pid}.png")
    
    if not os.path.exists(mask_path):
        print(f"Warning: {mask_path} not found!")
        continue
        
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(mask_path))
    rgb_bgr = cv2.imread(rgb_path)
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    comp_bgr = render_overlay(rgb_bgr, mask, alpha=0.45)
    comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
    
    # Store side-by-side pair for contact sheet: 512x256 each or 256x256
    pair = np.hstack([cv2.resize(rgb_rgb, (256, 256)), cv2.resize(comp_rgb, (256, 256))])
    # Add thin header on pair for contact sheet
    header = np.zeros((32, 512, 3), dtype=np.uint8) + 25
    cv2.putText(header, f"#{idx:02d}: {pid}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    tile_with_header = np.vstack([header, pair])
    contact_tiles.append(tile_with_header)
    
    # Calculate stats
    tot = mask.size
    class_stats = {}
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        cnt = (mask == c).sum()
        pct = cnt / tot * 100.0
        class_stats[c] = (cnt, pct)
        
    # Build single patch high-res figure (14 x 7.5 inches at 150 DPI)
    fig = plt.figure(figsize=(14, 7.5), dpi=150, facecolor="#141820")
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 0.18], hspace=0.15, wspace=0.06,
                          left=0.03, right=0.97, top=0.88, bottom=0.04)
                          
    fig.suptitle(f"Patch #{idx:02d}: {pid} — Human-Corrected Ground Truth (Version 3)\n"
                 f"Saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')} | Category: Human-Approved Ground Truth",
                 color="#FFFFFF", fontsize=13, fontweight="bold", y=0.96)
                 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_rgb)
    ax1.set_title("Original Cartosat-2E RGB (0.60 m GSD)", color="#A0B0C8", fontsize=11, pad=6, fontweight="bold")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(comp_rgb)
    ax2.set_title("Ground-Truth Annotation Overlay (7-Class Taxonomy)", color="#A0B0C8", fontsize=11, pad=6, fontweight="bold")
    ax2.axis("off")
    
    # Legend
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.set_facecolor("#1A202C")
    
    leg_handles = []
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        cnt, pct = class_stats[c]
        hex_c, name = PALETTE_HEX[c]
        label = f"{name}: {pct:>5.2f}% ({cnt:,} px)"
        leg_handles.append(Patch(facecolor=hex_c, edgecolor="#FFFFFF", label=label, linewidth=0.6))
        
    ax_leg.legend(handles=leg_handles, loc="center", ncol=4, fontsize=9.0,
                  framealpha=0.95, facecolor="#1F2937", edgecolor="#374151",
                  labelcolor="#F3F4F6", handlelength=1.3, handleheight=0.9, borderpad=0.7)
                  
    out_name = f"gt_{idx:02d}_{pid}.png"
    out_brain = os.path.join(GALLERY_DIR, out_name)
    out_local = os.path.join(LOCAL_GALLERY_DIR, out_name)
    plt.savefig(out_brain, facecolor=fig.get_facecolor(), dpi=150)
    plt.savefig(out_local, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"  [{idx}/20] Saved: {out_name}")

# Now build Master Contact Sheet (5 rows of 4 columns, or 4 rows of 5 columns)
# 4 columns, 5 rows
print("\nBuilding Master 20-Patch Contact Sheet...")
rows = []
for r in range(5):
    row_tiles = contact_tiles[r*4 : (r+1)*4]
    rows.append(np.hstack(row_tiles))
master_grid = np.vstack(rows)

# Add master header and legend
h_g, w_g = master_grid.shape[:2]
header_banner = np.zeros((90, w_g, 3), dtype=np.uint8) + 20
cv2.putText(header_banner, "Cartosat-2E Pune: 20 Human-Verified Ground-Truth Patches (Version 3)", (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(header_banner, "Left: 0.60m RGB Satellite Imagery  |  Right: Human-Approved 7-Class Annotation Overlay  |  Status: Training Halted",
            (32, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 180, 210), 1, cv2.LINE_AA)

final_contact_sheet = np.vstack([header_banner, master_grid])

contact_brain = os.path.join(GALLERY_DIR, "master_20_human_verified_overview.png")
contact_local = os.path.join(LOCAL_GALLERY_DIR, "master_20_human_verified_overview.png")
cv2.imwrite(contact_brain, cv2.cvtColor(final_contact_sheet, cv2.COLOR_RGB2BGR))
cv2.imwrite(contact_local, cv2.cvtColor(final_contact_sheet, cv2.COLOR_RGB2BGR))
print(f"Master contact sheet saved: {contact_brain} ({os.path.getsize(contact_brain)/(1024*1024):.2f} MB)")

