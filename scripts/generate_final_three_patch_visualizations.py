"""
Generate high-resolution side-by-side ground-truth visualizations for
the three user-specified patches using their latest manually saved masks:
1. test_pune_0572 (Image 1)
2. test_pune_0550 (Image 2)
3. test_pune_0408 (Image 3)
"""
import os
import shutil
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review"
RGB_DIR = os.path.join(BASE_DIR, "data", "images_rgb")
V3_DIR = os.path.join(BASE_DIR, "data", "masks_v3_human_verified")
OUT_DIR = BASE_DIR
BRAIN_DIR = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\builtup_reannotation_review"
os.makedirs(BRAIN_DIR, exist_ok=True)

TARGETS = [
    {
        "id": "test_pune_0572",
        "title": "Image 1: test_pune_0572 (Consolidated Industrial vs Residential Neighborhood)",
        "desc": "Industrial correctly pruned to genuine complex; Residential expanded across neighborhood"
    },
    {
        "id": "test_pune_0550",
        "title": "Image 2: test_pune_0550 (Dense Residential Neighborhood & Building Footprints)",
        "desc": "Scattered false industrial removed; Individual residential rooftops accurately annotated"
    },
    {
        "id": "test_pune_0408",
        "title": "Image 3: test_pune_0408 (Mixed Urban: Buildings, Roads, Vegetation & Barren)",
        "desc": "Individual residential buildings cleanly delineated from surrounding vegetation & soil"
    }
]

# Standard 7-Class Palette
PALETTE_HEX = {
    0: ("#1E78DC", "Water (Class 0)"),
    1: ("#229628", "Vegetation (Class 1)"),
    2: ("#D7B991", "Barren Land (Class 2)"),
    3: ("#E65A46", "Residential (Class 3)"),
    4: ("#8C32BE", "Industrial (Class 4)"),
    5: ("#FF7800", "Slum / Informal (Class 5)"),
    6: ("#FFE114", "Road Network (Class 6)"),
    255: ("#969696", "Ignore / Uncertain (255)")
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
    comp = cv2.addWeighted(rgb_bgr, 1.0 - alpha, color_mask, alpha, 0)
    return comp

for tgt in TARGETS:
    pid = tgt["id"]
    rgb_file = os.path.join(RGB_DIR, f"{pid}_rgb.png")
    mask_file = os.path.join(V3_DIR, f"{pid}.png")
    
    assert os.path.exists(rgb_file), f"RGB file not found: {rgb_file}"
    assert os.path.exists(mask_file), f"V3 Mask file not found: {mask_file}"
    
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(mask_file))
    print(f"\nProcessing {pid} (Mask timestamp: {mtime})...")
    
    rgb_bgr = cv2.imread(rgb_file)
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    # Render crisp overlay
    comp_bgr = render_overlay(rgb_bgr, mask, alpha=0.45)
    comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
    
    # Calculate exact class percentages
    total_pixels = mask.size
    class_stats = {}
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        cnt = (mask == c).sum()
        pct = (cnt / total_pixels) * 100.0
        class_stats[c] = (cnt, pct)
    
    # Create Figure: High Resolution (16 x 9 inches at 200 DPI = 3200 x 1800 px)
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="#141820")
    
    # Grid specification
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 0.16], hspace=0.15, wspace=0.08,
                          left=0.04, right=0.96, top=0.90, bottom=0.04)
    
    # Header Title
    title_text = f"Cartosat-2E Ground Truth — {pid}\n" \
                 f"CATEGORY: HUMAN-CORRECTED GROUND TRUTH (Version 3) | Saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')}"
    fig.suptitle(title_text, color="#FFFFFF", fontsize=15, fontweight="bold", y=0.96)
    
    # Panel 1: Original RGB Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_rgb)
    ax1.set_title("Original Satellite Imagery (0.60 m True-Color RGB)", color="#A0B0C8", fontsize=12, pad=8, fontweight="bold")
    ax1.axis("off")
    # Add border
    for spine in ax1.spines.values():
        spine.set_edgecolor("#303848")
        spine.set_linewidth(1.5)
        
    # Panel 2: Ground-Truth Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(comp_rgb)
    ax2.set_title("Ground-Truth Annotation Overlay (Seven-Class Taxonomy, Alpha = 0.45)", color="#A0B0C8", fontsize=12, pad=8, fontweight="bold")
    ax2.axis("off")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#303848")
        spine.set_linewidth(1.5)
        
    # Bottom: Legend & Quantitative Distribution
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.set_facecolor("#1A202C")
    
    # Create custom legend patches with exact percentages
    legend_elements = []
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        cnt, pct = class_stats[c]
        hex_col, name = PALETTE_HEX[c]
        label = f"{name}: {pct:>5.2f}% ({cnt:,} px)"
        # Use full patch
        legend_elements.append(Patch(facecolor=hex_col, edgecolor="#FFFFFF", label=label, linewidth=0.8))
        
    # Split into 2 rows of 4 items for perfect readability
    leg = ax_leg.legend(handles=legend_elements, loc="center", ncol=4, fontsize=9.5,
                        framealpha=0.95, facecolor="#1F2937", edgecolor="#374151",
                        labelcolor="#F3F4F6", handlelength=1.4, handleheight=1.0, borderpad=0.8)
    
    # Output file paths
    out_filename = f"{pid}_human_verified_final.png"
    out_path_local = os.path.join(OUT_DIR, out_filename)
    out_path_brain = os.path.join(BRAIN_DIR, out_filename)
    
    plt.savefig(out_path_local, facecolor=fig.get_facecolor(), edgecolor="none", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path_local}")
    
    shutil.copy2(out_path_local, out_path_brain)
    print(f"Copied to brain: {out_path_brain}")

print("\nAll 3 final visualizations generated successfully!")

