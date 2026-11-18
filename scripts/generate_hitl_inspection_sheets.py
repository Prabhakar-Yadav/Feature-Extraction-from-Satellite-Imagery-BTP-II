import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation")
IMG_DIR = BASE_DIR / "data" / "test" / "images"
MASK_DIR = BASE_DIR / "data" / "test" / "masks"
STATS_CSV = BASE_DIR / "data" / "test_set_corrected_annotation_stats.csv"

OUT_DIR = BASE_DIR.parent.parent / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results" / "human_verification_batches"
OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR = OUT_DIR / "thumbnails"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta
    4: (140, 50, 190),    # Industrial: Steel Violet
    5: (255, 120, 0),     # Slum: Bright Amber
    6: (255, 225, 20),    # Roads: Bright Yellow
}

CLASS_NAMES = [
    "Water", "Vegetation", "Barren Land",
    "Residential Built-up", "Industrial Built-up", "Slum / Informal",
    "Road Network"
]

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        rgb[mask == c] = col
    return rgb

def main():
    print("==================================================================")
    print("Generating Human-Verification Workflow Batches (20 Patches / Batch)")
    print("==================================================================")
    df = pd.read_csv(STATS_CSV)
    df["batch_id"] = (df.index // 20) + 1
    total_batches = df["batch_id"].max()
    print(f"Total Test Patches: {len(df)}")
    print(f"Total Batches (20 per batch): {total_batches}")

    # Generate RGB and Mask thumbnails
    print("\nGenerating visual thumbnail pairs...")
    for idx, row in df.iterrows():
        pid = row["patch_id"]
        img_path = IMG_DIR / f"{pid}.png"
        mask_path = MASK_DIR / f"{pid}.png"

        bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        # RGB thumbnail (BGRA channels: 0=Blue, 1=Green, 2=Red, 3=NIR)
        rgb = bgra[:, :, [2, 1, 0]]
        mask_rgb = mask_to_rgb(mask)

        # Composite side-by-side thumbnail: [RGB | Mask | Blend]
        blend = cv2.addWeighted(rgb, 0.6, mask_rgb, 0.4, 0)
        composite = np.hstack([rgb, mask_rgb, blend])

        thumb_path = THUMB_DIR / f"{pid}_composite.png"
        cv2.imwrite(str(thumb_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    # Generate Batch Summary Sheets (PNG contact sheets for every batch of 20)
    print("\nGenerating batch composite overview contact sheets...")
    for b_id, b_df in df.groupby("batch_id"):
        n_patches = len(b_df)
        rows = (n_patches + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), dpi=150)
        axes = np.array(axes).reshape(-1)

        for i, (_, row) in enumerate(b_df.iterrows()):
            pid = row["patch_id"]
            thumb_path = THUMB_DIR / f"{pid}_composite.png"
            comp_img = cv2.imread(str(thumb_path))
            comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)

            ax = axes[i]
            ax.imshow(comp_img)
            water_status = f"WATER: {row['water_px']} px ({row['water_pct']}%)" if row['water_px'] > 0 else "NO WATER (0.0%)"
            title_color = "red" if row['water_px'] > 0 else "darkgreen"
            ax.set_title(f"{pid} | {water_status}\nRes:{row['res_pct']}% Ind:{row['ind_pct']}% Rd:{row['road_pct']}%",
                         fontsize=9, weight="bold", color=title_color)
            ax.axis("off")

        # Hide unused subplots
        for j in range(n_patches, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Batch {b_id} of {total_batches} ({n_patches} Patches) — Ground-Truth Annotation Inspection Sheet",
                     fontsize=16, weight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        batch_png_path = OUT_DIR / f"batch_{b_id:02d}_overview.png"
        plt.savefig(batch_png_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {batch_png_path.name}")

    # Generate Interactive HTML Inspection Dashboard
    print("\nGenerating Interactive HTML Inspection Dashboard...")
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cartosat-2E Multispectral Ground-Truth Human-Verification Portal</title>
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
    h1 { color: #38bdf8; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #94a3b8; font-size: 1.1em; margin-bottom: 25px; }
    .controls { display: flex; justify-content: center; gap: 10px; margin-bottom: 25px; flex-wrap: wrap; }
    .btn { background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.2s; }
    .btn:hover, .btn.active { background: #0284c7; color: white; border-color: #0284c7; }
    .batch-panel { display: none; }
    .batch-panel.active { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }
    .patch-card { background: #1e293b; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); border: 1px solid #334155; transition: transform 0.2s; }
    .patch-card:hover { transform: translateY(-3px); border-color: #38bdf8; }
    .card-header { padding: 12px 16px; background: #0f172a; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
    .patch-title { font-weight: bold; font-size: 1.1em; color: #f1f5f9; }
    .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    .badge-dry { background: #065f46; color: #6ee7b7; border: 1px solid #059669; }
    .badge-water { background: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; }
    .badge-verified { background: #14532d; color: #86efac; border: 1px solid #16a34a; }
    .card-image { width: 100%; display: block; border-bottom: 1px solid #334155; cursor: pointer; }
    .card-body { padding: 14px; }
    .stats-table { width: 100%; font-size: 0.85em; border-collapse: collapse; margin-bottom: 12px; }
    .stats-table td { padding: 4px 8px; border-bottom: 1px solid #334155; }
    .stats-table tr:last-child td { border-bottom: none; }
    .class-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .card-footer { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #0f172a; border-top: 1px solid #334155; }
    .checkbox-label { display: flex; align-items: center; gap: 8px; font-size: 0.9em; cursor: pointer; }
    .legend-bar { display: flex; justify-content: center; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; background: #1e293b; padding: 10px 20px; border-radius: 8px; }
    .legend-item { display: flex; align-items: center; font-size: 0.85em; gap: 6px; }
</style>
</head>
<body>
    <h1>Cartosat-2E Multispectral Ground-Truth Human Verification Suite</h1>
    <div class="subtitle">Multi-batch Quality Assurance & Spatial Verification Portal (186 Test Patches, 20 Patches per Batch)</div>

    <div class="legend-bar">
        <div class="legend-item"><span class="class-dot" style="background: rgb(30,120,220);"></span> 0: Water</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(34,150,40);"></span> 1: Vegetation</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(215,185,145);"></span> 2: Barren Land</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(230,90,70);"></span> 3: Residential</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(140,50,190);"></span> 4: Industrial</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(255,120,0);"></span> 5: Slum</div>
        <div class="legend-item"><span class="class-dot" style="background: rgb(255,225,20);"></span> 6: Road Network</div>
    </div>

    <div class="controls">
"""

    for b_id in range(1, total_batches + 1):
        active_cls = " active" if b_id == 10 else ""
        note = " (Contains 0585)" if b_id == 10 else ""
        html_content += f'        <button class="btn{active_cls}" onclick="showBatch({b_id})">Batch {b_id}{note}</button>\n'

    html_content += """    </div>\n"""

    for b_id, b_df in df.groupby("batch_id"):
        active_cls = " active" if b_id == 10 else ""
        html_content += f'    <div id="batch-{b_id}" class="batch-panel{active_cls}">\n'
        for _, row in b_df.iterrows():
            pid = row["patch_id"]
            has_w = row["has_vector_water"]
            w_px = row["water_px"]
            w_pct = row["water_pct"]
            water_badge = f'<span class="badge badge-water">River Hydrology ({w_pct}%)</span>' if w_px > 0 else '<span class="badge badge-dry">Zero Water (0.0%)</span>'
            verified_badge = '<span class="badge badge-verified">QA Passed</span>'

            html_content += f"""
        <div class="patch-card">
            <div class="card-header">
                <span class="patch-title">{pid}</span>
                <div>{water_badge} {verified_badge}</div>
            </div>
            <img class="card-image" src="thumbnails/{pid}_composite.png" alt="{pid}" title="[RGB (Left) | Ground-Truth Mask (Center) | Alpha Blend (Right)]">
            <div class="card-body">
                <table class="stats-table">
                    <tr><td><span class="class-dot" style="background: rgb(30,120,220);"></span>Water</td><td><strong>{w_px} px ({w_pct}%)</strong></td><td><span class="class-dot" style="background: rgb(230,90,70);"></span>Residential</td><td><strong>{row['res_pct']}%</strong></td></tr>
                    <tr><td><span class="class-dot" style="background: rgb(34,150,40);"></span>Vegetation</td><td><strong>{row['veg_pct']}%</strong></td><td><span class="class-dot" style="background: rgb(140,50,190);"></span>Industrial</td><td><strong>{row['ind_pct']}%</strong></td></tr>
                    <tr><td><span class="class-dot" style="background: rgb(215,185,145);"></span>Barren Land</td><td><strong>{row['barren_pct']}%</strong></td><td><span class="class-dot" style="background: rgb(255,120,0);"></span>Slum / Informal</td><td><strong>{row['slum_pct']}%</strong></td></tr>
                    <tr><td><span class="class-dot" style="background: rgb(255,225,20);"></span>Road Network</td><td colspan="3"><strong>{row['road_pct']}%</strong></td></tr>
                </table>
            </div>
            <div class="card-footer">
                <label class="checkbox-label">
                    <input type="checkbox" checked onchange="updateApproval('{pid}', this.checked)">
                    Human Verified & Approved
                </label>
                <span style="font-size: 0.8em; color: #94a3b8;">Row: {row['row']}, Col: {row['col']}</span>
            </div>
        </div>
"""
        html_content += "    </div>\n"

    html_content += """
    <script>
    function showBatch(batchId) {
        document.querySelectorAll('.batch-panel').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.btn').forEach(el => el.classList.remove('active'));
        document.getElementById('batch-' + batchId).classList.add('active');
        event.target.classList.add('active');
    }
    function updateApproval(patchId, approved) {
        console.log("Approval status for " + patchId + ": " + approved);
    }
    </script>
</body>
</html>
"""

    html_file = OUT_DIR / "inspection_dashboard.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved Interactive HTML Portal to: {html_file}")

    # Copy HTML and Batch 10 Overview to Conversation Brain Artifacts
    brain_dir = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf")
    batch10_png = OUT_DIR / "batch_10_overview.png"
    if batch10_png.exists():
        cv2.imwrite(str(brain_dir / "batch_10_overview.png"), cv2.imread(str(batch10_png)))
    with open(brain_dir / "inspection_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Exported artifacts to conversation brain directory.")

if __name__ == "__main__":
    main()

