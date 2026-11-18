import re
from pathlib import Path
import shutil

OUT_DIR = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches")
BRAIN_DIR = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf")
html_path = OUT_DIR / "inspection_dashboard.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Add quick navigation toolbar and modal lightbox
toolbar_html = """
    <!-- Top Action Banner -->
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #38bdf8; border-radius: 10px; padding: 15px 20px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <h3 style="margin: 0 0 5px 0; color: #38bdf8; font-size: 1.2em;">⭐ Spotlight Review: Tile test_pune_0585 (Batch 10)</h3>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9em;">
                    Zero-Water Certified: <strong>0 Water Pixels (0.00%)</strong> | Reassigned: Road Network (31.7%) & Barren Land (42.7%)
                </p>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <a href="test_pune_0585_side_by_side_comparison.png" target="_blank" class="btn" style="text-decoration: none; background: #0284c7; color: white;">🔍 View Side-by-Side (0585)</a>
                <a href="ground_truth_fix_11panel_test_pune_0585.png" target="_blank" class="btn" style="text-decoration: none; background: #475569; color: white;">📊 11-Panel Diagnostic Suite</a>
                <a href="batch_10_overview.png" target="_blank" class="btn" style="text-decoration: none; background: #334155; color: white;">🖼️ Batch 10 Overview PNG</a>
                <a href="Cartosat2E_GroundTruth_Inspection_Package.zip" download class="btn" style="text-decoration: none; background: #16a34a; color: white;">💾 Download Full ZIP (250 MB)</a>
            </div>
        </div>
    </div>

    <!-- Batch Contact Sheets Quick Links -->
    <div style="background: #1e293b; border-radius: 8px; padding: 10px 15px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 0.85em;">
        <span style="font-weight: bold; color: #94a3b8;">High-Res Batch Contact Sheets (PNG):</span>
        <a href="batch_01_overview.png" target="_blank" style="color: #38bdf8;">Batch 01</a> |
        <a href="batch_02_overview.png" target="_blank" style="color: #38bdf8;">Batch 02</a> |
        <a href="batch_03_overview.png" target="_blank" style="color: #38bdf8;">Batch 03</a> |
        <a href="batch_04_overview.png" target="_blank" style="color: #38bdf8;">Batch 04</a> |
        <a href="batch_05_overview.png" target="_blank" style="color: #38bdf8;">Batch 05</a> |
        <a href="batch_06_overview.png" target="_blank" style="color: #38bdf8;">Batch 06</a> |
        <a href="batch_07_overview.png" target="_blank" style="color: #38bdf8;">Batch 07</a> |
        <a href="batch_08_overview.png" target="_blank" style="color: #38bdf8;">Batch 08</a> |
        <a href="batch_09_overview.png" target="_blank" style="color: #38bdf8;">Batch 09</a> |
        <a href="batch_10_overview.png" target="_blank" style="color: #4ade80; font-weight: bold;">Batch 10 (0585)</a> |
        <a href="test_set_per_patch_verification_report.csv" target="_blank" style="color: #fbbf24;">📄 Per-Patch CSV Report</a>
    </div>
"""

# Insert toolbar right after `<div class="legend-bar">...</div>`
legend_end = html.find('</div>\n\n    <div class="controls">')
if legend_end != -1:
    html = html[:legend_end+6] + toolbar_html + html[legend_end+6:]

# Add Lightbox modal
lightbox_html = """
    <!-- Modal Lightbox for Image Zoom -->
    <div id="lightbox" onclick="closeLightbox()" style="display: none; position: fixed; z-index: 9999; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.9); justify-content: center; align-items: center; cursor: pointer;">
        <img id="lightbox-img" style="max-width: 95vw; max-height: 90vh; border-radius: 8px; border: 2px solid #38bdf8; box-shadow: 0 0 25px rgba(56,189,248,0.5);">
    </div>
    <script>
    document.querySelectorAll('.card-image').forEach(img => {
        img.addEventListener('click', () => {
            document.getElementById('lightbox-img').src = img.src;
            document.getElementById('lightbox').style.display = 'flex';
        });
    });
    function closeLightbox() {
        document.getElementById('lightbox').style.display = 'none';
    }
    </script>
"""

body_end = html.rfind('</body>')
if body_end != -1:
    html = html[:body_end] + lightbox_html + html[body_end:]

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

shutil.copy2(html_path, BRAIN_DIR / "inspection_dashboard.html")
print("Enhanced inspection_dashboard.html successfully!")

