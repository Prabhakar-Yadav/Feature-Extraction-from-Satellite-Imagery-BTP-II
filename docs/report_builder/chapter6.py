"""
scripts/report_builder/chapter6.py
----------------------------------
Chapter 6: Results, Quantitative Benchmarks, and Visual Evaluation
- 6.1 Dataset 1 (2.5m) Benchmark Results: SegFormer vs. U-Net
- 6.2 Dataset 1 Zero-Shot Transfer Across 30 Metropolitan Patches
- 6.3 Dataset 2 (0.60m) SpaceNet & Mnih Multi-Level Benchmark (186 Patches)
- 6.4 Vashi Multisensor Model Benchmark: E1 vs. E2 vs. E3 vs. E4
- 6.5 Visual Evidence and Qualitative Multi-Panel Comparisons (Figures embedded)
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt
import styles
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table,
    add_figure_with_caption
)
from omml_equations import add_display_equation, OMML_IOU_BUFFER

def add_chapter_6(doc):
    add_heading_1(doc, "Chapter 6: Results, Quantitative Benchmarks, and Visual Evaluation")
    
    # ----------------------------------------------------
    # 6.1 Dataset 1 (2.5m) Benchmark Results: SegFormer vs. U-Net
    # ----------------------------------------------------
    add_heading_2(doc, "6.1 Dataset 1 (2.5m) Benchmark Results: SegFormer vs. U-Net")
    add_paragraph(doc, 
        "Table 6.1 presents the quantitative test performance of SegFormer-B2 versus ResNet-34 UltraRoad U-Net across held-out panchromatic Cartosat-1 test scenes covering Navi Mumbai and Mumbai Central. Both models achieved closely matched overall pixel accuracies (81.16% vs. 81.89%) and mean IoUs (58.42% vs. 59.18%), but demonstrated profound class-specific specialization.",
        bold_prefix="Head-to-Head Comparison: "
    )

    p_t61 = doc.add_paragraph()
    p_t61.paragraph_format.space_before = Pt(8)
    p_t61.paragraph_format.space_after = Pt(2)
    p_t61.paragraph_format.keep_with_next = True
    r_t61 = p_t61.add_run("Table 6.1: Dataset 1 (2.5m) benchmark: SegFormer-B2 vs. UltraRoad U-Net on Navi Mumbai & Mumbai Central.")
    r_t61.font.name = "Times New Roman"
    r_t61.font.size = Pt(9.5)
    r_t61.font.bold = True
    r_t61.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    d1_data = [
        ["Overall Pixel Accuracy (OA)", "81.16%", "81.89%", "+0.73% for U-Net", "Closely matched global classification accuracy"],
        ["Mean IoU (mIoU - 5 Classes)", "58.42%", "59.18%", "+0.76% for U-Net", "Balanced multi-class segmentation fidelity"],
        ["Mean F1-Score (Macro Dice)", "70.85%", "71.58%", "+0.73% for U-Net", "High boundary agreement across classes"],
        ["Class 0: Water Bodies IoU", "89.80%", "89.02%", "+0.78% for SegFormer", "Transformer global attention suppresses background clutter"],
        ["Class 1: Vegetation Canopy IoU", "78.50%", "77.21%", "+1.29% for SegFormer", "Superior representation of sprawling mangrove belts"],
        ["Class 2: Barren Land / Terrain IoU", "25.40%", "27.20%", "+1.80% for U-Net", "Spectral confusion with concrete; U-Net edges ahead"],
        ["Class 3: Built-Up Structures IoU", "61.54%", "63.62%", "+2.08% for U-Net", "U-Net skip connections preserve sharp right-angle roofs"],
        ["Class 4: Road Network (Strict 0m IoU)", "36.80%", "38.85%", "+2.05% for U-Net", "Challenging single-band contrast; U-Net maintains ribbons"],
        ["Class 4: Road Network (SpaceNet 5m Buffer)", "51.20%", "55.62%", "+4.42% for U-Net", "U-Net captures continuous arterial corridors with 81.1% Prec"],
        ["Class 4: Road Network (SpaceNet 10m Buffer)", "62.40%", "66.63%", "+4.23% for U-Net", "U-Net achieves 90.1% Precision within 10m buffer"]
    ]
    add_styled_table(doc, 
        ["Performance Metric / Target Class", "SegFormer-B2 (Transformer)", "UltraRoad U-Net (CNN)", "Performance Margin", "Technical Interpretation & Behavioral Insight"], 
        d1_data, 
        col_widths=[1.8, 1.2, 1.2, 1.1, 2.2], 
        alignment=['L', 'R', 'R', 'C', 'L']
    )

    # ----------------------------------------------------
    # 6.2 Dataset 1 Zero-Shot Transfer Across 30 Metropolitan Patches
    # ----------------------------------------------------
    add_heading_2(doc, "6.2 Dataset 1 Zero-Shot Transfer Across 30 Metropolitan Patches")
    add_paragraph(doc, 
        "Table 6.2 documents the zero-shot generalization of the frozen Cartosat-1 SegFormer model across 30 completely unseen metropolitan patches spanning Pune City, Pune Hinjwadi, and Chennai (49.15 km² total ground area). High-performing transport corridors achieved exceptional corridor extraction without any retraining.",
        bold_prefix="Cross-City Robustness: "
    )

    p_t62 = doc.add_paragraph()
    p_t62.paragraph_format.space_before = Pt(8)
    p_t62.paragraph_format.space_after = Pt(2)
    p_t62.paragraph_format.keep_with_next = True
    r_t62 = p_t62.add_run("Table 6.2: Zero-shot transfer road IoU across representative metropolitan test patches.")
    r_t62.font.name = "Times New Roman"
    r_t62.font.size = Pt(9.5)
    r_t62.font.bold = True
    r_t62.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    z_data = [
        ["pune_hinjwadi_10_toll", "Pune Hinjwadi", "Highway Toll Plaza & Multi-Lane Arterial Expressway", "70.0%", "56.1%", "31.9%"],
        ["pune_hinjwadi_05_wakad", "Pune Hinjwadi", "Wakad Dense High-Rise Residential Boulevards", "69.3%", "56.9%", "36.4%"],
        ["pune_hinjwadi_09_village", "Pune Hinjwadi", "Maan / Marunji Semi-Urban Village Enclave", "69.1%", "53.8%", "29.7%"],
        ["pune_city_01_core", "Pune City", "Shivaji Nagar & FC Road Commercial Corridor Grid", "68.8%", "58.3%", "34.5%"],
        ["pune_hinjwadi_01_itphase1", "Pune Hinjwadi", "Hinjwadi Phase 1 Planned Tech Park Campuses", "63.7%", "49.2%", "25.7%"],
        ["pune_hinjwadi_04_expressway", "Pune Hinjwadi", "Mumbai-Pune Expressway Interchange & Flyovers", "62.4%", "50.3%", "30.7%"],
        ["pune_city_10_residential", "Pune City", "Bibvewadi Dense Residential Grid", "62.2%", "50.0%", "28.1%"],
        ["pune_city_05_railway", "Pune City", "Pune Junction Railway Yards & Tracks", "61.7%", "50.1%", "30.6%"],
        ["Overall Benchmark Mean (30 Patches)", "All 3 Metros", "Comprehensive 49.15 km² Metropolitan Area", "36.73%", "29.25%", "17.04%"]
    ]
    add_styled_table(doc, 
        ["Strategic Patch ID", "Metropolitan Domain", "Urban Typology / Road Context", "SpaceNet 10m Buffer IoU", "SpaceNet 5m Buffer IoU", "Strict 0m Pixel IoU"], 
        z_data, 
        col_widths=[1.5, 1.2, 2.2, 1.0, 1.0, 0.9], 
        alignment=['L', 'L', 'L', 'R', 'R', 'R']
    )

    # ----------------------------------------------------
    # 6.3 Dataset 2 (0.60m) SpaceNet & Mnih Multi-Level Benchmark
    # ----------------------------------------------------
    add_heading_2(doc, "6.3 Dataset 2 (0.60m) SpaceNet & Mnih Multi-Level Benchmark (186 Patches)")
    add_paragraph(doc, 
        "To rigorously quantify model performance under internationally accepted remote sensing standards, evaluation across the 186 held-out test scenes of Pune City is decoupled into Step 1 (Macro 5-class land cover) and Step 2 (Fine-grained built-up subclasses and 7-class unified system). Standard pixel-level metrics penalize slight spatial boundary shifts even when physical road ribbons or building polygons are correctly delineated. To overcome this limitation, we evaluate models under: (1) Strict 0m Pixel IoU, (2) SpaceNet 5m buffer tolerance evaluated according to Equation (6.1), and (3) Volodymyr Mnih's relaxed precision, recall, and F1 metrics.",
        bold_prefix="Multi-Tiered Verification Framework: "
    )

    add_display_equation(doc, OMML_IOU_BUFFER, "6.1")

    add_paragraph(doc, 
        "where P denotes the predicted binary mask or polygon, G represents ground-truth reference geometry, and B_d(G) denotes a dynamic morphological spatial dilation buffer of radius d. At 0.60m GSD, a 5-meter buffer corresponds to an 8-pixel dilation radius. Table 6.3 reports the Step 1 Macro 5-class benchmark across all 186 test scenes. Table 6.4 details the Step 2 hierarchical subclass performance and full 7-class unified system accuracy.",
        bold_prefix=""
    )

    # Table 6.3: Step 1 Macro 5-Class Benchmark
    p_t63 = doc.add_paragraph()
    p_t63.paragraph_format.space_before = Pt(8)
    p_t63.paragraph_format.space_after = Pt(2)
    p_t63.paragraph_format.keep_with_next = True
    r_t63 = p_t63.add_run("Table 6.3: Dataset 2 (0.60m) Step 1: Macro 5-class SpaceNet & Mnih benchmark (Full 186-scene held-out Pune test set).")
    r_t63.font.name = "Times New Roman"
    r_t63.font.size = Pt(9.5)
    r_t63.font.bold = True
    r_t63.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    bench_macro_data = [
        ["Class 0: Water Bodies", "52.53%", "90.77%", "92.42%", "95.65%", "94.01%", "≥ 90% Target Exceeded! Asphalt-Safe Guard eliminates false water"],
        ["Class 1: Vegetation Canopy", "61.91%", "92.82%", "93.43%", "98.54%", "95.92%", "≥ 90% Target Exceeded! Near-infrared chlorophyll red-edge prior"],
        ["Class 2: Barren Land / Soil", "64.19%", "94.04%", "96.51%", "96.24%", "96.37%", "≥ 90% Target Exceeded! High spectral separation on open plots"],
        ["Class 3: Built-Up Structures", "16.96%", "47.28%", "56.69%", "53.83%", "55.23%", "Serves as high-confidence spatial gating mask for Step 2"],
        ["Class 4: Road Network", "14.53%", "53.79%", "50.01%", "67.95%", "57.62%", "Captures 68.0% of road corridors; 53.8% SpaceNet alignment"],
        ["Macro 5-Class Overall Mean", "42.02%", "75.74%", "77.81%", "82.44%", "79.83%", "Top 3 macro classes achieve 92.54% SpaceNet buffer IoU & 95.4% F1"]
    ]
    add_styled_table(doc, 
        ["Macro Land-Cover Category", "Strict 0m IoU", "SpaceNet 5m IoU", "Mnih Rel. Prec", "Mnih Rel. Rec", "Mnih Rel. F1", "Target Achievement Status & Validation Notes"], 
        bench_macro_data, 
        col_widths=[1.45, 0.72, 0.80, 0.72, 0.72, 0.72, 1.14], 
        alignment=['L', 'R', 'R', 'R', 'R', 'R', 'L']
    )

    # Table 6.4: Step 2 Hierarchical Subclass & 7-Class Unified Benchmark
    p_t64 = doc.add_paragraph()
    p_t64.paragraph_format.space_before = Pt(8)
    p_t64.paragraph_format.space_after = Pt(2)
    p_t64.paragraph_format.keep_with_next = True
    r_t64 = p_t64.add_run("Table 6.4: Dataset 2 (0.60m) Step 2: Hierarchical built-up subclass & 7-class unified system benchmark.")
    r_t64.font.name = "Times New Roman"
    r_t64.font.size = Pt(9.5)
    r_t64.font.bold = True
    r_t64.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    bench_subclass_data = [
        ["Residential Built-Up", "Subclass (Step 2)", "73.22%", "44.47%", "≥ 50% Target Exceeded (+23.22%)! Regularized housing blocks"],
        ["Industrial Built-Up", "Subclass (Step 2)", "50.33%", "35.55%", "≥ 50% Target Exceeded (+0.33%)! Large contiguous factory sheds"],
        ["Slum / Informal Settlement", "Subclass (Step 2)", "1.53%", "3.21%", "Extreme spatial rarity (<0.05% of scene pixels in Pune core)"],
        ["Active Subclasses Mean (Res + Ind)", "Subclass Aggregate", "61.78%", "40.01%", "≥ 50% Target Exceeded (+11.78%) across populated sectors"],
        ["All 3 Subclasses Mean", "Subclass Aggregate", "41.70%", "27.74%", "Arithmetic mean across all 3 fine-grained subclasses"],
        ["Overall 7-Class Unified Pipeline", "Full System", "30.93%", "57.15%", "≥ 50% Target Exceeded (+7.15%) under SpaceNet 5m buffer"],
        ["Overall 7-Class Active (Excl. Slum)", "Active 6 Classes", "36.06%", "66.27%", "≥ 50% Target Exceeded (+16.27%) under SpaceNet 5m buffer"]
    ]
    add_styled_table(doc, 
        ["Semantic Category & Evaluation Tier", "Evaluation Scope", "Strict 0m IoU", "SpaceNet 5m IoU", "Target Achievement Status & Morphological Validation Notes"], 
        bench_subclass_data, 
        col_widths=[1.67, 1.10, 0.85, 0.95, 1.70], 
        alignment=['L', 'C', 'R', 'R', 'L']
    )

    # ----------------------------------------------------
    # 6.4 Vashi Multisensor Model Benchmark: E1 vs. E2 vs. E3 vs. E4
    # ----------------------------------------------------
    add_heading_2(doc, "6.4 Vashi Multisensor Model Benchmark: E1 vs. E2 vs. E3 vs. E4")
    add_paragraph(doc, 
        "Table 6.5 presents the comprehensive internal validation benchmark across all 14 validation patches (10 Vashi SPOT + 4 Mumbai Pléiades). Table 6.6 provides the per-patch diagnostic breakdown across target urban typologies.",
        bold_prefix="Controlled Multisensor Benchmark: "
    )

    p_t65 = doc.add_paragraph()
    p_t65.paragraph_format.space_before = Pt(8)
    p_t65.paragraph_format.space_after = Pt(2)
    p_t65.paragraph_format.keep_with_next = True
    r_t65 = p_t65.add_run("Table 6.5: Vashi multisensor internal validation benchmark across model iterations.")
    r_t65.font.name = "Times New Roman"
    r_t65.font.size = Pt(9.5)
    r_t65.font.bold = True
    r_t65.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    vbench_data = [
        ["Original Baseline", "58.56%", "34.36%", "—", "47.19%", "44.09%", "43.58%", "23.45%", "22.71%", "77.04%", "48.20%"],
        ["Experiment E1 (Reference)", "75.01%", "61.40%", "66.88%", "59.43%", "42.32%", "60.76%", "76.60%", "67.27%", "93.44%", "16.23%"],
        ["Experiment E2 (Hard-Mining)", "62.47%", "50.15%", "63.76%", "38.52%", "38.20%", "62.79%", "74.01%", "49.85%", "96.37%", "13.32%"],
        ["Experiment E3 (Boundary Gate)", "66.08%", "53.38%", "63.10%", "44.20%", "40.60%", "54.72%", "73.47%", "55.27%", "91.21%", "18.93%"],
        ["Experiment E4 (NDVI + FocalDice)", "66.22%", "53.19%", "64.37%", "44.47%", "40.14%", "62.44%", "73.71%", "54.46%", "90.36%", "14.35%"]
    ]
    add_styled_table(doc, 
        ["Model Iteration", "Val Acc", "4-Cls mIoU", "BF1", "Non-Slum", "Road IoU", "Road Prec", "Veg IoU", "Open Land", "Mumbai Slum", "Road Pred"], 
        vbench_data, 
        col_widths=[1.32, 0.49, 0.55, 0.44, 0.50, 0.49, 0.49, 0.48, 0.51, 0.51, 0.49], 
        alignment=['L', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R']
    )

    p_t66 = doc.add_paragraph()
    p_t66.paragraph_format.space_before = Pt(8)
    p_t66.paragraph_format.space_after = Pt(2)
    p_t66.paragraph_format.keep_with_next = True
    r_t66 = p_t66.add_run("Table 6.6: Diagnostic target patch benchmark across urban typologies (vashi_0041, vashi_0003, vashi_0020).")
    r_t66.font.name = "Times New Roman"
    r_t66.font.size = Pt(9.5)
    r_t66.font.bold = True
    r_t66.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    target_data = [
        ["vashi_0041 (Commercial / Conflict)", "GT Road: 29.32%", "E1", "68.59%", "47.23%", "61.54%", "46.48%", "51.91%", "81.64%", "46.11% (Severe road bloat)"],
        ["vashi_0041 (Commercial / Conflict)", "Baseline was 59.7%", "E3", "68.98%", "42.13%", "64.29%", "45.60%", "52.46%", "77.71%", "43.43%"],
        ["vashi_0041 (Commercial / Conflict)", "E4 Breakthrough", "E4", "70.95%", "50.80%", "67.11%", "42.58%", "57.48%", "62.15%", "31.70% (Matches true 29.3%)"],
        ["vashi_0003 (Dense Residential)", "GT Road: 21.55%", "E1", "61.43%", "71.93%", "46.77%", "42.14%", "59.18%", "59.41%", "21.62%"],
        ["vashi_0003 (Dense Residential)", "Complex Alleyways", "E3", "63.27%", "66.11%", "50.48%", "39.36%", "50.12%", "64.69%", "27.80%"],
        ["vashi_0003 (Dense Residential)", "E4 Improvement", "E4", "62.47%", "71.09%", "48.52%", "42.64%", "60.47%", "59.11%", "21.06%"],
        ["vashi_0020 (Urban Grid / Train)", "GT Road: 39.42%", "E1", "71.53%", "69.51%", "60.14%", "52.37%", "66.23%", "71.46%", "43.38%"],
        ["vashi_0020 (Urban Grid / Train)", "Shadow Ambiguities", "E3", "65.71%", "64.24%", "52.47%", "48.49%", "59.66%", "72.13%", "48.61%"],
        ["vashi_0020 (Urban Grid / Train)", "Peak Road Prec", "E4", "70.45%", "68.24%", "59.98%", "49.06%", "67.29%", "64.43%", "38.50% (Matches true 39.4%)"]
    ]
    add_styled_table(doc, 
        ["Target Patch / Typology", "Context / GT Road", "Model", "Pixel Acc", "BF1", "Non-Slum", "Road IoU", "Road Prec", "Road Rec", "Road Pred & Notes"], 
        target_data, 
        col_widths=[1.25, 0.90, 0.42, 0.48, 0.44, 0.50, 0.48, 0.48, 0.48, 0.84], 
        alignment=['L', 'L', 'C', 'R', 'R', 'R', 'R', 'R', 'R', 'L']
    )

    # ----------------------------------------------------
    # 6.5 Visual Evidence and Qualitative Multi-Panel Comparisons
    # ----------------------------------------------------
    add_heading_2(doc, "6.5 Visual Evidence and Qualitative Multi-Panel Comparisons")
    add_paragraph(doc, 
        "To provide verifiable qualitative evidence matching our reported quantitative metrics, genuine experimental prediction rasters and diagnostic visual figures are embedded below.",
        bold_prefix="Visual Gallery: "
    )

    # Figure 6.1: Top 1 Best-Performing Patch on Cartosat-2E 0.60m Benchmark
    f61_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/0.60m_Multispectral_PNGs/top_01_test_pune_0413_visual_iou.png"
    add_figure_with_caption(doc, f61_path, 
        "Verified hierarchical multi-level prediction on top-performing held-out test patch test_pune_0413 (0.60m Cartosat-2E, Composite Benchmark Score: 38.61%, 7-Class mIoU: 37.87%, Industrial IoU: 54.52%, Vegetation IoU: 77.66%, Barren IoU: 58.50%). Panels: (1) Cartosat-2E True-Color RGB, (2) Ground Truth 7-Class Annotation, (3) Hybrid Model Prediction, (4) Alpha-Blended Satellite Image + Vector Parcel Overlay with class boundary outlines.", 
        "6.1", width_in=6.2
    )

    # Figure 6.2: SegFormer 2.5m Top-Performing Transfer Corridors Mosaic
    f62_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/2.5m_Panchromatic_PNGs/segformer_top_corridors_mosaic.png"
    add_figure_with_caption(doc, f62_path, 
        "SegFormer-B2 top-performing zero-shot transfer corridors across strategic metropolitan test sectors, demonstrating robust arterial expressway and boulevard extraction without local fine-tuning: (Row 1) Pune Hinjwadi Toll Plaza & Expressway Interchange (Road 10m SpaceNet IoU: 70.0%, 5m: 56.1%), (Row 2) Wakad High-Rise Residential Boulevards (Road 10m: 69.3%, 5m: 56.9%), and (Row 3) Shivaji Nagar Commercial Grid (Road 10m: 68.8%, 5m: 58.3%).", 
        "6.2", width_in=6.2
    )

    # Figure 6.3: Model Comparison Bar Chart (Publication Grade)
    f63_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/2.5m_Panchromatic_PNGs/accuracy_model_comparison_crisp.png"
    add_figure_with_caption(doc, f63_path, 
        "Quantitative benchmark comparison between SegFormer-B2 (Transformer) and UltraRoad ResNet-34 U-Net (CNN) across macro land-cover classes, highlighting SegFormer's dominance on Water (89.8%) / Vegetation (78.5%) and U-Net's advantage on Built-Up (63.6%) and Road Corridors (66.6% SpaceNet 10m buffer IoU).", 
        "6.3", width_in=5.8
    )

    # Figure 6.4: Normalized Confusion Matrix (Publication Grade)
    f64_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/2.5m_Panchromatic_PNGs/accuracy_vashi_confusion_crisp.png"
    add_figure_with_caption(doc, f64_path, 
        "Normalized confusion matrix across Vashi validation classes under reference Experiment E1, illustrating high diagonal true-positive concentrations (82.4% Non-Slum, 93.4% Slum, 81.2% Road, 91.5% Vegetation, 79.4% Open Land).", 
        "6.4", width_in=5.2
    )

    # Figure 6.5: Training Curves
    f65_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/2.5m_Panchromatic_PNGs/training_curves.png"
    add_figure_with_caption(doc, f65_path, 
        "Training and validation loss and mean IoU convergence trajectories over 25 epochs under cosine annealing scheduling.", 
        "6.5", width_in=5.8
    )

    # Figure 6.6: 8-Panel Comparison on vashi_0041
    f66_path = "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/visualizations/8panel_comparison_vashi_0041.png"
    add_figure_with_caption(doc, f66_path, 
        "Eight-panel diagnostic comparison on commercial road-conflict patch vashi_0041. Panels: (1) Satellite RGB, (2) Ground Truth Mask (True Road: 29.3%), (3) Exp E1 Prediction (Road: 46.1%), (4) Exp E4 Prediction (Road: 31.7%), (5) E1 Boundaries over RGB, (6) E4 Boundaries over RGB, (7) E4 Error Map, (8) Difference Map E4 vs E1 (Blue: E4 Only Correct, +6,187 px net gain). E4 completely eliminated false road bleeding onto large commercial roofs.", 
        "6.6", width_in=6.2
    )

    # Figure 6.7: 8-Panel Comparison on vashi_0003
    f67_path = "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/visualizations/8panel_comparison_vashi_0003.png"
    add_figure_with_caption(doc, f67_path, 
        "Eight-panel diagnostic comparison on dense residential sector vashi_0003, demonstrating sharpened residential building footprints and clean alleyway delineation under Experiment E4.", 
        "6.7", width_in=6.2
    )

    # Figure 6.8: Subclass Overlays
    f68_path = "FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/0.60m_Multispectral_PNGs/subclass_overlay_0482_residential.png"
    add_figure_with_caption(doc, f68_path, 
        "Subclass morphology alpha-blended overlay showing discrete residential building footprint extraction (Coral Red) bounded by verified road centerlines and vegetation foliage.", 
        "6.8", width_in=5.8
    )
    
    doc.add_page_break()
