"""
scripts/report_builder/front_matter.py
--------------------------------------
Generates the formal IIT Bombay front matter matching BTP_report_Part2.pdf:
- Page 1: Cover / Title Page with authentic IIT Bombay Crest logo and exact layout
- Page 2: Abstract (with Keywords, Roman page number 'i')
- Page 3: Contents (with native dot leaders, Roman page number 'ii')
- Page 4: List of Figures (with native dot leaders, Roman page number 'iii')
- Page 5: List of Tables (with native dot leaders, Roman page number 'iv')
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from styles import (
    COLOR_NAVY_DARK, COLOR_NAVY_PRIMARY, COLOR_STEEL_BLUE,
    COLOR_CHARCOAL, COLOR_MUTED_GRAY,
    add_page_number_field, add_toc_entry
)

def add_front_matter(doc):
    """
    Constructs the exact 5-page front matter sequence matching BTP_report_Part2.pdf:
    Section 1: Cover Page (Unnumbered, centered layout with IITB logo)
    Section 2: Front Matter Pages (Roman numerals i to iv: Abstract, Contents, Figures, Tables)
    """
    # =========================================================================
    # SECTION 1: COVER PAGE (EXACT LAYOUT FROM BTP_report_Part2.pdf)
    # =========================================================================
    sec1 = doc.sections[0]
    sec1.top_margin = Inches(0.8)
    sec1.bottom_margin = Inches(0.8)
    sec1.left_margin = Inches(1.0)
    sec1.right_margin = Inches(1.0)
    sec1.page_width = Inches(8.27)
    sec1.page_height = Inches(11.69)
    sec1.header.is_linked_to_previous = False
    sec1.footer.is_linked_to_previous = False
    for p in sec1.header.paragraphs:
        p.text = ""
    for p in sec1.footer.paragraphs:
        p.text = ""

    # 1. Main Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(16)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.paragraph_format.line_spacing = 1.15
    r_t = p_title.add_run("Urban Feature Extraction from\nHigh-Resolution Satellite Imagery")
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(21)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_NAVY_DARK

    # 2. Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(4)
    p_sub.paragraph_format.space_after = Pt(28)
    p_sub.paragraph_format.line_spacing = 1.15
    r_sub = p_sub.add_run("Multi-Resolution Land-Cover Segmentation and\nHierarchical Urban Feature Mapping")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(14)
    r_sub.font.color.rgb = COLOR_CHARCOAL

    # 3. Report Type
    p_rep = doc.add_paragraph()
    p_rep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_rep.paragraph_format.space_before = Pt(0)
    p_rep.paragraph_format.space_after = Pt(28)
    p_rep.paragraph_format.line_spacing = 1.15
    r_rep1 = p_rep.add_run("B.Tech Project Report\n")
    r_rep1.font.name = "Times New Roman"
    r_rep1.font.size = Pt(17)
    r_rep1.font.bold = True
    r_rep1.font.color.rgb = COLOR_NAVY_DARK
    r_rep2 = p_rep.add_run("Current-Semester Progress and Evaluation")
    r_rep2.font.name = "Times New Roman"
    r_rep2.font.size = Pt(14)
    r_rep2.font.color.rgb = COLOR_CHARCOAL

    # 4. Author Information
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(24)
    p_auth.paragraph_format.line_spacing = 1.15
    r_by = p_auth.add_run("Submitted by\n")
    r_by.font.name = "Times New Roman"
    r_by.font.size = Pt(14)
    r_by.font.color.rgb = COLOR_CHARCOAL
    r_name = p_auth.add_run("Prabhakar Yadav\n")
    r_name.font.name = "Times New Roman"
    r_name.font.size = Pt(17)
    r_name.font.bold = True
    r_name.font.color.rgb = COLOR_NAVY_PRIMARY
    r_roll = p_auth.add_run("Roll No. 23B0653")
    r_roll.font.name = "Times New Roman"
    r_roll.font.size = Pt(14)
    r_roll.font.color.rgb = COLOR_CHARCOAL

    # 5. Supervision Information
    p_sup = doc.add_paragraph()
    p_sup.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sup.paragraph_format.space_before = Pt(0)
    p_sup.paragraph_format.space_after = Pt(22)
    p_sup.paragraph_format.line_spacing = 1.15
    r_under = p_sup.add_run("Under the supervision of\n")
    r_under.font.name = "Times New Roman"
    r_under.font.size = Pt(14)
    r_under.font.color.rgb = COLOR_CHARCOAL
    r_prof = p_sup.add_run("Prof. Gopal Patil")
    r_prof.font.name = "Times New Roman"
    r_prof.font.size = Pt(17)
    r_prof.font.bold = True
    r_prof.font.color.rgb = COLOR_NAVY_PRIMARY

    # 6. Authentic Centered IIT Bombay Logo
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(0)
    p_logo.paragraph_format.space_after = Pt(22)
    logo_path = Path(__file__).resolve().parent / "assets" / "iitb_logo.png"
    if logo_path.exists():
        p_logo.add_run().add_picture(str(logo_path), width=Inches(1.25))

    # 7. Department & Institute
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(0)
    p_inst.paragraph_format.space_after = Pt(0)
    p_inst.paragraph_format.line_spacing = 1.15
    r_dept = p_inst.add_run("Department of Civil Engineering\n")
    r_dept.font.name = "Times New Roman"
    r_dept.font.size = Pt(14)
    r_dept.font.color.rgb = COLOR_CHARCOAL
    r_iit = p_inst.add_run("Indian Institute of Technology Bombay\n")
    r_iit.font.name = "Times New Roman"
    r_iit.font.size = Pt(14)
    r_iit.font.bold = True
    r_iit.font.color.rgb = COLOR_NAVY_PRIMARY
    r_date = p_inst.add_run("November 2026")
    r_date.font.name = "Times New Roman"
    r_date.font.size = Pt(14)
    r_date.font.color.rgb = COLOR_CHARCOAL

    # =========================================================================
    # SECTION 2: FRONT MATTER PAGES (ABSTRACT, CONTENTS, FIGURES, TABLES)
    # Roman page numbering: i, ii, iii, iv centered at bottom
    # =========================================================================
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    sec2.top_margin = Inches(0.85)
    sec2.bottom_margin = Inches(0.85)
    sec2.left_margin = Inches(1.0)
    sec2.right_margin = Inches(1.0)
    sec2.page_width = Inches(8.27)
    sec2.page_height = Inches(11.69)
    sec2.header.is_linked_to_previous = False
    sec2.footer.is_linked_to_previous = False

    # Clear header
    for p in sec2.header.paragraphs:
        p.text = ""

    # Configure lowerRoman page numbering starting at 1 (i)
    sectPr2 = sec2._sectPr
    pgNumType2 = OxmlElement('w:pgNumType')
    pgNumType2.set(qn('w:fmt'), 'lowerRoman')
    pgNumType2.set(qn('w:start'), '1')
    sectPr2.append(pgNumType2)

    # Center-aligned footer with Roman page number
    p_f2 = sec2.footer.paragraphs[0]
    p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_pf2 = p_f2.add_run()
    r_pf2.font.name = "Times New Roman"
    r_pf2.font.size = Pt(10)
    r_pf2.font.color.rgb = COLOR_CHARCOAL
    add_page_number_field(r_pf2)

    # -------------------------------------------------------------------------
    # PAGE 2: ABSTRACT
    # -------------------------------------------------------------------------
    p_ab_h = doc.add_paragraph()
    p_ab_h.paragraph_format.space_before = Pt(12)
    p_ab_h.paragraph_format.space_after = Pt(14)
    r_abh = p_ab_h.add_run("Abstract")
    r_abh.font.name = "Times New Roman"
    r_abh.font.size = Pt(17.2)
    r_abh.font.bold = True
    r_abh.font.color.rgb = COLOR_NAVY_PRIMARY

    p_ab1 = doc.add_paragraph()
    p_ab1.paragraph_format.line_spacing = 1.15
    p_ab1.paragraph_format.space_after = Pt(8)
    r_ab1 = p_ab1.add_run(
        "High-resolution Earth observation satellite imagery provides an essential foundation for urban monitoring, "
        "infrastructure assessment, and dynamic spatial planning. However, converting raw multi-sensor rasters into "
        "accurate, georeferenced thematic layers presents severe computational and spectral challenges: acute spectral "
        "overlap between concrete roofs and dry open ground, shadow occlusion along dense corridors, canopy obstruction "
        "over narrow roads, and extreme spatial heterogeneity in informal settlements. This report presents a comprehensive "
        "deep learning and geospatial engineering framework for multi-resolution land-cover segmentation and hierarchical "
        "urban feature mapping. Building directly upon the foundation established in the previous semester (April 2026), "
        "the current research delivers two major engineering advances: (1) high-resolution feature extraction on 2.5m "
        "Cartosat-1 panchromatic and 0.60m Cartosat-2E multispectral data, and (2) partial-label multi-sensor supervision on "
        "coastal SPOT 4-band and Pléiades urban imagery."
    )
    r_ab1.font.name = "Times New Roman"
    r_ab1.font.size = Pt(10.5)

    p_ab2 = doc.add_paragraph()
    p_ab2.paragraph_format.line_spacing = 1.15
    p_ab2.paragraph_format.space_after = Pt(8)
    r_ab2 = p_ab2.add_run(
        "For 2.5m Cartosat-1 panchromatic data, we automated OpenStreetMap (OSM) vector priority burning (Roads > Built-up > "
        "Water > Vegetation > Barren Land) and established a rigorous benchmark between Vision Transformers (SegFormer-B2) "
        "and Deep Convolutional Networks (UltraRoad ResNet-34 U-Net). The models exhibit complementary strengths: SegFormer "
        "excels in broad geographic and natural context (89.8% Water IoU, 78.5% Vegetation IoU), while U-Net achieves superior "
        "linear road continuity (81.89% Overall Accuracy, 66.63% SpaceNet 10m buffer road IoU). Zero-shot cross-city transfer "
        "across 30 held-out metropolitan patches (49.15 km²) in Pune and Chennai demonstrated strong generalisation, achieving "
        "up to 70.0% SpaceNet road IoU on highway corridors without regional fine-tuning."
    )
    r_ab2.font.name = "Times New Roman"
    r_ab2.font.size = Pt(10.5)

    p_ab3 = doc.add_paragraph()
    p_ab3.paragraph_format.line_spacing = 1.15
    p_ab3.paragraph_format.space_after = Pt(8)
    r_ab3 = p_ab3.add_run(
        "For 0.60m Cartosat-2E multispectral (RGB+NIR) data, we engineered a Unified Hybrid SegFormer-B2 + UltraRoad Attention "
        "U-Net coupled with a scSE Subclass Morphology Head. To eliminate structural false positives, we introduced an In-Epoch "
        "Decoupled Two-Step Joint Optimization scheme where macro land-cover logits serve as a detached spatial gating mask for "
        "fine-grained built-up categorization (Residential, Industrial, and Slums). Across 186 held-out test patches of Pune City, "
        "the model achieved 73.22% Residential IoU, 50.33% Industrial IoU (61.78% Active Subclass Mean), 94.04% Barren buffer IoU, "
        "92.82% Vegetation buffer IoU, 90.77% Water buffer IoU, and an overall 7-class SpaceNet buffer mIoU of 57.15%. A physics-based "
        "Asphalt-Safe Guard completely resolved dark pavement misclassification. On Vashi SPOT data, controlled ablations (E1 to E4) "
        "identified E1 as the primary reference checkpoint (75.01% Val Accuracy, 61.40% mIoU, 66.88% Boundary F1), while Experiment E4 "
        "successfully integrated 5-band NDVI and Focal-Dice loss to resolve long-standing commercial road bloat on sector vashi_0041 "
        "(lifting patch accuracy to 70.95% and Non-slum IoU to 67.11%)."
    )
    r_ab3.font.name = "Times New Roman"
    r_ab3.font.size = Pt(10.5)

    p_ab4 = doc.add_paragraph()
    p_ab4.paragraph_format.line_spacing = 1.15
    p_ab4.paragraph_format.space_after = Pt(14)
    r_ab4 = p_ab4.add_run(
        "The end-to-end software platform preserves native geospatial projections (UTM / EPSG:32643), performs tiled sliding-window "
        "inference with 2D Hann window blending to eliminate edge artefacts, and provides automated GIS export to GeoTIFF, ESRI Shapefiles, "
        "and GeoJSON verified directly in QGIS Desktop. Full-scene deployment processed over 18,000 × 17,000 pixel scenes efficiently "
        "within strict hardware constraints (laptop RTX 3050 4GB VRAM). The primary contribution of this semester is a unified, "
        "methodologically rigorous remote sensing platform that advances from weak-label rasterization to sub-meter subclass mapping."
    )
    r_ab4.font.name = "Times New Roman"
    r_ab4.font.size = Pt(10.5)

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.line_spacing = 1.15
    p_kw.paragraph_format.space_after = Pt(0)
    r_kw_lbl = p_kw.add_run("Keywords: ")
    r_kw_lbl.font.name = "Times New Roman"
    r_kw_lbl.font.size = Pt(10)
    r_kw_lbl.font.bold = True
    r_kw_txt = p_kw.add_run(
        "remote sensing; semantic segmentation; panchromatic imagery; multispectral imagery; Cartosat-2E; "
        "attention U-Net; SegFormer; hierarchical classification; subclass morphology; road extraction; "
        "partial-label supervision; GIS export."
    )
    r_kw_txt.font.name = "Times New Roman"
    r_kw_txt.font.size = Pt(10)

    doc.add_page_break()

    # -------------------------------------------------------------------------
    # PAGE 3: CONTENTS
    # -------------------------------------------------------------------------
    p_toc_h = doc.add_paragraph()
    p_toc_h.paragraph_format.space_before = Pt(12)
    p_toc_h.paragraph_format.space_after = Pt(14)
    r_toch = p_toc_h.add_run("Contents")
    r_toch.font.name = "Times New Roman"
    r_toch.font.size = Pt(17.2)
    r_toch.font.bold = True
    r_toch.font.color.rgb = COLOR_NAVY_PRIMARY

    add_toc_entry(doc, "Abstract", "i", level=1, is_bold=True)
    add_toc_entry(doc, "1 Introduction", "1", level=1, is_bold=True)
    add_toc_entry(doc, "1.1 Motivation and Problem Setting", "1", level=2)
    add_toc_entry(doc, "1.2 Previous-Semester Foundation (April 2026)", "2", level=2)
    add_toc_entry(doc, "1.3 Current-Semester Objectives and Core Contributions", "3", level=2)
    add_toc_entry(doc, "1.4 Research Questions and Scope", "4", level=2)
    add_toc_entry(doc, "1.5 Report Organization", "5", level=2)

    add_toc_entry(doc, "2 Background and Literature Review", "6", level=1, is_bold=True)
    add_toc_entry(doc, "2.1 High-Resolution Urban Satellite Imagery: Resolution, Radiometry and Ambiguity", "6", level=2)
    add_toc_entry(doc, "2.2 Spectral Reflectance Models and Biophysical Indices (NDVI, NDWI)", "7", level=2)
    add_toc_entry(doc, "2.3 Deep Encoder-Decoder Networks for Semantic Segmentation", "8", level=2)
    add_toc_entry(doc, "2.4 Vision Transformers and Attention Mechanisms in Earth Observation", "9", level=2)
    add_toc_entry(doc, "2.5 Hierarchical and Multi-Level Land-Cover Categorization", "10", level=2)
    add_toc_entry(doc, "2.6 Loss Formulations for Imbalanced, Boundary and Topological Structures", "11", level=2)
    add_toc_entry(doc, "2.7 Weak and Pseudo-Supervision from Open Geospatial Data", "12", level=2)
    add_toc_entry(doc, "2.8 Evaluation Protocols Beyond Pixel Accuracy", "13", level=2)

    add_toc_entry(doc, "3 Datasets and Ground-Truth Preparation", "14", level=1, is_bold=True)
    add_toc_entry(doc, "3.1 Multi-Sensor Remote Sensing Inventory", "14", level=2)
    add_toc_entry(doc, "3.2 Dataset 1: Cartosat-1 2.5m Panchromatic (Mumbai & Navi Mumbai)", "15", level=2)
    add_toc_entry(doc, "3.3 Dataset 2: Cartosat-2E 0.60m Multispectral (Pune & Chennai)", "17", level=2)
    add_toc_entry(doc, "3.4 Dataset 3: Vashi SPOT 4-Band & Mumbai Pléiades Cross-Sensor Benchmark", "19", level=2)
    add_toc_entry(doc, "3.5 Class Distribution and Statistical Imbalance Analysis", "21", level=2)

    add_toc_entry(doc, "4 Methodology and Model Architecture", "23", level=1, is_bold=True)
    add_toc_entry(doc, "4.1 Overall Architectural Philosophy", "23", level=2)
    add_toc_entry(doc, "4.2 Dataset 1 Architecture: SegFormer-B2 vs. ResNet-34 U-Net", "24", level=2)
    add_toc_entry(doc, "4.3 Dataset 2 Architecture: Unified Hybrid SegFormer + UltraRoad Attention U-Net", "26", level=2)
    add_toc_entry(doc, "4.4 In-Epoch Decoupled Two-Step Hierarchical Joint Optimization", "28", level=2)
    add_toc_entry(doc, "4.5 Physics-Based Asphalt-Safe Guard", "30", level=2)
    add_toc_entry(doc, "4.6 Composite Loss Formulations (Focal, Lovász-Softmax, Focal-Dice)", "31", level=2)

    add_toc_entry(doc, "5 Experiments and Model Development", "33", level=1, is_bold=True)
    add_toc_entry(doc, "5.1 Experimental Setup and Hardware Constraints (RTX 3050 4GB VRAM Cap)", "33", level=2)
    add_toc_entry(doc, "5.2 Training Configurations and Pacing Strategy", "34", level=2)
    add_toc_entry(doc, "5.3 Dataset 1 Experiments: SegFormer vs. U-Net Comparison", "35", level=2)
    add_toc_entry(doc, "5.4 Dataset 1 Zero-Shot Cross-City Transfer (Pune Core, Hinjwadi, Chennai)", "36", level=2)
    add_toc_entry(doc, "5.5 Dataset 2 Hierarchical Model Training Dynamics and Convergence", "37", level=2)
    add_toc_entry(doc, "5.6 Vashi Controlled Model Progression (Baseline to Experiment E4)", "38", level=2)
    add_toc_entry(doc, "5.7 Model Selection Protocol and Strict Decision Rules", "41", level=2)

    add_toc_entry(doc, "6 Results, Quantitative Benchmarks, and Visual Evaluation", "43", level=1, is_bold=True)
    add_toc_entry(doc, "6.1 Dataset 1 (2.5m) Quantitative Results: SegFormer vs. U-Net", "43", level=2)
    add_toc_entry(doc, "6.2 Dataset 1 Zero-Shot Transfer Across 30 Metropolitan Patches", "45", level=2)
    add_toc_entry(doc, "6.3 Dataset 2 (0.60m) Multi-Level Benchmark on 186 Held-Out Pune Patches", "46", level=2)
    add_toc_entry(doc, "6.4 Vashi Multisensor Model Benchmark: E1 vs. E2 vs. E3 vs. E4", "49", level=2)
    add_toc_entry(doc, "6.5 Visual Evidence and Qualitative Multi-Panel Comparisons", "51", level=2)

    add_toc_entry(doc, "7 Implementation and System Workflow", "55", level=1, is_bold=True)
    add_toc_entry(doc, "7.1 Software Repository Architecture", "55", level=2)
    add_toc_entry(doc, "7.2 Training Engine and Memory Optimization", "56", level=2)
    add_toc_entry(doc, "7.3 Large-Scale Sliding-Window Inference with 2D Hann Blending", "57", level=2)
    add_toc_entry(doc, "7.4 Georeferenced Vector and Raster Export (GeoTIFF, Shapefiles, GeoJSON)", "58", level=2)
    add_toc_entry(doc, "7.5 GIS Integration and Validation in QGIS Desktop", "59", level=2)
    add_toc_entry(doc, "7.6 Computational Efficiency, Runtime Analysis, and VRAM Profiling", "60", level=2)

    add_toc_entry(doc, "8 Conclusions and Future Work", "62", level=1, is_bold=True)
    add_toc_entry(doc, "8.1 Summary of Completed Research and Engineering Deliverables", "62", level=2)
    add_toc_entry(doc, "8.2 Answers to Research Questions", "63", level=2)
    add_toc_entry(doc, "8.3 Current Limitations and Technical Bottlenecks", "64", level=2)
    add_toc_entry(doc, "8.4 Prioritized Roadmap for Semester VIII", "65", level=2)

    add_toc_entry(doc, "References", "67", level=1, is_bold=True)
    add_toc_entry(doc, "Appendix A: Model Checkpoint Inventory and Verification Manifest", "72", level=1, is_bold=True)
    add_toc_entry(doc, "Appendix B: Software Environment and Hardware Specifications", "74", level=1, is_bold=True)
    add_toc_entry(doc, "Appendix C: Top 10 Best-Performing Test Patches on Cartosat-2E Benchmark", "75", level=1, is_bold=True)
    add_toc_entry(doc, "Appendix D: Formal Institute Submission Documents", "77", level=1, is_bold=True)

    doc.add_page_break()

    # -------------------------------------------------------------------------
    # PAGE 4: LIST OF FIGURES
    # -------------------------------------------------------------------------
    p_lof_h = doc.add_paragraph()
    p_lof_h.paragraph_format.space_before = Pt(12)
    p_lof_h.paragraph_format.space_after = Pt(14)
    r_lofh = p_lof_h.add_run("List of Figures")
    r_lofh.font.name = "Times New Roman"
    r_lofh.font.size = Pt(17.2)
    r_lofh.font.bold = True
    r_lofh.font.color.rgb = COLOR_NAVY_PRIMARY

    add_toc_entry(doc, "Figure 4.1: Unified Hybrid SegFormer-B2 + UltraRoad Attention U-Net Architecture", "27")
    add_toc_entry(doc, "Figure 4.2: In-Epoch Decoupled Two-Step Hierarchical Training Pipeline", "29")
    add_toc_entry(doc, "Figure 5.1: Training Loss and Validation IoU Trajectory for Hybrid Hierarchical Model", "37")
    add_toc_entry(doc, "Figure 6.1: Verified Hierarchical Multi-Level Prediction on Top Patch test_pune_0413 (0.60m)", "51")
    add_toc_entry(doc, "Figure 6.2: SegFormer-B2 Top-Performing Zero-Shot Transfer Corridors Mosaic", "52")
    add_toc_entry(doc, "Figure 6.3: Quantitative Benchmark Comparison (SegFormer-B2 vs. UltraRoad U-Net)", "53")
    add_toc_entry(doc, "Figure 6.4: Normalized Confusion Matrix on Vashi Multi-Class Validation Set", "53")
    add_toc_entry(doc, "Figure 6.5: Training Loss and Mean IoU Convergence Trajectories over 25 Epochs", "54")
    add_toc_entry(doc, "Figure 6.6: Eight-Panel Visual Diagnostic Comparison on Conflict Sector vashi_0041 (E4 vs. E1)", "54")
    add_toc_entry(doc, "Figure 6.7: Eight-Panel Visual Diagnostic Comparison on Dense Residential Sector vashi_0003", "55")
    add_toc_entry(doc, "Figure 6.8: Subclass Morphology Alpha-Blended Overlays: Residential Footprints & Boundaries", "55")
    add_toc_entry(doc, "Figure 7.1: Sliding-Window Hann-Window Blending Scheme for Large Satellite Scene Tiling", "57")
    add_toc_entry(doc, "Figure 7.2: QGIS Desktop Verification of Georeferenced Polygons Exported from GeoVision", "59")

    doc.add_page_break()

    # -------------------------------------------------------------------------
    # PAGE 5: LIST OF TABLES
    # -------------------------------------------------------------------------
    p_lot_h = doc.add_paragraph()
    p_lot_h.paragraph_format.space_before = Pt(12)
    p_lot_h.paragraph_format.space_after = Pt(14)
    r_loth = p_lot_h.add_run("List of Tables")
    r_loth.font.name = "Times New Roman"
    r_loth.font.size = Pt(17.2)
    r_loth.font.bold = True
    r_loth.font.color.rgb = COLOR_NAVY_PRIMARY

    add_toc_entry(doc, "Table 1.1: Previous-Semester Results Summary (April 2026 Reference Baseline)", "2")
    add_toc_entry(doc, "Table 3.1: Multi-Sensor Remote Sensing Satellite Imagery Inventory", "14")
    add_toc_entry(doc, "Table 3.2: Five-Class Macro Land-Cover Taxonomy (Cartosat-1 2.5m Dataset)", "16")
    add_toc_entry(doc, "Table 3.3: Seven-Class Hierarchical Taxonomy and Subclass Morphology (Cartosat-2E 0.60m)", "18")
    add_toc_entry(doc, "Table 3.4: Vashi SPOT 4-Band & Mumbai Geographic Dataset Partition Manifest", "20")
    add_toc_entry(doc, "Table 3.5: Class Pixel Frequencies and Imbalance Ratios Across Evaluation Sets", "22")
    add_toc_entry(doc, "Table 4.1: Architectural Hyperparameters: SegFormer-B2, UltraRoad U-Net, Subclass Head", "25")
    add_toc_entry(doc, "Table 5.1: Hardware Enforcement and Compute Duty Cycle Parameters (RTX 3050 Laptop GPU)", "34")
    add_toc_entry(doc, "Table 5.2: In-Epoch Training Loss Dynamics and Subclass Convergence Trajectory", "37")
    add_toc_entry(doc, "Table 5.3: Vashi Controlled Experimental Progression (Baseline to Experiment E4)", "40")
    add_toc_entry(doc, "Table 6.1: Dataset 1 (2.5m) Benchmark: SegFormer vs. U-Net on Navi Mumbai & Mumbai Central", "44")
    add_toc_entry(doc, "Table 6.2: Zero-Shot Transfer Road IoU Across 30 Metropolitan Test Patches (Pune & Chennai)", "45")
    add_toc_entry(doc, "Table 6.3: Dataset 2 (0.60m) Step 1: Macro 5-Class SpaceNet & Mnih Benchmark (186 Scenes)", "47")
    add_toc_entry(doc, "Table 6.4: Dataset 2 (0.60m) Step 2: Hierarchical Built-Up Subclass & 7-Class Unified Benchmark", "48")
    add_toc_entry(doc, "Table 6.5: Vashi Multisensor Internal Validation Benchmark (Reference E1 vs. E2 vs. E3 vs. E4)", "49")
    add_toc_entry(doc, "Table 6.6: Diagnostic Target Patch Benchmark Across Urban Typologies (vashi_0041, 0003, 0020)", "50")
    add_toc_entry(doc, "Table 7.1: Sliding-Window Inference Runtime, Latency, and Throughput Benchmarks", "60")
    add_toc_entry(doc, "Table A.1: Verified Model Checkpoint Inventory, Epoch Counts, and Storage Footprints", "72")
    add_toc_entry(doc, "Table B.1: Software Stack, Library Versions, and CUDA Driver Configuration", "74")
    add_toc_entry(doc, "Table C.1: Top 10 Best-Performing Test Patches on Held-Out Cartosat-2E Scene", "75")
