"""
Script to generate a 14-slide presentation in PPTX format:
'Urban Feature Extraction from High-Resolution Satellite Imagery'
Focusing on:
1. Previous work: Monochromatic Cartosat 2.5m (JNPA) and 1.0m data,
   pseudo-labeling, two architectures (U-Net vs Random Forest / ResNet),
   visual comparisons, test results, and web platform.
2. Last 2-3 slides: Transition to 0.60m multispectral Cartosat-2E data,
   NDVI physics, clDice topology-aware vectorizer, and test benchmark.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Clean White and Blue Executive Palette
    NAVY = RGBColor(10, 37, 64)       # Deep Navy Header
    STEEL = RGBColor(30, 96, 145)     # Steel Blue Accent
    SKY = RGBColor(0, 119, 182)       # Vivid Sky Blue
    BG_CARD = RGBColor(245, 248, 252) # Soft Ice Blue Card Fill
    BORDER_COL = RGBColor(218, 228, 240) # Card Border
    DARK_TEXT = RGBColor(28, 39, 76)  # Dark Slate Primary Text
    MUTED_TEXT = RGBColor(74, 85, 104) # Muted Secondary Text
    WHITE = RGBColor(255, 255, 255)
    BADGE_BG = RGBColor(230, 242, 254)

    def add_header(slide, kicker, title):
        # Kicker / Category Tracker
        k_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_k = k_box.text_frame
        tf_k.word_wrap = True
        tf_k.margin_left = tf_k.margin_right = tf_k.margin_top = tf_k.margin_bottom = 0
        p_k = tf_k.paragraphs[0]
        p_k.text = kicker.upper()
        p_k.font.size = Pt(11)
        p_k.font.bold = True
        p_k.font.color.rgb = SKY
        p_k.font.name = "Arial"

        # Main Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.6))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY
        p_t.font.name = "Arial"

        # Subtle divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COL
        line.line.color.rgb = BORDER_COL

    def add_card(slide, left, top, width, height, title, items, badge=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = BG_CARD
        card.line.color.rgb = BORDER_COL
        card.line.width = Pt(1.2)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.28)
        tf.margin_right = Inches(0.28)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.2)

        # Title
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(15)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = "Arial"
        p0.space_after = Pt(10)

        for item in items:
            p = tf.add_paragraph()
            p.space_after = Pt(7)
            if ":" in item:
                lead, rest = item.split(":", 1)
                r_lead = p.add_run()
                r_lead.text = f"•  {lead}:"
                r_lead.font.bold = True
                r_lead.font.size = Pt(13)
                r_lead.font.color.rgb = DARK_TEXT
                r_lead.font.name = "Arial"

                r_rest = p.add_run()
                r_rest.text = rest
                r_rest.font.bold = False
                r_rest.font.size = Pt(13)
                r_rest.font.color.rgb = MUTED_TEXT
                r_rest.font.name = "Arial"
            else:
                r = p.add_run()
                r.text = f"•  {item}"
                r.font.size = Pt(13)
                r.font.color.rgb = DARK_TEXT
                r.font.name = "Arial"

    # ==========================================
    # SLIDE 1: Title Slide (Hero Card)
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    hero = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.0), Inches(11.333), Inches(5.5))
    hero.fill.solid()
    hero.fill.fore_color.rgb = BG_CARD
    hero.line.color.rgb = STEEL
    hero.line.width = Pt(2.0)

    tf1 = hero.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.8)
    tf1.margin_right = Inches(0.8)
    tf1.margin_top = Inches(0.6)

    p1 = tf1.paragraphs[0]
    p1.text = "RESEARCH PROGRESS PRESENTATION".upper()
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = SKY
    p1.font.name = "Arial"
    p1.space_after = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Urban Feature Extraction from High-Resolution Satellite Imagery"
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Arial"
    p2.space_after = Pt(12)

    p3 = tf1.add_paragraph()
    p3.text = "Evolution from 2.5m Monochromatic Cartosat Modeling to 0.60m Multispectral Extraction"
    p3.font.size = Pt(17)
    p3.font.color.rgb = STEEL
    p3.font.name = "Arial"
    p3.space_after = Pt(36)

    p4 = tf1.add_paragraph()
    p4.text = "• Student: Prabhakar Yadav (Roll No: 23B0653)   |   Dept. of Civil Engineering, IIT Bombay\n• Supervisor: Prof. Pennan Chinnasamy   |   Dept. of Civil Engineering, IIT Bombay\n• Core Topics: Panchromatic Feature Engineering, Pseudo-Labeling, Dual Architectures & Web Platform"
    p4.font.size = Pt(13)
    p4.font.color.rgb = MUTED_TEXT
    p4.font.name = "Arial"

    # ==========================================
    # SLIDE 2: Research Motivation & Problem Statement
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Overview & Motivation", "Project Objectives & Challenges in Panchromatic Imagery")

    add_card(s2, 0.8, 1.6, 5.7, 5.2, "Core Research Objective", [
        "Goal: Automated extraction of urban features (water, buildings, vegetation, roads, open soil) from Indian satellite scenes.",
        "Ground Truth Constraint: Lack of hand-annotated, pixel-accurate ground truth masks for regional Indian satellite data.",
        "Automation Strategy: Implement an end-to-end unsupervised pseudo-labeling pipeline to train deep networks without manual labeling.",
        "Operational Target: Build an interactive, deployable GIS platform capable of processing massive GeoTIFF imagery."
    ])

    add_card(s2, 6.8, 1.6, 5.7, 5.2, "Key Challenges in Monochromatic Data", [
        "Single-Band Limitation: Grayscale imagery contains only intensity; zero multispectral color or Near-Infrared (NIR) data.",
        "High Spectral Confusion: Built-up concrete, dry open soil, and port tarmac exhibit almost identical grayscale reflectances.",
        "Shadow & Water Overlap: Deep building shadows and turbid water bodies share similar dark pixel intensities.",
        "Need for Spatial Context: Classical per-pixel thresholding fails; spatial textures and edge distributions are mandatory."
    ])

    # ==========================================
    # SLIDE 3: Monochromatic Dataset: Cartosat 2.5m & 1.0m
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Specifications", "Panchromatic Cartosat Imagery & Patch Pre-Processing")

    add_card(s3, 0.8, 1.6, 5.7, 5.2, "Satellite Sensor Characteristics", [
        "Primary Dataset: JNPA Port Region (JNPA_2_5.tif) at 2.5m Ground Sampling Distance (GSD).",
        "Secondary Dataset: Cartosat-1 Panchromatic scene (CARTOSAT_1M_PAN.tif) at 1.0m GSD.",
        "Radiometric Profile: Single-band grayscale GeoTIFFs, high bit-depth requiring dynamic range compression.",
        "Geographic Context: Complex coastal, industrial port, dense residential, and mangrove ecosystems."
    ])

    add_card(s3, 6.8, 1.6, 5.7, 5.2, "Grid Tiling & Quality Filtering", [
        "Patch Generation: Large GeoTIFF split into regular non-overlapping 256x256 pixel tiles.",
        "Total Grid Extent: 1,015 candidate patches extracted across the JNPA survey extent.",
        "Variance Thresholding: Low-information, completely uniform patches filtered out (Std Dev threshold > 0.045).",
        "Valid Data Fraction: Only patches with >85% valid pixel coverage retained (735 high-quality patches selected)."
    ])

    # ==========================================
    # SLIDE 4: Feature Extraction & Pseudo-Labeling Strategy
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Ground Truth Generation", "Multi-Feature Extraction & Unsupervised Pseudo-Labeling")

    add_card(s4, 0.8, 1.6, 3.7, 5.2, "Feature Engineering", [
        "Intensity Stats: Mean reflectance, standard deviation, and quartile distribution per patch.",
        "Texture (GLCM): Contrast, homogeneity, energy, and entropy capturing surface roughness.",
        "Edge Density: Canny edge detector (60/160 thresholds) capturing structural boundaries.",
        "PCA Compression: 25 principal components extracted per patch for compact representation."
    ])

    add_card(s4, 4.8, 1.6, 3.7, 5.2, "Clustering Engine", [
        "Unsupervised Grouping: K-Means and GMM applied over 25-dimensional PCA feature vectors.",
        "Cluster Interpretation: Domain verification comparing clusters against satellite reference.",
        "Stability Override: Added dark-water heuristic override to ensure deep water bodies stay clean.",
        "Mask Synthesis: Cluster assignments mapped to semantic pseudo-label masks."
    ])

    add_card(s4, 8.8, 1.6, 3.7, 5.2, "Target Land-Use Classes", [
        "Class 0 - Water Bodies: Deep open water, creeks, and harbor basins.",
        "Class 1 - Industrial / Port: Port berths, heavy container yards, terminals.",
        "Class 2 - Bare Land / Soil: Cleared land, exposed ground, unpaved areas.",
        "Class 3 - Vegetation: Mangrove forests, natural tree cover, green patches.",
        "Class 4 - Urban Built-up: Residential clusters, commercial structures, roads."
    ])

    # ==========================================
    # SLIDE 5: Visualizing Pseudo-Labels & Class Formation (Image Slide)
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Visual Demonstration", "Pseudo-Label Generation & Cluster Mapping Showcase")

    img_p5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dense_label_preview_panel.png")
    if os.path.exists(img_p5):
        s5.shapes.add_picture(img_p5, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s5, 8.3, 1.6, 4.2, 5.2, "Pseudo-Label Analysis", [
        "Visual Alignment: Grayscale satellite patches (left) accurately matched to semantic class masks (right).",
        "Water Separation: Water bodies cleanly segmented with zero confusion against land.",
        "High-Contrast Boundaries: High-frequency texture in container yards captured under Class 1 (Industrial).",
        "Training Ready: Provided hundreds of annotated patches without a single hour of manual tracing."
    ])

    # ==========================================
    # SLIDE 6: Architecture 1: Semantic Segmentation (U-Net)
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Deep Learning Modeling", "Architecture 1: End-to-End U-Net Semantic Segmentation")

    add_card(s6, 0.8, 1.6, 5.7, 5.2, "Model Structure & Loss", [
        "Encoder-Decoder Network: Multi-scale convolutional contraction with symmetric expanding decoder.",
        "Skip Connections: Concatenates low-level spatial detail directly to high-level semantic features.",
        "Loss Formulation: Joint Cross-Entropy + Soft Dice Loss for multi-class class imbalance handling.",
        "Data Augmentation: Random horizontal/vertical flips, 90-degree rotations, contrast jitter, and noise."
    ])

    add_card(s6, 6.8, 1.6, 5.7, 5.2, "V1 Baseline vs V128 Dense Upgrade", [
        "Initial 256x256 V1 Result: Pixel Accuracy 48.96%, Mean IoU 33.50% (best epoch 17).",
        "Why V1 Struggled: The U-Net attempted pixel-level boundary segmentation on patch-uniform pseudo-labels.",
        "The V128 Dense Upgrade: Retrained on dense 128x128 soft probability pseudo-masks.",
        "Surge in Performance: V128 achieved 91.07% Pixel Accuracy and 78.73% Mean IoU (+42.8% gain)."
    ])

    # ==========================================
    # SLIDE 7: Architecture 2: Patch Classification (Random Forest & ResNet)
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Alternative Architecture", "Architecture 2: Random Forest & Deep ResNet Classifiers")

    add_card(s7, 0.8, 1.6, 5.7, 5.2, "Random Forest Pipeline (JNPA & Cartosat)", [
        "Ensemble Architecture: 1,000 decision trees trained directly on the 25 PCA features + Canny edge stats.",
        "Task-Model Alignment: The prediction unit (256x256 patch) perfectly matched the pseudo-label unit.",
        "JNPA 2.5m Result: 87.76% Validation Accuracy, 85.56% Macro F1 Score.",
        "Cartosat 1m (4-Class): 89.78% Validation Accuracy, 74.64% Macro F1 (separating port from dense urban)."
    ])

    add_card(s7, 6.8, 1.6, 5.7, 5.2, "ResNet-18 Deep Feature Classifier", [
        "Transfer Learning Backbone: Pretrained ResNet-18 feature extractor with fine-tuned classification head.",
        "Deep Spatial Embeddings: Replaced hand-crafted GLCM features with learned convolutional representations.",
        "Validation Accuracy: Achieved 94.74% Overall Accuracy on Cartosat panchromatic test patches.",
        "Stability: Zero training instability; instantaneous inference ideal for large satellite mosaic streaming."
    ])

    # ==========================================
    # SLIDE 8: Visual Comparison & Training Results (Image Slide)
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Visual Demonstration", "Model Predictions vs Clustering Ground Truth")

    img_p8 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "comparison_summary_board.png")
    if os.path.exists(img_p8):
        s8.shapes.add_picture(img_p8, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s8, 8.3, 1.6, 4.2, 5.2, "Visual Comparison Insights", [
        "Clustering vs Learning: The trained models reproduce the major land-use distribution with high spatial fidelity.",
        "Noise Filtering: U-Net and Random Forest smoothed out isolated spurious pixel noise seen in raw clustering.",
        "Boundary Crispness: Dense U-Net V128 achieved continuous spatial corridors across coastal shorelines.",
        "Model Selection: Random Forest chosen for rapid patch scoring; Dense U-Net chosen for spatial mapping."
    ])

    # ==========================================
    # SLIDE 9: Quantitative Test Evaluation (Monochromatic Data)
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Performance Benchmark", "Quantitative Test Results Across Urban & Natural Classes")

    add_card(s9, 0.8, 1.6, 5.7, 5.2, "Held-Out Class Accuracies", [
        "Water Bodies: 97.30% accuracy (strongest absorption contrast across all models).",
        "Dense Urban Built-up: 94.74% accuracy (high edge frequency reliably detected).",
        "Industrial / Port: 80.00% accuracy (distinct tarmac and berth signatures).",
        "Vegetation / Mangroves: 100.00% accuracy on held-out patch classification tests.",
        "Bare Land / Soil: 53.33% accuracy (most challenging due to grayscale similarity with dry terrain)."
    ])

    img_p9 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Test_Model\results\accuracy_cartosat_confusion.png"
    if os.path.exists(img_p9):
        s9.shapes.add_picture(img_p9, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # ==========================================
    # SLIDE 10: GeoVision Web Platform Deployment
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Platform Engineering", "Interactive Web Platform for End-to-End GIS Inference")

    add_card(s10, 0.8, 1.6, 5.7, 5.2, "System Architecture & Features", [
        "Unified Web Application: Built with Flask / Streamlit for user-friendly browser access.",
        "Auto-Model Selection: Automatically detects image bands and resolution, routing to the best model.",
        "Large-Scale Tiling: Sliding-window inference with Hann-window blending eliminates tile seam artifacts.",
        "Multi-Format Ingestion: Supports GeoTIFF, PNG, JPEG, and raw NumPy arrays up to 500 MB."
    ])

    add_card(s10, 6.8, 1.6, 5.7, 5.2, "GIS Export & Practical Utility", [
        "GeoTIFF Export: Reconstructs full-resolution geo-referenced classification GeoTIFFs.",
        "Vector Shapefiles: Automated vector polygon extraction exported as ESRI Shapefiles and GeoJSON.",
        "CRS Preservation: Retains exact spatial reference systems (EPSG:4326 / EPSG:32643) for GIS integration.",
        "Area Statistics: Interactive dashboards displaying class-wise square kilometer coverage."
    ])

    # ==========================================
    # SLIDE 11: Limitations of Monochromatic 2.5m Data
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Diagnostic Transition", "Why We Moved from 2.5m Panchromatic to 0.60m Multispectral")

    add_card(s11, 0.8, 1.6, 3.7, 5.2, "1. Spatial Resolution Limit", [
        "Pixel Dimension: At 2.5m, major roads are only 2–4 pixels wide in total.",
        "Loss of Narrow Linear Paths: Narrow streets and corridors merge into adjacent buildings.",
        "Structural Fragmentation: The model produces disconnected 'jelly blobs' rather than continuous ribbons."
    ])

    add_card(s11, 4.8, 1.6, 3.7, 5.2, "2. Spectral Blindness", [
        "Single Band: Grayscale intensity cannot reliably separate green vegetation from dark shadows.",
        "Missing NIR: Near-Infrared is essential to compute physical vegetation indices like NDVI.",
        "Shadow Confusion: Deep building and cloud shadows share pixel values with asphalt roads."
    ])

    add_card(s11, 8.8, 1.6, 3.7, 5.2, "3. Path to Semester 2", [
        "Sensor Leap: Transition to Cartosat-2E at 0.60m resolution (>4x sharper).",
        "Multispectral Bands: 4 Bands (Red, Green, Blue, NIR) enabling physical ground truth.",
        "Topology-Aware Learning: Replace standard segmentation with connectivity-enforcing models."
    ])

    # ==========================================
    # SLIDE 12: Transition to Current Work: Cartosat-2E 0.60m Multispectral Data
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Current Work: Data & Physics", "Transition to 0.60m Cartosat-2E Multispectral Imagery")

    add_card(s12, 0.8, 1.6, 5.2, 5.2, "0.60m Multispectral Advantages", [
        "Sub-Meter Resolution: 0.60m GSD resolves individual road lanes, building footprints, and tree crowns.",
        "4 Spectral Channels: Red, Green, Blue, and Near-Infrared (NIR) bands.",
        "Physical NDVI Ground Truth: NDVI = (NIR - Red)/(NIR + Red) cleanly isolates 100% of vegetation automatically.",
        "OSM Vector Fusion: Fusing OpenStreetMap road centerlines with metric road widths for automated corridor ground truth."
    ])

    img_p12 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\test_2673727101_r15200_c11200_comparison.png"
    if os.path.exists(img_p12):
        s12.shapes.add_picture(img_p12, Inches(6.3), Inches(1.6), width=Inches(6.2))

    # ==========================================
    # SLIDE 13: Current Work: Topology-Aware Road Extraction Pipeline
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Current Work: Modeling & Results", "Topology-Aware Road Extraction & Benchmark Results")

    add_card(s13, 0.8, 1.6, 5.2, 5.2, "Key Pipeline Modifications", [
        "Multi-Task Geometric Vectorizer: Predicts centerlines, corridor edges, and constant widths simultaneously.",
        "Differentiable clDice Loss: Centerline Dice loss explicitly penalizes road disconnections under tree canopies.",
        "Affine Vector Alignment: Corrects 2-4m orthorectification shifts before rasterizing pseudo-labels.",
        "Test Performance (173 Patches): 86.99% Overall Accuracy, 60.95% Road Recall, and 83.57% Active Corridor mIoU."
    ])

    img_p13 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\constant_width_continuous_roads_preview.png"
    if os.path.exists(img_p13):
        s13.shapes.add_picture(img_p13, Inches(6.3), Inches(1.6), width=Inches(6.2))

    # ==========================================
    # SLIDE 14: Summary & Immediate Next Steps
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion & Future Roadmap", "Summary of Progress & Immediate Research Next Steps")

    add_card(s14, 0.8, 1.6, 5.7, 5.2, "Summary of Achievements", [
        "Panchromatic Milestone: Completed full pipeline for 2.5m & 1.0m monochromatic Cartosat imagery.",
        "Pseudo-Label Feasibility: Validated that unsupervised clustering + PCA can train high-accuracy networks (up to 94%).",
        "Deployed Platform: Successfully created the GeoVision web platform for automated multi-model inference.",
        "Sub-Meter Breakthrough: Transitioned to 0.60m multispectral data, solving road continuity with clDice topology modeling."
    ])

    add_card(s14, 6.8, 1.6, 5.7, 5.2, "Current Focus & Next Steps", [
        "Cross-City Generalization: Testing transferability on unseen Chennai multispectral scenes without retraining.",
        "Direct CAD/GIS Graph Export: Converting centerline predictions into topological vector graphs (GeoJSON / Shapefiles).",
        "Routability Evaluation: Measuring graph connectivity using APLS (Average Path Length Similarity) metric.",
        "Final Report & Thesis: Compiling the dual-stage framework for publication and final thesis defense."
    ])

    output_path = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Cartosat_Urban_Feature_Extraction_Progress.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()
