"""
Script to generate ONE unified, clean, simple, and easy-to-explain presentation:
'Urban_Feature_Extraction_Progress_Presentation.pptx'

Flow:
1. Title Slide (Student, Supervisor, Dept)
2. Project Motivation & Goal (Feature extraction without manual labels)
3. Dataset 1: Panchromatic Cartosat 2.5m (JNPA) & Pre-processing (tiling, filtering)
4. Pseudo-Labeling: Feature extraction (GLCM, Canny, PCA) + K-Means Clustering (5 classes)
5. Visualizing Pseudo-Labels (Image: dense_label_preview_panel.png)
6. Architecture 1: U-Net Semantic Segmentation (Dice + CE loss, V1 vs V128 dense upgrade)
7. Architecture 2: Random Forest & ResNet Patch Classifier (Why patch classification worked)
8. Visual Comparison & Training Results (Image: comparison_summary_board.png)
9. Test Evaluation on Monochromatic Data (Class-wise accuracies + Confusion matrix image)
10. Web Platform Deployment (GeoVision app: auto-selection, streaming, GIS export)
11. Limitations of Monochromatic 2.5m Data (Resolution ceiling, spectral blindness)
12. New Dataset: Cartosat-2E 0.60m Multispectral (4 bands: RGB+NIR, NDVI physics, OSM labels)
13. New Model Training & Results (clDice topology loss, continuous roads, 87% accuracy, result image)
14. Summary & Next Steps (Clear progress overview)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_unified_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Clean White and Blue Palette
    NAVY = RGBColor(10, 37, 64)          # Primary Deep Navy
    STEEL = RGBColor(30, 96, 145)        # Steel Blue Accent
    SKY = RGBColor(0, 119, 182)          # Category Accent Blue
    CARD_BG = RGBColor(245, 248, 252)    # Soft Ice-Blue Card Background
    CARD_BORDER = RGBColor(218, 228, 240)# Card Border
    DARK_TEXT = RGBColor(28, 39, 76)     # Primary Text
    MUTED_TEXT = RGBColor(70, 80, 95)    # Secondary Bullet Text

    def add_header(slide, category, title):
        # Category Tracker
        k_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_k = k_box.text_frame
        tf_k.word_wrap = True
        tf_k.margin_left = tf_k.margin_right = tf_k.margin_top = tf_k.margin_bottom = 0
        p_k = tf_k.paragraphs[0]
        p_k.text = category.upper()
        p_k.font.size = Pt(11)
        p_k.font.bold = True
        p_k.font.color.rgb = SKY
        p_k.font.name = "Calibri"

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
        p_t.font.name = "Calibri"

        # Dividing Line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.color.rgb = CARD_BORDER

    def add_simple_card(slide, left, top, width, height, card_title, bullets):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.3)
        tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.28)
        tf.margin_bottom = Inches(0.2)

        # Card Title
        p0 = tf.paragraphs[0]
        p0.text = card_title
        p0.font.size = Pt(16)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.font.name = "Calibri"
        p0.space_after = Pt(12)

        for b in bullets:
            p = tf.add_paragraph()
            p.space_after = Pt(10)
            if ":" in b:
                lead, rest = b.split(":", 1)
                r_lead = p.add_run()
                r_lead.text = f"• {lead}:"
                r_lead.font.bold = True
                r_lead.font.size = Pt(13)
                r_lead.font.color.rgb = DARK_TEXT
                r_lead.font.name = "Calibri"

                r_rest = p.add_run()
                r_rest.text = rest
                r_rest.font.bold = False
                r_rest.font.size = Pt(13)
                r_rest.font.color.rgb = MUTED_TEXT
                r_rest.font.name = "Calibri"
            else:
                r = p.add_run()
                r.text = f"• {b}"
                r.font.size = Pt(13)
                r.font.color.rgb = DARK_TEXT
                r.font.name = "Calibri"

    # ==========================================
    # SLIDE 1: Title Slide (Simple & Clean)
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    hero = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.1), Inches(11.333), Inches(5.3))
    hero.fill.solid()
    hero.fill.fore_color.rgb = CARD_BG
    hero.line.color.rgb = STEEL
    hero.line.width = Pt(2.0)

    tf1 = hero.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.8)
    tf1.margin_right = Inches(0.8)
    tf1.margin_top = Inches(0.7)

    p1 = tf1.paragraphs[0]
    p1.text = "RESEARCH PROGRESS PRESENTATION".upper()
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = SKY
    p1.font.name = "Calibri"
    p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "Urban Feature Extraction from Satellite Imagery"
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Calibri"
    p2.space_after = Pt(12)

    p3 = tf1.add_paragraph()
    p3.text = "Monochromatic Cartosat (2.5m) to High-Resolution Multispectral (0.60m) Modeling"
    p3.font.size = Pt(17)
    p3.font.color.rgb = STEEL
    p3.font.name = "Calibri"
    p3.space_after = Pt(36)

    p4 = tf1.add_paragraph()
    p4.text = "• Student: Prabhakar Yadav (23B0653)   |   Dept. of Civil Engineering, IIT Bombay\n• Supervisor: Prof. Pennan Chinnasamy   |   Dept. of Civil Engineering, IIT Bombay\n• Focus: Unsupervised Pseudo-Labeling, Dual Architectures, and Topology-Aware Extraction"
    p4.font.size = Pt(13)
    p4.font.color.rgb = MUTED_TEXT
    p4.font.name = "Calibri"

    # ==========================================
    # SLIDE 2: Problem Statement & Motivation
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Project Overview", "Problem Statement & The Need for Pseudo-Labeling")

    add_simple_card(s2, 0.8, 1.6, 5.7, 5.2, "What We Want to Solve", [
        "Core Goal: Automatically identify urban features (water, buildings, roads, vegetation, bare soil) from satellite imagery.",
        "Manual Labeling Problem: Hand-labeling satellite scenes pixel by pixel is extremely expensive and rarely available for Indian cities.",
        "Our Approach: Build an automated unsupervised pseudo-labeling pipeline to train deep learning models without manual supervision.",
        "End Result: An interactive, deployable platform that can process large satellite GeoTIFF files and export GIS vectors."
    ])

    add_simple_card(s2, 6.8, 1.6, 5.7, 5.2, "Challenges in Monochromatic Imagery", [
        "Single Band (Grayscale): Images contain only intensity values; no color and no Near-Infrared (NIR) band.",
        "Spectral Confusion: Concrete roofs, dry soil, and port pavement look almost identical in black-and-white.",
        "Shadow & Water Overlap: Building shadows and dark water bodies have similar low intensity values.",
        "Solution Required: We must use spatial texture and edge features, not just raw pixel intensity."
    ])

    # ==========================================
    # SLIDE 3: Monochromatic Dataset & Pre-Processing
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset 1", "Panchromatic Cartosat Imagery (2.5m & 1.0m)")

    add_simple_card(s3, 0.8, 1.6, 5.7, 5.2, "Dataset Characteristics", [
        "Primary Satellite Scene: JNPA Port Region GeoTIFF (JNPA_2_5.tif) at 2.5m resolution.",
        "Secondary Scene: Cartosat-1 panchromatic scene (CARTOSAT_1M_PAN.tif) at 1.0m resolution.",
        "Image Type: 1-Band grayscale imagery with high dynamic range.",
        "Urban Complexity: Contains coastal waters, port container yards, industrial facilities, bare soil, and mangroves."
    ])

    add_simple_card(s3, 6.8, 1.6, 5.7, 5.2, "Tiling & Data Cleaning Pipeline", [
        "Patch Generation: Cut the large satellite mosaic into manageable 256 x 256 pixel tiles.",
        "Total Grid: Extracted 1,015 candidate patches across the region.",
        "Information Filtering: Removed blank or uniform patches using variance thresholding (Std Dev > 0.045).",
        "Quality Patches Kept: Retained 735 high-information patches (>85% valid data) for model training."
    ])

    # ==========================================
    # SLIDE 4: Pseudo-Labeling Strategy
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Ground Truth Creation", "How Pseudo-Labels Were Generated (Clustering & Texture)")

    add_simple_card(s4, 0.8, 1.6, 5.7, 5.2, "1. Feature Extraction per Patch", [
        "Texture (GLCM): Contrast, homogeneity, and energy to capture surface roughness.",
        "Edge Density: Canny edge detector (thresholds 60/160) to measure structural boundaries.",
        "Intensity Distribution: Mean brightness and standard deviation per patch.",
        "PCA Compression: Reduced all features into 25 Principal Components for fast, clean clustering."
    ])

    add_simple_card(s4, 6.8, 1.6, 5.7, 5.2, "2. Clustering & Class Mapping", [
        "Unsupervised Grouping: K-Means and GMM grouped patches with similar textures together.",
        "Domain Interpretation: Clusters were verified and mapped into 5 real-world classes:",
        "Class 0 (Water): Deep water bodies and creeks (with dark-water heuristic override).",
        "Class 1 (Industrial / Port): High-texture container yards and port berths.",
        "Class 2 (Bare Soil / Land): Low-texture, cleared open land.",
        "Class 3 (Vegetation): Natural mangrove cover and green zones.",
        "Class 4 (Urban Built-up): Dense residential and commercial buildings."
    ])

    # ==========================================
    # SLIDE 5: Visualizing Pseudo-Labels (Image Slide)
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Pseudo-Labeling Results", "Visual Demonstration of Generated Pseudo-Labels")

    img_p5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dense_label_preview_panel.png")
    if os.path.exists(img_p5):
        s5.shapes.add_picture(img_p5, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_simple_card(s5, 8.3, 1.6, 4.2, 5.2, "Visual Insights", [
        "Accurate Grouping: Grayscale satellite patches (left) match clean semantic class masks (right).",
        "Clean Water Separation: Dark water bodies are identified with near-zero false alarms.",
        "Industrial Features: Container yards and port infrastructure are cleanly isolated by edge density.",
        "No Manual Work: Provided hundreds of reliable training masks automatically."
    ])

    # ==========================================
    # SLIDE 6: Architecture 1: U-Net Semantic Segmentation
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Architecture 1", "Deep Learning: U-Net Semantic Segmentation")

    add_simple_card(s6, 0.8, 1.6, 5.7, 5.2, "Model Structure & Training", [
        "U-Net Architecture: Encoder-decoder convolutional network with skip connections for spatial precision.",
        "Loss Function: Joint Dice Loss + Cross-Entropy Loss to handle class imbalance.",
        "Data Augmentation: Random flips, 90-degree rotations, contrast jitter, and noise.",
        "Prediction Type: Pixel-wise segmentation mask across all 5 classes."
    ])

    add_simple_card(s6, 6.8, 1.6, 5.7, 5.2, "Initial Result & Dense V128 Upgrade", [
        "Initial 256x256 V1 Result: 48.96% pixel accuracy, 33.50% Mean IoU.",
        "Why V1 Struggled: The U-Net was trying to predict sharp pixel boundaries from patch-level pseudo-labels (task-model mismatch).",
        "The V128 Dense Upgrade: Upgraded to dense 128x128 soft probability pseudo-masks.",
        "Major Improvement: V128 reached 91.07% Pixel Accuracy and 78.73% Mean IoU (+42.8% boost)."
    ])

    # ==========================================
    # SLIDE 7: Architecture 2: Patch Classification (Random Forest & ResNet)
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Architecture 2", "Patch Classification: Random Forest & ResNet-18")

    add_simple_card(s7, 0.8, 1.6, 5.7, 5.2, "Random Forest Patch Classifier", [
        "Model Setup: Ensemble of 1,000 decision trees trained directly on the 25 PCA features + Canny edge stats.",
        "Why It Worked So Well: The model's prediction unit (patch) perfectly matched the pseudo-label unit.",
        "JNPA 2.5m Performance: 87.76% Validation Accuracy, 85.56% Macro F1 Score.",
        "Cartosat 1m Performance: 89.78% Validation Accuracy, 74.64% Macro F1 across 4 refined classes."
    ])

    add_simple_card(s7, 6.8, 1.6, 5.7, 5.2, "ResNet-18 Deep Feature Classifier", [
        "Deep Transfer Learning: Pretrained ResNet-18 backbone fine-tuned for Cartosat patch classification.",
        "Learned Features: Replaced manual GLCM texture with deep convolutional feature representations.",
        "Outstanding Accuracy: Achieved 94.74% Overall Accuracy on Cartosat test patches.",
        "Speed: Fast, robust inference ideal for processing entire satellite scenes in seconds."
    ])

    # ==========================================
    # SLIDE 8: Visual Comparison & Training Results (Image Slide)
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Model Comparisons", "Visual Results: Predictions vs Clustering Baseline")

    img_p8 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "comparison_summary_board.png")
    if os.path.exists(img_p8):
        s8.shapes.add_picture(img_p8, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_simple_card(s8, 8.3, 1.6, 4.2, 5.2, "Visual Insights", [
        "Noise Removal: Trained models smoothed out isolated noisy pixels from raw clustering.",
        "Coastline Continuity: Dense U-Net V128 captured smooth spatial shorelines.",
        "Complementary Strengths: Random Forest gave quick, reliable land-use classification; U-Net gave detailed spatial maps.",
        "Validation: Proved that unsupervised pseudo-labels can effectively train deep models."
    ])

    # ==========================================
    # SLIDE 9: Test Evaluation on Monochromatic Data
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Evaluation", "Quantitative Test Results on Held-Out Satellite Data")

    add_simple_card(s9, 0.8, 1.6, 5.7, 5.2, "Class-Wise Test Accuracies", [
        "Water Bodies: 97.30% (distinct dark absorption contrast).",
        "Dense Urban Built-up: 94.74% (sharp structural edge density).",
        "Industrial / Port: 80.00% (separated from general urban).",
        "Vegetation / Mangroves: 100.00% (accurate patch-level classification).",
        "Bare Land / Soil: 53.33% (most challenging due to gray-level similarity with dry ground)."
    ])

    img_p9 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Test_Model\results\accuracy_cartosat_confusion.png"
    if os.path.exists(img_p9):
        s9.shapes.add_picture(img_p9, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # ==========================================
    # SLIDE 10: GeoVision Web Platform Deployment
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Platform Deployment", "Interactive GeoVision Web Platform for End Users")

    add_simple_card(s10, 0.8, 1.6, 5.7, 5.2, "Platform Capabilities", [
        "User Interface: Built with Flask / Streamlit for simple drag-and-drop satellite image upload.",
        "Auto Model Selection: Inspects image band count and resolution, automatically routing to the best model.",
        "Tiled Streaming: Uses sliding-window inference with Hann-window blending to eliminate edge seams.",
        "Handles Large Images: Smoothly processes GeoTIFF files up to 500 MB without memory crashes."
    ])

    add_simple_card(s10, 6.8, 1.6, 5.7, 5.2, "GIS Export Features", [
        "GeoTIFF Export: Saves full-resolution, georeferenced classification GeoTIFFs.",
        "Vector Shapefiles: Automatically converts raster prediction masks into ESRI Shapefiles and GeoJSON.",
        "Preserves Coordinates: Retains original map projections (EPSG:4326 / EPSG:32643) for QGIS and ArcGIS.",
        "Area Metrics: Displays real-time square kilometer coverage for each urban class."
    ])

    # ==========================================
    # SLIDE 11: Limitations of Monochromatic 2.5m Data (Transition)
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Transition to Phase 2", "Limitations of 2.5m Monochromatic Data")

    add_simple_card(s11, 0.8, 1.6, 3.7, 5.2, "1. Resolution Limit (2.5m)", [
        "Narrow Features Lost: Roads and alleys are only 2–3 pixels wide at 2.5m resolution.",
        "Corridor Disconnection: Roads merge into adjacent buildings or bare ground.",
        "Disconnected Blobs: Produces irregular 'jelly blobs' instead of continuous road networks."
    ])

    add_simple_card(s11, 4.8, 1.6, 3.7, 5.2, "2. Spectral Blindness", [
        "No Color / No NIR: Impossible to compute physical vegetation indices like NDVI.",
        "Shadow Confusion: Asphalt roads and deep tree/building shadows look identical.",
        "Vegetation Ambiguity: Dark soil cannot be reliably separated from dense green canopy."
    ])

    add_simple_card(s11, 8.8, 1.6, 3.7, 5.2, "3. Solution: New Dataset", [
        "Move to Cartosat-2E: Jump to 0.60m resolution (>4x sharper detail).",
        "Multispectral 4 Bands: Adds Red, Green, Blue, and Near-Infrared (NIR).",
        "Topology-Aware Loss: Use clDice loss to enforce road continuity."
    ])

    # ==========================================
    # SLIDE 12: New Dataset: Cartosat-2E 0.60m Multispectral
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "New Dataset (Current Work)", "Cartosat-2E 0.60m Multispectral Imagery & NDVI Physics")

    add_simple_card(s12, 0.8, 1.6, 5.2, 5.2, "What's New in This Dataset", [
        "Sub-Meter Resolution: 0.60m GSD clearly resolves individual road lanes, building footprints, and trees.",
        "4 Spectral Channels: Red, Green, Blue, and Near-Infrared (NIR) bands.",
        "Physical NDVI Ground Truth: NDVI = (NIR - Red) / (NIR + Red) cleanly isolates 100% of vegetation with zero manual work.",
        "OSM Corridor Fusion: Merged OpenStreetMap road centerlines with metric widths to create automated corridor masks."
    ])

    img_p12 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\test_2673727101_r15200_c11200_comparison.png"
    if os.path.exists(img_p12):
        s12.shapes.add_picture(img_p12, Inches(6.3), Inches(1.6), width=Inches(6.2))

    # ==========================================
    # SLIDE 13: New Model Training & Results (Continuous Roads)
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "New Training & Results", "Topology-Aware Road Extraction with clDice Loss")

    add_simple_card(s13, 0.8, 1.6, 5.2, 5.2, "Key Model Improvements", [
        "Topology Loss (clDice): Centerline Dice loss strictly penalizes gaps and breaks under tree shadows.",
        "Multi-Task Vectorizer: Predicts road centerlines, edges, and constant widths simultaneously.",
        "Affine Alignment: Corrects 2-4m GPS orthorectification shifts before rasterizing labels.",
        "Evaluation on 173 Unseen Patches: Achieved 86.99% Overall Accuracy, 60.95% Road Recall, and 83.57% Active Corridor mIoU."
    ])

    img_p13 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\constant_width_continuous_roads_preview.png"
    if os.path.exists(img_p13):
        s13.shapes.add_picture(img_p13, Inches(6.3), Inches(1.6), width=Inches(6.2))

    # ==========================================
    # SLIDE 14: Conclusion & Next Steps
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion & Roadmap", "Summary of Progress & Immediate Next Steps")

    add_simple_card(s14, 0.8, 1.6, 5.7, 5.2, "What We Have Achieved", [
        "Panchromatic Foundation: Built a complete unsupervised pseudo-labeling pipeline for 2.5m & 1.0m Cartosat data.",
        "Dual Architectures: Compared U-Net segmentation vs. Random Forest / ResNet patch classification (up to 94.7% accuracy).",
        "Operational Platform: Deployed models into the interactive GeoVision web platform with GIS export.",
        "New Breakthrough: Upgraded to 0.60m multispectral data, successfully extracting continuous road networks with clDice."
    ])

    add_simple_card(s14, 6.8, 1.6, 5.7, 5.2, "Next Steps for Thesis Completion", [
        "Cross-City Testing: Validating model transferability on unseen Chennai multispectral scenes.",
        "Direct CAD/GIS Graph Export: Converting road centerlines into topological vector graphs (Shapefile / GeoJSON).",
        "Routability Metric (APLS): Evaluating shortest-path network connectivity.",
        "Thesis Writing: Compiling all results and comparative analyses into the final BTP report."
    ])

    # Save to the single unified presentation file
    unified_path = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Urban_Feature_Extraction_Progress_Presentation.pptx"
    prs.save(unified_path)
    print(f"Unified presentation saved successfully to: {unified_path}")

    # Remove older duplicate/confusing presentation files if they exist
    old_files = [
        r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Cartosat2E_Urban_Feature_Extraction_Progress_Presentation.pptx",
        r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Cartosat_Urban_Feature_Extraction_Progress.pptx"
    ]
    for old_f in old_files:
        try:
            if os.path.exists(old_f):
                os.remove(old_f)
                print(f"Removed older file: {old_f}")
        except Exception as e:
            print(f"Could not remove {old_f}: {e}")

if __name__ == "__main__":
    create_unified_deck()
