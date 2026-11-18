"""
Script to build the improved, visually stunning, easy-to-explain presentation:
'Urban_Feature_Extraction_Progress_Presentation.pptx'

Key Design Upgrades:
1. Fixed Corporate Blue Background Template (inspired by user's image):
   - Diagonal blue geometric banners & corner trapezoids/triangles
   - Consistent modern top-header bar and branding accents
   - Clean white canvas with spacious, rounded card containers
2. Bigger, bolder typography:
   - Slide Titles: 26pt Bold
   - Section Headers: 18pt Bold
   - Body & Bullets: 14-15pt (spacious, high legibility)
   - Large Stat Callouts: 28-36pt Bold
3. Simplified, visual content:
   - No dense paragraphs
   - Embedded Charts: Dataset 1 accuracy bar chart & Dataset 1 vs 2 comparison chart
   - Embedded Images: Pseudo-labels, model comparisons, continuous road predictions
4. Narrative Alignment:
   - Framing around TWO distinct datasets (Dataset 1: 2.5m Monochromatic; Dataset 2: 0.60m Multispectral)
   - 2 slides on Dataset 2 (adaptation & results)
   - Dedicated cross-dataset comparison slide at the end
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Tokens
    PRIMARY_BLUE = RGBColor(21, 101, 192)   # Vibrant Royal Blue (from template)
    DARK_NAVY    = RGBColor(10, 37, 64)     # Deep Executive Navy
    LIGHT_BLUE   = RGBColor(227, 242, 253)  # Soft Sky Tint for accents
    CARD_BG      = RGBColor(248, 250, 252)  # Clean Card Fill
    CARD_BORDER  = RGBColor(226, 232, 240)  # Subtle Card Border
    DARK_TEXT    = RGBColor(15, 23, 42)     # High-contrast Slate Text
    MUTED_TEXT   = RGBColor(71, 85, 105)    # Slate Muted Text
    WHITE        = RGBColor(255, 255, 255)
    CYAN_ACCENT  = RGBColor(2, 132, 199)

    # -------------------------------------------------------------
    # TEMPLATE DECORATOR: Adds the fixed background geometry to every slide
    # -------------------------------------------------------------
    def apply_fixed_template(slide, is_title=False):
        if is_title:
            # Title slide large geometric background shapes
            # Large angled polygon banner on top-left / top
            top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.25))
            top_bar.fill.solid()
            top_bar.fill.fore_color.rgb = PRIMARY_BLUE
            top_bar.line.fill.background()

            # Right-side diagonal accent triangle
            tri = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(10.5), Inches(0), Inches(2.833), Inches(7.5))
            tri.fill.solid()
            tri.fill.fore_color.rgb = PRIMARY_BLUE
            tri.line.fill.background()

            # Accent small triangle
            tri2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(9.8), Inches(4.5), Inches(1.5), Inches(3.0))
            tri2.fill.solid()
            tri2.fill.fore_color.rgb = LIGHT_BLUE
            tri2.line.fill.background()
            return

        # Content slides background template:
        # 1. Top left blue accent block
        top_accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.4), Inches(1.4))
        top_accent.fill.solid()
        top_accent.fill.fore_color.rgb = PRIMARY_BLUE
        top_accent.line.fill.background()

        # 2. Bottom right diagonal accent triangle
        corner_tri = slide.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE, Inches(12.133), Inches(6.3), Inches(1.2), Inches(1.2))
        corner_tri.fill.solid()
        corner_tri.fill.fore_color.rgb = PRIMARY_BLUE
        corner_tri.line.fill.background()

        # 3. Bottom left subtle horizontal accent bar
        bottom_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.35), Inches(13.333), Inches(0.15))
        bottom_bar.fill.solid()
        bottom_bar.fill.fore_color.rgb = LIGHT_BLUE
        bottom_bar.line.fill.background()

        # 4. Thin subtle rule under header
        rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        rule.fill.solid()
        rule.fill.fore_color.rgb = CARD_BORDER
        rule.line.fill.background()

    def add_slide_header(slide, tracker, title):
        apply_fixed_template(slide, is_title=False)

        # Tracker / Category tag
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.7), Inches(0.35))
        tf_tr = t_box.text_frame
        tf_tr.word_wrap = True
        tf_tr.margin_left = tf_tr.margin_right = tf_tr.margin_top = tf_tr.margin_bottom = 0
        p_tr = tf_tr.paragraphs[0]
        p_tr.text = tracker.upper()
        p_tr.font.size = Pt(11)
        p_tr.font.bold = True
        p_tr.font.color.rgb = PRIMARY_BLUE
        p_tr.font.name = "Calibri"

        # Main Title (BIGGER FONT: 24-26pt bold)
        h_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.6))
        tf_h = h_box.text_frame
        tf_h.word_wrap = True
        tf_h.margin_left = tf_h.margin_right = tf_h.margin_top = tf_h.margin_bottom = 0
        p_h = tf_h.paragraphs[0]
        p_h.text = title
        p_h.font.size = Pt(25)
        p_h.font.bold = True
        p_h.font.color.rgb = DARK_NAVY
        p_h.font.name = "Calibri"

    def add_card(slide, left, top, width, height, card_title, bullets, badge_text=""):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.4)

        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.32)
        tf.margin_right = Inches(0.32)
        tf.margin_top = Inches(0.3)
        tf.margin_bottom = Inches(0.2)

        # Title
        p0 = tf.paragraphs[0]
        p0.text = card_title
        p0.font.size = Pt(17)
        p0.font.bold = True
        p0.font.color.rgb = DARK_NAVY
        p0.font.name = "Calibri"
        p0.space_after = Pt(14)

        for b in bullets:
            p = tf.add_paragraph()
            p.space_after = Pt(11)
            if ":" in b:
                lead, rest = b.split(":", 1)
                r_lead = p.add_run()
                r_lead.text = f"•  {lead}:"
                r_lead.font.bold = True
                r_lead.font.size = Pt(14)
                r_lead.font.color.rgb = DARK_TEXT
                r_lead.font.name = "Calibri"

                r_rest = p.add_run()
                r_rest.text = rest
                r_rest.font.bold = False
                r_rest.font.size = Pt(14)
                r_rest.font.color.rgb = MUTED_TEXT
                r_rest.font.name = "Calibri"
            else:
                r = p.add_run()
                r.text = f"•  {b}"
                r.font.size = Pt(14)
                r.font.color.rgb = DARK_TEXT
                r.font.name = "Calibri"

    def add_stat_box(slide, left, top, width, height, stat_number, label_text, sub_text=""):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT_BLUE
        box.line.color.rgb = PRIMARY_BLUE
        box.line.width = Pt(1.5)

        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.18)

        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        p1.text = stat_number
        p1.font.size = Pt(30)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY_BLUE
        p1.font.name = "Calibri"

        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.text = label_text
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = DARK_NAVY
        p2.font.name = "Calibri"

        if sub_text:
            p3 = tf.add_paragraph()
            p3.alignment = PP_ALIGN.CENTER
            p3.text = sub_text
            p3.font.size = Pt(11)
            p3.font.color.rgb = MUTED_TEXT
            p3.font.name = "Calibri"

    # =============================================================
    # SLIDE 1: Title Slide (Preserving User's exact updated text)
    # =============================================================
    s1 = prs.slides.add_slide(blank_layout)
    apply_fixed_template(s1, is_title=True)

    # Main Title Card
    t_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(1.1), Inches(9.2), Inches(5.3))
    t_card.fill.solid()
    t_card.fill.fore_color.rgb = WHITE
    t_card.line.color.rgb = PRIMARY_BLUE
    t_card.line.width = Pt(2.0)

    tf_title = t_card.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = Inches(0.6)
    tf_title.margin_right = Inches(0.6)
    tf_title.margin_top = Inches(0.6)

    p_sub = tf_title.paragraphs[0]
    p_sub.text = "RESEARCH PROGRESS PRESENTATION"
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = PRIMARY_BLUE
    p_sub.font.name = "Calibri"
    p_sub.space_after = Pt(16)

    p_main = tf_title.add_paragraph()
    p_main.text = "Urban Feature Extraction from Satellite Imagery"
    p_main.font.size = Pt(28)
    p_main.font.bold = True
    p_main.font.color.rgb = DARK_NAVY
    p_main.font.name = "Calibri"
    p_main.space_after = Pt(12)

    p_desc = tf_title.add_paragraph()
    p_desc.text = "Monochromatic Cartosat (2.5m) to High-Resolution Multispectral (0.60m)"
    p_desc.font.size = Pt(18)
    p_desc.font.bold = True
    p_desc.font.color.rgb = CYAN_ACCENT
    p_desc.font.name = "Calibri"
    p_desc.space_after = Pt(34)

    p_meta = tf_title.add_paragraph()
    p_meta.text = "• Student: Prabhakar Yadav (Roll No: 23B0653)\n• Department: Department of Civil Engineering, IIT Bombay\n• Supervisor: Prof. Pennan Chinnasamy\n• Scope: Cross-Dataset Urban Feature Modeling & Evaluation"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = DARK_TEXT
    p_meta.font.name = "Calibri"

    # =============================================================
    # SLIDE 2: Project Framework: Two Datasets, Two Distinct Setups
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Research Strategy", "Project Framework: Training & Testing Across Two Distinct Datasets")

    add_card(s2, 0.8, 1.6, 5.7, 5.3, "Dataset 1: Monochromatic 2.5m / 1m", [
        "Core Mission: Extract urban features from single-band panchromatic satellite data with zero manual ground truth.",
        "Engineering Scope: Did everything possible on this data — patch filtering, GLCM texture, PCA feature compression, unsupervised clustering.",
        "Model Architectures: Trained & evaluated U-Net semantic segmentation and Random Forest / ResNet patch classifiers.",
        "Platform Output: Deployed into a full interactive web GIS platform (GeoVision)."
    ])

    add_card(s2, 6.8, 1.6, 5.7, 5.3, "Dataset 2: Multispectral 0.60m (Current)", [
        "Core Mission: Advance to sub-meter 4-band imagery to resolve narrow linear urban structures like roads.",
        "Engineering Adaptation: Adapted the previous model with minor changes (4-band input stem, physical NDVI, clDice loss).",
        "Recent Testing: Trained and tested on 173 unseen patches, achieving continuous, constant-width road corridors.",
        "Final Goal: Direct side-by-side comparison of results from both data types."
    ])

    # =============================================================
    # SLIDE 3: Dataset 1: 2.5m Monochromatic Cartosat (JNPA)
    # =============================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Dataset 1 Details", "Panchromatic Cartosat Imagery & Patch Filtering Pipeline")

    add_card(s3, 0.8, 1.6, 5.7, 4.0, "Sensor & Image Characteristics", [
        "Primary Satellite Scene: JNPA Port Region GeoTIFF (JNPA_2_5.tif) at 2.5m resolution.",
        "Secondary Scene: Cartosat-1 panchromatic scene (CARTOSAT_1M_PAN.tif) at 1.0m resolution.",
        "Image Format: 1-Band grayscale imagery with high radiometric dynamic range.",
        "Urban Diversity: Industrial berths, container terminals, mangroves, creeks, and residential clusters."
    ])

    add_card(s3, 6.8, 1.6, 5.7, 4.0, "Patch Extraction & Quality Filtering", [
        "Regular Tiling: Partitioned the massive mosaic into non-overlapping 256 x 256 pixel tiles.",
        "Candidate Grid: 1,015 candidate tiles extracted across the geographic extent.",
        "Variance Filtering: Discarded blank and uniform water patches using standard deviation threshold (>0.045).",
        "Retained Dataset: 735 high-information patches (>85% valid pixels) used for training."
    ])

    # 3 Stat badges along the bottom
    add_stat_box(s3, 0.8, 5.8, 3.6, 1.2, "2.5m / 1.0m", "Spatial Resolution", "Panchromatic Grayscale")
    add_stat_box(s3, 4.85, 5.8, 3.6, 1.2, "1,015 Grid Tiles", "Extracted Extent", "256 x 256 Patch Size")
    add_stat_box(s3, 8.9, 5.8, 3.6, 1.2, "735 Patches", "Clean Dataset", "Filtered via Texture Variance")

    # =============================================================
    # SLIDE 4: Dataset 1: Automated Pseudo-Labeling Strategy
    # =============================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Ground Truth Generation", "Automated Pseudo-Labeling via Multi-Feature Clustering")

    add_card(s4, 0.8, 1.6, 5.7, 5.3, "1. Multi-Feature Extraction", [
        "Texture Analysis (GLCM): Contrast, homogeneity, and energy to quantify surface roughness.",
        "Edge Density: Canny edge detector (thresholds 60/160) to isolate man-made built structures.",
        "Intensity Metrics: Mean reflectance and standard deviation per patch.",
        "Dimensionality Reduction: Compressed all features into 25 Principal Components (PCA)."
    ])

    add_card(s4, 6.8, 1.6, 5.7, 5.3, "2. Unsupervised Cluster Mapping", [
        "Clustering Engine: K-Means and GMM grouped patches with shared structural signatures.",
        "Expert Verification: Mapped mathematical clusters to 5 distinct real-world classes:",
        "Class 0 (Water): Deep water bodies (with dark-water heuristic override).",
        "Class 1 (Industrial / Port): High-texture container terminals and berths.",
        "Class 2 (Bare Soil / Land): Cleared ground and unpaved terrain.",
        "Class 3 (Vegetation): Natural mangrove forest and tree canopies.",
        "Class 4 (Urban Built-up): Residential and commercial settlements."
    ])

    # =============================================================
    # SLIDE 5: Visualizing Pseudo-Labels on Dataset 1 (Embedded Image)
    # =============================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Pseudo-Labeling Results", "Visual Demonstration: Generated Pseudo-Labels from Panchromatic Data")

    img_p5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dense_label_preview_panel.png")
    if os.path.exists(img_p5):
        s5.shapes.add_picture(img_p5, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s5, 8.3, 1.6, 4.2, 5.3, "Visual Quality Assessment", [
        "Clean Semantic Alignment: Grayscale satellite patches (left) map cleanly to distinct class masks (right).",
        "Water-Land Boundary: Crisp coastline separation with zero false alarms across water bodies.",
        "Industrial Texture: Container yards and port infrastructure clearly grouped by edge frequency.",
        "No Hand-Labeling: Generated hundreds of supervised masks automatically."
    ])

    # =============================================================
    # SLIDE 6: Architecture 1 on Dataset 1: U-Net Semantic Segmentation
    # =============================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Architecture 1", "Deep Learning: U-Net Semantic Segmentation on Dataset 1")

    add_card(s6, 0.8, 1.6, 5.7, 4.0, "Model Architecture & Loss", [
        "Network Design: U-Net encoder-decoder with symmetric contracting and expanding paths.",
        "Skip Connections: Forward spatial details to recover high-resolution boundaries.",
        "Loss Function: Joint Dice Loss + Cross-Entropy Loss to handle class imbalance.",
        "Data Augmentation: Random horizontal/vertical flips, 90-degree rotations, contrast jitter."
    ])

    add_card(s6, 6.8, 1.6, 5.7, 4.0, "V1 Baseline vs V128 Dense Upgrade", [
        "Initial 256x256 V1 Result: 48.96% pixel accuracy, 33.50% Mean IoU.",
        "The Cause: U-Net was trained for pixel boundaries on patch-level uniform pseudo-labels.",
        "The V128 Dense Upgrade: Retrained on dense 128x128 soft probability pseudo-masks.",
        "Final Performance: Reached 91.07% Pixel Accuracy and 78.73% Mean IoU (+42.8% gain)."
    ])

    add_stat_box(s6, 0.8, 5.8, 3.6, 1.2, "U-Net", "Network Architecture", "Encoder-Decoder + Skips")
    add_stat_box(s6, 4.85, 5.8, 3.6, 1.2, "91.07%", "Validation Pixel Accuracy", "Dense V128 Soft Masks")
    add_stat_box(s6, 8.9, 5.8, 3.6, 1.2, "78.73%", "Mean IoU Score", "+42.8% Improvement")

    # =============================================================
    # SLIDE 7: Architecture 2 on Dataset 1: Patch Classifiers (RF & ResNet)
    # =============================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Architecture 2", "Patch Classifiers: Random Forest & Deep ResNet-18 on Dataset 1")

    add_card(s7, 0.8, 1.6, 5.7, 4.0, "Random Forest Patch Classifier", [
        "Model Setup: Ensemble of 1,000 decision trees trained on 25 PCA features + Canny stats.",
        "Task-Model Alignment: The prediction unit (patch) matched the pseudo-label resolution perfectly.",
        "JNPA 2.5m Result: 87.76% Validation Accuracy, 85.56% Macro F1 Score.",
        "Cartosat 1m Result: 89.78% Validation Accuracy, 74.64% Macro F1 across 4 classes."
    ])

    add_card(s7, 6.8, 1.6, 5.7, 4.0, "ResNet-18 Deep Feature Classifier", [
        "Transfer Learning: Pretrained ResNet-18 backbone fine-tuned for Cartosat patch classification.",
        "Feature Representation: Deep spatial embeddings replaced manual texture engineering.",
        "Top Performance: Achieved 94.74% Overall Accuracy on Cartosat test patches.",
        "Inference Speed: Millisecond inference per tile, ideal for full-scene streaming."
    ])

    add_stat_box(s7, 0.8, 5.8, 3.6, 1.2, "87.76%", "Random Forest Accuracy", "JNPA 2.5m (85.56% F1)")
    add_stat_box(s7, 4.85, 5.8, 3.6, 1.2, "89.78%", "Cartosat 1m Accuracy", "4 Refined Classes (74.6% F1)")
    add_stat_box(s7, 8.9, 5.8, 3.6, 1.2, "94.74%", "ResNet-18 Accuracy", "Deep Spatial Embeddings")

    # =============================================================
    # SLIDE 8: Visual Predictions & Model Comparison on Dataset 1 (Embedded Image)
    # =============================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Dataset 1 Results", "Model Predictions vs Clustering Ground Truth on Panchromatic Data")

    img_p8 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "comparison_summary_board.png")
    if os.path.exists(img_p8):
        s8.shapes.add_picture(img_p8, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s8, 8.3, 1.6, 4.2, 5.3, "Visual Insights", [
        "Noise Filtering: Trained models eliminated speckle noise found in raw unsupervised clusters.",
        "Coastline Continuity: U-Net V128 achieved continuous spatial corridors along water bodies.",
        "Complementary Roles: Random Forest gives quick land-use classification; U-Net gives spatial boundaries.",
        "Core Finding: Validated that unsupervised pseudo-labeling is a viable supervision signal."
    ])

    # =============================================================
    # SLIDE 9: Quantitative Test Evaluation on Dataset 1 (Embedded Chart)
    # =============================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Dataset 1 Evaluation", "Quantitative Test Results Across Urban & Environmental Classes")

    # Embed Chart 1 (Class Accuracy Bar Chart)
    img_chart1 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\dataset1_accuracy_chart.png"
    if os.path.exists(img_chart1):
        s9.shapes.add_picture(img_chart1, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s9, 7.9, 1.6, 4.6, 5.3, "Key Accuracy Highlights", [
        "Water Bodies (97.3%): Highest accuracy due to distinct absorption contrast in panchromatic data.",
        "Urban Built-up (94.7%): Sharp edge density reliably separated buildings from flat soil.",
        "Vegetation (100.0%): Cleanly identified on held-out patch validation tests.",
        "Industrial / Port (80.0%): Successfully distinguished container terminals from general urban.",
        "Bare Land / Soil (53.3%): Most challenging due to grayscale similarity with dry terrain."
    ])

    # =============================================================
    # SLIDE 10: Operational Platform: GeoVision Web App & GIS Export
    # =============================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Platform Deployment", "GeoVision Platform: Interactive Satellite Inference & GIS Vector Export")

    add_card(s10, 0.8, 1.6, 5.7, 4.0, "Core Platform Architecture", [
        "User Interface: Built with Flask / Streamlit for easy drag-and-drop satellite imagery upload.",
        "Auto-Model Selection: Automatically detects image bands and resolution to route to the best model.",
        "Large-Scale Tiling: Sliding-window inference with Hann blending eliminates border seams.",
        "Robust File Ingestion: Processes GeoTIFF, PNG, and JPEG files up to 500 MB without crashes."
    ])

    add_card(s10, 6.8, 1.6, 5.7, 4.0, "GIS Vector Export Features", [
        "GeoTIFF Export: Reconstructs full-resolution geo-referenced classification GeoTIFFs.",
        "Vector Shapefiles: Automated vector polygon extraction exported as ESRI Shapefiles & GeoJSON.",
        "CRS Preservation: Retains exact spatial reference systems (EPSG:4326 / EPSG:32643) for QGIS.",
        "Area Metrics: Displays real-time square kilometer coverage for each urban land-use class."
    ])

    add_stat_box(s10, 0.8, 5.8, 3.6, 1.2, "Auto Routing", "Band & Resolution", "Automatic Model Pick")
    add_stat_box(s10, 4.85, 5.8, 3.6, 1.2, "Hann Blending", "Sliding Window", "Zero Tile Seams")
    add_stat_box(s10, 8.9, 5.8, 3.6, 1.2, "GIS Export", "Shapefile & GeoJSON", "EPSG Georeferenced")

    # =============================================================
    # SLIDE 11: Dataset 2 (Current Work - 1/2): 0.60m Multispectral Cartosat-2E & Adaptation
    # =============================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Dataset 2 (Current Work - 1/2)", "Cartosat-2E 0.60m Multispectral Data & Model Adaptation")

    add_card(s11, 0.8, 1.6, 5.4, 5.3, "The New Dataset & Model Changes", [
        "Sub-Meter Resolution: 0.60m GSD resolves individual road lanes, trees, and small buildings.",
        "4 Spectral Channels: Red, Green, Blue, and Near-Infrared (NIR) bands.",
        "Physical NDVI Ground Truth: NDVI = (NIR - Red)/(NIR + Red) cleanly isolates 100% of vegetation.",
        "OSM Corridor Fusion: Merged OpenStreetMap centerlines with metric widths for road masks.",
        "Minor Model Adaptations: Modified previous U-Net stem from 1-band to 4-band input, and added topology-aware loss."
    ])

    img_p12 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\test_2673727101_r15200_c11200_comparison.png"
    if os.path.exists(img_p12):
        s11.shapes.add_picture(img_p12, Inches(6.5), Inches(1.6), width=Inches(6.0))

    # =============================================================
    # SLIDE 12: Dataset 2 (Current Work - 2/2): Training & Test Results
    # =============================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Dataset 2 (Current Work - 2/2)", "Training & Test Results: Topology-Aware Continuous Road Extraction")

    add_card(s12, 0.8, 1.6, 5.4, 5.3, "Training & Test Results on Dataset 2", [
        "Topology-Aware Loss (clDice): Centerline Dice loss strictly penalizes gaps and breaks under tree canopies.",
        "Continuous Constant Width: Eliminates wobbly 'jelly blobs', maintaining constant corridor widths.",
        "Affine Vector Alignment: Corrects 2–4m GPS orthorectification shifts before rasterizing labels.",
        "Benchmark on 173 Unseen Patches:",
        "  • 86.99% Overall Accuracy",
        "  • 60.95% Road Recall (strict pixel match)",
        "  • 83.57% Active Corridor mIoU"
    ])

    img_p13 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\constant_width_continuous_roads_preview.png"
    if os.path.exists(img_p13):
        s12.shapes.add_picture(img_p13, Inches(6.5), Inches(1.6), width=Inches(6.0))

    # =============================================================
    # SLIDE 13: Direct Comparison: Dataset 1 (2.5m) vs. Dataset 2 (0.60m)
    # =============================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "Cross-Dataset Synthesis", "Comparison Between Model Results Across the Two Datasets")

    # Embed Chart 2 (Comparison Bar Chart)
    img_chart2 = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\visual_results\dataset_comparison_chart.png"
    if os.path.exists(img_chart2):
        s13.shapes.add_picture(img_chart2, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s13, 7.9, 1.6, 4.6, 5.3, "Side-by-Side Comparison", [
        "Spatial Detail: 2.5m (coarse zones) vs 0.60m (individual road corridors and buildings).",
        "Spectral Bands: 1-Band grayscale vs 4-Band multispectral (RGB + NIR).",
        "Supervision: Unsupervised GLCM/PCA clustering vs Physical NDVI + OSM geometries.",
        "Road Extraction: Disconnected blobs (narrow paths lost) vs Continuous constant-width corridors.",
        "Key Conclusion: Both models succeed in their scope — Dataset 1 for broad land use, Dataset 2 for high-precision vector corridors."
    ])

    # =============================================================
    # SLIDE 14: Conclusion & Future Roadmap
    # =============================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_header(s14, "Conclusion & Roadmap", "Summary of Progress & Final Steps for Thesis Completion")

    add_card(s14, 0.8, 1.6, 5.7, 5.3, "Summary of Achievements", [
        "Dataset 1 Accomplished: Completed full pseudo-labeling, U-Net, Random Forest, and ResNet-18 pipeline on 2.5m panchromatic data.",
        "Functional Web Platform: Built and verified GeoVision for automated multi-model inference and GIS vector export.",
        "Dataset 2 Adapted & Tested: Adapted the pipeline for 0.60m multispectral data, solving road continuity with clDice loss (87.0% OA).",
        "Direct Comparison Validated: Demonstrated how sensor resolution and spectral channels dictate feature extraction strategy."
    ])

    add_card(s14, 6.8, 1.6, 5.7, 5.3, "Immediate Next Steps", [
        "Cross-City Validation: Test model transferability on unseen Chennai multispectral scenes without retraining.",
        "Direct CAD/GIS Graph Export: Convert predicted road centerlines into topological vector road networks (Shapefile / GeoJSON).",
        "Network Routability (APLS): Evaluate shortest-path network connectivity using Average Path Length Similarity.",
        "Final Report & Defense: Assemble all comparative findings and metrics into the BTP thesis."
    ])

    # Save to the single unified presentation file
    unified_path = r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main\Urban_Feature_Extraction_Progress_Presentation.pptx"
    prs.save(unified_path)
    print(f"Presentation saved successfully to: {unified_path}")

if __name__ == "__main__":
    build_presentation()
