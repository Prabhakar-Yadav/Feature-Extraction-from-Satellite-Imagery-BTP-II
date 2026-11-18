"""
scripts/report_builder/chapter2.py
----------------------------------
Chapter 2: Background and Literature Review
- 2.1 Remote Sensing Imagery: Spatial vs. Spectral Resolution Trade-Off
- 2.2 Spectral Reflectance Models and Biophysical Indices (NDVI, NDWI, ExG)
- 2.3 Deep Encoder-Decoder Networks for Semantic Segmentation
- 2.4 Vision Transformers and Attention Mechanisms in Earth Observation
- 2.5 Hierarchical and Multi-Level Land-Cover Categorization
- 2.6 Loss Formulations for Imbalanced, Boundary, and Topological Structures
- 2.7 Weak and Pseudo-Supervision from Open Geospatial Data
- 2.8 Evaluation Protocols Beyond Pixel Accuracy
"""

import docx
from docx.shared import Inches, Pt
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table, add_equation
)
from omml_equations import (
    add_display_equation, OMML_NDVI, OMML_NDWI, OMML_EXG,
    OMML_ATTENTION, OMML_LCE, OMML_LFOCAL
)

def add_chapter_2(doc):
    add_heading_1(doc, "Chapter 2: Background and Literature Review")
    
    # ----------------------------------------------------
    # 2.1 Remote Sensing Imagery: Spatial vs. Spectral Trade-Off
    # ----------------------------------------------------
    add_heading_2(doc, "2.1 Remote Sensing Imagery: Spatial vs. Spectral Resolution Trade-Off")
    add_paragraph(doc, 
        "Earth observation satellites operate under fundamental physical constraints imposed by orbital optics, sensor detector dimensions, and signal-to-noise ratios. A persistent dilemma in satellite remote sensing is the inverse relationship between spatial resolution (Ground Sampling Distance, GSD) and spectral resolution (the number and bandwidth of spectral channels). Panchromatic sensors, such as Cartosat-1 (2.5m GSD), integrate radiant energy across a single wide visible-to-near-infrared spectral band (0.50–0.85 µm). Because the detector collects all available photons into a single channel, panchromatic instruments can achieve high spatial resolution with small pixel sizes on the ground, making them ideal for resolving road widths, building outlines, and structural geometry.",
        bold_prefix="Optical Trade-Offs: "
    )
    add_paragraph(doc, 
        "However, panchromatic data lacks spectral discrimination. The digital number (DN) recorded by a panchromatic pixel represents integrated luminance, causing materials with distinct physical compositions but similar grayscale reflectance—such as dark asphalt roads, shaded building roofs, deep water, and dense tree shadows—to appear identical. Conversely, multispectral platforms like Sentinel-2 provide 13 discrete spectral bands, enabling unambiguous biophysical identification of chlorophyll absorption, water bodies, and minerals. Yet, Sentinel-2's visible and NIR bands are constrained to a 10m GSD, where narrow residential access roads (3–8m wide) and individual house roofs become mixed pixels (pixels containing mixtures of multiple land-cover types).",
        bold_prefix="Spectral Ambiguity vs. Mixed Pixels: "
    )
    add_paragraph(doc, 
        "Modern sub-meter multispectral platforms, such as Cartosat-2E (0.60m GSD with Red, Green, Blue, and NIR), represent the ideal convergence: providing high spatial resolution to delineate discrete building setbacks and road corridors alongside the Near-Infrared band necessary to isolate green vegetation and surface water. This project investigates models across this entire continuum—from 2.5m single-band panchromatic to 0.60m 4-band multispectral imagery.",
        bold_prefix="The Sub-Meter Multispectral Frontier: "
    )

    # ----------------------------------------------------
    # 2.2 Spectral Reflectance Models and Biophysical Indices
    # ----------------------------------------------------
    add_heading_2(doc, "2.2 Spectral Reflectance Models and Biophysical Indices")
    add_paragraph(doc, 
        "In optical remote sensing, land-cover classes exhibit characteristic spectral reflectance signatures dictated by molecular and cellular absorption properties. Healthy green vegetation exhibits low reflectance in the visible Red spectrum (0.62–0.68 µm) due to intense absorption by chlorophyll pigments (chlorophyll-a and chlorophyll-b for photosynthesis), but exhibits exceptionally high reflectance in the Near-Infrared (NIR) spectrum (0.77–0.86 µm) due to internal light scattering within the mesophyll spongy cell structure.",
        bold_prefix="Biophysical Foundations: "
    )
    add_paragraph(doc, 
        "To exploit this physical contrast, normalized mathematical ratios have been established in remote sensing literature. The Normalized Difference Vegetation Index (NDVI), formulated by Rouse et al. (1974), normalizes the contrast between NIR reflectance and Red absorption:",
        bold_prefix="NDVI Formulation: "
    )
    add_display_equation(doc, OMML_NDVI, "2.1")
    add_paragraph(doc, 
        "where ρ_NIR and ρ_Red denote calibrated top-of-atmosphere or surface reflectance in the respective bands, and ε = 10⁻⁵ is a small numerical stabilization constant. NDVI values range bounded within [-1.0, 1.0]. Vigorous green canopies typically yield NDVI values between 0.35 and 0.85, whereas non-vegetated impervious surfaces (asphalt, concrete) and bare soil cluster near zero (-0.1 to +0.15), and water surfaces exhibit negative values (< 0.0).",
        bold_prefix=""
    )
    add_paragraph(doc, 
        "Similarly, surface water bodies strongly absorb Near-Infrared radiation while reflecting visible Green light (0.52–0.59 µm). McFeeters (1996) defined the Normalized Difference Water Index (NDWI) to delineate open water features:",
        bold_prefix="NDWI Formulation: "
    )
    add_display_equation(doc, OMML_NDWI, "2.2")
    add_paragraph(doc, 
        "In 3-band visible RGB imagery where NIR is absent, vegetation can be enhanced using chromatic indices such as the Excess Green Index (ExG), defined by Woebbecke et al. (1995):",
        bold_prefix="Excess Green Index: "
    )
    add_display_equation(doc, OMML_EXG, "2.3")
    add_paragraph(doc, 
        "Integrating explicit biophysical indices directly into deep learning architectures provides inductive physical priors, significantly reducing the sample complexity required for convolutional kernels to learn vegetation and water boundaries.",
        bold_prefix="Physical Priors in Deep Learning: "
    )

    # ----------------------------------------------------
    # 2.3 Deep Encoder-Decoder Networks for Semantic Segmentation
    # ----------------------------------------------------
    add_heading_2(doc, "2.3 Deep Encoder-Decoder Networks for Semantic Segmentation")
    add_paragraph(doc, 
        "Semantic segmentation in remote sensing requires pixel-level spatial accuracy alongside high-level semantic understanding. Fully Convolutional Networks (FCNs), introduced by Long, Shelhamer, and Darrell (2015), replaced fully connected classification layers with convolutional operations, enabling dense prediction across arbitrary image dimensions.",
        bold_prefix="Evolution of Dense Prediction: "
    )
    add_paragraph(doc, 
        "Ronneberger, Fischer, and Brox (2015) introduced U-Net, an encoder-decoder architecture that remains a cornerstone of biomedical and satellite image segmentation. The contracting encoder path consists of repeated convolutional layers, non-linear activations (ReLU), and spatial downsampling operations (max pooling or strided convolutions), progressively increasing feature channel depth while expanding the receptive field to capture global contextual patterns. The expanding decoder path sequentially upsamples feature representations using transposed convolutions or bilinear interpolation.",
        bold_prefix="The U-Net Paradigm: "
    )
    add_paragraph(doc, 
        "The defining innovation of U-Net is its lateral skip connections, which concatenate high-resolution feature maps from encoder stages directly to their corresponding decoder stages. In satellite imagery, deep downsampling (e.g., \\(32\\times\\) stride reduction) discards fine spatial details, eroding narrow road ribbons and building corners. Skip connections bypass this spatial bottleneck, routing uncompressed edge, boundary, and coordinate information directly to the reconstruction head.",
        bold_prefix="Role of Skip Connections: "
    )
    add_paragraph(doc, 
        "In satellite segmentation, input patches are large (512×512 pixels), severely constraining mini-batch sizes (typically 1 to 4 samples per GPU). Ioffe and Szegedy's (2015) Batch Normalization (BatchNorm) computes mean and variance statistics across the mini-batch; at very small batch sizes, these batch statistics become noisy and erratic, leading to unstable gradient updates and poor convergence. Wu and He (2018) proposed Group Normalization (GroupNorm), which divides feature channels into predefined groups and normalizes activations within each sample independently of batch size. In our preliminary April 2026 work, GroupNorm with 8 groups stabilized training; in current-semester architectures with larger backbones, pre-trained frozen or synchronized batch statistics were leveraged.",
        bold_prefix="Normalization Under Hardware Constraints: "
    )

    # ----------------------------------------------------
    # 2.4 Vision Transformers and Attention Mechanisms
    # ----------------------------------------------------
    add_heading_2(doc, "2.4 Vision Transformers and Attention Mechanisms in Earth Observation")
    add_paragraph(doc, 
        "While CNNs possess an intrinsic inductive bias towards local spatial translation equivariance, their effective receptive field grows linearly with depth, making them struggle to capture long-range contextual relationships across large satellite tiles. Dosovitskiy et al. (2020) introduced the Vision Transformer (ViT), applying multi-head self-attention directly to flattened non-overlapping image patches.",
        bold_prefix="From CNNs to Vision Transformers: "
    )
    add_paragraph(doc, 
        "Standard ViT architectures suffer from two major limitations when applied to remote sensing: (1) dividing images into non-overlapping \\(16\\times 16\\) patches destroys sub-meter boundary continuity, and (2) standard self-attention exhibits quadratic computational complexity \\(\\mathcal{O}(N^2)\\) with respect to sequence length \\(N = H \\cdot W\\), rendering full-resolution 512×512 tile processing computationally intractable.",
        bold_prefix="ViT Limitations in Remote Sensing: "
    )
    add_paragraph(doc, 
        "To resolve these bottlenecks, Xie et al. (2021) designed SegFormer, a hierarchical transformer framework tailored for dense semantic segmentation. SegFormer introduces two structural advances: (1) Overlapped Patch Merging, which extracts overlapping patch embeddings, preserving localized spatial boundary continuity between adjacent tokens; and (2) Efficient Multi-Head Self-Attention, which sub-samples key and value sequences by a reduction ratio R, reducing attention complexity from O(N²) to O(N² / R) according to Equation (2.4):",
        bold_prefix="The SegFormer Architecture: "
    )
    add_display_equation(doc, OMML_ATTENTION, "2.4")
    add_paragraph(doc, 
        "where S_R(·) reduces spatial sequence length from N to N/R. Furthermore, SegFormer eliminates complex positional encodings by introducing a Depthwise 3×3 Convolution within the Feed-Forward Network (Mix-FFN), enabling seamless zero-shot evaluation on arbitrary image resolutions without interpolation artifacts. A lightweight All-MLP Decoder unifies multi-level hierarchical representations to predict dense class masks.",
        bold_prefix=""
    )
    add_paragraph(doc, 
        "Complementary to transformers, Roy, Navab, and Wachinger (2018) introduced Concurrent Spatial and Channel Squeeze-and-Excitation (scSE) blocks for CNNs. An scSE block combines Spatial Squeeze and Channel Excitation (sSE), which recalibrates feature channels based on global context, with Channel Squeeze and Spatial Excitation (cSE), which generates a 2D spatial attention map highlighting salient edge structures. In this project, scSE modules are embedded within U-Net skip connections and subclass morphology heads to refine sub-meter building perimeters.",
        bold_prefix="scSE Dual Attention Blocks: "
    )

    # ----------------------------------------------------
    # 2.5 Hierarchical and Multi-Level Land-Cover Categorization
    # ----------------------------------------------------
    add_heading_2(doc, "2.5 Hierarchical and Multi-Level Land-Cover Categorization")
    add_paragraph(doc, 
        "In geographic information science, land-cover taxonomies are inherently hierarchical (Anderson et al. 1976). At the macro level, Earth surfaces are categorized into fundamental biophysical envelopes: Water Bodies, Forest/Vegetation, Barren Land, Built-up Structures, and Transportation Networks. At the micro level, built-up environments split into functional and socioeconomic subclasses.",
        bold_prefix="Taxonomic Hierarchy: "
    )
    add_paragraph(doc, 
        "A profound research insight emerges regarding informal settlements (slums): Slums cannot be accurately modeled or segmented as isolated, individual building footprints. In high-density informal settlements (such as Dharavi in Mumbai or pockets along railway lines in Pune), residential shacks physically touch one another with zero building setback. Roads narrow into irregular pedestrian pathways (<2m wide), and roofs consist of weathered corrugated galvanized iron (CGI), rusty tin sheets, and plastic tarpaulins. Consequently, slums exhibit high spatial entropy and must be delineated as contiguous settlement zones or clusters, whereas formal residential and industrial structures exhibit distinct, regularized building perimeters with setbacks.",
        bold_prefix="The Slum Delineation Insight: "
    )
    add_paragraph(doc, 
        "If a flat 7-class segmentation network is trained naively on satellite imagery, models frequently predict false-positive slum pixels inside remote agricultural fields, dry riverbeds, or forest fringes due to subtle spectral similarities between weathered tin sheets and bare soil. Enforcing an architectural hierarchy—where the network first defines a high-confidence macro Built-up envelope, and subsequently classifies subclasses strictly within that spatial envelope—imposes a structural constraint that completely eliminates false positive slums in non-built terrain.",
        bold_prefix="Structural Spatial Gating: "
    )

    # ----------------------------------------------------
    # 2.6 Loss Formulations for Imbalanced, Boundary, and Topological Structures
    # ----------------------------------------------------
    add_heading_2(doc, "2.6 Loss Formulations for Imbalanced, Boundary, and Topological Structures")
    add_paragraph(doc, 
        "Standard semantic segmentation pipelines optimize pixel-wise Categorical Cross-Entropy (CE) according to Equation (2.5):",
        bold_prefix="Cross-Entropy Limitations: "
    )
    add_display_equation(doc, OMML_LCE, "2.5")
    add_paragraph(doc, 
        "In urban remote sensing scenes, class distributions are heavily skewed: background classes (Barren Land, Vegetation) often encompass 70–85% of total scene pixels, whereas narrow road networks comprise only 5–15%, and informal slums may represent <0.1%. When optimizing standard CE, gradient updates are dominated by easy, well-classified background pixels, causing the network to ignore minority classes.",
        bold_prefix=""
    )
    add_paragraph(doc, 
        "To mitigate this imbalance, Lin et al. (2017) formulated Focal Loss, introducing a dynamic modulating factor (1 - p_i,c)^γ with focusing parameter γ > 0 according to Equation (2.6):",
        bold_prefix="Focal Loss Formulation: "
    )
    add_display_equation(doc, OMML_LFOCAL, "2.6")
    add_paragraph(doc, 
        "As a pixel becomes well-classified (\\(P_{i, c} \\to 1\\)), the modulating factor approaches zero, heavily downweighting the loss contribution of easy background regions and focusing backpropagation on hard boundary ambiguities.",
        bold_prefix=""
    )
    add_paragraph(doc, 
        "To directly maximize the Intersection over Union (Jaccard Index) metric, Berman, Triki, and Blaschko (2018) introduced the Lovász-Softmax loss, which establishes a convex, smooth surrogate for the discrete Jaccard loss using the Lovász extension of submodular set functions. Lovász-Softmax directly optimizes global IoU without requiring heuristic threshold tuning.",
        bold_prefix="Lovász-Softmax Surrogate: "
    )
    add_paragraph(doc, 
        "For transport corridors, Shit et al. (2021) introduced Centerline Dice Loss (clDice). clDice applies soft morphological erosion operations to extract continuous 1D skeletons of both predicted masks and ground-truth roads, computing the harmonic mean of topological precision and topological recall. This strictly penalizes road centerline fractures, ensuring continuous network topology under tree overhangs.",
        bold_prefix="Topology-Preserving clDice: "
    )

    # ----------------------------------------------------
    # 2.7 Weak and Pseudo-Supervision from Open Geospatial Data
    # ----------------------------------------------------
    add_heading_2(doc, "2.7 Weak and Pseudo-Supervision from Open Geospatial Data")
    add_paragraph(doc, 
        "Generating manual polygon annotations for multi-square-kilometer satellite scenes is prohibitively expensive. Researchers have increasingly investigated OpenStreetMap (OSM)—a global collaborative open-source geospatial vector database—as a source of weak supervision (Kaiser et al. 2017; Vargas-Muñoz et al. 2020).",
        bold_prefix="OpenStreetMap as Supervision: "
    )
    add_paragraph(doc, 
        "However, raw OSM vector geometries cannot be naively rasterized into multi-class masks. In OpenStreetMap, roads are represented as 1D centerline polylines, building footprints are represented as 2D closed polygons, and natural areas (rivers, parks) are represented as multi-polygons. If polylines and polygons are burned without strict collision rules, broad building footprints frequently overwrite narrow street lanes, or road buffers obliterate roadside structures. A robust automated pipeline must implement strict Priority Burning Rules (Roads > Built-up > Water > Vegetation > Barren Land) and incorporate dynamic buffer widths based on highway functional hierarchies (motorway, primary, secondary, residential) to produce clean, non-conflicting supervision signals.",
        bold_prefix="Priority-Based Vector Burning: "
    )

    # ----------------------------------------------------
    # 2.8 Evaluation Protocols Beyond Pixel Accuracy
    # ----------------------------------------------------
    add_heading_2(doc, "2.8 Evaluation Protocols Beyond Pixel Accuracy")
    add_paragraph(doc, 
        "In remote sensing literature, reporting only Overall Accuracy (OA) is widely recognized as misleading. In a scene dominated by 80% barren land, a naive model that predicts 100% barren land achieves 80% Overall Accuracy while providing zero utility for urban infrastructure extraction. To ensure rigorous, publication-grade validation, contemporary benchmarks enforce multi-tiered evaluation protocols:",
        bold_prefix="The Metric Inadequacy of Pixel Accuracy: "
    )
    add_bullet(doc, 
        "Strict 0m Pixel IoU: Measures exact pixel-by-pixel intersection over union (|P ∩ G| / |P ∪ G|). Highly unforgiving of sub-meter boundary shifts caused by orthorectification discrepancies or vector burning offsets.",
        bold_prefix="1. Strict Pixel IoU: "
    )
    add_bullet(doc, 
        "SpaceNet Metric Buffer IoU: Formulated by the official SpaceNet challenge (Van Etten et al. 2018), this protocol dilates ground-truth building footprints and road centerlines by an operational spatial tolerance buffer (e.g., 5m or 10m ground radius). This rewards correct corridor discovery and structural localization without excessively penalizing minor cartographic boundary drawing offsets.",
        bold_prefix="2. SpaceNet Buffer Tolerance: "
    )
    add_bullet(doc, 
        "Mnih Relaxed Precision and Recall: Established by Volodymyr Mnih (2013) for aerial road extraction, a predicted road pixel is considered a true positive if it lies within a slack radius ρ (e.g., 8 px ≈ 4.8m) of any ground-truth road pixel. Mnih relaxed precision, recall, and F1 (Dice) measure the true operational completeness and correctness of the extracted network.",
        bold_prefix="3. Mnih Relaxed Metrics: "
    )
    add_bullet(doc, 
        "Boundary F1-Score (BF1): Formulated by Csurka et al. (2013), BF1 evaluates the 1-pixel morphological boundary contours of predicted objects against ground-truth contours within an operational distance threshold, quantifying boundary crispness independently of object area.",
        bold_prefix="4. Boundary Delineation (BF1): "
    )
    
    doc.add_page_break()

