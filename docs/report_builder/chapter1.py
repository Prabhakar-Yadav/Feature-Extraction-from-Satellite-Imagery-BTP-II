"""
scripts/report_builder/chapter1.py
----------------------------------
Chapter 1: Introduction
- 1.1 Motivation and Societal Context
- 1.2 Problem Definition and Urban Remote Sensing Challenges
- 1.3 Foundation Established in Previous Semester (April 2026)
- 1.4 Current-Semester Objectives and Core Contributions
- 1.5 Research Questions and Experimental Hypotheses
- 1.6 Report Organization
"""

import docx
from docx.shared import Inches, Pt
import styles
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table
)

def add_chapter_1(doc):
    add_heading_1(doc, "Chapter 1: Introduction")
    
    # ----------------------------------------------------
    # 1.1 Motivation and Societal Context
    # ----------------------------------------------------
    add_heading_2(doc, "1.1 Motivation and Societal Context")
    add_paragraph(doc, 
        "Modern metropolitan landscapes in developing nations, particularly in India, are undergoing unprecedented rates of physical transformation. Rapid urbanization, large-scale industrial corridor expansion, dynamic road infrastructure development, and the informal growth of dense settlements continuously reshape the metropolitan fabric across cities such as Mumbai, Pune, and Chennai. For municipal corporations, regional development authorities, environmental protection agencies, and disaster response teams, maintaining up-to-date, topologically coherent, and semantically detailed geospatial databases is a fundamental prerequisite for evidence-based governance.",
        bold_prefix="Urban Growth Dynamics: "
    )
    add_paragraph(doc, 
        "Traditional ground surveys, land-registry deed audits, and manual visual digitization from aerial photographs are exceedingly labor-intensive, cost-prohibitive, and inherently incapable of keeping pace with high-velocity urban sprawl. Spaceborne Earth observation platforms offer a scalable, repeatable, and non-invasive alternative, capturing vast geographical swathes at sub-meter Ground Sampling Distances (GSD).",
        bold_prefix="The Earth Observation Imperative: "
    )
    add_paragraph(doc, 
        "However, raw satellite rasters cannot be directly ingested into municipal decision workflows. Before satellite data can inform property tax appraisal, informal settlement rehabilitation, transport corridor planning, or green canopy conservation, high-dimensional radiometry must be converted into structured, georeferenced vector layers. Automating this thematic extraction via deep learning models that operate reliably across varying sensor modalities, radiometric calibrations, and spatial resolutions represents both a major scientific frontier and a vital engineering necessity.",
        bold_prefix="From Imagery to Intelligence: "
    )

    # ----------------------------------------------------
    # 1.2 Problem Definition and Urban Remote Sensing Challenges
    # ----------------------------------------------------
    add_heading_2(doc, "1.2 Problem Definition and Urban Remote Sensing Challenges")
    add_paragraph(doc, 
        "Given an input satellite scene captured across visible, panchromatic, or multispectral bands, the core computer vision objective is semantic segmentation: assigning a mutually exclusive categorical land-cover label to every individual pixel. While deep convolutional neural networks (CNNs) and vision transformers have revolutionized general computer vision benchmarks, remote sensing of complex urban environments poses distinct, domain-specific obstacles:",
        bold_prefix="Task Formulation: "
    )
    add_bullet(doc, 
        "Pervasive Spectral Confusion: In single-band panchromatic and 3-band visible RGB imagery, materials with vastly different functional roles exhibit nearly identical reflectance values. Aged asphalt road surfaces, weathered concrete flat roofs, compacted barren soil, and unpaved parking lots share overlapping gray-level histograms, causing naive classifiers to bleed road predictions onto adjacent building slabs.",
        bold_prefix="1. Severe Spectral Overlap: "
    )
    add_bullet(doc, 
        "Structural Shadow Occlusion: In high-resolution imagery, tall multi-story buildings and industrial warehouses cast deep, sharp radiometric shadows. These shadow corridors absorb visible light, reducing digital number values to near-zero levels. Standard models routinely misclassify dark building shadows as water bodies, or become confused along building-road interfaces where shadows obscure physical access lanes.",
        bold_prefix="2. Cast Shadows and Vector Offsets: "
    )
    add_bullet(doc, 
        "Linear Road Connectivity vs. Tree Canopies: Unlike expansive natural land-cover classes (forests, water bodies), transport networks form narrow, continuous linear topologies often spanning only 3 to 10 pixels in width. Street trees and roadside vegetation frequently overhand and obscure road pavements. Traditional pixel-wise cross-entropy losses treat road breaks as minor local errors, resulting in fragmented, disconnected road networks unsuitable for navigation.",
        bold_prefix="3. Topological Fragility: "
    )
    add_bullet(doc, 
        "Intra-Class Diversity in Built-Up Land: Urban structures vary drastically in scale, layout, and roofing material. Delineating a generic 'Built-up' class obscures critical socioeconomic realities: organized residential apartment blocks possess regular setbacks and internal gardens; sprawling industrial logistics parks span thousands of square meters with corrugated iron roofs; and informal settlements (slums) consist of highly congested, contiguous clusters with irregular tin and tarpaulin sheets and no formal street access.",
        bold_prefix="4. Subclass Granularity: "
    )
    add_bullet(doc, 
        "Severe Annotation Scarcity: Manually annotating pixel-accurate masks for 0.60m resolution satellite scenes containing hundreds of thousands of discrete roof footprints requires hundreds of human hours per square kilometer. Developing automated, weakly supervised, and physically guided annotation strategies is essential for practical scalability.",
        bold_prefix="5. Ground-Truth Bottleneck: "
    )

    # ----------------------------------------------------
    # 1.3 Foundation Established in Previous Semester (April 2026)
    # ----------------------------------------------------
    add_heading_2(doc, "1.3 Foundation Established in Previous Semester (April 2026)")
    add_paragraph(doc, 
        "The preliminary phase of this B.Tech Project, completed in April 2026 and documented in the initial project report ('Report_BTP (PRABHAKAR).pdf'), formulated an end-to-end multi-sensor baseline pipeline. That work investigated three distinct satellite datasets spanning different spatial resolutions and sensor modalities:",
        bold_prefix="Previous Milestone Overview: "
    )
    add_bullet(doc, 
        "High-Resolution Aerial RGB Imagery (GB-NM): 1.6m GSD, 3-band visible camera capturing 25.4 million pixels. Trained a standard U-Net with GroupNorm (6 classes), achieving 68.20% Overall Accuracy on independent test data using spectral-index-derived pseudo-labels.",
        bold_prefix="Dataset A: "
    )
    add_bullet(doc, 
        "Panchromatic Cartosat-1 Imagery: 1.0m GSD monochromatic data capturing 19.8 million pixels. Investigated deep-feature clustering using a frozen ImageNet-pretrained ResNet-18 backbone, training a patch classifier across 6 classes that achieved 91.05% validation accuracy and 94.74% test accuracy.",
        bold_prefix="Dataset B: "
    )
    add_bullet(doc, 
        "Moderate-Resolution Sentinel-2 Multispectral Imagery: 10m GSD, 4-band (Blue, Green, Red, NIR) scene covering Vashi, Navi Mumbai. Trained a U-Net semantic segmentation network across 7 classes, reaching 75.71% Overall Accuracy on manually annotated ground truth.",
        bold_prefix="Dataset C: "
    )
    add_bullet(doc, 
        "Instance Detection and GIS Web Platform: Implemented a Mask R-CNN model achieving 87.9% building instance recall (at IoU 0.5), and developed 'GeoVision', a Streamlit web platform featuring automatic model selection, sliding-window Hann-blended inference, and georeferenced GeoTIFF/Shapefile export.",
        bold_prefix="System Infrastructure: "
    )

    # Table 1.1: Previous Results
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(2)
    p_t1.paragraph_format.keep_with_next = True
    r_t1 = p_t1.add_run("Table 1.1: Previous-semester baseline results summary (April 2026 Reference Baseline).")
    r_t1.font.name = "Times New Roman"
    r_t1.font.size = Pt(9.5)
    r_t1.font.bold = True
    r_t1.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    prev_data = [
        ["U-Net (GroupNorm)", "Aerial RGB (1.6m)", "3 Bands (RGB)", "6 Classes", "69.65%", "68.20%", "Unsupervised Spectral KMeans Pseudo-Labels"],
        ["ResNet-18 Patch CNN", "Cartosat-1 (1.0m)", "1 Band (PAN)", "6 Classes", "91.05%", "94.74%", "ResNet-18 Deep-Feature Patch Clustering"],
        ["U-Net (GroupNorm)", "Sentinel-2 (10m)", "4 Bands (B,G,R,NIR)", "7 Classes", "78.40%", "75.71%", "Manually Drawn Ground Truth (Vashi Scene)"],
        ["Mask R-CNN (ResNet-50)", "Aerial RGB (1.6m)", "3 Bands (RGB)", "Building Instance", "—", "87.9% Recall", "Vector Building Footprint Polygons (IoU 0.5)"]
    ]
    add_styled_table(doc, 
        ["Model Architecture", "Satellite Sensor / Scene", "Input Channels", "Class Scheme", "Val Accuracy", "Test Accuracy", "Ground Truth Supervision Type"], 
        prev_data, 
        col_widths=[1.5, 1.2, 1.0, 0.9, 0.9, 1.0, 1.5], 
        alignment=['L', 'L', 'C', 'C', 'R', 'R', 'L']
    )

    add_callout(doc, 
        "CRITICAL LIMITATIONS IDENTIFIED IN THE APRIL 2026 WORK", 
        "1. Panchromatic Resolution Bottleneck: Cartosat-1 (1.0m/2.5m) imagery lacked Near-Infrared radiometry; spectral index methods (NDVI) could not be deployed, forcing reliance on spatial texture.\n"
        "2. Pseudo-Label Quality Degradation: Unsupervised KMeans clustering grouped pixels by raw spectral similarity rather than true semantic boundaries, introducing systematic label noise along building-road perimeters.\n"
        "3. Coarse 10m Sentinel-2 Limitations: While Sentinel-2 provided NIR, its 10m GSD was far too coarse to resolve individual urban roads, narrow street networks, or informal settlement clusters.\n"
        "4. Absence of Subclass Granularity: All built-up structures were lumped into a single generic 'Building' category, failing to distinguish planned residential areas, industrial sheds, and informal slums.\n"
        "5. Road Network Disconnection: Models trained with pixel-wise Cross-Entropy produced severely fragmented road centerlines under tree overhangs and building shadows."
    )

    # ----------------------------------------------------
    # 1.4 Current-Semester Objectives and Core Contributions
    # ----------------------------------------------------
    add_heading_2(doc, "1.4 Current-Semester Objectives and Core Contributions")
    add_paragraph(doc, 
        "To overcome the fundamental limitations of the preliminary April 2026 models, the current semester (Autumn 2026) executed a major technological and methodological pivot. The work transitioned from exploratory clustering to high-resolution multispectral remote sensing, rigorous vector burning, vision transformer benchmarking, and hierarchical two-step joint optimization. The core contributions are structured across two principal experimental thrusts:",
        bold_prefix="Methodological Leap: "
    )
    
    add_heading_3(doc, "Thrust A: High-Resolution Cartosat Feature Extraction (2.5m PAN & 0.60m Multispectral)")
    add_bullet(doc, 
        "Automated OpenStreetMap Vector Burning for 2.5m Panchromatic Cartosat-1: Engineered an automated geographic vector-rasterization engine utilizing the OSM Overpass API. Enforced a strict priority-burning hierarchy (Roads > Built-up > Water > Vegetation > Barren Land) to produce hundreds of verified 5-class ground-truth masks across Navi Mumbai and Mumbai Central without manual tracing.",
        bold_prefix="1. Vector-to-Raster Engine: "
    )
    add_bullet(doc, 
        "Vision Transformer (SegFormer) vs. CNN (U-Net) Benchmark: Executed a comprehensive head-to-head empirical comparison between hierarchical multi-scale Vision Transformers (SegFormer-B2 with MiT encoder) and deep residual CNNs (ResNet-34 U-Net) on 2.5m single-band data. Documented their complementary spatial vs. global contextual behaviors (SegFormer achieving 89.8% Water IoU; U-Net achieving 66.6% SpaceNet Road Buffer IoU and 81.89% OA).",
        bold_prefix="2. Transformer vs. CNN Study: "
    )
    add_bullet(doc, 
        "Zero-Shot Cross-City Transferability Evaluation: Evaluated the frozen 2.5m Cartosat-1 SegFormer model across 30 completely unseen strategic patches (49.15 km²) spanning Pune City, Pune Hinjwadi, and Chennai, observing up to 70.0% SpaceNet road IoU on planned arterial transit corridors without fine-tuning.",
        bold_prefix="3. Zero-Shot Generalization: "
    )
    add_bullet(doc, 
        "0.60m Sub-Meter Multispectral Adaptation (Cartosat-2E): Expanded the pipeline to 4-band multispectral imagery (Red, Green, Blue, NIR) over the Pune Region. Integrated biophysical NDVI thresholding to establish pristine vegetation boundaries.",
        bold_prefix="4. Sub-Meter Radiometry: "
    )
    add_bullet(doc, 
        "Unified Hybrid SegFormer + UltraRoad Attention U-Net: Designed a novel parallel architecture that dynamically fuses global transformer context and sub-meter CNN edge preservation via a learned gating coefficient.",
        bold_prefix="5. Hybrid Architecture: "
    )
    add_bullet(doc, 
        "In-Epoch Decoupled Two-Step Hierarchical Training: Solved the informal settlement problem by formulating a two-step joint optimization pipeline where macro 5-class land-cover is predicted first, and 3 fine-grained subclasses (Residential, Industrial, Slum) are trained strictly inside the predicted built-up area in the same epoch. Decoupling backward passes eliminated CUDA Out-of-Memory (OOM) failures, maintaining peak GPU VRAM below 1.74 GB on a strict 2.4 GB hardware cap.",
        bold_prefix="6. Hierarchical Optimization: "
    )
    add_bullet(doc, 
        "Physics-Based Asphalt-Safe Guard: Developed a hybrid spectral rule combining NDWI, NIR reflectance, and deep learning model priors to completely eliminate the historical bug where dark asphalt roads and shaded roofs were incorrectly classified as water bodies.",
        bold_prefix="7. Asphalt-Safe Guard: "
    )
    add_bullet(doc, 
        "Rigorously Verified Multi-Level Benchmark on 186 Held-Out Patches: Evaluated the hierarchical 7-class pipeline on an independent 186-patch Pune test scene under strict 0m pixel IoU, SpaceNet 5m buffer tolerance, and Mnih relaxed metrics. Exceeded the 50% target by achieving 73.22% Residential IoU, 50.33% Industrial IoU (61.78% active subclass mean), and an overall 7-class SpaceNet buffer mIoU of 57.15%.",
        bold_prefix="8. Official Benchmark: "
    )

    add_heading_3(doc, "Thrust B: VASHI SPOT 4-Band Multisensor Classification (Controlled Ablations)")
    add_bullet(doc, 
        "Strict Geographic Partitioning and Partial-Label Supervision: Curated a pristine 4-band SPOT (Vashi) and 3-band Pléiades (Mumbai) multi-sensor dataset with strictly held-out test splits (29 patches). Implemented source-aware conditional softmax and masked loss functions to enable joint training without penalizing unannotated classes.",
        bold_prefix="1. Multisensor Protocol: "
    )
    add_bullet(doc, 
        "Controlled Model Progression (Baseline -> E1 -> E2 -> E3 -> E4): Conducted disciplined, single-variable iterations from reference model E1 (ResNet18 + scSE dual stem, native shallow skip, uniform sampling) which achieved 75.01% Val Accuracy, 61.40% mIoU, and 66.88% Boundary F1.",
        bold_prefix="2. Systematic Iteration: "
    )
    add_bullet(doc, 
        "Rigorous Failure Diagnosis of Speculative Approaches: Proved empirically why hard-example mining (E2) failed (oversampling shadow noise and vector misalignments, degrading mIoU to 50.15%) and why guided boundary attention (E3) regressed (gate over-suppressing low-contrast building roofs, reducing mIoU to 53.38%).",
        bold_prefix="3. Honest Failure Analysis: "
    )
    add_bullet(doc, 
        "Physics-Based NDVI and Calibrated Focal-Dice (E4): Integrated 5-band radiometry (RGB + NIR + NDVI) and Focal-Dice loss (gamma=1.5) with calibrated class weights. Succeeded in eliminating road-widening bloat on diagnostic commercial conflict sectors (e.g., vashi_0041), boosting accuracy to 70.95% and Non-slum Building IoU to 67.11%, while calibrating Road prediction to 14.35% (matching ground truth 16.9%).",
        bold_prefix="4. Targeted Sector Solution: "
    )

    # ----------------------------------------------------
    # 1.5 Research Questions and Experimental Hypotheses
    # ----------------------------------------------------
    add_heading_2(doc, "1.5 Research Questions and Experimental Hypotheses")
    add_paragraph(doc, 
        "To guide the experimental methodology and ensure scientific rigor, this research addresses four primary research questions (RQs):",
        bold_prefix=""
    )
    add_paragraph(doc, 
        "Can hierarchical multi-head self-attention mechanisms in Vision Transformers (SegFormer) outperform deep convolutional networks with skip connections (U-Net) on single-band panchromatic satellite imagery where spectral color information is completely absent?\n"
        "Hypothesis: Transformers will excel on large, continuous natural classes (Water, Vegetation) by capturing long-range contextual relationships across the entire tile, whereas CNNs with direct skip connections will preserve sharper linear road edges and building right angles.",
        bold_prefix="RQ1 (Transformer vs. CNN on Panchromatic Imagery): "
    )
    add_paragraph(doc, 
        "Can a two-step in-epoch hierarchical training formulation eliminate false positive informal settlement (slum) detections across natural land-cover classes while resolving subclass morphology?\n"
        "Hypothesis: Enforcing hard spatial gating—such that subclass classifiers are supervised strictly within the predicted macro built-up envelope—will eliminate 100% of false slum detections in waterways, forests, and barren fields.",
        bold_prefix="RQ2 (Hierarchical Subclass Spatial Gating): "
    )
    add_paragraph(doc, 
        "Can explicit biophysical vegetation features (NDVI) combined with calibrated Focal-Dice loss prevent deep segmentation models from misclassifying flat concrete roofs as roads?\n"
        "Hypothesis: Integrating bounded NDVI as an explicit 5th channel and penalizing easy background predictions via Focal loss will break the spectral tie between asphalt roads and concrete roofs, narrowing predicted road widths to match true centerlines.",
        bold_prefix="RQ3 (Biophysical Indices and Road Bleed Resolution): "
    )
    add_paragraph(doc, 
        "Does hard-example mining and dynamic boundary attention improve feature extraction in dense urban centers, or does it destabilize training due to label noise?\n"
        "Hypothesis: In satellite imagery where building boundaries suffer from vector offset errors and cast-shadow ambiguity, oversampling 'hard' patches will amplify annotation noise, causing global performance to degrade relative to uniform balanced sampling.",
        bold_prefix="RQ4 (Sampling Dynamics vs. Ground-Truth Noise): "
    )

    # ----------------------------------------------------
    # 1.6 Report Organization
    # ----------------------------------------------------
    add_heading_2(doc, "1.6 Report Organization")
    add_paragraph(doc, 
        "The remainder of this report is organized as follows:",
        bold_prefix=""
    )
    add_bullet(doc, 
        "Chapter 2 reviews foundational literature in remote sensing, spectral indices, convolutional segmentation architectures, vision transformers, hierarchical classification, and topological loss functions.",
        bold_prefix="Chapter 2 (Literature Review): "
    )
    add_bullet(doc, 
        "Chapter 3 details the satellite datasets (Cartosat-1 2.5m, Cartosat-2E 0.60m multispectral, Vashi SPOT 4-band), radiometric normalization, automated OSM vector rasterization, NDVI vegetation masking, and geographic partitioning.",
        bold_prefix="Chapter 3 (Datasets & Ground Truth): "
    )
    add_bullet(doc, 
        "Chapter 4 presents the model architectures, including the SegFormer vs. U-Net models for 2.5m data, the Unified Hybrid SegFormer + UltraRoad U-Net for 0.60m data, the in-epoch two-step hierarchical training pipeline, memory decoupling, Asphalt-Safe Guard, and composite loss formulations.",
        bold_prefix="Chapter 4 (Methodology & Architecture): "
    )
    add_bullet(doc, 
        "Chapter 5 describes the experimental configurations, training dynamics, zero-shot transferability studies, and the controlled Vashi progression (E1 to E4), detailing root causes for failed ablations.",
        bold_prefix="Chapter 5 (Experimental Progression): "
    )
    add_bullet(doc, 
        "Chapter 6 provides comprehensive quantitative benchmark evaluations under SpaceNet and Mnih standards, multi-panel visual comparisons, per-patch diagnostic analyses, and qualitative failure interpretations.",
        bold_prefix="Chapter 6 (Results & Visual Evaluation): "
    )
    add_bullet(doc, 
        "Chapter 7 outlines the production repository implementation, sliding-window Hann-blended inference, GIS vectorization, runtime benchmarks, and GPU VRAM profiling.",
        bold_prefix="Chapter 7 (Implementation & GIS Workflow): "
    )
    add_bullet(doc, 
        "Chapter 8 summarizes the major scientific and engineering findings, answers the research questions, outlines current limitations, and defines the prioritized roadmap for Semester VIII.",
        bold_prefix="Chapter 8 (Conclusion & Future Work): "
    )
    
    doc.add_page_break()
