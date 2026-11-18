"""
scripts/report_builder/chapter8.py
----------------------------------
Chapter 8: Conclusions and Future Work
- 8.1 Summary of Completed Research and Engineering Deliverables
- 8.2 Answers to Research Questions
- 8.3 Current Limitations and Technical Bottlenecks
- 8.4 Prioritized Roadmap for Semester VIII
"""

import docx
from docx.shared import Inches, Pt
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout
)

def add_chapter_8(doc):
    add_heading_1(doc, "Chapter 8: Conclusions and Future Work")
    
    # ----------------------------------------------------
    # 8.1 Summary of Completed Research and Engineering Deliverables
    # ----------------------------------------------------
    add_heading_2(doc, "8.1 Summary of Completed Research and Engineering Deliverables")
    add_paragraph(doc, 
        "This B.Tech Project has established an end-to-end, reproducible deep learning framework for urban feature extraction and hierarchical subclass segmentation from multi-resolution, multi-sensor satellite imagery. Transitioning from the preliminary semi-supervised clustering baselines of April 2026, the current semester delivered major scientific and engineering advances across two primary thrusts: high-resolution Cartosat feature extraction and controlled multisensor Vashi SPOT classification.",
        bold_prefix="Semester VII Synthesis: "
    )
    add_bullet(doc, 
        "Automated OpenStreetMap Vector-to-Raster Engine: Solved the manual annotation bottleneck for 2.5m panchromatic Cartosat-1 data by designing an automated Overpass API rasterization engine with strict priority burning (Roads > Built-up > Water > Vegetation > Barren Land), producing hundreds of verified 5-class masks across Mumbai.",
        bold_prefix="1. Vector Burning Engine: "
    )
    add_bullet(doc, 
        "Rigorous Vision Transformer vs. CNN Comparative Benchmark: Proved that SegFormer-B2 and UltraRoad U-Net exhibit complementary spatial behaviors on single-band data. SegFormer achieved 89.8% Water IoU and 78.5% Vegetation IoU via global multi-head self-attention; UltraRoad U-Net achieved 81.89% Overall Accuracy and 66.63% SpaceNet 10m buffer Road IoU via high-resolution skip connections. Zero-shot transfer across 30 unseen patches in Pune and Chennai demonstrated up to 70.0% SpaceNet road IoU on highway corridors.",
        bold_prefix="2. Transformer vs. CNN Study: "
    )
    add_bullet(doc, 
        "Unified Hybrid SegFormer + UltraRoad Attention U-Net on 0.60m Multispectral Data: Fused parallel transformer global context and sub-meter CNN edge preservation via a learned dynamic gating coefficient α_macro, supported by an explicit NDVI vegetation extraction prior.",
        bold_prefix="3. Sub-Meter Hybrid Architecture: "
    )
    add_bullet(doc, 
        "In-Epoch Decoupled Two-Step Hierarchical Training Pipeline: Formulated a two-step joint optimization pipeline where macro 5-class land-cover gates fine-grained subclass categorization (Residential, Industrial, Slum), eliminating false positive slums in water and agricultural fields. Decoupled backward passes cut peak GPU memory from >3.2 GB to 1.74 GB, enabling full training on an RTX 3050 4GB laptop GPU within a strict 2.4 GB cap.",
        bold_prefix="4. Hierarchical Joint Optimization: "
    )
    add_bullet(doc, 
        "Physics-Based Asphalt-Safe Guard: Completely eliminated the long-standing failure mode of dark asphalt roads and shaded roofs being misclassified as water by enforcing joint NDWI, NIR, and deep model confidence priors.",
        bold_prefix="5. Asphalt-Safe Guard: "
    )
    add_bullet(doc, 
        "Exceeded Subclass Benchmark Targets on 186 Held-Out Patches: Evaluated on an independent 186-scene Pune test set under SpaceNet and Mnih standards. Exceeded the 50% target by reaching 73.22% Residential IoU, 50.33% Industrial IoU (61.78% active subclass mean), 94.04% Barren Land buffer IoU, 92.82% Vegetation buffer IoU, 90.77% Water buffer IoU, and an overall 7-class SpaceNet buffer mIoU of 57.15%.",
        bold_prefix="6. SpaceNet Multi-Tier Verification: "
    )
    add_bullet(doc, 
        "Disciplined Controlled Ablation Progression on Vashi SPOT Data: Proved empirically why hard-mining (E2) and guided boundary attention (E3) failed, while establishing Experiment E1 as the primary global reference checkpoint (75.01% Val Accuracy, 61.40% mIoU, 66.88% BF1) and Experiment E4 as the specialized commercial conflict model (70.95% accuracy and 67.11% Non-slum IoU on vashi_0041, eliminating road bloat).",
        bold_prefix="7. Vashi Controlled Progression: "
    )
    add_bullet(doc, 
        "Production-Grade GIS Export Platform: Delivered seamless sliding-window inference with 2D Hann blending, automated polygonization with area/geometry filtering, and export to georeferenced GeoTIFFs, Shapefiles, and GeoJSON validated in QGIS.",
        bold_prefix="8. Full GIS Platform: "
    )

    # ----------------------------------------------------
    # 8.2 Answers to Research Questions
    # ----------------------------------------------------
    add_heading_2(doc, "8.2 Answers to Research Questions")
    
    add_paragraph(doc, 
        "Can Vision Transformers (SegFormer) outperform deep CNNs (U-Net) on single-band panchromatic imagery?\n"
        "Empirical Answer: Neither architecture is unilaterally superior. SegFormer dominates on large continuous natural classes (Water: 89.8% vs. 89.0%; Vegetation: 78.5% vs. 77.2%) because its self-attention mechanism models global tile context from Stage 1, eliminating isolated noise in open terrain. Conversely, UltraRoad U-Net excels on sharp, compact built structures (63.6% vs. 61.5% IoU) and continuous transport corridors (66.6% vs. 62.4% SpaceNet 10m buffer IoU) because its direct skip connections bypass the spatial downsampling bottleneck. The optimal strategy is parallel hybridization.",
        bold_prefix="Answer to RQ1 (Transformer vs. CNN on Panchromatic Data): "
    )
    add_paragraph(doc, 
        "Can a two-step in-epoch hierarchical training formulation eliminate false positive slum detections across natural land-cover classes?\n"
        "Empirical Answer: Yes. Enforcing hard spatial gating—where the subclass head is supervised and evaluated strictly within the predicted macro built-up envelope M_builtup—completely eliminated false positive slum predictions across waterways, agricultural green fields, and barren land, achieving a clean 0% false positive rate across natural terrain.",
        bold_prefix="Answer to RQ2 (Hierarchical Subclass Spatial Gating): "
    )
    add_paragraph(doc, 
        "Can explicit biophysical vegetation features (NDVI) combined with calibrated Focal-Dice loss prevent deep segmentation models from misclassifying flat concrete roofs as roads?\n"
        "Empirical Answer: Yes. Integrating bounded NDVI as an explicit 5th channel alongside calibrated Focal-Dice loss (gamma=1.5, Road weight=0.65) successfully broke the spectral tie on severe commercial conflict sectors (vashi_0041). Patch accuracy rose from 68.59% to 70.95%, Non-slum building IoU surged from 61.54% to 67.11%, and predicted road percentage was perfectly calibrated down from 46.11% to 31.70% (matching true ground truth of 29.32%). However, because dry soil and concrete roofs both exhibit flat near-zero NDVI, this introduces an asymmetric global trade-off on barren open land.",
        bold_prefix="Answer to RQ3 (Biophysical Indices and Road Bleed Resolution): "
    )
    add_paragraph(doc, 
        "Does hard-example mining improve urban boundary precision, or does it destabilize training due to label noise?\n"
        "Empirical Answer: In satellite imagery containing vector misalignments and shadow ambiguities, hard-example mining severely degrades performance. Oversampling 'hard' patches (E2) amplified label noise in shadow corridors, causing validation accuracy to drop by -12.54% and Non-slum IoU to collapse from 59.43% to 38.52%. Uniform balanced sampling is strictly superior in the presence of real-world remote sensing annotation noise.",
        bold_prefix="Answer to RQ4 (Sampling Dynamics vs. Ground-Truth Noise): "
    )

    # ----------------------------------------------------
    # 8.3 Current Limitations and Technical Bottlenecks
    # ----------------------------------------------------
    add_heading_2(doc, "8.3 Current Limitations and Technical Bottlenecks")
    add_paragraph(doc, 
        "While the current models have achieved verified breakthroughs exceeding official benchmark targets, several technical bottlenecks remain:",
        bold_prefix="Honest Critical Assessment: "
    )
    add_bullet(doc, 
        "Geographical Scarcity of Slums in Master-Planned Scenes: In the Cartosat-2E Pune scene, informal slums comprise only 0.05% of ground-truth pixels. While the model achieved 100% precision in preventing false slums, evaluating statistical recall on such small sample sizes (<25,000 pixels total) is statistically noisy. Robust slum benchmarking requires training on high-density informal settlements such as Dharavi or Govandi in Mumbai.",
        bold_prefix="1. Extreme Slum Class Scarcity: "
    )
    add_bullet(doc, 
        "Residual Concrete Roof vs. Dry Bare Soil Confusion: In arid sectors during dry seasons, unpaved bare soil and weathered concrete roofs share nearly identical spectral reflectance across both visible and Near-Infrared wavelengths (NDVI ≈ 0.05 to 0.12). Separating these classes requires deep contextual texture cues or thermal infrared radiometry.",
        bold_prefix="2. Arid Soil-Roof Ambiguity: "
    )
    add_bullet(doc, 
        "Off-Nadir Building Lean and Vector Offset Discrepancies: Satellite imagery acquired at oblique off-nadir angles (>15°) exhibits severe building perspective displacement (building facades leaning away from nadir). OpenStreetMap building footprints represent ground cadastres. This causes physical roofs to be offset from ground vectors by 2 to 5 meters, penalizing strict 0m pixel metrics despite visually correct predictions.",
        bold_prefix="3. Orthorectification Displacement: "
    )
    add_bullet(doc, 
        "Hardware VRAM Constraints on Edge GPUs: Operating within a 4 GB VRAM envelope prevented scaling the SegFormer encoder to MiT-B4 or MiT-B5, and constrained batch sizes to 1 with gradient accumulation.",
        bold_prefix="4. Hardware Ceilings: "
    )

    # ----------------------------------------------------
    # 8.4 Prioritized Roadmap for Semester VIII
    # ----------------------------------------------------
    add_heading_2(doc, "8.4 Prioritized Roadmap for Semester VIII")
    add_paragraph(doc, 
        "To conclude this B.Tech Project in the forthcoming final semester (Semester VIII, Spring 2027), a prioritized five-point research and engineering roadmap is established:",
        bold_prefix="Future Research Directions: "
    )
    add_bullet(doc, 
        "Graph-Theoretic Road Centerline Vectorization and APLS Evaluation: Extend raster road predictions into fully connected, routable topological vector graphs using iterative skeleton tracing. Benchmark network routability against OpenStreetMap graphs using the Average Path Length Similarity (APLS) metric.",
        bold_prefix="1. Topological Road Graph Extraction: "
    )
    add_bullet(doc, 
        "Contrastive Supervised Learning for Soil-Roof Disambiguation: Train a contrastive feature embedding head (SimCLR / SupCon) specifically conditioned on building texture versus open soil geometry to permanently eliminate residual concrete-soil confusion.",
        bold_prefix="2. Contrastive Texture Separation: "
    )
    add_bullet(doc, 
        "Dedicated Multi-City Informal Settlement Callset: Incorporate targeted sub-meter imagery over major informal settlement zones in Mumbai (Dharavi, Mankhurd) and Ahmedabad to train a high-recall, multi-city Slum Specialist head.",
        bold_prefix="3. Multi-City Slum Expansion: "
    )
    add_bullet(doc, 
        "Multi-Modal Optical and Synthetic Aperture Radar (SAR) Fusion: Fuse optical multispectral imagery with Sentinel-1 / NISAR dual-polarization (VV/VH) radar backscatter. Radar backscatter is sensitive to physical surface roughness and double-bounce corner reflection from buildings, completely piercing cloud cover and shadow corridors.",
        bold_prefix="4. Optical-SAR Fusion: "
    )
    add_bullet(doc, 
        "Edge TensorRT / ONNX Deployment in GeoVision: Quantize the best hybrid hierarchical model to INT8 / FP16 using NVIDIA TensorRT, enabling real-time sliding-window inference at >60 FPS on edge workstations and municipal GIS servers.",
        bold_prefix="5. TensorRT Optimization: "
    )
    
    doc.add_page_break()

