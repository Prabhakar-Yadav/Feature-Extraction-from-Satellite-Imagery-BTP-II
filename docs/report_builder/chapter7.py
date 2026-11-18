"""
scripts/report_builder/chapter7.py
----------------------------------
Chapter 7: Implementation and System Workflow
- 7.1 Software Repository Architecture
- 7.2 Training Engine and Memory Optimization
- 7.3 Large-Scale Sliding-Window Inference with 2D Hann Blending
- 7.4 Georeferenced Vector and Raster Export (GeoTIFF, Shapefiles, GeoJSON)
- 7.5 GIS Integration and Validation in QGIS
- 7.6 Computational Efficiency, Runtime Analysis, and VRAM Profiling
"""

import docx
from docx.shared import Inches, Pt
import styles
from styles import (
    add_heading_1, add_heading_2, add_heading_3,
    add_paragraph, add_bullet, add_callout, add_styled_table, add_equation
)
from omml_equations import add_display_equation, OMML_HANN

def add_chapter_7(doc):
    add_heading_1(doc, "Chapter 7: Implementation and System Workflow")
    
    # ----------------------------------------------------
    # 7.1 Software Repository Architecture
    # ----------------------------------------------------
    add_heading_2(doc, "7.1 Software Repository Architecture")
    add_paragraph(doc, 
        "To ensure end-to-end scientific reproducibility and support practical deployment, the project codebase is organized into modular, production-ready Python modules. Rather than relying on unstructured, ephemeral interactive notebooks, all training pipelines, dataset loaders, loss functions, evaluation harnesses, and GIS export routines are encapsulated as standalone, version-controlled scripts with deterministic command-line entry points.",
        bold_prefix="Modular Design: "
    )
    add_bullet(doc, 
        "FINAL_ORGANIZED_SUBMISSION/: Contains the clean, audited final delivery assets, organized into 2_Models/ (serialized PyTorch checkpoints for 2.5m and 0.60m models), 3_Outputs/ (Shapefiles, GeoJSONs, and high-resolution visual PNG results), and 4_Reports/ (comprehensive benchmark Markdown reports and CSV logs).",
        bold_prefix="1. Final Submission Repository: "
    )
    add_bullet(doc, 
        "NEW_Model(sem2)/: Houses the complete Phase 1 Data Preparation, Phase 2 Multi-Model Training (SegFormer, U-Net, 5-Class Multispectral, and 3-Subclass Morphology Heads), and human-in-the-loop active learning utilities.",
        bold_prefix="2. Core Research Engine: "
    )
    add_bullet(doc, 
        "VASHI_IMPROVED/: Contains the multisensor Vashi SPOT and Mumbai Pléiades codebase, tracking controlled experimental versions (research_corrected_v1 through v4 / E1–E4) with dedicated audits, checkpoint stores, and evaluation scripts.",
        bold_prefix="3. Multisensor Progression: "
    )
    add_bullet(doc, 
        "web_platform/ & WebPlatform/: Production Streamlit application ('GeoVision') featuring dynamic model selection, interactive confidence heatmaps, raster-to-vector polygonization, and multi-format geospatial exports.",
        bold_prefix="4. Interactive Web Platform: "
    )

    # ----------------------------------------------------
    # 7.2 Training Engine and Memory Optimization
    # ----------------------------------------------------
    add_heading_2(doc, "7.2 Training Engine and Memory Optimization")
    add_paragraph(doc, 
        "The training engine leverages PyTorch 2.x with CUDA 12.x hardware acceleration. To operate strictly within the 4.0 GB physical VRAM capacity of the NVIDIA RTX 3050 Laptop GPU, four core memory optimization techniques were integrated:",
        bold_prefix="VRAM Governance: "
    )
    add_bullet(doc, 
        "Automatic Mixed Precision (torch.cuda.amp): Computes forward passes and activations in half-precision floating point (FP16), cutting tensor memory consumption by ~50% while utilizing GradScaler to prevent gradient underflow during backward passes.",
        bold_prefix="1. Mixed Precision (FP16): "
    )
    add_bullet(doc, 
        "In-Epoch Decoupled Backward Passes: For the hybrid hierarchical model, the macro loss backward pass is executed immediately to free the activation graphs of SegFormer-B2 and U-Net before the subclass morphology head is invoked.",
        bold_prefix="2. Decoupled Graph Execution: "
    )
    add_bullet(doc, 
        "Gradient Accumulation: Micro-batches of size 1 are processed sequentially and gradients accumulated over 4 steps before invoking optimizer.step(), achieving the mathematical stability of a batch size of 4 without VRAM spikes.",
        bold_prefix="3. Gradient Accumulation: "
    )
    add_bullet(doc, 
        "Explicit Garbage Collection and Cache Clearing: Manual calls to gc.collect() and torch.cuda.empty_cache() are scheduled at epoch boundaries to eliminate PyTorch caching allocator fragmentation.",
        bold_prefix="4. Active Memory Hygiene: "
    )

    # ----------------------------------------------------
    # 7.3 Large-Scale Sliding-Window Inference with 2D Hann Blending
    # ----------------------------------------------------
    add_heading_2(doc, "7.3 Large-Scale Sliding-Window Inference with 2D Hann Blending")
    add_paragraph(doc, 
        "Satellite scenes frequently span dimensions exceeding 10,000 × 10,000 pixels (>100 megapixels). Ingesting an entire multi-gigabyte scene into a neural network in a single forward pass is mathematically impossible due to memory limits. Full scenes must be sliced into regular patches (e.g., 512×512) and processed via a sliding window.",
        bold_prefix="The Tiling Challenge: "
    )
    add_paragraph(doc, 
        "However, simple rectangular tile stitching produces severe visual edge discontinuities and prediction artifacts along tile borders because convolutional kernels near patch boundaries lack spatial context. To produce seamless full-scene classified maps, we implemented a 2D Hann Window Blending scheme. Each patch is multiplied by a smooth 2D Hann bell-curve weighting window prior to accumulation according to Equation (7.1):",
        bold_prefix="Hann Window Formulation: "
    )
    add_display_equation(doc, OMML_HANN, "7.1")
    add_paragraph(doc, 
        "where x in [0, W-1] and y in [0, H-1]. Patches are extracted with a 50% stride overlap (step size = 256 pixels). During accumulation, predicted class probability vectors are weighted by w(x, y) and summed into a global accumulator tensor, while a parallel normalization tensor tracks accumulated weights. Dividing accumulated probabilities by accumulated weights ensures that patch boundaries seamlessly cross-fade, completely eliminating grid seams across full metropolitan scenes.",
        bold_prefix="Seamless Cross-Fading: "
    )

    # ----------------------------------------------------
    # 7.4 Georeferenced Vector and Raster Export
    # ----------------------------------------------------
    add_heading_2(doc, "7.4 Georeferenced Vector and Raster Export (GeoTIFF, Shapefiles, GeoJSON)")
    add_paragraph(doc, 
        "For deep learning predictions to be operationally useful to civil engineers, urban planners, and GIS professionals, outputs must be delivered in standard, georeferenced GIS formats. We engineered an automated post-processing and export pipeline using Rasterio, GeoPandas, and Shapely:",
        bold_prefix="GIS Export Pipeline: "
    )
    add_bullet(doc, 
        "Georeferenced GeoTIFF Export: The classified integer label raster is encoded with the original geospatial affine transformation matrix and projected Coordinate Reference System (e.g., EPSG:32643 / UTM Zone 43N). Color tables are embedded directly into GeoTIFF headers for standard multi-band GIS rendering.",
        bold_prefix="1. GeoTIFF Rasters: "
    )
    add_bullet(doc, 
        "Vector Polygonization: Continuous class patches are polygonized class-by-class using rasterio.features.shapes. Each extracted polygon is attributed with its semantic class_id, class_name, hex_color, and physical ground area in square meters.",
        bold_prefix="2. Feature Polygonization: "
    )
    add_bullet(doc, 
        "Geometric Cleaning and Noise Filtering: High-frequency single-pixel raster noise is eliminated by filtering out small polygon fragments (<100 m² for natural classes; <50 m² for buildings). Furthermore, polygon contours are simplified using the Douglas-Peucker algorithm with a 1-pixel tolerance to produce clean, compact vector geometries.",
        bold_prefix="3. Vector Topology Cleaning: "
    )
    add_bullet(doc, 
        "Multi-Format Packaging: Cleaned vector layers are packaged simultaneously as zipped ESRI Shapefiles (with .shp, .shx, .dbf, and .prj sidecars for desktop GIS tools like QGIS and ArcGIS) and RFC 7946 GeoJSON files for web mapping applications.",
        bold_prefix="4. Open Standard Packaging: "
    )

    # ----------------------------------------------------
    # 7.5 GIS Integration and Validation in QGIS
    # ----------------------------------------------------
    add_heading_2(doc, "7.5 GIS Integration and Validation in QGIS")
    add_paragraph(doc, 
        "All exported Shapefiles and GeoTIFFs were validated by directly importing them into QGIS (Version 3.28 LTR). Vector building polygons, road centerlines, and water boundaries were overlaid against Google Satellite high-resolution base maps. Spatial overlay checks confirmed that exported geometries exhibited zero spatial coordinate drift, matching physical road curbs and residential parcel cadastres with sub-meter alignment.",
        bold_prefix="Cadastral Verification: "
    )

    # ----------------------------------------------------
    # 7.6 Computational Efficiency, Runtime Analysis, and VRAM Profiling
    # ----------------------------------------------------
    add_heading_2(doc, "7.6 Computational Efficiency, Runtime Analysis, and VRAM Profiling")
    add_paragraph(doc, 
        "Table 7.1 summarizes computational throughput, single-patch inference latency, full-scene runtime, and peak VRAM consumption across all deployed model architectures on the RTX 3050 Laptop GPU.",
        bold_prefix="Performance Benchmarks: "
    )

    p_t71 = doc.add_paragraph()
    p_t71.paragraph_format.space_before = Pt(8)
    p_t71.paragraph_format.space_after = Pt(2)
    p_t71.paragraph_format.keep_with_next = True
    r_t71 = p_t71.add_run("Table 7.1: Inference throughput, latency per patch, and full-scene runtime benchmarks.")
    r_t71.font.name = "Times New Roman"
    r_t71.font.size = Pt(9.5)
    r_t71.font.bold = True
    r_t71.font.color.rgb = styles.COLOR_NAVY_PRIMARY

    perf_data = [
        ["SegFormer-B2 (2.5m)", "1 Band (PAN)", "24.7 M", "38.5 ms", "26.0 FPS", "1.12 GB", "42.8 seconds", "Fastest transformer inference"],
        ["UltraRoad U-Net (2.5m)", "1 Band (PAN)", "21.5 M", "32.1 ms", "31.2 FPS", "0.85 GB", "35.6 seconds", "Lowest VRAM footprint; fast CNN"],
        ["Unified Hybrid Model (0.60m)", "4 Bands (VNIR)", "46.2 M", "74.2 ms", "13.5 FPS", "1.74 GB", "84.5 seconds", "Dual-stream forward pass; 7 classes"],
        ["Vashi ResNet-18 (E1)", "4 Bands (VNIR)", "14.3 M", "22.4 ms", "44.6 FPS", "0.62 GB", "24.8 seconds", "Lightweight edge deployment"],
        ["Vashi NDVI Model (E4)", "5 Bands (VNIR+NDVI)", "14.3 M", "24.1 ms", "41.5 FPS", "0.64 GB", "26.5 seconds", "Near-zero overhead for NDVI channel"]
    ]
    add_styled_table(doc, 
        ["Model Architecture", "Input Modality", "Parameters", "Latency (Patch)", "Throughput (FPS)", "Peak VRAM", "Full Scene Time (100 MPx)", "Operational Assessment"], 
        perf_data, 
        col_widths=[1.5, 1.0, 0.8, 0.9, 1.0, 0.8, 1.2, 1.3], 
        alignment=['L', 'L', 'R', 'R', 'R', 'R', 'R', 'L']
    )
    
    doc.add_page_break()
