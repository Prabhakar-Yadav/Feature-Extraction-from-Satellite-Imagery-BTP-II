# Urban Feature Extraction — Visualizations Catalog

This directory contains all authentic visual predictions, multi-panel diagnostic comparisons, architecture flowcharts, and interactive dashboards generated across both production models developed for the B.Tech Project (BTP-II) at the Department of Civil Engineering, IIT Bombay.

---

## Folder Structure

```
visualizations/
├── 01_model1_vashi_mumbai/                    # Model 1: Vashi–Mumbai Sensor-Adaptive U-Net (SPOT 1.5m + Pléiades 0.5m)
│   ├── fig7_6_vashi_0041_e1_vs_e4.jpg         # [BEST RESULT] Commercial patch resolution (Report Figure 7.6)
│   ├── fig7_5_vashi_confusion_matrix.png      # Normalized 5-class confusion matrix (Report Figure 7.5)
│   ├── fig7_7_vashi_0003_dense_res.jpg        # Dense residential sector evaluation (Report Figure 7.7)
│   ├── chart_e1_vs_e4.png                     # Quantitative ablation comparison chart
│   ├── simple_model1_architecture.png         # HighResNDVIAdaptiveUNet architecture diagram
│   ├── 8panel_comparison_vashi_0003.png       # 8-panel diagnostic on residential sector
│   ├── 8panel_comparison_vashi_0020.png       # 8-panel diagnostic on vegetation & water boundary
│   ├── 8panel_comparison_vashi_0041.png       # 8-panel diagnostic on commercial sector
│   └── sector2_comparisons/                   # High-res Sector 2 multi-panel comparisons (vashi_0052 to vashi_0064)
│
├── 02_model2_pune_cartosat2e/                 # Model 2: Pune Hierarchical Hybrid (ISRO Cartosat-2E 0.60m)
│   ├── fig7_3_pune_0413_seven_class.jpg       # [BEST RESULT] 7-class thematic map with overlay (Report Figure 7.3)
│   ├── fig7_4_pune_0482_residential.jpg       # [BEST RESULT] Formal residential subclass delineation (Report Figure 7.4)
│   ├── chart_model2_benchmark.png             # 186 held-out test patches quantitative performance chart
│   ├── simple_model2_hierarchical.png         # Two-tier hierarchical spatial gating flowchart
│   ├── ground_truth_fix_11panel_test_pune_0585.png  # 11-panel full diagnostic verification
│   ├── seven_panel_water_diagnostic_test_pune_0585.png # 7-panel asphalt-safe water diagnostic
│   ├── subclass_overlays/                     # Visual prediction overlays for each subclass
│   │   ├── subclass_overlay_0438_industrial.png
│   │   ├── subclass_overlay_0462_slum.png
│   │   ├── subclass_overlay_0482_residential.png
│   │   ├── subclass_overlay_0563_triclass.png
│   │   └── subclass_overlay_0584_mixed.png
│   └── top_predictions/                       # Top 10 benchmark test predictions (IoU-ranked)
│       ├── top_01_test_pune_0413_visual_iou.png
│       ├── top_02_test_pune_0562_visual_iou.png
│       ├── top_03_test_pune_0459_visual_iou.png
│       └── ...
│
├── 03_system_architecture_and_deployment/     # Overall framework & GIS production pipeline
│   ├── simple_dual_pipeline_flow.png          # Dual-track operational framework diagram (Slide 5)
│   ├── simple_gis_deployment.png              # Tiled sliding window & GIS export flowchart (Slide 13)
│   ├── dual_pipeline_flow.png                 # Detailed dual-track architectural flow
│   └── gis_pipeline_slide16.png               # Detailed GIS deployment diagram
│
└── 04_interactive_dashboards/                 # Standalone interactive web inspection tools
    ├── inspection_dashboard.html              # Multi-batch test patch inspection dashboard
    └── builtup_reannotation_portal.html       # Built-up boundary review and validation portal
```

---

## Detailed Model Visualizations Catalog

### 1. Model 1 — Vashi–Mumbai Sensor-Adaptive U-Net
* **Primary Sensor Data:** SPOT-5/6 (1.50m, 4 multispectral bands) and Pléiades-1A (0.50m, 3-band RGB).
* **Taxonomy:** 5 Macro Classes (Road Network, Non-Slum Buildings, Informal Settlements/Slums, Water Bodies, Background).
* **Core Visual Highlight:**
  * **`fig7_6_vashi_0041_e1_vs_e4.jpg`:** Resolves severe commercial roof glare. While baseline E1 erroneously classified reflective warehouse roofs as wide asphalt roads (inflating road area to 46.11%), checkpoint E4 corrects predicted road area to **31.70%** (aligning with ground truth 29.32%) and improves building IoU from **61.54% to 67.11%** (+5.57% gain).
  * **`fig7_5_vashi_confusion_matrix.png`:** Confirms clean separation between water, vegetation, and roads, with remaining errors confined strictly to building-road borders.

### 2. Model 2 — Pune Cartosat-2E Hierarchical Hybrid
* **Primary Sensor Data:** ISRO Cartosat-2E (0.60m pan-sharpened multispectral: Blue, Green, Red, Near-Infrared).
* **Taxonomy:** Two-Level Hierarchy (Level 1: 5 Macro Classes; Level 2: Residential, Industrial, and Informal Settlements).
* **Core Visual Highlights:**
  * **`fig7_3_pune_0413_seven_class.jpg`:** Shows the 4-panel progression on test patch `test_pune_0413` (Original 0.60m Satellite Input, Reference Ground Truth, 7-Class Prediction, and Prediction Overlay).
  * **`fig7_4_pune_0482_residential.jpg`:** Shows clean formal residential building envelope extraction achieving **73.22% strict IoU** (80.15% buffer IoU).
  * **`seven_panel_water_diagnostic_test_pune_0585.png`:** Demonstrates the asphalt-safe physical water guard (NDWI + NIR reflectance thresholding) suppressing false-positive water predictions on dark asphalt roads and steep building shadows.

### 3. Interactive Web Tools
* **Streamlit Web Application (`app.py` in root):**
  Run `streamlit run app.py` to launch the interactive browser interface for model inference, side-by-side swipe overlays, and automated municipal land-use reports.
* **Inspection Dashboard (`04_interactive_dashboards/inspection_dashboard.html`):**
  Open directly in any modern browser to visually inspect human verification batches, error breakdowns, and spatial overlays.
