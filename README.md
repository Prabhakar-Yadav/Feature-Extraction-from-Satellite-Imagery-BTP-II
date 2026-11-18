# High-Resolution Satellite Urban Feature Extraction Platform

This repository provides an end-to-end framework for urban feature extraction, hierarchical land-cover segmentation, and fine-grained built-up characterization across multi-resolution satellite platforms (Cartosat-1 2.5m, Cartosat-2E 0.60m, SPOT-5, and Pléiades).

Developed as part of the B.Tech Project (BTP-II) at the Indian Institute of Technology Bombay (IITB).

---

## 1. The Three Primary Model Pipelines

The project centers around three production-grade model architectures evaluated across panchromatic and multispectral satellite imagery:

```
models/
├── 01_cartosat1_unet_2.5m/          # Model 1: Macro 5-class U-Net baseline (2.5m panchromatic)
├── 02_cartosat1_segformer_2.5m/     # Model 2: Transformer-based SegFormer benchmark (2.5m panchromatic)
├── 03_cartosat2e_hybrid_0.60m/      # Model 3: Hierarchical Dual-Stream Hybrid (0.60m multispectral)
├── 04_cartosat2e_subclasses_0.60m/  # Subclass specialist (Residential, Industrial, Slum)
└── 05_vashi_mumbai_e4_multisensor/  # Dual-stem multisensor model (SPOT-5 4-band + Pléiades RGB)
```

### Model Architecture Highlights

| Model | Ground Resolution | Spectral Bands | Architecture / Backbone | Output Classes | Primary Application |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **Model 1: Cartosat-1 U-Net** | 2.5m | 1 Band (Panchromatic, 3-ch) | `RobustUltraRoadUNet` (ResNet-34 + ASPP) | 5 Macro Classes | Rapid regional baseline land-cover mapping |
| **Model 2: Cartosat-1 SegFormer** | 2.5m | 1 Band (Panchromatic, 3-ch) | `RoadEnhancedSegFormer` (MiT-B2 Transformer) | 5 Macro Classes | High-precision road corridor & built-up delineation |
| **Model 3: Cartosat-2E Hybrid** | 0.60m | 4 Bands (VNIR: R, G, B, NIR) | Dual-Stream Hierarchical (Macro + Subclasses) | 7 Unified Urban Classes | Complete urban characterization & slum mapping |

### Hierarchical Coupling (Model 3 & Model 4)
* **Macro Stream (`MultispectralSegFormer5Class`):** Delineates broad land-cover categories (Background, Road, Vegetation, Water, Built-Up). The predicted built-up polygon forms a spatial gating mask.
* **Subclass Specialist (`MultispectralSegFormerSubclass`):** Focuses exclusively on the gated built-up pixels using Edge-Guided SegFormer-B2 with spatial/channel squeeze-and-excitation (scSE) attention to categorize urban fabric into:
  1. **Residential Built-Up** (formal organized housing layouts)
  2. **Industrial Built-Up** (large contiguous shed footprints, warehousing)
  3. **Slum / Informal Settlements** (high-density, irregular organic morphologies)
* **Unified 7-Class Output:** Seamlessly merges macro features and fine-grained subclasses into a single georeferenced raster and GIS vector layer.

---

## 2. Directory Structure

```text
├── models/                       # Self-contained model families
│   ├── 01_cartosat1_unet_2.5m/   # Model 1 architecture, dataset loader, eval script & config
│   ├── 02_cartosat1_segformer_2.5m/ # Model 2 architecture, dataset loader, eval script & config
│   ├── 03_cartosat2e_hybrid_0.60m/ # Model 3 hierarchical hybrid model & benchmark evaluators
│   ├── 04_cartosat2e_subclasses_0.60m/ # Model 4 standalone 3-subclass specialist
│   └── 05_vashi_mumbai_e4_multisensor/ # Model 5 dual-stem NDVI adaptive network
├── data/                         # Centralized dataset storage
│   ├── manifests/                # Verified CSV manifests (v4_dataset_manifest, autonomous_manifest)
│   ├── raw/                      # Raw satellite scenes (Cartosat, SPOT) [Local / Excluded from Git]
│   ├── processed/                # Harmonized tile arrays [Local / Excluded from Git]
│   └── ground_truth/             # Reference raster masks [Local / Excluded from Git]
├── outputs/                      # Deliverables and benchmark results
│   ├── sample_gis/               # 5 Complete demo GIS shapefile layers (.shp, .shx, .dbf, .prj, .cpg)
│   ├── shapefiles/               # Master full-scene GIS database (15 layers) [Local]
│   └── visual_comparisons/       # Multi-panel prediction mosaics and IoU overlays
├── docs/                         # Documentation, thesis report, and presentation
│   ├── reports/                  # BTP Final Thesis (23b0653.docx)
│   ├── presentations/            # Presentation slide deck (PPTX) & visual assets
│   └── report_builder/           # Automated Python DOCX thesis compilation suite
├── scripts/                      # Preprocessing, GIS vectorization, and data generation utilities
├── shared/                       # Shared evaluation metrics (SpaceNet buffer IoU, Mnih relaxed metrics)
├── requirements.txt              # Complete Python dependency specification
└── .gitignore                    # Watertight exclusion rules for large datasets and binaries
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+ (Recommended: Python 3.11 or 3.12)
* NVIDIA GPU with CUDA 11.8+ or 12.x (Inference also supported on CPU)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II.git
cd Feature-Extraction-from-Satellite-Imagery-BTP-II

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. How to Use: Training, Inference & Evaluation

### 4.1 Model 1: Cartosat-1 2.5m U-Net
* **Evaluate on Test Benchmark:**
  ```bash
  python models/01_cartosat1_unet_2.5m/evaluate.py --config models/01_cartosat1_unet_2.5m/config.yaml
  ```
* **Run Training Pipeline:**
  ```bash
  python models/01_cartosat1_unet_2.5m/dataset.py
  ```

### 4.2 Model 2: Cartosat-1 2.5m SegFormer
* **Evaluate SegFormer Benchmark:**
  ```bash
  python models/02_cartosat1_segformer_2.5m/evaluate.py --config models/02_cartosat1_segformer_2.5m/config.yaml
  ```

### 4.3 Model 3: Cartosat-2E 0.60m Hierarchical Hybrid (Flagship)
* **Run Full Hierarchical 7-Class Benchmark:**
  ```bash
  python models/03_cartosat2e_hybrid_0.60m/evaluate_hierarchical_benchmark_90plus.py
  ```
* **Verify Macro & Subclass Forward Passes (Self-Test):**
  ```bash
  python scratch/test_all_five_pipelines_v3.py
  ```

### 4.4 Inspecting GIS Deliverables
Demo GIS shapefiles are available in `outputs/sample_gis/` and can be loaded directly into **QGIS** or **ArcGIS**:
* `all_7class_urban_features.shp` — Complete 7-class urban feature map
* `residential_parcels.shp` — Formal housing boundary polygons
* `industrial_parcels.shp` — Industrial complex footprint polygons
* `slum_settlements.shp` — Delineated informal settlement boundaries
* `road_network.shp` — Vectorized urban transport corridor centerlines

---

## 5. Model Checkpoints & Large Data Hosting

GitHub enforces a strict **100 MB per-file limit** and recommends keeping repositories under **1–5 GB**. Consequently:
* **The Git repository** contains the complete source code, architectures, configuration files, dataset manifests, documentation, reports, presentation decks, and lightweight sample GIS vector layers.
* **Trained Weights (`.pth` files):** Checkpoints range from 104 MB to 521 MB and are hosted via **[GitHub Releases](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases)**:
  - `best_unet.pth` (314 MB) $\to$ Place in `models/01_cartosat1_unet_2.5m/`
  - `best_segformer.pth` (313.9 MB) $\to$ Place in `models/02_cartosat1_segformer_2.5m/`
  - `best_hybrid_hierarchical_model.pth` (521 MB) $\to$ Place in `models/03_cartosat2e_hybrid_0.60m/`
  - `best_3subclass_segformer.pth` (104.6 MB) $\to$ Place in `models/04_cartosat2e_subclasses_0.60m/`
  - `best_expE4_model.pth` (55.3 MB) $\to$ Place in `models/05_vashi_mumbai_e4_multisensor/checkpoints/`
* **Full Satellite Datasets (13.0 GB):** Raw Cartosat-1, Cartosat-2E, and SPOT-5 scenes are available upon request or via institutional cloud storage as governed by ISRO RSDP data dissemination guidelines.

---

## 6. Project Deliverables & Reports

* **Final BTP Report:** [`docs/reports/23b0653.docx`](docs/reports/23b0653.docx) (Comprehensive 8-chapter thesis with complete mathematical formulations, loss functions, and benchmark tables).
* **Final Presentation:** [`docs/presentations/Urban_Feature_Extraction_Progress_Presentation_Final.pptx`](docs/presentations/Urban_Feature_Extraction_Progress_Presentation_Final.pptx).
* **Benchmark IoU Visual Overlays:** Located in `outputs/visual_comparisons/`.

---

## 7. Authors & Acknowledgments

* **Author:** Prabhakar Yadav (Roll No. 23B0653), IIT Bombay
* **Advisor:** Prof. Biplab Banerjee, Centre of Studies in Resources Engineering (CSRE), IIT Bombay
* **Sponsoring Body:** Indian Space Research Organisation (ISRO) & CSRE / KReSIT, IIT Bombay
