# High-Resolution Satellite Urban Feature Extraction Platform

This repository provides an end-to-end framework for urban feature extraction, hierarchical land-cover segmentation, and fine-grained built-up characterization across multi-sensor satellite platforms (SPOT-5, Pléiades, and high-resolution multispectral imagery).

Developed as part of the B.Tech Project (BTP-II) at the Indian Institute of Technology Bombay (IITB).

🌐 **Deployed Streamlit Application:** [https://mpupqnx7annqnv6ejotxsl.streamlit.app/](https://mpupqnx7annqnv6ejotxsl.streamlit.app/)

---

## 1. Web Application: Two Production Model Pipelines

The Streamlit web application (`app.py`) exposes **strictly TWO selectable production model options** in the user interface:

### Option 1: Vashi–Mumbai E4 Sensor-Adaptive U-Net (PRIMARY / PREFERRED)
* **Status:** Primary production model, selected by default in the UI.
* **Directory:** `models/05_vashi_mumbai_e4_multisensor/`
* **Architecture:** `HighResNDVIAdaptiveUNet` (Pretrained ResNet-18 Encoder + 5-Stage Multi-Scale Decoder + Native Resolution $512 \times 512$ Shallow Skips).
* **Sensors Supported & Sensor-Adaptive Stems:**
  1. **SPOT-5 Multispectral (4-Band):** Takes Bands [Red, Green, Blue, NIR], computes physics-based NDVI on-the-fly, and feeds a calibrated 5-channel tensor `[R, G, B, NIR, NDVI]`.
  2. **Pléiades High-Resolution RGB (3-Band):** Takes high-resolution RGB imagery normalized with ImageNet statistics.
* **Loss Function:** Calibrated Multi-Task Focal Cross-Entropy + Masked Soft Dice Loss with inverse-frequency class weighting.
* **Output Classes (5):** Non-Slum Built-Up, Road Network, Vegetation, Open Land / Soil, Slum / Informal Settlement.
* **Checkpoint:** `best_expE4_model.pth` ([Release v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0), SHA-256: `bbd658ed027cc9fe9369a892f09b09ff084436bb442ccac857b2c8049a66e4c0`).

---

### Option 2: Pune Multispectral SegFormer + Attention U-Net Hybrid
* **Status:** Second production model, exposed as ONE unified hierarchical choice in the UI.
* **Primary Directory:** `models/03_cartosat2e_hybrid_0.60m/` (self-contained with Stage-1 models and Stage-2 subclass module).
* **Dataset:** Trained and benchmarked on the Pune-area high-resolution multispectral dataset.
* **Two-Stage Collaborative Architecture:**
  * **Stage 1 (Macro Ensemble):** Triple probability ensemble blending three specialized backbones:
    - **SegFormer Extended** (Weight: 0.35)
    - **SegFormer Lovasz** (Weight: 0.35, IoU specialist)
    - **Attention UltraRoad U-Net** (Weight: 0.30, boundary and corridor detail)
    - **Physical Multispectral Safeguards:** Real-time NDVI and NDWI physical constraints applied to eliminate false water predictions in shadowed asphalt and confirm genuine vegetation chlorophyll absorption.
    - **Macro Classes (5):** Water, Vegetation, Barren Land / Soil, Built-Up, Road Network.
  * **Stage 2 (Subclass Specialist):** `MultispectralSegFormerSubclass` with spatial/channel squeeze-and-excitation (scSE) attention, applied strictly within the Stage-1 predicted Built-Up footprint.
    - **Subclasses (3):** Residential Built-Up, Industrial Built-Up, Slum / Informal Settlement.
* **Strict Input Validation:** Requires genuine 4-band multispectral imagery (Blue, Green, Red, Near-Infrared). Fake or synthesized NIR is strictly disallowed to ensure scientific rigor.
* **Output Classes (7 Unified):** Water, Vegetation, Barren Land / Soil, Residential Built-Up, Industrial Built-Up, Slum / Informal Settlement, Road Network.
* **Checkpoints:**
  - `best_5class_segformer_extended.pth` ([Release v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1), SHA-256: `b664b90bde7da335465db2ee27125fd62ff60eb11ac93670c224657df8670d77`)
  - `best_5class_segformer_lovasz.pth` ([Release v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1), SHA-256: `6c9bbd80a423b3d77f7bd20799f5a474973c5ce0603540b85c6301e5a112df50`)
  - `best_5class_unet_multispectral.pth` ([Release v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1), SHA-256: `a813fe6cd69adfb00164e4e0865ab10e3f972f5273bf0618524f6d81f59151e2`)
  - `best_3subclass_segformer.pth` ([Release v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0), SHA-256: `1ce752382d1c2b6758deca50508fe73f6b52f2dc8c54e39570728680de15fdf9`)

---

## 2. Repository Architecture & Model Organization

The repository strictly separates the **two production models** from historical research benchmarks and modular components:

```text
├── models/                           # MAIN PRODUCTION MODELS (Strictly 2 Choices)
│   ├── 05_vashi_mumbai_e4_multisensor/  # [PRODUCTION OPTION 1 - PREFERRED] Vashi–Mumbai E4 Sensor-Adaptive U-Net
│   └── 03_cartosat2e_hybrid_0.60m/      # [PRODUCTION OPTION 2] Pune Multispectral SegFormer + Attention U-Net Hybrid
│
└── other_models/                     # HISTORICAL BENCHMARKS & EXPERIMENTAL MODULES
    ├── 01_cartosat1_unet_2.5m/          # Monochromatic Cartosat-1 2.5m ResNet-34 U-Net (Semester 1 Baseline)
    ├── 02_cartosat1_segformer_2.5m/     # Monochromatic Cartosat-1 2.5m MiT-B2 SegFormer (Semester 1 Baseline)
    └── 04_cartosat2e_subclasses_0.60m/  # Standalone 3-Subclass Specialist Training & TTA Module
```

### Clean Separation Rationale:
1. **Primary Production Focus:** `models/` contains exclusively the two production pipelines exposed in the interactive Streamlit platform. This avoids ambiguity for users, reviewers, and evaluators.
2. **Preservation of Academic & Experimental Lineage in `other_models/`:**
   - Directories `01` and `02` preserve the Semester 1 baseline research benchmarks evaluating monochromatic panchromatic Cartosat-1 2.5m imagery for thesis reproduction and ablation analysis.
   - Directory `04` preserves the standalone subclass training notebooks (`02_SegFormer_Subclass_Training_and_TTA.ipynb`) and experimental setups.
3. **Self-Contained Deployment:** The hybrid pipeline in `models/03_cartosat2e_hybrid_0.60m/` embeds its own `models_subclass.py` and checkpoint bundle, ensuring fast and robust inference without dependencies on historical folders.

---

## 3. Running the Streamlit Web Application

### Local Launch
```bash
# Clone the repository
git clone https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II.git
cd Feature-Extraction-from-Satellite-Imagery-BTP-II

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

### Cloud Deployment (Streamlit Community Cloud)
The app is deployed at: **[https://mpupqnx7annqnv6ejotxsl.streamlit.app/](https://mpupqnx7annqnv6ejotxsl.streamlit.app/)**
- The application automatically verifies local checkpoints. If run in a cloud container without local weights, it retrieves the required weights from GitHub Releases on-demand and validates their SHA-256 checksums before loading into memory.
- Comes preloaded with verified demo samples in `demo_samples/` for instantaneous one-click testing of both models.

---

## 4. Model Checkpoint Organization & Git LFS Storage

All five production model checkpoints are organized directly within their respective model directories and tracked via **Git LFS (Large File Storage)**. GitHub Releases (v1.0.0 and v1.0.1) remain permanently accessible as backup and cloud auto-download fallback endpoints.

### Production Checkpoint Directory Structure
```
models/
├── 05_vashi_mumbai_e4_multisensor/
│   └── checkpoints/
│       └── best_expE4_model.pth                  # Model 1 (E4 Multisensor Sensor-Adaptive U-Net)
│
└── 03_cartosat2e_hybrid_0.60m/
    ├── checkpoints/                              # Model 2 (Active Production Components)
    │   ├── best_5class_segformer_extended.pth    # Stage 1 Macro Ensemble (Weight: 0.35)
    │   ├── best_5class_segformer_lovasz.pth      # Stage 1 Macro Ensemble (Weight: 0.35)
    │   ├── best_5class_unet_multispectral.pth    # Stage 1 Macro Ensemble (Weight: 0.30)
    │   └── best_3subclass_segformer.pth          # Stage 2 Subclass Specialist (Gated in Built-Up)
    └── reference_checkpoints/
        ├── best_hybrid_hierarchical_model.pth    # Monolithic Combined Architecture (Epoch 6 Reference)
        └── README.md                             # Architectural Reference Documentation
```

### Production Model Checkpoint Inventory
| Checkpoint Path in Repository | Size | SHA-256 Checksum | Pipeline Role | Release Fallback |
| :--- | :---: | :---: | :--- | :---: |
| `models/05_vashi_mumbai_e4_multisensor/checkpoints/best_expE4_model.pth` | 55.33 MB | `bbd658ed027cc9fe9369a892f09b09ff084436bb442ccac857b2c8049a66e4c0` | **Model 1 (Primary):** E4 Sensor-Adaptive U-Net (SPOT 5-ch / Pléiades RGB) | [v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0) |
| `models/03_cartosat2e_hybrid_0.60m/checkpoints/best_5class_segformer_extended.pth` | 104.63 MB | `b664b90bde7da335465db2ee27125fd62ff60eb11ac93670c224657df8670d77` | **Model 2:** Stage 1 Macro SegFormer Extended (Weight 0.35) | [v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1) |
| `models/03_cartosat2e_hybrid_0.60m/checkpoints/best_5class_segformer_lovasz.pth` | 104.63 MB | `6c9bbd80a423b3d77f7bd20799f5a474973c5ce0603540b85c6301e5a112df50` | **Model 2:** Stage 1 Macro SegFormer Lovász (Weight 0.35) | [v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1) |
| `models/03_cartosat2e_hybrid_0.60m/checkpoints/best_5class_unet_multispectral.pth` | 102.24 MB | `a813fe6cd69adfb00164e4e0865ab10e3f972f5273bf0618524f6d81f59151e2` | **Model 2:** Stage 1 Macro UltraRoad ResNet-34 U-Net (Weight 0.30) | [v1.0.1](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.1) |
| `models/03_cartosat2e_hybrid_0.60m/checkpoints/best_3subclass_segformer.pth` | 104.62 MB | `1ce752382d1c2b6758deca50508fe73f6b52f2dc8c54e39570728680de15fdf9` | **Model 2:** Stage 2 Subclass Specialist (Residential, Industrial, Slum) | [v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0) |

### Why Model 2 (Hybrid) Uses Four Separate Components
The Pune–Chennai 7-class hybrid pipeline uses a two-stage hierarchical architecture:
1. **Stage 1 (Macro 5-Class Segmentation):** Blends global semantic transformer representations (SegFormer Extended and Lovász variants) with fine-grained road edge representations (UltraRoad ResNet-34 U-Net) using calibrated probability weights ($0.35 + 0.35 + 0.30$). Splitting the macro ensemble into three components allows independent 8-way dihedral test-time augmentation (TTA), modular benchmarking, and reduced peak GPU memory.
2. **Stage 2 (Built-Up Subclass Specialist):** Focuses specifically on separating built-up pixels into Residential, Industrial, and Informal Settlement / Slum classes. It is applied conditionally as a spatial gate over the macro built-up prediction, avoiding false positive subclass predictions in agricultural or barren regions.

### Reference Monolithic Hybrid Checkpoint
* `models/03_cartosat2e_hybrid_0.60m/reference_checkpoints/best_hybrid_hierarchical_model.pth` (521.0 MB, SHA-256: `db15c31848f40b24ce22182bd8a09681a8fc3e1c6c35643245e074e428fa0a53`): End-to-end trained model (Epoch 6) containing `alpha_macro`, full encoder-decoder, and subclass head state dictionaries. Preserved locally for training reproducibility and downloadable from [Release v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0).

### Git LFS Setup & How to Retrieve Model Weights
To clone the repository and retrieve the binary model checkpoints:
```bash
# 1. Install Git LFS (if not already installed)
git lfs install

# 2. Clone the repository
git clone https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II.git
cd Feature-Extraction-from-Satellite-Imagery-BTP-II

# 3. Pull all Git LFS binary files
git lfs pull
```

> **Automated Cloud Fallback:** If the repository is cloned in an environment without Git LFS (or where pointer text files are received), the Streamlit loader in `app.py` automatically detects pointer files, falls back to GitHub Releases v1.0.0 and v1.0.1, downloads the full binary weights, and validates their SHA-256 hashes before loading.

---

## 5. Offline Evaluation & Benchmark Scripts

* **Vashi–Mumbai E4 Multi-Sensor Evaluation:**
  ```bash
  python models/05_vashi_mumbai_e4_multisensor/scripts/train_experiment_E4.py --evaluate
  ```
* **Pune Multispectral Hierarchical 7-Class Benchmark:**
  ```bash
  python models/03_cartosat2e_hybrid_0.60m/evaluate_hierarchical_benchmark_90plus.py
  ```
* **Comprehensive Integration Pipeline Test (Self-Check):**
  ```bash
  python scratch/test_models.py
  ```

---

## 6. Project Deliverables & Reports

* **Final BTP Report:** [`BTP_Report_23B0653.docx`](BTP_Report_23B0653.docx) (also mirrored in [`docs/reports/BTP_Report_23B0653.docx`](docs/reports/BTP_Report_23B0653.docx)) — Comprehensive 8-chapter thesis entitled *"Urban Feature Extraction from High-Resolution Satellite Imagery: Multi-Resolution Land-Cover Segmentation and Hierarchical Urban Feature Mapping"*.
* **Final Presentation Deck:** [`docs/presentations/Urban_Feature_Extraction_Progress_Presentation_Final.pptx`](docs/presentations/Urban_Feature_Extraction_Progress_Presentation_Final.pptx).
* **Sample GIS Deliverables:** Located in `outputs/sample_gis/` (Vectorized Shapefiles for direct loading in QGIS/ArcGIS).

---

## 7. Authors & Acknowledgments

* **Author:** Prabhakar Yadav (Roll No. 23B0653), Department of Civil Engineering, Indian Institute of Technology Bombay
* **Supervisor:** Prof. Gopal Patil, Department of Civil Engineering, Indian Institute of Technology Bombay
