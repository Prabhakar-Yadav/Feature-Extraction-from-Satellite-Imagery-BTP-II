"""
High-Resolution Satellite Urban Feature Extraction Platform
============================================================
Streamlit Web Application for BTP-II (IIT Bombay)

Exposes exactly TWO production-grade models:
1. Vashi-Mumbai E4 Sensor-Adaptive U-Net (PRIMARY / PREFERRED)
2. Multispectral SegFormer-U-Net Hybrid (Hierarchical Macro + Subclasses)
"""

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Urban Feature Extraction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-preferred {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .badge-hierarchical {
        background-color: #3B82F6;
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .model-card {
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        background-color: #F8FAFC;
        margin-bottom: 1rem;
    }
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 8px;
        border-radius: 3px;
        vertical-align: middle;
        border: 1px solid #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

ROOT_DIR = Path(__file__).resolve().parent
REPO_RELEASE_V100 = "https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/download/v1.0.0"
REPO_RELEASE_V101 = "https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/download/v1.0.1"

# Add model subdirectories to system path
sys.path.insert(0, str(ROOT_DIR / "models" / "05_vashi_mumbai_e4_multisensor" / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "models" / "03_cartosat2e_hybrid_0.60m"))
sys.path.insert(0, str(ROOT_DIR / "models" / "04_cartosat2e_subclasses_0.60m"))

# -----------------------------------------------------------------------------
# 2. Checkpoint Auto-Downloader for Streamlit Cloud
# -----------------------------------------------------------------------------
def ensure_checkpoint(filename: str, candidate_paths: list, release_tag: str = "v1.0.0") -> Path:
    """Checks local candidate paths; if missing, downloads from GitHub Releases."""
    for p in candidate_paths:
        path_obj = Path(p)
        if path_obj.exists() and path_obj.stat().st_size > 1000:
            return path_obj

    # Destination in models folder
    dst = Path(candidate_paths[0])
    dst.parent.mkdir(parents=True, exist_ok=True)

    base_url = REPO_RELEASE_V101 if release_tag == "v1.0.1" else REPO_RELEASE_V100
    url = f"{base_url}/{filename}"

    with st.spinner(f"Downloading checkpoint '{filename}' from GitHub Releases ({release_tag})..."):
        try:
            urllib.request.urlretrieve(url, str(dst))
            st.success(f"Successfully downloaded {filename}!")
            return dst
        except Exception as e:
            st.error(f"Failed to download {filename} from {url}: {e}")
            raise e


# -----------------------------------------------------------------------------
# 3. Model 1 (E4 Multisensor) Loading & Preprocessing
# -----------------------------------------------------------------------------
E4_CLASSES = ["Non-Slum Built-Up", "Road Network", "Vegetation", "Open Land / Soil", "Slum / Informal Settlement"]
E4_COLORS = [
    [230, 90, 70],    # 0: Non-Slum Built-Up (Coral / Red)
    [255, 225, 20],   # 1: Road (Yellow)
    [34, 150, 40],    # 2: Vegetation (Green)
    [215, 185, 145],  # 3: Open Land (Tan)
    [140, 50, 190],   # 4: Slum (Purple)
]

E4_VASHI_MEANS = np.array([2154.0, 2043.0, 1782.0, 2410.0], dtype=np.float32).reshape(4, 1, 1)
E4_VASHI_STDS = np.array([820.0, 790.0, 750.0, 910.0], dtype=np.float32).reshape(4, 1, 1)
E4_MUMBAI_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
E4_MUMBAI_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@st.cache_resource(show_spinner="Loading Vashi-Mumbai E4 Model Checkpoint...")
def load_e4_model(device: str):
    import train_experiment_E4 as e4_module
    ckpt_path = ensure_checkpoint(
        "best_expE4_model.pth",
        [
            ROOT_DIR / "models" / "05_vashi_mumbai_e4_multisensor" / "checkpoints" / "best_expE4_model.pth",
            ROOT_DIR / "models" / "05_vashi_mumbai_e4_multisensor" / "best_expE4_model.pth"
        ],
        release_tag="v1.0.0"
    )
    net = e4_module.HighResNDVIAdaptiveUNet(num_classes=5)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = sd["model_state_dict"] if "model_state_dict" in sd else sd
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    return net


def preprocess_e4_vashi(raw_4band: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    Expects raw_4band as float32 array of shape [4, H, W] or [H, W, 4]:
    Bands: [Red, Green, Blue, NIR].
    Computes physics NDVI and returns stacked 5-channel tensor [R, G, B, NIR, NDVI].
    """
    if raw_4band.ndim == 3 and raw_4band.shape[2] == 4:
        raw_4band = raw_4band.transpose(2, 0, 1)

    if raw_4band.shape[1] != 512 or raw_4band.shape[2] != 512:
        # Resize each band
        bands = [cv2.resize(raw_4band[i], (512, 512), interpolation=cv2.INTER_LINEAR) for i in range(4)]
        raw_4band = np.stack(bands, axis=0)

    # Replace NoData/saturated
    data = np.where(raw_4band == 65535, E4_VASHI_MEANS, raw_4band)
    norm4 = (data - E4_VASHI_MEANS) / E4_VASHI_STDS

    red = data[0]
    nir = data[3]
    ndvi = (nir - red) / (nir + red + 1e-5)
    ndvi = np.clip(ndvi, -1.0, 1.0)[None, :, :]
    data5 = np.concatenate([norm4, ndvi], axis=0)
    t_img = torch.from_numpy(data5).float()

    # Create RGB preview: Bands [Red=0, Green=1, Blue=2]
    rgb_preview = np.stack([data[0], data[1], data[2]], axis=-1)
    rgb_preview = np.clip((rgb_preview - rgb_preview.min()) / (rgb_preview.max() - rgb_preview.min() + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    return t_img, rgb_preview


def preprocess_e4_mumbai(rgb_img: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Expects RGB image of shape [H, W, 3]. Normalizes with ImageNet stats."""
    if rgb_img.shape[:2] != (512, 512):
        rgb_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    norm3 = (rgb_img.astype(np.float32) / 255.0 - E4_MUMBAI_MEANS.squeeze(-1).transpose(1, 2, 0)) / E4_MUMBAI_STDS.squeeze(-1).transpose(1, 2, 0)
    t_img = torch.from_numpy(norm3.transpose(2, 0, 1)).float()
    return t_img, rgb_img


# -----------------------------------------------------------------------------
# 4. Model 2 (Multispectral SegFormer-U-Net Hybrid) Loading & Preprocessing
# -----------------------------------------------------------------------------
HYBRID_5C_CLASSES = ["Water", "Vegetation", "Barren Land / Soil", "Built-Up", "Roads"]
HYBRID_5C_COLORS = [
    [30, 120, 220],   # 0: Water (Blue)
    [34, 150, 40],    # 1: Vegetation (Green)
    [215, 185, 145],  # 2: Barren (Tan)
    [225, 45, 45],    # 3: Built-Up (Red)
    [255, 225, 20],   # 4: Roads (Yellow)
]

HYBRID_7C_CLASSES = [
    "Water",
    "Vegetation",
    "Barren Land / Soil",
    "Residential Built-Up",
    "Industrial Built-Up",
    "Slum / Informal Settlement",
    "Road Network"
]
HYBRID_7C_COLORS = [
    [30, 120, 220],   # 0: Water (Blue)
    [34, 150, 40],    # 1: Vegetation (Green)
    [215, 185, 145],  # 2: Barren (Tan)
    [230, 90, 70],    # 3: Residential (Salmon Red)
    [140, 50, 190],   # 4: Industrial (Deep Purple)
    [255, 120, 0],    # 5: Slum (Bright Orange)
    [255, 225, 20],   # 6: Road Network (Yellow)
]

BAND_MEANS = np.array([76.38, 77.01, 80.73, 85.92], dtype=np.float32)
BAND_STDS = np.array([45.16, 42.60, 46.12, 49.33], dtype=np.float32)


@st.cache_resource(show_spinner="Loading Multispectral SegFormer + Subclass Models...")
def load_hybrid_models(device: str):
    import models as m3_models
    from models_subclass import MultispectralSegFormerSubclass

    # 1. Macro Model
    macro_ckpt = ensure_checkpoint(
        "best_5class_segformer_extended.pth",
        [
            ROOT_DIR / "models" / "03_cartosat2e_hybrid_0.60m" / "checkpoints" / "best_5class_segformer_extended.pth",
            ROOT_DIR / "models" / "03_cartosat2e_hybrid_0.60m" / "best_5class_segformer_extended.pth"
        ],
        release_tag="v1.0.1"
    )
    macro_net = m3_models.MultispectralSegFormer5Class(num_classes=5, in_channels=4)
    sd_m = torch.load(macro_ckpt, map_location=device, weights_only=False)
    macro_net.load_state_dict(sd_m["model_state_dict"] if "model_state_dict" in sd_m else sd_m, strict=True)
    macro_net.to(device)
    macro_net.eval()

    # 2. Subclass Specialist
    sub_ckpt = ensure_checkpoint(
        "best_3subclass_segformer.pth",
        [
            ROOT_DIR / "models" / "04_cartosat2e_subclasses_0.60m" / "best_3subclass_segformer.pth",
            ROOT_DIR / "models" / "03_cartosat2e_hybrid_0.60m" / "checkpoints" / "best_3subclass_segformer.pth"
        ],
        release_tag="v1.0.0"
    )
    sub_net = MultispectralSegFormerSubclass(num_classes=3, in_channels=4)
    sd_s = torch.load(sub_ckpt, map_location=device, weights_only=False)
    sub_net.load_state_dict(sd_s["model_state_dict"] if "model_state_dict" in sd_s else sd_s, strict=True)
    sub_net.to(device)
    sub_net.eval()

    return macro_net, sub_net


def preprocess_multispectral_4band(raw_4band: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Expects 4-band array [H, W, 4] with bands [Blue, Green, Red, NIR]."""
    if raw_4band.ndim == 3 and raw_4band.shape[0] == 4:
        raw_4band = raw_4band.transpose(1, 2, 0)

    if raw_4band.shape[:2] != (512, 512):
        raw_4band = cv2.resize(raw_4band, (512, 512), interpolation=cv2.INTER_LINEAR)

    norm4 = (raw_4band.astype(np.float32) - BAND_MEANS) / BAND_STDS
    t_img = torch.from_numpy(norm4.transpose(2, 0, 1)).unsqueeze(0).float()

    # True Color preview (RGB: bands 2, 1, 0)
    rgb_preview = np.stack([raw_4band[:, :, 2], raw_4band[:, :, 1], raw_4band[:, :, 0]], axis=-1)
    rgb_preview = np.clip((rgb_preview.astype(float) / 255.0) * 255.0, 0, 255).astype(np.uint8)
    return t_img, rgb_preview


# -----------------------------------------------------------------------------
# 5. Helper Visualization & Statistics Functions
# -----------------------------------------------------------------------------
def colorize_mask(mask: np.ndarray, palette: list) -> np.ndarray:
    """Converts a 2D integer mask into an RGB colorized image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        colored[mask == class_id] = color
    return colored


def compute_class_statistics(mask: np.ndarray, class_names: list, palette: list, gsd_m: float = 0.6) -> pd.DataFrame:
    """Computes pixel count, percentage, and estimated area in hectares."""
    total_pixels = mask.size
    records = []
    pixel_area_m2 = gsd_m * gsd_m

    for class_id, name in enumerate(class_names):
        count = int((mask == class_id).sum())
        pct = (count / total_pixels) * 100.0
        area_ha = (count * pixel_area_m2) / 10000.0
        hex_color = f"#{palette[class_id][0]:02x}{palette[class_id][1]:02x}{palette[class_id][2]:02x}"
        records.append({
            "Class ID": class_id,
            "Class Name": name,
            "Color": hex_color,
            "Pixel Count": count,
            "Coverage (%)": round(pct, 2),
            "Estimated Area (ha)": round(area_ha, 3)
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 6. Streamlit User Interface
# -----------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

st.markdown('<div class="main-header">🛰️ High-Resolution Satellite Urban Feature Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Indian Institute of Technology Bombay — B.Tech Project (BTP-II) | Real-Time Inference Platform</div>', unsafe_allow_html=True)

# Sidebar System Specs
with st.sidebar:
    st.header("⚙️ System Status")
    st.info(f"**Compute Hardware:** `{device.upper()}`")
    st.markdown("---")
    st.header("📌 Model Architecture")
    st.markdown("""
    This platform provides **two production models**:
    * **Model 1 (Preferred):** Vashi-Mumbai Dual-Stem U-Net with ResNet-18 & on-the-fly NDVI.
    * **Model 2:** Multispectral SegFormer + Subclass Specialist Hybrid Ensemble.
    """)
    st.markdown("---")
    overlay_alpha = st.slider("Segmentation Overlay Opacity", min_value=0.0, max_value=1.0, value=0.50, step=0.05)


# -----------------------------------------------------------------------------
# 7. Model Selection (EXACTLY TWO CHOICES)
# -----------------------------------------------------------------------------
selected_model_option = st.radio(
    "Select Model Pipeline:",
    options=[
        "Option 1: Vashi–Mumbai E4 Sensor-Adaptive U-Net (PRIMARY / PREFERRED)",
        "Option 2: Multispectral SegFormer–U-Net Hybrid (Hierarchical 7-Class)"
    ],
    index=0,
    help="Option 1 is the primary multisensor model. Option 2 provides hierarchical macro-to-subclass urban refinement."
)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PIPELINE 1: Vashi-Mumbai E4 Multisensor Model
# =============================================================================
if selected_model_option.startswith("Option 1"):
    st.markdown("""
    <div class="model-card">
        <h3>Option 1: Vashi–Mumbai E4 Sensor-Adaptive U-Net <span class="badge-preferred">PRIMARY / PREFERRED</span></h3>
        <p><strong>Architecture:</strong> <code>HighResNDVIAdaptiveUNet</code> (Pretrained ResNet-18 Encoder + 5-Stage Multi-Scale Decoder + Native Resolution Shallow Skips).</p>
        <p><strong>Sensors Supported:</strong> SPOT-5 (4-Band Multispectral with on-the-fly physical NDVI) & Pléiades (3-Band High-Resolution RGB).</p>
        <p><strong>Output Classes (5):</strong> Non-Slum Built-Up, Road Network, Vegetation, Open Land / Soil, Slum / Informal Settlement.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sensor, col_source = st.columns([1, 1])
    with col_sensor:
        sensor_choice = st.selectbox(
            "Select Sensor & Mode:",
            options=["Vashi SPOT-5 (4-Band Multispectral + NDVI)", "Mumbai Pléiades (3-Band High-Resolution RGB)"]
        )

    with col_source:
        input_source = st.radio("Input Source:", ["Built-in Demo Sample Patch", "Upload Custom Image"], horizontal=True)

    raw_input = None
    input_type = "spot" if "SPOT-5" in sensor_choice else "mumbai"

    if input_source == "Built-in Demo Sample Patch":
        if input_type == "spot":
            demo_path = ROOT_DIR / "demo_samples" / "vashi_spot_sample.npy"
            if demo_path.exists():
                raw_input = np.load(str(demo_path))
                st.caption(f"Loaded demo SPOT-5 4-band patch: `{demo_path.name}` ({raw_input.shape})")
        else:
            demo_path = ROOT_DIR / "demo_samples" / "mumbai_pleiades_sample.png"
            if demo_path.exists():
                bgr = cv2.imread(str(demo_path))
                raw_input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                st.caption(f"Loaded demo Pléiades RGB patch: `{demo_path.name}` ({raw_input.shape})")
    else:
        uploaded_file = st.file_uploader("Upload Image File (.png, .jpg, .npy, .tif)", type=["png", "jpg", "jpeg", "npy", "tif", "tiff"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                raw_input = np.load(uploaded_file)
            else:
                image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                bgr = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
                if bgr.ndim == 3 and bgr.shape[2] == 4:
                    raw_input = bgr  # 4-band
                elif bgr.ndim == 3 and bgr.shape[2] == 3:
                    raw_input = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    raw_input = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)

    if raw_input is not None:
        # Preprocessing
        if input_type == "spot":
            t_img, rgb_preview = preprocess_e4_vashi(raw_input)
            source_tag = "vashi_spot"
        else:
            t_img, rgb_preview = preprocess_e4_mumbai(raw_input)
            source_tag = "mumbai_shapefile"

        # Inference
        net = load_e4_model(device)
        with torch.no_grad():
            t_img_in = t_img.to(device)
            logits = net([t_img_in], [source_tag])
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        colorized_mask = colorize_mask(pred_mask, E4_COLORS)
        overlay = cv2.addWeighted(rgb_preview, 1.0 - overlay_alpha, colorized_mask, overlay_alpha, 0)

        # Display Previews
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Satellite Input (RGB)")
            st.image(rgb_preview, use_column_width=True)

        with col2:
            st.subheader("2. Segmentation Mask (5-Class)")
            st.image(colorized_mask, use_column_width=True)

        with col3:
            st.subheader("3. Alpha-Blended Overlay")
            st.image(overlay, use_column_width=True)

        # Class Statistics
        st.markdown("### 📊 Class Distribution Analysis")
        stats_df = compute_class_statistics(pred_mask, E4_CLASSES, E4_COLORS, gsd_m=2.5 if input_type == "spot" else 0.5)

        col_stat, col_chart = st.columns([1, 1])
        with col_stat:
            st.dataframe(stats_df[["Class Name", "Pixel Count", "Coverage (%)", "Estimated Area (ha)"]], use_container_width=True)

        with col_chart:
            chart_df = stats_df.set_index("Class Name")["Coverage (%)"]
            st.bar_chart(chart_df)

        # Download Buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            res_png = Image.fromarray(colorized_mask)
            mask_bytes = Path(ROOT_DIR / "scratch" / "temp_pred_e4.png")
            mask_bytes.parent.mkdir(parents=True, exist_ok=True)
            res_png.save(mask_bytes)
            with open(mask_bytes, "rb") as f:
                st.download_button(
                    label="⬇️ Download Colorized Mask (PNG)",
                    data=f,
                    file_name="e4_segmentation_mask.png",
                    mime="image/png"
                )

        with col_dl2:
            st.download_button(
                label="⬇️ Download Classification Statistics (CSV)",
                data=stats_df.to_csv(index=False),
                file_name="e4_class_statistics.csv",
                mime="text/csv"
            )

# =============================================================================
# PIPELINE 2: Multispectral SegFormer–U-Net Hybrid
# =============================================================================
else:
    st.markdown("""
    <div class="model-card">
        <h3>Option 2: Multispectral SegFormer–U-Net Hybrid <span class="badge-hierarchical">HIERARCHICAL REFINEMENT</span></h3>
        <p><strong>Architecture:</strong> Dual-Stage Collaborative Pipeline:</p>
        <ul>
            <li><strong>Stage 1 (Macro):</strong> <code>MultispectralSegFormer5Class</code> (4-Band VNIR input &rarr; 5 Land-Cover Classes: Water, Vegetation, Barren, Built-Up, Roads).</li>
            <li><strong>Stage 2 (Subclass Specialist):</strong> <code>MultispectralSegFormerSubclass</code> with scSE Attention (Gated strictly inside Built-Up areas &rarr; Residential, Industrial, Slum).</li>
        </ul>
        <p><strong>Output Classes (7 Unified):</strong> Water, Vegetation, Barren Land, Residential Built-Up, Industrial Built-Up, Slum / Informal Settlement, Road Network.</p>
    </div>
    """, unsafe_allow_html=True)

    input_source = st.radio("Input Source:", ["Built-in Demo Multispectral Patch (4-Band)", "Upload Custom Image (4-Band TIFF/PNG/NPY)"], horizontal=True)

    raw_multi = None
    if input_source.startswith("Built-in"):
        demo_path = ROOT_DIR / "demo_samples" / "multispectral_sample.png"
        if demo_path.exists():
            raw_multi = cv2.imread(str(demo_path), cv2.IMREAD_UNCHANGED)
            st.caption(f"Loaded demo 4-band multispectral patch: `{demo_path.name}` ({raw_multi.shape})")
    else:
        uploaded_file = st.file_uploader("Upload 4-Band Multispectral File (.png, .npy, .tif)", type=["png", "npy", "tif", "tiff"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                raw_multi = np.load(uploaded_file)
            else:
                image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                raw_multi = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
                if raw_multi.ndim == 3 and raw_multi.shape[2] == 3:
                    st.warning("Uploaded image has only 3 channels (RGB). Synthesizing NIR channel from Green & Red for demonstration.")
                    nir_fake = ((raw_multi[:, :, 1].astype(float) + raw_multi[:, :, 2].astype(float)) / 2.0).astype(np.uint8)
                    raw_multi = np.concatenate([raw_multi, nir_fake[:, :, None]], axis=-1)

    if raw_multi is not None:
        t_img, rgb_preview = preprocess_multispectral_4band(raw_multi)

        # Load models
        macro_net, sub_net = load_hybrid_models(device)

        with torch.no_grad():
            t_in = t_img.to(device)
            macro_logits = macro_net(t_in)
            macro_pred = macro_logits.argmax(dim=1)[0].cpu().numpy()

            sub_logits = sub_net(t_in)
            sub_pred = sub_logits.argmax(dim=1)[0].cpu().numpy()

        # Combine into Unified 7-Class Output
        # Macro: 0=Water, 1=Veg, 2=Barren, 3=Built-Up, 4=Road
        # Subclass: 0=Residential, 1=Industrial, 2=Slum
        unified_mask = np.zeros_like(macro_pred)
        unified_mask[macro_pred == 0] = 0  # Water
        unified_mask[macro_pred == 1] = 1  # Vegetation
        unified_mask[macro_pred == 2] = 2  # Barren Land
        unified_mask[macro_pred == 4] = 6  # Road Network

        built_mask = (macro_pred == 3)
        unified_mask[built_mask & (sub_pred == 0)] = 3  # Residential
        unified_mask[built_mask & (sub_pred == 1)] = 4  # Industrial
        unified_mask[built_mask & (sub_pred == 2)] = 5  # Slum
        unified_mask[built_mask & (unified_mask == 0)] = 3  # Default residential if unassigned

        colorized_unified = colorize_mask(unified_mask, HYBRID_7C_COLORS)
        colorized_macro = colorize_mask(macro_pred, HYBRID_5C_COLORS)

        overlay_unified = cv2.addWeighted(rgb_preview, 1.0 - overlay_alpha, colorized_unified, overlay_alpha, 0)

        # Tabs for Hierarchical Views
        tab1, tab2 = st.tabs(["🌟 Unified 7-Class Map (Final)", "🔍 Stage 1 Macro Map (5-Class)"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("1. Satellite Image (RGB)")
                st.image(rgb_preview, use_column_width=True)

            with col2:
                st.subheader("2. Unified 7-Class Segmentation")
                st.image(colorized_unified, use_column_width=True)

            with col3:
                st.subheader("3. Unified Overlay")
                st.image(overlay_unified, use_column_width=True)

            st.markdown("### 📊 7-Class Urban Feature Statistics")
            stats_7c = compute_class_statistics(unified_mask, HYBRID_7C_CLASSES, HYBRID_7C_COLORS, gsd_m=0.6)

            col_stat, col_chart = st.columns([1, 1])
            with col_stat:
                st.dataframe(stats_7c[["Class Name", "Pixel Count", "Coverage (%)", "Estimated Area (ha)"]], use_container_width=True)

            with col_chart:
                st.bar_chart(stats_7c.set_index("Class Name")["Coverage (%)"])

        with tab2:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("Stage 1: Macro Land-Cover Mask")
                st.image(colorized_macro, use_column_width=True)
            with col_m2:
                st.subheader("Stage 1 Macro Statistics")
                stats_5c = compute_class_statistics(macro_pred, HYBRID_5C_CLASSES, HYBRID_5C_COLORS, gsd_m=0.6)
                st.dataframe(stats_5c[["Class Name", "Pixel Count", "Coverage (%)", "Estimated Area (ha)"]], use_container_width=True)

        # Download Buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            res_png = Image.fromarray(colorized_unified)
            mask_bytes = Path(ROOT_DIR / "scratch" / "temp_pred_hybrid.png")
            mask_bytes.parent.mkdir(parents=True, exist_ok=True)
            res_png.save(mask_bytes)
            with open(mask_bytes, "rb") as f:
                st.download_button(
                    label="⬇️ Download Unified 7-Class Mask (PNG)",
                    data=f,
                    file_name="unified_7class_mask.png",
                    mime="image/png"
                )

        with col_dl2:
            st.download_button(
                label="⬇️ Download 7-Class Statistics (CSV)",
                data=stats_7c.to_csv(index=False),
                file_name="unified_7class_statistics.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>IIT Bombay CSRE | Sponsoring Body: ISRO & CSRE / KReSIT | BTP-II Urban Feature Extraction Platform</p>", unsafe_allow_html=True)
