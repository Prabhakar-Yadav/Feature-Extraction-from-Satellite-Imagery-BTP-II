# Reference Combined Hybrid Checkpoint

## File Information
- **Filename:** `best_hybrid_hierarchical_model.pth`
- **File Size:** 546,311,517 bytes (521.0 MB)
- **SHA-256:** `db15c31848f40b24ce22182bd8a09681a8fc3e1c6c35643245e074e428fa0a53`
- **Training Epoch:** 6 (Best overall validation score: 60.26% active subclass mean, 40.46% subclass mIoU)
- **Primary Hosting:** [GitHub Release v1.0.0](https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II/releases/tag/v1.0.0)

## Architecture & Contents
This checkpoint contains the complete monolithic state dictionary of the dual-branch hierarchical network trained end-to-end:
1. `alpha_macro`: Learned scalar gate for macro logit fusion ($\sigma(\alpha_{\text{macro}}) Z_{\text{SegFormer}} + (1 - \sigma(\alpha_{\text{macro}})) Z_{\text{UNet}}$).
2. `segformer.*`: 4-band SegFormer-B2 encoder-decoder weights.
3. `unet.*`: 4-band UltraRoad ResNet-34 Attention U-Net weights.
4. `subclass_head.*`: Multi-scale spatial and channel squeeze-and-excitation (scSE) refinement head.
5. `optimizer_state_dict`: AdamW optimization states and learning rate scheduler states.

## Operational Note
For active production deployment and Streamlit inference (`app.py`), the system loads the decoupled, modular components located in `models/03_cartosat2e_hybrid_0.60m/checkpoints/`:
- `best_5class_segformer_extended.pth` (Stage 1 Macro)
- `best_5class_segformer_lovasz.pth` (Stage 1 Macro)
- `best_5class_unet_multispectral.pth` (Stage 1 Macro)
- `best_3subclass_segformer.pth` (Stage 2 Subclasses)

This decoupled setup allows per-model dihedral test-time augmentation (TTA), calibrated probability blending (0.35 + 0.35 + 0.30), and lower peak GPU/RAM overhead. This monolithic reference checkpoint is preserved for architectural verification and training continuity.
