"""
scripts/inspect_model_families.py
---------------------------------
Inspects the 4 required model families in detail:
1. Latest/best model
2. U-Net-only model
3. SegFormer-only model
4. Combined/Hybrid model
"""

import json
import torch
from pathlib import Path

WORKSPACE = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def inspect():
    # 1. Latest/Best Model: Experiment E4 (or Hybrid 0.60m)
    # Let's inspect both:
    # A) Dataset 3 (Vashi/Mumbai 5-class): research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth
    # B) Dataset 2 (Pune 0.60m Hierarchical): best_hybrid_hierarchical_model.pth
    # C) Dataset 1 (2.5m Cartosat): best_segformer.pth vs best_unet.pth

    candidates = {
        "expE4": WORKSPACE / "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
        "hybrid_060m": WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
        "segformer_25m": WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/SegFormer_2.5m/best_segformer.pth",
        "unet_25m": WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/UNet_2.5m/best_unet.pth",
        "segformer_subclass": WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Final_Subclasses_Model/best_3subclass_segformer.pth"
    }

    for name, p in candidates.items():
        print("=" * 80)
        print(f"CANDIDATE: {name}")
        print(f"Path: {p}")
        if not p.exists():
            print("DOES NOT EXIST!")
            continue
        sz_mb = p.stat().st_size / (1024*1024)
        print(f"Size: {sz_mb:.2f} MB")
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                print(f"Top keys: {list(ckpt.keys())[:15]}")
                for k in ["epoch", "val_iou", "best_iou", "val_acc", "val_loss", "metrics", "config"]:
                    if k in ckpt:
                        print(f"  {k}: {ckpt[k]}")
                if "model_state_dict" in ckpt:
                    sd = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                else:
                    sd = ckpt
                print(f"  Total param tensors: {len(sd)}")
                layer_names = list(sd.keys())
                print(f"  First 5 layers: {layer_names[:5]}")
                print(f"  Last 5 layers:  {layer_names[-5:]}")
            else:
                print(f"Direct object type: {type(ckpt)}")
        except Exception as e:
            print(f"Load error: {e}")

if __name__ == "__main__":
    inspect()

