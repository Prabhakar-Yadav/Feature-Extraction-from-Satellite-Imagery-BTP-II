"""
scripts/analyze_checkpoints.py
------------------------------
Deep inspection of all model checkpoints:
- Architecture detection
- State dict keys / metadata
- Loadability check via torch.load
- Classification into the 4 required model families:
  1. Latest/Best Model
  2. U-Net-only Model
  3. SegFormer-only Model
  4. Combined/Hybrid Model
"""

import os
import json
import torch
from pathlib import Path

WORKSPACE_ROOT = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def analyze():
    json_path = WORKSPACE_ROOT / "scratch" / "audit_storage_summary.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = data["checkpoints"]
    print(f"Total checkpoints to inspect: {len(checkpoints)}")

    results = []

    for idx, ck in enumerate(checkpoints):
        rel_p = ck["path"]
        full_p = WORKSPACE_ROOT / rel_p
        sz_mb = ck["size_mb"]

        entry = {
            "index": idx + 1,
            "path": rel_p,
            "size_mb": sz_mb,
            "mtime": ck["mtime"],
            "loadable": False,
            "keys": [],
            "num_params": 0,
            "arch_type": "Unknown",
            "notes": ""
        }

        if sz_mb < 0.01:
            entry["notes"] = "Empty or stub file (< 10 KB)"
            results.append(entry)
            continue

        try:
            # Safe load on CPU
            obj = torch.load(full_p, map_location="cpu", weights_only=False)
            entry["loadable"] = True

            state_dict = None
            if isinstance(obj, dict):
                entry["keys"] = list(obj.keys())
                if "model_state_dict" in obj:
                    state_dict = obj["model_state_dict"]
                elif "state_dict" in obj:
                    state_dict = obj["state_dict"]
                else:
                    state_dict = obj
            elif hasattr(obj, "state_dict"):
                state_dict = obj.state_dict()
            else:
                entry["notes"] = f"Unusual torch object type: {type(obj)}"

            if state_dict is not None and isinstance(state_dict, dict):
                param_keys = list(state_dict.keys())
                entry["param_keys_sample"] = param_keys[:10]
                entry["total_layers"] = len(param_keys)

                # Heuristic architecture detection
                keys_str = " ".join(param_keys).lower()
                if "segformer" in keys_str or "decode_head" in keys_str or "backbone" in keys_str and "linear_c" in keys_str:
                    if "unet" in keys_str or "encoder" in keys_str and "decoder" in keys_str and "segformer" in keys_str:
                        entry["arch_type"] = "Hybrid SegFormer + U-Net"
                    else:
                        entry["arch_type"] = "SegFormer"
                elif "unet" in keys_str or "up_conv" in keys_str or "down_conv" in keys_str or "d1" in keys_str or "encoder.conv1" in keys_str or "decoder.blocks" in keys_str:
                    if "scse" in keys_str:
                        entry["arch_type"] = "U-Net with scSE Attention"
                    else:
                        entry["arch_type"] = "U-Net"
                elif "resnet" in keys_str:
                    entry["arch_type"] = "ResNet Backbone / Encoder"
                elif "linear" in keys_str or "classifier" in keys_str:
                    entry["arch_type"] = "Classifier / Linear Head"
                else:
                    entry["arch_type"] = "PyTorch State Dict (General)"

                # Extract training metadata if present
                if isinstance(obj, dict):
                    for meta_key in ["epoch", "val_iou", "val_acc", "val_loss", "best_iou", "best_loss", "metrics", "args", "config"]:
                        if meta_key in obj:
                            entry[meta_key] = str(obj[meta_key])[:100]

        except Exception as e:
            entry["loadable"] = False
            entry["notes"] = f"Failed to load: {str(e)[:100]}"

        results.append(entry)

    out_json = WORKSPACE_ROOT / "scratch" / "checkpoint_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Analysis saved to {out_json}")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Idx':3s} | {'Size (MB)':9s} | {'Arch Type':25s} | {'Loadable':8s} | Path")
    print("=" * 100)
    for r in results:
        print(f"{r['index']:3d} | {r['size_mb']:8.2f}M | {r['arch_type']:25s} | {str(r['loadable']):8s} | {r['path']}")

if __name__ == "__main__":
    analyze()

