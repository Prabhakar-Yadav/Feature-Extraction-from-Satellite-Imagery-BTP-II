import os
import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

REPO_ROOT = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main")
os.chdir(str(REPO_ROOT))

# Append script paths
SUBCLASS_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation"
STAGE1_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training"
sys.path.insert(0, str(SUBCLASS_DIR / "scripts"))
sys.path.insert(0, str(STAGE1_DIR / "scripts"))

from models import MultispectralSegFormer5Class, MultispectralUltraRoadUNet5Class
from models_subclass import MultispectralSegFormerSubclass
from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep, regularize_subclass_blocks
from dataset_subclasses import BAND_MEANS, BAND_STDS, ORIG_CLASS_NAMES

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def scan_all_checkpoints():
    print("=" * 80)
    print("TASK 1: SCANNING ALL HISTORICAL CHECKPOINTS")
    print("=" * 80)
    ckpts = list(REPO_ROOT.glob("**/*.pth"))
    results = []
    for c in ckpts:
        sz_mb = c.stat().st_size / (1024 * 1024)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.stat().st_mtime))
        # Inspect keys without loading full weight tensor into GPU
        try:
            sd = torch.load(c, map_location='cpu', weights_only=False)
            if isinstance(sd, dict):
                keys = list(sd.keys())
                has_model_sd = 'model_state_dict' in sd
                epoch = sd.get('epoch', 'N/A')
                miou = sd.get('val_miou', sd.get('miou', sd.get('best_miou', 'N/A')))
            else:
                keys = ['raw_state_dict']
                has_model_sd = False
                epoch = 'raw'
                miou = 'N/A'
        except Exception as e:
            keys = [f"ERR: {e}"]
            has_model_sd = False
            epoch = 'err'
            miou = 'err'
        
        results.append({
            'path': str(c.relative_to(REPO_ROOT)),
            'size_mb': round(sz_mb, 1),
            'mtime': mtime,
            'epoch': epoch,
            'miou': miou,
            'has_model_state_dict': has_model_sd,
            'num_keys': len(keys)
        })
    df_ckpts = pd.DataFrame(results)
    print(df_ckpts.to_string())
    df_ckpts.to_csv("checkpoint_inventory.csv", index=False)
    return df_ckpts

def inspect_four_patches_raw_pipeline():
    print("\n" + "=" * 80)
    print("TASK 2 & 3: INSPECTING RAW LOGITS, UPSAMPLING, & SPATIAL BREAKDOWN")
    print("=" * 80)

    # Load 4 focal patches
    manifest_path = SUBCLASS_DIR / "data/autonomous_dataset_manifest.csv"
    df_man = pd.read_csv(manifest_path)
    four_pids = ['train_hinj_0098', 'train_hinj_0112', 'train_hinj_0009', 'train_chn_0317']

    # Load Model A components (Triple Merged Ensemble)
    ckpt_seg_5c = STAGE1_DIR / "checkpoints/best_5class_segformer_extended.pth"
    ckpt_seg_lov = STAGE1_DIR / "checkpoints/best_5class_segformer_lovasz.pth"
    ckpt_unet_5c = STAGE1_DIR / "checkpoints/best_5class_unet_multispectral.pth"
    ckpt_sub_3c = SUBCLASS_DIR / "checkpoints/best_3subclass_segformer.pth"

    segformer = MultispectralSegFormer5Class(num_classes=5, in_channels=4).to(DEVICE)
    segformer.load_state_dict(torch.load(ckpt_seg_5c, map_location='cpu', weights_only=False)['model_state_dict'])
    segformer.eval()

    unet = MultispectralUltraRoadUNet5Class(num_classes=5, in_channels=4).to(DEVICE)
    unet.load_state_dict(torch.load(ckpt_unet_5c, map_location='cpu', weights_only=False)['model_state_dict'])
    unet.eval()

    subclass_net = MultispectralSegFormerSubclass(num_classes=3, in_channels=4).to(DEVICE)
    subclass_net.load_state_dict(torch.load(ckpt_sub_3c, map_location='cpu', weights_only=False)['model_state_dict'])
    subclass_net.eval()

    # Load Model B (R05 checkpoint)
    ckpt_r05 = SUBCLASS_DIR / "checkpoints/checkpoint_round_05.pth"
    r05_model = HybridUNetSegFormerTwoStep().to(DEVICE)
    r05_sd = torch.load(ckpt_r05, map_location='cpu', weights_only=False)
    r05_model.load_state_dict(r05_sd['model_state_dict'])
    r05_model.eval()

    diagnostics = {}

    for pid in four_pids:
        row = df_man[df_man['patch_id'] == pid].iloc[0]
        raw_bgra = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        b, g, r, nir = raw_bgra[:, :, 0], raw_bgra[:, :, 1], raw_bgra[:, :, 2], raw_bgra[:, :, 3]
        img_4c = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_4c - BAND_MEANS) / BAND_STDS
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        gt_mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # Evaluate Stage 1 SegFormer raw logits vs UNet raw logits
        with torch.no_grad():
            out_seg_raw = segformer.segformer(tensor).logits # [1, 5, 128, 128]
            out_seg_refined = segformer(tensor)              # [1, 5, 512, 512]
            out_unet = unet(tensor)                          # [1, 5, 512, 512]

            # R05 outputs
            logits_5c_r05, logits_sub_r05, pred_bu_r05, prob_bu_r05 = r05_model(tensor)

        # Print spatial resolution & stats
        print(f"\n--- Patch: {pid} ---")
        print(f"  Input image shape: {raw_bgra.shape}, dtype: {raw_bgra.dtype}, min: {raw_bgra.min()}, max: {raw_bgra.max()}")
        print(f"  SegFormer backbone output shape (stride): {out_seg_raw.shape} (STRIDE 4: 128x128)")
        print(f"  SegFormer refined output shape: {out_seg_refined.shape}")
        print(f"  UltraRoad U-Net output shape: {out_unet.shape} (NATIVE 512x512)")
        print(f"  R05 alpha_macro: {torch.sigmoid(r05_model.alpha_macro).item():.3f} (SegFormer weight)")
        
        # Ground truth class distribution
        gt_dist = {ORIG_CLASS_NAMES[c]: int((gt_mask == c).sum()) for c in range(7)}
        print(f"  GT Pixel Counts: {gt_dist}")

        # Compare UNet vs SegFormer alone
        p_seg = torch.argmax(out_seg_refined, dim=1).cpu().numpy()[0]
        p_unet = torch.argmax(out_unet, dim=1).cpu().numpy()[0]
        
        # Measure agreement with GT 5-class
        gt_5c = gt_mask.copy()
        gt_5c[np.isin(gt_mask, [3, 4, 5])] = 3
        gt_5c[gt_mask == 6] = 4

        acc_seg = np.mean(p_seg == gt_5c) * 100
        acc_unet = np.mean(p_unet == gt_5c) * 100
        print(f"  5-Class Accuracy -> SegFormer Alone: {acc_seg:.2f}%, UltraRoad U-Net Alone: {acc_unet:.2f}%")

        # Check where built-up is predicted
        bu_seg_px = int((p_seg == 3).sum())
        bu_unet_px = int((p_unet == 3).sum())
        bu_gt_px = int((gt_5c == 3).sum())
        print(f"  Built-up Pixels -> True: {bu_gt_px}, SegFormer: {bu_seg_px} (ratio {bu_seg_px/(bu_gt_px+1e-5):.2f}), U-Net: {bu_unet_px} (ratio {bu_unet_px/(bu_gt_px+1e-5):.2f})")

        # Subclass prediction on GT builtup vs Predicted builtup
        p_sub_raw = torch.argmax(subclass_net(tensor), dim=1).cpu().numpy()[0]
        gt_bu_mask = (gt_5c == 3)
        if gt_bu_mask.any():
            sub_gt_dist = {i: int((p_sub_raw[gt_bu_mask] == i).sum()) for i in range(3)}
            print(f"  Subclass Pred inside GT Builtup: Res={sub_gt_dist[0]}, Ind={sub_gt_dist[1]}, Slum={sub_gt_dist[2]}")

        # Check R05 builtup inflation
        bu_r05_px = int(pred_bu_r05.cpu().numpy().sum())
        print(f"  R05 Stage 1 Builtup Mask Pixels: {bu_r05_px} (ratio {bu_r05_px/(bu_gt_px+1e-5):.2f})")

def compare_checkpoints_weights():
    print("\n" + "=" * 80)
    print("COMPARING WEIGHT TENSORS BETWEEN CHECKPOINTS")
    print("=" * 80)
    p_orig = REPO_ROOT / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth"
    p_phase2 = SUBCLASS_DIR / "checkpoints/best_hybrid_hierarchical_model.pth"
    p_base21 = SUBCLASS_DIR / "checkpoints/best_hybrid_hierarchical_model_30ep_baseline.pth"
    p_r05 = SUBCLASS_DIR / "checkpoints/checkpoint_round_05.pth"

    sd_orig = torch.load(p_orig, map_location='cpu', weights_only=False)
    sd_p2 = torch.load(p_phase2, map_location='cpu', weights_only=False)
    sd_b21 = torch.load(p_base21, map_location='cpu', weights_only=False)
    sd_r05 = torch.load(p_r05, map_location='cpu', weights_only=False)

    print(f"Original Submission Ckpt Epoch: {sd_orig.get('epoch')}, Keys: {list(sd_orig.keys())}")
    print(f"Phase 2 Ckpt Epoch: {sd_p2.get('epoch')}, Keys: {list(sd_p2.keys())}")
    print(f"Baseline 30ep Ckpt Epoch: {sd_b21.get('epoch')}, Keys: {list(sd_b21.keys())}")
    print(f"R05 Ckpt Epoch: {sd_r05.get('epoch')}, Keys: {list(sd_r05.keys())}")

    # Check differences in weights
    w_orig = sd_orig['model_state_dict']
    w_p2 = sd_p2['model_state_dict']
    w_b21 = sd_b21['model_state_dict']
    w_r05 = sd_r05['model_state_dict']

    diff_orig_vs_p2 = 0
    diff_orig_vs_r05 = 0
    diff_orig_vs_b21 = 0
    total_params = 0

    common_keys = [k for k in w_orig.keys() if k in w_r05]
    print(f"Common parameter tensors: {len(common_keys)} of {len(w_orig)} keys")

    for k in common_keys:
        t_orig = w_orig[k].float()
        t_p2 = w_p2[k].float()
        t_r05 = w_r05[k].float()
        t_b21 = w_b21[k].float()

        d_p2 = torch.norm(t_orig - t_p2).item()
        d_r05 = torch.norm(t_orig - t_r05).item()
        d_b21 = torch.norm(t_orig - t_b21).item()

        diff_orig_vs_p2 += d_p2
        diff_orig_vs_r05 += d_r05
        diff_orig_vs_b21 += d_b21
        total_params += t_orig.numel()

    print(f"Total parameter elements examined: {total_params:,}")
    print(f"L2 Weight Distance (Orig Submission vs Phase 2 Ckpt): {diff_orig_vs_p2:.4f}")
    print(f"L2 Weight Distance (Orig Submission vs Baseline 30ep): {diff_orig_vs_b21:.4f}")
    print(f"L2 Weight Distance (Orig Submission vs R05):         {diff_orig_vs_r05:.4f}")

if __name__ == "__main__":
    scan_all_checkpoints()
    compare_checkpoints_weights()
    inspect_four_patches_raw_pipeline()
