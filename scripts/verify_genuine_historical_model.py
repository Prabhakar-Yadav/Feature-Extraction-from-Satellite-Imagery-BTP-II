import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

REPO_ROOT = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main")
os.chdir(str(REPO_ROOT))

SUBCLASS_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation"
STAGE1_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training"
sys.path.insert(0, str(SUBCLASS_DIR / "scripts"))
sys.path.insert(0, str(STAGE1_DIR / "scripts"))

from models import MultispectralSegFormer5Class, MultispectralUltraRoadUNet5Class
from models_subclass import MultispectralSegFormerSubclass
from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep
from dataset_subclasses import BAND_MEANS, BAND_STDS, ORIG_CLASS_NAMES

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("GENUINE HISTORICAL CHECKPOINT BENCHMARK & DISCREPANCY ANALYSIS")
    print("=" * 80)

    # 1. Load Genuine Submission Checkpoint
    p_orig = REPO_ROOT / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth"
    orig_model = HybridUNetSegFormerTwoStep().to(DEVICE)
    sd_orig = torch.load(p_orig, map_location='cpu', weights_only=False)
    orig_model.load_state_dict(sd_orig['model_state_dict'])
    orig_model.eval()
    print("Genuine Original Submission Checkpoint loaded (Epoch 6).")

    # 2. Load R05 Checkpoint
    p_r05 = SUBCLASS_DIR / "checkpoints/checkpoint_round_05.pth"
    r05_model = HybridUNetSegFormerTwoStep().to(DEVICE)
    sd_r05 = torch.load(p_r05, map_location='cpu', weights_only=False)
    r05_model.load_state_dict(sd_r05['model_state_dict'])
    r05_model.eval()
    print("R05 Checkpoint loaded (Epoch 50).")

    # 3. Load Manifest
    df_man = pd.read_csv(SUBCLASS_DIR / "data/autonomous_dataset_manifest.csv")
    val_df = df_man[df_man['split'] == 'val'].reset_index(drop=True)
    four_pids = ['train_hinj_0098', 'train_hinj_0112', 'train_hinj_0009', 'train_chn_0317']

    print("\n--- Evaluating 4 Focal Patches ---")
    for pid in four_pids:
        row = val_df[val_df['patch_id'] == pid].iloc[0]
        raw_bgra = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        b, g, r, nir = raw_bgra[:, :, 0], raw_bgra[:, :, 1], raw_bgra[:, :, 2], raw_bgra[:, :, 3]
        img_4c = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_4c - BAND_MEANS) / BAND_STDS
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        gt_mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        with torch.no_grad():
            # Original Model Raw (No closing)
            l5_orig, lsub_orig, bu_mask_orig, _ = orig_model(tensor)
            p5_orig = torch.argmax(F.softmax(l5_orig, dim=1), dim=1).squeeze(0).cpu().numpy()
            psub_orig = torch.argmax(lsub_orig, dim=1).squeeze(0).cpu().numpy()
            p7_orig_raw = np.zeros((512, 512), dtype=np.uint8)
            p7_orig_raw[p5_orig == 0] = 0
            p7_orig_raw[p5_orig == 1] = 1
            p7_orig_raw[p5_orig == 2] = 2
            p7_orig_raw[p5_orig == 4] = 6
            is_bu_orig = (p5_orig == 3)
            p7_orig_raw[is_bu_orig & (psub_orig == 0)] = 3
            p7_orig_raw[is_bu_orig & (psub_orig == 1)] = 4
            p7_orig_raw[is_bu_orig & (psub_orig == 2)] = 5

            # R05 Raw (No closing)
            l5_r05, lsub_r05, bu_mask_r05, _ = r05_model(tensor)
            p5_r05 = torch.argmax(F.softmax(l5_r05, dim=1), dim=1).squeeze(0).cpu().numpy()
            psub_r05 = torch.argmax(lsub_r05, dim=1).squeeze(0).cpu().numpy()
            p7_r05_raw = np.zeros((512, 512), dtype=np.uint8)
            p7_r05_raw[p5_r05 == 0] = 0
            p7_r05_raw[p5_r05 == 1] = 1
            p7_r05_raw[p5_r05 == 2] = 2
            p7_r05_raw[p5_r05 == 4] = 6
            is_bu_r05 = (p5_r05 == 3)
            p7_r05_raw[is_bu_r05 & (psub_r05 == 0)] = 3
            p7_r05_raw[is_bu_r05 & (psub_r05 == 1)] = 4
            p7_r05_raw[is_bu_r05 & (psub_r05 == 2)] = 5

            # R05 with 9x9 block closing
            _, _, p7_r05_closed, _ = r05_model.predict_hierarchical_7class(tensor)
            p7_r05_closed = p7_r05_closed.squeeze(0).cpu().numpy()

        acc_orig_raw = np.mean(p7_orig_raw == gt_mask) * 100
        acc_r05_raw = np.mean(p7_r05_raw == gt_mask) * 100
        acc_r05_closed = np.mean(p7_r05_closed == gt_mask) * 100

        print(f"\nPatch {pid}:")
        print(f"  Orig Submission Model (Raw, No Closing) Acc: {acc_orig_raw:.2f}%")
        print(f"  R05 Checkpoint (Raw, No Closing) Acc:        {acc_r05_raw:.2f}%")
        print(f"  R05 Checkpoint (With 9x9 Closing) Acc:       {acc_r05_closed:.2f}%")

if __name__ == "__main__":
    main()

