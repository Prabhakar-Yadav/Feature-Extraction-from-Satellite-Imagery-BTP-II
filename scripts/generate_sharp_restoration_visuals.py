import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main")
os.chdir(str(REPO_ROOT))

SUBCLASS_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation"
STAGE1_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training"
sys.path.insert(0, str(SUBCLASS_DIR / "scripts"))
sys.path.insert(0, str(STAGE1_DIR / "scripts"))

from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep
from dataset_subclasses import BAND_MEANS, BAND_STDS, ORIG_CLASS_NAMES

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PALETTE_RGB = [
    [30, 120, 220],   # 0: Water (Blue)
    [34, 150, 40],    # 1: Vegetation (Forest Green)
    [215, 185, 145],  # 2: Barren Land (Tan)
    [230, 90, 70],    # 3: Residential (Soft Terracotta / Coral Red)
    [140, 50, 190],   # 4: Industrial (Steel Purple)
    [255, 120, 0],    # 5: Slum (Amber / Orange)
    [255, 225, 20]    # 6: Road Network (Bright Yellow)
]

def mask_to_color(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(7):
        rgb[mask == c] = PALETTE_RGB[c]
    return rgb

def main():
    print("=" * 80)
    print("GENERATING SHARP RESTORATION VISUALS (5-PANEL COMPARISONS)")
    print("=" * 80)

    # 1. Load Genuine Submission Checkpoint
    p_orig = REPO_ROOT / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth"
    orig_model = HybridUNetSegFormerTwoStep().to(DEVICE)
    sd_orig = torch.load(p_orig, map_location='cpu', weights_only=False)
    orig_model.load_state_dict(sd_orig['model_state_dict'])
    orig_model.eval()

    # 2. Load R05 Checkpoint
    p_r05 = SUBCLASS_DIR / "checkpoints/checkpoint_round_05.pth"
    r05_model = HybridUNetSegFormerTwoStep().to(DEVICE)
    sd_r05 = torch.load(p_r05, map_location='cpu', weights_only=False)
    r05_model.load_state_dict(sd_r05['model_state_dict'])
    r05_model.eval()

    # Output directories
    art_dir = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\restored_sharp_predictions")
    art_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO_ROOT / "NEW_Model(sem2)/FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/restored_sharp_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_man = pd.read_csv(SUBCLASS_DIR / "data/autonomous_dataset_manifest.csv")
    four_pids = ['train_hinj_0098', 'train_hinj_0112', 'train_hinj_0009', 'train_chn_0317']

    for idx, pid in enumerate(four_pids):
        row = df_man[df_man['patch_id'] == pid].iloc[0]
        raw_bgra = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        b, g, r, nir = raw_bgra[:, :, 0], raw_bgra[:, :, 1], raw_bgra[:, :, 2], raw_bgra[:, :, 3]
        img_4c = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_4c - BAND_MEANS) / BAND_STDS
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        gt_mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        rgb_vis = cv2.cvtColor(raw_bgra[:, :, :3], cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # 1. Orig Model Raw (alpha ~ 0.55)
            l5_orig, lsub_orig, _, _ = orig_model(tensor)
            p5_orig = torch.argmax(F.softmax(l5_orig, dim=1), dim=1).squeeze(0).cpu().numpy()
            psub_orig = torch.argmax(lsub_orig, dim=1).squeeze(0).cpu().numpy()
            p7_orig = np.zeros((512, 512), dtype=np.uint8)
            p7_orig[p5_orig == 0] = 0
            p7_orig[p5_orig == 1] = 1
            p7_orig[p5_orig == 2] = 2
            p7_orig[p5_orig == 4] = 6
            is_bu_orig = (p5_orig == 3)
            p7_orig[is_bu_orig & (psub_orig == 0)] = 3
            p7_orig[is_bu_orig & (psub_orig == 1)] = 4
            p7_orig[is_bu_orig & (psub_orig == 2)] = 5

            # 2. R05 Checkpoint with 9x9 block closing
            _, _, p7_r05, _ = r05_model.predict_hierarchical_7class(tensor)
            p7_r05 = p7_r05.squeeze(0).cpu().numpy()

            # 3. Restored Sharp Model: UltraRoad U-Net 70% + SegFormer 30%, strict gating, zero closing
            out_seg = orig_model.segformer(tensor)
            out_unet = orig_model.unet(tensor)
            p_seg = F.softmax(out_seg, dim=1)
            p_unet = F.softmax(out_unet, dim=1)
            # Invert alpha: 70% U-Net, 30% SegFormer
            p_sharp = 0.30 * p_seg + 0.70 * p_unet
            p5_sharp = torch.argmax(p_sharp, dim=1).squeeze(0).cpu().numpy()

            # Subclasses conditioned on U-Net feature sharpness
            p7_sharp = np.zeros((512, 512), dtype=np.uint8)
            p7_sharp[p5_sharp == 0] = 0
            p7_sharp[p5_sharp == 1] = 1
            p7_sharp[p5_sharp == 2] = 2
            p7_sharp[p5_sharp == 4] = 6
            is_bu_sharp = (p5_sharp == 3)
            p7_sharp[is_bu_sharp & (psub_orig == 0)] = 3
            p7_sharp[is_bu_sharp & (psub_orig == 1)] = 4
            p7_sharp[is_bu_sharp & (psub_orig == 2)] = 5

        acc_orig = np.mean(p7_orig == gt_mask) * 100
        acc_r05 = np.mean(p7_r05 == gt_mask) * 100
        acc_sharp = np.mean(p7_sharp == gt_mask) * 100

        # Plot 5-panel figure
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=140)

        # 1. RGB
        axes[0].imshow(rgb_vis)
        axes[0].set_title(f"1. Satellite RGB: {pid}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Human Ground Truth
        axes[1].imshow(mask_to_color(gt_mask))
        axes[1].set_title("2. Human Ground Truth", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. Original Submission Model
        axes[2].imshow(mask_to_color(p7_orig))
        axes[2].set_title(f"3. Original Model (Raw)\\nAcc: {acc_orig:.1f}%", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # 4. R05 Checkpoint (Coarse blocks)
        axes[3].imshow(mask_to_color(p7_r05))
        axes[3].set_title(f"4. R05 Checkpoint (Coarse Blocks)\\nAcc: {acc_r05:.1f}%", fontsize=11, fontweight="bold", color="red")
        axes[3].axis("off")

        # 5. Restored Sharp Model
        axes[4].imshow(mask_to_color(p7_sharp))
        axes[4].set_title(f"5. Restored Sharp Model\\n(70% U-Net, No Closing) Acc: {acc_sharp:.1f}%", fontsize=11, fontweight="bold", color="green")
        axes[4].axis("off")

        # Add Legend
        patches = [mpatches.Patch(color=np.array(PALETTE_RGB[c])/255.0, label=f"{ORIG_CLASS_NAMES[c]} ({c})") for c in range(7)]
        fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        fname = f"restored_sharp_0{idx+1}_{pid}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        fig.savefig(art_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname} (Orig: {acc_orig:.1f}%, R05: {acc_r05:.1f}%, Restored Sharp: {acc_sharp:.1f}%)")

if __name__ == "__main__":
    main()

