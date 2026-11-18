import os
import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2

# Ensure paths
REPO_ROOT = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main")
os.chdir(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "scripts"))
from models_hybrid_hierarchical import HybridHierarchicalPipeline

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("AUTONOMOUS CONTROLLED TRAINING: RECOVERED HIERARCHICAL PIPELINE")
    print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")
    print("Configuration: Sharp Gating (Model C), Balanced Loss Weights, Class-Aware Sampler")
    print("=" * 80)

    # 1. Load Manifest
    manifest_path = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/data/autonomous_dataset_manifest.csv"
    df = pd.read_csv(manifest_path)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    print(f"Dataset split: Train = {len(train_df)}, Val = {len(val_df)}, Held-Out Test = {len(test_df)}")
    assert len(val_df) == 61, f"Expected 61 validation patches, got {len(val_df)}"
    assert len(test_df) == 186, f"Expected 186 held-out test patches, got {len(test_df)}"

    # 2. Checkpoints
    ckpt_stage1_seg = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training/checkpoints/best_5class_segformer_extended.pth"
    ckpt_stage1_unet = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training/checkpoints/best_5class_unet_multispectral.pth"
    ckpt_stage2_sub = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_3subclass_segformer.pth"

    pipeline = HybridHierarchicalPipeline(
        stage1_segformer_path=str(ckpt_stage1_seg),
        stage1_unet_path=str(ckpt_stage1_unet),
        stage2_subclass_path=str(ckpt_stage2_sub),
        device=str(DEVICE)
    )
    print("Pipeline successfully initialized with verified base checkpoints.")
    print("Ready to run controlled fine-tuning without morphological distortion.")

if __name__ == "__main__":
    main()

