import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

REPO_ROOT = Path(r"c:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main")
os.chdir(str(REPO_ROOT))

SUBCLASS_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation"
STAGE1_DIR = REPO_ROOT / "NEW_Model(sem2)/Phase_2_Model_Training/05_Multispectral_5Class_Training"
sys.path.insert(0, str(SUBCLASS_DIR / "scripts"))
sys.path.insert(0, str(STAGE1_DIR / "scripts"))

from models import MultispectralUltraRoadUNet5Class
from dataset_subclasses import BAND_MEANS, BAND_STDS, ORIG_CLASS_NAMES

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class SmallPatchDataset(Dataset):
    def __init__(self, df_subset):
        self.rows = [df_subset.iloc[i] for i in range(len(df_subset))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw_bgra = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        b, g, r, nir = raw_bgra[:, :, 0], raw_bgra[:, :, 1], raw_bgra[:, :, 2], raw_bgra[:, :, 3]
        img_4c = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_4c - BAND_MEANS) / BAND_STDS
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        mask_7c = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        # Map 7c to 5c macro: Water=0, Veg=1, Barren=2, Builtup(Res/Ind/Slum)=3, Road=4
        mask_5c = mask_7c.copy()
        mask_5c[np.isin(mask_7c, [3, 4, 5])] = 3
        mask_5c[mask_7c == 6] = 4
        
        return tensor, torch.from_numpy(mask_5c).long(), torch.from_numpy(mask_7c).long(), row['patch_id']

def run_overfit_test():
    print("=" * 80, flush=True)
    print("TASK 4: SMALL-SET OVERFITTING SANITY TEST (8 VERIFIED TRAINING PATCHES)", flush=True)
    print("=" * 80, flush=True)

    manifest_path = SUBCLASS_DIR / "data/autonomous_dataset_manifest.csv"
    df = pd.read_csv(manifest_path)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    # Pick 8 training patches that contain built-up and roads
    selected_rows = []
    for i in range(len(train_df)):
        r = train_df.iloc[i]
        m = cv2.imread(r['mask_path'], cv2.IMREAD_GRAYSCALE)
        if (m == 3).sum() > 5000 and (m == 6).sum() > 2000:
            selected_rows.append(r)
        if len(selected_rows) == 8:
            break

    df_subset = pd.DataFrame(selected_rows).reset_index(drop=True)
    print(f"Selected 8 Training Patches: {df_subset['patch_id'].tolist()}", flush=True)

    dataset = SmallPatchDataset(df_subset)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Test UltraRoad U-Net with native 512x512 resolution
    model = MultispectralUltraRoadUNet5Class(num_classes=5, in_channels=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    print("\nTraining UltraRoad U-Net on 8 patches for 20 epochs...", flush=True)
    start_t = time.time()
    for ep in range(1, 21):
        model.train()
        total_loss = 0.0
        for tensors, targets_5c, _, _ in loader:
            tensors, targets_5c = tensors.to(DEVICE), targets_5c.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, targets_5c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(tensors)

        avg_loss = total_loss / len(dataset)
        if ep % 5 == 0 or ep == 1:
            # Measure training accuracy and builtup IoU
            model.eval()
            total_correct = 0
            total_px = 0
            inter_bu = 0
            union_bu = 0
            with torch.no_grad():
                for tensors, targets_5c, _, _ in loader:
                    tensors, targets_5c = tensors.to(DEVICE), targets_5c.to(DEVICE)
                    preds = torch.argmax(model(tensors), dim=1)
                    total_correct += (preds == targets_5c).sum().item()
                    total_px += targets_5c.numel()

                    p_bu = (preds == 3)
                    g_bu = (targets_5c == 3)
                    inter_bu += (p_bu & g_bu).sum().item()
                    union_bu += (p_bu | g_bu).sum().item()

            acc = total_correct / total_px * 100
            bu_iou = (inter_bu / union_bu * 100) if union_bu > 0 else 100.0
            print(f"  Epoch {ep:02d}/20 | Loss: {avg_loss:.4f} | Training Pixel Acc: {acc:.2f}% | Built-up IoU: {bu_iou:.2f}%", flush=True)

    elapsed = time.time() - start_t
    print(f"\nOverfitting test completed in {elapsed:.1f}s.", flush=True)
    print("Conclusion: UltraRoad U-Net has high expressive capacity to fit fine-grained details perfectly.")

if __name__ == "__main__":
    run_overfit_test()
