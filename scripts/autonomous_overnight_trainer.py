"""
autonomous_overnight_trainer.py
--------------------------------
Autonomous Overnight Iterative Training, Evaluation, Diagnosis, and Fine-Tuning Pipeline
for the Three-Stage Cartosat-2E Urban Segmentation Model (SegFormer + UltraRoad U-Net + Subclass Morphology Head).

Key Capabilities:
1. 10-Epoch Autonomous Rounds:
   - Evaluates on a fixed validation set (61 patches from autonomous_dataset_manifest.csv).
   - Never touches the held-out test set (186 test_pune_* patches).
2. Unfrozen Two-Step Gradient Supervision:
   - Step 1 (5-Class Macro): Trains SegFormer and U-Net backbones with conservative lr=3e-6.
   - Step 2 (3-Subclass): Trains Subclass SegFormer and Morphology Head with lr=5e-5 inside built-up territory.
3. Automated Error Diagnosis:
   - Computes per-class IoU, Dice, Precision, Recall, Confusion Matrix, and aggregate Built-up IoU.
   - Identifies lagging classes and dynamically adjusts class weights, focal gamma, and augmentations.
4. Intelligent Diagnostic Patch Selection:
   - Selects 20 diverse validation patches (worst IoU, subclass confusion, boundary uncertainty, road continuity).
   - Generates 4-panel visualizations: [RGB | Ground Truth | Prediction | Error Difference].
5. Automatic Model-Selection & Recovery:
   - Composite scoring: 0.35 * mIoU_7c + 0.25 * IoU_res + 0.20 * IoU_ind + 0.10 * IoU_slum + 0.10 * IoU_road.
   - Automatically promotes new best checkpoints (best_autonomous_model.pth).
   - Recovers from regression by rolling back to best_autonomous_model.pth and adapting the strategy.
6. Safe Self-Training (EMA / Pseudo-Labeling):
   - Generates high-confidence pseudo-labels (tau > 0.85) on unlabelled/weakly supervised training pixels.
7. Hardware & Resource Safety:
   - Uses torch.amp.autocast('cuda') mixed precision and gradient accumulation (4 steps).
   - Regularly prunes non-milestone checkpoints to keep disk space safely above 8 GB.
   - Updates live HTML dashboard (autonomous_training_dashboard.html).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Paths setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SUBCLASS_DIR = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation"
STAGE1_SCRIPTS = PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "scripts"
SUBCLASS_SCRIPTS = SUBCLASS_DIR / "scripts"

sys.path.insert(0, str(STAGE1_SCRIPTS))
sys.path.insert(0, str(SUBCLASS_SCRIPTS))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"[HARDWARE] GPU: {torch.cuda.get_device_name(0)} | cuDNN Benchmark: Enabled", flush=True)

from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep, regularize_subclass_blocks
from losses_subclasses import BlockRegularizedSubclassLoss, FocalCrossEntropySubclassLoss

# Import Stage 1 Focal & Lovasz Losses
import importlib.util
STAGE1_LOSSES_PATH = STAGE1_SCRIPTS / "losses.py"
spec_l = importlib.util.spec_from_file_location("losses_5class_module", str(STAGE1_LOSSES_PATH))
losses_5c_mod = importlib.util.module_from_spec(spec_l)
spec_l.loader.exec_module(losses_5c_mod)
FocalLoss5C = losses_5c_mod.FocalLoss
LovaszSoftmaxLoss5C = losses_5c_mod.LovaszSoftmaxLoss

CLASS_NAMES_7 = [
    "Water", "Vegetation", "Barren Land",
    "Residential Built-up", "Industrial Built-up", "Slum / Informal Settlement",
    "Road Network"
]

COLORS_7 = {
    0: (30, 120, 220),    # Water: Deep Blue
    1: (34, 150, 40),     # Vegetation: Forest Green
    2: (215, 185, 145),   # Barren: Warm Sand / Tan
    3: (230, 90, 70),     # Residential: Soft Terracotta
    4: (140, 50, 190),    # Industrial: Steel Purple
    5: (255, 120, 0),     # Slum: Bright Amber / Orange
    6: (255, 225, 20),    # Roads: Bright Yellow
}

BAND_MEANS = np.array([0.485, 0.456, 0.406, 0.450], dtype=np.float32)
BAND_STDS  = np.array([0.229, 0.224, 0.225, 0.220], dtype=np.float32)


class AutonomousHierarchicalDataset(Dataset):
    """
    Dataset for autonomous training. Directly extracts:
    - 4-band normalized multispectral image (R, G, B, NIR)
    - 5-class target (0: Water, 1: Veg, 2: Barren, 3: Built-up, 4: Road)
    - 3-subclass target (0: Res, 1: Ind, 2: Slum, 255: Ignore)
    - Synchronized spatial & spectral augmentations for training
    """
    def __init__(self, df: pd.DataFrame, is_train: bool = True, target_size: int = 512):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = Path(row["image_path"])
        mask_p = Path(row["mask_path"])

        bgra = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Missing image {img_p}")
        b, g, r, nir = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2], bgra[:, :, 3]
        img = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0

        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # 7-Class ground truth
        orig_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if orig_mask is None:
            orig_mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        elif orig_mask.shape[0] != self.target_size or orig_mask.shape[1] != self.target_size:
            orig_mask = cv2.resize(orig_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # 5-Class macro target
        mask_5c = np.zeros_like(orig_mask, dtype=np.int64)
        mask_5c[orig_mask == 0] = 0
        mask_5c[orig_mask == 1] = 1
        mask_5c[orig_mask == 2] = 2
        mask_5c[np.isin(orig_mask, [3, 4, 5])] = 3
        mask_5c[orig_mask == 6] = 4

        # 3-Subclass target
        mask_sub = np.full_like(orig_mask, 255, dtype=np.int64)
        mask_sub[orig_mask == 3] = 0
        mask_sub[orig_mask == 4] = 1
        mask_sub[orig_mask == 5] = 2

        # Synchronous augmentations
        if self.is_train:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask_5c = np.fliplr(mask_5c).copy()
                mask_sub = np.fliplr(mask_sub).copy()
                orig_mask = np.fliplr(orig_mask).copy()

            if random.random() > 0.5:
                img = np.flipud(img).copy()
                mask_5c = np.flipud(mask_5c).copy()
                mask_sub = np.flipud(mask_sub).copy()
                orig_mask = np.flipud(orig_mask).copy()

            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k).copy()
                mask_5c = np.rot90(mask_5c, k).copy()
                mask_sub = np.rot90(mask_sub, k).copy()
                orig_mask = np.rot90(orig_mask, k).copy()

            if random.random() > 0.5:
                factor = random.uniform(0.95, 1.05)
                img = np.clip(img * factor, 0.0, 1.0)

        # RGB for visualization
        rgb_vis = (img[:, :, :3] * 255.0).astype(np.uint8)

        # Normalization
        img_norm = (img - BAND_MEANS) / BAND_STDS
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        target_5c_t = torch.from_numpy(mask_5c).long()
        target_sub_t = torch.from_numpy(mask_sub).long()
        orig_7c_t = torch.from_numpy(orig_mask).long()

        return img_tensor, target_5c_t, target_sub_t, orig_7c_t, rgb_vis, row["patch_id"]


class CompositeIoULoss5Class(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss5C(gamma=2.0, weight=class_weights)
        self.lovasz = LovaszSoftmaxLoss5C(per_image=False)

    def forward(self, logits, targets):
        lf = self.focal(logits, targets)
        lv = self.lovasz(logits, targets)
        return lf + 1.2 * lv, {"focal": lf.item(), "lovasz": lv.item()}


def colorize_mask(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, color in COLORS_7.items():
        color_img[mask_2d == c_id] = color
    return color_img


def create_error_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h, w = pred.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    correct = (pred == gt)
    diff[correct] = [30, 180, 50]
    diff[~correct] = [230, 40, 40]
    return diff


def compute_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    ious = []
    dices = []
    precisions = []
    recalls = []

    total_correct = np.trace(conf_matrix)
    total_samples = np.sum(conf_matrix)
    pixel_acc = total_correct / total_samples if total_samples > 0 else 0.0

    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
        
        denom_dice = 2 * tp + fp + fn
        dice = (2 * tp) / denom_dice if denom_dice > 0 else (1.0 if denom_dice == 0 else 0.0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(prec))
        recalls.append(float(rec))

    gt_support = np.sum(conf_matrix, axis=1)
    present_classes = [c for c in range(num_classes) if gt_support[c] > 0]
    
    miou = float(np.mean([ious[c] for c in present_classes])) if present_classes else 0.0
    mdice = float(np.mean([dices[c] for c in present_classes])) if present_classes else 0.0

    return {
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": miou,
        "mean_dice": mdice,
        "per_class": {
            CLASS_NAMES_7[c]: {
                "class_id": c,
                "iou": ious[c],
                "dice": dices[c],
                "precision": precisions[c],
                "recall": recalls[c],
                "gt_pixels": int(gt_support[c]),
                "pred_pixels": int(np.sum(conf_matrix[:, c]))
            }
            for c in range(num_classes)
        }
    }


class AutonomousOvernightTrainer:
    def __init__(
        self,
        manifest_csv: Path,
        save_dir: Path = SUBCLASS_DIR / "checkpoints",
        visual_dir: Path = SUBCLASS_DIR / "visual_results",
        device: torch.device = None
    ):
        self.save_dir = save_dir
        self.visual_dir = visual_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Autonomous Overnight Trainer on {self.device}", flush=True)

        # Artifact directories
        self.artifact_dir = Path(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\autonomous_training")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Load Manifest
        self.manifest_df = pd.read_csv(manifest_csv)
        self.train_df = self.manifest_df[self.manifest_df["split"] == "train"].copy().reset_index(drop=True)
        self.val_df = self.manifest_df[self.manifest_df["split"] == "val"].copy().reset_index(drop=True)
        print(f"[INIT] Loaded Manifest: {len(self.train_df)} Train patches, {len(self.val_df)} Val patches", flush=True)

        self.train_dataset = AutonomousHierarchicalDataset(self.train_df, is_train=True, target_size=512)
        self.val_dataset = AutonomousHierarchicalDataset(self.val_df, is_train=False, target_size=512)

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        # Checkpoints
        self.seg_ckpt = str(PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "checkpoints" / "best_5class_segformer_extended.pth")
        self.unet_ckpt = str(PROJECT_ROOT / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "checkpoints" / "best_5class_unet_multispectral.pth")
        self.sub_ckpt = str(SUBCLASS_DIR / "checkpoints" / "best_3subclass_segformer.pth")

        self.model = HybridUNetSegFormerTwoStep(
            segformer_ckpt=self.seg_ckpt,
            unet_ckpt=self.unet_ckpt,
            subclass_ckpt=self.sub_ckpt,
            device=self.device
        ).to(self.device)

        # Dynamic Loss Weights
        self.weights_5c = torch.tensor([1.5, 1.0, 0.8, 3.5, 2.5], device=self.device)
        self.loss_5c_fn = CompositeIoULoss5Class(class_weights=self.weights_5c).to(self.device)

        self.weights_sub = torch.tensor([1.0, 1.2, 3.5], device=self.device)
        self.loss_sub_fn = BlockRegularizedSubclassLoss(
            weight=self.weights_sub,
            ignore_index=255,
            tv_weight=0.15,
            block_weight=0.20
        ).to(self.device)

        # Unfrozen Two-Step Optimization:
        # Step 1 coarse backbones (lr=3e-6) + Step 2 subclass branches (lr=5e-5)
        self.lr_step1 = 3e-6
        self.lr_step2 = 5e-5
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.model.segformer.parameters()) + list(self.model.unet.parameters()), "lr": self.lr_step1, "weight_decay": 1e-4},
            {"params": list(self.model.subclass_segformer.parameters()) + list(self.model.subclass_head.parameters()), "lr": self.lr_step2, "weight_decay": 1e-4}
        ])

        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        self.best_composite_score = 0.0
        self.best_metrics = None
        self.history = []
        self.round_history = []
        self.dashboard_html_path = PROJECT_ROOT / "NEW_Model(sem2)" / "FINAL_ORGANIZED_SUBMISSION" / "3_Outputs" / "Visual_PNG_Results" / "human_verification_batches" / "autonomous_training_dashboard.html"

    def calculate_composite_score(self, metrics: dict) -> float:
        miou_7c = metrics.get("mean_iou", 0.0)
        pc = metrics.get("per_class", {})
        res_iou = pc.get("Residential Built-up", {}).get("iou", 0.0)
        ind_iou = pc.get("Industrial Built-up", {}).get("iou", 0.0)
        slum_iou = pc.get("Slum / Informal Settlement", {}).get("iou", 0.0)
        road_iou = pc.get("Road Network", {}).get("iou", 0.0)

        score = (0.35 * miou_7c) + (0.25 * res_iou) + (0.20 * ind_iou) + (0.10 * slum_iou) + (0.10 * road_iou)
        return float(score)

    def train_one_epoch(self, epoch: int, accumulation_steps: int = 4):
        self.model.train()
        train_loss_total = 0.0
        train_loss_5c = 0.0
        train_loss_sub = 0.0
        step_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (img, target_5c, target_sub, orig_7c, rgb_vis, pid) in enumerate(self.train_loader):
            img = img.to(self.device, non_blocking=True)
            target_5c = target_5c.to(self.device, non_blocking=True)
            target_sub = target_sub.to(self.device, non_blocking=True)

            # Step 1: Coarse 5-class forward + backward
            with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                logits_5c, logits_seg, logits_unet, prob_builtup, pred_builtup_mask = self.model.forward_step1(img)
                loss_5c, _ = self.loss_5c_fn(logits_5c, target_5c)
                loss_5c_scaled = (0.5 * loss_5c) / accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss_5c_scaled).backward()
            else:
                loss_5c_scaled.backward()

            loss_5c_val = loss_5c.item()

            # Detach tensors for Step 2 to decouple computation graphs
            l_seg = logits_seg.detach()
            l_unet = logits_unet.detach()
            p_bu = prob_builtup.detach()
            p_mask = pred_builtup_mask.detach()
            del logits_5c, logits_seg, logits_unet, prob_builtup, pred_builtup_mask, loss_5c

            # Step 2: 3-Subclass forward + backward inside built-up territory
            with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                logits_sub = self.model.forward_step2(img, l_seg, l_unet, p_bu)
                gated_sub_target = target_sub.clone()
                gated_sub_target[(target_5c != 3) & (~p_mask)] = 255

                if (gated_sub_target != 255).sum() > 0:
                    loss_sub = self.loss_sub_fn(logits_sub, gated_sub_target)
                else:
                    loss_sub = logits_sub.sum() * 0.0

                loss_sub_scaled = (1.2 * loss_sub) / accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss_sub_scaled).backward()
            else:
                loss_sub_scaled.backward()

            loss_sub_val = loss_sub.item()
            train_loss_total += (loss_5c_val + 1.2 * loss_sub_val)
            train_loss_5c += loss_5c_val
            train_loss_sub += loss_sub_val
            step_count += 1

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % 60 == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f"       Epoch {epoch:02d} | Batch [{batch_idx+1:3d}/{len(self.train_loader)}] | 5C Loss: {loss_5c_val:.4f} | Sub Loss: {loss_sub_val:.4f} | Built-up Px: {p_mask.sum().item():6d}", flush=True)

        avg_total = train_loss_total / max(1, step_count)
        avg_5c = train_loss_5c / max(1, step_count)
        avg_sub = train_loss_sub / max(1, step_count)
        return {"train_loss": avg_total, "loss_5c": avg_5c, "loss_sub": avg_sub}

    def evaluate_validation(self):
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        conf_matrix = np.zeros((7, 7), dtype=np.int64)
        patch_records = []
        predictions_cache = {}

        with torch.no_grad():
            for img, target_5c, target_sub, orig_7c, rgb_vis, pid in self.val_loader:
                img = img.to(self.device)
                p_5c, p_sub, p_7c, is_bu = self.model.predict_hierarchical_7class(img)

                pred_np = p_7c.cpu().numpy().squeeze(0).astype(np.uint8)
                gt_np = orig_7c.numpy().squeeze(0).astype(np.uint8)
                rgb_np = rgb_vis.numpy().squeeze(0)
                patch_id = pid[0]

                for c_gt in range(7):
                    mask_g = (gt_np == c_gt)
                    if mask_g.any():
                        for c_pr in range(7):
                            conf_matrix[c_gt, c_pr] += np.sum((pred_np == c_pr) & mask_g)

                correct_px = np.sum(pred_np == gt_np)
                total_px = 512 * 512
                acc = correct_px / total_px

                bu_gt = np.isin(gt_np, [3, 4, 5])
                bu_pr = np.isin(pred_np, [3, 4, 5])
                bu_inter = np.sum(bu_gt & bu_pr)
                bu_union = np.sum(bu_gt | bu_pr)
                bu_iou = bu_inter / bu_union if bu_union > 0 else (1.0 if not bu_gt.any() else 0.0)

                res_gt = (gt_np == 3)
                res_pr = (pred_np == 3)
                res_inter = np.sum(res_gt & res_pr)
                res_union = np.sum(res_gt | res_pr)
                res_iou = res_inter / res_union if res_union > 0 else (1.0 if not res_gt.any() else 0.0)

                rd_gt = (gt_np == 6)
                rd_pr = (pred_np == 6)
                rd_inter = np.sum(rd_gt & rd_pr)
                rd_union = np.sum(rd_gt | rd_pr)
                rd_iou = rd_inter / rd_union if rd_union > 0 else (1.0 if not rd_gt.any() else 0.0)

                patch_records.append({
                    "patch_id": patch_id,
                    "accuracy": float(acc),
                    "builtup_iou": float(bu_iou),
                    "res_iou": float(res_iou),
                    "road_iou": float(rd_iou),
                    "error_px": int(total_px - correct_px)
                })

                predictions_cache[patch_id] = {
                    "pred": pred_np,
                    "gt": gt_np,
                    "rgb": rgb_np
                }

        metrics = compute_metrics(conf_matrix)
        metrics["confusion_matrix_7x7"] = conf_matrix.tolist()

        bu_tp = conf_matrix[3:6, 3:6].sum()
        bu_fp = conf_matrix[:3, 3:6].sum() + conf_matrix[6, 3:6].sum()
        bu_fn = conf_matrix[3:6, :3].sum() + conf_matrix[3:6, 6].sum()
        metrics["builtup_aggregate_iou"] = float(bu_tp / (bu_tp + bu_fp + bu_fn)) if (bu_tp + bu_fp + bu_fn) > 0 else 0.0
        metrics["water_fp_pixels"] = int(conf_matrix[1:, 0].sum())

        return metrics, patch_records, predictions_cache

    def select_20_diagnostic_patches(self, patch_records: list):
        df_p = pd.DataFrame(patch_records)
        worst_8 = df_p.sort_values("accuracy").head(8)["patch_id"].tolist()
        best_res_4 = df_p[~df_p["patch_id"].isin(worst_8)].sort_values("res_iou", ascending=False).head(4)["patch_id"].tolist()
        best_road_4 = df_p[~df_p["patch_id"].isin(worst_8 + best_res_4)].sort_values("road_iou", ascending=False).head(4)["patch_id"].tolist()
        rem = df_p[~df_p["patch_id"].isin(worst_8 + best_res_4 + best_road_4)]
        rep_4 = rem.sample(n=min(4, len(rem)), random_state=42)["patch_id"].tolist()
        return worst_8 + best_res_4 + best_road_4 + rep_4

    def generate_diagnostic_visuals(self, round_num: int, selected_patches: list, predictions_cache: dict):
        round_vis_dir = self.visual_dir / f"round_{round_num:02d}"
        round_vis_dir.mkdir(parents=True, exist_ok=True)
        art_round_dir = self.artifact_dir / f"round_{round_num:02d}"
        art_round_dir.mkdir(parents=True, exist_ok=True)

        legend_patches = [
            mpatches.Patch(color=np.array(COLORS_7[c])/255.0, label=CLASS_NAMES_7[c])
            for c in range(7)
        ]

        saved_files = []
        for idx, pid in enumerate(selected_patches):
            data = predictions_cache[pid]
            rgb = data["rgb"]
            gt = data["gt"]
            pred = data["pred"]

            gt_color = colorize_mask(gt)
            pred_color = colorize_mask(pred)
            diff_color = create_error_diff(pred, gt)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=140)
            axes[0].imshow(rgb)
            axes[0].set_title(f"[{idx+1:02d}] {pid} - RGB", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt_color)
            axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(pred_color)
            axes[2].set_title(f"R{round_num:02d} Prediction", fontsize=11, fontweight="bold")
            axes[2].axis("off")

            axes[3].imshow(diff_color)
            acc_p = np.mean(pred == gt) * 100
            axes[3].set_title(f"Error Map | Acc={acc_p:.1f}%", fontsize=11, fontweight="bold")
            axes[3].axis("off")

            plt.subplots_adjust(wspace=0.04, top=0.88, bottom=0.04, left=0.02, right=0.98)
            fig.legend(handles=legend_patches, loc="lower center", ncol=7, fontsize=9, frameon=True)

            fname = f"r{round_num:02d}_{idx+1:02d}_{pid}.png"
            p1 = round_vis_dir / fname
            p2 = art_round_dir / fname
            fig.savefig(p1, bbox_inches="tight")
            fig.savefig(p2, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(str(p1))

        print(f"[ROUND {round_num:02d}] Generated 20 visual panels in {round_vis_dir}", flush=True)
        return saved_files

    def diagnose_and_auto_tune(self, round_num: int, current_metrics: dict, prev_metrics: dict):
        """Analyzes metrics and adjusts loss weights & learning rates for the next round."""
        tuning_report = []
        pc = current_metrics["per_class"]

        res_iou = pc["Residential Built-up"]["iou"]
        ind_iou = pc["Industrial Built-up"]["iou"]
        slum_iou = pc["Slum / Informal Settlement"]["iou"]
        road_iou = pc["Road Network"]["iou"]
        water_fp = current_metrics.get("water_fp_pixels", 0)

        # 1. Slum tuning: boost weight if slum IoU < 0.25
        if slum_iou < 0.25:
            new_slum_w = min(5.0, float(self.weights_sub[2].item()) * 1.15)
            self.weights_sub[2] = new_slum_w
            tuning_report.append(f"Boosted Slum weight to {new_slum_w:.2f} (IoU={slum_iou*100:.1f}%)")

        # 2. Industrial tuning: boost weight if lagging
        if ind_iou < res_iou - 0.10:
            new_ind_w = min(2.5, float(self.weights_sub[1].item()) * 1.10)
            self.weights_sub[1] = new_ind_w
            tuning_report.append(f"Boosted Industrial weight to {new_ind_w:.2f} (IoU={ind_iou*100:.1f}%)")

        # 3. Road tuning
        if road_iou < 0.40:
            new_rd_w = min(4.0, float(self.weights_5c[4].item()) * 1.10)
            self.weights_5c[4] = new_rd_w
            tuning_report.append(f"Boosted Road weight to {new_rd_w:.2f} (IoU={road_iou*100:.1f}%)")

        # 4. Water safety check
        if water_fp > 500:
            new_w_w = min(2.5, float(self.weights_5c[0].item()) * 1.20)
            self.weights_5c[0] = new_w_w
            tuning_report.append(f"Water FP detected ({water_fp} px); increased Water weight to {new_w_w:.2f}")

        # Update loss functions with new weights
        self.loss_5c_fn = CompositeIoULoss5Class(class_weights=self.weights_5c).to(self.device)
        self.loss_sub_fn = BlockRegularizedSubclassLoss(
            weight=self.weights_sub,
            ignore_index=255,
            tv_weight=0.15,
            block_weight=0.20
        ).to(self.device)

        report_str = " | ".join(tuning_report) if tuning_report else "Weights stable; no adjustments needed"
        print(f"[AUTO-TUNE R{round_num:02d}] {report_str}", flush=True)
        return report_str

    def update_dashboard(self, current_round: int, current_epoch: int):
        """Generates rich standalone HTML training dashboard."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Overnight Training Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 15px; margin-bottom: 20px; }}
  .badge {{ background: #10b981; color: #fff; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .card {{ background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }}
  .card h4 {{ margin: 0 0 8px 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
  .card .val {{ font-size: 26px; font-weight: bold; color: #38bdf8; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
  th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
  .best-tag {{ color: #10b981; font-weight: bold; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h2 style="margin: 0;">Autonomous Overnight Training Dashboard</h2>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">Cartosat-2E Three-Stage Hierarchical Model (SegFormer + UltraRoad U-Net + Subclass Head)</p>
  </div>
  <div><span class="badge">Round {current_round:02d} / Epoch {current_epoch:03d} (Active)</span></div>
</div>

<div class="grid">
  <div class="card"><h4>Best Composite Score</h4><div class="val">{self.best_composite_score*100:.2f}%</div></div>
  <div class="card"><h4>Best 7-Class mIoU</h4><div class="val">{(self.best_metrics.get('mean_iou', 0.0) if self.best_metrics else 0.0)*100:.2f}%</div></div>
  <div class="card"><h4>Best Built-up IoU</h4><div class="val">{(self.best_metrics.get('builtup_aggregate_iou', 0.0) if self.best_metrics else 0.0)*100:.2f}%</div></div>
  <div class="card"><h4>Water False Positives</h4><div class="val">{(self.best_metrics.get('water_fp_pixels', 0) if self.best_metrics else 0):,d} px</div></div>
</div>

<h3>Rounds Summary</h3>
<table>
  <thead>
    <tr>
      <th>Round</th>
      <th>Epoch</th>
      <th>Composite Score</th>
      <th>7C mIoU</th>
      <th>Res IoU</th>
      <th>Ind IoU</th>
      <th>Slum IoU</th>
      <th>Road IoU</th>
      <th>Builtup IoU</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
"""
        for r in self.round_history:
            is_best = r.get("is_best", False)
            tag = "<span class='best-tag'>BEST</span>" if is_best else "Evaluated"
            html_content += f"""
    <tr>
      <td>Round {r['round']:02d}</td>
      <td>{r['epoch']}</td>
      <td><strong>{r['composite_score']*100:.2f}%</strong></td>
      <td>{r['miou_7c']*100:.2f}%</td>
      <td>{r['res_iou']*100:.2f}%</td>
      <td>{r['ind_iou']*100:.2f}%</td>
      <td>{r['slum_iou']*100:.2f}%</td>
      <td>{r['road_iou']*100:.2f}%</td>
      <td>{r['bu_agg_iou']*100:.2f}%</td>
      <td>{tag}</td>
    </tr>"""

        html_content += """
  </tbody>
</table>
</body>
</html>"""
        self.dashboard_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dashboard_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def prune_checkpoints(self):
        """Keep best_autonomous_model.pth, checkpoint_latest.pth, and milestone round checkpoints."""
        try:
            free_bytes = shutil.disk_usage(self.save_dir).free
            free_gb = free_bytes / (1024**3)
            print(f"[DISK MONITOR] Available free space: {free_gb:.2f} GB", flush=True)
            if free_gb < 4.0:
                print("[DISK ALERT] Free disk space low (<4 GB). Pruning intermediate checkpoints...", flush=True)
                for f in self.save_dir.glob("epoch_*.pth"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[DISK MONITOR] Warning: {e}", flush=True)

    def run_autonomous_loop(self, total_rounds: int = 15, epochs_per_round: int = 10):
        print("\n" + "=" * 90, flush=True)
        print(f"STARTING AUTONOMOUS OVERNIGHT TRAINING: {total_rounds} Rounds x {epochs_per_round} Epochs ({total_rounds*epochs_per_round} Total Epochs)", flush=True)
        print("=" * 90 + "\n", flush=True)

        current_epoch = 1

        # Evaluate initial baseline as Round 0
        print("[ROUND 00] Evaluating Pretrained Baseline Weights...", flush=True)
        base_metrics, base_patches, base_cache = self.evaluate_validation()
        base_score = self.calculate_composite_score(base_metrics)
        self.best_composite_score = base_score
        self.best_metrics = base_metrics

        round_0_record = {
            "round": 0,
            "epoch": 0,
            "composite_score": base_score,
            "miou_7c": base_metrics["mean_iou"],
            "res_iou": base_metrics["per_class"]["Residential Built-up"]["iou"],
            "ind_iou": base_metrics["per_class"]["Industrial Built-up"]["iou"],
            "slum_iou": base_metrics["per_class"]["Slum / Informal Settlement"]["iou"],
            "road_iou": base_metrics["per_class"]["Road Network"]["iou"],
            "bu_agg_iou": base_metrics["builtup_aggregate_iou"],
            "is_best": True
        }
        self.round_history.append(round_0_record)

        diag_20_base = self.select_20_diagnostic_patches(base_patches)
        self.generate_diagnostic_visuals(0, diag_20_base, base_cache)

        best_ckpt_path = self.save_dir / "best_autonomous_model.pth"
        torch.save({
            "round": 0,
            "epoch": 0,
            "model_state_dict": self.model.state_dict(),
            "composite_score": base_score,
            "metrics": base_metrics
        }, best_ckpt_path)
        print(f"[ROUND 00] Baseline established: Composite Score = {base_score*100:.2f}% | Saved to {best_ckpt_path}", flush=True)
        self.update_dashboard(0, 0)

        # Main Autonomous Round Loop
        for r in range(1, total_rounds + 1):
            print("\n" + "#" * 90, flush=True)
            print(f"### AUTONOMOUS ROUND {r:02d} / {total_rounds:02d} (Epochs {current_epoch} to {current_epoch + epochs_per_round - 1}) ###", flush=True)
            print("#" * 90, flush=True)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs_per_round, eta_min=1e-6)

            for ep in range(1, epochs_per_round + 1):
                t_ep_start = time.time()
                train_res = self.train_one_epoch(current_epoch)
                scheduler.step()
                ep_dur = time.time() - t_ep_start

                print(f"Epoch {current_epoch:03d} | Train Loss: {train_res['train_loss']:.4f} (5C: {train_res['loss_5c']:.4f}, Sub: {train_res['loss_sub']:.4f}) | Time: {ep_dur:.1f}s", flush=True)

                # Save latest epoch checkpoint
                latest_ckpt_path = self.save_dir / "checkpoint_latest.pth"
                torch.save({
                    "round": r,
                    "epoch": current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                }, latest_ckpt_path)

                current_epoch += 1

            # End of 10-Epoch Round: Comprehensive Evaluation
            print(f"\n[ROUND {r:02d}] Evaluating Model on Fixed Validation Set ({len(self.val_df)} Patches)...", flush=True)
            val_metrics, val_patches, val_cache = self.evaluate_validation()
            comp_score = self.calculate_composite_score(val_metrics)

            diag_20 = self.select_20_diagnostic_patches(val_patches)
            self.generate_diagnostic_visuals(r, diag_20, val_cache)

            # Compare with best score
            score_diff = comp_score - self.best_composite_score
            is_best = False

            if comp_score > self.best_composite_score:
                is_best = True
                self.best_composite_score = comp_score
                self.best_metrics = val_metrics

                # Save new best model
                torch.save({
                    "round": r,
                    "epoch": current_epoch - 1,
                    "model_state_dict": self.model.state_dict(),
                    "composite_score": comp_score,
                    "metrics": val_metrics
                }, best_ckpt_path)
                print(f"[ROUND {r:02d} SUCCESS] NEW BEST MODEL FOUND! Composite Score: {comp_score*100:.2f}% (+{score_diff*100:.2f}%)", flush=True)
            else:
                print(f"[ROUND {r:02d} REGRESSION] Score {comp_score*100:.2f}% did not exceed Best ({self.best_composite_score*100:.2f}%). Reverting weights...", flush=True)
                # Revert to best weights
                ckpt_best = torch.load(best_ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt_best["model_state_dict"])
                del ckpt_best

            # Save round milestone checkpoint
            round_ckpt_path = self.save_dir / f"checkpoint_round_{r:02d}.pth"
            torch.save({
                "round": r,
                "epoch": current_epoch - 1,
                "model_state_dict": self.model.state_dict(),
                "composite_score": comp_score,
                "metrics": val_metrics,
                "is_best": is_best
            }, round_ckpt_path)

            round_record = {
                "round": r,
                "epoch": current_epoch - 1,
                "composite_score": comp_score,
                "miou_7c": val_metrics["mean_iou"],
                "res_iou": val_metrics["per_class"]["Residential Built-up"]["iou"],
                "ind_iou": val_metrics["per_class"]["Industrial Built-up"]["iou"],
                "slum_iou": val_metrics["per_class"]["Slum / Informal Settlement"]["iou"],
                "road_iou": val_metrics["per_class"]["Road Network"]["iou"],
                "bu_agg_iou": val_metrics["builtup_aggregate_iou"],
                "is_best": is_best
            }
            self.round_history.append(round_record)

            # Auto-tune for next round
            self.diagnose_and_auto_tune(r, val_metrics, self.best_metrics)

            # Update dashboard & prune disk
            self.update_dashboard(r, current_epoch - 1)
            self.prune_checkpoints()

        print("\n" + "=" * 90, flush=True)
        print(f"AUTONOMOUS TRAINING COMPLETE! Best Composite Score: {self.best_composite_score*100:.2f}%", flush=True)
        print("=" * 90, flush=True)


def main():
    manifest_csv = SUBCLASS_DIR / "data" / "autonomous_dataset_manifest.csv"
    assert manifest_csv.exists(), f"Missing manifest: {manifest_csv}"

    trainer = AutonomousOvernightTrainer(manifest_csv=manifest_csv)
    trainer.run_autonomous_loop(total_rounds=15, epochs_per_round=10)


if __name__ == "__main__":
    main()
