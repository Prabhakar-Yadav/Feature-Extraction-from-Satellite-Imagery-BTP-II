import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path('.').resolve()
SUBCLASS_DIR = PROJECT_ROOT / 'NEW_Model(sem2)' / 'Phase_2_Model_Training' / '04_Multispectral_Subclass_Segmentation'
STAGE1_DIR = PROJECT_ROOT / 'NEW_Model(sem2)' / 'Phase_2_Model_Training' / '05_Multispectral_5Class_Training'

sys.path.insert(0, str(SUBCLASS_DIR / 'scripts'))
sys.path.insert(0, str(STAGE1_DIR / 'scripts'))

from dataset_subclasses import BAND_MEANS, BAND_STDS, ORIG_CLASS_NAMES, ORIG_CLASS_COLORS
from hierarchical_pipeline import HierarchicalUrbanPipeline
from models_hybrid_hierarchical import HybridUNetSegFormerTwoStep

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[INVESTIGATION] Running on {device}...')

# Checkpoints
OLD_SEG_CKPT = STAGE1_DIR / 'checkpoints' / 'best_5class_segformer_extended.pth'
OLD_UNET_CKPT = STAGE1_DIR / 'checkpoints' / 'best_5class_unet_multispectral.pth'
OLD_SUB_CKPT = SUBCLASS_DIR / 'checkpoints' / 'best_3subclass_segformer.pth'
R05_CKPT = SUBCLASS_DIR / 'checkpoints' / 'best_autonomous_model.pth'

print('Loading Previous Model (Model A)...')
old_pipeline = HierarchicalUrbanPipeline(
    stage1_segformer_ckpt=OLD_SEG_CKPT,
    stage1_unet_ckpt=OLD_UNET_CKPT,
    stage2_subclass_ckpt=OLD_SUB_CKPT,
    device=device
)

print('Loading R05 Model (Model B)...')
r05_model = HybridUNetSegFormerTwoStep(
    segformer_ckpt=str(OLD_SEG_CKPT),
    unet_ckpt=str(OLD_UNET_CKPT),
    subclass_ckpt=str(OLD_SUB_CKPT),
    device=device
).to(device)

sd_r05 = torch.load(R05_CKPT, map_location=device, weights_only=False)
r05_model.load_state_dict(sd_r05['model_state_dict'])
r05_model.eval()

# Manifest
manifest_path = SUBCLASS_DIR / 'data' / 'autonomous_dataset_manifest.csv'
df_manifest = pd.read_csv(manifest_path)
val_df = df_manifest[df_manifest['split'] == 'val'].copy()
print(f'Validation set size: {len(val_df)} patches')

# Palette
PALETTE_RGB = {
    0: (30, 120, 220),    # Water
    1: (34, 150, 40),     # Vegetation
    2: (215, 185, 145),   # Barren Land
    3: (230, 90, 70),     # Residential
    4: (140, 50, 190),    # Industrial
    5: (255, 120, 0),     # Slum
    6: (255, 225, 20)     # Road
}

def mask_to_color(m):
    h, w = m.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in PALETTE_RGB.items():
        c[m == cid] = col
    return c

def load_patch_data(row):
    img_p = Path(row['image_path'])
    mask_p = Path(row['mask_path'])
    bgra = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
    b, g, r, nir = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2], bgra[:, :, 3]
    img_4c = np.stack([r, g, b, nir], axis=-1).astype(np.float32) / 255.0
    gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if img_4c.shape[0] != 512 or img_4c.shape[1] != 512:
        img_4c = cv2.resize(img_4c, (512, 512), interpolation=cv2.INTER_LINEAR)
    if gt_mask.shape[0] != 512 or gt_mask.shape[1] != 512:
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    rgb_vis = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    if rgb_vis.shape[0] != 512 or rgb_vis.shape[1] != 512:
        rgb_vis = cv2.resize(rgb_vis, (512, 512), interpolation=cv2.INTER_LINEAR)
    return img_4c, gt_mask, rgb_vis

# Evaluation helpers
def run_model_a(img_4c):
    res = old_pipeline.predict_patch(img_4c)
    return res['pred_7c']

def run_model_b(img_4c):
    img_norm = (img_4c - BAND_MEANS) / BAND_STDS
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        _, _, p_7c, _ = r05_model.predict_hierarchical_7class(tensor)
    return p_7c.squeeze(0).cpu().numpy()

def run_model_c(img_4c):
    img_norm = (img_4c - BAND_MEANS) / BAND_STDS
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits_5c, logits_sub, pred_bu_mask, prob_bu = r05_model.forward(tensor)
        probs_5c = F.softmax(logits_5c, dim=1)
        pred_5c = torch.argmax(probs_5c, dim=1).squeeze(0).cpu().numpy()
        pred_sub = torch.argmax(logits_sub, dim=1).squeeze(0).cpu().numpy()

        p_7c = np.zeros((512, 512), dtype=np.uint8)
        p_7c[pred_5c == 0] = 0
        p_7c[pred_5c == 1] = 1
        p_7c[pred_5c == 2] = 2
        p_7c[pred_5c == 3] = 3
        p_7c[pred_5c == 4] = 6

        # Subclasses strictly where pred_5c is builtup without 9x9 block closing
        is_bu = (pred_5c == 3)
        p_7c[is_bu & (pred_sub == 0)] = 3
        p_7c[is_bu & (pred_sub == 1)] = 4
        p_7c[is_bu & (pred_sub == 2)] = 5
    return p_7c

print('Beginning 61 validation patch benchmark for Model A vs Model B vs Model C...')

def evaluate_model(pred_func, name):
    num_classes = 7
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0
    for idx, (_, row) in enumerate(val_df.iterrows()):
        img_4c, gt_mask, _ = load_patch_data(row)
        pred = pred_func(img_4c)
        valid = (gt_mask >= 0) & (gt_mask < num_classes)
        cm += np.bincount(num_classes * gt_mask[valid].astype(int) + pred[valid].astype(int),
                          minlength=num_classes**2).reshape(num_classes, num_classes)
        total_correct += np.sum((pred == gt_mask) & valid)
        total_pixels += np.sum(valid)
        if (idx + 1) % 15 == 0:
            print(f'  [{name}] Evaluated {idx+1}/{len(val_df)} patches...')
    
    inter = np.diag(cm)
    gts = cm.sum(axis=1)
    preds = cm.sum(axis=0)
    union = gts + preds - inter
    ious = np.array([inter[c] / union[c] if union[c] > 0 else 1.0 for c in range(num_classes)])
    precisions = np.array([inter[c] / preds[c] if preds[c] > 0 else 0.0 for c in range(num_classes)])
    recalls = np.array([inter[c] / gts[c] if gts[c] > 0 else 0.0 for c in range(num_classes)])
    
    # Built-up aggregate IoU (classes 3, 4, 5)
    gt_bu = np.isin(np.arange(num_classes), [3, 4, 5])
    pred_bu = np.isin(np.arange(num_classes), [3, 4, 5])
    inter_bu = np.sum(cm[3:6, 3:6])
    union_bu = np.sum(cm[3:6, :]) + np.sum(cm[:, 3:6]) - inter_bu
    bu_iou = inter_bu / union_bu if union_bu > 0 else 0.0

    acc = total_correct / total_pixels
    miou_7c = np.mean(ious)
    
    # 5-class macro mapping: Water, Veg, Barren, Built-up, Road
    # Combine classes [3, 4, 5] into class 3, class 6 into class 4
    cm_5c = np.zeros((5, 5), dtype=np.int64)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}
    for r in range(7):
        for c in range(7):
            cm_5c[mapping[r], mapping[c]] += cm[r, c]
    inter_5c = np.diag(cm_5c)
    union_5c = cm_5c.sum(axis=1) + cm_5c.sum(axis=0) - inter_5c
    ious_5c = np.array([inter_5c[c] / union_5c[c] if union_5c[c] > 0 else 1.0 for c in range(5)])
    miou_5c = np.mean(ious_5c)

    return {
        'name': name,
        'acc': acc,
        'miou_7c': miou_7c,
        'miou_5c': miou_5c,
        'bu_iou': bu_iou,
        'ious': ious,
        'precisions': precisions,
        'recalls': recalls,
        'cm': cm
    }

res_a = evaluate_model(run_model_a, 'Model A (Previous High-Performing)')
res_b = evaluate_model(run_model_b, 'Model B (R05 Checkpoint)')
res_c = evaluate_model(run_model_c, 'Model C (Recovered Clean Gating)')

print('\n' + '='*85)
print('COMPREHENSIVE FAIR VALIDATION COMPARISON (61 FIXED VALIDATION PATCHES)')
print('='*85)
print(f"{'Metric':<28} | {'Model A (Previous)':<20} | {'Model B (R05)':<20} | {'Model C (Recovered)':<20}")
print('-'*85)
print(f"{'Overall Pixel Accuracy':<28} | {res_a['acc']*100:>19.2f}% | {res_b['acc']*100:>19.2f}% | {res_c['acc']*100:>19.2f}%")
print(f"{'7-Class Mean IoU':<28} | {res_a['miou_7c']*100:>19.2f}% | {res_b['miou_7c']*100:>19.2f}% | {res_c['miou_7c']*100:>19.2f}%")
print(f"{'5-Class Macro Mean IoU':<28} | {res_a['miou_5c']*100:>19.2f}% | {res_b['miou_5c']*100:>19.2f}% | {res_c['miou_5c']*100:>19.2f}%")
print(f"{'Built-up Aggregate IoU':<28} | {res_a['bu_iou']*100:>19.2f}% | {res_b['bu_iou']*100:>19.2f}% | {res_c['bu_iou']*100:>19.2f}%")
print('-'*85)
for c in range(7):
    cname = ORIG_CLASS_NAMES[c]
    print(f"{cname:<28} | {res_a['ious'][c]*100:>19.2f}% | {res_b['ious'][c]*100:>19.2f}% | {res_c['ious'][c]*100:>19.2f}%")
print('='*85)

# Generate Side-by-Side Visual Comparison Panels for the 4 Focal Patches
four_patches = ['train_hinj_0098', 'train_hinj_0112', 'train_hinj_0009', 'train_chn_0317']
ART_DIR = Path(r'C:/Users/PRABHAKAR/.gemini/antigravity/brain/c89e20d2-5c79-420e-8b8e-ab1621aef4cf/investigation_comparisons')
ART_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path('NEW_Model(sem2)/FINAL_ORGANIZED_SUBMISSION/3_Outputs/Visual_PNG_Results/investigation_comparisons')
OUT_DIR.mkdir(parents=True, exist_ok=True)

legend_patches = [mpatches.Patch(color=np.array(PALETTE_RGB[c])/255.0, label=f'{ORIG_CLASS_NAMES[c]} ({c})') for c in range(7)]

for idx, pid in enumerate(four_patches):
    row = df_manifest[df_manifest['patch_id'] == pid].iloc[0]
    img_4c, gt_mask, rgb_vis = load_patch_data(row)
    
    pred_a = run_model_a(img_4c)
    pred_b = run_model_b(img_4c)
    pred_c = run_model_c(img_4c)

    acc_a = np.mean(pred_a == gt_mask) * 100
    acc_b = np.mean(pred_b == gt_mask) * 100
    acc_c = np.mean(pred_c == gt_mask) * 100

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), dpi=140)
    
    # 1. RGB
    axes[0].imshow(rgb_vis)
    axes[0].set_title(f'1. RGB: {pid}', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(mask_to_color(gt_mask))
    axes[1].set_title('2. Ground Truth', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    # 3. Model A (Previous)
    axes[2].imshow(mask_to_color(pred_a))
    axes[2].set_title(f'3. Previous Model\\n(Acc: {acc_a:.1f}%)', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    # 4. Model B (Current R05)
    axes[3].imshow(mask_to_color(pred_b))
    axes[3].set_title(f'4. R05 Checkpoint\\n(Acc: {acc_b:.1f}%)', fontsize=10, fontweight='bold', color='red')
    axes[3].axis('off')

    # 5. Model C (Recovered)
    axes[4].imshow(mask_to_color(pred_c))
    axes[4].set_title(f'5. Recovered Model\\n(Acc: {acc_c:.1f}%)', fontsize=10, fontweight='bold', color='green')
    axes[4].axis('off')

    # 6. Error Difference Map (Model C vs GT)
    diff = np.zeros((512, 512, 3), dtype=np.uint8)
    diff[pred_c == gt_mask] = (rgb_vis[pred_c == gt_mask] * 0.4).astype(np.uint8)
    diff[pred_c != gt_mask] = [230, 20, 20] # Red error
    axes[5].imshow(diff)
    axes[5].set_title(f'6. Error Map (Model C)', fontsize=10, fontweight='bold')
    axes[5].axis('off')

    plt.subplots_adjust(wspace=0.03, top=0.85, bottom=0.10, left=0.01, right=0.99)
    fig.legend(handles=legend_patches, loc='lower center', ncol=7, fontsize=9, frameon=True)

    fname = f'side_by_side_{idx+1:02d}_{pid}.png'
    p1 = OUT_DIR / fname
    p2 = ART_DIR / fname
    fig.savefig(p1, bbox_inches='tight')
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved visual comparison for {pid} to {p1} and {p2}')

# Dump detailed metrics to json for permanent reference
comparison_data = {
    'model_a': {
        'name': res_a['name'], 'acc': res_a['acc'], 'miou_7c': res_a['miou_7c'], 'miou_5c': res_a['miou_5c'],
        'bu_iou': res_a['bu_iou'], 'ious': res_a['ious'].tolist(), 'precisions': res_a['precisions'].tolist(), 'recalls': res_a['recalls'].tolist()
    },
    'model_b': {
        'name': res_b['name'], 'acc': res_b['acc'], 'miou_7c': res_b['miou_7c'], 'miou_5c': res_b['miou_5c'],
        'bu_iou': res_b['bu_iou'], 'ious': res_b['ious'].tolist(), 'precisions': res_b['precisions'].tolist(), 'recalls': res_b['recalls'].tolist()
    },
    'model_c': {
        'name': res_c['name'], 'acc': res_c['acc'], 'miou_7c': res_c['miou_7c'], 'miou_5c': res_c['miou_5c'],
        'bu_iou': res_c['bu_iou'], 'ious': res_c['ious'].tolist(), 'precisions': res_c['precisions'].tolist(), 'recalls': res_c['recalls'].tolist()
    }
}
import json
with open(OUT_DIR / 'comparison_metrics.json', 'w') as jf:
    json.dump(comparison_data, jf, indent=2)

print('\n[COMPLETE] Investigation execution finished successfully!')
