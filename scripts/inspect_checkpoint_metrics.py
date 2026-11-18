import torch
import json
from pathlib import Path

p = Path("NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_autonomous_model.pth")
ckpt = torch.load(str(p), map_location='cpu')

print("=" * 70)
print(f"CHECKPOINT: {p.name}")
print(f"Round: {ckpt.get('round')}")
print(f"Epoch: {ckpt.get('epoch')}")
print(f"Composite Score: {ckpt.get('composite_score', 0.0)*100:.2f}%")
print("=" * 70)

m = ckpt.get('metrics', {})
print(f"Pixel Accuracy:          {m.get('pixel_accuracy', 0.0)*100:.2f}%")
print(f"Mean IoU (7 Classes):    {m.get('mean_iou', 0.0)*100:.2f}%")
print(f"Mean Dice:               {m.get('mean_dice', 0.0)*100:.2f}%")
print(f"Built-up Aggregate IoU:  {m.get('builtup_aggregate_iou', 0.0)*100:.2f}%")
print(f"Water False Positives:   {m.get('water_fp_pixels', 0):,d} px")
print("-" * 70)
print(f"{'Class':30s} | {'IoU (%)':8s} | {'Dice (%)':8s} | {'Prec (%)':8s} | {'Rec (%)':8s} | {'GT Pixels':11s} | {'Pred Pixels':11s}")
print("-" * 70)

for c_name, c_data in m.get('per_class', {}).items():
    iou = c_data.get('iou', 0.0) * 100
    dice = c_data.get('dice', 0.0) * 100
    prec = c_data.get('precision', 0.0) * 100
    rec = c_data.get('recall', 0.0) * 100
    gt_px = c_data.get('gt_pixels', 0)
    pr_px = c_data.get('pred_pixels', 0)
    print(f"{c_name:30s} | {iou:8.2f} | {dice:8.2f} | {prec:8.2f} | {rec:8.2f} | {gt_px:11,d} | {pr_px:11,d}")

print("=" * 70)

