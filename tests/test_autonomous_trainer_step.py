import os
import sys
import time
from pathlib import Path
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation" / "scripts"))
sys.path.insert(0, str(root / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "05_Multispectral_5Class_Training" / "scripts"))

from autonomous_overnight_trainer import AutonomousHierarchicalDataset, AutonomousOvernightTrainer
import pandas as pd

print("[TEST 1] Initializing trainer...", flush=True)
manifest_csv = root / "NEW_Model(sem2)" / "Phase_2_Model_Training" / "04_Multispectral_Subclass_Segmentation" / "data" / "autonomous_dataset_manifest.csv"
trainer = AutonomousOvernightTrainer(manifest_csv=manifest_csv)
print("[TEST 2] Trainer initialized!", flush=True)

img, t5, tsub, o7, rgb, pid = trainer.train_dataset[0]
img = img.unsqueeze(0).to(trainer.device)
t5 = t5.unsqueeze(0).to(trainer.device)
tsub = tsub.unsqueeze(0).to(trainer.device)
print(f"[TEST 3] Batch loaded: {pid}", flush=True)

t0 = time.time()
print("[TEST 4] Running Step 1 forward...", flush=True)
with torch.amp.autocast('cuda'):
    logits_5c, logits_seg, logits_unet, prob_builtup, pred_builtup_mask = trainer.model.forward_step1(img)
    loss_5c, _ = trainer.loss_5c_fn(logits_5c, t5)
print(f"[TEST 5] Step 1 forward done in {time.time()-t0:.2f}s, loss={loss_5c.item():.4f}", flush=True)

t0 = time.time()
print("[TEST 6] Running Step 1 backward...", flush=True)
trainer.scaler.scale(loss_5c).backward()
print(f"[TEST 7] Step 1 backward done in {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
print("[TEST 8] Running Step 2 forward...", flush=True)
l_seg = logits_seg.detach()
l_unet = logits_unet.detach()
p_bu = prob_builtup.detach()
with torch.amp.autocast('cuda'):
    logits_sub = trainer.model.forward_step2(img, l_seg, l_unet, p_bu)
    loss_sub = trainer.loss_sub_fn(logits_sub, tsub)
print(f"[TEST 9] Step 2 forward done in {time.time()-t0:.2f}s, loss={loss_sub.item():.4f}", flush=True)

t0 = time.time()
print("[TEST 10] Running Step 2 backward...", flush=True)
trainer.scaler.scale(loss_sub).backward()
print(f"[TEST 11] Step 2 backward done in {time.time()-t0:.2f}s", flush=True)

trainer.scaler.step(trainer.optimizer)
trainer.scaler.update()
trainer.optimizer.zero_grad()
print("[TEST 12] Optimizer step completed! SUCCESS!", flush=True)
