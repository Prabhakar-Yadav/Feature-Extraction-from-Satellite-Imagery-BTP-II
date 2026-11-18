import os
import cv2
import numpy as np
import glob

# 1. Check screenshots in Pictures/Screenshots
sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc_files = [
    os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png"),
    os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png"),
    os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png"),
]

for p in sc_files:
    if os.path.exists(p):
        img = cv2.imread(p)
        print(f"Screenshot {os.path.basename(p)}: shape {img.shape}")
    else:
        print(f"Not found: {p}")

# 2. Check user_uploaded
up_dir = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded"
for f in os.listdir(up_dir):
    p = os.path.join(up_dir, f)
    img = cv2.imread(p)
    print(f"User uploaded {f}: shape {img.shape if img is not None else None}")

