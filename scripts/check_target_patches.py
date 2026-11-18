import os
import glob
import datetime
import cv2
import numpy as np

v3_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v3_human_verified"

for pid in ["test_pune_0436", "test_pune_0550", "test_pune_0408"]:
    raw_p = os.path.join(v3_dir, f"{pid}.png")
    col_p = os.path.join(v3_dir, f"{pid}_color.png")
    
    print(f"\n=== Patch {pid} in masks_v3_human_verified ===")
    if os.path.exists(raw_p):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(raw_p))
        sz = os.path.getsize(raw_p)
        print(f"Raw mask exists! mtime: {mtime}, size: {sz} bytes")
        img = cv2.imread(raw_p, cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(img, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
    else:
        print(f"Raw mask DOES NOT EXIST at: {raw_p}")
        
    if os.path.exists(col_p):
        mtime_c = datetime.datetime.fromtimestamp(os.path.getmtime(col_p))
        print(f"Color mask exists! mtime: {mtime_c}")
    else:
        print(f"Color mask DOES NOT EXIST at: {col_p}")

# Also check Downloads or other locations for any test_pune_0436 / 0550 / 0408 files
print("\n=== Searching Downloads for these patch IDs ===")
for pid in ["test_pune_0436", "test_pune_0550", "test_pune_0408"]:
    pattern = os.path.join(r"C:\Users\PRABHAKAR\Downloads", f"*{pid}*")
    matches = glob.glob(pattern)
    print(f"Matches for {pid} in Downloads:", matches)

