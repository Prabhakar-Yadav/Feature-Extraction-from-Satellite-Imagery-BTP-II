import os
import cv2
import numpy as np
import datetime

v3_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v3_human_verified"
v1_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v1_original"
v4_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v4_learned_candidate"

CLASS_NAMES = {
    0: "Water", 1: "Vegetation", 2: "Barren Land",
    3: "Residential", 4: "Industrial", 5: "Slum / Informal",
    6: "Road Network", 255: "Ignore / Uncertain"
}

for pid in ["test_pune_0572", "test_pune_0550", "test_pune_0408"]:
    print(f"\n==========================================")
    print(f"PATCH: {pid}")
    print(f"==========================================")
    f3 = os.path.join(v3_dir, f"{pid}.png")
    f1 = os.path.join(v1_dir, f"{pid}.png")
    f4 = os.path.join(v4_dir, f"{pid}.png")
    
    mtime3 = datetime.datetime.fromtimestamp(os.path.getmtime(f3))
    print(f"Latest Human-Verified Mask timestamp: {mtime3}")
    
    m3 = cv2.imread(f3, cv2.IMREAD_GRAYSCALE)
    m1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    m4 = cv2.imread(f4, cv2.IMREAD_GRAYSCALE)
    
    print("\nClass breakdown (pixels and %):")
    print(f"{'Class ID & Name':<28} | {'Original (v1)':<16} | {'Learned (v4)':<16} | {'User Human (v3)':<16}")
    print("-" * 84)
    for c in [0, 1, 2, 3, 4, 5, 6, 255]:
        c1 = (m1 == c).sum()
        c4 = (m4 == c).sum()
        c3 = (m3 == c).sum()
        if c1 > 0 or c4 > 0 or c3 > 0:
            name = f"{c}: {CLASS_NAMES.get(c, 'Unknown')}"
            print(f"{name:<28} | {c1:>6} ({c1/m1.size*100:>5.2f}%) | {c4:>6} ({c4/m4.size*100:>5.2f}%) | {c3:>6} ({c3/m3.size*100:>5.2f}%)")
    
    # Differences
    diff_3_vs_4 = (m3 != m4).sum()
    diff_3_vs_1 = (m3 != m1).sum()
    print(f"\nDifferences:")
    print(f"  v3 (Human) vs v4 (Candidate): {diff_3_vs_4} pixels changed ({diff_3_vs_4/m3.size*100:.2f}%)")
    print(f"  v3 (Human) vs v1 (Original):  {diff_3_vs_1} pixels changed ({diff_3_vs_1/m3.size*100:.2f}%)")

