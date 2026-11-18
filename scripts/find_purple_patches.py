import os
import cv2
import numpy as np
import glob

# Load media_1793989943145_74b1352e.jpg
up1_p = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg"
up1 = cv2.imread(up1_p)
print("up1 shape:", up1.shape)

# Let's extract purple pixels in up1:
# In the studio palette, purple is #8C32BE -> BGR: [190, 50, 140]
# With alpha 0.45 blending over image:
# Blue is moderate/high, Red is moderate/high, Green is low!
b, g, r = cv2.split(up1)
purple_mask = (b > 100) & (r > 80) & (g < 60)
print(f"Total purple pixels in up1: {purple_mask.sum()}")

# Let's check which patches have a large contiguous purple region in masks_v4_learned_candidate or masks_v1_original:
base_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data"
for f in sorted(glob.glob(os.path.join(base_dir, "masks_v4_learned_candidate", "*.png"))):
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    ind_count = (m == 4).sum()
    if ind_count > 5000:
        pid = os.path.basename(f).replace(".png", "")
        print(f"Candidate {pid}: Industrial (class 4) = {ind_count} pixels ({ind_count/m.size*100:.2f}%)")

for f in sorted(glob.glob(os.path.join(base_dir, "masks_v1_original", "*.png"))):
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    ind_count = (m == 4).sum()
    if ind_count > 5000:
        pid = os.path.basename(f).replace(".png", "")
        print(f"Original {pid}: Industrial (class 4) = {ind_count} pixels ({ind_count/m.size*100:.2f}%)")

