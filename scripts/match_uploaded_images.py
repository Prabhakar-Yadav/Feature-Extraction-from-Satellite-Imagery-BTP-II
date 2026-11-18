import os
import cv2
import numpy as np
import glob

img1_path = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg"
img2_path = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793990007163_3831517d.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

print("Img1 shape:", img1.shape if img1 is not None else None)
print("Img2 shape:", img2.shape if img2 is not None else None)

# In img1, notice the roads (yellow: BGR ~ [0, 220, 255] or similar) and black padding on the right!
# Let's see: the black padding on the right means it's a cropped canvas or full canvas.
# Let's compare with data/images_rgb/
rgb_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\images_rgb"
all_rgb_files = glob.glob(os.path.join(rgb_dir, "*_rgb.png"))

print(f"Total RGB files in studio: {len(all_rgb_files)}")

# Also check all 186 patches in dataset if needed

