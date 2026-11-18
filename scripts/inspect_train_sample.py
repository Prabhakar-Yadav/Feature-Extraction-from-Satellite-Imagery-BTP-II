import os
import cv2
import numpy as np
import pandas as pd

train_img_dir = r"NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation\data\train\images"
train_mask_dir = r"NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation\data\train\masks"

img_sample = os.path.join(train_img_dir, "train_hinj_0000.png")
mask_sample = os.path.join(train_mask_dir, "train_hinj_0000.png")

img = cv2.imread(img_sample, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_sample, cv2.IMREAD_UNCHANGED)

print("Train Image shape:", img.shape if img is not None else None)
print("Train Image dtype:", img.dtype if img is not None else None)
print("Train Mask shape:", mask.shape if mask is not None else None)
print("Train Mask dtype:", mask.dtype if mask is not None else None)
print("Train Mask unique classes:", np.unique(mask) if mask is not None else None)

# Check total files in train/images
all_train_imgs = os.listdir(train_img_dir)
all_train_masks = os.listdir(train_mask_dir)
print(f"Total train images: {len(all_train_imgs)}, Total train masks: {len(all_train_masks)}")

