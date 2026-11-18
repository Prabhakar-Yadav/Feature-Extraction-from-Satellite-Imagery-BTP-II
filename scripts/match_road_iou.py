import os
import cv2
import numpy as np

# Load the 3 screenshots
sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # ~00:02
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # ~00:03
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # ~00:08

# In sc1, sc2, sc3, roads are yellow.
# Yellow in BGR: B is low (<80), G is high (>150), R is high (>180)
def extract_yellow(img):
    b, g, r = cv2.split(img)
    mask = (b < 100) & (g > 140) & (r > 170)
    return mask.astype(np.uint8) * 255

y1 = extract_yellow(sc1)
y2 = extract_yellow(sc2)
y3 = extract_yellow(sc3)

visited_patches = [
    "test_pune_0575", "test_pune_0572", "test_pune_0564", "test_pune_0550",
    "test_pune_0560", "test_pune_0516", "test_pune_0471", "test_pune_0460",
    "test_pune_0458", "test_pune_0456", "test_pune_0429", "test_pune_0423",
    "test_pune_0409", "test_pune_0408", "test_pune_0407", "test_pune_0505",
    "test_pune_0506", "test_pune_0507", "test_pune_0508", "test_pune_0538"
]

mask_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v1_original"

for idx, (y_sc, sc_name) in enumerate([(y1, "Screenshot 1"), (y2, "Screenshot 2"), (y3, "Screenshot 3")]):
    print(f"\n--- Matching {sc_name} ---")
    h_s, w_s = y_sc.shape[:2]
    # In the portal, the canvas is 512x512, but it might be drawn with CSS zoom/scale or browser zoom
    # Notice sc1 has a black band on the right! Let's crop out the black band on the right.
    # Where does the content end?
    row_sums = np.sum(sc1 if idx==0 else (sc2 if idx==1 else sc3), axis=(0, 2))
    valid_cols = np.where(row_sums > 0)[0]
    if len(valid_cols) > 0:
        c_min, c_max = valid_cols[0], valid_cols[-1]
    else:
        c_min, c_max = 0, w_s - 1
    
    cropped_y = y_sc[:, c_min:c_max+1]
    # Resize cropped_y to 512x512
    resized_y = cv2.resize(cropped_y, (512, 512))
    
    best_iou = -1
    best_p = None
    for pid in visited_patches:
        mf = os.path.join(mask_dir, f"{pid}.png")
        if not os.path.exists(mf):
            continue
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        road_gt = (m == 6).astype(np.uint8) * 255
        
        # Calculate intersection over union
        inter = np.logical_and(resized_y > 0, road_gt > 0).sum()
        union = np.logical_or(resized_y > 0, road_gt > 0).sum()
        iou = inter / (union + 1e-6)
        if iou > best_iou:
            best_iou = iou
            best_p = pid
        print(f"  {pid}: road IoU = {iou:.4f}")
    print(f"==> {sc_name} BEST MATCH: {best_p} (IoU={best_iou:.4f})")

