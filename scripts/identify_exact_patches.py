import os
import cv2
import numpy as np
import glob

# Load the three screenshots
sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # ~00:02
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # ~00:03
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # ~00:08

# Also check user_uploaded
up1 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg")
up2 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793990007163_3831517d.jpg")

print(f"sc1: {sc1.shape if sc1 is not None else None}")
print(f"sc2: {sc2.shape if sc2 is not None else None}")
print(f"sc3: {sc3.shape if sc3 is not None else None}")
print(f"up1: {up1.shape if up1 is not None else None}")
print(f"up2: {up2.shape if up2 is not None else None}")

# Let's match each screenshot with RGB patches
rgb_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\images_rgb"

def find_best_match(query_img, label):
    best_id = None
    best_score = -1
    # query_img might be cropped or resized. Let's find template match
    # Convert query to gray
    h_q, w_q = query_img.shape[:2]
    # In up1 / up2 or sc, it's an overlay on 512x512
    # Let's check which RGB patch matches best
    for f in glob.glob(os.path.join(rgb_dir, "*_rgb.png")):
        pid = os.path.basename(f).replace("_rgb.png", "")
        rgb = cv2.imread(f)
        # Check normalized cross-correlation or feature match
        # Let's resize rgb to match or check direct template match
        res = cv2.matchTemplate(rgb, cv2.resize(query_img[:, :min(w_q, h_q)], (256, 256)), cv2.TM_CCOEFF_NORMED) if rgb.shape[0]>=256 else 0
        score = np.max(res)
        if score > best_score:
            best_score = score
            best_id = pid
    print(f"[{label}] Best match: {best_id} with score {best_score}")

find_best_match(up1, "up1")
find_best_match(up2, "up2")
if sc3 is not None:
    find_best_match(sc3, "sc3")

