import os
import cv2
import numpy as np

sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # ~00:02
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # ~00:03
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # ~00:08

rgb_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\images_rgb"

candidates = [
    ("sc1", sc1, ["test_pune_0575", "test_pune_0572", "test_pune_0564"]),
    ("sc2", sc2, ["test_pune_0564", "test_pune_0550", "test_pune_0560", "test_pune_0516"]),
    ("sc3", sc3, ["test_pune_0407", "test_pune_0408", "test_pune_0409", "test_pune_0423"]),
]

for sc_name, sc_img, pids in candidates:
    print(f"\n=================== {sc_name} ===================")
    # The screenshot has the canvas, plus black background if browser/canvas margins exist
    # Let's crop out any pure black borders
    gray = cv2.cvtColor(sc_img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = sc_img[y:y+h, x:x+w]
    print(f"Cropped {sc_name} bounding rect: {x},{y},{w},{h}")
    
    # Compare with each pid RGB
    for pid in pids:
        rf = os.path.join(rgb_dir, f"{pid}_rgb.png")
        rgb = cv2.imread(rf)
        # Resize cropped to 512x512
        resized = cv2.resize(cropped, (512, 512))
        
        # In the screenshot, pixels where mask is transparent show the underlying RGB image!
        # Underneath roads/buildings, there is an alpha overlay.
        # But image features (like edges, contrast) correlate strongly with original RGB.
        diff = cv2.matchTemplate(rgb, cv2.resize(cropped, (256, 256)), cv2.TM_CCOEFF_NORMED)
        print(f"  {pid}: max correlation = {np.max(diff):.4f}")

