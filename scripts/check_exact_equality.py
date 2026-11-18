import os
import cv2
import numpy as np

sc_dir = r"C:\Users\PRABHAKAR\Pictures\Screenshots"
sc1 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000209.png")) # Image 1
sc2 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000322.png")) # Image 2
sc3 = cv2.imread(os.path.join(sc_dir, "Screenshot 2026-11-07 000838.png")) # Image 3

# Also let's check up1 and up2 directly:
up1 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793989943145_74b1352e.jpg")
up2 = cv2.imread(r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf\.user_uploaded\media_1793990007163_3831517d.jpg")

# Notice that sc1 is identical to up1!
print("Is sc1 identical to up1?", np.array_equal(sc1, up1) if (sc1 is not None and up1 is not None) else False)
print("Is sc2 identical to up2?", np.array_equal(sc2, up2) if (sc2 is not None and up2 is not None) else False)

# Let's crop sc1, sc2, sc3 to remove black padding:
# In sc1: shape is (486, 546). The canvas is on the left, 486 x 486!
# Let's check where the image ends on the right:
col_sums_1 = np.sum(sc1, axis=(0, 2))
black_cols_1 = np.where(col_sums_1 == 0)[0]
print("sc1 black cols on right:", black_cols_1[:10] if len(black_cols_1)>0 else "none")

# In sc2: shape is (746, 900)
col_sums_2 = np.sum(sc2, axis=(0, 2))
black_cols_2 = np.where(col_sums_2 == 0)[0]
print("sc2 black cols on right:", black_cols_2[:10] if len(black_cols_2)>0 else "none")

# In sc3: shape is (698, 788)
col_sums_3 = np.sum(sc3, axis=(0, 2))
black_cols_3 = np.where(col_sums_3 == 0)[0]
print("sc3 black cols on right:", black_cols_3[:10] if len(black_cols_3)>0 else "none")

