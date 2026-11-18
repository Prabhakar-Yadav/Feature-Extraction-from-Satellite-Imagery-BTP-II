import os
import glob
import datetime

v3_dir = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v3_human_verified"

files = sorted(glob.glob(os.path.join(v3_dir, "*.png")), key=os.path.getmtime, reverse=True)

print("Recently modified masks in masks_v3_human_verified:")
for f in files:
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    if mtime.day == 7 or (mtime.day == 6 and mtime.hour == 23 and mtime.minute > 50):
        print(f"{mtime} | {os.path.getsize(f):>8} bytes | {os.path.basename(f)}")

