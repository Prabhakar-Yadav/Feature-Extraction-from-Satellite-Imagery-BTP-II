import os
import glob
import datetime

def check_dir(d, title):
    print(f"=== {title}: {d} ===")
    if not os.path.exists(d):
        print("Directory does not exist")
        return
    files = sorted(glob.glob(os.path.join(d, "*")), key=os.path.getmtime, reverse=True)
    for f in files[:20]:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        sz = os.path.getsize(f)
        print(f"{mtime} | {sz:>10} bytes | {os.path.basename(f)}")

check_dir(r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review\data\masks_v3_human_verified", "masks_v3_human_verified")
check_dir(r"C:\Users\PRABHAKAR\Downloads", "Downloads")
check_dir(r"C:\Users\PRABHAKAR\Pictures\Screenshots", "Screenshots")
check_dir(r"C:\Users\PRABHAKAR\Desktop", "Desktop")

