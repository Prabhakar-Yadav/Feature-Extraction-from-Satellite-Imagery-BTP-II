"""
scripts/inspect_subdirs.py
--------------------------
Inspects directory sizes and references for NEW_Model(sem2), VASHI_IMPROVED, Opencode,
and checks if NISAR .h5 file is referenced in any code.
"""

import os
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def get_folder_size(path):
    total = 0
    cnt = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            try:
                total += os.path.getsize(p)
                cnt += 1
            except: pass
    return total, cnt

def analyze_subdirs():
    print("=" * 80)
    print("SUBDIRECTORY ANALYSIS")
    print("=" * 80)

    for target in ["NEW_Model(sem2)", "VASHI_IMPROVED", "Opencode", "FINAL_ORGANIZED_SUBMISSION", "Test_Model"]:
        p = WORKSPACE / target
        if not p.exists(): continue
        print(f"\n--- {target} ---")
        for sub in sorted(p.iterdir()):
            if sub.is_dir():
                sz, cnt = get_folder_size(sub)
                print(f"  [DIR]  {sub.name:45s} | {sz/(1024**3):6.2f} GB ({sz/(1024**2):8.1f} MB) | {cnt:5d} files")
            else:
                sz = sub.stat().st_size
                print(f"  [FILE] {sub.name:45s} | {sz/(1024**2):8.2f} MB")

    # Search for references to NISAR
    print("\n" + "=" * 80)
    print("CHECKING CODE REFERENCES TO NISAR HDF5 FILE...")
    print("=" * 80)
    nisar_matches = []
    for root, dirs, files in os.walk(WORKSPACE):
        if ".git" in root: continue
        for f in files:
            if f.endswith((".py", ".ipynb", ".json", ".sh", ".bat", ".md")):
                fp = Path(root) / f
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    if "NISAR" in text or "025_113_A" in text:
                        nisar_matches.append(str(fp.relative_to(WORKSPACE)))
                except: pass

    if nisar_matches:
        print(f"Found {len(nisar_matches)} files referencing NISAR:")
        for m in nisar_matches:
            print(f"  - {m}")
    else:
        print("NO CODE OR NOTEBOOK REFERENCES FOUND TO NISAR FILE!")

if __name__ == "__main__":
    analyze_subdirs()

