"""
scripts/inspect_data_and_dupes.py
---------------------------------
Inspects Test_Data, Satellite_Data, and exact duplicates across checkpoints and submissions.
"""

import os
from pathlib import Path

WORKSPACE = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def inspect():
    # 1. Test_Data
    td = WORKSPACE / "NEW_Model(sem2)/Test_Data"
    print("=" * 80)
    print("INSPECTING NEW_Model(sem2)/Test_Data:")
    print("=" * 80)
    for p in sorted(td.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            sz = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  [DIR]  {p.name:35s} | {sz/(1024**3):6.2f} GB ({sz/(1024**2):8.1f} MB) | {len(files):5d} files")
        else:
            print(f"  [FILE] {p.name:35s} | {p.stat().st_size/(1024**2):8.2f} MB")

    # 2. Satellite_Data
    sd = WORKSPACE / "NEW_Model(sem2)/Satellite_Data"
    print("\n" + "=" * 80)
    print("INSPECTING NEW_Model(sem2)/Satellite_Data:")
    print("=" * 80)
    for p in sorted(sd.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            sz = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  [DIR]  {p.name:35s} | {sz/(1024**3):6.2f} GB ({sz/(1024**2):8.1f} MB) | {len(files):5d} files")
        else:
            print(f"  [FILE] {p.name:35s} | {p.stat().st_size/(1024**2):8.2f} MB")

    # 3. Compare root FINAL_ORGANIZED_SUBMISSION vs NEW_Model(sem2)/FINAL_ORGANIZED_SUBMISSION
    f1 = WORKSPACE / "FINAL_ORGANIZED_SUBMISSION"
    f2 = WORKSPACE / "NEW_Model(sem2)/FINAL_ORGANIZED_SUBMISSION"
    print("\n" + "=" * 80)
    print("COMPARING ROOT vs NESTED FINAL_ORGANIZED_SUBMISSION:")
    print("=" * 80)
    print(f"Root:   {sum(f.stat().st_size for f in f1.rglob('*') if f.is_file()) / (1024**2):.2f} MB, {len(list(f1.rglob('*')))} items")
    print(f"Nested: {sum(f.stat().st_size for f in f2.rglob('*') if f.is_file()) / (1024**2):.2f} MB, {len(list(f2.rglob('*')))} items")

    # 4. Check duplicate checkpoints
    print("\n" + "=" * 80)
    print("EXACT DUPLICATE CHECKPOINTS:")
    print("=" * 80)
    import hashlib
    ck_paths = [
        WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
        WORKSPACE / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_hybrid_hierarchical_model.pth",
        WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/SegFormer_2.5m/best_segformer.pth",
        WORKSPACE / "NEW_Model(sem2)/Phase_2_Model_Training/01_SegFormer/checkpoints/best_segformer.pth",
        WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/UNet_2.5m/best_unet.pth",
        WORKSPACE / "NEW_Model(sem2)/Phase_2_Model_Training/02_UNet/checkpoints/best_unet.pth",
        WORKSPACE / "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Final_Subclasses_Model/best_3subclass_segformer.pth",
        WORKSPACE / "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_3subclass_segformer.pth",
        WORKSPACE / "Opencode/checkpoints/checkpoint_latest_v2.pth",
        WORKSPACE / "Opencode/checkpoints_smoke/checkpoint_latest_v2.pth",
    ]
    hashes = {}
    for p in ck_paths:
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            print(f"{p.stat().st_size/(1024*1024):8.2f} MB | {h[:16]} | {p.relative_to(WORKSPACE)}")

if __name__ == "__main__":
    inspect()

