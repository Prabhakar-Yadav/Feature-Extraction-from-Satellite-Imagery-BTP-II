"""
scripts/verify_deletion_candidates.py
-------------------------------------
Fast verification of deletion candidates:
Pre-loads all code files (.py, .ipynb, .json, .md) once and checks candidates against them.
"""

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def get_hash(p):
    if not p.exists(): return None
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1048576):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("Pre-indexing code files...")
    code_contents = {}
    for root, dirs, files in os.walk(WORKSPACE):
        if ".git" in root or "scratch" in root: continue
        for f in files:
            if f.endswith((".py", ".ipynb", ".json", ".sh", ".bat", ".md")):
                fp = Path(root) / f
                try:
                    rel_p = str(fp.relative_to(WORKSPACE))
                    code_contents[rel_p] = fp.read_text(encoding="utf-8", errors="ignore")
                except: pass
    print(f"Pre-indexed {len(code_contents)} code files.")

    def find_refs(target_path, target_name=None):
        refs = []
        t1 = target_path.replace("\\", "/")
        t2 = target_path.replace("/", "\\")
        tname = target_name or Path(target_path).name
        for rel_f, text in code_contents.items():
            if t1 in text or t2 in text or (tname and tname in text):
                refs.append(rel_f)
        return refs

    # Candidates
    candidates = [
        # Group A: Exact duplicate checkpoints
        {
            "category": "A. Exact Duplicate Model Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/01_SegFormer/checkpoints/best_segformer.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/SegFormer_2.5m/best_segformer.pth",
            "reason": "Exact byte-for-byte binary duplicate (SHA-256 matches retained copy)."
        },
        {
            "category": "A. Exact Duplicate Model Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/02_UNet/checkpoints/best_unet.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/2.5m_Resolution/UNet_2.5m/best_unet.pth",
            "reason": "Exact byte-for-byte binary duplicate (SHA-256 matches retained copy)."
        },
        {
            "category": "A. Exact Duplicate Model Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_3subclass_segformer.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Final_Subclasses_Model/best_3subclass_segformer.pth",
            "reason": "Exact byte-for-byte binary duplicate (SHA-256 matches retained copy)."
        },
        {
            "category": "A. Exact Duplicate Model Checkpoint",
            "candidate": "Opencode/checkpoints_smoke/checkpoint_latest_v2.pth",
            "retained": "Opencode/checkpoints/checkpoint_latest_v2.pth",
            "reason": "Duplicate smoke test checkpoint identical to full latest checkpoint."
        },
        # Group A: Exact Duplicate Root File
        {
            "category": "A. Exact Duplicate Report Copy",
            "candidate": "BTP_Report_Prabhakar_Yadav_23B0653.docx",
            "retained": "23b0653.docx",
            "reason": "Exact binary duplicate of the primary submission report file."
        },
        # Group B: Intermediate Round Checkpoints
        {
            "category": "B. Intermediate Round Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_01.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
            "reason": "Intermediate training round 1; fully superseded by best_hybrid_hierarchical_model.pth."
        },
        {
            "category": "B. Intermediate Round Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_02.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
            "reason": "Intermediate training round 2; fully superseded by best_hybrid_hierarchical_model.pth."
        },
        {
            "category": "B. Intermediate Round Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_03.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
            "reason": "Intermediate training round 3; fully superseded by best_hybrid_hierarchical_model.pth."
        },
        {
            "category": "B. Intermediate Round Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_04.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
            "reason": "Intermediate training round 4; fully superseded by best_hybrid_hierarchical_model.pth."
        },
        {
            "category": "B. Intermediate Round Checkpoint",
            "candidate": "NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_05.pth",
            "retained": "FINAL_ORGANIZED_SUBMISSION/2_Models/0.60m_Multispectral/Combined_UNet_SegFormer_Hybrid/best_hybrid_hierarchical_model.pth",
            "reason": "Intermediate training round 5; fully superseded by best_hybrid_hierarchical_model.pth."
        },
        # Group B: Early Vashi Intermediate Checkpoints
        {
            "category": "B. Intermediate Vashi Checkpoint",
            "candidate": "VASHI_IMPROVED/checkpoints/vashi_level2_128_best_model.pth",
            "retained": "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
            "reason": "Early patch-resolution exploration checkpoint; superseded by corrected V4."
        },
        {
            "category": "B. Intermediate Vashi Checkpoint",
            "candidate": "VASHI_IMPROVED/checkpoints/vashi_level2_256_best_model.pth",
            "retained": "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
            "reason": "Early patch-resolution exploration checkpoint; superseded by corrected V4."
        },
        {
            "category": "B. Intermediate Vashi Checkpoint",
            "candidate": "VASHI_IMPROVED/checkpoints/vashi_remediated_best_model.pth",
            "retained": "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
            "reason": "Intermediate remediation run; superseded by research_corrected_v4 models."
        },
        {
            "category": "B. Intermediate Vashi Checkpoint",
            "candidate": "VASHI_IMPROVED/checkpoints/vashi_remediated_latest.pth",
            "retained": "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
            "reason": "Intermediate latest snapshot of remediation run; superseded by E4."
        },
        {
            "category": "B. Intermediate Vashi Checkpoint",
            "candidate": "VASHI_IMPROVED/checkpoints/latest_vashi_checkpoint.pth",
            "retained": "VASHI_IMPROVED/FIVE_CLASS_MUMBAI_VASHI/research_corrected_v4/experiment_E4/checkpoints/best_expE4_model.pth",
            "reason": "Unfinished training snapshot; superseded by E4."
        },
        # Group C: Disposable Scratch DOCX Files
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_3_sections.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from page number formatting test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_all_math.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from equation verification test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_all_omml.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from OMML equation test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_cover.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from cover page formatting test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_math.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from math layout test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_math_table.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from equation table test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_pgnum.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from page number type test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_sections.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from section break test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_tab.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from tab alignment test."
        },
        {
            "category": "C. Disposable Scratch Test DOCX",
            "candidate": "test_toc.docx",
            "retained": "23b0653.docx",
            "reason": "Scratch test document from TOC dot leader test."
        },
        # Group C: Nested Duplicate Submission Directory
        {
            "category": "C. Nested Duplicate Directory",
            "candidate": "NEW_Model(sem2)/FINAL_ORGANIZED_SUBMISSION",
            "retained": "FINAL_ORGANIZED_SUBMISSION",
            "reason": "Entire directory (1,430 MB, 1035 files) is a redundant nested copy of root FINAL_ORGANIZED_SUBMISSION."
        }
    ]

    results = []
    for c in candidates:
        cp = WORKSPACE / c["candidate"]
        rp = WORKSPACE / c["retained"]
        
        is_dir = cp.is_dir()
        if is_dir:
            sz_mb = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file()) / (1024*1024)
            hash_match = "N/A (Directory)"
        else:
            sz_mb = cp.stat().st_size / (1024*1024) if cp.exists() else 0
            if "Exact Duplicate" in c["category"]:
                ch = get_hash(cp)
                rh = get_hash(rp)
                hash_match = "Verified Match" if (ch == rh and ch is not None) else f"Mismatch ({ch[:8]} vs {rh[:8]})"
            else:
                hash_match = "N/A (Superseded/Test)"

        refs = find_refs(c["candidate"])
        # Filter out audit scripts from refs
        code_refs = [r for r in refs if not ("audit" in r or "inspect" in r or "verify" in r)]

        entry = {
            "category": c["category"],
            "candidate": c["candidate"],
            "size_mb": sz_mb,
            "retained": c["retained"],
            "hash_verification": hash_match,
            "reason": c["reason"],
            "code_refs": code_refs,
            "dep_check": "PASSED (0 breaking code references)" if len(code_refs) == 0 else f"NOTE: Referenced in {len(code_refs)} script(s)"
        }
        results.append(entry)

    with open(WORKSPACE / "scratch" / "verified_deletion_list.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 110)
    print("VERIFIED DELETION CANDIDATES SUMMARY:")
    print("=" * 110)
    total_sz = 0
    for i, r in enumerate(results):
        total_sz += r["size_mb"]
        print(f"{i+1:2d}. [{r['category'][:22]:22s}] {r['size_mb']:8.2f} MB | {r['candidate']}")
        print(f"    Retained Copy: {r['retained']}")
        print(f"    Hash Check:    {r['hash_verification']} | Dep Check: {r['dep_check']}")
    print(f"\nTotal potential recoverable storage: {total_sz:.2f} MB ({total_sz/1024:.2f} GB)")

if __name__ == "__main__":
    main()
