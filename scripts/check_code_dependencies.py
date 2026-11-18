"""
scripts/check_code_dependencies.py
----------------------------------
Searches all .py and .ipynb files in the repository for string references
to checkpoints, folders, and datasets to guarantee no active pipeline is broken.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def check_dependencies():
    # Load all checkpoints from audit
    with open(WORKSPACE / "scratch" / "checkpoint_analysis.json") as f:
        checkpoints = json.load(f)

    # Collect all code files (.py, .ipynb)
    code_files = []
    for root, dirs, files in os.walk(WORKSPACE):
        if ".git" in root or ".commandcode" in root or "scratch" in root:
            continue
        for f in files:
            if f.endswith((".py", ".ipynb")):
                code_files.append(Path(root) / f)

    print(f"Total code files scanned: {len(code_files)}")

    # Map each checkpoint filename and relative path to references in code
    ckpt_refs = defaultdict(list)

    for cf in code_files:
        rel_cf = str(cf.relative_to(WORKSPACE))
        try:
            content = cf.read_text(encoding="utf-8", errors="ignore")
        except:
            continue

        for ck in checkpoints:
            ck_p = ck["path"]
            ck_name = Path(ck_p).name
            if ck_name in content or ck_p in content or ck_p.replace("\\", "/") in content:
                ckpt_refs[ck_p].append(rel_cf)

    print("\n" + "=" * 90)
    print("CHECKPOINT CODE DEPENDENCY CROSS-REFERENCE:")
    print("=" * 90)
    for ck in checkpoints:
        ck_p = ck["path"]
        refs = ckpt_refs[ck_p]
        status = f"REFERENCED in {len(refs)} file(s)" if refs else "NO DIRECT CODE REFERENCE"
        print(f"[{status:26s}] {ck['size_mb']:7.2f} MB | {ck_p}")
        if refs:
            for r in refs[:3]:
                print(f"    <- {r}")
            if len(refs) > 3:
                print(f"    <- ... and {len(refs)-3} more")

    # Save to json
    out_p = WORKSPACE / "scratch" / "checkpoint_code_dependencies.json"
    with open(out_p, "w") as f:
        json.dump({k: v for k, v in ckpt_refs.items()}, f, indent=2)
    print(f"\nDependency map saved to {out_p}")

if __name__ == "__main__":
    check_dependencies()

