"""
scripts/inspect_audit.py
------------------------
Processes audit_storage_summary.json and prints comprehensive analytics.
"""

import json
from pathlib import Path

def main():
    json_path = Path("scratch/audit_storage_summary.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    print("=" * 80)
    print(f"WORKSPACE STORAGE OVERVIEW")
    print("=" * 80)
    print(f"Total Storage:       {data['total_size_gb']:.2f} GB ({data['total_size_mb']:.2f} MB)")
    print(f"Total File Count:    {data['total_files']:,}")
    print(f"Total Directory Count: {data['total_dirs']:,}")
    print(f"Rasters (.tif/.h5):  {data['rasters_count']}")
    print(f"Arrays (.npy/.npz):  {data['arrays_count']}")
    print(f"Notebooks (.ipynb):  {data['notebooks_count']}")
    print(f"Python Scripts:      {data['scripts_count']}")

    print("\n" + "=" * 80)
    print("TOP-LEVEL DIRECTORY BREAKDOWN")
    print("=" * 80)
    for k, v in data["top_level_breakdown"].items():
        print(f"{k:40s} | {v['size_gb']:7.2f} GB ({v['size_mb']:9.1f} MB) | {v['file_count']:6,d} files")

    print("\n" + "=" * 80)
    print("STORAGE BY FILE EXTENSION (TOP 20)")
    print("=" * 80)
    for ext, v in list(data["extension_breakdown"].items())[:20]:
        print(f"{ext or '[no ext]':15s} | {v['size_mb']/1024:7.2f} GB ({v['size_mb']:9.1f} MB) | {v['count']:6,d} files")

    print("\n" + "=" * 80)
    print(f"MODEL CHECKPOINTS INVENTORY ({len(data['checkpoints'])} FOUND)")
    print("=" * 80)
    for ck in data["checkpoints"]:
        print(f"{ck['size_mb']:8.2f} MB | {ck['mtime']} | {ck['path']}")

    print("\n" + "=" * 80)
    print(f"LARGEST 25 FILES IN WORKSPACE")
    print("=" * 80)
    for lf in data["large_files"][:25]:
        print(f"{lf['size_mb']:8.2f} MB | {lf['path']}")

    total_dupe_waste = sum((d["count"] - 1) * d["size_mb"] for d in data["duplicates"])
    print("\n" + "=" * 80)
    print(f"DUPLICATE FILES ANALYSIS ({len(data['duplicates'])} duplicate sets, {total_dupe_waste:.2f} MB / {total_dupe_waste/1024:.2f} GB recoverable)")
    print("=" * 80)
    for i, d in enumerate(data["duplicates"][:25]):
        waste = (d["count"] - 1) * d["size_mb"]
        print(f"\nGroup {i+1}: {d['size_mb']:.2f} MB each x {d['count']} copies -> Waste: {waste:.2f} MB")
        for f in d["files"]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()

