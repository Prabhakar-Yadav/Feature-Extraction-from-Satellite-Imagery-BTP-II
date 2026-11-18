"""
scripts/audit_project_storage.py
---------------------------------
Comprehensive read-only project audit script for storage analysis,
checkpoint inspection, duplicate identification, and dependency cross-referencing.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import defaultdict

WORKSPACE_ROOT = Path(r"C:\Users\PRABHAKAR\Documents\U-NET_Urban_Feature_Extraction-main").resolve()

def get_file_hash(path, chunk_size=65536, max_size=100 * 1024 * 1024):
    """Calculates SHA256 of a file. For very large files (>100MB), hashes first and last 50MB."""
    sz = os.path.getsize(path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        if sz <= max_size:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        else:
            # Read first 50MB
            hasher.update(f.read(50 * 1024 * 1024))
            # Read last 50MB
            f.seek(-50 * 1024 * 1024, os.SEEK_END)
            hasher.update(f.read(50 * 1024 * 1024))
            hasher.update(str(sz).encode("utf-8"))
    return hasher.hexdigest()

def scan_project():
    print(f"Scanning workspace: {WORKSPACE_ROOT}...")
    start_t = time.time()
    
    total_files = 0
    total_dirs = 0
    total_bytes = 0
    
    dir_sizes = defaultdict(int)
    dir_file_counts = defaultdict(int)
    ext_sizes = defaultdict(int)
    ext_counts = defaultdict(int)
    
    large_files = [] # > 10MB
    checkpoints = [] # .pth, .pt, .ckpt
    datasets_rasters = [] # .tif, .tiff, .h5
    arrays = [] # .npy, .npz
    notebooks = [] # .ipynb
    scripts = [] # .py
    
    # For duplicate detection: group by file size first
    size_to_files = defaultdict(list)
    
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        total_dirs += len(dirs)
        rel_root = Path(root).relative_to(WORKSPACE_ROOT)
        top_dir = rel_root.parts[0] if rel_root.parts else "."
        
        for f in files:
            p = Path(root) / f
            try:
                sz = p.stat().st_size
                mtime = p.stat().st_mtime
            except Exception as e:
                continue
                
            total_files += 1
            total_bytes += sz
            
            dir_sizes[top_dir] += sz
            dir_file_counts[top_dir] += 1
            
            ext = p.suffix.lower()
            ext_sizes[ext] += sz
            ext_counts[ext] += 1
            
            rel_path_str = str(p.relative_to(WORKSPACE_ROOT))
            
            if sz > 10 * 1024 * 1024:
                large_files.append({
                    "path": rel_path_str,
                    "size_mb": sz / (1024 * 1024),
                    "size_bytes": sz,
                    "mtime": time.ctime(mtime),
                    "ext": ext
                })
                
            if ext in [".pth", ".pt", ".ckpt"]:
                checkpoints.append({
                    "path": rel_path_str,
                    "size_mb": sz / (1024 * 1024),
                    "size_bytes": sz,
                    "mtime": time.ctime(mtime)
                })
                
            if ext in [".tif", ".tiff", ".h5"]:
                datasets_rasters.append({
                    "path": rel_path_str,
                    "size_mb": sz / (1024 * 1024),
                    "size_bytes": sz,
                    "mtime": time.ctime(mtime)
                })
                
            if ext in [".npy", ".npz"]:
                arrays.append({
                    "path": rel_path_str,
                    "size_mb": sz / (1024 * 1024),
                    "size_bytes": sz,
                    "mtime": time.ctime(mtime)
                })
                
            if ext == ".ipynb":
                notebooks.append(rel_path_str)
            if ext == ".py":
                scripts.append(rel_path_str)
                
            # Track files > 100 KB for potential duplicates
            if sz > 100 * 1024:
                size_to_files[sz].append(rel_path_str)
                
    elapsed = time.time() - start_t
    print(f"Scan complete in {elapsed:.2f}s. Total files: {total_files}, Total bytes: {total_bytes / (1024**3):.2f} GB")
    
    # Duplicate detection for identical size files
    print("Checking for duplicate files...")
    duplicates = []
    potential_dupes = {sz: flist for sz, flist in size_to_files.items() if len(flist) > 1}
    for sz, flist in potential_dupes.items():
        hash_to_paths = defaultdict(list)
        for rpath in flist:
            full_p = WORKSPACE_ROOT / rpath
            try:
                h = get_file_hash(full_p)
                hash_to_paths[h].append(rpath)
            except Exception as e:
                pass
        for h, dupe_paths in hash_to_paths.items():
            if len(dupe_paths) > 1:
                duplicates.append({
                    "size_mb": sz / (1024 * 1024),
                    "size_bytes": sz,
                    "count": len(dupe_paths),
                    "files": dupe_paths
                })
                
    duplicates.sort(key=lambda x: x["size_bytes"] * (x["count"] - 1), reverse=True)
    
    summary = {
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_size_gb": total_bytes / (1024 ** 3),
        "total_size_mb": total_bytes / (1024 ** 2),
        "top_level_breakdown": {
            td: {
                "size_gb": dir_sizes[td] / (1024 ** 3),
                "size_mb": dir_sizes[td] / (1024 ** 2),
                "file_count": dir_file_counts[td]
            }
            for td in sorted(dir_sizes.keys(), key=lambda k: dir_sizes[k], reverse=True)
        },
        "extension_breakdown": {
            ext: {
                "size_mb": ext_sizes[ext] / (1024 ** 2),
                "count": ext_counts[ext]
            }
            for ext in sorted(ext_sizes.keys(), key=lambda k: ext_sizes[k], reverse=True)[:30]
        },
        "checkpoints": checkpoints,
        "large_files": sorted(large_files, key=lambda x: x["size_bytes"], reverse=True),
        "duplicates": duplicates,
        "rasters_count": len(datasets_rasters),
        "arrays_count": len(arrays),
        "notebooks_count": len(notebooks),
        "scripts_count": len(scripts)
    }
    
    out_json = WORKSPACE_ROOT / "scratch" / "audit_storage_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary written to {out_json}")
    return summary

if __name__ == "__main__":
    scan_project()

