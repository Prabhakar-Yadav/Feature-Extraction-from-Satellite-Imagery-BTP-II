"""
Inspect project directory and find training datasets, image paths,
and existing ground-truth mask directories.
"""
import os
import glob

print("Current working directory:", os.getcwd())

# Search for train/val split files or directories
for root, dirs, files in os.walk("."):
    # Don't recurse into .git or venv
    if ".git" in root or "venv" in root or "__pycache__" in root:
        continue
    for f in files:
        if any(kw in f.lower() for kw in ["train", "split", "dataset", "patches", "metadata"]):
            if f.endswith((".csv", ".json", ".txt", ".py")) and not f.endswith(".pyc"):
                p = os.path.join(root, f)
                print(f"File: {p}")

