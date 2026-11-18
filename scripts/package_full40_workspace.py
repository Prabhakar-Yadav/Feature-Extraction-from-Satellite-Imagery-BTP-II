"""
Package complete 40-patch workspace into a clean standalone ZIP archive
for offline inspection and distribution.
"""
import os
import zipfile
import shutil

SOURCE_DIR = r"NEW_Model(sem2)\FINAL_ORGANIZED_SUBMISSION\3_Outputs\Visual_PNG_Results\human_verification_batches\builtup_reannotation_review"
ZIP_NAME = "Cartosat2E_Full40_Annotation_Workspace.zip"
TARGET_ZIP_LOCAL = os.path.join(SOURCE_DIR, ZIP_NAME)
BRAIN_DIR = r"C:\Users\PRABHAKAR\.gemini\antigravity\brain\c89e20d2-5c79-420e-8b8e-ab1621aef4cf"
TARGET_ZIP_BRAIN = os.path.join(BRAIN_DIR, ZIP_NAME)

print(f"Creating ZIP archive: {TARGET_ZIP_LOCAL}")

# We will include:
# 1. HTML portal
# 2. CSV summary files
# 3. Overview PNGs
# 4. new_20_random_patch_comparisons/
# 5. patch_comparisons/
# 6. data/ directory (images_rgb, images_cir, masks_v1, masks_v2, masks_v3, masks_v4 for the 40 patches + meta)

with zipfile.ZipFile(TARGET_ZIP_LOCAL, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # Skip any existing large zip files
            if file.endswith(".zip"):
                continue
            
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, SOURCE_DIR)
            
            # If in masks_v4_learned_candidate, only include the 40 active studio patches to keep zip size reasonable
            if "masks_v4_learned_candidate" in rel_path:
                # We can include all or 40. There are 186 pngs (each ~10KB). That's only ~2MB total! So include all!
                pass
                
            zipf.write(full_path, arcname=rel_path)

zip_size_mb = os.path.getsize(TARGET_ZIP_LOCAL) / (1024 * 1024)
print(f"Successfully created ZIP ({zip_size_mb:.2f} MB)")

# Copy to brain artifact directory as well
print(f"Copying to brain directory: {TARGET_ZIP_BRAIN}")
shutil.copy2(TARGET_ZIP_LOCAL, TARGET_ZIP_BRAIN)
print("Packaging complete!")

