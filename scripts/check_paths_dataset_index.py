import pandas as pd

csv_path = r"NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation\data\dataset_index.csv"
df = pd.read_csv(csv_path)

print("Unique splits:", df['split'].value_counts())
print("\nSample train paths:")
for i, r in df[df['split'] == 'train'].head(5).iterrows():
    print(" ", r['patch_id'], "|", r['image_path'], "|", r['mask_path'])

print("\nSample test paths:")
for i, r in df[df['split'] == 'test'].head(5).iterrows():
    print(" ", r['patch_id'], "|", r['image_path'], "|", r['mask_path'])

