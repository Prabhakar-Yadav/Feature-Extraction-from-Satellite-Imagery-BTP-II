import os
import pandas as pd

idx_paths = [
    r"NEW_Model(sem2)\Phase_2_Model_Training\04_Multispectral_Subclass_Segmentation\data\dataset_index.csv",
    r"NEW_Model(sem2)\Phase_1_Data_Prep_and_Labeling\data\patches\dataset_index.csv",
    r"NEW_Model(sem2)\Phase_2_Model_Training\03_Cartosat2E_SegFormer_Training\data\dataset_index.csv"
]

for p in idx_paths:
    if os.path.exists(p):
        print(f"\n=== {p} ===")
        df = pd.read_csv(p)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        if "split" in df.columns:
            print("Split value counts:\n", df["split"].value_counts())
        print("First 2 rows:\n", df.head(2))

