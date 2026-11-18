import torch
import numpy as np

CLASS_NAMES_7 = [
    "Water", "Vegetation", "Barren Land",
    "Residential Built-up", "Industrial Built-up", "Slum / Informal Settlement",
    "Road Network"
]

def analyze_ckpt(path, name):
    ckpt = torch.load(path, map_location='cpu')
    m = ckpt.get('metrics', {})
    conf = np.array(m.get('confusion_matrix_7x7'))
    
    print("\n" + "="*80)
    print(f"CONFUSION MATRIX: {name} (Round {ckpt.get('round')}, Epoch {ckpt.get('epoch')})")
    print(f"Composite Score: {ckpt.get('composite_score', 0)*100:.2f}% | Pixel Acc: {m.get('pixel_accuracy', 0)*100:.2f}% | 7C mIoU: {m.get('mean_iou', 0)*100:.2f}%")
    print("="*80)
    
    header = f"{'Ground Truth \\ Predicted':28s}" + "".join([f"{n[:8]:>10s}" for n in CLASS_NAMES_7])
    print(header)
    print("-" * 100)
    for r in range(7):
        row_str = f"{CLASS_NAMES_7[r]:28s}" + "".join([f"{int(conf[r, c]):10,d}" for c in range(7)])
        print(row_str)
        
    print("\n" + "-"*40 + " ROW-NORMALIZED (RECALL %) " + "-"*40)
    print(header)
    print("-" * 100)
    for r in range(7):
        row_tot = np.sum(conf[r, :])
        if row_tot > 0:
            row_str = f"{CLASS_NAMES_7[r]:28s}" + "".join([f"{conf[r, c]/row_tot*100:9.1f}%" for c in range(7)])
        else:
            row_str = f"{CLASS_NAMES_7[r]:28s}" + "".join([f"{'0.0%':>10s}" for _ in range(7)])
        print(row_str)

    print("\n" + "-"*40 + " COLUMN-NORMALIZED (PRECISION %) " + "-"*40)
    print(header)
    print("-" * 100)
    for r in range(7):
        row_str = f"{CLASS_NAMES_7[r]:28s}"
        for c in range(7):
            col_tot = np.sum(conf[:, c])
            if col_tot > 0:
                row_str += f"{conf[r, c]/col_tot*100:9.1f}%"
            else:
                row_str += f"{'0.0%':>10s}"
        print(row_str)

    # Built-up aggregate analysis:
    bu_tp = conf[3:6, 3:6].sum()
    bu_fp = conf[:3, 3:6].sum() + conf[6, 3:6].sum()
    bu_fn = conf[3:6, :3].sum() + conf[3:6, 6].sum()
    bu_iou = bu_tp / (bu_tp + bu_fp + bu_fn) if (bu_tp + bu_fp + bu_fn) > 0 else 0.0
    print(f"\nBuilt-up Aggregate TP: {bu_tp:,d} | FP: {bu_fp:,d} | FN: {bu_fn:,d} | Aggregate Built-up IoU: {bu_iou*100:.2f}%")
    print(f"Built-up False Positives Breakdown (Pixels mistakenly called Built-up):")
    print(f"  From Water:       {conf[0, 3:6].sum():10,d} ({conf[0, 3:6].sum()/max(1, bu_fp)*100:5.1f}%)")
    print(f"  From Vegetation:  {conf[1, 3:6].sum():10,d} ({conf[1, 3:6].sum()/max(1, bu_fp)*100:5.1f}%)")
    print(f"  From Barren Land: {conf[2, 3:6].sum():10,d} ({conf[2, 3:6].sum()/max(1, bu_fp)*100:5.1f}%)")
    print(f"  From Road:        {conf[6, 3:6].sum():10,d} ({conf[6, 3:6].sum()/max(1, bu_fp)*100:5.1f}%)")
    print(f"Built-up False Negatives Breakdown (True Built-up missed by model):")
    print(f"  Missed as Water:       {conf[3:6, 0].sum():10,d} ({conf[3:6, 0].sum()/max(1, bu_fn)*100:5.1f}%)")
    print(f"  Missed as Vegetation:  {conf[3:6, 1].sum():10,d} ({conf[3:6, 1].sum()/max(1, bu_fn)*100:5.1f}%)")
    print(f"  Missed as Barren Land: {conf[3:6, 2].sum():10,d} ({conf[3:6, 2].sum()/max(1, bu_fn)*100:5.1f}%)")
    print(f"  Missed as Road:        {conf[3:6, 6].sum():10,d} ({conf[3:6, 6].sum()/max(1, bu_fn)*100:5.1f}%)")

analyze_ckpt("NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/checkpoint_round_01.pth", "ROUND 01")
analyze_ckpt("NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/checkpoints/best_autonomous_model.pth", "ROUND 05 (LATEST BEST)")

