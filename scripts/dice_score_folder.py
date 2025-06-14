import nibabel as nib
import numpy as np
import pandas as pd
import os

def load_nifti(path):
    return nib.load(path).get_fdata()

def dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 1.0
    return 2. * intersection / total

# === CONFIGURATION ===
ground_truth_path = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations_middle/30_middle_tumour.nii.gz"
folder_path = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations_middle"
itv_csv_path = "C:/AMIGOpy/regular_20amplitude_scan1/structure_stats_2.csv"

# === Load volume reference CSV ===
volume_df = pd.read_csv(itv_csv_path)
volume_df = volume_df[['series_id', 'name', 'volume']].dropna()

# === Load ground truth mask ===
gt_mask = load_nifti(ground_truth_path) > 0.5
gt_filename = os.path.basename(ground_truth_path)

# === Compare with each other file ===
results = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    if not (file.endswith(".nii") or file.endswith(".nii.gz")):
        continue
    if file == gt_filename:
        continue

    try:
        pred_mask = load_nifti(file_path) > 0.5
        score = dice_score(gt_mask, pred_mask)

        # Extract series_id and tumour name from filename
        parts = file.replace(".nii.gz", "").split("_", 1)
        series_id = int(parts[0])
        tumour_name = parts[1]

        # Find matching row in CSV
        row = volume_df.loc[
            (volume_df['series_id'] == series_id) &
            (volume_df['name'] == tumour_name)
        ]

        itv_volume = row['volume'].values[0] if not row.empty else None

        results.append({
            "filename": file,
            "dice_score": round(score, 4),
            "itv_volume_mm3": round(itv_volume, 1) if itv_volume is not None else "N/A"
        })

    except Exception as e:
        results.append({
            "filename": file,
            "dice_score": "ERROR",
            "itv_volume_mm3": f"❌ {e}"
        })

# Save to CSV
results_df = pd.DataFrame(results)
output_path = os.path.join(folder_path, "dice_itv_summary.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")
