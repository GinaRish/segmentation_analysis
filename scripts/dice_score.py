import nibabel as nib
import numpy as np

def load_nifti(path):
    """Load NIfTI image and return data as NumPy array."""
    return nib.load(path).get_fdata()

def dice_score(seg1, seg2):
    """Calculate DICE similarity coefficient between two binary masks."""
    seg1 = seg1.astype(bool)
    seg2 = seg2.astype(bool)

    intersection = np.logical_and(seg1, seg2).sum()
    total = seg1.sum() + seg2.sum()

    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2. * intersection / total

# === Replace with your actual file paths ===
path_gt = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/30_upper_tumour.nii.gz"
path_pred = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/31_upper_tumour.nii.gz"

# === Load and calculate ===
gt = load_nifti(path_gt)
pred = load_nifti(path_pred)

# Optional: apply thresholding if the masks aren't strictly 0 and 1
gt = gt > 0.5
pred = pred > 0.5

dice = dice_score(gt, pred)
print(f"DICE Score: {dice:.4f}")


