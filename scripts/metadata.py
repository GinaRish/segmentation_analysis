import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Load a NIfTI file and print its header
nifti_file_path = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/30_upper_tumour.nii.gz"  # Replace with the actual file path
brain_vol = nib.load(nifti_file_path)
print(brain_vol.header)