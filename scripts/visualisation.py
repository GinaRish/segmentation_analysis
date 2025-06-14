import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

def visualize_nifti(file_path):
    # Load the NIfTI file
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()

    # Check the shape of the data
    print(f"Data shape: {data.shape}")

    # Visualize the entire NIfTI image as a single projection (maximum intensity projection)
    projection = np.max(data, axis=2)  # Maximum intensity projection along the third axis
    plt.imshow(projection, cmap="gray")
    plt.title("Maximum Intensity Projection")
    plt.axis("off")
    plt.show()

nifti_file = "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/32_middle_tumour.nii.gz"  # Replace with your file path
visualize_nifti(nifti_file)