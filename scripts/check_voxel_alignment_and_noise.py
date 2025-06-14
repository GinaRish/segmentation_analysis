import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops

def check_alignment(path1, path2):
    img1 = nib.load(path1)
    img2 = nib.load(path2)

    shape1 = img1.shape
    shape2 = img2.shape
    affine1 = img1.affine
    affine2 = img2.affine

    print("Shape 1:", shape1)
    print("Shape 2:", shape2)
    print("Affine 1:\n", affine1)
    print("Affine 2:\n", affine2)

    if shape1 != shape2:
        print("⚠️ Shapes are different!")
    else:
        print("✅ Shapes match.")

    if not np.allclose(affine1, affine2, atol=1e-5):
        print("⚠️ Affine matrices (spacing/orientation) differ!")
    else:
        print("✅ Affines match — images are aligned voxel-wise.")

check_alignment(
    "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/32_lower_tumour.nii.gz",
    "C:/AMIGOpy/regular_20amplitude_scan1/segmentations/31_lower_tumour.nii.gz"
)

def check_connected_components(binary_mask):
    labeled, _ = label(binary_mask)
    regions = regionprops(labeled)

    print(f"Number of components: {len(regions)}")
    if len(regions) == 0:
        print("⚠️ No foreground detected.")
        return

    largest = max(regions, key=lambda r: r.area)
    print(f"Largest component area: {largest.area} voxels")

    if len(regions) > 1:
        print("⚠️ Multiple components detected — might contain speckles or noise.")
    else:
        print("✅ Only one component — likely clean.")

# Example usage:
data = nib.load("C:/AMIGOpy/regular_20amplitude_scan1/segmentations/30_lower_tumour.nii.gz").get_fdata()
binary = data > 0.5
check_connected_components(binary)

