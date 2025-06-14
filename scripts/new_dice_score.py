import nibabel as nib
import numpy as np
import os
import pandas as pd
import argparse
import glob

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

def find_segmentation_file(base_dir, modality, series_id, tumor_name):
    """
    Find the segmentation file for a specific tumor.
    
    Args:
        base_dir: Base directory to search in
        modality: Modality name (e.g., '4DCT', 'HS')
        series_id: Series ID (e.g., '9__0', '39')
        tumor_name: Name of tumor (e.g., 'tumour1')
    
    Returns:
        Path to the segmentation file if found, None otherwise
    """
    # Common patterns for file paths
    patterns = [
        # Pattern 1: {base_dir}/{modality}/{series_id}/segmentations/{tumor_name}.nii.gz
        os.path.join(base_dir, modality, series_id, "segmentations", f"{tumor_name}.nii.gz"),
        
        # Pattern 2: {base_dir}/{modality}/segmentations/{series_id}_{tumor_name}.nii.gz
        os.path.join(base_dir, modality, "segmentations", f"{series_id}_{tumor_name}.nii.gz"),
        
        # Pattern 3: {base_dir}/segmentations/{modality}_{series_id}_{tumor_name}.nii.gz
        os.path.join(base_dir, "segmentations", f"{modality}_{series_id}_{tumor_name}.nii.gz"),
        
        # Pattern 4: Just search recursively for any file matching the tumor name
        os.path.join(base_dir, f"**/*{tumor_name}*.nii.gz")
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    
    return None

def compare_tumors(base_dir, modality1, series_id1, modality2, series_id2, tumor_name, 
                  csv_path1=None, csv_path2=None):
    """
    Calculate DICE score between segmentations from different modalities.
    
    Args:
        base_dir: Base directory containing segmentations
        modality1: First modality (e.g., '4DCT')
        series_id1: First series ID (e.g., '9__0')
        modality2: Second modality (e.g., 'HS')
        series_id2: Second series ID (e.g., '39')
        tumor_name: Name of tumor to analyze (e.g., 'tumour1')
        csv_path1: Path to CSV file for first modality
        csv_path2: Path to CSV file for second modality
    
    Returns:
        DICE score for the specified tumor
    """
    # Determine CSV paths if not provided
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if csv_path1 is None:
        # Look for CSV with modality1 in the name
        csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and modality1.lower() in f.lower()]
        if csv_files:
            csv_path1 = os.path.join(parent_dir, csv_files[0])
        else:
            print(f"No CSV file found for {modality1}. Please specify CSV path.")
            return None
    
    if csv_path2 is None:
        # Look for CSV with modality2 in the name
        csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and modality2.lower() in f.lower()]
        if csv_files:
            csv_path2 = os.path.join(parent_dir, csv_files[0])
        else:
            print(f"No CSV file found for {modality2}. Please specify CSV path.")
            return None
    
    # Read tumor info from CSVs
    tumor_data1 = pd.read_csv(csv_path1)
    tumor_data2 = pd.read_csv(csv_path2)
    
    # Verify series exists in the CSV
    if series_id1 not in tumor_data1['series_id'].astype(str).values:
        print(f"Series ID {series_id1} not found in {csv_path1}")
        return None
    
    if series_id2 not in tumor_data2['series_id'].astype(str).values:
        print(f"Series ID {series_id2} not found in {csv_path2}")
        return None
    
    # Filter for the specified series
    series1_data = tumor_data1[tumor_data1['series_id'].astype(str) == str(series_id1)]
    series2_data = tumor_data2[tumor_data2['series_id'].astype(str) == str(series_id2)]
    
    # Verify tumor exists in both series
    if tumor_name not in series1_data['name'].values:
        print(f"Tumor {tumor_name} not found in series {series_id1}")
        return None
    
    if tumor_name not in series2_data['name'].values:
        print(f"Tumor {tumor_name} not found in series {series_id2}")
        return None
    
    # Find segmentation files
    path_seg1 = find_segmentation_file(base_dir, modality1, series_id1, tumor_name)
    path_seg2 = find_segmentation_file(base_dir, modality2, series_id2, tumor_name)
    
    if not path_seg1:
        print(f"Could not find segmentation file for {modality1}, series {series_id1}, tumor {tumor_name}")
        return None
    
    if not path_seg2:
        print(f"Could not find segmentation file for {modality2}, series {series_id2}, tumor {tumor_name}")
        return None
    
    print(f"Found segmentation files:")
    print(f"  {path_seg1}")
    print(f"  {path_seg2}")
    
    # Load segmentations
    try:
        seg1 = load_nifti(path_seg1)
        seg2 = load_nifti(path_seg2)
        
        # Apply thresholding if needed
        seg1 = seg1 > 0.5
        seg2 = seg2 > 0.5
        
        # Calculate DICE score
        dice = dice_score(seg1, seg2)
        print(f"DICE Score for {tumor_name}: {dice:.4f}")
        
        # Get tumor volumes from CSV for reference
        vol1 = series1_data[series1_data['name'] == tumor_name]['volume'].values[0]
        vol2 = series2_data[series2_data['name'] == tumor_name]['volume'].values[0]
        print(f"Volumes: {modality1}={vol1:.2f}, {modality2}={vol2:.2f}, Difference={vol1-vol2:.2f}")
        
        return dice
    except Exception as e:
        print(f"Error processing {tumor_name}: {e}")
        return None

def analyze_all_tumors(base_dir, modality1, series_id1, modality2, series_id2, 
                      csv_path1=None, csv_path2=None):
    """
    Calculate DICE scores for all common tumors between two series.
    """
    # Determine CSV paths if not provided
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if csv_path1 is None:
        # Look for CSV with modality1 in the name
        csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and modality1.lower() in f.lower()]
        if csv_files:
            csv_path1 = os.path.join(parent_dir, csv_files[0])
        else:
            print(f"No CSV file found for {modality1}. Please specify CSV path.")
            return {}
    
    if csv_path2 is None:
        # Look for CSV with modality2 in the name
        csv_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and modality2.lower() in f.lower()]
        if csv_files:
            csv_path2 = os.path.join(parent_dir, csv_files[0])
        else:
            print(f"No CSV file found for {modality2}. Please specify CSV path.")
            return {}
    
    # Read tumor info from CSVs
    tumor_data1 = pd.read_csv(csv_path1)
    tumor_data2 = pd.read_csv(csv_path2)
    
    # Filter for the specified series
    series1_data = tumor_data1[tumor_data1['series_id'].astype(str) == str(series_id1)]
    series2_data = tumor_data2[tumor_data2['series_id'].astype(str) == str(series_id2)]
    
    # Find common tumors
    tumors1 = set(series1_data['name'].values)
    tumors2 = set(series2_data['name'].values)
    common_tumors = tumors1.intersection(tumors2)
    
    results = {}
    
    print(f"Found {len(common_tumors)} common tumors between {modality1} series {series_id1} and {modality2} series {series_id2}")
    
    for tumor in common_tumors:
        dice = compare_tumors(base_dir, modality1, series_id1, modality2, series_id2, tumor, csv_path1, csv_path2)
        if dice is not None:
            results[tumor] = dice
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate DICE scores between tumor segmentations from different modalities')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing segmentations')
    parser.add_argument('--modality1', type=str, required=True, help='First modality (e.g., 4DCT)')
    parser.add_argument('--series1', type=str, required=True, help='First series ID (e.g., 9__0)')
    parser.add_argument('--modality2', type=str, required=True, help='Second modality (e.g., HS)')
    parser.add_argument('--series2', type=str, required=True, help='Second series ID (e.g., 39)')
    parser.add_argument('--tumor', type=str, help='Specific tumor to analyze (e.g., tumour1)')
    parser.add_argument('--csv1', type=str, help='Path to CSV with tumor information for first modality')
    parser.add_argument('--csv2', type=str, help='Path to CSV with tumor information for second modality')
    
    args = parser.parse_args()
    
    if args.tumor:
        # Analyze specific tumor
        compare_tumors(args.base_dir, args.modality1, args.series1, args.modality2, args.series2, 
                      args.tumor, args.csv1, args.csv2)
    else:
        # Analyze all common tumors
        analyze_all_tumors(args.base_dir, args.modality1, args.series1, args.modality2, args.series2,
                          args.csv1, args.csv2)