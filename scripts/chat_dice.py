import os
import pandas as pd
import nibabel as nib
import numpy as np
import argparse

def load_nifti(path):
    return nib.load(path).get_fdata()

def dice_score(seg1, seg2):
    seg1 = seg1.astype(bool)
    seg2 = seg2.astype(bool)
    intersection = np.logical_and(seg1, seg2).sum()
    total = seg1.sum() + seg2.sum()
    if total == 0:
        return 1.0
    return 2. * intersection / total

def enhanced_dice_evaluation(base_dir, modality1, series_id1, modality2, series_id2, tumor_list, patient_id=11):
    results = []
    
    # Construct correct directory paths based on your folder structure and patient ID
    modality1_dir = f"{modality1}_pat{patient_id}_segmentations"
    modality2_dir = f"{modality2}_pat{patient_id}_segmentations"
    
    print(f"Looking for files in:")
    print(f"  {os.path.join(base_dir, modality1_dir)}")
    print(f"  {os.path.join(base_dir, modality2_dir)}")
    
    for tumor in tumor_list:
        # Updated paths to match your directory structure
        path1 = os.path.join(base_dir, modality1_dir, f"{series_id1}_{tumor}.nii.gz")
        path2 = os.path.join(base_dir, modality2_dir, f"{series_id2}_{tumor}.nii.gz")

        if not os.path.exists(path1) or not os.path.exists(path2):
            results.append({
                "tumor": tumor,
                "modality1_path": path1 if os.path.exists(path1) else "❌ Not found",
                "modality2_path": path2 if os.path.exists(path2) else "❌ Not found",
                "dice_score": "N/A",
                "note": "Missing file(s)"
            })
            continue

        seg1 = load_nifti(path1) > 0.5
        seg2 = load_nifti(path2) > 0.5

        if seg1.shape != seg2.shape:
            results.append({
                "tumor": tumor,
                "modality1_path": path1,
                "modality2_path": path2,
                "dice_score": "N/A",
                "note": f"Shape mismatch {seg1.shape} vs {seg2.shape}"
            })
            continue

        dice = dice_score(seg1, seg2)
        results.append({
            "tumor": tumor,
            "modality1_path": path1,
            "modality2_path": path2,
            "dice_score": round(dice, 4),
            "note": "OK"
        })

    results_df = pd.DataFrame(results)
    output_path = os.path.join(base_dir, f"dice_summary_pat{patient_id}_{modality1}_{series_id1}_vs_{modality2}_{series_id2}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Dice summary saved to {output_path}")
    return results_df

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate DICE scores between tumor segmentations")
    parser.add_argument("--base_dir", type=str, default="C:/Users/ginar/Documents/Master Health an Digital Transformation/Internship/segmentation_analysis",
                        help="Base directory containing segmentations")
    parser.add_argument("--modality1", type=str, default="4DCT", help="First modality (e.g., 4DCT)")
    parser.add_argument("--series1", type=str, default="9__0", help="First series ID (e.g., 9__0)")
    parser.add_argument("--modality2", type=str, default="HS", help="Second modality (e.g., HS)")
    parser.add_argument("--series2", type=str, default="39", help="Second series ID (e.g., 39)")
    parser.add_argument("--patient", type=int, default=11, help="Patient ID number (e.g., 11)")
    parser.add_argument("--tumors", type=str, default="tumour1,tumour2,tumour3,tumour4,tumour5",
                        help="Comma-separated list of tumors to analyze")
    
    args = parser.parse_args()
    
    # Parse tumor list from comma-separated string
    tumor_list = [t.strip() for t in args.tumors.split(",")]
    
    print(f"Comparing patient {args.patient} data:")
    print(f"  {args.modality1} series {args.series1} vs {args.modality2} series {args.series2}")
    print(f"  Analyzing {len(tumor_list)} tumors: {', '.join(tumor_list)}")
    
    # Run the dice evaluation
    results = enhanced_dice_evaluation(
        args.base_dir, 
        args.modality1, 
        args.series1, 
        args.modality2, 
        args.series2, 
        tumor_list,
        args.patient
    )
    
    # Print a summary of the results
    print("\nResults Summary:")
    for index, row in results.iterrows():
        if row["dice_score"] != "N/A":
            print(f"{row['tumor']}: Dice Score = {row['dice_score']}")
        else:
            print(f"{row['tumor']}: {row['note']}")
# comment 