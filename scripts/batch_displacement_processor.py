"""
Description:
    This script processes multiple AmigoPy-style CSV files from tumour segmentations,
    computes the superior–inferior (I–S, Z-axis) displacement using center of mass (COM),
    and generates a displacement plot for each scan across breathing phases.

Usage:
    - Place your AmigoPy output CSVs in a folder.
    - Each CSV should include columns: "phase", "axis", "com".
    - The script will calculate displacement relative to a reference phase (default = '0in').
    - Output: PNG plot files showing displacement across breathing phases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def compute_displacement(df, reference_phase='0in'):
    """
    Computes Z-direction displacement relative to a reference phase.

    Parameters:
    - df: DataFrame with 'phase', 'axis', 'com' columns
    - reference_phase: Phase label to use as baseline, e.g., '0in'

    Returns:
    - DataFrame with added 'z_displacement' column
    """
    df = df[df['axis'] == 'z'].copy()
    df = df.sort_values(by='phase')

    if reference_phase not in df['phase'].values:
        raise ValueError(f"Reference phase '{reference_phase}' not found in data.")

    ref_val = df[df['phase'] == reference_phase]['com'].values[0]
    df['z_displacement'] = df['com'] - ref_val
    return df[['phase', 'z_displacement']]

def plot_displacement(df, label, output_path):
    """
    Plots I–S displacement curve for one scan.

    Parameters:
    - df: DataFrame with 'phase' and 'z_displacement'
    - label: Title/label for the plot
    - output_path: Path to save the output plot
    """
    plt.figure()
    plt.plot(df['phase'], df['z_displacement'], marker='o', label=label)
    plt.xlabel('Breathing Phase')
    plt.ylabel('Displacement (mm)')
    plt.title(f'I–S Tumour Displacement: {label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_displacement_folder(folder_path, reference_phase='0in'):
    """
    Processes all CSV files in a folder to compute and plot I–S displacement.

    Parameters:
    - folder_path: Directory containing AmigoPy-style CSVs
    - reference_phase: Phase to use as displacement baseline
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_disp = compute_displacement(df, reference_phase=reference_phase)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(folder_path, f"{base_name}_displacement_plot.png")
            plot_displacement(df_disp, base_name, output_path)
            print(f"Processed and saved plot for {base_name}")
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")

# Example usage
if __name__ == "__main__":
    # Edit this path to the folder with your AmigoPy segmentation CSVs
    scan_folder = "C:/AMIGOpy/regular_20amplitude_scan1"
    os.makedirs(scan_folder, exist_ok=True)

    # Run processing
    process_displacement_folder(scan_folder)
