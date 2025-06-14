"""
Tumour Displacement Plotter (Real Data Format)
Author: ChatGPT for breathing irregularity thesis
Description:
    Processes AmigoPy structure_stats CSVs to compute I–S tumour displacement
    for a selected tumour (e.g. 'middle_tumour'), and plots phase-wise Z (SI) movement.

    Adapted for CSVs with columns: 'series_id', 'name', 'z' (centre of mass Z).

Usage:
    - Place your structure_stats_X.csv files in a folder.
    - Update scan_folder and tumour_name in the main section.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Define phase order to match expected breathing cycle sequence
PHASE_ORDER = ['0in', '25in', '50in', '75in', '100in', '75ex', '50ex', '25ex']

def process_structure_csv(file_path, tumour_name='middle_tumour', reference_phase='0in'):
    """
    Process one structure_stats CSV file to extract Z displacement for a specific tumour.

    Parameters:
    - file_path: Path to CSV
    - tumour_name: e.g., 'middle_tumour'
    - reference_phase: baseline phase for displacement (e.g., '0in')

    Returns:
    - DataFrame with phases and displacement
    """
    df = pd.read_csv(file_path)
    df = df[df['name'] == tumour_name].copy()

    # Strip spaces and lowercase for consistent phase labels
    df['series_id'] = df['series_id'].str.strip().str.lower()
    df = df[df['series_id'].isin(PHASE_ORDER)]

    df['series_id'] = pd.Categorical(df['series_id'], categories=PHASE_ORDER, ordered=True)
    df = df.sort_values('series_id')

    if reference_phase not in df['series_id'].values:
        raise ValueError(f"Reference phase '{reference_phase}' not found in {file_path}")

    ref_z = df[df['series_id'] == reference_phase]['z'].values[0]
    df['z_displacement'] = df['z'] - ref_z

    return df[['series_id', 'z_displacement']]

def plot_displacement(df, label, output_path):
    """
    Plot Z-displacement curve.

    Parameters:
    - df: DataFrame with 'series_id' and 'z_displacement'
    - label: plot title
    - output_path: PNG save location
    """
    plt.figure()
    plt.plot(df['series_id'], df['z_displacement'], marker='o', label=label)
    plt.xlabel('Breathing Phase')
    plt.ylabel('Displacement (mm)')
    plt.title(f'I–S Tumour Displacement: {label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_folder(folder_path, tumour_name='middle_tumour', reference_phase='0in'):
    """
    Batch process all CSVs in a folder for a specific tumour name.

    Parameters:
    - folder_path: where the structure_stats files are
    - tumour_name: which tumour to extract (e.g., 'middle_tumour')
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process...")

    for csv_file in csv_files:
        try:
            df_disp = process_structure_csv(csv_file, tumour_name=tumour_name, reference_phase=reference_phase)
            label = os.path.splitext(os.path.basename(csv_file))[0]
            plot_path = os.path.join(folder_path, f"{label}_disp_{tumour_name}.png")
            plot_displacement(df_disp, label, plot_path)
            print(f"✅ Processed {label}")
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")

# Example usage
if __name__ == "__main__":
    scan_folder = "C:/AMIGOpy/regular_20amplitude_scan1"  # update this path
    tumour_name = "middle_tumour"
    process_folder(scan_folder, tumour_name=tumour_name)
