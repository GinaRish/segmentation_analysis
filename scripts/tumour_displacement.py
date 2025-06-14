import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("C:/AMIGOpy/HyperSight Phantom/structure_stats.csv")  # Adjust path as needed
df = df[['series_id', 'name', 'x', 'y', 'z']].dropna()

# Filter out rows where series_id contains "average" (case-insensitive)
if df['series_id'].dtype == object:  # Check if series_id is string type
    df = df[~df['series_id'].str.lower().str.contains('average')]
    print(f"Filtered data to remove 'average' entries. Remaining rows: {len(df)}")

# Check if series_id is numeric or breathing-phase-like
def is_numeric(s):
    try:
        float(s)
        return True
    except:
        return False

# Determine handling strategy
if 'series_id' in df.columns and not df['series_id'].apply(is_numeric).all():
    # Corrected phase order - properly formatted as a list with quotes
    phase_order = ['0in', '25in', '50in', '75in', '100in', '75ex', '50ex', '25ex']
    
    # Filter the phase order list to only include phases that exist in the data
    existing_phases = [phase for phase in phase_order if phase in df['series_id'].unique()]
    
    # Print for debugging
    print(f"Found phases in data: {df['series_id'].unique().tolist()}")
    print(f"Using phase order: {existing_phases}")
    
    # Convert to categorical for proper sorting
    df['series_id'] = pd.Categorical(df['series_id'], categories=existing_phases, ordered=True)
    df = df.sort_values('series_id')
else:  
    if df['series_id'].apply(is_numeric).all():
        df['series_id'] = df['series_id'].astype(float)
    df = df.sort_values('series_id')

# Tumour types
tumour_names = ['lower_tumour', 'middle_tumour', 'upper_tumour']

# Create figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Tumour Displacement Across Breathing Phases or Numeric Series', fontsize=16)

# Plot Euclidean displacement for each tumour
for i, tumour in enumerate(tumour_names):
    tumour_data = df[df['name'] == tumour]
    
    if len(tumour_data) > 0:
        displacement = (tumour_data[['x', 'y', 'z']]**2).sum(axis=1)**0.5
        axs[i].plot(tumour_data['series_id'], displacement, marker='o')
        axs[i].set_title(tumour.replace('_', ' ').title())
        axs[i].set_ylabel('Displacement (mm)')
        axs[i].grid(True)
    else:
        axs[i].text(0.5, 0.5, f"No data for {tumour.replace('_', ' ').title()}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axs[i].transAxes)
        axs[i].set_title(tumour.replace('_', ' ').title())
        axs[i].set_ylabel('Displacement (mm)')
        axs[i].grid(True)

axs[-1].set_xlabel('Series ID (Phase or Numeric)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()