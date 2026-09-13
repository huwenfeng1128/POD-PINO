#绘制所有数据的直方分布图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random # 导入 random 模块以设置种子

# --- Configuration ---
# SIM_DATA_BASE_DIR: Base directory for simulation data
SIM_DATA_BASE_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'

# OUTPUT_DIR_DATA: Directory to save all histogram data
OUTPUT_DIR_DATA = r'D:\PINN\zenodo\POD_deeponet\compare\result\all_histogram_data'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR_DATA, exist_ok=True)

# --- Matplotlib Style Settings ---
plt.rcParams['font.sans-serif'] = ['SimSun']  # Set font to Song Ti (宋体)
plt.rcParams['axes.unicode_minus'] = False     # Correctly display minus sign
plt.rcParams['font.size'] = 10                 # Default font size (can be overridden)


# --- Main Execution ---

# No longer reading parameter CSVs as only histogram data is being extracted.
# The merge step and related parameter processing have been removed.

print(f"--- Generating histograms for all available files ---")

# Dynamically determine the number of subplots needed for a grid layout.
# We'll aim for a roughly square grid.
# First, let's find all potential simulation data files.
# This assumes that each subdirectory within SIM_DATA_BASE_DIR might contain relevant CSVs.
# If SIM_DATA_BASE_DIR directly contains the CSVs, this part would need adjustment.
# For now, we'll assume it lists files directly or we can infer files from it.

# A more robust way to get all relevant files if they are directly in SIM_DATA_BASE_DIR
# or in subfolders. If they are consistently named and in one place,
# we can infer the list of files from the simulation data directory.
# For simplicity, and based on the previous script's logic which merged based on filenames,
# we'll list all CSV files that have corresponding simulation data.
# However, without the parameter files, we can't *filter* by common filenames anymore.
# So, we'll process ALL simulation data files found in SIM_DATA_BASE_DIR.

all_sim_files = [f for f in os.listdir(SIM_DATA_BASE_DIR) if f.endswith('.csv')]

if not all_sim_files:
    print(f"Error: No CSV files found in the simulation data directory: {SIM_DATA_BASE_DIR}")
    exit()

print(f"Found {len(all_sim_files)} potential simulation data files.")

n_files = len(all_sim_files)
n_cols = int(np.ceil(np.sqrt(n_files)))
n_rows = int(np.ceil(n_files / n_cols))

# Create figure and axes, ensuring it's always a 2D array even if n_rows or n_cols is 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
axes = axes.flatten() # Flatten the axes array for easy iteration

# Remove any unused subplots
for j in range(n_files, len(axes)):
    fig.delaxes(axes[j])

hist_bins = np.linspace(0, 8, 100) # Define histogram bins once for consistency

for i, current_filename in enumerate(all_sim_files):
    ax = axes[i]
    # Clean filename for use in paths and titles
    clean_filename = current_filename.replace('.csv', '').replace('(', '').replace(')', '').replace(',', '_')

    DATA_FILE_SIM = os.path.join(SIM_DATA_BASE_DIR, current_filename)

    print(f"Processing {current_filename} for subplot {i + 1}...")

    # 1. Read Data
    try:
        df_sim = pd.read_csv(DATA_FILE_SIM)
        # Assuming the column name for amplitude is 'Envelope' based on previous script
        if 'Envelope' not in df_sim.columns:
            raise KeyError("'Envelope' column not found.")
        envelope_data = df_sim['Envelope'].values
    except FileNotFoundError:
        print(f"Error: Simulation data file not found at {DATA_FILE_SIM}. Skipping subplot {i + 1}.")
        ax.text(0.5, 0.5, "Data Not Found", transform=ax.transAxes, color='red',
                fontsize=12, ha='center', va='center')
        continue
    except KeyError as e:
        print(f"Error: {e} in {DATA_FILE_SIM}. Skipping subplot {i + 1}.")
        ax.text(0.5, 0.5, f"Column Error: {e}", transform=ax.transAxes, color='red',
                fontsize=12, ha='center', va='center')
        continue
    except Exception as e:
        print(f"An unexpected error occurred while reading simulation data for {current_filename}: {e}. Skipping subplot {i + 1}.")
        ax.text(0.5, 0.5, f"Read Error: {str(e)[:20]}...", transform=ax.transAxes, color='red',
                fontsize=12, ha='center', va='center')
        continue

    # 2. Plot Data Histogram and capture histogram data
    hist_counts, bin_edges = np.histogram(envelope_data, bins=hist_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.hist(envelope_data, bins=hist_bins, density=True, alpha=0.7, label='Simulation Data', color='skyblue')

    # --- Save Data for this subplot ---
    df_hist_data = pd.DataFrame({
        'Amplitude_Bin_Center': bin_centers,
        'Probability_Density_Histogram': hist_counts
    })
    hist_data_path = os.path.join(OUTPUT_DIR_DATA, f'{clean_filename}_histogram_data.csv')
    try:
        df_hist_data.to_csv(hist_data_path, index=False)
        print(f"Saved histogram data to {hist_data_path}")
    except Exception as e:
        print(f"Error saving histogram data to {hist_data_path}: {e}")

    # --- Subplot Styling ---
    ax.set_title(f'{clean_filename}', fontsize=10) # Title for each subplot
    ax.tick_params(axis='both', which='major', labelsize=10) # Axis numbers size
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.2) # Ensure consistent Y-axis limits

    # Only add legend to the first subplot to avoid clutter
    if i == 0:
        ax.legend(fontsize=10, loc='upper right')


# --- Global X and Y Labels ---
# Hide individual axis labels and set a single shared label for the entire figure
for ax in axes:
    ax.set_xlabel('') # Hide individual x-labels
    ax.set_ylabel('') # Hide individual y-labels

# Add shared labels for the entire figure
fig.supxlabel('Amplitude (A)', fontsize=16, fontfamily='SimSun', y=0.02) # Global X-label at the bottom
fig.supylabel('Probability Density', fontsize=16, fontfamily='SimSun', x=0.03) # Global Y-label on the left

# Adjust layout and save the merged plot
# Use tight_layout with rect to make space for supxlabel/supylabel
plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95]) # Adjust rect to make space for global labels

output_plot_path_merged = os.path.join(OUTPUT_DIR_DATA, "all_histogram_comparison.svg") # Changed to SVG
try:
    plt.savefig(output_plot_path_merged, format='svg', dpi=300) # Save as SVG
    print(f"\nMerged histogram comparison plot saved to {output_plot_path_merged}")
except Exception as e:
    print(f"Error saving merged histogram comparison plot: {e}")

plt.show()
plt.close(fig)

print("\nScript finished.")