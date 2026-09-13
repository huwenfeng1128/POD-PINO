import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad  # For numerical integration to find normalization constant
import os

# --- Configuration ---
# Path to the Excel file containing the filenames and parameters
PARAMETERS_EXCEL_FILE = r'D:\PINN\zenodo\POD_deeponet\SI_result_v3_matched\batch_summary_v3.xlsx'  # Corrected path and implied file type
SIM_DATA_BASE_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'  # Base directory for simulation data
OUTPUT_DIR_PLOT = r'D:\PINN\zenodo\POD_deeponet\PDF_comparison'  # Directory to save plots for batch processing

# Ensure output directory exists 
os.makedirs(OUTPUT_DIR_PLOT, exist_ok=True)


# --- Function to calculate Stationary PDF ---
def stationary_pdf(a, nu, kappa, d_diffusion):
    """Calculates the unnormalized stationary PDF for the given parameters."""
    if d_diffusion <= 0:
        return np.zeros_like(a)  # Handle non-positive diffusion
    # Ensure a is non-negative for log(a) and physical meaning
    a = np.maximum(a, 1e-9)  # Avoid log(0)
    return a * np.exp((nu / (2 * d_diffusion)) * a ** 2 - (kappa / (32 * d_diffusion)) * a ** 4)


def normalized_stationary_pdf(a, nu, kappa, d_diffusion):
    """Calculates the normalized stationary PDF."""
    # Define the unnormalized function for integration
    unnormalized_func = lambda x: stationary_pdf(x, nu, kappa, d_diffusion)

    # Perform numerical integration from 0 to infinity to find the normalization constant
    # Use a sufficiently large upper limit for integration
    try:
        integral_value, _ = quad(unnormalized_func, 0, 50, limit=1000)  # Integrate up to a reasonable limit
        if integral_value <= 0:
            print(
                f"Warning: Integral for normalization is zero or negative for params nu={nu}, k={kappa}, d={d_diffusion}. Returning zero PDF.")
            return np.zeros_like(a)
        normalization_constant = 1.0 / integral_value
    except Exception as e:
        print(f"Error during numerical integration for normalization (params nu={nu}, k={kappa}, d={d_diffusion}): {e}")
        return np.zeros_like(a)  # Return zero PDF on integration error

    # Calculate the normalized PDF
    return normalization_constant * stationary_pdf(a, nu, kappa, d_diffusion)


# --- Main Execution ---

# Read the parameters Excel file
print(f"Reading parameters from {PARAMETERS_EXCEL_FILE}...")
try:
    # *** IMPORTANT CHANGE HERE: Use pd.read_excel() for .xlsx files ***
    df_params = pd.read_excel(PARAMETERS_EXCEL_FILE)
    print("Parameters Excel file read successfully.")
except FileNotFoundError:
    print(f"Error: Parameters Excel file not found at {PARAMETERS_EXCEL_FILE}")
    exit()
except Exception as e:
    print(f"An error occurred while reading parameters Excel file: {e}")
    exit()

# Iterate through each row in the parameters DataFrame
for index, row in df_params.iterrows():
    current_filename = row['filename']

    # Construct the full path for the simulation data file
    DATA_FILE_SIM = os.path.join(SIM_DATA_BASE_DIR, current_filename)

    # Update parameters for DeepOnet based on 'nu_optimz', 'kappa_opt', 'd_diffusion'
    params_deeponet = {
        'nu': row['nu_optimized'],
        'kappa': row['kappa_optimized'],
        'd_diffusion': row['d_diffusion_optimized']
    }

    # Update parameters for Filename based on 'n
    params_filename = {
        'nu': row['nu_standard'],
        'kappa': row['kappa_standard'],
        'd_diffusion': row['D_standard']
    }

    print(f"\n--- Processing {current_filename} ---")
    print(f"DeepOnet Params: {params_deeponet}")
    print(f"Filename Params: {params_filename}")

    # 1. Read Data
    print(f"Reading simulation data from {DATA_FILE_SIM}...")
    try:
        # This part correctly uses pd.read_excel(), assuming your sim_data files are also .xlsx
        # If your sim_data files are .csv, you should change this back to pd.read_csv()
        df_sim = pd.read_csv(DATA_FILE_SIM) # Reverting to pd.read_csv as per original problem description for sim_data
        envelope_data = df_sim['Envelope'].values
        print("Data read successfully.")
    except FileNotFoundError:
        print(f"Error: Simulation data file not found at {DATA_FILE_SIM}. Skipping this file.")
        continue  # Skip to the next iteration if file not found
    except KeyError:
        print(f"Error: 'Envelope' column not found in {DATA_FILE_SIM}. Skipping this file.")
        continue
    except Exception as e:
        print(f"An error occurred while reading simulation data for {current_filename}: {e}. Skipping this file.")
        continue

    # 2. Plot Data Histogram
    print("Plotting data histogram...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))  # Slightly larger figure for curves

    # Plot normalized histogram (density=True gives probability density)
    plt.hist(envelope_data, bins=100, density=True, alpha=0.7, label='Data Histogram (Normalized)', color='skyblue')

    # 3. Calculate and Plot Stationary PDFs
    print("Calculating and plotting stationary PDFs from DeepOnet and Filename parameters...")

    # Define a range of amplitude values for plotting the PDF curves
    a_values = np.linspace(0, np.max(envelope_data) * 1.2, 500)  # Extend slightly beyond max data amplitude

    # DeepOnet Parameters Case
    pdf_deeponet = normalized_stationary_pdf(a_values, **params_deeponet)
    plt.plot(a_values, pdf_deeponet, label=f'Stationary PDF (DeepOnet)', color='red', linewidth=2)

    # Filename Parameters Case
    pdf_filename = normalized_stationary_pdf(a_values, **params_filename)
    plt.plot(a_values, pdf_filename, label=f'Stationary PDF (Filename Params)', color='purple', linewidth=2,
             linestyle=':')

    # 4. Finalize Plot
    plt.title(f'Comparison for {current_filename.replace(".csv", "")}')
    plt.xlabel('Amplitude (A)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim(0, np.max(envelope_data) * 1.3)  # Set x-limit based on data
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    # Create a more descriptive filename based on the current_filename
    output_plot_path_pdf = os.path.join(OUTPUT_DIR_PLOT, f"pdf_comparison_{current_filename.replace('.csv', '')}.png")
    try:
        plt.savefig(output_plot_path_pdf, dpi=300)
        print(f"Comparison plot saved to {output_plot_path_pdf}")
    except Exception as e:
        print(f"Error saving comparison plot for {current_filename}: {e}")

    plt.close()  # Close the plot figure to free up memory

print("\nBatch processing finished.")