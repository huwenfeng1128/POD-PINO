# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import hilbert # 导入 hilbert
import time
from tqdm import tqdm
import math
from joblib import Parallel, delayed
import numpy.fft
import traceback

# --- START: Force Joblib Temp Folder to Pure ASCII Path ---
# This part is crucial if you encounter issues with Joblib's temporary files
joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp'
print(f"Setting joblib temporary folder to: {joblib_temp_folder}")
try:
    os.makedirs(joblib_temp_folder, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    print(f"Successfully set JOBLIB_TEMP_FOLDER environment variable.")
except Exception as e:
    print(f"ERROR: Could not create or set joblib temp folder '{joblib_temp_folder}'.")
    print(f"Please manually create this folder or choose a different ASCII path where you have write permissions.")
    print(f"Error details: {e}")
# --- END: Force Joblib Temp Folder ---


# --- Configuration for Data KM Calculation ---
# Input directory containing the simulation result CSV files
INPUT_DIR_SIM_DATA = r'/AFP/P(A,t)_data/sim_data/sim_data_for_test'
# Output directory for KM data
OUTPUT_DIR_KM_DATA = r'/AFP/P(A,t)_data/km_data_for_test'
os.makedirs(OUTPUT_DIR_KM_DATA, exist_ok=True) # Create output dir if needed

# Parameters for tau and A selection
N_tau = 10          # Number of time delays
N_A = 49            # Number of amplitude points
F_FILTER = 100       # Filter width (Hz). Set to None if unknown or not applicable.

# Parallel processing settings for Data KM calculation
N_JOBS_FILE_PROCESSING = 12 # Use all available CPU cores for processing files
N_JOBS_KM_CALCULATION = 2 # Set to 1 as inner parallelization is handled by joblib within _compute_km_for_single_tau

# ==============================================
# 2. 镜像延拓函数定义 (重新添加)
# ==============================================
def mirror_extension(signal, extension_len=500):
    """对信号进行镜像延拓以抑制边界效应"""
    left_ext = signal[:extension_len][::-1]  # 左镜像
    right_ext = signal[-extension_len:][::-1]  # 右镜像
    return np.concatenate([left_ext, signal, right_ext])


# --- Helper Functions (Relevant to KM calculation) ---

def calculate_autocorrelation(signal):
    """Calculates the normalized autocorrelation using FFT."""
    n = len(signal)
    signal_detrended = signal - np.mean(signal)
    fft_signal = np.fft.fft(signal_detrended, n=2*n-1)
    autocorr = np.fft.ifft(fft_signal * np.conj(fft_signal))
    autocorr = autocorr[:n].real
    if autocorr[0] == 0: return np.zeros_like(autocorr), 0
    autocorr /= autocorr[0]
    return autocorr, autocorr[0]

def find_tau_A(autocorr, dt):
    """Finds tau_A where autocorrelation drops to 1/4."""
    try:
        indices = np.where(autocorr <= 0.25)[0]
        if len(indices) > 0:
            first_index = indices[0]
            if first_index == 0 and len(indices) > 1: first_index = indices[1]
            elif first_index == 0:
                 # print("Warning: Autocorrelation drops below 0.25 immediately. Using default tau_A.")
                 return 10 * dt # Use a small default if it drops immediately
            return first_index * dt
        else:
            # print("Warning: Autocorrelation never dropped to 1/4. Using half signal length for tau_A estimate.")
            return (len(autocorr) / 2) * dt # Fallback
    except IndexError:
        # print("Warning: Could not find tau_A. Using default value.")
        return (len(autocorr) / 2) * dt # Fallback

def _compute_km_for_single_tau(tau_idx, A_original_local, A_selected_local, dt_local):
    """Helper function to compute KM coeffs for one tau_idx (for parallelization)."""
    if tau_idx >= len(A_original_local): return tau_idx, None
    A_t = A_original_local[:-tau_idx]
    A_t_tau = A_original_local[tau_idx:]
    A_min_kde, A_max_kde = np.min(A_original_local), np.max(A_original_local)
    try:
        data_stack = np.vstack([A_t, A_t_tau])
        if len(A_t) < 2: return tau_idx, None
        # Add small noise if data is constant to avoid KDE errors
        if np.all(A_t == A_t[0]) and np.all(A_t_tau == A_t_tau[0]):
             data_stack += np.random.normal(0, 1e-6 * (A_max_kde - A_min_kde + 1e-9), data_stack.shape)
        # Check for zero standard deviation
        if np.std(A_t) < 1e-9 or np.std(A_t_tau) < 1e-9: return tau_idx, None
        kernel = gaussian_kde(data_stack)
        kde_marginal = gaussian_kde(A_t)
    except (np.linalg.LinAlgError, ValueError) as e:
        # print(f"Error computing KDE for tau={tau_idx*dt_local:.4f}s (index {tau_idx}): {e}. Skipping.")
        return tau_idx, None

    D1_tau, D2_tau = [], []
    for A in A_selected_local:
        y_range = A_max_kde - A_min_kde
        # Estimate local standard deviation for y_grid range
        A_neighborhood = A_t[np.abs(A_t - A) < y_range * 0.1]
        y_std_est = np.std(A_t_tau[np.isin(A_t, A_neighborhood)]) if len(np.isin(A_t, A_neighborhood)) > 1 else y_range * 0.1
        y_std_est = max(y_std_est, y_range * 0.01) # Ensure a minimum range
        y_grid_min = max(A_min_kde, A - 5*y_std_est)
        y_grid_max = min(A_max_kde, A + 5*y_std_est)
        y_grid = np.linspace(y_grid_min, y_grid_max, 300) # Use a reasonable number of points
        dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 0
        if dy <= 0:
            D1_tau.append(np.nan)
            D2_tau.append(np.nan)
            continue

        joint_points = np.vstack([np.full_like(y_grid, A), y_grid])
        pdf_joint_values = kernel(joint_points)
        marginal_at_A = kde_marginal(A)[0]
        if marginal_at_A > 1e-10:
            pdf_conditional = pdf_joint_values / marginal_at_A
        else:
            pdf_conditional = np.zeros_like(pdf_joint_values)
        integral_cond = np.trapz(pdf_conditional, y_grid)
        if integral_cond > 1e-10:
            pdf_conditional /= integral_cond # Normalize the conditional PDF
        else:
            pdf_conditional = np.zeros_like(pdf_conditional)

        delta_t_actual = tau_idx * dt_local
        M1 = np.trapz((y_grid - A) * pdf_conditional, y_grid)
        M2 = np.trapz((y_grid - A)**2 * pdf_conditional, y_grid)
        if delta_t_actual > 1e-15: # Avoid division by near-zero tau
            D1_finite = M1 / (1 * delta_t_actual)
            D2_finite = M2 / (2 * delta_t_actual)
        else:
            D1_finite = np.nan
            D2_finite = np.nan
        D1_tau.append(D1_finite)
        D2_tau.append(D2_finite)

    result_dict = {'A': A_selected_local, 'D1': np.array(D1_tau), 'D2': np.array(D2_tau)}
    return tau_idx, result_dict

def compute_finite_time_km_coefficients_and_save(file_path, N_tau, N_A, F_FILTER, n_jobs_km, output_dir):
    """
    Computes finite-time KM coefficients for a single data file and saves them.
    Returns True if successful, False otherwise.
    """
    original_file_name = os.path.basename(file_path)
    output_file_name = f"{os.path.splitext(original_file_name)[0]}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        # print(f"Skipping {original_file_name}: KM data already exists at {output_file_path}")
        return True # Indicate success as it's already done

    try:
        df = pd.read_csv(file_path)
        # Assuming the envelope column is named 'Envelope' or 'eta_filtered_envelope'
        # You might need to adjust this based on your simulation output
        if 'Envelope' in df.columns:
            A_original = df['Envelope'].values
        elif 'eta_filtered' in df.columns:
             # If only filtered eta is available, compute envelope from it
            # print(f"Warning: 'Envelope' column not found in {os.path.basename(file_path)}. Computing envelope from 'eta_filtered'.")
            extended_eta_filtered = mirror_extension(df['eta_filtered'].values, 500)
            analytic_signal_filtered = hilbert(extended_eta_filtered)
            A_original = np.abs(analytic_signal_filtered)[500:-500]
        elif 'Eta' in df.columns:
            # If only original eta is available, compute envelope from it
            # print(f"Warning: 'Envelope' or 'eta_filtered' column not found in {os.path.basename(file_path)}. Computing envelope from 'Eta'.")
            extended_eta = mirror_extension(df['Eta'].values, 500)
            analytic_signal = hilbert(extended_eta)
            A_original = np.abs(analytic_signal)[500:-500]
        else:
            print(f"Error: Could not find a suitable column ('Envelope', 'eta_filtered', or 'Eta') in {original_file_name}. Skipping.")
            return False

        if 'Time (s)' in df.columns:
            dt = np.mean(np.diff(df['Time (s)'].values))
        else:
            dt = 0.0001 # Fallback if time column is missing

        if len(A_original) < 1000: # Basic check for sufficient data points
             # print(f"Warning: Data in {original_file_name} is too short ({len(A_original)} points). Skipping KM calculation.")
             return False

        # Determine Time Delays (tau)
        autocorr, kAA0 = calculate_autocorrelation(A_original)
        tau_A_sec = find_tau_A(autocorr, dt)
        tau_min_sampling = dt
        tau_min_filter = (1.0 / F_FILTER) if F_FILTER is not None and F_FILTER > 0 else 0
        tau_min_sec = max(tau_min_sampling, tau_min_filter)
        tau_max_sec = 2 * tau_A_sec
        max_possible_tau_sec = (len(A_original) - 1) * dt
        if tau_max_sec <= tau_min_sec:
            # print(f"Warning: Calculated tau_max <= tau_min for {original_file_name}. Adjusting tau_max.")
            tau_max_sec = min(max(tau_min_sec * 10, tau_max_sec), max_possible_tau_sec * 0.5)
        if tau_max_sec > max_possible_tau_sec:
             # print(f"Warning: Calculated tau_max exceeds data length for {original_file_name}. Clamping tau_max.")
             tau_max_sec = max_possible_tau_sec

        tau_values_sec = np.linspace(tau_min_sec, tau_max_sec, N_tau)
        tau_indices = (tau_values_sec / dt).astype(int)
        tau_indices = np.maximum(1, tau_indices)
        tau_indices = np.unique(tau_indices)
        N_tau_actual = len(tau_indices)

        # Select Amplitude Points (A)
        A_min_data, A_max_data = np.min(A_original), np.max(A_original)
        if A_max_data - A_min_data < 1e-9: # Handle cases with nearly constant amplitude
             # print(f"Warning: Amplitude is nearly constant in {original_file_name}. Skipping KM calculation.")
             return False
        A_selected = np.linspace(A_min_data + 0.15 * (A_max_data - A_min_data),
                                 A_max_data - 0.15* (A_max_data - A_min_data),
                                 N_A)

        # Compute Finite-Time KM Coefficients from Data (Parallelized over tau)
        results_list = Parallel(n_jobs=n_jobs_km, backend='loky')(
            delayed(_compute_km_for_single_tau)(tau_idx, A_original, A_selected, dt)
            for tau_idx in tau_indices
        )

        finite_time_km = {}
        valid_tau_indices = []
        for tau_idx, result_data in results_list:
            if result_data is not None:
                 nan_frac_d1 = np.sum(np.isnan(result_data['D1'])) / len(result_data['D1']) if len(result_data['D1']) > 0 else 1.0
                 nan_frac_d2 = np.sum(np.isnan(result_data['D2'])) / len(result_data['D2']) if len(result_data['D2']) > 0 else 1.0
                 if nan_frac_d1 < 0.9 and nan_frac_d2 < 0.9: # Consider valid if < 90% NaNs
                     finite_time_km[tau_idx] = result_data
                     valid_tau_indices.append(tau_idx)
                 # else:
                     # print(f"Warning: Excessive NaNs in KM results for tau_idx={tau_idx} in {original_file_name}. Skipping.")

        if not finite_time_km:
            # print(f"Warning: No valid KM coefficients computed for {original_file_name}. Skipping.")
            return False

        # Create DataFrame for KM coefficients
        data_km_df_list = []
        for tau_idx in sorted(finite_time_km.keys()):
            data = finite_time_km[tau_idx]
            tau_sec = tau_idx * dt
            temp_df = pd.DataFrame({'tau_index': tau_idx, 'tau_sec': tau_sec, 'A': data['A'],
                                   'D1_data': data['D1'], 'D2_data': data['D2']})
            data_km_df_list.append(temp_df)

        if data_km_df_list:
            all_data_km_df = pd.concat(data_km_df_list, ignore_index=True)
            # Save the result immediately
            all_data_km_df.to_csv(output_file_path, index=False)
            # print(f"Successfully computed and saved KM for {original_file_name} to {output_file_path}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error processing file {original_file_name}: {e}")
        traceback.print_exc() # Print detailed traceback
        return False

# --- Main Execution for Data KM Calculation ---

if __name__ == '__main__': # Ensure this block runs only when the script is executed directly
    print("--- Starting Batch Data KM Coefficient Calculation ---")

    # Get list of simulation data files
    sim_data_files = [os.path.join(INPUT_DIR_SIM_DATA, f) for f in os.listdir(INPUT_DIR_SIM_DATA) if f.endswith('.csv')]
    print(f"Found {len(sim_data_files)} simulation data files in {INPUT_DIR_SIM_DATA}")

    if not sim_data_files:
        print("No simulation data files found. Exiting.")
        exit()

    # Filter out files that already have KM data
    files_to_process = []
    skipped_count = 0
    for file_path in sim_data_files:
        original_file_name = os.path.basename(file_path)
        output_file_name = f"{os.path.splitext(original_file_name)[0]}.csv"
        output_file_path = os.path.join(OUTPUT_DIR_KM_DATA, output_file_name)
        if os.path.exists(output_file_path):
            skipped_count += 1
            # print(f"Skipping {original_file_name}: KM data already exists.") # Optional: print skipped files
        else:
            files_to_process.append(file_path)

    print(f"Skipped {skipped_count} files (KM data already exists).")
    print(f"Will process {len(files_to_process)} new/remaining files.")

    if not files_to_process:
        print("No new files to process. Exiting.")
        exit()

    print(f"\n--- Step 1: Processing {len(files_to_process)} files in parallel using {N_JOBS_FILE_PROCESSING} jobs ---")
    start_batch_time = time.time()

    # Use Parallel to distribute the work. Each worker will call compute_finite_time_km_coefficients_and_save
    # which now handles saving internally.
    results = Parallel(n_jobs=N_JOBS_FILE_PROCESSING, backend='loky')(
        delayed(compute_finite_time_km_coefficients_and_save)(
            file_path, N_tau, N_A, F_FILTER, N_JOBS_KM_CALCULATION, OUTPUT_DIR_KM_DATA
        ) for file_path in tqdm(files_to_process, desc="Processing files")
    )

    successful_count = sum(results)
    failed_count = len(results) - successful_count
    # Note: `failed_files` list can't be directly populated here without modifying the worker to return file names on failure.
    # For now, we'll just report counts.

    end_batch_time = time.time()
    print(f"\n--- Batch Data KM Coefficient Calculation Finished ---")
    print(f"Total files considered: {len(sim_data_files)}")
    print(f"Files skipped (already processed): {skipped_count}")
    print(f"Files attempted for processing: {len(files_to_process)}")
    print(f"Successfully computed and saved KM for {successful_count} files.")
    print(f"Failed to compute KM for {failed_count} files.")
    print(f"Total batch processing time: {end_batch_time - start_batch_time:.2f} seconds.")