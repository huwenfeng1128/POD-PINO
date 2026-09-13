#批量化计算所有数据的FD识别参数

# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.sparse import diags, identity, csc_matrix, lil_matrix # Use LIL for building
from scipy.sparse.linalg import spsolve # For solving sparse linear systems
import time
from tqdm import tqdm
import math
from joblib import Parallel, delayed # For parallel processing
import numpy.fft
import traceback # For detailed error printing

# --- START: Force Joblib Temp Folder to Pure ASCII Path ---
# Define and create the required ASCII path for joblib temporary files
joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp' # Explicit ASCII path

print(f"Setting joblib temporary folder to: {joblib_temp_folder}")
# Create the directory if it doesn't exist
try:
    os.makedirs(joblib_temp_folder, exist_ok=True)
    # Set the environment variable for joblib
    os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    print(f"Successfully set JOBLIB_TEMP_FOLDER environment variable.")
except Exception as e:
    print(f"ERROR: Could not create or set joblib temp folder '{joblib_temp_folder}'.")
    print(f"Please manually create this folder or choose a different ASCII path where you have write permissions.")
    print(f"Error details: {e}")
    # Exit if we cannot guarantee a safe temp folder for joblib
    exit()
# --- END: Force Joblib Temp Folder ---


# --- Configuration ---
# Base directory for all KM data files (e.g., (18.18,4.54,18.18).csv)
BASE_KM_DATA_DIR = r'/AFP/P(A,t)_data/km_data_4'
# Base directory for all raw simulation data files (e.g., simulation_results_nu18.18_kappa4.54_D18.18.csv)
BASE_RAW_DATA_DIR = r'/AFP/P(A,t)_data/sim_data/sim_data_4_best'
# Base directory for all output results
BASE_OUTPUT_DIR = r'/AFP/result/grid_fp_batch_run_4'  # New base output directory for batch runs
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True) # Create base output dir if needed

# Parameters for tau selection (these will be determined per file)
F_FILTER = 60    # Filter width (Hz). Set to None if unknown or not applicable.
OMEGA_0 = 2 * math.pi * 150 # Central frequency (rad/s) - Used only for equivalent gamma estimate

# Optimization settings
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {'maxiter': 300, 'disp': True, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}

# Parallel processing settings
N_JOBS = -1 # Use all available CPU cores (-1 means all cores)

# Fokker-Planck Solver settings (Adjusted for potential stiffness due to 1/A term)
FP_N_GRID = 300    # Increased spatial resolution
CN_DT_STEP = 1e-3   # Significantly decreased time step for stability
D_DIFFUSION_UPPER_BOUND = 100.0 # Upper bound constraint for d_diffusion in objective func

# --- Helper Functions ---

def parse_params_from_filename(filepath):
    """
    Parses nu, kappa, D from filenames like '(18.18,4.54,18.18).csv' using split.
    This function is intended for KM data filenames, not raw simulation data filenames.
    Returns (nu, kappa, D) as floats, or (None, None, None) if parsing fails.
    """
    filename = os.path.basename(filepath)
    nu_std, kappa_std, D_std = None, None, None

    try:
        # This part is for KM data files like (18.18,4.54,18.18).csv
        if filename.startswith('(') and filename.endswith('.csv'):
            inner_part = filename[1:-5] # Remove '(' and ').csv'
            values = inner_part.split(',')
            if len(values) == 3:
                nu_std = float(values[0])
                kappa_std = float(values[1])
                D_std = float(values[2])
        # The 'simulation_results_nuX_kappaY_D.csv' format is NOT parsed here for standard values.
        # It's assumed that the KM data filename provides the ground truth parameters.
    except (ValueError, IndexError) as e:
        # print(f"Warning: Could not parse parameters from filename '{filename}'. Error: {e}") # Suppress for batch
        nu_std, kappa_std, D_std = None, None, None
    return nu_std, kappa_std, D_std

# --- Theoretical KM Coefficients and Forward FP Solver (Encapsulated) ---

def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """Theoretical Drift Coefficient D1(A) using d_diffusion/A form."""
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    term_nu = nu * A
    term_kappa = (kappa / 8) * A**3
    return term_nu - term_kappa + term_gamma

def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """Theoretical Diffusion Coefficient D2(A) using d_diffusion."""
    A = np.asarray(A)
    return np.full_like(A, max(0.0, d_diffusion))

def create_grid(A_max_data, N_grid=FP_N_GRID):
    """Creates the spatial grid for the FP solver."""
    A_grid_max = 1.5 * A_max_data
    A_grid_min = 0
    A_grid = np.linspace(A_grid_min, A_grid_max, N_grid)
    dA = A_grid[1] - A_grid[0]
    return A_grid, dA, A_grid_max

class FPSolver:
    """Solves the Forward Fokker-Planck equation dP/dt = L P using Crank-Nicolson."""
    def __init__(self, A_grid, dA, nu, kappa, d_diffusion):
        self.A_grid = A_grid
        self.dA = dA
        self.N_grid = len(A_grid)
        self.nu = nu
        self.kappa = kappa
        self.d_diffusion = max(0.0, d_diffusion)
        self.L = None

        try:
            self.D1_values = theoretical_D1_d(self.A_grid, nu, kappa, self.d_diffusion)
            self.D2_values = theoretical_D2_d(self.A_grid, nu, kappa, self.d_diffusion)
            self.D2_values = np.maximum(self.D2_values, 1e-15) # Ensure D2 is strictly positive
        except Exception as e:
            raise ValueError(f"Failed to calculate valid D1/D2: {e}") from e

        try:
            self.L = self._build_forward_operator_matrix()
            if self.L is None or not isinstance(self.L, csc_matrix):
                 raise ValueError("Operator matrix L was not built correctly.")
            if not np.all(np.isfinite(self.L.data)):
                 self.L.data[~np.isfinite(self.L.data)] = 0.0
                 self.L.eliminate_zeros()
        except Exception as e:
            raise ValueError(f"Failed to build operator matrix L: {e}") from e

    def _build_forward_operator_matrix(self):
        N = self.N_grid
        dA = self.dA
        dA2 = dA**2
        L = lil_matrix((N, N), dtype=float)

        safe_D1 = np.nan_to_num(self.D1_values, nan=0.0, posinf=1e10, neginf=-1e10)
        safe_D2 = self.D2_values

        dD1_dA = np.gradient(safe_D1, dA, edge_order=1)
        dD2_dA = np.gradient(safe_D2, dA, edge_order=1)
        d2D2_dA2 = np.gradient(dD2_dA, dA, edge_order=1)

        dD1_dA = np.nan_to_num(dD1_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        dD2_dA = np.nan_to_num(dD2_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        d2D2_dA2 = np.nan_to_num(d2D2_dA2, nan=0.0, posinf=1e10, neginf=-1e10)

        coeff_p = -(dD1_dA - d2D2_dA2)
        coeff_dp_dA = -safe_D1 + 2 * dD2_dA
        coeff_d2p_dA2 = safe_D2

        for i in range(1, N - 1):
            L[i, i-1] = -coeff_dp_dA[i] / (2 * dA) + coeff_d2p_dA2[i] / dA2
            L[i, i]   = coeff_p[i] - 2 * coeff_d2p_dA2[i] / dA2
            L[i, i+1] = coeff_dp_dA[i] / (2 * dA) + coeff_d2p_dA2[i] / dA2

        L[0, :] = 0
        L[0, 0] = 0
        L[N-1, :] = 0
        L[N-1, N-1] = 0

        return L.tocsc()

    def initial_condition_delta(self, A_target):
        p0 = np.zeros(self.N_grid)
        idx = np.argmin(np.abs(self.A_grid - A_target))
        idx = max(0, min(self.N_grid - 1, idx))
        if self.dA > 1e-15:
            p0[idx] = 1.0 / self.dA
        else:
            p0[idx] = 1.0
        return p0

    def solve_forward_cn(self, A_target, tau, dt_step=CN_DT_STEP):
        if self.L is None: return None, "Operator L not built"
        if tau <= 0: return None, f"Invalid tau: {tau}"
        if dt_step <= 0: dt_step = tau / 100.0

        num_steps = max(1, int(round(tau / dt_step)))
        actual_dt_step = tau / num_steps

        P_current = self.initial_condition_delta(A_target)
        if np.sum(P_current) < 1e-10:
             return None, f"Initial condition is near zero for A_target={A_target:.3f}"

        Id = identity(self.N_grid, format='csc')
        try:
            LHS = Id - 0.5 * actual_dt_step * self.L
            RHS = Id + 0.5 * actual_dt_step * self.L
            if not np.all(np.isfinite(LHS.data)) or not np.all(np.isfinite(RHS.data)):
                raise ValueError("NaN/Inf detected in LHS/RHS matrices.")
        except Exception as e:
             return None, f"Failed to build LHS/RHS: {e}"

        try:
            for step in range(num_steps):
                b = RHS.dot(P_current)
                if np.isnan(b).any() or np.isinf(b).any():
                    return None, f"NaN/Inf in RHS vector 'b' at step {step+1}/{num_steps} for A_target={A_target:.3f}, tau={tau:.4f}"

                P_next = spsolve(LHS, b)

                if np.isnan(P_next).any() or np.isinf(P_next).any():
                    return None, f"Instability (NaN/Inf) detected in CN solver at step {step+1}/{num_steps} for A_target={A_target:.3f}, tau={tau:.4f}"
                if np.linalg.norm(P_next) > 1e6 * np.linalg.norm(P_current) and np.linalg.norm(P_current) > 1e-6:
                    pass # Optional: return None, "Potential instability (norm increase)"

                P_current = P_next

        except np.linalg.LinAlgError as e:
             return None, f"Linear algebra error during spsolve at step {step+1}/{num_steps} (matrix may be singular): {e}"
        except Exception as e:
             return None, f"Error during Crank-Nicolson solve step {step+1}/{num_steps}: {e}"

        P_final = np.maximum(P_current, 0)
        integral_P_final = np.trapz(P_final, self.A_grid)
        if integral_P_final > 1e-10:
            P_final /= integral_P_final
        else:
            P_final = np.zeros_like(P_final)
        return P_final, None

    def calculate_km_from_solution(self, P_final, A_target, tau):
        if P_final is None or tau <= 0: return np.nan, np.nan
        if np.sum(P_final) < 1e-10: return 0.0, 0.0

        M1 = np.trapz((self.A_grid - A_target) * P_final, self.A_grid)
        M2 = np.trapz((self.A_grid - A_target)**2 * P_final, self.A_grid)

        if tau > 1e-15:
            D1_fp = M1 / (1 * tau)
            D2_fp = M2 / (2 * tau)
        else:
            D1_fp = np.nan
            D2_fp = np.nan

        D2_fp = max(0.0, D2_fp) if not np.isnan(D2_fp) else np.nan
        return D1_fp, D2_fp

# --- Compute FP-based KM Coefficients Function ---
def compute_fp_km_coefficients(params, A_selected, tau_indices, dt, A_max_data, n_jobs, optimization_history_local):
    """
    Computes KM coefficients by solving the Forward FP equation for given params.
    optimization_history_local is passed to allow printing iteration info.
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1 # Get current iteration for printing

    if np.isnan(nu) or np.isinf(nu) or np.isnan(kappa) or np.isinf(kappa) or np.isnan(d_diffusion) or np.isinf(d_diffusion):
        return None
    if d_diffusion < 0 or kappa < 0:
        return None

    solver = None
    try:
        A_grid, dA, _ = create_grid(A_max_data, N_grid=FP_N_GRID)
        solver = FPSolver(A_grid, dA, nu, kappa, d_diffusion)
    except ValueError as e:
        # print(f"  Iter {current_iter} Error: Failed creating FPSolver with params {params}: {e}") # Suppress for batch
        return None
    except Exception as e:
        # print(f"  Iter {current_iter} Error: Unexpected error creating FPSolver with params {params}: {e}") # Suppress for batch
        return None

    results = {}
    fp_compute_start_time = time.time()

    # Define helper function for parallel execution over A_selected for *this* tau
    def solve_and_calc_km_for_A(A_target_val):
        P_final, error_msg = solver.solve_forward_cn(A_target_val, tau_sec, dt_step=CN_DT_STEP)
        if P_final is None:
            return np.nan, np.nan
        D1, D2 = solver.calculate_km_from_solution(P_final, A_target_val, tau_sec)
        return D1, D2

    for tau_idx in tau_indices:
        tau_sec = tau_idx * dt
        if tau_sec <= 0: continue

        # Parallelize the FP solves over the selected amplitude points for the current tau
        km_results_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(solve_and_calc_km_for_A)(A) for A in A_selected
        )

        D1_fp = np.array([res[0] for res in km_results_list])
        D2_fp = np.array([res[1] for res in km_results_list])

        nan_count_d1 = np.sum(np.isnan(D1_fp))
        nan_count_d2 = np.sum(np.isnan(D2_fp))
        if nan_count_d1 > len(A_selected) * 0.5 or nan_count_d2 > len(A_selected) * 0.5:
             pass # print(f"  Iter {current_iter} Warning: High NaN count ({nan_count_d1} D1, {nan_count_d2} D2) in FP results for tau={tau_sec:.4f}s. Params: {params}") # Suppress for batch

        results[tau_idx] = {'A': A_selected, 'D1': D1_fp, 'D2': D2_fp}

    fp_compute_duration = time.time() - fp_compute_start_time
    results['fp_duration'] = fp_compute_duration
    return results

# --- Objective Function (using FP Solver) ---
def objective_function(params, data_km, A_selected, tau_indices, dt, A_max_data, n_jobs, optimization_history_local):
    """
    Objective function comparing data KM and Forward FP KM.
    optimization_history_local is passed to allow printing iteration info.
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    # Constraints and Parameter Validity Checks
    if np.isnan(nu) or np.isinf(nu) or np.isnan(kappa) or np.isinf(kappa) or np.isnan(d_diffusion) or np.isinf(d_diffusion):
        # print(f"  Iter {current_iter} Eval: Invalid params (NaN/Inf). Cost: 1e12") # Suppress for batch
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e12, 'fp_duration_sec': 0.0, 'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e12

    penalty = 0.0
    if d_diffusion < 0:
        penalty += (abs(d_diffusion) + 1)**2 * 1e6
    if d_diffusion > D_DIFFUSION_UPPER_BOUND:
        penalty += (d_diffusion - D_DIFFUSION_UPPER_BOUND)**2 * 1e4
    if kappa < 0:
        penalty += (abs(kappa) + 1)**2 * 1e6

    if penalty > 0:
        # print(f"  Iter {current_iter} Eval: Params nu={nu:.4f}, k={kappa:.4f}, d={d_diffusion:.4f}. Constraint penalty: {penalty:.2e}. Returning high cost.") # Suppress for batch
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e10 + penalty, 'fp_duration_sec': 0.0, 'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e10 + penalty

    start_fp_time = time.time()
    fp_km_results = compute_fp_km_coefficients(params, A_selected, tau_indices, dt, A_max_data, n_jobs, optimization_history_local)
    end_fp_time = time.time()
    fp_duration = end_fp_time - start_fp_time

    if fp_km_results is None:
        # print(f"  Iter {current_iter} Eval: FP computation failed for params {params}. Cost: 1e11") # Suppress for batch
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e11, 'fp_duration_sec': fp_duration, 'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e11

    fp_duration = fp_km_results.pop('fp_duration', fp_duration)

    total_sq_error = 0
    num_points = 0
    for tau_idx in tau_indices:
        if tau_idx not in data_km or tau_idx not in fp_km_results:
            continue

        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']
        D1_fp = fp_km_results[tau_idx]['D1']
        D2_fp = fp_km_results[tau_idx]['D2']

        if D1_data.shape != D1_fp.shape or D2_data.shape != D2_fp.shape:
            continue

        valid_mask_d1 = ~np.isnan(D1_data) & ~np.isnan(D1_fp)
        valid_mask_d2 = ~np.isnan(D2_data) & ~np.isnan(D2_fp)

        error_D1 = np.sum(((D1_data - D1_fp)**2)[valid_mask_d1])
        error_D2 = np.sum(((D2_data - D2_fp)**2)[valid_mask_d2])

        total_sq_error += error_D1 + error_D2
        num_points += np.sum(valid_mask_d1) + np.sum(valid_mask_d2)

    if num_points == 0:
        mean_sq_error = 1e10
    else:
        mean_sq_error = total_sq_error / num_points

    iteration_data = {
        'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
        'mse': mean_sq_error, 'fp_duration_sec': fp_duration, 'num_compared_points': num_points
    }
    optimization_history_local.append(iteration_data)

    # print(f"  Iter {current_iter} Eval: nu={nu:.5f}, k={kappa:.5f}, d={d_diffusion:.5f} -> MSE={mean_sq_error:.6e} (FP time: {fp_duration:.2f}s, {num_points} pts)") # Suppress for batch

    if np.isnan(mean_sq_error) or np.isinf(mean_sq_error):
        return 1e12

    return mean_sq_error


# --- Main Processing Function for Each KM File (FP Version) ---
def process_km_file_fp(km_file_path, base_raw_data_dir, base_output_dir):
    print(f"\n--- Processing file: {os.path.basename(km_file_path)} ---")

    # 1. Get standard parameters from KM data filename
    nu_standard, kappa_standard, D_standard = parse_params_from_filename(km_file_path)
    if nu_standard is None or kappa_standard is None or D_standard is None:
        print(f"WARNING: Could not extract standard parameters from KM data filename '{os.path.basename(km_file_path)}'. Skipping this file.")
        return None

    # Create a specific output directory for this file
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0].replace('.', '_') # Replace . with _ for valid folder name
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this file will be saved to: {output_dir}")

    # 2. Read Raw Data (for dt and A_max_data)
    # Construct raw data file path using the standard parameters (nu, kappa, D) from KM filename
    # Assuming raw data filenames are consistently formatted as 'simulation_results_nuX_kappaY_D.csv'
    raw_data_filename = f"({nu_standard},{kappa_standard},{D_standard}).csv"
    raw_data_file_path = os.path.join(base_raw_data_dir, raw_data_filename)

    A_max_data = None
    dt = None
    try:
        df_raw = pd.read_csv(raw_data_file_path)
        A_original = df_raw['Envelope'].values
        if 'time' in df_raw.columns:
            dt = np.mean(np.diff(df_raw['time'].values))
        else:
            dt = 0.0001 # Fallback
        A_max_data = np.max(A_original)
        print(f"  Inferred dt: {dt:.6f} s, Max A: {A_max_data:.4f} from raw data '{raw_data_filename}'.")
    except FileNotFoundError:
        print(f"  ERROR: Raw data file '{raw_data_file_path}' not found for '{os.path.basename(km_file_path)}'. Please ensure raw data files exist and match KM data parameters. Skipping this file.")
        return None
    except Exception as e:
        print(f"  Error reading raw data file '{raw_data_file_path}': {e}. Skipping.")
        traceback.print_exc()
        return None

    # 3. Read Pre-calculated Data KM Coefficients
    finite_time_km = {}
    tau_indices = []
    A_selected = None
    N_tau = 0
    N_A = 0

    try:
        finite_time_km_df = pd.read_csv(km_file_path)
        unique_tau_indices = finite_time_km_df['tau_index'].unique()
        unique_A_values = finite_time_km_df['A'].unique()

        if len(unique_tau_indices) == 0 or len(unique_A_values) == 0:
            print(f"  Error: Data KM file '{os.path.basename(km_file_path)}' is empty or missing columns. Skipping.")
            return None

        tau_indices = sorted(unique_tau_indices)
        A_selected = np.sort(unique_A_values)
        N_tau = len(tau_indices)
        N_A = len(A_selected)

        for tau_idx in tau_indices:
            tau_data = finite_time_km_df[finite_time_km_df['tau_index'] == tau_idx].sort_values(by='A')
            if not tau_data.empty:
                finite_time_km[tau_idx] = {
                    'A': tau_data['A'].values,
                    'D1': tau_data['D1_data'].values,
                    'D2': tau_data['D2_data'].values
                }
        if not finite_time_km:
            print(f"  Error: No valid data KM coefficients loaded from {os.path.basename(km_file_path)}. Skipping.")
            return None

        print(f"  Loaded {N_tau} tau values and {N_A} amplitude points from KM data.")

    except Exception as e:
        print(f"  Error reading data KM from {os.path.basename(km_file_path)}: {e}. Skipping.")
        traceback.print_exc()
        return None

    # 4. Parameter Optimization
    # Initial Guess Estimation
    try:
        valid_d2_means = []
        for tau_idx in finite_time_km:
            d2_values = finite_time_km[tau_idx]['D2']
            valid_d2 = d2_values[~np.isnan(d2_values)]
            if len(valid_d2) > 0:
                valid_d2_means.append(np.mean(valid_d2))

        if valid_d2_means:
            avg_D2 = np.mean(valid_d2_means)
            d_diffusion_0 = max(1e-9, avg_D2)
            d_diffusion_0 = min(d_diffusion_0, D_DIFFUSION_UPPER_BOUND * 0.9)
        else:
            d_diffusion_0 = 0.01
    except Exception as e:
         print(f"  Error estimating initial d_diffusion for {os.path.basename(km_file_path)}: {e}. Using default.")
         d_diffusion_0 = 0.01

    nu_0 = 0.1
    kappa_0 = 0.1
    params_0 = [nu_0, kappa_0, d_diffusion_0]

    optimization_history_local = [] # Local history for this file's optimization

    print(f"  Starting optimization for {os.path.basename(km_file_path)}...")
    start_time_opt = time.time()
    result = None
    try:
        result = minimize(
            objective_function,
            params_0,
            args=(finite_time_km, A_selected, tau_indices, dt, A_max_data, N_JOBS, optimization_history_local),
            method=OPTIMIZER_METHOD,
            options=OPTIMIZER_OPTIONS,
        )
    except KeyboardInterrupt:
        print(f"  Optimization for {os.path.basename(km_file_path)} interrupted by user.")
    except Exception as e:
        print(f"  Error during optimization for {os.path.basename(km_file_path)}: {e}")
        traceback.print_exc()

    end_time_opt = time.time()
    print(f"  Optimization for {os.path.basename(km_file_path)} finished in {end_time_opt - start_time_opt:.2f} seconds.")

    # Process Optimization Results
    optimization_successful = False
    final_cost = np.nan
    nu_opt, kappa_opt, d_diffusion_opt = params_0

    if result and hasattr(result, 'success'):
        optimization_successful = result.success
        final_cost = result.fun
        if optimization_successful:
            nu_opt, kappa_opt, d_diffusion_opt = result.x
        else:
            if hasattr(result, 'x'):
                nu_opt, kappa_opt, d_diffusion_opt = result.x
    else:
        if optimization_history_local:
            history_df_temp = pd.DataFrame(optimization_history_local)
            if not history_df_temp.empty and 'mse' in history_df_temp.columns:
                best_iter = history_df_temp.loc[history_df_temp['mse'].idxmin()]
                nu_opt, kappa_opt, d_diffusion_opt = best_iter['nu'], best_iter['kappa'], best_iter['d_diffusion']
                final_cost = best_iter['mse']
                print(f"  Using best parameters from iteration {int(best_iter['iteration'])} for {os.path.basename(km_file_path)}.")
        else:
            print(f"  No optimization history for {os.path.basename(km_file_path)}. Using initial parameters.")

    # Ensure final parameters are physically plausible and within bounds
    if np.isnan(d_diffusion_opt) or d_diffusion_opt < 0: d_diffusion_opt = 1e-9
    d_diffusion_opt = min(d_diffusion_opt, D_DIFFUSION_UPPER_BOUND)
    if np.isnan(kappa_opt) or kappa_opt < 0: kappa_opt = 1e-9
    if np.isnan(nu_opt): nu_opt = 0.0

    # Calculate equivalent gamma
    gamma_equiv = np.nan
    try:
        gamma_equiv = d_diffusion_opt * 4 * OMEGA_0**2
    except Exception:
        pass # Keep as NaN if calculation fails

    # Calculate Relative Errors
    nu_rel_err, kappa_rel_err, D_rel_err = np.nan, np.nan, np.nan

    if nu_standard is not None:
        if nu_standard != 0:
            nu_rel_err = abs((nu_opt - nu_standard) / nu_standard)
        elif nu_opt != 0: nu_rel_err = np.inf
        else: nu_rel_err = 0.0

    if kappa_standard is not None:
        if kappa_standard != 0:
            kappa_rel_err = abs((kappa_opt - kappa_standard) / kappa_standard)
        elif kappa_opt != 0: kappa_rel_err = np.inf
        else: kappa_rel_err = 0.0

    if D_standard is not None:
        if D_standard != 0:
            D_rel_err = abs((d_diffusion_opt - D_standard) / D_standard)
        elif d_diffusion_opt != 0: D_rel_err = np.inf
        else: D_rel_err = 0.0

    print(f"  Optimized: ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_diffusion_opt:.4f}")
    if nu_standard is not None:
        print(f"  Standard:  ν={nu_standard:.4f}, κ={kappa_standard:.4f}, D={D_standard:.4f}")
        print(f"  Rel. Errors: ν={nu_rel_err:.2%}, κ={kappa_rel_err:.2%}, D={D_rel_err:.2%}")

    # 5. Recalculate FP & Theoretical KM with Optimal Params and Plot
    fp_km_opt = None
    try:
        fp_km_opt_results = compute_fp_km_coefficients(
            [nu_opt, kappa_opt, d_diffusion_opt],
            A_selected, tau_indices, dt, A_max_data, N_JOBS, optimization_history_local # Pass local history
        )
        if fp_km_opt_results is not None:
            fp_km_opt = {k: v for k, v in fp_km_opt_results.items() if k != 'fp_duration'}
    except Exception as e:
        print(f"  Error recalculating final FP KM coeffs for {os.path.basename(km_file_path)}: {e}")

    A_plot = np.linspace(max(1e-6, A_selected[0] * 0.8), A_selected[-1] * 1.2, 200)
    D1_theory, D2_theory = None, None
    try:
        D1_theory = theoretical_D1_d(A_plot, nu_opt, kappa_opt, d_diffusion_opt)
        D2_theory = theoretical_D2_d(A_plot, nu_opt, kappa_opt, d_diffusion_opt)
        D1_theory = np.nan_to_num(D1_theory, nan=np.nan, posinf=np.nan, neginf=np.nan)
    except Exception as e:
        print(f"  Error calculating theoretical KM coeffs for {os.path.basename(km_file_path)}: {e}")

    # Plotting Final Comparison
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(14, 10))
        colors_plot = plt.cm.viridis(np.linspace(0, 1, max(1, N_tau)))

        ax1 = plt.subplot(2, 1, 1)
        plotted_data_legend = False
        plotted_fp_legend = False
        for i, tau_idx in enumerate(tau_indices):
            if tau_idx not in finite_time_km: continue
            tau_sec = tau_idx * dt
            color = colors_plot[i]
            label_data = f'Data D1 (All τ)' if not plotted_data_legend else "_nolegend_"
            valid_data_d1 = ~np.isnan(finite_time_km[tau_idx]['D1'])
            ax1.scatter(finite_time_km[tau_idx]['A'][valid_data_d1], finite_time_km[tau_idx]['D1'][valid_data_d1],
                        alpha=0.6, color=color, marker='o', s=30, label=label_data)
            plotted_data_legend = True

            if fp_km_opt and tau_idx in fp_km_opt:
                valid_fp_d1 = ~np.isnan(fp_km_opt[tau_idx]['D1'])
                if np.any(valid_fp_d1):
                     label_fp = f'FP D1 (Optimized, All τ)' if not plotted_fp_legend else "_nolegend_"
                     ax1.plot(fp_km_opt[tau_idx]['A'][valid_fp_d1], fp_km_opt[tau_idx]['D1'][valid_fp_d1],
                              '--', marker='x', markersize=5, color=color, label=label_fp)
                     plotted_fp_legend = True

        if D1_theory is not None:
            valid_theory_d1 = ~np.isnan(D1_theory)
            ax1.plot(A_plot[valid_theory_d1], D1_theory[valid_theory_d1], 'k-', linewidth=2.5, label='Theoretical D1 (τ→0)', zorder=N_tau+1)

        ax1.set_title(f'Drift Coefficient (D1) Comparison (N_τ={N_tau}, N_A={N_A}) - Forward FP')
        ax1.set_xlabel('Amplitude (A)')
        ax1.set_ylabel('D1')
        ax1.legend(fontsize='small', ncol=2)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2 = plt.subplot(2, 1, 2)
        plotted_data_legend = False
        plotted_fp_legend = False
        for i, tau_idx in enumerate(tau_indices):
            if tau_idx not in finite_time_km: continue
            tau_sec = tau_idx * dt
            color = colors_plot[i]
            label_data = f'Data D2 (All τ)' if not plotted_data_legend else "_nolegend_"
            valid_data_d2 = ~np.isnan(finite_time_km[tau_idx]['D2'])
            ax2.scatter(finite_time_km[tau_idx]['A'][valid_data_d2], finite_time_km[tau_idx]['D2'][valid_data_d2],
                        alpha=0.6, color=color, marker='o', s=30, label=label_data)
            plotted_data_legend = True

            if fp_km_opt and tau_idx in fp_km_opt:
                valid_fp_d2 = ~np.isnan(fp_km_opt[tau_idx]['D2'])
                if np.any(valid_fp_d2):
                    label_fp = f'FP D2 (Optimized, All τ)' if not plotted_fp_legend else "_nolegend_"
                    ax2.plot(fp_km_opt[tau_idx]['A'][valid_fp_d2], fp_km_opt[tau_idx]['D2'][valid_fp_d2],
                             '--', marker='x', markersize=5, color=color, label=label_fp)
                    plotted_fp_legend = True

        if D2_theory is not None:
            valid_theory_d2 = ~np.isnan(D2_theory)
            ax2.plot(A_plot[valid_theory_d2], D2_theory[valid_theory_d2], 'k-', linewidth=2.5, label='Theoretical D2 (τ→0)', zorder=N_tau+1)

        ax2.set_title(f'Diffusion Coefficient (D2) Comparison')
        ax2.set_xlabel('Amplitude (A)')
        ax2.set_ylabel('D2')
        ax2.legend(fontsize='small', ncol=2)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.axhline(0, color='gray', linewidth=0.5)
        if 'finite_time_km_df' in locals() and not finite_time_km_df.empty:
             y_max_d2 = np.nanpercentile(finite_time_km_df['D2_data'], 98) * 1.2 if not finite_time_km_df['D2_data'].isnull().all() else 1.0
             ax2.set_ylim([max(-0.05 * y_max_d2, -0.1), max(y_max_d2, 0.1)])

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        title_str = f'Forward FP Parameter Estimation Results for {os.path.basename(km_file_path)}\n' \
                    f'Optimized: d_diff={d_diffusion_opt:.4f}, ν={nu_opt:.3f}, κ={kappa_opt:.3f} (MSE: {final_cost:.3e})\n'
        if nu_standard is not None and kappa_standard is not None and D_standard is not None:
            title_str += f'Standard: d_diff={D_standard:.4f}, ν={nu_standard:.3f}, κ={kappa_standard:.3f}\n' \
                         f'Relative Errors: d_diff={D_rel_err:.2%}, ν={nu_rel_err:.2%}, κ={kappa_rel_err:.2%}'
        else:
            title_str += 'Standard parameters not available for relative error calculation.'

        plt.suptitle(title_str, fontsize=14, y=0.99)

        output_plot_path = os.path.join(output_dir, f"fp_param_est_comparison.png")
        try:
            plt.savefig(output_plot_path, dpi=300)
            print(f"  Comparison plot saved to {output_plot_path}")
        except Exception as e:
            print(f"  Error saving comparison plot for {os.path.basename(km_file_path)}: {e}")
        plt.close()

    except Exception as e:
        print(f"  An error occurred during final plotting for {os.path.basename(km_file_path)}: {e}")
        traceback.print_exc()

    # Save Optimization Iteration History
    if optimization_history_local:
        history_df = pd.DataFrame(optimization_history_local)
        output_history_path = os.path.join(output_dir, f"fp_optimization_history.csv")
        try:
            history_df.to_csv(output_history_path, index=False)
            print(f"  Optimization history saved to {output_history_path}")
        except Exception as e:
            print(f"  Error saving optimization history for {os.path.basename(km_file_path)}: {e}")

    # Save Detailed Comparison Data
    try:
        comparison_data = {'A_plot': A_plot}
        if D1_theory is not None: comparison_data['D1_theoretical_opt'] = D1_theory
        if D2_theory is not None: comparison_data['D2_theoretical_opt'] = D2_theory

        for tau_idx in tau_indices:
             tau_sec = tau_idx * dt
             if tau_idx in finite_time_km:
                 D1_data_interp = interp1d(finite_time_km[tau_idx]['A'], finite_time_km[tau_idx]['D1'], kind='linear', bounds_error=False, fill_value=np.nan)
                 D2_data_interp = interp1d(finite_time_km[tau_idx]['A'], finite_time_km[tau_idx]['D2'], kind='linear', bounds_error=False, fill_value=np.nan)
                 comparison_data[f'D1_data_tau_{tau_sec:.4f}s'] = D1_data_interp(A_plot)
                 comparison_data[f'D2_data_tau_{tau_sec:.4f}s'] = D2_data_interp(A_plot)
             if fp_km_opt and tau_idx in fp_km_opt:
                 D1_fp_interp = interp1d(fp_km_opt[tau_idx]['A'], fp_km_opt[tau_idx]['D1'], kind='linear', bounds_error=False, fill_value=np.nan)
                 D2_fp_interp = interp1d(fp_km_opt[tau_idx]['A'], fp_km_opt[tau_idx]['D2'], kind='linear', bounds_error=False, fill_value=np.nan)
                 comparison_data[f'D1_fp_opt_tau_{tau_sec:.4f}s'] = D1_fp_interp(A_plot)
                 comparison_data[f'D2_fp_opt_tau_{tau_sec:.4f}s'] = D2_fp_interp(A_plot)

        comparison_df = pd.DataFrame(comparison_data)
        output_comp_path = os.path.join(output_dir, f"km_coefficients_comparison_detailed.csv")
        comparison_df.to_csv(output_comp_path, index=False)
        print(f"  Detailed comparison data saved to {output_comp_path}")
    except Exception as e:
        print(f"  Could not save detailed comparison data for {os.path.basename(km_file_path)}: {e}")

    # Collect results for overall summary
    return {
        'filename': os.path.basename(km_file_path),
        'nu_standard': nu_standard,
        'kappa_standard': kappa_standard,
        'D_standard': D_standard,
        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_diffusion_opt,
        'nu_relative_error': nu_rel_err,
        'kappa_relative_error': kappa_rel_err,
        'D_relative_error': D_rel_err,
        'final_cost_mse': final_cost,
        'optimization_successful': optimization_successful,
        'gamma_equivalent': gamma_equiv,
        'N_tau': N_tau,
        'N_A': N_A,
        'dt': dt
    }


# --- Main Batch Processing Logic ---
if __name__ == "__main__":
    all_results_summary = []
    km_files = [f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')]

    if not km_files:
        print(f"No .csv files found in {BASE_KM_DATA_DIR}. Exiting.")
    else:
        print(f"\n--- Starting batch processing of {len(km_files)} KM data files using FP Solver ---")
        for i, filename in tqdm(enumerate(km_files), total=len(km_files), desc="Overall Batch Progress"):
            km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
            # Pass the BASE_RAW_DATA_DIR to the processing function
            result = process_km_file_fp(km_file_path, BASE_RAW_DATA_DIR, BASE_OUTPUT_DIR)
            if result:
                all_results_summary.append(result)
            # print(f"--- Finished processing: {filename} ---") # Suppress this line from tqdm description

        # Save overall summary
        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary)
            overall_summary_path = os.path.join(BASE_OUTPUT_DIR, "fp_batch_optimization_summary.csv")
            try:
                summary_df.to_csv(overall_summary_path, index=False)
                print(f"\n--- Overall optimization summary saved to {overall_summary_path} ---")
            except Exception as e:
                print(f"ERROR: Could not save overall summary file: {e}")
        else:
            print("\nNo successful optimization results to summarize.")

    print("\n--- Batch processing finished ---")