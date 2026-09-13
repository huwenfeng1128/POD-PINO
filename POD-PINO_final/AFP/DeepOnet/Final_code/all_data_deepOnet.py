#这个代码没有考虑加权损失函数，因此不太适用
#批量化计算所有的DeepOnet识别参数
# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
from tqdm import tqdm
import math
import traceback

# --- PyTorch Imports for DeepONet ---
import torch
import torch.nn as nn

# --- Configuration ---
# Base directory for all KM data files
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
# Base directory for all output results
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\Comparison_Identification\grid_deeponet_batch_run_without_clipping_1000'  # New base output directory for batch runs
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)  # Create base output dir if needed

# --- DeepONet Configuration ---
DEEPONET_MODEL_PATH = r'D:\PINN\zenodo\POD_deeponet\Comparison_Final_Result\model_standard_deeponet.pth'
DEEPONET_SCALER_PATH = r'D:\PINN\zenodo\AFP\result\DeepOnet_train_result\scalers_earlystop_2500.pth'
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_TRUNK_INPUT_DIM = 2
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_OUTPUT_FEATURES = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Match training precision

# Parameters for tau selection (These will be inferred from the loaded data KM)
# N_tau, N_A, dt will be determined per file

F_FILTER = 100  # Filter width (Hz). Set to None if unknown or not applicable.
OMEGA_0 = 2 * math.pi * 150  # Central frequency (rad/s) - Used only for equivalent gamma estimate

# Optimization settings
OPTIMIZER_METHOD = 'Nelder-Mead' # Keep Nelder-Mead for now
OPTIMIZER_OPTIONS = {'maxiter': 3000, 'disp': True, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}

# --- Parameter Bounds and Penalty Configuration ---
PARAMETER_UPPER_BOUND = 100 # Upper bound for absolute value of parameters
PENALTY_FACTOR = 1e8  # Factor to multiply penalty by
# New: Penalty for D < 0
D_NEGATIVE_PENALTY_FACTOR = 1e3 # Much larger penalty for D < 0

# --- Global Variable for Optimization History ---
# This will now be reset for each file, and a summary collected
# optimization_history = [] # No longer global, passed and collected

# --- Helper Functions ---

def parse_params_from_filename(filepath):
    """
    Parses nu, kappa, D from filenames like 'simulation_results_nu18.18_kappa4.54_D18.18.csv'
    or '(18.18,4.54,18.18).csv' using split.
    Returns (nu, kappa, D) as floats, or (None, None, None) if parsing fails.
    """
    filename = os.path.basename(filepath)
    nu_std, kappa_std, D_std = None, None, None

    try:
        if 'simulation_results_nu' in filename and 'kappa' in filename and 'D' in filename:
            parts = filename.replace('.csv', '').split('_')
            for part in parts:
                if part.startswith('nu'):
                    nu_std = float(part[2:])
                elif part.startswith('kappa'):
                    kappa_std = float(part[5:])
                elif part.startswith('D'):
                    D_std = float(part[1:])
        elif filename.startswith('(') and filename.endswith('.csv'):
            # Example: (18.18,4.54,18.18).csv
            inner_part = filename[1:-5]  # Remove '(' and ').csv'
            values = inner_part.split(',')
            if len(values) == 3:
                nu_std = float(values[0])
                kappa_std = float(values[1])
                D_std = float(values[2])
    except (ValueError, IndexError) as e:
        # print(f"Warning: Could not parse parameters from filename '{filename}'. Error: {e}") # Suppress for batch
        nu_std, kappa_std, D_std = None, None, None
    return nu_std, kappa_std, D_std


# --- DeepONet Model Definition (Copied from Training Script) ---
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron (used for Branch and Trunk)."""

    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units, dtype=DTYPE))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units, dtype=DTYPE))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, output_dim, dtype=DTYPE))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepONet(nn.Module):
    """DeepONet Architecture."""

    def __init__(self, branch_input_dim, trunk_input_dim, hidden_units, num_hidden_layers, output_features):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, output_features)
        self.trunk1 = MLP(trunk_input_dim, hidden_units, num_hidden_layers, output_features)
        self.trunk2 = MLP(trunk_input_dim, hidden_units, num_hidden_layers, output_features)
        self.bias1 = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        self.bias2 = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, branch_x, trunk_x):
        branch_out = self.branch(branch_x)
        trunk1_out = self.trunk1(trunk_x)
        trunk2_out = self.trunk2(trunk_x)
        # Ensure batch dimension exists for broadcasting bias, even if batch size is 1
        if branch_out.dim() == 1: branch_out = branch_out.unsqueeze(0)
        if trunk1_out.dim() == 1: trunk1_out = trunk1_out.unsqueeze(0)
        if trunk2_out.dim() == 1: trunk2_out = trunk2_out.unsqueeze(0)

        output1 = torch.sum(branch_out * trunk1_out, dim=1, keepdim=True) + self.bias1
        output2 = torch.sum(branch_out * trunk2_out, dim=1, keepdim=True) + self.bias2
        return torch.cat((output1, output2), dim=1)


# --- Manual Scaler Functions (Copied from Training Script) ---
def manual_scaler_transform(data, mean, std):
    """Applies pre-computed scaling."""
    mean_t = torch.tensor(mean, dtype=DTYPE, device=data.device)
    std_t = torch.tensor(std, dtype=DTYPE, device=data.device)
    std_t[std_t < 1e-10] = 1.0  # Avoid division by zero
    return (data - mean_t) / std_t


def manual_unscaler_transform(scaled_data, mean, std):
    """Applies pre-computed unscaling."""
    mean_t = torch.tensor(mean, dtype=DTYPE, device=scaled_data.device)
    std_t = torch.tensor(std, dtype=DTYPE, device=scaled_data.device)
    return scaled_data * std_t + mean_t


# --- Theoretical KM Coefficients (for tau->0 plot) ---
def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """Theoretical Drift Coefficient D1(A) using d_diffusion/A form."""
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    term_nu = nu * A
    term_kappa = (kappa / 8) * A ** 3
    return term_nu - term_kappa + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """Theoretical Diffusion Coefficient D2(A) using d_diffusion."""
    A = np.asarray(A)
    return np.full_like(A, d_diffusion)


# --- DeepONet KM Coefficient Computation Function ---
def compute_deeponet_km_coefficients(params, A_selected, tau_indices, dt, model, scalers, device,
                                     optimization_history_local):
    """
    Computes KM coefficients using the pre-trained DeepONet model.
    optimization_history_local is passed to allow printing iteration info.
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1  # Get current iteration number for printing

    # --- Parameter Validation (NaN/Inf check only) ---
    if np.isnan(nu) or np.isinf(nu) or \
            np.isnan(kappa) or np.isinf(kappa) or \
            np.isnan(d_diffusion) or np.isinf(d_diffusion):
        return None  # Invalid parameters

    # --- Prepare Scalers ---
    branch_mean = scalers['branch_mean'].numpy()
    branch_std = scalers['branch_std'].numpy()
    trunk_mean = scalers['trunk_mean'].numpy()
    trunk_std = scalers['trunk_std'].numpy()
    y_mean = scalers['y_mean'].numpy()
    y_std = scalers['y_std'].numpy()

    results = {}
    deeponet_compute_start_time = time.time()

    try:
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            for tau_idx in tau_indices:
                tau_sec = tau_idx * dt
                if tau_sec <= 0: continue

                num_A = len(A_selected)
                branch_input_np = np.tile(np.array([[nu, kappa, d_diffusion]]), (num_A, 1))
                trunk_input_np = np.column_stack((A_selected, np.full(num_A, tau_sec)))

                branch_input_scaled_np = (branch_input_np - branch_mean) / branch_std
                trunk_input_scaled_np = (trunk_input_np - trunk_mean) / trunk_std

                branch_t = torch.tensor(branch_input_scaled_np, dtype=DTYPE).to(device)
                trunk_t = torch.tensor(trunk_input_scaled_np, dtype=DTYPE).to(device)

                y_pred_scaled_t = model(branch_t, trunk_t)
                y_pred_t = manual_unscaler_transform(y_pred_scaled_t, y_mean, y_std)
                y_pred_np = y_pred_t.cpu().numpy()

                D1_pred = y_pred_np[:, 0]
                D2_pred = y_pred_np[:, 1]

                results[tau_idx] = {'A': A_selected, 'D1': D1_pred, 'D2': D2_pred}

    except Exception as e:
        # print(f"  Iter {current_iter} Error: DeepONet prediction failed for params {params}: {e}") # Suppress for batch
        return None

    deeponet_compute_duration = time.time() - deeponet_compute_start_time
    results['deeponet_duration'] = deeponet_compute_duration
    return results


# --- Objective Function (using DeepONet) ---
def objective_function(params, data_km, A_selected, tau_indices, dt, model, scalers, device,
                       optimization_history_local):
    """Objective function comparing data KM and DeepONet KM predictions with parameter penalty."""
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    # --- Parameter Validity Checks (NaN/Inf and D < 0) ---
    if np.isnan(nu) or np.isinf(nu) or \
            np.isnan(kappa) or np.isinf(kappa) or \
            np.isnan(d_diffusion) or np.isinf(d_diffusion):
        # print(f"  Iter {current_iter} Eval: Invalid params (NaN/Inf). Cost: 1e12") # Suppress for batch
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e12, 'penalty': 0.0, 'total_cost': 1e12, 'pred_duration_sec': 0.0, 'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e12

    # New: Penalty for d_diffusion < 0
    d_negative_penalty = 0.0
    if d_diffusion < 0:
        d_negative_penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion) # Linear penalty for now, could be quadratic
        # print(f"  Iter {current_iter} Eval: d_diffusion < 0 ({d_diffusion:.5f}). Adding penalty {d_negative_penalty:.2e}") # Suppress for batch
        # If d_diffusion is very negative, we might want to return a very high cost immediately
        # to guide the optimizer away from this region quickly.
        # For Nelder-Mead, a very high value is usually enough.
        if d_diffusion < -100: # Arbitrary large negative threshold
            total_cost = 1e15 + d_negative_penalty # Extremely high cost
            iteration_data = {
                'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
                'mse': 0.0, 'penalty': d_negative_penalty, 'total_cost': total_cost,
                'pred_duration_sec': 0.0, 'num_compared_points': 0
            }
            optimization_history_local.append(iteration_data)
            return total_cost


    start_pred_time = time.time()
    deeponet_km_results = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices, dt, model, scalers, device, optimization_history_local  # Pass local history
    )
    end_pred_time = time.time()
    pred_duration = end_pred_time - start_pred_time

    if deeponet_km_results is None:
        # print(f"  Iter {current_iter} Eval: DeepONet prediction failed for params {params}. Cost: 1e11") # Suppress for batch
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e11, 'penalty': 0.0, 'total_cost': 1e11, 'pred_duration_sec': pred_duration,
            'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e11 + d_negative_penalty # Add penalty even if DeepONet fails

    pred_duration = deeponet_km_results.pop('deeponet_duration', pred_duration)

    total_sq_error = 0
    num_points = 0
    for tau_idx in tau_indices:
        if tau_idx not in data_km or tau_idx not in deeponet_km_results:
            continue

        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']
        D1_pred = deeponet_km_results[tau_idx]['D1']
        D2_pred = deeponet_km_results[tau_idx]['D2']

        if D1_data.shape != D1_pred.shape or D2_data.shape != D2_pred.shape:
            continue

        valid_mask_d1 = ~np.isnan(D1_data) & ~np.isnan(D1_pred)
        valid_mask_d2 = ~np.isnan(D2_data) & ~np.isnan(D2_pred)

        error_D1 = np.sum(((D1_data - D1_pred) ** 2)[valid_mask_d1])
        error_D2 = np.sum(((D2_data - D2_pred) ** 2)[valid_mask_d2])

        total_sq_error += error_D1 + error_D2
        num_points += np.sum(valid_mask_d1) + np.sum(valid_mask_d2)

    if num_points == 0:
        mean_sq_error = 1e10
    else:
        mean_sq_error = total_sq_error / num_points

    penalty = 0.0
    for param in params:
        if abs(param) > PARAMETER_UPPER_BOUND:
            penalty += PENALTY_FACTOR * (abs(param) - PARAMETER_UPPER_BOUND) ** 2

    total_cost = mean_sq_error + penalty + d_negative_penalty # Add the new penalty here

    iteration_data = {
        'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
        'mse': mean_sq_error, 'penalty': penalty + d_negative_penalty, 'total_cost': total_cost, # Report combined penalty
        'pred_duration_sec': pred_duration, 'num_compared_points': num_points
    }
    optimization_history_local.append(iteration_data)

    # print(f"  Iter {current_iter} Eval: nu={nu:.5f}, k={kappa:.5f}, d={d_diffusion:.5f} -> MSE={mean_sq_error:.6e}, Penalty={penalty:.6e}, D_neg_Penalty={d_negative_penalty:.6e}, Total Cost={total_cost:.6e} (Pred time: {pred_duration:.4f}s)") # Suppress for batch

    if np.isnan(total_cost) or np.isinf(total_cost):
        return 1e12

    return total_cost


# --- Main Processing Function for Each KM File ---
def process_km_file(km_file_path, deeponet_model, scalers, device, base_output_dir):
    print(f"\n--- Processing file: {os.path.basename(km_file_path)} ---")

    # 1. Get standard parameters from filename
    nu_standard, kappa_standard, D_standard = parse_params_from_filename(km_file_path)
    if nu_standard is None or kappa_standard is None or D_standard is None:
        print(
            f"WARNING: Could not extract standard parameters from '{os.path.basename(km_file_path)}'. Skipping this file.")
        return None  # Skip this file if standard params can't be found

    # Create a specific output directory for this file
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0].replace('.',
                                                                                       '_')  # Replace . with _ for valid folder name
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this file will be saved to: {output_dir}")

    # 2. Load Data KM Coefficients
    finite_time_km = {}
    tau_indices = []
    A_selected = None
    dt = None

    try:
        data_km_df = pd.read_csv(km_file_path)

        tau_indices_loaded = sorted(data_km_df['tau_index'].unique())
        tau_sec_loaded = sorted(data_km_df['tau_sec'].unique())
        A_selected = sorted(data_km_df['A'].unique())

        if len(tau_indices_loaded) > 1:
            dt = (tau_sec_loaded[1] - tau_sec_loaded[0]) / (tau_indices_loaded[1] - tau_indices_loaded[0])
        elif len(tau_indices_loaded) == 1 and tau_indices_loaded[0] > 0:
            dt = tau_sec_loaded[0] / tau_indices_loaded[0]
        else:
            dt = 0.0001  # Fallback

        N_tau = len(tau_indices_loaded)
        N_A = len(A_selected)
        tau_indices = tau_indices_loaded

        for tau_idx in tau_indices:
            tau_data = data_km_df[data_km_df['tau_index'] == tau_idx]
            if not tau_data.empty:
                finite_time_km[tau_idx] = {
                    'A': tau_data['A'].values,
                    'D1': tau_data['D1_data'].values,
                    'D2': tau_data['D2_data'].values
                }
        if not finite_time_km:
            print(f"Error: No valid data KM coefficients loaded from {os.path.basename(km_file_path)}. Skipping.")
            return None

        print(f"  Loaded {N_tau} tau values and {N_A} amplitude points from data.")

    except Exception as e:
        print(f"Error loading data KM from {os.path.basename(km_file_path)}: {e}. Skipping.")
        traceback.print_exc()
        return None

    # 3. Parameter Optimization
    # Initial Guess Estimation (Using loaded data's D2 mean)
    try:
        valid_d2_means = []
        for tau in tau_indices:
            if tau in finite_time_km:
                d2_values = finite_time_km[tau]['D2']
                valid_d2 = d2_values[~np.isnan(d2_values)]
                if len(valid_d2) > 0:
                    valid_d2_means.append(np.mean(valid_d2))
        if valid_d2_means:
            avg_D2 = np.mean(valid_d2_means)
            d_diffusion_0 = max(0.001, avg_D2) # Ensure initial guess for D is not negative
        else:
            d_diffusion_0 = 0.01
    except Exception as e:
        print(f"  Error estimating initial d_diffusion for {os.path.basename(km_file_path)}: {e}. Using default.")
        d_diffusion_0 = 0.01

    nu_0 = 0.1
    kappa_0 = 0.1
    params_0 = [nu_0, kappa_0, d_diffusion_0]

    optimization_history_local = []  # Local history for this file's optimization

    print(f"  Starting optimization for {os.path.basename(km_file_path)}...")
    start_time_opt = time.time()
    result = None
    try:
        result = minimize(
            objective_function,
            params_0,
            args=(
            finite_time_km, A_selected, tau_indices, dt, deeponet_model, scalers, device, optimization_history_local),
            method=OPTIMIZER_METHOD,
            options=OPTIMIZER_OPTIONS,
        )
    except KeyboardInterrupt:
        print(f"  Optimization for {os.path.basename(km_file_path)} interrupted by user.")
    except Exception as e:
        print(f"  Error during optimization for {os.path.basename(km_file_path)}: {e}")
        traceback.print_exc()

    end_time_opt = time.time()
    print(
        f"  Optimization for {os.path.basename(km_file_path)} finished in {end_time_opt - start_time_opt:.2f} seconds.")

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
            if not history_df_temp.empty and 'total_cost' in history_df_temp.columns:
                best_iter = history_df_temp.loc[history_df_temp['total_cost'].idxmin()]
                nu_opt, kappa_opt, d_diffusion_opt = best_iter['nu'], best_iter['kappa'], best_iter['d_diffusion']
                final_cost = best_iter['total_cost']
                print(
                    f"  Using best parameters from iteration {int(best_iter['iteration'])} for {os.path.basename(km_file_path)}.")
        else:
            print(f"  No optimization history for {os.path.basename(km_file_path)}. Using initial parameters.")

    # Handle potential NaNs from optimization failure, set to 0 as a basic fallback
    if np.isnan(nu_opt): nu_opt = 0.0
    if np.isnan(kappa_opt): kappa_opt = 0.0
    if np.isnan(d_diffusion_opt): d_diffusion_opt = 0.0

    # Ensure d_diffusion_opt is not negative after optimization, even if optimization
    # didn't perfectly converge to a non-negative value due to Nelder-Mead's nature.
    d_diffusion_opt = max(0.0, d_diffusion_opt)


    # Calculate final MSE part for reporting
    final_mse_part = np.nan
    if optimization_history_local:
        history_df_temp = pd.DataFrame(optimization_history_local)
        # Find the row that corresponds to the chosen optimal parameters (nu_opt, kappa_opt, d_diffusion_opt)
        # Due to max(0.0, d_diffusion_opt) adjustment, it might not perfectly match a history entry.
        # So, we re-evaluate the objective function with the final adjusted parameters to get the exact MSE.
        # This is important for accurate reporting of the final MSE component.
        final_params_for_mse_calc = [nu_opt, kappa_opt, d_diffusion_opt]
        # Create a temporary history list for this single evaluation to avoid polluting the main one
        temp_history_for_final_mse = []
        _ = objective_function(final_params_for_mse_calc, finite_time_km, A_selected, tau_indices, dt,
                               deeponet_model, scalers, device, temp_history_for_final_mse)
        if temp_history_for_final_mse:
            final_mse_part = temp_history_for_final_mse[0]['mse']


    # Calculate equivalent gamma
    gamma_equiv = np.nan
    try:
        gamma_equiv = d_diffusion_opt * 4 * OMEGA_0 ** 2
    except Exception:
        pass  # Keep as NaN if calculation fails

    # Calculate Relative Errors
    nu_rel_err, kappa_rel_err, D_rel_err = np.nan, np.nan, np.nan

    if nu_standard is not None:
        if nu_standard != 0:
            nu_rel_err = abs((nu_opt - nu_standard) / nu_standard)
        elif nu_opt != 0:
            nu_rel_err = np.inf
        else:
            nu_rel_err = 0.0

    if kappa_standard is not None:
        if kappa_standard != 0:
            kappa_rel_err = abs((kappa_opt - kappa_standard) / kappa_standard)
        elif kappa_opt != 0:
            kappa_rel_err = np.inf
        else:
            kappa_rel_err = 0.0

    if D_standard is not None:
        if D_standard != 0:
            D_rel_err = abs((d_diffusion_opt - D_standard) / D_standard)
        elif d_diffusion_opt != 0:
            D_rel_err = np.inf
        else:
            D_rel_err = 0.0

    print(f"  Optimized: ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_diffusion_opt:.4f}")
    if nu_standard is not None:
        print(f"  Standard:  ν={nu_standard:.4f}, κ={kappa_standard:.4f}, D={D_standard:.4f}")
        print(f"  Rel. Errors: ν={nu_rel_err:.2%}, κ={kappa_rel_err:.2%}, D={D_rel_err:.2%}")

    # 4. Recalculate DeepONet & Theoretical KM with Optimal Params and Plot
    deeponet_km_opt = None
    try:
        deeponet_km_opt_results = compute_deeponet_km_coefficients(
            [nu_opt, kappa_opt, d_diffusion_opt],
            A_selected, tau_indices, dt, deeponet_model, scalers, device, optimization_history_local
            # Pass local history
        )
        if deeponet_km_opt_results is not None:
            deeponet_km_opt = {k: v for k, v in deeponet_km_opt_results.items() if k != 'deeponet_duration'}
    except Exception as e:
        print(f"  Error recalculating final DeepONet KM coeffs for {os.path.basename(km_file_path)}: {e}")

    A_plot = np.linspace(min(A_selected) * 0.9, max(A_selected) * 1.1, 200)
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
        plotted_deeponet_legend = False
        for i, tau_idx in enumerate(tau_indices):
            if tau_idx not in finite_time_km: continue
            tau_sec = tau_idx * dt
            color = colors_plot[i]
            label_data = f'Data D1 (All τ)' if not plotted_data_legend else "_nolegend_"
            valid_data_d1 = ~np.isnan(finite_time_km[tau_idx]['D1'])
            A_data_filtered_d1 = np.compress(valid_data_d1, finite_time_km[tau_idx]['A'])
            D1_data_filtered = np.compress(valid_data_d1, finite_time_km[tau_idx]['D1'])
            ax1.scatter(A_data_filtered_d1, D1_data_filtered,
                        alpha=0.6, color=color, marker='o', s=30, label=label_data)
            plotted_data_legend = True

            if deeponet_km_opt and tau_idx in deeponet_km_opt:
                valid_pred_d1 = ~np.isnan(deeponet_km_opt[tau_idx]['D1'])
                if np.any(valid_pred_d1):
                    label_pred = f'DeepONet D1 (Optimized, All τ)' if not plotted_deeponet_legend else "_nolegend_"
                    A_pred_filtered_d1 = np.compress(valid_pred_d1, deeponet_km_opt[tau_idx]['A'])
                    D1_pred_filtered = np.compress(valid_pred_d1, deeponet_km_opt[tau_idx]['D1'])
                    ax1.plot(A_pred_filtered_d1, D1_pred_filtered,
                             '--', marker='x', markersize=5, color=color, label=label_pred)
                    plotted_deeponet_legend = True

        if D1_theory is not None:
            valid_theory_d1 = ~np.isnan(D1_theory)
            ax1.plot(A_plot[valid_theory_d1], D1_theory[valid_theory_d1], 'k-', linewidth=2.5,
                     label='Theoretical D1 (τ→0)', zorder=N_tau + 1)

        ax1.set_title(f'Drift Coefficient (D1) Comparison (N_τ={N_tau}, N_A={N_A}) - DeepONet (With Penalty)')
        ax1.set_xlabel('Amplitude (A)')
        ax1.set_ylabel('D1')
        ax1.legend(fontsize='small', ncol=2)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.axhline(0, color='gray', linewidth=0.5)

        ax2 = plt.subplot(2, 1, 2)
        plotted_data_legend = False
        plotted_deeponet_legend = False
        for i, tau_idx in enumerate(tau_indices):
            if tau_idx not in finite_time_km: continue
            tau_sec = tau_idx * dt
            color = colors_plot[i]
            label_data = f'Data D2 (All τ)' if not plotted_data_legend else "_nolegend_"
            valid_data_d2 = ~np.isnan(finite_time_km[tau_idx]['D2'])
            A_data_filtered_d2 = np.compress(valid_data_d2, finite_time_km[tau_idx]['A'])
            D2_data_filtered = np.compress(valid_data_d2, finite_time_km[tau_idx]['D2'])
            ax2.scatter(A_data_filtered_d2, D2_data_filtered,
                        alpha=0.6, color=color, marker='o', s=30, label=label_data)
            plotted_data_legend = True

            if deeponet_km_opt and tau_idx in deeponet_km_opt:
                valid_pred_d2 = ~np.isnan(deeponet_km_opt[tau_idx]['D2'])
                if np.any(valid_pred_d2):
                    label_pred = f'DeepONet D2 (Optimized, All τ)' if not plotted_deeponet_legend else "_nolegend_"
                    A_pred_filtered_d2 = np.compress(valid_pred_d2, deeponet_km_opt[tau_idx]['A'])
                    D2_pred_filtered = np.compress(valid_pred_d2, deeponet_km_opt[tau_idx]['D2'])
                    ax2.plot(A_pred_filtered_d2, D2_pred_filtered,
                             '--', marker='x', markersize=5, color=color, label=label_pred)
                    plotted_deeponet_legend = True

        if D2_theory is not None:
            valid_theory_d2 = ~np.isnan(D2_theory)
            ax2.plot(A_plot[valid_theory_d2], D2_theory[valid_theory_d2], 'k-', linewidth=2.5,
                     label='Theoretical D2 (τ→0)', zorder=N_tau + 1)

        ax2.set_title(f'Diffusion Coefficient (D2) Comparison')
        ax2.set_xlabel('Amplitude (A)')
        ax2.set_ylabel('D2')
        ax2.legend(fontsize='small', ncol=2)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.axhline(0, color='gray', linewidth=0.5)
        if 'data_km_df' in locals() and not data_km_df.empty:
            y_min_d2_data = np.nanpercentile(data_km_df['D2_data'], 2) if not data_km_df[
                'D2_data'].isnull().all() else -0.1
            y_max_d2_data = np.nanpercentile(data_km_df['D2_data'], 98) if not data_km_df[
                'D2_data'].isnull().all() else 1.0
            y_range = y_max_d2_data - y_min_d2_data
            ax2.set_ylim([y_min_d2_data - 0.1 * y_range, y_max_d2_data + 0.1 * y_range])

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        title_str = f'DeepONet Parameter Estimation Results for {os.path.basename(km_file_path)}\n' \
                    f'Optimized: d_diff={d_diffusion_opt:.4f}, ν={nu_opt:.3f}, κ={kappa_opt:.3f} (Total Cost: {final_cost:.3e})\n'
        if nu_standard is not None and kappa_standard is not None and D_standard is not None:
            title_str += f'Standard: d_diff={D_standard:.4f}, ν={nu_standard:.3f}, κ={kappa_standard:.3f}\n' \
                         f'Relative Errors: d_diff={D_rel_err:.2%}, ν={nu_rel_err:.2%}, κ={kappa_rel_err:.2%}'
        else:
            title_str += 'Standard parameters not available for relative error calculation.'

        plt.suptitle(title_str, fontsize=14, y=0.99)

        output_plot_path = os.path.join(output_dir, f"deeponet_param_est_comparison_with_penalty.png")
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
        output_history_path = os.path.join(output_dir, f"deeponet_optimization_history_with_penalty.csv")
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
                D1_data_interp = interp1d(finite_time_km[tau_idx]['A'], finite_time_km[tau_idx]['D1'], kind='linear',
                                          bounds_error=False, fill_value=np.nan)
                D2_data_interp = interp1d(finite_time_km[tau_idx]['A'], finite_time_km[tau_idx]['D2'], kind='linear',
                                          bounds_error=False, fill_value=np.nan)
                comparison_data[f'D1_data_tau_{tau_sec:.4f}s'] = D1_data_interp(A_plot)
                comparison_data[f'D2_data_tau_{tau_sec:.4f}s'] = D2_data_interp(A_plot)
            if deeponet_km_opt and tau_idx in deeponet_km_opt:
                D1_pred_interp = interp1d(deeponet_km_opt[tau_idx]['A'], deeponet_km_opt[tau_idx]['D1'], kind='linear',
                                          bounds_error=False, fill_value=np.nan)
                D2_pred_interp = interp1d(deeponet_km_opt[tau_idx]['A'], deeponet_km_opt[tau_idx]['D2'], kind='linear',
                                          bounds_error=False, fill_value=np.nan)
                comparison_data[f'D1_deeponet_opt_tau_{tau_sec:.4f}s'] = D1_pred_interp(A_plot)
                comparison_data[f'D2_deeponet_opt_tau_{tau_sec:.4f}s'] = D2_pred_interp(A_plot)

        comparison_df = pd.DataFrame(comparison_data)
        output_comp_path = os.path.join(output_dir, f"km_coefficients_comparison_detailed_with_penalty.csv")
        comparison_df.to_csv(output_comp_path, index=False)
        print(f"  Detailed comparison data saved to {output_comp_path}")
    except Exception as e:
        print(f"  Could not save detailed comparison data for {os.path.basename(km_file_path)}: {e}")

    # Plot Optimization Evolution
    if optimization_history_local:
        history_df = pd.DataFrame(optimization_history_local)
        if not history_df.empty:
            plt.style.use('seaborn-v0_8-whitegrid')

            # Parameter Evolution
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(history_df['iteration'], history_df['nu'], 'o-', label='ν (nu)', markersize=4)
                plt.plot(history_df['iteration'], history_df['kappa'], 's-', label='κ (kappa)', markersize=4)

                use_secondary_axis = False
                if 'd_diffusion' in history_df.columns and 'nu' in history_df.columns and 'kappa' in history_df.columns:
                    param_ranges = history_df[['nu', 'kappa', 'd_diffusion']].max() - history_df[
                        ['nu', 'kappa', 'd_diffusion']].min()
                    all_ranges = param_ranges.dropna()
                    if len(all_ranges) > 1:
                        max_range = all_ranges.max()
                        min_range = all_ranges.min()
                        if max_range > 0 and min_range > 0 and (
                                max_range / min_range > 10 or min_range / max_range < 0.1):
                            if abs(param_ranges['d_diffusion']) > 5 * abs(param_ranges['nu']) or \
                                    abs(param_ranges['d_diffusion']) > 5 * abs(param_ranges['kappa']) or \
                                    abs(param_ranges['nu']) > 5 * abs(param_ranges['d_diffusion']) or \
                                    abs(param_ranges['kappa']) > 5 * abs(param_ranges['d_diffusion']):
                                use_secondary_axis = True

                if use_secondary_axis:
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    line3 = ax2.plot(history_df['iteration'], history_df['d_diffusion'], '^-',
                                     label='d_diffusion (right axis)', markersize=4, color='green')
                    ax2.set_ylabel('d_diffusion Value', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                    ax1.set_ylabel('Parameter Value (ν, κ)')
                else:
                    plt.plot(history_df['iteration'], history_df['d_diffusion'], '^-', label='d_diffusion',
                             markersize=4)
                    plt.legend(loc='best')
                    plt.ylabel('Parameter Value (ν, κ, d_diffusion)')

                plt.title(f'Parameter Evolution for {os.path.basename(km_file_path)}')
                plt.xlabel('Iteration Number')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                output_param_evo_path = os.path.join(output_dir,
                                                     f"deeponet_optimization_parameter_evolution_with_penalty.png")
                plt.savefig(output_param_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting parameter evolution for {os.path.basename(km_file_path)}: {e}")

            # Total Cost Evolution
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(history_df['iteration'], history_df['total_cost'], 'o-', label='Total Cost (MSE + Penalty)',
                         markersize=4, color='red')
                plt.yscale('log')
                plt.title(f'Objective Function (Total Cost) Evolution for {os.path.basename(km_file_path)}')
                plt.xlabel('Iteration Number')
                plt.ylabel('Total Cost (log scale)')
                plt.legend(loc='best')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                output_total_cost_evo_path = os.path.join(output_dir,
                                                          f"deeponet_optimization_total_cost_evolution_with_penalty.png")
                plt.savefig(output_total_cost_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting Total Cost evolution for {os.path.basename(km_file_path)}: {e}")

            # MSE vs Penalty Evolution
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(history_df['iteration'], history_df['mse'], 'o-', label='MSE', markersize=4, color='blue')
                plt.plot(history_df['iteration'], history_df['penalty'], 's-', label='Penalty', markersize=4,
                         color='orange')
                plt.yscale('log')
                plt.title(f'MSE and Penalty Evolution for {os.path.basename(km_file_path)}')
                plt.xlabel('Iteration Number')
                plt.ylabel('Value (log scale)')
                plt.legend(loc='best')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                output_mse_penalty_evo_path = os.path.join(output_dir,
                                                           f"deeponet_optimization_mse_penalty_evolution_with_penalty.png")
                plt.savefig(output_mse_penalty_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting MSE and Penalty evolution for {os.path.basename(km_file_path)}: {e}")

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
        'final_total_cost': final_cost,
        'final_mse_part': final_mse_part,
        'optimization_successful': optimization_successful,
        'gamma_equivalent': gamma_equiv,
        'N_tau': N_tau,
        'N_A': N_A,
        'dt': dt
    }


# --- Main Batch Processing Logic ---
if __name__ == "__main__":
    # --- Force Joblib Temp Folder to Pure ASCII Path (if needed, though not used in DeepONet part) ---
    joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp'
    print(f"Setting joblib temporary folder to: {joblib_temp_folder}")
    try:
        os.makedirs(joblib_temp_folder, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
        print(f"Successfully set JOBLIB_TEMP_FOLDER environment variable.")
    except Exception as e:
        print(f"ERROR: Could not create or set joblib temp folder '{joblib_temp_folder}'.")
        print(f"Error details: {e}")
        exit()
    # --- END: Force Joblib Temp Folder ---

    print("\n--- Loading Pre-trained DeepONet Model and Scalers (Global Load) ---")
    deeponet_model = None
    scalers = None
    try:
        scalers = torch.load(DEEPONET_SCALER_PATH)
        deeponet_model = DeepONet(
            DEEPONET_BRANCH_INPUT_DIM, DEEPONET_TRUNK_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS, DEEPONET_NUM_HIDDEN_LAYERS, DEEPONET_OUTPUT_FEATURES
        )
        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE).to(DTYPE)
        deeponet_model.eval()
        print("DeepONet model and scalers loaded successfully for batch processing.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load DeepONet model or scalers. Exiting.")
        traceback.print_exc()
        exit()

    all_results_summary = []
    km_files = [f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')]

    if not km_files:
        print(f"No .csv files found in {BASE_KM_DATA_DIR}. Exiting.")
    else:
        print(f"\n--- Starting batch processing of {len(km_files)} KM data files ---")
        for i, filename in enumerate(km_files):
            km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
            print(f"\n--- ({i + 1}/{len(km_files)}) Processing: {filename} ---")

            # Pass the globally loaded model and scalers
            result = process_km_file(km_file_path, deeponet_model, scalers, DEVICE, BASE_OUTPUT_DIR)
            if result:
                all_results_summary.append(result)
            print(f"--- Finished processing: {filename} ---")

        # Save overall summary
        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary)
            overall_summary_path = os.path.join(BASE_OUTPUT_DIR, "deeponet_batch_optimization_summary.csv")
            try:
                summary_df.to_csv(overall_summary_path, index=False)
                print(f"\n--- Overall optimization summary saved to {overall_summary_path} ---")
            except Exception as e:
                print(f"ERROR: Could not save overall summary file: {e}")
        else:
            print("\nNo successful optimization results to summarize.")

    print("\n--- Batch processing finished ---")