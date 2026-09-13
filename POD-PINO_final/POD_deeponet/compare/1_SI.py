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
from scipy.signal import hilbert

# --- PyTorch Imports for DeepONet ---
import torch
import torch.nn as nn

# --- Configuration ---
# Base directory for all KM data files (where the calculated KM data is stored)
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data_out_KM'
# Base directory for the ORIGINAL simulation data files (needed for P(a_i) calculation)
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_out'
# Base directory for all output results
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\compare\result\Out_DeepONet_result'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- DeepONet Configuration (Aligned with Training Code) ---
DEEPONET_MODEL_PATH = r'D:\PINN\zenodo\POD_deeponet\compare\result\Vanilla_DeepONet_v3\best_model.pth' # 请确保路径指向的是根据新代码训练出的 best_model.pth
DEEPONET_SCALER_PATH = r'D:\PINN\zenodo\POD_deeponet\compare\result\Vanilla_DeepONet_v3\scalers.pth' # 请确保路径指向的是根据新代码训练出的 scalers.pth
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_TRUNK_INPUT_DIM = 2
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4  # Modified to match Training Code (was 5)
DEEPONET_OUTPUT_FEATURES = 128
DEEPONET_DROPOUT_RATE = 0.1     # Added to match Training Code
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Match training precision

# Parameters for tau selection
F_FILTER = 100
OMEGA_0 = 2 * math.pi * 150

# Optimization settings
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {'maxiter': 3000, 'disp': True, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}

# --- Parameter Bounds and Penalty Configuration ---
PARAMETER_UPPER_BOUND = 100
PENALTY_FACTOR = 1e8
D_NEGATIVE_PENALTY_FACTOR = 1e3

# --- Helper Functions ---

def parse_params_from_filename(filepath):
    """
    Parses nu, kappa, D from filenames.
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
            inner_part = filename[1:-5]
            values = inner_part.split(',')
            if len(values) == 3:
                nu_std = float(values[0])
                kappa_std = float(values[1])
                D_std = float(values[2])
    except (ValueError, IndexError) as e:
        nu_std, kappa_std, D_std = None, None, None
    return nu_std, kappa_std, D_std


# --- DeepONet Model Definition (Strictly Aligned with Training Code) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE)
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepONet(nn.Module):
    # Renamed from VanillaDeepONet in training code to DeepONet here for consistency with existing calls,
    # but internal structure matches the training code perfectly.
    def __init__(self, branch_dim, trunk_dim, hidden, layers, p, dropout):
        super().__init__()
        self.branch = MLP(branch_dim, hidden, layers, p, dropout)
        # Training code uses trunk_d1 and trunk_d2. MUST match these names for load_state_dict.
        self.trunk_d1 = MLP(trunk_dim, hidden, layers, p, dropout)
        self.trunk_d2 = MLP(trunk_dim, hidden, layers, p, dropout)
        self.b1 = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        self.b2 = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, x_b, x_t):
        b_out = self.branch(x_b)
        t1_out = self.trunk_d1(x_t)
        t2_out = self.trunk_d2(x_t)
        # Dot product operation
        out1 = torch.sum(b_out * t1_out, dim=1, keepdim=True) + self.b1
        out2 = torch.sum(b_out * t2_out, dim=1, keepdim=True) + self.b2
        return torch.cat([out1, out2], dim=1)


# --- Manual Scaler Functions ---
def manual_scaler_transform(data, mean, std):
    """Applies pre-computed scaling."""
    mean_t = torch.tensor(mean, dtype=DTYPE, device=data.device)
    std_t = torch.tensor(std, dtype=DTYPE, device=data.device)
    std_t[std_t < 1e-10] = 1.0
    return (data - mean_t) / std_t


def manual_unscaler_transform(scaled_data, mean, std):
    """Applies pre-computed unscaling."""
    mean_t = torch.tensor(mean, dtype=DTYPE, device=scaled_data.device)
    std_t = torch.tensor(std, dtype=DTYPE, device=scaled_data.device)
    return scaled_data * std_t + mean_t


# --- Theoretical KM Coefficients ---
def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    term_nu = nu * A
    term_kappa = (kappa / 8) * A ** 3
    return term_nu - term_kappa + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    return np.full_like(A, d_diffusion)


# --- DeepONet KM Coefficient Computation Function ---
def compute_deeponet_km_coefficients(params, A_selected, tau_indices, dt, model, scalers, device,
                                     optimization_history_local):
    """
    Computes KM coefficients using the pre-trained DeepONet model.
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    # --- Parameter Validation ---
    if np.isnan(nu) or np.isinf(nu) or \
            np.isnan(kappa) or np.isinf(kappa) or \
            np.isnan(d_diffusion) or np.isinf(d_diffusion):
        return None

    # --- Prepare Scalers (UPDATED KEYS to match Training Code) ---
    # Training code saves keys: 'b_mean', 'b_std', 't_mean', 't_std', 'y_mean', 'y_std'
    try:
        branch_mean = scalers['b_mean']
        branch_std = scalers['b_std']
        trunk_mean = scalers['t_mean']
        trunk_std = scalers['t_std']
        y_mean = scalers['y_mean']
        y_std = scalers['y_std']
    except KeyError as e:
        # Fallback for old scaler files if necessary, but primarily target the new code
        # print(f"Key error loading scalers: {e}. Checking for alternative keys.")
        branch_mean = scalers['branch_mean']
        branch_std = scalers['branch_std']
        trunk_mean = scalers['trunk_mean']
        trunk_std = scalers['trunk_std']
        # y_mean and y_std are usually consistent

    # Convert to numpy if they are tensors (Training code saves numpy arrays usually, but let's be safe)
    if isinstance(branch_mean, torch.Tensor): branch_mean = branch_mean.cpu().numpy()
    if isinstance(branch_std, torch.Tensor): branch_std = branch_std.cpu().numpy()
    if isinstance(trunk_mean, torch.Tensor): trunk_mean = trunk_mean.cpu().numpy()
    if isinstance(trunk_std, torch.Tensor): trunk_std = trunk_std.cpu().numpy()
    if isinstance(y_mean, torch.Tensor): y_mean = y_mean.cpu().numpy()
    if isinstance(y_std, torch.Tensor): y_std = y_std.cpu().numpy()

    results = {}
    deeponet_compute_start_time = time.time()

    try:
        model.eval()
        with torch.no_grad():
            for tau_idx in tau_indices:
                tau_sec = tau_idx * dt
                if tau_sec <= 0: continue

                num_A = len(A_selected)
                branch_input_np = np.tile(np.array([[nu, kappa, d_diffusion]]), (num_A, 1))
                trunk_input_np = np.column_stack((A_selected, np.full(num_A, tau_sec)))

                # Apply standardization
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
        return None

    deeponet_compute_duration = time.time() - deeponet_compute_start_time
    results['deeponet_duration'] = deeponet_compute_duration
    return results


# --- Objective Function ---
def objective_function(params, data_km, A_selected, tau_indices, dt, model, scalers, device,
                       optimization_history_local, N_A_from_km_data, p_a_weights_map):
    """Objective function comparing data KM and DeepONet KM predictions."""
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    # --- Parameter Validity Checks ---
    if np.isnan(nu) or np.isinf(nu) or \
            np.isnan(kappa) or np.isinf(kappa) or \
            np.isnan(d_diffusion) or np.isinf(d_diffusion):
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e12, 'penalty': 0.0, 'total_cost': 1e12, 'pred_duration_sec': 0.0, 'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e12

    # Penalty for d_diffusion < 0
    d_negative_penalty = 0.0
    if d_diffusion < 0:
        d_negative_penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion)
        if d_diffusion < -100:
            total_cost = 1e15 + d_negative_penalty
            iteration_data = {
                'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
                'mse': 0.0, 'penalty': d_negative_penalty, 'total_cost': total_cost,
                'pred_duration_sec': 0.0, 'num_compared_points': 0
            }
            optimization_history_local.append(iteration_data)
            return total_cost

    start_pred_time = time.time()
    deeponet_km_results = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices, dt, model, scalers, device, optimization_history_local
    )
    end_pred_time = time.time()
    pred_duration = end_pred_time - start_pred_time

    if deeponet_km_results is None:
        iteration_data = {
            'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
            'mse': 1e11, 'penalty': 0.0, 'total_cost': 1e11, 'pred_duration_sec': pred_duration,
            'num_compared_points': 0
        }
        optimization_history_local.append(iteration_data)
        return 1e11 + d_negative_penalty

    pred_duration = deeponet_km_results.pop('deeponet_duration', pred_duration)

    total_weighted_sq_error = 0
    N_tau = len(tau_indices)

    for tau_idx in tau_indices:
        if tau_idx not in data_km or tau_idx not in deeponet_km_results:
            continue

        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']
        D1_pred = deeponet_km_results[tau_idx]['D1']
        D2_pred = deeponet_km_results[tau_idx]['D2']
        A_current_tau = data_km[tau_idx]['A']

        if D1_data.shape != D1_pred.shape or D2_data.shape != D2_pred.shape:
            continue

        current_A_weights = np.array([p_a_weights_map.get(a, 0.0) for a in A_current_tau])

        valid_mask_d1 = ~np.isnan(D1_data) & ~np.isnan(D1_pred)
        valid_mask_d2 = ~np.isnan(D2_data) & ~np.isnan(D2_pred)

        error_D1 = np.sum(((D1_data - D1_pred) ** 2 * current_A_weights)[valid_mask_d1])
        error_D2 = np.sum(((D2_data - D2_pred) ** 2 * current_A_weights)[valid_mask_d2])

        total_weighted_sq_error += error_D1 + error_D2

    N_total_points_in_formula = N_A_from_km_data * N_tau

    if N_total_points_in_formula == 0:
        mean_weighted_sq_error = 1e10
    else:
        mean_weighted_sq_error = total_weighted_sq_error / (2 * N_total_points_in_formula)

    penalty = 0.0
    for param in params:
        if abs(param) > PARAMETER_UPPER_BOUND:
            penalty += PENALTY_FACTOR * (abs(param) - PARAMETER_UPPER_BOUND) ** 2

    total_cost = mean_weighted_sq_error + penalty + d_negative_penalty

    iteration_data = {
        'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
        'mse': mean_weighted_sq_error, 'penalty': penalty + d_negative_penalty, 'total_cost': total_cost,
        'pred_duration_sec': pred_duration, 'num_compared_points': N_total_points_in_formula * 2
    }
    optimization_history_local.append(iteration_data)

    if np.isnan(total_cost) or np.isinf(total_cost):
        return 1e12

    return total_cost


# --- Main Processing Function for Each KM File ---
def process_km_file(km_file_path, deeponet_model, scalers, device, base_output_dir, input_dir_sim_data):
    print(f"\n--- Processing file: {os.path.basename(km_file_path)} ---")

    # 1. Get standard parameters from filename
    nu_standard, kappa_standard, D_standard = parse_params_from_filename(km_file_path)
    if nu_standard is None or kappa_standard is None or D_standard is None:
        print(
            f"WARNING: Could not extract standard parameters from '{os.path.basename(km_file_path)}'. Skipping this file.")
        return None

    # Create a specific output directory for this file
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0].replace('.', '_')
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results for this file will be saved to: {output_dir}")

    # 2. Load Data KM Coefficients
    finite_time_km = {}
    tau_indices = []
    A_selected = None
    dt = None
    N_tau_actual_from_km_data = 0
    N_A_actual_from_km_data = 0

    try:
        data_km_df = pd.read_csv(km_file_path)

        tau_indices_loaded = sorted(data_km_df['tau_index'].unique())
        tau_sec_loaded = sorted(data_km_df['tau_sec'].unique())
        A_selected = sorted(data_km_df['A'].unique())
        A_selected = np.array(sorted(data_km_df['A'].unique()))
        if len(tau_indices_loaded) > 1:
            dt = (tau_sec_loaded[1] - tau_sec_loaded[0]) / (tau_indices_loaded[1] - tau_indices_loaded[0])
        elif len(tau_indices_loaded) == 1 and tau_indices_loaded[0] > 0:
            dt = tau_sec_loaded[0] / tau_indices_loaded[0]
        else:
            dt = 0.0001

        N_tau_actual_from_km_data = len(tau_indices_loaded)
        N_A_actual_from_km_data = len(A_selected)

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

        print(f"  Loaded {N_tau_actual_from_km_data} tau values and {N_A_actual_from_km_data} amplitude points from KM data.")

    except Exception as e:
        print(f"Error loading data KM from {os.path.basename(km_file_path)}: {e}. Skipping.")
        traceback.print_exc()
        return None

    # --- Calculate P_tilde(a_i) from original simulation data ---
    original_sim_file_name = os.path.basename(km_file_path)
    original_sim_file_path = os.path.join(input_dir_sim_data, original_sim_file_name)

    p_a_weights_map = {}
    if not os.path.exists(original_sim_file_path):
        print(f"WARNING: Original simulation data file '{original_sim_file_path}' not found. Using uniform weights.")
        p_a_values_fallback = np.full(len(A_selected), 1.0 / len(A_selected))
        p_a_weights_arr = p_a_values_fallback * N_A_actual_from_km_data
        p_a_weights_map = {A: p_a_weights_arr[i] for i, A in enumerate(A_selected)}

    else:
        try:
            df_sim = pd.read_csv(original_sim_file_path)
            A_original_from_sim = None
            if 'Envelope' in df_sim.columns:
                A_original_from_sim = df_sim['Envelope'].values
            elif 'eta_filtered' in df_sim.columns:
                A_original_from_sim = np.abs(hilbert(df_sim['eta_filtered'].values))
            elif 'Eta' in df_sim.columns:
                A_original_from_sim = np.abs(hilbert(df_sim['Eta'].values))
            else:
                print(f"WARNING: Could not find a suitable amplitude column. Using uniform weights.")
                p_a_values_fallback = np.full(len(A_selected), 1.0 / len(A_selected))
                p_a_weights_arr = p_a_values_fallback * N_A_actual_from_km_data
                p_a_weights_map = {A: p_a_weights_arr[i] for i, A in enumerate(A_selected)}
                A_original_from_sim = None

            if A_original_from_sim is not None and len(A_original_from_sim) > 0:
                min_A_selected = np.min(A_selected)
                max_A_selected = np.max(A_selected)
                min_A_original = np.min(A_original_from_sim)
                max_A_original = np.max(A_original_from_sim)

                avg_spacing = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0

                if len(A_selected) > 1:
                    bin_edges = np.concatenate(([A_selected[0] - avg_spacing / 2],
                                                (A_selected[:-1] + A_selected[1:]) / 2,
                                                [A_selected[-1] + avg_spacing / 2]))
                else:
                    bin_edges = np.array([A_selected[0] - 0.5, A_selected[0] + 0.5])

                bin_edges[0] = min(bin_edges[0], min_A_original - avg_spacing / 2)
                bin_edges[-1] = max(bin_edges[-1], max_A_original + avg_spacing / 2)

                hist, _ = np.histogram(A_original_from_sim, bins=bin_edges)
                total_hist_count = np.sum(hist)
                if total_hist_count > 0:
                    p_a_values = hist / total_hist_count
                else:
                    p_a_values = np.zeros_like(hist, dtype=float)

                p_a_weights_arr = p_a_values * N_A_actual_from_km_data

                if len(p_a_weights_arr) == len(A_selected):
                    p_a_weights_map = {A_selected[i]: p_a_weights_arr[i] for i in range(len(A_selected))}
                else:
                    print(f"WARNING: Mismatch in length. Using uniform weights.")
                    p_a_values_fallback = np.full(len(A_selected), 1.0 / len(A_selected))
                    p_a_weights_arr = p_a_values_fallback * N_A_actual_from_km_data
                    p_a_weights_map = {A: p_a_weights_arr[i] for i, A in enumerate(A_selected)}

            else:
                print(f"WARNING: A_original_from_sim is empty. Using uniform weights.")
                p_a_values_fallback = np.full(len(A_selected), 1.0 / len(A_selected))
                p_a_weights_arr = p_a_values_fallback * N_A_actual_from_km_data
                p_a_weights_map = {A: p_a_weights_arr[i] for i, A in enumerate(A_selected)}


        except Exception as e:
            print(f"WARNING: Error processing original simulation data: {e}. Using uniform weights.")
            traceback.print_exc()
            p_a_values_fallback = np.full(len(A_selected), 1.0 / len(A_selected))
            p_a_weights_arr = p_a_values_fallback * N_A_actual_from_km_data
            p_a_weights_map = {A: p_a_weights_arr[i] for i, A in enumerate(A_selected)}

    # 3. Parameter Optimization
    try:
        valid_d2_means = []
        for tau in tau_indices:
            if tau in finite_time_km:
                d2_values = finite_time_km[tau]['D2']
                valid_d2 = d2_values[~np.isnan(d2_values)]
                if len(valid_d2) > 0:
                    avg_D2_for_tau = np.mean(valid_d2)
                    valid_A_for_d2 = finite_time_km[tau]['A'][~np.isnan(d2_values)]
                    weights_for_d2 = np.array([p_a_weights_map.get(a, 0.0) for a in valid_A_for_d2])
                    if np.sum(weights_for_d2) > 0:
                        avg_D2_for_tau = np.average(valid_d2, weights=weights_for_d2)
                    valid_d2_means.append(avg_D2_for_tau)

        if valid_d2_means:
            avg_D2 = np.mean(valid_d2_means)
            d_diffusion_0 = max(0.001, avg_D2)
        else:
            d_diffusion_0 = 0.01
    except Exception as e:
        print(f"  Error estimating initial d_diffusion: {e}. Using default.")
        d_diffusion_0 = 0.01

    nu_0 = 0.1
    kappa_0 = 0.1
    params_0 = [nu_0, kappa_0, d_diffusion_0]

    optimization_history_local = []

    print(f"  Starting optimization for {os.path.basename(km_file_path)}...")
    start_time_opt = time.time()
    result = None
    try:
        result = minimize(
            objective_function,
            params_0,
            args=(
            finite_time_km, A_selected, tau_indices, dt, deeponet_model, scalers, device, optimization_history_local,
            N_A_actual_from_km_data, p_a_weights_map),
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
                    f"  Using best parameters from iteration {int(best_iter['iteration'])}.")
        else:
            print(f"  No optimization history. Using initial parameters.")

    if np.isnan(nu_opt): nu_opt = 0.0
    if np.isnan(kappa_opt): kappa_opt = 0.0
    if np.isnan(d_diffusion_opt): d_diffusion_opt = 0.0

    d_diffusion_opt = max(0.0, d_diffusion_opt)


    # Calculate final MSE part for reporting
    final_mse_part = np.nan
    if optimization_history_local:
        final_params_for_mse_calc = [nu_opt, kappa_opt, d_diffusion_opt]
        temp_history_for_final_mse = []
        _ = objective_function(final_params_for_mse_calc, finite_time_km, A_selected, tau_indices, dt,
                               deeponet_model, scalers, device, temp_history_for_final_mse, N_A_actual_from_km_data, p_a_weights_map)
        if temp_history_for_final_mse:
            final_mse_part = temp_history_for_final_mse[0]['mse']


    # Calculate equivalent gamma
    gamma_equiv = np.nan
    try:
        gamma_equiv = d_diffusion_opt * 4 * OMEGA_0 ** 2
    except Exception:
        pass

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
        )
        if deeponet_km_opt_results is not None:
            deeponet_km_opt = {k: v for k, v in deeponet_km_opt_results.items() if k != 'deeponet_duration'}
    except Exception as e:
        print(f"  Error recalculating final DeepONet KM coeffs: {e}")

    A_plot = np.linspace(min(A_selected) * 0.9, max(A_selected) * 1.1, 200)
    D1_theory, D2_theory = None, None
    try:
        D1_theory = theoretical_D1_d(A_plot, nu_opt, kappa_opt, d_diffusion_opt)
        D2_theory = theoretical_D2_d(A_plot, nu_opt, kappa_opt, d_diffusion_opt)
        D1_theory = np.nan_to_num(D1_theory, nan=np.nan, posinf=np.nan, neginf=np.nan)
    except Exception as e:
        print(f"  Error calculating theoretical KM coeffs: {e}")

    # Plotting Final Comparison
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(14, 10))
        colors_plot = plt.cm.viridis(np.linspace(0, 1, max(1, N_tau_actual_from_km_data)))

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
                     label='Theoretical D1 (τ→0)', zorder=N_tau_actual_from_km_data + 1)

        ax1.set_title(f'Drift Coefficient (D1) Comparison (N_τ={N_tau_actual_from_km_data}, N_A={N_A_actual_from_km_data}) - DeepONet (With Weighted Loss)')
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
                     label='Theoretical D2 (τ→0)', zorder=N_tau_actual_from_km_data + 1)

        ax2.set_title(f'Diffusion Coefficient (D2) Comparison')
        ax2.set_xlabel('Amplitude (A)')
        ax2.set_ylabel('D2')
        ax2.legend(fontsize='small', ncol=2)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.axhline(0, color='gray', linewidth=0.5)
        if 'data_km_df' in locals() and not data_km_df.empty:
            valid_d2_data_for_plot_ylim = data_km_df['D2_data'].dropna()
            if not valid_d2_data_for_plot_ylim.empty:
                y_min_d2_data = np.nanpercentile(valid_d2_data_for_plot_ylim, 2)
                y_max_d2_data = np.nanpercentile(valid_d2_data_for_plot_ylim, 98)
                y_range = y_max_d2_data - y_min_d2_data
                if y_range > 1e-9:
                    ax2.set_ylim([y_min_d2_data - 0.1 * y_range, y_max_d2_data + 0.1 * y_range])
            else:
                ax2.set_ylim([-0.1, 1.0])


        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        title_str = f'DeepONet Parameter Estimation Results for {os.path.basename(km_file_path)}\n' \
                    f'Optimized: d_diff={d_diffusion_opt:.4f}, ν={nu_opt:.3f}, κ={kappa_opt:.3f} (Total Cost: {final_cost:.3e})\n'
        if nu_standard is not None and kappa_standard is not None and D_standard is not None:
            title_str += f'Standard: d_diff={D_standard:.4f}, ν={nu_standard:.3f}, κ={kappa_standard:.3f}\n' \
                         f'Relative Errors: d_diff={D_rel_err:.2%}, ν={nu_rel_err:.2%}, κ={kappa_rel_err:.2%}'
        else:
            title_str += 'Standard parameters not available for relative error calculation.'

        plt.suptitle(title_str, fontsize=14, y=0.99)

        output_plot_path = os.path.join(output_dir, f"deeponet_param_est_comparison_with_weighted_loss.png")
        try:
            plt.savefig(output_plot_path, dpi=300)
            print(f"  Comparison plot saved to {output_plot_path}")
        except Exception as e:
            print(f"  Error saving comparison plot: {e}")
        plt.close()

    except Exception as e:
        print(f"  An error occurred during final plotting: {e}")
        traceback.print_exc()

    # Save Optimization Iteration History
    if optimization_history_local:
        history_df = pd.DataFrame(optimization_history_local)
        output_history_path = os.path.join(output_dir, f"deeponet_optimization_history_with_weighted_loss.csv")
        try:
            history_df.to_csv(output_history_path, index=False)
            print(f"  Optimization history saved to {output_history_path}")
        except Exception as e:
            print(f"  Error saving optimization history: {e}")

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
        output_comp_path = os.path.join(output_dir, f"km_coefficients_comparison_detailed_with_weighted_loss.csv")
        comparison_df.to_csv(output_comp_path, index=False)
        print(f"  Detailed comparison data saved to {output_comp_path}")
    except Exception as e:
        print(f"  Could not save detailed comparison data: {e}")

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
                                                     f"deeponet_optimization_parameter_evolution_with_weighted_loss.png")
                plt.savefig(output_param_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting parameter evolution: {e}")

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
                                                          f"deeponet_optimization_total_cost_evolution_with_weighted_loss.png")
                plt.savefig(output_total_cost_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting Total Cost evolution: {e}")

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
                                                           f"deeponet_optimization_mse_penalty_evolution_with_weighted_loss.png")
                plt.savefig(output_mse_penalty_evo_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"  Error plotting MSE and Penalty evolution: {e}")

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
        'N_tau': N_tau_actual_from_km_data,
        'N_A': N_A_actual_from_km_data,
        'dt': dt
    }


# --- Main Batch Processing Logic ---
if __name__ == "__main__":
    joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp'
    print(f"Setting joblib temporary folder to: {joblib_temp_folder}")
    try:
        os.makedirs(joblib_temp_folder, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    except Exception as e:
        print(f"ERROR: Could not create or set joblib temp folder '{joblib_temp_folder}'.")
        print(f"Error details: {e}")
        exit()

    print("\n--- Loading Pre-trained DeepONet Model and Scalers (Global Load) ---")
    deeponet_model = None
    scalers = None
    try:
        scalers = torch.load(DEEPONET_SCALER_PATH)
        # Initialize model with matched architecture and dropout
        deeponet_model = DeepONet(
            DEEPONET_BRANCH_INPUT_DIM, DEEPONET_TRUNK_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS, DEEPONET_NUM_HIDDEN_LAYERS,
            DEEPONET_OUTPUT_FEATURES, DEEPONET_DROPOUT_RATE
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

            result = process_km_file(km_file_path, deeponet_model, scalers, DEVICE, BASE_OUTPUT_DIR, INPUT_DIR_SIM_DATA)
            if result:
                all_results_summary.append(result)
            print(f"--- Finished processing: {filename} ---")

        if all_results_summary:
            summary_df = pd.DataFrame(all_results_summary)
            overall_summary_path = os.path.join(BASE_OUTPUT_DIR, "deeponet_batch_optimization_summary_weighted.csv")
            try:
                summary_df.to_csv(overall_summary_path, index=False)
                print(f"\n--- Overall optimization summary saved to {overall_summary_path} ---")
            except Exception as e:
                print(f"ERROR: Could not save overall summary file: {e}")
        else:
            print("\nNo successful optimization results to summarize.")

    print("\n--- Batch processing finished ---")