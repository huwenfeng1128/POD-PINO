# POD-DeepONet参数辨识程序
import os
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
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'
# Modified BASE_OUTPUT_DIR to reflect single run, multiple method processing
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\compare\result\POD-DeepONet_Time'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Point to the V3 training outputs
V3_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\train_result_v4'
RUN_ID = "pod_v3_cosine_warmup_layernorm_weightedloss"

DEEPONET_MODEL_PATH = os.path.join(V3_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(V3_RESULT_DIR, f'scalers_{RUN_ID}.pth')
DEEPONET_POD_PARAMS_PATH = os.path.join(V3_RESULT_DIR, f'pod_params_{RUN_ID}.pth')

# DeepONet Configuration (Must match training script)
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# --- Optimization Settings ---
OMEGA_0 = 2 * math.pi * 150
# Define different optimization configurations
OPTIMIZERS = {
    'Nelder-Mead': {
        'method': 'Nelder-Mead',
        'options': {'maxiter': 3000, 'disp': False, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}
    },
    'L-BFGS-B': {
        'method': 'L-BFGS-B',
        'options': {'maxiter': 3000, 'disp': False, 'maxcor': 10, 'ftol': 1e-10, 'gtol': 1e-8}
    }
}
D_NEGATIVE_PENALTY_FACTOR = 1e5

# --- Single Run Configuration ---
# Set to a specific filename (e.g., 'simulation_results_nu0.1_kappa0.1_D0.01.csv')
# If None, the first CSV file found in BASE_KM_DATA_DIR will be processed.
SELECTED_KM_FILE_FOR_SINGLE_RUN = None
RANDOM_SEED = 42  # Fixed random seed for reproducibility


# --- Model Definitions (Exact match with V3 training script) ---
class MLP(nn.Module):
    """MLP with GELU and LayerNorm - exactly as in V3 training script."""

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
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
    """PODDeepONet - exactly as in V3 training script."""

    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes,
                 pod_basis, y_mean_pod_scaled, dropout_rate):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, dropout_rate)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod_scaled, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        """Outputs prediction in SCALED space."""
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
        """Outputs prediction in ORIGINAL PHYSICAL space."""
        y_pred_scaled = self.forward(branch_x)
        y_mean_scaler_t = torch.tensor(y_mean_scaler, dtype=DTYPE, device=self.pod_basis.device)
        y_std_scaler_t = torch.tensor(y_std_scaler, dtype=DTYPE, device=self.pod_basis.device)
        return y_pred_scaled * y_std_scaler_t + y_mean_scaler_t


# --- Helper Functions ---
def manual_scaler_transform(data, mean, std):
    """Applies pre-computed scaling."""
    if isinstance(data, torch.Tensor):
        device = data.device
        data_t = data
    else:
        device = DEVICE
        data_t = torch.tensor(data, dtype=DTYPE, device=device)

    mean_t = torch.tensor(mean, dtype=DTYPE, device=device)
    std_t = torch.tensor(std, dtype=DTYPE, device=device)
    std_t[std_t < 1e-10] = 1.0  # Avoid division by zero
    return (data_t - mean_t) / std_t


def parse_params_from_filename(filepath):
    """Extract nu, kappa, d from filename like '(0.1,0.2,0.3).csv'"""
    filename = os.path.basename(filepath)
    try:
        if filename.startswith('(') and filename.endswith('.csv'):
            values = filename[1:-5].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except (ValueError, IndexError):
        pass
    return None, None, None


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """Theoretical D1 coefficient."""
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """Theoretical D2 coefficient."""
    return np.full_like(np.asarray(A), d_diffusion)


def compute_deeponet_km_coefficients(params, A_selected, tau_indices_map, model, scalers,
                                     unified_a_grid, unified_tau_grid):
    """Compute KM coefficients using pre-trained DeepONet."""
    nu, kappa, d_diffusion = params

    if any(np.isnan(p) or np.isinf(p) for p in params):
        return None

    start_time = time.time()
    try:
        model.eval()
        with torch.no_grad():
            # Prepare branch input
            branch_input_np = np.array([[nu, kappa, d_diffusion]])
            branch_input_scaled_t = manual_scaler_transform(
                branch_input_np, scalers['branch_mean'], scalers['branch_std']
            )

            # Get prediction in physical space
            y_pred_t = model.predict(
                branch_input_scaled_t,
                scalers['y_mean_scaler'],
                scalers['y_std_scaler']
            )
            y_pred_np = y_pred_t.cpu().numpy().flatten()

            # Reconstruct D1 and D2 fields
            field_len_per_type = len(y_pred_np) // 2
            N_tau_points = len(unified_tau_grid)
            N_a_points = len(unified_a_grid)

            D1_field_pred = y_pred_np[:field_len_per_type].reshape(N_tau_points, N_a_points)
            D2_field_pred = y_pred_np[field_len_per_type:].reshape(N_tau_points, N_a_points)

            # Interpolate for each required tau
            results_for_tau = {}

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                # Find closest tau in unified grid
                closest_tau_grid_idx = np.argmin(np.abs(unified_tau_grid - tau_sec_km))

                # Get predicted slices
                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                # Interpolate to KM file's A-grid
                interp_d1 = interp1d(unified_a_grid, D1_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                interp_d2 = interp1d(unified_a_grid, D2_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)

                results_for_tau[tau_idx_km] = {
                    'A': A_selected,
                    'D1': interp_d1(A_selected),
                    'D2': interp_d2(A_selected)
                }
            pred_duration = time.time() - start_time
            return results_for_tau, pred_duration

    except Exception as e:
        print(f"  DeepONet prediction error for params {params}: {e}")
        traceback.print_exc()
        return None, time.time() - start_time


def objective_function(params, data_km, A_selected, tau_indices_map, model, scalers,
                       optimization_history_local, p_a_weights_map, unified_a_grid, unified_tau_grid,
                       start_time_iteration):
    """Objective function for optimization."""
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    # Penalty for negative diffusion
    d_negative_penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion) if d_diffusion < 0 else 0.0

    # Get DeepONet predictions
    deeponet_km_results, pred_duration = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices_map, model, scalers, unified_a_grid, unified_tau_grid
    )

    if deeponet_km_results is None:
        cost = 1e11 + d_negative_penalty
        mean_weighted_sq_error = np.nan
    else:
        total_weighted_sq_error = 0
        num_compared_points = 0

        for tau_idx in tau_indices_map.keys():
            if tau_idx not in data_km or tau_idx not in deeponet_km_results:
                continue

            D1_data = data_km[tau_idx]['D1']
            D2_data = data_km[tau_idx]['D2']
            D1_pred = deeponet_km_results[tau_idx]['D1']
            D2_pred = deeponet_km_results[tau_idx]['D2']

            # Valid points mask
            valid_mask = (~np.isnan(D1_data) & ~np.isnan(D1_pred) &
                          ~np.isnan(D2_data) & ~np.isnan(D2_pred))

            if not np.any(valid_mask):
                continue

            A_current = data_km[tau_idx]['A'][valid_mask]
            weights = np.array([p_a_weights_map.get(a, 0.0) for a in A_current])

            error_d1 = (D1_data[valid_mask] - D1_pred[valid_mask]) ** 2
            error_d2 = (D2_data[valid_mask] - D2_pred[valid_mask]) ** 2

            total_weighted_sq_error += np.sum(weights * (error_d1 + error_d2))
            num_compared_points += len(A_current)

        mean_weighted_sq_error = (total_weighted_sq_error / num_compared_points
                                  if num_compared_points > 0 else 1e10)
        cost = mean_weighted_sq_error + d_negative_penalty

    # Log history
    optimization_history_local.append({
        'iteration': current_iter,
        'nu': nu,
        'kappa': kappa,
        'd_diffusion': d_diffusion,
        'mse': mean_weighted_sq_error,
        'penalty': d_negative_penalty,
        'total_cost': cost,
        'pred_duration_sec': pred_duration,
        'iteration_time_sec': time.time() - start_time_iteration,
        'total_elapsed_time_sec': time.time() - start_time_iteration
        # This will be total time as objective is called by optimizer
    })

    return cost if not (np.isnan(cost) or np.isinf(cost)) else 1e12


def plot_comparison(data_km, deeponet_results, theo_results, A_selected, tau_indices_map,
                    nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std, output_dir):
    """Generate comparison plots."""
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Comparison: Optimized vs Standard vs Data\n'
                 f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
                 f'Std: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
                 fontsize=12)

    for i, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        if tau_idx not in data_km:
            continue

        A_data = data_km[tau_idx]['A']
        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']

        # D1 plot
        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)

        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(A_selected, deeponet_results[tau_idx]['D1'], 'b-',
                       label=f'DeepONet Opt', linewidth=2)

        if theo_results and tau_idx in theo_results:
            ax_d1.plot(A_selected, theo_results[tau_idx]['D1'], 'r--',
                       label=f'Theoretical Std', linewidth=2)

        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at τ={tau_sec:.4f}s')
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)

        # D2 plot
        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)

        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(A_selected, deeponet_results[tau_idx]['D2'], 'b-',
                       label=f'DeepONet Opt', linewidth=2)

        if theo_results and tau_idx in theo_results:
            ax_d2.plot(A_selected, theo_results[tau_idx]['D2'], 'r--',
                       label=f'Theoretical Std', linewidth=2)

        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at τ={tau_sec:.4f}s')
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=150)
    plt.close()


def process_km_file(km_file_path, deeponet_model, scalers, unified_a_grid, unified_tau_grid,
                    base_output_dir, input_dir_sim_data, optimizer_config, random_seed):
    """Process a single KM file for parameter identification with specified optimizer and seed."""
    print(f"\n--- Processing: {os.path.basename(km_file_path)} ---")
    print(f"  Optimizer: {optimizer_config['method']}, Seed: {random_seed}")

    # Set random seed for this run
    np.random.seed(random_seed)
    # If using PyTorch's random functions for initialization etc., you might also need:
    # torch.manual_seed(random_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(random_seed)

    # Parse standard parameters
    nu_std, kappa_std, D_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        print(f"WARNING: Could not parse params from '{os.path.basename(km_file_path)}'. Skipping.")
        return None

    # Create output directory for this file and optimizer
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0]
    optimizer_method_name = optimizer_config['method']
    output_dir = os.path.join(base_output_dir, optimizer_method_name, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load KM data
    try:
        data_km_df = pd.read_csv(km_file_path)
        finite_time_km = {
            tau_idx: {
                'A': group['A'].values,
                'D1': group['D1_data'].values,
                'D2': group['D2_data'].values,
                'tau_sec': group['tau_sec'].iloc[0]
            }
            for tau_idx, group in data_km_df.groupby('tau_index')
        }
        A_selected = sorted(data_km_df['A'].unique())
        tau_indices_map = {
            idx: ts for idx, ts in
            data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }
    except Exception as e:
        print(f"Error loading KM data: {e}");
        return None

    # Load simulation data for weights
    p_a_weights_map = {a: 1.0 / len(A_selected) for a in A_selected}  # Uniform weights by default
    sim_file_path = os.path.join(input_dir_sim_data, os.path.basename(km_file_path))

    if os.path.exists(sim_file_path):
        try:
            df_sim = pd.read_csv(sim_file_path)
            # Try to find amplitude column, default to 'Eta' if available
            sim_envelope = None
            if 'Eta' in df_sim.columns:
                sim_envelope = np.abs(hilbert(df_sim['Eta'].values))
            elif 'Envelope' in df_sim.columns:
                sim_envelope = df_sim['Envelope'].values
            elif 'eta_filtered' in df_sim.columns:  # Fallback if eta_filtered is present
                sim_envelope = np.abs(hilbert(df_sim['eta_filtered'].values))

            if sim_envelope is not None and len(sim_envelope) > 0:
                bin_width = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0
                bin_edges = np.concatenate([
                    [A_selected[0] - bin_width / 2],
                    A_selected[:-1] + bin_width / 2,
                    [A_selected[-1] + bin_width / 2]
                ])

                hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
                p_a_weights_map = {A_selected[i]: hist[i] for i in range(len(A_selected))}
            else:
                print(
                    f"WARNING: Amplitude column not found or empty in '{os.path.basename(sim_file_path)}'. Using uniform weights.")
        except Exception as e:
            print(f"WARNING: Could not process sim file for weights: {e}. Using uniform weights.")
            traceback.print_exc()
    else:
        print(f"WARNING: Simulation data file '{sim_file_path}' not found. Using uniform weights.")

    # Initialize optimization
    # Estimate initial 'd' from mean of D2 data across all taus
    initial_d_estimates = []
    for tau_idx in tau_indices_map.keys():
        if tau_idx in finite_time_km:
            valid_d2 = finite_time_km[tau_idx]['D2'][~np.isnan(finite_time_km[tau_idx]['D2'])]
            if len(valid_d2) > 0:
                initial_d_estimates.append(np.mean(valid_d2))

    initial_d = np.nanmean(initial_d_estimates) if initial_d_estimates else 0.01
    params_0 = [0.1, 0.1, max(0.01, initial_d)]  # Ensure initial D > 0

    optimization_history = []

    print(f"  Initial guess: ν={params_0[0]:.4f}, κ={params_0[1]:.4f}, D={params_0[2]:.4f}")

    # Run optimization
    start_time_total = time.time()
    result = None
    try:
        result = minimize(
            objective_function, params_0,
            args=(finite_time_km, A_selected, tau_indices_map, deeponet_model, scalers,
                  optimization_history, p_a_weights_map, unified_a_grid, unified_tau_grid, start_time_total),
            method=optimizer_config['method'],
            options=optimizer_config['options']
        )
    except Exception as e:
        print(f"  Error during optimization: {e}")
        traceback.print_exc()

    elapsed_time_total = time.time() - start_time_total

    # Extract optimized parameters
    nu_opt, kappa_opt, d_opt = params_0  # Default to initial guess
    final_cost = np.nan
    optimization_successful = False

    if result and result.success:
        nu_opt, kappa_opt, d_opt = result.x
        final_cost = result.fun
        optimization_successful = True
    elif result and hasattr(result, 'x'):  # Use best found parameters even if not converged
        print(f"  Optimization did not converge. Using best from history.")
        nu_opt, kappa_opt, d_opt = result.x
        final_cost = result.fun
    elif optimization_history:  # Fallback to best from logged history if result.x is not available
        print(f"  Optimizer result incomplete. Using best from logged history.")
        best_idx = np.argmin([h['total_cost'] for h in optimization_history])
        nu_opt = optimization_history[best_idx]['nu']
        kappa_opt = optimization_history[best_idx]['kappa']
        d_opt = optimization_history[best_idx]['d_diffusion']
        final_cost = optimization_history[best_idx]['total_cost']

    d_opt = max(0.0, d_opt)  # Ensure non-negative

    print(f"  Optimization finished in {elapsed_time_total:.2f}s")
    print(f"  Optimized: ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}")
    print(f"  Standard:  ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f}")

    # --- Error Calculation ---
    nu_error = abs(nu_opt - nu_std)
    kappa_error = abs(kappa_opt - kappa_std)
    d_error = abs(d_opt - D_std)

    # Relative Errors (handle potential zero division)
    nu_rel_error = nu_error / abs(nu_std) if nu_std != 0 else 0.0
    kappa_rel_error = kappa_error / abs(kappa_std) if kappa_std != 0 else 0.0
    d_rel_error = d_error / abs(D_std) if D_std != 0 else 0.0

    # Compute final predictions with optimized parameters
    deeponet_opt_results, pred_duration_opt = compute_deeponet_km_coefficients(
        [nu_opt, kappa_opt, d_opt], A_selected, tau_indices_map,
        deeponet_model, scalers, unified_a_grid, unified_tau_grid
    )

    # Compute theoretical predictions with standard parameters
    theo_std_results = {}
    for tau_idx in tau_indices_map.keys():
        if tau_idx in finite_time_km:  # Only compute for taus present in data
            theo_std_results[tau_idx] = {
                'A': A_selected,
                'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
                'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
            }

    # Generate plots
    plot_comparison(finite_time_km, deeponet_opt_results, theo_std_results,
                    A_selected, tau_indices_map, nu_opt, kappa_opt, d_opt,
                    nu_std, kappa_std, D_std, output_dir)

    # Save optimization history
    if optimization_history:
        history_df = pd.DataFrame(optimization_history)
        # Add total elapsed time to the last iteration record
        if not history_df.empty:
            history_df.loc[
                history_df['iteration'] == history_df['iteration'].max(), 'total_elapsed_time_sec'] = elapsed_time_total

        history_df.to_csv(os.path.join(output_dir, f'optimization_history_{optimizer_method_name}.csv'), index=False)

    # Return results dictionary
    return {
        'filename': os.path.basename(km_file_path),
        'nu_standard': nu_std, 'kappa_standard': kappa_std, 'D_standard': D_std,
        'nu_optimized': nu_opt, 'kappa_optimized': kappa_opt, 'd_diffusion_optimized': d_opt,
        'nu_error': nu_error,
        'kappa_error': kappa_error,
        'd_error': d_error,
        'nu_rel_error': nu_rel_error,
        'kappa_rel_error': kappa_rel_error,
        'd_rel_error': d_rel_error,
        'optimization_time_sec': elapsed_time_total,
        'final_cost': final_cost,
        'optimization_successful': optimization_successful,
        'optimizer_method': optimizer_method_name,
        'random_seed': random_seed,
        'total_pred_duration_opt_sec': pred_duration_opt if deeponet_opt_results else np.nan
    }


# --- Main Script ---
if __name__ == "__main__":
    # Configure joblib temp folder (optional but good practice)
    joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp'
    try:
        os.makedirs(joblib_temp_folder, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
        print(f"Set joblib temporary folder to: {joblib_temp_folder}")
    except Exception as e:
        print(f"WARNING: Could not set joblib temp folder '{joblib_temp_folder}': {e}")

    print("\n=== Loading Pre-trained DeepONet Model ===")

    try:
        # Load scalers
        print("Loading scalers...")
        scalers = torch.load(DEEPONET_SCALER_PATH, map_location=DEVICE)

        # Load POD parameters
        print("Loading POD parameters...")
        pod_params = torch.load(DEEPONET_POD_PARAMS_PATH, map_location=DEVICE)

        pod_basis = pod_params['pod_basis']
        y_mean_pod_scaled = pod_params['y_mean_pod_scaled']
        num_pod_modes = pod_params['num_pod_modes']
        unified_a_grid = pod_params['unified_a_grid']
        unified_tau_grid = pod_params['unified_tau_grid']

        # Convert to numpy if they are tensors
        if isinstance(unified_a_grid, torch.Tensor):
            unified_a_grid = unified_a_grid.cpu().numpy()
        if isinstance(unified_tau_grid, torch.Tensor):
            unified_tau_grid = unified_tau_grid.cpu().numpy()

        print(f"  POD modes: {num_pod_modes}")
        print(f"  A-grid points: {len(unified_a_grid)}")
        print(f"  Tau-grid points: {len(unified_tau_grid)}")

        # Instantiate model
        print("Building model...")
        deeponet_model = PODDeepONet(
            DEEPONET_BRANCH_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS,
            DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes,
            pod_basis,
            y_mean_pod_scaled,
            DEEPONET_DROPOUT_RATE
        )

        # Load trained weights
        print("Loading model weights...")
        # Use map_location='cpu' first to ensure compatibility, then move to DEVICE
        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location='cpu'))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        print("✓ Model loaded successfully!\n")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Model or scaler file not found. Please check paths.")
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load DeepONet model or parameters.")
        print(f"Error: {e}")
        traceback.print_exc()
        exit(1)

    # --- Batch Processing ---
    print("=== Starting Batch Processing ===")
    all_results = []
    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])

    if not km_files:
        print(f"ERROR: No .csv files found in {BASE_KM_DATA_DIR}. Exiting.")
        exit()

    # Determine which files to process
    files_to_process = []
    if SELECTED_KM_FILE_FOR_SINGLE_RUN:
        if SELECTED_KM_FILE_FOR_SINGLE_RUN in km_files:
            files_to_process = [SELECTED_KM_FILE_FOR_SINGLE_RUN]
            print(f"--- Processing ONLY the selected file: {SELECTED_KM_FILE_FOR_SINGLE_RUN} ---")
        else:
            print(f"WARNING: Selected file '{SELECTED_KM_FILE_FOR_SINGLE_RUN}' not found in '{BASE_KM_DATA_DIR}'.")
            print(f"         Will process the first found file instead: {km_files[0]}")
            files_to_process = [km_files[0]]
    else:
        files_to_process = [km_files[0]]  # Process the first file if none is selected
        print(f"--- Processing the first found file: {files_to_process[0]} ---")

    print(f"Total files selected for processing: {len(files_to_process)}")

    # Loop through selected files and then through optimizers for each file
    for filename in tqdm(files_to_process, desc="Processing Selected KM Files"):
        km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)

        # Loop through each optimizer configuration
        for optimizer_name, optimizer_config in OPTIMIZERS.items():
            result = process_km_file(
                km_file_path, deeponet_model, scalers,
                unified_a_grid, unified_tau_grid,
                BASE_OUTPUT_DIR, INPUT_DIR_SIM_DATA,
                optimizer_config,  # Pass the full config dict
                RANDOM_SEED  # Pass the fixed random seed
            )
            if result:
                all_results.append(result)

    # --- Save Summary ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, f"batch_summary_single_run_seed{RANDOM_SEED}.csv")
        try:
            summary_df.to_csv(summary_path, index=False)
            print(f"\n=== Summary Statistics ===")
            print(f"Total runs completed (file x optimizer): {len(all_results)}")

            # Calculate average errors grouped by optimizer method
            grouped_summary = summary_df.groupby('optimizer_method')

            print("\n--- Average Metrics Per Optimizer Method ---")
            for method, group in grouped_summary:
                print(f"\nOptimizer: {method}")
                print(f"  Number of runs: {len(group)}")
                print(f"  Mean optimization time: {group['optimization_time_sec'].mean():.2f}s")
                print(f"  Mean final cost: {group['final_cost'].mean():.3e}")
                print(f"  Mean absolute errors:")
                print(f"    ν error:  {group['nu_error'].mean():.6f}")
                print(f"    κ error:  {group['kappa_error'].mean():.6f}")
                print(f"    D error:  {group['d_error'].mean():.6f}")
                print(f"  Mean relative errors:")
                print(f"    ν rel error:  {group['nu_rel_error'].mean():.2%}")
                print(f"    κ rel error:  {group['kappa_rel_error'].mean():.2%}")
                print(f"    D rel error:  {group['d_rel_error'].mean():.2%}")

            print(f"\nFull summary saved to: {summary_path}")

        except Exception as e:
            print(f"\nERROR: Could not save summary file to {summary_path}: {e}")
            traceback.print_exc()

    else:
        print("\nNo successful runs completed to generate a summary.")

    print("\n=== Batch processing completed ===")