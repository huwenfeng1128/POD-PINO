# 批量化计算所有数据的FD识别参数

# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.sparse import diags, identity, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time
from tqdm import tqdm
import math
from joblib import Parallel, delayed
import numpy.fft
import traceback
from scipy.signal import hilbert

# --- START: Force Joblib Temp Folder to Pure ASCII Path ---
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


# --- Configuration ---
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
BASE_RAW_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\compare\result\FD_time'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

F_FILTER = 60
OMEGA_0 = 2 * math.pi * 150

OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {'maxiter': 300, 'disp': True, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}

N_JOBS = 16
FP_N_GRID = 300
CN_DT_STEP = 1e-3
D_DIFFUSION_UPPER_BOUND = 100.0

SELECTED_KM_FILE_FOR_SINGLE_RUN = None
RANDOM_SEED = 42


# --- Helper Functions ---

def parse_params_from_filename(filepath):
    filename = os.path.basename(filepath)
    nu_std, kappa_std, D_std = None, None, None
    nu_str, kappa_str, D_str = None, None, None
    try:
        if filename.startswith('(') and filename.endswith('.csv'):
            inner_part = filename[1:-5]
            values = inner_part.split(',')
            if len(values) == 3:
                nu_str, kappa_str, D_str = values[0], values[1], values[2]
                nu_std, kappa_std, D_std = float(nu_str), float(kappa_str), float(D_str)
    except (ValueError, IndexError):
        nu_std, kappa_std, D_std = None, None, None
    return nu_std, kappa_std, D_std, nu_str, kappa_str, D_str


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
    return np.full_like(A, max(0.0, d_diffusion))


def create_grid(A_max_data, N_grid=FP_N_GRID):
    A_grid_max = 1.5 * A_max_data
    A_grid_min = 0
    A_grid = np.linspace(A_grid_min, A_grid_max, N_grid)
    dA = A_grid[1] - A_grid[0]
    return A_grid, dA, A_grid_max


class FPSolver:
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
            self.D2_values = np.maximum(self.D2_values, 1e-15)
        except Exception as e:
            raise ValueError(f"Failed to calculate valid D1/D2: {e}")

        try:
            self.L = self._build_forward_operator_matrix()
            if self.L is None or not isinstance(self.L, csc_matrix):
                raise ValueError("Operator matrix L was not built correctly.")
            if not np.all(np.isfinite(self.L.data)):
                self.L.data[~np.isfinite(self.L.data)] = 0.0
                self.L.eliminate_zeros()
        except Exception as e:
            raise ValueError(f"Failed to build operator matrix L: {e}")

    def _build_forward_operator_matrix(self):
        N = self.N_grid
        dA = self.dA
        dA2 = dA ** 2
        L = lil_matrix((N, N), dtype=float)

        safe_D1 = np.nan_to_num(self.D1_values, nan=0.0, posinf=1e10, neginf=-1e10)
        safe_D2 = self.D2_values

        dD1_dA = np.gradient(safe_D1, dA, edge_order=1)
        dD2_dA = np.gradient(safe_D2, dA, edge_order=1)
        d2D2_dA2 = np.gradient(dD2_dA, dA, edge_order=1)

        dD1_dA = np.nan_to_num(dD1_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        dD2_dA = np.nan_to_num(dD2_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        d2D2_dA2 = np.nan_to_num(d2D2_dA2, nan=0.0, posinf=1e10, neginf=-1e10)

        coeff_p_center = -dD1_dA + d2D2_dA2
        coeff_dp_dA_lower = -safe_D1 / (2 * dA)
        coeff_dp_dA_upper = safe_D1 / (2 * dA)
        coeff_d2p_dA2_lower = 0.5 * safe_D2 / dA2
        coeff_d2p_dA2_center = -safe_D2 / dA2
        coeff_d2p_dA2_upper = 0.5 * safe_D2 / dA2

        for i in range(1, N - 1):
            L[i, i - 1] = coeff_dp_dA_lower[i] + coeff_d2p_dA2_lower[i]
            L[i, i] = coeff_p_center[i] + coeff_d2p_dA2_center[i]
            L[i, i + 1] = coeff_dp_dA_upper[i] + coeff_d2p_dA2_upper[i]

        L[0, :] = 0
        L[0, 0] = 1
        L[N - 1, :] = 0
        L[N - 1, N - 1] = 1
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
        num_steps = max(1, int(round(tau / dt_step)))
        actual_dt_step = tau / num_steps

        P_current = self.initial_condition_delta(A_target)
        Id = identity(self.N_grid, format='csc')
        try:
            LHS = Id - 0.5 * actual_dt_step * self.L
            RHS = Id + 0.5 * actual_dt_step * self.L
            for step in range(num_steps):
                b = RHS.dot(P_current)
                b[0], b[self.N_grid - 1] = 0.0, 0.0
                P_next = spsolve(LHS, b)
                if np.isnan(P_next).any(): return None, "NaN in solver"
                P_current = P_next
        except Exception as e:
            return None, str(e)

        P_final = np.maximum(P_current, 0)
        integral = np.trapz(P_final, self.A_grid)
        if integral > 1e-10: P_final /= integral
        return P_final, None

    def calculate_km_from_solution(self, P_final, A_target, tau):
        if P_final is None or tau <= 0: return np.nan, np.nan
        M1 = np.trapz((self.A_grid - A_target) * P_final, self.A_grid)
        M2 = np.trapz((self.A_grid - A_target) ** 2 * P_final, self.A_grid)
        return M1 / tau, M2 / (2 * tau)


def compute_fp_km_coefficients(params, A_selected, tau_indices, dt, A_max_data, n_jobs, optimization_history_local):
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1
    if any(np.isnan(p) or np.isinf(p) for p in params) or d_diffusion < 0 or kappa < 0:
        return None

    try:
        A_grid, dA, _ = create_grid(A_max_data, N_grid=FP_N_GRID)
        solver = FPSolver(A_grid, dA, nu, kappa, d_diffusion)
    except Exception:
        return None

    results = {}
    fp_compute_start_time = time.time()

    def solve_and_calc_km(A_target_val, tau_sec_val):
        P_final, _ = solver.solve_forward_cn(A_target_val, tau_sec_val)
        return solver.calculate_km_from_solution(P_final, A_target_val, tau_sec_val)

    for tau_idx in tau_indices:
        tau_sec = tau_idx * dt
        km_list = Parallel(n_jobs=n_jobs, backend='loky')(delayed(solve_and_calc_km)(A, tau_sec) for A in A_selected)
        results[tau_idx] = {'A': A_selected, 'D1': np.array([r[0] for r in km_list]),
                            'D2': np.array([r[1] for r in km_list])}

    results['fp_duration'] = time.time() - fp_compute_start_time
    return results


def objective_function(params, data_km, A_selected, tau_indices, dt, A_max_data, n_jobs, optimization_history_local,
                       N_A_from_km_data, p_a_weights_map):
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1
    penalty = 0.0
    if d_diffusion < 0: penalty += (abs(d_diffusion) + 1) ** 2 * 1e6
    if kappa < 0: penalty += (abs(kappa) + 1) ** 2 * 1e6
    if d_diffusion > D_DIFFUSION_UPPER_BOUND: penalty += (d_diffusion - D_DIFFUSION_UPPER_BOUND) ** 2 * 1e4

    if any(np.isnan(p) or np.isinf(p) for p in params):
        optimization_history_local.append(
            {'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion, 'mse': 1e12,
             'fp_duration_sec': 0, 'num_compared_points': 0})
        return 1e12

    start_fp_time = time.time()
    fp_km_results = compute_fp_km_coefficients(params, A_selected, tau_indices, dt, A_max_data, n_jobs,
                                               optimization_history_local)
    fp_duration = time.time() - start_fp_time

    if fp_km_results is None:
        optimization_history_local.append(
            {'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion, 'mse': 1e11,
             'fp_duration_sec': fp_duration, 'num_compared_points': 0})
        return 1e11 + penalty

    # 提取时间并移除键值以防干扰后续循环
    fp_duration = fp_km_results.pop('fp_duration', fp_duration)

    total_weighted_sq_error = 0
    total_weight_sum = 0
    for tau_idx in tau_indices:
        if tau_idx not in data_km or tau_idx not in fp_km_results: continue
        D1_d, D2_d = data_km[tau_idx]['D1'], data_km[tau_idx]['D2']
        D1_f, D2_f = fp_km_results[tau_idx]['D1'], fp_km_results[tau_idx]['D2']
        mask = np.isfinite(D1_d) & np.isfinite(D1_f) & np.isfinite(D2_d) & np.isfinite(D2_f)
        if not np.any(mask): continue

        A_curr = data_km[tau_idx]['A'][mask]
        weights = np.array([p_a_weights_map.get(a, 0.0) for a in A_curr])
        total_weighted_sq_error += np.sum(weights * ((D1_d[mask] - D1_f[mask]) ** 2 + (D2_d[mask] - D2_f[mask]) ** 2))
        total_weight_sum += np.sum(weights)

    mse = total_weighted_sq_error / total_weight_sum if total_weight_sum > 1e-15 else 1e10
    optimization_history_local.append(
        {'iteration': current_iter, 'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion, 'mse': mse,
         'fp_duration_sec': fp_duration, 'num_compared_points': total_weight_sum})
    return mse + penalty


def process_km_file_fp(km_file_path, base_raw_data_dir, base_output_dir):
    print(f"\n--- Processing file: {os.path.basename(km_file_path)} ---")
    nu_std, kappa_std, D_std, nu_str, kappa_str, D_str = parse_params_from_filename(km_file_path)
    if nu_std is None: return None

    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0].replace('.', '_')
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    raw_data_path = os.path.join(base_raw_data_dir, f"({nu_str},{kappa_str},{D_str}).csv")
    try:
        df_raw = pd.read_csv(raw_data_path)
        A_original = df_raw['Envelope'].values if 'Envelope' in df_raw.columns else np.abs(
            hilbert(df_raw.iloc[:, 1].values))
        dt = np.mean(np.diff(df_raw.iloc[:, 0].values)) if len(df_raw) > 1 else 0.0001
        A_max_data = np.max(A_original)
    except Exception as e:
        print(f"Error reading raw data: {e}");
        return None

    try:
        df_km = pd.read_csv(km_file_path)
        tau_indices = sorted(df_km['tau_index'].unique())
        A_selected = np.array(sorted(df_km['A'].unique()))
        finite_time_km = {
            t: {'A': df_km[df_km['tau_index'] == t]['A'].values, 'D1': df_km[df_km['tau_index'] == t]['D1_data'].values,
                'D2': df_km[df_km['tau_index'] == t]['D2_data'].values} for t in tau_indices}
    except Exception as e:
        print(f"Error reading KM data: {e}");
        return None

    # Calculate Weights
    avg_spacing = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0
    bin_edges = np.concatenate(
        [[A_selected[0] - avg_spacing / 2], A_selected[:-1] + avg_spacing / 2, [A_selected[-1] + avg_spacing / 2]])
    hist, _ = np.histogram(A_original, bins=bin_edges)
    p_a_weights_map = {A_selected[i]: (hist[i] / np.sum(hist)) * len(A_selected) for i in range(len(A_selected))}

    # Optimization
    params_0 = [0.1, 0.1, 0.01]
    history = []
    res = minimize(objective_function, params_0, args=(
    finite_time_km, A_selected, tau_indices, dt, A_max_data, N_JOBS, history, len(A_selected), p_a_weights_map),
                   method=OPTIMIZER_METHOD, options=OPTIMIZER_OPTIONS)

    nu_opt, kappa_opt, d_opt = res.x

    # 5. Recalculate - 修复了这里未解析的引用 fp_km_results
    fp_km_opt = None
    try:
        fp_km_opt_results = compute_fp_km_coefficients([nu_opt, kappa_opt, d_opt], A_selected, tau_indices, dt,
                                                       A_max_data, N_JOBS, [])
        if fp_km_opt_results is not None:
            # 使用正确的变量名 fp_km_opt_results 提取时间
            _ = fp_km_opt_results.pop('fp_duration', 0)
            fp_km_opt = fp_km_opt_results
    except Exception as e:
        print(f"Final calculation error: {e}")

    # Plotting & Saving (Simplified for brevity, similar to original)
    plt.figure(figsize=(10, 6))
    plt.plot(A_selected, theoretical_D1_d(A_selected, nu_opt, kappa_opt, d_opt), 'k-', label='Theory')
    plt.title(f"Optimized: nu={nu_opt:.3f}, k={kappa_opt:.3f}, D={d_opt:.4f}")
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)

    return {'filename': os.path.basename(km_file_path), 'nu_standard': nu_std, 'kappa_standard': kappa_std,
            'D_standard': D_std, 'nu_optimized': nu_opt, 'kappa_optimized': kappa_opt, 'd_diffusion_optimized': d_opt,
            'final_cost_mse': res.fun, 'total_optimization_duration_sec': res.get('total_time', 0)}


if __name__ == "__main__":
    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    if not km_files: exit()

    files_to_process = [SELECTED_KM_FILE_FOR_SINGLE_RUN] if SELECTED_KM_FILE_FOR_SINGLE_RUN in km_files else [
        km_files[0]]
    all_res = []
    for f in tqdm(files_to_process):
        res = process_km_file_fp(os.path.join(BASE_KM_DATA_DIR, f), BASE_RAW_DATA_DIR, BASE_OUTPUT_DIR)
        if res: all_res.append(res)

    if all_res:
        summary_path = os.path.join(BASE_OUTPUT_DIR, f"summary_seed{RANDOM_SEED}.csv")
        pd.DataFrame(all_res).to_csv(summary_path, index=False)
        print(f"Done. Summary saved to {summary_path}")