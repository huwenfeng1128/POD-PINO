# -*- coding: utf-8 -*-
"""
Physics-Informed POD-DeepONet 参数辨识程序：Surrogate Error 相关性分析
=================================================================

功能：
1. 批量参数辨识；
2. 保存每个样本的对比图；
3. 保存每个样本的可复绘数据；
4. 统计 true 参数下 surrogate prediction MSE 与最终参数误差的相关性；
5. 保存全局散点图、回归线数据和相关性统计；
6. 保留原 Surrogate_error 实验数据，只把模型换成 physics-informed POD-DeepONet。

说明：
- 使用代码一的 physics-informed 模型加载方式：model_{RUN_ID}.pth + scalers_{RUN_ID}.pth；
- 不再使用 pod_params_{RUN_ID}.pth，POD basis / grid 信息统一从 scalers 文件读取；
- 默认使用代码一的“加权归一化 MSE”作为辨识目标函数；
- 如需退回原始加权 MSE，把 USE_NORMALIZED_OBJECTIVE 改为 False。
"""

import os
import math
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from scipy.stats import pearsonr, linregress

# --- PyTorch Imports for DeepONet ---
import torch
import torch.nn as nn


# =========================
# Configuration
# =========================
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\SIlubangxing\surrogate_error\result_2'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 使用代码一的 physics-informed 训练输出
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# DeepONet Configuration (Must match training script)
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Optimization Settings
# 如果你论文文字里强调 L-BFGS-B，可把下面 method 改成 'L-BFGS-B'
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {
    'maxiter': 3000,
    'disp': True,
    'adaptive': True,
    'xatol': 1e-8,
    'fatol': 1e-8
}
D_NEGATIVE_PENALTY_FACTOR = 1e5

OBJECTIVE_LOG_FREQUENCY = 20

# True  = 使用代码一的 D1/D2 加权归一化 MSE；
# False = 使用原脚本的原始加权 MSE。
USE_NORMALIZED_OBJECTIVE = False


# =========================
# Model Definitions
# =========================
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
        self.eval()
        with torch.no_grad():
            y_pred_scaled = self.forward(branch_x)

            if not torch.is_tensor(y_mean_scaler):
                y_mean_scaler = torch.tensor(y_mean_scaler, dtype=DTYPE, device=branch_x.device)
            if not torch.is_tensor(y_std_scaler):
                y_std_scaler = torch.tensor(y_std_scaler, dtype=DTYPE, device=branch_x.device)

            return y_pred_scaled * y_std_scaler + y_mean_scaler


# =========================
# Helper Functions
# =========================
def to_numpy(x):
    """将 tensor / list / ndarray 统一转成 numpy.ndarray。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def manual_scaler_transform(data, mean, std):
    """Applies pre-computed scaling."""
    if isinstance(data, torch.Tensor):
        device = data.device
        data_t = data
    else:
        device = DEVICE
        data_t = torch.tensor(data, dtype=DTYPE, device=device)

    mean_t = torch.tensor(to_numpy(mean), dtype=DTYPE, device=device)
    std_t = torch.tensor(to_numpy(std), dtype=DTYPE, device=device)
    std_t = torch.where(std_t < 1e-10, torch.ones_like(std_t), std_t)
    return (data_t - mean_t) / std_t


def parse_params_from_filename(filepath):
    """
    从文件名中提取参数。

    支持：
    1. (0.1,0.2,0.3).csv
    2. data_nu_neg18_388_kappa_0_648_d_10_613.csv
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    # 旧格式: (nu,kappa,D).csv
    try:
        if stem.startswith('(') and stem.endswith(')'):
            values = stem[1:-1].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except Exception:
        pass

    # 新格式: data_nu_neg18_388_kappa_0_648_d_10_613.csv
    pattern = r'(?:data_)?nu_(neg)?(\d+)_(\d+)_kappa_(neg)?(\d+)_(\d+)_d_(neg)?(\d+)_(\d+)'
    m = re.match(pattern, stem)
    if m:
        nu_sign, nu_i, nu_f, k_sign, k_i, k_f, d_sign, d_i, d_f = m.groups()
        nu = float(f"{nu_i}.{nu_f}")
        kappa = float(f"{k_i}.{k_f}")
        d_diffusion = float(f"{d_i}.{d_f}")

        if nu_sign == 'neg':
            nu = -nu
        if k_sign == 'neg':
            kappa = -kappa
        if d_sign == 'neg':
            d_diffusion = -d_diffusion
        return nu, kappa, d_diffusion

    return None, None, None


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """Theoretical D1 coefficient."""
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8.0) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """Theoretical D2 coefficient."""
    return np.full_like(np.asarray(A), d_diffusion, dtype=float)


def compute_deeponet_km_coefficients(params, A_selected, tau_indices_map, model, scalers,
                                     unified_a_grid, unified_tau_grid):
    """Compute KM coefficients using pre-trained DeepONet."""
    nu, kappa, d_diffusion = params

    if any(np.isnan(p) or np.isinf(p) for p in params):
        return None

    try:
        model.eval()
        with torch.no_grad():
            # Prepare branch input
            branch_input_np = np.array([[nu, kappa, d_diffusion]], dtype=np.float64)
            branch_input_scaled_t = manual_scaler_transform(
                branch_input_np, scalers['branch_mean'], scalers['branch_std']
            )

            # Get prediction in physical space
            y_pred_t = model.predict(
                branch_input_scaled_t,
                scalers['y_mean_scaler'],
                scalers['y_std_scaler']
            )
            y_pred_np = y_pred_t.detach().cpu().numpy().flatten()

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
                interp_d1 = interp1d(
                    unified_a_grid, D1_pred_slice,
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                interp_d2 = interp1d(
                    unified_a_grid, D2_pred_slice,
                    kind='linear', bounds_error=False, fill_value=np.nan
                )

                results_for_tau[tau_idx_km] = {
                    'A': np.asarray(A_selected, dtype=float),
                    'D1': interp_d1(A_selected),
                    'D2': interp_d2(A_selected)
                }

            return results_for_tau

    except Exception as e:
        print(f"  DeepONet prediction error for params {params}: {e}")
        traceback.print_exc()
        return None


def compute_surrogate_prediction_mse(nu_true, kappa_true, d_true,
                                     data_km, A_selected, tau_indices_map,
                                     model, scalers, unified_a_grid, unified_tau_grid,
                                     p_a_weights_map=None):
    """
    Compute surrogate-model prediction MSE at the TRUE parameters:
    compare DeepONet-predicted KM field with the KM data field.

    USE_NORMALIZED_OBJECTIVE=True 时，和辨识目标函数一致，
    分别用当前 tau 下 KM 数据的 D1/D2 标准差归一化。
    """
    true_params = [nu_true, kappa_true, d_true]

    pred_results = compute_deeponet_km_coefficients(
        true_params, A_selected, tau_indices_map, model, scalers,
        unified_a_grid, unified_tau_grid
    )

    if pred_results is None:
        return np.nan

    total_weighted_sq_error = 0.0
    num_compared_points = 0

    for tau_idx in tau_indices_map.keys():
        if tau_idx not in data_km or tau_idx not in pred_results:
            continue

        D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
        D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)
        D1_pred = np.asarray(pred_results[tau_idx]['D1'], dtype=float)
        D2_pred = np.asarray(pred_results[tau_idx]['D2'], dtype=float)

        valid_mask = (
            ~np.isnan(D1_data) & ~np.isnan(D1_pred) &
            ~np.isnan(D2_data) & ~np.isnan(D2_pred)
        )

        if not np.any(valid_mask):
            continue

        A_current = np.asarray(data_km[tau_idx]['A'], dtype=float)[valid_mask]
        if p_a_weights_map is None:
            weights = np.ones_like(A_current, dtype=float)
        else:
            weights = np.array([p_a_weights_map.get(a, 1.0) for a in A_current], dtype=float)

        D1_data_valid = D1_data[valid_mask]
        D2_data_valid = D2_data[valid_mask]
        D1_pred_valid = D1_pred[valid_mask]
        D2_pred_valid = D2_pred[valid_mask]

        if USE_NORMALIZED_OBJECTIVE:
            D1_scale = np.nanstd(D1_data_valid)
            D2_scale = np.nanstd(D2_data_valid)

            if np.isnan(D1_scale) or np.isinf(D1_scale) or D1_scale < 1e-12:
                D1_scale = 1.0
            if np.isnan(D2_scale) or np.isinf(D2_scale) or D2_scale < 1e-12:
                D2_scale = 1.0

            error_d1 = ((D1_data_valid - D1_pred_valid) / D1_scale) ** 2
            error_d2 = ((D2_data_valid - D2_pred_valid) / D2_scale) ** 2
        else:
            error_d1 = (D1_data_valid - D1_pred_valid) ** 2
            error_d2 = (D2_data_valid - D2_pred_valid) ** 2

        total_weighted_sq_error += np.sum(weights * (error_d1 + error_d2))
        num_compared_points += len(A_current)

    if num_compared_points <= 0:
        return np.nan

    surrogate_mse = total_weighted_sq_error / num_compared_points
    return float(surrogate_mse)


def objective_function(params, data_km, A_selected, tau_indices_map, model, scalers,
                       optimization_history_local, p_a_weights_map, unified_a_grid, unified_tau_grid):
    """Objective function for optimization."""
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    d_negative_penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion) if d_diffusion < 0 else 0.0

    deeponet_km_results = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices_map, model, scalers, unified_a_grid, unified_tau_grid
    )

    if deeponet_km_results is None:
        cost = 1e11 + d_negative_penalty
        mean_weighted_sq_error = np.nan
        num_compared_points = 0
    else:
        total_weighted_sq_error = 0.0
        num_compared_points = 0

        for tau_idx in tau_indices_map.keys():
            if tau_idx not in data_km or tau_idx not in deeponet_km_results:
                continue

            D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
            D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)
            D1_pred = np.asarray(deeponet_km_results[tau_idx]['D1'], dtype=float)
            D2_pred = np.asarray(deeponet_km_results[tau_idx]['D2'], dtype=float)

            valid_mask = (
                ~np.isnan(D1_data) & ~np.isnan(D1_pred) &
                ~np.isnan(D2_data) & ~np.isnan(D2_pred)
            )

            if not np.any(valid_mask):
                continue

            A_current = np.asarray(data_km[tau_idx]['A'], dtype=float)[valid_mask]
            weights = np.array([p_a_weights_map.get(a, 0.0) for a in A_current], dtype=float)

            D1_data_valid = D1_data[valid_mask]
            D2_data_valid = D2_data[valid_mask]
            D1_pred_valid = D1_pred[valid_mask]
            D2_pred_valid = D2_pred[valid_mask]

            if USE_NORMALIZED_OBJECTIVE:
                D1_scale = np.nanstd(D1_data_valid)
                D2_scale = np.nanstd(D2_data_valid)

                if np.isnan(D1_scale) or np.isinf(D1_scale) or D1_scale < 1e-12:
                    D1_scale = 1.0
                if np.isnan(D2_scale) or np.isinf(D2_scale) or D2_scale < 1e-12:
                    D2_scale = 1.0

                error_d1 = ((D1_data_valid - D1_pred_valid) / D1_scale) ** 2
                error_d2 = ((D2_data_valid - D2_pred_valid) / D2_scale) ** 2
            else:
                error_d1 = (D1_data_valid - D1_pred_valid) ** 2
                error_d2 = (D2_data_valid - D2_pred_valid) ** 2

            total_weighted_sq_error += np.sum(weights * (error_d1 + error_d2))
            num_compared_points += len(A_current)

        mean_weighted_sq_error = (
            total_weighted_sq_error / num_compared_points
            if num_compared_points > 0 else 1e10
        )
        cost = mean_weighted_sq_error + d_negative_penalty

    optimization_history_local.append({
        'iteration': current_iter,
        'nu': nu,
        'kappa': kappa,
        'd_diffusion': d_diffusion,
        'mse': mean_weighted_sq_error,
        'penalty': d_negative_penalty,
        'num_compared_points': num_compared_points,
        'total_cost': cost
    })

    if current_iter == 1 or current_iter % OBJECTIVE_LOG_FREQUENCY == 0:
        print(f"[OPT] Eval={current_iter} | nu={nu:.6f}, kappa={kappa:.6f}, D={d_diffusion:.6f}, "
              f"mse={mean_weighted_sq_error:.6e}, penalty={d_negative_penalty:.3e}, total_cost={cost:.6e}")

    if np.isnan(cost) or np.isinf(cost):
        return 1e12
    return cost


def save_comparison_plot_data(data_km, deeponet_results, theo_results, output_dir):
    """
    Save all comparison-plot underlying data for secondary plotting.
    """
    rows = []

    tau_keys = sorted(set(list(data_km.keys()) +
                          list(deeponet_results.keys()) if deeponet_results else list(data_km.keys()) +
                          list(theo_results.keys()) if theo_results else list(data_km.keys())))

    # 更稳妥处理
    tau_keys = sorted(set(list(data_km.keys()) +
                          (list(deeponet_results.keys()) if deeponet_results else []) +
                          (list(theo_results.keys()) if theo_results else [])))

    for tau_idx in tau_keys:
        tau_sec = None
        A_union = []

        if tau_idx in data_km:
            tau_sec = data_km[tau_idx].get('tau_sec', tau_sec)
            A_union.extend(list(np.asarray(data_km[tau_idx]['A']).tolist()))
        if deeponet_results and tau_idx in deeponet_results:
            A_union.extend(list(np.asarray(deeponet_results[tau_idx]['A']).tolist()))
        if theo_results and tau_idx in theo_results:
            A_union.extend(list(np.asarray(theo_results[tau_idx]['A']).tolist()))

        if len(A_union) == 0:
            continue

        A_union = np.unique(np.asarray(A_union, dtype=float))

        data_A = np.asarray(data_km[tau_idx]['A'], dtype=float) if tau_idx in data_km else np.array([])
        data_D1 = np.asarray(data_km[tau_idx]['D1'], dtype=float) if tau_idx in data_km else np.array([])
        data_D2 = np.asarray(data_km[tau_idx]['D2'], dtype=float) if tau_idx in data_km else np.array([])

        opt_A = np.asarray(deeponet_results[tau_idx]['A'], dtype=float) if (deeponet_results and tau_idx in deeponet_results) else np.array([])
        opt_D1 = np.asarray(deeponet_results[tau_idx]['D1'], dtype=float) if (deeponet_results and tau_idx in deeponet_results) else np.array([])
        opt_D2 = np.asarray(deeponet_results[tau_idx]['D2'], dtype=float) if (deeponet_results and tau_idx in deeponet_results) else np.array([])

        theo_A = np.asarray(theo_results[tau_idx]['A'], dtype=float) if (theo_results and tau_idx in theo_results) else np.array([])
        theo_D1 = np.asarray(theo_results[tau_idx]['D1'], dtype=float) if (theo_results and tau_idx in theo_results) else np.array([])
        theo_D2 = np.asarray(theo_results[tau_idx]['D2'], dtype=float) if (theo_results and tau_idx in theo_results) else np.array([])

        # 建映射，方便按A写表
        data_map_D1 = {a: v for a, v in zip(data_A, data_D1)}
        data_map_D2 = {a: v for a, v in zip(data_A, data_D2)}
        opt_map_D1 = {a: v for a, v in zip(opt_A, opt_D1)}
        opt_map_D2 = {a: v for a, v in zip(opt_A, opt_D2)}
        theo_map_D1 = {a: v for a, v in zip(theo_A, theo_D1)}
        theo_map_D2 = {a: v for a, v in zip(theo_A, theo_D2)}

        for a in A_union:
            rows.append({
                'tau_index': tau_idx,
                'tau_sec': tau_sec,
                'A': a,
                'D1_data': data_map_D1.get(a, np.nan),
                'D2_data': data_map_D2.get(a, np.nan),
                'D1_deeponet_opt': opt_map_D1.get(a, np.nan),
                'D2_deeponet_opt': opt_map_D2.get(a, np.nan),
                'D1_theoretical_std': theo_map_D1.get(a, np.nan),
                'D2_theoretical_std': theo_map_D2.get(a, np.nan)
            })

    df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, 'comparison_plot_data.csv')
    df.to_csv(save_path, index=False)
    return save_path


def plot_comparison(data_km, deeponet_results, theo_results, A_selected, tau_indices_map,
                    nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std, output_dir):
    """Generate and save comparison plots."""
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Comparison: Optimized vs Standard vs Data\n'
        f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
        f'Std: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
        fontsize=12
    )

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
            ax_d1.plot(
                deeponet_results[tau_idx]['A'],
                deeponet_results[tau_idx]['D1'],
                'b-',
                label='DeepONet Opt',
                linewidth=2
            )

        if theo_results and tau_idx in theo_results:
            ax_d1.plot(
                theo_results[tau_idx]['A'],
                theo_results[tau_idx]['D1'],
                'r--',
                label='Theoretical Std',
                linewidth=2
            )

        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at τ={tau_sec:.4f}s')
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)

        # D2 plot
        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)

        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(
                deeponet_results[tau_idx]['A'],
                deeponet_results[tau_idx]['D2'],
                'b-',
                label='DeepONet Opt',
                linewidth=2
            )

        if theo_results and tau_idx in theo_results:
            ax_d2.plot(
                theo_results[tau_idx]['A'],
                theo_results[tau_idx]['D2'],
                'r--',
                label='Theoretical Std',
                linewidth=2
            )

        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at τ={tau_sec:.4f}s')
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def plot_surrogate_vs_param_error(summary_df, output_dir,
                                  x_col='surrogate_mse',
                                  y_col='param_rel_error_percent'):
    """
    Scatter plot:
      X-axis: surrogate prediction MSE (log scale)
      Y-axis: final parameter identification relative error (%)
    Also save plot data and regression data for secondary plotting.
    """
    plot_df = summary_df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    plot_df = plot_df[plot_df[x_col] > 0]

    scatter_data_path = os.path.join(output_dir, 'surrogate_vs_param_error_data.csv')
    regression_data_path = os.path.join(output_dir, 'surrogate_vs_param_error_regression.csv')
    fig_path = os.path.join(output_dir, 'surrogate_vs_param_error_scatter.png')

    if len(plot_df) < 3:
        print("Not enough valid samples to plot surrogate-vs-parameter-error correlation.")
        plot_df.to_csv(scatter_data_path, index=False)
        return None

    x = plot_df[x_col].values
    y = plot_df[y_col].values

    # Pearson correlation
    r_value, p_value = pearsonr(x, y)

    # Linear regression in original x-space
    reg = linregress(x, y)
    x_line = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    y_line = reg.slope * x_line + reg.intercept

    # Save scatter raw data
    scatter_cols = [
        'filename',
        'surrogate_mse',
        'param_rel_error_percent',
        'nu_standard', 'kappa_standard', 'D_standard',
        'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized',
        'nu_rel_error', 'kappa_rel_error', 'd_rel_error'
    ]
    scatter_cols = [c for c in scatter_cols if c in plot_df.columns]
    plot_df[scatter_cols].to_csv(scatter_data_path, index=False)

    # Save regression data
    reg_df = pd.DataFrame({
        'x_line': x_line,
        'y_line': y_line
    })
    reg_df.to_csv(regression_data_path, index=False)

    # Draw figure
    plt.figure(figsize=(8, 6))
    plt.scatter(
        x, y,
        color='royalblue',
        alpha=0.6,
        s=45,
        edgecolors='none',
        label='Samples'
    )
    plt.plot(
        x_line, y_line,
        'r--',
        linewidth=2,
        label=f'Trend line (slope={reg.slope:.3e})'
    )

    plt.xscale('log')
    plt.xlabel('Surrogate Model Prediction MSE', fontsize=12)
    plt.ylabel('Parameter Identification Relative Error (%)', fontsize=12)
    plt.title('Correlation between Surrogate Error and Parameter Identification Error', fontsize=13)
    plt.grid(True, which='both', alpha=0.3)

    text_str = (
        f'Pearson R = {r_value:.3f}\n'
        f'p = {p_value:.3e}\n'
        f'N = {len(plot_df)}'
    )
    plt.text(
        0.05, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Correlation scatter plot saved to: {fig_path}")
    print(f"Scatter raw data saved to: {scatter_data_path}")
    print(f"Regression data saved to: {regression_data_path}")
    print(f"Pearson R = {r_value:.4f}, p = {p_value:.4e}, N = {len(plot_df)}")

    return {
        'figure_path': fig_path,
        'scatter_data_path': scatter_data_path,
        'regression_data_path': regression_data_path,
        'pearson_r': r_value,
        'pearson_p': p_value,
        'n_samples': len(plot_df)
    }


def process_km_file(km_file_path, deeponet_model, scalers, unified_a_grid, unified_tau_grid,
                    base_output_dir, input_dir_sim_data):
    """Process a single KM file for parameter identification."""
    print(f"\n--- Processing: {os.path.basename(km_file_path)} ---")

    # Parse standard parameters
    nu_std, kappa_std, D_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        print(f"WARNING: Could not parse params from '{os.path.basename(km_file_path)}'. Skipping.")
        return None

    # Create output directory
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0]
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
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
        print(f"Error loading KM data: {e}")
        return None

    # Load simulation data for weights
    p_a_weights_map = {a: 1.0 / len(A_selected) for a in A_selected}
    sim_file_path = os.path.join(input_dir_sim_data, os.path.basename(km_file_path))

    if os.path.exists(sim_file_path):
        try:
            df_sim = pd.read_csv(sim_file_path)
            if 'Eta' in df_sim.columns:
                sim_envelope = np.abs(hilbert(df_sim['Eta'].values))
            elif 'Envelope' in df_sim.columns:
                sim_envelope = df_sim['Envelope'].values
            else:
                sim_envelope = None

            if sim_envelope is not None and len(A_selected) > 0:
                bin_width = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0
                bin_edges = np.concatenate([
                    [A_selected[0] - bin_width / 2.0],
                    np.array(A_selected[:-1]) + bin_width / 2.0,
                    [A_selected[-1] + bin_width / 2.0]
                ])

                hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
                if np.sum(hist) > 0:
                    p_a_weights_map = {A_selected[i]: hist[i] for i in range(len(A_selected))}
        except Exception as e:
            print(f"Warning: Could not process sim file for weights: {e}")

    # Initialize optimization
    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    params_0 = [0.1, 0.1, max(0.01, initial_d if not np.isnan(initial_d) else 0.01)]
    optimization_history = []

    print(f"  Initial guess: ν={params_0[0]:.4f}, κ={params_0[1]:.4f}, D={params_0[2]:.4f}")

    # Run optimization
    start_time = time.time()
    result = minimize(
        objective_function,
        params_0,
        args=(
            finite_time_km, A_selected, tau_indices_map, deeponet_model, scalers,
            optimization_history, p_a_weights_map, unified_a_grid, unified_tau_grid
        ),
        method=OPTIMIZER_METHOD,
        options=OPTIMIZER_OPTIONS
    )
    elapsed_time = time.time() - start_time

    # Extract optimized parameters
    if result.success:
        nu_opt, kappa_opt, d_opt = result.x
    else:
        print(f"  Optimization did not converge. Using best from history.")
        if len(optimization_history) > 0:
            best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
            nu_opt = optimization_history[best_idx]['nu']
            kappa_opt = optimization_history[best_idx]['kappa']
            d_opt = optimization_history[best_idx]['d_diffusion']
        else:
            nu_opt, kappa_opt, d_opt = params_0

    d_opt = max(0.0, d_opt)  # Ensure non-negative

    print(f"  Optimization finished in {elapsed_time:.2f}s")
    print(f"  Optimized: ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}")
    print(f"  Standard:  ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f}")

    # --- Surrogate prediction error at TRUE parameters ---
    surrogate_mse = compute_surrogate_prediction_mse(
        nu_std, kappa_std, D_std,
        finite_time_km, A_selected, tau_indices_map,
        deeponet_model, scalers, unified_a_grid, unified_tau_grid,
        p_a_weights_map=p_a_weights_map
    )

    # --- Error Calculation ---
    nu_error = abs(nu_opt - nu_std)
    kappa_error = abs(kappa_opt - kappa_std)
    d_error = abs(d_opt - D_std)

    nu_rel_error = nu_error / abs(nu_std) if abs(nu_std) > 1e-15 else 0.0
    kappa_rel_error = kappa_error / abs(kappa_std) if abs(kappa_std) > 1e-15 else 0.0
    d_rel_error = d_error / abs(D_std) if abs(D_std) > 1e-15 else 0.0

    # Final parameter identification relative error (%)
    param_rel_error_percent = (nu_rel_error + kappa_rel_error + d_rel_error) / 3.0 * 100.0

    # Compute final predictions
    deeponet_opt_results = compute_deeponet_km_coefficients(
        [nu_opt, kappa_opt, d_opt], A_selected, tau_indices_map,
        deeponet_model, scalers, unified_a_grid, unified_tau_grid
    )

    # Compute theoretical predictions with standard params
    theo_std_results = {}
    for tau_idx in tau_indices_map.keys():
        theo_std_results[tau_idx] = {
            'A': np.asarray(A_selected, dtype=float),
            'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
            'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
        }

    # Generate plots
    comparison_fig_path = plot_comparison(
        finite_time_km, deeponet_opt_results, theo_std_results,
        A_selected, tau_indices_map, nu_opt, kappa_opt, d_opt,
        nu_std, kappa_std, D_std, output_dir
    )

    # Save optimization history
    optimization_history_path = None
    if optimization_history:
        history_df = pd.DataFrame(optimization_history)
        optimization_history_path = os.path.join(output_dir, 'optimization_history.csv')
        history_df.to_csv(optimization_history_path, index=False)

    # Save underlying comparison data for secondary plotting
    comparison_data_path = save_comparison_plot_data(
        finite_time_km, deeponet_opt_results, theo_std_results, output_dir
    )

    # Save sample metrics
    sample_metrics = pd.DataFrame([{
        'filename': os.path.basename(km_file_path),

        'nu_standard': nu_std,
        'kappa_standard': kappa_std,
        'D_standard': D_std,

        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,

        'nu_error': nu_error,
        'kappa_error': kappa_error,
        'd_error': d_error,

        'nu_rel_error': nu_rel_error,
        'kappa_rel_error': kappa_rel_error,
        'd_rel_error': d_rel_error,

        'param_rel_error_percent': param_rel_error_percent,
        'surrogate_mse': surrogate_mse,

        'optimization_time': elapsed_time,
        'optimization_success': bool(result.success),
        'normalized_objective': int(USE_NORMALIZED_OBJECTIVE)
    }])
    sample_metrics_path = os.path.join(output_dir, 'sample_metrics.csv')
    sample_metrics.to_csv(sample_metrics_path, index=False)

    return {
        'filename': os.path.basename(km_file_path),

        'nu_standard': nu_std,
        'kappa_standard': kappa_std,
        'D_standard': D_std,

        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,

        'nu_error': nu_error,
        'kappa_error': kappa_error,
        'd_error': d_error,

        'nu_rel_error': nu_rel_error,
        'kappa_rel_error': kappa_rel_error,
        'd_rel_error': d_rel_error,

        'param_rel_error_percent': param_rel_error_percent,
        'surrogate_mse': surrogate_mse,

        'optimization_time': elapsed_time,
        'optimization_success': bool(result.success),
        'normalized_objective': int(USE_NORMALIZED_OBJECTIVE),

        'comparison_fig_path': comparison_fig_path,
        'comparison_data_path': comparison_data_path,
        'sample_metrics_path': sample_metrics_path,
        'optimization_history_path': optimization_history_path
    }


# =========================
# Main Script
# =========================
if __name__ == "__main__":
    print("\n=== Loading Pre-trained Physics-Informed POD-DeepONet Model ===")

    try:
        # ---------------------------------------------------------------------
        # 1) 加载统一保存的 scaler / POD / grid 信息
        # ---------------------------------------------------------------------
        print("Loading scaler payload...")
        scaler_payload = torch.load(DEEPONET_SCALER_PATH, map_location=DEVICE)

        required_keys = [
            'branch_mean', 'branch_std',
            'y_mean_scaler', 'y_std_scaler',
            'y_mean_pod_scaled', 'pod_basis',
            'actual_num_modes',
            'unified_a_grid', 'unified_tau_grid'
        ]
        missing = [k for k in required_keys if k not in scaler_payload]
        if missing:
            raise KeyError(f"scaler payload missing keys: {missing}")

        scalers = {
            'branch_mean': to_numpy(scaler_payload['branch_mean']),
            'branch_std': to_numpy(scaler_payload['branch_std']),
            'y_mean_scaler': to_numpy(scaler_payload['y_mean_scaler']),
            'y_std_scaler': to_numpy(scaler_payload['y_std_scaler'])
        }

        pod_basis = to_numpy(scaler_payload['pod_basis'])
        y_mean_pod_scaled = to_numpy(scaler_payload['y_mean_pod_scaled'])
        num_pod_modes = int(scaler_payload['actual_num_modes'])
        unified_a_grid = to_numpy(scaler_payload['unified_a_grid']).astype(np.float64)
        unified_tau_grid = to_numpy(scaler_payload['unified_tau_grid']).astype(np.float64)

        print(f"  POD modes: {num_pod_modes}")
        print(f"  A-grid points: {len(unified_a_grid)}")
        print(f"  Tau-grid points: {len(unified_tau_grid)}")

        # ---------------------------------------------------------------------
        # 2) 构建模型
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 3) 加载模型权重
        # ---------------------------------------------------------------------
        print("Loading model weights...")
        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        print("✓ Physics-informed model loaded successfully!\n")

    except Exception as e:
        print("CRITICAL ERROR: Failed to load DeepONet model.")
        print(f"Error: {e}")
        traceback.print_exc()
        raise

    # Batch processing
    print("=== Starting Batch Processing ===")
    all_results = []
    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])

    print(f"Found {len(km_files)} KM files to process\n")

    for filename in tqdm(km_files, desc="Processing KM Files"):
        km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
        result = process_km_file(
            km_file_path, deeponet_model, scalers,
            unified_a_grid, unified_tau_grid,
            BASE_OUTPUT_DIR, INPUT_DIR_SIM_DATA
        )
        if result is not None:
            all_results.append(result)

    # Save summary and global plot
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, "batch_summary_physics_informed.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n=== Summary Statistics ===")
        print(f"Total files processed: {len(all_results)}")

        print(f"\nMean Absolute Errors:")
        print(f"  ν error:  {summary_df['nu_error'].mean():.6f}")
        print(f"  κ error:  {summary_df['kappa_error'].mean():.6f}")
        print(f"  D error:  {summary_df['d_error'].mean():.6f}")

        print(f"\nMean Relative Errors:")
        print(f"  ν rel error:  {summary_df['nu_rel_error'].mean():.2%}")
        print(f"  κ rel error:  {summary_df['kappa_rel_error'].mean():.2%}")
        print(f"  D rel error:  {summary_df['d_rel_error'].mean():.2%}")

        if 'param_rel_error_percent' in summary_df.columns:
            print(f"\nMean Final Parameter Identification Relative Error:")
            print(f"  ΔP mean: {summary_df['param_rel_error_percent'].mean():.2f}%")
            print(f"  ΔP median: {summary_df['param_rel_error_percent'].median():.2f}%")

        if 'surrogate_mse' in summary_df.columns:
            valid_surrogate = summary_df['surrogate_mse'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_surrogate) > 0:
                print(f"\nSurrogate Prediction MSE:")
                print(f"  mean: {valid_surrogate.mean():.6e}")
                print(f"  median: {valid_surrogate.median():.6e}")

        print(f"\nSummary saved to: {summary_path}")

        # Global scatter plot
        corr_info = plot_surrogate_vs_param_error(summary_df, BASE_OUTPUT_DIR)

        # Save correlation stats
        if corr_info is not None:
            corr_stats_df = pd.DataFrame([{
                'pearson_r': corr_info['pearson_r'],
                'pearson_p': corr_info['pearson_p'],
                'n_samples': corr_info['n_samples'],
                'figure_path': corr_info['figure_path'],
                'scatter_data_path': corr_info['scatter_data_path'],
                'regression_data_path': corr_info['regression_data_path']
            }])
            corr_stats_path = os.path.join(BASE_OUTPUT_DIR, 'surrogate_vs_param_error_stats.csv')
            corr_stats_df.to_csv(corr_stats_path, index=False)
            print(f"Correlation stats saved to: {corr_stats_path}")

    print("\n=== Batch processing completed ===")