# -*- coding: utf-8 -*-
r"""
实验脚本 3：不同滤波带宽下的参数辨识（Physics-Informed POD-DeepONet 版）
============================================================================

改动目标：
--------
1. 保留原“不同带宽”实验的数据：
       D:\PINN\zenodo\POD_deeponet\Revise_experiment\bandwith_experiment\km_data
2. 将原脚本中的 v3 POD-DeepONet 模型替换为代码一中的 physics-informed POD-DeepONet 模型；
3. 使用 physics-informed 训练脚本的统一 scaler/POD 保存格式：
       scalers_{RUN_ID}.pth
   不再加载 pod_params_{RUN_ID}.pth；
4. 对不同 bandwidth 下的 KM 数据逐个做参数辨识；
5. 保存：
   - 单样本 result_summary.csv
   - optimization_history.csv
   - optimization_convergence.png
   - comparison_plot.png
   - curve_data_for_plot.csv
   - 总汇总 identification_summary_physics_informed.csv
   - 按 group_id + bandwidth 汇总 identification_group_bandwidth_stats_physics_informed.csv
   - 按 bandwidth 汇总 summary_by_bandwidth_physics_informed.csv
   - plot-ready 长表 error_metrics_long_format_physics_informed.csv

说明：
----
- 默认目标函数使用代码一的“归一化 MSE”形式，使 D1/D2 尺度更平衡；
- 如果你想让目标函数完全等同原带宽脚本中的原始 MSE，把 USE_NORMALIZED_OBJECTIVE 改为 False。
"""

import os
import re
import math
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import interp1d

import torch
import torch.nn as nn


# =============================================================================
# 0. 路径与全局配置
# =============================================================================

# --- 保持原“不同带宽”实验的数据目录不变 ---
OUTPUT_ROOT = r"D:\PINN\zenodo\POD_deeponet\Revise_experiment\bandwith_experiment"
INPUT_DIR_KM_DATA = os.path.join(OUTPUT_ROOT, "km_data")

# --- physics-informed 模型在不同带宽数据上的参数辨识输出目录 ---
BASE_OUTPUT_DIR = os.path.join('D:\\PINN\\zenodo\\POD_deeponet\\PI-POD-DeepONet\\result\\lubangxing\\shujulubangxing\\bandwith\\result')
OUTPUT_DIR_IDENT = os.path.join(BASE_OUTPUT_DIR, "single_case_results")
OUTPUT_DIR_IDENT_PLOTS = os.path.join(BASE_OUTPUT_DIR, "plots")
OUTPUT_DIR_IDENT_PLOT_DATA = os.path.join(BASE_OUTPUT_DIR, "plot_data")

for _dir in [BASE_OUTPUT_DIR, OUTPUT_DIR_IDENT, OUTPUT_DIR_IDENT_PLOTS, OUTPUT_DIR_IDENT_PLOT_DATA]:
    os.makedirs(_dir, exist_ok=True)

# --- 使用代码一的 physics-informed 训练结果 ---
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# --- 模型结构参数：必须与你的 physics-informed 训练代码一致 ---
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# --- 优化参数 ---
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

# True  = 使用代码一的归一化 MSE；
# False = 使用原带宽脚本的原始 MSE。
USE_NORMALIZED_OBJECTIVE = True


# =============================================================================
# 1. 日志函数
# =============================================================================

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 2. 文件名解析
# =============================================================================

def sanitize_stem(stem):
    """
    将文件 stem 转为更安全的输出文件/目录名。
    """
    stem = stem.replace('.', 'p').replace(',', '_')
    stem = stem.replace('(', '').replace(')', '')
    stem = re.sub(r'[^0-9a-zA-Z_\-]+', '_', stem)
    return stem.strip('_')


def parse_filename_info(filename):
    """
    解析实验 2 生成的 KM 文件名。

    预期格式：
        group1_bw20_(nu,kappa,D).csv
    例如：
        group1_bw10_(0.100,0.200,0.030).csv
        group2_bw50_(-18.388,0.648,10.613).csv
    """
    pattern = (
        r"group(\d+)_bw(\d+)_"
        r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\.csv$"
    )
    m = re.match(pattern, filename)
    if m:
        return {
            "group_id": int(m.group(1)),
            "bandwidth": int(m.group(2)),
            "nu": float(m.group(3)),
            "kappa": float(m.group(4)),
            "D": float(m.group(5)),
        }
    return None


# =============================================================================
# 3. 网络定义：代码一 physics-informed POD-DeepONet 结构
# =============================================================================

class MLP(nn.Module):
    """
    与 physics-informed 训练脚本一致的 MLP：
    Linear -> [GELU + LayerNorm + Dropout + Linear] * N
           -> GELU + LayerNorm + Dropout + Linear
    """
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
    """
    与代码一一致的 POD-DeepONet。
    """
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers,
                 num_pod_modes, pod_basis, y_mean_pod_scaled, dropout_rate):
        super().__init__()

        self.branch = MLP(
            input_dim=branch_input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            output_dim=num_pod_modes,
            dropout_rate=dropout_rate
        )

        self.pod_basis = nn.Parameter(
            torch.tensor(pod_basis, dtype=DTYPE),
            requires_grad=False
        )
        self.y_mean_pod_scaled = nn.Parameter(
            torch.tensor(y_mean_pod_scaled, dtype=DTYPE),
            requires_grad=False
        )

    def forward(self, branch_x):
        """
        输出标准化空间中的预测。
        """
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
        """
        输出物理空间中的预测。
        """
        self.eval()
        with torch.no_grad():
            y_pred_scaled = self.forward(branch_x)

            if not torch.is_tensor(y_mean_scaler):
                y_mean_scaler = torch.tensor(y_mean_scaler, dtype=DTYPE, device=branch_x.device)
            else:
                y_mean_scaler = y_mean_scaler.to(dtype=DTYPE, device=branch_x.device)

            if not torch.is_tensor(y_std_scaler):
                y_std_scaler = torch.tensor(y_std_scaler, dtype=DTYPE, device=branch_x.device)
            else:
                y_std_scaler = y_std_scaler.to(dtype=DTYPE, device=branch_x.device)

            y_pred = y_pred_scaled * y_std_scaler + y_mean_scaler
            return y_pred


# =============================================================================
# 4. 工具函数
# =============================================================================

def safe_torch_load(path, map_location):
    """
    兼容不同 PyTorch 版本。
    新版 torch.load 在部分环境中可能需要显式 weights_only=False。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_numpy(x):
    """
    将 tensor / list / ndarray 统一转成 numpy.ndarray。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def manual_scaler_transform(data, mean, std):
    """
    使用训练阶段保存好的 mean/std 对输入做标准化。
    """
    if isinstance(data, torch.Tensor):
        device = data.device
        data_t = data.to(dtype=DTYPE)
    else:
        device = DEVICE
        data_t = torch.tensor(data, dtype=DTYPE, device=device)

    mean_t = torch.tensor(to_numpy(mean), dtype=DTYPE, device=device)
    std_t = torch.tensor(to_numpy(std), dtype=DTYPE, device=device)
    std_t = torch.where(std_t < 1e-10, torch.ones_like(std_t), std_t)

    return (data_t - mean_t) / std_t


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """
    理论 D1：
        D1 = nu*A - (kappa/8)*A^3 + D/A
    """
    A = np.asarray(A, dtype=float)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_mask] = d_diffusion / A[non_zero_mask]
    return (nu * A) - ((kappa / 8.0) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """
    理论 D2：
        D2 = D
    """
    return np.full_like(np.asarray(A, dtype=float), d_diffusion)


# =============================================================================
# 5. DeepONet 预测
# =============================================================================

def compute_deeponet_km_coefficients(params,
                                     A_selected,
                                     tau_indices_map,
                                     model,
                                     scalers,
                                     unified_a_grid,
                                     unified_tau_grid):
    """
    给定参数 [nu, kappa, d_diffusion]，用 physics-informed POD-DeepONet 预测 D1/D2 场，
    再按 KM 文件中的 tau 和 A 网格抽取/插值。
    """
    nu, kappa, d_diffusion = params

    if any(np.isnan(p) or np.isinf(p) for p in params):
        return None

    try:
        model.eval()
        with torch.no_grad():
            branch_input_np = np.array([[nu, kappa, d_diffusion]], dtype=np.float64)
            branch_input_scaled_t = manual_scaler_transform(
                branch_input_np,
                scalers['branch_mean'],
                scalers['branch_std']
            )

            y_pred_t = model.predict(
                branch_input_scaled_t,
                scalers['y_mean_scaler'],
                scalers['y_std_scaler']
            )
            y_pred_np = y_pred_t.cpu().numpy().flatten()

            field_len_per_type = len(y_pred_np) // 2
            n_tau = len(unified_tau_grid)
            n_a = len(unified_a_grid)

            D1_field_pred = y_pred_np[:field_len_per_type].reshape(n_tau, n_a)
            D2_field_pred = y_pred_np[field_len_per_type:].reshape(n_tau, n_a)

            results_for_tau = {}

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                closest_tau_grid_idx = int(np.argmin(np.abs(unified_tau_grid - tau_sec_km)))

                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                interp_d1 = interp1d(
                    unified_a_grid, D1_pred_slice,
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                interp_d2 = interp1d(
                    unified_a_grid, D2_pred_slice,
                    kind='linear', bounds_error=False, fill_value=np.nan
                )

                A_selected_np = np.asarray(A_selected, dtype=float)
                results_for_tau[int(tau_idx_km)] = {
                    'A': A_selected_np,
                    'D1': interp_d1(A_selected_np),
                    'D2': interp_d2(A_selected_np)
                }

            return results_for_tau

    except Exception as e:
        log(f"[PRED-ERROR] params={params} -> DeepONet 预测失败: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# 6. 目标函数
# =============================================================================

def objective_function(params,
                       data_km,
                       A_selected,
                       tau_indices_map,
                       model,
                       scalers,
                       optimization_history_local,
                       unified_a_grid,
                       unified_tau_grid):
    """
    优化目标函数。

    默认：归一化 MSE
        ((D1_data - D1_pred) / std(D1_data))^2
      + ((D2_data - D2_pred) / std(D2_data))^2

    如果 USE_NORMALIZED_OBJECTIVE=False，则退回原脚本的原始 MSE。
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion) if d_diffusion < 0 else 0.0

    deeponet_km_results = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices_map, model, scalers, unified_a_grid, unified_tau_grid
    )

    if deeponet_km_results is None:
        mse_val = np.nan
        total_cost = 1e11 + penalty
        num_compared_points = 0
    else:
        total_sq_error = 0.0
        num_compared_points = 0

        for tau_idx in tau_indices_map.keys():
            tau_idx = int(tau_idx)
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

            total_sq_error += np.sum(error_d1 + error_d2)
            num_compared_points += int(np.sum(valid_mask))

        mse_val = total_sq_error / num_compared_points if num_compared_points > 0 else 1e10
        total_cost = mse_val + penalty

    optimization_history_local.append({
        'iteration': current_iter,
        'nu': nu,
        'kappa': kappa,
        'd_diffusion': d_diffusion,
        'mse': mse_val,
        'penalty': penalty,
        'num_compared_points': num_compared_points,
        'total_cost': total_cost
    })

    if current_iter == 1 or current_iter % OBJECTIVE_LOG_FREQUENCY == 0:
        log(
            f"[OPT] Eval={current_iter} | "
            f"nu={nu:.6f}, kappa={kappa:.6f}, D={d_diffusion:.6f}, "
            f"mse={mse_val:.6e}, penalty={penalty:.3e}, total_cost={total_cost:.6e}"
        )

    if np.isnan(total_cost) or np.isinf(total_cost):
        return 1e12
    return total_cost


# =============================================================================
# 7. 保存收敛图和数据
# =============================================================================

def save_convergence_plot_and_data(history_df, output_dir):
    plot_path = os.path.join(output_dir, "optimization_convergence.png")
    data_path = os.path.join(output_dir, "optimization_convergence_data.csv")

    if history_df.empty:
        return plot_path, data_path

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["iteration"], history_df["total_cost"], label="Total Cost")
    plt.plot(history_df["iteration"], history_df["mse"], label="MSE", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Convergence")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    history_df.to_csv(data_path, index=False)
    return plot_path, data_path


# =============================================================================
# 8. 保存对比图和曲线数据
# =============================================================================

def save_comparison_plot_and_data(data_km,
                                  deeponet_results,
                                  theo_results,
                                  A_selected,
                                  tau_indices_map,
                                  nu_opt,
                                  kappa_opt,
                                  d_opt,
                                  nu_std,
                                  kappa_std,
                                  D_std,
                                  output_dir,
                                  title_suffix=""):
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    curve_path = os.path.join(output_dir, 'curve_data_for_plot.csv')

    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Comparison: Optimized vs True vs Data {title_suffix}\n'
        f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
        f'True: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
        fontsize=12
    )

    all_rows = []

    for i, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        tau_idx = int(tau_idx)
        if tau_idx not in data_km:
            continue

        A_data = np.asarray(data_km[tau_idx]['A'], dtype=float)
        D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
        D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)

        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(A_selected, deeponet_results[tau_idx]['D1'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d1.plot(A_selected, theo_results[tau_idx]['D1'], 'r--', label='Theory True', linewidth=2)
        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at τ={tau_sec:.6f}s')
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)

        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(A_selected, deeponet_results[tau_idx]['D2'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d2.plot(A_selected, theo_results[tau_idx]['D2'], 'r--', label='Theory True', linewidth=2)
        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at τ={tau_sec:.6f}s')
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)

        d1_pred = (
            deeponet_results[tau_idx]['D1']
            if (deeponet_results and tau_idx in deeponet_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )
        d2_pred = (
            deeponet_results[tau_idx]['D2']
            if (deeponet_results and tau_idx in deeponet_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )
        d1_theory = (
            theo_results[tau_idx]['D1']
            if (theo_results and tau_idx in theo_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )
        d2_theory = (
            theo_results[tau_idx]['D2']
            if (theo_results and tau_idx in theo_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )

        for j in range(len(A_data)):
            all_rows.append({
                "tau_index": tau_idx,
                "tau_sec": tau_sec,
                "A": A_data[j],
                "D1_data": D1_data[j],
                "D2_data": D2_data[j],
                "D1_pred_opt": d1_pred[j] if j < len(d1_pred) else np.nan,
                "D2_pred_opt": d2_pred[j] if j < len(d2_pred) else np.nan,
                "D1_theory_true": d1_theory[j] if j < len(d1_theory) else np.nan,
                "D2_theory_true": d2_theory[j] if j < len(d2_theory) else np.nan
            })

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(plot_path, dpi=150)
    plt.close()

    pd.DataFrame(all_rows).to_csv(curve_path, index=False)
    return plot_path, curve_path


# =============================================================================
# 9. 处理单个 KM 文件
# =============================================================================

def process_km_file(km_file_path,
                    deeponet_model,
                    scalers,
                    unified_a_grid,
                    unified_tau_grid):
    original_file_name = os.path.basename(km_file_path)
    original_stem = os.path.splitext(original_file_name)[0]
    safe_stem = sanitize_stem(original_stem)
    info = parse_filename_info(original_file_name)

    log("=" * 90)
    log(f"[IDENT-START] 开始处理: {original_file_name}")

    if info is None:
        log(f"[IDENT-ERROR] {original_file_name} -> 文件名解析失败，跳过")
        return None

    nu_std = info["nu"]
    kappa_std = info["kappa"]
    D_std = info["D"]
    group_id = info["group_id"]
    bandwidth = info["bandwidth"]

    case_output_dir = os.path.join(OUTPUT_DIR_IDENT, f"group{group_id}_bw{bandwidth}", safe_stem)
    os.makedirs(case_output_dir, exist_ok=True)

    t0 = time.time()

    try:
        log(
            f"[META] group={group_id}, bandwidth={bandwidth}, "
            f"true=(nu={nu_std:.6f}, kappa={kappa_std:.6f}, D={D_std:.6f})"
        )

        # ---------------------------------------------------------------------
        # 读取 KM 数据
        # ---------------------------------------------------------------------
        log(f"[LOAD] 读取 KM 数据: {km_file_path}")
        data_km_df = pd.read_csv(km_file_path)

        required_cols = ['tau_index', 'tau_sec', 'A', 'D1_data', 'D2_data']
        missing = [c for c in required_cols if c not in data_km_df.columns]
        if missing:
            raise ValueError(f"KM file missing required columns: {missing}")

        finite_time_km = {
            int(tau_idx): {
                'A': group['A'].values.astype(float),
                'D1': group['D1_data'].values.astype(float),
                'D2': group['D2_data'].values.astype(float),
                'tau_sec': float(group['tau_sec'].iloc[0])
            }
            for tau_idx, group in data_km_df.groupby('tau_index')
        }

        A_selected = np.asarray(sorted(data_km_df['A'].unique()), dtype=float)
        tau_indices_map = {
            int(idx): float(ts)
            for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }

        log(f"[LOAD] tau 个数={len(tau_indices_map)}, A 点数={len(A_selected)}, 行数={len(data_km_df)}")

        # ---------------------------------------------------------------------
        # 参数优化
        # ---------------------------------------------------------------------
        initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
        params_0 = [0.1, 0.1, max(0.01, initial_d if not np.isnan(initial_d) else 0.01)]
        optimization_history = []

        log(
            f"[OPT] 初值: nu={params_0[0]:.6f}, "
            f"kappa={params_0[1]:.6f}, D={params_0[2]:.6f}"
        )
        log(f"[OPT] 开始 {OPTIMIZER_METHOD} 优化")

        result = minimize(
            objective_function,
            params_0,
            args=(
                finite_time_km,
                A_selected,
                tau_indices_map,
                deeponet_model,
                scalers,
                optimization_history,
                unified_a_grid,
                unified_tau_grid
            ),
            method=OPTIMIZER_METHOD,
            options=OPTIMIZER_OPTIONS
        )

        if result.success:
            nu_opt, kappa_opt, d_opt = result.x
            log("[OPT] 优化成功")
        else:
            log("[OPT-WARN] 优化未正常收敛，采用历史最佳点")
            if optimization_history:
                best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
                nu_opt = optimization_history[best_idx]['nu']
                kappa_opt = optimization_history[best_idx]['kappa']
                d_opt = optimization_history[best_idx]['d_diffusion']
            else:
                nu_opt, kappa_opt, d_opt = params_0

        d_opt = max(0.0, float(d_opt))
        nu_opt = float(nu_opt)
        kappa_opt = float(kappa_opt)

        elapsed = time.time() - t0
        log(
            f"[OPT] 完成 | 耗时={elapsed:.2f}s | "
            f"opt=(nu={nu_opt:.6f}, kappa={kappa_opt:.6f}, D={d_opt:.6f})"
        )

        # ---------------------------------------------------------------------
        # 保存优化历史与收敛图
        # ---------------------------------------------------------------------
        history_df = pd.DataFrame(optimization_history)
        history_csv = os.path.join(case_output_dir, 'optimization_history.csv')
        history_df.to_csv(history_csv, index=False)
        convergence_plot, convergence_data = save_convergence_plot_and_data(history_df, case_output_dir)
        log(f"[SAVE] 优化历史已保存: {history_csv}")

        # ---------------------------------------------------------------------
        # 最优参数预测与理论曲线
        # ---------------------------------------------------------------------
        log("[PRED] 生成最终 DeepONet 预测与理论曲线")
        deeponet_opt_results = compute_deeponet_km_coefficients(
            [nu_opt, kappa_opt, d_opt],
            A_selected,
            tau_indices_map,
            deeponet_model,
            scalers,
            unified_a_grid,
            unified_tau_grid
        )

        theo_std_results = {}
        for tau_idx in tau_indices_map.keys():
            tau_idx = int(tau_idx)
            theo_std_results[tau_idx] = {
                'A': A_selected,
                'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
                'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
            }

        comparison_plot, curve_data_csv = save_comparison_plot_and_data(
            finite_time_km,
            deeponet_opt_results,
            theo_std_results,
            A_selected,
            tau_indices_map,
            nu_opt,
            kappa_opt,
            d_opt,
            nu_std,
            kappa_std,
            D_std,
            case_output_dir,
            title_suffix=f"(group={group_id}, bandwidth={bandwidth})"
        )
        log(f"[SAVE] 对比图已保存: {comparison_plot}")
        log(f"[SAVE] 曲线数据已保存: {curve_data_csv}")

        # ---------------------------------------------------------------------
        # 单样本摘要
        # ---------------------------------------------------------------------
        result_record = {
            "file": original_file_name,
            "status": "success",
            "group_id": group_id,
            "bandwidth": bandwidth,

            "nu_true": nu_std,
            "kappa_true": kappa_std,
            "D_true": D_std,

            "nu_opt": nu_opt,
            "kappa_opt": kappa_opt,
            "D_opt": d_opt,

            "nu_abs_error": abs(nu_opt - nu_std),
            "kappa_abs_error": abs(kappa_opt - kappa_std),
            "D_abs_error": abs(d_opt - D_std),

            "nu_rel_error": abs(nu_opt - nu_std) / max(abs(nu_std), 1e-12),
            "kappa_rel_error": abs(kappa_opt - kappa_std) / max(abs(kappa_std), 1e-12),
            "D_rel_error": abs(d_opt - D_std) / max(abs(D_std), 1e-12),

            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "final_fun": float(result.fun) if hasattr(result, "fun") else np.nan,
            "final_objective": history_df['total_cost'].iloc[-1] if not history_df.empty else np.nan,
            "n_iterations": len(optimization_history),
            "elapsed_sec": elapsed,

            "case_output_dir": case_output_dir,
            "history_csv": history_csv,
            "convergence_plot": convergence_plot,
            "comparison_plot": comparison_plot,
            "curve_data_csv": curve_data_csv
        }

        result_summary_csv = os.path.join(case_output_dir, "result_summary.csv")
        pd.DataFrame([result_record]).to_csv(result_summary_csv, index=False)
        log(f"[SAVE] 单样本摘要已保存: {result_summary_csv}")

        log(f"[IDENT-END] {original_file_name} -> 完成")
        return result_record

    except Exception as e:
        elapsed = time.time() - t0
        log(f"[IDENT-ERROR] {original_file_name} -> 失败，耗时 {elapsed:.2f}s，错误: {e}")
        traceback.print_exc()
        return {
            "file": original_file_name,
            "status": f"failed_exception: {e}",
            "group_id": group_id,
            "bandwidth": bandwidth,
            "elapsed_sec": elapsed
        }


# =============================================================================
# 10. 绘制带宽比较图和保存 plot-ready 数据
# =============================================================================

def save_bandwidth_summary_plots(summary_df):
    if summary_df.empty:
        return

    metrics = [
        "nu_abs_error", "kappa_abs_error", "D_abs_error",
        "nu_rel_error", "kappa_rel_error", "D_rel_error",
        "final_fun", "final_objective"
    ]

    for metric in metrics:
        if metric not in summary_df.columns:
            continue

        plt.figure(figsize=(9, 6))
        for group_id in sorted(summary_df["group_id"].dropna().unique()):
            sub = summary_df[summary_df["group_id"] == group_id].sort_values("bandwidth")
            plt.plot(sub["bandwidth"], sub[metric], marker='o', label=f"group{int(group_id)}")
        plt.xlabel("Bandwidth")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Bandwidth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR_IDENT_PLOTS, f"{metric}_vs_bandwidth.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        plot_df = summary_df[["group_id", "bandwidth", metric]].copy()
        plot_df.to_csv(
            os.path.join(OUTPUT_DIR_IDENT_PLOT_DATA, f"{metric}_vs_bandwidth_data.csv"),
            index=False
        )


# =============================================================================
# 11. 主程序
# =============================================================================

if __name__ == "__main__":
    log("========== 实验 3：不同带宽参数辨识开始（Physics-Informed POD-DeepONet） ==========")
    log(f"[MAIN] KM 输入目录: {INPUT_DIR_KM_DATA}")
    log(f"[MAIN] 输出目录: {BASE_OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # 1) 加载 physics-informed 模型和统一 scaler/POD payload
    # -------------------------------------------------------------------------
    try:
        log("[MODEL] 加载 physics-informed scaler/POD payload")
        scaler_payload = safe_torch_load(DEEPONET_SCALER_PATH, map_location=DEVICE)

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

        log(f"[MODEL] POD modes={num_pod_modes}")
        log(f"[MODEL] A-grid points={len(unified_a_grid)}")
        log(f"[MODEL] tau-grid points={len(unified_tau_grid)}")

        log("[MODEL] 构建 PODDeepONet")
        deeponet_model = PODDeepONet(
            branch_input_dim=DEEPONET_BRANCH_INPUT_DIM,
            hidden_units=DEEPONET_HIDDEN_UNITS,
            num_hidden_layers=DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes=num_pod_modes,
            pod_basis=pod_basis,
            y_mean_pod_scaled=y_mean_pod_scaled,
            dropout_rate=DEEPONET_DROPOUT_RATE
        )

        log("[MODEL] 加载模型权重")
        deeponet_model.load_state_dict(safe_torch_load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()
        log("[MODEL] 模型加载成功")

    except Exception as e:
        log(f"[MAIN-ERROR] 模型加载失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 2) 扫描 KM 文件
    # -------------------------------------------------------------------------
    if not os.path.exists(INPUT_DIR_KM_DATA):
        log(f"[MAIN-ERROR] KM 输入目录不存在: {INPUT_DIR_KM_DATA}")
        raise SystemExit(1)

    km_files = [
        os.path.join(INPUT_DIR_KM_DATA, f)
        for f in sorted(os.listdir(INPUT_DIR_KM_DATA))
        if f.endswith(".csv")
    ]

    log(f"[MAIN] 找到 {len(km_files)} 个 KM 文件")
    if not km_files:
        log("[MAIN] 未找到 KM 文件，程序退出")
        raise SystemExit(0)

    # -------------------------------------------------------------------------
    # 3) 批量处理
    # -------------------------------------------------------------------------
    all_results = []
    batch_t0 = time.time()

    for i, km_file in enumerate(km_files, 1):
        log("")
        log(f"[MAIN] 全局进度 {i}/{len(km_files)} -> {os.path.basename(km_file)}")
        result = process_km_file(
            km_file,
            deeponet_model,
            scalers,
            unified_a_grid,
            unified_tau_grid
        )
        if result is not None:
            all_results.append(result)

    batch_elapsed = time.time() - batch_t0

    # -------------------------------------------------------------------------
    # 4) 保存总汇总和按 bandwidth 汇总
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(BASE_OUTPUT_DIR, "identification_summary_physics_informed.csv")
    summary_df.to_csv(summary_csv, index=False)
    log(f"[SAVE] 总汇总已保存: {summary_csv}")

    if not summary_df.empty and "status" in summary_df.columns:
        success_df = summary_df[summary_df["status"] == "success"].copy()

        if not success_df.empty:
            group_stats = success_df.groupby(["group_id", "bandwidth"]).agg({
                "nu_abs_error": ["mean", "std", "max"],
                "kappa_abs_error": ["mean", "std", "max"],
                "D_abs_error": ["mean", "std", "max"],
                "nu_rel_error": ["mean", "std", "max"],
                "kappa_rel_error": ["mean", "std", "max"],
                "D_rel_error": ["mean", "std", "max"],
                "final_fun": ["mean", "std"],
                "final_objective": ["mean", "std"],
                "elapsed_sec": ["mean", "std"]
            })
            group_stats.columns = ["_".join(col) for col in group_stats.columns]
            group_stats = group_stats.reset_index()
            group_stats_csv = os.path.join(BASE_OUTPUT_DIR, "identification_group_bandwidth_stats_physics_informed.csv")
            group_stats.to_csv(group_stats_csv, index=False)
            log(f"[SAVE] group + bandwidth 汇总已保存: {group_stats_csv}")

            summary_by_bandwidth = success_df.groupby("bandwidth", dropna=False).agg({
                "nu_abs_error": ["mean", "std", "max"],
                "kappa_abs_error": ["mean", "std", "max"],
                "D_abs_error": ["mean", "std", "max"],
                "nu_rel_error": ["mean", "std", "max"],
                "kappa_rel_error": ["mean", "std", "max"],
                "D_rel_error": ["mean", "std", "max"],
                "final_fun": ["mean", "std"],
                "final_objective": ["mean", "std"],
                "elapsed_sec": ["mean", "std"]
            })
            summary_by_bandwidth.columns = ["_".join(col) for col in summary_by_bandwidth.columns]
            summary_by_bandwidth = summary_by_bandwidth.reset_index()
            summary_by_bandwidth_csv = os.path.join(BASE_OUTPUT_DIR, "summary_by_bandwidth_physics_informed.csv")
            summary_by_bandwidth.to_csv(summary_by_bandwidth_csv, index=False)
            log(f"[SAVE] 按 bandwidth 汇总已保存: {summary_by_bandwidth_csv}")

            error_long = success_df.melt(
                id_vars=['file', 'group_id', 'bandwidth'],
                value_vars=[
                    'nu_abs_error', 'kappa_abs_error', 'D_abs_error',
                    'nu_rel_error', 'kappa_rel_error', 'D_rel_error'
                ],
                var_name='metric',
                value_name='value'
            )
            error_long_csv = os.path.join(BASE_OUTPUT_DIR, "error_metrics_long_format_physics_informed.csv")
            error_long.to_csv(error_long_csv, index=False)
            log(f"[SAVE] plot-ready 长表已保存: {error_long_csv}")

            save_bandwidth_summary_plots(success_df)
            log("[SAVE] bandwidth 对比图和底层数据已保存")

            log("===== Summary Statistics =====")
            log(f"成功样本数: {len(success_df)}")
            log(f"nu_abs_error 平均: {success_df['nu_abs_error'].mean():.6e}")
            log(f"kappa_abs_error 平均: {success_df['kappa_abs_error'].mean():.6e}")
            log(f"D_abs_error 平均: {success_df['D_abs_error'].mean():.6e}")
        else:
            log("[MAIN-WARN] 没有成功样本，跳过统计图和分组汇总")

    log(f"[MAIN] 参数辨识完成，总耗时 {batch_elapsed:.2f} 秒")
    log("========== 实验 3：不同带宽参数辨识结束 ==========")
