# -*- coding: utf-8 -*-
"""
PI-POD-DeepONet 参数辨识程序：初值敏感性实验
目的：
1. 从 KM 数据目录中随机抽取 3 组不同参数文件
2. 每组参数设置 5 组不同初值
3. 使用 physics-informed POD-DeepONet 模型探究参数辨识对初值的敏感性
4. 保存详细优化历史、识别结果、图片与汇总统计

说明：
- 数据目录保持原初值敏感性实验不变；
- 模型加载方式改为新版 physics-informed 训练输出：
  model_{RUN_ID}.pth + scalers_{RUN_ID}.pth；
- POD basis、POD mean、统一 A/tau 网格均从 scalers_{RUN_ID}.pth 中读取；
- 不再依赖 pod_params_{RUN_ID}.pth。
"""

import os
import json
import math
import time
import random
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from tqdm import tqdm

# --- PyTorch Imports for DeepONet ---
import torch
import torch.nn as nn


# =========================================================
# 0. 配置区
# =========================================================
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\SIlubangxing\initial\result'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# DeepONet 训练结果路径：改为 physics-informed 模型
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# DeepONet 配置（必须与训练一致）
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# 优化参数
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {
    'maxiter': 3000,
    'disp': False,
    'adaptive': True,
    'xatol': 1e-8,
    'fatol': 1e-8
}
D_NEGATIVE_PENALTY_FACTOR = 1e5

# 实验设置
RANDOM_SEED = 2026
N_RANDOM_FILES = 3
N_INITIAL_GUESSES = 5

# 初值采样范围（可按你的训练参数分布再调整）
NU_INIT_RANGE = (0.05, 20.0)
KAPPA_INIT_RANGE = (0.05, 8.0)
D_INIT_RANGE = (0.05, 20.0)

# 是否额外加入一个“基于 D2 均值”的启发式初值
USE_HEURISTIC_INITIAL = True

# True: 使用代码一 physics-informed 版本中的 D1/D2 归一化加权 MSE
# False: 使用原初值敏感性脚本中的原始加权 MSE
USE_NORMALIZED_OBJECTIVE = False


# =========================================================
# 1. 日志函数
# =========================================================
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def to_numpy(x):
    """将 tensor / list / ndarray 统一转为 numpy.ndarray。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================
# 2. 模型定义（与 physics-informed 训练脚本一致）
# =========================================================
class MLP(nn.Module):
    """MLP with GELU and LayerNorm"""

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
    """PODDeepONet"""

    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes,
                 pod_basis, y_mean_pod_scaled, dropout_rate):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, dropout_rate)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod_scaled, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
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

            return y_pred_scaled * y_std_scaler + y_mean_scaler


# =========================================================
# 3. 工具函数
# =========================================================
def manual_scaler_transform(data, mean, std):
    """
    使用训练阶段保存好的 mean/std 对 branch 输入做标准化。
    """
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
    从文件名中提取真实参数。

    支持：
    1) (nu,kappa,D).csv
       例如 (4.5,2.0,4.5).csv
    2) data_nu_neg18_388_kappa_0_648_d_10_613.csv
       或 nu_neg18_388_kappa_0_648_d_10_613.csv
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    # 旧格式：(4.5,2.0,4.5).csv
    try:
        if stem.startswith('(') and stem.endswith(')'):
            values = stem[1:-1].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except (ValueError, IndexError):
        pass

    # 新格式：data_nu_neg18_388_kappa_0_648_d_10_613
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
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8.0) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    return np.full_like(np.asarray(A), d_diffusion)


def compute_deeponet_km_coefficients(params, A_selected, tau_indices_map, model, scalers,
                                     unified_a_grid, unified_tau_grid):
    nu, kappa, d_diffusion = params

    if any(np.isnan(p) or np.isinf(p) for p in params):
        return None

    try:
        model.eval()
        with torch.no_grad():
            branch_input_np = np.array([[nu, kappa, d_diffusion]])
            branch_input_scaled_t = manual_scaler_transform(
                branch_input_np, scalers['branch_mean'], scalers['branch_std']
            )

            y_pred_t = model.predict(
                branch_input_scaled_t,
                scalers['y_mean_scaler'],
                scalers['y_std_scaler']
            )
            y_pred_np = y_pred_t.cpu().numpy().flatten()

            field_len_per_type = len(y_pred_np) // 2
            n_tau_points = len(unified_tau_grid)
            n_a_points = len(unified_a_grid)

            D1_field_pred = y_pred_np[:field_len_per_type].reshape(n_tau_points, n_a_points)
            D2_field_pred = y_pred_np[field_len_per_type:].reshape(n_tau_points, n_a_points)

            results_for_tau = {}

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                closest_tau_grid_idx = np.argmin(np.abs(unified_tau_grid - tau_sec_km))

                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                interp_d1 = interp1d(unified_a_grid, D1_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                interp_d2 = interp1d(unified_a_grid, D2_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)

                results_for_tau[tau_idx_km] = {
                    'A': np.array(A_selected),
                    'D1': interp_d1(A_selected),
                    'D2': interp_d2(A_selected)
                }

            return results_for_tau

    except Exception as e:
        log(f"[PRED-ERROR] params={params} -> {e}")
        traceback.print_exc()
        return None


def build_pa_weights_map(sim_file_path, A_selected):
    """
    从模拟信号中构造 p(A) 权重；若失败则退化为均匀权重
    """
    p_a_weights_map = {a: 1.0 / len(A_selected) for a in A_selected}

    if not os.path.exists(sim_file_path):
        return p_a_weights_map

    try:
        df_sim = pd.read_csv(sim_file_path)

        if 'Envelope' in df_sim.columns:
            sim_envelope = df_sim['Envelope'].values
        elif 'Eta' in df_sim.columns:
            sim_envelope = np.abs(hilbert(df_sim['Eta'].values))
        else:
            return p_a_weights_map

        if len(A_selected) > 1:
            bin_width = np.mean(np.diff(A_selected))
        else:
            bin_width = 1.0

        bin_edges = np.concatenate([
            [A_selected[0] - bin_width / 2],
            np.array(A_selected[:-1]) + bin_width / 2,
            [A_selected[-1] + bin_width / 2]
        ])

        hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
        p_a_weights_map = {A_selected[i]: hist[i] for i in range(len(A_selected))}
        return p_a_weights_map

    except Exception as e:
        log(f"[WEIGHT-WARN] 计算 p(A) 权重失败，改用均匀权重: {e}")
        return p_a_weights_map


# =========================================================
# 4. 目标函数
# =========================================================
def objective_function(params, data_km, A_selected, tau_indices_map, model, scalers,
                       optimization_history_local, p_a_weights_map, unified_a_grid, unified_tau_grid):
    """
    目标函数：
    - 默认使用 physics-informed 代码一中的加权归一化 MSE；
    - 可通过 USE_NORMALIZED_OBJECTIVE=False 退回原始加权 MSE；
    - 对 D < 0 添加惩罚。
    """
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

    if np.isnan(cost) or np.isinf(cost):
        return 1e12
    return cost


# =========================================================
# 5. 初值生成
# =========================================================
def generate_initial_guesses(finite_time_km, seed):
    """
    生成 5 组初值
    """
    rng = np.random.default_rng(seed)

    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    if np.isnan(initial_d) or np.isinf(initial_d):
        initial_d = 1.0

    guesses = []

    if USE_HEURISTIC_INITIAL:
        guesses.append([
            0.1,
            0.1,
            max(0.01, float(initial_d))
        ])

    while len(guesses) < N_INITIAL_GUESSES:
        nu0 = rng.uniform(*NU_INIT_RANGE)
        kappa0 = rng.uniform(*KAPPA_INIT_RANGE)
        d0 = rng.uniform(*D_INIT_RANGE)
        guesses.append([float(nu0), float(kappa0), float(d0)])

    return guesses[:N_INITIAL_GUESSES]


# =========================================================
# 6. 绘图函数
# =========================================================
def plot_single_run_comparison(data_km, deeponet_results, theo_results, A_selected, tau_indices_map,
                               nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std, output_path):
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Comparison: Optimized vs Standard vs KM Data\n'
        f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
        f'True: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
        fontsize=12
    )

    for i, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        A_data = data_km[tau_idx]['A']
        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']

        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(A_selected, deeponet_results[tau_idx]['D1'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d1.plot(A_selected, theo_results[tau_idx]['D1'], 'r--', label='Theoretical True', linewidth=2)
        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at τ={tau_sec:.4f}s')
        ax_d1.grid(True, alpha=0.3)
        ax_d1.legend()

        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(A_selected, deeponet_results[tau_idx]['D2'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d2.plot(A_selected, theo_results[tau_idx]['D2'], 'r--', label='Theoretical True', linewidth=2)
        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at τ={tau_sec:.4f}s')
        ax_d2.grid(True, alpha=0.3)
        ax_d2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_optimization_histories(all_histories_dict, output_path):
    plt.figure(figsize=(10, 6))
    for run_name, history_df in all_histories_dict.items():
        if not history_df.empty:
            plt.plot(history_df["iteration"], history_df["total_cost"], label=run_name)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    plt.title("Optimization Histories for Different Initial Guesses")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_initial_sensitivity(result_df, nu_std, kappa_std, d_std, output_dir):
    """
    单个样本：画 5 组初值下最终参数和误差
    """
    if result_df.empty:
        return

    # 图1：最终识别参数 vs run_id
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(result_df["run_id"], result_df["nu_optimized"], marker='o', label='nu_opt')
    axes[0].axhline(nu_std, linestyle='--', label='nu_true')
    axes[0].set_ylabel("nu")
    axes[0].set_title("Sensitivity of nu to Initial Guesses")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result_df["run_id"], result_df["kappa_optimized"], marker='o', label='kappa_opt')
    axes[1].axhline(kappa_std, linestyle='--', label='kappa_true')
    axes[1].set_ylabel("kappa")
    axes[1].set_title("Sensitivity of kappa to Initial Guesses")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(result_df["run_id"], result_df["d_diffusion_optimized"], marker='o', label='D_opt')
    axes[2].axhline(d_std, linestyle='--', label='D_true')
    axes[2].set_ylabel("D")
    axes[2].set_xlabel("Run ID")
    axes[2].set_title("Sensitivity of D to Initial Guesses")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "initial_sensitivity_parameters.png"), dpi=150)
    plt.close()

    # 图2：误差柱状图
    x = np.arange(len(result_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, result_df["nu_error"], width=width, label="nu_error")
    plt.bar(x, result_df["kappa_error"], width=width, label="kappa_error")
    plt.bar(x + width, result_df["d_error"], width=width, label="D_error")
    plt.xticks(x, result_df["run_id"], rotation=0)
    plt.ylabel("Absolute Error")
    plt.title("Absolute Errors under Different Initial Guesses")
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "initial_sensitivity_errors.png"), dpi=150)
    plt.close()

    # 图3：final cost
    plt.figure(figsize=(10, 6))
    plt.plot(result_df["run_id"], result_df["final_cost"], marker='o')
    plt.yscale("log")
    plt.xlabel("Run ID")
    plt.ylabel("Final Cost")
    plt.title("Final Cost under Different Initial Guesses")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "initial_sensitivity_final_cost.png"), dpi=150)
    plt.close()


def plot_global_summary(summary_df, output_dir):
    """
    全局汇总图：3个样本 × 5个初值
    """
    if summary_df.empty:
        return

    # 按文件分别画误差散点
    for metric in ["nu_error", "kappa_error", "d_error", "final_cost"]:
        plt.figure(figsize=(12, 6))
        for filename in summary_df["filename"].unique():
            sub = summary_df[summary_df["filename"] == filename]
            plt.plot(sub["run_id"], sub[metric], marker='o', label=filename)
        if metric == "final_cost":
            plt.yscale("log")
        plt.xlabel("Run ID")
        plt.ylabel(metric)
        plt.title(f"{metric} under Different Initial Guesses")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"global_{metric}_by_run.png"), dpi=150)
        plt.close()

    # 箱线图
    for metric in ["nu_error", "kappa_error", "d_error"]:
        plt.figure(figsize=(10, 6))
        data = [summary_df[summary_df["filename"] == fn][metric].values for fn in summary_df["filename"].unique()]
        plt.boxplot(data, tick_labels=summary_df["filename"].unique(), showmeans=True)
        plt.ylabel(metric)
        plt.title(f"{metric} Distribution across Initial Guesses")
        plt.xticks(rotation=20)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"global_{metric}_boxplot.png"), dpi=150)
        plt.close()


# =========================================================
# 7. 单个初值下的辨识
# =========================================================
def run_single_identification(run_id, init_params, finite_time_km, A_selected, tau_indices_map,
                              deeponet_model, scalers, p_a_weights_map,
                              unified_a_grid, unified_tau_grid,
                              nu_std, kappa_std, D_std, output_dir):
    log(f"[RUN-START] {run_id} -> 初值: nu0={init_params[0]:.6f}, kappa0={init_params[1]:.6f}, D0={init_params[2]:.6f}")

    optimization_history = []
    t0 = time.time()

    result = minimize(
        objective_function, init_params,
        args=(finite_time_km, A_selected, tau_indices_map, deeponet_model, scalers,
              optimization_history, p_a_weights_map, unified_a_grid, unified_tau_grid),
        method=OPTIMIZER_METHOD,
        options=OPTIMIZER_OPTIONS
    )

    elapsed_time = time.time() - t0

    if result.success:
        nu_opt, kappa_opt, d_opt = result.x
    else:
        log(f"[RUN-WARN] {run_id} -> 优化未正常收敛，采用历史最佳点")
        best_idx = np.argmin([h['total_cost'] for h in optimization_history])
        nu_opt = optimization_history[best_idx]['nu']
        kappa_opt = optimization_history[best_idx]['kappa']
        d_opt = optimization_history[best_idx]['d_diffusion']

    d_opt = max(0.0, d_opt)

    history_df = pd.DataFrame(optimization_history)
    history_path = os.path.join(output_dir, f"{run_id}_optimization_history.csv")
    history_df.to_csv(history_path, index=False)

    final_cost = history_df["total_cost"].iloc[-1] if not history_df.empty else np.nan

    deeponet_opt_results = compute_deeponet_km_coefficients(
        [nu_opt, kappa_opt, d_opt],
        A_selected, tau_indices_map,
        deeponet_model, scalers,
        unified_a_grid, unified_tau_grid
    )

    theo_std_results = {}
    for tau_idx in tau_indices_map.keys():
        theo_std_results[tau_idx] = {
            'A': A_selected,
            'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
            'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
        }

    comparison_plot_path = os.path.join(output_dir, f"{run_id}_comparison_plot.png")
    plot_single_run_comparison(
        finite_time_km, deeponet_opt_results, theo_std_results,
        A_selected, tau_indices_map,
        nu_opt, kappa_opt, d_opt,
        nu_std, kappa_std, D_std,
        comparison_plot_path
    )

    log(f"[RUN-END] {run_id} -> 用时 {elapsed_time:.2f}s | "
        f"nu={nu_opt:.6f}, kappa={kappa_opt:.6f}, D={d_opt:.6f}")

    return {
        'run_id': run_id,
        'nu_init': init_params[0],
        'kappa_init': init_params[1],
        'd_init': init_params[2],
        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,
        'nu_error': abs(nu_opt - nu_std),
        'kappa_error': abs(kappa_opt - kappa_std),
        'd_error': abs(d_opt - D_std),
        'optimization_time': elapsed_time,
        'optimizer_success': bool(result.success),
        'final_cost': final_cost,
        'history_csv': history_path,
        'comparison_plot': comparison_plot_path
    }, history_df


# =========================================================
# 8. 单个 KM 文件处理
# =========================================================
def process_km_file_with_initial_sensitivity(km_file_path, deeponet_model, scalers,
                                             unified_a_grid, unified_tau_grid,
                                             base_output_dir, input_dir_sim_data, file_seed):
    filename = os.path.basename(km_file_path)
    log("=" * 100)
    log(f"[FILE-START] 开始处理: {filename}")

    nu_std, kappa_std, D_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        log(f"[FILE-WARN] 无法从文件名解析参数，跳过: {filename}")
        return []

    file_output_dir_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 KM 数据
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
        A_selected = np.array(sorted(data_km_df['A'].unique()))
        tau_indices_map = {
            int(idx): float(ts)
            for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }
    except Exception as e:
        log(f"[FILE-ERROR] 读取 KM 数据失败: {filename} -> {e}")
        traceback.print_exc()
        return []

    # 保存基础元信息
    meta = {
        "filename": filename,
        "nu_true": nu_std,
        "kappa_true": kappa_std,
        "D_true": D_std,
        "n_tau": int(len(tau_indices_map)),
        "n_A": int(len(A_selected)),
        "file_seed": int(file_seed)
    }
    with open(os.path.join(output_dir, "file_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # p(A) 权重
    sim_file_path = os.path.join(input_dir_sim_data, filename)
    p_a_weights_map = build_pa_weights_map(sim_file_path, A_selected)

    # 初值生成
    initial_guesses = generate_initial_guesses(finite_time_km, seed=file_seed)
    init_df = pd.DataFrame(initial_guesses, columns=["nu_init", "kappa_init", "d_init"])
    init_df["run_id"] = [f"run_{i+1}" for i in range(len(init_df))]
    init_df.to_csv(os.path.join(output_dir, "initial_guesses.csv"), index=False)

    log(f"[FILE] {filename} -> 共生成 {len(initial_guesses)} 组初值")

    file_results = []
    all_histories_dict = {}

    for i, init_params in enumerate(initial_guesses, 1):
        run_id = f"run_{i}"
        result_row, history_df = run_single_identification(
            run_id, init_params,
            finite_time_km, A_selected, tau_indices_map,
            deeponet_model, scalers, p_a_weights_map,
            unified_a_grid, unified_tau_grid,
            nu_std, kappa_std, D_std, output_dir
        )
        result_row["filename"] = filename
        result_row["nu_standard"] = nu_std
        result_row["kappa_standard"] = kappa_std
        result_row["D_standard"] = D_std
        file_results.append(result_row)
        all_histories_dict[run_id] = history_df

    # 保存单文件汇总
    result_df = pd.DataFrame(file_results)
    result_df.to_csv(os.path.join(output_dir, "initial_sensitivity_results.csv"), index=False)

    # 画优化历史总图
    plot_optimization_histories(
        all_histories_dict,
        os.path.join(output_dir, "all_optimization_histories.png")
    )

    # 画单文件敏感性图
    plot_initial_sensitivity(result_df, nu_std, kappa_std, D_std, output_dir)

    # 保存统计量
    stats = {
        "filename": filename,
        "nu_error_mean": float(result_df["nu_error"].mean()),
        "nu_error_std": float(result_df["nu_error"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "kappa_error_mean": float(result_df["kappa_error"].mean()),
        "kappa_error_std": float(result_df["kappa_error"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "d_error_mean": float(result_df["d_error"].mean()),
        "d_error_std": float(result_df["d_error"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "nu_opt_std": float(result_df["nu_optimized"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "kappa_opt_std": float(result_df["kappa_optimized"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "d_opt_std": float(result_df["d_diffusion_optimized"].std(ddof=1)) if len(result_df) > 1 else 0.0,
        "final_cost_mean": float(result_df["final_cost"].mean()),
        "final_cost_std": float(result_df["final_cost"].std(ddof=1)) if len(result_df) > 1 else 0.0,
    }
    with open(os.path.join(output_dir, "initial_sensitivity_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log(f"[FILE-END] 完成: {filename}")
    return file_results


# =========================================================
# 9. 主程序
# =========================================================
if __name__ == "__main__":
    log("========== 初值敏感性实验开始 ==========")
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # -------------------------------
    # 9.1 加载模型
    # -------------------------------
    log("[MAIN] 加载 DeepONet 模型与相关参数")

    try:
        log("[MAIN] 加载 physics-informed scaler/POD/grid payload")
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

        log(f"[MAIN] POD modes={num_pod_modes}, len(A-grid)={len(unified_a_grid)}, len(tau-grid)={len(unified_tau_grid)}")

        deeponet_model = PODDeepONet(
            DEEPONET_BRANCH_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS,
            DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes,
            pod_basis,
            y_mean_pod_scaled,
            DEEPONET_DROPOUT_RATE
        )

        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        log("[MAIN] physics-informed 模型加载成功")

    except Exception as e:
        log(f"[MAIN-ERROR] 模型加载失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)


    # -------------------------------
    # 9.2 随机抽取 3 个 KM 文件
    # -------------------------------
    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    log(f"[MAIN] 找到 {len(km_files)} 个 KM 文件")

    if len(km_files) < N_RANDOM_FILES:
        log("[MAIN-ERROR] KM 文件数量不足，无法抽取 3 组")
        raise SystemExit(1)

    selected_km_files = random.sample(km_files, N_RANDOM_FILES)
    log("[MAIN] 随机抽取的 3 个文件如下：")
    for i, fn in enumerate(selected_km_files, 1):
        log(f"  {i}. {fn}")

    pd.DataFrame({"selected_filename": selected_km_files}).to_csv(
        os.path.join(BASE_OUTPUT_DIR, "selected_random_files.csv"),
        index=False
    )

    # -------------------------------
    # 9.3 逐文件做 5 组初值敏感性辨识
    # -------------------------------
    all_results = []
    batch_t0 = time.time()

    for idx, filename in enumerate(tqdm(selected_km_files, desc="Processing Selected KM Files"), 1):
        km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
        file_seed = RANDOM_SEED + idx * 100

        file_results = process_km_file_with_initial_sensitivity(
            km_file_path,
            deeponet_model,
            scalers,
            unified_a_grid,
            unified_tau_grid,
            BASE_OUTPUT_DIR,
            INPUT_DIR_SIM_DATA,
            file_seed
        )
        all_results.extend(file_results)

    total_elapsed = time.time() - batch_t0

    # -------------------------------
    # 9.4 保存全局汇总
    # -------------------------------
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, "initial_sensitivity_summary_physics_informed.csv")
        summary_df.to_csv(summary_path, index=False)

        # 全局统计
        stats_df = summary_df.groupby("filename").agg({
            "nu_error": ["mean", "std"],
            "kappa_error": ["mean", "std"],
            "d_error": ["mean", "std"],
            "nu_optimized": ["mean", "std"],
            "kappa_optimized": ["mean", "std"],
            "d_diffusion_optimized": ["mean", "std"],
            "final_cost": ["mean", "std"],
            "optimization_time": ["mean", "std"]
        })
        stats_df.columns = ["_".join(col) for col in stats_df.columns]
        stats_df = stats_df.reset_index()
        stats_path = os.path.join(BASE_OUTPUT_DIR, "initial_sensitivity_group_stats_physics_informed.csv")
        stats_df.to_csv(stats_path, index=False)

        # 全局图
        plot_global_summary(summary_df, BASE_OUTPUT_DIR)

        # 额外保存一个总 JSON
        global_meta = {
            "random_seed": RANDOM_SEED,
            "n_random_files": N_RANDOM_FILES,
            "n_initial_guesses": N_INITIAL_GUESSES,
            "total_runs": int(len(summary_df)),
            "total_elapsed_sec": float(total_elapsed),
            "selected_files": selected_km_files,
            "model_run_id": RUN_ID,
            "objective_mode": "normalized_weighted_mse" if USE_NORMALIZED_OBJECTIVE else "raw_weighted_mse"
        }
        with open(os.path.join(BASE_OUTPUT_DIR, "experiment_meta_physics_informed.json"), "w", encoding="utf-8") as f:
            json.dump(global_meta, f, ensure_ascii=False, indent=2)

        log("========== 实验完成 ==========")
        log(f"[MAIN] 总运行数: {len(summary_df)}")
        log(f"[MAIN] 总耗时: {total_elapsed:.2f} 秒")
        log(f"[MAIN] 汇总文件: {summary_path}")
        log(f"[MAIN] 分组统计: {stats_path}")

        log("[MAIN] 平均绝对误差：")
        log(f"  nu_error_mean    = {summary_df['nu_error'].mean():.6f}")
        log(f"  kappa_error_mean = {summary_df['kappa_error'].mean():.6f}")
        log(f"  d_error_mean     = {summary_df['d_error'].mean():.6f}")

    else:
        log("[MAIN-WARN] 没有得到有效结果")

    log("========== 初值敏感性实验结束 ==========")