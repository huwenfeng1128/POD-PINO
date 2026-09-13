# -*- coding: utf-8 -*-
"""
Physics-Informed POD-DeepONet 的 SNR 参数辨识评估脚本
====================================================

改动目标：
--------
1. 保留代码一的 physics-informed POD-DeepONet 模型、scaler/POD 统一加载方式；
2. 改成代码二一样的 SNR 实验数据结构：
       SNR_experiment/km_experiment/{clean,snr_xxdb}/xxx.csv
       SNR_experiment/sim_experiment/{clean,snr_xxdb}/xxx.csv
3. 按 clean + 不同 SNR 子目录逐个做参数辨识；
4. 输出：
   - 每个样本的 optimization_history.csv
   - comparison_plot.png
   - curve_data_for_plot.csv
   - single_case_summary.csv
   - 总汇总 batch_summary_all_cases.csv
   - 按 SNR 汇总 summary_by_snr.csv
   - plot-ready 长表 error_metrics_long_format.csv

说明：
----
- 默认保留代码一中的“加权归一化 MSE”目标函数，更适合继续对比 physics-informed 模型；
- 如果你想让误差项完全等同代码二，把 USE_NORMALIZED_OBJECTIVE 改为 False。
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
from scipy.signal import hilbert

import torch
import torch.nn as nn


# =============================================================================
# 0. 配置区
# =============================================================================

# --- 使用代码二一样的 SNR 实验数据 ---
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\Revise_experiment\SNR_experiment\km_experiment'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\POD_deeponet\Revise_experiment\SNR_experiment\sim_experiment'

# --- physics-informed 模型的 SNR 评估输出目录 ---
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\shujulubangxing\SNR\result'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- 使用代码一的 physics-informed 模型 ---
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# --- DeepONet 配置：必须与 physics-informed 训练脚本一致 ---
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# --- 优化设置 ---
OMEGA_0 = 2 * math.pi * 150
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

# True  = 代码一的加权归一化 MSE；
# False = 代码二的原始加权 MSE。
USE_NORMALIZED_OBJECTIVE = True


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 1. 模型定义：保持代码一 physics-informed POD-DeepONet 结构
# =============================================================================

class MLP(nn.Module):
    """
    与代码一/训练代码一致的 MLP：
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
# 2. 工具函数
# =============================================================================

def safe_torch_load(path, map_location):
    """
    兼容不同 PyTorch 版本。
    新版 torch.load 的 weights_only 默认值变化时，直接加载 dict/scaler 可能报错。
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


def parse_params_from_filename(filepath):
    """
    从文件名中解析标准参数。

    兼容：
    1) 代码二 SNR 数据常用旧格式：
       (0.1,0.2,0.3).csv

    2) 代码一新格式：
       data_nu_neg18_388_kappa_0_648_d_10_613.csv
       或
       nu_neg18_388_kappa_0_648_d_10_613.csv
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]

    # ---------- 旧格式 ----------
    try:
        if stem.startswith('(') and stem.endswith(')'):
            values = stem[1:-1].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except Exception:
        pass

    # ---------- 新格式 ----------
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


def infer_snr_from_path(filepath):
    """
    从父目录名识别 SNR：
      clean        -> np.inf
      snr_20db     -> 20.0
      snr_-5db     -> -5.0
    """
    folder = os.path.basename(os.path.dirname(filepath)).lower()

    if folder == "clean":
        return np.inf

    m = re.search(r"snr_(-?\d+(?:\.\d+)?)db", folder)
    if m:
        return float(m.group(1))

    return np.nan


def sanitize_file_stem(filename):
    stem = os.path.splitext(filename)[0]
    return (
        stem.replace('.', 'p')
            .replace(',', '_')
            .replace('(', '')
            .replace(')', '')
            .replace(' ', '_')
    )


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """
    理论 D1：
        D1 = nu*A - (kappa/8)*A^3 + d/A
    """
    A = np.asarray(A, dtype=float)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8.0) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """
    理论 D2：
        D2 = d_diffusion
    """
    return np.full_like(np.asarray(A, dtype=float), d_diffusion)


# =============================================================================
# 3. DeepONet 预测函数
# =============================================================================

def compute_deeponet_km_coefficients(params,
                                     A_selected,
                                     tau_indices_map,
                                     model,
                                     scalers,
                                     unified_a_grid,
                                     unified_tau_grid):
    """
    给定参数 [nu, kappa, d_diffusion]，用 physics-informed POD-DeepONet
    预测整张 D1/D2 场，再抽取目标 tau，并插值到 KM 文件使用的 A 网格上。

    返回：
    {
        tau_idx: {
            'A': A_selected,
            'D1': ...,
            'D2': ...
        },
        ...
    }
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

            expected_len = 2 * n_tau * n_a
            if len(y_pred_np) != expected_len:
                raise ValueError(
                    f"模型输出长度不匹配：got {len(y_pred_np)}, "
                    f"expected {expected_len}=2*{n_tau}*{n_a}"
                )

            D1_field_pred = y_pred_np[:field_len_per_type].reshape(n_tau, n_a)
            D2_field_pred = y_pred_np[field_len_per_type:].reshape(n_tau, n_a)

            results_for_tau = {}
            A_selected = np.asarray(A_selected, dtype=float)

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                closest_tau_grid_idx = int(np.argmin(np.abs(unified_tau_grid - tau_sec_km)))

                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                interp_d1 = interp1d(
                    unified_a_grid, D1_pred_slice,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                interp_d2 = interp1d(
                    unified_a_grid, D2_pred_slice,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                results_for_tau[int(tau_idx_km)] = {
                    'A': A_selected,
                    'D1': interp_d1(A_selected),
                    'D2': interp_d2(A_selected)
                }

            return results_for_tau

    except Exception as e:
        log(f"[DeepONet] 预测失败 for params={params}: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# 4. 目标函数
# =============================================================================

def objective_function(params,
                       data_km,
                       A_selected,
                       tau_indices_map,
                       model,
                       scalers,
                       optimization_history_local,
                       p_a_weights_map,
                       unified_a_grid,
                       unified_tau_grid):
    """
    优化目标函数：
      cost = weighted MSE + D<0 penalty

    若 USE_NORMALIZED_OBJECTIVE=True：
      使用代码一版本：
      ((D1_data - D1_pred) / std(D1_data))^2
      +
      ((D2_data - D2_pred) / std(D2_data))^2

    若 USE_NORMALIZED_OBJECTIVE=False：
      使用代码二版本：
      (D1_data - D1_pred)^2 + (D2_data - D2_pred)^2
    """
    nu, kappa, d_diffusion = params
    current_iter = len(optimization_history_local) + 1

    d_negative_penalty = D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion) if d_diffusion < 0 else 0.0

    deeponet_km_results = compute_deeponet_km_coefficients(
        params, A_selected, tau_indices_map,
        model, scalers, unified_a_grid, unified_tau_grid
    )

    if deeponet_km_results is None:
        cost = 1e11 + d_negative_penalty
        mean_weighted_sq_error = np.nan
        num_compared_points = 0
    else:
        total_weighted_sq_error = 0.0
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

            A_current = np.asarray(data_km[tau_idx]['A'], dtype=float)[valid_mask]
            weights = np.array([p_a_weights_map.get(float(a), 0.0) for a in A_current], dtype=float)

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
        log(
            f"[OPT] Eval={current_iter} | "
            f"nu={nu:.6f}, kappa={kappa:.6f}, D={d_diffusion:.6f}, "
            f"mse={mean_weighted_sq_error:.6e}, penalty={d_negative_penalty:.3e}, "
            f"total_cost={cost:.6e}"
        )

    if np.isnan(cost) or np.isinf(cost):
        return 1e12
    return cost


# =============================================================================
# 5. 绘图与保存函数
# =============================================================================

def plot_comparison(data_km,
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
    """
    生成 D1 / D2 对比图：
    - KM 数据
    - DeepONet 优化后结果
    - 标准参数理论曲线
    """
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Comparison: Optimized vs Standard vs Data {title_suffix}\n'
        f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
        f'Std: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
        fontsize=12
    )

    for i, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        tau_idx = int(tau_idx)
        if tau_idx not in data_km:
            continue

        A_data = data_km[tau_idx]['A']
        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']

        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)

        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(
                A_selected,
                deeponet_results[tau_idx]['D1'],
                'b-',
                label='DeepONet Opt',
                linewidth=2
            )

        if theo_results and tau_idx in theo_results:
            ax_d1.plot(
                A_selected,
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

        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)

        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(
                A_selected,
                deeponet_results[tau_idx]['D2'],
                'b-',
                label='DeepONet Opt',
                linewidth=2
            )

        if theo_results and tau_idx in theo_results:
            ax_d2.plot(
                A_selected,
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
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    log(f"[SAVE] 对比图已保存: {plot_path}")


def save_plot_ready_curve_data(data_km,
                               deeponet_results,
                               theo_results,
                               tau_indices_map,
                               output_dir):
    """
    保存画图用长表，便于后续统一画论文图。
    """
    records = []

    for tau_idx, tau_sec in sorted(tau_indices_map.items()):
        tau_idx = int(tau_idx)
        if tau_idx not in data_km:
            continue

        A_data = np.asarray(data_km[tau_idx]['A'], dtype=float)
        D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
        D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)

        d1_pred = (
            np.asarray(deeponet_results[tau_idx]['D1'], dtype=float)
            if (deeponet_results and tau_idx in deeponet_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )
        d2_pred = (
            np.asarray(deeponet_results[tau_idx]['D2'], dtype=float)
            if (deeponet_results and tau_idx in deeponet_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )

        d1_theory = (
            np.asarray(theo_results[tau_idx]['D1'], dtype=float)
            if (theo_results and tau_idx in theo_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )
        d2_theory = (
            np.asarray(theo_results[tau_idx]['D2'], dtype=float)
            if (theo_results and tau_idx in theo_results)
            else np.full_like(A_data, np.nan, dtype=float)
        )

        for i in range(len(A_data)):
            records.append({
                "tau_index": tau_idx,
                "tau_sec": tau_sec,
                "A": A_data[i],
                "D1_data": D1_data[i],
                "D2_data": D2_data[i],
                "D1_pred_opt": d1_pred[i] if i < len(d1_pred) else np.nan,
                "D2_pred_opt": d2_pred[i] if i < len(d2_pred) else np.nan,
                "D1_theory_std": d1_theory[i] if i < len(d1_theory) else np.nan,
                "D2_theory_std": d2_theory[i] if i < len(d2_theory) else np.nan
            })

    curve_path = os.path.join(output_dir, "curve_data_for_plot.csv")
    pd.DataFrame(records).to_csv(curve_path, index=False)
    log(f"[SAVE] 曲线数据已保存: {curve_path}")


# =============================================================================
# 6. 单文件处理
# =============================================================================

def process_km_file(km_file_path,
                    deeponet_model,
                    scalers,
                    unified_a_grid,
                    unified_tau_grid,
                    base_output_dir,
                    input_dir_sim_base):
    """
    对单个 KM 文件执行参数辨识。
    """
    log("")
    log(f"===== 开始处理 KM 文件: {km_file_path} =====")

    # 解析标准参数
    nu_std, kappa_std, D_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        log(f"[SKIP] 文件名解析失败: {os.path.basename(km_file_path)}")
        return None

    # SNR 元信息与输出目录
    snr_db = infer_snr_from_path(km_file_path)
    parent_folder = os.path.basename(os.path.dirname(km_file_path))
    file_stem = sanitize_file_stem(os.path.basename(km_file_path))
    output_dir = os.path.join(base_output_dir, parent_folder, file_stem)
    os.makedirs(output_dir, exist_ok=True)

    log(f"[META] 标准参数: nu={nu_std:.6f}, kappa={kappa_std:.6f}, D={D_std:.6f}, SNR={snr_db}")

    # 读取 KM 数据
    try:
        data_km_df = pd.read_csv(km_file_path)
        log(f"[LOAD] KM 文件读取成功，行数={len(data_km_df)}")

        required_cols = ['tau_index', 'tau_sec', 'A', 'D1_data', 'D2_data']
        missing = [c for c in required_cols if c not in data_km_df.columns]
        if missing:
            raise ValueError(f"KM file missing required columns: {missing}")

        # 按 A 排序，保证 data、prediction、theory 在保存曲线时一一对应。
        finite_time_km = {}
        for tau_idx, group in data_km_df.groupby('tau_index'):
            group = group.sort_values('A')
            tau_idx_int = int(tau_idx)

            finite_time_km[tau_idx_int] = {
                'A': group['A'].values.astype(float),
                'D1': group['D1_data'].values.astype(float),
                'D2': group['D2_data'].values.astype(float),
                'tau_sec': float(group['tau_sec'].iloc[0])
            }

        A_selected = np.sort(data_km_df['A'].unique()).astype(float)

        tau_indices_map = {
            int(idx): float(ts)
            for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }

        log(f"[LOAD] tau 个数={len(tau_indices_map)}, A 点数={len(A_selected)}")

    except Exception as e:
        log(f"[ERROR] KM 数据读取失败: {e}")
        traceback.print_exc()
        return None

    # 使用对应观测/仿真数据构造 p(A) 权重
    p_a_weights_map = {float(a): 1.0 / len(A_selected) for a in A_selected}
    sim_file_path = os.path.join(input_dir_sim_base, parent_folder, os.path.basename(km_file_path))
    log(f"[WEIGHT] 尝试加载权重源文件: {sim_file_path}")

    if os.path.exists(sim_file_path):
        try:
            df_sim = pd.read_csv(sim_file_path)

            if 'Envelope' in df_sim.columns:
                sim_envelope = df_sim['Envelope'].values
                log("[WEIGHT] 使用 Envelope 计算 p(A) 权重")
            elif 'Eta' in df_sim.columns:
                sim_envelope = np.abs(hilbert(df_sim['Eta'].values))
                log("[WEIGHT] 使用 Eta 的 Hilbert 包络计算 p(A) 权重")
            else:
                sim_envelope = None
                log("[WEIGHT] 权重源文件没有 Envelope 或 Eta 列，使用均匀权重")

            if sim_envelope is not None and len(sim_envelope) > 0:
                bin_width = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0
                bin_edges = np.concatenate([
                    [A_selected[0] - bin_width / 2],
                    A_selected[:-1] + bin_width / 2,
                    [A_selected[-1] + bin_width / 2]
                ])

                hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
                hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

                if np.sum(hist) > 0:
                    p_a_weights_map = {
                        float(A_selected[i]): float(hist[i])
                        for i in range(len(A_selected))
                    }
                    log("[WEIGHT] p(A) 权重构建完成")
                else:
                    log("[WEIGHT] histogram 全零，使用均匀权重")

        except Exception as e:
            log(f"[WEIGHT] 权重构建失败，退回均匀权重: {e}")
            traceback.print_exc()
    else:
        log("[WEIGHT] 找不到对应权重源文件，使用均匀权重")

    # 初值：沿用代码一/代码二逻辑
    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    params_0 = [0.1, 0.1, max(0.01, initial_d if not np.isnan(initial_d) else 0.01)]
    optimization_history = []

    log(f"[OPT] 初值: nu={params_0[0]:.6f}, kappa={params_0[1]:.6f}, D={params_0[2]:.6f}")
    log(f"[OPT] 开始 {OPTIMIZER_METHOD} 优化；normalized_objective={USE_NORMALIZED_OBJECTIVE}")

    start_time = time.time()

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
            p_a_weights_map,
            unified_a_grid,
            unified_tau_grid
        ),
        method=OPTIMIZER_METHOD,
        options=OPTIMIZER_OPTIONS
    )

    elapsed_time = time.time() - start_time

    if result.success:
        nu_opt, kappa_opt, d_opt = result.x
        log("[OPT] 优化成功")
    else:
        log("[OPT] 优化未正常收敛，改用历史最好点")
        if len(optimization_history) == 0:
            nu_opt, kappa_opt, d_opt = params_0
        else:
            best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
            nu_opt = optimization_history[best_idx]['nu']
            kappa_opt = optimization_history[best_idx]['kappa']
            d_opt = optimization_history[best_idx]['d_diffusion']

    d_opt = max(0.0, d_opt)

    log(
        f"[OPT] 完成 | 耗时={elapsed_time:.2f}s | "
        f"opt=(nu={nu_opt:.6f}, kappa={kappa_opt:.6f}, D={d_opt:.6f})"
    )
    log(
        f"[STD] true=(nu={nu_std:.6f}, kappa={kappa_std:.6f}, D={D_std:.6f})"
    )

    # 用最优参数再预测一次
    deeponet_opt_results = compute_deeponet_km_coefficients(
        [nu_opt, kappa_opt, d_opt],
        A_selected,
        tau_indices_map,
        deeponet_model,
        scalers,
        unified_a_grid,
        unified_tau_grid
    )

    # 标准参数理论曲线
    theo_std_results = {}
    for tau_idx in tau_indices_map.keys():
        tau_idx = int(tau_idx)
        theo_std_results[tau_idx] = {
            'A': A_selected,
            'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
            'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
        }

    # 保存图
    log("[PLOT] 开始保存 comparison_plot.png")
    plot_comparison(
        finite_time_km,
        deeponet_opt_results,
        theo_std_results,
        A_selected,
        tau_indices_map,
        nu_opt, kappa_opt, d_opt,
        nu_std, kappa_std, D_std,
        output_dir,
        title_suffix=f"(SNR={snr_db})"
    )

    # 保存优化历史
    if len(optimization_history) > 0:
        history_df = pd.DataFrame(optimization_history)
        history_path = os.path.join(output_dir, 'optimization_history.csv')
        history_df.to_csv(history_path, index=False)
        log(f"[SAVE] 优化历史已保存: {history_path}")

    # 保存画图长表
    save_plot_ready_curve_data(
        finite_time_km,
        deeponet_opt_results,
        theo_std_results,
        tau_indices_map,
        output_dir
    )

    # 单样本摘要
    result_record = {
        'km_subdir': parent_folder,
        'filename': os.path.basename(km_file_path),
        'snr_db': snr_db,

        'nu_standard': nu_std,
        'kappa_standard': kappa_std,
        'D_standard': D_std,

        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,

        'nu_error_abs': abs(nu_opt - nu_std),
        'kappa_error_abs': abs(kappa_opt - kappa_std),
        'd_error_abs': abs(d_opt - D_std),

        'nu_error_rel': abs(nu_opt - nu_std) / max(abs(nu_std), 1e-12),
        'kappa_error_rel': abs(kappa_opt - kappa_std) / max(abs(kappa_std), 1e-12),
        'd_error_rel': abs(d_opt - D_std) / max(abs(D_std), 1e-12),

        'optimization_time_sec': elapsed_time,
        'optimizer_success': int(result.success),
        'optimizer_message': str(result.message),
        'num_objective_evals': len(optimization_history),
        'final_objective': optimization_history[-1]['total_cost'] if optimization_history else np.nan,
        'normalized_objective': int(USE_NORMALIZED_OBJECTIVE)
    }

    result_path = os.path.join(output_dir, "single_case_summary.csv")
    pd.DataFrame([result_record]).to_csv(result_path, index=False)
    log(f"[SAVE] 单样本摘要已保存: {result_path}")

    log(f"===== 文件处理完成: {km_file_path} =====")
    return result_record


# =============================================================================
# 7. 主程序
# =============================================================================

if __name__ == "__main__":
    log("===== 开始 Physics-Informed POD-DeepONet SNR 参数辨识评估 =====")

    try:
        # ---------------------------------------------------------------------
        # 1) 加载代码一格式的 unified scaler / POD / grid payload
        # ---------------------------------------------------------------------
        log("[MODEL] 加载 physics-informed scaler payload")
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

        log(
            f"[MODEL] POD modes={num_pod_modes}, "
            f"len(A-grid)={len(unified_a_grid)}, len(tau-grid)={len(unified_tau_grid)}"
        )

        # ---------------------------------------------------------------------
        # 2) 构建模型
        # ---------------------------------------------------------------------
        log("[MODEL] 构建 physics-informed POD-DeepONet")
        deeponet_model = PODDeepONet(
            branch_input_dim=DEEPONET_BRANCH_INPUT_DIM,
            hidden_units=DEEPONET_HIDDEN_UNITS,
            num_hidden_layers=DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes=num_pod_modes,
            pod_basis=pod_basis,
            y_mean_pod_scaled=y_mean_pod_scaled,
            dropout_rate=DEEPONET_DROPOUT_RATE
        )

        # ---------------------------------------------------------------------
        # 3) 加载模型权重
        # ---------------------------------------------------------------------
        log("[MODEL] 加载 physics-informed 模型权重")
        model_state = safe_torch_load(DEEPONET_MODEL_PATH, map_location=DEVICE)
        deeponet_model.load_state_dict(model_state)
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        log("[MODEL] 模型加载成功")

    except Exception as e:
        log(f"[CRITICAL] 模型加载失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 4) 扫描 SNR 子目录
    # -------------------------------------------------------------------------
    all_results = []

    if not os.path.exists(BASE_KM_DATA_DIR):
        log(f"[CRITICAL] KM data directory not found: {BASE_KM_DATA_DIR}")
        raise SystemExit(1)

    subdirs = [
        d for d in os.listdir(BASE_KM_DATA_DIR)
        if os.path.isdir(os.path.join(BASE_KM_DATA_DIR, d))
    ]
    subdirs = sorted(subdirs, key=lambda x: (x.lower() != "clean", x.lower()))

    log(f"[SCAN] 检测到 KM 子目录: {subdirs}")

    total_files = 0
    for subdir in subdirs:
        subdir_path = os.path.join(BASE_KM_DATA_DIR, subdir)
        total_files += len([
            f for f in os.listdir(subdir_path)
            if f.lower().endswith('.csv')
        ])

    log(f"[SCAN] 待处理 KM 文件总数: {total_files}")

    processed_counter = 0

    for subdir in subdirs:
        subdir_path = os.path.join(BASE_KM_DATA_DIR, subdir)
        km_files = sorted([
            f for f in os.listdir(subdir_path)
            if f.lower().endswith('.csv')
        ])

        log("")
        log(f"===== 开始处理子目录 {subdir} | 文件数={len(km_files)} =====")

        for filename in km_files:
            processed_counter += 1
            km_file_path = os.path.join(subdir_path, filename)
            log(f"[BATCH] 全局进度 {processed_counter}/{total_files} | 当前文件={km_file_path}")

            try:
                result = process_km_file(
                    km_file_path=km_file_path,
                    deeponet_model=deeponet_model,
                    scalers=scalers,
                    unified_a_grid=unified_a_grid,
                    unified_tau_grid=unified_tau_grid,
                    base_output_dir=BASE_OUTPUT_DIR,
                    input_dir_sim_base=INPUT_DIR_SIM_DATA
                )
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                log(f"[ERROR] 当前文件处理失败: {e}")
                traceback.print_exc()

    # -------------------------------------------------------------------------
    # 5) 保存批量汇总
    # -------------------------------------------------------------------------
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)

        # 总表
        summary_all_path = os.path.join(BASE_OUTPUT_DIR, "batch_summary_all_cases.csv")
        summary_df.to_csv(summary_all_path, index=False)
        log(f"[SAVE] 总汇总已保存: {summary_all_path}")

        # 按 SNR 统计
        summary_by_snr = summary_df.groupby('snr_db', dropna=False).agg({
            'nu_error_abs': ['mean', 'std', 'max'],
            'kappa_error_abs': ['mean', 'std', 'max'],
            'd_error_abs': ['mean', 'std', 'max'],
            'nu_error_rel': ['mean', 'std', 'max'],
            'kappa_error_rel': ['mean', 'std', 'max'],
            'd_error_rel': ['mean', 'std', 'max'],
            'optimization_time_sec': ['mean', 'std'],
            'num_objective_evals': ['mean', 'std'],
            'final_objective': ['mean', 'std']
        })

        summary_by_snr.columns = ['_'.join(col) for col in summary_by_snr.columns]
        summary_by_snr = summary_by_snr.reset_index()
        summary_by_snr_path = os.path.join(BASE_OUTPUT_DIR, "summary_by_snr.csv")
        summary_by_snr.to_csv(summary_by_snr_path, index=False)
        log(f"[SAVE] 按 SNR 汇总已保存: {summary_by_snr_path}")

        # plot-ready 长表
        error_long = summary_df.melt(
            id_vars=['km_subdir', 'filename', 'snr_db'],
            value_vars=[
                'nu_error_abs', 'kappa_error_abs', 'd_error_abs',
                'nu_error_rel', 'kappa_error_rel', 'd_error_rel'
            ],
            var_name='metric',
            value_name='value'
        )

        error_long_path = os.path.join(BASE_OUTPUT_DIR, "error_metrics_long_format.csv")
        error_long.to_csv(error_long_path, index=False)
        log(f"[SAVE] 长表误差数据已保存: {error_long_path}")

        log("===== Summary Statistics =====")
        log(f"总样本数: {len(summary_df)}")
        log(f"nu_error_abs 平均: {summary_df['nu_error_abs'].mean():.6e}")
        log(f"kappa_error_abs 平均: {summary_df['kappa_error_abs'].mean():.6e}")
        log(f"d_error_abs 平均: {summary_df['d_error_abs'].mean():.6e}")

    else:
        log("[WARN] 没有成功处理的结果。")

    log("===== Physics-Informed POD-DeepONet SNR 参数辨识评估完成 =====")
