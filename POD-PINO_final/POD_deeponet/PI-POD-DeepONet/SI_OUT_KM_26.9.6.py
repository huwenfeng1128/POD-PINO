# -*- coding: utf-8 -*-
"""
PI-POD-DeepONet 参数辨识程序
==========================

功能：
----
1. 加载已经训练好的 PI-POD-DeepONet 模型；
2. 对每个 KM 数据文件进行参数反演（nu, kappa, d_diffusion）；
3. 通过最小化：
       DeepONet 预测的有限时 KM 系数
   与
       外部 KM 数据
   之间的加权误差，
   来识别最优参数；
4. 输出：
   - 每个文件的优化历史
   - 拟合结果对比图
   - 批量汇总结果
   - 每组参数的 KM 系数宽表 KM_coefficients_all_tau.csv

与旧版相比的关键适配：
---------------------
- 适配新的训练输出目录与命名；
- 适配新的 scaler 保存格式（所有 POD/scaler 信息统一保存在 scalers_{RUN_ID}.pth）；
- 适配新的训练模型结构；
- 适配 data_nu_xxx_kappa_xxx_d_xxx.csv 命名格式；
- 保持原有参数辨识大逻辑不变。
"""

import os
import re
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

# --- PyTorch ---
import torch
import torch.nn as nn


# =============================================================================
# 1. 配置区
# =============================================================================

# --- 外部 KM 数据目录 ---
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data_out_KM'

# --- 对应仿真数据目录（用于构造 A 权重）---
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_out'

# --- 结果输出目录 ---
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\SI_OUT_result_physics_informed_with_KM'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- 指向你最新的 physics-informed 训练结果 ---
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# --- DeepONet 配置（必须与你训练脚本一致）---
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# --- 优化设置 ---
OMEGA_0 = 2 * math.pi * 150   # 保留原配置，当前逻辑未直接使用
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {
    'maxiter': 3000,
    'disp': True,
    'adaptive': True,
    'xatol': 1e-8,
    'fatol': 1e-8
}

# D < 0 的惩罚系数
D_NEGATIVE_PENALTY_FACTOR = 1e5


# =============================================================================
# 2. 模型定义（与训练代码保持一致）
# =============================================================================

class MLP(nn.Module):
    """
    与训练代码一致的 MLP：
    Linear -> [GELU + LayerNorm + Dropout + Linear] * N -> GELU + LN + Dropout + Linear
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
    与训练代码一致的 POD-DeepONet。
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
            if not torch.is_tensor(y_std_scaler):
                y_std_scaler = torch.tensor(y_std_scaler, dtype=DTYPE, device=branch_x.device)

            y_pred = y_pred_scaled * y_std_scaler + y_mean_scaler
            return y_pred


# =============================================================================
# 3. 工具函数
# =============================================================================

def to_numpy(x):
    """
    将 tensor / list / ndarray 统一转成 numpy.ndarray
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
    从文件名中解析标准参数。

    兼容两类文件名：
    1) 旧格式：
       (0.1,0.2,0.3).csv

    2) 新格式：
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
    # 例如:
    # data_nu_neg18_388_kappa_0_648_d_10_613
    # data_nu_1_250_kappa_2_500_d_3_750
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
    """
    理论 D1：
        D1 = nu*A - (kappa/8)*A^3 + d/A
    """
    A = np.asarray(A, dtype=float)
    A_safe = np.maximum(np.abs(A), 1e-9)
    return nu * A - (kappa / 8.0) * A**3 + d_diffusion / A_safe


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """
    理论 D2：
        D2 = d_diffusion
    """
    return np.full_like(np.asarray(A, dtype=float), d_diffusion)


# =============================================================================
# 3.1 KM 宽表保存函数（新增；不参与优化、权重、误差或绘图）
# =============================================================================

def save_km_wide_table(output_dir,
                       A_selected,
                       finite_time_km,
                       deeponet_opt_results,
                       tau_indices_map,
                       nu_opt,
                       kappa_opt,
                       d_opt):
    """
    将当前参数组的 KM 系数保存为宽表：

        A_plot
        D1_theoretical_opt
        D2_theoretical_opt
        D1_data_tau_0.0100s
        D2_data_tau_0.0100s
        D1_deeponet_opt_tau_0.0100s
        D2_deeponet_opt_tau_0.0100s
        ...

    说明：
    1. D1_theoretical_opt / D2_theoretical_opt 使用最终优化参数
       (nu_opt, kappa_opt, d_opt) 计算；
    2. 原始 KM data 不做插值，只按 A 值对齐；若某个 A 在某个 tau 中缺失，则写 NaN；
    3. DeepONet 结果直接保存最终优化参数对应的预测；
    4. 本函数只负责额外写 CSV，不修改目标函数、权重、优化、误差计算或绘图逻辑。
    """
    A_plot = np.asarray(A_selected, dtype=float)

    save_dict = {
        'A_plot': A_plot,
        'D1_theoretical_opt': theoretical_D1_d(
            A_plot, nu_opt, kappa_opt, d_opt
        ),
        'D2_theoretical_opt': theoretical_D2_d(
            A_plot, nu_opt, kappa_opt, d_opt
        )
    }

    # 按 tau 时间从小到大输出列
    for tau_idx, tau_sec in sorted(
            tau_indices_map.items(), key=lambda item: item[1]):

        tau_name = f'{float(tau_sec):.4f}s'

        # ---------------------------------------------------------------------
        # 外部 KM 原始数据：只做 A 值对齐，不插值
        # ---------------------------------------------------------------------
        if tau_idx in finite_time_km:
            A_data = np.asarray(finite_time_km[tau_idx]['A'], dtype=float)
            D1_data = np.asarray(finite_time_km[tau_idx]['D1'], dtype=float)
            D2_data = np.asarray(finite_time_km[tau_idx]['D2'], dtype=float)

            d1_map = {
                float(a): float(v) if np.isfinite(v) else np.nan
                for a, v in zip(A_data, D1_data)
            }
            d2_map = {
                float(a): float(v) if np.isfinite(v) else np.nan
                for a, v in zip(A_data, D2_data)
            }

            save_dict[f'D1_data_tau_{tau_name}'] = np.asarray(
                [d1_map.get(float(a), np.nan) for a in A_plot],
                dtype=float
            )
            save_dict[f'D2_data_tau_{tau_name}'] = np.asarray(
                [d2_map.get(float(a), np.nan) for a in A_plot],
                dtype=float
            )
        else:
            save_dict[f'D1_data_tau_{tau_name}'] = np.full(
                len(A_plot), np.nan, dtype=float
            )
            save_dict[f'D2_data_tau_{tau_name}'] = np.full(
                len(A_plot), np.nan, dtype=float
            )

        # ---------------------------------------------------------------------
        # 最终优化参数对应的 DeepONet KM 系数
        # ---------------------------------------------------------------------
        if (deeponet_opt_results is not None and
                tau_idx in deeponet_opt_results):

            D1_deeponet = np.asarray(
                deeponet_opt_results[tau_idx]['D1'], dtype=float
            )
            D2_deeponet = np.asarray(
                deeponet_opt_results[tau_idx]['D2'], dtype=float
            )

            if len(D1_deeponet) != len(A_plot) or len(D2_deeponet) != len(A_plot):
                raise ValueError(
                    f'DeepONet output length mismatch at tau={tau_sec}: '
                    f'len(A)={len(A_plot)}, len(D1)={len(D1_deeponet)}, '
                    f'len(D2)={len(D2_deeponet)}'
                )

            save_dict[f'D1_deeponet_opt_tau_{tau_name}'] = D1_deeponet
            save_dict[f'D2_deeponet_opt_tau_{tau_name}'] = D2_deeponet
        else:
            save_dict[f'D1_deeponet_opt_tau_{tau_name}'] = np.full(
                len(A_plot), np.nan, dtype=float
            )
            save_dict[f'D2_deeponet_opt_tau_{tau_name}'] = np.full(
                len(A_plot), np.nan, dtype=float
            )

    km_wide_df = pd.DataFrame(save_dict)
    km_wide_path = os.path.join(output_dir, 'KM_coefficients_all_tau.csv')
    km_wide_df.to_csv(km_wide_path, index=False)

    print(f'  KM wide table saved to: {km_wide_path}')
    return km_wide_df


# =============================================================================
# 4. DeepONet 预测函数
# =============================================================================

def compute_deeponet_km_coefficients(params,
                                     A_selected,
                                     tau_indices_map,
                                     model,
                                     scalers,
                                     unified_a_grid,
                                     unified_tau_grid):
    """
    给定参数 [nu, kappa, d_diffusion]，用训练好的 POD-DeepONet 预测整张 D1/D2 场，
    再抽取目标 tau，并插值到 KM 文件使用的 A 网格上。

    返回格式：
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
            # 构造 branch 输入，并按训练阶段 scaler 进行标准化
            branch_input_np = np.array([[nu, kappa, d_diffusion]], dtype=np.float64)
            branch_input_scaled_t = manual_scaler_transform(
                branch_input_np,
                scalers['branch_mean'],
                scalers['branch_std']
            )

            # DeepONet 输出物理空间预测
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
                # 找最近的 tau 网格点
                closest_tau_grid_idx = int(np.argmin(np.abs(unified_tau_grid - tau_sec_km)))

                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                # 插值到 KM 使用的 A 网格
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


# =============================================================================
# 5. 目标函数
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
        cost = 加权归一化 MSE + D<0 惩罚

    其中误差由以下组成：
        ((D1_data - D1_pred) / D1_scale)^2
        +
        ((D2_data - D2_pred) / D2_scale)^2

    D1_scale 和 D2_scale 使用当前 tau 下 KM 数据的标准差；
    若标准差过小，则设为 1，避免除零。
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

            # ================================================================
            # D1 / D2 归一化尺度
            # 只修改目标函数误差项，其他逻辑保持不变
            # ================================================================
            D1_scale = np.nanstd(D1_data_valid)
            D2_scale = np.nanstd(D2_data_valid)

            if np.isnan(D1_scale) or np.isinf(D1_scale) or D1_scale < 1e-12:
                D1_scale = 1.0
            if np.isnan(D2_scale) or np.isinf(D2_scale) or D2_scale < 1e-12:
                D2_scale = 1.0

            error_d1 = ((D1_data_valid - D1_pred_valid) / D1_scale) ** 2
            error_d2 = ((D2_data_valid - D2_pred_valid) / D2_scale) ** 2

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
        'total_cost': cost
    })

    if np.isnan(cost) or np.isinf(cost):
        return 1e12
    return cost


# =============================================================================
# 6. 绘图函数
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
                    output_dir):
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

        # --- D1 ---
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

        # --- D2 ---
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
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=180)
    plt.close()


# =============================================================================
# 7. 单文件处理
# =============================================================================

def process_km_file(km_file_path,
                    deeponet_model,
                    scalers,
                    unified_a_grid,
                    unified_tau_grid,
                    base_output_dir,
                    input_dir_sim_data):
    """
    对单个 KM 文件执行参数辨识。
    """
    print(f"\n--- Processing: {os.path.basename(km_file_path)} ---")

    # 解析标准参数
    nu_std, kappa_std, D_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        print(f"WARNING: Could not parse params from '{os.path.basename(km_file_path)}'. Skipping.")
        return None

    # 输出目录
    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0]
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 读取 KM 数据
    try:
        data_km_df = pd.read_csv(km_file_path)

        required_cols = ['tau_index', 'tau_sec', 'A', 'D1_data', 'D2_data']
        missing = [c for c in required_cols if c not in data_km_df.columns]
        if missing:
            raise ValueError(f"KM file missing required columns: {missing}")

        finite_time_km = {
            tau_idx: {
                'A': group['A'].values.astype(float),
                'D1': group['D1_data'].values.astype(float),
                'D2': group['D2_data'].values.astype(float),
                'tau_sec': float(group['tau_sec'].iloc[0])
            }
            for tau_idx, group in data_km_df.groupby('tau_index')
        }

        A_selected = np.sort(data_km_df['A'].unique()).astype(float)

        tau_indices_map = {
            int(idx): float(ts)
            for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }

    except Exception as e:
        print(f"Error loading KM data: {e}")
        return None

    # 使用对应仿真数据构造 A 权重
    p_a_weights_map = {float(a): 1.0 / len(A_selected) for a in A_selected}
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
                    p_a_weights_map = {float(A_selected[i]): float(hist[i]) for i in range(len(A_selected))}
        except Exception as e:
            print(f"Warning: Could not process sim file for weights: {e}")

    # 初值：保留原来的大致逻辑
    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    params_0 = [0.1, 0.1, max(0.01, initial_d if not np.isnan(initial_d) else 0.01)]
    optimization_history = []

    print(f"  Initial guess: ν={params_0[0]:.4f}, κ={params_0[1]:.4f}, D={params_0[2]:.4f}")

    # 优化
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

    # 提取最优参数
    if result.success:
        nu_opt, kappa_opt, d_opt = result.x
    else:
        print("  Optimization did not converge. Using best from history.")
        if len(optimization_history) == 0:
            nu_opt, kappa_opt, d_opt = params_0
        else:
            best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
            nu_opt = optimization_history[best_idx]['nu']
            kappa_opt = optimization_history[best_idx]['kappa']
            d_opt = optimization_history[best_idx]['d_diffusion']

    d_opt = max(0.0, d_opt)

    print(f"  Optimization finished in {elapsed_time:.2f}s")
    print(f"  Optimized: ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}")
    print(f"  Standard:  ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f}")

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
        theo_std_results[tau_idx] = {
            'A': A_selected,
            'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
            'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
        }

    # 画图
    plot_comparison(
        finite_time_km,
        deeponet_opt_results,
        theo_std_results,
        A_selected,
        tau_indices_map,
        nu_opt, kappa_opt, d_opt,
        nu_std, kappa_std, D_std,
        output_dir
    )

    # -------------------------------------------------------------------------
    # 新增：保存当前参数组全部 KM 系数宽表
    # 仅导出，不改变原有优化、权重、误差计算和绘图逻辑
    # -------------------------------------------------------------------------
    save_km_wide_table(
        output_dir=output_dir,
        A_selected=A_selected,
        finite_time_km=finite_time_km,
        deeponet_opt_results=deeponet_opt_results,
        tau_indices_map=tau_indices_map,
        nu_opt=nu_opt,
        kappa_opt=kappa_opt,
        d_opt=d_opt
    )

    # 保存优化历史
    if len(optimization_history) > 0:
        history_df = pd.DataFrame(optimization_history)
        history_df.to_csv(os.path.join(output_dir, 'optimization_history.csv'), index=False)

    # 保存总结
    result_summary = {
        'filename': os.path.basename(km_file_path),
        'nu_standard': nu_std,
        'kappa_standard': kappa_std,
        'D_standard': D_std,
        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,
        'nu_error': abs(nu_opt - nu_std),
        'kappa_error': abs(kappa_opt - kappa_std),
        'd_error': abs(d_opt - D_std),
        'optimization_time': elapsed_time,
        'optimizer_success': bool(result.success),
        'optimizer_message': str(result.message)
    }

    pd.DataFrame([result_summary]).to_csv(
        os.path.join(output_dir, 'result_summary.csv'),
        index=False
    )

    return result_summary


# =============================================================================
# 8. 主程序
# =============================================================================

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

        print(f"  POD modes     : {num_pod_modes}")
        print(f"  A-grid points : {len(unified_a_grid)}")
        print(f"  Tau-grid points: {len(unified_tau_grid)}")

        # ---------------------------------------------------------------------
        # 2) 构建模型
        # ---------------------------------------------------------------------
        print("Building model...")
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
        print("Loading model weights...")
        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        print("✓ Model loaded successfully!\n")

    except Exception as e:
        print("CRITICAL ERROR: Failed to load DeepONet model.")
        print(f"Error: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    # -------------------------------------------------------------------------
    # 4) 批量处理 KM 文件
    # -------------------------------------------------------------------------
    print("=== Starting Batch Processing ===")
    all_results = []

    if not os.path.exists(BASE_KM_DATA_DIR):
        print(f"KM data directory not found: {BASE_KM_DATA_DIR}")
        raise SystemExit(1)

    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(km_files)} KM files to process\n")

    for filename in tqdm(km_files, desc="Processing KM Files"):
        km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)

        try:
            result = process_km_file(
                km_file_path=km_file_path,
                deeponet_model=deeponet_model,
                scalers=scalers,
                unified_a_grid=unified_a_grid,
                unified_tau_grid=unified_tau_grid,
                base_output_dir=BASE_OUTPUT_DIR,
                input_dir_sim_data=INPUT_DIR_SIM_DATA
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed processing {filename}: {e}")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # 5) 保存批量汇总
    # -------------------------------------------------------------------------
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, "batch_summary_physics_informed.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n=== Summary Statistics ===")
        print(f"Total files processed successfully: {len(all_results)}")
        print("\nMean absolute errors:")
        print(f"  ν error: {summary_df['nu_error'].mean():.6f}")
        print(f"  κ error: {summary_df['kappa_error'].mean():.6f}")
        print(f"  D error: {summary_df['d_error'].mean():.6f}")
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("\nNo successful results were produced.")

    print("\n=== Batch processing completed ===")
