# -*- coding: utf-8 -*-
"""
Physics-Informed POD-DeepONet 的不同信号长度参数辨识评估脚本
=============================================================

改动目标：
--------
1. 保留原“不同信号长度”实验的数据：
       signal_length_experiment/km_data
       signal_length_experiment/sim_data
2. 将原脚本中的 v3 POD-DeepONet 模型替换为代码一中的 physics-informed POD-DeepONet 模型；
3. 使用 physics-informed 训练脚本的统一 scaler/POD 保存格式：
       scalers_{RUN_ID}.pth
   不再加载 pod_params_{RUN_ID}.pth；
4. 对不同信号长度 T=50s / 100s / 200s / 300s / 400s 的 KM 数据逐个做参数辨识；
5. 保存：
   - 每个样本的 optimization_history.csv
   - comparison_plot.png
   - curve_data_for_plot.csv
   - single_case_summary.csv
   - 总汇总 batch_summary_all_cases_physics_informed.csv
   - 按信号长度汇总 summary_by_signal_length_physics_informed.csv
   - plot-ready 长表 error_metrics_long_format_physics_informed.csv

说明：
----
- 默认保留代码一中的“加权归一化 MSE”目标函数，更适合继续对比 physics-informed 模型；
- 如果你想让误差项完全等同原信号长度脚本，把 USE_NORMALIZED_OBJECTIVE 改为 False。
"""

import os
import re
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

# =========================
# 0. 配置
# =========================
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\Revise_experiment\signal_length_experiment\km_data'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\POD_deeponet\Revise_experiment\signal_length_experiment\sim_data'
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\shujulubangxing\length\result'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- 使用代码一的 physics-informed 训练结果 ---
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 128
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {'maxiter': 3000, 'disp': True, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8}
D_NEGATIVE_PENALTY_FACTOR = 1e5

OBJECTIVE_LOG_FREQUENCY = 20  # 每 20 次目标函数评估打印一次

# True  = 使用代码一的加权归一化 MSE；
# False = 使用原信号长度脚本的原始加权 MSE。
USE_NORMALIZED_OBJECTIVE = True


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


# =========================
# 1. 模型定义（保持原代码风格）
# =========================
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
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
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

            return y_pred_scaled * y_std_scaler + y_mean_scaler


# =========================
# 2. 工具函数
# =========================
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


def parse_params_and_length_from_filename(filepath):
    """
    支持两种文件名格式：
    1. (nu=2.50,kappa=1.60,D=2.50,T=50s).csv
    2. (2.50,1.60,2.50,T=50s).csv  或包含 T=50s 的类似变体
    """
    filename = os.path.basename(filepath)

    # 格式1：带键名
    m1 = re.search(
        r'nu=([\-0-9\.]+),\s*kappa=([\-0-9\.]+),\s*D=([\-0-9\.]+),\s*T=([0-9]+)s',
        filename
    )
    if m1:
        return float(m1.group(1)), float(m1.group(2)), float(m1.group(3)), int(m1.group(4))

    # 格式2：前三个是参数，后面带 T=xxs
    m2 = re.search(
        r'\(([\-0-9\.]+),([\-\d\.]+),([\-\d\.]+),\s*T=([0-9]+)s\)\.csv',
        filename
    )
    if m2:
        return float(m2.group(1)), float(m2.group(2)), float(m2.group(3)), int(m2.group(4))

    # 更宽松：提取 T=xxs，同时取前三个数字为参数
    t_match = re.search(r'T=([0-9]+)s', filename)
    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', filename)
    if t_match and len(nums) >= 4:
        # 最后一个数字可能是 T，前三个数字作为参数
        T_sec = int(t_match.group(1))
        # 从开头找前三个浮点/整数作为参数
        nu_std = float(nums[0])
        kappa_std = float(nums[1])
        D_std = float(nums[2])
        return nu_std, kappa_std, D_std, T_sec

    return None, None, None, None


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8) * A ** 3) + term_gamma


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
            N_tau_points = len(unified_tau_grid)
            N_a_points = len(unified_a_grid)

            expected_len = 2 * N_tau_points * N_a_points
            if len(y_pred_np) != expected_len:
                raise ValueError(
                    f"模型输出长度不匹配：got {len(y_pred_np)}, "
                    f"expected {expected_len}=2*{N_tau_points}*{N_a_points}"
                )

            D1_field_pred = y_pred_np[:field_len_per_type].reshape(N_tau_points, N_a_points)
            D2_field_pred = y_pred_np[field_len_per_type:].reshape(N_tau_points, N_a_points)

            results_for_tau = {}

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                closest_tau_grid_idx = np.argmin(np.abs(unified_tau_grid - tau_sec_km))

                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                interp_d1 = interp1d(unified_a_grid, D1_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                interp_d2 = interp1d(unified_a_grid, D2_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)

                results_for_tau[int(tau_idx_km)] = {
                    'A': np.asarray(A_selected, dtype=float),
                    'D1': interp_d1(np.asarray(A_selected, dtype=float)),
                    'D2': interp_d2(np.asarray(A_selected, dtype=float))
                }

            return results_for_tau

    except Exception as e:
        log(f"[DeepONet] 预测失败 for params={params}: {e}")
        traceback.print_exc()
        return None


def objective_function(params, data_km, A_selected, tau_indices_map, model, scalers,
                       optimization_history_local, p_a_weights_map, unified_a_grid, unified_tau_grid):
    """
    优化目标函数：
      cost = weighted MSE + D<0 penalty

    若 USE_NORMALIZED_OBJECTIVE=True：
      使用代码一版本：
      ((D1_data - D1_pred) / std(D1_data))^2
      +
      ((D2_data - D2_pred) / std(D2_data))^2

    若 USE_NORMALIZED_OBJECTIVE=False：
      使用原信号长度脚本版本：
      (D1_data - D1_pred)^2 + (D2_data - D2_pred)^2
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
            tau_idx = int(tau_idx)
            if tau_idx not in data_km or tau_idx not in deeponet_km_results:
                continue

            D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
            D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)
            D1_pred = np.asarray(deeponet_km_results[tau_idx]['D1'], dtype=float)
            D2_pred = np.asarray(deeponet_km_results[tau_idx]['D2'], dtype=float)

            valid_mask = (~np.isnan(D1_data) & ~np.isnan(D1_pred) &
                          ~np.isnan(D2_data) & ~np.isnan(D2_pred))

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

        mean_weighted_sq_error = total_weighted_sq_error / num_compared_points if num_compared_points > 0 else 1e10
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
        log(f"[OPT] Eval={current_iter} | nu={nu:.6f}, kappa={kappa:.6f}, D={d_diffusion:.6f}, "
            f"mse={mean_weighted_sq_error:.6e}, penalty={d_negative_penalty:.3e}, total_cost={cost:.6e}")

    return cost if not (np.isnan(cost) or np.isinf(cost)) else 1e12


def plot_comparison(data_km, deeponet_results, theo_results, A_selected, tau_indices_map,
                    nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std, output_dir, title_suffix=""):
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
        if tau_idx not in data_km:
            continue

        A_data = data_km[tau_idx]['A']
        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']

        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(A_selected, deeponet_results[tau_idx]['D1'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d1.plot(A_selected, theo_results[tau_idx]['D1'], 'r--', label='Theoretical Std', linewidth=2)
        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at τ={tau_sec:.4f}s')
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)

        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(A_selected, deeponet_results[tau_idx]['D2'], 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d2.plot(A_selected, theo_results[tau_idx]['D2'], 'r--', label='Theoretical Std', linewidth=2)
        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at τ={tau_sec:.4f}s')
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=150)
    plt.close()


def save_plot_ready_curve_data(data_km, deeponet_results, theo_results, tau_indices_map, output_dir):
    records = []

    for tau_idx, tau_sec in sorted(tau_indices_map.items()):
        if tau_idx not in data_km:
            continue

        A_data = data_km[tau_idx]['A']
        D1_data = data_km[tau_idx]['D1']
        D2_data = data_km[tau_idx]['D2']

        d1_pred = deeponet_results[tau_idx]['D1'] if (deeponet_results and tau_idx in deeponet_results) else np.full_like(A_data, np.nan)
        d2_pred = deeponet_results[tau_idx]['D2'] if (deeponet_results and tau_idx in deeponet_results) else np.full_like(A_data, np.nan)

        d1_theory = theo_results[tau_idx]['D1'] if (theo_results and tau_idx in theo_results) else np.full_like(A_data, np.nan)
        d2_theory = theo_results[tau_idx]['D2'] if (theo_results and tau_idx in theo_results) else np.full_like(A_data, np.nan)

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


def process_km_file(km_file_path, deeponet_model, scalers, unified_a_grid, unified_tau_grid,
                    base_output_dir, input_dir_sim_base):
    log("")
    log(f"===== 开始处理 KM 文件: {km_file_path} =====")

    nu_std, kappa_std, D_std, signal_length_sec = parse_params_and_length_from_filename(km_file_path)
    if nu_std is None:
        log(f"[SKIP] 文件名解析失败: {os.path.basename(km_file_path)}")
        return None

    file_stem = os.path.splitext(os.path.basename(km_file_path))[0].replace('.', 'p').replace(',', '_')
    output_dir = os.path.join(base_output_dir, f"T_{signal_length_sec}s", file_stem)
    os.makedirs(output_dir, exist_ok=True)

    log(f"[META] 标准参数: nu={nu_std:.4f}, kappa={kappa_std:.4f}, D={D_std:.4f}, T={signal_length_sec}s")

    # 读取 KM 数据
    try:
        data_km_df = pd.read_csv(km_file_path)
        log(f"[LOAD] KM 文件读取成功，行数={len(data_km_df)}")

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

        log(f"[LOAD] tau 个数={len(tau_indices_map)}, A 点数={len(A_selected)}")
    except Exception as e:
        log(f"[ERROR] KM 数据读取失败: {e}")
        return None

    # 从对应观测数据中读取 p(A) 权重
    p_a_weights_map = {float(a): 1.0 / len(A_selected) for a in A_selected}
    sim_file_path = os.path.join(input_dir_sim_base, os.path.basename(km_file_path))
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
                log("[WEIGHT] 未找到 Envelope/Eta，退回均匀权重")

            if sim_envelope is not None and len(A_selected) > 0:
                bin_width = np.mean(np.diff(A_selected)) if len(A_selected) > 1 else 1.0
                bin_edges = np.concatenate([
                    [A_selected[0] - bin_width / 2],
                    np.array(A_selected[:-1]) + bin_width / 2,
                    [A_selected[-1] + bin_width / 2]
                ])
                hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
                hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

                if np.sum(hist) > 0:
                    p_a_weights_map = {float(A_selected[i]): float(hist[i]) for i in range(len(A_selected))}
                    log("[WEIGHT] p(A) 权重构建完成")
                else:
                    log("[WEIGHT] histogram 全零，退回均匀权重")
        except Exception as e:
            log(f"[WEIGHT] 权重构建失败，退回均匀权重: {e}")
    else:
        log("[WEIGHT] 未找到对应 sim 文件，退回均匀权重")

    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    params_0 = [0.1, 0.1, max(0.01, initial_d if not np.isnan(initial_d) else 0.01)]
    optimization_history = []

    log(f"[OPT] 初值: nu={params_0[0]:.4f}, kappa={params_0[1]:.4f}, D={params_0[2]:.4f}")
    log(f"[OPT] 开始 {OPTIMIZER_METHOD} 优化；normalized_objective={USE_NORMALIZED_OBJECTIVE}")

    start_time = time.time()
    result = minimize(
        objective_function, params_0,
        args=(finite_time_km, A_selected, tau_indices_map, deeponet_model, scalers,
              optimization_history, p_a_weights_map, unified_a_grid, unified_tau_grid),
        method=OPTIMIZER_METHOD,
        options=OPTIMIZER_OPTIONS
    )
    elapsed_time = time.time() - start_time

    if result.success:
        nu_opt, kappa_opt, d_opt = result.x
        log("[OPT] 优化成功")
    else:
        log("[OPT] 优化未正常收敛，改用历史最好点")
        best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
        nu_opt = optimization_history[best_idx]['nu']
        kappa_opt = optimization_history[best_idx]['kappa']
        d_opt = optimization_history[best_idx]['d_diffusion']

    d_opt = max(0.0, d_opt)

    log(f"[OPT] 完成 | 耗时={elapsed_time:.2f}s | "
        f"opt=(nu={nu_opt:.6f}, kappa={kappa_opt:.6f}, D={d_opt:.6f})")

    deeponet_opt_results = compute_deeponet_km_coefficients(
        [nu_opt, kappa_opt, d_opt], A_selected, tau_indices_map,
        deeponet_model, scalers, unified_a_grid, unified_tau_grid
    )

    theo_std_results = {}
    for tau_idx in tau_indices_map.keys():
        theo_std_results[tau_idx] = {
            'A': A_selected,
            'D1': theoretical_D1_d(A_selected, nu_std, kappa_std, D_std),
            'D2': theoretical_D2_d(A_selected, nu_std, kappa_std, D_std)
        }

    log("[PLOT] 开始保存 comparison_plot.png")
    plot_comparison(
        finite_time_km, deeponet_opt_results, theo_std_results,
        A_selected, tau_indices_map,
        nu_opt, kappa_opt, d_opt,
        nu_std, kappa_std, D_std,
        output_dir,
        title_suffix=f"(T={signal_length_sec}s)"
    )

    if optimization_history:
        history_df = pd.DataFrame(optimization_history)
        history_path = os.path.join(output_dir, 'optimization_history.csv')
        history_df.to_csv(history_path, index=False)
        log(f"[SAVE] 优化历史已保存: {history_path}")

    save_plot_ready_curve_data(finite_time_km, deeponet_opt_results, theo_std_results, tau_indices_map, output_dir)

    result_record = {
        'filename': os.path.basename(km_file_path),
        'signal_length_sec': signal_length_sec,

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
        'num_objective_evals': len(optimization_history),
        'final_objective': optimization_history[-1]['total_cost'] if optimization_history else np.nan,
        'normalized_objective': int(USE_NORMALIZED_OBJECTIVE)
    }

    result_path = os.path.join(output_dir, "single_case_summary.csv")
    pd.DataFrame([result_record]).to_csv(result_path, index=False)
    log(f"[SAVE] 单样本摘要已保存: {result_path}")

    log(f"===== 文件处理完成: {km_file_path} =====")
    return result_record


if __name__ == "__main__":
    log("===== 开始 Physics-Informed POD-DeepONet 信号长度参数辨识评估 =====")

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

        log(f"[MODEL] POD modes={num_pod_modes}, len(A-grid)={len(unified_a_grid)}, len(tau-grid)={len(unified_tau_grid)}")

        log("[MODEL] 构建 physics-informed PODDeepONet")
        deeponet_model = PODDeepONet(
            DEEPONET_BRANCH_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS,
            DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes,
            pod_basis,
            y_mean_pod_scaled,
            DEEPONET_DROPOUT_RATE
        )

        log("[MODEL] 加载 physics-informed 模型权重")
        deeponet_model.load_state_dict(safe_torch_load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        log("[MODEL] 模型加载成功")
    except Exception as e:
        log(f"[CRITICAL] 模型加载失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    all_results = []

    km_files = sorted([
        os.path.join(BASE_KM_DATA_DIR, f)
        for f in os.listdir(BASE_KM_DATA_DIR)
        if f.endswith('.csv')
    ])

    log(f"[SCAN] 待处理 KM 文件总数: {len(km_files)}")
    for i, p in enumerate(km_files, 1):
        log(f"[SCAN] 文件 {i:02d}: {p}")

    processed_counter = 0

    for km_file_path in km_files:
        processed_counter += 1
        log("")
        log(f"[BATCH] 全局进度 {processed_counter}/{len(km_files)} | 当前文件={km_file_path}")

        try:
            result = process_km_file(
                km_file_path, deeponet_model, scalers,
                unified_a_grid, unified_tau_grid,
                BASE_OUTPUT_DIR, INPUT_DIR_SIM_DATA
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            log(f"[ERROR] 当前文件处理失败: {e}")
            traceback.print_exc()

    if all_results:
        summary_df = pd.DataFrame(all_results)

        # 总表
        summary_all_path = os.path.join(BASE_OUTPUT_DIR, "batch_summary_all_cases_physics_informed.csv")
        summary_df.to_csv(summary_all_path, index=False)
        log(f"[SAVE] 总汇总已保存: {summary_all_path}")

        # 按信号长度统计
        summary_by_len = summary_df.groupby('signal_length_sec', dropna=False).agg({
            'nu_error_abs': ['mean', 'std', 'max'],
            'kappa_error_abs': ['mean', 'std', 'max'],
            'd_error_abs': ['mean', 'std', 'max'],
            'nu_error_rel': ['mean', 'std', 'max'],
            'kappa_error_rel': ['mean', 'std', 'max'],
            'd_error_rel': ['mean', 'std', 'max'],
            'optimization_time_sec': ['mean', 'std'],
            'final_objective': ['mean', 'std'],
            'num_objective_evals': ['mean', 'std']
        })
        summary_by_len.columns = ['_'.join(col) for col in summary_by_len.columns]
        summary_by_len = summary_by_len.reset_index()
        summary_by_len_path = os.path.join(BASE_OUTPUT_DIR, "summary_by_signal_length_physics_informed.csv")
        summary_by_len.to_csv(summary_by_len_path, index=False)
        log(f"[SAVE] 按信号长度汇总已保存: {summary_by_len_path}")

        # plot-ready 长表
        error_long = summary_df.melt(
            id_vars=['filename', 'signal_length_sec'],
            value_vars=[
                'nu_error_abs', 'kappa_error_abs', 'd_error_abs',
                'nu_error_rel', 'kappa_error_rel', 'd_error_rel'
            ],
            var_name='metric',
            value_name='value'
        )
        error_long_path = os.path.join(BASE_OUTPUT_DIR, "error_metrics_long_format_physics_informed.csv")
        error_long.to_csv(error_long_path, index=False)
        log(f"[SAVE] 长表误差数据已保存: {error_long_path}")

        log("===== Summary Statistics =====")
        log(f"总样本数: {len(summary_df)}")
        log(f"nu_error_abs 平均: {summary_df['nu_error_abs'].mean():.6e}")
        log(f"kappa_error_abs 平均: {summary_df['kappa_error_abs'].mean():.6e}")
        log(f"d_error_abs 平均: {summary_df['d_error_abs'].mean():.6e}")

    log("===== Step 3 完成 =====")