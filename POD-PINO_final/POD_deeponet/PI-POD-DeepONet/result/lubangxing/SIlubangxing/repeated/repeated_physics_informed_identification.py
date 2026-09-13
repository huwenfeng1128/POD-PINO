# -*- coding: utf-8 -*-
"""
PI-POD-DeepONet 参数辨识程序：重复实验
====================================

代码目的：
1）对 repeated_experiment 中 3 组参数 × 每组 5 次重复 的 KM 数据逐个做参数辨识；
2）实时详细输出当前辨识进度与优化过程；
3）保存优化历史、对比图、对比图数据、收敛图和收敛图数据；
4）汇总 15 个样本的辨识误差，便于后续画箱线图、误差条形图、散点图。

与原重复实验脚本相比：
- 数据目录和 group/repeat 文件名解析逻辑保持不变；
- 模型切换为 physics-informed POD-DeepONet；
- 使用 model_{RUN_ID}.pth + scalers_{RUN_ID}.pth；
- POD basis、POD mean、统一 A/tau 网格均从 scalers_{RUN_ID}.pth 读取；
- 不再依赖 pod_params_{RUN_ID}.pth；
- 默认使用代码一中的 D1/D2 归一化加权 MSE，可通过 USE_NORMALIZED_OBJECTIVE 切回原始加权 MSE。
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

# ---- PyTorch ----
import torch
import torch.nn as nn


# =========================================================
# 0. 路径配置
# =========================================================
ROOT_DIR = r"D:\PINN\zenodo\POD_deeponet\Revise_experiment\repeated_experiment"
BASE_KM_DATA_DIR = os.path.join(ROOT_DIR, "km_data")
INPUT_DIR_SIM_DATA = os.path.join(ROOT_DIR, "sim_data")
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\SIlubangxing\repeated\result'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ---- physics-informed 训练结果路径 ----
PI_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed'
RUN_ID = "pod_physics_informed_v2"

DEEPONET_MODEL_PATH = os.path.join(PI_RESULT_DIR, f'model_{RUN_ID}.pth')
DEEPONET_SCALER_PATH = os.path.join(PI_RESULT_DIR, f'scalers_{RUN_ID}.pth')

# 模型结构参数（需与 physics-informed 训练一致）
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
    'disp': True,
    'adaptive': True,
    'xatol': 1e-8,
    'fatol': 1e-8
}
D_NEGATIVE_PENALTY_FACTOR = 1e5
OBJECTIVE_LOG_FREQUENCY = 20

# True: 使用 physics-informed 代码一中的 D1/D2 归一化加权 MSE
# False: 使用原重复实验脚本中的原始加权 MSE
USE_NORMALIZED_OBJECTIVE = True


# =========================================================
# 1. 日志函数与通用转换
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
    """使用训练阶段保存好的 mean/std 对 branch 输入做标准化。"""
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
    解析重复实验文件名：
        group1_rep3_(4.50,2.00,4.50).csv
    """
    filename = os.path.basename(filepath)
    pattern = r"group(\d+)_rep(\d+)_\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)\)\.csv"
    m = re.match(pattern, filename)
    if m:
        return {
            "group_id": int(m.group(1)),
            "repeat_id": int(m.group(2)),
            "nu": float(m.group(3)),
            "kappa": float(m.group(4)),
            "D": float(m.group(5))
        }
    return None


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A, dtype=float)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    return (nu * A) - ((kappa / 8.0) * A ** 3) + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    return np.full_like(np.asarray(A, dtype=float), d_diffusion)


def compute_deeponet_km_coefficients(params, A_selected, tau_indices_map, model, scalers,
                                     unified_a_grid, unified_tau_grid):
    nu, kappa, d_diffusion = params

    if any(np.isnan(p) or np.isinf(p) for p in params):
        return None

    try:
        model.eval()
        with torch.no_grad():
            branch_input_np = np.array([[nu, kappa, d_diffusion]], dtype=np.float64)
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
            A_selected_arr = np.asarray(A_selected, dtype=float)

            for tau_idx_km, tau_sec_km in tau_indices_map.items():
                closest_tau_grid_idx = int(np.argmin(np.abs(unified_tau_grid - tau_sec_km)))
                D1_pred_slice = D1_field_pred[closest_tau_grid_idx, :]
                D2_pred_slice = D2_field_pred[closest_tau_grid_idx, :]

                interp_d1 = interp1d(unified_a_grid, D1_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                interp_d2 = interp1d(unified_a_grid, D2_pred_slice, kind='linear',
                                     bounds_error=False, fill_value=np.nan)

                results_for_tau[tau_idx_km] = {
                    'A': A_selected_arr,
                    'D1': interp_d1(A_selected_arr),
                    'D2': interp_d2(A_selected_arr)
                }
            return results_for_tau

    except Exception as e:
        log(f"[PRED-ERROR] params={params} -> DeepONet 预测失败: {e}")
        traceback.print_exc()
        return None


def build_pa_weights_map(sim_file_path, A_selected, output_dir=None):
    """从模拟信号中构造 p(A) 权重；若失败则退化为均匀权重。"""
    A_selected = np.asarray(A_selected, dtype=float)
    p_a_weights_map = {float(a): 1.0 / len(A_selected) for a in A_selected}

    if not os.path.exists(sim_file_path):
        log("[WEIGHT] 未找到对应 sim 文件，使用均匀权重")
        return p_a_weights_map

    try:
        log("[WEIGHT] 尝试根据模拟信号构造 p(A) 权重")
        df_sim = pd.read_csv(sim_file_path)

        if 'Eta' in df_sim.columns:
            sim_envelope = np.abs(hilbert(df_sim['Eta'].values))
            log("[WEIGHT] 使用 Eta 的 Hilbert 包络计算 p(A) 权重")
        elif 'Envelope' in df_sim.columns:
            sim_envelope = df_sim['Envelope'].values
            log("[WEIGHT] 使用 Envelope 计算 p(A) 权重")
        else:
            log("[WEIGHT] sim 文件没有 Eta/Envelope 列，使用均匀权重")
            return p_a_weights_map

        if len(A_selected) > 1:
            bin_width = float(np.mean(np.diff(A_selected)))
        else:
            bin_width = 1.0

        bin_edges = np.concatenate([
            [A_selected[0] - bin_width / 2],
            A_selected[:-1] + bin_width / 2,
            [A_selected[-1] + bin_width / 2]
        ])

        hist, _ = np.histogram(sim_envelope, bins=bin_edges, density=True)
        hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

        if np.sum(hist) > 0:
            p_a_weights_map = {float(A_selected[i]): float(hist[i]) for i in range(len(A_selected))}

            if output_dir is not None:
                pd.DataFrame({
                    "A": A_selected,
                    "p_A_weight": hist
                }).to_csv(os.path.join(output_dir, "pA_weights.csv"), index=False)
                log("[WEIGHT] p(A) 权重已保存")

        return p_a_weights_map

    except Exception as e:
        log(f"[WEIGHT-WARN] 构造 p(A) 权重失败，退回等权: {e}")
        return p_a_weights_map


# =========================================================
# 4. 目标函数
# =========================================================
def objective_function(params, data_km, A_selected, tau_indices_map, model, scalers,
                       optimization_history_local, p_a_weights_map, unified_a_grid, unified_tau_grid):
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


# =========================================================
# 5. 绘图与数据保存
# =========================================================
def save_convergence_plot_and_data(history_df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["iteration"], history_df["total_cost"], label="Total Cost")
    plt.plot(history_df["iteration"], history_df["mse"], label="MSE", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Convergence")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_convergence.png"), dpi=150)
    plt.close()

    history_df.to_csv(os.path.join(output_dir, "optimization_convergence_data.csv"), index=False)


def save_comparison_plot_and_data(data_km, deeponet_results, theo_results,
                                  A_selected, tau_indices_map, output_dir,
                                  nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std):
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(14, 5 * num_tau))
    if num_tau == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'Comparison - Repeated Experiment (Physics-Informed)\n'
        f'Opt: (ν={nu_opt:.4f}, κ={kappa_opt:.4f}, D={d_opt:.4f}) | '
        f'True: (ν={nu_std:.4f}, κ={kappa_std:.4f}, D={D_std:.4f})',
        fontsize=12
    )

    comparison_rows = []
    A_selected = np.asarray(A_selected, dtype=float)

    for i, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        if tau_idx not in data_km:
            continue

        A_data = np.asarray(data_km[tau_idx]['A'], dtype=float)
        D1_data = np.asarray(data_km[tau_idx]['D1'], dtype=float)
        D2_data = np.asarray(data_km[tau_idx]['D2'], dtype=float)

        D1_pred = (deeponet_results[tau_idx]['D1']
                   if (deeponet_results and tau_idx in deeponet_results)
                   else np.full_like(A_data, np.nan, dtype=float))
        D2_pred = (deeponet_results[tau_idx]['D2']
                   if (deeponet_results and tau_idx in deeponet_results)
                   else np.full_like(A_data, np.nan, dtype=float))
        D1_theory = (theo_results[tau_idx]['D1']
                     if (theo_results and tau_idx in theo_results)
                     else np.full_like(A_data, np.nan, dtype=float))
        D2_theory = (theo_results[tau_idx]['D2']
                     if (theo_results and tau_idx in theo_results)
                     else np.full_like(A_data, np.nan, dtype=float))

        ax_d1 = axes[i, 0]
        ax_d1.plot(A_data, D1_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d1.plot(A_selected, D1_pred, 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d1.plot(A_selected, D1_theory, 'r--', label='Theory True', linewidth=2)
        ax_d1.set_xlabel('A')
        ax_d1.set_ylabel('D1')
        ax_d1.set_title(f'D1 at tau={tau_sec:.6f}s')
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)

        ax_d2 = axes[i, 1]
        ax_d2.plot(A_data, D2_data, 'ko', label='KM Data', markersize=4)
        if deeponet_results and tau_idx in deeponet_results:
            ax_d2.plot(A_selected, D2_pred, 'b-', label='DeepONet Opt', linewidth=2)
        if theo_results and tau_idx in theo_results:
            ax_d2.plot(A_selected, D2_theory, 'r--', label='Theory True', linewidth=2)
        ax_d2.set_xlabel('A')
        ax_d2.set_ylabel('D2')
        ax_d2.set_title(f'D2 at tau={tau_sec:.6f}s')
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)

        for j, a_val in enumerate(A_data):
            comparison_rows.append({
                "tau_idx": tau_idx,
                "tau_sec": tau_sec,
                "A": a_val,
                "D1_data": D1_data[j],
                "D2_data": D2_data[j],
                "D1_pred_opt": D1_pred[j] if j < len(D1_pred) else np.nan,
                "D2_pred_opt": D2_pred[j] if j < len(D2_pred) else np.nan,
                "D1_theory_true": D1_theory[j] if j < len(D1_theory) else np.nan,
                "D2_theory_true": D2_theory[j] if j < len(D2_theory) else np.nan,
            })

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=150)
    plt.close()

    pd.DataFrame(comparison_rows).to_csv(os.path.join(output_dir, "comparison_plot_data.csv"), index=False)


def save_batch_summary_plots(summary_df, output_dir):
    """保存重复实验整体箱线图和 true-vs-identified 图。"""
    # 误差箱线图
    for metric in ["nu_error", "kappa_error", "d_error"]:
        plt.figure(figsize=(8, 6))
        summary_df.boxplot(column=metric, by="group_id")
        plt.title(f"{metric} by Group - Physics-Informed")
        plt.suptitle("")
        plt.xlabel("Group ID")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"), dpi=150)
        plt.close()

    # 每组 5 次重复的识别值散点图
    for param_true, param_opt, label in [
        ("nu_standard", "nu_optimized", "nu"),
        ("kappa_standard", "kappa_optimized", "kappa"),
        ("D_standard", "d_diffusion_optimized", "D"),
    ]:
        plt.figure(figsize=(8, 6))
        for gid in sorted(summary_df["group_id"].unique()):
            sub = summary_df[summary_df["group_id"] == gid]
            plt.scatter(sub[param_true], sub[param_opt], label=f"group {gid}", s=60)
        minv = min(summary_df[param_true].min(), summary_df[param_opt].min())
        maxv = max(summary_df[param_true].max(), summary_df[param_opt].max())
        plt.plot([minv, maxv], [minv, maxv], 'k--', label='y=x')
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Identified {label}")
        plt.title(f"True vs Identified {label} - Physics-Informed")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"true_vs_identified_{label}.png"), dpi=150)
        plt.close()


# =========================================================
# 6. 单文件参数辨识
# =========================================================
def process_km_file(km_file_path, deeponet_model, scalers, unified_a_grid, unified_tau_grid,
                    base_output_dir, input_dir_sim_data):
    log("")
    log("=" * 100)
    log(f"[FILE-START] 开始参数辨识: {os.path.basename(km_file_path)}")
    log("=" * 100)

    parsed = parse_params_from_filename(km_file_path)
    if parsed is None:
        log("[SKIP] 无法解析文件名，跳过")
        return None

    group_id = parsed["group_id"]
    repeat_id = parsed["repeat_id"]
    nu_std = parsed["nu"]
    kappa_std = parsed["kappa"]
    D_std = parsed["D"]

    file_output_dir_name = os.path.splitext(os.path.basename(km_file_path))[0]
    output_dir = os.path.join(base_output_dir, file_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        log("[LOAD] 读取 KM 数据")
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

        A_selected = np.array(sorted(data_km_df['A'].unique()), dtype=float)
        tau_indices_map = {
            int(idx): float(ts)
            for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values
        }
        log(f"[LOAD] 读取完成: tau数={len(tau_indices_map)}, A点数={len(A_selected)}")

    except Exception as e:
        log(f"[ERROR] KM 数据读取失败: {e}")
        traceback.print_exc()
        return None

    # 权重
    sim_file_path = os.path.join(input_dir_sim_data, os.path.basename(km_file_path))
    p_a_weights_map = build_pa_weights_map(sim_file_path, A_selected, output_dir=output_dir)

    # 初值
    initial_d = np.nanmean([np.nanmean(v['D2']) for v in finite_time_km.values()])
    params_0 = [0.1, 0.1, max(0.01, float(initial_d) if not np.isnan(initial_d) else 0.01)]
    optimization_history = []

    log(f"[OPT] 初值: nu={params_0[0]:.6f}, kappa={params_0[1]:.6f}, D={params_0[2]:.6f}")
    log(f"[OPT] 开始优化: {OPTIMIZER_METHOD}")

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
        log("[OPT] 优化未正常收敛，从历史中选择最佳点")
        if optimization_history:
            best_idx = int(np.argmin([h['total_cost'] for h in optimization_history]))
            nu_opt = optimization_history[best_idx]['nu']
            kappa_opt = optimization_history[best_idx]['kappa']
            d_opt = optimization_history[best_idx]['d_diffusion']
        else:
            nu_opt, kappa_opt, d_opt = params_0

    d_opt = max(0.0, float(d_opt))

    log(f"[OPT] 完成 | 用时={elapsed_time:.2f}s")
    log(f"[RESULT] 真值: nu={nu_std:.6f}, kappa={kappa_std:.6f}, D={D_std:.6f}")
    log(f"[RESULT] 识别: nu={nu_opt:.6f}, kappa={kappa_opt:.6f}, D={d_opt:.6f}")

    # 保存优化历史与收敛图
    if optimization_history:
        history_df = pd.DataFrame(optimization_history)
        history_df.to_csv(os.path.join(output_dir, 'optimization_history.csv'), index=False)
        save_convergence_plot_and_data(history_df, output_dir)
    else:
        history_df = pd.DataFrame()

    # 最优参数预测
    log("[PLOT] 生成最终对比结果")
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

    save_comparison_plot_and_data(
        finite_time_km, deeponet_opt_results, theo_std_results,
        A_selected, tau_indices_map, output_dir,
        nu_opt, kappa_opt, d_opt, nu_std, kappa_std, D_std
    )

    final_objective = optimization_history[-1]['total_cost'] if optimization_history else np.nan

    # 保存样本级 summary
    sample_result = {
        'filename': os.path.basename(km_file_path),
        'group_id': group_id,
        'repeat_id': repeat_id,
        'nu_standard': nu_std,
        'kappa_standard': kappa_std,
        'D_standard': D_std,
        'nu_optimized': nu_opt,
        'kappa_optimized': kappa_opt,
        'd_diffusion_optimized': d_opt,
        'nu_error': abs(nu_opt - nu_std),
        'kappa_error': abs(kappa_opt - kappa_std),
        'd_error': abs(d_opt - D_std),
        'nu_error_rel': abs(nu_opt - nu_std) / max(abs(nu_std), 1e-12),
        'kappa_error_rel': abs(kappa_opt - kappa_std) / max(abs(kappa_std), 1e-12),
        'd_error_rel': abs(d_opt - D_std) / max(abs(D_std), 1e-12),
        'optimization_time': elapsed_time,
        'optimizer_success': int(result.success),
        'final_fun': float(result.fun) if hasattr(result, 'fun') else np.nan,
        'final_objective': final_objective,
        'n_iterations_logged': len(optimization_history),
        'objective_normalized': int(USE_NORMALIZED_OBJECTIVE)
    }

    pd.DataFrame([sample_result]).to_csv(os.path.join(output_dir, "sample_identification_result.csv"), index=False)
    log(f"[SAVE] 样本级结果已保存: {output_dir}")

    return sample_result


# =========================================================
# 7. 主程序
# =========================================================
if __name__ == "__main__":
    log("")
    log("#" * 100)
    log("[MAIN] 开始加载 Physics-Informed POD-DeepONet 模型")
    log("#" * 100)

    try:
        log("[MODEL] 加载统一 scalers / POD / grid payload")
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
            raise KeyError(f"scalers payload missing keys: {missing}")

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

        log("[MODEL] 构建模型")
        deeponet_model = PODDeepONet(
            DEEPONET_BRANCH_INPUT_DIM,
            DEEPONET_HIDDEN_UNITS,
            DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes,
            pod_basis,
            y_mean_pod_scaled,
            DEEPONET_DROPOUT_RATE
        )

        log("[MODEL] 加载模型权重")
        deeponet_model.load_state_dict(torch.load(DEEPONET_MODEL_PATH, map_location=DEVICE))
        deeponet_model.to(DEVICE)
        deeponet_model.eval()

        log("[MODEL] 模型加载成功")

    except Exception as e:
        log(f"[CRITICAL] 模型加载失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)

    log("")
    log("#" * 100)
    log("[MAIN] 开始批量参数辨识")
    log("#" * 100)

    if not os.path.exists(BASE_KM_DATA_DIR):
        log(f"[MAIN-ERROR] KM 数据目录不存在: {BASE_KM_DATA_DIR}")
        raise SystemExit(1)

    km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    log(f"[MAIN] KM 文件总数: {len(km_files)}")

    all_results = []
    total_start = time.time()

    for idx, filename in enumerate(km_files, start=1):
        km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
        log("")
        log(f"[MAIN] 进度 {idx}/{len(km_files)} -> {filename}")

        try:
            result = process_km_file(
                km_file_path, deeponet_model, scalers,
                unified_a_grid, unified_tau_grid,
                BASE_OUTPUT_DIR, INPUT_DIR_SIM_DATA
            )
            if result:
                all_results.append(result)
        except Exception as e:
            log(f"[MAIN-ERROR] 处理文件失败: {filename} | {e}")
            traceback.print_exc()

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(BASE_OUTPUT_DIR, "batch_summary_repeated_experiment_physics_informed.csv")
        summary_df.to_csv(summary_path, index=False)

        # 组级统计
        group_stats = summary_df.groupby("group_id").agg({
            "nu_error": ["mean", "std", "min", "max"],
            "kappa_error": ["mean", "std", "min", "max"],
            "d_error": ["mean", "std", "min", "max"],
            "nu_error_rel": ["mean", "std", "min", "max"],
            "kappa_error_rel": ["mean", "std", "min", "max"],
            "d_error_rel": ["mean", "std", "min", "max"],
            "optimization_time": ["mean", "std"],
            "final_objective": ["mean", "std"],
            "n_iterations_logged": ["mean", "std"]
        })
        group_stats.columns = ["_".join(col) for col in group_stats.columns]
        group_stats = group_stats.reset_index()
        group_stats_path = os.path.join(BASE_OUTPUT_DIR, "group_statistics_physics_informed.csv")
        group_stats.to_csv(group_stats_path, index=False)

        # 可直接用于绘图的数据
        summary_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "boxplot_scatter_ready_data_physics_informed.csv"), index=False)

        # plot-ready 长表
        error_long = summary_df.melt(
            id_vars=['filename', 'group_id', 'repeat_id'],
            value_vars=[
                'nu_error', 'kappa_error', 'd_error',
                'nu_error_rel', 'kappa_error_rel', 'd_error_rel'
            ],
            var_name='metric',
            value_name='value'
        )
        error_long_path = os.path.join(BASE_OUTPUT_DIR, "error_metrics_long_format_physics_informed.csv")
        error_long.to_csv(error_long_path, index=False)

        save_batch_summary_plots(summary_df, BASE_OUTPUT_DIR)

        total_elapsed = time.time() - total_start

        log("")
        log("#" * 100)
        log("[MAIN] 所有参数辨识完成")
        log(f"[MAIN] 总用时: {total_elapsed:.2f} 秒")
        log(f"[MAIN] 汇总表: {summary_path}")
        log(f"[MAIN] 组统计表: {group_stats_path}")
        log(f"[MAIN] 长表误差数据: {error_long_path}")
        log(f"[MAIN] ν 平均绝对误差: {summary_df['nu_error'].mean():.6f}")
        log(f"[MAIN] κ 平均绝对误差: {summary_df['kappa_error'].mean():.6f}")
        log(f"[MAIN] D 平均绝对误差: {summary_df['d_error'].mean():.6f}")
        log("#" * 100)
    else:
        log("[MAIN-WARN] 没有成功得到任何参数辨识结果")
