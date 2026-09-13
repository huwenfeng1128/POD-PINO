# -*- coding: utf-8 -*-
"""
两个 POD-DeepONet 模型统一参数辨识对比
========================================

目标：
1. 对 no_pde_residual 与 with_pde_residual 两个模型；
2. 使用完全相同的 KM 数据、仿真权重、参数初值、优化器及停止条件；
3. 分别辨识 (nu, kappa, d_diffusion)；
4. 输出逐文件结果、优化历史、对比图以及总体统计，便于公平比较。

本程序沿用上一版参数辨识程序的基本思路：
候选参数 -> 模型预测完整 D1/D2 场 -> 选取/插值至 KM 网格 ->
按 p(A) 加权计算 D1 与 D2 误差 -> Nelder-Mead 反演参数。
"""

import os
import math
import time
import traceback
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import hilbert
from tqdm import tqdm

import torch
import torch.nn as nn


# =============================================================================
# 1. 路径与统一辨识配置
# =============================================================================

# 第二份训练程序的结果目录，目录下应包含：
# shared_preprocessing.pth
# no_pde_residual/best_model.pth
# with_pde_residual/best_model.pth
TRAIN_RESULT_DIR = (
    r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\xiaorongshiyan\train_result_physics_ablation_2'
)

SHARED_PREPROCESSING_PATH = os.path.join(
    TRAIN_RESULT_DIR, 'shared_preprocessing.pth'
)

MODEL_PATHS = {
    'no_pde_residual': os.path.join(
        TRAIN_RESULT_DIR, 'no_pde_residual', 'best_model.pth'
    ),
    'with_pde_residual': os.path.join(
        TRAIN_RESULT_DIR, 'with_pde_residual', 'best_model.pth'
    ),
}

# 与上一版参数辨识程序一致的数据目录
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'

BASE_OUTPUT_DIR = os.path.join(
    TRAIN_RESULT_DIR, 'parameter_identification_comparison'
)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 必须与训练程序一致
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
DROPOUT_RATE = 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

# 两个模型严格共用同一优化设置
OPTIMIZER_METHOD = 'Nelder-Mead'
OPTIMIZER_OPTIONS = {
    'maxiter': 3000,
    'disp': False,
    'adaptive': True,
    'xatol': 1e-8,
    'fatol': 1e-8,
}
D_NEGATIVE_PENALTY_FACTOR = 1e5

# 是否对 tau 方向也进行线性插值。
# True 比“最近 tau 点”更平滑，但两个模型始终使用相同设置。
INTERPOLATE_TAU = True

# 为了与上一版完全一致，默认仍按比较点数归一化。
# 若设为 True，则改为除以有效权重和。
NORMALIZE_BY_WEIGHT_SUM = False


# =============================================================================
# 2. 模型定义：必须与训练程序一致
# =============================================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE),
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        num_pod_modes: int,
        pod_basis: np.ndarray,
        y_mean_pod_scaled: np.ndarray,
        dropout_rate: float,
    ):
        super().__init__()
        self.branch = MLP(
            branch_input_dim,
            hidden_units,
            num_hidden_layers,
            num_pod_modes,
            dropout_rate,
        )
        self.register_buffer(
            'pod_basis', torch.as_tensor(pod_basis, dtype=DTYPE)
        )
        self.register_buffer(
            'y_mean_pod_scaled',
            torch.as_tensor(y_mean_pod_scaled, dtype=DTYPE),
        )

    def forward(self, branch_x_scaled: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(branch_x_scaled)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


# =============================================================================
# 3. 加载工具
# =============================================================================

def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def safe_load_state_dict(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def manual_scaler_transform(data, mean, std) -> torch.Tensor:
    data_t = torch.as_tensor(data, dtype=DTYPE, device=DEVICE)
    mean_t = torch.as_tensor(mean, dtype=DTYPE, device=DEVICE)
    std_t = torch.as_tensor(std, dtype=DTYPE, device=DEVICE).clone()
    std_t[std_t < 1e-10] = 1.0
    return (data_t - mean_t) / std_t


def load_shared_preprocessing(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'共享预处理文件不存在: {path}')

    payload = safe_torch_load(path)
    required = [
        'branch_mean', 'branch_std',
        'y_mean_scaler', 'y_std_scaler',
        'y_mean_pod_scaled', 'pod_basis',
        'actual_num_modes', 'unified_a_grid', 'unified_tau_grid',
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f'共享预处理文件缺少字段: {missing}')

    result = dict(payload)
    for key in required:
        if key != 'actual_num_modes':
            result[key] = as_numpy(result[key]).astype(np.float64)
    result['actual_num_modes'] = int(result['actual_num_modes'])
    return result


def build_and_load_models(shared: Dict[str, np.ndarray]) -> Dict[str, PODDeepONet]:
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'{model_name} 模型不存在: {model_path}')

        model = PODDeepONet(
            branch_input_dim=BRANCH_INPUT_DIM,
            hidden_units=HIDDEN_UNITS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_pod_modes=shared['actual_num_modes'],
            pod_basis=shared['pod_basis'],
            y_mean_pod_scaled=shared['y_mean_pod_scaled'],
            dropout_rate=DROPOUT_RATE,
        ).to(DEVICE)

        model.load_state_dict(safe_load_state_dict(model_path), strict=True)
        model.eval()
        models[model_name] = model
        print(f'✓ 已加载模型: {model_name}')

    return models


# =============================================================================
# 4. 数据与理论函数
# =============================================================================

def parse_params_from_filename(filepath: str):
    """从形如 '(0.1,0.2,0.3).csv' 的文件名提取标准参数。"""
    filename = os.path.basename(filepath)
    try:
        if filename.startswith('(') and filename.endswith('.csv'):
            values = filename[1:-5].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except (ValueError, IndexError):
        pass
    return None, None, None


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A, dtype=np.float64)
    term_gamma = np.zeros_like(A)
    mask = np.abs(A) > 1e-15
    term_gamma[mask] = d_diffusion / A[mask]
    return nu * A - (kappa / 8.0) * A ** 3 + term_gamma


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    return np.full_like(np.asarray(A, dtype=np.float64), d_diffusion)


def load_km_data(km_file_path: str):
    df = pd.read_csv(km_file_path)
    required = ['A', 'D1_data', 'D2_data', 'tau_index', 'tau_sec']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'KM 文件缺少列: {missing}')

    finite_time_km = {
        tau_idx: {
            'A': group['A'].to_numpy(dtype=np.float64),
            'D1': group['D1_data'].to_numpy(dtype=np.float64),
            'D2': group['D2_data'].to_numpy(dtype=np.float64),
            'tau_sec': float(group['tau_sec'].iloc[0]),
        }
        for tau_idx, group in df.groupby('tau_index')
    }

    A_selected = np.sort(df['A'].unique().astype(np.float64))
    tau_indices_map = {
        int(row.tau_index): float(row.tau_sec)
        for row in df[['tau_index', 'tau_sec']].drop_duplicates().itertuples(index=False)
    }
    return finite_time_km, A_selected, tau_indices_map


def build_probability_weights(
    km_file_path: str,
    A_selected: np.ndarray,
) -> Dict[float, float]:
    # 默认均匀权重
    weights = {float(a): 1.0 / len(A_selected) for a in A_selected}
    sim_file_path = os.path.join(
        INPUT_DIR_SIM_DATA, os.path.basename(km_file_path)
    )
    if not os.path.isfile(sim_file_path):
        return weights

    try:
        df_sim = pd.read_csv(sim_file_path)
        if 'Eta' in df_sim.columns:
            envelope = np.abs(hilbert(df_sim['Eta'].to_numpy(dtype=np.float64)))
        elif 'Envelope' in df_sim.columns:
            envelope = df_sim['Envelope'].to_numpy(dtype=np.float64)
        else:
            raise ValueError("仿真文件既没有 'Eta' 也没有 'Envelope' 列。")

        bin_width = (
            float(np.mean(np.diff(A_selected))) if len(A_selected) > 1 else 1.0
        )
        bin_edges = np.concatenate([
            [A_selected[0] - bin_width / 2.0],
            A_selected[:-1] + bin_width / 2.0,
            [A_selected[-1] + bin_width / 2.0],
        ])
        hist, _ = np.histogram(envelope, bins=bin_edges, density=True)
        if np.all(np.isfinite(hist)) and np.sum(hist) > 0:
            weights = {float(A_selected[i]): float(hist[i]) for i in range(len(A_selected))}
    except Exception as exc:
        print(f'  [警告] 无法读取仿真权重，使用均匀权重: {exc}')
    return weights


# =============================================================================
# 5. 正向预测：两个模型调用完全相同的函数
# =============================================================================

def interpolate_field_at_tau_and_a(
    field: np.ndarray,
    unified_tau_grid: np.ndarray,
    unified_a_grid: np.ndarray,
    tau_target: float,
    A_target: np.ndarray,
) -> np.ndarray:
    """先获得目标 tau 切片，再插值到目标 A 网格。"""
    if INTERPOLATE_TAU and len(unified_tau_grid) >= 2:
        # 对每个 A 点沿 tau 插值，得到目标 tau 下的一维 A 切片
        tau_interp = interp1d(
            unified_tau_grid,
            field,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        field_at_tau = np.asarray(tau_interp(tau_target), dtype=np.float64)
    else:
        closest_idx = int(np.argmin(np.abs(unified_tau_grid - tau_target)))
        field_at_tau = field[closest_idx, :]

    a_interp = interp1d(
        unified_a_grid,
        field_at_tau,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(a_interp(A_target), dtype=np.float64)


def compute_model_km_coefficients(
    params,
    A_selected,
    tau_indices_map,
    model,
    shared,
):
    if any(not np.isfinite(p) for p in params):
        return None

    try:
        branch_input = np.asarray([params], dtype=np.float64)
        branch_scaled = manual_scaler_transform(
            branch_input,
            shared['branch_mean'],
            shared['branch_std'],
        )

        model.eval()
        with torch.no_grad():
            y_scaled = model(branch_scaled)
            y_mean = torch.as_tensor(
                shared['y_mean_scaler'], dtype=DTYPE, device=DEVICE
            )
            y_std = torch.as_tensor(
                shared['y_std_scaler'], dtype=DTYPE, device=DEVICE
            )
            y_physical = y_scaled * y_std + y_mean

        y = y_physical.detach().cpu().numpy().reshape(-1)
        num_tau = len(shared['unified_tau_grid'])
        num_a = len(shared['unified_a_grid'])
        field_len = num_tau * num_a
        if len(y) != 2 * field_len:
            raise ValueError(
                f'模型输出长度 {len(y)} 与预期 {2 * field_len} 不一致。'
            )

        d1_field = y[:field_len].reshape(num_tau, num_a)
        d2_field = y[field_len:].reshape(num_tau, num_a)

        results = {}
        for tau_idx, tau_sec in tau_indices_map.items():
            results[tau_idx] = {
                'A': np.asarray(A_selected),
                'D1': interpolate_field_at_tau_and_a(
                    d1_field,
                    shared['unified_tau_grid'],
                    shared['unified_a_grid'],
                    tau_sec,
                    A_selected,
                ),
                'D2': interpolate_field_at_tau_and_a(
                    d2_field,
                    shared['unified_tau_grid'],
                    shared['unified_a_grid'],
                    tau_sec,
                    A_selected,
                ),
            }
        return results

    except Exception as exc:
        print(f'  模型预测失败，params={params}: {exc}')
        return None


# =============================================================================
# 6. 统一目标函数与优化
# =============================================================================

def objective_function(
    params,
    data_km,
    A_selected,
    tau_indices_map,
    model,
    shared,
    optimization_history,
    p_a_weights_map,
):
    nu, kappa, d_diffusion = [float(v) for v in params]

    penalty = (
        D_NEGATIVE_PENALTY_FACTOR * (-d_diffusion)
        if d_diffusion < 0.0 else 0.0
    )

    predictions = compute_model_km_coefficients(
        params, A_selected, tau_indices_map, model, shared
    )

    if predictions is None:
        fit_error = np.nan
        cost = 1e11 + penalty
        valid_count = 0
        weight_sum = 0.0
    else:
        weighted_error_sum = 0.0
        valid_count = 0
        weight_sum = 0.0

        for tau_idx in tau_indices_map:
            if tau_idx not in data_km or tau_idx not in predictions:
                continue

            obs = data_km[tau_idx]
            pred = predictions[tau_idx]
            valid = (
                np.isfinite(obs['D1']) & np.isfinite(obs['D2']) &
                np.isfinite(pred['D1']) & np.isfinite(pred['D2'])
            )
            if not np.any(valid):
                continue

            A_valid = obs['A'][valid]
            weights = np.asarray(
                [p_a_weights_map.get(float(a), 0.0) for a in A_valid],
                dtype=np.float64,
            )
            err_d1 = (obs['D1'][valid] - pred['D1'][valid]) ** 2
            err_d2 = (obs['D2'][valid] - pred['D2'][valid]) ** 2
            weighted_error_sum += float(np.sum(weights * (err_d1 + err_d2)))
            valid_count += int(np.sum(valid))
            weight_sum += float(np.sum(weights))

        denominator = weight_sum if NORMALIZE_BY_WEIGHT_SUM else valid_count
        fit_error = (
            weighted_error_sum / denominator
            if denominator > 0 else 1e10
        )
        cost = fit_error + penalty

    optimization_history.append({
        'evaluation': len(optimization_history) + 1,
        'nu': nu,
        'kappa': kappa,
        'd_diffusion': d_diffusion,
        'fit_error': fit_error,
        'negative_D_penalty': penalty,
        'total_cost': cost,
        'valid_point_count': valid_count,
        'weight_sum': weight_sum,
    })

    return float(cost) if np.isfinite(cost) else 1e12


def common_initial_guess(data_km) -> np.ndarray:
    d2_means = [np.nanmean(v['D2']) for v in data_km.values()]
    initial_d = np.nanmean(d2_means)
    if not np.isfinite(initial_d):
        initial_d = 0.01
    return np.asarray([0.1, 0.1, max(0.01, initial_d)], dtype=np.float64)


def identify_one_model(
    model_name: str,
    model: PODDeepONet,
    data_km,
    A_selected,
    tau_indices_map,
    shared,
    p_a_weights_map,
    params_0,
):
    history = []
    start = time.time()
    result = minimize(
        objective_function,
        x0=np.asarray(params_0, dtype=np.float64).copy(),
        args=(
            data_km,
            A_selected,
            tau_indices_map,
            model,
            shared,
            history,
            p_a_weights_map,
        ),
        method=OPTIMIZER_METHOD,
        options=dict(OPTIMIZER_OPTIONS),
    )
    elapsed = time.time() - start

    if result.success and np.all(np.isfinite(result.x)):
        raw_params = np.asarray(result.x, dtype=np.float64)
        source = 'scipy_result'
    elif history:
        best = min(history, key=lambda row: row['total_cost'])
        raw_params = np.asarray(
            [best['nu'], best['kappa'], best['d_diffusion']],
            dtype=np.float64,
        )
        source = 'best_history'
    else:
        raw_params = np.asarray(params_0, dtype=np.float64)
        source = 'initial_guess_fallback'

    final_params = raw_params.copy()
    final_params[2] = max(0.0, final_params[2])

    final_predictions = compute_model_km_coefficients(
        final_params, A_selected, tau_indices_map, model, shared
    )

    return {
        'model_name': model_name,
        'params': final_params,
        'raw_params': raw_params,
        'success': bool(result.success),
        'message': str(result.message),
        'nfev': int(getattr(result, 'nfev', len(history))),
        'nit': int(getattr(result, 'nit', -1)),
        'elapsed_time': elapsed,
        'selection_source': source,
        'history': history,
        'predictions': final_predictions,
        'final_objective_reported': float(result.fun) if np.isfinite(result.fun) else np.nan,
    }


# =============================================================================
# 7. 公平对比输出
# =============================================================================

def calculate_fit_metrics(data_km, predictions, tau_indices_map):
    d1_true, d1_pred, d2_true, d2_pred = [], [], [], []
    if predictions is None:
        return {
            'D1_rmse': np.nan, 'D1_mae': np.nan,
            'D2_rmse': np.nan, 'D2_mae': np.nan,
            'joint_rmse': np.nan,
        }

    for tau_idx in tau_indices_map:
        if tau_idx not in data_km or tau_idx not in predictions:
            continue
        obs, pred = data_km[tau_idx], predictions[tau_idx]
        valid = (
            np.isfinite(obs['D1']) & np.isfinite(obs['D2']) &
            np.isfinite(pred['D1']) & np.isfinite(pred['D2'])
        )
        d1_true.extend(obs['D1'][valid])
        d1_pred.extend(pred['D1'][valid])
        d2_true.extend(obs['D2'][valid])
        d2_pred.extend(pred['D2'][valid])

    if not d1_true:
        return {
            'D1_rmse': np.nan, 'D1_mae': np.nan,
            'D2_rmse': np.nan, 'D2_mae': np.nan,
            'joint_rmse': np.nan,
        }

    d1_true, d1_pred = np.asarray(d1_true), np.asarray(d1_pred)
    d2_true, d2_pred = np.asarray(d2_true), np.asarray(d2_pred)
    e1, e2 = d1_pred - d1_true, d2_pred - d2_true
    return {
        'D1_rmse': float(np.sqrt(np.mean(e1 ** 2))),
        'D1_mae': float(np.mean(np.abs(e1))),
        'D2_rmse': float(np.sqrt(np.mean(e2 ** 2))),
        'D2_mae': float(np.mean(np.abs(e2))),
        'joint_rmse': float(np.sqrt(np.mean(np.concatenate([e1, e2]) ** 2))),
    }


def plot_two_model_comparison(
    data_km,
    model_results,
    A_selected,
    tau_indices_map,
    standard_params,
    output_path,
):
    num_tau = len(tau_indices_map)
    fig, axes = plt.subplots(num_tau, 2, figsize=(15, 4.8 * num_tau))
    if num_tau == 1:
        axes = np.asarray(axes).reshape(1, 2)

    nu_std, kappa_std, d_std = standard_params
    fig.suptitle(
        'Unified Parameter Identification: no-PDE vs with-PDE\n'
        f'Standard: nu={nu_std:.5g}, kappa={kappa_std:.5g}, D={d_std:.5g}',
        fontsize=13,
    )

    line_styles = {
        'no_pde_residual': ('b-', 'POD-DeepONet (no PDE)'),
        'with_pde_residual': ('g-', 'PI-POD-DeepONet (with PDE)'),
    }

    for row, (tau_idx, tau_sec) in enumerate(sorted(tau_indices_map.items())):
        obs = data_km[tau_idx]
        ax1, ax2 = axes[row, 0], axes[row, 1]
        ax1.plot(obs['A'], obs['D1'], 'ko', markersize=4, label='KM data')
        ax2.plot(obs['A'], obs['D2'], 'ko', markersize=4, label='KM data')

        for model_name, result in model_results.items():
            preds = result['predictions']
            if preds is None or tau_idx not in preds:
                continue
            style, label = line_styles.get(model_name, ('-', model_name))
            p = result['params']
            full_label = f'{label}: ({p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g})'
            ax1.plot(A_selected, preds[tau_idx]['D1'], style, lw=2, label=full_label)
            ax2.plot(A_selected, preds[tau_idx]['D2'], style, lw=2, label=full_label)

        ax1.plot(
            A_selected,
            theoretical_D1_d(A_selected, nu_std, kappa_std, d_std),
            'r--', lw=1.8, label='Theoretical standard',
        )
        ax2.plot(
            A_selected,
            theoretical_D2_d(A_selected, nu_std, kappa_std, d_std),
            'r--', lw=1.8, label='Theoretical standard',
        )

        ax1.set_title(f'D1, tau={tau_sec:.6g} s')
        ax2.set_title(f'D2, tau={tau_sec:.6g} s')
        for ax, ylabel in [(ax1, 'D1'), (ax2, 'D2')]:
            ax.set_xlabel('A')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_optimization_history(model_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, result in model_results.items():
        history = result['history']
        if not history:
            continue
        x = [h['evaluation'] for h in history]
        y = np.maximum([h['total_cost'] for h in history], 1e-30)
        ax.semilogy(x, y, label=model_name)
    ax.set_xlabel('Objective evaluation')
    ax.set_ylabel('Total cost')
    ax.set_title('Unified optimization history')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_one_km_file(km_file_path, models, shared):
    filename = os.path.basename(km_file_path)
    print(f'\n--- 统一辨识: {filename} ---')

    nu_std, kappa_std, d_std = parse_params_from_filename(km_file_path)
    if nu_std is None:
        print('  [跳过] 无法从文件名解析标准参数。')
        return []

    output_dir = os.path.join(
        BASE_OUTPUT_DIR, os.path.splitext(filename)[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    data_km, A_selected, tau_indices_map = load_km_data(km_file_path)
    p_a_weights_map = build_probability_weights(km_file_path, A_selected)

    # 关键公平性：两个模型使用同一个初始点对象的副本
    params_0 = common_initial_guess(data_km)
    print(
        f'  统一初值: nu={params_0[0]:.6g}, '
        f'kappa={params_0[1]:.6g}, D={params_0[2]:.6g}'
    )

    model_results = {}
    records = []

    for model_name, model in models.items():
        result = identify_one_model(
            model_name=model_name,
            model=model,
            data_km=data_km,
            A_selected=A_selected,
            tau_indices_map=tau_indices_map,
            shared=shared,
            p_a_weights_map=p_a_weights_map,
            params_0=params_0,
        )
        model_results[model_name] = result

        history_path = os.path.join(
            output_dir, f'optimization_history_{model_name}.csv'
        )
        pd.DataFrame(result['history']).to_csv(history_path, index=False)

        metrics = calculate_fit_metrics(
            data_km, result['predictions'], tau_indices_map
        )
        p = result['params']
        record = {
            'filename': filename,
            'model': model_name,
            'nu_standard': nu_std,
            'kappa_standard': kappa_std,
            'D_standard': d_std,
            'nu_optimized': p[0],
            'kappa_optimized': p[1],
            'd_diffusion_optimized': p[2],
            'nu_abs_error': abs(p[0] - nu_std),
            'kappa_abs_error': abs(p[1] - kappa_std),
            'D_abs_error': abs(p[2] - d_std),
            'optimizer_success': result['success'],
            'optimizer_message': result['message'],
            'selection_source': result['selection_source'],
            'nfev': result['nfev'],
            'nit': result['nit'],
            'optimization_time': result['elapsed_time'],
            'initial_nu': params_0[0],
            'initial_kappa': params_0[1],
            'initial_D': params_0[2],
            **metrics,
        }
        records.append(record)
        print(
            f"  {model_name}: nu={p[0]:.6g}, kappa={p[1]:.6g}, "
            f"D={p[2]:.6g}, joint RMSE={metrics['joint_rmse']:.4e}"
        )

    pd.DataFrame(records).to_csv(
        os.path.join(output_dir, 'two_model_identification_result.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    plot_two_model_comparison(
        data_km=data_km,
        model_results=model_results,
        A_selected=A_selected,
        tau_indices_map=tau_indices_map,
        standard_params=(nu_std, kappa_std, d_std),
        output_path=os.path.join(output_dir, 'two_model_comparison.png'),
    )
    plot_optimization_history(
        model_results,
        os.path.join(output_dir, 'optimization_history_comparison.png'),
    )
    return records


def save_global_summaries(all_records):
    if not all_records:
        print('没有成功完成的辨识结果。')
        return

    df = pd.DataFrame(all_records)
    df.to_csv(
        os.path.join(BASE_OUTPUT_DIR, 'all_identification_results.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    metric_cols = [
        'nu_abs_error', 'kappa_abs_error', 'D_abs_error',
        'D1_rmse', 'D1_mae', 'D2_rmse', 'D2_mae',
        'joint_rmse', 'optimization_time',
    ]
    summary = df.groupby('model')[metric_cols].agg(['mean', 'median', 'std'])
    summary.to_csv(
        os.path.join(BASE_OUTPUT_DIR, 'model_comparison_summary.csv'),
        encoding='utf-8-sig',
    )

    # 每个文件内直接比较两个模型，正值表示 with-PDE 的误差更小
    pivot = df.pivot(index='filename', columns='model', values=metric_cols)
    pair_records = []
    for filename in pivot.index:
        row = {'filename': filename}
        for metric in metric_cols:
            try:
                no_pde = pivot.loc[filename, (metric, 'no_pde_residual')]
                with_pde = pivot.loc[filename, (metric, 'with_pde_residual')]
                row[f'{metric}_no_pde'] = no_pde
                row[f'{metric}_with_pde'] = with_pde
                row[f'{metric}_improvement_with_pde'] = no_pde - with_pde
            except KeyError:
                continue
        pair_records.append(row)

    pair_df = pd.DataFrame(pair_records)
    pair_df.to_csv(
        os.path.join(BASE_OUTPUT_DIR, 'paired_model_comparison.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print('\n=== 两模型总体辨识对比 ===')
    for model_name, group in df.groupby('model'):
        print(f'\n{model_name}')
        print(f"  nu MAE    : {group['nu_abs_error'].mean():.6e}")
        print(f"  kappa MAE : {group['kappa_abs_error'].mean():.6e}")
        print(f"  D MAE     : {group['D_abs_error'].mean():.6e}")
        print(f"  joint RMSE: {group['joint_rmse'].mean():.6e}")
        print(f"  mean time : {group['optimization_time'].mean():.3f} s")


# =============================================================================
# 8. 主程序
# =============================================================================

def main():
    print('=' * 88)
    print('Two-model unified parameter identification')
    print('=' * 88)
    print(f'Device: {DEVICE}')
    print(f'KM data: {BASE_KM_DATA_DIR}')
    print(f'Output : {BASE_OUTPUT_DIR}')

    shared = load_shared_preprocessing(SHARED_PREPROCESSING_PATH)
    print(f"POD modes: {shared['actual_num_modes']}")
    print(f"A-grid   : {len(shared['unified_a_grid'])}")
    print(f"tau-grid : {len(shared['unified_tau_grid'])}")

    models = build_and_load_models(shared)

    if not os.path.isdir(BASE_KM_DATA_DIR):
        raise FileNotFoundError(f'KM 数据目录不存在: {BASE_KM_DATA_DIR}')

    km_files = sorted(
        os.path.join(BASE_KM_DATA_DIR, f)
        for f in os.listdir(BASE_KM_DATA_DIR)
        if f.lower().endswith('.csv')
    )
    print(f'待处理 KM 文件数: {len(km_files)}')

    all_records = []
    failed = []
    for km_file_path in tqdm(km_files, desc='KM files'):
        try:
            all_records.extend(
                process_one_km_file(km_file_path, models, shared)
            )
        except Exception as exc:
            failed.append({
                'file': km_file_path,
                'error': str(exc),
                'traceback': traceback.format_exc(),
            })
            print(f'\n[失败] {km_file_path}: {exc}')

    if failed:
        pd.DataFrame(failed).to_csv(
            os.path.join(BASE_OUTPUT_DIR, 'failed_identification_files.csv'),
            index=False,
            encoding='utf-8-sig',
        )

    save_global_summaries(all_records)
    print('\n=== 参数辨识对比完成 ===')


if __name__ == '__main__':
    main()
