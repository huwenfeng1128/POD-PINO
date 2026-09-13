# -*- coding: utf-8 -*-
"""
POD-DeepONet + Physics-Informed Diagnostics + 自动对照实验版
==========================================================

本脚本目标
----------
这个版本不是只训练一个模型，而是自动跑两组实验并进行对照：

1) data_only
   - 纯数据驱动
   - PHYSICS_LOSS_WEIGHT = 0.0

2) physics_informed
   - 数据 + 物理约束
   - PHYSICS_LOSS_WEIGHT = 0.1

这样做的目的，是帮助你回答一个非常关键的问题：
“物理信息约束到底有没有真的发挥作用，而不是只是数据拟合在主导？”

本脚本会自动记录并输出：
------------------------
一、常规训练与验证指标
- train_total_loss
- val_total_loss
- val_data_loss
- val_d1_data
- val_d2_data
- val_raw_pos_loss
- val_raw_pde_loss
- val_raw_pde_r1
- val_raw_pde_r2

二、真正进入总损失的“加权贡献”
- weighted_pos
- weighted_pde
- weighted_physics_total
- data / pos / pde 在 total loss 中的占比

三、梯度贡献审计（最关键）
- grad_norm_data_loss
- grad_norm_weighted_pos
- grad_norm_weighted_pde
- grad_norm_total_loss
- ratio_grad_pde_over_data
- ratio_grad_pos_over_data
- ratio_grad_*_over_total

这些量可以帮助你判断：
- PDE 项是不是只是数值上存在，但几乎不影响参数更新
- 还是它真的在推着模型学

四、样本级统计
- 每个验证样本的 RMSE(D1), RMSE(D2)
- mean |R1|, mean |R2|
- max |R1|, max |R2|
- D2 负值比例
- sample-level weighted physics

五、网格级统计
- 验证集所有样本上的 |R1| / |R2| 的 mean heatmap / std heatmap
- 正性违约 heatmap

六、自动对照图
- 两组实验的 val_data_loss 对比
- 两组实验的 val_raw_pde_loss 对比
- 两组实验的 weighted_pde 对比
- 两组实验的 grad_ratio_pde_over_data 对比
- 两组实验的样本级分布对比
- 两组实验的最优结果汇总

重要说明
--------
1. 当前实现默认 A 网格、tau 网格是统一且均匀的。
2. 物理损失采用你已经接受的层级 PDE 形式：
   R1 = dU1/dtau - (D1_th*dU1/da + D2_th*d2U1/da2) - D1_th
   R2 = dU2/dtau - (D1_th*dU2/da + D2_th*d2U2/da2) - D1_th*U1 - 2*D2_th*dU1/da - D2_th
   其中 U1 = tau * D1_pred, U2 = tau * D2_pred

3. 你如果要真正“严格证明 physics 有独立贡献”，最关键看的不是 raw PDE loss，
   而是：
   - 加权后贡献 weighted_pde
   - 梯度范数比值 ratio_grad_pde_over_data
   - 自动对照实验结果

4. 这个版本为了强调诊断与可解释性，代码会比普通训练脚本长很多。
"""

import os
import glob
import math
import json
import random
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR


# =============================================================================
# 1. 全局总配置
# =============================================================================

# -------------------------
# 你自己的数据路径
# -------------------------
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA'

# -------------------------
# 总输出目录
# -------------------------
MASTER_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_compare_physics_3'
os.makedirs(MASTER_RESULT_DIR, exist_ok=True)

# -------------------------
# 总运行 ID
# -------------------------
MASTER_RUN_ID = 'pod_compare_data_only_vs_physics_v1'

# -------------------------
# 设备与精度
# -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

# -------------------------
# 数据与训练公共配置
# -------------------------
GLOBAL_CONFIG = {
    'BRANCH_INPUT_DIM': 3,
    'HIDDEN_UNITS': 128,
    'NUM_HIDDEN_LAYERS': 4,
    'REQUESTED_NUM_POD_MODES': 100,
    'DROPOUT_RATE': 0.1,
    'WEIGHT_DECAY': 1e-6,

    'ADAM_LR': 3e-4,
    'ADAM_BATCH_SIZE': 64,
    'ADAM_ITERATIONS': 100000,
    'D2_LOSS_WEIGHT': 0.2,
    'PDE_IN_TOTAL_WEIGHT': 0.01,

    'VALIDATION_SPLIT': 0.2,
    'SEED': 24,
    'VALIDATION_FREQUENCY': 200,
    'EARLY_STOPPING_PATIENCE': 5,
    'WARMUP_STEPS': 5000,
    'TARGET_NUM_FILES': 2500,

    'GRAD_AUDIT_EVERY': 200,
    'GRAD_AUDIT_MAX_SAMPLES': 64,
    'NUM_VIS_SAMPLES': 4,
}

# -------------------------
# 自动对照实验配置
# -------------------------
EXPERIMENTS = [
    {
        'exp_name': 'data_only',
        'physics_loss_weight': 0.0
    },
    {
        'exp_name': 'physics_informed',
        'physics_loss_weight': 0.1
    }
]


# =============================================================================
# 2. 通用工具函数
# =============================================================================

def set_seed(seed: int = 24):
    """固定随机种子，提升可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_required_columns(df: pd.DataFrame, required_cols, file_path: str):
    """检查 CSV 是否包含必要字段。"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'文件 {file_path} 缺少必要列: {missing}')


def is_uniform_grid(arr: np.ndarray, tol: float = 1e-12) -> bool:
    """检查 1D 网格是否均匀。"""
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], atol=tol, rtol=tol)


def to_float(x):
    """把 torch scalar / numpy scalar 安全转成 float。"""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def compute_grad_norm(model: nn.Module) -> float:
    """计算当前模型参数梯度的 L2 范数。"""
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sq_sum += float(torch.sum(g * g).item())
    return math.sqrt(max(sq_sum, 0.0))


def save_json(path: str, payload: dict):
    """保存 JSON。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text(path: str, text: str):
    """保存 txt。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def safe_div(a, b, eps=1e-30):
    """安全除法。"""
    return a / max(b, eps)


def get_exp_result_dir(master_dir: str, exp_name: str) -> str:
    """每组实验单独的结果目录。"""
    d = os.path.join(master_dir, exp_name)
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# 3. 数据加载与预处理
# =============================================================================

def load_unified_data(data_dir, target_num_files):
    """
    从 data_dir 中加载 data_*.csv 文件，要求：
    - 所有文件的 A 网格一致
    - 所有文件的 tau 网格一致
    - 都是完整 tensor-product 网格
    - 且 A/tau 都是均匀网格

    返回
    ----
    branch_inputs_np : [N, 3]
        每个样本的输入参数 [nu, kappa, d_diffusion]

    y_snapshots_np : [N, 2 * (num_tau * num_a)]
        输出快照：[D1_flat, D2_flat]

    unified_a_grid : [num_a]
    unified_tau_grid : [num_tau]
    """
    print(f'从 {data_dir} 加载统一数据...')

    all_available_files = glob.glob(os.path.join(data_dir, 'data_*.csv'))
    if not all_available_files:
        raise FileNotFoundError(f'在 {data_dir} 中未找到任何 data_*.csv 文件。')

    selected_files = random.sample(
        all_available_files,
        min(len(all_available_files), target_num_files)
    )

    branch_inputs_list = []
    y_snapshots_list = []

    unified_a_grid = None
    unified_tau_grid = None

    required_cols = ['A', 'tau', 'nu', 'kappa', 'd_diffusion', 'D1_adj', 'D2_adj']
    num_failed = 0

    for f in tqdm(selected_files, desc='加载快照'):
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            ensure_required_columns(df, required_cols, f)

            # 删除关键输出为 NaN 的行
            df = df.dropna(subset=['D1_adj', 'D2_adj'])
            if df.empty:
                raise ValueError('删除缺失值后为空表。')

            a_grid = np.sort(df['A'].unique())
            tau_grid = np.sort(df['tau'].unique())

            if len(a_grid) < 3 or len(tau_grid) < 3:
                raise ValueError('A 和 tau 的网格点数都至少需要 >= 3。')

            if unified_a_grid is None:
                unified_a_grid = a_grid
                unified_tau_grid = tau_grid
            else:
                if not np.array_equal(a_grid, unified_a_grid):
                    raise ValueError('A 网格与统一网格不一致。')
                if not np.array_equal(tau_grid, unified_tau_grid):
                    raise ValueError('tau 网格与统一网格不一致。')

            if not is_uniform_grid(a_grid):
                raise ValueError('A 网格不是均匀网格。')
            if not is_uniform_grid(tau_grid):
                raise ValueError('tau 网格不是均匀网格。')

            # 按 (tau, A) 排序，确保 flatten 顺序固定
            df_sorted = df.sort_values(by=['tau', 'A']).reset_index(drop=True)

            expected_len = len(unified_a_grid) * len(unified_tau_grid)
            if len(df_sorted) != expected_len:
                raise ValueError(f'网格点数不完整或有重复。期望 {expected_len}，实际 {len(df_sorted)}。')

            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values.astype(np.float64)

            final_snapshot = np.concatenate([
                df_sorted['D1_adj'].values.astype(np.float64),
                df_sorted['D2_adj'].values.astype(np.float64)
            ])

            branch_inputs_list.append(params)
            y_snapshots_list.append(final_snapshot)

        except Exception as e:
            num_failed += 1
            print(f'\n[警告] 文件读取失败: {f}')
            print(f'原因: {e}')

    if len(branch_inputs_list) == 0:
        raise ValueError('没有成功加载任何有效样本。')

    branch_inputs_np = np.array(branch_inputs_list, dtype=np.float64)
    y_snapshots_np = np.array(y_snapshots_list, dtype=np.float64)

    print(f'\n成功加载样本数: {len(branch_inputs_np)}')
    print(f'失败文件数: {num_failed}')
    print(f'A 网格点数: {len(unified_a_grid)}, tau 网格点数: {len(unified_tau_grid)}')
    print(f'输出场维数: {y_snapshots_np.shape[1]}')

    return (
        branch_inputs_np,
        y_snapshots_np,
        np.array(unified_a_grid, dtype=np.float64),
        np.array(unified_tau_grid, dtype=np.float64)
    )


def manual_scaler(data, mean=None, std=None):
    """
    手工标准化。
    若 mean/std 未提供，则根据当前 data 估计。
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def inverse_manual_scaler(data_scaled, mean, std):
    """标准化逆变换。"""
    return data_scaled * std + mean


def pod(y_data_scaled, requested_num_modes):
    """
    对输出场做 POD(SVD)。
    """
    y_mean_pod_scaled = np.mean(y_data_scaled, axis=0)
    centered = y_data_scaled - y_mean_pod_scaled
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    actual_num_modes = min(requested_num_modes, Vt.shape[0])
    pod_basis = Vt.T[:, :actual_num_modes]
    return y_mean_pod_scaled, pod_basis, S, actual_num_modes


# =============================================================================
# 4. 模型定义
# =============================================================================

class MLP(nn.Module):
    """
    多层感知机：
        输入参数 -> POD 系数
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
    POD-DeepONet 的简化形式：
        branch_x -> MLP -> POD coefficients
        POD coefficients @ POD basis^T + POD mean
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
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
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
# 5. 物理损失定义
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    这个类负责：
    1. 数据损失
    2. D2 非负约束
    3. PDE 残差
    4. 各种“可审计”的拆分量

    为什么要拆这么细？
    ----------------
    因为你真正想知道的是：
    - PDE 项有没有被算
    - PDE 项有没有进 total loss
    - PDE 项有没有产生实际梯度
    - PDE 项是否改变了模型行为

    所以必须同时记录：
    - raw physics loss
    - weighted physics contribution
    - gradient norms
    """
    def __init__(self,
                 unified_a_grid,
                 unified_tau_grid,
                 branch_mean,
                 branch_std,
                 y_mean_scaler,
                 y_std_scaler,
                 d2_weight=0.5,
                 phys_weight=0.1,
                 pde_in_total_weight=0.001):
        super().__init__()

        self.d2_weight = d2_weight
        self.phys_weight = phys_weight
        self.pde_in_total_weight = pde_in_total_weight
        self.mse = nn.MSELoss()

        self.a_grid = torch.tensor(unified_a_grid, dtype=DTYPE, device=DEVICE)
        self.tau_grid = torch.tensor(unified_tau_grid, dtype=DTYPE, device=DEVICE)

        self.num_a = len(unified_a_grid)
        self.num_tau = len(unified_tau_grid)

        if self.num_a < 3 or self.num_tau < 3:
            raise ValueError('中心差分要求 A 和 tau 的网格点数至少为 3。')

        self.da = self.a_grid[1] - self.a_grid[0]
        self.dtau = self.tau_grid[1] - self.tau_grid[0]

        self.branch_mean = torch.tensor(branch_mean, dtype=DTYPE, device=DEVICE)
        self.branch_std = torch.tensor(branch_std, dtype=DTYPE, device=DEVICE)
        self.y_mean_scaler = torch.tensor(y_mean_scaler, dtype=DTYPE, device=DEVICE)
        self.y_std_scaler = torch.tensor(y_std_scaler, dtype=DTYPE, device=DEVICE)

    def decompose_prediction(self, y_pred_scaled, branch_x_scaled):
        """
        把网络输出拆成物理空间中的 D1_pred / D2_pred，并构造理论系数 D1_th / D2_th。
        """
        field_len = self.num_a * self.num_tau

        # 反标准化到真实物理空间
        y_pred = y_pred_scaled * self.y_std_scaler + self.y_mean_scaler
        branch_x = branch_x_scaled * self.branch_std + self.branch_mean

        D1_pred = y_pred[:, :field_len].view(-1, self.num_tau, self.num_a)
        D2_pred = y_pred[:, field_len:].view(-1, self.num_tau, self.num_a)

        nu = branch_x[:, 0].view(-1, 1, 1)
        kappa = branch_x[:, 1].view(-1, 1, 1)
        d_diff = branch_x[:, 2].view(-1, 1, 1)

        A = self.a_grid.view(1, 1, -1)
        Tau = self.tau_grid.view(1, -1, 1)

        # 避免 d/a 在 a=0 奇异
        A_safe = torch.clamp(A, min=1e-9)

        # 理论生成元系数
        D1_th = nu * A - (kappa / 8.0) * (A ** 3) + (d_diff / A_safe)
        D2_th = d_diff.expand_as(D1_th)

        return {
            'field_len': field_len,
            'D1_pred': D1_pred,
            'D2_pred': D2_pred,
            'D1_th': D1_th,
            'D2_th': D2_th,
            'A': A,
            'Tau': Tau,
        }

    def compute_residual_maps(self, y_pred_scaled, branch_x_scaled):
        """
        构造 PDE 残差 R1 / R2 以及 D2 非负违约图。
        """
        parts = self.decompose_prediction(y_pred_scaled, branch_x_scaled)

        D1_pred = parts['D1_pred']
        D2_pred = parts['D2_pred']
        D1_th = parts['D1_th']
        D2_th = parts['D2_th']
        Tau = parts['Tau']

        # U1 = tau * D1_tau, U2 = tau * D2_tau
        U1 = Tau * D1_pred
        U2 = Tau * D2_pred

        # 只在内部点上做中心差分，剔除边界
        U1_in = U1[:, 1:-1, 1:-1]
        U2_in = U2[:, 1:-1, 1:-1]
        D1_th_in = D1_th[:, :, 1:-1]
        D2_th_in = D2_th[:, :, 1:-1]

        dU1_dtau = (U1[:, 2:, 1:-1] - U1[:, :-2, 1:-1]) / (2.0 * self.dtau)
        dU2_dtau = (U2[:, 2:, 1:-1] - U2[:, :-2, 1:-1]) / (2.0 * self.dtau)

        dU1_da = (U1[:, 1:-1, 2:] - U1[:, 1:-1, :-2]) / (2.0 * self.da)
        dU2_da = (U2[:, 1:-1, 2:] - U2[:, 1:-1, :-2]) / (2.0 * self.da)

        d2U1_da2 = (
            U1[:, 1:-1, 2:] - 2.0 * U1[:, 1:-1, 1:-1] + U1[:, 1:-1, :-2]
        ) / (self.da ** 2)

        d2U2_da2 = (
            U2[:, 1:-1, 2:] - 2.0 * U2[:, 1:-1, 1:-1] + U2[:, 1:-1, :-2]
        ) / (self.da ** 2)

        # 一阶层级 PDE 残差
        R1 = dU1_dtau - (D1_th_in * dU1_da + D2_th_in * d2U1_da2) - D1_th_in

        # 二阶层级 PDE 残差
        R2 = (
            dU2_dtau
            - (D1_th_in * dU2_da + D2_th_in * d2U2_da2)
            - (D1_th_in * U1_in)
            - 2.0 * D2_th_in * dU1_da
            - D2_th_in
        )

        # D2 非负约束
        pos_violation = torch.relu(-D2_pred)

        return {
            'R1': R1,
            'R2': R2,
            'pos_violation': pos_violation,
            'D1_pred': D1_pred,
            'D2_pred': D2_pred,
            'D1_th': D1_th,
            'D2_th': D2_th,
        }

    def compute_all_losses(self, y_pred_scaled, y_true_scaled, branch_x_scaled):
        """
        计算所有损失、各项占比，以及原始残差图。
        """
        field_len = self.num_a * self.num_tau

        # ---------------------------------------------------------------------
        # A. 数据损失（标准化空间）
        # ---------------------------------------------------------------------
        loss_d1_data = self.mse(y_pred_scaled[:, :field_len], y_true_scaled[:, :field_len])
        loss_d2_data = self.mse(y_pred_scaled[:, field_len:], y_true_scaled[:, field_len:])

        data_loss = (1.0 - self.d2_weight) * loss_d1_data + self.d2_weight * loss_d2_data

        # ---------------------------------------------------------------------
        # B. 物理残差（真实物理空间）
        # ---------------------------------------------------------------------
        residual_parts = self.compute_residual_maps(y_pred_scaled, branch_x_scaled)
        R1 = residual_parts['R1']
        R2 = residual_parts['R2']
        pos_violation = residual_parts['pos_violation']

        loss_pos = torch.mean(pos_violation ** 2)
        loss_pde_r1 = torch.mean(R1 ** 2)
        loss_pde_r2 = torch.mean(R2 ** 2)
        loss_pde = loss_pde_r1 + loss_pde_r2

        # ---------------------------------------------------------------------
        # C. 真正进入 total loss 的“加权后贡献”
        # ---------------------------------------------------------------------
        weighted_pos = self.phys_weight * loss_pos
        weighted_pde = self.phys_weight * self.pde_in_total_weight * loss_pde
        weighted_physics_total = weighted_pos + weighted_pde

        total_loss = data_loss + weighted_physics_total

        # ---------------------------------------------------------------------
        # D. 各项在 total loss 中的占比
        # ---------------------------------------------------------------------
        denom = total_loss.detach() + 1e-30
        ratio_data_in_total = data_loss / denom
        ratio_weighted_pos_in_total = weighted_pos / denom
        ratio_weighted_pde_in_total = weighted_pde / denom

        return {
            'total_loss': total_loss,

            'data_loss': data_loss,
            'loss_d1_data': loss_d1_data,
            'loss_d2_data': loss_d2_data,

            'loss_pos': loss_pos,
            'loss_pde': loss_pde,
            'loss_pde_r1': loss_pde_r1,
            'loss_pde_r2': loss_pde_r2,

            'weighted_pos': weighted_pos,
            'weighted_pde': weighted_pde,
            'weighted_physics_total': weighted_physics_total,

            'ratio_data_in_total': ratio_data_in_total,
            'ratio_weighted_pos_in_total': ratio_weighted_pos_in_total,
            'ratio_weighted_pde_in_total': ratio_weighted_pde_in_total,

            'R1': R1,
            'R2': R2,
            'pos_violation': pos_violation,
            'D1_pred': residual_parts['D1_pred'],
            'D2_pred': residual_parts['D2_pred'],
            'D1_th': residual_parts['D1_th'],
            'D2_th': residual_parts['D2_th'],
        }

    def forward(self, y_pred_scaled, y_true_scaled, branch_x_scaled):
        """
        与普通 loss 接口兼容的简单返回形式。
        """
        out = self.compute_all_losses(y_pred_scaled, y_true_scaled, branch_x_scaled)
        return (
            out['total_loss'],
            out['data_loss'].detach(),
            out['loss_pos'].detach(),
            out['loss_pde'].detach(),
            out['loss_d1_data'].detach(),
            out['loss_d2_data'].detach()
        )


# =============================================================================
# 6. 学习率调度器
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    先线性 warmup，再 cosine 衰减。
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# =============================================================================
# 7. 梯度贡献审计
# =============================================================================

def audit_gradient_contributions(model: nn.Module,
                                 loss_fn: PhysicsInformedLoss,
                                 branch_batch: torch.Tensor,
                                 y_batch_scaled: torch.Tensor,
                                 max_samples: int) -> dict:
    """
    这个函数非常关键。

    它分别计算：
    - data_loss 单独反向传播时的梯度范数
    - weighted_pos 单独反向传播时的梯度范数
    - weighted_pde 单独反向传播时的梯度范数
    - total_loss 反向传播时的梯度范数

    通过这些比值，你就可以判断：
    “physics 约束到底只是存在于 loss 里，还是它真的在推动参数更新”
    """
    model.eval()

    if branch_batch.shape[0] > max_samples:
        idx = torch.randperm(branch_batch.shape[0], device=branch_batch.device)[:max_samples]
        branch_batch = branch_batch[idx]
        y_batch_scaled = y_batch_scaled[idx]

    y_pred_scaled = model(branch_batch)
    parts = loss_fn.compute_all_losses(y_pred_scaled, y_batch_scaled, branch_batch)

    grad_stats = {}

    # 1) 只看数据损失的梯度
    model.zero_grad(set_to_none=True)
    parts['data_loss'].backward(retain_graph=True)
    grad_stats['grad_norm_data_loss'] = compute_grad_norm(model)

    # 2) 只看 weighted_pos 的梯度
    model.zero_grad(set_to_none=True)
    parts['weighted_pos'].backward(retain_graph=True)
    grad_stats['grad_norm_weighted_pos'] = compute_grad_norm(model)

    # 3) 只看 weighted_pde 的梯度
    model.zero_grad(set_to_none=True)
    parts['weighted_pde'].backward(retain_graph=True)
    grad_stats['grad_norm_weighted_pde'] = compute_grad_norm(model)

    # 4) 看 total loss 的梯度
    model.zero_grad(set_to_none=True)
    parts['total_loss'].backward()
    grad_stats['grad_norm_total_loss'] = compute_grad_norm(model)

    # 清空，避免污染外部训练
    model.zero_grad(set_to_none=True)

    total = max(grad_stats['grad_norm_total_loss'], 1e-30)
    data_g = grad_stats['grad_norm_data_loss']
    pos_g = grad_stats['grad_norm_weighted_pos']
    pde_g = grad_stats['grad_norm_weighted_pde']

    grad_stats['ratio_grad_data_over_total'] = data_g / total
    grad_stats['ratio_grad_weighted_pos_over_total'] = pos_g / total
    grad_stats['ratio_grad_weighted_pde_over_total'] = pde_g / total

    grad_stats['ratio_grad_pde_over_data'] = pde_g / max(data_g, 1e-30)
    grad_stats['ratio_grad_pos_over_data'] = pos_g / max(data_g, 1e-30)

    return grad_stats


# =============================================================================
# 8. 验证与统计
# =============================================================================

def evaluate_on_loader(model: nn.Module,
                       loss_fn: PhysicsInformedLoss,
                       loader: DataLoader):
    """
    在整个验证集上计算：
    1. 平均验证指标
    2. 网格级统计（R1/R2/正性）
    3. 样本级统计 DataFrame
    """
    model.eval()

    total_n = 0

    agg = {
        'total_loss': 0.0,
        'data_loss': 0.0,
        'loss_d1_data': 0.0,
        'loss_d2_data': 0.0,
        'loss_pos': 0.0,
        'loss_pde': 0.0,
        'loss_pde_r1': 0.0,
        'loss_pde_r2': 0.0,
        'weighted_pos': 0.0,
        'weighted_pde': 0.0,
        'weighted_physics_total': 0.0,
        'ratio_data_in_total': 0.0,
        'ratio_weighted_pos_in_total': 0.0,
        'ratio_weighted_pde_in_total': 0.0,
    }

    all_R1_abs = []
    all_R2_abs = []
    all_pos = []

    sample_records = []

    with torch.no_grad():
        for branch_x, y_true_scaled in loader:
            y_pred_scaled = model(branch_x)
            parts = loss_fn.compute_all_losses(y_pred_scaled, y_true_scaled, branch_x)

            bs = branch_x.shape[0]
            total_n += bs

            for k in agg.keys():
                agg[k] += to_float(parts[k]) * bs

            R1 = parts['R1']
            R2 = parts['R2']
            pos_v = parts['pos_violation']
            D2_pred = parts['D2_pred']

            field_len = loss_fn.num_a * loss_fn.num_tau

            y_pred = y_pred_scaled * loss_fn.y_std_scaler + loss_fn.y_mean_scaler
            y_true = y_true_scaled * loss_fn.y_std_scaler + loss_fn.y_mean_scaler

            y_pred_d1 = y_pred[:, :field_len].view(-1, loss_fn.num_tau, loss_fn.num_a)
            y_pred_d2 = y_pred[:, field_len:].view(-1, loss_fn.num_tau, loss_fn.num_a)

            y_true_d1 = y_true[:, :field_len].view(-1, loss_fn.num_tau, loss_fn.num_a)
            y_true_d2 = y_true[:, field_len:].view(-1, loss_fn.num_tau, loss_fn.num_a)

            # 样本级 RMSE
            rmse_d1_per_sample = torch.sqrt(torch.mean((y_pred_d1 - y_true_d1) ** 2, dim=(1, 2)))
            rmse_d2_per_sample = torch.sqrt(torch.mean((y_pred_d2 - y_true_d2) ** 2, dim=(1, 2)))

            # 样本级 PDE 残差统计
            mean_abs_r1 = torch.mean(torch.abs(R1), dim=(1, 2))
            mean_abs_r2 = torch.mean(torch.abs(R2), dim=(1, 2))
            max_abs_r1 = torch.amax(torch.abs(R1), dim=(1, 2))
            max_abs_r2 = torch.amax(torch.abs(R2), dim=(1, 2))

            # 正性统计
            mean_pos = torch.mean(pos_v, dim=(1, 2))
            max_pos = torch.amax(pos_v, dim=(1, 2))
            frac_neg_d2 = torch.mean((D2_pred < 0.0).to(DTYPE), dim=(1, 2))

            # 样本级综合指标
            per_sample_data_mse = torch.mean((y_pred_scaled - y_true_scaled) ** 2, dim=1)
            per_sample_r1 = torch.mean(R1 ** 2, dim=(1, 2))
            per_sample_r2 = torch.mean(R2 ** 2, dim=(1, 2))
            per_sample_pde = per_sample_r1 + per_sample_r2
            per_sample_weighted_physics = loss_fn.phys_weight * (
                torch.mean(pos_v ** 2, dim=(1, 2)) + loss_fn.pde_in_total_weight * per_sample_pde
            )

            for i in range(bs):
                sample_records.append({
                    'rmse_d1': to_float(rmse_d1_per_sample[i]),
                    'rmse_d2': to_float(rmse_d2_per_sample[i]),
                    'mean_abs_r1': to_float(mean_abs_r1[i]),
                    'mean_abs_r2': to_float(mean_abs_r2[i]),
                    'max_abs_r1': to_float(max_abs_r1[i]),
                    'max_abs_r2': to_float(max_abs_r2[i]),
                    'mean_pos_violation': to_float(mean_pos[i]),
                    'max_pos_violation': to_float(max_pos[i]),
                    'frac_negative_d2': to_float(frac_neg_d2[i]),
                    'sample_data_mse_scaled': to_float(per_sample_data_mse[i]),
                    'sample_pde_r1_mse': to_float(per_sample_r1[i]),
                    'sample_pde_r2_mse': to_float(per_sample_r2[i]),
                    'sample_pde_total_mse': to_float(per_sample_pde[i]),
                    'sample_weighted_physics': to_float(per_sample_weighted_physics[i]),
                })

            all_R1_abs.append(torch.abs(R1).detach().cpu().numpy())
            all_R2_abs.append(torch.abs(R2).detach().cpu().numpy())
            all_pos.append(pos_v.detach().cpu().numpy())

    metrics = {f'val_{k}': agg[k] / max(total_n, 1) for k in agg.keys()}

    R1_all = np.concatenate(all_R1_abs, axis=0)
    R2_all = np.concatenate(all_R2_abs, axis=0)
    POS_all = np.concatenate(all_pos, axis=0)

    grid_stats = {
        'R1_abs_mean': np.mean(R1_all, axis=0),
        'R1_abs_std': np.std(R1_all, axis=0),
        'R2_abs_mean': np.mean(R2_all, axis=0),
        'R2_abs_std': np.std(R2_all, axis=0),
        'POS_mean': np.mean(POS_all, axis=0),
        'POS_std': np.std(POS_all, axis=0),
    }

    sample_df = pd.DataFrame(sample_records)
    return metrics, grid_stats, sample_df


# =============================================================================
# 9. 单实验可视化函数
# =============================================================================

def plot_loss_overview(train_df: pd.DataFrame, val_df: pd.DataFrame, save_path: str, warmup_steps: int):
    plt.figure(figsize=(15, 8))

    plt.plot(train_df['iter'], train_df['train_total_loss'], label='Train Total Loss', alpha=0.55)
    plt.plot(val_df['iter'], val_df['val_total_loss'], label='Val Total Loss', linewidth=2.0)
    plt.plot(val_df['iter'], val_df['val_data_loss'], label='Val Data Loss', linestyle='--')
    plt.plot(val_df['iter'], val_df['val_loss_d1_data'], label='Val D1 Data', linestyle=':')
    plt.plot(val_df['iter'], val_df['val_loss_d2_data'], label='Val D2 Data', linestyle=':')

    plt.axvline(x=warmup_steps, color='gray', linestyle='--', label='Warmup End')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Overview')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_raw_physics_components(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(15, 8))

    plt.plot(val_df['iter'], val_df['val_loss_pos'], label='Val Raw Pos Loss')
    plt.plot(val_df['iter'], val_df['val_loss_pde'], label='Val Raw PDE Loss')
    plt.plot(val_df['iter'], val_df['val_loss_pde_r1'], label='Val Raw PDE R1')
    plt.plot(val_df['iter'], val_df['val_loss_pde_r2'], label='Val Raw PDE R2')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Raw Loss Value')
    plt.title('Raw Physics Components')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_weighted_contributions(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(15, 8))

    plt.plot(val_df['iter'], val_df['val_data_loss'], label='Data Loss', linewidth=2.0)
    plt.plot(val_df['iter'], val_df['val_weighted_pos'], label='Weighted Pos Contribution', linewidth=2.0)
    plt.plot(val_df['iter'], val_df['val_weighted_pde'], label='Weighted PDE Contribution', linewidth=2.0)
    plt.plot(val_df['iter'], val_df['val_weighted_physics_total'], label='Weighted Physics Total', linewidth=2.0)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Contribution Actually Entering Total Loss')
    plt.title('Weighted Contributions')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_contribution_ratios(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(15, 8))

    x = val_df['iter'].to_numpy()
    d = val_df['val_ratio_data_in_total'].to_numpy()
    p = val_df['val_ratio_weighted_pos_in_total'].to_numpy()
    q = val_df['val_ratio_weighted_pde_in_total'].to_numpy()

    plt.stackplot(
        x, d, p, q,
        labels=['Data Ratio', 'Weighted Pos Ratio', 'Weighted PDE Ratio'],
        alpha=0.8
    )

    plt.xlabel('Iteration')
    plt.ylabel('Ratio in Total Loss')
    plt.title('Relative Contribution Ratios in Total Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gradient_norms(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(15, 8))

    plt.plot(val_df['iter'], val_df['grad_norm_data_loss'], label='Grad Norm Data')
    plt.plot(val_df['iter'], val_df['grad_norm_weighted_pos'], label='Grad Norm Weighted Pos')
    plt.plot(val_df['iter'], val_df['grad_norm_weighted_pde'], label='Grad Norm Weighted PDE')
    plt.plot(val_df['iter'], val_df['grad_norm_total_loss'], label='Grad Norm Total', linewidth=2.0)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Audit')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gradient_ratios(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(15, 8))

    x = val_df['iter'].to_numpy()
    a = val_df['ratio_grad_data_over_total'].to_numpy()
    b = val_df['ratio_grad_weighted_pos_over_total'].to_numpy()
    c = val_df['ratio_grad_weighted_pde_over_total'].to_numpy()

    plt.stackplot(
        x, a, b, c,
        labels=['Grad(Data)/Grad(Total)', 'Grad(Pos)/Grad(Total)', 'Grad(PDE)/Grad(Total)'],
        alpha=0.8
    )

    plt.xlabel('Iteration')
    plt.ylabel('Gradient Ratio')
    plt.title('Gradient Composition Ratios')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_data_vs_pde_scatter(val_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 8))

    sc = plt.scatter(
        val_df['val_data_loss'],
        val_df['val_loss_pde'],
        c=val_df['iter'],
        cmap='viridis',
        s=55
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Validation Data Loss')
    plt.ylabel('Validation Raw PDE Loss')
    plt.title('Data Loss vs PDE Loss Through Training')
    plt.grid(True, which='both', alpha=0.3)

    cbar = plt.colorbar(sc)
    cbar.set_label('Iteration')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_comparison(model,
                               branch_val_scaled,
                               y_val_np,
                               unified_a_grid,
                               unified_tau_grid,
                               y_mean_scaler,
                               y_std_scaler,
                               save_path,
                               num_samples=4):
    """
    绘制多个验证样本的：
    - D1 真值 / 预测 / 误差
    - D2 真值 / 预测 / 误差
    """
    num_val = len(branch_val_scaled)
    if num_val == 0:
        return

    num_samples = min(num_samples, num_val)
    indices = np.random.choice(num_val, num_samples, replace=False)

    num_a = len(unified_a_grid)
    num_tau = len(unified_tau_grid)

    fig, axes = plt.subplots(num_samples, 6, figsize=(30, 5.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle('Prediction vs Ground Truth vs Absolute Error', fontsize=18, y=0.995)

    def plot_field(ax, data, title, vmin=None, vmax=None, cmap='viridis'):
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=[
                unified_a_grid.min(), unified_a_grid.max(),
                unified_tau_grid.min(), unified_tau_grid.max()
            ],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('A', fontsize=10)
        ax.set_ylabel('tau', fontsize=10)
        return im

    with torch.no_grad():
        for row_i, idx in enumerate(indices):
            branch_input_scaled = torch.from_numpy(branch_val_scaled[idx]).unsqueeze(0).to(DEVICE).to(DTYPE)
            y_pred_np = model.predict(branch_input_scaled, y_mean_scaler, y_std_scaler).cpu().numpy().flatten()
            y_true_np = y_val_np[idx]

            field_len_half = len(y_true_np) // 2

            y_pred_d1 = y_pred_np[:field_len_half].reshape(num_tau, num_a)
            y_pred_d2 = y_pred_np[field_len_half:].reshape(num_tau, num_a)

            y_true_d1 = y_true_np[:field_len_half].reshape(num_tau, num_a)
            y_true_d2 = y_true_np[field_len_half:].reshape(num_tau, num_a)

            err_d1 = np.abs(y_pred_d1 - y_true_d1)
            err_d2 = np.abs(y_pred_d2 - y_true_d2)

            vmin_d1 = min(y_true_d1.min(), y_pred_d1.min())
            vmax_d1 = max(y_true_d1.max(), y_pred_d1.max())

            vmin_d2 = min(y_true_d2.min(), y_pred_d2.min())
            vmax_d2 = max(y_true_d2.max(), y_pred_d2.max())

            ims = [
                plot_field(axes[row_i, 0], y_true_d1, f'Sample {idx} - D1 True', vmin=vmin_d1, vmax=vmax_d1, cmap='viridis'),
                plot_field(axes[row_i, 1], y_pred_d1, f'Sample {idx} - D1 Pred', vmin=vmin_d1, vmax=vmax_d1, cmap='viridis'),
                plot_field(axes[row_i, 2], err_d1, f'Sample {idx} - |D1 Error|', cmap='magma'),
                plot_field(axes[row_i, 3], y_true_d2, f'Sample {idx} - D2 True', vmin=vmin_d2, vmax=vmax_d2, cmap='plasma'),
                plot_field(axes[row_i, 4], y_pred_d2, f'Sample {idx} - D2 Pred', vmin=vmin_d2, vmax=vmax_d2, cmap='plasma'),
                plot_field(axes[row_i, 5], err_d2, f'Sample {idx} - |D2 Error|', cmap='magma'),
            ]

            for j, im in enumerate(ims):
                fig.colorbar(im, ax=axes[row_i, j], fraction=0.046, pad=0.04)

            rmse_d1 = np.sqrt(np.mean((y_pred_d1 - y_true_d1) ** 2))
            rmse_d2 = np.sqrt(np.mean((y_pred_d2 - y_true_d2) ** 2))
            axes[row_i, 0].text(
                0.02, 1.08,
                f'RMSE(D1)={rmse_d1:.3e}, RMSE(D2)={rmse_d2:.3e}',
                transform=axes[row_i, 0].transAxes,
                fontsize=10,
                verticalalignment='bottom'
            )

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_maps(grid_stats: dict, unified_a_grid, unified_tau_grid, save_path: str):
    """
    绘制网格级 residual 统计图：
    - R1 |abs| mean/std
    - R2 |abs| mean/std
    - Pos violation mean/std
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    items = [
        ('R1 |abs| mean', grid_stats['R1_abs_mean']),
        ('R1 |abs| std', grid_stats['R1_abs_std']),
        ('R2 |abs| mean', grid_stats['R2_abs_mean']),
        ('R2 |abs| std', grid_stats['R2_abs_std']),
        ('Pos violation mean', grid_stats['POS_mean']),
        ('Pos violation std', grid_stats['POS_std']),
    ]

    for ax, (title, data) in zip(axes.flatten(), items):
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=[unified_a_grid[1], unified_a_grid[-2], unified_tau_grid[1], unified_tau_grid[-2]]
        )
        ax.set_title(title)
        ax.set_xlabel('A')
        ax.set_ylabel('tau')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sample_statistics(sample_df: pd.DataFrame, save_path: str):
    """
    绘制样本级分布图：
    可以看出 physics 是否让“整体分布”发生变化，而不只是个别样本。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    cols = [
        ('rmse_d1', 'RMSE D1'),
        ('rmse_d2', 'RMSE D2'),
        ('mean_abs_r1', 'Mean |R1|'),
        ('mean_abs_r2', 'Mean |R2|'),
        ('frac_negative_d2', 'Fraction Negative D2'),
        ('sample_weighted_physics', 'Weighted Physics per Sample'),
    ]

    for ax, (col, title) in zip(axes.flatten(), cols):
        ax.hist(sample_df[col].values, bins=30)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(sample_df: pd.DataFrame, save_path: str):
    """
    样本级指标相关性图：
    用于看“数据误差”和“物理残差”是否强相关，还是相对独立。
    """
    corr = sample_df.corr(numeric_only=True)

    plt.figure(figsize=(13, 11))
    im = plt.imshow(corr.values, aspect='auto')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Correlation Heatmap of Sample-Level Diagnostics')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# 10. 对照实验图
# =============================================================================

def plot_compare_two_curves(df_a, df_b, col, label_a, label_b, title, ylabel, save_path, yscale='log'):
    plt.figure(figsize=(12, 7))

    plt.plot(df_a['iter'], df_a[col], label=label_a, linewidth=2.0)
    plt.plot(df_b['iter'], df_b[col], label=label_b, linewidth=2.0)

    if yscale is not None:
        plt.yscale(yscale)

    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_compare_hist(sample_df_a, sample_df_b, col, label_a, label_b, title, save_path, bins=30):
    plt.figure(figsize=(10, 7))

    plt.hist(sample_df_a[col].values, bins=bins, alpha=0.6, label=label_a)
    plt.hist(sample_df_b[col].values, bins=bins, alpha=0.6, label=label_b)

    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_compare_box(sample_df_a, sample_df_b, col, label_a, label_b, title, save_path):
    plt.figure(figsize=(8, 7))

    plt.boxplot(
        [sample_df_a[col].values, sample_df_b[col].values],
        labels=[label_a, label_b],
        showfliers=False
    )

    plt.ylabel(col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_compare_summary_table(summary_a: dict, summary_b: dict):
    """
    汇总两组实验的关键指标。
    """
    rows = []

    def add_metric(name, va, vb):
        rows.append({
            'metric': name,
            'data_only': va,
            'physics_informed': vb
        })

    add_metric('best_iter', summary_a['best_iter'], summary_b['best_iter'])
    add_metric('best_val_total_loss', summary_a['best_metrics']['val_total_loss'], summary_b['best_metrics']['val_total_loss'])
    add_metric('best_val_data_loss', summary_a['best_metrics']['val_data_loss'], summary_b['best_metrics']['val_data_loss'])
    add_metric('best_val_d1_data', summary_a['best_metrics']['val_loss_d1_data'], summary_b['best_metrics']['val_loss_d1_data'])
    add_metric('best_val_d2_data', summary_a['best_metrics']['val_loss_d2_data'], summary_b['best_metrics']['val_loss_d2_data'])
    add_metric('best_val_raw_pos_loss', summary_a['best_metrics']['val_loss_pos'], summary_b['best_metrics']['val_loss_pos'])
    add_metric('best_val_raw_pde_loss', summary_a['best_metrics']['val_loss_pde'], summary_b['best_metrics']['val_loss_pde'])
    add_metric('best_val_raw_pde_r1', summary_a['best_metrics']['val_loss_pde_r1'], summary_b['best_metrics']['val_loss_pde_r1'])
    add_metric('best_val_raw_pde_r2', summary_a['best_metrics']['val_loss_pde_r2'], summary_b['best_metrics']['val_loss_pde_r2'])
    add_metric('best_val_weighted_pos', summary_a['best_metrics']['val_weighted_pos'], summary_b['best_metrics']['val_weighted_pos'])
    add_metric('best_val_weighted_pde', summary_a['best_metrics']['val_weighted_pde'], summary_b['best_metrics']['val_weighted_pde'])
    add_metric('best_val_weighted_physics_total', summary_a['best_metrics']['val_weighted_physics_total'], summary_b['best_metrics']['val_weighted_physics_total'])

    return pd.DataFrame(rows)


# =============================================================================
# 11. 单个实验完整流程
# =============================================================================

def run_single_experiment(exp_name: str,
                          physics_loss_weight: float,
                          global_cfg: dict,
                          data_bundle: dict,
                          master_result_dir: str):
    """
    跑单个实验：
    - data_only 或 physics_informed
    - 保存完整日志、图像、模型
    """
    print('\n' + '=' * 100)
    print(f'开始实验: {exp_name}')
    print(f'PHYSICS_LOSS_WEIGHT = {physics_loss_weight}')
    print('=' * 100)

    result_dir = get_exp_result_dir(master_result_dir, exp_name)

    run_id = f'{MASTER_RUN_ID}_{exp_name}'

    # -------------------------
    # 文件路径
    # -------------------------
    MODEL_SAVE_PATH = os.path.join(result_dir, f'model_{run_id}.pth')
    SCALER_SAVE_PATH = os.path.join(result_dir, f'scalers_{run_id}.pth')

    TRAIN_LOG_CSV_PATH = os.path.join(result_dir, f'train_log_{run_id}.csv')
    VAL_LOG_CSV_PATH = os.path.join(result_dir, f'val_log_{run_id}.csv')
    SAMPLE_METRICS_CSV_PATH = os.path.join(result_dir, f'sample_metrics_best_model_{run_id}.csv')
    GRID_STATS_NPZ_PATH = os.path.join(result_dir, f'grid_residual_stats_best_model_{run_id}.npz')
    SUMMARY_JSON_PATH = os.path.join(result_dir, f'run_summary_{run_id}.json')
    SUMMARY_TXT_PATH = os.path.join(result_dir, f'summary_{run_id}.txt')

    LOSS_PLOT_SAVE_PATH = os.path.join(result_dir, f'loss_overview_{run_id}.png')
    RAW_PHYSICS_PLOT_SAVE_PATH = os.path.join(result_dir, f'raw_physics_components_{run_id}.png')
    WEIGHTED_CONTRIB_PLOT_SAVE_PATH = os.path.join(result_dir, f'weighted_contributions_{run_id}.png')
    CONTRIB_RATIO_PLOT_SAVE_PATH = os.path.join(result_dir, f'contribution_ratios_{run_id}.png')
    GRAD_NORM_PLOT_SAVE_PATH = os.path.join(result_dir, f'gradient_norms_{run_id}.png')
    GRAD_RATIO_PLOT_SAVE_PATH = os.path.join(result_dir, f'gradient_ratios_{run_id}.png')
    DATA_VS_PDE_SCATTER_PATH = os.path.join(result_dir, f'data_vs_pde_scatter_{run_id}.png')
    PREDICTION_PLOT_PATH = os.path.join(result_dir, f'prediction_compare_{run_id}.png')
    RESIDUAL_MAP_PLOT_PATH = os.path.join(result_dir, f'residual_maps_best_model_{run_id}.png')
    SAMPLE_STATS_PLOT_PATH = os.path.join(result_dir, f'sample_stats_best_model_{run_id}.png')
    CORRELATION_PLOT_PATH = os.path.join(result_dir, f'correlation_heatmap_best_model_{run_id}.png')

    # -------------------------
    # 解包数据
    # -------------------------
    branch_train_np = data_bundle['branch_train_np']
    branch_val_np = data_bundle['branch_val_np']
    y_train_np = data_bundle['y_train_np']
    y_val_np = data_bundle['y_val_np']

    branch_train_scaled = data_bundle['branch_train_scaled']
    branch_val_scaled = data_bundle['branch_val_scaled']

    y_train_scaled = data_bundle['y_train_scaled']
    y_val_scaled = data_bundle['y_val_scaled']

    branch_mean = data_bundle['branch_mean']
    branch_std = data_bundle['branch_std']
    y_mean_scaler = data_bundle['y_mean_scaler']
    y_std_scaler = data_bundle['y_std_scaler']

    y_mean_pod_scaled = data_bundle['y_mean_pod_scaled']
    pod_basis = data_bundle['pod_basis']
    S = data_bundle['S']
    actual_num_modes = data_bundle['actual_num_modes']

    unified_a_grid = data_bundle['unified_a_grid']
    unified_tau_grid = data_bundle['unified_tau_grid']
    retained_energy = data_bundle['retained_energy']
    y_snapshots_np = data_bundle['y_snapshots_np']

    # -------------------------
    # DataLoader
    # -------------------------
    train_dataset = TensorDataset(
        torch.from_numpy(branch_train_scaled).to(DEVICE).to(DTYPE),
        torch.from_numpy(y_train_scaled).to(DEVICE).to(DTYPE)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(branch_val_scaled).to(DEVICE).to(DTYPE),
        torch.from_numpy(y_val_scaled).to(DEVICE).to(DTYPE)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=global_cfg['ADAM_BATCH_SIZE'],
        shuffle=True
    )

    # 一个小验证 loader，用于普通验证频率
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(len(val_dataset), global_cfg['ADAM_BATCH_SIZE']),
        shuffle=False
    )

    # 一个全量验证 loader，用于关键节点做全量诊断
    val_loader_full = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False
    )

    # -------------------------
    # 模型
    # -------------------------
    model = PODDeepONet(
        branch_input_dim=global_cfg['BRANCH_INPUT_DIM'],
        hidden_units=global_cfg['HIDDEN_UNITS'],
        num_hidden_layers=global_cfg['NUM_HIDDEN_LAYERS'],
        num_pod_modes=actual_num_modes,
        pod_basis=pod_basis,
        y_mean_pod_scaled=y_mean_pod_scaled,
        dropout_rate=global_cfg['DROPOUT_RATE']
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=global_cfg['ADAM_LR'],
        weight_decay=global_cfg['WEIGHT_DECAY']
    )

    loss_fn = PhysicsInformedLoss(
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
        branch_mean=branch_mean,
        branch_std=branch_std,
        y_mean_scaler=y_mean_scaler,
        y_std_scaler=y_std_scaler,
        d2_weight=global_cfg['D2_LOSS_WEIGHT'],
        phys_weight=physics_loss_weight,
        pde_in_total_weight=global_cfg['PDE_IN_TOTAL_WEIGHT']
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        global_cfg['WARMUP_STEPS'],
        global_cfg['ADAM_ITERATIONS']
    )

    # -------------------------
    # 日志容器
    # -------------------------
    train_logs = {
        'iter': [],
        'train_total_loss': [],
        'train_data_loss': [],
        'train_loss_pos': [],
        'train_loss_pde': [],
        'train_loss_pde_r1': [],
        'train_loss_pde_r2': [],
        'train_weighted_pos': [],
        'train_weighted_pde': [],
        'train_weighted_physics_total': [],
        'train_ratio_data_in_total': [],
        'train_ratio_weighted_pos_in_total': [],
        'train_ratio_weighted_pde_in_total': [],
        'train_lr': [],
    }

    val_logs = {
        'iter': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_loss_d1_data': [],
        'val_loss_d2_data': [],
        'val_loss_pos': [],
        'val_loss_pde': [],
        'val_loss_pde_r1': [],
        'val_loss_pde_r2': [],
        'val_weighted_pos': [],
        'val_weighted_pde': [],
        'val_weighted_physics_total': [],
        'val_ratio_data_in_total': [],
        'val_ratio_weighted_pos_in_total': [],
        'val_ratio_weighted_pde_in_total': [],
        'grad_norm_data_loss': [],
        'grad_norm_weighted_pos': [],
        'grad_norm_weighted_pde': [],
        'grad_norm_total_loss': [],
        'ratio_grad_data_over_total': [],
        'ratio_grad_weighted_pos_over_total': [],
        'ratio_grad_weighted_pde_over_total': [],
        'ratio_grad_pde_over_data': [],
        'ratio_grad_pos_over_data': [],
    }

    min_val_loss = float('inf')
    best_iter = -1
    early_stop_counter = 0

    # -------------------------
    # 开始训练
    # -------------------------
    print(f'\n--- 开始训练: {exp_name} ---')
    pbar = tqdm(total=global_cfg['ADAM_ITERATIONS'], desc=f'训练进度 [{exp_name}]')

    global_iter = 0
    done = False

    while not done:
        for batch_branch, batch_y_scaled in train_loader:
            if global_iter >= global_cfg['ADAM_ITERATIONS']:
                done = True
                break

            model.train()
            optimizer.zero_grad()

            y_pred_scaled = model(batch_branch)
            parts = loss_fn.compute_all_losses(y_pred_scaled, batch_y_scaled, batch_branch)

            loss = parts['total_loss']
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            train_logs['iter'].append(global_iter)
            train_logs['train_total_loss'].append(to_float(parts['total_loss']))
            train_logs['train_data_loss'].append(to_float(parts['data_loss']))
            train_logs['train_loss_pos'].append(to_float(parts['loss_pos']))
            train_logs['train_loss_pde'].append(to_float(parts['loss_pde']))
            train_logs['train_loss_pde_r1'].append(to_float(parts['loss_pde_r1']))
            train_logs['train_loss_pde_r2'].append(to_float(parts['loss_pde_r2']))
            train_logs['train_weighted_pos'].append(to_float(parts['weighted_pos']))
            train_logs['train_weighted_pde'].append(to_float(parts['weighted_pde']))
            train_logs['train_weighted_physics_total'].append(to_float(parts['weighted_physics_total']))
            train_logs['train_ratio_data_in_total'].append(to_float(parts['ratio_data_in_total']))
            train_logs['train_ratio_weighted_pos_in_total'].append(to_float(parts['ratio_weighted_pos_in_total']))
            train_logs['train_ratio_weighted_pde_in_total'].append(to_float(parts['ratio_weighted_pde_in_total']))
            train_logs['train_lr'].append(current_lr)

            pbar.update(1)

            # -------------------------------------------------------------
            # 每隔一段做一次全量验证与梯度审计
            # -------------------------------------------------------------
            if global_iter % global_cfg['VALIDATION_FREQUENCY'] == 0:
                val_metrics, _, _ = evaluate_on_loader(model, loss_fn, val_loader_full)

                # 梯度审计基于整验证集或截断后的验证集
                val_branch_full, val_y_full = next(iter(val_loader_full))
                grad_stats = audit_gradient_contributions(
                    model=model,
                    loss_fn=loss_fn,
                    branch_batch=val_branch_full,
                    y_batch_scaled=val_y_full,
                    max_samples=global_cfg['GRAD_AUDIT_MAX_SAMPLES']
                )

                val_logs['iter'].append(global_iter)

                for k, v in val_metrics.items():
                    val_logs[k].append(v)
                for k, v in grad_stats.items():
                    val_logs[k].append(v)

                print(
                    f"\n[{exp_name}] "
                    f"Iter={global_iter:6d} | "
                    f"TrainTot={to_float(parts['total_loss']):.4e} | "
                    f"ValTot={val_metrics['val_total_loss']:.4e} | "
                    f"ValData={val_metrics['val_data_loss']:.4e} | "
                    f"ValRawPDE={val_metrics['val_loss_pde']:.4e} | "
                    f"ValW_PDE={val_metrics['val_weighted_pde']:.4e} | "
                    f"GradPDE/Data={grad_stats['ratio_grad_pde_over_data']:.4e} | "
                    f"LR={current_lr:.3e}"
                )

                if val_metrics['val_total_loss'] < min_val_loss:
                    min_val_loss = val_metrics['val_total_loss']
                    best_iter = global_iter
                    early_stop_counter = 0
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= global_cfg['EARLY_STOPPING_PATIENCE']:
                    print(f'\n*** Early stopping triggered for {exp_name}. ***')
                    done = True
                    break

            global_iter += 1

    pbar.close()

    # -------------------------
    # 保存基础信息
    # -------------------------
    torch.save({
        'branch_mean': branch_mean,
        'branch_std': branch_std,
        'y_mean_scaler': y_mean_scaler,
        'y_std_scaler': y_std_scaler,
        'y_mean_pod_scaled': y_mean_pod_scaled,
        'pod_basis': pod_basis,
        'singular_values': S,
        'actual_num_modes': actual_num_modes,
        'unified_a_grid': unified_a_grid,
        'unified_tau_grid': unified_tau_grid,
        'physics_loss_weight': physics_loss_weight,
    }, SCALER_SAVE_PATH)

    train_df = pd.DataFrame(train_logs)
    val_df = pd.DataFrame(val_logs)

    train_df.to_csv(TRAIN_LOG_CSV_PATH, index=False, encoding='utf-8-sig')
    val_df.to_csv(VAL_LOG_CSV_PATH, index=False, encoding='utf-8-sig')

    # -------------------------
    # 加载最佳模型，做最终详细分析
    # -------------------------
    print(f'\n--- 加载最佳模型并做详细验证: {exp_name} ---')

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    best_metrics, best_grid_stats, sample_df = evaluate_on_loader(model, loss_fn, val_loader_full)

    sample_df.to_csv(SAMPLE_METRICS_CSV_PATH, index=False, encoding='utf-8-sig')
    np.savez(GRID_STATS_NPZ_PATH, **best_grid_stats)

    # -------------------------
    # 单实验图像
    # -------------------------
    plot_loss_overview(
        train_df=train_df,
        val_df=val_df,
        save_path=LOSS_PLOT_SAVE_PATH,
        warmup_steps=global_cfg['WARMUP_STEPS']
    )

    plot_raw_physics_components(
        val_df=val_df,
        save_path=RAW_PHYSICS_PLOT_SAVE_PATH
    )

    plot_weighted_contributions(
        val_df=val_df,
        save_path=WEIGHTED_CONTRIB_PLOT_SAVE_PATH
    )

    plot_contribution_ratios(
        val_df=val_df,
        save_path=CONTRIB_RATIO_PLOT_SAVE_PATH
    )

    plot_gradient_norms(
        val_df=val_df,
        save_path=GRAD_NORM_PLOT_SAVE_PATH
    )

    plot_gradient_ratios(
        val_df=val_df,
        save_path=GRAD_RATIO_PLOT_SAVE_PATH
    )

    plot_data_vs_pde_scatter(
        val_df=val_df,
        save_path=DATA_VS_PDE_SCATTER_PATH
    )

    plot_prediction_comparison(
        model=model,
        branch_val_scaled=branch_val_scaled,
        y_val_np=y_val_np,
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
        y_mean_scaler=y_mean_scaler,
        y_std_scaler=y_std_scaler,
        save_path=PREDICTION_PLOT_PATH,
        num_samples=global_cfg['NUM_VIS_SAMPLES']
    )

    plot_residual_maps(
        grid_stats=best_grid_stats,
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
        save_path=RESIDUAL_MAP_PLOT_PATH
    )

    plot_sample_statistics(
        sample_df=sample_df,
        save_path=SAMPLE_STATS_PLOT_PATH
    )

    plot_correlation_heatmap(
        sample_df=sample_df,
        save_path=CORRELATION_PLOT_PATH
    )

    # -------------------------
    # 汇总
    # -------------------------
    summary = {
        'exp_name': exp_name,
        'physics_loss_weight': physics_loss_weight,
        'train_samples': int(len(branch_train_np)),
        'val_samples': int(len(branch_val_np)),
        'A_points': int(len(unified_a_grid)),
        'tau_points': int(len(unified_tau_grid)),
        'output_dim': int(y_snapshots_np.shape[1]),
        'actual_pod_modes': int(actual_num_modes),
        'retained_energy': float(retained_energy),
        'best_val_loss': float(min_val_loss),
        'best_iter': int(best_iter),
        'best_metrics': {k: float(v) for k, v in best_metrics.items()},
        'files': {
            'model': MODEL_SAVE_PATH,
            'scaler': SCALER_SAVE_PATH,
            'train_log_csv': TRAIN_LOG_CSV_PATH,
            'val_log_csv': VAL_LOG_CSV_PATH,
            'sample_metrics_csv': SAMPLE_METRICS_CSV_PATH,
            'grid_stats_npz': GRID_STATS_NPZ_PATH,
            'loss_plot': LOSS_PLOT_SAVE_PATH,
            'raw_physics_plot': RAW_PHYSICS_PLOT_SAVE_PATH,
            'weighted_contrib_plot': WEIGHTED_CONTRIB_PLOT_SAVE_PATH,
            'contrib_ratio_plot': CONTRIB_RATIO_PLOT_SAVE_PATH,
            'grad_norm_plot': GRAD_NORM_PLOT_SAVE_PATH,
            'grad_ratio_plot': GRAD_RATIO_PLOT_SAVE_PATH,
            'data_vs_pde_scatter': DATA_VS_PDE_SCATTER_PATH,
            'prediction_plot': PREDICTION_PLOT_PATH,
            'residual_map_plot': RESIDUAL_MAP_PLOT_PATH,
            'sample_stats_plot': SAMPLE_STATS_PLOT_PATH,
            'correlation_plot': CORRELATION_PLOT_PATH,
        }
    }

    save_json(SUMMARY_JSON_PATH, summary)

    summary_text = (
        f'exp_name: {exp_name}\n'
        f'physics_loss_weight: {physics_loss_weight}\n'
        f'train_samples: {len(branch_train_np)}\n'
        f'val_samples: {len(branch_val_np)}\n'
        f'A points: {len(unified_a_grid)}\n'
        f'tau points: {len(unified_tau_grid)}\n'
        f'output_dim: {y_snapshots_np.shape[1]}\n'
        f'actual_pod_modes: {actual_num_modes}\n'
        f'retained_energy: {retained_energy:.8f}\n'
        f'best_val_loss: {min_val_loss:.8e}\n'
        f'best_iter: {best_iter}\n'
        f'best_val_data_loss: {best_metrics["val_data_loss"]:.8e}\n'
        f'best_val_raw_pde_loss: {best_metrics["val_loss_pde"]:.8e}\n'
        f'best_val_weighted_pde: {best_metrics["val_weighted_pde"]:.8e}\n'
        f'model: {MODEL_SAVE_PATH}\n'
    )
    save_text(SUMMARY_TXT_PATH, summary_text)

    print(f'\n实验完成: {exp_name}')
    print(f'最佳模型: {MODEL_SAVE_PATH}')
    print(f'训练日志: {TRAIN_LOG_CSV_PATH}')
    print(f'验证日志: {VAL_LOG_CSV_PATH}')

    return {
        'exp_name': exp_name,
        'result_dir': result_dir,
        'summary': summary,
        'train_df': train_df,
        'val_df': val_df,
        'sample_df': sample_df,
        'grid_stats': best_grid_stats,
    }


# =============================================================================
# 12. 自动对照实验汇总
# =============================================================================

def run_compare_analysis(exp_result_a: dict, exp_result_b: dict, compare_dir: str):
    """
    对两组实验自动做对照分析：
    - 曲线对比
    - 分布对比
    - 总结表
    """
    os.makedirs(compare_dir, exist_ok=True)

    label_a = exp_result_a['exp_name']
    label_b = exp_result_b['exp_name']

    val_a = exp_result_a['val_df']
    val_b = exp_result_b['val_df']

    sample_a = exp_result_a['sample_df']
    sample_b = exp_result_b['sample_df']

    summary_a = exp_result_a['summary']
    summary_b = exp_result_b['summary']

    # ---------------------------------------------------------
    # 1. 关键曲线对比
    # ---------------------------------------------------------
    plot_compare_two_curves(
        val_a, val_b,
        col='val_data_loss',
        label_a=label_a,
        label_b=label_b,
        title='Compare: Validation Data Loss',
        ylabel='Validation Data Loss',
        save_path=os.path.join(compare_dir, 'compare_val_data_loss.png')
    )

    plot_compare_two_curves(
        val_a, val_b,
        col='val_loss_pde',
        label_a=label_a,
        label_b=label_b,
        title='Compare: Validation Raw PDE Loss',
        ylabel='Validation Raw PDE Loss',
        save_path=os.path.join(compare_dir, 'compare_val_raw_pde_loss.png')
    )

    plot_compare_two_curves(
        val_a, val_b,
        col='val_weighted_pde',
        label_a=label_a,
        label_b=label_b,
        title='Compare: Validation Weighted PDE Contribution',
        ylabel='Validation Weighted PDE',
        save_path=os.path.join(compare_dir, 'compare_val_weighted_pde.png')
    )

    plot_compare_two_curves(
        val_a, val_b,
        col='ratio_grad_pde_over_data',
        label_a=label_a,
        label_b=label_b,
        title='Compare: Gradient Ratio PDE/Data',
        ylabel='grad_pde / grad_data',
        save_path=os.path.join(compare_dir, 'compare_grad_ratio_pde_over_data.png'),
        yscale='log'
    )

    plot_compare_two_curves(
        val_a, val_b,
        col='val_ratio_weighted_pde_in_total',
        label_a=label_a,
        label_b=label_b,
        title='Compare: Weighted PDE Ratio in Total Loss',
        ylabel='Ratio',
        save_path=os.path.join(compare_dir, 'compare_weighted_pde_ratio_in_total.png'),
        yscale=None
    )

    # ---------------------------------------------------------
    # 2. 样本分布对比
    # ---------------------------------------------------------
    plot_compare_hist(
        sample_a, sample_b,
        col='rmse_d1',
        label_a=label_a,
        label_b=label_b,
        title='Compare Sample Distribution: RMSE D1',
        save_path=os.path.join(compare_dir, 'compare_hist_rmse_d1.png')
    )

    plot_compare_hist(
        sample_a, sample_b,
        col='rmse_d2',
        label_a=label_a,
        label_b=label_b,
        title='Compare Sample Distribution: RMSE D2',
        save_path=os.path.join(compare_dir, 'compare_hist_rmse_d2.png')
    )

    plot_compare_hist(
        sample_a, sample_b,
        col='mean_abs_r1',
        label_a=label_a,
        label_b=label_b,
        title='Compare Sample Distribution: Mean |R1|',
        save_path=os.path.join(compare_dir, 'compare_hist_mean_abs_r1.png')
    )

    plot_compare_hist(
        sample_a, sample_b,
        col='mean_abs_r2',
        label_a=label_a,
        label_b=label_b,
        title='Compare Sample Distribution: Mean |R2|',
        save_path=os.path.join(compare_dir, 'compare_hist_mean_abs_r2.png')
    )

    plot_compare_hist(
        sample_a, sample_b,
        col='frac_negative_d2',
        label_a=label_a,
        label_b=label_b,
        title='Compare Sample Distribution: Fraction Negative D2',
        save_path=os.path.join(compare_dir, 'compare_hist_frac_negative_d2.png')
    )

    plot_compare_box(
        sample_a, sample_b,
        col='sample_weighted_physics',
        label_a=label_a,
        label_b=label_b,
        title='Compare Boxplot: Sample Weighted Physics',
        save_path=os.path.join(compare_dir, 'compare_box_sample_weighted_physics.png')
    )

    # ---------------------------------------------------------
    # 3. 汇总表
    # ---------------------------------------------------------
    compare_table = create_compare_summary_table(summary_a, summary_b)
    compare_table_path = os.path.join(compare_dir, 'compare_summary_table.csv')
    compare_table.to_csv(compare_table_path, index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------
    # 4. 汇总文字结论
    # ---------------------------------------------------------
    text = []
    text.append('自动对照实验总结\n')
    text.append('=' * 60 + '\n')
    text.append(f'实验A: {label_a}\n')
    text.append(f'实验B: {label_b}\n\n')

    for _, row in compare_table.iterrows():
        text.append(
            f"{row['metric']}: "
            f"{label_a}={row['data_only']}, "
            f"{label_b}={row['physics_informed']}\n"
        )

    # 加一点更直观的结论提示
    text.append('\n' + '=' * 60 + '\n')
    text.append('你看 physics 是否真的发挥作用，建议重点观察：\n')
    text.append('1. compare_val_data_loss.png\n')
    text.append('2. compare_val_raw_pde_loss.png\n')
    text.append('3. compare_val_weighted_pde.png\n')
    text.append('4. compare_grad_ratio_pde_over_data.png\n')
    text.append('5. compare_hist_mean_abs_r1.png\n')
    text.append('6. compare_hist_mean_abs_r2.png\n')
    text.append('\n')
    text.append('若 physics_informed 同时满足：\n')
    text.append('- val_data_loss 不变或更好\n')
    text.append('- raw/weighted PDE 指标更低\n')
    text.append('- grad_pde_over_data 不是长期接近 0\n')
    text.append('- 样本级 mean|R1| / mean|R2| 分布整体左移\n')
    text.append('那么就能较有力地说明 physics 确实发挥了作用，而不是只有数据项在主导。\n')

    compare_summary_txt = os.path.join(compare_dir, 'compare_summary.txt')
    save_text(compare_summary_txt, ''.join(text))

    compare_summary_json = os.path.join(compare_dir, 'compare_summary.json')
    save_json(compare_summary_json, {
        'experiment_a': summary_a,
        'experiment_b': summary_b,
        'summary_table_csv': compare_table_path,
        'summary_text': compare_summary_txt
    })

    print('\n自动对照分析完成。')
    print(f'对照目录: {compare_dir}')
    print(f'对照表: {compare_table_path}')
    print(f'对照说明: {compare_summary_txt}')


# =============================================================================
# 13. 主程序
# =============================================================================

if __name__ == '__main__':
    try:
        print('=' * 100)
        print('POD-DeepONet + Physics-Informed Diagnostics + 自动对照实验')
        print('=' * 100)
        print(f'Device: {DEVICE}')
        print(f'Dtype : {DTYPE}')
        print(f'DATA_DIR: {DATA_DIR}')
        print(f'MASTER_RESULT_DIR: {MASTER_RESULT_DIR}')

        # ---------------------------------------------------------
        # 0. 固定随机种子
        # ---------------------------------------------------------
        set_seed(GLOBAL_CONFIG['SEED'])

        # ---------------------------------------------------------
        # 1. 只加载一次数据，并固定 train/val split
        #    这样两组实验才是公平对照
        # ---------------------------------------------------------
        branch_inputs_np, y_snapshots_np, unified_a_grid, unified_tau_grid = load_unified_data(
            DATA_DIR, GLOBAL_CONFIG['TARGET_NUM_FILES']
        )

        branch_train_np, branch_val_np, y_train_np, y_val_np = train_test_split(
            branch_inputs_np,
            y_snapshots_np,
            test_size=GLOBAL_CONFIG['VALIDATION_SPLIT'],
            random_state=GLOBAL_CONFIG['SEED']
        )

        print('\n数据划分完成：')
        print(f'Train samples: {len(branch_train_np)}')
        print(f'Val samples  : {len(branch_val_np)}')

        # ---------------------------------------------------------
        # 2. 标准化（只做一次，保证两组实验完全一致）
        # ---------------------------------------------------------
        branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
        branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)

        y_train_scaled, y_mean_scaler, y_std_scaler = manual_scaler(y_train_np)
        y_val_scaled = manual_scaler(y_val_np, y_mean_scaler, y_std_scaler)

        # ---------------------------------------------------------
        # 3. POD（也只做一次，保证可比性）
        # ---------------------------------------------------------
        y_mean_pod_scaled, pod_basis, S, actual_num_modes = pod(
            y_train_scaled,
            GLOBAL_CONFIG['REQUESTED_NUM_POD_MODES']
        )

        energy = (S ** 2) / np.sum(S ** 2)
        cumulative_energy = np.cumsum(energy)
        retained_energy = cumulative_energy[actual_num_modes - 1]

        print('\nPOD 分解完成：')
        print(f"Requested POD modes: {GLOBAL_CONFIG['REQUESTED_NUM_POD_MODES']}")
        print(f'Actual POD modes   : {actual_num_modes}')
        print(f'Retained energy    : {retained_energy:.6f}')

        # ---------------------------------------------------------
        # 4. 统一数据包
        # ---------------------------------------------------------
        data_bundle = {
            'branch_train_np': branch_train_np,
            'branch_val_np': branch_val_np,
            'y_train_np': y_train_np,
            'y_val_np': y_val_np,

            'branch_train_scaled': branch_train_scaled,
            'branch_val_scaled': branch_val_scaled,
            'y_train_scaled': y_train_scaled,
            'y_val_scaled': y_val_scaled,

            'branch_mean': branch_mean,
            'branch_std': branch_std,
            'y_mean_scaler': y_mean_scaler,
            'y_std_scaler': y_std_scaler,

            'y_mean_pod_scaled': y_mean_pod_scaled,
            'pod_basis': pod_basis,
            'S': S,
            'actual_num_modes': actual_num_modes,
            'retained_energy': retained_energy,

            'unified_a_grid': unified_a_grid,
            'unified_tau_grid': unified_tau_grid,
            'y_snapshots_np': y_snapshots_np,
        }

        # ---------------------------------------------------------
        # 5. 逐个跑实验
        # ---------------------------------------------------------
        results = []

        for exp_cfg in EXPERIMENTS:
            # 为了最大限度保持可比性，每次实验前都重新固定 seed
            set_seed(GLOBAL_CONFIG['SEED'])

            result = run_single_experiment(
                exp_name=exp_cfg['exp_name'],
                physics_loss_weight=exp_cfg['physics_loss_weight'],
                global_cfg=deepcopy(GLOBAL_CONFIG),
                data_bundle=data_bundle,
                master_result_dir=MASTER_RESULT_DIR
            )
            results.append(result)

        # ---------------------------------------------------------
        # 6. 自动对照分析
        # ---------------------------------------------------------
        result_map = {r['exp_name']: r for r in results}
        compare_dir = os.path.join(MASTER_RESULT_DIR, 'compare_analysis')
        os.makedirs(compare_dir, exist_ok=True)

        if 'data_only' in result_map and 'physics_informed' in result_map:
            run_compare_analysis(
                exp_result_a=result_map['data_only'],
                exp_result_b=result_map['physics_informed'],
                compare_dir=compare_dir
            )
        else:
            print('未找到 data_only 与 physics_informed 两组结果，跳过对照分析。')

        # ---------------------------------------------------------
        # 7. 总汇总
        # ---------------------------------------------------------
        master_summary = {
            'master_run_id': MASTER_RUN_ID,
            'device': str(DEVICE),
            'dtype': str(DTYPE),
            'global_config': GLOBAL_CONFIG,
            'experiments': [r['summary'] for r in results],
            'compare_dir': compare_dir,
        }

        save_json(
            os.path.join(MASTER_RESULT_DIR, f'master_summary_{MASTER_RUN_ID}.json'),
            master_summary
        )

        save_text(
            os.path.join(MASTER_RESULT_DIR, f'master_summary_{MASTER_RUN_ID}.txt'),
            (
                f'MASTER_RUN_ID: {MASTER_RUN_ID}\n'
                f'Device: {DEVICE}\n'
                f'Dtype: {DTYPE}\n'
                f'Experiments: {[e["exp_name"] for e in EXPERIMENTS]}\n'
                f'Compare dir: {compare_dir}\n'
            )
        )

        print('\n' + '=' * 100)
        print('全部实验完成。')
        print(f'总目录: {MASTER_RESULT_DIR}')
        print(f'对照分析目录: {compare_dir}')
        print('=' * 100)

    except Exception:
        print('\n程序运行失败，异常信息如下：')
        traceback.print_exc()
        raise