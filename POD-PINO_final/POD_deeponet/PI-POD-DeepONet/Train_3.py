# -*- coding: utf-8 -*-
"""
POD-DeepONet + Physics-Informed Loss
===================================

功能概述
--------
1. 从多个 CSV 文件中读取统一网格上的 D1 / D2 快照数据；
2. 对输入参数和输出场进行标准化；
3. 对输出场做 POD 降维；
4. 使用 MLP 学习“参数 -> POD 系数”的映射；
5. 在训练损失中加入：
   - 数据拟合损失
   - D2 非负性约束
   - 基于有限差分的 PDE 物理残差约束
6. 使用 AdamW + Warmup + Cosine Scheduler 训练；
7. 保存最佳模型、scaler、训练日志，并绘制：
   - 训练/验证损失曲线
   - 验证物理损失分项曲线
   - 多样本预测对比图
   - 误差热图

说明
----
本代码在尽量不改变原始算法逻辑的前提下，修复了原代码中的以下问题：
- 修复日志字段缺失导致的 KeyError
- 修复 model.predict() 不存在导致的 AttributeError
- 补充 PREDICTION_PLOT_PATH 定义
- 修复可能的缩进问题
- 增强 CSV 数据完整性检查
- 增强网格一致性 / 均匀性检查
- 增强绘图细节和保存逻辑
- 增强异常信息提示，避免静默吞错

作者建议
--------
若数据量较大、显存有限，可以将 TensorDataset 保留在 CPU，再在 batch 级别搬运到 GPU。
当前版本为尽量贴近你原始代码风格，保留了“先整体转 DEVICE”的写法。
"""

import os
import glob
import math
import json
import random
import traceback

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
# 1. 全局配置
# =============================================================================

# --- 路径配置 ---
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physics_informed_2'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA'
os.makedirs(RESULT_DIR, exist_ok=True)

RUN_ID = "pod_physics_informed_v3"

MODEL_SAVE_PATH = os.path.join(RESULT_DIR, f'model_{RUN_ID}.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, f'scalers_{RUN_ID}.pth')
LOG_CSV_PATH = os.path.join(RESULT_DIR, f'logs_{RUN_ID}.csv')
LOSS_PLOT_SAVE_PATH = os.path.join(RESULT_DIR, f'training_loss_{RUN_ID}.png')
PHYSICS_PLOT_SAVE_PATH = os.path.join(RESULT_DIR, f'physics_loss_{RUN_ID}.png')
PREDICTION_PLOT_PATH = os.path.join(RESULT_DIR, f'prediction_compare_{RUN_ID}.png')
SUMMARY_TXT_PATH = os.path.join(RESULT_DIR, f'summary_{RUN_ID}.txt')

# --- 模型超参数 ---
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
REQUESTED_NUM_POD_MODES = 100
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6

# --- 训练超参数 ---
ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 64
ADAM_ITERATIONS = 100000
D2_LOSS_WEIGHT = 0.2
PHYSICS_LOSS_WEIGHT = 0.1

VALIDATION_SPLIT = 0.2
SEED = 24
VALIDATION_FREQUENCY = 200

# 注意：这是“验证次数级别”的 patience，而不是 iteration 级别
EARLY_STOPPING_PATIENCE = 5

WARMUP_STEPS = 5000
TARGET_NUM_FILES = 2500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# =============================================================================
# 2. 工具函数
# =============================================================================

def set_seed(seed: int = 24):
    """
    固定随机种子，提升结果可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_required_columns(df: pd.DataFrame, required_cols, file_path: str):
    """
    检查 DataFrame 是否包含必须字段。
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"文件 {file_path} 缺少必要列: {missing}")


def is_uniform_grid(arr: np.ndarray, tol: float = 1e-12) -> bool:
    """
    检查 1D 网格是否为均匀网格。
    """
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], atol=tol, rtol=tol)


def safe_mkdir(path: str):
    """
    安全创建目录。
    """
    os.makedirs(path, exist_ok=True)


def save_text_summary(path: str, text: str):
    """
    保存文本总结。
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# =============================================================================
# 3. 数据加载与预处理
# =============================================================================

def load_unified_data(data_dir, target_num_files):
    """
    从 data_dir 中加载形如 data_*.csv 的数据文件，构建统一训练样本。

    每个 CSV 文件应至少包含以下列：
    - A
    - tau
    - nu
    - kappa
    - d_diffusion
    - D1_adj
    - D2_adj

    返回
    ----
    branch_inputs_np : shape [N, 3]
        每个样本对应的分支网络输入参数 [nu, kappa, d_diffusion]

    y_snapshots_np : shape [N, 2 * (num_tau * num_a)]
        拼接后的输出快照 [D1_flat, D2_flat]

    unified_a_grid : shape [num_a]
    unified_tau_grid : shape [num_tau]
    """
    print(f"从 {data_dir} 加载统一数据...")

    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_available_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何 data_*.csv 文件。")

    # 随机抽取指定数量样本文件
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

    for f in tqdm(selected_files, desc="加载快照"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            ensure_required_columns(df, required_cols, f)

            # 去掉关键输出为空的行
            df = df.dropna(subset=['D1_adj', 'D2_adj'])

            if df.empty:
                raise ValueError("删除缺失值后为空表。")

            # 提取当前文件网格
            a_grid = np.sort(df['A'].unique())
            tau_grid = np.sort(df['tau'].unique())

            if len(a_grid) < 3 or len(tau_grid) < 3:
                raise ValueError("网格点数量不足，中心差分至少要求 A 和 tau 两个方向各不少于 3 个点。")

            # 建立统一网格；后续文件必须一致
            if unified_a_grid is None:
                unified_a_grid = a_grid
                unified_tau_grid = tau_grid
            else:
                if not np.array_equal(a_grid, unified_a_grid):
                    raise ValueError("A 网格与统一网格不一致。")
                if not np.array_equal(tau_grid, unified_tau_grid):
                    raise ValueError("tau 网格与统一网格不一致。")

            # 检查均匀网格（你的 PDE 差分推导默认均匀网格）
            if not is_uniform_grid(a_grid):
                raise ValueError("A 网格不是均匀网格，当前实现默认均匀网格。")
            if not is_uniform_grid(tau_grid):
                raise ValueError("tau 网格不是均匀网格，当前实现默认均匀网格。")

            # 依据 (tau, A) 排序，以确保 flatten 顺序一致
            df_sorted = df.sort_values(by=['tau', 'A']).reset_index(drop=True)

            # 理论上应为完整 tensor-product 网格
            expected_len = len(unified_a_grid) * len(unified_tau_grid)
            if len(df_sorted) != expected_len:
                raise ValueError(
                    f"网格点数不完整或有重复。期望 {expected_len}，实际 {len(df_sorted)}。"
                )

            # 参数取文件第一行即可（默认同文件内部参数恒定）
            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values.astype(np.float64)

            # 输出快照拼接：[D1_flat, D2_flat]
            final_snapshot = np.concatenate([
                df_sorted['D1_adj'].values.astype(np.float64),
                df_sorted['D2_adj'].values.astype(np.float64)
            ])

            branch_inputs_list.append(params)
            y_snapshots_list.append(final_snapshot)

        except Exception as e:
            num_failed += 1
            print(f"\n[警告] 文件读取失败: {f}")
            print(f"原因: {e}")

    if len(branch_inputs_list) == 0 or len(y_snapshots_list) == 0:
        raise ValueError("未成功加载任何有效样本，请检查 CSV 数据格式、列名和网格一致性。")

    branch_inputs_np = np.array(branch_inputs_list, dtype=np.float64)
    y_snapshots_np = np.array(y_snapshots_list, dtype=np.float64)

    print(f"\n成功加载样本数: {len(branch_inputs_np)}")
    print(f"失败文件数: {num_failed}")
    print(f"A 网格点数: {len(unified_a_grid)}, tau 网格点数: {len(unified_tau_grid)}")
    print(f"输出场维数: {y_snapshots_np.shape[1]}")

    return (
        branch_inputs_np,
        y_snapshots_np,
        np.array(unified_a_grid, dtype=np.float64),
        np.array(unified_tau_grid, dtype=np.float64)
    )


def manual_scaler(data, mean=None, std=None):
    """
    手工标准化：
        x_scaled = (x - mean) / std

    如果未提供 mean/std，则基于当前 data 估计；
    若 std 太小，则置为 1，避免除零或放大数值噪声。
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std

    return (data - mean) / std


def inverse_manual_scaler(data_scaled, mean, std):
    """
    标准化的逆变换。
    """
    return data_scaled * std + mean


def pod(y_data_scaled, requested_num_modes):
    """
    对标准化后的输出场做 POD（SVD）。

    输入
    ----
    y_data_scaled : [N, output_dim]
    requested_num_modes : int

    返回
    ----
    y_mean_pod_scaled : [output_dim]
        POD 中使用的均值中心

    pod_basis : [output_dim, num_modes]
        POD 基底

    S : 奇异值序列
    actual_num_modes : 实际模式数
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
    多层感知机，用于学习：
        参数输入 -> POD 系数
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
    POD-DeepONet 的一个简化实现。

    结构：
    ----
    branch_x -> MLP -> POD coefficients
    POD coefficients @ POD basis^T + POD mean -> scaled field prediction
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

        # 固定 POD 基底与均值，不参与训练
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
        输入：
            branch_x: [B, branch_input_dim]

        输出：
            y_pred_scaled: [B, output_dim]
        """
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict_scaled(self, branch_x):
        """
        返回标准化空间中的预测。
        """
        self.eval()
        with torch.no_grad():
            return self.forward(branch_x)

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
        """
        返回反标准化后的物理量空间预测。

        参数
        ----
        branch_x : torch.Tensor
            标准化后的输入参数

        y_mean_scaler, y_std_scaler : np.ndarray 或 torch.Tensor
            输出场标准化参数
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
# 5. 物理损失定义
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    物理约束损失类。

    总损失由三部分构成：
    1. 数据拟合损失
    2. D2 非负性约束
    3. PDE 残差损失

    说明：
    ----
    - 数据损失在“标准化空间”中计算，数值更稳；
    - 物理损失在“真实物理量纲空间”中计算，因此会对预测输出先做反标准化；
    - PDE 残差基于统一均匀网格上的中心差分近似。
    """
    def __init__(self,
                 unified_a_grid,
                 unified_tau_grid,
                 branch_mean,
                 branch_std,
                 y_mean_scaler,
                 y_std_scaler,
                 d2_weight=0.5,
                 phys_weight=0.1):
        super().__init__()

        self.d2_weight = d2_weight
        self.phys_weight = phys_weight
        self.mse = nn.MSELoss()

        # 保存网格信息
        self.a_grid = torch.tensor(unified_a_grid, dtype=DTYPE, device=DEVICE)
        self.tau_grid = torch.tensor(unified_tau_grid, dtype=DTYPE, device=DEVICE)
        self.num_a = len(unified_a_grid)
        self.num_tau = len(unified_tau_grid)

        if self.num_a < 3 or self.num_tau < 3:
            raise ValueError("中心差分要求 unified_a_grid 和 unified_tau_grid 的长度至少为 3。")

        # 默认均匀网格
        self.da = self.a_grid[1] - self.a_grid[0]
        self.dtau = self.tau_grid[1] - self.tau_grid[0]

        # 保存标准化参数
        self.branch_mean = torch.tensor(branch_mean, dtype=DTYPE, device=DEVICE)
        self.branch_std = torch.tensor(branch_std, dtype=DTYPE, device=DEVICE)
        self.y_mean_scaler = torch.tensor(y_mean_scaler, dtype=DTYPE, device=DEVICE)
        self.y_std_scaler = torch.tensor(y_std_scaler, dtype=DTYPE, device=DEVICE)

    def forward(self, y_pred_scaled, y_true_scaled, branch_x_scaled):
        """
        输入：
        ----
        y_pred_scaled : [B, output_dim]
        y_true_scaled : [B, output_dim]
        branch_x_scaled : [B, 3]

        返回：
        ----
        total_loss
        data_loss
        loss_pos
        loss_pde
        loss_d1_data
        loss_d2_data
        """

        # ---------------------------------------------------------------------
        # Part A: 数据驱动损失（标准化空间）
        # ---------------------------------------------------------------------
        field_len = self.num_a * self.num_tau

        loss_d1_data = self.mse(y_pred_scaled[:, :field_len], y_true_scaled[:, :field_len])
        loss_d2_data = self.mse(y_pred_scaled[:, field_len:], y_true_scaled[:, field_len:])

        data_loss = (1.0 - self.d2_weight) * loss_d1_data + self.d2_weight * loss_d2_data

        # ---------------------------------------------------------------------
        # Part B: 物理损失在真实物理空间中计算
        # ---------------------------------------------------------------------
        y_pred = y_pred_scaled * self.y_std_scaler + self.y_mean_scaler
        branch_x = branch_x_scaled * self.branch_std + self.branch_mean

        # [B, Tau, A]
        D1_pred = y_pred[:, :field_len].view(-1, self.num_tau, self.num_a)
        D2_pred = y_pred[:, field_len:].view(-1, self.num_tau, self.num_a)

        # 当前 batch 的物理参数
        nu = branch_x[:, 0].view(-1, 1, 1)
        kappa = branch_x[:, 1].view(-1, 1, 1)
        d_diff = branch_x[:, 2].view(-1, 1, 1)

        # 网格张量
        A = self.a_grid.view(1, 1, -1)
        Tau = self.tau_grid.view(1, -1, 1)

        # 避免 a=0 导致 d/a 奇异
        A_safe = torch.clamp(A, min=1e-9)

        # ---------------------------------------------------------------------
        # 物理约束 A：扩散项严格非负约束
        # ---------------------------------------------------------------------
        loss_pos = torch.mean(torch.relu(-D2_pred) ** 2)

        # ---------------------------------------------------------------------
        # 物理约束 B：伴随 FP 方程 PDE 残差
        # ---------------------------------------------------------------------
        # 理论漂移/扩散项
        # D1_th = nu*a - (kappa/8)*a^3 + d/a
        # D2_th = d
        D1_th = nu * A - (kappa / 8.0) * (A ** 3) + (d_diff / A_safe)
        D2_th = d_diff.expand_as(D1_th)

        # U = tau * D_tau
        U1 = Tau * D1_pred
        U2 = Tau * D2_pred

        # 内部点（中心差分，剔除边界）
        U1_in = U1[:, 1:-1, 1:-1]
        U2_in = U2[:, 1:-1, 1:-1]
        D1_th_in = D1_th[:, :, 1:-1]
        D2_th_in = D2_th[:, :, 1:-1]

        # dU/dtau
        dU1_dtau = (U1[:, 2:, 1:-1] - U1[:, :-2, 1:-1]) / (2.0 * self.dtau)
        dU2_dtau = (U2[:, 2:, 1:-1] - U2[:, :-2, 1:-1]) / (2.0 * self.dtau)

        # dU/da
        dU1_da = (U1[:, 1:-1, 2:] - U1[:, 1:-1, :-2]) / (2.0 * self.da)
        dU2_da = (U2[:, 1:-1, 2:] - U2[:, 1:-1, :-2]) / (2.0 * self.da)

        # d2U/da2
        d2U1_da2 = (U1[:, 1:-1, 2:] - 2.0 * U1[:, 1:-1, 1:-1] + U1[:, 1:-1, :-2]) / (self.da ** 2)
        d2U2_da2 = (U2[:, 1:-1, 2:] - 2.0 * U2[:, 1:-1, 1:-1] + U2[:, 1:-1, :-2]) / (self.da ** 2)

        # PDE 残差
        R1 = dU1_dtau - (D1_th_in * dU1_da + D2_th_in * d2U1_da2) - D1_th_in

        R2 = (
            dU2_dtau
            - (D1_th_in * dU2_da + D2_th_in * d2U2_da2)
            - (D1_th_in * U1_in)
            - 2.0 * D2_th_in * dU1_da
            - D2_th_in
        )

        loss_pde = torch.mean(R1 ** 2) + torch.mean(R2 ** 2)

        # 总损失
        total_loss = data_loss + self.phys_weight * (loss_pos + 0.01 * loss_pde)

        return (
            total_loss,
            data_loss.detach(),
            loss_pos.detach(),
            loss_pde.detach(),
            loss_d1_data.detach(),
            loss_d2_data.detach()
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
# 7. 可视化工具函数
# =============================================================================

def save_logs_to_csv(logs: dict, path: str):
    """
    将 logs 字典保存为 CSV。

    由于训练日志和验证日志长度可能不同，这里拆成两个 DataFrame 再合并保存。
    """
    train_df = pd.DataFrame({
        'iter': logs['iter'],
        'train_loss': logs['train_loss'],
        'train_lr': logs['train_lr']
    })

    val_df = pd.DataFrame({
        'val_iter': logs['val_iter'],
        'val_loss': logs['val_loss'],
        'val_data': logs['val_data'],
        'val_d1': logs['val_d1'],
        'val_d2': logs['val_d2'],
        'val_pos': logs['val_pos'],
        'val_pde': logs['val_pde']
    })

    # 为方便查看，按最长长度对齐保存
    max_len = max(len(train_df), len(val_df))

    def pad_df(df, max_len):
        if len(df) < max_len:
            extra = pd.DataFrame(index=range(max_len - len(df)), columns=df.columns)
            df = pd.concat([df, extra], ignore_index=True)
        return df

    train_df = pad_df(train_df, max_len)
    val_df = pad_df(val_df, max_len)

    merged = pd.concat([train_df, val_df], axis=1)
    merged.to_csv(path, index=False, encoding='utf-8-sig')


def plot_loss_curves(logs, save_path, warmup_steps):
    """
    绘制训练/验证总损失及数据分项损失曲线。
    """
    plt.figure(figsize=(14, 8))

    # 总损失
    plt.plot(logs['iter'], logs['train_loss'], label='Training Total Loss', alpha=0.55, linewidth=1.4)
    plt.plot(logs['val_iter'], logs['val_loss'], label='Validation Total Loss', marker='o', markersize=4, linewidth=2.0)

    # 数据分项
    plt.plot(logs['val_iter'], logs['val_d1'], label='Validation Data Loss D1', linestyle='--', linewidth=1.6)
    plt.plot(logs['val_iter'], logs['val_d2'], label='Validation Data Loss D2', linestyle='--', linewidth=1.6)
    plt.plot(logs['val_iter'], logs['val_data'], label='Validation Data Loss (Weighted)', linestyle='-.', linewidth=1.8)

    # 最佳点
    if len(logs['val_loss']) > 0:
        best_idx = int(np.argmin(logs['val_loss']))
        best_iter = logs['val_iter'][best_idx]
        best_val_loss = logs['val_loss'][best_idx]

        plt.scatter(
            [best_iter], [best_val_loss],
            color='red', s=120, zorder=10,
            label=f'Best Val @ Iter {best_iter}'
        )

    # warmup 结束位置
    plt.axvline(x=warmup_steps, color='gray', linestyle=':', linewidth=1.8, label='Warmup End')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.title('Training / Validation Loss Curves', fontsize=15)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_physics_components(logs, save_path):
    """
    绘制验证集上的物理损失分量。
    """
    plt.figure(figsize=(14, 7))

    plt.plot(logs['val_iter'], logs['val_pos'], marker='s', markersize=4, label='Validation Positivity Loss')
    plt.plot(logs['val_iter'], logs['val_pde'], marker='^', markersize=4, label='Validation PDE Loss')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss Component', fontsize=12)
    plt.yscale('log')
    plt.title('Validation Physics-Informed Loss Components', fontsize=15)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


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
    绘制多个验证样本的“真值 vs 预测 vs 误差”图。

    每个样本绘制 6 张图：
    - D1 True
    - D1 Pred
    - |D1 Error|
    - D2 True
    - D2 Pred
    - |D2 Error|

    整体布局为 [num_samples, 6]
    """
    num_val = len(branch_val_scaled)
    if num_val == 0:
        print("验证集为空，跳过预测对比图。")
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
            extent=[unified_a_grid.min(), unified_a_grid.max(),
                    unified_tau_grid.min(), unified_tau_grid.max()],
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
            branch_input_scaled = torch.from_numpy(branch_val_scaled[idx]).unsqueeze(0).to(DEVICE)
            y_pred_np = model.predict(branch_input_scaled, y_mean_scaler, y_std_scaler).cpu().numpy().flatten()
            y_true_np = y_val_np[idx]

            field_len_half = len(y_true_np) // 2

            y_pred_d1 = y_pred_np[:field_len_half].reshape(num_tau, num_a)
            y_pred_d2 = y_pred_np[field_len_half:].reshape(num_tau, num_a)

            y_true_d1 = y_true_np[:field_len_half].reshape(num_tau, num_a)
            y_true_d2 = y_true_np[field_len_half:].reshape(num_tau, num_a)

            err_d1 = np.abs(y_pred_d1 - y_true_d1)
            err_d2 = np.abs(y_pred_d2 - y_true_d2)

            # 使用统一色标便于真值和预测比较
            vmin_d1 = min(y_true_d1.min(), y_pred_d1.min())
            vmax_d1 = max(y_true_d1.max(), y_pred_d1.max())

            vmin_d2 = min(y_true_d2.min(), y_pred_d2.min())
            vmax_d2 = max(y_true_d2.max(), y_pred_d2.max())

            # D1
            im1 = plot_field(axes[row_i, 0], y_true_d1, f'Sample {idx} - D1 True', vmin=vmin_d1, vmax=vmax_d1, cmap='viridis')
            im2 = plot_field(axes[row_i, 1], y_pred_d1, f'Sample {idx} - D1 Pred', vmin=vmin_d1, vmax=vmax_d1, cmap='viridis')
            im3 = plot_field(axes[row_i, 2], err_d1, f'Sample {idx} - |D1 Error|', cmap='magma')

            # D2
            im4 = plot_field(axes[row_i, 3], y_true_d2, f'Sample {idx} - D2 True', vmin=vmin_d2, vmax=vmax_d2, cmap='plasma')
            im5 = plot_field(axes[row_i, 4], y_pred_d2, f'Sample {idx} - D2 Pred', vmin=vmin_d2, vmax=vmax_d2, cmap='plasma')
            im6 = plot_field(axes[row_i, 5], err_d2, f'Sample {idx} - |D2 Error|', cmap='magma')

            # 为每张图加 colorbar
            fig.colorbar(im1, ax=axes[row_i, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[row_i, 1], fraction=0.046, pad=0.04)
            fig.colorbar(im3, ax=axes[row_i, 2], fraction=0.046, pad=0.04)
            fig.colorbar(im4, ax=axes[row_i, 3], fraction=0.046, pad=0.04)
            fig.colorbar(im5, ax=axes[row_i, 4], fraction=0.046, pad=0.04)
            fig.colorbar(im6, ax=axes[row_i, 5], fraction=0.046, pad=0.04)

            # 在行首补充一些简单误差信息
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
    plt.show()


# =============================================================================
# 8. 主程序
# =============================================================================

if __name__ == "__main__":
    try:
        print("=" * 80)
        print("POD-DeepONet + Physics-Informed Training")
        print("=" * 80)
        print(f"Device: {DEVICE}")
        print(f"Dtype : {DTYPE}")
        print(f"Result dir: {RESULT_DIR}")

        set_seed(SEED)

        # ---------------------------------------------------------------------
        # Step 1: 加载数据
        # ---------------------------------------------------------------------
        branch_inputs_np, y_snapshots_np, unified_a_grid, unified_tau_grid = load_unified_data(
            DATA_DIR, TARGET_NUM_FILES
        )

        # ---------------------------------------------------------------------
        # Step 2: 划分训练 / 验证集
        # ---------------------------------------------------------------------
        branch_train_np, branch_val_np, y_train_np, y_val_np = train_test_split(
            branch_inputs_np,
            y_snapshots_np,
            test_size=VALIDATION_SPLIT,
            random_state=SEED
        )

        print("\n数据划分完成：")
        print(f"Train samples: {len(branch_train_np)}")
        print(f"Val samples  : {len(branch_val_np)}")

        # ---------------------------------------------------------------------
        # Step 3: 标准化
        # ---------------------------------------------------------------------
        branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
        branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)

        y_train_scaled, y_mean_scaler, y_std_scaler = manual_scaler(y_train_np)
        y_val_scaled = manual_scaler(y_val_np, y_mean_scaler, y_std_scaler)

        # ---------------------------------------------------------------------
        # Step 4: POD 分解
        # ---------------------------------------------------------------------
        y_mean_pod_scaled, pod_basis, S, actual_num_modes = pod(
            y_train_scaled,
            REQUESTED_NUM_POD_MODES
        )

        energy = (S ** 2) / np.sum(S ** 2)
        cumulative_energy = np.cumsum(energy)
        retained_energy = cumulative_energy[actual_num_modes - 1]

        print("\nPOD 分解完成：")
        print(f"Requested POD modes: {REQUESTED_NUM_POD_MODES}")
        print(f"Actual POD modes   : {actual_num_modes}")
        print(f"Retained energy    : {retained_energy:.6f}")

        # ---------------------------------------------------------------------
        # Step 5: 构建 DataLoader
        # ---------------------------------------------------------------------
        # 保持原风格：直接移动到 DEVICE
        train_dataset = TensorDataset(
            torch.from_numpy(branch_train_scaled).to(DEVICE),
            torch.from_numpy(y_train_scaled).to(DEVICE)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(branch_val_scaled).to(DEVICE),
            torch.from_numpy(y_val_scaled).to(DEVICE)
        )

        if len(val_dataset) == 0:
            raise ValueError("验证集为空，无法进行训练监控。请检查数据划分。")

        train_loader = DataLoader(train_dataset, batch_size=ADAM_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        # ---------------------------------------------------------------------
        # Step 6: 初始化模型、优化器、损失函数、调度器
        # ---------------------------------------------------------------------
        model = PODDeepONet(
            branch_input_dim=BRANCH_INPUT_DIM,
            hidden_units=HIDDEN_UNITS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_pod_modes=actual_num_modes,
            pod_basis=pod_basis,
            y_mean_pod_scaled=y_mean_pod_scaled,
            dropout_rate=DROPOUT_RATE
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)

        loss_fn = PhysicsInformedLoss(
            unified_a_grid=unified_a_grid,
            unified_tau_grid=unified_tau_grid,
            branch_mean=branch_mean,
            branch_std=branch_std,
            y_mean_scaler=y_mean_scaler,
            y_std_scaler=y_std_scaler,
            d2_weight=D2_LOSS_WEIGHT,
            phys_weight=PHYSICS_LOSS_WEIGHT
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            WARMUP_STEPS,
            ADAM_ITERATIONS
        )

        # ---------------------------------------------------------------------
        # Step 7: 日志容器
        # ---------------------------------------------------------------------
        logs = {
            'iter': [],
            'train_loss': [],
            'train_lr': [],

            'val_iter': [],
            'val_loss': [],
            'val_data': [],
            'val_d1': [],
            'val_d2': [],
            'val_pos': [],
            'val_pde': []
        }

        min_val_loss = float('inf')
        best_iter = -1
        early_stop_counter = 0

        # ---------------------------------------------------------------------
        # Step 8: 训练
        # ---------------------------------------------------------------------
        print("\n--- 开始 Physics-Informed AdamW 训练 ---")
        pbar = tqdm(total=ADAM_ITERATIONS, desc="训练进度")
        global_iter = 0
        done = False

        while not done:
            for batch_branch, batch_y_scaled in train_loader:
                if global_iter >= ADAM_ITERATIONS:
                    done = True
                    break

                model.train()
                optimizer.zero_grad()

                y_pred_scaled = model(batch_branch)

                (
                    loss,
                    loss_data,
                    loss_pos,
                    loss_pde,
                    loss_d1_data,
                    loss_d2_data
                ) = loss_fn(y_pred_scaled, batch_y_scaled, batch_branch)

                loss.backward()
                optimizer.step()
                scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']

                logs['iter'].append(global_iter)
                logs['train_loss'].append(loss.item())
                logs['train_lr'].append(current_lr)

                pbar.update(1)

                # 每隔固定频率做一次验证
                if global_iter % VALIDATION_FREQUENCY == 0:
                    model.eval()
                    with torch.no_grad():
                        val_branch_all, val_y_all_scaled = next(iter(val_loader))
                        val_y_pred_scaled = model(val_branch_all)

                        (
                            val_loss,
                            val_data,
                            val_pos,
                            val_pde,
                            val_d1,
                            val_d2
                        ) = loss_fn(val_y_pred_scaled, val_y_all_scaled, val_branch_all)

                        logs['val_iter'].append(global_iter)
                        logs['val_loss'].append(val_loss.item())
                        logs['val_data'].append(val_data.item())
                        logs['val_d1'].append(val_d1.item())
                        logs['val_d2'].append(val_d2.item())
                        logs['val_pos'].append(val_pos.item())
                        logs['val_pde'].append(val_pde.item())

                        print(
                            f"\nIter: {global_iter:6d} | "
                            f"Train: {loss.item():.4e} | "
                            f"ValTot: {val_loss.item():.4e} | "
                            f"ValData: {val_data.item():.4e} | "
                            f"ValD1: {val_d1.item():.4e} | "
                            f"ValD2: {val_d2.item():.4e} | "
                            f"ValPos: {val_pos.item():.4e} | "
                            f"ValPDE: {val_pde.item():.4e} | "
                            f"LR: {current_lr:.3e}"
                        )

                    # 保存最佳模型
                    if val_loss.item() < min_val_loss:
                        min_val_loss = val_loss.item()
                        best_iter = global_iter
                        early_stop_counter = 0

                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                        print("\n*** Early stopping triggered. ***")
                        done = True
                        break

                global_iter += 1

        pbar.close()

        # ---------------------------------------------------------------------
        # Step 9: 保存 scaler / POD / 元信息
        # ---------------------------------------------------------------------
        scaler_payload = {
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
            'config': {
                'BRANCH_INPUT_DIM': BRANCH_INPUT_DIM,
                'HIDDEN_UNITS': HIDDEN_UNITS,
                'NUM_HIDDEN_LAYERS': NUM_HIDDEN_LAYERS,
                'REQUESTED_NUM_POD_MODES': REQUESTED_NUM_POD_MODES,
                'ACTUAL_NUM_POD_MODES': actual_num_modes,
                'DROPOUT_RATE': DROPOUT_RATE,
                'WEIGHT_DECAY': WEIGHT_DECAY,
                'ADAM_LR': ADAM_LR,
                'ADAM_BATCH_SIZE': ADAM_BATCH_SIZE,
                'ADAM_ITERATIONS': ADAM_ITERATIONS,
                'D2_LOSS_WEIGHT': D2_LOSS_WEIGHT,
                'PHYSICS_LOSS_WEIGHT': PHYSICS_LOSS_WEIGHT,
                'VALIDATION_SPLIT': VALIDATION_SPLIT,
                'SEED': SEED,
                'VALIDATION_FREQUENCY': VALIDATION_FREQUENCY,
                'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
                'WARMUP_STEPS': WARMUP_STEPS,
                'TARGET_NUM_FILES': TARGET_NUM_FILES,
                'DEVICE': str(DEVICE),
                'DTYPE': str(DTYPE)
            }
        }
        torch.save(scaler_payload, SCALER_SAVE_PATH)

        # 保存日志 CSV
        save_logs_to_csv(logs, LOG_CSV_PATH)

        # 保存摘要说明
        summary_text = (
            f"RUN_ID: {RUN_ID}\n"
            f"Train samples: {len(branch_train_np)}\n"
            f"Val samples: {len(branch_val_np)}\n"
            f"A points: {len(unified_a_grid)}\n"
            f"tau points: {len(unified_tau_grid)}\n"
            f"Output dim: {y_snapshots_np.shape[1]}\n"
            f"Requested POD modes: {REQUESTED_NUM_POD_MODES}\n"
            f"Actual POD modes: {actual_num_modes}\n"
            f"Retained energy: {retained_energy:.8f}\n"
            f"Best val loss: {min_val_loss:.8e}\n"
            f"Best iter: {best_iter}\n"
            f"Device: {DEVICE}\n"
            f"Dtype: {DTYPE}\n"
            f"Model saved to: {MODEL_SAVE_PATH}\n"
            f"Scaler saved to: {SCALER_SAVE_PATH}\n"
            f"Logs saved to: {LOG_CSV_PATH}\n"
        )
        save_text_summary(SUMMARY_TXT_PATH, summary_text)

        # ---------------------------------------------------------------------
        # Step 10: 绘图（训练曲线 / 物理损失曲线）
        # ---------------------------------------------------------------------
        print("\n--- 绘制训练曲线 ---")
        plot_loss_curves(logs, LOSS_PLOT_SAVE_PATH, WARMUP_STEPS)

        print("\n--- 绘制物理损失分项曲线 ---")
        plot_physics_components(logs, PHYSICS_PLOT_SAVE_PATH)

        # ---------------------------------------------------------------------
        # Step 11: 加载最佳模型并做可视化验证
        # ---------------------------------------------------------------------
        print("\n--- 可视化验证（加载最佳模型）---")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.eval()

        plot_prediction_comparison(
            model=model,
            branch_val_scaled=branch_val_scaled,
            y_val_np=y_val_np,
            unified_a_grid=unified_a_grid,
            unified_tau_grid=unified_tau_grid,
            y_mean_scaler=y_mean_scaler,
            y_std_scaler=y_std_scaler,
            save_path=PREDICTION_PLOT_PATH,
            num_samples=4
        )

        print("\n训练完成，结果已保存。")
        print(f"最佳模型: {MODEL_SAVE_PATH}")
        print(f"Scaler/POD信息: {SCALER_SAVE_PATH}")
        print(f"训练日志: {LOG_CSV_PATH}")
        print(f"训练曲线图: {LOSS_PLOT_SAVE_PATH}")
        print(f"物理损失图: {PHYSICS_PLOT_SAVE_PATH}")
        print(f"预测对比图: {PREDICTION_PLOT_PATH}")
        print(f"运行摘要: {SUMMARY_TXT_PATH}")

    except Exception as e:
        print("\n程序运行失败，异常信息如下：")
        traceback.print_exc()
        raise