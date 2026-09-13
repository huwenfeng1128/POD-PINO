# -*- coding: utf-8 -*-
"""
POD-DeepONet 物理残差消融实验
================================

实验目标
--------
In完全相同的：
1. 数据文件与训练/验证划分；
2. 输入/输出标准化参数；
3. POD 基底；
4. 网络初始权重；
5. 优化器、学习率计划与随机种子；

条件下训练两个模型：

A. no_pde_residual
   总损失 = 数据损失

B. with_pde_residual
   总损失 = 数据损失 + PDE 物理残差

注意
----
- 为保证模型选择标准公平，两个实验都使用：
      validation common objective = validation data loss
  保存最佳模型。PDE 项只参与 with_pde_residual 的训练梯度，不参与两组共同的
  最佳模型选择指标。
- 本程序同时记录两类“物理梯度”：
  1. PDE 损失对网络参数的梯度范数、有效加权梯度范数、与数据梯度的夹角；
  2. 预测物理场 U=tau*D 的有限差分导数误差，包括 dU/dtau、dU/dA、d2U/dA2。
- 程序不会预设“物理约束一定有效”。运行结束后会根据实际指标自动生成报告：
  若 PDE 残差、物理导数误差下降，且 PDE 有效梯度非零，则报告支持物理约束发挥作用；
  否则报告会如实说明证据不足。

主要输出
--------
RESULT_DIR/
├─ shared_preprocessing.pth
├─ data_split_manifest.csv
├─ config.json
├─ no_pde_residual/
│  ├─ best_model.pth
│  ├─ train_logs.csv
│  ├─ validation_logs.csv
│  ├─ gradient_diagnostics.csv
│  ├─ validation_predictions.npz
│  ├─ sample_metrics.csv
│  └─ evaluation_metrics.json
├─ with_pde_residual/
│  └─ ...
├─ ablation_summary.csv
├─ physical_gradient_metrics.csv
├─ loss_ablation_comparison.png
├─ parameter_gradient_comparison.png
├─ physical_field_gradient_comparison.png
├─ metric_ablation_comparison.png
├─ prediction_D1_ablation.png
├─ prediction_D2_ablation.png
├─ pde_residual_map_ablation.png
└─ ablation_report.txt
"""

import os
import glob
import math
import json
import copy
import random
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

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
RESULT_DIR = (
    r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\xiaorongshiyan\train_result_physics_ablation'
)
DATA_DIR = (
    r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA'
)
RUN_ID = "pod_physics_residual_ablation_v1"

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
EVAL_BATCH_SIZE = 128
ADAM_ITERATIONS = 200000
VALIDATION_SPLIT = 0.2
VALIDATION_FREQUENCY = 200
WARMUP_STEPS = 5000
SEED = 24
TARGET_NUM_FILES = 2500

# 严格消融默认使用相同训练步数，避免两个实验因早停步数不同而引入额外变量。
# 若显式开启早停，两个实验仍使用相同的 common objective 作为监控指标。
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20
MIN_ITERATIONS_BEFORE_EARLY_STOPPING = 10000

# --- 损失权重 ---
# D1/D2 数据项内部权重：data=(1-D2_LOSS_WEIGHT)*D1 + D2_LOSS_WEIGHT*D2
D2_LOSS_WEIGHT = 0.2

# PDE 原始残差权重
PHYSICS_LOSS_WEIGHT = 0.1

# --- 梯度诊断 ---
# 每隔多少次验证计算一次参数梯度诊断。该诊断需要额外反向传播，设为 5 可控制开销。
GRADIENT_DIAGNOSTIC_EVERY_N_VALIDATIONS = 5
GRADIENT_DIAGNOSTIC_BATCH_SIZE = 32
GRAD_EPS = 1e-30

# --- 输出配置 ---
NUM_PLOT_SAMPLES = 4
PLOT_RANDOM_SEED = 2026
SAVE_FULL_VALIDATION_PREDICTIONS = True
FIG_DPI = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PIN_MEMORY = torch.cuda.is_available()

EXPERIMENTS = {
    "no_pde_residual": False,
    "with_pde_residual": True,
}

os.makedirs(RESULT_DIR, exist_ok=True)


# =============================================================================
# 2. 基础工具
# =============================================================================

def set_seed(seed: int = 24) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 提升可复现性。某些 CUDA 算子在严格确定性模式下可能较慢。
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    file_path: str,
) -> None:
    """检查 CSV 是否包含必需字段。"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"文件 {file_path} 缺少必要列: {missing}")


def is_uniform_grid(arr: np.ndarray, tol: float = 1e-12) -> bool:
    """检查一维网格是否均匀。"""
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], atol=tol, rtol=tol)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    """保存 JSON，兼容 NumPy 标量和数组。"""

    def converter(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.dtype):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=converter)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_torch_load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """兼容不同 PyTorch 版本加载纯 state_dict。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(DEVICE, non_blocking=PIN_MEMORY)


def positive_for_log(values, floor: float = 1e-30) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.maximum(arr, floor)


def moving_average(values: List[float], window: int = 200) -> np.ndarray:
    """用于绘图的移动平均，不改变原始保存数据。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if len(arr) < window:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()


# =============================================================================
# 3. 数据加载与预处理
# =============================================================================

def load_unified_data(
    data_dir: str,
    target_num_files: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict]]:
    """
    从多个 data_*.csv 文件读取统一网格快照。

    返回
    ----
    branch_inputs_np : [N, 3]
    y_snapshots_np   : [N, 2*num_tau*num_a]
    unified_a_grid   : [num_a]
    unified_tau_grid : [num_tau]
    valid_files      : 成功加载的文件路径
    failed_records   : 失败文件及原因
    """
    print(f"从 {data_dir} 加载统一数据...")
    all_available_files = sorted(glob.glob(os.path.join(data_dir, "data_*.csv")))
    if not all_available_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何 data_*.csv 文件。")

    rng = random.Random(seed)
    selected_files = rng.sample(
        all_available_files,
        min(len(all_available_files), target_num_files),
    )

    branch_inputs_list: List[np.ndarray] = []
    y_snapshots_list: List[np.ndarray] = []
    valid_files: List[str] = []
    failed_records: List[Dict] = []

    unified_a_grid: Optional[np.ndarray] = None
    unified_tau_grid: Optional[np.ndarray] = None

    required_cols = [
        "A",
        "tau",
        "nu",
        "kappa",
        "d_diffusion",
        "D1_adj",
        "D2_adj",
    ]

    for file_path in tqdm(selected_files, desc="加载快照"):
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")
            ensure_required_columns(df, required_cols, file_path)

            # 删除关键字段缺失行。
            df = df.dropna(subset=required_cols)
            if df.empty:
                raise ValueError("删除缺失值后为空表。")

            a_grid = np.sort(df["A"].unique().astype(np.float64))
            tau_grid = np.sort(df["tau"].unique().astype(np.float64))

            if len(a_grid) < 3 or len(tau_grid) < 3:
                raise ValueError(
                    "网格点数量不足；中心差分要求 A 和 tau 两个方向均至少 3 点。"
                )

            if not is_uniform_grid(a_grid):
                raise ValueError("A 网格不是均匀网格。")
            if not is_uniform_grid(tau_grid):
                raise ValueError("tau 网格不是均匀网格。")

            if unified_a_grid is None:
                unified_a_grid = a_grid
                unified_tau_grid = tau_grid
            else:
                if not np.allclose(a_grid, unified_a_grid, atol=1e-12, rtol=1e-12):
                    raise ValueError("A 网格与统一网格不一致。")
                if not np.allclose(
                    tau_grid,
                    unified_tau_grid,
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError("tau 网格与统一网格不一致。")

            # 检查同一文件中参数是否恒定。
            for col in ["nu", "kappa", "d_diffusion"]:
                if df[col].nunique(dropna=True) != 1:
                    raise ValueError(f"同一文件内参数 {col} 不是常数。")

            # 检查 (tau, A) 是否为完整且无重复的张量积网格。
            duplicated = df.duplicated(subset=["tau", "A"]).any()
            if duplicated:
                raise ValueError("存在重复的 (tau, A) 网格点。")

            expected_len = len(unified_a_grid) * len(unified_tau_grid)
            df_sorted = df.sort_values(by=["tau", "A"]).reset_index(drop=True)
            if len(df_sorted) != expected_len:
                raise ValueError(
                    f"网格不完整。期望 {expected_len} 行，实际 {len(df_sorted)} 行。"
                )

            params = df_sorted[["nu", "kappa", "d_diffusion"]].iloc[0]
            params_np = params.to_numpy(dtype=np.float64)

            snapshot = np.concatenate(
                [
                    df_sorted["D1_adj"].to_numpy(dtype=np.float64),
                    df_sorted["D2_adj"].to_numpy(dtype=np.float64),
                ]
            )

            if not np.all(np.isfinite(params_np)):
                raise ValueError("输入参数包含 NaN 或 Inf。")
            if not np.all(np.isfinite(snapshot)):
                raise ValueError("输出快照包含 NaN 或 Inf。")

            branch_inputs_list.append(params_np)
            y_snapshots_list.append(snapshot)
            valid_files.append(file_path)

        except Exception as exc:
            failed_records.append(
                {
                    "file": file_path,
                    "reason": str(exc),
                }
            )
            print(f"\n[警告] 文件读取失败: {file_path}\n原因: {exc}")

    if not branch_inputs_list:
        raise ValueError("未成功加载任何有效样本，请检查数据格式与网格。")

    branch_inputs_np = np.asarray(branch_inputs_list, dtype=np.float64)
    y_snapshots_np = np.asarray(y_snapshots_list, dtype=np.float64)

    assert unified_a_grid is not None
    assert unified_tau_grid is not None

    print(f"\n成功加载样本数: {len(branch_inputs_np)}")
    print(f"失败文件数: {len(failed_records)}")
    print(f"A 网格点数: {len(unified_a_grid)}")
    print(f"tau 网格点数: {len(unified_tau_grid)}")
    print(f"输出场维数: {y_snapshots_np.shape[1]}")

    return (
        branch_inputs_np,
        y_snapshots_np,
        np.asarray(unified_a_grid, dtype=np.float64),
        np.asarray(unified_tau_grid, dtype=np.float64),
        valid_files,
        failed_records,
    )


def manual_scaler(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
):
    """手工标准化。"""
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.asarray(std, dtype=np.float64)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def pod(
    y_data_scaled: np.ndarray,
    requested_num_modes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """对训练集标准化快照执行 POD/SVD。"""
    y_mean_pod_scaled = np.mean(y_data_scaled, axis=0)
    centered = y_data_scaled - y_mean_pod_scaled
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    actual_num_modes = min(requested_num_modes, vt.shape[0])
    pod_basis = vt.T[:, :actual_num_modes]
    return y_mean_pod_scaled, pod_basis, singular_values, actual_num_modes


# =============================================================================
# 4. 模型定义
# =============================================================================

class MLP(nn.Module):
    """参数输入到 POD 系数的映射。"""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()

        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_units, dtype=DTYPE)
        ]

        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.GELU(),
                    nn.LayerNorm(hidden_units, dtype=DTYPE),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(hidden_units, hidden_units, dtype=DTYPE),
                ]
            )

        layers.extend(
            [
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_units, output_dim, dtype=DTYPE),
            ]
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PODDeepONet(nn.Module):
    """简化 POD-DeepONet：参数 -> POD 系数 -> 场。"""

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
            input_dim=branch_input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            output_dim=num_pod_modes,
            dropout_rate=dropout_rate,
        )

        # register_buffer 会随 model.to(device) 自动迁移，但不参与优化。
        self.register_buffer(
            "pod_basis",
            torch.tensor(pod_basis, dtype=DTYPE),
        )
        self.register_buffer(
            "y_mean_pod_scaled",
            torch.tensor(y_mean_pod_scaled, dtype=DTYPE),
        )

    def forward(self, branch_x: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


# =============================================================================
# 5. 物理损失与残差
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    统一计算数据项和 PDE 项。

    use_pde=False 时，仅将 PDE 项从训练总损失中移除；PDE 残差仍会被计算和记录，
    因此可以直接比较无物理约束模型的物理一致性。
    """

    def __init__(
        self,
        unified_a_grid: np.ndarray,
        unified_tau_grid: np.ndarray,
        branch_mean: np.ndarray,
        branch_std: np.ndarray,
        y_mean_scaler: np.ndarray,
        y_std_scaler: np.ndarray,
        d2_data_weight: float,
        physics_weight: float,
    ):
        super().__init__()

        self.d2_data_weight = float(d2_data_weight)
        self.physics_weight = float(physics_weight)
        self.mse = nn.MSELoss()

        self.num_a = len(unified_a_grid)
        self.num_tau = len(unified_tau_grid)
        if self.num_a < 3 or self.num_tau < 3:
            raise ValueError("中心差分要求 A 和 tau 网格长度至少为 3。")

        self.register_buffer(
            "a_grid",
            torch.tensor(unified_a_grid, dtype=DTYPE),
        )
        self.register_buffer(
            "tau_grid",
            torch.tensor(unified_tau_grid, dtype=DTYPE),
        )
        self.register_buffer(
            "branch_mean",
            torch.tensor(branch_mean, dtype=DTYPE),
        )
        self.register_buffer(
            "branch_std",
            torch.tensor(branch_std, dtype=DTYPE),
        )
        self.register_buffer(
            "y_mean_scaler",
            torch.tensor(y_mean_scaler, dtype=DTYPE),
        )
        self.register_buffer(
            "y_std_scaler",
            torch.tensor(y_std_scaler, dtype=DTYPE),
        )

        self.da = float(unified_a_grid[1] - unified_a_grid[0])
        self.dtau = float(unified_tau_grid[1] - unified_tau_grid[0])

    @property
    def effective_pde_coefficient(self) -> float:
        return self.physics_weight

    def compute_residual_maps(
        self,
        y_pred_scaled: torch.Tensor,
        branch_x_scaled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """返回 R1、R2 以及预测场的中心差分物理导数。"""
        field_len = self.num_a * self.num_tau

        y_pred = y_pred_scaled * self.y_std_scaler + self.y_mean_scaler
        branch_x = branch_x_scaled * self.branch_std + self.branch_mean

        d1_pred = y_pred[:, :field_len].view(-1, self.num_tau, self.num_a)
        d2_pred = y_pred[:, field_len:].view(-1, self.num_tau, self.num_a)

        nu = branch_x[:, 0].view(-1, 1, 1)
        kappa = branch_x[:, 1].view(-1, 1, 1)
        d_diff = branch_x[:, 2].view(-1, 1, 1)

        a = self.a_grid.view(1, 1, -1)
        tau = self.tau_grid.view(1, -1, 1)
        a_safe = torch.clamp(a, min=1e-9)

        d1_th = nu * a - (kappa / 8.0) * (a ** 3) + d_diff / a_safe
        d2_th = d_diff.expand_as(d1_th)

        u1 = tau * d1_pred
        u2 = tau * d2_pred

        u1_in = u1[:, 1:-1, 1:-1]

        d1_th_in = d1_th[:, :, 1:-1]
        d2_th_in = d2_th[:, :, 1:-1]

        du1_dtau = (u1[:, 2:, 1:-1] - u1[:, :-2, 1:-1]) / (2.0 * self.dtau)
        du2_dtau = (u2[:, 2:, 1:-1] - u2[:, :-2, 1:-1]) / (2.0 * self.dtau)

        du1_da = (u1[:, 1:-1, 2:] - u1[:, 1:-1, :-2]) / (2.0 * self.da)
        du2_da = (u2[:, 1:-1, 2:] - u2[:, 1:-1, :-2]) / (2.0 * self.da)

        d2u1_da2 = (
            u1[:, 1:-1, 2:]
            - 2.0 * u1[:, 1:-1, 1:-1]
            + u1[:, 1:-1, :-2]
        ) / (self.da ** 2)

        d2u2_da2 = (
            u2[:, 1:-1, 2:]
            - 2.0 * u2[:, 1:-1, 1:-1]
            + u2[:, 1:-1, :-2]
        ) / (self.da ** 2)

        r1 = (
            du1_dtau
            - (d1_th_in * du1_da + d2_th_in * d2u1_da2)
            - d1_th_in
        )

        r2 = (
            du2_dtau
            - (d1_th_in * du2_da + d2_th_in * d2u2_da2)
            - d1_th_in * u1_in
            - 2.0 * d2_th_in * du1_da
            - d2_th_in
        )

        derivatives = {
            "du1_dtau": du1_dtau,
            "du1_da": du1_da,
            "d2u1_da2": d2u1_da2,
            "du2_dtau": du2_dtau,
            "du2_da": du2_da,
            "d2u2_da2": d2u2_da2,
        }
        return r1, r2, derivatives

    def compute_components(
        self,
        y_pred_scaled: torch.Tensor,
        y_true_scaled: torch.Tensor,
        branch_x_scaled: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        field_len = self.num_a * self.num_tau

        loss_d1 = self.mse(
            y_pred_scaled[:, :field_len],
            y_true_scaled[:, :field_len],
        )
        loss_d2 = self.mse(
            y_pred_scaled[:, field_len:],
            y_true_scaled[:, field_len:],
        )
        data_loss = (
            (1.0 - self.d2_data_weight) * loss_d1
            + self.d2_data_weight * loss_d2
        )

        r1, r2, _ = self.compute_residual_maps(y_pred_scaled, branch_x_scaled)
        pde_r1 = torch.mean(r1 ** 2)
        pde_r2 = torch.mean(r2 ** 2)
        pde_loss = pde_r1 + pde_r2

        weighted_pde_loss = self.effective_pde_coefficient * pde_loss

        return {
            "data": data_loss,
            "d1": loss_d1,
            "d2": loss_d2,
            "pde": pde_loss,
            "pde_r1": pde_r1,
            "pde_r2": pde_r2,
            "common": data_loss,
            "weighted_pde": weighted_pde_loss,
        }

    def total_loss(
        self,
        components: Dict[str, torch.Tensor],
        use_pde: bool,
    ) -> torch.Tensor:
        if use_pde:
            return components["common"] + components["weighted_pde"]
        return components["common"]


# =============================================================================
# 6. 学习率调度器
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """线性 warmup + cosine 衰减。"""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# =============================================================================
# 7. 数据结构
# =============================================================================

@dataclass
class SharedData:
    branch_train_scaled: np.ndarray
    branch_val_scaled: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    branch_train_physical: np.ndarray
    branch_val_physical: np.ndarray
    y_train_physical: np.ndarray
    y_val_physical: np.ndarray
    branch_mean: np.ndarray
    branch_std: np.ndarray
    y_mean_scaler: np.ndarray
    y_std_scaler: np.ndarray
    y_mean_pod_scaled: np.ndarray
    pod_basis: np.ndarray
    singular_values: np.ndarray
    actual_num_modes: int
    unified_a_grid: np.ndarray
    unified_tau_grid: np.ndarray


@dataclass
class ExperimentResult:
    name: str
    use_pde: bool
    model_path: str
    train_log_path: str
    val_log_path: str
    grad_log_path: str
    metrics_path: str
    predictions_path: str
    sample_metrics_path: str
    best_iter: int
    best_common_objective: float
    metrics: Dict[str, float]
    train_logs: Dict[str, List[float]]
    val_logs: Dict[str, List[float]]
    grad_logs: Dict[str, List[float]]
    predictions: np.ndarray
    sample_metrics: pd.DataFrame


# =============================================================================
# 8. 训练与梯度诊断
# =============================================================================

def build_model(shared: SharedData) -> PODDeepONet:
    return PODDeepONet(
        branch_input_dim=BRANCH_INPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_pod_modes=shared.actual_num_modes,
        pod_basis=shared.pod_basis,
        y_mean_pod_scaled=shared.y_mean_pod_scaled,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)


def build_loss(shared: SharedData) -> PhysicsInformedLoss:
    return PhysicsInformedLoss(
        unified_a_grid=shared.unified_a_grid,
        unified_tau_grid=shared.unified_tau_grid,
        branch_mean=shared.branch_mean,
        branch_std=shared.branch_std,
        y_mean_scaler=shared.y_mean_scaler,
        y_std_scaler=shared.y_std_scaler,
        d2_data_weight=D2_LOSS_WEIGHT,
        physics_weight=PHYSICS_LOSS_WEIGHT,
    ).to(DEVICE)


def make_train_loader(shared: SharedData, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(shared.branch_train_scaled),
        torch.from_numpy(shared.y_train_scaled),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=ADAM_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        generator=generator,
        drop_last=False,
    )


def make_val_loader(shared: SharedData) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(shared.branch_val_scaled),
        torch.from_numpy(shared.y_val_scaled),
    )
    return DataLoader(
        dataset,
        batch_size=min(EVAL_BATCH_SIZE, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def empty_train_logs() -> Dict[str, List[float]]:
    return {
        "iter": [],
        "train_total": [],
        "train_common": [],
        "train_data": [],
        "train_d1": [],
        "train_d2": [],
        "train_pde": [],
        "train_pde_r1": [],
        "train_pde_r2": [],
        "train_weighted_pde": [],
        "lr": [],
    }


def empty_val_logs() -> Dict[str, List[float]]:
    return {
        "iter": [],
        "val_total": [],
        "val_common": [],
        "val_data": [],
        "val_d1": [],
        "val_d2": [],
        "val_pde": [],
        "val_pde_r1": [],
        "val_pde_r2": [],
        "val_weighted_pde": [],
    }


def empty_grad_logs() -> Dict[str, List[float]]:
    return {
        "iter": [],
        "data_grad_norm": [],
        "raw_pde_grad_norm": [],
        "effective_pde_grad_norm": [],
        "total_grad_norm": [],
        "raw_pde_to_data_ratio": [],
        "effective_pde_to_data_ratio": [],
        "data_pde_cosine": [],
        "diagnostic_data_loss": [],
        "diagnostic_pde_loss": [],
    }


def append_train_log(
    logs: Dict[str, List[float]],
    step: int,
    total: torch.Tensor,
    comp: Dict[str, torch.Tensor],
    lr: float,
) -> None:
    logs["iter"].append(step)
    logs["train_total"].append(float(total.detach().cpu()))
    logs["train_common"].append(float(comp["common"].detach().cpu()))
    logs["train_data"].append(float(comp["data"].detach().cpu()))
    logs["train_d1"].append(float(comp["d1"].detach().cpu()))
    logs["train_d2"].append(float(comp["d2"].detach().cpu()))
    logs["train_pde"].append(float(comp["pde"].detach().cpu()))
    logs["train_pde_r1"].append(float(comp["pde_r1"].detach().cpu()))
    logs["train_pde_r2"].append(float(comp["pde_r2"].detach().cpu()))
    logs["train_weighted_pde"].append(float(comp["weighted_pde"].detach().cpu()))
    logs["lr"].append(float(lr))


def evaluate_loss_components(
    model: PODDeepONet,
    loss_fn: PhysicsInformedLoss,
    val_loader: DataLoader,
    use_pde: bool,
) -> Dict[str, float]:
    """按样本数加权汇总验证损失。"""
    model.eval()
    sums = {
        "total": 0.0,
        "common": 0.0,
        "data": 0.0,
        "d1": 0.0,
        "d2": 0.0,
        "pde": 0.0,
        "pde_r1": 0.0,
        "pde_r2": 0.0,
        "weighted_pde": 0.0,
    }
    n_total = 0

    with torch.no_grad():
        for branch_batch, y_batch in val_loader:
            branch_batch = to_device(branch_batch)
            y_batch = to_device(y_batch)
            pred = model(branch_batch)
            comp = loss_fn.compute_components(pred, y_batch, branch_batch)
            total = loss_fn.total_loss(comp, use_pde=use_pde)

            batch_n = len(branch_batch)
            n_total += batch_n
            sums["total"] += float(total.cpu()) * batch_n
            for key in [
                "common",
                "data",
                "d1",
                "d2",
                "pde",
                "pde_r1",
                "pde_r2",
                "weighted_pde",
            ]:
                sums[key] += float(comp[key].cpu()) * batch_n

    if n_total == 0:
        raise ValueError("验证集为空。")

    return {key: value / n_total for key, value in sums.items()}


def append_val_log(
    logs: Dict[str, List[float]],
    step: int,
    metrics: Dict[str, float],
) -> None:
    logs["iter"].append(step)
    for key in [
        "total",
        "common",
        "data",
        "d1",
        "d2",
        "pde",
        "pde_r1",
        "pde_r2",
        "weighted_pde",
    ]:
        logs[f"val_{key}"].append(float(metrics[key]))


def gradients_to_vector(
    grads: Tuple[Optional[torch.Tensor], ...],
    params: List[torch.nn.Parameter],
) -> torch.Tensor:
    """将 autograd.grad 返回值转换为统一扁平向量；未使用参数补零。"""
    vectors = []
    for grad, param in zip(grads, params):
        if grad is None:
            vectors.append(torch.zeros_like(param).reshape(-1))
        else:
            vectors.append(grad.reshape(-1))
    if not vectors:
        return torch.zeros(1, dtype=DTYPE, device=DEVICE)
    return torch.cat(vectors)


def compute_gradient_diagnostics(
    model: PODDeepONet,
    loss_fn: PhysicsInformedLoss,
    branch_diag: torch.Tensor,
    y_diag: torch.Tensor,
    use_pde: bool,
) -> Dict[str, float]:
    """
    计算各损失分量对可训练参数的梯度。

    raw_pde_grad_norm：原始 PDE 残差梯度范数，用于判断模型所在位置的物理敏感性。
    effective_pde_grad_norm：训练时真正注入优化器的加权 PDE 梯度；无 PDE 组恒为 0。
    data_pde_cosine：数据梯度与原始 PDE 梯度余弦，相反方向时为负。
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    pred = model(branch_diag)
    comp = loss_fn.compute_components(pred, y_diag, branch_diag)
    total = loss_fn.total_loss(comp, use_pde=use_pde)

    g_data = torch.autograd.grad(
        comp["data"],
        params,
        retain_graph=True,
        allow_unused=True,
    )
    g_pde = torch.autograd.grad(
        comp["pde"],
        params,
        retain_graph=True,
        allow_unused=True,
    )
    g_total = torch.autograd.grad(
        total,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    v_data = gradients_to_vector(g_data, params)
    v_pde = gradients_to_vector(g_pde, params)
    v_total = gradients_to_vector(g_total, params)

    data_norm = torch.linalg.vector_norm(v_data)
    raw_pde_norm = torch.linalg.vector_norm(v_pde)
    total_norm = torch.linalg.vector_norm(v_total)

    effective_coeff = loss_fn.effective_pde_coefficient if use_pde else 0.0
    effective_pde_norm = abs(effective_coeff) * raw_pde_norm

    cosine = torch.dot(v_data, v_pde) / (
        data_norm * raw_pde_norm + GRAD_EPS
    )

    result = {
        "data_grad_norm": float(data_norm.detach().cpu()),
        "raw_pde_grad_norm": float(raw_pde_norm.detach().cpu()),
        "effective_pde_grad_norm": float(effective_pde_norm.detach().cpu()),
        "total_grad_norm": float(total_norm.detach().cpu()),
        "raw_pde_to_data_ratio": float(
            (raw_pde_norm / (data_norm + GRAD_EPS)).detach().cpu()
        ),
        "effective_pde_to_data_ratio": float(
            (effective_pde_norm / (data_norm + GRAD_EPS)).detach().cpu()
        ),
        "data_pde_cosine": float(cosine.detach().cpu()),
        "diagnostic_data_loss": float(comp["data"].detach().cpu()),
        "diagnostic_pde_loss": float(comp["pde"].detach().cpu()),
    }

    model.zero_grad(set_to_none=True)
    return result


def append_grad_log(
    logs: Dict[str, List[float]],
    step: int,
    metrics: Dict[str, float],
) -> None:
    logs["iter"].append(step)
    for key, value in metrics.items():
        logs[key].append(float(value))


def train_one_experiment(
    name: str,
    use_pde: bool,
    shared: SharedData,
    initial_state_dict: Dict[str, torch.Tensor],
) -> Tuple[
    PODDeepONet,
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    int,
    float,
    str,
]:
    """训练单个实验并返回最佳模型。"""
    exp_dir = os.path.join(RESULT_DIR, name)
    safe_mkdir(exp_dir)
    model_path = os.path.join(exp_dir, "best_model.pth")

    # 每组重新固定随机种子，并载入完全相同的初始权重。
    set_seed(SEED)
    model = build_model(shared)
    model.load_state_dict(copy.deepcopy(initial_state_dict))

    loss_fn = build_loss(shared)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ADAM_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        WARMUP_STEPS,
        ADAM_ITERATIONS,
    )

    train_loader = make_train_loader(shared, seed=SEED)
    val_loader = make_val_loader(shared)
    train_iterator = iter(train_loader)

    # 固定诊断批次，两个实验使用完全相同样本。
    diag_n = min(GRADIENT_DIAGNOSTIC_BATCH_SIZE, len(shared.branch_val_scaled))
    branch_diag = to_device(
        torch.from_numpy(shared.branch_val_scaled[:diag_n])
    )
    y_diag = to_device(
        torch.from_numpy(shared.y_val_scaled[:diag_n])
    )

    train_logs = empty_train_logs()
    val_logs = empty_val_logs()
    grad_logs = empty_grad_logs()

    best_common = float("inf")
    best_iter = -1
    early_stop_counter = 0
    validation_count = 0

    print("\n" + "=" * 88)
    print(f"开始实验: {name}")
    print(f"PDE residual enabled: {use_pde}")
    print("=" * 88)

    pbar = tqdm(total=ADAM_ITERATIONS, desc=name)

    for step in range(ADAM_ITERATIONS):
        try:
            branch_batch, y_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            branch_batch, y_batch = next(train_iterator)

        branch_batch = to_device(branch_batch)
        y_batch = to_device(y_batch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred = model(branch_batch)
        comp = loss_fn.compute_components(pred, y_batch, branch_batch)
        total = loss_fn.total_loss(comp, use_pde=use_pde)

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"实验 {name} 在 step={step} 出现非有限损失: {float(total.detach().cpu())}"
            )

        total.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        append_train_log(train_logs, step, total, comp, current_lr)
        pbar.update(1)

        should_validate = (
            step % VALIDATION_FREQUENCY == 0
            or step == ADAM_ITERATIONS - 1
        )
        if not should_validate:
            continue

        val_metrics = evaluate_loss_components(
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            use_pde=use_pde,
        )
        append_val_log(val_logs, step, val_metrics)

        # 参数梯度诊断不在 no_grad 下执行。
        should_diagnose = (
            validation_count % GRADIENT_DIAGNOSTIC_EVERY_N_VALIDATIONS == 0
            or step == ADAM_ITERATIONS - 1
        )
        if should_diagnose:
            grad_metrics = compute_gradient_diagnostics(
                model=model,
                loss_fn=loss_fn,
                branch_diag=branch_diag,
                y_diag=y_diag,
                use_pde=use_pde,
            )
            append_grad_log(grad_logs, step, grad_metrics)

        validation_count += 1

        print(
            f"\n[{name}] Iter {step:6d} | "
            f"TrainTot={float(total.detach().cpu()):.4e} | "
            f"ValCommon={val_metrics['common']:.4e} | "
            f"ValData={val_metrics['data']:.4e} | "
            f"ValPDE={val_metrics['pde']:.4e} | "
            f"ValR1={val_metrics['pde_r1']:.4e} | "
            f"ValR2={val_metrics['pde_r2']:.4e} | "
            f"LR={current_lr:.3e}"
        )

        # 两组统一按 common objective (即 validation data loss) 保存最佳模型。
        current_common = val_metrics["common"]
        if current_common < best_common:
            best_common = current_common
            best_iter = step
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1

        if (
            USE_EARLY_STOPPING
            and step >= MIN_ITERATIONS_BEFORE_EARLY_STOPPING
            and early_stop_counter >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"\n[{name}] 触发早停：连续 {EARLY_STOPPING_PATIENCE} 次验证 "
                "common objective 未改善。"
            )
            break

    pbar.close()

    if best_iter < 0:
        # 理论上 step=0 即会验证；此处为防御性处理。
        torch.save(model.state_dict(), model_path)
        best_iter = len(train_logs["iter"]) - 1
        best_common = float("nan")

    model.load_state_dict(safe_torch_load_state_dict(model_path, DEVICE))
    model.eval()

    return (
        model,
        train_logs,
        val_logs,
        grad_logs,
        best_iter,
        best_common,
        model_path,
    )


# =============================================================================
# 9. 预测与定量评价
# =============================================================================

def predict_scaled_batched(
    model: PODDeepONet,
    branch_scaled: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(branch_scaled))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )

    outputs = []
    model.eval()
    with torch.no_grad():
        for (branch_batch,) in loader:
            branch_batch = to_device(branch_batch)
            outputs.append(model(branch_batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def inverse_output_scaler(
    y_scaled: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    return y_scaled * y_std + y_mean


def safe_relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.linalg.norm(y_true.ravel())
    if denominator < 1e-30:
        return float("nan")
    return float(np.linalg.norm((y_pred - y_true).ravel()) / denominator)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    denominator = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    if denominator < 1e-30:
        return float("nan")
    numerator = np.sum((y_true_flat - y_pred_flat) ** 2)
    return float(1.0 - numerator / denominator)


def field_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    error = y_pred - y_true
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_max_abs": float(np.max(np.abs(error))),
        f"{prefix}_rel_l2": safe_relative_l2(y_true, y_pred),
        f"{prefix}_r2": safe_r2(y_true, y_pred),
    }


def compute_numpy_residual_maps(
    y_physical: np.ndarray,
    branch_physical: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy 版本 PDE 残差，用于评价与绘图。"""
    num_samples = y_physical.shape[0]
    num_a = len(a_grid)
    num_tau = len(tau_grid)
    field_len = num_a * num_tau

    d1 = y_physical[:, :field_len].reshape(num_samples, num_tau, num_a)
    d2 = y_physical[:, field_len:].reshape(num_samples, num_tau, num_a)

    nu = branch_physical[:, 0][:, None, None]
    kappa = branch_physical[:, 1][:, None, None]
    d_diff = branch_physical[:, 2][:, None, None]

    a = a_grid[None, None, :]
    tau = tau_grid[None, :, None]
    a_safe = np.maximum(a, 1e-9)

    d1_th = nu * a - (kappa / 8.0) * (a ** 3) + d_diff / a_safe
    d2_th = np.broadcast_to(d_diff, d1_th.shape)

    u1 = tau * d1
    u2 = tau * d2

    da = float(a_grid[1] - a_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])

    u1_in = u1[:, 1:-1, 1:-1]
    d1_th_in = d1_th[:, :, 1:-1]
    d2_th_in = d2_th[:, :, 1:-1]

    du1_dtau = (u1[:, 2:, 1:-1] - u1[:, :-2, 1:-1]) / (2.0 * dtau)
    du2_dtau = (u2[:, 2:, 1:-1] - u2[:, :-2, 1:-1]) / (2.0 * dtau)

    du1_da = (u1[:, 1:-1, 2:] - u1[:, 1:-1, :-2]) / (2.0 * da)
    du2_da = (u2[:, 1:-1, 2:] - u2[:, 1:-1, :-2]) / (2.0 * da)

    d2u1_da2 = (
        u1[:, 1:-1, 2:]
        - 2.0 * u1[:, 1:-1, 1:-1]
        + u1[:, 1:-1, :-2]
    ) / (da ** 2)

    d2u2_da2 = (
        u2[:, 1:-1, 2:]
        - 2.0 * u2[:, 1:-1, 1:-1]
        + u2[:, 1:-1, :-2]
    ) / (da ** 2)

    r1 = (
        du1_dtau
        - (d1_th_in * du1_da + d2_th_in * d2u1_da2)
        - d1_th_in
    )

    r2 = (
        du2_dtau
        - (d1_th_in * du2_da + d2_th_in * d2u2_da2)
        - d1_th_in * u1_in
        - 2.0 * d2_th_in * du1_da
        - d2_th_in
    )

    return r1, r2


def compute_u_derivatives(
    y_physical: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """计算 U1=tau*D1、U2=tau*D2 的中心差分导数。"""
    num_samples = y_physical.shape[0]
    num_a = len(a_grid)
    num_tau = len(tau_grid)
    field_len = num_a * num_tau

    d1 = y_physical[:, :field_len].reshape(num_samples, num_tau, num_a)
    d2 = y_physical[:, field_len:].reshape(num_samples, num_tau, num_a)

    tau = tau_grid[None, :, None]
    u1 = tau * d1
    u2 = tau * d2

    da = float(a_grid[1] - a_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])

    return {
        "du1_dtau": (u1[:, 2:, 1:-1] - u1[:, :-2, 1:-1]) / (2.0 * dtau),
        "du1_da": (u1[:, 1:-1, 2:] - u1[:, 1:-1, :-2]) / (2.0 * da),
        "d2u1_da2": (
            u1[:, 1:-1, 2:]
            - 2.0 * u1[:, 1:-1, 1:-1]
            + u1[:, 1:-1, :-2]
        ) / (da ** 2),
        "du2_dtau": (u2[:, 2:, 1:-1] - u2[:, :-2, 1:-1]) / (2.0 * dtau),
        "du2_da": (u2[:, 1:-1, 2:] - u2[:, 1:-1, :-2]) / (2.0 * da),
        "d2u2_da2": (
            u2[:, 1:-1, 2:]
            - 2.0 * u2[:, 1:-1, 1:-1]
            + u2[:, 1:-1, :-2]
        ) / (da ** 2),
    }


def compute_physical_gradient_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Dict[str, float]:
    true_derivatives = compute_u_derivatives(y_true, a_grid, tau_grid)
    pred_derivatives = compute_u_derivatives(y_pred, a_grid, tau_grid)

    metrics: Dict[str, float] = {}
    for key in true_derivatives:
        true_value = true_derivatives[key]
        pred_value = pred_derivatives[key]
        error = pred_value - true_value
        metrics[f"grad_{key}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        metrics[f"grad_{key}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"grad_{key}_rel_l2"] = safe_relative_l2(true_value, pred_value)
        metrics[f"pred_{key}_rms"] = float(np.sqrt(np.mean(pred_value ** 2)))

    rmse_keys = [
        "grad_du1_dtau_rmse",
        "grad_du1_da_rmse",
        "grad_d2u1_da2_rmse",
        "grad_du2_dtau_rmse",
        "grad_du2_da_rmse",
        "grad_d2u2_da2_rmse",
    ]
    metrics["mean_physical_gradient_rmse"] = float(
        np.mean([metrics[k] for k in rmse_keys])
    )
    return metrics


def compute_sample_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    branch_physical: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> pd.DataFrame:
    num_a = len(a_grid)
    num_tau = len(tau_grid)
    field_len = num_a * num_tau

    r1, r2 = compute_numpy_residual_maps(
        y_pred,
        branch_physical,
        a_grid,
        tau_grid,
    )

    records = []
    for i in range(len(y_true)):
        d1_true = y_true[i, :field_len]
        d1_pred = y_pred[i, :field_len]
        d2_true = y_true[i, field_len:]
        d2_pred = y_pred[i, field_len:]

        records.append(
            {
                "sample_index": i,
                "nu": branch_physical[i, 0],
                "kappa": branch_physical[i, 1],
                "d_diffusion": branch_physical[i, 2],
                "d1_rmse": np.sqrt(np.mean((d1_pred - d1_true) ** 2)),
                "d1_mae": np.mean(np.abs(d1_pred - d1_true)),
                "d1_rel_l2": safe_relative_l2(d1_true, d1_pred),
                "d2_rmse": np.sqrt(np.mean((d2_pred - d2_true) ** 2)),
                "d2_mae": np.mean(np.abs(d2_pred - d2_true)),
                "d2_rel_l2": safe_relative_l2(d2_true, d2_pred),
                "pde_r1_mse": np.mean(r1[i] ** 2),
                "pde_r2_mse": np.mean(r2[i] ** 2),
                "pde_total_mse": np.mean(r1[i] ** 2) + np.mean(r2[i] ** 2),
                "d2_min_prediction": np.min(d2_pred),
            }
        )

    return pd.DataFrame(records)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    branch_physical: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    num_a = len(a_grid)
    num_tau = len(tau_grid)
    field_len = num_a * num_tau

    d1_true = y_true[:, :field_len]
    d1_pred = y_pred[:, :field_len]
    d2_true = y_true[:, field_len:]
    d2_pred = y_pred[:, field_len:]

    metrics: Dict[str, float] = {}
    metrics.update(field_error_metrics(d1_true, d1_pred, "d1"))
    metrics.update(field_error_metrics(d2_true, d2_pred, "d2"))

    r1, r2 = compute_numpy_residual_maps(
        y_pred,
        branch_physical,
        a_grid,
        tau_grid,
    )
    metrics["pde_r1_mse"] = float(np.mean(r1 ** 2))
    metrics["pde_r2_mse"] = float(np.mean(r2 ** 2))
    metrics["pde_total_mse"] = metrics["pde_r1_mse"] + metrics["pde_r2_mse"]
    metrics["pde_r1_rmse"] = float(np.sqrt(metrics["pde_r1_mse"]))
    metrics["pde_r2_rmse"] = float(np.sqrt(metrics["pde_r2_mse"]))

    metrics["d2_min_prediction"] = float(np.min(d2_pred))

    metrics.update(
        compute_physical_gradient_metrics(
            y_true=y_true,
            y_pred=y_pred,
            a_grid=a_grid,
            tau_grid=tau_grid,
        )
    )

    sample_df = compute_sample_metrics(
        y_true=y_true,
        y_pred=y_pred,
        branch_physical=branch_physical,
        a_grid=a_grid,
        tau_grid=tau_grid,
    )
    return metrics, sample_df


def finalize_experiment(
    name: str,
    use_pde: bool,
    model: PODDeepONet,
    shared: SharedData,
    train_logs: Dict[str, List[float]],
    val_logs: Dict[str, List[float]],
    grad_logs: Dict[str, List[float]],
    best_iter: int,
    best_common: float,
    model_path: str,
) -> ExperimentResult:
    exp_dir = os.path.join(RESULT_DIR, name)

    train_log_path = os.path.join(exp_dir, "train_logs.csv")
    val_log_path = os.path.join(exp_dir, "validation_logs.csv")
    grad_log_path = os.path.join(exp_dir, "gradient_diagnostics.csv")
    metrics_path = os.path.join(exp_dir, "evaluation_metrics.json")
    predictions_path = os.path.join(exp_dir, "validation_predictions.npz")
    sample_metrics_path = os.path.join(exp_dir, "sample_metrics.csv")

    pd.DataFrame(train_logs).to_csv(
        train_log_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(val_logs).to_csv(
        val_log_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(grad_logs).to_csv(
        grad_log_path,
        index=False,
        encoding="utf-8-sig",
    )

    pred_scaled = predict_scaled_batched(
        model,
        shared.branch_val_scaled,
        EVAL_BATCH_SIZE,
    )
    pred_physical = inverse_output_scaler(
        pred_scaled,
        shared.y_mean_scaler,
        shared.y_std_scaler,
    )

    metrics, sample_df = evaluate_predictions(
        y_true=shared.y_val_physical,
        y_pred=pred_physical,
        branch_physical=shared.branch_val_physical,
        a_grid=shared.unified_a_grid,
        tau_grid=shared.unified_tau_grid,
    )

    metrics["best_iter"] = int(best_iter)
    metrics["best_common_objective"] = float(best_common)
    metrics["use_pde"] = bool(use_pde)
    metrics["effective_pde_coefficient"] = (
        PHYSICS_LOSS_WEIGHT if use_pde else 0.0
    )

    if grad_logs["effective_pde_grad_norm"]:
        metrics["mean_effective_pde_grad_norm"] = float(
            np.mean(grad_logs["effective_pde_grad_norm"])
        )
        metrics["max_effective_pde_grad_norm"] = float(
            np.max(grad_logs["effective_pde_grad_norm"])
        )
        metrics["mean_raw_pde_grad_norm"] = float(
            np.mean(grad_logs["raw_pde_grad_norm"])
        )
        metrics["mean_data_pde_cosine"] = float(
            np.mean(grad_logs["data_pde_cosine"])
        )
    else:
        metrics["mean_effective_pde_grad_norm"] = float("nan")
        metrics["max_effective_pde_grad_norm"] = float("nan")
        metrics["mean_raw_pde_grad_norm"] = float("nan")
        metrics["mean_data_pde_cosine"] = float("nan")

    save_json(metrics_path, metrics)
    sample_df.to_csv(sample_metrics_path, index=False, encoding="utf-8-sig")

    if SAVE_FULL_VALIDATION_PREDICTIONS:
        np.savez_compressed(
            predictions_path,
            branch_val_physical=shared.branch_val_physical,
            branch_val_scaled=shared.branch_val_scaled,
            y_true_physical=shared.y_val_physical,
            y_pred_physical=pred_physical,
            y_pred_scaled=pred_scaled,
            a_grid=shared.unified_a_grid,
            tau_grid=shared.unified_tau_grid,
        )
    else:
        # 即使不保存完整预测，也保存必要网格与指标占位。
        np.savez_compressed(
            predictions_path,
            branch_val_physical=shared.branch_val_physical,
            a_grid=shared.unified_a_grid,
            tau_grid=shared.unified_tau_grid,
        )

    return ExperimentResult(
        name=name,
        use_pde=use_pde,
        model_path=model_path,
        train_log_path=train_log_path,
        val_log_path=val_log_path,
        grad_log_path=grad_log_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        sample_metrics_path=sample_metrics_path,
        best_iter=best_iter,
        best_common_objective=best_common,
        metrics=metrics,
        train_logs=train_logs,
        val_logs=val_logs,
        grad_logs=grad_logs,
        predictions=pred_physical,
        sample_metrics=sample_df,
    )


# =============================================================================
# 10. 可视化
# =============================================================================

def plot_loss_ablation(
    results: Dict[str, ExperimentResult],
    save_path: str,
) -> None:
    labels = {
        "no_pde_residual": "Without PDE residual",
        "with_pde_residual": "With PDE residual",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for name, result in results.items():
        train = result.train_logs
        val = result.val_logs
        label = labels[name]

        axes[0, 0].plot(
            train["iter"],
            positive_for_log(moving_average(train["train_total"], 200)),
            linewidth=1.5,
            label=label,
        )
        axes[0, 1].plot(
            val["iter"],
            positive_for_log(val["val_common"]),
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=label,
        )
        axes[1, 0].plot(
            val["iter"],
            positive_for_log(val["val_data"]),
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=label,
        )
        axes[1, 1].plot(
            val["iter"],
            positive_for_log(val["val_pde"]),
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=label,
        )

    axes[0, 0].set_title("Training objective (moving average)")
    axes[0, 1].set_title("Validation common objective")
    axes[1, 0].set_title("Validation data loss")
    axes[1, 1].set_title("Validation raw PDE residual loss")

    for ax in axes.ravel():
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.axvline(WARMUP_STEPS, linestyle=":", linewidth=1.2, alpha=0.7)

    fig.suptitle("Physics-residual ablation: loss comparison", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_gradient_comparison(
    results: Dict[str, ExperimentResult],
    save_path: str,
) -> None:
    labels = {
        "no_pde_residual": "Without PDE residual",
        "with_pde_residual": "With PDE residual",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for name, result in results.items():
        grad = result.grad_logs
        label = labels[name]
        if not grad["iter"]:
            continue

        axes[0, 0].plot(
            grad["iter"],
            positive_for_log(grad["raw_pde_grad_norm"]),
            marker="o",
            markersize=4,
            label=label,
        )
        axes[0, 1].plot(
            grad["iter"],
            positive_for_log(grad["effective_pde_grad_norm"]),
            marker="o",
            markersize=4,
            label=label,
        )
        axes[1, 0].plot(
            grad["iter"],
            positive_for_log(grad["effective_pde_to_data_ratio"]),
            marker="o",
            markersize=4,
            label=label,
        )
        axes[1, 1].plot(
            grad["iter"],
            grad["data_pde_cosine"],
            marker="o",
            markersize=4,
            label=label,
        )

    axes[0, 0].set_title("Raw PDE gradient norm")
    axes[0, 1].set_title("Effective weighted PDE gradient norm")
    axes[1, 0].set_title("Effective PDE / data gradient norm")
    axes[1, 1].set_title("Cosine(data gradient, PDE gradient)")

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].axhline(0.0, linestyle=":", linewidth=1.2)

    for ax in axes.ravel():
        ax.set_xlabel("Iteration")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[0, 0].set_ylabel("Gradient norm")
    axes[0, 1].set_ylabel("Gradient norm")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 1].set_ylabel("Cosine similarity")

    fig.suptitle("Parameter-gradient diagnostics", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_physical_field_gradient_comparison(
    results: Dict[str, ExperimentResult],
    save_path: str,
) -> None:
    metric_keys = [
        "grad_du1_dtau_rmse",
        "grad_du1_da_rmse",
        "grad_d2u1_da2_rmse",
        "grad_du2_dtau_rmse",
        "grad_du2_da_rmse",
        "grad_d2u2_da2_rmse",
    ]
    tick_labels = [
        "dU1/dtau",
        "dU1/dA",
        "d2U1/dA2",
        "dU2/dtau",
        "dU2/dA",
        "d2U2/dA2",
    ]

    base = results["no_pde_residual"]
    phys = results["with_pde_residual"]
    x = np.arange(len(metric_keys))
    width = 0.36

    base_values = [base.metrics[k] for k in metric_keys]
    phys_values = [phys.metrics[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width / 2, base_values, width, label="Without PDE residual")
    ax.bar(x + width / 2, phys_values, width, label="With PDE residual")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE of physical finite-difference derivative")
    ax.set_title("Physical-field gradient error comparison")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_ablation(
    results: Dict[str, ExperimentResult],
    save_path: str,
) -> None:
    base = results["no_pde_residual"]
    phys = results["with_pde_residual"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    width = 0.36

    # 场误差
    field_keys = ["d1_rmse", "d2_rmse", "d1_rel_l2", "d2_rel_l2"]
    field_labels = ["D1 RMSE", "D2 RMSE", "D1 Rel-L2", "D2 Rel-L2"]
    x = np.arange(len(field_keys))
    axes[0].bar(
        x - width / 2,
        [base.metrics[k] for k in field_keys],
        width,
        label="Without PDE",
    )
    axes[0].bar(
        x + width / 2,
        [phys.metrics[k] for k in field_keys],
        width,
        label="With PDE",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(field_labels, rotation=25, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_title("Prediction errors")
    axes[0].grid(True, axis="y", which="both", alpha=0.3)
    axes[0].legend()

    # PDE 残差
    pde_keys = ["pde_r1_mse", "pde_r2_mse", "pde_total_mse"]
    pde_labels = ["R1 MSE", "R2 MSE", "R1+R2 MSE"]
    x2 = np.arange(len(pde_keys))
    axes[1].bar(
        x2 - width / 2,
        [base.metrics[k] for k in pde_keys],
        width,
        label="Without PDE",
    )
    axes[1].bar(
        x2 + width / 2,
        [phys.metrics[k] for k in pde_keys],
        width,
        label="With PDE",
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pde_labels, rotation=20, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_title("Physical residual errors")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Final validation metrics", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_field_ablation(
    field_name: str,
    field_index: int,
    results: Dict[str, ExperimentResult],
    shared: SharedData,
    sample_indices: np.ndarray,
    save_path: str,
) -> None:
    """绘制 True / No-PDE / With-PDE / 两组误差。"""
    num_a = len(shared.unified_a_grid)
    num_tau = len(shared.unified_tau_grid)
    field_len = num_a * num_tau

    start = 0 if field_index == 0 else field_len
    end = field_len if field_index == 0 else 2 * field_len

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    extent = [
        shared.unified_a_grid.min(),
        shared.unified_a_grid.max(),
        shared.unified_tau_grid.min(),
        shared.unified_tau_grid.max(),
    ]

    base_pred_all = results["no_pde_residual"].predictions
    phys_pred_all = results["with_pde_residual"].predictions

    for row, idx in enumerate(sample_indices):
        true_field = shared.y_val_physical[idx, start:end].reshape(num_tau, num_a)
        base_field = base_pred_all[idx, start:end].reshape(num_tau, num_a)
        phys_field = phys_pred_all[idx, start:end].reshape(num_tau, num_a)

        base_err = np.abs(base_field - true_field)
        phys_err = np.abs(phys_field - true_field)

        vmin = min(true_field.min(), base_field.min(), phys_field.min())
        vmax = max(true_field.max(), base_field.max(), phys_field.max())
        err_max = max(base_err.max(), phys_err.max(), 1e-30)

        fields = [true_field, base_field, phys_field, base_err, phys_err]
        titles = [
            f"{field_name} true",
            f"{field_name} pred: no PDE",
            f"{field_name} pred: with PDE",
            f"|error|: no PDE",
            f"|error|: with PDE",
        ]

        for col, (data, title) in enumerate(zip(fields, titles)):
            is_error = col >= 3
            image = axes[row, col].imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="magma" if is_error else "viridis",
                vmin=0.0 if is_error else vmin,
                vmax=err_max if is_error else vmax,
            )
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("A")
            axes[row, col].set_ylabel("tau")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

        params = shared.branch_val_physical[idx]
        base_rmse = np.sqrt(np.mean((base_field - true_field) ** 2))
        phys_rmse = np.sqrt(np.mean((phys_field - true_field) ** 2))
        axes[row, 0].text(
            0.0,
            1.11,
            (
                f"sample={idx}, nu={params[0]:.4g}, kappa={params[1]:.4g}, "
                f"d={params[2]:.4g}; RMSE(no PDE)={base_rmse:.3e}, "
                f"RMSE(with PDE)={phys_rmse:.3e}"
            ),
            transform=axes[row, 0].transAxes,
            fontsize=10,
        )

    fig.suptitle(f"{field_name} prediction ablation", fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pde_residual_maps(
    results: Dict[str, ExperimentResult],
    shared: SharedData,
    sample_indices: np.ndarray,
    save_path: str,
) -> None:
    base_pred = results["no_pde_residual"].predictions
    phys_pred = results["with_pde_residual"].predictions

    base_r1, base_r2 = compute_numpy_residual_maps(
        base_pred,
        shared.branch_val_physical,
        shared.unified_a_grid,
        shared.unified_tau_grid,
    )
    phys_r1, phys_r2 = compute_numpy_residual_maps(
        phys_pred,
        shared.branch_val_physical,
        shared.unified_a_grid,
        shared.unified_tau_grid,
    )

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 4, figsize=(21, 5.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    extent = [
        shared.unified_a_grid[1],
        shared.unified_a_grid[-2],
        shared.unified_tau_grid[1],
        shared.unified_tau_grid[-2],
    ]

    for row, idx in enumerate(sample_indices):
        maps = [
            np.abs(base_r1[idx]),
            np.abs(phys_r1[idx]),
            np.abs(base_r2[idx]),
            np.abs(phys_r2[idx]),
        ]
        titles = [
            "|R1|: no PDE",
            "|R1|: with PDE",
            "|R2|: no PDE",
            "|R2|: with PDE",
        ]

        r1_max = max(maps[0].max(), maps[1].max(), 1e-30)
        r2_max = max(maps[2].max(), maps[3].max(), 1e-30)

        for col, (data, title) in enumerate(zip(maps, titles)):
            vmax = r1_max if col < 2 else r2_max
            image = axes[row, col].imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("A")
            axes[row, col].set_ylabel("tau")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

        axes[row, 0].text(
            0.0,
            1.11,
            (
                f"sample={idx}; R1-MSE(no/with)="
                f"{np.mean(base_r1[idx] ** 2):.3e}/"
                f"{np.mean(phys_r1[idx] ** 2):.3e}; "
                f"R2-MSE(no/with)="
                f"{np.mean(base_r2[idx] ** 2):.3e}/"
                f"{np.mean(phys_r2[idx] ** 2):.3e}"
            ),
            transform=axes[row, 0].transAxes,
            fontsize=10,
        )

    fig.suptitle("PDE residual-map ablation", fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 11. 汇总、改善率与自动报告
# =============================================================================

def percent_improvement(baseline: float, physics: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-30:
        return float("nan")
    return float((baseline - physics) / abs(baseline) * 100.0)


def build_ablation_summary(
    results: Dict[str, ExperimentResult],
) -> pd.DataFrame:
    base = results["no_pde_residual"]
    phys = results["with_pde_residual"]

    key_metrics = [
        "d1_rmse",
        "d2_rmse",
        "d1_rel_l2",
        "d2_rel_l2",
        "pde_r1_mse",
        "pde_r2_mse",
        "pde_total_mse",
        "mean_physical_gradient_rmse",
        "grad_du1_dtau_rmse",
        "grad_du1_da_rmse",
        "grad_d2u1_da2_rmse",
        "grad_du2_dtau_rmse",
        "grad_du2_da_rmse",
        "grad_d2u2_da2_rmse",
        "mean_effective_pde_grad_norm",
        "mean_raw_pde_grad_norm",
        "mean_data_pde_cosine",
        "best_common_objective",
    ]

    rows = []
    for key in key_metrics:
        base_value = base.metrics.get(key, float("nan"))
        phys_value = phys.metrics.get(key, float("nan"))
        rows.append(
            {
                "metric": key,
                "without_pde": base_value,
                "with_pde": phys_value,
                "improvement_percent_positive_is_better": percent_improvement(
                    base_value,
                    phys_value,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_physical_gradient_table(
    results: Dict[str, ExperimentResult],
) -> pd.DataFrame:
    keys = [
        "grad_du1_dtau_rmse",
        "grad_du1_da_rmse",
        "grad_d2u1_da2_rmse",
        "grad_du2_dtau_rmse",
        "grad_du2_da_rmse",
        "grad_d2u2_da2_rmse",
        "mean_physical_gradient_rmse",
    ]
    rows = []
    for name, result in results.items():
        row = {"experiment": name}
        for key in keys:
            row[key] = result.metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_report(
    results: Dict[str, ExperimentResult],
    retained_energy: float,
) -> str:
    base = results["no_pde_residual"]
    phys = results["with_pde_residual"]

    pde_improvement = percent_improvement(
        base.metrics["pde_total_mse"],
        phys.metrics["pde_total_mse"],
    )
    gradient_improvement = percent_improvement(
        base.metrics["mean_physical_gradient_rmse"],
        phys.metrics["mean_physical_gradient_rmse"],
    )
    d1_improvement = percent_improvement(
        base.metrics["d1_rmse"],
        phys.metrics["d1_rmse"],
    )
    d2_improvement = percent_improvement(
        base.metrics["d2_rmse"],
        phys.metrics["d2_rmse"],
    )

    effective_grad = phys.metrics.get("mean_effective_pde_grad_norm", float("nan"))
    has_active_physics_gradient = np.isfinite(effective_grad) and effective_grad > 0.0

    supports_effect = (
        np.isfinite(pde_improvement)
        and pde_improvement > 0.0
        and np.isfinite(gradient_improvement)
        and gradient_improvement > 0.0
        and has_active_physics_gradient
    )

    if supports_effect:
        conclusion = (
            "结论：结果支持 PDE 物理残差在训练中发挥了实际作用。"
            "依据是：with_pde_residual 组存在非零的有效 PDE 参数梯度，"
            "且最终 PDE 残差和物理场有限差分导数误差均低于无 PDE 组。"
        )
    else:
        conclusion = (
            "结论：当前运行结果尚不足以确认 PDE 物理残差带来稳定改善。"
            "请结合损失权重、POD 截断误差、训练步数及梯度冲突指标进一步分析，"
            "不要仅凭是否加入物理项作先验结论。"
        )

    report = f"""POD-DeepONet 物理残差消融实验报告
{'=' * 64}

1. 实验公平性
- 两组使用相同数据文件、训练/验证划分、标准化参数和 POD 基底。
- 两组从完全相同的网络初始权重开始。
- 两组使用相同 AdamW、batch size、学习率计划和随机种子。
- 唯一核心差别：with_pde_residual 训练总损失中包含 PDE 残差项。
- 两组最佳模型均按 validation data loss 选择。

2. POD 信息
- 实际 POD 模态数: {base.metrics.get('actual_num_modes', '见 shared_preprocessing.pth')}
- POD 累积保留能量: {retained_energy:.8f}

3. 验证集预测误差
- D1 RMSE: no PDE={base.metrics['d1_rmse']:.8e}, with PDE={phys.metrics['d1_rmse']:.8e}, 改善率={d1_improvement:.3f}%
- D2 RMSE: no PDE={base.metrics['d2_rmse']:.8e}, with PDE={phys.metrics['d2_rmse']:.8e}, 改善率={d2_improvement:.3f}%

4. 物理一致性
- PDE total MSE: no PDE={base.metrics['pde_total_mse']:.8e}, with PDE={phys.metrics['pde_total_mse']:.8e}, 改善率={pde_improvement:.3f}%
- Mean physical-gradient RMSE: no PDE={base.metrics['mean_physical_gradient_rmse']:.8e}, with PDE={phys.metrics['mean_physical_gradient_rmse']:.8e}, 改善率={gradient_improvement:.3f}%

5. 物理梯度是否真正参与优化
- no PDE 组平均有效 PDE 梯度范数: {base.metrics['mean_effective_pde_grad_norm']:.8e}
- with PDE 组平均有效 PDE 梯度范数: {phys.metrics['mean_effective_pde_grad_norm']:.8e}
- with PDE 组平均原始 PDE 梯度范数: {phys.metrics['mean_raw_pde_grad_norm']:.8e}
- with PDE 组数据梯度与 PDE 梯度平均余弦: {phys.metrics['mean_data_pde_cosine']:.8e}

7. 自动判定
{conclusion}

说明：
- “改善率”为 (无 PDE 指标 - 有 PDE 指标) / |无 PDE 指标| * 100%。
- 对误差与残差类指标，正值表示加入 PDE 后下降。
- 若数据误差略升但 PDE 残差显著下降，说明存在数据拟合与物理一致性的权衡；
  应结合 data_pde_cosine 和 effective_pde_to_data_ratio 调整物理权重。
"""
    return report


# =============================================================================
# 12. 主程序
# =============================================================================

def main() -> None:
    print("=" * 96)
    print("POD-DeepONet Physics Residual Ablation")
    print("=" * 96)
    print(f"Device     : {DEVICE}")
    print(f"Dtype      : {DTYPE}")
    print(f"Result dir : {RESULT_DIR}")
    print(f"Data dir   : {DATA_DIR}")

    set_seed(SEED)

    # -------------------------------------------------------------------------
    # Step 1: 加载数据
    # -------------------------------------------------------------------------
    (
        branch_inputs_np,
        y_snapshots_np,
        unified_a_grid,
        unified_tau_grid,
        valid_files,
        failed_records,
    ) = load_unified_data(DATA_DIR, TARGET_NUM_FILES, SEED)

    if failed_records:
        pd.DataFrame(failed_records).to_csv(
            os.path.join(RESULT_DIR, "failed_files.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Step 2: 统一划分，保存文件清单
    # -------------------------------------------------------------------------
    all_indices = np.arange(len(branch_inputs_np))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        shuffle=True,
    )

    branch_train_np = branch_inputs_np[train_idx]
    branch_val_np = branch_inputs_np[val_idx]
    y_train_np = y_snapshots_np[train_idx]
    y_val_np = y_snapshots_np[val_idx]

    manifest_records = []
    train_set = set(train_idx.tolist())
    for i, file_path in enumerate(valid_files):
        manifest_records.append(
            {
                "sample_index_before_split": i,
                "file": file_path,
                "split": "train" if i in train_set else "validation",
                "nu": branch_inputs_np[i, 0],
                "kappa": branch_inputs_np[i, 1],
                "d_diffusion": branch_inputs_np[i, 2],
            }
        )
    pd.DataFrame(manifest_records).to_csv(
        os.path.join(RESULT_DIR, "data_split_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n数据划分完成：")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples  : {len(val_idx)}")

    # -------------------------------------------------------------------------
    # Step 3: 标准化，仅用训练集统计量
    # -------------------------------------------------------------------------
    branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
    branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)

    y_train_scaled, y_mean_scaler, y_std_scaler = manual_scaler(y_train_np)
    y_val_scaled = manual_scaler(y_val_np, y_mean_scaler, y_std_scaler)

    # -------------------------------------------------------------------------
    # Step 4: POD，仅用训练集
    # -------------------------------------------------------------------------
    (
        y_mean_pod_scaled,
        pod_basis,
        singular_values,
        actual_num_modes,
    ) = pod(y_train_scaled, REQUESTED_NUM_POD_MODES)

    singular_energy = singular_values ** 2
    total_energy = np.sum(singular_energy)
    if total_energy <= 0.0:
        raise ValueError("POD 奇异值能量为零，无法构造有效 POD 基底。")
    cumulative_energy = np.cumsum(singular_energy / total_energy)
    retained_energy = float(cumulative_energy[actual_num_modes - 1])

    print("\nPOD 分解完成：")
    print(f"Requested POD modes: {REQUESTED_NUM_POD_MODES}")
    print(f"Actual POD modes   : {actual_num_modes}")
    print(f"Retained energy    : {retained_energy:.8f}")

    shared = SharedData(
        branch_train_scaled=branch_train_scaled,
        branch_val_scaled=branch_val_scaled,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        branch_train_physical=branch_train_np,
        branch_val_physical=branch_val_np,
        y_train_physical=y_train_np,
        y_val_physical=y_val_np,
        branch_mean=branch_mean,
        branch_std=branch_std,
        y_mean_scaler=y_mean_scaler,
        y_std_scaler=y_std_scaler,
        y_mean_pod_scaled=y_mean_pod_scaled,
        pod_basis=pod_basis,
        singular_values=singular_values,
        actual_num_modes=actual_num_modes,
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
    )

    # -------------------------------------------------------------------------
    # Step 5: 保存共享预处理信息与配置
    # -------------------------------------------------------------------------
    shared_payload = {
        "branch_mean": branch_mean,
        "branch_std": branch_std,
        "y_mean_scaler": y_mean_scaler,
        "y_std_scaler": y_std_scaler,
        "y_mean_pod_scaled": y_mean_pod_scaled,
        "pod_basis": pod_basis,
        "singular_values": singular_values,
        "actual_num_modes": actual_num_modes,
        "retained_energy": retained_energy,
        "unified_a_grid": unified_a_grid,
        "unified_tau_grid": unified_tau_grid,
        "train_indices": train_idx,
        "validation_indices": val_idx,
    }
    torch.save(
        shared_payload,
        os.path.join(RESULT_DIR, "shared_preprocessing.pth"),
    )

    config = {
        "RUN_ID": RUN_ID,
        "RESULT_DIR": RESULT_DIR,
        "DATA_DIR": DATA_DIR,
        "BRANCH_INPUT_DIM": BRANCH_INPUT_DIM,
        "HIDDEN_UNITS": HIDDEN_UNITS,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "REQUESTED_NUM_POD_MODES": REQUESTED_NUM_POD_MODES,
        "ACTUAL_NUM_POD_MODES": actual_num_modes,
        "DROPOUT_RATE": DROPOUT_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "ADAM_LR": ADAM_LR,
        "ADAM_BATCH_SIZE": ADAM_BATCH_SIZE,
        "EVAL_BATCH_SIZE": EVAL_BATCH_SIZE,
        "ADAM_ITERATIONS": ADAM_ITERATIONS,
        "VALIDATION_SPLIT": VALIDATION_SPLIT,
        "VALIDATION_FREQUENCY": VALIDATION_FREQUENCY,
        "WARMUP_STEPS": WARMUP_STEPS,
        "SEED": SEED,
        "TARGET_NUM_FILES": TARGET_NUM_FILES,
        "USE_EARLY_STOPPING": USE_EARLY_STOPPING,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "MIN_ITERATIONS_BEFORE_EARLY_STOPPING": MIN_ITERATIONS_BEFORE_EARLY_STOPPING,
        "D2_LOSS_WEIGHT": D2_LOSS_WEIGHT,
        "PHYSICS_LOSS_WEIGHT": PHYSICS_LOSS_WEIGHT,
        "EFFECTIVE_PDE_COEFFICIENT": PHYSICS_LOSS_WEIGHT,
        "GRADIENT_DIAGNOSTIC_EVERY_N_VALIDATIONS": GRADIENT_DIAGNOSTIC_EVERY_N_VALIDATIONS,
        "GRADIENT_DIAGNOSTIC_BATCH_SIZE": GRADIENT_DIAGNOSTIC_BATCH_SIZE,
        "DEVICE": str(DEVICE),
        "DTYPE": str(DTYPE),
        "NUM_TRAIN_SAMPLES": len(train_idx),
        "NUM_VALIDATION_SAMPLES": len(val_idx),
        "RETAINED_POD_ENERGY": retained_energy,
    }
    save_json(os.path.join(RESULT_DIR, "config.json"), config)

    # -------------------------------------------------------------------------
    # Step 6: 构造一次初始模型，两个实验载入同一初始 state_dict
    # -------------------------------------------------------------------------
    set_seed(SEED)
    initial_model = build_model(shared)
    initial_state_dict = copy.deepcopy(initial_model.state_dict())
    torch.save(
        initial_state_dict,
        os.path.join(RESULT_DIR, "shared_initial_state_dict.pth"),
    )
    del initial_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 7: 依次训练两个实验
    # -------------------------------------------------------------------------
    results: Dict[str, ExperimentResult] = {}

    for name, use_pde in EXPERIMENTS.items():
        (
            model,
            train_logs,
            val_logs,
            grad_logs,
            best_iter,
            best_common,
            model_path,
        ) = train_one_experiment(
            name=name,
            use_pde=use_pde,
            shared=shared,
            initial_state_dict=initial_state_dict,
        )

        result = finalize_experiment(
            name=name,
            use_pde=use_pde,
            model=model,
            shared=shared,
            train_logs=train_logs,
            val_logs=val_logs,
            grad_logs=grad_logs,
            best_iter=best_iter,
            best_common=best_common,
            model_path=model_path,
        )
        result.metrics["actual_num_modes"] = actual_num_modes
        save_json(result.metrics_path, result.metrics)
        results[name] = result

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 8: 汇总数据
    # -------------------------------------------------------------------------
    summary_df = build_ablation_summary(results)
    summary_path = os.path.join(RESULT_DIR, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    physical_gradient_df = build_physical_gradient_table(results)
    physical_gradient_path = os.path.join(
        RESULT_DIR,
        "physical_gradient_metrics.csv",
    )
    physical_gradient_df.to_csv(
        physical_gradient_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 将两组逐样本指标横向合并，便于配对统计和散点分析。
    paired_sample_metrics = results["no_pde_residual"].sample_metrics.merge(
        results["with_pde_residual"].sample_metrics,
        on=["sample_index", "nu", "kappa", "d_diffusion"],
        suffixes=("_without_pde", "_with_pde"),
        how="inner",
    )
    paired_sample_metrics.to_csv(
        os.path.join(RESULT_DIR, "paired_sample_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Step 9: 统一选择可视化样本
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(PLOT_RANDOM_SEED)
    num_plot = min(NUM_PLOT_SAMPLES, len(shared.branch_val_scaled))
    sample_indices = np.sort(
        rng.choice(len(shared.branch_val_scaled), size=num_plot, replace=False)
    )
    pd.DataFrame({"validation_sample_index": sample_indices}).to_csv(
        os.path.join(RESULT_DIR, "selected_plot_indices.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Step 10: 绘图
    # -------------------------------------------------------------------------
    print("\n开始绘制消融对比图...")

    plot_loss_ablation(
        results,
        os.path.join(RESULT_DIR, "loss_ablation_comparison.png"),
    )
    plot_parameter_gradient_comparison(
        results,
        os.path.join(RESULT_DIR, "parameter_gradient_comparison.png"),
    )
    plot_physical_field_gradient_comparison(
        results,
        os.path.join(RESULT_DIR, "physical_field_gradient_comparison.png"),
    )
    plot_metric_ablation(
        results,
        os.path.join(RESULT_DIR, "metric_ablation_comparison.png"),
    )
    plot_prediction_field_ablation(
        field_name="D1",
        field_index=0,
        results=results,
        shared=shared,
        sample_indices=sample_indices,
        save_path=os.path.join(RESULT_DIR, "prediction_D1_ablation.png"),
    )
    plot_prediction_field_ablation(
        field_name="D2",
        field_index=1,
        results=results,
        shared=shared,
        sample_indices=sample_indices,
        save_path=os.path.join(RESULT_DIR, "prediction_D2_ablation.png"),
    )
    plot_pde_residual_maps(
        results=results,
        shared=shared,
        sample_indices=sample_indices,
        save_path=os.path.join(RESULT_DIR, "pde_residual_map_ablation.png"),
    )

    # -------------------------------------------------------------------------
    # Step 11: 自动生成结论报告
    # -------------------------------------------------------------------------
    report = build_ablation_report(results, retained_energy)
    report_path = os.path.join(RESULT_DIR, "ablation_report.txt")
    save_text(report_path, report)

    print("\n" + report)
    print("\n实验完成，主要输出：")
    print(f"汇总指标       : {summary_path}")
    print(f"物理梯度指标   : {physical_gradient_path}")
    print(f"消融报告       : {report_path}")
    print(f"损失对比图     : {os.path.join(RESULT_DIR, 'loss_ablation_comparison.png')}")
    print(f"参数梯度图     : {os.path.join(RESULT_DIR, 'parameter_gradient_comparison.png')}")
    print(f"场物理梯度图   : {os.path.join(RESULT_DIR, 'physical_field_gradient_comparison.png')}")
    print(f"PDE 残差图     : {os.path.join(RESULT_DIR, 'pde_residual_map_ablation.png')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n程序运行失败，异常信息如下：")
        traceback.print_exc()
        raise