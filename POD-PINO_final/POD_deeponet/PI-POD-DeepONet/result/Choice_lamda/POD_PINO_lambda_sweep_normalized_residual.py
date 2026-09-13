# -*- coding: utf-8 -*-
"""
POD-DeepONet + Physics-Informed Loss: lambda sweep
===================================================

本程序用于系统分析归一化 PDE 残差权重 lambda 对模型精度与物理一致性的影响。

核心目标
--------
1. 所有 lambda 共用完全相同的数据、训练/验证划分、标准化参数和 POD 基底；
2. 所有 lambda 使用相同的随机种子、初始网络参数和 mini-batch 顺序；
3. 对每个 lambda 独立训练，并保存：
   - 最佳模型参数；
   - 最终模型及优化器/调度器状态；
   - 可直接推理的完整模型包；
   - 训练与验证日志；
   - 验证集逐样本误差；
   - 验证集预测结果、误差和 PDE 残差图；
   - 单组 lambda 的训练曲线、物理损失曲线和预测对比图；
4. 汇总不同 lambda 的预测精度、物理残差和非负性违约指标；
5. 自动选取最优 lambda，并保存最优模型。

lambda 的定义
-------------
    L_total = L_data + FIXED_POSITIVITY_WEIGHT * L_pos + lambda * L_pde_norm

其中，lambda 仅控制归一化 PDE 残差，不控制 D2 非负约束；
D2 非负约束权重在所有 lambda 实验中保持固定。

最优 lambda 选择标准
-------------------
不同 lambda 的 L_total 不可直接横向比较，因为损失函数本身随 lambda 改变。

因此，本程序采用“预测精度约束下的物理一致性最优”原则：

1. 用验证集物理量空间中的加权 NRMSE 衡量预测精度：

       E = (1-w_D2) * NRMSE_D1 + w_D2 * NRMSE_D2

2. 找到平均 E 最小的 lambda，并计算该模型逐样本 E 的标准误 SE；
3. 将满足 E <= E_min + SE 的 lambda 视为预测精度统计上接近；
4. 在这些候选 lambda 中，选择无量纲 PDE 残差和 D2 非负违约综合指标最小者。

这种 one-standard-error 规则可以：
- 避免因为只看数据误差而机械地选择 lambda=0；
- 避免因为 lambda 过大而明显牺牲预测精度；
- 在精度相近的模型中选择物理一致性更好的结果。
"""

import os
import glob
import math
import json
import time
import random
import shutil
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
# 1. 全局配置
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 路径配置
# -----------------------------------------------------------------------------

BASE_RESULT_DIR = (
    r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\Choice_lamda\train_result_lambda_sweep_2_0.5'
)

DATA_DIR = (
    r'D:\PINN\zenodo\POD_deeponet'
    r'\PI-POD-DeepONet\Data_AdjointFP_FixedA'
)

RUN_ID = 'pod_physics_lambda_sweep_normalized_residual_v2'

RESULT_DIR = os.path.join(BASE_RESULT_DIR, RUN_ID)
os.makedirs(RESULT_DIR, exist_ok=True)

COMMON_DATA_PATH = os.path.join(
    RESULT_DIR,
    'common_dataset_and_preprocess.npz'
)

COMMON_META_PATH = os.path.join(
    RESULT_DIR,
    'common_metadata.pth'
)

SELECTED_FILES_PATH = os.path.join(
    RESULT_DIR,
    'selected_files.csv'
)

SWEEP_SUMMARY_CSV = os.path.join(
    RESULT_DIR,
    'lambda_sweep_summary.csv'
)

SWEEP_SUMMARY_JSON = os.path.join(
    RESULT_DIR,
    'lambda_sweep_summary.json'
)

SELECTION_JSON_PATH = os.path.join(
    RESULT_DIR,
    'optimal_lambda_selection.json'
)

OPTIMAL_MODEL_BUNDLE_PATH = os.path.join(
    RESULT_DIR,
    'optimal_model_bundle.pth'
)

OPTIMAL_MODEL_STATE_PATH = os.path.join(
    RESULT_DIR,
    'optimal_model_state_dict.pth'
)

OPTIMAL_LAMBDA_TXT_PATH = os.path.join(
    RESULT_DIR,
    'optimal_lambda.txt'
)


# -----------------------------------------------------------------------------
# 1.2 待搜索的 lambda
# -----------------------------------------------------------------------------
#
# lambda 仅控制归一化 PDE 残差：
#
#     total_loss
#     =
#     data_loss
#     + FIXED_POSITIVITY_WEIGHT * positivity_loss
#     + lambda * normalized_pde_loss
#
# normalized_pde_loss 由 R1、R2 各自按方程组成项能量归一化后直接相加，
# 不使用 beta1、beta2，也不再使用额外的 PDE_LOSS_SCALE。
#
# 建议先进行对数粗搜索，再在较优区间内加密。
#
LAMBDA_VALUES = [
    0.0,
    1.0e-7,
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0
]


# -----------------------------------------------------------------------------
# 1.3 固定非负约束权重与归一化稳定项
# -----------------------------------------------------------------------------

# 所有 lambda 实验均保持 D2 非负约束权重不变，
# 从而保证不同实验之间唯一变化的是 PDE 残差权重。
FIXED_POSITIVITY_WEIGHT = 0.1

# 归一化分母稳定项，防止方程组成项能量接近零时数值发散。
NORMALIZED_RESIDUAL_EPS = 1.0e-12


# -----------------------------------------------------------------------------
# 1.4 模型超参数
# -----------------------------------------------------------------------------

BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4

REQUESTED_NUM_POD_MODES = 100

DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6


# -----------------------------------------------------------------------------
# 1.5 训练超参数
# -----------------------------------------------------------------------------

ADAM_LR = 3e-4

ADAM_BATCH_SIZE = 64
VAL_BATCH_SIZE = 256

ADAM_ITERATIONS = 100000

# D1、D2 数据损失权重：
#
# data_loss
# =
# (1-D2_DATA_WEIGHT) * loss_D1
# +
# D2_DATA_WEIGHT * loss_D2
#
D2_DATA_WEIGHT = 0.5

VALIDATION_SPLIT = 0.2
SEED = 24

VALIDATION_FREQUENCY = 200

# 每若干次验证将日志保存一次，防止程序意外终止后日志全部丢失。
LOG_SAVE_EVERY_N_VALIDATIONS = 10


# -----------------------------------------------------------------------------
# 1.6 早停设置
# -----------------------------------------------------------------------------
#
# 早停监控指标不是带 lambda 的总损失，
# 而是与 lambda 无关的验证集 balanced NRMSE。
#
EARLY_STOPPING_PATIENCE = 20

WARMUP_STEPS = 5000

# 原代码可能在 warmup 尚未结束时提前停止。
# 这里规定至少在该步数后才允许早停。
EARLY_STOP_START_ITER = max(
    WARMUP_STEPS,
    10000
)

MIN_IMPROVEMENT = 1e-8


# -----------------------------------------------------------------------------
# 1.7 数据设置
# -----------------------------------------------------------------------------

TARGET_NUM_FILES = 2500


# -----------------------------------------------------------------------------
# 1.8 保存与绘图选项
# -----------------------------------------------------------------------------

# 是否保存完整验证集的预测值、真值和误差。
SAVE_FULL_VALIDATION_ARRAYS = True

# 是否保存验证集 PDE 残差图。
SAVE_RESIDUAL_MAPS = True

# 是否为每个 lambda 都画验证集预测结果。
PLOT_PREDICTIONS_FOR_EACH_LAMBDA = True

NUM_PREDICTION_SAMPLES = 4


# -----------------------------------------------------------------------------
# 1.9 计算设备与数据类型
# -----------------------------------------------------------------------------

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

DTYPE = torch.float64

EPS = 1e-12


# =============================================================================
# 2. 基础工具函数
# =============================================================================

def set_seed(seed: int = 24):
    """
    固定随机种子，尽量保证不同 lambda 使用完全相同的：

    1. 网络初始参数；
    2. DataLoader 打乱顺序；
    3. Dropout 随机序列；
    4. NumPy 随机操作；
    5. Python random 随机操作。
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 尽量保证可复现。
    # 某些 GPU 算子启用确定性后可能速度下降。
    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def safe_mkdir(path: str):
    """
    安全创建文件夹。
    """

    os.makedirs(
        path,
        exist_ok=True
    )


def lambda_tag(lambda_value: float) -> str:
    """
    将 lambda 转换为适合作为目录名的字符串。

    例如：
        0        -> lambda_0
        0.001    -> lambda_1p000e-03
        0.1      -> lambda_1p000e-01
    """

    if lambda_value == 0:
        return 'lambda_0'

    return (
        f'lambda_{lambda_value:.3e}'
        .replace('+', '')
        .replace('.', 'p')
    )


def ensure_required_columns(
    df: pd.DataFrame,
    required_cols,
    file_path: str
):
    """
    检查数据文件是否包含所有必要字段。
    """

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f'文件 {file_path} 缺少必要列: {missing}'
        )


def is_uniform_grid(
    arr: np.ndarray,
    tol: float = 1e-12
) -> bool:
    """
    检查一维网格是否为均匀网格。
    """

    if len(arr) < 2:
        return False

    diffs = np.diff(arr)

    return np.allclose(
        diffs,
        diffs[0],
        atol=tol,
        rtol=tol
    )


def save_json(
    path: str,
    payload
):
    """
    将包含 NumPy、Torch 对象的数据保存为 JSON。
    """

    def convert(obj):

        if isinstance(obj, dict):
            return {
                str(key): convert(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [
                convert(value)
                for value in obj
            ]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, torch.Tensor):
            return (
                obj.detach()
                .cpu()
                .numpy()
                .tolist()
            )

        if isinstance(obj, torch.device):
            return str(obj)

        return obj

    with open(
        path,
        'w',
        encoding='utf-8'
    ) as file:

        json.dump(
            convert(payload),
            file,
            ensure_ascii=False,
            indent=2
        )


def save_text(
    path: str,
    text: str
):
    """
    保存文本文件。
    """

    with open(
        path,
        'w',
        encoding='utf-8'
    ) as file:

        file.write(text)


def clone_state_dict_to_cpu(
    model: nn.Module
):
    """
    将模型 state_dict 深拷贝到 CPU。

    避免后续训练覆盖当前最佳模型，
    同时避免最佳参数长期占用 GPU 显存。
    """

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def torch_load_compat(
    path,
    map_location=None
):
    """
    兼容不同版本 PyTorch 的 torch.load。
    """

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location
        )


def count_trainable_parameters(
    model: nn.Module
) -> int:
    """
    统计模型可训练参数数量。
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# =============================================================================
# 3. 数据加载与预处理
# =============================================================================

def load_unified_data(
    data_dir,
    target_num_files
):
    """
    从 data_dir 中读取 data_*.csv 文件。

    每个文件应至少包含以下列：

    - A
    - tau
    - nu
    - kappa
    - d_diffusion
    - D1_adj
    - D2_adj

    所有文件必须使用完全相同的 A、tau 网格。

    返回
    ----
    branch_inputs_np
        shape = [N, 3]

        每个样本的输入参数：
            [nu, kappa, d_diffusion]

    y_snapshots_np
        shape = [N, 2 * num_tau * num_a]

        输出顺序为：
            [D1_flat, D2_flat]

    unified_a_grid
        A 方向统一网格。

    unified_tau_grid
        tau 方向统一网格。

    valid_files
        成功读取的文件路径。
    """

    print(
        f'从 {data_dir} 加载统一数据...'
    )

    all_available_files = sorted(
        glob.glob(
            os.path.join(
                data_dir,
                'data_*.csv'
            )
        )
    )

    if not all_available_files:
        raise FileNotFoundError(
            f'在 {data_dir} 中未找到任何 data_*.csv 文件。'
        )

    selected_files = random.sample(
        all_available_files,
        min(
            len(all_available_files),
            target_num_files
        )
    )

    branch_inputs_list = []
    y_snapshots_list = []

    valid_files = []
    failed_records = []

    unified_a_grid = None
    unified_tau_grid = None

    required_cols = [
        'A',
        'tau',
        'nu',
        'kappa',
        'd_diffusion',
        'D1_adj',
        'D2_adj'
    ]

    for file_path in tqdm(
        selected_files,
        desc='加载快照'
    ):

        try:
            df = pd.read_csv(
                file_path,
                on_bad_lines='skip'
            )

            ensure_required_columns(
                df,
                required_cols,
                file_path
            )

            df = df.dropna(
                subset=[
                    'D1_adj',
                    'D2_adj'
                ]
            )

            if df.empty:
                raise ValueError(
                    '删除缺失值后为空表。'
                )

            a_grid = np.sort(
                df['A'].unique()
            )

            tau_grid = np.sort(
                df['tau'].unique()
            )

            if (
                len(a_grid) < 3
                or len(tau_grid) < 3
            ):
                raise ValueError(
                    '中心差分要求 A 和 tau 两个方向均至少有 3 个点。'
                )

            if unified_a_grid is None:

                unified_a_grid = a_grid
                unified_tau_grid = tau_grid

            else:

                if not np.array_equal(
                    a_grid,
                    unified_a_grid
                ):
                    raise ValueError(
                        'A 网格与统一网格不一致。'
                    )

                if not np.array_equal(
                    tau_grid,
                    unified_tau_grid
                ):
                    raise ValueError(
                        'tau 网格与统一网格不一致。'
                    )

            if not is_uniform_grid(a_grid):
                raise ValueError(
                    'A 网格不是均匀网格。'
                )

            if not is_uniform_grid(tau_grid):
                raise ValueError(
                    'tau 网格不是均匀网格。'
                )

            # 检查同一个文件内部物理参数是否为常数。
            for col in [
                'nu',
                'kappa',
                'd_diffusion'
            ]:

                if df[col].nunique(
                    dropna=True
                ) != 1:

                    raise ValueError(
                        f'参数 {col} 在同一文件内部并非常数。'
                    )

            # 统一按照 tau、A 排序。
            df_sorted = (
                df.sort_values(
                    by=[
                        'tau',
                        'A'
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            expected_len = (
                len(unified_a_grid)
                * len(unified_tau_grid)
            )

            if len(df_sorted) != expected_len:
                raise ValueError(
                    '网格点数不完整或存在重复。'
                    f'期望 {expected_len}，'
                    f'实际 {len(df_sorted)}。'
                )

            # 检查每一个 (tau, A) 网格点是否唯一。
            if df_sorted.duplicated(
                subset=[
                    'tau',
                    'A'
                ]
            ).any():

                raise ValueError(
                    '存在重复的 (tau, A) 网格点。'
                )

            params = (
                df_sorted[
                    [
                        'nu',
                        'kappa',
                        'd_diffusion'
                    ]
                ]
                .iloc[0]
                .values
                .astype(np.float64)
            )

            final_snapshot = np.concatenate(
                [
                    df_sorted[
                        'D1_adj'
                    ].values.astype(np.float64),

                    df_sorted[
                        'D2_adj'
                    ].values.astype(np.float64)
                ]
            )

            if not np.all(
                np.isfinite(final_snapshot)
            ):
                raise ValueError(
                    '输出快照包含 NaN 或 Inf。'
                )

            if not np.all(
                np.isfinite(params)
            ):
                raise ValueError(
                    '输入参数包含 NaN 或 Inf。'
                )

            branch_inputs_list.append(
                params
            )

            y_snapshots_list.append(
                final_snapshot
            )

            valid_files.append(
                file_path
            )

        except Exception as exception:

            failed_records.append(
                {
                    'file': file_path,
                    'status': 'failed',
                    'reason': str(exception)
                }
            )

    if not branch_inputs_list:
        raise ValueError(
            '未成功加载任何有效样本。'
        )

    branch_inputs_np = np.asarray(
        branch_inputs_list,
        dtype=np.float64
    )

    y_snapshots_np = np.asarray(
        y_snapshots_list,
        dtype=np.float64
    )

    file_records = [
        {
            'file': file_path,
            'status': 'valid',
            'reason': ''
        }
        for file_path in valid_files
    ]

    file_records.extend(
        failed_records
    )

    pd.DataFrame(
        file_records
    ).to_csv(
        SELECTED_FILES_PATH,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        f'\n成功加载样本数: {len(branch_inputs_np)}'
    )

    print(
        f'失败文件数: {len(failed_records)}'
    )

    print(
        f'A 网格点数: {len(unified_a_grid)}'
    )

    print(
        f'tau 网格点数: {len(unified_tau_grid)}'
    )

    print(
        f'输出场维数: {y_snapshots_np.shape[1]}'
    )

    return (
        branch_inputs_np,
        y_snapshots_np,
        np.asarray(
            unified_a_grid,
            dtype=np.float64
        ),
        np.asarray(
            unified_tau_grid,
            dtype=np.float64
        ),
        valid_files
    )


def manual_scaler(
    data,
    mean=None,
    std=None
):
    """
    手工标准化：

        scaled = (data - mean) / std

    当 mean、std 未提供时，仅使用当前 data 估计。
    """

    if mean is None or std is None:

        mean = np.mean(
            data,
            axis=0
        )

        std = np.std(
            data,
            axis=0
        )

        std = np.asarray(
            std,
            dtype=np.float64
        )

        std[
            std < 1e-10
        ] = 1.0

        scaled = (
            data - mean
        ) / std

        return (
            scaled,
            mean,
            std
        )

    return (
        data - mean
    ) / std


def inverse_manual_scaler(
    data_scaled,
    mean,
    std
):
    """
    标准化逆变换。
    """

    return (
        data_scaled * std
        + mean
    )


def pod(
    y_data_scaled,
    requested_num_modes
):
    """
    对训练集标准化输出进行 POD/SVD。

    注意：
    POD 只能使用训练集拟合，不能使用验证集，
    否则会造成验证信息泄漏。
    """

    y_mean_pod_scaled = np.mean(
        y_data_scaled,
        axis=0
    )

    centered = (
        y_data_scaled
        - y_mean_pod_scaled
    )

    _, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False
    )

    actual_num_modes = min(
        requested_num_modes,
        vt.shape[0]
    )

    pod_basis = (
        vt.T[
            :,
            :actual_num_modes
        ]
    )

    return (
        y_mean_pod_scaled,
        pod_basis,
        singular_values,
        actual_num_modes
    )


# =============================================================================
# 4. 模型定义
# =============================================================================

class MLP(nn.Module):
    """
    参数到 POD 系数的多层感知机。
    """

    def __init__(
        self,
        input_dim,
        hidden_units,
        num_hidden_layers,
        output_dim,
        dropout_rate
    ):
        super().__init__()

        layers = [
            nn.Linear(
                input_dim,
                hidden_units,
                dtype=DTYPE
            )
        ]

        for _ in range(
            num_hidden_layers
        ):

            layers.extend(
                [
                    nn.GELU(),

                    nn.LayerNorm(
                        hidden_units,
                        dtype=DTYPE
                    ),

                    nn.Dropout(
                        p=dropout_rate
                    ),

                    nn.Linear(
                        hidden_units,
                        hidden_units,
                        dtype=DTYPE
                    )
                ]
            )

        layers.extend(
            [
                nn.GELU(),

                nn.LayerNorm(
                    hidden_units,
                    dtype=DTYPE
                ),

                nn.Dropout(
                    p=dropout_rate
                ),

                nn.Linear(
                    hidden_units,
                    output_dim,
                    dtype=DTYPE
                )
            ]
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x
    ):
        return self.network(x)


class PODDeepONet(nn.Module):
    """
    简化的 POD-DeepONet：

        branch input
             ↓
            MLP
             ↓
        POD coefficients
             ↓
        POD reconstruction
             ↓
        scaled output field
    """

    def __init__(
        self,
        branch_input_dim,
        hidden_units,
        num_hidden_layers,
        num_pod_modes,
        pod_basis,
        y_mean_pod_scaled,
        dropout_rate
    ):
        super().__init__()

        self.branch = MLP(
            input_dim=branch_input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            output_dim=num_pod_modes,
            dropout_rate=dropout_rate
        )

        # POD 基底固定，不参与训练。
        self.register_buffer(
            'pod_basis',
            torch.tensor(
                pod_basis,
                dtype=DTYPE
            )
        )

        # POD 中心固定，不参与训练。
        self.register_buffer(
            'y_mean_pod_scaled',
            torch.tensor(
                y_mean_pod_scaled,
                dtype=DTYPE
            )
        )

    def forward(
        self,
        branch_x
    ):
        """
        输入
        ----
        branch_x:
            shape = [batch, 3]

        返回
        ----
        y_pred_scaled:
            shape = [batch, output_dim]
        """

        branch_coeffs = self.branch(
            branch_x
        )

        y_pred_scaled = (
            torch.matmul(
                branch_coeffs,
                self.pod_basis.T
            )
            + self.y_mean_pod_scaled
        )

        return y_pred_scaled

    def predict_scaled(
        self,
        branch_x
    ):
        """
        返回标准化空间中的预测。
        """

        self.eval()

        with torch.no_grad():
            return self.forward(
                branch_x
            )

    def predict(
        self,
        branch_x,
        y_mean_scaler,
        y_std_scaler
    ):
        """
        返回真实物理量空间中的预测。
        """

        self.eval()

        with torch.no_grad():

            y_pred_scaled = self.forward(
                branch_x
            )

            y_mean_t = torch.as_tensor(
                y_mean_scaler,
                dtype=DTYPE,
                device=branch_x.device
            )

            y_std_t = torch.as_tensor(
                y_std_scaler,
                dtype=DTYPE,
                device=branch_x.device
            )

            y_pred = (
                y_pred_scaled
                * y_std_t
                + y_mean_t
            )

            return y_pred


# =============================================================================
# 5. 物理损失与残差定义
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    总损失：

        L_total
        =
        L_data
        + FIXED_POSITIVITY_WEIGHT * L_pos
        + lambda_phys * L_pde_normalized

    其中：

        L_data
        =
        (1-w_D2) * L_D1
        +
        w_D2 * L_D2

        L_pos
        =
        mean(ReLU(-D2_pred)^2)

        L_pde
        =
        mean(R1^2)
        +
        mean(R2^2)
    """

    def __init__(
        self,
        unified_a_grid,
        unified_tau_grid,
        branch_mean,
        branch_std,
        y_mean_scaler,
        y_std_scaler,
        d2_data_weight=0.2,
        lambda_phys=0.1,
        positivity_weight=0.1,
        normalized_residual_eps=1.0e-12
    ):
        super().__init__()

        self.d2_data_weight = float(
            d2_data_weight
        )

        self.lambda_phys = float(
            lambda_phys
        )

        self.positivity_weight = float(
            positivity_weight
        )

        self.normalized_residual_eps = float(
            normalized_residual_eps
        )

        self.mse = nn.MSELoss()

        self.register_buffer(
            'a_grid',
            torch.tensor(
                unified_a_grid,
                dtype=DTYPE
            )
        )

        self.register_buffer(
            'tau_grid',
            torch.tensor(
                unified_tau_grid,
                dtype=DTYPE
            )
        )

        self.register_buffer(
            'branch_mean',
            torch.tensor(
                branch_mean,
                dtype=DTYPE
            )
        )

        self.register_buffer(
            'branch_std',
            torch.tensor(
                branch_std,
                dtype=DTYPE
            )
        )

        self.register_buffer(
            'y_mean_scaler',
            torch.tensor(
                y_mean_scaler,
                dtype=DTYPE
            )
        )

        self.register_buffer(
            'y_std_scaler',
            torch.tensor(
                y_std_scaler,
                dtype=DTYPE
            )
        )

        self.num_a = len(
            unified_a_grid
        )

        self.num_tau = len(
            unified_tau_grid
        )

        self.field_len = (
            self.num_a
            * self.num_tau
        )

        if (
            self.num_a < 3
            or self.num_tau < 3
        ):
            raise ValueError(
                '中心差分要求 A 和 tau 网格长度至少为 3。'
            )

        self.da = (
            self.a_grid[1]
            - self.a_grid[0]
        )

        self.dtau = (
            self.tau_grid[1]
            - self.tau_grid[0]
        )

    def decompose_prediction(
        self,
        y_pred_scaled,
        branch_x_scaled
    ):
        """
        将标准化预测恢复到物理量空间，并拆分 D1、D2。
        """

        y_pred = (
            y_pred_scaled
            * self.y_std_scaler
            + self.y_mean_scaler
        )

        branch_x = (
            branch_x_scaled
            * self.branch_std
            + self.branch_mean
        )

        d1_pred = (
            y_pred[
                :,
                :self.field_len
            ]
            .view(
                -1,
                self.num_tau,
                self.num_a
            )
        )

        d2_pred = (
            y_pred[
                :,
                self.field_len:
            ]
            .view(
                -1,
                self.num_tau,
                self.num_a
            )
        )

        nu = (
            branch_x[
                :,
                0
            ]
            .view(
                -1,
                1,
                1
            )
        )

        kappa = (
            branch_x[
                :,
                1
            ]
            .view(
                -1,
                1,
                1
            )
        )

        d_diff = (
            branch_x[
                :,
                2
            ]
            .view(
                -1,
                1,
                1
            )
        )

        a = self.a_grid.view(
            1,
            1,
            -1
        )

        tau = self.tau_grid.view(
            1,
            -1,
            1
        )

        a_safe = torch.clamp(
            a,
            min=1e-9
        )

        # 理论漂移项：
        #
        # D1_th = nu*a - (kappa/8)*a^3 + d/a
        #
        d1_th = (
            nu * a
            - (kappa / 8.0) * (a ** 3)
            + d_diff / a_safe
        )

        # 理论扩散项：
        #
        # D2_th = d_diffusion
        #
        d2_th = d_diff.expand_as(
            d1_th
        )

        return {
            'y_pred': y_pred,
            'branch_x': branch_x,
            'D1_pred': d1_pred,
            'D2_pred': d2_pred,
            'D1_th': d1_th,
            'D2_th': d2_th,
            'Tau': tau
        }

    def compute_residual_maps(
        self,
        y_pred_scaled,
        branch_x_scaled
    ):
        """
        计算 PDE 残差 R1、R2 和 D2 非负违约图。

        返回内容
        --------
        R1:
            第一条 PDE 残差。

        R2:
            第二条 PDE 残差。

        R1_scale_sq:
            R1 各组成项平方和，用于构造无量纲相对残差。

        R2_scale_sq:
            R2 各组成项平方和，用于构造无量纲相对残差。

        negative_D2:
            ReLU(-D2_pred)，表示 D2 小于零的违约幅值。

        相对 PDE 残差定义为：

            relative_R
            =
            sqrt(
                sum(R^2)
                /
                sum(term_1^2 + ... + term_n^2)
            )

        该指标不依赖 lambda 的权重，
        适合用于不同 lambda 之间横向比较。
        """

        parts = self.decompose_prediction(
            y_pred_scaled,
            branch_x_scaled
        )

        d1_pred = parts['D1_pred']
        d2_pred = parts['D2_pred']

        d1_th = parts['D1_th']
        d2_th = parts['D2_th']

        tau = parts['Tau']

        # U1 = tau * D1_tau
        # U2 = tau * D2_tau
        u1 = tau * d1_pred
        u2 = tau * d2_pred

        # 中心差分只在内部点计算。
        u1_in = u1[
            :,
            1:-1,
            1:-1
        ]

        d1_th_in = d1_th[
            :,
            :,
            1:-1
        ]

        d2_th_in = d2_th[
            :,
            :,
            1:-1
        ]

        # ---------------------------------------------------------------------
        # 对 tau 的一阶中心差分
        # ---------------------------------------------------------------------

        du1_dtau = (
            u1[
                :,
                2:,
                1:-1
            ]
            -
            u1[
                :,
                :-2,
                1:-1
            ]
        ) / (
            2.0 * self.dtau
        )

        du2_dtau = (
            u2[
                :,
                2:,
                1:-1
            ]
            -
            u2[
                :,
                :-2,
                1:-1
            ]
        ) / (
            2.0 * self.dtau
        )

        # ---------------------------------------------------------------------
        # 对 A 的一阶中心差分
        # ---------------------------------------------------------------------

        du1_da = (
            u1[
                :,
                1:-1,
                2:
            ]
            -
            u1[
                :,
                1:-1,
                :-2
            ]
        ) / (
            2.0 * self.da
        )

        du2_da = (
            u2[
                :,
                1:-1,
                2:
            ]
            -
            u2[
                :,
                1:-1,
                :-2
            ]
        ) / (
            2.0 * self.da
        )

        # ---------------------------------------------------------------------
        # 对 A 的二阶中心差分
        # ---------------------------------------------------------------------

        d2u1_da2 = (
            u1[
                :,
                1:-1,
                2:
            ]
            -
            2.0
            * u1[
                :,
                1:-1,
                1:-1
            ]
            +
            u1[
                :,
                1:-1,
                :-2
            ]
        ) / (
            self.da ** 2
        )

        d2u2_da2 = (
            u2[
                :,
                1:-1,
                2:
            ]
            -
            2.0
            * u2[
                :,
                1:-1,
                1:-1
            ]
            +
            u2[
                :,
                1:-1,
                :-2
            ]
        ) / (
            self.da ** 2
        )

        # ---------------------------------------------------------------------
        # 第一条 PDE：
        #
        # R1
        # =
        # dU1/dtau
        # -
        # D1_th * dU1/dA
        # -
        # D2_th * d2U1/dA2
        # -
        # D1_th
        # ---------------------------------------------------------------------

        r1_t1 = du1_dtau

        r1_t2 = (
            d1_th_in
            * du1_da
        )

        r1_t3 = (
            d2_th_in
            * d2u1_da2
        )

        r1_t4 = d1_th_in.expand_as(
            r1_t1
        )

        r1 = (
            r1_t1
            - r1_t2
            - r1_t3
            - r1_t4
        )

        r1_scale_sq = (
            r1_t1.square()
            + r1_t2.square()
            + r1_t3.square()
            + r1_t4.square()
        )

        # ---------------------------------------------------------------------
        # 第二条 PDE：
        #
        # R2
        # =
        # dU2/dtau
        # -
        # D1_th * dU2/dA
        # -
        # D2_th * d2U2/dA2
        # -
        # D1_th * U1
        # -
        # 2 * D2_th * dU1/dA
        # -
        # D2_th
        # ---------------------------------------------------------------------

        r2_t1 = du2_dtau

        r2_t2 = (
            d1_th_in
            * du2_da
        )

        r2_t3 = (
            d2_th_in
            * d2u2_da2
        )

        r2_t4 = (
            d1_th_in
            * u1_in
        )

        r2_t5 = (
            2.0
            * d2_th_in
            * du1_da
        )

        r2_t6 = d2_th_in.expand_as(
            r2_t1
        )

        r2 = (
            r2_t1
            - r2_t2
            - r2_t3
            - r2_t4
            - r2_t5
            - r2_t6
        )

        r2_scale_sq = (
            r2_t1.square()
            + r2_t2.square()
            + r2_t3.square()
            + r2_t4.square()
            + r2_t5.square()
            + r2_t6.square()
        )

        negative_d2 = torch.relu(
            -d2_pred
        )

        return {
            'R1': r1,
            'R2': r2,
            'R1_scale_sq': r1_scale_sq,
            'R2_scale_sq': r2_scale_sq,
            'negative_D2': negative_d2,
            'D2_pred': d2_pred
        }

    def forward(
        self,
        y_pred_scaled,
        y_true_scaled,
        branch_x_scaled
    ):
        """
        计算训练损失。
        """

        # ---------------------------------------------------------------------
        # 数据损失：标准化空间
        # ---------------------------------------------------------------------

        loss_d1_data = self.mse(
            y_pred_scaled[
                :,
                :self.field_len
            ],
            y_true_scaled[
                :,
                :self.field_len
            ]
        )

        loss_d2_data = self.mse(
            y_pred_scaled[
                :,
                self.field_len:
            ],
            y_true_scaled[
                :,
                self.field_len:
            ]
        )

        data_loss = (
            (
                1.0
                - self.d2_data_weight
            )
            * loss_d1_data
            +
            self.d2_data_weight
            * loss_d2_data
        )

        # ---------------------------------------------------------------------
        # 物理损失：真实物理量空间
        # ---------------------------------------------------------------------

        residuals = self.compute_residual_maps(
            y_pred_scaled,
            branch_x_scaled
        )

        loss_pos = torch.mean(
            residuals[
                'negative_D2'
            ].square()
        )

        # 原始残差 MSE，仅用于诊断与报告。
        loss_pde_r1_raw = torch.mean(
            residuals['R1'].square()
        )

        loss_pde_r2_raw = torch.mean(
            residuals['R2'].square()
        )

        # 归一化残差：每条方程分别用其各组成项平方和进行局部归一化。
        # 这样可以消除 R1、R2 量级差异，避免某一条方程主导训练。
        loss_pde_r1 = torch.mean(
            residuals['R1'].square()
            / (
                residuals['R1_scale_sq']
                + self.normalized_residual_eps
            )
        )

        loss_pde_r2 = torch.mean(
            residuals['R2'].square()
            / (
                residuals['R2_scale_sq']
                + self.normalized_residual_eps
            )
        )

        loss_pde = loss_pde_r1 + loss_pde_r2

        # 非负约束权重固定；lambda_phys 仅控制归一化 PDE 残差。
        physics_unweighted = (
            self.positivity_weight * loss_pos
            + loss_pde
        )

        total_loss = (
            data_loss
            + self.positivity_weight * loss_pos
            + self.lambda_phys * loss_pde
        )

        return {
            'total': total_loss,
            'data': data_loss,
            'd1_data': loss_d1_data,
            'd2_data': loss_d2_data,
            'pos': loss_pos,
            'pde': loss_pde,
            'pde_r1': loss_pde_r1,
            'pde_r2': loss_pde_r2,
            'pde_r1_raw': loss_pde_r1_raw,
            'pde_r2_raw': loss_pde_r2_raw,
            'physics_unweighted': physics_unweighted
        }


# =============================================================================
# 6. 学习率调度器
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """
    线性 warmup + cosine 衰减。
    """

    def lr_lambda(
        current_step
    ):

        if current_step < num_warmup_steps:

            return (
                float(current_step)
                /
                float(
                    max(
                        1,
                        num_warmup_steps
                    )
                )
            )

        progress = (
            float(
                current_step
                - num_warmup_steps
            )
            /
            float(
                max(
                    1,
                    num_training_steps
                    - num_warmup_steps
                )
            )
        )

        progress = min(
            max(
                progress,
                0.0
            ),
            1.0
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch
    )


# =============================================================================
# 7. DataLoader 与验证评估
# =============================================================================

def make_train_loader(
    branch_train_scaled,
    y_train_scaled,
    seed
):
    """
    创建训练集 DataLoader。

    使用独立 torch.Generator 固定 shuffle 顺序，
    确保每个 lambda 使用相同的 mini-batch 顺序。
    """

    dataset = TensorDataset(
        torch.from_numpy(
            branch_train_scaled
        ),
        torch.from_numpy(
            y_train_scaled
        )
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=ADAM_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=False
    )


def make_val_loader(
    branch_val_scaled,
    y_val_scaled
):
    """
    创建验证集 DataLoader。
    """

    dataset = TensorDataset(
        torch.from_numpy(
            branch_val_scaled
        ),
        torch.from_numpy(
            y_val_scaled
        )
    )

    return DataLoader(
        dataset,
        batch_size=min(
            VAL_BATCH_SIZE,
            len(dataset)
        ),
        shuffle=False,
        drop_last=False
    )


def evaluate_model(
    model,
    val_loader,
    loss_fn,
    y_mean_scaler,
    y_std_scaler,
    d2_data_weight,
    collect_arrays=False,
    collect_residual_maps=False
):
    """
    在完整验证集上计算模型指标。

    该函数同时计算两类指标：

    1. 与当前 lambda 相关的训练损失指标；
    2. 与 lambda 权重无关的模型精度和物理一致性指标。

    最优 lambda 不直接根据 val_total_loss 选择，
    因为不同 lambda 的 val_total_loss 定义不同。
    """

    model.eval()

    y_pred_scaled_batches = []
    y_true_scaled_batches = []
    branch_scaled_batches = []

    loss_sums = {
        'total': 0.0,
        'data': 0.0,
        'd1_data': 0.0,
        'd2_data': 0.0,
        'pos': 0.0,
        'pde': 0.0,
        'pde_r1': 0.0,
        'pde_r2': 0.0,
        'physics_unweighted': 0.0
    }

    total_samples = 0

    # PDE 残差累计量。
    r1_sq_sum = 0.0
    r2_sq_sum = 0.0

    r1_scale_sq_sum = 0.0
    r2_scale_sq_sum = 0.0

    r1_count = 0
    r2_count = 0

    # D2 非负约束累计量。
    negative_sq_sum = 0.0
    negative_count = 0

    negative_violation_count = 0
    d2_pred_count = 0

    residual_r1_batches = []
    residual_r2_batches = []
    negative_d2_batches = []

    with torch.no_grad():

        for (
            branch_cpu,
            y_true_cpu
        ) in val_loader:

            branch = branch_cpu.to(
                DEVICE,
                non_blocking=True
            )

            y_true_scaled = y_true_cpu.to(
                DEVICE,
                non_blocking=True
            )

            y_pred_scaled = model(
                branch
            )

            losses = loss_fn(
                y_pred_scaled,
                y_true_scaled,
                branch
            )

            batch_size = branch.shape[0]
            total_samples += batch_size

            for key in loss_sums:

                loss_sums[key] += (
                    float(
                        losses[key].item()
                    )
                    * batch_size
                )

            residuals = loss_fn.compute_residual_maps(
                y_pred_scaled,
                branch
            )

            r1 = residuals['R1']
            r2 = residuals['R2']

            r1_scale_sq = residuals[
                'R1_scale_sq'
            ]

            r2_scale_sq = residuals[
                'R2_scale_sq'
            ]

            negative_d2 = residuals[
                'negative_D2'
            ]

            d2_pred_tensor = residuals[
                'D2_pred'
            ]

            r1_sq_sum += float(
                torch.sum(
                    r1.square()
                ).item()
            )

            r2_sq_sum += float(
                torch.sum(
                    r2.square()
                ).item()
            )

            r1_scale_sq_sum += float(
                torch.sum(
                    r1_scale_sq
                ).item()
            )

            r2_scale_sq_sum += float(
                torch.sum(
                    r2_scale_sq
                ).item()
            )

            r1_count += r1.numel()
            r2_count += r2.numel()

            negative_sq_sum += float(
                torch.sum(
                    negative_d2.square()
                ).item()
            )

            negative_count += (
                negative_d2.numel()
            )

            negative_violation_count += int(
                torch.sum(
                    d2_pred_tensor < 0
                ).item()
            )

            d2_pred_count += (
                d2_pred_tensor.numel()
            )

            y_pred_scaled_batches.append(
                y_pred_scaled.cpu().numpy()
            )

            y_true_scaled_batches.append(
                y_true_scaled.cpu().numpy()
            )

            branch_scaled_batches.append(
                branch.cpu().numpy()
            )

            if collect_residual_maps:

                residual_r1_batches.append(
                    r1.cpu().numpy()
                )

                residual_r2_batches.append(
                    r2.cpu().numpy()
                )

                negative_d2_batches.append(
                    negative_d2.cpu().numpy()
                )

    y_pred_scaled_np = np.concatenate(
        y_pred_scaled_batches,
        axis=0
    )

    y_true_scaled_np = np.concatenate(
        y_true_scaled_batches,
        axis=0
    )

    branch_scaled_np = np.concatenate(
        branch_scaled_batches,
        axis=0
    )

    # 恢复到真实物理量空间。
    y_pred_np = (
        y_pred_scaled_np
        * y_std_scaler
        + y_mean_scaler
    )

    y_true_np = (
        y_true_scaled_np
        * y_std_scaler
        + y_mean_scaler
    )

    field_len = (
        y_true_np.shape[1]
        // 2
    )

    d1_true = y_true_np[
        :,
        :field_len
    ]

    d2_true = y_true_np[
        :,
        field_len:
    ]

    d1_pred = y_pred_np[
        :,
        :field_len
    ]

    d2_pred = y_pred_np[
        :,
        field_len:
    ]

    d1_err = (
        d1_pred
        - d1_true
    )

    d2_err = (
        d2_pred
        - d2_true
    )

    # -------------------------------------------------------------------------
    # 逐样本 RMSE
    # -------------------------------------------------------------------------

    d1_rmse_sample = np.sqrt(
        np.mean(
            d1_err ** 2,
            axis=1
        )
    )

    d2_rmse_sample = np.sqrt(
        np.mean(
            d2_err ** 2,
            axis=1
        )
    )

    # 使用整个验证集真值标准差作为归一化尺度。
    #
    # 不使用每个样本自己的标准差，
    # 避免某些接近常量的样本导致分母过小。
    d1_scale = float(
        np.std(d1_true)
    )

    d2_scale = float(
        np.std(d2_true)
    )

    if d1_scale < EPS:

        d1_scale = float(
            np.sqrt(
                np.mean(
                    d1_true ** 2
                )
            )
            + EPS
        )

    if d2_scale < EPS:

        d2_scale = float(
            np.sqrt(
                np.mean(
                    d2_true ** 2
                )
            )
            + EPS
        )

    d1_nrmse_sample = (
        d1_rmse_sample
        /
        (
            d1_scale
            + EPS
        )
    )

    d2_nrmse_sample = (
        d2_rmse_sample
        /
        (
            d2_scale
            + EPS
        )
    )

    # 与数据损失中的 D1、D2 权重保持一致。
    balanced_nrmse_sample = (
        (
            1.0
            - d2_data_weight
        )
        * d1_nrmse_sample
        +
        d2_data_weight
        * d2_nrmse_sample
    )

    n_val = len(
        balanced_nrmse_sample
    )

    balanced_mean = float(
        np.mean(
            balanced_nrmse_sample
        )
    )

    if n_val > 1:

        balanced_std = float(
            np.std(
                balanced_nrmse_sample,
                ddof=1
            )
        )

    else:
        balanced_std = 0.0

    balanced_se = (
        balanced_std
        /
        math.sqrt(
            max(
                n_val,
                1
            )
        )
    )

    # -------------------------------------------------------------------------
    # 相对 L2 误差
    # -------------------------------------------------------------------------

    d1_rel_l2 = float(
        np.linalg.norm(
            d1_err.ravel()
        )
        /
        (
            np.linalg.norm(
                d1_true.ravel()
            )
            + EPS
        )
    )

    d2_rel_l2 = float(
        np.linalg.norm(
            d2_err.ravel()
        )
        /
        (
            np.linalg.norm(
                d2_true.ravel()
            )
            + EPS
        )
    )

    # -------------------------------------------------------------------------
    # PDE 物理一致性指标
    # -------------------------------------------------------------------------

    pde_r1_relative = math.sqrt(
        r1_sq_sum
        /
        (
            r1_scale_sq_sum
            + EPS
        )
    )

    pde_r2_relative = math.sqrt(
        r2_sq_sum
        /
        (
            r2_scale_sq_sum
            + EPS
        )
    )

    pde_r1_rmse = math.sqrt(
        r1_sq_sum
        /
        max(
            r1_count,
            1
        )
    )

    pde_r2_rmse = math.sqrt(
        r2_sq_sum
        /
        max(
            r2_count,
            1
        )
    )

    # -------------------------------------------------------------------------
    # D2 非负违约指标
    # -------------------------------------------------------------------------

    negative_d2_rmse = math.sqrt(
        negative_sq_sum
        /
        max(
            negative_count,
            1
        )
    )

    d2_true_rms = float(
        np.sqrt(
            np.mean(
                d2_true ** 2
            )
        )
    )

    negative_d2_nrmse = (
        negative_d2_rmse
        /
        (
            d2_true_rms
            + EPS
        )
    )

    negative_d2_rate = (
        negative_violation_count
        /
        max(
            d2_pred_count,
            1
        )
    )

    # 该指标仅用于预测精度相近的模型之间进行物理一致性比较。
    #
    # PDE 两条残差取平均，再加上 D2 负值违约。
    physics_score = (
        0.5
        * (
            pde_r1_relative
            + pde_r2_relative
        )
        +
        negative_d2_nrmse
    )

    metrics = {
        # 与 lambda 相关的损失。
        'val_total_loss': (
            loss_sums['total']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_data_loss_scaled': (
            loss_sums['data']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_d1_loss_scaled': (
            loss_sums['d1_data']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_d2_loss_scaled': (
            loss_sums['d2_data']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_pos_loss': (
            loss_sums['pos']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_pde_loss': (
            loss_sums['pde']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_pde_r1_loss': (
            loss_sums['pde_r1']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_pde_r2_loss': (
            loss_sums['pde_r2']
            /
            max(
                total_samples,
                1
            )
        ),

        'val_physics_unweighted': (
            loss_sums['physics_unweighted']
            /
            max(
                total_samples,
                1
            )
        ),

        # 与 lambda 权重无关的真实空间预测精度。
        'd1_rmse': float(
            np.sqrt(
                np.mean(
                    d1_err ** 2
                )
            )
        ),

        'd2_rmse': float(
            np.sqrt(
                np.mean(
                    d2_err ** 2
                )
            )
        ),

        'd1_mae': float(
            np.mean(
                np.abs(
                    d1_err
                )
            )
        ),

        'd2_mae': float(
            np.mean(
                np.abs(
                    d2_err
                )
            )
        ),

        'd1_rel_l2': d1_rel_l2,
        'd2_rel_l2': d2_rel_l2,

        'd1_nrmse_mean': float(
            np.mean(
                d1_nrmse_sample
            )
        ),

        'd2_nrmse_mean': float(
            np.mean(
                d2_nrmse_sample
            )
        ),

        'balanced_nrmse_mean': balanced_mean,
        'balanced_nrmse_std': balanced_std,
        'balanced_nrmse_se': balanced_se,

        # 物理一致性。
        'pde_r1_relative': pde_r1_relative,
        'pde_r2_relative': pde_r2_relative,

        'pde_r1_rmse': pde_r1_rmse,
        'pde_r2_rmse': pde_r2_rmse,

        'negative_d2_rmse': negative_d2_rmse,
        'negative_d2_nrmse': negative_d2_nrmse,
        'negative_d2_rate': negative_d2_rate,

        'physics_score': physics_score,

        'num_val_samples': int(
            n_val
        )
    }

    per_sample_df = pd.DataFrame(
        {
            'sample_index_in_validation': np.arange(
                n_val
            ),

            'd1_rmse': d1_rmse_sample,

            'd2_rmse': d2_rmse_sample,

            'd1_nrmse': d1_nrmse_sample,

            'd2_nrmse': d2_nrmse_sample,

            'balanced_nrmse': balanced_nrmse_sample
        }
    )

    arrays = None

    if collect_arrays:

        arrays = {
            'branch_scaled': branch_scaled_np,

            'y_true_scaled': y_true_scaled_np,

            'y_pred_scaled': y_pred_scaled_np,

            'y_true': y_true_np,

            'y_pred': y_pred_np,

            'error': (
                y_pred_np
                - y_true_np
            ),

            'absolute_error': np.abs(
                y_pred_np
                - y_true_np
            )
        }

        if collect_residual_maps:

            arrays.update(
                {
                    'R1': np.concatenate(
                        residual_r1_batches,
                        axis=0
                    ),

                    'R2': np.concatenate(
                        residual_r2_batches,
                        axis=0
                    ),

                    'negative_D2': np.concatenate(
                        negative_d2_batches,
                        axis=0
                    )
                }
            )

    return (
        metrics,
        per_sample_df,
        arrays
    )


# =============================================================================
# 8. 日志与绘图
# =============================================================================

def save_training_logs(
    logs,
    path
):
    """
    保存训练与验证日志。
    """

    pd.DataFrame(
        logs
    ).to_csv(
        path,
        index=False,
        encoding='utf-8-sig'
    )


def plot_single_lambda_training(
    log_df,
    save_path,
    lambda_phys
):
    """
    绘制单个 lambda 的训练历史。
    """

    fig, ax = plt.subplots(
        figsize=(
            13,
            8
        )
    )

    train_rows = log_df[
        log_df[
            'record_type'
        ] == 'train'
    ]

    val_rows = log_df[
        log_df[
            'record_type'
        ] == 'validation'
    ]

    ax.plot(
        train_rows[
            'iteration'
        ],
        train_rows[
            'train_total_loss'
        ],
        label='Training total loss',
        alpha=0.55,
        linewidth=1.2
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'val_total_loss'
        ],
        label='Validation total loss',
        linewidth=2.0,
        marker='o',
        markersize=3
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'balanced_nrmse_mean'
        ],
        label='Validation balanced NRMSE',
        linewidth=2.0,
        linestyle='--'
    )

    if not val_rows.empty:

        best_row = val_rows.loc[
            val_rows[
                'balanced_nrmse_mean'
            ].idxmin()
        ]

        ax.scatter(
            [
                best_row[
                    'iteration'
                ]
            ],
            [
                best_row[
                    'balanced_nrmse_mean'
                ]
            ],
            s=100,
            zorder=5,
            label=(
                'Best accuracy @ '
                f'{int(best_row["iteration"])}'
            )
        )

    ax.axvline(
        WARMUP_STEPS,
        linestyle=':',
        linewidth=1.5,
        label='Warmup end'
    )

    ax.axvline(
        EARLY_STOP_START_ITER,
        linestyle='-.',
        linewidth=1.5,
        label='Early-stop enabled'
    )

    ax.set_xlabel(
        'Iteration'
    )

    ax.set_ylabel(
        'Metric'
    )

    ax.set_yscale(
        'log'
    )

    ax.set_title(
        f'Training history, lambda={lambda_phys:g}'
    )

    ax.grid(
        True,
        which='both',
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300
    )

    plt.close(fig)


def plot_single_lambda_physics(
    log_df,
    save_path,
    lambda_phys
):
    """
    绘制单个 lambda 的验证集物理一致性指标。
    """

    val_rows = log_df[
        log_df[
            'record_type'
        ] == 'validation'
    ]

    if val_rows.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            13,
            7
        )
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'pde_r1_relative'
        ],
        marker='o',
        markersize=3,
        label='Relative R1 residual'
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'pde_r2_relative'
        ],
        marker='s',
        markersize=3,
        label='Relative R2 residual'
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'negative_d2_nrmse'
        ],
        marker='^',
        markersize=3,
        label='Normalized D2 negativity'
    )

    ax.plot(
        val_rows[
            'iteration'
        ],
        val_rows[
            'physics_score'
        ],
        linewidth=2.0,
        label='Physics score'
    )

    ax.set_xlabel(
        'Iteration'
    )

    ax.set_ylabel(
        'Dimensionless metric'
    )

    ax.set_yscale(
        'log'
    )

    ax.set_title(
        f'Validation normalized-physics metrics, lambda={lambda_phys:g}'
    )

    ax.grid(
        True,
        which='both',
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300
    )

    plt.close(fig)


def plot_prediction_comparison(
    model,
    branch_val_scaled,
    y_val_np,
    unified_a_grid,
    unified_tau_grid,
    y_mean_scaler,
    y_std_scaler,
    sample_indices,
    save_path,
    lambda_phys
):
    """
    绘制验证集若干样本的：

    - D1 真值；
    - D1 预测；
    - D1 绝对误差；
    - D2 真值；
    - D2 预测；
    - D2 绝对误差。
    """

    num_samples = len(
        sample_indices
    )

    num_a = len(
        unified_a_grid
    )

    num_tau = len(
        unified_tau_grid
    )

    fig, axes = plt.subplots(
        num_samples,
        6,
        figsize=(
            30,
            5.2 * num_samples
        )
    )

    if num_samples == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    def draw(
        ax,
        data,
        title,
        vmin=None,
        vmax=None,
        cmap='viridis'
    ):

        image = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=[
                unified_a_grid.min(),
                unified_a_grid.max(),
                unified_tau_grid.min(),
                unified_tau_grid.max()
            ],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            'A'
        )

        ax.set_ylabel(
            'tau'
        )

        return image

    model.eval()

    with torch.no_grad():

        for (
            row_idx,
            sample_idx
        ) in enumerate(
            sample_indices
        ):

            branch_input = (
                torch.from_numpy(
                    branch_val_scaled[
                        sample_idx
                    ]
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            prediction = (
                model.predict(
                    branch_input,
                    y_mean_scaler,
                    y_std_scaler
                )
                .cpu()
                .numpy()
                .ravel()
            )

            true = y_val_np[
                sample_idx
            ]

            half = (
                len(true)
                // 2
            )

            d1_true = (
                true[
                    :half
                ]
                .reshape(
                    num_tau,
                    num_a
                )
            )

            d2_true = (
                true[
                    half:
                ]
                .reshape(
                    num_tau,
                    num_a
                )
            )

            d1_pred = (
                prediction[
                    :half
                ]
                .reshape(
                    num_tau,
                    num_a
                )
            )

            d2_pred = (
                prediction[
                    half:
                ]
                .reshape(
                    num_tau,
                    num_a
                )
            )

            d1_error = np.abs(
                d1_pred
                - d1_true
            )

            d2_error = np.abs(
                d2_pred
                - d2_true
            )

            d1_vmin = min(
                d1_true.min(),
                d1_pred.min()
            )

            d1_vmax = max(
                d1_true.max(),
                d1_pred.max()
            )

            d2_vmin = min(
                d2_true.min(),
                d2_pred.min()
            )

            d2_vmax = max(
                d2_true.max(),
                d2_pred.max()
            )

            images = [
                draw(
                    axes[
                        row_idx,
                        0
                    ],
                    d1_true,
                    f'Sample {sample_idx}: D1 true',
                    d1_vmin,
                    d1_vmax
                ),

                draw(
                    axes[
                        row_idx,
                        1
                    ],
                    d1_pred,
                    f'Sample {sample_idx}: D1 pred',
                    d1_vmin,
                    d1_vmax
                ),

                draw(
                    axes[
                        row_idx,
                        2
                    ],
                    d1_error,
                    f'Sample {sample_idx}: |D1 error|',
                    cmap='magma'
                ),

                draw(
                    axes[
                        row_idx,
                        3
                    ],
                    d2_true,
                    f'Sample {sample_idx}: D2 true',
                    d2_vmin,
                    d2_vmax,
                    cmap='plasma'
                ),

                draw(
                    axes[
                        row_idx,
                        4
                    ],
                    d2_pred,
                    f'Sample {sample_idx}: D2 pred',
                    d2_vmin,
                    d2_vmax,
                    cmap='plasma'
                ),

                draw(
                    axes[
                        row_idx,
                        5
                    ],
                    d2_error,
                    f'Sample {sample_idx}: |D2 error|',
                    cmap='magma'
                )
            ]

            for (
                col_idx,
                image
            ) in enumerate(
                images
            ):

                fig.colorbar(
                    image,
                    ax=axes[
                        row_idx,
                        col_idx
                    ],
                    fraction=0.046,
                    pad=0.04
                )

    fig.suptitle(
        f'Validation predictions, lambda={lambda_phys:g}',
        fontsize=18,
        y=0.995
    )

    fig.tight_layout(
        rect=[
            0,
            0.01,
            1,
            0.985
        ]
    )

    fig.savefig(
        save_path,
        dpi=300
    )

    plt.close(fig)


def plot_lambda_sweep(
    summary_df,
    selection_info,
    result_dir
):
    """
    绘制不同 lambda 的综合对比图。
    """

    labels = [
        f'{value:g}'
        for value in summary_df[
            'lambda_phys'
        ]
    ]

    x = np.arange(
        len(summary_df)
    )

    # -------------------------------------------------------------------------
    # 图 1：lambda 与预测精度
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            13,
            7
        )
    )

    ax.errorbar(
        x,
        summary_df[
            'balanced_nrmse_mean'
        ],
        yerr=summary_df[
            'balanced_nrmse_se'
        ],
        marker='o',
        capsize=4,
        label='Balanced NRMSE ± SE'
    )

    ax.plot(
        x,
        summary_df[
            'd1_nrmse_mean'
        ],
        marker='s',
        label='D1 NRMSE'
    )

    ax.plot(
        x,
        summary_df[
            'd2_nrmse_mean'
        ],
        marker='^',
        label='D2 NRMSE'
    )

    ax.axhline(
        selection_info[
            'accuracy_threshold'
        ],
        linestyle='--',
        label='One-SE accuracy threshold'
    )

    ax.scatter(
        [
            selection_info[
                'optimal_row_index'
            ]
        ],
        [
            selection_info[
                'optimal_balanced_nrmse'
            ]
        ],
        s=140,
        zorder=6,
        label=(
            'Selected lambda='
            f'{selection_info["optimal_lambda"]:g}'
        )
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha='right'
    )

    ax.set_xlabel(
        'lambda'
    )

    ax.set_ylabel(
        'Normalized prediction error'
    )

    ax.set_yscale(
        'log'
    )

    ax.set_title(
        'Prediction accuracy versus lambda'
    )

    ax.grid(
        True,
        which='both',
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            result_dir,
            'lambda_vs_accuracy.png'
        ),
        dpi=300
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 图 2：lambda 与物理一致性
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            13,
            7
        )
    )

    ax.plot(
        x,
        summary_df[
            'pde_r1_relative'
        ],
        marker='o',
        label='Relative R1 residual'
    )

    ax.plot(
        x,
        summary_df[
            'pde_r2_relative'
        ],
        marker='s',
        label='Relative R2 residual'
    )

    ax.plot(
        x,
        summary_df[
            'negative_d2_nrmse'
        ],
        marker='^',
        label='Normalized D2 negativity'
    )

    ax.plot(
        x,
        summary_df[
            'physics_score'
        ],
        marker='D',
        linewidth=2.0,
        label='Physics score'
    )

    ax.scatter(
        [
            selection_info[
                'optimal_row_index'
            ]
        ],
        [
            selection_info[
                'optimal_physics_score'
            ]
        ],
        s=140,
        zorder=6,
        label=(
            'Selected lambda='
            f'{selection_info["optimal_lambda"]:g}'
        )
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha='right'
    )

    ax.set_xlabel(
        'lambda'
    )

    ax.set_ylabel(
        'Dimensionless physics metric'
    )

    ax.set_yscale(
        'log'
    )

    ax.set_title(
        'Physics consistency versus lambda'
    )

    ax.grid(
        True,
        which='both',
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            result_dir,
            'lambda_vs_physics.png'
        ),
        dpi=300
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # 图 3：预测精度—物理一致性 Pareto 图
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            10,
            8
        )
    )

    ax.scatter(
        summary_df[
            'balanced_nrmse_mean'
        ],
        summary_df[
            'physics_score'
        ],
        s=90
    )

    for _, row in summary_df.iterrows():

        ax.annotate(
            f'λ={row["lambda_phys"]:g}',
            (
                row[
                    'balanced_nrmse_mean'
                ],
                row[
                    'physics_score'
                ]
            ),
            xytext=(
                6,
                5
            ),
            textcoords='offset points',
            fontsize=9
        )

    ax.scatter(
        [
            selection_info[
                'optimal_balanced_nrmse'
            ]
        ],
        [
            selection_info[
                'optimal_physics_score'
            ]
        ],
        s=180,
        marker='*',
        label='Selected model'
    )

    ax.axvline(
        selection_info[
            'accuracy_threshold'
        ],
        linestyle='--',
        label='One-SE threshold'
    )

    ax.set_xlabel(
        'Balanced validation NRMSE (lower is better)'
    )

    ax.set_ylabel(
        'Physics score (lower is better)'
    )

    ax.set_xscale(
        'log'
    )

    ax.set_yscale(
        'log'
    )

    ax.set_title(
        'Accuracy–physics trade-off'
    )

    ax.grid(
        True,
        which='both',
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            result_dir,
            'lambda_pareto_accuracy_physics.png'
        ),
        dpi=300
    )

    plt.close(fig)


# =============================================================================
# 9. 单个 lambda 训练
# =============================================================================

def train_one_lambda(
    lambda_phys,
    branch_train_scaled,
    y_train_scaled,
    branch_val_scaled,
    y_val_scaled,
    y_val_np,
    common_payload,
    prediction_sample_indices
):
    """
    独立训练一个 lambda，并保存该 lambda 的全部结果。
    """

    lambda_dir = os.path.join(
        RESULT_DIR,
        lambda_tag(
            lambda_phys
        )
    )

    safe_mkdir(
        lambda_dir
    )

    best_state_path = os.path.join(
        lambda_dir,
        'best_model_state_dict.pth'
    )

    best_checkpoint_path = os.path.join(
        lambda_dir,
        'best_checkpoint.pth'
    )

    final_checkpoint_path = os.path.join(
        lambda_dir,
        'final_checkpoint.pth'
    )

    inference_bundle_path = os.path.join(
        lambda_dir,
        'inference_bundle.pth'
    )

    log_path = os.path.join(
        lambda_dir,
        'training_log.csv'
    )

    metrics_json_path = os.path.join(
        lambda_dir,
        'best_validation_metrics.json'
    )

    per_sample_path = os.path.join(
        lambda_dir,
        'validation_per_sample_metrics.csv'
    )

    predictions_path = os.path.join(
        lambda_dir,
        'validation_predictions_and_residuals.npz'
    )

    config_json_path = os.path.join(
        lambda_dir,
        'run_config.json'
    )

    print(
        '\n'
        + '=' * 88
    )

    print(
        f'开始训练 lambda = {lambda_phys:g}'
    )

    print(
        f'输出目录: {lambda_dir}'
    )

    print(
        '=' * 88
    )

    # 每个 lambda 都重置同一个随机种子。
    #
    # 这样保证：
    # 1. 初始网络参数一致；
    # 2. Dropout 随机序列一致；
    # 3. mini-batch 打乱顺序一致。
    set_seed(
        SEED
    )

    train_loader = make_train_loader(
        branch_train_scaled,
        y_train_scaled,
        SEED
    )

    val_loader = make_val_loader(
        branch_val_scaled,
        y_val_scaled
    )

    model = PODDeepONet(
        branch_input_dim=BRANCH_INPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_pod_modes=common_payload[
            'actual_num_modes'
        ],
        pod_basis=common_payload[
            'pod_basis'
        ],
        y_mean_pod_scaled=common_payload[
            'y_mean_pod_scaled'
        ],
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=ADAM_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        WARMUP_STEPS,
        ADAM_ITERATIONS
    )

    loss_fn = PhysicsInformedLoss(
        unified_a_grid=common_payload[
            'unified_a_grid'
        ],
        unified_tau_grid=common_payload[
            'unified_tau_grid'
        ],
        branch_mean=common_payload[
            'branch_mean'
        ],
        branch_std=common_payload[
            'branch_std'
        ],
        y_mean_scaler=common_payload[
            'y_mean_scaler'
        ],
        y_std_scaler=common_payload[
            'y_std_scaler'
        ],
        d2_data_weight=D2_DATA_WEIGHT,
        lambda_phys=lambda_phys,
        positivity_weight=FIXED_POSITIVITY_WEIGHT,
        normalized_residual_eps=NORMALIZED_RESIDUAL_EPS
    ).to(DEVICE)

    run_config = {
        'lambda_phys': lambda_phys,

        'FIXED_POSITIVITY_WEIGHT': FIXED_POSITIVITY_WEIGHT,

        'NORMALIZED_RESIDUAL_EPS': NORMALIZED_RESIDUAL_EPS,

        'BRANCH_INPUT_DIM': BRANCH_INPUT_DIM,

        'HIDDEN_UNITS': HIDDEN_UNITS,

        'NUM_HIDDEN_LAYERS': NUM_HIDDEN_LAYERS,

        'REQUESTED_NUM_POD_MODES': REQUESTED_NUM_POD_MODES,

        'ACTUAL_NUM_POD_MODES': common_payload[
            'actual_num_modes'
        ],

        'DROPOUT_RATE': DROPOUT_RATE,

        'WEIGHT_DECAY': WEIGHT_DECAY,

        'ADAM_LR': ADAM_LR,

        'ADAM_BATCH_SIZE': ADAM_BATCH_SIZE,

        'VAL_BATCH_SIZE': VAL_BATCH_SIZE,

        'ADAM_ITERATIONS': ADAM_ITERATIONS,

        'D2_DATA_WEIGHT': D2_DATA_WEIGHT,

        'VALIDATION_FREQUENCY': VALIDATION_FREQUENCY,

        'LOG_SAVE_EVERY_N_VALIDATIONS': LOG_SAVE_EVERY_N_VALIDATIONS,

        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,

        'WARMUP_STEPS': WARMUP_STEPS,

        'EARLY_STOP_START_ITER': EARLY_STOP_START_ITER,

        'SEED': SEED,

        'DEVICE': str(
            DEVICE
        ),

        'DTYPE': str(
            DTYPE
        ),

        'trainable_parameters': count_trainable_parameters(
            model
        )
    }

    save_json(
        config_json_path,
        run_config
    )

    log_records = []

    best_monitor = float(
        'inf'
    )

    best_iter = -1
    best_metrics = None
    best_state_cpu = None

    early_stop_counter = 0
    validation_counter = 0

    train_iterator = iter(
        train_loader
    )

    start_time = time.time()

    pbar = tqdm(
        range(
            ADAM_ITERATIONS
        ),
        desc=f'lambda={lambda_phys:g}'
    )

    for global_iter in pbar:

        try:
            (
                batch_branch_cpu,
                batch_y_cpu
            ) = next(
                train_iterator
            )

        except StopIteration:

            train_iterator = iter(
                train_loader
            )

            (
                batch_branch_cpu,
                batch_y_cpu
            ) = next(
                train_iterator
            )

        batch_branch = batch_branch_cpu.to(
            DEVICE,
            non_blocking=True
        )

        batch_y = batch_y_cpu.to(
            DEVICE,
            non_blocking=True
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        y_pred_scaled = model(
            batch_branch
        )

        train_losses = loss_fn(
            y_pred_scaled,
            batch_y,
            batch_branch
        )

        train_losses[
            'total'
        ].backward()

        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[
            0
        ][
            'lr'
        ]

        log_records.append(
            {
                'record_type': 'train',

                'iteration': global_iter,

                'learning_rate': current_lr,

                'train_total_loss': float(
                    train_losses[
                        'total'
                    ].item()
                ),

                'train_data_loss': float(
                    train_losses[
                        'data'
                    ].item()
                ),

                'train_d1_loss': float(
                    train_losses[
                        'd1_data'
                    ].item()
                ),

                'train_d2_loss': float(
                    train_losses[
                        'd2_data'
                    ].item()
                ),

                'train_pos_loss': float(
                    train_losses[
                        'pos'
                    ].item()
                ),

                'train_pde_loss': float(
                    train_losses[
                        'pde'
                    ].item()
                ),

                'val_total_loss': np.nan,

                'balanced_nrmse_mean': np.nan,

                'balanced_nrmse_se': np.nan,

                'pde_r1_relative': np.nan,

                'pde_r2_relative': np.nan,

                'negative_d2_nrmse': np.nan,

                'physics_score': np.nan
            }
        )

        should_validate = (
            global_iter
            % VALIDATION_FREQUENCY
            == 0
            or
            global_iter
            == ADAM_ITERATIONS - 1
        )

        if should_validate:

            validation_counter += 1

            (
                val_metrics,
                _,
                _
            ) = evaluate_model(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                y_mean_scaler=common_payload[
                    'y_mean_scaler'
                ],
                y_std_scaler=common_payload[
                    'y_std_scaler'
                ],
                d2_data_weight=D2_DATA_WEIGHT,
                collect_arrays=False,
                collect_residual_maps=False
            )

            # 使用与 lambda 无关的真实空间 balanced NRMSE
            # 作为 checkpoint 和早停监控指标。
            monitor = val_metrics[
                'balanced_nrmse_mean'
            ]

            log_records.append(
                {
                    'record_type': 'validation',

                    'iteration': global_iter,

                    'learning_rate': current_lr,

                    'train_total_loss': np.nan,

                    'train_data_loss': np.nan,

                    'train_d1_loss': np.nan,

                    'train_d2_loss': np.nan,

                    'train_pos_loss': np.nan,

                    'train_pde_loss': np.nan,

                    'val_total_loss': val_metrics[
                        'val_total_loss'
                    ],

                    'balanced_nrmse_mean': monitor,

                    'balanced_nrmse_se': val_metrics[
                        'balanced_nrmse_se'
                    ],

                    'pde_r1_relative': val_metrics[
                        'pde_r1_relative'
                    ],

                    'pde_r2_relative': val_metrics[
                        'pde_r2_relative'
                    ],

                    'negative_d2_nrmse': val_metrics[
                        'negative_d2_nrmse'
                    ],

                    'physics_score': val_metrics[
                        'physics_score'
                    ]
                }
            )

            pbar.set_postfix(
                {
                    'val_nrmse': (
                        f'{monitor:.3e}'
                    ),

                    'physics': (
                        f'{val_metrics["physics_score"]:.3e}'
                    ),

                    'lr': (
                        f'{current_lr:.2e}'
                    )
                }
            )

            print(
                f'\nIter {global_iter:6d} | '
                f'lambda={lambda_phys:.3e} | '
                f'Val NRMSE={monitor:.5e} | '
                f'R1rel={val_metrics["pde_r1_relative"]:.5e} | '
                f'R2rel={val_metrics["pde_r2_relative"]:.5e} | '
                f'D2neg={val_metrics["negative_d2_nrmse"]:.5e} | '
                f'LR={current_lr:.3e}'
            )

            if (
                monitor
                <
                best_monitor
                - MIN_IMPROVEMENT
            ):

                best_monitor = monitor
                best_iter = global_iter

                best_metrics = deepcopy(
                    val_metrics
                )

                best_state_cpu = clone_state_dict_to_cpu(
                    model
                )

                early_stop_counter = 0

                # 仅保存模型参数。
                torch.save(
                    best_state_cpu,
                    best_state_path
                )

                # 保存完整最佳 checkpoint。
                torch.save(
                    {
                        'iteration': best_iter,

                        'lambda_phys': lambda_phys,

                        'monitor_name': (
                            'balanced_nrmse_mean'
                        ),

                        'monitor_value': best_monitor,

                        'model_state_dict': best_state_cpu,

                        'optimizer_state_dict': optimizer.state_dict(),

                        'scheduler_state_dict': scheduler.state_dict(),

                        'validation_metrics': best_metrics,

                        'run_config': run_config
                    },
                    best_checkpoint_path
                )

            elif (
                global_iter
                >= EARLY_STOP_START_ITER
            ):

                early_stop_counter += 1

            # 定期保存日志。
            if (
                validation_counter
                % LOG_SAVE_EVERY_N_VALIDATIONS
                == 0
            ):

                save_training_logs(
                    log_records,
                    log_path
                )

            # warmup 和指定最小步数之后才允许早停。
            if (
                global_iter
                >= EARLY_STOP_START_ITER
                and
                early_stop_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    '\nEarly stopping: '
                    f'lambda={lambda_phys:g}, '
                    f'best_iter={best_iter}, '
                    f'best_balanced_nrmse={best_monitor:.6e}'
                )

                break

    elapsed_seconds = (
        time.time()
        - start_time
    )

    # 保存完整日志。
    save_training_logs(
        log_records,
        log_path
    )

    # 保存训练终止时的完整状态。
    torch.save(
        {
            'iteration': global_iter,

            'lambda_phys': lambda_phys,

            'model_state_dict': clone_state_dict_to_cpu(
                model
            ),

            'optimizer_state_dict': optimizer.state_dict(),

            'scheduler_state_dict': scheduler.state_dict(),

            'run_config': run_config,

            'elapsed_seconds': elapsed_seconds
        },
        final_checkpoint_path
    )

    if best_state_cpu is None:
        raise RuntimeError(
            f'lambda={lambda_phys:g} 未产生有效最佳模型。'
        )

    # -------------------------------------------------------------------------
    # 加载最佳精度 checkpoint，进行最终完整验证
    # -------------------------------------------------------------------------

    model.load_state_dict(
        best_state_cpu
    )

    model.eval()

    (
        final_metrics,
        per_sample_df,
        arrays
    ) = evaluate_model(
        model=model,
        val_loader=val_loader,
        loss_fn=loss_fn,
        y_mean_scaler=common_payload[
            'y_mean_scaler'
        ],
        y_std_scaler=common_payload[
            'y_std_scaler'
        ],
        d2_data_weight=D2_DATA_WEIGHT,
        collect_arrays=SAVE_FULL_VALIDATION_ARRAYS,
        collect_residual_maps=(
            SAVE_FULL_VALIDATION_ARRAYS
            and SAVE_RESIDUAL_MAPS
        )
    )

    final_metrics.update(
        {
            'lambda_phys': lambda_phys,

            'best_iter': best_iter,

            'elapsed_seconds': elapsed_seconds,

            'lambda_directory': lambda_dir,

            'best_state_path': best_state_path,

            'best_checkpoint_path': best_checkpoint_path,

            'final_checkpoint_path': final_checkpoint_path,

            'inference_bundle_path': inference_bundle_path
        }
    )

    save_json(
        metrics_json_path,
        final_metrics
    )

    per_sample_df.to_csv(
        per_sample_path,
        index=False,
        encoding='utf-8-sig'
    )

    if arrays is not None:

        np.savez_compressed(
            predictions_path,
            **arrays
        )

    # -------------------------------------------------------------------------
    # 保存自包含推理包
    # -------------------------------------------------------------------------
    #
    # 每个 lambda 均保存：
    # 1. 模型参数；
    # 2. 模型结构配置；
    # 3. 输入输出 scaler；
    # 4. POD 基底与 POD 均值；
    # 5. 网格；
    # 6. 验证指标；
    # 7. 训练配置。
    #
    inference_bundle = {
        'lambda_phys': lambda_phys,

        'model_state_dict': best_state_cpu,

        'model_config': {
            'branch_input_dim': BRANCH_INPUT_DIM,

            'hidden_units': HIDDEN_UNITS,

            'num_hidden_layers': NUM_HIDDEN_LAYERS,

            'num_pod_modes': common_payload[
                'actual_num_modes'
            ],

            'dropout_rate': DROPOUT_RATE
        },

        'branch_mean': common_payload[
            'branch_mean'
        ],

        'branch_std': common_payload[
            'branch_std'
        ],

        'y_mean_scaler': common_payload[
            'y_mean_scaler'
        ],

        'y_std_scaler': common_payload[
            'y_std_scaler'
        ],

        'y_mean_pod_scaled': common_payload[
            'y_mean_pod_scaled'
        ],

        'pod_basis': common_payload[
            'pod_basis'
        ],

        'singular_values': common_payload[
            'singular_values'
        ],

        'actual_num_modes': common_payload[
            'actual_num_modes'
        ],

        'unified_a_grid': common_payload[
            'unified_a_grid'
        ],

        'unified_tau_grid': common_payload[
            'unified_tau_grid'
        ],

        'validation_metrics': final_metrics,

        'training_config': run_config
    }

    torch.save(
        inference_bundle,
        inference_bundle_path
    )

    # -------------------------------------------------------------------------
    # 绘制该 lambda 的训练图
    # -------------------------------------------------------------------------

    log_df = pd.DataFrame(
        log_records
    )

    plot_single_lambda_training(
        log_df,
        os.path.join(
            lambda_dir,
            'training_history.png'
        ),
        lambda_phys
    )

    plot_single_lambda_physics(
        log_df,
        os.path.join(
            lambda_dir,
            'physics_history.png'
        ),
        lambda_phys
    )

    if PLOT_PREDICTIONS_FOR_EACH_LAMBDA:

        plot_prediction_comparison(
            model=model,
            branch_val_scaled=branch_val_scaled,
            y_val_np=y_val_np,
            unified_a_grid=common_payload[
                'unified_a_grid'
            ],
            unified_tau_grid=common_payload[
                'unified_tau_grid'
            ],
            y_mean_scaler=common_payload[
                'y_mean_scaler'
            ],
            y_std_scaler=common_payload[
                'y_std_scaler'
            ],
            sample_indices=prediction_sample_indices,
            save_path=os.path.join(
                lambda_dir,
                'prediction_comparison.png'
            ),
            lambda_phys=lambda_phys
        )

    # -------------------------------------------------------------------------
    # 清理当前 lambda 的资源
    # -------------------------------------------------------------------------

    del model
    del optimizer
    del scheduler
    del loss_fn
    del train_loader
    del val_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics


# =============================================================================
# 10. 最优 lambda 选择
# =============================================================================

def select_optimal_lambda(
    summary_df: pd.DataFrame
):
    """
    使用以下规则选择最优 lambda：

    第一步：寻找 balanced NRMSE 最小的模型。

    第二步：计算 one-standard-error 阈值：

        threshold
        =
        minimum balanced NRMSE
        +
        该模型 balanced NRMSE 的标准误

    第三步：将所有满足

        balanced NRMSE <= threshold

    的模型视为预测精度相近。

    第四步：在这些候选模型中选择 physics_score 最小者。

    若 physics_score 相同，则依次选择：

    1. balanced NRMSE 更小者；
    2. lambda 更小者。
    """

    if summary_df.empty:
        raise ValueError(
            'lambda 汇总结果为空。'
        )

    best_accuracy_idx = summary_df[
        'balanced_nrmse_mean'
    ].idxmin()

    best_accuracy_row = summary_df.loc[
        best_accuracy_idx
    ]

    accuracy_threshold = (
        float(
            best_accuracy_row[
                'balanced_nrmse_mean'
            ]
        )
        +
        float(
            best_accuracy_row[
                'balanced_nrmse_se'
            ]
        )
    )

    eligible_mask = (
        summary_df[
            'balanced_nrmse_mean'
        ]
        <=
        accuracy_threshold
        + EPS
    )

    eligible_df = summary_df.loc[
        eligible_mask
    ].copy()

    if eligible_df.empty:

        eligible_df = summary_df.loc[
            [
                best_accuracy_idx
            ]
        ].copy()

    eligible_df = eligible_df.sort_values(
        by=[
            'physics_score',
            'balanced_nrmse_mean',
            'lambda_phys'
        ],
        ascending=[
            True,
            True,
            True
        ]
    )

    optimal_row = eligible_df.iloc[
        0
    ]

    optimal_original_index = int(
        optimal_row.name
    )

    optimal_row_position = int(
        summary_df.index.get_loc(
            optimal_row.name
        )
    )

    selection_info = {
        'selection_rule': (
            '先用验证集 balanced NRMSE 的 one-standard-error '
            '规则筛选精度相近模型，再从候选模型中选择 '
            'physics_score 最小者。'
        ),

        'best_accuracy_lambda': float(
            best_accuracy_row[
                'lambda_phys'
            ]
        ),

        'best_accuracy_mean': float(
            best_accuracy_row[
                'balanced_nrmse_mean'
            ]
        ),

        'best_accuracy_se': float(
            best_accuracy_row[
                'balanced_nrmse_se'
            ]
        ),

        'accuracy_threshold': float(
            accuracy_threshold
        ),

        'eligible_lambdas': (
            eligible_df[
                'lambda_phys'
            ]
            .astype(float)
            .tolist()
        ),

        'optimal_lambda': float(
            optimal_row[
                'lambda_phys'
            ]
        ),

        'optimal_balanced_nrmse': float(
            optimal_row[
                'balanced_nrmse_mean'
            ]
        ),

        'optimal_physics_score': float(
            optimal_row[
                'physics_score'
            ]
        ),

        'optimal_original_index': optimal_original_index,

        'optimal_row_index': optimal_row_position,

        'optimal_lambda_directory': str(
            optimal_row[
                'lambda_directory'
            ]
        ),

        'optimal_inference_bundle_path': str(
            optimal_row[
                'inference_bundle_path'
            ]
        ),

        'optimal_best_state_path': str(
            optimal_row[
                'best_state_path'
            ]
        )
    }

    return selection_info


# =============================================================================
# 11. 主程序
# =============================================================================

if __name__ == '__main__':

    try:
        print(
            '=' * 88
        )

        print(
            'POD-DeepONet Physics-Weight Lambda Sweep'
        )

        print(
            '=' * 88
        )

        print(
            f'Device: {DEVICE}'
        )

        print(
            f'Dtype: {DTYPE}'
        )

        print(
            f'Result directory: {RESULT_DIR}'
        )

        print(
            f'Lambda values: {LAMBDA_VALUES}'
        )

        set_seed(
            SEED
        )

        # ---------------------------------------------------------------------
        # Step 1：加载一次数据，所有 lambda 共用
        # ---------------------------------------------------------------------

        (
            branch_inputs_np,
            y_snapshots_np,
            unified_a_grid,
            unified_tau_grid,
            valid_files
        ) = load_unified_data(
            DATA_DIR,
            TARGET_NUM_FILES
        )

        # ---------------------------------------------------------------------
        # Step 2：固定训练集与验证集索引
        # ---------------------------------------------------------------------

        all_indices = np.arange(
            len(
                branch_inputs_np
            )
        )

        (
            train_indices,
            val_indices
        ) = train_test_split(
            all_indices,
            test_size=VALIDATION_SPLIT,
            random_state=SEED,
            shuffle=True
        )

        branch_train_np = branch_inputs_np[
            train_indices
        ]

        branch_val_np = branch_inputs_np[
            val_indices
        ]

        y_train_np = y_snapshots_np[
            train_indices
        ]

        y_val_np = y_snapshots_np[
            val_indices
        ]

        print(
            f'\nTrain samples: {len(train_indices)}'
        )

        print(
            f'Validation samples: {len(val_indices)}'
        )

        if len(train_indices) == 0:
            raise ValueError(
                '训练集为空。'
            )

        if len(val_indices) == 0:
            raise ValueError(
                '验证集为空。'
            )

        # ---------------------------------------------------------------------
        # Step 3：只用训练集拟合 scaler
        # ---------------------------------------------------------------------

        (
            branch_train_scaled,
            branch_mean,
            branch_std
        ) = manual_scaler(
            branch_train_np
        )

        branch_val_scaled = manual_scaler(
            branch_val_np,
            branch_mean,
            branch_std
        )

        (
            y_train_scaled,
            y_mean_scaler,
            y_std_scaler
        ) = manual_scaler(
            y_train_np
        )

        y_val_scaled = manual_scaler(
            y_val_np,
            y_mean_scaler,
            y_std_scaler
        )

        # ---------------------------------------------------------------------
        # Step 4：只用训练集进行 POD 分解
        # ---------------------------------------------------------------------

        (
            y_mean_pod_scaled,
            pod_basis,
            singular_values,
            actual_num_modes
        ) = pod(
            y_train_scaled,
            REQUESTED_NUM_POD_MODES
        )

        singular_energy_sum = np.sum(
            singular_values ** 2
        )

        if singular_energy_sum <= 0:
            raise ValueError(
                'POD 奇异值能量为零，无法计算累计能量。'
            )

        energy = (
            singular_values ** 2
            /
            singular_energy_sum
        )

        cumulative_energy = np.cumsum(
            energy
        )

        retained_energy = float(
            cumulative_energy[
                actual_num_modes - 1
            ]
        )

        print(
            f'Actual POD modes: {actual_num_modes}'
        )

        print(
            f'Retained POD energy: {retained_energy:.8f}'
        )

        # ---------------------------------------------------------------------
        # Step 5：固定所有 lambda 使用相同的预测绘图样本
        # ---------------------------------------------------------------------

        rng = np.random.default_rng(
            SEED
        )

        prediction_sample_indices = np.sort(
            rng.choice(
                len(branch_val_np),
                size=min(
                    NUM_PREDICTION_SAMPLES,
                    len(branch_val_np)
                ),
                replace=False
            )
        )

        # ---------------------------------------------------------------------
        # Step 6：保存公共数据、划分、scaler 和 POD
        # ---------------------------------------------------------------------

        np.savez_compressed(
            COMMON_DATA_PATH,

            branch_inputs=branch_inputs_np,

            y_snapshots=y_snapshots_np,

            train_indices=train_indices,

            val_indices=val_indices,

            branch_train=branch_train_np,

            branch_val=branch_val_np,

            y_train=y_train_np,

            y_val=y_val_np,

            branch_train_scaled=branch_train_scaled,

            branch_val_scaled=branch_val_scaled,

            y_train_scaled=y_train_scaled,

            y_val_scaled=y_val_scaled,

            branch_mean=branch_mean,

            branch_std=branch_std,

            y_mean_scaler=y_mean_scaler,

            y_std_scaler=y_std_scaler,

            y_mean_pod_scaled=y_mean_pod_scaled,

            pod_basis=pod_basis,

            singular_values=singular_values,

            unified_a_grid=unified_a_grid,

            unified_tau_grid=unified_tau_grid,

            prediction_sample_indices=prediction_sample_indices
        )

        common_payload = {
            'branch_mean': branch_mean,

            'branch_std': branch_std,

            'y_mean_scaler': y_mean_scaler,

            'y_std_scaler': y_std_scaler,

            'y_mean_pod_scaled': y_mean_pod_scaled,

            'pod_basis': pod_basis,

            'singular_values': singular_values,

            'actual_num_modes': actual_num_modes,

            'retained_energy': retained_energy,

            'unified_a_grid': unified_a_grid,

            'unified_tau_grid': unified_tau_grid,

            'train_indices': train_indices,

            'val_indices': val_indices,

            'valid_files': valid_files,

            'prediction_sample_indices': prediction_sample_indices,

            'global_config': {
                'RUN_ID': RUN_ID,

                'LAMBDA_VALUES': LAMBDA_VALUES,

                'FIXED_POSITIVITY_WEIGHT': FIXED_POSITIVITY_WEIGHT,

        'NORMALIZED_RESIDUAL_EPS': NORMALIZED_RESIDUAL_EPS,

                'SEED': SEED,

                'VALIDATION_SPLIT': VALIDATION_SPLIT,

                'TARGET_NUM_FILES': TARGET_NUM_FILES,

                'DEVICE': str(
                    DEVICE
                ),

                'DTYPE': str(
                    DTYPE
                )
            }
        }

        torch.save(
            common_payload,
            COMMON_META_PATH
        )

        # ---------------------------------------------------------------------
        # Step 7：逐个 lambda 独立训练
        # ---------------------------------------------------------------------

        all_results = []

        for lambda_phys in LAMBDA_VALUES:

            result = train_one_lambda(
                lambda_phys=lambda_phys,

                branch_train_scaled=branch_train_scaled,

                y_train_scaled=y_train_scaled,

                branch_val_scaled=branch_val_scaled,

                y_val_scaled=y_val_scaled,

                y_val_np=y_val_np,

                common_payload=common_payload,

                prediction_sample_indices=prediction_sample_indices
            )

            all_results.append(
                result
            )

            # 每完成一个 lambda 就更新总表，
            # 避免长时间训练中途终止后全部结果丢失。
            pd.DataFrame(
                all_results
            ).to_csv(
                SWEEP_SUMMARY_CSV,
                index=False,
                encoding='utf-8-sig'
            )

            save_json(
                SWEEP_SUMMARY_JSON,
                all_results
            )

        summary_df = (
            pd.DataFrame(
                all_results
            )
            .reset_index(
                drop=True
            )
        )

        summary_df.to_csv(
            SWEEP_SUMMARY_CSV,
            index=False,
            encoding='utf-8-sig'
        )

        save_json(
            SWEEP_SUMMARY_JSON,
            all_results
        )

        # ---------------------------------------------------------------------
        # Step 8：自动选择最优 lambda
        # ---------------------------------------------------------------------

        selection_info = select_optimal_lambda(
            summary_df
        )

        save_json(
            SELECTION_JSON_PATH,
            selection_info
        )

        optimal_bundle_source = selection_info[
            'optimal_inference_bundle_path'
        ]

        optimal_state_source = selection_info[
            'optimal_best_state_path'
        ]

        shutil.copy2(
            optimal_bundle_source,
            OPTIMAL_MODEL_BUNDLE_PATH
        )

        shutil.copy2(
            optimal_state_source,
            OPTIMAL_MODEL_STATE_PATH
        )

        optimal_text = (
            f'Optimal lambda: '
            f'{selection_info["optimal_lambda"]:.12g}\n'

            f'Selection rule: '
            f'{selection_info["selection_rule"]}\n'

            f'Best-accuracy lambda: '
            f'{selection_info["best_accuracy_lambda"]:.12g}\n'

            f'Minimum balanced NRMSE: '
            f'{selection_info["best_accuracy_mean"]:.8e}\n'

            f'Best-accuracy SE: '
            f'{selection_info["best_accuracy_se"]:.8e}\n'

            f'One-SE threshold: '
            f'{selection_info["accuracy_threshold"]:.8e}\n'

            f'Selected balanced NRMSE: '
            f'{selection_info["optimal_balanced_nrmse"]:.8e}\n'

            f'Selected physics score: '
            f'{selection_info["optimal_physics_score"]:.8e}\n'

            f'Eligible lambdas: '
            f'{selection_info["eligible_lambdas"]}\n'

            f'Optimal model bundle: '
            f'{OPTIMAL_MODEL_BUNDLE_PATH}\n'

            f'Optimal model state: '
            f'{OPTIMAL_MODEL_STATE_PATH}\n'
        )

        save_text(
            OPTIMAL_LAMBDA_TXT_PATH,
            optimal_text
        )

        # ---------------------------------------------------------------------
        # Step 9：绘制跨 lambda 对比图
        # ---------------------------------------------------------------------

        plot_lambda_sweep(
            summary_df,
            selection_info,
            RESULT_DIR
        )

        print(
            '\n'
            + '=' * 88
        )

        print(
            'Lambda sweep 完成'
        )

        print(
            '=' * 88
        )

        print(
            '最优 lambda: '
            f'{selection_info["optimal_lambda"]:.12g}'
        )

        print(
            f'最优模型包: {OPTIMAL_MODEL_BUNDLE_PATH}'
        )

        print(
            f'最优模型参数: {OPTIMAL_MODEL_STATE_PATH}'
        )

        print(
            f'汇总表: {SWEEP_SUMMARY_CSV}'
        )

        print(
            f'选择说明: {SELECTION_JSON_PATH}'
        )

        print(
            f'公共数据: {COMMON_DATA_PATH}'
        )

    except Exception:

        print(
            '\n程序运行失败，异常信息如下：'
        )

        traceback.print_exc()

        raise