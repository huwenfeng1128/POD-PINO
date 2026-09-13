# -*- coding: utf-8 -*-
"""
POD-DeepONet / POD-PINO 双模型前向预测与综合评价程序
=======================================================

用途
----
本程序用于第四章 4.2.2 节的前向预测对比，严格对应物理残差消融实验：

1. 同时加载两套模型：
   - no_pde_residual：不含 PDE 物理残差的 POD-DeepONet；
   - with_pde_residual：含归一化 PDE 物理残差的 POD-PINO。
2. 两套模型共享相同的：
   - 输入/输出标准化参数；
   - POD 基底与 POD 中心；
   - A、tau 网格；
   - 网络结构。
3. 对测试目录下每一组 data_*.csv 数据完成前向预测，并保存：
   - 完整真值场、两模型预测场、绝对误差场、相对误差场；
   - R1、R2 原始残差与归一化残差；
   - 每个样本的完整 CSV/NPZ/JSON 指标；
   - D1、D2 场域对比图；
   - R1、R2 残差对比图；
   - 固定 tau 截面曲线；
   - 固定 A 截面曲线；
   - 误差随 tau、A 的分布曲线。
4. 生成全测试集汇总：
   - sample_metrics.csv；
   - overall_metrics.csv / overall_metrics.json；
   - 各指标箱线图、CDF、散点图、改善率分布图；
   - 参数空间误差分布图；
   - 代表性样本自动选择清单。

归一化 PDE 残差
----------------
与权重选择程序和消融程序一致：

    U1 = tau * D1_tau
    U2 = tau * D2_tau

    R1 = dU1/dtau
         - D1_inf * dU1/dA
         - D2_inf * d2U1/dA2
         - D1_inf

    R2 = dU2/dtau
         - D1_inf * dU2/dA
         - D2_inf * d2U2/dA2
         - D1_inf * U1
         - 2 * D2_inf * dU1/dA
         - D2_inf

归一化残差点值：

    R1_norm_sq = R1^2 / (sum_i term_R1_i^2 + eps)
    R2_norm_sq = R2^2 / (sum_i term_R2_i^2 + eps)

注意
----
1. 本程序只进行前向预测，不执行参数反演。
2. 测试数据文件应包含：A, tau, nu, kappa, d_diffusion, D1_adj, D2_adj。
3. 默认读取消融程序输出目录中的 shared_preprocessing.pth 与两个 best_model.pth。
4. 若模型文件名不同，只需修改配置区路径。
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import random
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# =============================================================================
# 1. 配置区
# =============================================================================

# --- 消融实验结果目录 ---
ABLATION_RESULT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\xiaorongshiyan"
    r"\train_result_physics_ablation_2"
)

# 两个模型权重。
NO_PDE_MODEL_PATH = ABLATION_RESULT_DIR / "no_pde_residual" / "best_model.pth"
WITH_PDE_MODEL_PATH = ABLATION_RESULT_DIR / "with_pde_residual" / "best_model.pth"

# 消融程序保存的统一预处理文件。
# 程序兼容 shared_preprocessing.pth、common_metadata.pth、model_bundle.pth 等常见形式。
PREPROCESSING_PATH = ABLATION_RESULT_DIR / "shared_preprocessing.pth"

# 最优 lambda 选择信息，仅用于记录，不参与前向预测。
OPTIMAL_LAMBDA_SELECTION_PATH = ABLATION_RESULT_DIR / "optimal_lambda_selection_used.json"

# --- 前向预测测试数据目录 ---
# 每个文件是一组参数对应的完整 A-tau 张量积网格。
TEST_DATA_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA"
)

# 文件匹配模式。
TEST_FILE_GLOB = "data_*.csv"

# --- 结果输出目录 ---
OUTPUT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\forward_predict"
    r"\forward_prediction_comparison"
)

# 若只想测试部分文件，设置正整数；None 表示全部。
MAX_TEST_FILES: Optional[int] = None

# 固定抽样种子。仅用于文件抽样和代表样本选择。
SEED = 2026

# 网络结构，必须与训练程序一致。
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
DROPOUT_RATE = 0.1

# 计算设置。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
INFERENCE_BATCH_SIZE = 256
EPS = 1.0e-12

# 绘图设置。
FIG_DPI = 300
NUM_TAU_SLICES = 4
NUM_A_SLICES = 4
NUM_REPRESENTATIVE_SAMPLES = 6
SAVE_NPZ = True
SAVE_PER_SAMPLE_PLOTS = True
SAVE_PER_SAMPLE_CSV = True

# 代表样本选择策略：中位样本、困难样本、改善最大/最小、边界样本等。
REPRESENTATIVE_SELECTION = True

# 图中相对误差分母下限，避免真值接近 0 时产生巨大伪相对误差。
RELATIVE_ERROR_FLOOR_RATIO = 1.0e-3

# 数据文件必需列。
REQUIRED_COLUMNS = [
    "A",
    "tau",
    "nu",
    "kappa",
    "d_diffusion",
    "D1_adj",
    "D2_adj",
]

# 输出目录结构。
SAMPLE_OUTPUT_DIR = OUTPUT_DIR / "samples"
SUMMARY_FIG_DIR = OUTPUT_DIR / "summary_figures"
SUMMARY_DATA_DIR = OUTPUT_DIR / "summary_data"


# =============================================================================
# 2. 可复现性与通用工具
# =============================================================================


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def json_converter(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_converter)


def safe_torch_load(path: Path, map_location: torch.device = DEVICE) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_load_state_dict(path: Path, map_location: torch.device = DEVICE) -> Dict[str, torch.Tensor]:
    payload = safe_torch_load(path, map_location)
    if isinstance(payload, dict):
        for key in ["state_dict", "model_state_dict", "best_state_dict", "model"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    raise ValueError(f"无法从 {path} 中解析模型 state_dict。")


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def percent_improvement(baseline: float, physics: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < EPS:
        return float("nan")
    return float((baseline - physics) / abs(baseline) * 100.0)


def safe_rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.linalg.norm(y_true.ravel()))
    if denominator < EPS:
        return float("nan")
    return float(np.linalg.norm((y_pred - y_true).ravel()) / denominator)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    denominator = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    if denominator < EPS:
        return float("nan")
    numerator = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    return 1.0 - numerator / denominator


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# =============================================================================
# 3. 网络定义
# =============================================================================


class MLP(nn.Module):
    """参数输入到 POD 系数的映射网络。"""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout_rate: float,
    ) -> None:
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
    """参数 -> POD 系数 -> 标准化物理场。"""

    def __init__(
        self,
        branch_input_dim: int,
        hidden_units: int,
        num_hidden_layers: int,
        num_pod_modes: int,
        pod_basis: np.ndarray,
        y_mean_pod_scaled: np.ndarray,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.branch = MLP(
            input_dim=branch_input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            output_dim=num_pod_modes,
            dropout_rate=dropout_rate,
        )

        self.register_buffer(
            "pod_basis",
            torch.tensor(pod_basis, dtype=DTYPE),
        )
        self.register_buffer(
            "y_mean_pod_scaled",
            torch.tensor(y_mean_pod_scaled, dtype=DTYPE),
        )

    def forward(self, branch_x_scaled: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(branch_x_scaled)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


# =============================================================================
# 4. 数据结构
# =============================================================================


@dataclass
class SharedPreprocessing:
    branch_mean: np.ndarray
    branch_std: np.ndarray
    y_mean_scaler: np.ndarray
    y_std_scaler: np.ndarray
    y_mean_pod_scaled: np.ndarray
    pod_basis: np.ndarray
    actual_num_modes: int
    unified_a_grid: np.ndarray
    unified_tau_grid: np.ndarray


@dataclass
class TestSample:
    file_path: Path
    sample_name: str
    params: np.ndarray
    a_grid: np.ndarray
    tau_grid: np.ndarray
    d1_true: np.ndarray
    d2_true: np.ndarray


# =============================================================================
# 5. 预处理与模型加载
# =============================================================================


def _extract_first(payload: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"预处理文件缺少任一字段: {keys}")


def load_shared_preprocessing(path: Path) -> SharedPreprocessing:
    if not path.exists():
        alternatives = [
            ABLATION_RESULT_DIR / "common_metadata.pth",
            ABLATION_RESULT_DIR / "common_dataset_and_preprocess.npz",
            ABLATION_RESULT_DIR / "shared_data.pth",
            ABLATION_RESULT_DIR / "model_bundle.pth",
        ]
        found = next((p for p in alternatives if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"未找到预处理文件 {path}，也未找到候选文件: {alternatives}"
            )
        path = found

    if path.suffix.lower() == ".npz":
        npz = np.load(path, allow_pickle=True)
        payload = {key: npz[key] for key in npz.files}
    else:
        payload = safe_torch_load(path, DEVICE)

    if not isinstance(payload, dict):
        raise TypeError(f"预处理文件 {path} 内容不是字典。")

    # 某些程序将真正信息放在 shared、preprocessing、scalers 等子字典中。
    for wrapper_key in ["shared", "preprocessing", "scalers", "metadata", "bundle"]:
        if wrapper_key in payload and isinstance(payload[wrapper_key], dict):
            merged = dict(payload)
            merged.update(payload[wrapper_key])
            payload = merged

    branch_mean = to_numpy(_extract_first(payload, ["branch_mean", "x_mean"]))
    branch_std = to_numpy(_extract_first(payload, ["branch_std", "x_std"]))
    y_mean_scaler = to_numpy(_extract_first(payload, ["y_mean_scaler", "output_mean", "y_mean"]))
    y_std_scaler = to_numpy(_extract_first(payload, ["y_std_scaler", "output_std", "y_std"]))
    y_mean_pod_scaled = to_numpy(
        _extract_first(payload, ["y_mean_pod_scaled", "pod_mean", "snapshot_mean"])
    )
    pod_basis = to_numpy(_extract_first(payload, ["pod_basis", "basis", "phi"]))
    actual_num_modes = int(
        np.asarray(_extract_first(payload, ["actual_num_modes", "num_pod_modes", "n_modes"])).item()
    )
    unified_a_grid = to_numpy(_extract_first(payload, ["unified_a_grid", "a_grid", "A_grid"]))
    unified_tau_grid = to_numpy(
        _extract_first(payload, ["unified_tau_grid", "tau_grid", "Tau_grid"])
    )

    return SharedPreprocessing(
        branch_mean=np.asarray(branch_mean, dtype=np.float64),
        branch_std=np.asarray(branch_std, dtype=np.float64),
        y_mean_scaler=np.asarray(y_mean_scaler, dtype=np.float64),
        y_std_scaler=np.asarray(y_std_scaler, dtype=np.float64),
        y_mean_pod_scaled=np.asarray(y_mean_pod_scaled, dtype=np.float64),
        pod_basis=np.asarray(pod_basis, dtype=np.float64),
        actual_num_modes=actual_num_modes,
        unified_a_grid=np.asarray(unified_a_grid, dtype=np.float64),
        unified_tau_grid=np.asarray(unified_tau_grid, dtype=np.float64),
    )


def build_model(shared: SharedPreprocessing, model_path: Path) -> PODDeepONet:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = PODDeepONet(
        branch_input_dim=BRANCH_INPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_pod_modes=shared.actual_num_modes,
        pod_basis=shared.pod_basis,
        y_mean_pod_scaled=shared.y_mean_pod_scaled,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    state_dict = safe_load_state_dict(model_path, DEVICE)

    # 兼容旧代码将 buffer 保存为 Parameter 的 state_dict，不影响 key 名。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"加载模型 {model_path} 时缺少参数: {missing}")
    if unexpected:
        print(f"[警告] 模型 {model_path} 存在未使用参数: {unexpected}")

    model.eval()
    return model


# =============================================================================
# 6. 测试数据读取
# =============================================================================


def is_uniform_grid(arr: np.ndarray, tol: float = 1.0e-10) -> bool:
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return bool(np.allclose(diffs, diffs[0], atol=tol, rtol=tol))


def load_test_sample(file_path: Path, shared: SharedPreprocessing) -> TestSample:
    df = pd.read_csv(file_path, on_bad_lines="skip")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"文件缺少字段 {missing}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("删除缺失值后数据为空。")

    for column in ["nu", "kappa", "d_diffusion"]:
        if df[column].nunique(dropna=True) != 1:
            raise ValueError(f"同一文件内参数 {column} 不是常数。")

    if df.duplicated(subset=["tau", "A"]).any():
        raise ValueError("存在重复 (tau, A) 网格点。")

    a_grid = np.sort(df["A"].unique().astype(np.float64))
    tau_grid = np.sort(df["tau"].unique().astype(np.float64))

    if len(a_grid) < 3 or len(tau_grid) < 3:
        raise ValueError("A、tau 网格点数均需至少为 3。")
    if not is_uniform_grid(a_grid):
        raise ValueError("A 网格不是均匀网格。")
    if not is_uniform_grid(tau_grid):
        raise ValueError("tau 网格不是均匀网格。")

    # 前向模型输出固定统一网格，因此测试数据必须使用相同网格。
    if not np.allclose(a_grid, shared.unified_a_grid, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("测试文件 A 网格与训练统一网格不一致。")
    if not np.allclose(tau_grid, shared.unified_tau_grid, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("测试文件 tau 网格与训练统一网格不一致。")

    expected_len = len(a_grid) * len(tau_grid)
    df_sorted = df.sort_values(["tau", "A"]).reset_index(drop=True)
    if len(df_sorted) != expected_len:
        raise ValueError(f"网格不完整：期望 {expected_len} 行，实际 {len(df_sorted)} 行。")

    params = df_sorted[["nu", "kappa", "d_diffusion"]].iloc[0].to_numpy(dtype=np.float64)
    d1_true = df_sorted["D1_adj"].to_numpy(dtype=np.float64).reshape(len(tau_grid), len(a_grid))
    d2_true = df_sorted["D2_adj"].to_numpy(dtype=np.float64).reshape(len(tau_grid), len(a_grid))

    if not np.all(np.isfinite(params)):
        raise ValueError("参数包含 NaN 或 Inf。")
    if not np.all(np.isfinite(d1_true)) or not np.all(np.isfinite(d2_true)):
        raise ValueError("真值场包含 NaN 或 Inf。")

    return TestSample(
        file_path=file_path,
        sample_name=sanitize_name(file_path.stem),
        params=params,
        a_grid=a_grid,
        tau_grid=tau_grid,
        d1_true=d1_true,
        d2_true=d2_true,
    )


# =============================================================================
# 7. 前向预测
# =============================================================================


def scale_branch(params: np.ndarray, shared: SharedPreprocessing) -> torch.Tensor:
    std = np.asarray(shared.branch_std, dtype=np.float64).copy()
    std[np.abs(std) < 1.0e-10] = 1.0
    scaled = (params - shared.branch_mean) / std
    return torch.tensor(scaled, dtype=DTYPE, device=DEVICE)


def predict_batch(
    model: PODDeepONet,
    params_batch: np.ndarray,
    shared: SharedPreprocessing,
) -> np.ndarray:
    x_scaled = scale_branch(params_batch, shared)
    outputs: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(x_scaled), INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, len(x_scaled))
            y_scaled = model(x_scaled[start:end])
            y_physical = (
                y_scaled
                * torch.tensor(shared.y_std_scaler, dtype=DTYPE, device=DEVICE)
                + torch.tensor(shared.y_mean_scaler, dtype=DTYPE, device=DEVICE)
            )
            outputs.append(y_physical.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def split_prediction_fields(
    y_pred: np.ndarray,
    n_tau: int,
    n_a: int,
) -> Tuple[np.ndarray, np.ndarray]:
    field_len = n_tau * n_a
    if y_pred.shape[1] != 2 * field_len:
        raise ValueError(
            f"模型输出维数 {y_pred.shape[1]} 与 2*n_tau*n_a={2 * field_len} 不一致。"
        )
    d1 = y_pred[:, :field_len].reshape(-1, n_tau, n_a)
    d2 = y_pred[:, field_len:].reshape(-1, n_tau, n_a)
    return d1, d2


# =============================================================================
# 8. 物理残差与导数评价
# =============================================================================


def compute_residual_components(
    d1_field: np.ndarray,
    d2_field: np.ndarray,
    params: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    输入形状均为 [n_tau, n_a]，返回内部点 [n_tau-2, n_a-2]。
    """
    nu, kappa, d_diffusion = [float(value) for value in params]

    a = a_grid[None, :]
    tau = tau_grid[:, None]
    a_safe = np.maximum(a, 1.0e-9)

    d1_inf = nu * a - (kappa / 8.0) * a**3 + d_diffusion / a_safe
    d2_inf = np.full_like(d1_inf, d_diffusion)

    u1 = tau * d1_field
    u2 = tau * d2_field

    da = float(a_grid[1] - a_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])

    u1_in = u1[1:-1, 1:-1]
    d1_inf_in = np.broadcast_to(d1_inf[:, 1:-1], u1_in.shape)
    d2_inf_in = np.broadcast_to(d2_inf[:, 1:-1], u1_in.shape)

    du1_dtau = (u1[2:, 1:-1] - u1[:-2, 1:-1]) / (2.0 * dtau)
    du2_dtau = (u2[2:, 1:-1] - u2[:-2, 1:-1]) / (2.0 * dtau)

    du1_da = (u1[1:-1, 2:] - u1[1:-1, :-2]) / (2.0 * da)
    du2_da = (u2[1:-1, 2:] - u2[1:-1, :-2]) / (2.0 * da)

    d2u1_da2 = (
        u1[1:-1, 2:] - 2.0 * u1[1:-1, 1:-1] + u1[1:-1, :-2]
    ) / (da**2)
    d2u2_da2 = (
        u2[1:-1, 2:] - 2.0 * u2[1:-1, 1:-1] + u2[1:-1, :-2]
    ) / (da**2)

    # R1 分项。
    r1_t1 = du1_dtau
    r1_t2 = d1_inf_in * du1_da
    r1_t3 = d2_inf_in * d2u1_da2
    r1_t4 = d1_inf_in
    r1 = r1_t1 - r1_t2 - r1_t3 - r1_t4
    r1_scale_sq = r1_t1**2 + r1_t2**2 + r1_t3**2 + r1_t4**2

    # R2 分项。
    r2_t1 = du2_dtau
    r2_t2 = d1_inf_in * du2_da
    r2_t3 = d2_inf_in * d2u2_da2
    r2_t4 = d1_inf_in * u1_in
    r2_t5 = 2.0 * d2_inf_in * du1_da
    r2_t6 = d2_inf_in
    r2 = r2_t1 - r2_t2 - r2_t3 - r2_t4 - r2_t5 - r2_t6
    r2_scale_sq = (
        r2_t1**2 + r2_t2**2 + r2_t3**2 + r2_t4**2 + r2_t5**2 + r2_t6**2
    )

    r1_norm_sq = r1**2 / (r1_scale_sq + EPS)
    r2_norm_sq = r2**2 / (r2_scale_sq + EPS)

    return {
        "R1": r1,
        "R2": r2,
        "R1_scale_sq": r1_scale_sq,
        "R2_scale_sq": r2_scale_sq,
        "R1_norm_sq": r1_norm_sq,
        "R2_norm_sq": r2_norm_sq,
        "du1_dtau": du1_dtau,
        "du1_da": du1_da,
        "d2u1_da2": d2u1_da2,
        "du2_dtau": du2_dtau,
        "du2_da": du2_da,
        "d2u2_da2": d2u2_da2,
    }


def derivative_error_metrics(
    true_components: Dict[str, np.ndarray],
    pred_components: Dict[str, np.ndarray],
    prefix: str,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    derivative_keys = [
        "du1_dtau",
        "du1_da",
        "d2u1_da2",
        "du2_dtau",
        "du2_da",
        "d2u2_da2",
    ]

    rmse_values = []
    for key in derivative_keys:
        true_value = true_components[key]
        pred_value = pred_components[key]
        error = pred_value - true_value
        rmse = float(np.sqrt(np.mean(error**2)))
        metrics[f"{prefix}_{key}_rmse"] = rmse
        metrics[f"{prefix}_{key}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{prefix}_{key}_rel_l2"] = safe_rel_l2(true_value, pred_value)
        rmse_values.append(rmse)

    metrics[f"{prefix}_mean_derivative_rmse"] = float(np.mean(rmse_values))
    return metrics


# =============================================================================
# 9. 误差指标
# =============================================================================


def field_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    true_scale = float(np.std(y_true))
    if true_scale < EPS:
        true_scale = float(np.sqrt(np.mean(y_true**2)) + EPS)

    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_nrmse": float(np.sqrt(np.mean(error**2)) / true_scale),
        f"{prefix}_mae": float(np.mean(abs_error)),
        f"{prefix}_max_abs": float(np.max(abs_error)),
        f"{prefix}_median_abs": float(np.median(abs_error)),
        f"{prefix}_p95_abs": float(np.percentile(abs_error, 95.0)),
        f"{prefix}_rel_l2": safe_rel_l2(y_true, y_pred),
        f"{prefix}_r2": safe_r2(y_true, y_pred),
        f"{prefix}_corr": safe_corrcoef(y_true, y_pred),
        f"{prefix}_bias": float(np.mean(error)),
    }


def residual_metrics(components: Dict[str, np.ndarray], prefix: str) -> Dict[str, float]:
    r1 = components["R1"]
    r2 = components["R2"]
    r1_norm_sq = components["R1_norm_sq"]
    r2_norm_sq = components["R2_norm_sq"]

    return {
        f"{prefix}_r1_mse": float(np.mean(r1**2)),
        f"{prefix}_r2_mse": float(np.mean(r2**2)),
        f"{prefix}_pde_total_mse": float(np.mean(r1**2) + np.mean(r2**2)),
        f"{prefix}_r1_rmse": float(np.sqrt(np.mean(r1**2))),
        f"{prefix}_r2_rmse": float(np.sqrt(np.mean(r2**2))),
        f"{prefix}_r1_normalized_mse": float(np.mean(r1_norm_sq)),
        f"{prefix}_r2_normalized_mse": float(np.mean(r2_norm_sq)),
        f"{prefix}_normalized_pde_loss": float(np.mean(r1_norm_sq) + np.mean(r2_norm_sq)),
        f"{prefix}_r1_normalized_rmse": float(np.sqrt(np.mean(r1_norm_sq))),
        f"{prefix}_r2_normalized_rmse": float(np.sqrt(np.mean(r2_norm_sq))),
        f"{prefix}_r1_max_abs": float(np.max(np.abs(r1))),
        f"{prefix}_r2_max_abs": float(np.max(np.abs(r2))),
    }


def calculate_sample_metrics(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, np.ndarray]]]:
    true_comp = compute_residual_components(
        sample.d1_true, sample.d2_true, sample.params, sample.a_grid, sample.tau_grid
    )
    no_comp = compute_residual_components(
        d1_no, d2_no, sample.params, sample.a_grid, sample.tau_grid
    )
    phys_comp = compute_residual_components(
        d1_phys, d2_phys, sample.params, sample.a_grid, sample.tau_grid
    )

    metrics: Dict[str, float] = {
        "sample_name": sample.sample_name,
        "source_file": str(sample.file_path),
        "nu": float(sample.params[0]),
        "kappa": float(sample.params[1]),
        "d_diffusion": float(sample.params[2]),
    }

    metrics.update(field_metrics(sample.d1_true, d1_no, "no_pde_d1"))
    metrics.update(field_metrics(sample.d2_true, d2_no, "no_pde_d2"))
    metrics.update(field_metrics(sample.d1_true, d1_phys, "with_pde_d1"))
    metrics.update(field_metrics(sample.d2_true, d2_phys, "with_pde_d2"))

    metrics.update(residual_metrics(no_comp, "no_pde"))
    metrics.update(residual_metrics(phys_comp, "with_pde"))

    metrics.update(derivative_error_metrics(true_comp, no_comp, "no_pde"))
    metrics.update(derivative_error_metrics(true_comp, phys_comp, "with_pde"))

    # D2 非负性。
    metrics["no_pde_d2_negative_fraction"] = float(np.mean(d2_no < 0.0))
    metrics["with_pde_d2_negative_fraction"] = float(np.mean(d2_phys < 0.0))
    metrics["no_pde_d2_negative_mse"] = float(np.mean(np.minimum(d2_no, 0.0) ** 2))
    metrics["with_pde_d2_negative_mse"] = float(np.mean(np.minimum(d2_phys, 0.0) ** 2))

    # 改善率。
    improvement_pairs = {
        "d1_rmse_improvement_percent": ("no_pde_d1_rmse", "with_pde_d1_rmse"),
        "d2_rmse_improvement_percent": ("no_pde_d2_rmse", "with_pde_d2_rmse"),
        "d1_rel_l2_improvement_percent": ("no_pde_d1_rel_l2", "with_pde_d1_rel_l2"),
        "d2_rel_l2_improvement_percent": ("no_pde_d2_rel_l2", "with_pde_d2_rel_l2"),
        "normalized_pde_improvement_percent": (
            "no_pde_normalized_pde_loss",
            "with_pde_normalized_pde_loss",
        ),
        "raw_pde_improvement_percent": ("no_pde_pde_total_mse", "with_pde_pde_total_mse"),
        "derivative_rmse_improvement_percent": (
            "no_pde_mean_derivative_rmse",
            "with_pde_mean_derivative_rmse",
        ),
    }
    for output_key, (baseline_key, physics_key) in improvement_pairs.items():
        metrics[output_key] = percent_improvement(metrics[baseline_key], metrics[physics_key])

    # 综合场误差，用于代表样本排序。
    metrics["no_pde_balanced_nrmse"] = 0.8 * metrics["no_pde_d1_nrmse"] + 0.2 * metrics[
        "no_pde_d2_nrmse"
    ]
    metrics["with_pde_balanced_nrmse"] = 0.8 * metrics["with_pde_d1_nrmse"] + 0.2 * metrics[
        "with_pde_d2_nrmse"
    ]
    metrics["balanced_nrmse_improvement_percent"] = percent_improvement(
        metrics["no_pde_balanced_nrmse"], metrics["with_pde_balanced_nrmse"]
    )

    components = {
        "true": true_comp,
        "no_pde": no_comp,
        "with_pde": phys_comp,
    }
    return metrics, components


# =============================================================================
# 10. 每个样本的数据保存
# =============================================================================


def build_field_dataframe(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
) -> pd.DataFrame:
    aa, tt = np.meshgrid(sample.a_grid, sample.tau_grid)

    d1_floor = max(float(np.max(np.abs(sample.d1_true))) * RELATIVE_ERROR_FLOOR_RATIO, EPS)
    d2_floor = max(float(np.max(np.abs(sample.d2_true))) * RELATIVE_ERROR_FLOOR_RATIO, EPS)

    return pd.DataFrame(
        {
            "A": aa.ravel(),
            "tau": tt.ravel(),
            "nu": float(sample.params[0]),
            "kappa": float(sample.params[1]),
            "d_diffusion": float(sample.params[2]),
            "D1_true": sample.d1_true.ravel(),
            "D1_no_pde": d1_no.ravel(),
            "D1_with_pde": d1_phys.ravel(),
            "D1_error_no_pde": (d1_no - sample.d1_true).ravel(),
            "D1_error_with_pde": (d1_phys - sample.d1_true).ravel(),
            "D1_abs_error_no_pde": np.abs(d1_no - sample.d1_true).ravel(),
            "D1_abs_error_with_pde": np.abs(d1_phys - sample.d1_true).ravel(),
            "D1_relative_error_no_pde": (
                np.abs(d1_no - sample.d1_true) / np.maximum(np.abs(sample.d1_true), d1_floor)
            ).ravel(),
            "D1_relative_error_with_pde": (
                np.abs(d1_phys - sample.d1_true) / np.maximum(np.abs(sample.d1_true), d1_floor)
            ).ravel(),
            "D2_true": sample.d2_true.ravel(),
            "D2_no_pde": d2_no.ravel(),
            "D2_with_pde": d2_phys.ravel(),
            "D2_error_no_pde": (d2_no - sample.d2_true).ravel(),
            "D2_error_with_pde": (d2_phys - sample.d2_true).ravel(),
            "D2_abs_error_no_pde": np.abs(d2_no - sample.d2_true).ravel(),
            "D2_abs_error_with_pde": np.abs(d2_phys - sample.d2_true).ravel(),
            "D2_relative_error_no_pde": (
                np.abs(d2_no - sample.d2_true) / np.maximum(np.abs(sample.d2_true), d2_floor)
            ).ravel(),
            "D2_relative_error_with_pde": (
                np.abs(d2_phys - sample.d2_true) / np.maximum(np.abs(sample.d2_true), d2_floor)
            ).ravel(),
        }
    )


def build_residual_dataframe(
    sample: TestSample,
    components: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    a_in = sample.a_grid[1:-1]
    tau_in = sample.tau_grid[1:-1]
    aa, tt = np.meshgrid(a_in, tau_in)

    data: Dict[str, np.ndarray] = {
        "A": aa.ravel(),
        "tau": tt.ravel(),
        "nu": np.full(aa.size, sample.params[0]),
        "kappa": np.full(aa.size, sample.params[1]),
        "d_diffusion": np.full(aa.size, sample.params[2]),
    }

    for group_name, comp in components.items():
        for key in [
            "R1",
            "R2",
            "R1_scale_sq",
            "R2_scale_sq",
            "R1_norm_sq",
            "R2_norm_sq",
            "du1_dtau",
            "du1_da",
            "d2u1_da2",
            "du2_dtau",
            "du2_da",
            "d2u2_da2",
        ]:
            data[f"{group_name}_{key}"] = comp[key].ravel()

    return pd.DataFrame(data)


# =============================================================================
# 11. 每个样本的绘图
# =============================================================================


def add_colorbar(fig: plt.Figure, ax: plt.Axes, image: Any) -> None:
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_field_comparison(
    sample: TestSample,
    true_field: np.ndarray,
    no_field: np.ndarray,
    phys_field: np.ndarray,
    field_name: str,
    save_path: Path,
) -> None:
    no_abs = np.abs(no_field - true_field)
    phys_abs = np.abs(phys_field - true_field)
    no_signed = no_field - true_field
    phys_signed = phys_field - true_field

    extent = [sample.a_grid.min(), sample.a_grid.max(), sample.tau_grid.min(), sample.tau_grid.max()]
    field_min = min(true_field.min(), no_field.min(), phys_field.min())
    field_max = max(true_field.max(), no_field.max(), phys_field.max())
    abs_max = max(no_abs.max(), phys_abs.max(), EPS)
    signed_max = max(np.abs(no_signed).max(), np.abs(phys_signed).max(), EPS)

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))

    top_data = [true_field, no_field, phys_field]
    top_titles = [
        f"{field_name} reference",
        f"{field_name} POD-DeepONet",
        f"{field_name} POD-PINO",
    ]
    for col, (data, title) in enumerate(zip(top_data, top_titles)):
        image = axes[0, col].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=field_min,
            vmax=field_max,
            cmap="viridis",
        )
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("A")
        axes[0, col].set_ylabel(r"$\tau$")
        add_colorbar(fig, axes[0, col], image)

    diff_image = axes[0, 3].imshow(
        phys_abs - no_abs,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-abs_max,
        vmax=abs_max,
        cmap="coolwarm",
    )
    axes[0, 3].set_title("Absolute-error change\n(POD-PINO minus POD-DeepONet)")
    axes[0, 3].set_xlabel("A")
    axes[0, 3].set_ylabel(r"$\tau$")
    add_colorbar(fig, axes[0, 3], diff_image)

    bottom_data = [no_abs, phys_abs, no_signed, phys_signed]
    bottom_titles = [
        "Absolute error: POD-DeepONet",
        "Absolute error: POD-PINO",
        "Signed error: POD-DeepONet",
        "Signed error: POD-PINO",
    ]
    for col, (data, title) in enumerate(zip(bottom_data, bottom_titles)):
        signed = col >= 2
        image = axes[1, col].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=-signed_max if signed else 0.0,
            vmax=signed_max if signed else abs_max,
            cmap="coolwarm" if signed else "magma",
        )
        axes[1, col].set_title(title)
        axes[1, col].set_xlabel("A")
        axes[1, col].set_ylabel(r"$\tau$")
        add_colorbar(fig, axes[1, col], image)

    no_rmse = np.sqrt(np.mean((no_field - true_field) ** 2))
    phys_rmse = np.sqrt(np.mean((phys_field - true_field) ** 2))
    improvement = percent_improvement(no_rmse, phys_rmse)

    fig.suptitle(
        f"{field_name} forward prediction | "
        f"nu={sample.params[0]:.5g}, kappa={sample.params[1]:.5g}, d={sample.params[2]:.5g}\n"
        f"RMSE: POD-DeepONet={no_rmse:.4e}, POD-PINO={phys_rmse:.4e}, "
        f"improvement={improvement:.2f}%",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residual_comparison(
    sample: TestSample,
    components: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    extent = [
        sample.a_grid[1],
        sample.a_grid[-2],
        sample.tau_grid[1],
        sample.tau_grid[-2],
    ]

    no_comp = components["no_pde"]
    phys_comp = components["with_pde"]

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))

    raw_maps = [
        np.abs(no_comp["R1"]),
        np.abs(phys_comp["R1"]),
        np.abs(no_comp["R2"]),
        np.abs(phys_comp["R2"]),
    ]
    raw_titles = [
        r"$|R^{(1)}|$: POD-DeepONet",
        r"$|R^{(1)}|$: POD-PINO",
        r"$|R^{(2)}|$: POD-DeepONet",
        r"$|R^{(2)}|$: POD-PINO",
    ]
    r1_max = max(raw_maps[0].max(), raw_maps[1].max(), EPS)
    r2_max = max(raw_maps[2].max(), raw_maps[3].max(), EPS)

    for col, (data, title) in enumerate(zip(raw_maps, raw_titles)):
        vmax = r1_max if col < 2 else r2_max
        image = axes[0, col].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("A")
        axes[0, col].set_ylabel(r"$\tau$")
        add_colorbar(fig, axes[0, col], image)

    norm_maps = [
        np.sqrt(no_comp["R1_norm_sq"]),
        np.sqrt(phys_comp["R1_norm_sq"]),
        np.sqrt(no_comp["R2_norm_sq"]),
        np.sqrt(phys_comp["R2_norm_sq"]),
    ]
    norm_titles = [
        r"Normalized $|R^{(1)}|$: POD-DeepONet",
        r"Normalized $|R^{(1)}|$: POD-PINO",
        r"Normalized $|R^{(2)}|$: POD-DeepONet",
        r"Normalized $|R^{(2)}|$: POD-PINO",
    ]
    nr1_max = max(norm_maps[0].max(), norm_maps[1].max(), EPS)
    nr2_max = max(norm_maps[2].max(), norm_maps[3].max(), EPS)

    for col, (data, title) in enumerate(zip(norm_maps, norm_titles)):
        vmax = nr1_max if col < 2 else nr2_max
        image = axes[1, col].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )
        axes[1, col].set_title(title)
        axes[1, col].set_xlabel("A")
        axes[1, col].set_ylabel(r"$\tau$")
        add_colorbar(fig, axes[1, col], image)

    no_norm = np.mean(no_comp["R1_norm_sq"]) + np.mean(no_comp["R2_norm_sq"])
    phys_norm = np.mean(phys_comp["R1_norm_sq"]) + np.mean(phys_comp["R2_norm_sq"])
    improvement = percent_improvement(no_norm, phys_norm)

    fig.suptitle(
        f"PDE residual comparison | "
        f"nu={sample.params[0]:.5g}, kappa={sample.params[1]:.5g}, d={sample.params[2]:.5g}\n"
        f"Normalized PDE loss: POD-DeepONet={no_norm:.4e}, POD-PINO={phys_norm:.4e}, "
        f"improvement={improvement:.2f}%",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def choose_even_indices(length: int, number: int) -> np.ndarray:
    number = min(max(1, number), length)
    return np.unique(np.linspace(0, length - 1, number, dtype=int))


def plot_tau_slices(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
    save_path: Path,
) -> None:
    tau_indices = choose_even_indices(len(sample.tau_grid), NUM_TAU_SLICES)
    fig, axes = plt.subplots(len(tau_indices), 2, figsize=(15, 4.2 * len(tau_indices)))
    if len(tau_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, tau_idx in enumerate(tau_indices):
        tau_value = sample.tau_grid[tau_idx]

        axes[row, 0].plot(sample.a_grid, sample.d1_true[tau_idx], linewidth=2.2, label="Reference")
        axes[row, 0].plot(sample.a_grid, d1_no[tau_idx], linestyle="--", linewidth=1.8, label="POD-DeepONet")
        axes[row, 0].plot(sample.a_grid, d1_phys[tau_idx], linestyle=":", linewidth=2.2, label="POD-PINO")
        axes[row, 0].set_title(rf"$D_\tau^{{(1)}}$ at $\tau={tau_value:.4g}$")
        axes[row, 0].set_xlabel("A")
        axes[row, 0].set_ylabel(r"$D_\tau^{(1)}$")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].legend()

        axes[row, 1].plot(sample.a_grid, sample.d2_true[tau_idx], linewidth=2.2, label="Reference")
        axes[row, 1].plot(sample.a_grid, d2_no[tau_idx], linestyle="--", linewidth=1.8, label="POD-DeepONet")
        axes[row, 1].plot(sample.a_grid, d2_phys[tau_idx], linestyle=":", linewidth=2.2, label="POD-PINO")
        axes[row, 1].set_title(rf"$D_\tau^{{(2)}}$ at $\tau={tau_value:.4g}$")
        axes[row, 1].set_xlabel("A")
        axes[row, 1].set_ylabel(r"$D_\tau^{(2)}$")
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].legend()

    fig.suptitle(
        f"Fixed-tau slices | nu={sample.params[0]:.5g}, "
        f"kappa={sample.params[1]:.5g}, d={sample.params[2]:.5g}",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_a_slices(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
    save_path: Path,
) -> None:
    a_indices = choose_even_indices(len(sample.a_grid), NUM_A_SLICES)
    fig, axes = plt.subplots(len(a_indices), 2, figsize=(15, 4.2 * len(a_indices)))
    if len(a_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, a_idx in enumerate(a_indices):
        a_value = sample.a_grid[a_idx]

        axes[row, 0].plot(sample.tau_grid, sample.d1_true[:, a_idx], linewidth=2.2, label="Reference")
        axes[row, 0].plot(sample.tau_grid, d1_no[:, a_idx], linestyle="--", linewidth=1.8, label="POD-DeepONet")
        axes[row, 0].plot(sample.tau_grid, d1_phys[:, a_idx], linestyle=":", linewidth=2.2, label="POD-PINO")
        axes[row, 0].set_title(rf"$D_\tau^{{(1)}}$ at $A={a_value:.4g}$")
        axes[row, 0].set_xlabel(r"$\tau$")
        axes[row, 0].set_ylabel(r"$D_\tau^{(1)}$")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].legend()

        axes[row, 1].plot(sample.tau_grid, sample.d2_true[:, a_idx], linewidth=2.2, label="Reference")
        axes[row, 1].plot(sample.tau_grid, d2_no[:, a_idx], linestyle="--", linewidth=1.8, label="POD-DeepONet")
        axes[row, 1].plot(sample.tau_grid, d2_phys[:, a_idx], linestyle=":", linewidth=2.2, label="POD-PINO")
        axes[row, 1].set_title(rf"$D_\tau^{{(2)}}$ at $A={a_value:.4g}$")
        axes[row, 1].set_xlabel(r"$\tau$")
        axes[row, 1].set_ylabel(r"$D_\tau^{(2)}$")
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].legend()

    fig.suptitle(
        f"Fixed-A slices | nu={sample.params[0]:.5g}, "
        f"kappa={sample.params[1]:.5g}, d={sample.params[2]:.5g}",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_error_profiles(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
    save_path: Path,
) -> None:
    errors = {
        "d1_no": np.abs(d1_no - sample.d1_true),
        "d1_phys": np.abs(d1_phys - sample.d1_true),
        "d2_no": np.abs(d2_no - sample.d2_true),
        "d2_phys": np.abs(d2_phys - sample.d2_true),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(sample.tau_grid, np.mean(errors["d1_no"], axis=1), label="POD-DeepONet")
    axes[0, 0].plot(sample.tau_grid, np.mean(errors["d1_phys"], axis=1), label="POD-PINO")
    axes[0, 0].set_title(r"Mean $D_\tau^{(1)}$ absolute error versus $\tau$")
    axes[0, 0].set_xlabel(r"$\tau$")
    axes[0, 0].set_ylabel("Mean absolute error")

    axes[0, 1].plot(sample.a_grid, np.mean(errors["d1_no"], axis=0), label="POD-DeepONet")
    axes[0, 1].plot(sample.a_grid, np.mean(errors["d1_phys"], axis=0), label="POD-PINO")
    axes[0, 1].set_title(r"Mean $D_\tau^{(1)}$ absolute error versus $A$")
    axes[0, 1].set_xlabel("A")
    axes[0, 1].set_ylabel("Mean absolute error")

    axes[1, 0].plot(sample.tau_grid, np.mean(errors["d2_no"], axis=1), label="POD-DeepONet")
    axes[1, 0].plot(sample.tau_grid, np.mean(errors["d2_phys"], axis=1), label="POD-PINO")
    axes[1, 0].set_title(r"Mean $D_\tau^{(2)}$ absolute error versus $\tau$")
    axes[1, 0].set_xlabel(r"$\tau$")
    axes[1, 0].set_ylabel("Mean absolute error")

    axes[1, 1].plot(sample.a_grid, np.mean(errors["d2_no"], axis=0), label="POD-DeepONet")
    axes[1, 1].plot(sample.a_grid, np.mean(errors["d2_phys"], axis=0), label="POD-PINO")
    axes[1, 1].set_title(r"Mean $D_\tau^{(2)}$ absolute error versus $A$")
    axes[1, 1].set_xlabel("A")
    axes[1, 1].set_ylabel("Mean absolute error")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Error profiles | nu={sample.params[0]:.5g}, "
        f"kappa={sample.params[1]:.5g}, d={sample.params[2]:.5g}",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 12. 汇总图
# =============================================================================


def plot_metric_boxplots(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    pairs = [
        ("no_pde_d1_rmse", "with_pde_d1_rmse", r"$D_\tau^{(1)}$ RMSE"),
        ("no_pde_d2_rmse", "with_pde_d2_rmse", r"$D_\tau^{(2)}$ RMSE"),
        ("no_pde_balanced_nrmse", "with_pde_balanced_nrmse", "Balanced NRMSE"),
        ("no_pde_normalized_pde_loss", "with_pde_normalized_pde_loss", "Normalized PDE loss"),
        ("no_pde_pde_total_mse", "with_pde_pde_total_mse", "Raw PDE total MSE"),
        ("no_pde_mean_derivative_rmse", "with_pde_mean_derivative_rmse", "Mean derivative RMSE"),
    ]

    for ax, (base_key, phys_key, title) in zip(axes.ravel(), pairs):
        values = [summary_df[base_key].dropna().values, summary_df[phys_key].dropna().values]
        ax.boxplot(values, labels=["POD-DeepONet", "POD-PINO"], showmeans=True)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.3)

    fig.suptitle("Forward-prediction metric distributions", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return values, values
    probabilities = np.arange(1, len(values) + 1) / len(values)
    return values, probabilities


def plot_metric_cdfs(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [
        ("no_pde_d1_rel_l2", "with_pde_d1_rel_l2", r"$D_\tau^{(1)}$ relative $L_2$ error"),
        ("no_pde_d2_rel_l2", "with_pde_d2_rel_l2", r"$D_\tau^{(2)}$ relative $L_2$ error"),
        ("no_pde_balanced_nrmse", "with_pde_balanced_nrmse", "Balanced NRMSE"),
        ("no_pde_normalized_pde_loss", "with_pde_normalized_pde_loss", "Normalized PDE loss"),
    ]

    for ax, (base_key, phys_key, title) in zip(axes.ravel(), pairs):
        x_base, y_base = empirical_cdf(summary_df[base_key].to_numpy(dtype=float))
        x_phys, y_phys = empirical_cdf(summary_df[phys_key].to_numpy(dtype=float))
        ax.plot(x_base, y_base, linewidth=2, label="POD-DeepONet")
        ax.plot(x_phys, y_phys, linewidth=2, label="POD-PINO")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Metric value")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Empirical cumulative distributions", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_distributions(summary_df: pd.DataFrame, save_path: Path) -> None:
    keys = [
        "d1_rmse_improvement_percent",
        "d2_rmse_improvement_percent",
        "balanced_nrmse_improvement_percent",
        "normalized_pde_improvement_percent",
        "raw_pde_improvement_percent",
        "derivative_rmse_improvement_percent",
    ]
    labels = [
        "D1 RMSE",
        "D2 RMSE",
        "Balanced NRMSE",
        "Normalized PDE",
        "Raw PDE",
        "Derivative RMSE",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, key, label in zip(axes.ravel(), keys, labels):
        values = summary_df[key].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        ax.hist(values, bins=30, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1.5)
        if len(values) > 0:
            ax.axvline(np.median(values), linestyle=":", linewidth=2, label=f"Median={np.median(values):.2f}%")
            ax.legend()
        ax.set_title(f"{label} improvement")
        ax.set_xlabel("Improvement (%)")
        ax.set_ylabel("Sample count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sample-wise improvement distributions", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_error_scatter(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    pairs = [
        ("no_pde_d1_rmse", "with_pde_d1_rmse", r"$D_\tau^{(1)}$ RMSE"),
        ("no_pde_d2_rmse", "with_pde_d2_rmse", r"$D_\tau^{(2)}$ RMSE"),
        ("no_pde_balanced_nrmse", "with_pde_balanced_nrmse", "Balanced NRMSE"),
        ("no_pde_normalized_pde_loss", "with_pde_normalized_pde_loss", "Normalized PDE loss"),
    ]

    for ax, (base_key, phys_key, title) in zip(axes.ravel(), pairs):
        x = summary_df[base_key].to_numpy(dtype=float)
        y = summary_df[phys_key].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        ax.scatter(x, y, alpha=0.65, s=25)
        if len(x) > 0:
            lower = min(np.min(x), np.min(y))
            upper = max(np.max(x), np.max(y))
            ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5, label="Equal performance")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("POD-DeepONet")
        ax.set_ylabel("POD-PINO")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Sample-wise metric comparison", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_space_errors(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    parameters = ["nu", "kappa", "d_diffusion"]
    metrics = [
        ("balanced_nrmse_improvement_percent", "Balanced NRMSE improvement (%)"),
        ("normalized_pde_improvement_percent", "Normalized PDE improvement (%)"),
    ]

    for row, (metric_key, metric_label) in enumerate(metrics):
        for col, parameter in enumerate(parameters):
            ax = axes[row, col]
            x = summary_df[parameter].to_numpy(dtype=float)
            y = summary_df[metric_key].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.65, s=28)
            ax.axhline(0.0, linestyle="--", linewidth=1.2)
            ax.set_xlabel(parameter)
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            corr = safe_corrcoef(x, y)
            ax.set_title(f"{metric_label} vs {parameter}\nPearson r={corr:.3f}")

    fig.suptitle("Performance changes across parameter space", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_error_profiles(
    samples: List[TestSample],
    all_predictions: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    d1_true = np.stack([sample.d1_true for sample in samples], axis=0)
    d2_true = np.stack([sample.d2_true for sample in samples], axis=0)
    d1_no = all_predictions["d1_no"]
    d2_no = all_predictions["d2_no"]
    d1_phys = all_predictions["d1_phys"]
    d2_phys = all_predictions["d2_phys"]

    a_grid = samples[0].a_grid
    tau_grid = samples[0].tau_grid

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(tau_grid, np.mean(np.abs(d1_no - d1_true), axis=(0, 2)), label="POD-DeepONet")
    axes[0, 0].plot(tau_grid, np.mean(np.abs(d1_phys - d1_true), axis=(0, 2)), label="POD-PINO")
    axes[0, 0].set_title(r"Dataset-mean $D_\tau^{(1)}$ error versus $\tau$")
    axes[0, 0].set_xlabel(r"$\tau$")
    axes[0, 0].set_ylabel("Mean absolute error")

    axes[0, 1].plot(a_grid, np.mean(np.abs(d1_no - d1_true), axis=(0, 1)), label="POD-DeepONet")
    axes[0, 1].plot(a_grid, np.mean(np.abs(d1_phys - d1_true), axis=(0, 1)), label="POD-PINO")
    axes[0, 1].set_title(r"Dataset-mean $D_\tau^{(1)}$ error versus $A$")
    axes[0, 1].set_xlabel("A")
    axes[0, 1].set_ylabel("Mean absolute error")

    axes[1, 0].plot(tau_grid, np.mean(np.abs(d2_no - d2_true), axis=(0, 2)), label="POD-DeepONet")
    axes[1, 0].plot(tau_grid, np.mean(np.abs(d2_phys - d2_true), axis=(0, 2)), label="POD-PINO")
    axes[1, 0].set_title(r"Dataset-mean $D_\tau^{(2)}$ error versus $\tau$")
    axes[1, 0].set_xlabel(r"$\tau$")
    axes[1, 0].set_ylabel("Mean absolute error")

    axes[1, 1].plot(a_grid, np.mean(np.abs(d2_no - d2_true), axis=(0, 1)), label="POD-DeepONet")
    axes[1, 1].plot(a_grid, np.mean(np.abs(d2_phys - d2_true), axis=(0, 1)), label="POD-PINO")
    axes[1, 1].set_title(r"Dataset-mean $D_\tau^{(2)}$ error versus $A$")
    axes[1, 1].set_xlabel("A")
    axes[1, 1].set_ylabel("Mean absolute error")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Dataset-level error profiles", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 13. 汇总统计与代表样本
# =============================================================================


def aggregate_numeric_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
    rows = []
    for column in numeric_columns:
        values = summary_df[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append(
            {
                "metric": column,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    return pd.DataFrame(rows)


def build_method_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_suffixes = [
        "d1_rmse",
        "d2_rmse",
        "d1_rel_l2",
        "d2_rel_l2",
        "balanced_nrmse",
        "normalized_pde_loss",
        "pde_total_mse",
        "mean_derivative_rmse",
        "d2_negative_fraction",
    ]

    rows = []
    for method_prefix, method_name in [
        ("no_pde", "POD-DeepONet"),
        ("with_pde", "POD-PINO"),
    ]:
        row: Dict[str, Any] = {"method": method_name}
        for suffix in metric_suffixes:
            key = f"{method_prefix}_{suffix}"
            if key not in summary_df.columns:
                continue
            values = summary_df[key].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            row[f"{suffix}_mean"] = float(np.mean(values))
            row[f"{suffix}_median"] = float(np.median(values))
            row[f"{suffix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_representative_samples(summary_df: pd.DataFrame) -> pd.DataFrame:
    selected_records = []
    used_indices = set()

    def add_index(index: int, reason: str) -> None:
        if index in used_indices:
            return
        used_indices.add(index)
        record = summary_df.loc[index].to_dict()
        record["selection_reason"] = reason
        selected_records.append(record)

    balanced = summary_df["no_pde_balanced_nrmse"].to_numpy(dtype=float)
    finite_indices = np.where(np.isfinite(balanced))[0]
    if len(finite_indices) > 0:
        median_value = np.nanmedian(balanced)
        median_index = int(np.nanargmin(np.abs(balanced - median_value)))
        add_index(median_index, "POD-DeepONet balanced NRMSE near dataset median")
        add_index(int(np.nanargmax(balanced)), "Largest POD-DeepONet balanced NRMSE")

    improvement = summary_df["balanced_nrmse_improvement_percent"].to_numpy(dtype=float)
    if np.any(np.isfinite(improvement)):
        add_index(int(np.nanargmax(improvement)), "Largest balanced-NRMSE improvement")
        add_index(int(np.nanargmin(improvement)), "Largest balanced-NRMSE degradation")

    pde_improvement = summary_df["normalized_pde_improvement_percent"].to_numpy(dtype=float)
    if np.any(np.isfinite(pde_improvement)):
        add_index(int(np.nanargmax(pde_improvement)), "Largest normalized-PDE improvement")

    # 参数空间边界样本：到参数中心的标准化距离最大。
    params = summary_df[["nu", "kappa", "d_diffusion"]].to_numpy(dtype=float)
    center = np.nanmean(params, axis=0)
    scale = np.nanstd(params, axis=0)
    scale[scale < EPS] = 1.0
    distance = np.sqrt(np.sum(((params - center) / scale) ** 2, axis=1))
    add_index(int(np.nanargmax(distance)), "Parameter-space boundary sample")

    return pd.DataFrame(selected_records).head(NUM_REPRESENTATIVE_SAMPLES)


# =============================================================================
# 14. 单样本处理
# =============================================================================


def process_single_sample(
    sample: TestSample,
    d1_no: np.ndarray,
    d2_no: np.ndarray,
    d1_phys: np.ndarray,
    d2_phys: np.ndarray,
) -> Dict[str, float]:
    sample_dir = SAMPLE_OUTPUT_DIR / sample.sample_name
    safe_mkdir(sample_dir)

    metrics, components = calculate_sample_metrics(
        sample, d1_no, d2_no, d1_phys, d2_phys
    )

    # 保存完整场数据。
    if SAVE_PER_SAMPLE_CSV:
        field_df = build_field_dataframe(sample, d1_no, d2_no, d1_phys, d2_phys)
        field_df.to_csv(sample_dir / "field_predictions_and_errors.csv", index=False)

        residual_df = build_residual_dataframe(sample, components)
        residual_df.to_csv(sample_dir / "pde_residuals_and_derivatives.csv", index=False)

    pd.DataFrame([metrics]).to_csv(sample_dir / "sample_metrics.csv", index=False)
    save_json(sample_dir / "sample_metrics.json", metrics)

    if SAVE_NPZ:
        np.savez_compressed(
            sample_dir / "complete_forward_prediction.npz",
            A=sample.a_grid,
            tau=sample.tau_grid,
            params=sample.params,
            D1_true=sample.d1_true,
            D2_true=sample.d2_true,
            D1_no_pde=d1_no,
            D2_no_pde=d2_no,
            D1_with_pde=d1_phys,
            D2_with_pde=d2_phys,
            **{
                f"{group}_{key}": value
                for group, comp in components.items()
                for key, value in comp.items()
            },
        )

    if SAVE_PER_SAMPLE_PLOTS:
        plot_field_comparison(
            sample,
            sample.d1_true,
            d1_no,
            d1_phys,
            "D1",
            sample_dir / "D1_field_comparison.png",
        )
        plot_field_comparison(
            sample,
            sample.d2_true,
            d2_no,
            d2_phys,
            "D2",
            sample_dir / "D2_field_comparison.png",
        )
        plot_residual_comparison(
            sample,
            components,
            sample_dir / "PDE_residual_comparison.png",
        )
        plot_tau_slices(
            sample,
            d1_no,
            d2_no,
            d1_phys,
            d2_phys,
            sample_dir / "fixed_tau_slices.png",
        )
        plot_a_slices(
            sample,
            d1_no,
            d2_no,
            d1_phys,
            d2_phys,
            sample_dir / "fixed_A_slices.png",
        )
        plot_error_profiles(
            sample,
            d1_no,
            d2_no,
            d1_phys,
            d2_phys,
            sample_dir / "error_profiles.png",
        )

    return metrics


# =============================================================================
# 15. 主程序
# =============================================================================


def main() -> None:
    set_seed(SEED)
    safe_mkdir(OUTPUT_DIR)
    safe_mkdir(SAMPLE_OUTPUT_DIR)
    safe_mkdir(SUMMARY_FIG_DIR)
    safe_mkdir(SUMMARY_DATA_DIR)

    print("=" * 96)
    print("POD-DeepONet / POD-PINO Forward Prediction Comparison")
    print("=" * 96)
    print(f"Device                 : {DEVICE}")
    print(f"Ablation result dir    : {ABLATION_RESULT_DIR}")
    print(f"No-PDE model           : {NO_PDE_MODEL_PATH}")
    print(f"With-PDE model         : {WITH_PDE_MODEL_PATH}")
    print(f"Preprocessing          : {PREPROCESSING_PATH}")
    print(f"Test data dir          : {TEST_DATA_DIR}")
    print(f"Output dir             : {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Step 1：加载共享预处理与模型
    # -------------------------------------------------------------------------
    shared = load_shared_preprocessing(PREPROCESSING_PATH)
    no_pde_model = build_model(shared, NO_PDE_MODEL_PATH)
    with_pde_model = build_model(shared, WITH_PDE_MODEL_PATH)

    print("\n共享预处理信息：")
    print(f"POD modes              : {shared.actual_num_modes}")
    print(f"A-grid points          : {len(shared.unified_a_grid)}")
    print(f"tau-grid points        : {len(shared.unified_tau_grid)}")
    print(f"Output dimension       : {len(shared.y_mean_scaler)}")

    lambda_info: Dict[str, Any] = {}
    if OPTIMAL_LAMBDA_SELECTION_PATH.exists():
        with OPTIMAL_LAMBDA_SELECTION_PATH.open("r", encoding="utf-8") as file:
            lambda_info = json.load(file)
        print(f"Optimal lambda used    : {lambda_info.get('optimal_lambda', 'unknown')}")

    # 保存运行配置。
    run_config = {
        "ablation_result_dir": ABLATION_RESULT_DIR,
        "no_pde_model_path": NO_PDE_MODEL_PATH,
        "with_pde_model_path": WITH_PDE_MODEL_PATH,
        "preprocessing_path": PREPROCESSING_PATH,
        "test_data_dir": TEST_DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "seed": SEED,
        "model_hyperparameters": {
            "branch_input_dim": BRANCH_INPUT_DIM,
            "hidden_units": HIDDEN_UNITS,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "num_pod_modes": shared.actual_num_modes,
        },
        "optimal_lambda_info": lambda_info,
    }
    save_json(OUTPUT_DIR / "run_config.json", run_config)

    # -------------------------------------------------------------------------
    # Step 2：读取全部测试样本
    # -------------------------------------------------------------------------
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {TEST_DATA_DIR}")

    test_files = sorted(TEST_DATA_DIR.glob(TEST_FILE_GLOB))
    if not test_files:
        raise FileNotFoundError(
            f"目录 {TEST_DATA_DIR} 中未找到匹配 {TEST_FILE_GLOB} 的文件。"
        )

    if MAX_TEST_FILES is not None and len(test_files) > MAX_TEST_FILES:
        rng = random.Random(SEED)
        test_files = sorted(rng.sample(test_files, MAX_TEST_FILES))

    samples: List[TestSample] = []
    failed_records: List[Dict[str, str]] = []

    print(f"\n读取测试文件，共 {len(test_files)} 个候选文件……")
    for file_path in test_files:
        try:
            samples.append(load_test_sample(file_path, shared))
        except Exception as exc:
            failed_records.append({"file": str(file_path), "reason": str(exc)})
            print(f"[警告] 跳过 {file_path.name}: {exc}")

    if failed_records:
        pd.DataFrame(failed_records).to_csv(
            SUMMARY_DATA_DIR / "failed_test_files.csv", index=False
        )

    if not samples:
        raise RuntimeError("没有成功读取任何测试样本。")

    print(f"成功读取样本数        : {len(samples)}")
    print(f"失败样本数            : {len(failed_records)}")

    params_batch = np.stack([sample.params for sample in samples], axis=0)

    # -------------------------------------------------------------------------
    # Step 3：批量前向预测
    # -------------------------------------------------------------------------
    print("\n执行 POD-DeepONet 前向预测……")
    start = time.time()
    y_no = predict_batch(no_pde_model, params_batch, shared)
    no_elapsed = time.time() - start

    print("执行 POD-PINO 前向预测……")
    start = time.time()
    y_phys = predict_batch(with_pde_model, params_batch, shared)
    phys_elapsed = time.time() - start

    n_tau = len(shared.unified_tau_grid)
    n_a = len(shared.unified_a_grid)
    d1_no, d2_no = split_prediction_fields(y_no, n_tau, n_a)
    d1_phys, d2_phys = split_prediction_fields(y_phys, n_tau, n_a)

    timing_info = {
        "num_samples": len(samples),
        "pod_deeponet_total_seconds": no_elapsed,
        "pod_pino_total_seconds": phys_elapsed,
        "pod_deeponet_seconds_per_sample": no_elapsed / len(samples),
        "pod_pino_seconds_per_sample": phys_elapsed / len(samples),
    }
    save_json(SUMMARY_DATA_DIR / "forward_timing.json", timing_info)
    pd.DataFrame([timing_info]).to_csv(
        SUMMARY_DATA_DIR / "forward_timing.csv", index=False
    )

    # 保存批量预测数组。
    if SAVE_NPZ:
        np.savez_compressed(
            SUMMARY_DATA_DIR / "all_forward_predictions.npz",
            params=params_batch,
            A=shared.unified_a_grid,
            tau=shared.unified_tau_grid,
            D1_true=np.stack([sample.d1_true for sample in samples], axis=0),
            D2_true=np.stack([sample.d2_true for sample in samples], axis=0),
            D1_no_pde=d1_no,
            D2_no_pde=d2_no,
            D1_with_pde=d1_phys,
            D2_with_pde=d2_phys,
            sample_names=np.asarray([sample.sample_name for sample in samples]),
        )

    # -------------------------------------------------------------------------
    # Step 4：逐样本评价、保存和绘图
    # -------------------------------------------------------------------------
    print("\n逐样本保存结果和绘图……")
    sample_metric_rows: List[Dict[str, float]] = []

    for index, sample in enumerate(samples):
        print(f"[{index + 1:4d}/{len(samples):4d}] {sample.sample_name}")
        try:
            metrics = process_single_sample(
                sample,
                d1_no[index],
                d2_no[index],
                d1_phys[index],
                d2_phys[index],
            )
            metrics["sample_index"] = index
            sample_metric_rows.append(metrics)
        except Exception as exc:
            traceback.print_exc()
            failed_records.append({"file": str(sample.file_path), "reason": f"processing: {exc}"})

    if not sample_metric_rows:
        raise RuntimeError("所有样本处理均失败。")

    summary_df = pd.DataFrame(sample_metric_rows).sort_values("sample_index")
    summary_df.to_csv(SUMMARY_DATA_DIR / "sample_metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # Step 5：汇总统计
    # -------------------------------------------------------------------------
    numeric_summary_df = aggregate_numeric_metrics(summary_df)
    numeric_summary_df.to_csv(
        SUMMARY_DATA_DIR / "all_numeric_metric_statistics.csv", index=False
    )

    method_summary_df = build_method_summary(summary_df)
    method_summary_df.to_csv(SUMMARY_DATA_DIR / "method_summary.csv", index=False)
    save_json(
        SUMMARY_DATA_DIR / "method_summary.json",
        {
            "methods": method_summary_df.to_dict(orient="records"),
            "timing": timing_info,
            "optimal_lambda_info": lambda_info,
        },
    )

    # 关键改善率统计。
    improvement_keys = [
        "d1_rmse_improvement_percent",
        "d2_rmse_improvement_percent",
        "balanced_nrmse_improvement_percent",
        "normalized_pde_improvement_percent",
        "raw_pde_improvement_percent",
        "derivative_rmse_improvement_percent",
    ]
    improvement_rows = []
    for key in improvement_keys:
        values = summary_df[key].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        improvement_rows.append(
            {
                "metric": key,
                "mean_improvement_percent": float(np.mean(values)),
                "median_improvement_percent": float(np.median(values)),
                "positive_improvement_fraction": float(np.mean(values > 0.0)),
                "negative_improvement_fraction": float(np.mean(values < 0.0)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    pd.DataFrame(improvement_rows).to_csv(
        SUMMARY_DATA_DIR / "improvement_statistics.csv", index=False
    )

    # -------------------------------------------------------------------------
    # Step 6：全测试集汇总图
    # -------------------------------------------------------------------------
    print("生成汇总图片……")
    plot_metric_boxplots(summary_df, SUMMARY_FIG_DIR / "metric_boxplots.png")
    plot_metric_cdfs(summary_df, SUMMARY_FIG_DIR / "metric_cdfs.png")
    plot_improvement_distributions(
        summary_df, SUMMARY_FIG_DIR / "improvement_distributions.png"
    )
    plot_error_scatter(summary_df, SUMMARY_FIG_DIR / "sample_metric_scatter.png")
    plot_parameter_space_errors(
        summary_df, SUMMARY_FIG_DIR / "parameter_space_performance.png"
    )
    plot_mean_error_profiles(
        samples,
        {
            "d1_no": d1_no,
            "d2_no": d2_no,
            "d1_phys": d1_phys,
            "d2_phys": d2_phys,
        },
        SUMMARY_FIG_DIR / "dataset_mean_error_profiles.png",
    )

    # -------------------------------------------------------------------------
    # Step 7：代表性样本清单与复制索引
    # -------------------------------------------------------------------------
    if REPRESENTATIVE_SELECTION:
        representative_df = select_representative_samples(summary_df)
        representative_df.to_csv(
            SUMMARY_DATA_DIR / "representative_samples.csv", index=False
        )

        representative_manifest = []
        for _, row in representative_df.iterrows():
            representative_manifest.append(
                {
                    "sample_name": row["sample_name"],
                    "selection_reason": row["selection_reason"],
                    "sample_directory": str(SAMPLE_OUTPUT_DIR / row["sample_name"]),
                    "D1_plot": str(SAMPLE_OUTPUT_DIR / row["sample_name"] / "D1_field_comparison.png"),
                    "D2_plot": str(SAMPLE_OUTPUT_DIR / row["sample_name"] / "D2_field_comparison.png"),
                    "PDE_plot": str(SAMPLE_OUTPUT_DIR / row["sample_name"] / "PDE_residual_comparison.png"),
                }
            )
        save_json(
            SUMMARY_DATA_DIR / "representative_samples_manifest.json",
            {"samples": representative_manifest},
        )

    # 最终报告。
    report_lines = [
        "POD-DeepONet / POD-PINO 前向预测对比报告",
        "=" * 72,
        f"成功样本数: {len(summary_df)}",
        f"失败样本数: {len(failed_records)}",
        f"POD-DeepONet 单样本平均推理时间: {timing_info['pod_deeponet_seconds_per_sample']:.8e} s",
        f"POD-PINO 单样本平均推理时间: {timing_info['pod_pino_seconds_per_sample']:.8e} s",
        "",
    ]

    for row in improvement_rows:
        report_lines.append(
            f"{row['metric']}: mean={row['mean_improvement_percent']:.4f}%, "
            f"median={row['median_improvement_percent']:.4f}%, "
            f"positive fraction={row['positive_improvement_fraction']:.4f}"
        )

    report_lines.extend(
        [
            "",
            "主要输出：",
            f"- 每组样本目录: {SAMPLE_OUTPUT_DIR}",
            f"- 全样本指标: {SUMMARY_DATA_DIR / 'sample_metrics.csv'}",
            f"- 方法汇总: {SUMMARY_DATA_DIR / 'method_summary.csv'}",
            f"- 改善率汇总: {SUMMARY_DATA_DIR / 'improvement_statistics.csv'}",
            f"- 汇总图片: {SUMMARY_FIG_DIR}",
        ]
    )
    with (OUTPUT_DIR / "forward_prediction_report.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    if failed_records:
        pd.DataFrame(failed_records).to_csv(
            SUMMARY_DATA_DIR / "all_failed_records.csv", index=False
        )

    print("\n" + "=" * 96)
    print("前向预测与综合评价完成")
    print("=" * 96)
    print(f"成功样本数          : {len(summary_df)}")
    print(f"每样本完整结果目录  : {SAMPLE_OUTPUT_DIR}")
    print(f"汇总数据目录        : {SUMMARY_DATA_DIR}")
    print(f"汇总图片目录        : {SUMMARY_FIG_DIR}")
    print(f"最终报告            : {OUTPUT_DIR / 'forward_prediction_report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n程序运行失败，异常信息如下：")
        traceback.print_exc()
        raise
