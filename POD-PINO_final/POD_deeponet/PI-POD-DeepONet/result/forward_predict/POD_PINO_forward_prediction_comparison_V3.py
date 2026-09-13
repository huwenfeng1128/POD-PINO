# -*- coding: utf-8 -*-
"""
POD-DeepONet / POD-PINO 双模型前向预测 V2与消融对比分析 (网格化场数据增强版)
====================================================================

功能
----
1. 严格加载消融训练输出：
   - shared_preprocessing.pth
   - config.json
   - no_pde_residual/best_model.pth
   - with_pde_residual/best_model.pth

2. 对每一组模拟数据同时执行：
   - 不含 PDE 残差模型（no_pde_residual）
   - 含 PDE 残差模型（with_pde_residual）

3. 支持两类测试数据：
   A. 训练格式：
      A, tau, nu, kappa, d_diffusion, D1_adj, D2_adj
   B. KM 格式：
      A, tau_index, tau_sec, D1_data, D2_data
      参数优先从 CSV 列读取；若无参数列，则从文件名
      “(nu,kappa,d_diffusion).csv”中解析。

4. 每个样本保存：
   - 参照标准网格场结构导出的统一 CSV (field_comparison_data_*.csv)，
     包含 A, Tau 网格、真值二维插值(D1/D2_True_Interp)、双模型预测场、误差场及差值场，
     极度方便 Origin/Python 直接读取绘图。
   - 原始网格完整预测场 NPZ
   - 对齐到真实数据离散点的逐点 CSV
   - 逐 tau 指标 CSV
   - 样本总体指标 JSON
   - D1、D2 真值/双模型预测/误差场图
   - 双模型差值图
   - PDE R1、R2 残差图
   - 典型 tau 剖面对比图
   - D2 负值分布图

5. 全局保存：
   - all_sample_metrics.csv
   - paired_sample_metrics.csv
   - per_tau_metrics_all.csv
   - pointwise_predictions_all.csv（可关闭）
   - summary_statistics.csv
   - improvement_statistics.csv
   - parameter_error_correlations.csv
   - model_win_rates.csv
   - failed_files.csv
   - 多种总体统计图和自动分析报告

注意
----
- 前向预测阶段“有物理信息”和“无物理信息”的区别来自两套训练完成的权重。
  预测时不会再次把 PDE 残差加到输出中。
- 输出向量严格按 [D1_flat | D2_flat] 排列，并恢复为 [tau, A]。
- 默认对 tau 与 A 均做线性插值到真实数据网格；超出训练网格的点默认记为 NaN。
"""

import os
import re
import json
import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.stats import wilcoxon
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 1. 用户配置：通常只需修改本节
# =============================================================================

# 消融训练输出根目录
ABLATION_RESULT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet"
    r"\result\xiaorongshiyan\train_result_physics_ablation_2"
)

# 待预测模拟数据目录
TEST_DATA_DIR = Path(
    r"D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4"
)

# 前向预测输出目录
OUTPUT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\forward_predict\forward_prediction_physics_ablation_3"
)

# 文件匹配模式。KM 文件可用 "*.csv"；训练格式可用 "data_*.csv"
TEST_FILE_PATTERN = "*.csv"

# None 表示处理所有文件；整数表示固定随机抽取指定数量
MAX_TEST_FILES: Optional[int] = None
RANDOM_SEED = 2026

# 是否保存可能较大的全体逐点汇总表
SAVE_GLOBAL_POINTWISE_TABLE = True

# 每个样本都生成完整图片。数据很多时会产生较多图片。
PLOT_EVERY_SAMPLE = True

# 剖面图最多选择多少个 tau
NUM_PROFILE_TAU = 6

# 绘图分辨率
FIG_DPI = 220

# 插值方法：当前使用规则网格线性插值
# 超出模型训练 A/tau 范围时：
#   False -> 输出 NaN，避免不可靠外推
#   True  -> 允许线性外推
ALLOW_EXTRAPOLATION = False

# 推理批量大小
INFERENCE_BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

MODEL_NAMES = ("no_pde_residual", "with_pde_residual")
MODEL_LABELS = {
    "no_pde_residual": "Without PDE residual",
    "with_pde_residual": "With PDE residual",
}


# =============================================================================
# 2. 通用工具
# =============================================================================

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(value: Any, dtype=np.float64) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def json_converter(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"无法序列化类型: {type(obj)}")


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_converter,
            allow_nan=False,
        )


def safe_torch_load(path: Path, map_location=DEVICE):
    """兼容新旧 PyTorch，并允许加载包含 NumPy 数组的预处理字典。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_load_state_dict(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def sanitize_name(name: str, max_length: int = 120) -> str:
    stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*\s]+', "_", stem)
    return stem[:max_length]


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_median(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def percent_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or abs(baseline) < 1e-30:
        return float("nan")
    return float((baseline - candidate) / abs(baseline) * 100.0)


def safe_relative_l2(y_true, y_pred) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    denominator = np.linalg.norm(np.asarray(y_true)[mask].ravel())
    if denominator < 1e-30:
        return float("nan")
    return float(
        np.linalg.norm((np.asarray(y_pred)[mask] - np.asarray(y_true)[mask]).ravel())
        / denominator
    )


def safe_r2(y_true, y_pred) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return float("nan")
    yt = np.asarray(y_true)[mask].ravel()
    yp = np.asarray(y_pred)[mask].ravel()
    denominator = np.sum((yt - np.mean(yt)) ** 2)
    if denominator < 1e-30:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / denominator)


def error_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    result = {f"{prefix}n_valid": int(np.sum(mask))}
    if not np.any(mask):
        for key in ("mse", "rmse", "mae", "max_abs", "bias", "rel_l2", "r2"):
            result[f"{prefix}{key}"] = float("nan")
        return result
    err = yp[mask] - yt[mask]
    result.update({
        f"{prefix}mse": float(np.mean(err ** 2)),
        f"{prefix}rmse": float(np.sqrt(np.mean(err ** 2))),
        f"{prefix}mae": float(np.mean(np.abs(err))),
        f"{prefix}max_abs": float(np.max(np.abs(err))),
        f"{prefix}bias": float(np.mean(err)),
        f"{prefix}rel_l2": safe_relative_l2(yt, yp),
        f"{prefix}r2": safe_r2(yt, yp),
    })
    return result


# =============================================================================
# 3. 与消融训练完全一致的模型结构
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
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_units, dtype=DTYPE)
        ]
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
            "pod_basis",
            torch.as_tensor(pod_basis, dtype=DTYPE),
        )
        self.register_buffer(
            "y_mean_pod_scaled",
            torch.as_tensor(y_mean_pod_scaled, dtype=DTYPE),
        )

    def forward(self, branch_x: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


# =============================================================================
# 4. 加载训练资产
# =============================================================================

def load_training_assets() -> Tuple[Dict, Dict, Dict[str, PODDeepONet]]:
    preprocessing_path = ABLATION_RESULT_DIR / "shared_preprocessing.pth"
    config_path = ABLATION_RESULT_DIR / "config.json"

    required = [preprocessing_path, config_path]
    for model_name in MODEL_NAMES:
        required.append(ABLATION_RESULT_DIR / model_name / "best_model.pth")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少必要训练文件：\n" + "\n".join(missing))

    preprocessing_raw = safe_torch_load(preprocessing_path, map_location="cpu")
    preprocessing = {
        key: to_numpy(value) if key not in ("actual_num_modes", "retained_energy") else value
        for key, value in preprocessing_raw.items()
    }
    preprocessing["actual_num_modes"] = int(preprocessing_raw["actual_num_modes"])
    preprocessing["retained_energy"] = float(preprocessing_raw.get("retained_energy", np.nan))

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    branch_dim = int(config["BRANCH_INPUT_DIM"])
    hidden_units = int(config["HIDDEN_UNITS"])
    hidden_layers = int(config["NUM_HIDDEN_LAYERS"])
    dropout = float(config["DROPOUT_RATE"])
    num_modes = int(preprocessing["actual_num_modes"])

    models: Dict[str, PODDeepONet] = {}
    for model_name in MODEL_NAMES:
        model = PODDeepONet(
            branch_input_dim=branch_dim,
            hidden_units=hidden_units,
            num_hidden_layers=hidden_layers,
            num_pod_modes=num_modes,
            pod_basis=preprocessing["pod_basis"],
            y_mean_pod_scaled=preprocessing["y_mean_pod_scaled"],
            dropout_rate=dropout,
        ).to(DEVICE)
        state_path = ABLATION_RESULT_DIR / model_name / "best_model.pth"
        state = safe_load_state_dict(state_path)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[model_name] = model

    field_len = len(preprocessing["unified_a_grid"]) * len(
        preprocessing["unified_tau_grid"]
    )
    expected_output = 2 * field_len
    actual_output = len(preprocessing["y_mean_scaler"])
    if expected_output != actual_output:
        raise ValueError(
            f"输出维度不一致：网格推断为 {expected_output}，"
            f"y_mean_scaler 长度为 {actual_output}。"
        )

    return preprocessing, config, models


# =============================================================================
# 5. 数据读取与格式自动识别
# =============================================================================

def parse_params_from_filename(file_path: Path) -> Optional[Tuple[float, float, float]]:
    stem = file_path.stem.strip()
    match = re.search(
        r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        stem,
    )
    if not match:
        return None
    try:
        return tuple(float(match.group(i)) for i in (1, 2, 3))
    except ValueError:
        return None


def extract_constant_params(df: pd.DataFrame, file_path: Path) -> Tuple[float, float, float]:
    param_cols = ["nu", "kappa", "d_diffusion"]
    if all(c in df.columns for c in param_cols):
        values = []
        for c in param_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(vals) != 1:
                raise ValueError(f"{file_path.name} 中参数列 {c} 不是唯一常数。")
            values.append(float(vals[0]))
        return tuple(values)

    parsed = parse_params_from_filename(file_path)
    if parsed is None:
        raise ValueError(
            "无法取得 (nu, kappa, d_diffusion)：CSV 中没有完整参数列，"
            "文件名也不符合 '(nu,kappa,d).csv'。"
        )
    return parsed


def load_test_sample(file_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(file_path, on_bad_lines="skip")
    if df.empty:
        raise ValueError("CSV 为空。")

    params = extract_constant_params(df, file_path)

    # 训练格式
    training_cols = {"A", "tau", "D1_adj", "D2_adj"}
    # KM 格式
    km_cols = {"A", "tau_sec", "D1_data", "D2_data"}

    if training_cols.issubset(df.columns):
        data_format = "training_format"
        work = df[["A", "tau", "D1_adj", "D2_adj"]].copy()
        work.columns = ["A", "tau", "D1_true", "D2_true"]
        if "tau_index" in df.columns:
            work["tau_index"] = df["tau_index"].values
        else:
            unique_tau = sorted(pd.to_numeric(work["tau"], errors="coerce").dropna().unique())
            tau_map = {v: i for i, v in enumerate(unique_tau)}
            work["tau_index"] = work["tau"].map(tau_map)

    elif km_cols.issubset(df.columns):
        data_format = "km_format"
        work = df[["A", "tau_sec", "D1_data", "D2_data"]].copy()
        work.columns = ["A", "tau", "D1_true", "D2_true"]
        if "tau_index" in df.columns:
            work["tau_index"] = df["tau_index"].values
        else:
            unique_tau = sorted(pd.to_numeric(work["tau"], errors="coerce").dropna().unique())
            tau_map = {v: i for i, v in enumerate(unique_tau)}
            work["tau_index"] = work["tau"].map(tau_map)
    else:
        raise ValueError(
            "无法识别数据格式。需要训练格式列 "
            "[A,tau,D1_adj,D2_adj] 或 KM 格式列 "
            "[A,tau_sec,D1_data,D2_data]。"
        )

    for c in ["A", "tau", "D1_true", "D2_true"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["A", "tau"]).copy()
    if work.empty:
        raise ValueError("删除无效 A/tau 后没有数据。")

    if work.duplicated(["tau", "A"]).any():
        # 对重复网格点取均值，避免插值与绘图失败
        work = (
            work.groupby(["tau", "A"], as_index=False)
            .agg({
                "D1_true": "mean",
                "D2_true": "mean",
                "tau_index": "first",
            })
        )

    work = work.sort_values(["tau", "A"]).reset_index(drop=True)
    return {
        "file_path": file_path,
        "filename": file_path.name,
        "sample_id": sanitize_name(file_path.name),
        "format": data_format,
        "params": params,
        "points": work,
    }


# =============================================================================
# 6. 模型预测、插值和物理残差
# =============================================================================

def predict_full_fields(
        params_batch: np.ndarray,
        models: Dict[str, PODDeepONet],
        preprocessing: Dict,
) -> Dict[str, Dict[str, np.ndarray]]:
    params_batch = np.asarray(params_batch, dtype=np.float64).reshape(-1, 3)
    branch_mean = preprocessing["branch_mean"]
    branch_std = preprocessing["branch_std"].copy()
    branch_std[np.abs(branch_std) < 1e-10] = 1.0

    branch_scaled = (params_batch - branch_mean) / branch_std
    tensor_dataset = TensorDataset(torch.from_numpy(branch_scaled))
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(INFERENCE_BATCH_SIZE, len(tensor_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    num_tau = len(preprocessing["unified_tau_grid"])
    num_a = len(preprocessing["unified_a_grid"])
    field_len = num_tau * num_a

    results = {}
    for model_name, model in models.items():
        scaled_chunks = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(
                    DEVICE,
                    dtype=DTYPE,
                    non_blocking=torch.cuda.is_available(),
                )
                scaled_chunks.append(model(batch).cpu().numpy())
        y_scaled = np.concatenate(scaled_chunks, axis=0)
        y_physical = (
                y_scaled * preprocessing["y_std_scaler"]
                + preprocessing["y_mean_scaler"]
        )
        results[model_name] = {
            "y_scaled": y_scaled,
            "y_physical": y_physical,
            "D1": y_physical[:, :field_len].reshape(-1, num_tau, num_a),
            "D2": y_physical[:, field_len:].reshape(-1, num_tau, num_a),
        }
    return results


def build_interpolator(
        tau_grid: np.ndarray,
        a_grid: np.ndarray,
        field: np.ndarray,
) -> RegularGridInterpolator:
    fill_value = None if ALLOW_EXTRAPOLATION else np.nan
    return RegularGridInterpolator(
        (tau_grid, a_grid),
        field,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def interpolate_field_to_points(
        field: np.ndarray,
        tau_grid: np.ndarray,
        a_grid: np.ndarray,
        target_tau: np.ndarray,
        target_a: np.ndarray,
) -> np.ndarray:
    interp = build_interpolator(tau_grid, a_grid, field)
    query = np.column_stack([target_tau, target_a])
    return np.asarray(interp(query), dtype=np.float64)


def compute_residual_maps(
        D1: np.ndarray,
        D2: np.ndarray,
        params: Tuple[float, float, float],
        a_grid: np.ndarray,
        tau_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    与训练代码一致的中心差分 PDE 残差。
    返回内部网格 [tau[1:-1], A[1:-1]] 上的 R1、R2。
    """
    nu, kappa, d_diff = params
    a = a_grid[None, :]
    tau = tau_grid[:, None]
    a_safe = np.maximum(a, 1e-9)

    d1_th = nu * a - (kappa / 8.0) * (a ** 3) + d_diff / a_safe
    d2_th = np.full_like(d1_th, d_diff)

    u1 = tau * D1
    u2 = tau * D2

    da = float(a_grid[1] - a_grid[0])
    dtau = float(tau_grid[1] - tau_grid[0])

    u1_in = u1[1:-1, 1:-1]
    d1_th_in = d1_th[:, 1:-1]
    d2_th_in = d2_th[:, 1:-1]

    du1_dtau = (u1[2:, 1:-1] - u1[:-2, 1:-1]) / (2.0 * dtau)
    du2_dtau = (u2[2:, 1:-1] - u2[:-2, 1:-1]) / (2.0 * dtau)
    du1_da = (u1[1:-1, 2:] - u1[1:-1, :-2]) / (2.0 * da)
    du2_da = (u2[1:-1, 2:] - u2[1:-1, :-2]) / (2.0 * da)
    d2u1_da2 = (
                       u1[1:-1, 2:] - 2.0 * u1[1:-1, 1:-1] + u1[1:-1, :-2]
               ) / da ** 2
    d2u2_da2 = (
                       u2[1:-1, 2:] - 2.0 * u2[1:-1, 1:-1] + u2[1:-1, :-2]
               ) / da ** 2

    r1 = (
            du1_dtau
            - d1_th_in * du1_da
            - d2_th_in * d2u1_da2
            - d1_th_in
    )
    r2 = (
            du2_dtau
            - d1_th_in * du2_da
            - d2_th_in * d2u2_da2
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


# =============================================================================
# 7. 单样本指标与保存
# =============================================================================

def calculate_per_tau_metrics(point_df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    records = []
    for tau_value, group in point_df.groupby("tau", sort=True):
        for model_name in MODEL_NAMES:
            row = {
                "sample_id": sample_id,
                "tau": float(tau_value),
                "tau_index": group["tau_index"].iloc[0],
                "model": model_name,
            }
            row.update(error_metrics(
                group["D1_true"].to_numpy(),
                group[f"D1_pred_{model_name}"].to_numpy(),
                prefix="d1_",
            ))
            row.update(error_metrics(
                group["D2_true"].to_numpy(),
                group[f"D2_pred_{model_name}"].to_numpy(),
                prefix="d2_",
            ))
            row["total_mse"] = row["d1_mse"] + row["d2_mse"]
            records.append(row)
    return pd.DataFrame(records)


def calculate_sample_metrics(
        point_df: pd.DataFrame,
        full_predictions: Dict[str, Dict[str, np.ndarray]],
        params: Tuple[float, float, float],
        preprocessing: Dict,
        sample_meta: Dict,
) -> List[Dict]:
    records = []
    a_grid = preprocessing["unified_a_grid"]
    tau_grid = preprocessing["unified_tau_grid"]

    for model_name in MODEL_NAMES:
        D1 = full_predictions[model_name]["D1"][0]
        D2 = full_predictions[model_name]["D2"][0]
        r1, r2, derivatives = compute_residual_maps(
            D1, D2, params, a_grid, tau_grid
        )

        row = {
            "sample_id": sample_meta["sample_id"],
            "filename": sample_meta["filename"],
            "data_format": sample_meta["format"],
            "model": model_name,
            "nu": params[0],
            "kappa": params[1],
            "d_diffusion": params[2],
            "n_points": len(point_df),
            "n_tau": point_df["tau"].nunique(),
            "n_a": point_df["A"].nunique(),
            "train_a_min": float(np.min(a_grid)),
            "train_a_max": float(np.max(a_grid)),
            "train_tau_min": float(np.min(tau_grid)),
            "train_tau_max": float(np.max(tau_grid)),
            "target_a_outside_fraction": float(np.mean(
                (point_df["A"] < np.min(a_grid)) |
                (point_df["A"] > np.max(a_grid))
            )),
            "target_tau_outside_fraction": float(np.mean(
                (point_df["tau"] < np.min(tau_grid)) |
                (point_df["tau"] > np.max(tau_grid))
            )),
        }

        row.update(error_metrics(
            point_df["D1_true"].to_numpy(),
            point_df[f"D1_pred_{model_name}"].to_numpy(),
            prefix="d1_",
        ))
        row.update(error_metrics(
            point_df["D2_true"].to_numpy(),
            point_df[f"D2_pred_{model_name}"].to_numpy(),
            prefix="d2_",
        ))

        row["total_mse"] = row["d1_mse"] + row["d2_mse"]
        row["total_rmse_rss"] = float(np.sqrt(row["d1_rmse"] ** 2 + row["d2_rmse"] ** 2))
        row["pde_r1_mse"] = float(np.mean(r1 ** 2))
        row["pde_r2_mse"] = float(np.mean(r2 ** 2))
        row["pde_total_mse"] = row["pde_r1_mse"] + row["pde_r2_mse"]
        row["pde_r1_rmse"] = float(np.sqrt(row["pde_r1_mse"]))
        row["pde_r2_rmse"] = float(np.sqrt(row["pde_r2_mse"]))  # 修复了键名引用的 Bug
        row["d2_negative_fraction_full_grid"] = float(np.mean(D2 < 0.0))
        row["d2_min_full_grid"] = float(np.min(D2))
        row["d2_negative_magnitude_mse"] = float(np.mean(np.maximum(-D2, 0.0) ** 2))

        for key, value in derivatives.items():
            row[f"pred_{key}_rms"] = float(np.sqrt(np.mean(value ** 2)))

        records.append(row)
    return records


def full_grid_long_table(
        full_predictions: Dict[str, Dict[str, np.ndarray]],
        params: Tuple[float, float, float],
        preprocessing: Dict,
        point_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """生成包含网格坐标 A, Tau、插值真值、双模型预测场、绝对误差场及模型差值场的标准化 CSV。

    参照可视化代码的 2D 网格建表标准（利用 scipy.interpolate.griddata 将散点真值
    插值投影到统一规则构格上），产生极便于 Origin 或 Python 快速绘图与处理的平坦长表。
    """
    a_grid = preprocessing["unified_a_grid"]
    tau_grid = preprocessing["unified_tau_grid"]
    tau_mesh, a_mesh = np.meshgrid(tau_grid, a_grid, indexing="ij")

    df = pd.DataFrame({
        "A": a_mesh.ravel(),
        "Tau": tau_mesh.ravel(),
        "nu": params[0],
        "kappa": params[1],
        "d_diffusion": params[2],
    })

    # 插值真值到统一规则网格上 (参照对比绘图规范)
    if point_df is not None:
        valid_d1 = point_df.dropna(subset=["D1_true", "A", "tau"])
        if len(valid_d1) >= 4:
            pts_d1 = (valid_d1["A"].values, valid_d1["tau"].values)
            vals_d1 = valid_d1["D1_true"].values
            d1_true_interp = griddata(
                pts_d1, vals_d1, (a_mesh, tau_mesh), method="linear", fill_value=np.nan
            )
        else:
            d1_true_interp = np.full_like(a_mesh, np.nan)
        df["D1_True_Interp"] = d1_true_interp.ravel()

        valid_d2 = point_df.dropna(subset=["D2_true", "A", "tau"])
        if len(valid_d2) >= 4:
            pts_d2 = (valid_d2["A"].values, valid_d2["tau"].values)
            vals_d2 = valid_d2["D2_true"].values
            d2_true_interp = griddata(
                pts_d2, vals_d2, (a_mesh, tau_mesh), method="linear", fill_value=np.nan
            )
        else:
            d2_true_interp = np.full_like(a_mesh, np.nan)
        df["D2_True_Interp"] = d2_true_interp.ravel()
    else:
        df["D1_True_Interp"] = np.nan
        df["D2_True_Interp"] = np.nan

    # 预测场列
    for model_name in MODEL_NAMES:
        df[f"D1_Pred_{model_name}"] = full_predictions[model_name]["D1"][0].ravel()
        df[f"D2_Pred_{model_name}"] = full_predictions[model_name]["D2"][0].ravel()

    # 绝对误差场
    df["D1_Error_no_pde_residual"] = np.abs(df["D1_Pred_no_pde_residual"] - df["D1_True_Interp"])
    df["D1_Error_with_pde_residual"] = np.abs(df["D1_Pred_with_pde_residual"] - df["D1_True_Interp"])
    df["D2_Error_no_pde_residual"] = np.abs(df["D2_Pred_no_pde_residual"] - df["D2_True_Interp"])
    df["D2_Error_with_pde_residual"] = np.abs(df["D2_Pred_with_pde_residual"] - df["D2_True_Interp"])

    # 双模型差值场
    df["D1_Diff_with_minus_without"] = (
            df["D1_Pred_with_pde_residual"] - df["D1_Pred_no_pde_residual"]
    )
    df["D2_Diff_with_minus_without"] = (
            df["D2_Pred_with_pde_residual"] - df["D2_Pred_no_pde_residual"]
    )
    return df


# =============================================================================
# 8. 单样本绘图
# =============================================================================

def pivot_field(
        point_df: pd.DataFrame,
        value_col: str,
        a_values: Optional[np.ndarray] = None,
        tau_values: Optional[np.ndarray] = None,
):
    if a_values is None:
        a_values = np.sort(point_df["A"].dropna().unique().astype(float))
    if tau_values is None:
        tau_values = np.sort(point_df["tau"].dropna().unique().astype(float))

    pivot = point_df.pivot_table(
        index="tau",
        columns="A",
        values=value_col,
        aggfunc="mean",
        dropna=False,
    )
    pivot = pivot.reindex(index=tau_values, columns=a_values)
    return np.asarray(a_values), np.asarray(tau_values), pivot.to_numpy(dtype=float)


def build_target_grid_diagnostics(point_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tau_value, group in point_df.groupby("tau", sort=True, dropna=False):
        n = len(group)
        d1_valid = int(np.isfinite(group["D1_true"].to_numpy(dtype=float)).sum())
        d2_valid = int(np.isfinite(group["D2_true"].to_numpy(dtype=float)).sum())
        both_valid = int((
                                 np.isfinite(group["D1_true"].to_numpy(dtype=float))
                                 & np.isfinite(group["D2_true"].to_numpy(dtype=float))
                         ).sum())
        rows.append({
            "tau": float(tau_value),
            "tau_index": group["tau_index"].iloc[0],
            "n_grid_points": int(n),
            "d1_valid_points": d1_valid,
            "d2_valid_points": d2_valid,
            "both_valid_points": both_valid,
            "d1_coverage": d1_valid / n if n else np.nan,
            "d2_coverage": d2_valid / n if n else np.nan,
            "both_coverage": both_valid / n if n else np.nan,
            "d1_entire_layer_missing": bool(d1_valid == 0),
            "d2_entire_layer_missing": bool(d2_valid == 0),
        })
    return pd.DataFrame(rows)


def imshow_field(ax, a, tau, z, title, cmap=None, symmetric=False):
    finite = np.asarray(z)[np.isfinite(z)]
    kwargs = {}
    if symmetric and len(finite):
        vmax = np.max(np.abs(finite))
        if vmax > 0:
            kwargs["norm"] = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.pcolormesh(a, tau, z, shading="auto", cmap=cmap, **kwargs)
    ax.set_xlabel("A")
    ax.set_ylabel("tau")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, shrink=0.88)


def plot_field_comparison(
        point_df: pd.DataFrame,
        field_name: str,
        save_path: Path,
) -> None:
    true_col = f"{field_name}_true"
    no_col = f"{field_name}_pred_no_pde_residual"
    with_col = f"{field_name}_pred_with_pde_residual"

    a_values = np.sort(point_df["A"].dropna().unique().astype(float))
    tau_values = np.sort(point_df["tau"].dropna().unique().astype(float))
    a, tau, true_z = pivot_field(point_df, true_col, a_values, tau_values)
    _, _, no_z = pivot_field(point_df, no_col, a_values, tau_values)
    _, _, with_z = pivot_field(point_df, with_col, a_values, tau_values)
    no_err = no_z - true_z
    with_err = with_z - true_z
    difference = with_z - no_z

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    imshow_field(axes[0, 0], a, tau, true_z, f"{field_name} truth")
    imshow_field(axes[0, 1], a, tau, no_z, f"{field_name} without PDE")
    imshow_field(axes[0, 2], a, tau, with_z, f"{field_name} with PDE")
    imshow_field(
        axes[1, 0], a, tau, no_err,
        f"Error: without PDE - truth", cmap="coolwarm", symmetric=True
    )
    imshow_field(
        axes[1, 1], a, tau, with_err,
        f"Error: with PDE - truth", cmap="coolwarm", symmetric=True
    )
    imshow_field(
        axes[1, 2], a, tau, difference,
        f"With PDE - without PDE", cmap="coolwarm", symmetric=True
    )
    fig.suptitle(f"{field_name} forward-prediction ablation", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residual_comparison(
        full_predictions: Dict[str, Dict[str, np.ndarray]],
        params: Tuple[float, float, float],
        preprocessing: Dict,
        save_path: Path,
) -> None:
    a_in = preprocessing["unified_a_grid"][1:-1]
    tau_in = preprocessing["unified_tau_grid"][1:-1]

    residuals = {}
    for model_name in MODEL_NAMES:
        residuals[model_name] = compute_residual_maps(
            full_predictions[model_name]["D1"][0],
            full_predictions[model_name]["D2"][0],
            params,
            preprocessing["unified_a_grid"],
            preprocessing["unified_tau_grid"],
        )[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    r1_no, r2_no = residuals["no_pde_residual"]
    r1_with, r2_with = residuals["with_pde_residual"]

    imshow_field(axes[0, 0], a_in, tau_in, r1_no, "R1 without PDE",
                 cmap="coolwarm", symmetric=True)
    imshow_field(axes[0, 1], a_in, tau_in, r1_with, "R1 with PDE",
                 cmap="coolwarm", symmetric=True)
    imshow_field(axes[0, 2], a_in, tau_in, r1_with - r1_no, "R1 difference",
                 cmap="coolwarm", symmetric=True)
    imshow_field(axes[1, 0], a_in, tau_in, r2_no, "R2 without PDE",
                 cmap="coolwarm", symmetric=True)
    imshow_field(axes[1, 1], a_in, tau_in, r2_with, "R2 with PDE",
                 cmap="coolwarm", symmetric=True)
    imshow_field(axes[1, 2], a_in, tau_in, r2_with - r2_no, "R2 difference",
                 cmap="coolwarm", symmetric=True)

    fig.suptitle("PDE residual-map comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def choose_profile_taus(point_df: pd.DataFrame, max_count: int) -> np.ndarray:
    taus = np.sort(point_df["tau"].dropna().unique())
    if len(taus) <= max_count:
        return taus
    indices = np.unique(np.linspace(0, len(taus) - 1, max_count).round().astype(int))
    return taus[indices]


def plot_profiles(point_df: pd.DataFrame, save_path: Path) -> None:
    selected_taus = choose_profile_taus(point_df, NUM_PROFILE_TAU)
    n = len(selected_taus)
    fig, axes = plt.subplots(n, 2, figsize=(15, max(4, 3.4 * n)), squeeze=False)

    for row_idx, tau_value in enumerate(selected_taus):
        group = point_df[np.isclose(point_df["tau"], tau_value)].sort_values("A")
        for col_idx, field in enumerate(("D1", "D2")):
            ax = axes[row_idx, col_idx]
            ax.plot(group["A"], group[f"{field}_true"], "o-", label="Truth", ms=3)
            ax.plot(
                group["A"], group[f"{field}_pred_no_pde_residual"],
                "--", label="Without PDE"
            )
            ax.plot(
                group["A"], group[f"{field}_pred_with_pde_residual"],
                "-.", label="With PDE"
            )
            ax.set_title(f"{field}, tau={tau_value:.6g}")
            ax.set_xlabel("A")
            ax.set_ylabel(field)
            ax.grid(True, alpha=0.3)
            ax.legend()

    fig.suptitle("Representative tau-profile comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_d2_positivity(
        full_predictions: Dict[str, Dict[str, np.ndarray]],
        preprocessing: Dict,
        save_path: Path,
) -> None:
    a = preprocessing["unified_a_grid"]
    tau = preprocessing["unified_tau_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    no = full_predictions["no_pde_residual"]["D2"][0]
    with_pde = full_predictions["with_pde_residual"]["D2"][0]
    imshow_field(axes[0], a, tau, np.minimum(no, 0.0), "Negative D2: without PDE")
    imshow_field(axes[1], a, tau, np.minimum(with_pde, 0.0), "Negative D2: with PDE")
    imshow_field(
        axes[2], a, tau, with_pde - no,
        "D2 with PDE - without PDE", cmap="coolwarm", symmetric=True
    )
    fig.suptitle("D2 positivity diagnostics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 9. 处理一个样本
# =============================================================================

def process_one_sample(
        sample: Dict[str, Any],
        models: Dict[str, PODDeepONet],
        preprocessing: Dict,
) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame]:
    sample_dir = OUTPUT_DIR / "samples" / sample["sample_id"]
    data_dir = sample_dir / "data"
    fig_dir = sample_dir / "figures"
    safe_mkdir(data_dir)
    safe_mkdir(fig_dir)

    params = sample["params"]
    point_df = sample["points"].copy()
    full_predictions = predict_full_fields(
        np.asarray(params).reshape(1, 3),
        models,
        preprocessing,
    )

    target_tau = point_df["tau"].to_numpy(dtype=np.float64)
    target_a = point_df["A"].to_numpy(dtype=np.float64)

    for model_name in MODEL_NAMES:
        for field_name in ("D1", "D2"):
            field = full_predictions[model_name][field_name][0]
            point_df[f"{field_name}_pred_{model_name}"] = interpolate_field_to_points(
                field,
                preprocessing["unified_tau_grid"],
                preprocessing["unified_a_grid"],
                target_tau,
                target_a,
            )
            point_df[f"{field_name}_error_{model_name}"] = (
                    point_df[f"{field_name}_pred_{model_name}"]
                    - point_df[f"{field_name}_true"]
            )

    point_df["D1_pred_difference_with_minus_without"] = (
            point_df["D1_pred_with_pde_residual"]
            - point_df["D1_pred_no_pde_residual"]
    )
    point_df["D2_pred_difference_with_minus_without"] = (
            point_df["D2_pred_with_pde_residual"]
            - point_df["D2_pred_no_pde_residual"]
    )

    point_df["D1_truth_valid"] = np.isfinite(point_df["D1_true"])
    point_df["D2_truth_valid"] = np.isfinite(point_df["D2_true"])
    point_df["truth_both_valid"] = point_df["D1_truth_valid"] & point_df["D2_truth_valid"]
    for model_name in MODEL_NAMES:
        point_df[f"D1_prediction_valid_{model_name}"] = np.isfinite(
            point_df[f"D1_pred_{model_name}"]
        )
        point_df[f"D2_prediction_valid_{model_name}"] = np.isfinite(
            point_df[f"D2_pred_{model_name}"]
        )
        point_df[f"D1_metric_valid_{model_name}"] = (
                point_df["D1_truth_valid"]
                & point_df[f"D1_prediction_valid_{model_name}"]
        )
        point_df[f"D2_metric_valid_{model_name}"] = (
                point_df["D2_truth_valid"]
                & point_df[f"D2_prediction_valid_{model_name}"]
        )
    point_df.insert(0, "sample_id", sample["sample_id"])
    point_df.insert(1, "filename", sample["filename"])
    point_df["nu"] = params[0]
    point_df["kappa"] = params[1]
    point_df["d_diffusion"] = params[2]

    per_tau_df = calculate_per_tau_metrics(point_df, sample["sample_id"])
    grid_diagnostics_df = build_target_grid_diagnostics(point_df)
    sample_metrics = calculate_sample_metrics(
        point_df, full_predictions, params, preprocessing, sample
    )
    for row in sample_metrics:
        row["target_grid_rows"] = int(len(point_df))
        row["target_tau_levels"] = int(point_df["tau"].nunique())
        row["target_a_levels"] = int(point_df["A"].nunique())
        row["d1_truth_valid_fraction"] = float(point_df["D1_truth_valid"].mean())
        row["d2_truth_valid_fraction"] = float(point_df["D2_truth_valid"].mean())
        row["truth_both_valid_fraction"] = float(point_df["truth_both_valid"].mean())
        row["d1_entire_missing_tau_layers"] = int(grid_diagnostics_df["d1_entire_layer_missing"].sum())
        row["d2_entire_missing_tau_layers"] = int(grid_diagnostics_df["d2_entire_layer_missing"].sum())

    # 保存对齐真实数据的逐点离散结果
    point_df.to_csv(
        data_dir / "pointwise_truth_and_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_tau_df.to_csv(
        data_dir / "per_tau_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grid_diagnostics_df.to_csv(
        data_dir / "target_grid_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(sample_metrics).to_csv(
        data_dir / "sample_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 按照标准化网格保存平坦化的 2D 场数据 CSV (优化方便第三方软件绘图)
    full_grid_df = full_grid_long_table(
        full_predictions, params, preprocessing, point_df=point_df
    )

    # 按照参照格式命名导出场数据
    field_csv_filename = f"field_comparison_data_{sample['sample_id']}.csv"
    full_grid_df.to_csv(
        data_dir / field_csv_filename,
        index=False,
        encoding="utf-8-sig",
    )
    # 同时在全局的 field_data_samples 目录保存一份
    global_fields_dir = OUTPUT_DIR / "field_data_samples"
    safe_mkdir(global_fields_dir)
    full_grid_df.to_csv(
        global_fields_dir / field_csv_filename,
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        data_dir / "full_model_grid_predictions.npz",
        params=np.asarray(params),
        a_grid=preprocessing["unified_a_grid"],
        tau_grid=preprocessing["unified_tau_grid"],
        D1_no_pde_residual=full_predictions["no_pde_residual"]["D1"][0],
        D2_no_pde_residual=full_predictions["no_pde_residual"]["D2"][0],
        D1_with_pde_residual=full_predictions["with_pde_residual"]["D1"][0],
        D2_with_pde_residual=full_predictions["with_pde_residual"]["D2"][0],
        y_scaled_no_pde_residual=full_predictions["no_pde_residual"]["y_scaled"][0],
        y_scaled_with_pde_residual=full_predictions["with_pde_residual"]["y_scaled"][0],
    )

    metric_dict = {
        "sample_information": {
            "sample_id": sample["sample_id"],
            "filename": sample["filename"],
            "format": sample["format"],
            "nu": params[0],
            "kappa": params[1],
            "d_diffusion": params[2],
        },
        "metrics": sample_metrics,
    }
    save_json(data_dir / "sample_metrics.json", metric_dict)

    if PLOT_EVERY_SAMPLE:
        plot_field_comparison(point_df, "D1", fig_dir / "D1_field_comparison.png")
        plot_field_comparison(point_df, "D2", fig_dir / "D2_field_comparison.png")
        plot_residual_comparison(
            full_predictions, params, preprocessing,
            fig_dir / "PDE_residual_comparison.png"
        )
        plot_profiles(point_df, fig_dir / "representative_tau_profiles.png")
        plot_d2_positivity(
            full_predictions, preprocessing,
            fig_dir / "D2_positivity_diagnostics.png"
        )

    return sample_metrics, per_tau_df, point_df


# =============================================================================
# 10. 全局汇总统计与绘图
# =============================================================================

def make_paired_metrics(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "sample_id", "filename", "data_format",
        "nu", "kappa", "d_diffusion",
    ]
    metric_cols = [
        c for c in sample_metrics_df.columns
        if c not in id_cols + ["model"]
    ]
    no = sample_metrics_df[
        sample_metrics_df["model"] == "no_pde_residual"
        ][id_cols + metric_cols].copy()
    with_pde = sample_metrics_df[
        sample_metrics_df["model"] == "with_pde_residual"
        ][id_cols + metric_cols].copy()

    paired = no.merge(
        with_pde,
        on=id_cols,
        suffixes=("_without_pde", "_with_pde"),
        how="inner",
    )

    for metric in [
        "d1_mse", "d1_rmse", "d1_mae", "d1_rel_l2",
        "d2_mse", "d2_rmse", "d2_mae", "d2_rel_l2",
        "total_mse", "pde_total_mse",
        "d2_negative_fraction_full_grid",
        "d2_negative_magnitude_mse",
    ]:
        a = f"{metric}_without_pde"
        b = f"{metric}_with_pde"
        if a in paired.columns and b in paired.columns:
            paired[f"{metric}_improvement_percent"] = [
                percent_improvement(x, y)
                for x, y in zip(paired[a], paired[b])
            ]
            paired[f"{metric}_with_minus_without"] = paired[b] - paired[a]
            paired[f"{metric}_winner"] = np.where(
                paired[b] < paired[a], "with_pde",
                np.where(paired[b] > paired[a], "without_pde", "tie")
            )
    return paired


def build_summary_statistics(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "d1_mse", "d1_rmse", "d1_mae", "d1_rel_l2", "d1_r2",
        "d2_mse", "d2_rmse", "d2_mae", "d2_rel_l2", "d2_r2",
        "total_mse", "pde_r1_mse", "pde_r2_mse", "pde_total_mse",
        "d2_negative_fraction_full_grid", "d2_min_full_grid",
        "d2_negative_magnitude_mse",
    ]
    rows = []
    for model_name, group in sample_metrics_df.groupby("model"):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append({
                "model": model_name,
                "metric": metric,
                "count": len(finite),
                "mean": finite.mean() if len(finite) else np.nan,
                "std": finite.std(ddof=1) if len(finite) > 1 else np.nan,
                "median": finite.median() if len(finite) else np.nan,
                "q25": finite.quantile(0.25) if len(finite) else np.nan,
                "q75": finite.quantile(0.75) if len(finite) else np.nan,
                "min": finite.min() if len(finite) else np.nan,
                "max": finite.max() if len(finite) else np.nan,
            })
    return pd.DataFrame(rows)


def build_improvement_statistics(paired_df: pd.DataFrame) -> pd.DataFrame:
    improvement_cols = [c for c in paired_df if c.endswith("_improvement_percent")]
    rows = []
    for col in improvement_cols:
        values = pd.to_numeric(paired_df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        base_metric = col.replace("_improvement_percent", "")
        winner_col = f"{base_metric}_winner"
        wins = paired_df[winner_col].value_counts() if winner_col in paired_df else {}
        rows.append({
            "metric": base_metric,
            "n": len(finite),
            "mean_improvement_percent": finite.mean() if len(finite) else np.nan,
            "median_improvement_percent": finite.median() if len(finite) else np.nan,
            "std_improvement_percent": finite.std(ddof=1) if len(finite) > 1 else np.nan,
            "with_pde_win_count": int(wins.get("with_pde", 0)),
            "without_pde_win_count": int(wins.get("without_pde", 0)),
            "tie_count": int(wins.get("tie", 0)),
            "with_pde_win_rate": float(wins.get("with_pde", 0) / len(paired_df))
            if len(paired_df) else np.nan,
        })
    return pd.DataFrame(rows)


def build_parameter_correlations(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["d1_rmse", "d2_rmse", "total_mse", "pde_total_mse"]
    for model_name, group in sample_metrics_df.groupby("model"):
        for param in ["nu", "kappa", "d_diffusion"]:
            for metric in metrics:
                pair = group[[param, metric]].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append({
                    "model": model_name,
                    "parameter": param,
                    "metric": metric,
                    "pearson_correlation": pair[param].corr(pair[metric], method="pearson")
                    if len(pair) >= 2 else np.nan,
                    "spearman_correlation": pair[param].corr(pair[metric], method="spearman")
                    if len(pair) >= 2 else np.nan,
                    "n": len(pair),
                })
    return pd.DataFrame(rows)


def build_wilcoxon_results(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["d1_rmse", "d2_rmse", "total_mse", "pde_total_mse"]:
        col_no = f"{metric}_without_pde"
        col_with = f"{metric}_with_pde"
        pair = paired_df[[col_no, col_with]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(pair) < 2 or np.allclose(pair[col_no], pair[col_with]):
            statistic = pvalue = np.nan
        else:
            try:
                result = wilcoxon(pair[col_with], pair[col_no], alternative="two-sided")
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic = pvalue = np.nan
        rows.append({
            "metric": metric,
            "n": len(pair),
            "wilcoxon_statistic": statistic,
            "p_value": pvalue,
            "median_without_pde": pair[col_no].median() if len(pair) else np.nan,
            "median_with_pde": pair[col_with].median() if len(pair) else np.nan,
            "median_improvement_percent": finite_median([
                percent_improvement(x, y)
                for x, y in zip(pair[col_no], pair[col_with])
            ]),
        })
    return pd.DataFrame(rows)


def plot_global_metric_boxplots(sample_metrics_df: pd.DataFrame, save_path: Path):
    metrics = ["d1_rmse", "d2_rmse", "total_mse", "pde_total_mse"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, metric in zip(axes.ravel(), metrics):
        groups = [
            sample_metrics_df.loc[
                sample_metrics_df["model"] == m, metric
            ].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            for m in MODEL_NAMES
        ]
        ax.boxplot(groups, tick_labels=["Without PDE", "With PDE"], showfliers=True)
        if all(np.all(g > 0) for g in groups if len(g)):
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Global error-distribution comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scatter(paired_df: pd.DataFrame, save_path: Path):
    metrics = ["d1_rmse", "d2_rmse", "total_mse", "pde_total_mse"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, metric in zip(axes.ravel(), metrics):
        x = paired_df[f"{metric}_without_pde"].to_numpy()
        y = paired_df[f"{metric}_with_pde"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, alpha=0.65)
        if len(x):
            low = min(np.min(x), np.min(y))
            high = max(np.max(x), np.max(y))
            if high > low:
                ax.plot([low, high], [low, high], "--", linewidth=1.2)
            if low > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
        ax.set_xlabel("Without PDE")
        ax.set_ylabel("With PDE")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Paired sample comparison: points below diagonal favor PDE", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_histograms(paired_df: pd.DataFrame, save_path: Path):
    metrics = ["d1_rmse", "d2_rmse", "total_mse", "pde_total_mse"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, metric in zip(axes.ravel(), metrics):
        values = paired_df[f"{metric}_improvement_percent"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1.2)
        ax.axvline(values.median() if len(values) else 0.0, linestyle=":", linewidth=1.5)
        ax.set_xlabel("Improvement (%)")
        ax.set_ylabel("Sample count")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Improvement distribution: positive values favor PDE", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_error_maps(sample_metrics_df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    params = ["nu", "kappa", "d_diffusion"]
    for row, model_name in enumerate(MODEL_NAMES):
        group = sample_metrics_df[sample_metrics_df["model"] == model_name]
        for col, param in enumerate(params):
            ax = axes[row, col]
            sc = ax.scatter(
                group[param], group["total_mse"],
                c=group["pde_total_mse"], alpha=0.75
            )
            ax.set_xlabel(param)
            ax.set_ylabel("total_mse")
            if np.all(group["total_mse"].dropna() > 0):
                ax.set_yscale("log")
            ax.set_title(f"{MODEL_LABELS[model_name]}: {param}")
            ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="PDE total MSE")
    fig.suptitle("Parameter–error relationships", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_tau(per_tau_df: pd.DataFrame, save_path: Path):
    grouped = (
        per_tau_df.groupby(["model", "tau"], as_index=False)
        .agg(
            mean_d1_rmse=("d1_rmse", "mean"),
            median_d1_rmse=("d1_rmse", "median"),
            mean_d2_rmse=("d2_rmse", "mean"),
            median_d2_rmse=("d2_rmse", "median"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for model_name in MODEL_NAMES:
        g = grouped[grouped["model"] == model_name].sort_values("tau")
        axes[0].plot(g["tau"], g["mean_d1_rmse"], marker="o",
                     label=MODEL_LABELS[model_name])
        axes[1].plot(g["tau"], g["mean_d2_rmse"], marker="o",
                     label=MODEL_LABELS[model_name])
    for ax, field in zip(axes, ("D1", "D2")):
        ax.set_xlabel("tau")
        ax.set_ylabel("Mean RMSE")
        ax.set_title(f"{field} error versus tau")
        ax.grid(True, alpha=0.3)
        ax.legend()
        positive = [line.get_ydata() for line in ax.lines]
        if positive and all(np.all(np.asarray(v) > 0) for v in positive):
            ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_summary_excel(
        path: Path,
        sample_metrics_df: pd.DataFrame,
        paired_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        improvement_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        wilcoxon_df: pd.DataFrame,
        per_tau_df: pd.DataFrame,
        failed_df: pd.DataFrame,
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sample_metrics_df.to_excel(writer, sheet_name="all_sample_metrics", index=False)
            paired_df.to_excel(writer, sheet_name="paired_metrics", index=False)
            summary_df.to_excel(writer, sheet_name="summary_statistics", index=False)
            improvement_df.to_excel(writer, sheet_name="improvement", index=False)
            correlation_df.to_excel(writer, sheet_name="parameter_corr", index=False)
            wilcoxon_df.to_excel(writer, sheet_name="wilcoxon", index=False)
            per_tau_df.to_excel(writer, sheet_name="per_tau_metrics", index=False)
            failed_df.to_excel(writer, sheet_name="failed_files", index=False)
    except ImportError:
        print("[提示] 未安装 openpyxl，已跳过 Excel 汇总；CSV 输出不受影响。")
    except Exception as exc:
        print(f"[提示] Excel 汇总保存失败：{exc}；CSV 输出不受影响。")


def build_analysis_report(
        sample_metrics_df: pd.DataFrame,
        paired_df: pd.DataFrame,
        improvement_df: pd.DataFrame,
        wilcoxon_df: pd.DataFrame,
        failed_df: pd.DataFrame,
        elapsed_seconds: float,
        preprocessing: Dict,
        config: Dict,
) -> str:
    lines = [
        "POD-DeepONet / POD-PINO 双模型前向预测 V2分析报告",
        "=" * 70,
        "",
        "1. 运行概况",
        f"- 成功样本数：{paired_df['sample_id'].nunique()}",
        f"- 失败文件数：{len(failed_df)}",
        f"- 设备：{DEVICE}",
        f"- 总耗时：{elapsed_seconds:.2f} s",
        f"- POD 模态数：{preprocessing['actual_num_modes']}",
        f"- POD 保留能量：{preprocessing.get('retained_energy', np.nan):.8f}",
        f"- 训练物理权重 lambda：{config.get('PHYSICS_LOSS_WEIGHT', 'unknown')}",
        "",
        "2. 两套模型说明",
        "- no_pde_residual：训练损失不含 PDE 物理残差，但保留 D2 非负约束。",
        "- with_pde_residual：训练损失包含归一化 PDE 物理残差，也保留 D2 非负约束。",
        "- 前向预测阶段仅加载两套最佳权重，不在输出端人为添加物理修正。",
        "",
        "3. 关键指标",
    ]
    for _, row in improvement_df.iterrows():
        lines.append(
            f"- {row['metric']}: 中位改善率="
            f"{row['median_improvement_percent']:.4g}%，"
            f"平均改善率={row['mean_improvement_percent']:.4g}%，"
            f"with-PDE 胜率={100 * row['with_pde_win_rate']:.2f}%"
        )

    lines.extend(["", "4. 配对非参数检验"])
    for _, row in wilcoxon_df.iterrows():
        lines.append(
            f"- {row['metric']}: n={int(row['n'])}, "
            f"p={row['p_value']:.6g}, "
            f"median(no PDE)={row['median_without_pde']:.6e}, "
            f"median(with PDE)={row['median_with_pde']:.6e}"
        )

    lines.extend([
        "",
        "5. 结果解释原则",
        "- 改善率为正，表示 with_pde_residual 的误差低于 no_pde_residual。",
        "- 配对散点落在 y=x 下方，表示该样本中物理残差模型更好。",
        "- PDE 残差下降不必然意味着数据拟合误差同步下降，应同时考察 D1、D2、PDE 和 D2 非负性。",
        "- 若目标 A 或 tau 超出训练网格且未允许外推，相应逐点预测会保存为 NaN。",
        "- V2 对每个字段都显式重建公共目标网格；整层真值缺失会保留为 NaN，不再删除 tau 层。",
        "- 参数相关性用于发现误差集中区，不直接证明因果关系。",
        "",
        "6. 建议优先检查的输出",
        "- field_data_samples/field_comparison_data_*.csv：标准二维网格插值场数据，极利于 Origin 绘图。",
        "- paired_sample_metrics.csv：逐样本两模型配对结果和改善率。",
        "- summary_statistics.csv：总体分布统计。",
        "- improvement_statistics.csv：胜率与改善率。",
        "- per_tau_metrics_all.csv：误差随 tau 的变化。",
        "- samples/<sample>/：每个样本的完整数据、场图和残差图。",
    ])
    return "\n".join(lines)


# =============================================================================
# 11. 主程序
# =============================================================================

def main() -> None:
    start_time = time.time()
    safe_mkdir(OUTPUT_DIR)
    safe_mkdir(OUTPUT_DIR / "samples")
    safe_mkdir(OUTPUT_DIR / "field_data_samples")
    safe_mkdir(OUTPUT_DIR / "summary_figures")
    safe_mkdir(OUTPUT_DIR / "summary_tables")

    print("=" * 88)
    print("POD-DeepONet / POD-PINO 双模型前向预测 V2")
    print(f"Device: {DEVICE}")
    print("=" * 88)

    preprocessing, config, models = load_training_assets()

    # 保存本次运行配置
    run_config = {
        "ABLATION_RESULT_DIR": ABLATION_RESULT_DIR,
        "TEST_DATA_DIR": TEST_DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "TEST_FILE_PATTERN": TEST_FILE_PATTERN,
        "MAX_TEST_FILES": MAX_TEST_FILES,
        "RANDOM_SEED": RANDOM_SEED,
        "SAVE_GLOBAL_POINTWISE_TABLE": SAVE_GLOBAL_POINTWISE_TABLE,
        "PLOT_EVERY_SAMPLE": PLOT_EVERY_SAMPLE,
        "NUM_PROFILE_TAU": NUM_PROFILE_TAU,
        "FIG_DPI": FIG_DPI,
        "ALLOW_EXTRAPOLATION": ALLOW_EXTRAPOLATION,
        "INFERENCE_BATCH_SIZE": INFERENCE_BATCH_SIZE,
        "DEVICE": str(DEVICE),
        "DTYPE": str(DTYPE),
        "TRAINING_CONFIG": config,
    }
    save_json(OUTPUT_DIR / "forward_run_config.json", run_config)

    files = sorted(TEST_DATA_DIR.glob(TEST_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"在 {TEST_DATA_DIR} 中未找到 {TEST_FILE_PATTERN}"
        )
    if MAX_TEST_FILES is not None and len(files) > MAX_TEST_FILES:
        rng = np.random.default_rng(RANDOM_SEED)
        selected = rng.choice(len(files), size=MAX_TEST_FILES, replace=False)
        files = [files[i] for i in sorted(selected)]

    print(f"待处理文件数：{len(files)}")

    all_sample_metrics: List[Dict] = []
    all_per_tau: List[pd.DataFrame] = []
    all_pointwise: List[pd.DataFrame] = []
    failures: List[Dict] = []

    for file_path in tqdm(files, desc="双模型前向预测"):
        try:
            sample = load_test_sample(file_path)
            sample_metrics, per_tau_df, point_df = process_one_sample(
                sample, models, preprocessing
            )
            all_sample_metrics.extend(sample_metrics)
            all_per_tau.append(per_tau_df)
            if SAVE_GLOBAL_POINTWISE_TABLE:
                all_pointwise.append(point_df)
        except Exception as exc:
            failures.append({
                "filename": file_path.name,
                "file_path": str(file_path),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"\n[警告] {file_path.name} 处理失败：{exc}")

    if not all_sample_metrics:
        failed_df = pd.DataFrame(failures)
        failed_df.to_csv(
            OUTPUT_DIR / "summary_tables" / "failed_files.csv",
            index=False, encoding="utf-8-sig"
        )
        raise RuntimeError("没有任何样本成功完成前向预测。")

    tables_dir = OUTPUT_DIR / "summary_tables"
    figures_dir = OUTPUT_DIR / "summary_figures"

    sample_metrics_df = pd.DataFrame(all_sample_metrics)
    per_tau_df = pd.concat(all_per_tau, ignore_index=True)
    failed_df = pd.DataFrame(failures)

    sample_metrics_df.to_csv(
        tables_dir / "all_sample_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    per_tau_df.to_csv(
        tables_dir / "per_tau_metrics_all.csv",
        index=False, encoding="utf-8-sig"
    )
    failed_df.to_csv(
        tables_dir / "failed_files.csv",
        index=False, encoding="utf-8-sig"
    )

    if SAVE_GLOBAL_POINTWISE_TABLE and all_pointwise:
        pd.concat(all_pointwise, ignore_index=True).to_csv(
            tables_dir / "pointwise_predictions_all.csv",
            index=False, encoding="utf-8-sig"
        )

    paired_df = make_paired_metrics(sample_metrics_df)
    summary_df = build_summary_statistics(sample_metrics_df)
    improvement_df = build_improvement_statistics(paired_df)
    correlation_df = build_parameter_correlations(sample_metrics_df)
    wilcoxon_df = build_wilcoxon_results(paired_df)

    paired_df.to_csv(
        tables_dir / "paired_sample_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        tables_dir / "summary_statistics.csv",
        index=False, encoding="utf-8-sig"
    )
    improvement_df.to_csv(
        tables_dir / "improvement_statistics.csv",
        index=False, encoding="utf-8-sig"
    )
    correlation_df.to_csv(
        tables_dir / "parameter_error_correlations.csv",
        index=False, encoding="utf-8-sig"
    )
    wilcoxon_df.to_csv(
        tables_dir / "paired_wilcoxon_tests.csv",
        index=False, encoding="utf-8-sig"
    )
    save_summary_excel(
        tables_dir / "forward_prediction_summary.xlsx",
        sample_metrics_df,
        paired_df,
        summary_df,
        improvement_df,
        correlation_df,
        wilcoxon_df,
        per_tau_df,
        failed_df,
    )

    # 汇总图
    plot_global_metric_boxplots(
        sample_metrics_df, figures_dir / "global_metric_boxplots.png"
    )
    plot_paired_scatter(
        paired_df, figures_dir / "paired_metric_scatter.png"
    )
    plot_improvement_histograms(
        paired_df, figures_dir / "improvement_histograms.png"
    )
    plot_parameter_error_maps(
        sample_metrics_df, figures_dir / "parameter_error_relationships.png"
    )
    plot_error_vs_tau(
        per_tau_df, figures_dir / "mean_error_vs_tau.png"
    )

    elapsed = time.time() - start_time
    report = build_analysis_report(
        sample_metrics_df=sample_metrics_df,
        paired_df=paired_df,
        improvement_df=improvement_df,
        wilcoxon_df=wilcoxon_df,
        failed_df=failed_df,
        elapsed_seconds=elapsed,
        preprocessing=preprocessing,
        config=config,
    )
    with (OUTPUT_DIR / "forward_prediction_report.txt").open(
            "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print("\n" + report)
    print("\n主要输出目录：")
    print(f"- 每组样本：{OUTPUT_DIR / 'samples'}")
    print(f"- 标准网格场数据：{OUTPUT_DIR / 'field_data_samples'}")
    print(f"- 汇总数据：{tables_dir}")
    print(f"- 汇总图片：{figures_dir}")
    print(f"- 自动报告：{OUTPUT_DIR / 'forward_prediction_report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n程序运行失败，异常如下：")
        traceback.print_exc()
        raise