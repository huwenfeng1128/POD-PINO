# -*- coding: utf-8 -*-
"""
公平比较：POD-DeepONet（无 PDE）vs PI-POD-DeepONet（含 PDE）的 KM 参数辨识
============================================================================

核心原则
--------
1. 两套模型使用完全相同的 KM 数据、A 权重、tau 对齐、目标函数、优化器、
   初始点、多起点策略、惩罚项、评价指标与绘图方式。
2. 两套模型只允许在“模型权重”上不同。推荐两套模型共享同一个
   preprocessing/scaler/POD 文件；脚本会检查预处理是否一致。
3. 不仅比较数据拟合误差，还定量比较：
   - A 方向一阶/二阶粗糙度；
   - tau 方向一阶/二阶粗糙度；
   - 总变差 TV；
   - PDE 残差；
   - D2 负值比例；
   - 参数辨识误差。
4. 输出逐点数据、逐 tau 指标、优化历史、配对统计、Excel 汇总和论文级图片。

建议目录结构（严格消融）
------------------------
ABLATION_RESULT_DIR/
    shared_preprocessing.pth
    config.json
    no_pde_residual/best_model.pth
    with_pde_residual/best_model.pth

KM 文件格式
-----------
至少包含：
    tau_index, tau_sec, A, D1_data, D2_data

参数标准值优先从 CSV 列：
    nu, kappa, d_diffusion
读取；若不存在，则从文件名解析：
    (nu,kappa,d).csv
或：
    data_nu_neg18_388_kappa_0_648_d_10_613.csv

依赖
----
numpy, pandas, scipy, matplotlib, torch, tqdm, openpyxl
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.signal import hilbert
from scipy.stats import wilcoxon
from tqdm import tqdm

import torch
import torch.nn as nn


# =============================================================================
# 1. 用户配置
# =============================================================================

# 严格消融训练输出目录
ABLATION_RESULT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet"
    r"\result\xiaorongshiyan\train_result_physics_ablation_2"
)

# 两套权重：除权重外，其他训练资产必须一致
MODEL_PATHS = {
    "without_pde": ABLATION_RESULT_DIR / "no_pde_residual" / "best_model.pth",
    "with_pde": ABLATION_RESULT_DIR / "with_pde_residual" / "best_model.pth",
}
SHARED_PREPROCESSING_PATH = ABLATION_RESULT_DIR / "shared_preprocessing.pth"
CONFIG_PATH = ABLATION_RESULT_DIR / "config.json"

# 同一批 KM 数据
KM_DATA_DIR = Path(r"D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4")
KM_FILE_PATTERN = "*.csv"

# 可选：用于构造 A 概率权重的原始仿真数据；不存在时使用均匀权重
SIM_DATA_DIR: Optional[Path] = Path(
    r"D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best"
)

OUTPUT_DIR = Path(
    r"D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\wendingxing\fair_km_identification_ablation"
)

# 测试文件数量；None 表示全部
MAX_FILES: Optional[int] = None
RANDOM_SEED = 2026

# 模型设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# 优化设置：两套模型完全共用
OPTIMIZER_METHOD = "Nelder-Mead"
OPTIMIZER_OPTIONS = {
    "maxiter": 3500,
    "xatol": 1e-9,
    "fatol": 1e-9,
    "adaptive": True,
    "disp": False,
}

# 多起点数量。使用同一组起点分别优化两套模型，降低局部极值影响。
N_MULTI_START = 6

# 参数边界；Nelder-Mead 无原生边界，因此目标函数中使用平滑外部惩罚
PARAM_BOUNDS = {
    "nu": (-50.0, 50.0),
    "kappa": (-20.0, 20.0),
    "d_diffusion": (0.0, 30.0),
}
BOUND_PENALTY_FACTOR = 1e6
D_NEGATIVE_PENALTY_FACTOR = 1e6

# 目标函数：
# normalized_mse = 每个 tau 下分别按 D1/D2 标准差归一化，再做加权 MSE
OBJECTIVE_MODE = "normalized_mse"
EPS_SCALE = 1e-12

# D1、D2 在总目标中的相同权重
FIELD_WEIGHTS = {"D1": 1.0, "D2": 1.0}

# tau 处理方式：
# "linear"：在训练 tau 网格上进行线性插值，推荐；
# "nearest"：取最近 tau 点，与旧代码兼容。
TAU_INTERPOLATION = "linear"

# 模型训练网格外不外推
ALLOW_EXTRAPOLATION = False

# 画图设置
FIG_DPI = 240
NUM_PROFILE_TAU = 10
PLOT_EVERY_SAMPLE = True
SAVE_EXCEL = True

MODEL_NAMES = ("without_pde", "with_pde")
MODEL_LABELS = {
    "without_pde": "Without PDE residual",
    "with_pde": "With PDE residual",
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


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=DEVICE)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def sanitize_name(name: str, max_length: int = 140) -> str:
    stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*\s]+', "_", stem)
    return stem[:max_length]


def json_converter(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Unsupported JSON type: {type(obj)}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_converter,
            allow_nan=False,
        )


def array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else np.nan


def percent_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return np.nan
    if abs(baseline) < 1e-30:
        return np.nan
    return float((baseline - candidate) / abs(baseline) * 100.0)


# =============================================================================
# 3. 与训练脚本一致的网络
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
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE),
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Dropout(dropout_rate),
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
        return coeffs @ self.pod_basis.T + self.y_mean_pod_scaled


# =============================================================================
# 4. 训练资产加载与公平性检查
# =============================================================================

@dataclass
class Assets:
    preprocessing: Dict[str, Any]
    config: Dict[str, Any]
    models: Dict[str, PODDeepONet]


def load_assets() -> Assets:
    required = [
        SHARED_PREPROCESSING_PATH,
        CONFIG_PATH,
        *MODEL_PATHS.values(),
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required assets:\n" + "\n".join(missing)
        )

    raw = safe_torch_load(SHARED_PREPROCESSING_PATH, map_location="cpu")
    preprocessing: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("actual_num_modes",):
            preprocessing[key] = int(value)
        elif key in ("retained_energy",):
            preprocessing[key] = float(value)
        else:
            preprocessing[key] = to_numpy(value)

    required_keys = [
        "branch_mean", "branch_std",
        "y_mean_scaler", "y_std_scaler",
        "pod_basis", "y_mean_pod_scaled",
        "actual_num_modes",
        "unified_a_grid", "unified_tau_grid",
    ]
    missing_keys = [k for k in required_keys if k not in preprocessing]
    if missing_keys:
        raise KeyError(f"shared_preprocessing missing keys: {missing_keys}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
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
        model.load_state_dict(
            safe_load_state_dict(MODEL_PATHS[model_name]),
            strict=True,
        )
        model.eval()
        models[model_name] = model

    n_tau = len(preprocessing["unified_tau_grid"])
    n_a = len(preprocessing["unified_a_grid"])
    expected = 2 * n_tau * n_a
    if len(preprocessing["y_mean_scaler"]) != expected:
        raise ValueError(
            f"Output dimension mismatch: expected {expected}, "
            f"got {len(preprocessing['y_mean_scaler'])}."
        )

    # 保存可复核的公平性指纹
    preprocessing["_fingerprints"] = {
        "a_grid_sha256": array_digest(preprocessing["unified_a_grid"]),
        "tau_grid_sha256": array_digest(preprocessing["unified_tau_grid"]),
        "pod_basis_sha256": array_digest(preprocessing["pod_basis"]),
        "branch_mean_sha256": array_digest(preprocessing["branch_mean"]),
        "branch_std_sha256": array_digest(preprocessing["branch_std"]),
        "y_mean_sha256": array_digest(preprocessing["y_mean_scaler"]),
        "y_std_sha256": array_digest(preprocessing["y_std_scaler"]),
    }
    return Assets(preprocessing, config, models)


# =============================================================================
# 5. KM 数据读取
# =============================================================================

def parse_params_from_filename(path: Path) -> Optional[Tuple[float, float, float]]:
    stem = path.stem.strip()

    old = re.search(
        r"\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        stem,
    )
    if old:
        return tuple(float(old.group(i)) for i in (1, 2, 3))

    pattern = (
        r"(?:data_)?nu_(neg)?(\d+)_(\d+)_"
        r"kappa_(neg)?(\d+)_(\d+)_"
        r"d_(neg)?(\d+)_(\d+)"
    )
    m = re.match(pattern, stem)
    if m:
        groups = m.groups()
        values = []
        for sign, integer, frac in (
            groups[0:3], groups[3:6], groups[6:9]
        ):
            val = float(f"{integer}.{frac}")
            if sign == "neg":
                val = -val
            values.append(val)
        return tuple(values)
    return None


def extract_standard_params(
    df: pd.DataFrame,
    file_path: Path,
) -> Tuple[float, float, float]:
    cols = ["nu", "kappa", "d_diffusion"]
    if all(c in df.columns for c in cols):
        out = []
        for c in cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(vals) != 1:
                raise ValueError(f"{c} is not a unique constant in {file_path.name}")
            out.append(float(vals[0]))
        return tuple(out)

    parsed = parse_params_from_filename(file_path)
    if parsed is None:
        return (np.nan, np.nan, np.nan)
    return parsed


@dataclass
class KMSample:
    file_path: Path
    sample_id: str
    standard_params: Tuple[float, float, float]
    points: pd.DataFrame
    a_values: np.ndarray
    tau_values: np.ndarray
    tau_indices: np.ndarray
    a_weights: Dict[float, float]


def build_a_weights(
    a_values: np.ndarray,
    km_filename: str,
) -> Dict[float, float]:
    # 默认均匀权重，且归一化到和为 1
    weights = np.ones(len(a_values), dtype=float)

    if SIM_DATA_DIR is not None:
        sim_path = SIM_DATA_DIR / km_filename
        if sim_path.is_file():
            try:
                df = pd.read_csv(sim_path)
                if "Eta" in df.columns:
                    envelope = np.abs(hilbert(
                        pd.to_numeric(df["Eta"], errors="coerce").dropna().to_numpy()
                    ))
                elif "Envelope" in df.columns:
                    envelope = pd.to_numeric(
                        df["Envelope"], errors="coerce"
                    ).dropna().to_numpy()
                else:
                    envelope = np.array([])

                if len(envelope):
                    if len(a_values) > 1:
                        mids = 0.5 * (a_values[:-1] + a_values[1:])
                        first = a_values[0] - (mids[0] - a_values[0])
                        last = a_values[-1] + (a_values[-1] - mids[-1])
                        edges = np.concatenate([[first], mids, [last]])
                    else:
                        edges = np.array([a_values[0] - 0.5, a_values[0] + 0.5])

                    hist, _ = np.histogram(envelope, bins=edges, density=False)
                    hist = hist.astype(float)
                    if hist.sum() > 0:
                        weights = hist
            except Exception as exc:
                print(f"[Warning] A weights fallback to uniform for {km_filename}: {exc}")

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 0:
        weights[:] = 1.0
    weights /= weights.sum()
    return {float(a): float(w) for a, w in zip(a_values, weights)}


def load_km_sample(file_path: Path) -> KMSample:
    df = pd.read_csv(file_path)
    required = ["tau_index", "tau_sec", "A", "D1_data", "D2_data"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing KM columns: {missing}")

    work = df[required].copy()
    work.columns = ["tau_index", "tau", "A", "D1_true", "D2_true"]
    for c in ["tau_index", "tau", "A", "D1_true", "D2_true"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["tau_index", "tau", "A"]).copy()

    # 重复点取均值
    work = (
        work.groupby(["tau_index", "tau", "A"], as_index=False)
        .agg(D1_true=("D1_true", "mean"), D2_true=("D2_true", "mean"))
        .sort_values(["tau", "A"])
        .reset_index(drop=True)
    )
    if work.empty:
        raise ValueError("No valid KM points")

    a_values = np.sort(work["A"].unique().astype(float))
    tau_table = (
        work[["tau_index", "tau"]]
        .drop_duplicates()
        .sort_values("tau")
    )
    tau_values = tau_table["tau"].to_numpy(dtype=float)
    tau_indices = tau_table["tau_index"].to_numpy(dtype=int)

    return KMSample(
        file_path=file_path,
        sample_id=sanitize_name(file_path.name),
        standard_params=extract_standard_params(df, file_path),
        points=work,
        a_values=a_values,
        tau_values=tau_values,
        tau_indices=tau_indices,
        a_weights=build_a_weights(a_values, file_path.name),
    )


# =============================================================================
# 6. 模型前向预测与统一插值
# =============================================================================

def predict_full_field(
    params: Sequence[float],
    model: PODDeepONet,
    preprocessing: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    params_np = np.asarray(params, dtype=np.float64).reshape(1, 3)
    mean = preprocessing["branch_mean"]
    std = preprocessing["branch_std"].copy()
    std[np.abs(std) < 1e-10] = 1.0
    x_scaled = (params_np - mean) / std

    x = torch.as_tensor(x_scaled, dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        y_scaled = model(x).cpu().numpy()[0]
    y = y_scaled * preprocessing["y_std_scaler"] + preprocessing["y_mean_scaler"]

    n_tau = len(preprocessing["unified_tau_grid"])
    n_a = len(preprocessing["unified_a_grid"])
    n_field = n_tau * n_a
    D1 = y[:n_field].reshape(n_tau, n_a)
    D2 = y[n_field:].reshape(n_tau, n_a)
    return D1, D2


def interpolate_to_points(
    field: np.ndarray,
    source_tau: np.ndarray,
    source_a: np.ndarray,
    target_tau: np.ndarray,
    target_a: np.ndarray,
) -> np.ndarray:
    fill_value = None if ALLOW_EXTRAPOLATION else np.nan

    if TAU_INTERPOLATION == "nearest":
        tau_indices = np.argmin(
            np.abs(source_tau[:, None] - target_tau[None, :]),
            axis=0,
        )
        values = np.empty(len(target_tau), dtype=float)
        for i, (ti, a) in enumerate(zip(tau_indices, target_a)):
            values[i] = np.interp(
                a,
                source_a,
                field[ti],
                left=np.nan if not ALLOW_EXTRAPOLATION else None,
                right=np.nan if not ALLOW_EXTRAPOLATION else None,
            )
        return values

    interpolator = RegularGridInterpolator(
        (source_tau, source_a),
        field,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return np.asarray(
        interpolator(np.column_stack([target_tau, target_a])),
        dtype=float,
    )


def predict_at_km_points(
    params: Sequence[float],
    model: PODDeepONet,
    sample: KMSample,
    preprocessing: Dict[str, Any],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    D1_full, D2_full = predict_full_field(params, model, preprocessing)
    pts = sample.points.copy()
    target_tau = pts["tau"].to_numpy(dtype=float)
    target_a = pts["A"].to_numpy(dtype=float)
    source_tau = preprocessing["unified_tau_grid"]
    source_a = preprocessing["unified_a_grid"]

    pts["D1_pred"] = interpolate_to_points(
        D1_full, source_tau, source_a, target_tau, target_a
    )
    pts["D2_pred"] = interpolate_to_points(
        D2_full, source_tau, source_a, target_tau, target_a
    )
    return pts, D1_full, D2_full


# =============================================================================
# 7. 公平的辨识目标函数
# =============================================================================

def bound_penalty(params: Sequence[float]) -> float:
    total = 0.0
    for value, key in zip(params, ("nu", "kappa", "d_diffusion")):
        low, high = PARAM_BOUNDS[key]
        if value < low:
            total += (low - value) ** 2
        elif value > high:
            total += (value - high) ** 2
    if params[2] < 0:
        total += D_NEGATIVE_PENALTY_FACTOR / BOUND_PENALTY_FACTOR * (-params[2]) ** 2
    return BOUND_PENALTY_FACTOR * total


def objective_from_prediction(
    point_df: pd.DataFrame,
    a_weights: Dict[float, float],
) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    total_weight = 0.0
    d1_acc = 0.0
    d2_acc = 0.0

    for _, group in point_df.groupby("tau", sort=True):
        d1_true = group["D1_true"].to_numpy(dtype=float)
        d2_true = group["D2_true"].to_numpy(dtype=float)
        d1_pred = group["D1_pred"].to_numpy(dtype=float)
        d2_pred = group["D2_pred"].to_numpy(dtype=float)
        a = group["A"].to_numpy(dtype=float)

        valid = (
            np.isfinite(d1_true) & np.isfinite(d2_true) &
            np.isfinite(d1_pred) & np.isfinite(d2_pred)
        )
        if not np.any(valid):
            continue

        d1_true = d1_true[valid]
        d2_true = d2_true[valid]
        d1_pred = d1_pred[valid]
        d2_pred = d2_pred[valid]
        a = a[valid]

        w = np.asarray([a_weights.get(float(x), 0.0) for x in a], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        if OBJECTIVE_MODE == "normalized_mse":
            s1 = np.nanstd(d1_true)
            s2 = np.nanstd(d2_true)
            if not np.isfinite(s1) or s1 < EPS_SCALE:
                s1 = 1.0
            if not np.isfinite(s2) or s2 < EPS_SCALE:
                s2 = 1.0
        else:
            s1 = s2 = 1.0

        e1 = ((d1_pred - d1_true) / s1) ** 2
        e2 = ((d2_pred - d2_true) / s2) ** 2

        d1_term = float(np.sum(w * e1))
        d2_term = float(np.sum(w * e2))
        tau_term = FIELD_WEIGHTS["D1"] * d1_term + FIELD_WEIGHTS["D2"] * d2_term

        total += tau_term
        d1_acc += d1_term
        d2_acc += d2_term
        total_weight += 1.0

    if total_weight == 0:
        return 1e12, {"data_cost": 1e12, "d1_cost": np.nan, "d2_cost": np.nan}

    return (
        total / total_weight,
        {
            "data_cost": total / total_weight,
            "d1_cost": d1_acc / total_weight,
            "d2_cost": d2_acc / total_weight,
        },
    )


def make_objective(
    model: PODDeepONet,
    sample: KMSample,
    preprocessing: Dict[str, Any],
    history: List[Dict[str, float]],
):
    def objective(params: np.ndarray) -> float:
        iteration = len(history) + 1
        penalty = bound_penalty(params)
        try:
            pred_df, _, _ = predict_at_km_points(
                params, model, sample, preprocessing
            )
            data_cost, details = objective_from_prediction(
                pred_df, sample.a_weights
            )
            total_cost = data_cost + penalty
        except Exception:
            data_cost = 1e12
            total_cost = 1e12 + penalty
            details = {"d1_cost": np.nan, "d2_cost": np.nan}

        history.append({
            "iteration": iteration,
            "nu": float(params[0]),
            "kappa": float(params[1]),
            "d_diffusion": float(params[2]),
            "d1_cost": float(details.get("d1_cost", np.nan)),
            "d2_cost": float(details.get("d2_cost", np.nan)),
            "data_cost": float(data_cost),
            "bound_penalty": float(penalty),
            "total_cost": float(total_cost),
        })
        return float(total_cost) if np.isfinite(total_cost) else 1e12

    return objective


def generate_initial_points(
    sample: KMSample,
) -> List[np.ndarray]:
    d2_mean = float(np.nanmean(sample.points["D2_true"]))
    if not np.isfinite(d2_mean):
        d2_mean = 1.0

    base = np.array([0.1, 0.1, max(0.01, d2_mean)], dtype=float)
    starts = [base]

    rng = np.random.default_rng(RANDOM_SEED)
    scales = np.array([3.0, 1.0, max(0.5, 0.35 * abs(base[2]))])
    while len(starts) < N_MULTI_START:
        candidate = base + rng.normal(size=3) * scales
        candidate[2] = max(0.01, candidate[2])
        for i, key in enumerate(("nu", "kappa", "d_diffusion")):
            low, high = PARAM_BOUNDS[key]
            candidate[i] = np.clip(candidate[i], low, high)
        starts.append(candidate)
    return starts


@dataclass
class IdentificationResult:
    model_name: str
    params: np.ndarray
    objective: float
    success: bool
    message: str
    elapsed_seconds: float
    history: pd.DataFrame
    prediction_points: pd.DataFrame
    D1_full: np.ndarray
    D2_full: np.ndarray
    start_results: pd.DataFrame


def identify_one_model(
    model_name: str,
    model: PODDeepONet,
    sample: KMSample,
    preprocessing: Dict[str, Any],
    initial_points: Sequence[np.ndarray],
) -> IdentificationResult:
    start_time = time.time()
    all_histories: List[pd.DataFrame] = []
    start_rows: List[Dict[str, Any]] = []
    best_payload = None

    for start_id, x0 in enumerate(initial_points):
        history: List[Dict[str, float]] = []
        objective = make_objective(model, sample, preprocessing, history)
        result = minimize(
            objective,
            np.asarray(x0, dtype=float),
            method=OPTIMIZER_METHOD,
            options=OPTIMIZER_OPTIONS,
        )

        hist_df = pd.DataFrame(history)
        hist_df.insert(0, "start_id", start_id)
        all_histories.append(hist_df)

        if len(hist_df):
            idx = int(hist_df["total_cost"].idxmin())
            best_hist_row = hist_df.loc[idx]
            candidate_params = np.array([
                best_hist_row["nu"],
                best_hist_row["kappa"],
                best_hist_row["d_diffusion"],
            ])
            candidate_cost = float(best_hist_row["total_cost"])
        else:
            candidate_params = np.asarray(result.x, dtype=float)
            candidate_cost = float(result.fun)

        start_rows.append({
            "start_id": start_id,
            "x0_nu": float(x0[0]),
            "x0_kappa": float(x0[1]),
            "x0_d": float(x0[2]),
            "result_success": bool(result.success),
            "result_message": str(result.message),
            "result_fun": float(result.fun),
            "best_history_cost": candidate_cost,
            "best_nu": float(candidate_params[0]),
            "best_kappa": float(candidate_params[1]),
            "best_d": float(candidate_params[2]),
            "nfev": int(getattr(result, "nfev", -1)),
            "nit": int(getattr(result, "nit", -1)),
        })

        if best_payload is None or candidate_cost < best_payload["cost"]:
            best_payload = {
                "params": candidate_params,
                "cost": candidate_cost,
                "success": bool(result.success),
                "message": str(result.message),
            }

    assert best_payload is not None
    params = best_payload["params"].copy()
    params[2] = max(0.0, params[2])

    prediction_points, D1_full, D2_full = predict_at_km_points(
        params, model, sample, preprocessing
    )

    return IdentificationResult(
        model_name=model_name,
        params=params,
        objective=float(best_payload["cost"]),
        success=bool(best_payload["success"]),
        message=str(best_payload["message"]),
        elapsed_seconds=time.time() - start_time,
        history=pd.concat(all_histories, ignore_index=True),
        prediction_points=prediction_points,
        D1_full=D1_full,
        D2_full=D2_full,
        start_results=pd.DataFrame(start_rows),
    )


# =============================================================================
# 8. 误差、平滑性和 PDE 残差指标
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {
            "n": 0, "mse": np.nan, "rmse": np.nan, "mae": np.nan,
            "max_abs": np.nan, "bias": np.nan, "rel_l2": np.nan, "r2": np.nan,
        }
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    err = yp - yt
    denom = np.linalg.norm(yt)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return {
        "n": int(len(yt)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "rel_l2": float(np.linalg.norm(err) / denom) if denom > 1e-30 else np.nan,
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-30 else np.nan,
    }


def nonuniform_first_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, x, edge_order=2 if len(x) >= 3 else 1)


def nonuniform_second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.full_like(y, np.nan, dtype=float)
    first = np.gradient(y, x, edge_order=2)
    return np.gradient(first, x, edge_order=2)


def smoothness_metrics_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 3:
        return {
            "first_derivative_rms": np.nan,
            "second_derivative_rms": np.nan,
            "second_derivative_mae": np.nan,
            "total_variation": np.nan,
            "normalized_total_variation": np.nan,
            "curvature_energy": np.nan,
            "sign_change_rate_second_derivative": np.nan,
        }

    order = np.argsort(x)
    x, y = x[order], y[order]
    d1 = nonuniform_first_derivative(x, y)
    d2 = nonuniform_second_derivative(x, y)
    tv = float(np.sum(np.abs(np.diff(y))))
    amplitude = float(np.nanmax(y) - np.nanmin(y))
    signs = np.sign(d2)
    sign_changes = np.sum(signs[1:] * signs[:-1] < 0)

    return {
        "first_derivative_rms": float(np.sqrt(np.mean(d1 ** 2))),
        "second_derivative_rms": float(np.sqrt(np.mean(d2 ** 2))),
        "second_derivative_mae": float(np.mean(np.abs(d2))),
        "total_variation": tv,
        "normalized_total_variation": tv / max(amplitude, 1e-12),
        "curvature_energy": float(np.trapz(d2 ** 2, x)),
        "sign_change_rate_second_derivative": float(sign_changes / max(len(d2) - 1, 1)),
    }


def compute_pde_residual_maps(
    D1: np.ndarray,
    D2: np.ndarray,
    params: Sequence[float],
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nu, kappa, d_diff = params
    if len(a_grid) < 3 or len(tau_grid) < 3:
        raise ValueError("PDE residual requires at least 3 A and 3 tau points")

    a = a_grid[None, :]
    tau = tau_grid[:, None]
    a_safe = np.maximum(np.abs(a), 1e-9)

    d1_th = nu * a - (kappa / 8.0) * a ** 3 + d_diff / a_safe
    d2_th = np.full_like(d1_th, d_diff)

    u1 = tau * D1
    u2 = tau * D2

    # np.gradient 支持非均匀网格
    du1_dtau = np.gradient(u1, tau_grid, axis=0, edge_order=2)
    du2_dtau = np.gradient(u2, tau_grid, axis=0, edge_order=2)
    du1_da = np.gradient(u1, a_grid, axis=1, edge_order=2)
    du2_da = np.gradient(u2, a_grid, axis=1, edge_order=2)
    d2u1_da2 = np.gradient(du1_da, a_grid, axis=1, edge_order=2)
    d2u2_da2 = np.gradient(du2_da, a_grid, axis=1, edge_order=2)

    r1 = du1_dtau - d1_th * du1_da - d2_th * d2u1_da2 - d1_th
    r2 = (
        du2_dtau
        - d1_th * du2_da
        - d2_th * d2u2_da2
        - d1_th * u1
        - 2.0 * d2_th * du1_da
        - d2_th
    )
    return r1[1:-1, 1:-1], r2[1:-1, 1:-1]


def build_pointwise_comparison(
    sample: KMSample,
    results: Dict[str, IdentificationResult],
) -> pd.DataFrame:
    out = sample.points.copy()
    for model_name, result in results.items():
        pred = result.prediction_points[
            ["tau_index", "tau", "A", "D1_pred", "D2_pred"]
        ].copy()
        pred = pred.rename(columns={
            "D1_pred": f"D1_pred_{model_name}",
            "D2_pred": f"D2_pred_{model_name}",
        })
        out = out.merge(pred, on=["tau_index", "tau", "A"], how="left")
        for field in ("D1", "D2"):
            out[f"{field}_error_{model_name}"] = (
                out[f"{field}_pred_{model_name}"] - out[f"{field}_true"]
            )

    out["D1_with_minus_without"] = (
        out["D1_pred_with_pde"] - out["D1_pred_without_pde"]
    )
    out["D2_with_minus_without"] = (
        out["D2_pred_with_pde"] - out["D2_pred_without_pde"]
    )
    return out


def build_per_tau_metrics(
    pointwise: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for tau_value, group in pointwise.groupby("tau", sort=True):
        tau_index = int(group["tau_index"].iloc[0])
        for model_name in MODEL_NAMES:
            row = {
                "tau": float(tau_value),
                "tau_index": tau_index,
                "model": model_name,
            }
            for field in ("D1", "D2"):
                metrics = regression_metrics(
                    group[f"{field}_true"].to_numpy(dtype=float),
                    group[f"{field}_pred_{model_name}"].to_numpy(dtype=float),
                )
                for key, value in metrics.items():
                    row[f"{field.lower()}_{key}"] = value

                smooth = smoothness_metrics_1d(
                    group["A"].to_numpy(dtype=float),
                    group[f"{field}_pred_{model_name}"].to_numpy(dtype=float),
                )
                for key, value in smooth.items():
                    row[f"{field.lower()}_{key}"] = value

            row["total_mse"] = row["d1_mse"] + row["d2_mse"]
            row["combined_curvature_energy"] = (
                row["d1_curvature_energy"] + row["d2_curvature_energy"]
            )
            row["combined_second_derivative_rms"] = math.sqrt(
                row["d1_second_derivative_rms"] ** 2
                + row["d2_second_derivative_rms"] ** 2
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_sample_metrics(
    sample: KMSample,
    pointwise: pd.DataFrame,
    per_tau: pd.DataFrame,
    results: Dict[str, IdentificationResult],
    preprocessing: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    std_nu, std_kappa, std_d = sample.standard_params
    a_grid = preprocessing["unified_a_grid"]
    tau_grid = preprocessing["unified_tau_grid"]

    for model_name, result in results.items():
        row: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "filename": sample.file_path.name,
            "model": model_name,
            "nu_standard": std_nu,
            "kappa_standard": std_kappa,
            "d_standard": std_d,
            "nu_identified": result.params[0],
            "kappa_identified": result.params[1],
            "d_identified": result.params[2],
            "nu_abs_error": abs(result.params[0] - std_nu) if np.isfinite(std_nu) else np.nan,
            "kappa_abs_error": abs(result.params[1] - std_kappa) if np.isfinite(std_kappa) else np.nan,
            "d_abs_error": abs(result.params[2] - std_d) if np.isfinite(std_d) else np.nan,
            "objective": result.objective,
            "optimizer_success": result.success,
            "optimizer_message": result.message,
            "elapsed_seconds": result.elapsed_seconds,
        }

        for field in ("D1", "D2"):
            metrics = regression_metrics(
                pointwise[f"{field}_true"].to_numpy(dtype=float),
                pointwise[f"{field}_pred_{model_name}"].to_numpy(dtype=float),
            )
            for key, value in metrics.items():
                row[f"{field.lower()}_{key}"] = value

        subset = per_tau[per_tau["model"] == model_name]
        smooth_cols = [
            c for c in subset.columns
            if (
                "derivative" in c or
                "variation" in c or
                "curvature" in c
            )
        ]
        for c in smooth_cols:
            row[f"mean_{c}"] = finite_mean(subset[c])
            row[f"median_{c}"] = finite_median(subset[c])

        r1, r2 = compute_pde_residual_maps(
            result.D1_full,
            result.D2_full,
            result.params,
            a_grid,
            tau_grid,
        )
        row["pde_r1_mse"] = float(np.mean(r1 ** 2))
        row["pde_r2_mse"] = float(np.mean(r2 ** 2))
        row["pde_total_mse"] = row["pde_r1_mse"] + row["pde_r2_mse"]
        row["d2_negative_fraction"] = float(np.mean(result.D2_full < 0))
        row["d2_min"] = float(np.min(result.D2_full))
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 9. 单样本可视化
# =============================================================================

def choose_profile_taus(tau_values: np.ndarray, max_count: int) -> np.ndarray:
    tau_values = np.sort(np.asarray(tau_values, dtype=float))
    if len(tau_values) <= max_count:
        return tau_values
    idx = np.unique(
        np.linspace(0, len(tau_values) - 1, max_count)
        .round()
        .astype(int)
    )
    return tau_values[idx]


def theoretical_D1(
    A: np.ndarray,
    params: Sequence[float],
) -> np.ndarray:
    nu, kappa, d = params
    a_safe = np.maximum(np.abs(A), 1e-9)
    return nu * A - (kappa / 8.0) * A ** 3 + d / a_safe


def theoretical_D2(
    A: np.ndarray,
    params: Sequence[float],
) -> np.ndarray:
    return np.full_like(A, float(params[2]), dtype=float)


def plot_profile_matrix(
    sample: KMSample,
    pointwise: pd.DataFrame,
    results: Dict[str, IdentificationResult],
    save_path: Path,
) -> None:
    taus = choose_profile_taus(sample.tau_values, NUM_PROFILE_TAU)
    n = len(taus)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(15, max(4.5, 3.15 * n)),
        squeeze=False,
    )

    for i, tau in enumerate(taus):
        g = pointwise[np.isclose(pointwise["tau"], tau)].sort_values("A")
        A = g["A"].to_numpy(dtype=float)
        for j, field in enumerate(("D1", "D2")):
            ax = axes[i, j]
            ax.scatter(
                A, g[f"{field}_true"],
                s=16, marker="o", label="KM data", zorder=4
            )
            ax.plot(
                A, g[f"{field}_pred_without_pde"],
                linewidth=1.6, label="Without PDE"
            )
            ax.plot(
                A, g[f"{field}_pred_with_pde"],
                linewidth=2.0, linestyle="--", label="With PDE"
            )

            if all(np.isfinite(sample.standard_params)):
                theo = (
                    theoretical_D1(A, sample.standard_params)
                    if field == "D1"
                    else theoretical_D2(A, sample.standard_params)
                )
                ax.plot(
                    A, theo, linewidth=1.2, linestyle=":",
                    label="Theoretical standard"
                )

            ax.set_title(f"{field}, tau={tau:.6g} s")
            ax.set_xlabel("A")
            ax.set_ylabel(field)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.legend(ncol=2, fontsize=8)

    title = (
        "Fair KM identification comparison\n"
        f"Without PDE: ({results['without_pde'].params[0]:.4g}, "
        f"{results['without_pde'].params[1]:.4g}, "
        f"{results['without_pde'].params[2]:.4g}) | "
        f"With PDE: ({results['with_pde'].params[0]:.4g}, "
        f"{results['with_pde'].params[1]:.4g}, "
        f"{results['with_pde'].params[2]:.4g})"
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_second_derivative_profiles(
    sample: KMSample,
    pointwise: pd.DataFrame,
    save_path: Path,
) -> None:
    taus = choose_profile_taus(sample.tau_values, min(NUM_PROFILE_TAU, 6))
    fig, axes = plt.subplots(
        len(taus), 2,
        figsize=(15, max(4.5, 3.2 * len(taus))),
        squeeze=False,
    )

    for i, tau in enumerate(taus):
        g = pointwise[np.isclose(pointwise["tau"], tau)].sort_values("A")
        A = g["A"].to_numpy(dtype=float)
        for j, field in enumerate(("D1", "D2")):
            ax = axes[i, j]
            for model_name, linestyle in (
                ("without_pde", "-"),
                ("with_pde", "--"),
            ):
                y = g[f"{field}_pred_{model_name}"].to_numpy(dtype=float)
                d2 = nonuniform_second_derivative(A, y)
                ax.plot(
                    A, d2,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=MODEL_LABELS[model_name],
                )
            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(
                rf"$\partial^2 {field}/\partial A^2$, tau={tau:.6g} s"
            )
            ax.set_xlabel("A")
            ax.set_ylabel("Second derivative")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Direct smoothness evidence: second derivatives versus A",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness_vs_tau(
    per_tau: pd.DataFrame,
    save_path: Path,
) -> None:
    metrics = [
        ("combined_second_derivative_rms", "Combined second-derivative RMS"),
        ("combined_curvature_energy", "Combined curvature energy"),
        ("d1_normalized_total_variation", "D1 normalized total variation"),
        ("d2_normalized_total_variation", "D2 normalized total variation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for model_name, linestyle in (
            ("without_pde", "-"),
            ("with_pde", "--"),
        ):
            g = per_tau[per_tau["model"] == model_name].sort_values("tau")
            ax.plot(
                g["tau"], g[metric],
                marker="o", markersize=3,
                linestyle=linestyle,
                label=MODEL_LABELS[model_name],
            )
        ax.set_title(title)
        ax.set_xlabel("tau (s)")
        ax.set_ylabel(metric)
        positive = per_tau[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(positive) and np.all(positive > 0):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Quantitative smoothness comparison across finite-time KM coefficients",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_error_smoothness_tradeoff(
    per_tau: pd.DataFrame,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, rough_metric, field in (
        (axes[0], "d1_second_derivative_rms", "D1"),
        (axes[1], "d2_second_derivative_rms", "D2"),
    ):
        for model_name, marker in (("without_pde", "o"), ("with_pde", "^")):
            g = per_tau[per_tau["model"] == model_name]
            ax.scatter(
                g[f"{field.lower()}_rmse"],
                g[rough_metric],
                marker=marker,
                alpha=0.8,
                label=MODEL_LABELS[model_name],
            )
        ax.set_xlabel(f"{field} RMSE")
        ax.set_ylabel(f"{field} second-derivative RMS")
        ax.set_title(f"{field}: accuracy–smoothness trade-off")
        ax.grid(alpha=0.25)
        ax.legend()
        x = per_tau[f"{field.lower()}_rmse"].dropna()
        y = per_tau[rough_metric].dropna()
        if len(x) and np.all(x > 0):
            ax.set_xscale("log")
        if len(y) and np.all(y > 0):
            ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pde_residual_maps(
    results: Dict[str, IdentificationResult],
    preprocessing: Dict[str, Any],
    save_path: Path,
) -> None:
    a = preprocessing["unified_a_grid"][1:-1]
    tau = preprocessing["unified_tau_grid"][1:-1]

    residuals = {}
    for model_name, result in results.items():
        residuals[model_name] = compute_pde_residual_maps(
            result.D1_full,
            result.D2_full,
            result.params,
            preprocessing["unified_a_grid"],
            preprocessing["unified_tau_grid"],
        )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, residual_name in enumerate(("R1", "R2")):
        z_no = residuals["without_pde"][row]
        z_with = residuals["with_pde"][row]
        z_diff = z_with - z_no
        for col, (z, title) in enumerate((
            (z_no, f"{residual_name}: without PDE"),
            (z_with, f"{residual_name}: with PDE"),
            (z_diff, f"{residual_name}: with - without"),
        )):
            finite = z[np.isfinite(z)]
            kwargs = {}
            if len(finite):
                vmax = np.max(np.abs(finite))
                if vmax > 0:
                    kwargs["norm"] = TwoSlopeNorm(
                        vmin=-vmax, vcenter=0.0, vmax=vmax
                    )
            im = axes[row, col].pcolormesh(
                a, tau, z,
                shading="auto",
                cmap="coolwarm",
                **kwargs,
            )
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("A")
            axes[row, col].set_ylabel("tau")
            plt.colorbar(im, ax=axes[row, col], shrink=0.86)

    fig.suptitle("PDE residual maps at the identified parameters", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_optimization_convergence(
    results: Dict[str, IdentificationResult],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, model_name in zip(axes, MODEL_NAMES):
        history = results[model_name].history
        for start_id, g in history.groupby("start_id"):
            best_so_far = np.minimum.accumulate(
                g["total_cost"].to_numpy(dtype=float)
            )
            ax.plot(
                np.arange(1, len(best_so_far) + 1),
                best_so_far,
                linewidth=1.0,
                alpha=0.75,
                label=f"Start {start_id}",
            )
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel("Function evaluation")
        ax.set_ylabel("Best objective so far")
        finite = history["total_cost"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(finite) and np.all(finite > 0):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Identical multi-start optimization protocol", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 10. 全局配对统计和绘图
# =============================================================================

def make_paired_sample_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "sample_id", "filename",
        "nu_standard", "kappa_standard", "d_standard",
    ]
    no = sample_metrics[
        sample_metrics["model"] == "without_pde"
    ].drop(columns=["model"]).copy()
    wp = sample_metrics[
        sample_metrics["model"] == "with_pde"
    ].drop(columns=["model"]).copy()

    paired = no.merge(
        wp,
        on=id_cols,
        suffixes=("_without_pde", "_with_pde"),
        how="inner",
    )

    compare_metrics = [
        "d1_rmse", "d2_rmse", "objective",
        "mean_combined_second_derivative_rms",
        "mean_combined_curvature_energy",
        "mean_d1_normalized_total_variation",
        "mean_d2_normalized_total_variation",
        "pde_total_mse",
        "d2_negative_fraction",
        "nu_abs_error", "kappa_abs_error", "d_abs_error",
    ]
    for metric in compare_metrics:
        a = f"{metric}_without_pde"
        b = f"{metric}_with_pde"
        if a in paired.columns and b in paired.columns:
            paired[f"{metric}_improvement_percent"] = [
                percent_improvement(x, y)
                for x, y in zip(paired[a], paired[b])
            ]
            paired[f"{metric}_winner"] = np.where(
                paired[b] < paired[a], "with_pde",
                np.where(paired[b] > paired[a], "without_pde", "tie"),
            )
    return paired


def build_wilcoxon_table(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "d1_rmse", "d2_rmse",
        "mean_combined_second_derivative_rms",
        "mean_combined_curvature_energy",
        "pde_total_mse",
        "nu_abs_error", "kappa_abs_error", "d_abs_error",
    ]
    rows = []
    for metric in metrics:
        a = f"{metric}_without_pde"
        b = f"{metric}_with_pde"
        if a not in paired or b not in paired:
            continue
        pair = paired[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2 or np.allclose(pair[a], pair[b]):
            statistic = pvalue = np.nan
        else:
            try:
                test = wilcoxon(pair[b], pair[a], alternative="two-sided")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic = pvalue = np.nan
        improvements = [
            percent_improvement(x, y)
            for x, y in zip(pair[a], pair[b])
        ]
        rows.append({
            "metric": metric,
            "n": len(pair),
            "median_without_pde": pair[a].median() if len(pair) else np.nan,
            "median_with_pde": pair[b].median() if len(pair) else np.nan,
            "median_improvement_percent": finite_median(improvements),
            "wilcoxon_statistic": statistic,
            "p_value": pvalue,
        })
    return pd.DataFrame(rows)


def plot_global_paired_scatter(
    paired: pd.DataFrame,
    save_path: Path,
) -> None:
    metrics = [
        ("d1_rmse", "D1 RMSE"),
        ("d2_rmse", "D2 RMSE"),
        ("mean_combined_second_derivative_rms", "Combined roughness"),
        ("mean_combined_curvature_energy", "Curvature energy"),
        ("pde_total_mse", "PDE residual MSE"),
        ("objective", "Identification objective"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        x = paired[f"{metric}_without_pde"].to_numpy(dtype=float)
        y = paired[f"{metric}_with_pde"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, alpha=0.75)
        if len(x):
            low = min(x.min(), y.min())
            high = max(x.max(), y.max())
            if high > low:
                ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
            if low > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
        ax.set_xlabel("Without PDE")
        ax.set_ylabel("With PDE")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Paired comparison: points below the diagonal favor the PDE model",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_global_improvement_boxplots(
    paired: pd.DataFrame,
    save_path: Path,
) -> None:
    metrics = [
        ("d1_rmse", "D1 RMSE"),
        ("d2_rmse", "D2 RMSE"),
        ("mean_combined_second_derivative_rms", "Roughness"),
        ("mean_combined_curvature_energy", "Curvature energy"),
        ("pde_total_mse", "PDE residual"),
        ("objective", "Objective"),
    ]
    data = []
    labels = []
    for metric, label in metrics:
        col = f"{metric}_improvement_percent"
        if col in paired:
            values = paired[col].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().to_numpy()
            data.append(values)
            labels.append(label)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_ylabel("Improvement of with-PDE model (%)")
    ax.set_title(
        "Positive values mean lower error or smoother KM coefficients with PDE"
    )
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 11. 单样本主流程
# =============================================================================

def process_one_sample(
    sample: KMSample,
    assets: Assets,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_dir = OUTPUT_DIR / "samples" / sample.sample_id
    data_dir = sample_dir / "data"
    fig_dir = sample_dir / "figures"
    safe_mkdir(data_dir)
    safe_mkdir(fig_dir)

    # 同一组多起点用于两个模型
    initial_points = generate_initial_points(sample)

    results: Dict[str, IdentificationResult] = {}
    for model_name in MODEL_NAMES:
        results[model_name] = identify_one_model(
            model_name,
            assets.models[model_name],
            sample,
            assets.preprocessing,
            initial_points,
        )

    pointwise = build_pointwise_comparison(sample, results)
    per_tau = build_per_tau_metrics(pointwise)
    sample_metrics = build_sample_metrics(
        sample,
        pointwise,
        per_tau,
        results,
        assets.preprocessing,
    )

    # 保存数据
    pointwise.to_csv(
        data_dir / "pointwise_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_tau.to_csv(
        data_dir / "per_tau_accuracy_and_smoothness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample_metrics.to_csv(
        data_dir / "sample_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "A": sample.a_values,
        "weight": [sample.a_weights[float(a)] for a in sample.a_values],
    }).to_csv(
        data_dir / "A_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez_compressed(
        data_dir / "full_grid_predictions.npz",
        a_grid=assets.preprocessing["unified_a_grid"],
        tau_grid=assets.preprocessing["unified_tau_grid"],
        without_pde_params=results["without_pde"].params,
        with_pde_params=results["with_pde"].params,
        D1_without_pde=results["without_pde"].D1_full,
        D2_without_pde=results["without_pde"].D2_full,
        D1_with_pde=results["with_pde"].D1_full,
        D2_with_pde=results["with_pde"].D2_full,
    )

    for model_name, result in results.items():
        result.history.to_csv(
            data_dir / f"optimization_history_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result.start_results.to_csv(
            data_dir / f"multi_start_summary_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_json(
        data_dir / "identification_summary.json",
        {
            "sample_id": sample.sample_id,
            "filename": sample.file_path.name,
            "standard_params": sample.standard_params,
            "without_pde": {
                "params": results["without_pde"].params,
                "objective": results["without_pde"].objective,
                "success": results["without_pde"].success,
                "message": results["without_pde"].message,
            },
            "with_pde": {
                "params": results["with_pde"].params,
                "objective": results["with_pde"].objective,
                "success": results["with_pde"].success,
                "message": results["with_pde"].message,
            },
        },
    )

    if PLOT_EVERY_SAMPLE:
        plot_profile_matrix(
            sample, pointwise, results,
            fig_dir / "01_KM_profiles_fair_comparison.png",
        )
        plot_second_derivative_profiles(
            sample, pointwise,
            fig_dir / "02_second_derivative_direct_evidence.png",
        )
        plot_smoothness_vs_tau(
            per_tau,
            fig_dir / "03_smoothness_metrics_vs_tau.png",
        )
        plot_error_smoothness_tradeoff(
            per_tau,
            fig_dir / "04_accuracy_smoothness_tradeoff.png",
        )
        plot_pde_residual_maps(
            results, assets.preprocessing,
            fig_dir / "05_PDE_residual_maps.png",
        )
        plot_optimization_convergence(
            results,
            fig_dir / "06_identical_multistart_convergence.png",
        )

    return sample_metrics, per_tau, pointwise


# =============================================================================
# 12. 主程序与全局输出
# =============================================================================

def save_summary_excel(
    path: Path,
    sample_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    wilcoxon_table: pd.DataFrame,
    per_tau: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    if not SAVE_EXCEL:
        return
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sample_metrics.to_excel(
                writer, sheet_name="sample_metrics", index=False
            )
            paired.to_excel(
                writer, sheet_name="paired_metrics", index=False
            )
            wilcoxon_table.to_excel(
                writer, sheet_name="wilcoxon", index=False
            )
            per_tau.to_excel(
                writer, sheet_name="per_tau", index=False
            )
            failures.to_excel(
                writer, sheet_name="failures", index=False
            )
    except Exception as exc:
        print(f"[Warning] Excel output failed: {exc}")


def main() -> None:
    start_time = time.time()
    safe_mkdir(OUTPUT_DIR)
    safe_mkdir(OUTPUT_DIR / "samples")
    safe_mkdir(OUTPUT_DIR / "summary_tables")
    safe_mkdir(OUTPUT_DIR / "summary_figures")

    print("=" * 88)
    print("Fair KM identification ablation: without PDE vs with PDE")
    print(f"Device: {DEVICE}")
    print("=" * 88)

    assets = load_assets()

    run_config = {
        "ABLATION_RESULT_DIR": ABLATION_RESULT_DIR,
        "MODEL_PATHS": MODEL_PATHS,
        "SHARED_PREPROCESSING_PATH": SHARED_PREPROCESSING_PATH,
        "CONFIG_PATH": CONFIG_PATH,
        "KM_DATA_DIR": KM_DATA_DIR,
        "SIM_DATA_DIR": SIM_DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MAX_FILES": MAX_FILES,
        "RANDOM_SEED": RANDOM_SEED,
        "OPTIMIZER_METHOD": OPTIMIZER_METHOD,
        "OPTIMIZER_OPTIONS": OPTIMIZER_OPTIONS,
        "N_MULTI_START": N_MULTI_START,
        "PARAM_BOUNDS": PARAM_BOUNDS,
        "OBJECTIVE_MODE": OBJECTIVE_MODE,
        "FIELD_WEIGHTS": FIELD_WEIGHTS,
        "TAU_INTERPOLATION": TAU_INTERPOLATION,
        "ALLOW_EXTRAPOLATION": ALLOW_EXTRAPOLATION,
        "preprocessing_fingerprints": assets.preprocessing["_fingerprints"],
        "training_config": assets.config,
    }
    save_json(OUTPUT_DIR / "run_config_and_fairness_audit.json", run_config)

    files = sorted(KM_DATA_DIR.glob(KM_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No files matching {KM_FILE_PATTERN} in {KM_DATA_DIR}"
        )

    if MAX_FILES is not None and len(files) > MAX_FILES:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(len(files), MAX_FILES, replace=False))
        files = [files[i] for i in idx]

    all_sample_metrics = []
    all_per_tau = []
    all_pointwise = []
    failures = []

    for file_path in tqdm(files, desc="Fair paired identification"):
        try:
            sample = load_km_sample(file_path)
            sample_metrics, per_tau, pointwise = process_one_sample(
                sample, assets
            )
            all_sample_metrics.append(sample_metrics)
            per_tau.insert(0, "sample_id", sample.sample_id)
            per_tau.insert(1, "filename", file_path.name)
            all_per_tau.append(per_tau)
            pointwise.insert(0, "sample_id", sample.sample_id)
            pointwise.insert(1, "filename", file_path.name)
            all_pointwise.append(pointwise)
        except Exception as exc:
            failures.append({
                "filename": file_path.name,
                "path": str(file_path),
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"\n[Error] {file_path.name}: {exc}")

    if not all_sample_metrics:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "summary_tables" / "failures.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError("No KM file was processed successfully.")

    sample_metrics_df = pd.concat(all_sample_metrics, ignore_index=True)
    per_tau_df = pd.concat(all_per_tau, ignore_index=True)
    pointwise_df = pd.concat(all_pointwise, ignore_index=True)
    failures_df = pd.DataFrame(failures)

    paired_df = make_paired_sample_metrics(sample_metrics_df)
    wilcoxon_df = build_wilcoxon_table(paired_df)

    tables = OUTPUT_DIR / "summary_tables"
    figures = OUTPUT_DIR / "summary_figures"

    sample_metrics_df.to_csv(
        tables / "all_sample_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_df.to_csv(
        tables / "paired_sample_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_tau_df.to_csv(
        tables / "all_per_tau_accuracy_and_smoothness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pointwise_df.to_csv(
        tables / "all_pointwise_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wilcoxon_df.to_csv(
        tables / "paired_wilcoxon_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures_df.to_csv(
        tables / "failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_summary_excel(
        tables / "fair_KM_identification_summary.xlsx",
        sample_metrics_df,
        paired_df,
        wilcoxon_df,
        per_tau_df,
        failures_df,
    )

    plot_global_paired_scatter(
        paired_df,
        figures / "01_global_paired_scatter.png",
    )
    plot_global_improvement_boxplots(
        paired_df,
        figures / "02_global_improvement_boxplots.png",
    )

    report_lines = [
        "Fair KM identification ablation report",
        "=" * 70,
        f"Successful samples: {paired_df['sample_id'].nunique()}",
        f"Failed files: {len(failures_df)}",
        f"Elapsed seconds: {time.time() - start_time:.2f}",
        "",
        "Fairness conditions:",
        "- Same KM files",
        "- Same shared preprocessing/POD basis/scalers",
        "- Same A weights",
        "- Same tau interpolation",
        "- Same normalized objective",
        "- Same parameter bounds and penalties",
        "- Same deterministic multi-start initial points",
        "- Same optimizer and stopping criteria",
        "",
        "Interpretation:",
        "- Positive improvement means the with-PDE model has a lower value.",
        "- For roughness, curvature energy and total variation, lower means smoother.",
        "- A persuasive PDE benefit requires lower roughness/PDE residual without a",
        "  substantial deterioration in D1/D2 RMSE.",
        "",
        "Paired Wilcoxon results:",
    ]
    for _, row in wilcoxon_df.iterrows():
        report_lines.append(
            f"- {row['metric']}: n={int(row['n'])}, "
            f"median improvement={row['median_improvement_percent']:.4g}%, "
            f"p={row['p_value']:.6g}"
        )

    (OUTPUT_DIR / "analysis_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nCompleted.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Successful samples: {paired_df['sample_id'].nunique()}")
    print(f"Failed files: {len(failures_df)}")


if __name__ == "__main__":
    main()
