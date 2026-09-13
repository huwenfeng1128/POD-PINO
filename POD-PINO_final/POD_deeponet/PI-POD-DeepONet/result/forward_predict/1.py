
# -*- coding: utf-8 -*-
"""
POD-DeepONet / POD-PINO 外部数据前向预测与物理残差消融对比
================================================================

用途
----
本程序不重新训练模型，而是直接加载“物理残差消融实验”已经保存的：

1. shared_preprocessing.pth
2. config.json
3. no_pde_residual/best_model.pth
4. with_pde_residual/best_model.pth

然后对完全独立于训练集和验证集的外部 KM 数据进行前向预测，并使用对应
仿真数据构造 A 方向权重，比较：

A. no_pde_residual
   训练时未加入 PDE 物理残差，但保留了 D2 非负约束；

B. with_pde_residual
   训练时加入归一化 PDE 物理残差，同时保留 D2 非负约束。

重要说明
--------
1. 前向预测阶段不会再次优化模型，也不会把 PDE 残差加入预测值；
2. 两组模型的区别来自训练得到的参数不同；
3. PDE 残差在本程序中用于外部物理一致性评价；
4. 仿真数据构造的 A 权重仅用于加权误差评价，不会修改预测场；
5. 默认期望外部数据为 120 组，但实际发现数量不等于 120 时仍会处理全部成功配对组；
6. 如果外部 KM 网格与训练网格不同，程序可在外部网格覆盖训练网格的前提下，
   将 KM 真值插值到模型输出网格；模型本身仍只在训练时保存的统一网格上输出。

建议目录结构
------------
KM_DATA_DIR/
├─ data_1.csv
├─ data_2.csv
└─ ...

SIM_DATA_DIR/
├─ data_1.csv
├─ data_2.csv
└─ ...

文件名不必完全相同，但去掉 km/sim/data/best/result 等常见前后缀后应能匹配。
程序会把最终匹配关系保存到 group_pair_manifest.csv，务必检查该文件。

KM 数据最低要求
---------------
长表格式，每行对应一个 (tau, A) 网格点，至少能够识别：
- A
- tau 或 time/t
- D1
- D2
- nu、kappa、d_diffusion 三个常参数

参数列可以位于 KM 文件或对应仿真文件中。常见列名会自动识别；若实际列名不同，
请修改下方 COLUMN_ALIASES。

仿真数据构造 A 权重
-------------------
程序优先寻找 P/probability/density/count/weight 等权重列：
- 若存在，则按 A 聚合并归一化；
- 若不存在，则把 A 列视为仿真样本，按模型 A 网格做直方图计数；
- 最终 A 权重归一化为和等于 1。

主要输出
--------
OUTPUT_DIR/
├─ run_config.json
├─ group_pair_manifest.csv
├─ failed_groups.csv
├─ external_sample_metrics.csv
├─ paired_model_comparison.csv
├─ aggregate_metric_summary.csv
├─ parameter_and_extrapolation_summary.csv
├─ external_prediction_report.txt
├─ overall_metric_comparison.png
├─ paired_metric_scatter.png
├─ improvement_distribution.png
├─ parameter_space_error.png
├─ pde_residual_comparison.png
├─ group_000_xxx/
│  ├─ source_and_parameter_info.json
│  ├─ a_weights.csv
│  ├─ sim_data_processed.csv
│  ├─ prediction_fields.csv
│  ├─ physics_diagnostics_interior.csv
│  ├─ metrics_by_model.csv
│  ├─ predictions_and_diagnostics.npz
│  ├─ A_weight.png
│  ├─ D1_field_comparison.png
│  ├─ D2_field_comparison.png
│  ├─ D1_tau_slices.png
│  ├─ D2_tau_slices.png
│  ├─ PDE_residual_comparison.png
│  └─ model_difference_fields.png
└─ ...

依赖
----
numpy, pandas, matplotlib, torch, tqdm
"""

from __future__ import annotations

import os
import re
import glob
import json
import math
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =============================================================================
# 1. 路径与运行配置
# =============================================================================

# --- 原消融实验训练结果目录 ---
TRAIN_RESULT_DIR = (
    r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\xiaorongshiyan\train_result_physics_ablation_2'
)

# --- 外部 KM 数据目录 ---
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'

# --- 对应仿真数据目录（用于构造 A 权重）---
INPUT_DIR_SIM_DATA = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'

# --- 外部预测输出目录 ---
OUTPUT_DIR = os.path.join(
    TRAIN_RESULT_DIR,
    'external_forward_prediction_km_data_4_ablation',
)

# 两组模型必须与原消融实验目录名完全一致。
MODEL_SPECS = {
    'no_pde_residual': {
        'label': 'Without PDE residual',
        'checkpoint': os.path.join(
            TRAIN_RESULT_DIR,
            'no_pde_residual',
            'best_model.pth',
        ),
    },
    'with_pde_residual': {
        'label': 'With PDE residual',
        'checkpoint': os.path.join(
            TRAIN_RESULT_DIR,
            'with_pde_residual',
            'best_model.pth',
        ),
    },
}

SHARED_PREPROCESSING_PATH = os.path.join(
    TRAIN_RESULT_DIR,
    'shared_preprocessing.pth',
)
CONFIG_PATH = os.path.join(TRAIN_RESULT_DIR, 'config.json')
DATA_SPLIT_MANIFEST_PATH = os.path.join(
    TRAIN_RESULT_DIR,
    'data_split_manifest.csv',
)

# 外部数据文件搜索。
KM_FILE_PATTERNS = ('*.csv',)
SIM_FILE_PATTERNS = ('*.csv',)
RECURSIVE_FILE_SEARCH = True
EXPECTED_NUM_GROUPS = 120

# 文件配对策略。
# False 更安全：文件 key 无法匹配时直接记录失败，不按排序强行配对。
# 只有在确认两个目录排序后严格一一对应时，才建议改为 True。
ALLOW_SORTED_ORDER_FALLBACK = False

# 外部 KM 网格与模型网格不同是否允许二维线性插值。
ALLOW_KM_GRID_INTERPOLATION = True
GRID_ATOL = 1e-10
GRID_RTOL = 1e-10

# 若目标网格超出 KM 源网格范围，是否禁止外推。
# 强烈建议保持 True，避免用边界值伪装外推。
FORBID_KM_EXTRAPOLATION = True

# 仿真 A 权重配置。
A_WEIGHT_EPS = 1e-30
CLIP_NEGATIVE_SIM_WEIGHTS = True
SMOOTH_A_WEIGHT_WINDOW = 1

# 可视化配置。
FIG_DPI = 300
NUM_TAU_SLICES = 4
CMAP_FIELD = 'viridis'
CMAP_ERROR = 'magma'
CMAP_RESIDUAL = 'magma'

# 推理配置。
EVAL_BATCH_SIZE = 128
SEED = 24
NORMALIZED_RESIDUAL_EPS = 1e-12
DTYPE = torch.float64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = torch.cuda.is_available()

# 输入超出训练分布的简单诊断：标准化参数绝对值超过此阈值时标记。
# 这不是严格 OOD 判定，只用于提醒。
STANDARDIZED_INPUT_WARNING_THRESHOLD = 3.0

# 是否把每一组完整预测与诊断保存为 npz。
SAVE_COMPRESSED_NPZ_PER_GROUP = True

# 是否保存每组图片。
SAVE_PER_GROUP_FIGURES = True

# 汇总图中最多标注多少个极端组，防止文字过密。
MAX_ANNOTATED_GROUPS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. 列名别名
# =============================================================================

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    'A': (
        'A', 'a', 'amplitude', 'Amplitude', 'amp', 'state', 'x',
    ),
    'tau': (
        'tau', 'Tau', 'TAU', 't', 'time', 'Time', 'scaled_time',
    ),
    'nu': (
        'nu', 'Nu', 'NU', 'mu', 'linear_coefficient', 'linear_coeff',
    ),
    'kappa': (
        'kappa', 'Kappa', 'KAPPA', 'beta', 'nonlinear_coefficient',
        'nonlinear_coeff',
    ),
    'd_diffusion': (
        'd_diffusion', 'D_diffusion', 'diffusion', 'diffusion_coefficient',
        'noise_intensity', 'D', 'd', 'sigma2',
    ),
    'D1': (
        'D1_adj', 'D1', 'd1', 'D1_km', 'd1_km', 'D1_est', 'd1_est',
        'drift', 'drift_coefficient', 'KM_D1',
    ),
    'D2': (
        'D2_adj', 'D2', 'd2', 'D2_km', 'd2_km', 'D2_est', 'd2_est',
        'diffusion_km', 'KM_D2', 'second_km',
    ),
    'sim_weight': (
        'P', 'p', 'probability', 'Probability', 'prob', 'pdf', 'PDF',
        'density', 'Density', 'count', 'Count', 'counts', 'weight',
        'Weight', 'frequency', 'occupancy', 'P_A_t', 'P(A,t)',
    ),
}


# =============================================================================
# 3. 数据结构
# =============================================================================

@dataclass
class SharedPreprocessing:
    branch_mean: np.ndarray
    branch_std: np.ndarray
    y_mean_scaler: np.ndarray
    y_std_scaler: np.ndarray
    y_mean_pod_scaled: np.ndarray
    pod_basis: np.ndarray
    singular_values: np.ndarray
    actual_num_modes: int
    retained_energy: float
    a_grid: np.ndarray
    tau_grid: np.ndarray


@dataclass
class PairedFiles:
    group_key: str
    km_file: str
    sim_file: str
    pairing_method: str


@dataclass
class ExternalGroup:
    group_key: str
    km_file: str
    sim_file: str
    params: np.ndarray
    d1_true: np.ndarray
    d2_true: np.ndarray
    a_weights: np.ndarray
    sim_processed: pd.DataFrame
    km_source_a_grid: np.ndarray
    km_source_tau_grid: np.ndarray
    km_was_interpolated: bool
    parameter_sources: Dict[str, str]
    weight_build_info: Dict[str, Any]


# =============================================================================
# 4. 基础工具
# =============================================================================


def set_seed(seed: int = 24) -> None:
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


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_converter(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_converter,
        )


def save_text(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_torch_load(path: str, device: torch.device = torch.device('cpu')):
    """兼容不同 PyTorch 版本，并允许 shared_preprocessing 中的 NumPy 对象。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = safe_torch_load(path, device)

    if isinstance(payload, dict):
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
        for key in ('state_dict', 'model_state_dict', 'model'):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and all(
                isinstance(v, torch.Tensor) for v in candidate.values()
            ):
                return candidate

    raise ValueError(
        f'检查点 {path} 不是可识别的纯 state_dict 或嵌套 state_dict。'
    )


def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(DEVICE, non_blocking=PIN_MEMORY)


def finite_float(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} 不是有限数: {value!r}')
    return value


def is_uniform_grid(arr: np.ndarray, tol: float = 1e-10) -> bool:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < 2:
        return False
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], atol=tol, rtol=tol)


def make_safe_name(value: str, max_len: int = 80) -> str:
    value = re.sub(r'[^0-9a-zA-Z_\-\.]+', '_', str(value)).strip('_')
    return (value or 'group')[:max_len]


def positive_for_log(values, floor: float = 1e-30) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), floor)


def percent_improvement(baseline: float, physics: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-30:
        return float('nan')
    return float((baseline - physics) / abs(baseline) * 100.0)


def safe_relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.linalg.norm(np.asarray(y_true).ravel())
    if denominator < 1e-30:
        return float('nan')
    return float(
        np.linalg.norm((np.asarray(y_pred) - np.asarray(y_true)).ravel())
        / denominator
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).ravel()
    denominator = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    if denominator < 1e-30:
        return float('nan')
    numerator = np.sum((y_true_flat - y_pred_flat) ** 2)
    return float(1.0 - numerator / denominator)


def normalized_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in df.columns:
        normalized = re.sub(r'[^0-9a-z]+', '', str(col).lower())
        lookup.setdefault(normalized, col)
    return lookup


def find_column(
    df: pd.DataFrame,
    semantic_name: str,
    required: bool = True,
) -> Optional[str]:
    aliases = COLUMN_ALIASES[semantic_name]

    # 先精确匹配，防止 d_diffusion 被错误识别为 D。
    for alias in aliases:
        if alias in df.columns:
            return alias

    lookup = normalized_column_lookup(df)
    for alias in aliases:
        key = re.sub(r'[^0-9a-z]+', '', alias.lower())
        if key in lookup:
            return lookup[key]

    if required:
        raise KeyError(
            f'无法在列 {list(df.columns)} 中识别 {semantic_name}。'
            f'请在 COLUMN_ALIASES[{semantic_name!r}] 中加入实际列名。'
        )
    return None


def read_csv_numeric(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='gb18030', on_bad_lines='skip')

    if df.empty:
        raise ValueError(f'文件为空: {path}')
    return df


# =============================================================================
# 5. 外部文件发现与配对
# =============================================================================


def collect_files(
    directory: str,
    patterns: Tuple[str, ...],
    recursive: bool,
) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'目录不存在: {directory}')

    files: List[str] = []
    for pattern in patterns:
        if recursive:
            files.extend(
                glob.glob(os.path.join(directory, '**', pattern), recursive=True)
            )
        else:
            files.extend(glob.glob(os.path.join(directory, pattern)))

    return sorted(set(os.path.abspath(p) for p in files if os.path.isfile(p)))


def canonical_group_key(path: str) -> str:
    """
    从文件名构造匹配 key。

    示例：
    - km_data_001.csv -> 001
    - sim_data_001_best.csv -> 001
    - result_case_12.csv -> case_12
    """
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    stem = re.sub(
        r'(^|[_\-])(km|kramersmoyal|sim|simulation|data|result|results|best|output|field)(?=$|[_\-])',
        '_',
        stem,
    )
    stem = re.sub(r'[_\-]+', '_', stem).strip('_')
    return stem or os.path.splitext(os.path.basename(path))[0].lower()


def last_integer_key(path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    numbers = re.findall(r'\d+', stem)
    if not numbers:
        return None
    return str(int(numbers[-1]))


def unique_map_by_key(
    files: List[str],
    key_fn,
    source_name: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    temp: Dict[str, List[str]] = {}
    for path in files:
        key = key_fn(path)
        if key is None:
            continue
        temp.setdefault(str(key), []).append(path)

    unique: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}
    for key, paths in temp.items():
        if len(paths) == 1:
            unique[key] = paths[0]
        else:
            duplicates[key] = paths
            print(
                f'[警告] {source_name} 中 key={key!r} 对应多个文件，'
                '这些文件不会自动配对。'
            )
    return unique, duplicates


def pair_external_files(
    km_files: List[str],
    sim_files: List[str],
) -> Tuple[List[PairedFiles], pd.DataFrame]:
    paired: List[PairedFiles] = []
    manifest_rows: List[Dict[str, Any]] = []

    km_primary, km_primary_dup = unique_map_by_key(
        km_files, canonical_group_key, 'KM 目录'
    )
    sim_primary, sim_primary_dup = unique_map_by_key(
        sim_files, canonical_group_key, 'SIM 目录'
    )

    used_km = set()
    used_sim = set()

    common_primary = sorted(set(km_primary) & set(sim_primary))
    for key in common_primary:
        km_path = km_primary[key]
        sim_path = sim_primary[key]
        paired.append(PairedFiles(key, km_path, sim_path, 'canonical_name'))
        used_km.add(km_path)
        used_sim.add(sim_path)

    # 第二层：仅对尚未配对文件使用末尾整数。
    remaining_km = [p for p in km_files if p not in used_km]
    remaining_sim = [p for p in sim_files if p not in used_sim]

    km_numeric, km_numeric_dup = unique_map_by_key(
        remaining_km, last_integer_key, 'KM 目录末尾数字'
    )
    sim_numeric, sim_numeric_dup = unique_map_by_key(
        remaining_sim, last_integer_key, 'SIM 目录末尾数字'
    )

    common_numeric = sorted(
        set(km_numeric) & set(sim_numeric),
        key=lambda x: int(x),
    )
    for key in common_numeric:
        km_path = km_numeric[key]
        sim_path = sim_numeric[key]
        paired.append(PairedFiles(key, km_path, sim_path, 'last_integer'))
        used_km.add(km_path)
        used_sim.add(sim_path)

    remaining_km = [p for p in km_files if p not in used_km]
    remaining_sim = [p for p in sim_files if p not in used_sim]

    if ALLOW_SORTED_ORDER_FALLBACK and remaining_km and remaining_sim:
        if len(remaining_km) != len(remaining_sim):
            print(
                '[警告] 无法启用排序回退配对：剩余 KM 与 SIM 文件数量不相等。'
            )
        else:
            for idx, (km_path, sim_path) in enumerate(
                zip(sorted(remaining_km), sorted(remaining_sim))
            ):
                key = f'sorted_{idx:04d}'
                paired.append(PairedFiles(key, km_path, sim_path, 'sorted_fallback'))
                used_km.add(km_path)
                used_sim.add(sim_path)

    for item in paired:
        manifest_rows.append(
            {
                'group_key': item.group_key,
                'status': 'paired',
                'pairing_method': item.pairing_method,
                'km_file': item.km_file,
                'sim_file': item.sim_file,
                'km_canonical_key': canonical_group_key(item.km_file),
                'sim_canonical_key': canonical_group_key(item.sim_file),
            }
        )

    for path in km_files:
        if path not in used_km:
            manifest_rows.append(
                {
                    'group_key': canonical_group_key(path),
                    'status': 'unmatched_km',
                    'pairing_method': '',
                    'km_file': path,
                    'sim_file': '',
                    'km_canonical_key': canonical_group_key(path),
                    'sim_canonical_key': '',
                }
            )

    for path in sim_files:
        if path not in used_sim:
            manifest_rows.append(
                {
                    'group_key': canonical_group_key(path),
                    'status': 'unmatched_sim',
                    'pairing_method': '',
                    'km_file': '',
                    'sim_file': path,
                    'km_canonical_key': '',
                    'sim_canonical_key': canonical_group_key(path),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)

    # 去重保护。
    pair_keys = [p.group_key for p in paired]
    if len(pair_keys) != len(set(pair_keys)):
        renamed: List[PairedFiles] = []
        counts: Dict[str, int] = {}
        for item in paired:
            counts[item.group_key] = counts.get(item.group_key, 0) + 1
            new_key = f'{item.group_key}_{counts[item.group_key]:02d}'
            renamed.append(
                PairedFiles(
                    group_key=new_key,
                    km_file=item.km_file,
                    sim_file=item.sim_file,
                    pairing_method=item.pairing_method,
                )
            )
        paired = renamed

    return paired, manifest_df


# =============================================================================
# 6. 加载共享预处理与模型
# =============================================================================


def load_shared_preprocessing(path: str) -> SharedPreprocessing:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'未找到共享预处理文件: {path}')

    payload = safe_torch_load(path, torch.device('cpu'))
    required = [
        'branch_mean',
        'branch_std',
        'y_mean_scaler',
        'y_std_scaler',
        'y_mean_pod_scaled',
        'pod_basis',
        'singular_values',
        'actual_num_modes',
        'unified_a_grid',
        'unified_tau_grid',
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f'{path} 缺少字段: {missing}')

    singular_values = np.asarray(payload['singular_values'], dtype=np.float64)
    actual_num_modes = int(payload['actual_num_modes'])

    retained_energy = payload.get('retained_energy', None)
    if retained_energy is None:
        energy = singular_values ** 2
        total = float(np.sum(energy))
        retained_energy = (
            float(np.sum(energy[:actual_num_modes]) / total)
            if total > 0.0
            else float('nan')
        )

    shared = SharedPreprocessing(
        branch_mean=np.asarray(payload['branch_mean'], dtype=np.float64),
        branch_std=np.asarray(payload['branch_std'], dtype=np.float64),
        y_mean_scaler=np.asarray(payload['y_mean_scaler'], dtype=np.float64),
        y_std_scaler=np.asarray(payload['y_std_scaler'], dtype=np.float64),
        y_mean_pod_scaled=np.asarray(
            payload['y_mean_pod_scaled'], dtype=np.float64
        ),
        pod_basis=np.asarray(payload['pod_basis'], dtype=np.float64),
        singular_values=singular_values,
        actual_num_modes=actual_num_modes,
        retained_energy=float(retained_energy),
        a_grid=np.asarray(payload['unified_a_grid'], dtype=np.float64),
        tau_grid=np.asarray(payload['unified_tau_grid'], dtype=np.float64),
    )

    if shared.branch_mean.shape != (3,) or shared.branch_std.shape != (3,):
        raise ValueError(
            '本脚本假设 branch 输入顺序为 [nu, kappa, d_diffusion]，'
            f'但保存形状为 mean={shared.branch_mean.shape}, '
            f'std={shared.branch_std.shape}。'
        )

    expected_field_len = len(shared.a_grid) * len(shared.tau_grid)
    expected_output_dim = 2 * expected_field_len
    if len(shared.y_mean_scaler) != expected_output_dim:
        raise ValueError(
            f'输出标准化维数 {len(shared.y_mean_scaler)} 与网格推导维数 '
            f'{expected_output_dim} 不一致。'
        )

    if shared.pod_basis.shape != (
        expected_output_dim,
        shared.actual_num_modes,
    ):
        raise ValueError(
            f'POD basis 形状 {shared.pod_basis.shape} 不符合 '
            f'({expected_output_dim}, {shared.actual_num_modes})。'
        )

    return shared


class MLP(nn.Module):
    """与原消融实验完全一致的 branch 网络。"""

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
    """与原消融实验完全一致的 POD-DeepONet。"""

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
        self.register_buffer(
            'pod_basis',
            torch.tensor(pod_basis, dtype=DTYPE),
        )
        self.register_buffer(
            'y_mean_pod_scaled',
            torch.tensor(y_mean_pod_scaled, dtype=DTYPE),
        )

    def forward(self, branch_x: torch.Tensor) -> torch.Tensor:
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


def build_models(
    shared: SharedPreprocessing,
    config: Dict[str, Any],
) -> Dict[str, PODDeepONet]:
    branch_input_dim = int(config.get('BRANCH_INPUT_DIM', 3))
    hidden_units = int(config.get('HIDDEN_UNITS', 128))
    num_hidden_layers = int(config.get('NUM_HIDDEN_LAYERS', 4))
    dropout_rate = float(config.get('DROPOUT_RATE', 0.1))

    models: Dict[str, PODDeepONet] = {}
    for name, spec in MODEL_SPECS.items():
        checkpoint = spec['checkpoint']
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'未找到模型检查点: {checkpoint}')

        model = PODDeepONet(
            branch_input_dim=branch_input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            num_pod_modes=shared.actual_num_modes,
            pod_basis=shared.pod_basis,
            y_mean_pod_scaled=shared.y_mean_pod_scaled,
            dropout_rate=dropout_rate,
        ).to(DEVICE)

        state_dict = safe_load_state_dict(checkpoint, DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models[name] = model

    return models


# =============================================================================
# 7. KM 真值读取、网格重排与插值
# =============================================================================


def grid_matches(source: np.ndarray, target: np.ndarray) -> bool:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return source.shape == target.shape and np.allclose(
        source,
        target,
        atol=GRID_ATOL,
        rtol=GRID_RTOL,
    )


def ensure_strictly_increasing(grid: np.ndarray, name: str) -> None:
    grid = np.asarray(grid, dtype=np.float64)
    if len(grid) < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f'{name} 必须严格递增，实际为 {grid}')


def interpolate_rectilinear_2d(
    source_a: np.ndarray,
    source_tau: np.ndarray,
    source_field: np.ndarray,
    target_a: np.ndarray,
    target_tau: np.ndarray,
) -> np.ndarray:
    """
    在规则矩形网格上做分离式二维线性插值。

    source_field 形状为 [num_source_tau, num_source_a]。
    不依赖 scipy。
    """
    source_a = np.asarray(source_a, dtype=np.float64)
    source_tau = np.asarray(source_tau, dtype=np.float64)
    source_field = np.asarray(source_field, dtype=np.float64)
    target_a = np.asarray(target_a, dtype=np.float64)
    target_tau = np.asarray(target_tau, dtype=np.float64)

    ensure_strictly_increasing(source_a, 'source A grid')
    ensure_strictly_increasing(source_tau, 'source tau grid')
    ensure_strictly_increasing(target_a, 'target A grid')
    ensure_strictly_increasing(target_tau, 'target tau grid')

    if source_field.shape != (len(source_tau), len(source_a)):
        raise ValueError(
            f'source_field 形状 {source_field.shape} 与 '
            f'({len(source_tau)}, {len(source_a)}) 不一致。'
        )

    if FORBID_KM_EXTRAPOLATION:
        if (
            target_a.min() < source_a.min() - GRID_ATOL
            or target_a.max() > source_a.max() + GRID_ATOL
            or target_tau.min() < source_tau.min() - GRID_ATOL
            or target_tau.max() > source_tau.max() + GRID_ATOL
        ):
            raise ValueError(
                '模型目标网格超出了 KM 源网格范围，且 FORBID_KM_EXTRAPOLATION=True。'
                f' source A=[{source_a.min()}, {source_a.max()}],'
                f' target A=[{target_a.min()}, {target_a.max()}],'
                f' source tau=[{source_tau.min()}, {source_tau.max()}],'
                f' target tau=[{target_tau.min()}, {target_tau.max()}]。'
            )

    # 先沿 A 插值。
    temp = np.empty((len(source_tau), len(target_a)), dtype=np.float64)
    for i in range(len(source_tau)):
        temp[i] = np.interp(
            target_a,
            source_a,
            source_field[i],
            left=np.nan if FORBID_KM_EXTRAPOLATION else source_field[i, 0],
            right=np.nan if FORBID_KM_EXTRAPOLATION else source_field[i, -1],
        )

    # 再沿 tau 插值。
    result = np.empty((len(target_tau), len(target_a)), dtype=np.float64)
    for j in range(len(target_a)):
        result[:, j] = np.interp(
            target_tau,
            source_tau,
            temp[:, j],
            left=np.nan if FORBID_KM_EXTRAPOLATION else temp[0, j],
            right=np.nan if FORBID_KM_EXTRAPOLATION else temp[-1, j],
        )

    if not np.all(np.isfinite(result)):
        raise ValueError('二维插值结果包含 NaN/Inf。')
    return result


def extract_constant_parameter(
    semantic_name: str,
    km_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    km_file: str,
    sim_file: str,
) -> Tuple[float, str]:
    """依次从 KM 文件、SIM 文件和文件名中读取参数。"""
    for source_name, df in (('km_file', km_df), ('sim_file', sim_df)):
        col = find_column(df, semantic_name, required=False)
        if col is None:
            continue

        values = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(
            dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue

        unique = np.unique(values)
        if len(unique) > 1:
            spread = float(np.max(unique) - np.min(unique))
            scale = max(float(np.max(np.abs(unique))), 1.0)
            if spread > 1e-10 * scale:
                raise ValueError(
                    f'{source_name} 的参数 {semantic_name} 不是常数：'
                    f'min={unique.min()}, max={unique.max()}。'
                )
        return finite_float(np.mean(values), semantic_name), f'{source_name}:{col}'

    # 文件名回退。支持 nu_0.1 / nu=0.1 / kappa-2e-3 等写法。
    patterns = {
        'nu': (r'(?:^|[_\-])nu[_=\-]?([-+]?\d*\.?\d+(?:e[-+]?\d+)?)',),
        'kappa': (
            r'(?:^|[_\-])kappa[_=\-]?([-+]?\d*\.?\d+(?:e[-+]?\d+)?)',
        ),
        'd_diffusion': (
            r'(?:^|[_\-])d(?:iffusion)?[_=\-]?([-+]?\d*\.?\d+(?:e[-+]?\d+)?)',
        ),
    }

    for path_name, path in (('km_filename', km_file), ('sim_filename', sim_file)):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        for pattern in patterns[semantic_name]:
            match = re.search(pattern, stem, flags=re.IGNORECASE)
            if match:
                return finite_float(match.group(1), semantic_name), path_name

    raise KeyError(
        f'无法从 KM/SIM 文件列或文件名读取参数 {semantic_name}。'
        '请确保 nu、kappa、d_diffusion 至少在一个对应文件中为常数列，'
        '或扩展 COLUMN_ALIASES。'
    )


def read_km_truth_to_model_grid(
    km_df: pd.DataFrame,
    target_a: np.ndarray,
    target_tau: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    a_col = find_column(km_df, 'A', required=True)
    tau_col = find_column(km_df, 'tau', required=True)
    d1_col = find_column(km_df, 'D1', required=True)
    d2_col = find_column(km_df, 'D2', required=True)

    work = pd.DataFrame(
        {
            'A': pd.to_numeric(km_df[a_col], errors='coerce'),
            'tau': pd.to_numeric(km_df[tau_col], errors='coerce'),
            'D1': pd.to_numeric(km_df[d1_col], errors='coerce'),
            'D2': pd.to_numeric(km_df[d2_col], errors='coerce'),
        }
    ).dropna()

    if work.empty:
        raise ValueError('KM 文件关键列转换为数值并删除 NaN 后为空。')

    # 重复网格点用平均值合并，同时记录为显式行为。
    grouped = (
        work.groupby(['tau', 'A'], as_index=False)[['D1', 'D2']]
        .mean()
        .sort_values(['tau', 'A'])
        .reset_index(drop=True)
    )

    source_a = np.sort(grouped['A'].unique().astype(np.float64))
    source_tau = np.sort(grouped['tau'].unique().astype(np.float64))

    if len(source_a) < 3 or len(source_tau) < 3:
        raise ValueError(
            f'KM 网格太小：A={len(source_a)}, tau={len(source_tau)}；'
            'PDE 中心差分至少需要 3×3。'
        )

    expected = len(source_a) * len(source_tau)
    if len(grouped) != expected:
        missing = expected - len(grouped)
        raise ValueError(
            f'KM 数据不是完整张量积网格：期望 {expected} 点，'
            f'实际 {len(grouped)} 点，缺少 {missing} 点。'
        )

    d1_pivot = grouped.pivot(index='tau', columns='A', values='D1')
    d2_pivot = grouped.pivot(index='tau', columns='A', values='D2')
    d1_pivot = d1_pivot.reindex(index=source_tau, columns=source_a)
    d2_pivot = d2_pivot.reindex(index=source_tau, columns=source_a)

    d1_source = d1_pivot.to_numpy(dtype=np.float64)
    d2_source = d2_pivot.to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(d1_source)) or not np.all(np.isfinite(d2_source)):
        raise ValueError('KM 网格重排后 D1/D2 包含 NaN/Inf。')

    same_grid = grid_matches(source_a, target_a) and grid_matches(
        source_tau, target_tau
    )

    if same_grid:
        return d1_source, d2_source, source_a, source_tau, False

    if not ALLOW_KM_GRID_INTERPOLATION:
        raise ValueError(
            'KM 网格与模型网格不一致，且 ALLOW_KM_GRID_INTERPOLATION=False。'
        )

    d1_target = interpolate_rectilinear_2d(
        source_a,
        source_tau,
        d1_source,
        target_a,
        target_tau,
    )
    d2_target = interpolate_rectilinear_2d(
        source_a,
        source_tau,
        d2_source,
        target_a,
        target_tau,
    )
    return d1_target, d2_target, source_a, source_tau, True


# =============================================================================
# 8. 从仿真数据构造 A 权重
# =============================================================================


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode='same')


def grid_bin_edges(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    ensure_strictly_increasing(grid, 'A grid')
    mids = 0.5 * (grid[:-1] + grid[1:])
    left = grid[0] - 0.5 * (grid[1] - grid[0])
    right = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    return np.concatenate([[left], mids, [right]])


def build_a_weights_from_sim(
    sim_df: pd.DataFrame,
    target_a: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    a_col = find_column(sim_df, 'A', required=True)
    weight_col = find_column(sim_df, 'sim_weight', required=False)

    a_values = pd.to_numeric(sim_df[a_col], errors='coerce').to_numpy(
        dtype=np.float64
    )
    finite_a_mask = np.isfinite(a_values)

    if np.sum(finite_a_mask) == 0:
        raise ValueError('仿真文件 A 列没有有限数。')

    info: Dict[str, Any] = {
        'a_column': a_col,
        'weight_column': weight_col,
        'method': '',
        'negative_weight_count_before_clip': 0,
        'num_source_rows': int(len(sim_df)),
        'num_finite_a_rows': int(np.sum(finite_a_mask)),
    }

    if weight_col is not None:
        raw_weight = pd.to_numeric(sim_df[weight_col], errors='coerce').to_numpy(
            dtype=np.float64
        )
        mask = finite_a_mask & np.isfinite(raw_weight)
        a_used = a_values[mask]
        w_used = raw_weight[mask]

        if len(a_used) == 0:
            raise ValueError('仿真 A/权重列没有共同有限行。')

        negative_count = int(np.sum(w_used < 0.0))
        info['negative_weight_count_before_clip'] = negative_count
        if negative_count > 0:
            if CLIP_NEGATIVE_SIM_WEIGHTS:
                w_used = np.maximum(w_used, 0.0)
            else:
                raise ValueError('仿真权重包含负数。')

        source = pd.DataFrame({'A': a_used, 'raw_weight': w_used})
        source = (
            source.groupby('A', as_index=False)['raw_weight']
            .sum()
            .sort_values('A')
            .reset_index(drop=True)
        )
        source_a = source['A'].to_numpy(dtype=np.float64)
        source_w = source['raw_weight'].to_numpy(dtype=np.float64)

        if np.sum(source_w) <= A_WEIGHT_EPS:
            raise ValueError('仿真权重列聚合后总和为零。')

        # 把源权重看作 A 上的非负密度/质量，再插值到模型网格。
        if len(source_a) == 1:
            target_w = np.zeros_like(target_a, dtype=np.float64)
            target_w[np.argmin(np.abs(target_a - source_a[0]))] = source_w[0]
        else:
            target_w = np.interp(
                target_a,
                source_a,
                source_w,
                left=0.0,
                right=0.0,
            )

        info['method'] = 'aggregated_explicit_weight_then_interpolated'
        processed_source = source.copy()

    else:
        # 无显式权重列：把所有 A 行作为仿真访问样本，按模型网格统计占据频率。
        a_used = a_values[finite_a_mask]
        edges = grid_bin_edges(target_a)
        target_w, _ = np.histogram(a_used, bins=edges)
        target_w = target_w.astype(np.float64)

        processed_source = pd.DataFrame(
            {
                'A_sample': a_used,
            }
        )
        info['method'] = 'histogram_of_raw_A_samples'

    target_w = np.asarray(target_w, dtype=np.float64)
    target_w = np.maximum(target_w, 0.0)
    target_w = smooth_1d(target_w, SMOOTH_A_WEIGHT_WINDOW)
    target_w = np.maximum(target_w, 0.0)

    total = float(np.sum(target_w))
    if total <= A_WEIGHT_EPS:
        raise ValueError(
            '构造出的 A 权重总和为零。可能原因：仿真 A 完全超出模型网格范围，'
            '或显式权重列全为零。'
        )

    target_w /= total

    info.update(
        {
            'target_weight_sum': float(np.sum(target_w)),
            'target_weight_min': float(np.min(target_w)),
            'target_weight_max': float(np.max(target_w)),
            'target_nonzero_fraction': float(np.mean(target_w > 0.0)),
            'smooth_window': int(SMOOTH_A_WEIGHT_WINDOW),
        }
    )

    return target_w, processed_source, info


# =============================================================================
# 9. 外部组加载
# =============================================================================


def load_external_group(
    pair: PairedFiles,
    shared: SharedPreprocessing,
) -> ExternalGroup:
    km_df = read_csv_numeric(pair.km_file)
    sim_df = read_csv_numeric(pair.sim_file)

    d1_true, d2_true, source_a, source_tau, interpolated = (
        read_km_truth_to_model_grid(
            km_df,
            shared.a_grid,
            shared.tau_grid,
        )
    )

    params = []
    parameter_sources: Dict[str, str] = {}
    for semantic_name in ('nu', 'kappa', 'd_diffusion'):
        value, source = extract_constant_parameter(
            semantic_name,
            km_df,
            sim_df,
            pair.km_file,
            pair.sim_file,
        )
        params.append(value)
        parameter_sources[semantic_name] = source

    a_weights, sim_processed, weight_info = build_a_weights_from_sim(
        sim_df,
        shared.a_grid,
    )

    return ExternalGroup(
        group_key=pair.group_key,
        km_file=pair.km_file,
        sim_file=pair.sim_file,
        params=np.asarray(params, dtype=np.float64),
        d1_true=np.asarray(d1_true, dtype=np.float64),
        d2_true=np.asarray(d2_true, dtype=np.float64),
        a_weights=np.asarray(a_weights, dtype=np.float64),
        sim_processed=sim_processed,
        km_source_a_grid=np.asarray(source_a, dtype=np.float64),
        km_source_tau_grid=np.asarray(source_tau, dtype=np.float64),
        km_was_interpolated=bool(interpolated),
        parameter_sources=parameter_sources,
        weight_build_info=weight_info,
    )


# =============================================================================
# 10. 前向预测
# =============================================================================


def scale_branch_input(
    branch_physical: np.ndarray,
    shared: SharedPreprocessing,
) -> np.ndarray:
    return (branch_physical - shared.branch_mean) / shared.branch_std


def inverse_output_scaler(
    y_scaled: np.ndarray,
    shared: SharedPreprocessing,
) -> np.ndarray:
    return y_scaled * shared.y_std_scaler + shared.y_mean_scaler


def predict_one_group(
    model: PODDeepONet,
    params: np.ndarray,
    shared: SharedPreprocessing,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch_physical = np.asarray(params, dtype=np.float64).reshape(1, 3)
    branch_scaled = scale_branch_input(branch_physical, shared)

    x = to_device(torch.from_numpy(branch_scaled))
    model.eval()
    with torch.no_grad():
        y_scaled = model(x).cpu().numpy()

    y_physical = inverse_output_scaler(y_scaled, shared)
    field_len = len(shared.a_grid) * len(shared.tau_grid)
    d1 = y_physical[0, :field_len].reshape(
        len(shared.tau_grid), len(shared.a_grid)
    )
    d2 = y_physical[0, field_len:].reshape(
        len(shared.tau_grid), len(shared.a_grid)
    )
    return d1, d2, branch_scaled[0]


# =============================================================================
# 11. PDE 残差与物理导数
# =============================================================================


def compute_physics_diagnostics_single(
    d1: np.ndarray,
    d2: np.ndarray,
    params: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    a_grid = np.asarray(a_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)

    expected_shape = (len(tau_grid), len(a_grid))
    if d1.shape != expected_shape or d2.shape != expected_shape:
        raise ValueError(
            f'D1/D2 形状必须为 {expected_shape}，实际为 {d1.shape}/{d2.shape}。'
        )

    if len(a_grid) < 3 or len(tau_grid) < 3:
        raise ValueError('PDE 中心差分要求 A 和 tau 均至少 3 点。')

    da_array = np.diff(a_grid)
    dtau_array = np.diff(tau_grid)
    if not np.allclose(da_array, da_array[0], atol=1e-10, rtol=1e-10):
        raise ValueError('模型 A 网格不是均匀网格，无法复现训练中的中心差分。')
    if not np.allclose(
        dtau_array, dtau_array[0], atol=1e-10, rtol=1e-10
    ):
        raise ValueError('模型 tau 网格不是均匀网格，无法复现训练中的中心差分。')

    da = float(da_array[0])
    dtau = float(dtau_array[0])

    nu, kappa, d_diff = params
    a = a_grid[None, :]
    tau = tau_grid[:, None]
    a_safe = np.maximum(a, 1e-9)

    d1_th = nu * a - (kappa / 8.0) * (a ** 3) + d_diff / a_safe
    d2_th = np.broadcast_to(d_diff, d1_th.shape)

    u1 = tau * d1
    u2 = tau * d2

    u1_in = u1[1:-1, 1:-1]
    d1_th_in = d1_th[:, 1:-1]
    d2_th_in = d2_th[:, 1:-1]

    du1_dtau = (u1[2:, 1:-1] - u1[:-2, 1:-1]) / (2.0 * dtau)
    du2_dtau = (u2[2:, 1:-1] - u2[:-2, 1:-1]) / (2.0 * dtau)

    du1_da = (u1[1:-1, 2:] - u1[1:-1, :-2]) / (2.0 * da)
    du2_da = (u2[1:-1, 2:] - u2[1:-1, :-2]) / (2.0 * da)

    d2u1_da2 = (
        u1[1:-1, 2:]
        - 2.0 * u1[1:-1, 1:-1]
        + u1[1:-1, :-2]
    ) / (da ** 2)

    d2u2_da2 = (
        u2[1:-1, 2:]
        - 2.0 * u2[1:-1, 1:-1]
        + u2[1:-1, :-2]
    ) / (da ** 2)

    r1_t1 = du1_dtau
    r1_t2 = d1_th_in * du1_da
    r1_t3 = d2_th_in * d2u1_da2
    r1_t4 = np.broadcast_to(d1_th_in, r1_t1.shape)
    r1 = r1_t1 - r1_t2 - r1_t3 - r1_t4
    r1_scale_sq = r1_t1**2 + r1_t2**2 + r1_t3**2 + r1_t4**2

    r2_t1 = du2_dtau
    r2_t2 = d1_th_in * du2_da
    r2_t3 = d2_th_in * d2u2_da2
    r2_t4 = d1_th_in * u1_in
    r2_t5 = 2.0 * d2_th_in * du1_da
    r2_t6 = np.broadcast_to(d2_th_in, r2_t1.shape)
    r2 = r2_t1 - r2_t2 - r2_t3 - r2_t4 - r2_t5 - r2_t6
    r2_scale_sq = (
        r2_t1**2
        + r2_t2**2
        + r2_t3**2
        + r2_t4**2
        + r2_t5**2
        + r2_t6**2
    )

    return {
        'u1': u1,
        'u2': u2,
        'du1_dtau': du1_dtau,
        'du1_da': du1_da,
        'd2u1_da2': d2u1_da2,
        'du2_dtau': du2_dtau,
        'du2_da': du2_da,
        'd2u2_da2': d2u2_da2,
        'r1': r1,
        'r2': r2,
        'r1_scale_sq': r1_scale_sq,
        'r2_scale_sq': r2_scale_sq,
        'r1_normalized_sq': r1**2 / (
            r1_scale_sq + NORMALIZED_RESIDUAL_EPS
        ),
        'r2_normalized_sq': r2**2 / (
            r2_scale_sq + NORMALIZED_RESIDUAL_EPS
        ),
    }


# =============================================================================
# 12. 误差指标
# =============================================================================


def make_2d_weights(a_weights: np.ndarray, num_tau: int) -> np.ndarray:
    a_weights = np.asarray(a_weights, dtype=np.float64)
    if np.any(a_weights < 0.0):
        raise ValueError('A 权重不能为负。')
    total = float(np.sum(a_weights))
    if total <= A_WEIGHT_EPS:
        raise ValueError('A 权重总和为零。')
    a_weights = a_weights / total
    return np.broadcast_to(
        a_weights[None, :] / float(num_tau),
        (num_tau, len(a_weights)),
    ).copy()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not np.any(mask):
        return float('nan')
    w = weights[mask]
    total = float(np.sum(w))
    if total <= A_WEIGHT_EPS:
        return float('nan')
    return float(np.sum(values[mask] * w) / total)


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        math.sqrt(
            max(
                weighted_mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2, weights),
                0.0,
            )
        )
    )


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    return weighted_mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)), weights)


def weighted_relative_l2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    numerator = weighted_mean(
        (np.asarray(y_pred) - np.asarray(y_true)) ** 2,
        weights,
    )
    denominator = weighted_mean(np.asarray(y_true) ** 2, weights)
    if not np.isfinite(denominator) or denominator <= A_WEIGHT_EPS:
        return float('nan')
    return float(math.sqrt(max(numerator, 0.0) / denominator))


def field_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights_2d: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {
        f'{prefix}_rmse': float(np.sqrt(np.mean(error**2))),
        f'{prefix}_mae': float(np.mean(np.abs(error))),
        f'{prefix}_max_abs': float(np.max(np.abs(error))),
        f'{prefix}_rel_l2': safe_relative_l2(y_true, y_pred),
        f'{prefix}_r2': safe_r2(y_true, y_pred),
        f'{prefix}_weighted_rmse': weighted_rmse(
            y_true, y_pred, weights_2d
        ),
        f'{prefix}_weighted_mae': weighted_mae(
            y_true, y_pred, weights_2d
        ),
        f'{prefix}_weighted_rel_l2': weighted_relative_l2(
            y_true, y_pred, weights_2d
        ),
    }


def interior_weights(
    a_weights: np.ndarray,
    num_tau: int,
) -> np.ndarray:
    if num_tau < 3 or len(a_weights) < 3:
        raise ValueError('内部权重至少需要 3×3 网格。')
    a_in = np.asarray(a_weights[1:-1], dtype=np.float64)
    total = float(np.sum(a_in))
    if total <= A_WEIGHT_EPS:
        # 如果全部概率集中在边界，为物理残差评价回退到内部均匀权重。
        a_in = np.ones_like(a_in, dtype=np.float64)
        total = float(np.sum(a_in))
    a_in /= total
    return np.broadcast_to(
        a_in[None, :] / float(num_tau - 2),
        (num_tau - 2, len(a_in)),
    ).copy()


def compute_model_metrics(
    model_name: str,
    d1_true: np.ndarray,
    d2_true: np.ndarray,
    d1_pred: np.ndarray,
    d2_pred: np.ndarray,
    true_physics: Dict[str, np.ndarray],
    pred_physics: Dict[str, np.ndarray],
    a_weights: np.ndarray,
    params: np.ndarray,
    branch_scaled: np.ndarray,
    group_key: str,
    km_interpolated: bool,
) -> Dict[str, Any]:
    num_tau, num_a = d1_true.shape
    weights_2d = make_2d_weights(a_weights, num_tau)
    weights_in = interior_weights(a_weights, num_tau)

    metrics: Dict[str, Any] = {
        'group_key': group_key,
        'model': model_name,
        'nu': float(params[0]),
        'kappa': float(params[1]),
        'd_diffusion': float(params[2]),
        'nu_scaled': float(branch_scaled[0]),
        'kappa_scaled': float(branch_scaled[1]),
        'd_diffusion_scaled': float(branch_scaled[2]),
        'max_abs_standardized_input': float(np.max(np.abs(branch_scaled))),
        'input_warning_gt_threshold': bool(
            np.max(np.abs(branch_scaled)) > STANDARDIZED_INPUT_WARNING_THRESHOLD
        ),
        'km_was_interpolated_to_model_grid': bool(km_interpolated),
    }

    metrics.update(field_metrics(d1_true, d1_pred, weights_2d, 'd1'))
    metrics.update(field_metrics(d2_true, d2_pred, weights_2d, 'd2'))

    # D2 非负性。
    negative = np.maximum(-d2_pred, 0.0)
    metrics['d2_negative_fraction'] = float(np.mean(d2_pred < 0.0))
    metrics['d2_weighted_negative_fraction'] = weighted_mean(
        (d2_pred < 0.0).astype(np.float64),
        weights_2d,
    )
    metrics['d2_min_prediction'] = float(np.min(d2_pred))
    metrics['d2_negative_magnitude_mse'] = float(np.mean(negative**2))
    metrics['d2_weighted_negative_magnitude_mse'] = weighted_mean(
        negative**2,
        weights_2d,
    )

    # PDE 残差：原始 MSE 与训练同定义的归一化残差。
    for r_name in ('r1', 'r2'):
        r = pred_physics[r_name]
        metrics[f'pde_{r_name}_mse'] = float(np.mean(r**2))
        metrics[f'pde_{r_name}_rmse'] = float(np.sqrt(np.mean(r**2)))
        metrics[f'pde_{r_name}_weighted_mse'] = weighted_mean(r**2, weights_in)
        metrics[f'pde_{r_name}_normalized_mean'] = float(
            np.mean(pred_physics[f'{r_name}_normalized_sq'])
        )
        metrics[f'pde_{r_name}_weighted_normalized_mean'] = weighted_mean(
            pred_physics[f'{r_name}_normalized_sq'],
            weights_in,
        )

    metrics['pde_total_mse'] = (
        metrics['pde_r1_mse'] + metrics['pde_r2_mse']
    )
    metrics['pde_total_weighted_mse'] = (
        metrics['pde_r1_weighted_mse']
        + metrics['pde_r2_weighted_mse']
    )
    metrics['pde_total_normalized_mean'] = (
        metrics['pde_r1_normalized_mean']
        + metrics['pde_r2_normalized_mean']
    )
    metrics['pde_total_weighted_normalized_mean'] = (
        metrics['pde_r1_weighted_normalized_mean']
        + metrics['pde_r2_weighted_normalized_mean']
    )

    # KM 真值本身的残差，用于判断真值噪声/离散误差基线。
    metrics['km_truth_pde_total_mse'] = float(
        np.mean(true_physics['r1'] ** 2)
        + np.mean(true_physics['r2'] ** 2)
    )
    metrics['km_truth_pde_total_normalized_mean'] = float(
        np.mean(true_physics['r1_normalized_sq'])
        + np.mean(true_physics['r2_normalized_sq'])
    )

    # 物理场导数误差。
    derivative_keys = (
        'du1_dtau',
        'du1_da',
        'd2u1_da2',
        'du2_dtau',
        'du2_da',
        'd2u2_da2',
    )
    derivative_rmses = []
    derivative_weighted_rmses = []
    for key in derivative_keys:
        true_value = true_physics[key]
        pred_value = pred_physics[key]
        error = pred_value - true_value
        rmse = float(np.sqrt(np.mean(error**2)))
        wrmse = weighted_rmse(true_value, pred_value, weights_in)
        derivative_rmses.append(rmse)
        derivative_weighted_rmses.append(wrmse)
        metrics[f'grad_{key}_rmse'] = rmse
        metrics[f'grad_{key}_mae'] = float(np.mean(np.abs(error)))
        metrics[f'grad_{key}_rel_l2'] = safe_relative_l2(
            true_value, pred_value
        )
        metrics[f'grad_{key}_weighted_rmse'] = wrmse
        metrics[f'grad_{key}_weighted_rel_l2'] = weighted_relative_l2(
            true_value, pred_value, weights_in
        )

    metrics['mean_physical_gradient_rmse'] = float(np.mean(derivative_rmses))
    metrics['mean_weighted_physical_gradient_rmse'] = float(
        np.mean(derivative_weighted_rmses)
    )

    return metrics


# =============================================================================
# 13. 每组数据保存
# =============================================================================


def pad_interior_to_full(
    interior: np.ndarray,
    full_shape: Tuple[int, int],
) -> np.ndarray:
    full = np.full(full_shape, np.nan, dtype=np.float64)
    full[1:-1, 1:-1] = interior
    return full


def save_group_tabular_outputs(
    group_dir: str,
    group: ExternalGroup,
    shared: SharedPreprocessing,
    predictions: Dict[str, Dict[str, np.ndarray]],
    physics: Dict[str, Dict[str, np.ndarray]],
    metrics_rows: List[Dict[str, Any]],
    branch_scaled: np.ndarray,
) -> None:
    num_tau = len(shared.tau_grid)
    num_a = len(shared.a_grid)
    a_mesh, tau_mesh = np.meshgrid(shared.a_grid, shared.tau_grid)
    weights_2d = make_2d_weights(group.a_weights, num_tau)

    field_df = pd.DataFrame(
        {
            'tau': tau_mesh.ravel(),
            'A': a_mesh.ravel(),
            'A_weight': np.broadcast_to(
                group.a_weights[None, :], (num_tau, num_a)
            ).ravel(),
            'evaluation_weight_2d': weights_2d.ravel(),
            'D1_KM_true': group.d1_true.ravel(),
            'D2_KM_true': group.d2_true.ravel(),
        }
    )

    for model_name, pred in predictions.items():
        d1_pred = pred['d1']
        d2_pred = pred['d2']
        field_df[f'D1_pred_{model_name}'] = d1_pred.ravel()
        field_df[f'D2_pred_{model_name}'] = d2_pred.ravel()
        field_df[f'D1_error_{model_name}'] = (
            d1_pred - group.d1_true
        ).ravel()
        field_df[f'D2_error_{model_name}'] = (
            d2_pred - group.d2_true
        ).ravel()
        field_df[f'D1_abs_error_{model_name}'] = np.abs(
            d1_pred - group.d1_true
        ).ravel()
        field_df[f'D2_abs_error_{model_name}'] = np.abs(
            d2_pred - group.d2_true
        ).ravel()
        field_df[f'D1_weighted_sq_error_{model_name}'] = (
            weights_2d * (d1_pred - group.d1_true) ** 2
        ).ravel()
        field_df[f'D2_weighted_sq_error_{model_name}'] = (
            weights_2d * (d2_pred - group.d2_true) ** 2
        ).ravel()
        field_df[f'D2_negative_part_{model_name}'] = np.maximum(
            -d2_pred, 0.0
        ).ravel()

    field_df[
        'D1_pred_difference_with_minus_without'
    ] = (
        predictions['with_pde_residual']['d1']
        - predictions['no_pde_residual']['d1']
    ).ravel()
    field_df[
        'D2_pred_difference_with_minus_without'
    ] = (
        predictions['with_pde_residual']['d2']
        - predictions['no_pde_residual']['d2']
    ).ravel()

    field_df.to_csv(
        os.path.join(group_dir, 'prediction_fields.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    pd.DataFrame(
        {
            'A': shared.a_grid,
            'A_weight': group.a_weights,
            'cumulative_A_weight': np.cumsum(group.a_weights),
        }
    ).to_csv(
        os.path.join(group_dir, 'a_weights.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    group.sim_processed.to_csv(
        os.path.join(group_dir, 'sim_data_processed.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(group_dir, 'metrics_by_model.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 物理内部网格数据。
    a_in_mesh, tau_in_mesh = np.meshgrid(
        shared.a_grid[1:-1], shared.tau_grid[1:-1]
    )
    weights_in = interior_weights(group.a_weights, num_tau)
    physics_df = pd.DataFrame(
        {
            'tau': tau_in_mesh.ravel(),
            'A': a_in_mesh.ravel(),
            'evaluation_weight_interior': weights_in.ravel(),
        }
    )

    for source_name, diag in physics.items():
        for key in (
            'r1',
            'r2',
            'r1_normalized_sq',
            'r2_normalized_sq',
            'du1_dtau',
            'du1_da',
            'd2u1_da2',
            'du2_dtau',
            'du2_da',
            'd2u2_da2',
        ):
            physics_df[f'{key}_{source_name}'] = diag[key].ravel()

    physics_df.to_csv(
        os.path.join(group_dir, 'physics_diagnostics_interior.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    info = {
        'group_key': group.group_key,
        'km_file': group.km_file,
        'sim_file': group.sim_file,
        'parameters': {
            'nu': group.params[0],
            'kappa': group.params[1],
            'd_diffusion': group.params[2],
        },
        'standardized_parameters': {
            'nu_scaled': branch_scaled[0],
            'kappa_scaled': branch_scaled[1],
            'd_diffusion_scaled': branch_scaled[2],
            'max_abs_standardized_input': float(
                np.max(np.abs(branch_scaled))
            ),
            'warning_threshold': STANDARDIZED_INPUT_WARNING_THRESHOLD,
        },
        'parameter_sources': group.parameter_sources,
        'weight_build_info': group.weight_build_info,
        'km_source_a_grid': group.km_source_a_grid,
        'km_source_tau_grid': group.km_source_tau_grid,
        'model_a_grid': shared.a_grid,
        'model_tau_grid': shared.tau_grid,
        'km_was_interpolated_to_model_grid': group.km_was_interpolated,
    }
    save_json(
        os.path.join(group_dir, 'source_and_parameter_info.json'),
        info,
    )

    if SAVE_COMPRESSED_NPZ_PER_GROUP:
        np.savez_compressed(
            os.path.join(group_dir, 'predictions_and_diagnostics.npz'),
            group_key=np.asarray(group.group_key),
            params=group.params,
            branch_scaled=branch_scaled,
            a_grid=shared.a_grid,
            tau_grid=shared.tau_grid,
            a_weights=group.a_weights,
            d1_true=group.d1_true,
            d2_true=group.d2_true,
            d1_pred_no_pde=predictions['no_pde_residual']['d1'],
            d2_pred_no_pde=predictions['no_pde_residual']['d2'],
            d1_pred_with_pde=predictions['with_pde_residual']['d1'],
            d2_pred_with_pde=predictions['with_pde_residual']['d2'],
            r1_true=physics['km_truth']['r1'],
            r2_true=physics['km_truth']['r2'],
            r1_no_pde=physics['no_pde_residual']['r1'],
            r2_no_pde=physics['no_pde_residual']['r2'],
            r1_with_pde=physics['with_pde_residual']['r1'],
            r2_with_pde=physics['with_pde_residual']['r2'],
            r1_norm_sq_true=physics['km_truth']['r1_normalized_sq'],
            r2_norm_sq_true=physics['km_truth']['r2_normalized_sq'],
            r1_norm_sq_no_pde=physics['no_pde_residual'][
                'r1_normalized_sq'
            ],
            r2_norm_sq_no_pde=physics['no_pde_residual'][
                'r2_normalized_sq'
            ],
            r1_norm_sq_with_pde=physics['with_pde_residual'][
                'r1_normalized_sq'
            ],
            r2_norm_sq_with_pde=physics['with_pde_residual'][
                'r2_normalized_sq'
            ],
        )


# =============================================================================
# 14. 每组图片
# =============================================================================


def common_extent(a_grid: np.ndarray, tau_grid: np.ndarray) -> List[float]:
    return [
        float(a_grid.min()),
        float(a_grid.max()),
        float(tau_grid.min()),
        float(tau_grid.max()),
    ]


def add_image_colorbar(fig, ax, image) -> None:
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_a_weight(
    a_grid: np.ndarray,
    a_weights: np.ndarray,
    save_path: str,
    group_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(a_grid, a_weights, marker='o', linewidth=1.7, markersize=3)
    ax.fill_between(a_grid, 0.0, a_weights, alpha=0.25)
    ax.set_xlabel('A')
    ax.set_ylabel('Normalized A weight')
    ax.set_title(f'A-weight distribution | group={group_key}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_field_comparison(
    field_name: str,
    true_field: np.ndarray,
    no_field: np.ndarray,
    with_field: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
    save_path: str,
    title_suffix: str,
) -> None:
    no_err = np.abs(no_field - true_field)
    with_err = np.abs(with_field - true_field)

    vmin = float(min(true_field.min(), no_field.min(), with_field.min()))
    vmax = float(max(true_field.max(), no_field.max(), with_field.max()))
    err_max = float(max(no_err.max(), with_err.max(), 1e-30))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.2))
    extent = common_extent(a_grid, tau_grid)

    panels = [
        (true_field, f'{field_name} KM truth', CMAP_FIELD, vmin, vmax),
        (no_field, f'{field_name} no PDE', CMAP_FIELD, vmin, vmax),
        (with_field, f'{field_name} with PDE', CMAP_FIELD, vmin, vmax),
        (no_err, f'|error| no PDE', CMAP_ERROR, 0.0, err_max),
        (with_err, f'|error| with PDE', CMAP_ERROR, 0.0, err_max),
    ]

    for ax, (data, title, cmap, pmin, pmax) in zip(axes, panels):
        image = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap,
            vmin=pmin,
            vmax=pmax,
        )
        ax.set_title(title)
        ax.set_xlabel('A')
        ax.set_ylabel('tau')
        add_image_colorbar(fig, ax, image)

    no_rmse = np.sqrt(np.mean((no_field - true_field) ** 2))
    with_rmse = np.sqrt(np.mean((with_field - true_field) ** 2))
    fig.suptitle(
        f'{field_name} external prediction | {title_suffix} | '
        f'RMSE(no/with)={no_rmse:.3e}/{with_rmse:.3e}',
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def select_tau_indices(num_tau: int, num_slices: int) -> np.ndarray:
    num_slices = min(max(1, num_slices), num_tau)
    return np.unique(
        np.round(np.linspace(0, num_tau - 1, num_slices)).astype(int)
    )


def plot_tau_slices(
    field_name: str,
    true_field: np.ndarray,
    no_field: np.ndarray,
    with_field: np.ndarray,
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
    save_path: str,
    title_suffix: str,
) -> None:
    tau_indices = select_tau_indices(len(tau_grid), NUM_TAU_SLICES)
    n = len(tau_indices)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, idx in zip(axes, tau_indices):
        ax.plot(a_grid, true_field[idx], linewidth=2.0, label='KM truth')
        ax.plot(a_grid, no_field[idx], linewidth=1.5, label='No PDE')
        ax.plot(a_grid, with_field[idx], linewidth=1.5, label='With PDE')
        ax.set_title(f'tau={tau_grid[idx]:.6g}')
        ax.set_xlabel('A')
        ax.set_ylabel(field_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f'{field_name} A-direction slices | {title_suffix}', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_pde_residual_comparison_group(
    physics: Dict[str, Dict[str, np.ndarray]],
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
    save_path: str,
    title_suffix: str,
) -> None:
    extent = [
        float(a_grid[1]),
        float(a_grid[-2]),
        float(tau_grid[1]),
        float(tau_grid[-2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    source_order = ('km_truth', 'no_pde_residual', 'with_pde_residual')
    source_titles = ('KM truth', 'No PDE', 'With PDE')

    for row, residual_name in enumerate(('r1', 'r2')):
        maps = [np.abs(physics[s][residual_name]) for s in source_order]
        vmax = max(float(np.max(x)) for x in maps)
        vmax = max(vmax, 1e-30)

        for col, (data, source_title) in enumerate(zip(maps, source_titles)):
            image = axes[row, col].imshow(
                data,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap=CMAP_RESIDUAL,
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, col].set_title(f'|{residual_name.upper()}| {source_title}')
            axes[row, col].set_xlabel('A')
            axes[row, col].set_ylabel('tau')
            add_image_colorbar(fig, axes[row, col], image)

    fig.suptitle(f'PDE residual comparison | {title_suffix}', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_model_difference_fields(
    predictions: Dict[str, Dict[str, np.ndarray]],
    a_grid: np.ndarray,
    tau_grid: np.ndarray,
    save_path: str,
    title_suffix: str,
) -> None:
    d1_diff = (
        predictions['with_pde_residual']['d1']
        - predictions['no_pde_residual']['d1']
    )
    d2_diff = (
        predictions['with_pde_residual']['d2']
        - predictions['no_pde_residual']['d2']
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    extent = common_extent(a_grid, tau_grid)

    for ax, data, title in (
        (axes[0], d1_diff, 'D1: with PDE - no PDE'),
        (axes[1], d2_diff, 'D2: with PDE - no PDE'),
    ):
        vmax = max(float(np.max(np.abs(data))), 1e-30)
        image = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='coolwarm',
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel('A')
        ax.set_ylabel('tau')
        add_image_colorbar(fig, ax, image)

    fig.suptitle(f'Inter-model field difference | {title_suffix}', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def save_group_figures(
    group_dir: str,
    group: ExternalGroup,
    shared: SharedPreprocessing,
    predictions: Dict[str, Dict[str, np.ndarray]],
    physics: Dict[str, Dict[str, np.ndarray]],
) -> None:
    title_suffix = (
        f'group={group.group_key}, '
        f'nu={group.params[0]:.4g}, '
        f'kappa={group.params[1]:.4g}, '
        f'd={group.params[2]:.4g}'
    )

    plot_a_weight(
        shared.a_grid,
        group.a_weights,
        os.path.join(group_dir, 'A_weight.png'),
        group.group_key,
    )

    for field_name, true_field, key in (
        ('D1', group.d1_true, 'd1'),
        ('D2', group.d2_true, 'd2'),
    ):
        no_field = predictions['no_pde_residual'][key]
        with_field = predictions['with_pde_residual'][key]
        plot_field_comparison(
            field_name,
            true_field,
            no_field,
            with_field,
            shared.a_grid,
            shared.tau_grid,
            os.path.join(group_dir, f'{field_name}_field_comparison.png'),
            title_suffix,
        )
        plot_tau_slices(
            field_name,
            true_field,
            no_field,
            with_field,
            shared.a_grid,
            shared.tau_grid,
            os.path.join(group_dir, f'{field_name}_tau_slices.png'),
            title_suffix,
        )

    plot_pde_residual_comparison_group(
        physics,
        shared.a_grid,
        shared.tau_grid,
        os.path.join(group_dir, 'PDE_residual_comparison.png'),
        title_suffix,
    )

    plot_model_difference_fields(
        predictions,
        shared.a_grid,
        shared.tau_grid,
        os.path.join(group_dir, 'model_difference_fields.png'),
        title_suffix,
    )


# =============================================================================
# 15. 单组处理
# =============================================================================


def process_one_group(
    pair: PairedFiles,
    group_index: int,
    shared: SharedPreprocessing,
    models: Dict[str, PODDeepONet],
) -> List[Dict[str, Any]]:
    group = load_external_group(pair, shared)

    group_name = f'group_{group_index:03d}_{make_safe_name(group.group_key)}'
    group_dir = os.path.join(OUTPUT_DIR, group_name)
    safe_mkdir(group_dir)

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    branch_scaled_reference: Optional[np.ndarray] = None

    for model_name, model in models.items():
        d1_pred, d2_pred, branch_scaled = predict_one_group(
            model,
            group.params,
            shared,
        )
        predictions[model_name] = {
            'd1': d1_pred,
            'd2': d2_pred,
        }
        if branch_scaled_reference is None:
            branch_scaled_reference = branch_scaled
        elif not np.allclose(
            branch_scaled_reference, branch_scaled, atol=0.0, rtol=0.0
        ):
            raise RuntimeError('两组模型得到的标准化 branch 输入不一致。')

    assert branch_scaled_reference is not None

    physics: Dict[str, Dict[str, np.ndarray]] = {
        'km_truth': compute_physics_diagnostics_single(
            group.d1_true,
            group.d2_true,
            group.params,
            shared.a_grid,
            shared.tau_grid,
        )
    }

    for model_name in models:
        physics[model_name] = compute_physics_diagnostics_single(
            predictions[model_name]['d1'],
            predictions[model_name]['d2'],
            group.params,
            shared.a_grid,
            shared.tau_grid,
        )

    metrics_rows: List[Dict[str, Any]] = []
    for model_name in models:
        row = compute_model_metrics(
            model_name=model_name,
            d1_true=group.d1_true,
            d2_true=group.d2_true,
            d1_pred=predictions[model_name]['d1'],
            d2_pred=predictions[model_name]['d2'],
            true_physics=physics['km_truth'],
            pred_physics=physics[model_name],
            a_weights=group.a_weights,
            params=group.params,
            branch_scaled=branch_scaled_reference,
            group_key=group.group_key,
            km_interpolated=group.km_was_interpolated,
        )
        row['group_output_dir'] = group_dir
        row['km_file'] = group.km_file
        row['sim_file'] = group.sim_file
        row['pairing_method'] = pair.pairing_method
        metrics_rows.append(row)

    save_group_tabular_outputs(
        group_dir,
        group,
        shared,
        predictions,
        physics,
        metrics_rows,
        branch_scaled_reference,
    )

    if SAVE_PER_GROUP_FIGURES:
        save_group_figures(
            group_dir,
            group,
            shared,
            predictions,
            physics,
        )

    return metrics_rows


# =============================================================================
# 16. 汇总表
# =============================================================================


def build_paired_model_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    id_cols = [
        'group_key',
        'nu',
        'kappa',
        'd_diffusion',
        'nu_scaled',
        'kappa_scaled',
        'd_diffusion_scaled',
        'max_abs_standardized_input',
        'input_warning_gt_threshold',
        'km_was_interpolated_to_model_grid',
        'km_file',
        'sim_file',
        'group_output_dir',
        'pairing_method',
    ]

    metric_cols = [
        c
        for c in metrics_df.columns
        if c not in id_cols + ['model']
        and pd.api.types.is_numeric_dtype(metrics_df[c])
    ]

    no_df = metrics_df[metrics_df['model'] == 'no_pde_residual'].copy()
    with_df = metrics_df[metrics_df['model'] == 'with_pde_residual'].copy()

    no_keep = ['group_key'] + metric_cols
    with_keep = ['group_key'] + metric_cols

    no_df = no_df[no_keep].rename(
        columns={c: f'{c}_without_pde' for c in metric_cols}
    )
    with_df = with_df[with_keep].rename(
        columns={c: f'{c}_with_pde' for c in metric_cols}
    )

    meta = (
        metrics_df[id_cols]
        .drop_duplicates(subset=['group_key'])
        .reset_index(drop=True)
    )
    paired = meta.merge(no_df, on='group_key', how='inner').merge(
        with_df, on='group_key', how='inner'
    )

    lower_is_better_metrics = [
        c
        for c in metric_cols
        if any(
            token in c
            for token in (
                'rmse',
                'mae',
                'max_abs',
                'rel_l2',
                'mse',
                'negative_fraction',
                'normalized_mean',
            )
        )
        and not c.startswith('km_truth_')
    ]

    for metric in lower_is_better_metrics:
        no_col = f'{metric}_without_pde'
        with_col = f'{metric}_with_pde'
        paired[f'{metric}_delta_with_minus_without'] = (
            paired[with_col] - paired[no_col]
        )
        paired[f'{metric}_improvement_percent'] = [
            percent_improvement(a, b)
            for a, b in zip(paired[no_col], paired[with_col])
        ]
        paired[f'{metric}_with_pde_better'] = paired[with_col] < paired[no_col]

    return paired


def build_aggregate_metric_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    selected_metrics = [
        'd1_rmse',
        'd2_rmse',
        'd1_weighted_rmse',
        'd2_weighted_rmse',
        'd1_rel_l2',
        'd2_rel_l2',
        'd1_weighted_rel_l2',
        'd2_weighted_rel_l2',
        'pde_total_mse',
        'pde_total_weighted_mse',
        'pde_total_normalized_mean',
        'pde_total_weighted_normalized_mean',
        'mean_physical_gradient_rmse',
        'mean_weighted_physical_gradient_rmse',
        'd2_negative_fraction',
        'd2_weighted_negative_fraction',
    ]

    rows: List[Dict[str, Any]] = []
    for model_name, group in metrics_df.groupby('model'):
        for metric in selected_metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors='coerce')
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            rows.append(
                {
                    'model': model_name,
                    'metric': metric,
                    'count': int(len(values)),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    'min': float(values.min()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'max': float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def build_parameter_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    cols = [
        'group_key',
        'nu',
        'kappa',
        'd_diffusion',
        'nu_scaled',
        'kappa_scaled',
        'd_diffusion_scaled',
        'max_abs_standardized_input',
        'input_warning_gt_threshold',
        'km_was_interpolated_to_model_grid',
        'km_file',
        'sim_file',
    ]
    return (
        metrics_df[cols]
        .drop_duplicates(subset=['group_key'])
        .sort_values('group_key')
        .reset_index(drop=True)
    )


# =============================================================================
# 17. 汇总图片
# =============================================================================


def mean_metric_by_model(
    metrics_df: pd.DataFrame,
    metric: str,
) -> Tuple[float, float]:
    no = metrics_df.loc[
        metrics_df['model'] == 'no_pde_residual', metric
    ].astype(float)
    with_pde = metrics_df.loc[
        metrics_df['model'] == 'with_pde_residual', metric
    ].astype(float)
    return float(no.mean()), float(with_pde.mean())


def plot_overall_metric_comparison(
    metrics_df: pd.DataFrame,
    save_path: str,
) -> None:
    metrics = [
        ('d1_rmse', 'D1 RMSE'),
        ('d2_rmse', 'D2 RMSE'),
        ('d1_weighted_rmse', 'D1 weighted RMSE'),
        ('d2_weighted_rmse', 'D2 weighted RMSE'),
        ('pde_total_mse', 'PDE total MSE'),
        ('pde_total_weighted_mse', 'PDE weighted MSE'),
        ('mean_physical_gradient_rmse', 'Gradient RMSE'),
        (
            'mean_weighted_physical_gradient_rmse',
            'Weighted gradient RMSE',
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        no, with_pde = mean_metric_by_model(metrics_df, metric)
        ax.bar([0, 1], [max(no, 1e-30), max(with_pde, 1e-30)])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No PDE', 'With PDE'])
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True, axis='y', which='both', alpha=0.3)
        improvement = percent_improvement(no, with_pde)
        ax.text(
            0.5,
            0.95,
            f'Mean improvement={improvement:.2f}%',
            transform=ax.transAxes,
            ha='center',
            va='top',
        )

    fig.suptitle('External KM-data aggregate metric comparison', fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_paired_metric_scatter(
    paired_df: pd.DataFrame,
    save_path: str,
) -> None:
    metrics = [
        ('d1_rmse', 'D1 RMSE'),
        ('d2_rmse', 'D2 RMSE'),
        ('d1_weighted_rmse', 'D1 weighted RMSE'),
        ('d2_weighted_rmse', 'D2 weighted RMSE'),
        ('pde_total_mse', 'PDE total MSE'),
        ('pde_total_weighted_mse', 'PDE weighted MSE'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        x = paired_df[f'{metric}_without_pde'].to_numpy(dtype=np.float64)
        y = paired_df[f'{metric}_with_pde'].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x = x[mask]
        y = y[mask]

        ax.scatter(x, y, s=28, alpha=0.7)
        if len(x) > 0:
            low = max(min(x.min(), y.min()), 1e-30)
            high = max(x.max(), y.max())
            ax.plot([low, high], [low, high], linestyle='--', linewidth=1.2)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Without PDE residual')
        ax.set_ylabel('With PDE residual')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        'Paired external-group comparison; points below diagonal favor PDE model',
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_improvement_distribution(
    paired_df: pd.DataFrame,
    save_path: str,
) -> None:
    metrics = [
        ('d1_rmse', 'D1 RMSE'),
        ('d2_rmse', 'D2 RMSE'),
        ('d1_weighted_rmse', 'D1 weighted RMSE'),
        ('d2_weighted_rmse', 'D2 weighted RMSE'),
        ('pde_total_mse', 'PDE total MSE'),
        ('pde_total_weighted_mse', 'PDE weighted MSE'),
    ]

    data = []
    labels = []
    for metric, label in metrics:
        col = f'{metric}_improvement_percent'
        values = pd.to_numeric(paired_df[col], errors='coerce')
        values = values[np.isfinite(values)]
        data.append(values.to_numpy(dtype=np.float64))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(0.0, linestyle='--', linewidth=1.2)
    ax.set_ylabel('Improvement percent; positive favors PDE model')
    ax.set_title('Per-group improvement distribution')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_parameter_space_error(
    paired_df: pd.DataFrame,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    params = [
        ('nu', 'nu'),
        ('kappa', 'kappa'),
        ('d_diffusion', 'd_diffusion'),
    ]

    for col, (param, label) in enumerate(params):
        improvement = paired_df[
            'd1_weighted_rmse_improvement_percent'
        ].to_numpy(dtype=np.float64)
        x = paired_df[param].to_numpy(dtype=np.float64)
        scatter = axes[0, col].scatter(
            x,
            improvement,
            c=paired_df['max_abs_standardized_input'],
            cmap='viridis',
            s=38,
            alpha=0.8,
        )
        axes[0, col].axhline(0.0, linestyle='--', linewidth=1.0)
        axes[0, col].set_xlabel(label)
        axes[0, col].set_ylabel('D1 weighted RMSE improvement (%)')
        axes[0, col].grid(True, alpha=0.3)
        fig.colorbar(
            scatter,
            ax=axes[0, col],
            fraction=0.046,
            pad=0.04,
            label='max |standardized input|',
        )

        improvement_d2 = paired_df[
            'd2_weighted_rmse_improvement_percent'
        ].to_numpy(dtype=np.float64)
        scatter2 = axes[1, col].scatter(
            x,
            improvement_d2,
            c=paired_df['max_abs_standardized_input'],
            cmap='viridis',
            s=38,
            alpha=0.8,
        )
        axes[1, col].axhline(0.0, linestyle='--', linewidth=1.0)
        axes[1, col].set_xlabel(label)
        axes[1, col].set_ylabel('D2 weighted RMSE improvement (%)')
        axes[1, col].grid(True, alpha=0.3)
        fig.colorbar(
            scatter2,
            ax=axes[1, col],
            fraction=0.046,
            pad=0.04,
            label='max |standardized input|',
        )

    fig.suptitle('Physics-model improvement across parameter space', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_pde_residual_comparison_summary(
    paired_df: pd.DataFrame,
    save_path: str,
) -> None:
    metrics = [
        ('pde_r1_mse', 'R1 MSE'),
        ('pde_r2_mse', 'R2 MSE'),
        ('pde_total_mse', 'Total PDE MSE'),
        ('pde_r1_weighted_mse', 'Weighted R1 MSE'),
        ('pde_r2_weighted_mse', 'Weighted R2 MSE'),
        ('pde_total_weighted_mse', 'Weighted total PDE MSE'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        x = paired_df[f'{metric}_without_pde'].to_numpy(dtype=np.float64)
        y = paired_df[f'{metric}_with_pde'].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x = x[mask]
        y = y[mask]
        ax.scatter(x, y, s=30, alpha=0.75)
        if len(x):
            low = max(min(x.min(), y.min()), 1e-30)
            high = max(x.max(), y.max())
            ax.plot([low, high], [low, high], linestyle='--', linewidth=1.0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('No PDE model')
        ax.set_ylabel('With PDE model')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('External PDE-residual paired comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def save_summary_figures(
    metrics_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> None:
    if metrics_df.empty or paired_df.empty:
        return

    plot_overall_metric_comparison(
        metrics_df,
        os.path.join(OUTPUT_DIR, 'overall_metric_comparison.png'),
    )
    plot_paired_metric_scatter(
        paired_df,
        os.path.join(OUTPUT_DIR, 'paired_metric_scatter.png'),
    )
    plot_improvement_distribution(
        paired_df,
        os.path.join(OUTPUT_DIR, 'improvement_distribution.png'),
    )
    plot_parameter_space_error(
        paired_df,
        os.path.join(OUTPUT_DIR, 'parameter_space_error.png'),
    )
    plot_pde_residual_comparison_summary(
        paired_df,
        os.path.join(OUTPUT_DIR, 'pde_residual_comparison.png'),
    )


# =============================================================================
# 18. 自动报告
# =============================================================================


def metric_report_line(
    metrics_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    metric: str,
    label: str,
) -> str:
    no_mean, with_mean = mean_metric_by_model(metrics_df, metric)
    improvement = percent_improvement(no_mean, with_mean)
    win_col = f'{metric}_with_pde_better'
    win_rate = (
        float(paired_df[win_col].mean() * 100.0)
        if win_col in paired_df.columns and len(paired_df) > 0
        else float('nan')
    )
    return (
        f'- {label}: no PDE mean={no_mean:.8e}, '
        f'with PDE mean={with_mean:.8e}, '
        f'mean improvement={improvement:.3f}%, '
        f'with-PDE win rate={win_rate:.2f}%'
    )


def build_external_report(
    metrics_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    pair_manifest: pd.DataFrame,
    shared: SharedPreprocessing,
    config: Dict[str, Any],
) -> str:
    num_groups = int(metrics_df['group_key'].nunique()) if not metrics_df.empty else 0
    num_failed = int(len(failed_df))
    num_unmatched = int(
        np.sum(pair_manifest['status'] != 'paired')
        if not pair_manifest.empty
        else 0
    )
    num_interpolated = int(
        metrics_df[
            ['group_key', 'km_was_interpolated_to_model_grid']
        ]
        .drop_duplicates()['km_was_interpolated_to_model_grid']
        .sum()
        if not metrics_df.empty
        else 0
    )
    num_input_warnings = int(
        metrics_df[['group_key', 'input_warning_gt_threshold']]
        .drop_duplicates()['input_warning_gt_threshold']
        .sum()
        if not metrics_df.empty
        else 0
    )

    lines = [
        'POD-DeepONet / POD-PINO 外部 KM 数据前向预测报告',
        '=' * 72,
        '',
        '1. 模型与预处理一致性',
        f'- 训练结果目录: {TRAIN_RESULT_DIR}',
        f'- 共享预处理文件: {SHARED_PREPROCESSING_PATH}',
        f'- no-PDE 模型: {MODEL_SPECS["no_pde_residual"]["checkpoint"]}',
        f'- with-PDE 模型: {MODEL_SPECS["with_pde_residual"]["checkpoint"]}',
        f'- 实际 POD 模态数: {shared.actual_num_modes}',
        f'- POD 保留能量: {shared.retained_energy:.10f}',
        f'- 模型网格: tau={len(shared.tau_grid)} 点, A={len(shared.a_grid)} 点',
        f'- 训练时物理权重 lambda: {config.get("PHYSICS_LOSS_WEIGHT", "未在 config.json 中找到")}',
        '',
        '2. 外部数据处理',
        f'- 期望组数: {EXPECTED_NUM_GROUPS}',
        f'- 成功完成组数: {num_groups}',
        f'- 已配对但处理失败组数: {num_failed}',
        f'- 未匹配文件记录数: {num_unmatched}',
        f'- KM 网格插值到模型网格的组数: {num_interpolated}',
        f'- max|标准化输入|>{STANDARDIZED_INPUT_WARNING_THRESHOLD:g} 的组数: {num_input_warnings}',
        '- A 权重只用于评价；没有修改模型输出。',
        '',
        '3. 主要外部指标',
    ]

    if metrics_df.empty or paired_df.empty:
        lines.extend(
            [
                '- 没有成功生成可汇总指标。请检查 failed_groups.csv 和 group_pair_manifest.csv。',
            ]
        )
    else:
        for metric, label in (
            ('d1_rmse', 'D1 RMSE'),
            ('d2_rmse', 'D2 RMSE'),
            ('d1_weighted_rmse', 'D1 A-weighted RMSE'),
            ('d2_weighted_rmse', 'D2 A-weighted RMSE'),
            ('pde_total_mse', 'PDE total MSE'),
            ('pde_total_weighted_mse', 'PDE A-weighted total MSE'),
            (
                'mean_physical_gradient_rmse',
                'Mean physical-gradient RMSE',
            ),
            (
                'mean_weighted_physical_gradient_rmse',
                'Mean A-weighted physical-gradient RMSE',
            ),
        ):
            lines.append(metric_report_line(metrics_df, paired_df, metric, label))

        pde_imp = paired_df['pde_total_mse_improvement_percent']
        d1_imp = paired_df['d1_rmse_improvement_percent']
        d2_imp = paired_df['d2_rmse_improvement_percent']

        supports_physics = (
            np.nanmedian(pde_imp) > 0.0
            and np.mean(paired_df['pde_total_mse_with_pde_better']) > 0.5
        )

        lines.extend(['', '4. 自动判定'])
        if supports_physics:
            lines.append(
                '- 外部结果支持“训练时加入 PDE 残差提升了多数样本的物理一致性”：'
                'PDE total MSE 的组间中位改善率为正，且超过一半外部组由 with-PDE 模型获胜。'
            )
        else:
            lines.append(
                '- 外部结果不足以支持“加入 PDE 残差稳定提升物理一致性”：'
                '请检查逐组残差、参数外推程度、KM 真值自身残差和 lambda 权重。'
            )

        if np.nanmedian(d1_imp) > 0.0 and np.nanmedian(d2_imp) > 0.0:
            lines.append(
                '- D1 与 D2 的组间中位预测误差改善率均为正，'
                '说明物理约束没有只改善残差而普遍牺牲数据拟合。'
            )
        else:
            lines.append(
                '- D1 或 D2 至少一个场的组间中位误差改善率不为正，'
                '说明数据精度与物理一致性之间可能存在权衡。'
            )

    lines.extend(
        [
            '',
            '5. 解释注意事项',
            '- KM 真值通常含估计噪声，其 PDE 残差不必严格为零；报告同时保存了 KM 真值残差基线。',
            '- 外部参数标准化绝对值很大时属于明显分布外推，模型误差应与该诊断联合解读。',
            '- 如果 KM 网格被插值，误差中包含插值误差；可在逐组 JSON 中确认。',
            '- A-weighted 指标强调仿真高概率区域；未加权指标仍保留，用于防止低概率区域误差被完全忽略。',
            '- 请先人工核对 group_pair_manifest.csv，确认 KM 与 SIM 文件一一对应。',
        ]
    )

    return '\n'.join(lines) + '\n'


# =============================================================================
# 19. 可选：检查外部文件是否与训练/验证清单路径重复
# =============================================================================


def check_path_overlap_with_training_manifest(
    paired: List[PairedFiles],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(DATA_SPLIT_MANIFEST_PATH):
        return pd.DataFrame(rows)

    try:
        manifest = pd.read_csv(DATA_SPLIT_MANIFEST_PATH)
    except Exception as exc:
        print(f'[警告] 无法读取训练划分清单进行路径重复检查: {exc}')
        return pd.DataFrame(rows)

    if 'file' not in manifest.columns:
        return pd.DataFrame(rows)

    train_paths = {
        os.path.normcase(os.path.abspath(str(p)))
        for p in manifest['file'].dropna().tolist()
    }

    for item in paired:
        km_norm = os.path.normcase(os.path.abspath(item.km_file))
        sim_norm = os.path.normcase(os.path.abspath(item.sim_file))
        rows.append(
            {
                'group_key': item.group_key,
                'km_exact_path_in_training_manifest': km_norm in train_paths,
                'sim_exact_path_in_training_manifest': sim_norm in train_paths,
                'km_file': item.km_file,
                'sim_file': item.sim_file,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 20. 主程序
# =============================================================================


def main() -> None:
    set_seed(SEED)
    safe_mkdir(OUTPUT_DIR)

    print('=' * 96)
    print('External KM Forward Prediction: PDE Residual Ablation')
    print('=' * 96)
    print(f'Device              : {DEVICE}')
    print(f'Dtype               : {DTYPE}')
    print(f'Training result dir : {TRAIN_RESULT_DIR}')
    print(f'KM data dir         : {BASE_KM_DATA_DIR}')
    print(f'SIM data dir        : {INPUT_DIR_SIM_DATA}')
    print(f'Output dir          : {OUTPUT_DIR}')

    config = load_json(CONFIG_PATH)
    shared = load_shared_preprocessing(SHARED_PREPROCESSING_PATH)
    models = build_models(shared, config)

    print('\n共享预处理与模型加载完成：')
    print(f'POD modes       : {shared.actual_num_modes}')
    print(f'Retained energy : {shared.retained_energy:.10f}')
    print(f'A grid points   : {len(shared.a_grid)}')
    print(f'tau grid points : {len(shared.tau_grid)}')

    km_files = collect_files(
        BASE_KM_DATA_DIR,
        KM_FILE_PATTERNS,
        RECURSIVE_FILE_SEARCH,
    )
    sim_files = collect_files(
        INPUT_DIR_SIM_DATA,
        SIM_FILE_PATTERNS,
        RECURSIVE_FILE_SEARCH,
    )

    print('\n外部文件发现：')
    print(f'KM CSV count  : {len(km_files)}')
    print(f'SIM CSV count : {len(sim_files)}')

    if not km_files:
        raise FileNotFoundError(f'KM 目录中未找到 CSV: {BASE_KM_DATA_DIR}')
    if not sim_files:
        raise FileNotFoundError(f'SIM 目录中未找到 CSV: {INPUT_DIR_SIM_DATA}')

    paired, pair_manifest = pair_external_files(km_files, sim_files)
    pair_manifest_path = os.path.join(OUTPUT_DIR, 'group_pair_manifest.csv')
    pair_manifest.to_csv(
        pair_manifest_path,
        index=False,
        encoding='utf-8-sig',
    )

    if not paired:
        raise ValueError(
            '没有成功配对任何 KM/SIM 文件。请检查 group_pair_manifest.csv、'
            '文件名规则或 ALLOW_SORTED_ORDER_FALLBACK。'
        )

    print(f'成功配对组数: {len(paired)}')
    if len(paired) != EXPECTED_NUM_GROUPS:
        print(
            f'[警告] 期望 {EXPECTED_NUM_GROUPS} 组，但实际配对 {len(paired)} 组。'
            '程序仍会处理全部已配对组。'
        )

    overlap_df = check_path_overlap_with_training_manifest(paired)
    if not overlap_df.empty:
        overlap_df.to_csv(
            os.path.join(OUTPUT_DIR, 'exact_path_overlap_check.csv'),
            index=False,
            encoding='utf-8-sig',
        )
        if overlap_df[
            [
                'km_exact_path_in_training_manifest',
                'sim_exact_path_in_training_manifest',
            ]
        ].to_numpy().any():
            print(
                '[警告] 至少一个外部文件的完整路径出现在训练/验证清单中。'
                '请检查 exact_path_overlap_check.csv。'
            )

    run_config = {
        'TRAIN_RESULT_DIR': TRAIN_RESULT_DIR,
        'BASE_KM_DATA_DIR': BASE_KM_DATA_DIR,
        'INPUT_DIR_SIM_DATA': INPUT_DIR_SIM_DATA,
        'OUTPUT_DIR': OUTPUT_DIR,
        'MODEL_SPECS': MODEL_SPECS,
        'SHARED_PREPROCESSING_PATH': SHARED_PREPROCESSING_PATH,
        'CONFIG_PATH': CONFIG_PATH,
        'EXPECTED_NUM_GROUPS': EXPECTED_NUM_GROUPS,
        'NUM_DISCOVERED_KM_FILES': len(km_files),
        'NUM_DISCOVERED_SIM_FILES': len(sim_files),
        'NUM_PAIRED_GROUPS': len(paired),
        'ALLOW_SORTED_ORDER_FALLBACK': ALLOW_SORTED_ORDER_FALLBACK,
        'ALLOW_KM_GRID_INTERPOLATION': ALLOW_KM_GRID_INTERPOLATION,
        'FORBID_KM_EXTRAPOLATION': FORBID_KM_EXTRAPOLATION,
        'STANDARDIZED_INPUT_WARNING_THRESHOLD': STANDARDIZED_INPUT_WARNING_THRESHOLD,
        'SMOOTH_A_WEIGHT_WINDOW': SMOOTH_A_WEIGHT_WINDOW,
        'FIG_DPI': FIG_DPI,
        'DEVICE': str(DEVICE),
        'DTYPE': str(DTYPE),
        'actual_num_modes': shared.actual_num_modes,
        'retained_energy': shared.retained_energy,
        'a_grid': shared.a_grid,
        'tau_grid': shared.tau_grid,
        'training_config': config,
    }
    save_json(os.path.join(OUTPUT_DIR, 'run_config.json'), run_config)

    all_metric_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for group_index, pair in enumerate(
        tqdm(paired, desc='外部组前向预测与评价')
    ):
        try:
            rows = process_one_group(
                pair=pair,
                group_index=group_index,
                shared=shared,
                models=models,
            )
            all_metric_rows.extend(rows)
        except Exception as exc:
            failed_rows.append(
                {
                    'group_index': group_index,
                    'group_key': pair.group_key,
                    'km_file': pair.km_file,
                    'sim_file': pair.sim_file,
                    'pairing_method': pair.pairing_method,
                    'error_type': type(exc).__name__,
                    'reason': str(exc),
                    'traceback': traceback.format_exc(),
                }
            )
            print(
                f'\n[警告] group={pair.group_key} 处理失败：'
                f'{type(exc).__name__}: {exc}'
            )

    failed_df = pd.DataFrame(failed_rows)
    failed_path = os.path.join(OUTPUT_DIR, 'failed_groups.csv')
    failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = os.path.join(OUTPUT_DIR, 'external_sample_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    paired_df = build_paired_model_comparison(metrics_df)
    paired_path = os.path.join(OUTPUT_DIR, 'paired_model_comparison.csv')
    paired_df.to_csv(paired_path, index=False, encoding='utf-8-sig')

    aggregate_df = build_aggregate_metric_summary(metrics_df)
    aggregate_path = os.path.join(OUTPUT_DIR, 'aggregate_metric_summary.csv')
    aggregate_df.to_csv(
        aggregate_path,
        index=False,
        encoding='utf-8-sig',
    )

    parameter_df = build_parameter_summary(metrics_df)
    parameter_path = os.path.join(
        OUTPUT_DIR,
        'parameter_and_extrapolation_summary.csv',
    )
    parameter_df.to_csv(
        parameter_path,
        index=False,
        encoding='utf-8-sig',
    )

    if not paired_df.empty:
        # 关键指标排序，便于快速找到改善最大和退化最大的组。
        ranking_cols = [
            'group_key',
            'nu',
            'kappa',
            'd_diffusion',
            'max_abs_standardized_input',
            'd1_rmse_improvement_percent',
            'd2_rmse_improvement_percent',
            'd1_weighted_rmse_improvement_percent',
            'd2_weighted_rmse_improvement_percent',
            'pde_total_mse_improvement_percent',
            'pde_total_weighted_mse_improvement_percent',
            'group_output_dir',
        ]
        ranking_cols = [c for c in ranking_cols if c in paired_df.columns]
        paired_df[ranking_cols].sort_values(
            'pde_total_mse_improvement_percent',
            ascending=False,
        ).to_csv(
            os.path.join(OUTPUT_DIR, 'group_improvement_ranking.csv'),
            index=False,
            encoding='utf-8-sig',
        )

    save_summary_figures(metrics_df, paired_df)

    report = build_external_report(
        metrics_df,
        paired_df,
        failed_df,
        pair_manifest,
        shared,
        config,
    )
    report_path = os.path.join(OUTPUT_DIR, 'external_prediction_report.txt')
    save_text(report_path, report)

    print('\n' + report)
    print('主要输出：')
    print(f'- 配对清单       : {pair_manifest_path}')
    print(f'- 失败清单       : {failed_path}')
    print(f'- 逐组逐模型指标 : {metrics_path}')
    print(f'- 配对对比指标   : {paired_path}')
    print(f'- 汇总统计       : {aggregate_path}')
    print(f'- 参数外推诊断   : {parameter_path}')
    print(f'- 自动报告       : {report_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('\n程序运行失败，异常信息如下：')
        traceback.print_exc()
        raise


