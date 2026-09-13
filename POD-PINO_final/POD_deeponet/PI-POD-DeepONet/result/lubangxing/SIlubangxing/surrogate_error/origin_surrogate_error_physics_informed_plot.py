# -*- coding: utf-8 -*-
"""
Physics-informed Surrogate error 与参数辨识误差相关性作图脚本
================================================================

用途：
1. 读取 physics-informed 版本的 Surrogate_error 参数辨识结果；
2. 生成 Origin 可直接使用的数据表；
3. 将散点数据与三条线性拟合线数据放在同一个 CSV 中；
4. 保存每个参数误差与 surrogate_mse 的 Pearson / Spearman / 线性拟合统计；
5. 生成与原脚本风格一致的三联图。

说明：
- 本脚本不重新跑 DeepONet，也不重新做参数辨识；
- 它是针对已经运行完的 surrogate_error_physics_informed.py 的后处理脚本；
- 默认读取：
      D:\\PINN\\zenodo\\POD_deeponet\\Revise_experiment\\Surrogate_error\\result_physics_informed
- 优先读取 batch_summary_physics_informed.csv；
- 如果没有总表，会自动从各子目录 sample_metrics.csv 重建汇总表。

输出：
1. origin_plot_table_with_fit.csv
2. origin_plot_fit_stats.csv
3. origin_style_plot_linearx.png
"""

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress


# =========================
# Configuration
# =========================
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\SIlubangxing\surrogate_error\result_2'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

FIG_DPI = 300

# physics-informed 主脚本默认保存的总表名；后面两个是兼容旧命名/手动改名
SUMMARY_CANDIDATE_FILES = [
    "batch_summary_physics_informed.csv",
    "batch_summary_pi.csv",
    "batch_summary_v3.csv",
]

# 误差百分比定义：
# "range"    : |pred - true| / (max(true)-min(true)) * 100，保持原作图脚本定义；
# "relative" : |pred - true| / max(|true|, eps) * 100。
ERROR_PERCENT_MODE = "range"
EPS = 1e-12

# 输出文件名保持原脚本命名，因为现在输出目录已经是 result_physics_informed
ORIGIN_TABLE_NAME = "origin_plot_table_with_fit.csv"
ORIGIN_STATS_NAME = "origin_plot_fit_stats.csv"
ORIGIN_FIG_NAME = "origin_style_plot_linearx.png"


# =========================
# Safe Save Helpers
# =========================
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def add_timestamp_to_path(path):
    base, ext = os.path.splitext(path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


def safe_to_csv(df, path, index=False):
    ensure_parent_dir(path)
    try:
        df.to_csv(path, index=index)
        print(f"CSV saved to: {path}")
        return path
    except PermissionError:
        alt_path = add_timestamp_to_path(path)
        df.to_csv(alt_path, index=index)
        print(f"Permission denied for: {path}")
        print(f"CSV saved instead to: {alt_path}")
        return alt_path


def safe_savefig(fig, path, dpi=300, bbox_inches='tight'):
    ensure_parent_dir(path)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {path}")
        return path
    except PermissionError:
        alt_path = add_timestamp_to_path(path)
        fig.savefig(alt_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Permission denied for: {path}")
        print(f"Figure saved instead to: {alt_path}")
        return alt_path


# =========================
# Load Existing Results
# =========================
def load_existing_summary(base_output_dir):
    """
    优先读取 physics-informed 总表；
    若不存在，则扫描所有子目录中的 sample_metrics.csv 并重建总表。
    """
    # 1) 优先读取指定候选总表
    for filename in SUMMARY_CANDIDATE_FILES:
        summary_path = os.path.join(base_output_dir, filename)
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            print(f"Loaded summary from: {summary_path}")
            return df, summary_path

    # 2) 兼容其他 batch_summary*.csv：若有多个，取最近修改的一个
    pattern = os.path.join(base_output_dir, "batch_summary*.csv")
    extra_candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if extra_candidates:
        summary_path = extra_candidates[0]
        df = pd.read_csv(summary_path)
        print(f"Loaded summary from fallback batch summary: {summary_path}")
        return df, summary_path

    # 3) 从子目录 sample_metrics.csv 重建
    print("No batch summary found. Collecting sample_metrics.csv from subfolders...")

    rows = []
    for root, dirs, files in os.walk(base_output_dir):
        if "sample_metrics.csv" not in files:
            continue

        sample_metrics_path = os.path.join(root, "sample_metrics.csv")
        try:
            df_one = pd.read_csv(sample_metrics_path)
            if len(df_one) > 0:
                row = df_one.iloc[0].to_dict()
                row["sample_metrics_path"] = sample_metrics_path
                rows.append(row)
        except Exception as e:
            print(f"Warning: failed to read {sample_metrics_path}: {e}")

    if len(rows) == 0:
        raise FileNotFoundError(
            "没有找到 batch_summary*.csv，也没有在子目录中找到 sample_metrics.csv。\n"
            "请先运行 surrogate_error_physics_informed.py 完成参数辨识。"
        )

    df = pd.DataFrame(rows)
    rebuilt_path = os.path.join(base_output_dir, "batch_summary_physics_informed_rebuilt.csv")
    saved_path = safe_to_csv(df, rebuilt_path, index=False)
    print(f"Rebuilt summary and saved to: {saved_path}")
    return df, saved_path


def ensure_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in summary data: {missing}")


def _finite_series(values):
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _safe_param_range(values, name):
    s = _finite_series(values)
    if len(s) == 0:
        raise ValueError(f"Cannot compute parameter range for {name}: no finite values")
    r = float(s.max() - s.min())
    return max(r, EPS)


# =========================
# Build Plot Data
# =========================
def build_origin_plot_table(summary_df):
    """
    生成图中真正需要的散点表：
        filename, surrogate_mse, nu_error_percent, kappa_error_percent, d_error_percent

    默认误差定义与原脚本一致：
        |pred - true| / (max(true)-min(true)) * 100%
    """
    required_cols = [
        'surrogate_mse',
        'nu_standard', 'kappa_standard', 'D_standard',
        'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized'
    ]
    ensure_required_columns(summary_df, required_cols)

    df = summary_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if ERROR_PERCENT_MODE == "range":
        nu_denom = _safe_param_range(df['nu_standard'], 'nu_standard')
        kappa_denom = _safe_param_range(df['kappa_standard'], 'kappa_standard')
        d_denom = _safe_param_range(df['D_standard'], 'D_standard')

        df['nu_error_percent'] = (
            np.abs(df['nu_optimized'] - df['nu_standard']) / nu_denom * 100.0
        )
        df['kappa_error_percent'] = (
            np.abs(df['kappa_optimized'] - df['kappa_standard']) / kappa_denom * 100.0
        )
        df['d_error_percent'] = (
            np.abs(df['d_diffusion_optimized'] - df['D_standard']) / d_denom * 100.0
        )

    elif ERROR_PERCENT_MODE == "relative":
        df['nu_error_percent'] = (
            np.abs(df['nu_optimized'] - df['nu_standard']) /
            np.maximum(np.abs(df['nu_standard']), EPS) * 100.0
        )
        df['kappa_error_percent'] = (
            np.abs(df['kappa_optimized'] - df['kappa_standard']) /
            np.maximum(np.abs(df['kappa_standard']), EPS) * 100.0
        )
        df['d_error_percent'] = (
            np.abs(df['d_diffusion_optimized'] - df['D_standard']) /
            np.maximum(np.abs(df['D_standard']), EPS) * 100.0
        )

    else:
        raise ValueError("ERROR_PERCENT_MODE must be 'range' or 'relative'")

    keep_cols = ['surrogate_mse', 'nu_error_percent', 'kappa_error_percent', 'd_error_percent']
    if 'filename' in df.columns:
        keep_cols = ['filename'] + keep_cols

    plot_df = df[keep_cols].copy()
    plot_df = plot_df.dropna(
        subset=['surrogate_mse', 'nu_error_percent', 'kappa_error_percent', 'd_error_percent']
    )
    plot_df = plot_df[plot_df['surrogate_mse'] > 0]
    plot_df = plot_df.sort_values(by='surrogate_mse').reset_index(drop=True)

    if len(plot_df) < 2:
        raise ValueError("有效样本少于 2 个，无法做线性拟合和相关性分析。")

    return plot_df


def _fit_one_line(x, y, x_fit):
    """对单个参数误差拟合；若 x 或 y 无变化，则返回 NaN 统计和水平线。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        slope = 0.0
        intercept = float(np.nanmean(y)) if len(y) else np.nan
        y_fit = np.full_like(x_fit, intercept, dtype=float)
        return y_fit, {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'slope': slope,
            'intercept': intercept,
            'n_samples': len(x)
        }

    reg = linregress(x, y)
    pearson_val = pearsonr(x, y)
    spearman_val = spearmanr(x, y)
    y_fit = reg.slope * x_fit + reg.intercept

    return y_fit, {
        'pearson_r': pearson_val[0],
        'pearson_p': pearson_val[1],
        'spearman_rho': spearman_val[0],
        'spearman_p': spearman_val[1],
        'slope': reg.slope,
        'intercept': reg.intercept,
        'n_samples': len(x)
    }


def add_fit_columns_to_same_table(plot_df, n_fit=300):
    """
    在同一个表里追加三条拟合线数据：
        fit_x, fit_nu_error_percent, fit_kappa_error_percent, fit_d_error_percent
    """
    x = plot_df['surrogate_mse'].values.astype(float)
    x_fit = np.linspace(np.min(x), np.max(x), n_fit)

    y_fit_nu, stats_nu = _fit_one_line(x, plot_df['nu_error_percent'].values, x_fit)
    y_fit_kappa, stats_kappa = _fit_one_line(x, plot_df['kappa_error_percent'].values, x_fit)
    y_fit_d, stats_d = _fit_one_line(x, plot_df['d_error_percent'].values, x_fit)

    fit_df = pd.DataFrame({
        'fit_x': x_fit,
        'fit_nu_error_percent': y_fit_nu,
        'fit_kappa_error_percent': y_fit_kappa,
        'fit_d_error_percent': y_fit_d
    })

    max_len = max(len(plot_df), len(fit_df))
    scatter_pad = plot_df.reindex(range(max_len))
    fit_pad = fit_df.reindex(range(max_len))
    merged_df = pd.concat([scatter_pad, fit_pad], axis=1)

    stats_nu['parameter'] = 'nu'
    stats_kappa['parameter'] = 'kappa'
    stats_d['parameter'] = 'd'
    stats_df = pd.DataFrame([stats_nu, stats_kappa, stats_d])
    stats_df = stats_df[[
        'parameter', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p',
        'slope', 'intercept', 'n_samples'
    ]]

    return merged_df, stats_df


# =========================
# Plot from Same Table
# =========================
def plot_from_origin_table(plot_df, output_dir):
    fig_path = os.path.join(output_dir, ORIGIN_FIG_NAME)

    param_configs = [
        ('nu_error_percent', r'$\nu$ error (%)', 'nu'),
        ('kappa_error_percent', r'$\kappa$ error (%)', 'kappa'),
        ('d_error_percent', r'$d$ error (%)', 'd')
    ]

    x = plot_df['surrogate_mse'].values.astype(float)
    x_line = np.linspace(np.min(x), np.max(x), 300)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for ax, (y_col, y_label, short_name) in zip(axes, param_configs):
        y = plot_df[y_col].values.astype(float)
        y_line, stat = _fit_one_line(x, y, x_line)

        ax.scatter(
            x, y,
            s=28,
            alpha=0.70,
            color='#4C72B0',
            edgecolors='none',
            zorder=2
        )

        ax.plot(
            x_line, y_line,
            '--',
            color='#D62728',
            linewidth=2.0,
            zorder=3
        )

        ax.set_xlabel('Surrogate Model Prediction MSE', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, alpha=0.20, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'({short_name})', fontsize=12, pad=8)

        pearson_text = "nan" if np.isnan(stat['pearson_r']) else f"{stat['pearson_r']:.3f}"
        spearman_text = "nan" if np.isnan(stat['spearman_rho']) else f"{stat['spearman_rho']:.3f}"
        text_str = (
            f'Pearson r = {pearson_text}\n'
            f'Spearman ρ = {spearman_text}\n'
            f'N = {len(plot_df)}'
        )
        ax.text(
            0.04, 0.96,
            text_str,
            transform=ax.transAxes,
            fontsize=9.5,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor='0.6', alpha=0.9)
        )

    fig.suptitle('Physics-informed Surrogate Error vs Parameter Errors', fontsize=14, y=1.02)
    fig.tight_layout()

    fig_path = safe_savefig(fig, fig_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    return fig_path


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Build Origin-ready table for physics-informed surrogate-error analysis ===")
    print(f"Base output dir: {BASE_OUTPUT_DIR}")

    summary_df, summary_path = load_existing_summary(BASE_OUTPUT_DIR)
    print(f"Total samples loaded: {len(summary_df)}")
    print(f"Summary source: {summary_path}")

    plot_df = build_origin_plot_table(summary_df)
    merged_df, stats_df = add_fit_columns_to_same_table(plot_df, n_fit=300)

    table_path = os.path.join(BASE_OUTPUT_DIR, ORIGIN_TABLE_NAME)
    stats_path = os.path.join(BASE_OUTPUT_DIR, ORIGIN_STATS_NAME)

    table_path = safe_to_csv(merged_df, table_path, index=False)
    stats_path = safe_to_csv(stats_df, stats_path, index=False)
    fig_path = plot_from_origin_table(plot_df, BASE_OUTPUT_DIR)

    print(f"Origin-ready table with fit saved to: {table_path}")
    print(f"Fit stats saved to: {stats_path}")
    print(f"Plot saved to: {fig_path}")
    print("=== Done ===")
