# -*- coding: utf-8 -*-
"""
Physics-informed Surrogate error 与参数辨识误差相关性作图脚本（修正版）
========================================================================

用途：
1. 读取 surrogate_error_physics_informed.py 生成的 physics-informed 参数辨识结果；
2. 使用第一个脚本已经输出的相对误差列 nu_rel_error / kappa_rel_error / d_rel_error；
3. 生成 Origin 可直接使用的数据表，并把三条拟合线数据放在同一个 CSV 中；
4. 使用 log10(surrogate_mse) 做相关性与拟合，作图横轴显示为 log scale；
5. 保存 Pearson / Spearman / 线性拟合统计和三联图。

关键修正：
- 不再默认用 |pred-true|/(max(true)-min(true))*100 重新定义误差；
- 优先使用第一个脚本输出的 nu_rel_error、kappa_rel_error、d_rel_error；
- 拟合在 log10(surrogate_mse) 空间中进行，避免少数大 MSE 点把线性横轴拟合线拉歪；
- 输出文件名带 fixed/logx/relative，避免覆盖旧图。
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
# 必须与 surrogate_error_physics_informed.py 的 BASE_OUTPUT_DIR 一致
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\lubangxing\SIlubangxing\surrogate_error\result_2'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

FIG_DPI = 300
EPS = 1e-12

SUMMARY_CANDIDATE_FILES = [
    "batch_summary_physics_informed.csv",
    "batch_summary_physics_informed_rebuilt.csv",
    "batch_summary_pi.csv",
    "batch_summary_v3.csv",
]

ORIGIN_TABLE_NAME = "origin_plot_table_with_fit_logx_relative_fixed.csv"
ORIGIN_STATS_NAME = "origin_plot_fit_stats_logx_relative_fixed.csv"
ORIGIN_FIG_NAME = "origin_style_plot_logx_relative_fixed.png"


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
    """优先读取第一个 physics-informed 主脚本生成的总表；否则从 sample_metrics.csv 重建。"""
    for filename in SUMMARY_CANDIDATE_FILES:
        summary_path = os.path.join(base_output_dir, filename)
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            print(f"Loaded summary from: {summary_path}")
            return df, summary_path

    pattern = os.path.join(base_output_dir, "batch_summary*.csv")
    extra_candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if extra_candidates:
        summary_path = extra_candidates[0]
        df = pd.read_csv(summary_path)
        print(f"Loaded summary from fallback batch summary: {summary_path}")
        return df, summary_path

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


# =========================
# Build Plot Data
# =========================
def build_origin_plot_table(summary_df):
    """
    生成图中真正需要的散点表。

    优先使用主脚本输出的：
        nu_rel_error, kappa_rel_error, d_rel_error
    并转换为百分比。

    如果这些列不存在，才用 true/optimized 参数回退计算：
        |pred - true| / max(|true|, eps) * 100
    """
    required_base = [
        'surrogate_mse',
        'nu_standard', 'kappa_standard', 'D_standard',
        'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized'
    ]
    ensure_required_columns(summary_df, required_base)

    df = summary_df.copy().replace([np.inf, -np.inf], np.nan)

    if all(c in df.columns for c in ['nu_rel_error', 'kappa_rel_error', 'd_rel_error']):
        df['nu_error_percent'] = df['nu_rel_error'] * 100.0
        df['kappa_error_percent'] = df['kappa_rel_error'] * 100.0
        df['d_error_percent'] = df['d_rel_error'] * 100.0
        error_source = 'summary_rel_error_columns'
    else:
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
        error_source = 'computed_relative_error'

    keep_cols = [
        'surrogate_mse',
        'nu_error_percent', 'kappa_error_percent', 'd_error_percent'
    ]
    if 'filename' in df.columns:
        keep_cols = ['filename'] + keep_cols
    if 'param_rel_error_percent' in df.columns:
        keep_cols.append('param_rel_error_percent')
    if 'normalized_objective' in df.columns:
        keep_cols.append('normalized_objective')

    plot_df = df[keep_cols].copy()
    plot_df = plot_df.dropna(
        subset=['surrogate_mse', 'nu_error_percent', 'kappa_error_percent', 'd_error_percent']
    )
    plot_df = plot_df[plot_df['surrogate_mse'] > 0].copy()
    plot_df['log10_surrogate_mse'] = np.log10(np.maximum(plot_df['surrogate_mse'], EPS))
    plot_df['error_source'] = error_source
    plot_df = plot_df.sort_values(by='surrogate_mse').reset_index(drop=True)

    if len(plot_df) < 2:
        raise ValueError("有效样本少于 2 个，无法做拟合和相关性分析。")

    return plot_df


def _fit_one_line_logx(plot_df, y_col, x_fit_log):
    """在 log10(surrogate_mse) 空间拟合 y = a*log10(MSE)+b。"""
    x_log = plot_df['log10_surrogate_mse'].values.astype(float)
    y = plot_df[y_col].values.astype(float)

    finite_mask = np.isfinite(x_log) & np.isfinite(y)
    x_log = x_log[finite_mask]
    y = y[finite_mask]

    if len(x_log) < 2 or np.nanstd(x_log) < EPS or np.nanstd(y) < EPS:
        slope = 0.0
        intercept = float(np.nanmean(y)) if len(y) else np.nan
        y_fit = np.full_like(x_fit_log, intercept, dtype=float)
        return y_fit, {
            'pearson_r_logx': np.nan,
            'pearson_p_logx': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'slope_vs_log10_mse': slope,
            'intercept': intercept,
            'n_samples': len(x_log)
        }

    reg = linregress(x_log, y)
    pearson_val = pearsonr(x_log, y)
    spearman_val = spearmanr(x_log, y)
    y_fit = reg.slope * x_fit_log + reg.intercept

    # 参数误差不能为负，图上拟合线裁剪到 0 以上，避免视觉误导。
    y_fit = np.maximum(y_fit, 0.0)

    return y_fit, {
        'pearson_r_logx': pearson_val[0],
        'pearson_p_logx': pearson_val[1],
        'spearman_rho': spearman_val[0],
        'spearman_p': spearman_val[1],
        'slope_vs_log10_mse': reg.slope,
        'intercept': reg.intercept,
        'n_samples': len(x_log)
    }


def add_fit_columns_to_same_table(plot_df, n_fit=300):
    """把散点数据和 log-x 拟合线数据合并到同一个 CSV。"""
    x_min = max(float(plot_df['surrogate_mse'].min()), EPS)
    x_max = max(float(plot_df['surrogate_mse'].max()), x_min * (1.0 + 1e-6))
    fit_x = np.logspace(np.log10(x_min), np.log10(x_max), n_fit)
    fit_x_log = np.log10(fit_x)

    y_fit_nu, stats_nu = _fit_one_line_logx(plot_df, 'nu_error_percent', fit_x_log)
    y_fit_kappa, stats_kappa = _fit_one_line_logx(plot_df, 'kappa_error_percent', fit_x_log)
    y_fit_d, stats_d = _fit_one_line_logx(plot_df, 'd_error_percent', fit_x_log)

    fit_df = pd.DataFrame({
        'fit_x_surrogate_mse': fit_x,
        'fit_x_log10_surrogate_mse': fit_x_log,
        'fit_nu_error_percent': y_fit_nu,
        'fit_kappa_error_percent': y_fit_kappa,
        'fit_d_error_percent': y_fit_d,
    })

    max_len = max(len(plot_df), len(fit_df))
    merged_df = pd.concat(
        [plot_df.reindex(range(max_len)), fit_df.reindex(range(max_len))],
        axis=1
    )

    stats_nu['parameter'] = 'nu'
    stats_kappa['parameter'] = 'kappa'
    stats_d['parameter'] = 'd'
    stats_df = pd.DataFrame([stats_nu, stats_kappa, stats_d])
    stats_df = stats_df[[
        'parameter', 'pearson_r_logx', 'pearson_p_logx', 'spearman_rho', 'spearman_p',
        'slope_vs_log10_mse', 'intercept', 'n_samples'
    ]]

    return merged_df, stats_df, fit_df


# =========================
# Plot
# =========================
def plot_from_origin_table(plot_df, fit_df, stats_df, output_dir):
    fig_path = os.path.join(output_dir, ORIGIN_FIG_NAME)

    param_configs = [
        ('nu_error_percent', 'fit_nu_error_percent', r'$\nu$ relative error (%)', 'nu'),
        ('kappa_error_percent', 'fit_kappa_error_percent', r'$\kappa$ relative error (%)', 'kappa'),
        ('d_error_percent', 'fit_d_error_percent', r'$d$ relative error (%)', 'd'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for ax, (y_col, fit_col, y_label, short_name) in zip(axes, param_configs):
        x = plot_df['surrogate_mse'].values.astype(float)
        y = plot_df[y_col].values.astype(float)

        stat_row = stats_df[stats_df['parameter'] == short_name].iloc[0]
        pearson_r = stat_row['pearson_r_logx']
        spearman_rho = stat_row['spearman_rho']

        ax.scatter(
            x, y,
            s=28,
            alpha=0.70,
            color='#4C72B0',
            edgecolors='none',
            zorder=2
        )
        ax.plot(
            fit_df['fit_x_surrogate_mse'], fit_df[fit_col],
            '--',
            color='#D62728',
            linewidth=2.0,
            zorder=3
        )

        ax.set_xscale('log')
        ax.set_xlabel('Surrogate Model Prediction MSE', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, which='both', alpha=0.20, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'({short_name})', fontsize=12, pad=8)

        pearson_text = 'nan' if pd.isna(pearson_r) else f'{pearson_r:.3f}'
        spearman_text = 'nan' if pd.isna(spearman_rho) else f'{spearman_rho:.3f}'
        text_str = (
            f'Pearson r(log x) = {pearson_text}\n'
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

    fig.suptitle('Physics-informed Surrogate Error vs Parameter Relative Errors', fontsize=14, y=1.02)
    fig.tight_layout()

    fig_path = safe_savefig(fig, fig_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    return fig_path


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Build corrected Origin-ready table for physics-informed surrogate-error analysis ===")
    print(f"Base output dir: {BASE_OUTPUT_DIR}")

    summary_df, summary_path = load_existing_summary(BASE_OUTPUT_DIR)
    print(f"Total samples loaded: {len(summary_df)}")
    print(f"Summary source: {summary_path}")

    plot_df = build_origin_plot_table(summary_df)
    merged_df, stats_df, fit_df = add_fit_columns_to_same_table(plot_df, n_fit=300)

    table_path = os.path.join(BASE_OUTPUT_DIR, ORIGIN_TABLE_NAME)
    stats_path = os.path.join(BASE_OUTPUT_DIR, ORIGIN_STATS_NAME)

    table_path = safe_to_csv(merged_df, table_path, index=False)
    stats_path = safe_to_csv(stats_df, stats_path, index=False)
    fig_path = plot_from_origin_table(plot_df, fit_df, stats_df, BASE_OUTPUT_DIR)

    print(f"Origin-ready table with fit saved to: {table_path}")
    print(f"Fit stats saved to: {stats_path}")
    print(f"Plot saved to: {fig_path}")
    print("=== Done ===")
