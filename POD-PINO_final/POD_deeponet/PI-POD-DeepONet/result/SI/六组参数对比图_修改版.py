import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad


# --- Configuration (配置) ---

# 1. 仿真数据与输出路径
SIM_DATA_BASE_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'
OUTPUT_DIR_PLOT = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\result\SI\1'

# 2. 只从这个文件读取 filename 和标准参数
# 必须包含列：filename, nu_standard, kappa_standard, D_standard
STANDARD_PARAMS_FILE = (
    r'D:\PINN\zenodo\POD_deeponet\SI_result_v3_matched\batch_summary_v3.csv'
)

os.makedirs(OUTPUT_DIR_PLOT, exist_ok=True)


# --- 六组参数 ---
# 标准参数仍用于在 STANDARD_PARAMS_FILE 中匹配对应的 filename；
# 各方法的辨识参数直接采用图片表格中的数值，不再读取辨识结果文件。
CASES = [
    {
        'standard': {'nu': 5.32, 'kappa': 2.13, 'd_diffusion': 5.32},
        'FD': {'nu': 5.46, 'kappa': 2.19, 'd_diffusion': 5.41},
        'DeepONet': {'nu': 5.05, 'kappa': 2.10, 'd_diffusion': 6.33},
        'POD-DeepONet': {'nu': 4.94, 'kappa': 2.05, 'd_diffusion': 5.06},
        'POD-PINNO': {'nu': 5.36, 'kappa': 2.13, 'd_diffusion': 5.67},
    },
    {
        'standard': {'nu': 7.74, 'kappa': 2.58, 'd_diffusion': 7.74},
        'FD': {'nu': 7.83, 'kappa': 2.61, 'd_diffusion': 7.74},
        'DeepONet': {'nu': 7.57, 'kappa': 2.55, 'd_diffusion': 8.28},
        'POD-DeepONet': {'nu': 7.34, 'kappa': 2.50, 'd_diffusion': 6.91},
        'POD-PINNO': {'nu': 7.71, 'kappa': 2.56, 'd_diffusion': 7.83},
    },
    {
        'standard': {'nu': 11.24, 'kappa': 3.24, 'd_diffusion': 11.24},
        'FD': {'nu': 10.92, 'kappa': 3.14, 'd_diffusion': 11.32},
        'DeepONet': {'nu': 10.54, 'kappa': 3.05, 'd_diffusion': 11.51},
        'POD-DeepONet': {'nu': 10.92, 'kappa': 3.21, 'd_diffusion': 10.54},
        'POD-PINNO': {'nu': 10.72, 'kappa': 3.05, 'd_diffusion': 11.62},
    },
    {
        'standard': {'nu': 15.54, 'kappa': 4.05, 'd_diffusion': 15.54},
        'FD': {'nu': 17.08, 'kappa': 4.45, 'd_diffusion': 17.05},
        'DeepONet': {'nu': 15.81, 'kappa': 4.13, 'd_diffusion': 15.88},
        'POD-DeepONet': {'nu': 16.21, 'kappa': 4.31, 'd_diffusion': 14.91},
        'POD-PINNO': {'nu': 16.74, 'kappa': 4.36, 'd_diffusion': 15.94},
    },
    {
        'standard': {'nu': 16.62, 'kappa': 4.25, 'd_diffusion': 16.62},
        'FD': {'nu': 17.41, 'kappa': 4.47, 'd_diffusion': 17.31},
        'DeepONet': {'nu': 16.15, 'kappa': 4.16, 'd_diffusion': 16.16},
        'POD-DeepONet': {'nu': 16.75, 'kappa': 4.40, 'd_diffusion': 15.16},
        'POD-PINNO': {'nu': 17.19, 'kappa': 4.42, 'd_diffusion': 14.37},
    },
    {
        'standard': {'nu': 17.29, 'kappa': 4.37, 'd_diffusion': 17.29},
        'FD': {'nu': 19.63, 'kappa': 4.96, 'd_diffusion': 18.79},
        'DeepONet': {'nu': 17.97, 'kappa': 4.56, 'd_diffusion': 16.90},
        'POD-DeepONet': {'nu': 18.92, 'kappa': 4.93, 'd_diffusion': 14.54},
        'POD-PINNO': {'nu': 18.91, 'kappa': 4.89, 'd_diffusion': 17.80},
    },
]


# --- Functions (数学函数) ---

def stationary_pdf(a, nu, kappa, d_diffusion):
    """计算未归一化的平稳 PDF。"""
    if d_diffusion <= 0:
        return np.zeros_like(a)
    a = np.maximum(a, 1e-9)
    return a * np.exp(
        (nu / (2 * d_diffusion)) * a ** 2
        - (kappa / (32 * d_diffusion)) * a ** 4
    )


def normalized_stationary_pdf(a, nu, kappa, d_diffusion):
    """计算归一化后的平稳 PDF。"""
    unnormalized_func = lambda x: stationary_pdf(x, nu, kappa, d_diffusion)
    try:
        integral_value, _ = quad(unnormalized_func, 0, 50, limit=1000)
        if integral_value <= 0 or not np.isfinite(integral_value):
            return np.zeros_like(a)
        return stationary_pdf(a, nu, kappa, d_diffusion) / integral_value
    except Exception:
        return np.zeros_like(a)


def find_matching_row(df, standard_params, atol=1e-6):
    """按三项标准参数找到对应算例所在行。"""
    mask = (
        np.isclose(df['nu_standard'], standard_params['nu'], atol=atol, rtol=0)
        & np.isclose(df['kappa_standard'], standard_params['kappa'], atol=atol, rtol=0)
        & np.isclose(df['D_standard'], standard_params['d_diffusion'], atol=atol, rtol=0)
    )
    matched = df.loc[mask]

    if matched.empty:
        return None
    if len(matched) > 1:
        print(
            f"  Warning: 标准参数 {standard_params} 匹配到 {len(matched)} 行，"
            "默认使用第一行。"
        )
    return matched.iloc[0]


# --- Main Execution (主程序) ---

print('Step 1: Reading standard parameter CSV file...')

try:
    df_standard = pd.read_csv(STANDARD_PARAMS_FILE)
except FileNotFoundError as exc:
    raise SystemExit(f'Error: 标准参数文件不存在：{exc}')

required_columns = {'filename', 'nu_standard', 'kappa_standard', 'D_standard'}
missing_columns = required_columns - set(df_standard.columns)
if missing_columns:
    raise SystemExit(
        'Error: 标准参数文件缺少列：' + ', '.join(sorted(missing_columns))
    )

print(f'Loaded standard parameter table: {len(df_standard)} rows')
print('Only the six parameter groups in the table will be processed.\n')


# 曲线样式
METHOD_STYLES = {
    'POD-DeepONet': dict(color='#d62728', linewidth=2.5, linestyle='-'),
    'DeepONet': dict(color='#1f77b4', linewidth=2.0, linestyle='--'),
    'FD': dict(color='#2ca02c', linewidth=2.0, linestyle='-.'),
    'POD-PINNO': dict(color='#ff7f0e', linewidth=2.0, linestyle=(0, (5, 1))),
}

processed_count = 0

for case_index, case in enumerate(CASES, start=1):
    params_standard = case['standard']
    matched_row = find_matching_row(df_standard, params_standard)

    if matched_row is None:
        print(
            f'Case {case_index}: 未在标准参数文件中找到 '
            f"(nu={params_standard['nu']}, kappa={params_standard['kappa']}, "
            f"D={params_standard['d_diffusion']})，跳过。"
        )
        continue

    current_filename = str(matched_row['filename'])
    data_file_sim = os.path.join(SIM_DATA_BASE_DIR, current_filename)
    print(f'Case {case_index}/6 - Processing: {current_filename}')

    try:
        df_sim = pd.read_csv(data_file_sim)
        if 'Envelope' not in df_sim.columns:
            raise KeyError("缺少 'Envelope' 列")
        envelope_data = df_sim['Envelope'].dropna().to_numpy()
        if envelope_data.size == 0:
            raise ValueError('Envelope 数据为空')
    except Exception as exc:
        print(f'  Warning: 无法读取仿真数据，跳过。({exc})')
        continue

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        envelope_data,
        bins=100,
        density=True,
        alpha=0.6,
        label='Simulation Data (Hist)',
        color='lightgray',
        edgecolor='white',
    )

    a_max = np.max(envelope_data) * 1.3
    a_values = np.linspace(0, a_max, 500)

    # 四种辨识方法：参数直接来自图片表格
    pdf_results = {}
    for method_name in ('POD-DeepONet', 'DeepONet', 'FD', 'POD-PINNO'):
        pdf_results[method_name] = normalized_stationary_pdf(
            a_values, **case[method_name]
        )
        ax.plot(
            a_values,
            pdf_results[method_name],
            label=method_name,
            **METHOD_STYLES[method_name],
        )

    # 标准参数仍由文件确定对应算例，但数值使用匹配行中的标准参数
    params_standard_from_file = {
        'nu': float(matched_row['nu_standard']),
        'kappa': float(matched_row['kappa_standard']),
        'd_diffusion': float(matched_row['D_standard']),
    }
    pdf_standard = normalized_stationary_pdf(
        a_values, **params_standard_from_file
    )
    ax.plot(
        a_values,
        pdf_standard,
        label='Standard Parameters',
        color='purple',
        linewidth=1.5,
        linestyle=':',
    )

    case_name = os.path.splitext(current_filename)[0]
    ax.set_title(f'PDF Comparison: {case_name}', fontsize=14)
    ax.set_xlabel('Amplitude (A)', fontsize=12)
    ax.set_ylabel('Probability Density P(A)', fontsize=12)
    ax.set_xlim(0, a_max)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 保存图片：保存方式与原程序一致
    output_plot_path = os.path.join(
        OUTPUT_DIR_PLOT, f'pdf_compare_{case_name}.png'
    )
    try:
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    except Exception as exc:
        print(f'  Error saving plot: {exc}')

    # 保存曲线数据：仍然每个算例输出一个 CSV
    csv_output_path = os.path.join(
        OUTPUT_DIR_PLOT, f'pdf_data_{case_name}.csv'
    )
    try:
        df_export = pd.DataFrame({
            'Amplitude': a_values,
            'PDF_POD_DeepONet': pdf_results['POD-DeepONet'],
            'PDF_DeepONet': pdf_results['DeepONet'],
            'PDF_FD': pdf_results['FD'],
            'PDF_POD_PINNO': pdf_results['POD-PINNO'],
            'PDF_Standard': pdf_standard,
        })
        df_export.to_csv(csv_output_path, index=False)
    except Exception as exc:
        print(f'  Error saving CSV data: {exc}')

    plt.close(fig)
    processed_count += 1

print(f'\nBatch processing finished. Successfully processed {processed_count}/6 cases.')
