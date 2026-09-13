import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

# --- Configuration (配置) ---

# 1. 基础路径设置
SIM_DATA_BASE_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\sim_data\sim_data_4_best'  # 仿真数据(.csv)文件夹
OUTPUT_DIR_PLOT = r'D:\PINN\zenodo\POD_deeponet\PDF_comparison_All_Methods_CSV'  # 结果保存路径

# 2. 参数文件路径 (请修改为实际的 .csv 文件路径)
# 注意：这里假设这些文件都是 CSV 格式
POD_PARAMS_FILE = r'D:\PINN\zenodo\POD_deeponet\SI_result_v3_matched\batch_summary_v3.csv'
DEEPONET_PARAMS_FILE = r'D:\PINN\zenodo\POD_deeponet\compare\result\DeepONet_result\deeponet_batch_optimization_summary_weighted.csv'
FD_PARAMS_FILE = r'D:\PINN\zenodo\POD_deeponet\compare\result\POD-DeepONet_result\batch_summary_v3.csv'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR_PLOT, exist_ok=True)


# --- Functions (数学函数) ---

def stationary_pdf(a, nu, kappa, d_diffusion):
    """计算未归一化的平稳 PDF"""
    if d_diffusion <= 0:
        return np.zeros_like(a)
    a = np.maximum(a, 1e-9)
    return a * np.exp((nu / (2 * d_diffusion)) * a ** 2 - (kappa / (32 * d_diffusion)) * a ** 4)


def normalized_stationary_pdf(a, nu, kappa, d_diffusion):
    """计算归一化后的平稳 PDF"""
    unnormalized_func = lambda x: stationary_pdf(x, nu, kappa, d_diffusion)
    try:
        # 积分上限设为 50，通常足够覆盖包络范围
        integral_value, _ = quad(unnormalized_func, 0, 50, limit=1000)
        if integral_value <= 0:
            return np.zeros_like(a)
        normalization_constant = 1.0 / integral_value
    except Exception as e:
        # 遇到积分错误时打印警告，但不中断程序
        # print(f"Integration warning (nu={nu:.2f}, k={kappa:.2f}, d={d_diffusion:.2f}): {e}")
        return np.zeros_like(a)
    return normalization_constant * stationary_pdf(a, nu, kappa, d_diffusion)


# --- Main Execution (主程序) ---

print("Step 1: Reading Parameter CSV Files...")

try:
    # 1. 读取 POD-DeepONet 数据 (CSV)
    # 假设它是主表，包含标准参数(Standard)和POD参数
    df_pod = pd.read_csv(POD_PARAMS_FILE)
    print(f"Loaded POD params: {len(df_pod)} rows")

    # 2. 读取 DeepONet 数据 (CSV)
    df_don = pd.read_csv(DEEPONET_PARAMS_FILE)
    # 为了避免列名冲突，重命名识别出的参数列
    df_don = df_don[['filename', 'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized']].rename(
        columns={
            'nu_optimized': 'nu_don',
            'kappa_optimized': 'kappa_don',
            'd_diffusion_optimized': 'd_don'
        }
    )
    print(f"Loaded DeepONet params: {len(df_don)} rows")

    # 3. 读取 FD 数据 (CSV)
    df_fd = pd.read_csv(FD_PARAMS_FILE)
    # 为了避免列名冲突，重命名识别出的参数列
    df_fd = df_fd[['filename', 'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized']].rename(
        columns={
            'nu_optimized': 'nu_fd',
            'kappa_optimized': 'kappa_fd',
            'd_diffusion_optimized': 'd_fd'
        }
    )
    print(f"Loaded FD params: {len(df_fd)} rows")

    # 4. 合并数据表 (Merge)
    # 基于 'filename' 列进行合并，确保不同文件中同一算例的参数对齐
    print("Merging dataframes based on filename...")
    df_merged = pd.merge(df_pod, df_don, on='filename', how='inner')
    df_merged = pd.merge(df_merged, df_fd, on='filename', how='inner')

    print(f"Successfully merged. Total files to process: {len(df_merged)}")

except FileNotFoundError as e:
    print(f"Error: Could not find one of the CSV files. Details: {e}")
    exit()
except KeyError as e:
    print(f"Error: Missing expected column in CSV file. Details: {e}")
    print(
        "Please ensure input CSVs have columns: 'filename', 'nu_optimized', 'kappa_optimized', 'd_diffusion_optimized'")
    exit()

# --- Iterate and Process (循环处理绘图) ---

for index, row in df_merged.iterrows():
    current_filename = row['filename']
    DATA_FILE_SIM = os.path.join(SIM_DATA_BASE_DIR, current_filename)

    print(f"Processing: {current_filename}")

    # --- Extract Parameters ---

    # POD-DeepONet Params
    params_pod = {
        'nu': row['nu_optimized'],
        'kappa': row['kappa_optimized'],
        'd_diffusion': row['d_diffusion_optimized']
    }

    # DeepONet Params (重命名后的列)
    params_don = {
        'nu': row['nu_don'],
        'kappa': row['kappa_don'],
        'd_diffusion': row['d_don']
    }

    # FD Params (重命名后的列)
    params_fd = {
        'nu': row['nu_fd'],
        'kappa': row['kappa_fd'],
        'd_diffusion': row['d_fd']
    }

    # Standard/True Params (来自 POD 文件中的标准值)
    params_standard = {
        'nu': row['nu_standard'],
        'kappa': row['kappa_standard'],
        'd_diffusion': row['D_standard']
    }

    # --- Read Simulation Data ---
    try:
        # 读取仿真数据的 CSV
        df_sim = pd.read_csv(DATA_FILE_SIM)
        envelope_data = df_sim['Envelope'].values
    except Exception as e:
        print(f"  Warning: Could not read simulation data for {current_filename}. Skipping. ({e})")
        continue

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. 绘制数据直方图
    # density=True 确保面积为1，变成概率密度
    ax.hist(envelope_data, bins=100, density=True, alpha=0.6,
            label='Simulation Data (Hist)', color='lightgray', edgecolor='white')

    # 定义绘图的横坐标范围 (幅度 A)
    a_max = np.max(envelope_data) * 1.3
    a_values = np.linspace(0, a_max, 500)

    # 2. 绘制各方法的理论 PDF 曲线

    # POD-DeepONet (红色实线)
    pdf_pod = normalized_stationary_pdf(a_values, **params_pod)
    ax.plot(a_values, pdf_pod, label='POD-DeepONet', color='#d62728', linewidth=2.5)

    # DeepONet (蓝色虚线)
    pdf_don = normalized_stationary_pdf(a_values, **params_don)
    ax.plot(a_values, pdf_don, label='DeepONet', color='#1f77b4', linewidth=2, linestyle='--')

    # FD Identification (绿色点划线)
    pdf_fd = normalized_stationary_pdf(a_values, **params_fd)
    ax.plot(a_values, pdf_fd, label='FD Identification', color='#2ca02c', linewidth=2, linestyle='-.')

    # Standard Parameters (紫色点线 - 真值参考)
    pdf_standard = normalized_stationary_pdf(a_values, **params_standard)
    ax.plot(a_values, pdf_standard, label='Standard Parameters', color='purple', linewidth=1.5, linestyle=':')

    # 图表设置
    ax.set_title(f'PDF Comparison: {current_filename.replace(".csv", "")}', fontsize=14)
    ax.set_xlabel('Amplitude (A)', fontsize=12)
    ax.set_ylabel('Probability Density P(A)', fontsize=12)
    ax.set_xlim(0, a_max)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Save Plot ---
    output_plot_path = os.path.join(OUTPUT_DIR_PLOT, f"pdf_compare_{current_filename.replace('.csv', '')}.png")
    try:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        # print(f"  Plot saved: {output_plot_path}")
    except Exception as e:
        print(f"  Error saving plot: {e}")

    # --- Save Curve Data to CSV ---
    # 保存曲线数据，方便后续用 Origin 等软件作图
    csv_output_path = os.path.join(OUTPUT_DIR_PLOT, f"pdf_data_{current_filename.replace('.csv', '')}.csv")
    try:
        df_export = pd.DataFrame({
            'Amplitude': a_values,
            'PDF_POD_DeepONet': pdf_pod,
            'PDF_DeepONet': pdf_don,
            'PDF_FD': pdf_fd,
            'PDF_Standard': pdf_standard
        })
        df_export.to_csv(csv_output_path, index=False)
    except Exception as e:
        print(f"  Error saving CSV data: {e}")

    plt.close(fig)  # 关闭图像释放内存

print("\nBatch processing finished.")