# -*- coding: utf-8 -*-
#保存了CSV文件但是没有保存相对误差
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d, griddata
from scipy.stats import wilcoxon, ttest_rel
import time
from tqdm import tqdm
import traceback
import gc

# --- PyTorch Imports for DeepONet ---
import torch
import torch.nn as nn

# --- Configuration ---
BASE_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'

# 修改：创建分层输出目录
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\data_study_analysis_v3'
DIR_METRICS = os.path.join(BASE_OUTPUT_DIR, 'metrics_per_size')
DIR_FIELDS = os.path.join(BASE_OUTPUT_DIR, 'field_data_samples')
DIR_PLOTS = os.path.join(BASE_OUTPUT_DIR, 'plots')
DIR_STATS = os.path.join(BASE_OUTPUT_DIR, 'statistics')

for d in [DIR_METRICS, DIR_FIELDS, DIR_PLOTS, DIR_STATS]:
    os.makedirs(d, exist_ok=True)

V3_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\train_result_v3_data_study'
BASE_RUN_ID = "pod_v3_cosine_warmup_layernorm_weightedloss"

# Define the data sizes used in training
DATA_SIZES = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# DeepONet Configuration
DEEPONET_BRANCH_INPUT_DIM = 3
DEEPONET_HIDDEN_UNITS = 256
DEEPONET_NUM_HIDDEN_LAYERS = 4
DEEPONET_DROPOUT_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Test Configuration
NUM_TEST_FILES = 120
NUM_VISUALIZATION_SAMPLES = 3


# --- Model Definitions ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE)
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes,
                 pod_basis, y_mean_pod_scaled, dropout_rate):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, dropout_rate)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod_scaled, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled

    def predict(self, branch_x, y_mean_scaler, y_std_scaler):
        y_pred_scaled = self.forward(branch_x)
        y_mean_scaler_t = torch.tensor(y_mean_scaler, dtype=DTYPE, device=self.pod_basis.device)
        y_std_scaler_t = torch.tensor(y_std_scaler, dtype=DTYPE, device=self.pod_basis.device)
        return y_pred_scaled * y_std_scaler_t + y_mean_scaler_t


# --- Helper Functions ---
def manual_scaler_transform(data, mean, std):
    if isinstance(data, torch.Tensor):
        device = data.device
        data_t = data
    else:
        device = DEVICE
        data_t = torch.tensor(data, dtype=DTYPE, device=device)
    mean_t = torch.tensor(mean, dtype=DTYPE, device=device)
    std_t = torch.tensor(std, dtype=DTYPE, device=device)
    std_t[std_t < 1e-10] = 1.0
    return (data_t - mean_t) / std_t


def parse_params_from_filename(filepath):
    filename = os.path.basename(filepath)
    try:
        if filename.startswith('(') and filename.endswith('.csv'):
            values = filename[1:-5].split(',')
            return float(values[0]), float(values[1]), float(values[2])
    except (ValueError, IndexError):
        pass
    return None, None, None


def load_model_for_data_size(data_size, v3_result_dir, base_run_id):
    run_id = f"{base_run_id}_data_{data_size}"
    model_path = os.path.join(v3_result_dir, f'model_{run_id}.pth')
    scaler_path = os.path.join(v3_result_dir, f'scalers_{run_id}.pth')
    pod_params_path = os.path.join(v3_result_dir, f'pod_params_{run_id}.pth')

    try:
        scalers = torch.load(scaler_path, map_location=DEVICE)
        pod_params = torch.load(pod_params_path, map_location=DEVICE)

        pod_basis = pod_params['pod_basis']
        y_mean_pod_scaled = pod_params['y_mean_pod_scaled']
        num_pod_modes = pod_params['num_pod_modes']
        unified_a_grid = pod_params['unified_a_grid']
        unified_tau_grid = pod_params['unified_tau_grid']

        if isinstance(unified_a_grid, torch.Tensor): unified_a_grid = unified_a_grid.cpu().numpy()
        if isinstance(unified_tau_grid, torch.Tensor): unified_tau_grid = unified_tau_grid.cpu().numpy()

        model = PODDeepONet(
            DEEPONET_BRANCH_INPUT_DIM, DEEPONET_HIDDEN_UNITS, DEEPONET_NUM_HIDDEN_LAYERS,
            num_pod_modes, pod_basis, y_mean_pod_scaled, DEEPONET_DROPOUT_RATE
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return {'model': model, 'scalers': scalers, 'unified_a_grid': unified_a_grid,
                'unified_tau_grid': unified_tau_grid}
    except Exception as e:
        print(f"    ✗ Failed to load model for data size {data_size}: {e}")
        return None


# --- Core Analysis Functions ---

def process_single_file(km_file_path, model_info, true_params, return_fields=False):
    try:
        data_km_df = pd.read_csv(km_file_path)
        tau_indices_map = {idx: ts for idx, ts in data_km_df[['tau_index', 'tau_sec']].drop_duplicates().values}

        model = model_info['model']
        scalers = model_info['scalers']
        unified_tau = model_info['unified_tau_grid']
        unified_a = model_info['unified_a_grid']

        nu, kappa, d_val = true_params
        branch_input_np = np.array([[nu, kappa, d_val]])
        branch_input_scaled_t = manual_scaler_transform(branch_input_np, scalers['branch_mean'], scalers['branch_std'])

        with torch.no_grad():
            y_pred_t = model.predict(branch_input_scaled_t, scalers['y_mean_scaler'], scalers['y_std_scaler'])
            y_pred_np = y_pred_t.cpu().numpy().flatten()

        field_len = len(y_pred_np) // 2
        D1_pred_full = y_pred_np[:field_len].reshape(len(unified_tau), len(unified_a))
        D2_pred_full = y_pred_np[field_len:].reshape(len(unified_tau), len(unified_a))

        mse_d1_accum = 0
        mse_d2_accum = 0
        count = 0

        grouped = data_km_df.groupby('tau_index')

        for tau_idx, group in grouped:
            tau_val = tau_indices_map[tau_idx]
            tau_grid_idx = np.argmin(np.abs(unified_tau - tau_val))

            d1_slice_pred = D1_pred_full[tau_grid_idx, :]
            d2_slice_pred = D2_pred_full[tau_grid_idx, :]

            f_d1 = interp1d(unified_a, d1_slice_pred, kind='linear', fill_value="extrapolate")
            f_d2 = interp1d(unified_a, d2_slice_pred, kind='linear', fill_value="extrapolate")

            d1_pred_vals = f_d1(group['A'].values)
            d2_pred_vals = f_d2(group['A'].values)

            valid = ~np.isnan(group['D1_data'].values) & ~np.isnan(d1_pred_vals)
            if np.any(valid):
                mse_d1_accum += np.sum((group['D1_data'].values[valid] - d1_pred_vals[valid]) ** 2)
                mse_d2_accum += np.sum((group['D2_data'].values[valid] - d2_pred_vals[valid]) ** 2)
                count += np.sum(valid)

        if count == 0: return None

        mse_d1 = mse_d1_accum / count
        mse_d2 = mse_d2_accum / count

        result = {
            'mse_d1': mse_d1,
            'mse_d2': mse_d2,
            'mse_total': mse_d1 + mse_d2
        }

        if return_fields:
            # 保存真实数据
            true_data_df = data_km_df[['tau_sec', 'A', 'D1_data', 'D2_data']].copy()

            # 创建预测场的DataFrame
            A_mesh, Tau_mesh = np.meshgrid(unified_a, unified_tau)
            pred_field_df = pd.DataFrame({
                'tau': Tau_mesh.flatten(),
                'a': A_mesh.flatten(),
                'D1_pred': D1_pred_full.flatten(),
                'D2_pred': D2_pred_full.flatten()
            })

            field_data = {
                'unified_a': unified_a,
                'unified_tau': unified_tau,
                'pred_field_df': pred_field_df,
                'true_data_df': true_data_df,
                'params': true_params
            }
            return result, field_data

        return result, None

    except Exception as e:
        print(f"Error processing file {km_file_path}: {e}")
        traceback.print_exc()
        return None, None


def perform_statistical_tests(results_map, data_sizes):
    stats_results = []
    sorted_sizes = sorted(data_sizes)
    for i in range(len(sorted_sizes) - 1):
        size_a = sorted_sizes[i]
        size_b = sorted_sizes[i + 1]

        if size_a not in results_map or size_b not in results_map:
            continue

        df_a = results_map[size_a].sort_values('filename')
        df_b = results_map[size_b].sort_values('filename')

        merged = pd.merge(df_a[['filename', 'mse_total']], df_b[['filename', 'mse_total']],
                          on='filename', suffixes=('_A', '_B'))

        if len(merged) < 2: continue

        try:
            stat_w, p_w = wilcoxon(merged['mse_total_A'], merged['mse_total_B'], alternative='greater')
            stat_t, p_t = ttest_rel(merged['mse_total_A'], merged['mse_total_B'], alternative='greater')

            stats_results.append({
                'Size_A': size_a,
                'Size_B': size_b,
                'Mean_Diff': (merged['mse_total_A'] - merged['mse_total_B']).mean(),
                'Wilcoxon_p': p_w,
                'Ttest_p': p_t,
                'Significant_0.05': p_w < 0.05
            })
        except ValueError:
            pass

    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        df_stats.to_csv(os.path.join(DIR_STATS, 'significance_tests.csv'), index=False)
        print("\n=== Statistical Significance (Partial) ===")
        print(df_stats[['Size_A', 'Size_B', 'Wilcoxon_p', 'Significant_0.05']].to_string(index=False))


def save_field_data_to_csv(field_data, filename, data_size, output_dir):
    """保存场数据为CSV文件"""
    base_name = os.path.splitext(filename)[0]

    # 保存预测场
    pred_field_df = field_data['pred_field_df']
    pred_csv_path = os.path.join(output_dir, f'pred_field_{base_name}_size_{data_size}.csv')
    pred_field_df.to_csv(pred_csv_path, index=False)

    # 保存真实数据
    true_data_df = field_data['true_data_df']
    true_csv_path = os.path.join(output_dir, f'true_data_{base_name}_size_{data_size}.csv')
    true_data_df.to_csv(true_csv_path, index=False)

    # 保存网格信息
    grid_info = {
        'unified_a': field_data['unified_a'].tolist(),
        'unified_tau': field_data['unified_tau'].tolist(),
        'params': field_data['params']
    }

    grid_df = pd.DataFrame({
        'param': ['unified_a', 'unified_tau', 'nu', 'kappa', 'd'],
        'value': [
            str(field_data['unified_a'].tolist()),
            str(field_data['unified_tau'].tolist()),
            str(field_data['params'][0]),
            str(field_data['params'][1]),
            str(field_data['params'][2])
        ]
    })

    grid_csv_path = os.path.join(output_dir, f'grid_info_{base_name}_size_{data_size}.csv')
    grid_df.to_csv(grid_csv_path, index=False)

    return {
        'pred_path': pred_csv_path,
        'true_path': true_csv_path,
        'grid_path': grid_csv_path
    }


def load_field_data_from_csv(filename, data_size, input_dir):
    """从CSV文件加载场数据"""
    base_name = os.path.splitext(filename)[0]

    # 加载预测场
    pred_csv_path = os.path.join(input_dir, f'pred_field_{base_name}_size_{data_size}.csv')
    pred_field_df = pd.read_csv(pred_csv_path)

    # 加载真实数据
    true_csv_path = os.path.join(input_dir, f'true_data_{base_name}_size_{data_size}.csv')
    true_data_df = pd.read_csv(true_csv_path)

    # 加载网格信息
    grid_csv_path = os.path.join(input_dir, f'grid_info_{base_name}_size_{data_size}.csv')
    grid_df = pd.read_csv(grid_csv_path)

    # 解析网格信息
    grid_info = {}
    for _, row in grid_df.iterrows():
        if row['param'] == 'unified_a':
            unified_a = np.array(eval(row['value']))
        elif row['param'] == 'unified_tau':
            unified_tau = np.array(eval(row['value']))
        elif row['param'] == 'nu':
            nu = float(row['value'])
        elif row['param'] == 'kappa':
            kappa = float(row['value'])
        elif row['param'] == 'd':
            d_val = float(row['value'])

    params = (nu, kappa, d_val)

    # 重新构造D1_pred_full和D2_pred_full
    n_tau = len(unified_tau)
    n_a = len(unified_a)

    D1_pred_full = pred_field_df['D1_pred'].values.reshape(n_tau, n_a)
    D2_pred_full = pred_field_df['D2_pred'].values.reshape(n_tau, n_a)

    return {
        'unified_a': unified_a,
        'unified_tau': unified_tau,
        'D1_pred_full': D1_pred_full,
        'D2_pred_full': D2_pred_full,
        'true_data_df': true_data_df,
        'params': params,
        'data_size': data_size
    }


def plot_field_comparison(field_data_size_min, field_data_size_max, filename, output_dir):
    """
    绘制场对比图 - 从CSV文件加载数据
    """
    # 1. 提取数据
    true_df = field_data_size_max['true_data_df']
    a_grid = field_data_size_max['unified_a']
    tau_grid = field_data_size_max['unified_tau']

    # 确保 grid 是 1D 数组
    if a_grid.ndim > 1: a_grid = a_grid.flatten()
    if tau_grid.ndim > 1: tau_grid = tau_grid.flatten()

    A_mesh, Tau_mesh = np.meshgrid(a_grid, tau_grid)

    # 2. 对真实数据进行插值 (Griddata)
    valid_points = true_df.dropna(subset=['D1_data', 'A', 'tau_sec'])

    if len(valid_points) < 4:
        print(f"Not enough valid points to plot {filename}")
        return

    points = (valid_points['A'].values, valid_points['tau_sec'].values)
    values_d1 = valid_points['D1_data'].values

    # 使用 linear 插值
    D1_true_interp = griddata(points, values_d1, (A_mesh, Tau_mesh), method='linear')

    # 3. 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- D1 Comparison ---
    # True Field
    im1 = axes[0, 0].pcolormesh(A_mesh, Tau_mesh, D1_true_interp, shading='auto', cmap='viridis')
    axes[0, 0].set_title(f'True D1 (Interp) - {filename}')
    fig.colorbar(im1, ax=axes[0, 0])

    # Pred (Small Data)
    im2 = axes[0, 1].pcolormesh(A_mesh, Tau_mesh, field_data_size_min['D1_pred_full'], shading='auto', cmap='viridis')
    axes[0, 1].set_title(f'Pred D1 (N={field_data_size_min["data_size"]})')
    fig.colorbar(im2, ax=axes[0, 1])

    # Pred (Large Data)
    im3 = axes[0, 2].pcolormesh(A_mesh, Tau_mesh, field_data_size_max['D1_pred_full'], shading='auto', cmap='viridis')
    axes[0, 2].set_title(f'Pred D1 (N={field_data_size_max["data_size"]})')
    fig.colorbar(im3, ax=axes[0, 2])

    # --- Error Plots ---
    err_small = np.abs(D1_true_interp - field_data_size_min['D1_pred_full'])
    err_large = np.abs(D1_true_interp - field_data_size_max['D1_pred_full'])

    vmax_err = max(np.nanmax(err_small) if not np.all(np.isnan(err_small)) else 1.0,
                   np.nanmax(err_large) if not np.all(np.isnan(err_large)) else 1.0)

    im4 = axes[1, 1].pcolormesh(A_mesh, Tau_mesh, err_small, shading='auto', cmap='inferno', vmin=0, vmax=vmax_err)
    axes[1, 1].set_title(f'Abs Error (N={field_data_size_min["data_size"]})')
    fig.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(A_mesh, Tau_mesh, err_large, shading='auto', cmap='inferno', vmin=0, vmax=vmax_err)
    axes[1, 2].set_title(f'Abs Error (N={field_data_size_max["data_size"]})')
    fig.colorbar(im5, ax=axes[1, 2])

    # Scatter plot of original data
    axes[1, 0].scatter(valid_points['A'], valid_points['tau_sec'], c=values_d1, s=5, cmap='viridis')
    axes[1, 0].set_title('True Data Points (Scatter)')
    axes[1, 0].set_xlim(a_grid.min(), a_grid.max())
    axes[1, 0].set_ylim(tau_grid.min(), tau_grid.max())

    for ax in axes.flat:
        ax.set_xlabel('A')
        ax.set_ylabel('tau')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'field_comparison_{filename[:-4]}.png'), dpi=150)
    plt.close()


# --- Main Script ---
if __name__ == "__main__":
    print("\n=== Data Study V2: Analysis & Visualization (CSV Format) ===")

    # 1. Select Test Files
    all_km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    np.random.seed(42)
    if len(all_km_files) > NUM_TEST_FILES:
        test_files = np.random.choice(all_km_files, NUM_TEST_FILES, replace=False)
    else:
        test_files = all_km_files

    vis_files = np.random.choice(test_files, min(len(test_files), NUM_VISUALIZATION_SAMPLES), replace=False)
    print(f"Test files: {len(test_files)}")
    print(f"Visualization samples: {vis_files}")

    results_map = {}

    # 2. Processing Loop
    for data_size in DATA_SIZES:
        print(f"\n--- Processing Data Size: {data_size} ---")

        model_info = load_model_for_data_size(data_size, V3_RESULT_DIR, BASE_RUN_ID)
        if model_info is None: continue

        current_size_results = []

        for filename in tqdm(test_files, desc=f"Size {data_size}"):
            km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
            params = parse_params_from_filename(filename)
            if params[0] is None: continue

            is_vis_sample = filename in vis_files

            metrics, field_data = process_single_file(km_file_path, model_info, params, return_fields=is_vis_sample)

            if metrics:
                entry = metrics.copy()
                entry['filename'] = filename
                entry['data_size'] = data_size
                entry['nu'], entry['kappa'], entry['d'] = params
                current_size_results.append(entry)

                if is_vis_sample and field_data:
                    # 保存为CSV格式
                    save_field_data_to_csv(field_data, filename, data_size, DIR_FIELDS)

        if current_size_results:
            df_size = pd.DataFrame(current_size_results)
            save_csv_path = os.path.join(DIR_METRICS, f'metrics_size_{data_size}.csv')
            df_size.to_csv(save_csv_path, index=False)
            results_map[data_size] = df_size

        del model_info
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Aggregate Data
    print("\n=== Generating Aggregated Plots ===")
    if results_map:
        all_data = []
        for size, df in results_map.items():
            all_data.append(df)
        full_df = pd.concat(all_data, ignore_index=True)

        # Plot Boxplot
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='data_size', y='mse_total', data=full_df, palette="Blues")
        plt.yscale('log')
        plt.title('Distribution of Total MSE vs Training Data Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(DIR_PLOTS, 'boxplot_mse_distribution.png'), dpi=200)
        plt.close()

        # Plot Trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='data_size', y='mse_total', data=full_df, marker='o', errorbar=('ci', 95))
        plt.yscale('log')
        plt.title('Mean MSE Trend with 95% Confidence Interval')
        plt.ylabel('Total MSE (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(DIR_PLOTS, 'trend_mse_ci.png'), dpi=200)
        plt.close()

        # Stats
        perform_statistical_tests(results_map, DATA_SIZES)
    else:
        print("No results to plot.")

    # 4. Field Visualization (从CSV文件加载)
    print("\n=== Generating Field Reconstructions ===")
    min_size = min(DATA_SIZES)
    max_size = max(DATA_SIZES)

    for vis_file in vis_files:
        try:
            # 从CSV文件加载场数据
            field_data_min = load_field_data_from_csv(vis_file, min_size, DIR_FIELDS)
            field_data_max = load_field_data_from_csv(vis_file, max_size, DIR_FIELDS)

            plot_field_comparison(field_data_min, field_data_max, vis_file, DIR_PLOTS)
            print(f"Plot generated for {vis_file}")
        except Exception as e:
            print(f"Failed to plot fields for {vis_file}: {e}")
            traceback.print_exc()

    print(f"\nAnalysis Complete. All outputs in: {BASE_OUTPUT_DIR}")