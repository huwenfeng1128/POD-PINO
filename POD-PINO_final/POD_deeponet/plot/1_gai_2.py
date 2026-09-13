# -*- coding: utf-8 -*-
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
BASE_OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\data_study_analysis_v2'
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

        if count == 0: return None, None

        mse_d1 = mse_d1_accum / count
        mse_d2 = mse_d2_accum / count

        result = {
            'mse_d1': mse_d1,
            'mse_d2': mse_d2,
            'mse_total': mse_d1 + mse_d2
        }

        if return_fields:
            field_data = {
                'unified_a': unified_a,
                'unified_tau': unified_tau,
                'D1_pred_full': D1_pred_full,
                'D2_pred_full': D2_pred_full,
                'params': true_params,
                'true_data_df': data_km_df[['tau_sec', 'A', 'D1_data', 'D2_data']].to_dict('list') # 存成 dict 形式
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
            # alternative='greater' tests if A is greater than B, i.e., A's error is higher.
            # We expect larger data size (B) to have smaller error, so A's error > B's error.
            stat_w, p_w = wilcoxon(merged['mse_total_A'], merged['mse_total_B'], alternative='greater')
            stat_t, p_t = ttest_rel(merged['mse_total_A'], merged['mse_total_B'], alternative='greater')

            stats_results.append({
                'Size_A': size_a,
                'Size_B': size_b,
                'Mean_MSE_A': merged['mse_total_A'].mean(),
                'Mean_MSE_B': merged['mse_total_B'].mean(),
                'Mean_Diff': (merged['mse_total_A'] - merged['mse_total_B']).mean(),
                'Wilcoxon_p': p_w,
                'Ttest_p': p_t,
                'Significant_0.05_Wilcoxon': p_w < 0.05,
                'Significant_0.05_Ttest': p_t < 0.05
            })
        except ValueError as e:
            print(f"Skipping statistical test between {size_a} and {size_b} due to error: {e}")
            pass

    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        df_stats.to_csv(os.path.join(DIR_STATS, 'significance_tests.csv'), index=False)
        print("\n=== Statistical Significance (Partial) ===")
        print(df_stats[['Size_A', 'Size_B', 'Mean_Diff', 'Wilcoxon_p', 'Significant_0.05_Wilcoxon']].to_string(index=False))
    else:
        print("\nNo statistical test results to display.")


def save_field_data_and_plot_comparison(field_data_size_min, field_data_size_max, filename, output_dir):
    """
    保存场数据到 CSV 并绘制 D1 场对比图。
    """
    try:
        # 1. 提取真实数据 (处理 0-d array 问题)
        true_data_dict = field_data_size_max['true_data_df'].item() if field_data_size_max['true_data_df'].ndim == 0 else field_data_size_max['true_data_df']
        true_df = pd.DataFrame(true_data_dict)

        a_grid = field_data_size_max['unified_a']
        tau_grid = field_data_size_max['unified_tau']

        # 确保 grid 是 1D 数组
        if a_grid.ndim > 1: a_grid = a_grid.flatten()
        if tau_grid.ndim > 1: tau_grid = tau_grid.flatten()

        A_mesh, Tau_mesh = np.meshgrid(a_grid, tau_grid)

        # 初始化一个DataFrame来存储所有场信息，方便Origin绘图
        field_df_to_save = pd.DataFrame({
            'A': A_mesh.flatten(),
            'Tau': Tau_mesh.flatten()
        })

        # --- 处理 D1 场 ---
        valid_points_d1 = true_df.dropna(subset=['D1_data', 'A', 'tau_sec'])
        D1_true_interp = None
        if len(valid_points_d1) >= 4: # griddata 需要至少4个点
            points_d1 = (valid_points_d1['A'].values, valid_points_d1['tau_sec'].values)
            values_d1 = valid_points_d1['D1_data'].values
            D1_true_interp = griddata(points_d1, values_d1, (A_mesh, Tau_mesh), method='linear', fill_value=np.nan)
        else:
            print(f"Warning: Not enough valid D1 data points ({len(valid_points_d1)}) for interpolation in {filename}.")
            D1_true_interp = np.full_like(A_mesh, np.nan)

        field_df_to_save['D1_True_Interp'] = D1_true_interp.flatten()

        # 预测 D1 场
        D1_pred_full_min = field_data_size_min['D1_pred_full']
        D1_pred_full_max = field_data_size_max['D1_pred_full']
        field_df_to_save[f'D1_Pred_Size{field_data_size_min["data_size"]}'] = D1_pred_full_min.flatten()
        field_df_to_save[f'D1_Pred_Size{field_data_size_max["data_size"]}'] = D1_pred_full_max.flatten()

        # D1 绝对误差
        err_d1_small = np.abs(D1_true_interp - D1_pred_full_min)
        err_d1_large = np.abs(D1_true_interp - D1_pred_full_max)
        field_df_to_save[f'D1_Error_Size{field_data_size_min["data_size"]}'] = err_d1_small.flatten()
        field_df_to_save[f'D1_Error_Size{field_data_size_max["data_size"]}'] = err_d1_large.flatten()

        # --- 处理 D2 场 ---
        valid_points_d2 = true_df.dropna(subset=['D2_data', 'A', 'tau_sec'])
        D2_true_interp = None
        if len(valid_points_d2) >= 4:
            points_d2 = (valid_points_d2['A'].values, valid_points_d2['tau_sec'].values)
            values_d2 = valid_points_d2['D2_data'].values
            D2_true_interp = griddata(points_d2, values_d2, (A_mesh, Tau_mesh), method='linear', fill_value=np.nan)
        else:
            print(f"Warning: Not enough valid D2 data points ({len(valid_points_d2)}) for interpolation in {filename}.")
            D2_true_interp = np.full_like(A_mesh, np.nan)

        field_df_to_save['D2_True_Interp'] = D2_true_interp.flatten()

        # 预测 D2 场
        D2_pred_full_min = field_data_size_min['D2_pred_full']
        D2_pred_full_max = field_data_size_max['D2_pred_full']
        field_df_to_save[f'D2_Pred_Size{field_data_size_min["data_size"]}'] = D2_pred_full_min.flatten()
        field_df_to_save[f'D2_Pred_Size{field_data_size_max["data_size"]}'] = D2_pred_full_max.flatten()

        # D2 绝对误差
        err_d2_small = np.abs(D2_true_interp - D2_pred_full_min)
        err_d2_large = np.abs(D2_true_interp - D2_pred_full_max)
        field_df_to_save[f'D2_Error_Size{field_data_size_min["data_size"]}'] = err_d2_small.flatten()
        field_df_to_save[f'D2_Error_Size{field_data_size_max["data_size"]}'] = err_d2_large.flatten()

        # --- 保存场数据到 CSV ---
        csv_filename = f'field_comparison_data_{filename[:-4]}_Nmin_{field_data_size_min["data_size"]}_Nmax_{field_data_size_max["data_size"]}.csv'
        field_df_to_save.to_csv(os.path.join(DIR_FIELDS, csv_filename), index=False)
        print(f"Field data for {filename} saved to {csv_filename}")

        # --- 绘图 (保留 D1 的原绘图逻辑) ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # --- D1 Comparison ---
        # True Field
        if not np.all(np.isnan(D1_true_interp)):
            im1 = axes[0, 0].pcolormesh(A_mesh, Tau_mesh, D1_true_interp, shading='auto', cmap='viridis')
            axes[0, 0].set_title(f'True D1 (Interp) - {filename}')
            fig.colorbar(im1, ax=axes[0, 0])
        else:
            axes[0, 0].set_title(f'True D1 (Interp) - {filename}\n(Not enough data for interp)')
            axes[0, 0].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)


        # Pred (Small Data)
        im2 = axes[0, 1].pcolormesh(A_mesh, Tau_mesh, D1_pred_full_min, shading='auto', cmap='viridis')
        axes[0, 1].set_title(f'Pred D1 (N={field_data_size_min["data_size"]})')
        fig.colorbar(im2, ax=axes[0, 1])

        # Pred (Large Data)
        im3 = axes[0, 2].pcolormesh(A_mesh, Tau_mesh, D1_pred_full_max, shading='auto', cmap='viridis')
        axes[0, 2].set_title(f'Pred D1 (N={field_data_size_max["data_size"]})')
        fig.colorbar(im3, ax=axes[0, 2])

        # --- Error Plots ---
        # 计算误差 (注意: D1_true_interp 包含 NaN，减法会传播 NaN，这是正确的)
        vmax_err = max(np.nanmax(err_d1_small) if not np.all(np.isnan(err_d1_small)) else 1e-6,
                       np.nanmax(err_d1_large) if not np.all(np.isnan(err_d1_large)) else 1e-6)
        if vmax_err == 0: vmax_err = 1e-6 # Avoid zero vmax

        im4 = axes[1, 1].pcolormesh(A_mesh, Tau_mesh, err_d1_small, shading='auto', cmap='inferno', vmin=0, vmax=vmax_err)
        axes[1, 1].set_title(f'Abs Error (N={field_data_size_min["data_size"]})')
        fig.colorbar(im4, ax=axes[1, 1])

        im5 = axes[1, 2].pcolormesh(A_mesh, Tau_mesh, err_d1_large, shading='auto', cmap='inferno', vmin=0, vmax=vmax_err)
        axes[1, 2].set_title(f'Abs Error (N={field_data_size_max["data_size"]})')
        fig.colorbar(im5, ax=axes[1, 2])

        # Scatter plot of original data
        if not valid_points_d1.empty:
            scatter = axes[1, 0].scatter(valid_points_d1['A'], valid_points_d1['tau_sec'], c=values_d1, s=5, cmap='viridis')
            axes[1, 0].set_title('True D1 Data Points (Scatter)')
            fig.colorbar(scatter, ax=axes[1,0])
        else:
            axes[1, 0].set_title('True D1 Data Points (Scatter)\n(No valid data)')
            axes[1, 0].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)

        axes[1, 0].set_xlim(a_grid.min(), a_grid.max())
        axes[1, 0].set_ylim(tau_grid.min(), tau_grid.max())


        for ax in axes.flat:
            ax.set_xlabel('A')
            ax.set_ylabel('tau')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'field_comparison_{filename}.png'), dpi=150)
        plt.close()

    except Exception as e:
        print(f"Failed to save field data or plot for {filename}: {e}")
        traceback.print_exc()


# --- Main Script ---
if __name__ == "__main__":
    print("\n=== Data Study V2: Analysis & Visualization (Fixed) ===")

    # 1. Select Test Files
    all_km_files = sorted([f for f in os.listdir(BASE_KM_DATA_DIR) if f.endswith('.csv')])
    np.random.seed(42) # For reproducibility
    if len(all_km_files) > NUM_TEST_FILES:
        test_files = np.random.choice(all_km_files, NUM_TEST_FILES, replace=False)
    else:
        test_files = all_km_files

    vis_files = np.random.choice(test_files, min(len(test_files), NUM_VISUALIZATION_SAMPLES), replace=False)
    print(f"Selected {len(test_files)} test files.")
    print(f"Selected {len(vis_files)} visualization samples: {', '.join(vis_files)}")

    results_map = {}

    # 2. Processing Loop
    for data_size in DATA_SIZES:
        print(f"\n--- Processing Data Size: {data_size} ---")

        model_info = load_model_for_data_size(data_size, V3_RESULT_DIR, BASE_RUN_ID)
        if model_info is None:
            print(f"Skipping data size {data_size} due to model loading failure.")
            continue

        current_size_results = []
        field_data_for_vis = {} # Store field data for the visualization samples for this size

        for filename in tqdm(test_files, desc=f"Size {data_size}"):
            km_file_path = os.path.join(BASE_KM_DATA_DIR, filename)
            params = parse_params_from_filename(filename)
            if params[0] is None:
                # print(f"Warning: Could not parse parameters from filename {filename}. Skipping.")
                continue

            is_vis_sample = filename in vis_files

            metrics, field_data = process_single_file(km_file_path, model_info, params, return_fields=is_vis_sample)

            if metrics:
                entry = metrics.copy()
                entry['filename'] = filename
                entry['data_size'] = data_size
                entry['nu'], entry['kappa'], entry['d'] = params
                current_size_results.append(entry)

                if is_vis_sample and field_data:
                    field_data['data_size'] = data_size # Add data_size for saving purposes
                    field_data_for_vis[filename] = field_data
                    # Save field_data as npz for debugging if needed, but primary output is CSV
                    # save_path_npz = os.path.join(DIR_FIELDS, f'field_{filename[:-4]}_size_{data_size}.npz')
                    # np.savez(save_path_npz, **field_data)


        if current_size_results:
            df_size = pd.DataFrame(current_size_results)
            save_csv_path = os.path.join(DIR_METRICS, f'metrics_size_{data_size}.csv')
            df_size.to_csv(save_csv_path, index=False)
            results_map[data_size] = df_size
        else:
            print(f"No results generated for data size {data_size}.")


        # Store field_data_for_vis for later plot_field_comparison call
        # We need to store this for min_size and max_size across the loop.
        # This will be handled in step 4 below by reloading npz or directly accessing `field_data_for_vis`.
        # For this to work with the current `save_field_data_and_plot_comparison` signature,
        # we still need to store these per-file-per-size for later retrieval.
        # The existing npz saving for visualization samples already serves this purpose.

        del model_info
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Aggregate Data & Plots
    print("\n=== Generating Aggregated Plots & Saving Metrics CSV ===")
    if results_map:
        all_data = []
        for size, df in results_map.items():
            all_data.append(df)
        full_df = pd.concat(all_data, ignore_index=True)

        # Save aggregated metrics to CSV
        aggregated_metrics_path = os.path.join(DIR_METRICS, 'aggregated_mse_metrics_per_size.csv')
        full_df.to_csv(aggregated_metrics_path, index=False)
        print(f"Aggregated MSE metrics saved to: {aggregated_metrics_path}")


        # Plot Boxplot
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='data_size', y='mse_total', data=full_df, palette="Blues")
        plt.yscale('log')
        plt.title('Distribution of Total MSE vs Training Data Size')
        plt.xlabel('Training Data Size')
        plt.ylabel('Total MSE (Log Scale)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(DIR_PLOTS, 'boxplot_mse_distribution.png'), dpi=200)
        plt.close()

        # Plot Trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='data_size', y='mse_total', data=full_df, marker='o', errorbar=('ci', 95))
        plt.yscale('log')
        plt.title('Mean MSE Trend with 95% Confidence Interval')
        plt.xlabel('Training Data Size')
        plt.ylabel('Total MSE (Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(os.path.join(DIR_PLOTS, 'trend_mse_ci.png'), dpi=200)
        plt.close()

        # Stats
        perform_statistical_tests(results_map, DATA_SIZES)
    else:
        print("No results to plot or aggregate.")

    # 4. Field Visualization & Data Saving
    print("\n=== Generating Field Reconstructions & Saving Field Data CSV ===")
    if not DATA_SIZES:
        print("No data sizes defined for field visualization.")
    else:
        min_size = min(DATA_SIZES)
        max_size = max(DATA_SIZES)

        for vis_file in vis_files:
            try:
                # Load npz files for min and max data sizes for the visualization samples
                path_min_npz = os.path.join(DIR_FIELDS, f'field_{vis_file[:-4]}_size_{min_size}.npz')
                path_max_npz = os.path.join(DIR_FIELDS, f'field_{vis_file[:-4]}_size_{max_size}.npz')

                # Need to ensure these .npz files were actually saved during the processing loop.
                # The original code only saved if `is_vis_sample` was true.
                # Let's ensure they exist.
                if not os.path.exists(path_min_npz) or not os.path.exists(path_max_npz):
                    print(f"Warning: NPZ files for visualization sample {vis_file} at min/max sizes not found. Skipping plot/data save.")
                    continue

                data_min = np.load(path_min_npz, allow_pickle=True)
                data_max = np.load(path_max_npz, allow_pickle=True)

                # Now call the modified function
                save_field_data_and_plot_comparison(data_min, data_max, vis_file, DIR_PLOTS)
                print(f"Plot and field data CSV generated for {vis_file}")
            except Exception as e:
                print(f"Failed to process fields for {vis_file}: {e}")
                traceback.print_exc()

    print(f"\nAnalysis Complete. All outputs in: {BASE_OUTPUT_DIR}")