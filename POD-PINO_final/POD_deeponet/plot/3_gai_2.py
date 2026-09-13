# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.interpolate import RegularGridInterpolator, griddata

# ==========================================
# 1. 全局配置 (请修改为你的实际路径)
# ==========================================

# 结果保存路径 (必须与训练时的输出路径一致，用于读取 model.pth)
BASE_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\full_sensitivity_analysis_v1'

# 原始训练数据路径 (用于重新计算 Scaler 和 POD 基底，保证推理一致性)
TRAIN_DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'

# 外部测试数据路径 (用于新的评估)
TEST_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'

# 分析结果输出路径
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, 'final_external_analysis_1')
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_OUTPUT_DIR, 'csv_data'), exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_OUTPUT_DIR, 'plots'), exist_ok=True)

# 基础参数 (必须与训练时一致)
POD_MODES = 100
DEFAULT_CONFIG = {
    "d2_weight": 0.5,
    "layers": 4,
    "hidden_units": 128
}

# 评估设置
TRAIN_FILE_LIMIT = 2000  # 必须要和训练时用的数量一致，保证 POD 基底一致
NUM_VIS_SAMPLES = 3  # 选3个外部文件画图 (这个现在主要用于决定哪些样本会被绘制成图，而不是保存数据的范围)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# ==========================================
# 2. 实验设计 (必须与训练时定义的一致)
# ==========================================
def get_experiment_configs():
    experiments = {}

    # Study 1: 损失权重
    weight_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    experiments['Study_Weight'] = []
    for w in weight_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['d2_weight'] = w
        cfg['id'] = f"Weight_{w}"
        experiments['Study_Weight'].append(cfg)

    # Study 2: 网络深度
    depth_values = [2, 4, 6, 8]
    experiments['Study_Depth'] = []
    for d in depth_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['layers'] = d
        cfg['id'] = f"Depth_{d}"
        experiments['Study_Depth'].append(cfg)

    # Study 3: 网络宽度
    width_values = [32, 64, 128, 256]
    experiments['Study_Width'] = []
    for u in width_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['hidden_units'] = u
        cfg['id'] = f"Width_{u}"
        experiments['Study_Width'].append(cfg)

    return experiments


# ==========================================
# 3. 模型定义
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE)
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, pod_basis, y_mean_pod_scaled):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, 0.0)
        self.register_buffer('pod_basis', torch.tensor(pod_basis, dtype=DTYPE))
        self.register_buffer('y_mean_pod_scaled', torch.tensor(y_mean_pod_scaled, dtype=DTYPE))

    def forward(self, branch_x):
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


# ==========================================
# 4. 数据恢复与预处理 (Context Restoration)
# ==========================================
def restore_training_context():
    """
    重要：虽然不训练，但我们需要加载训练数据来计算 Mean, Std 和 POD Basis。
    这样才能保证我们加载的模型能正确反归一化数据。
    """
    print(">>> 正在重建训练上下文 (Scalers & POD Basis)...")
    files = glob.glob(os.path.join(TRAIN_DATA_DIR, "data_*.csv"))
    random.seed(SEED)
    selected = random.sample(files, min(len(files), TRAIN_FILE_LIMIT))

    b_list, y_list = [], []
    ua, ut = None, None

    for f in tqdm(selected, desc="Loading Train Data"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip').dropna(subset=['D1_fp', 'D2_fp'])
            if df.empty: continue
            if ua is None: ua, ut = sorted(df['A'].unique()), sorted(df['tau'].unique())
            b_list.append(df[['nu', 'kappa', 'd_diffusion']].iloc[0].values)

            df_sorted = df.sort_values(by=['tau', 'A'])
            y_list.append(np.concatenate([df_sorted['D1_fp'].values, df_sorted['D2_fp'].values]))
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    b_np, y_np = np.array(b_list), np.array(y_list)

    # 计算 Scalers
    b_mean, b_std = b_np.mean(0), b_np.std(0)
    y_mean_scaler, y_std_scaler = y_np.mean(0), y_np.std(0)
    y_std_scaler[y_std_scaler < 1e-10] = 1.0

    # 归一化用于 POD
    y_scaled = (y_np - y_mean_scaler) / y_std_scaler

    # 划分 (为了保持和训练时一致的 POD 基底，必须只用训练集部分计算 POD)
    _, _, y_train_s, _ = train_test_split(b_np, y_scaled, test_size=0.2, random_state=SEED)

    # POD 计算
    print("正在计算 SVD...")
    y_mean_pod = np.mean(y_train_s, axis=0)
    U, S, Vt = np.linalg.svd(y_train_s - y_mean_pod, full_matrices=False)
    pod_basis = Vt.T[:, :POD_MODES]

    context = {
        'b_mean': b_mean, 'b_std': b_std,
        'y_mean_scaler': y_mean_scaler, 'y_std_scaler': y_std_scaler,
        'pod_basis': pod_basis,
        'y_mean_pod': y_mean_pod,
        'grid_a': np.array(ua), 'grid_tau': np.array(ut)
    }
    print("上下文重建完成。")
    return context


def manual_scaler_transform(data, mean, std, device=DEVICE, dtype=DTYPE):
    return (torch.tensor(data, dtype=dtype, device=device) - torch.tensor(mean, dtype=dtype,
                                                                          device=device)) / torch.tensor(std,
                                                                                                         dtype=dtype,
                                                                                                         device=device)


# ==========================================
# 5. 核心分析逻辑
# ==========================================

def analyze_study_group(study_name, config_list, context):
    print(f"\n{'=' * 60}")
    print(f"分析研究组: {study_name}")
    print(f"{'=' * 60}")

    # 结果保存目录
    study_plot_dir = os.path.join(ANALYSIS_OUTPUT_DIR, 'plots', study_name)
    study_csv_dir = os.path.join(ANALYSIS_OUTPUT_DIR, 'csv_data', study_name)
    os.makedirs(study_plot_dir, exist_ok=True)
    os.makedirs(study_csv_dir, exist_ok=True)

    # --- 1. 收集损失函数历史数据 ---
    loss_history_data = []
    for config in tqdm(config_list, desc=f"Collecting Loss History for {study_name}"):
        model_dir = os.path.join(BASE_RESULT_DIR, study_name, config['id'])
        loss_hist_path = os.path.join(model_dir, 'loss_history.csv')
        if os.path.exists(loss_hist_path):
            try:
                hdf = pd.read_csv(loss_hist_path)
                # 添加参数信息到 loss data
                for index, row in hdf.iterrows():
                    loss_entry = row.to_dict()
                    loss_entry['study_id'] = study_name
                    # --- 添加 'id' 列 ---
                    loss_entry['id'] = config['id']  # <-- 添加这一行
                    # 根据 study_name 添加对应的参数列
                    if study_name == 'Study_Weight':
                        loss_entry['weight'] = config['d2_weight']
                    elif study_name == 'Study_Depth':
                        loss_entry['depth'] = config['layers']
                    elif study_name == 'Study_Width':
                        loss_entry['width'] = config['hidden_units']
                    loss_history_data.append(loss_entry)
            except Exception as e:
                print(f"Error reading loss history from {loss_hist_path}: {e}")
        else:
            print(f"Warning: Loss history not found at {loss_hist_path}")

    if loss_history_data:
        loss_df = pd.DataFrame(loss_history_data)
        loss_output_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'csv_data', f'{study_name}_loss_history.csv')
        loss_df.to_csv(loss_output_path, index=False)
        print(f"损失函数历史数据已保存至: {loss_output_path}")

        # 绘制损失函数收敛曲线
        plt.figure(figsize=(10, 6))
        for cfg_id in [c['id'] for c in config_list]:
            cfg_data = loss_df[loss_df['id'] == cfg_id].sort_values('iter')
            if not cfg_data.empty:
                # 尝试平滑曲线，如果数据点足够多
                if len(cfg_data) > 100:
                    plt.plot(cfg_data['iter'], cfg_data['total_loss'].rolling(100).mean(), label=cfg_id)
                else:
                    plt.plot(cfg_data['iter'], cfg_data['total_loss'], label=cfg_id)
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title(f'{study_name}: Training Convergence')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(os.path.join(study_plot_dir, 'loss_comparison.png'), dpi=200)
        plt.close()

    # --- 2. 准备测试数据 ---
    all_test_files = glob.glob(os.path.join(TEST_KM_DATA_DIR, "*.csv"))
    random.seed(SEED + 999)  # 确保可视化的三个文件是固定的，这部分是用于绘图的子集
    vis_files_path_for_plots = random.sample(all_test_files, min(len(all_test_files), NUM_VIS_SAMPLES))
    vis_filenames_for_plots = [os.path.basename(f) for f in vis_files_path_for_plots]
    print(f"选定用于绘图的样本: {vis_filenames_for_plots}")

    # --- 3. 收集 MSE 数据 ---
    all_mse_results = []

    # 预计算网格点用于插值 (用于保存场数据)
    ga, gt = context['grid_a'], context['grid_tau']
    GA, GT = np.meshgrid(ga, gt)
    grid_points = np.array([GA.flatten(), GT.flatten()]).T

    # 存储所有样本的真实场数据，方便后续与其他预测场合并
    all_true_field_data = {}  # {fname: {'true_d1': ..., 'true_d2': ...}}

    print(f"\n>>> 开始处理和评估所有 {len(all_test_files)} 个外部测试文件...")

    # 逐个测试文件处理
    for fpath in tqdm(all_test_files, desc=f"Processing All Test Files for {study_name}"):
        try:
            fname = os.path.basename(fpath)
            params_str = fname.replace('(', '').replace(').csv', '').split(',')
            params = np.array([[float(p) for p in params_str]])  # [nu, kappa, d]

            df = pd.read_csv(fpath)

            # --- 准备真实值场数据 ---
            valid_true_d1 = df.dropna(subset=['D1_data'])
            valid_true_d2 = df.dropna(subset=['D2_data'])

            true_d1_grid_orig, true_d2_grid_orig = None, None
            if len(valid_true_d1) >= 2:
                # Use griddata to interpolate true D1 field to the full grid
                true_d1_grid_orig = griddata((valid_true_d1['A'], valid_true_d1['tau_sec']), valid_true_d1['D1_data'],
                                             grid_points, method='linear').reshape(len(gt), len(ga))
            if len(valid_true_d2) >= 2:
                # Use griddata to interpolate true D2 field to the full grid
                true_d2_grid_orig = griddata((valid_true_d2['A'], valid_true_d2['tau_sec']), valid_true_d2['D2_data'],
                                             grid_points, method='linear').reshape(len(gt), len(ga))

            all_true_field_data[fname] = {
                'true_d1': true_d1_grid_orig,
                'true_d2': true_d2_grid_orig,
                'param_value': params[0],  # Store parameters for later sorting
                'original_test_points': df[['tau_sec', 'A']].values  # Store original points for interpolation
            }

            # --- 预测并计算 MSE ---
            for config in config_list:
                model_path = os.path.join(BASE_RESULT_DIR, study_name, config['id'], 'model.pth')
                if not os.path.exists(model_path):
                    continue  # Skip if model doesn't exist

                # Load Model
                model = PODDeepONet(3, config['hidden_units'], config['layers'], POD_MODES,
                                    context['pod_basis'], context['y_mean_pod']).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()

                y_mean_t = torch.tensor(context['y_mean_scaler'], device=DEVICE)
                y_std_t = torch.tensor(context['y_std_scaler'], device=DEVICE)

                inp = manual_scaler_transform(params, context['b_mean'], context['b_std'])
                with torch.no_grad():
                    pred_s = model(inp)
                    pred_full = (pred_s * y_std_t + y_mean_t).cpu().numpy().flatten()

                mid = len(pred_full) // 2
                d1_pred_field_fullgrid = pred_full[:mid].reshape(len(gt), len(ga))
                d2_pred_field_fullgrid = pred_full[mid:].reshape(len(gt), len(ga))

                # Interpolate predictions to the original test points for MSE calculation
                interp_d1 = RegularGridInterpolator((gt, ga), d1_pred_field_fullgrid, bounds_error=False,
                                                    fill_value=np.nan)
                interp_d2 = RegularGridInterpolator((gt, ga), d2_pred_field_fullgrid, bounds_error=False,
                                                    fill_value=np.nan)

                pts = df[['tau_sec', 'A']].values
                d1_p = interp_d1(pts)
                d2_p = interp_d2(pts)

                # Calculate MSE for this file and config
                mask_d1 = ~np.isnan(df['D1_data']) & ~np.isnan(d1_p)
                mask_d2 = ~np.isnan(df['D2_data']) & ~np.isnan(d2_p)

                mse_d1 = np.mean((df['D1_data'][mask_d1] - d1_p[mask_d1]) ** 2) if np.sum(mask_d1) > 0 else np.nan
                mse_d2 = np.mean((df['D2_data'][mask_d2] - d2_p[mask_d2]) ** 2) if np.sum(mask_d2) > 0 else np.nan

                # Determine the primary parameter value from the filename for MSE summary sorting
                param_val_for_mse = np.nan
                if 'nu=' in fname:
                    param_val_for_mse = float(fname.split('nu=')[1].split(',')[0])
                elif 'kappa=' in fname:
                    param_val_for_mse = float(fname.split('kappa=')[1].split(',')[0])
                elif 'd_diffusion=' in fname:
                    param_val_for_mse = float(fname.split('d_diffusion=')[1].split(',')[0])
                # Add more conditions if your test files have different parameter naming conventions

                all_mse_results.append({
                    'study_id': study_name,
                    'config_id': config['id'],
                    'param_value': param_val_for_mse,  # Parameter value of the test case for this MSE entry
                    'mse_d1': mse_d1,
                    'mse_d2': mse_d2,
                    'total_mse': (mse_d1 + mse_d2) / 2 if not np.isnan(mse_d1) and not np.isnan(mse_d2) else np.nan
                })

                # --- Store field data for saving (all files, not just vis_files_path_for_plots) ---
                # We store predicted fields on the full grid here.
                if fname not in all_true_field_data:  # Should not happen if true data was processed, but as a safeguard
                    print(f"Warning: True field data not found for {fname} when processing predictions.")
                    continue

                all_true_field_data[fname][config['id']] = {
                    'pred_d1': d1_pred_field_fullgrid,
                    'pred_d2': d2_pred_field_fullgrid,
                    'abs_err_d1': np.abs(
                        true_d1_grid_orig - d1_pred_field_fullgrid) if true_d1_grid_orig is not None else np.nan,
                    'abs_err_d2': np.abs(
                        true_d2_grid_orig - d2_pred_field_fullgrid) if true_d2_grid_orig is not None else np.nan
                }

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
            continue

    # --- Save MSE Data ---
    if all_mse_results:
        mse_df = pd.DataFrame(all_mse_results)
        mse_output_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'csv_data', f'{study_name}_mse_summary.csv')
        mse_df.to_csv(mse_output_path, index=False)
        print(f"MSE 汇总数据已保存至: {mse_output_path}")

    # --- 4. 绘制 MSE 灵敏度曲线 ---
    if all_mse_results:
        mse_df = pd.DataFrame(all_mse_results)
        # Aggregate MSE for plotting sensitivity curves
        # Group by test file parameter value AND config parameter value to get a more granular sensitivity
        # However, for a general sensitivity plot per study, we usually average over test files for each config.
        # Let's group by config_id first.
        avg_mse_per_config = mse_df.groupby(['study_id', 'config_id']).agg(
            avg_mse_d1=('mse_d1', lambda x: np.nanmean(x)),
            avg_mse_d2=('mse_d2', lambda x: np.nanmean(x)),
            avg_total_mse=('total_mse', lambda x: np.nanmean(x))
        ).reset_index()

        # Extract the varying parameter value from the config_id for plotting the x-axis
        def get_param_value_from_config_id(row):
            id_parts = row['config_id'].split('_')
            if id_parts[0] == 'Weight': return float(id_parts[1])
            if id_parts[0] == 'Depth': return int(id_parts[1])
            if id_parts[0] == 'Width': return int(id_parts[1])
            return np.nan

        avg_mse_per_config['param_value'] = avg_mse_per_config.apply(get_param_value_from_config_id, axis=1)
        avg_mse_per_config = avg_mse_per_config.dropna(subset=['param_value'])
        avg_mse_per_config = avg_mse_per_config.sort_values('param_value')

        plt.figure(figsize=(8, 6))
        plt.plot(avg_mse_per_config['param_value'], avg_mse_per_config['avg_mse_d1'], 'o-', label='Avg D1 External MSE')
        plt.plot(avg_mse_per_config['param_value'], avg_mse_per_config['avg_mse_d2'], 's-', label='Avg D2 External MSE')
        plt.plot(avg_mse_per_config['param_value'], avg_mse_per_config['avg_total_mse'], 'k--', label='Avg Total MSE',
                 alpha=0.5)
        plt.xlabel(f'{study_name} Parameter Value')
        plt.ylabel('Average Mean Squared Error (External Test)')
        plt.title(f'{study_name}: Generalization Sensitivity')
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(study_plot_dir, 'sensitivity_curve.png'), dpi=200)
        plt.close()

    # --- 5. 绘制场对比图 (针对少量选定样本，用于快速可视化) ---
    # 这个部分只用于快速查看几个代表性样本的效果，不保存所有样本的图
    print(f"\n>>> 正在生成可视化对比图 (针对 {len(vis_files_path_for_plots)} 个选定样本)...")
    if all_true_field_data:  # Check if any data was processed
        # Determine the configurations to visualize (e.g., low, mid, high parameter values)
        # These are the configurations for which we have predictions
        configs_with_predictions = sorted(
            [c for c in config_list if os.path.exists(os.path.join(BASE_RESULT_DIR, study_name, c['id'], 'model.pth'))],
            key=lambda c: float(c['id'].split('_')[-1]) if '_' in c['id'] else float('inf'))  # Robust sorting

        if len(configs_with_predictions) > 3:
            selected_configs_for_plotting = [configs_with_predictions[0],
                                             configs_with_predictions[len(configs_with_predictions) // 2],
                                             configs_with_predictions[-1]]
        else:
            selected_configs_for_plotting = configs_with_predictions

        for fname in vis_files_path_for_plots:  # Iterate only through the sampled files for plotting
            fname = os.path.basename(fname)  # Ensure we are using the base filename
            if fname in all_true_field_data:
                data = all_true_field_data[fname]  # Get all collected data for this sample

                # Check if we have predictions for this file and selected configs
                configs_to_plot_for_this_sample = [c for c in selected_configs_for_plotting if c['id'] in data]

                if not configs_to_plot_for_this_sample: continue  # Skip if no predictions for selected configs

                cols = 1 + len(configs_to_plot_for_this_sample)  # True + selected configs

                # --- Plot D1 ---
                fig_d1, axes_d1 = plt.subplots(2, cols, figsize=(4 * cols, 7), constrained_layout=True)
                fig_d1.suptitle(f"{study_name} - Sample: {fname} (D1 Field Comparison)", fontsize=16)

                vmin_d1, vmax_d1 = None, None
                if data['true_d1'] is not None:
                    vmin_d1, vmax_d1 = np.nanmin(data['true_d1']), np.nanmax(data['true_d1'])

                max_err_d1 = 0.0

                # Plot True D1
                im0 = axes_d1[0, 0].pcolormesh(GA, GT, data['true_d1'], shading='auto', cmap='viridis', vmin=vmin_d1,
                                               vmax=vmax_d1)
                axes_d1[0, 0].set_title("Ground Truth D1")
                axes_d1[1, 0].axis('off')
                fig_d1.colorbar(im0, ax=axes_d1[0, 0], location='bottom', pad=0.1)

                # Plot Configs D1
                current_col_d1 = 1
                for config in configs_to_plot_for_this_sample:
                    config_data = data[config['id']]
                    if config_data['abs_err_d1'] is not np.nan:
                        max_err_d1 = max(max_err_d1, np.nanmax(config_data['abs_err_d1']))

                    # Field
                    im = axes_d1[0, current_col_d1].pcolormesh(GA, GT, config_data['pred_d1'], shading='auto',
                                                               cmap='viridis', vmin=vmin_d1, vmax=vmax_d1)
                    axes_d1[0, current_col_d1].set_title(f"{config['id']}")
                    fig_d1.colorbar(im, ax=axes_d1[0, current_col_d1], location='bottom', pad=0.1)

                    # Error
                    # Handle case where max_err_d1 might be 0 if all errors are 0
                    error_vmax_d1 = max(max_err_d1, 1e-9)
                    im_e = axes_d1[1, current_col_d1].pcolormesh(GA, GT, config_data['abs_err_d1'], shading='auto',
                                                                 cmap='inferno', vmin=0, vmax=error_vmax_d1)
                    axes_d1[1, current_col_d1].set_title(f"Abs Error D1")
                    fig_d1.colorbar(im_e, ax=axes_d1[1, current_col_d1], location='bottom', pad=0.1)

                    current_col_d1 += 1
                plt.savefig(os.path.join(study_plot_dir, f'compare_field_d1_{fname}.png'), dpi=200)
                plt.close(fig_d1)

                # --- Plot D2 ---
                fig_d2, axes_d2 = plt.subplots(2, cols, figsize=(4 * cols, 7), constrained_layout=True)
                fig_d2.suptitle(f"{study_name} - Sample: {fname} (D2 Field Comparison)", fontsize=16)

                vmin_d2, vmax_d2 = None, None
                if data['true_d2'] is not None:
                    vmin_d2, vmax_d2 = np.nanmin(data['true_d2']), np.nanmax(data['true_d2'])

                max_err_d2 = 0.0

                # Plot True D2
                im0 = axes_d2[0, 0].pcolormesh(GA, GT, data['true_d2'], shading='auto', cmap='viridis', vmin=vmin_d2,
                                               vmax=vmax_d2)
                axes_d2[0, 0].set_title("Ground Truth D2")
                axes_d2[1, 0].axis('off')
                fig_d2.colorbar(im0, ax=axes_d2[0, 0], location='bottom', pad=0.1)

                # Plot Configs D2
                current_col_d2 = 1
                for config in configs_to_plot_for_this_sample:
                    config_data = data[config['id']]
                    if config_data['abs_err_d2'] is not np.nan:
                        max_err_d2 = max(max_err_d2, np.nanmax(config_data['abs_err_d2']))

                    # Field
                    im = axes_d2[0, current_col_d2].pcolormesh(GA, GT, config_data['pred_d2'], shading='auto',
                                                               cmap='viridis', vmin=vmin_d2, vmax=vmax_d2)
                    axes_d2[0, current_col_d2].set_title(f"{config['id']}")
                    fig_d2.colorbar(im, ax=axes_d2[0, current_col_d2], location='bottom', pad=0.1)

                    # Error
                    error_vmax_d2 = max(max_err_d2, 1e-9)
                    im_e = axes_d2[1, current_col_d2].pcolormesh(GA, GT, config_data['abs_err_d2'], shading='auto',
                                                                 cmap='inferno', vmin=0, vmax=error_vmax_d2)
                    axes_d2[1, current_col_d2].set_title(f"Abs Error D2")
                    fig_d2.colorbar(im_e, ax=axes_d2[1, current_col_d2], location='bottom', pad=0.1)

                    current_col_d2 += 1
                plt.savefig(os.path.join(study_plot_dir, f'compare_field_d2_{fname}.png'), dpi=200)
                plt.close(fig_d2)

    # --- 6. 保存所有 Field Data for Vis (D1 and D2) ---
    # 这个部分会遍历所有处理过的测试文件，并为每个文件保存所有模型的预测数据
    print(f"\n>>> 正在保存所有 ({len(all_true_field_data)} 个样本) 的场数据...")
    if all_true_field_data:
        for fname, data in all_true_field_data.items():
            # Check if we have any predictions for this file
            # A file has predictions if it contains keys other than the true data and param_value
            configs_present = [key for key in data.keys() if
                               key not in ['true_d1', 'true_d2', 'param_value', 'original_test_points']]

            if not configs_present: continue  # Skip if no predictions were made for this file

            # Prepare D1 data for saving
            d1_save_data = {'A': GA.flatten(), 'tau': GT.flatten()}
            d1_save_data['True_D1'] = data['true_d1'].flatten() if data['true_d1'] is not None else np.full_like(
                GA.flatten(), np.nan)

            for config_id in configs_present:
                config_data = data[config_id]
                d1_save_data[f'{config_id}_Pred_D1'] = config_data['pred_d1'].flatten()
                d1_save_data[f'{config_id}_AbsErr_D1'] = config_data['abs_err_d1'].flatten()

            d1_field_df = pd.DataFrame(d1_save_data)
            d1_field_df.to_csv(os.path.join(study_csv_dir, f'field_data_d1_{fname}.csv'), index=False)

            # Prepare D2 data for saving
            d2_save_data = {'A': GA.flatten(), 'tau': GT.flatten()}
            d2_save_data['True_D2'] = data['true_d2'].flatten() if data['true_d2'] is not None else np.full_like(
                GA.flatten(), np.nan)

            for config_id in configs_present:
                config_data = data[config_id]
                d2_save_data[f'{config_id}_Pred_D2'] = config_data['pred_d2'].flatten()
                d2_save_data[f'{config_id}_AbsErr_D2'] = config_data['abs_err_d2'].flatten()

            d2_field_df = pd.DataFrame(d2_save_data)
            d2_field_df.to_csv(os.path.join(study_csv_dir, f'field_data_d2_{fname}.csv'), index=False)


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 恢复上下文
    context = restore_training_context()

    # 2. 获取实验配置
    all_experiments = get_experiment_configs()

    # 3. 逐个研究组分析
    for study_name, config_list in all_experiments.items():
        analyze_study_group(study_name, config_list, context)

    print(f"\n全部分析完成！结果保存在: {ANALYSIS_OUTPUT_DIR}")