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
from scipy.interpolate import RegularGridInterpolator

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
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, 'final_external_analysis')
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
NUM_VIS_SAMPLES = 3  # 选3个外部文件画图
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
        except:
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


def manual_scaler_transform(data, mean, std):
    return (torch.tensor(data, dtype=DTYPE, device=DEVICE) - torch.tensor(mean, dtype=DTYPE,
                                                                          device=DEVICE)) / torch.tensor(std,
                                                                                                         dtype=DTYPE,
                                                                                                         device=DEVICE)


# ==========================================
# 5. 核心分析逻辑
# ==========================================

def analyze_study_group(study_name, config_list, context):
    print(f"\n{'=' * 60}")
    print(f"分析研究组: {study_name}")
    print(f"{'=' * 60}")

    study_plot_dir = os.path.join(ANALYSIS_OUTPUT_DIR, 'plots', study_name)
    study_csv_dir = os.path.join(ANALYSIS_OUTPUT_DIR, 'csv_data', study_name)
    os.makedirs(study_plot_dir, exist_ok=True)
    os.makedirs(study_csv_dir, exist_ok=True)

    # 1. 准备测试数据
    all_test_files = glob.glob(os.path.join(TEST_KM_DATA_DIR, "*.csv"))
    random.seed(SEED + 999)  # 确保可视化的三个文件是固定的
    vis_files_path = random.sample(all_test_files, min(len(all_test_files), NUM_VIS_SAMPLES))
    vis_filenames = [os.path.basename(f) for f in vis_files_path]
    print(f"选定可视化样本: {vis_filenames}")

    # 2. 选取对比配置 (Low, Mid, High)
    indices = [0, len(config_list) // 2, len(config_list) - 1]
    selected_configs = [config_list[i] for i in indices]
    print(f"对比配置: {[c['id'] for c in selected_configs]}")

    # 3. 收集灵敏度数据 (Sensitivity Data)
    sensitivity_results = []

    # 4. 对每个配置进行评估
    for config in tqdm(config_list, desc=f"Evaluating {study_name}"):
        model_path = os.path.join(BASE_RESULT_DIR, study_name, config['id'], 'model.pth')
        if not os.path.exists(model_path):
            print(f"警告: 模型未找到 {model_path}")
            continue

        # 加载模型
        model = PODDeepONet(3, config['hidden_units'], config['layers'], POD_MODES,
                            context['pod_basis'], context['y_mean_pod']).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # 准备反归一化参数
        y_mean_t = torch.tensor(context['y_mean_scaler'], device=DEVICE)
        y_std_t = torch.tensor(context['y_std_scaler'], device=DEVICE)

        # --- A. 全量外部测试 (平均误差) ---
        total_d1_err, total_d2_err = 0, 0
        file_count = 0

        # 为了速度，只随机抽50个文件计算平均误差，或者全部
        eval_files = random.sample(all_test_files, min(len(all_test_files), 120))

        for fpath in eval_files:
            try:
                fname = os.path.basename(fpath)
                params_str = fname.replace('(', '').replace(').csv', '').split(',')
                params = np.array([[float(p) for p in params_str]])  # [nu, kappa, d]

                df = pd.read_csv(fpath)

                # Predict
                inp = manual_scaler_transform(params, context['b_mean'], context['b_std'])
                with torch.no_grad():
                    pred_s = model(inp)
                    pred_full = (pred_s * y_std_t + y_mean_t).cpu().numpy().flatten()

                # Reshape & Interp
                ga, gt = context['grid_a'], context['grid_tau']
                mid = len(pred_full) // 2
                d1_field = pred_full[:mid].reshape(len(gt), len(ga))
                d2_field = pred_full[mid:].reshape(len(gt), len(ga))

                interp_d1 = RegularGridInterpolator((gt, ga), d1_field, bounds_error=False, fill_value=None)
                interp_d2 = RegularGridInterpolator((gt, ga), d2_field, bounds_error=False, fill_value=None)

                pts = df[['tau_sec', 'A']].values
                d1_p = interp_d1(pts)
                d2_p = interp_d2(pts)

                mask = ~np.isnan(df['D1_data']) & ~np.isnan(d1_p)
                if np.sum(mask) > 0:
                    total_d1_err += np.mean((df['D1_data'][mask] - d1_p[mask]) ** 2)
                    total_d2_err += np.mean((df['D2_data'][mask] - d2_p[mask]) ** 2)
                    file_count += 1

                # --- B. 保存可视化数据 (仅针对选定的3个配置 和 选定的3个文件) ---
                if (fpath in vis_files_path) and (config in selected_configs):
                    # Save Pred CSV
                    GA, GT = np.meshgrid(ga, gt)
                    save_df = pd.DataFrame({
                        'A': GA.flatten(), 'tau': GT.flatten(),
                        'D1_pred': d1_field.flatten(), 'D2_pred': d2_field.flatten()
                    })
                    csv_name = f"PRED_{fname[:-4]}_{config['id']}.csv"
                    save_df.to_csv(os.path.join(study_csv_dir, csv_name), index=False)

                    # 如果还没存过 True Data，存一份
                    true_csv_path = os.path.join(study_csv_dir, f"TRUE_{fname[:-4]}.csv")
                    if not os.path.exists(true_csv_path):
                        df[['tau_sec', 'A', 'D1_data', 'D2_data']].to_csv(true_csv_path, index=False)

            except Exception as e:
                continue

        if file_count > 0:
            val_val = float(config['id'].split('_')[1])
            sensitivity_results.append({
                'param_value': val_val,
                'd1_mse': total_d1_err / file_count,
                'd2_mse': total_d2_err / file_count
            })

    # 5. 绘制灵敏度曲线 (Sensitivity Plot)
    if sensitivity_results:
        sdf = pd.DataFrame(sensitivity_results).sort_values('param_value')
        sdf.to_csv(os.path.join(study_csv_dir, 'sensitivity_summary.csv'), index=False)

        plt.figure(figsize=(8, 6))
        plt.plot(sdf['param_value'], sdf['d1_mse'], 'o-', label='D1 External MSE')
        plt.plot(sdf['param_value'], sdf['d2_mse'], 's-', label='D2 External MSE')
        plt.plot(sdf['param_value'], sdf['d1_mse'] + sdf['d2_mse'], 'k--', label='Total MSE', alpha=0.5)
        plt.xlabel('Parameter Value')
        plt.ylabel('Mean Squared Error (External Test)')
        plt.title(f'{study_name}: Generalization Sensitivity')
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(study_plot_dir, 'sensitivity_curve.png'), dpi=200)
        plt.close()

    # 6. 绘制场对比图 (针对 3 个可视化文件)
    # 对于每个文件，生成一张图： Row1: True | Low | Mid | High, Row2: Blank | Err | Err | Err

    for fpath in vis_files_path:
        fname = os.path.basename(fpath)[:-4]  # remove .csv

        # 加载真实值
        true_path = os.path.join(study_csv_dir, f"TRUE_{fname}.csv")
        if not os.path.exists(true_path): continue
        df_true = pd.read_csv(true_path)

        # 准备网格插值 True Data
        ga, gt = context['grid_a'], context['grid_tau']
        GA, GT = np.meshgrid(ga, gt)

        # griddata interpolation for plotting True Field
        from scipy.interpolate import griddata
        valid = df_true.dropna(subset=['D1_data'])
        d1_true_grid = griddata((valid['A'], valid['tau_sec']), valid['D1_data'], (GA, GT), method='linear')

        # Setup Plot
        cols = 1 + len(selected_configs)
        fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7), constrained_layout=True)

        vmin, vmax = np.nanmin(d1_true_grid), np.nanmax(d1_true_grid)

        # Plot True
        im0 = axes[0, 0].pcolormesh(GA, GT, d1_true_grid, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Ground Truth")
        axes[1, 0].axis('off')
        fig.colorbar(im0, ax=axes[0, 0], location='bottom', pad=0.1)

        # 预计算最大误差用于 unified error scale
        max_err = 0
        pred_maps = []

        # Load all preds first
        for config in selected_configs:
            p_csv = os.path.join(study_csv_dir, f"PRED_{fname}_{config['id']}.csv")
            if os.path.exists(p_csv):
                pdf = pd.read_csv(p_csv)
                # Reshape
                d1_pred = pdf['D1_pred'].values.reshape(len(gt), len(ga))
                err = np.abs(d1_true_grid - d1_pred)
                max_err = max(max_err, np.nanmax(err))
                pred_maps.append((d1_pred, err))
            else:
                pred_maps.append((None, None))

        # Plot Configs
        for i, config in enumerate(selected_configs):
            col = i + 1
            d1_pred, err = pred_maps[i]

            if d1_pred is not None:
                # Field
                im = axes[0, col].pcolormesh(GA, GT, d1_pred, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, col].set_title(f"{config['id']}")
                fig.colorbar(im, ax=axes[0, col], location='bottom', pad=0.1)

                # Error
                im_e = axes[1, col].pcolormesh(GA, GT, err, shading='auto', cmap='inferno', vmin=0, vmax=max_err)
                axes[1, col].set_title(f"Abs Error")
                fig.colorbar(im_e, ax=axes[1, col], location='bottom', pad=0.1)

        plt.suptitle(f"{study_name} - Sample: {fname} (D1 Field)", fontsize=16)
        plt.savefig(os.path.join(study_plot_dir, f'compare_field_{fname}.png'), dpi=200)
        plt.close()

    # 7. 绘制 Loss History (读取训练时保存的 CSV)
    plt.figure(figsize=(10, 6))
    for config in config_list:
        hist_path = os.path.join(BASE_RESULT_DIR, study_name, config['id'], 'loss_history.csv')
        if os.path.exists(hist_path):
            hdf = pd.read_csv(hist_path)
            # Smooth
            plt.plot(hdf['iter'], hdf['total_loss'].rolling(100).mean(), label=config['id'])

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss (Smoothed)')
    plt.title(f'{study_name}: Training Convergence')
    plt.legend()
    plt.savefig(os.path.join(study_plot_dir, 'loss_comparison.png'), dpi=200)
    plt.close()


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