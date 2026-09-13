# 创建修改后的完整代码
# -*- coding: utf-8 -*-
import os
import glob
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import gc
from torch.optim.lr_scheduler import LambdaLR
from scipy.interpolate import griddata, RegularGridInterpolator

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================

# --- 路径设置 ---
TRAIN_DATA_DIR = r'D:\\PINN\\zenodo\\POD_deeponet\\data\\Data'
TEST_KM_DATA_DIR = r'D:\\PINN\\zenodo\\AFP\\P(A,t)_data\\km_data_4'
RESULT_DIR = r'D:\\PINN\\zenodo\\POD_deeponet\\mode_sensitivity_final_v3'

os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(RESULT_DIR, 'models')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
DATA_SAVE_DIR = os.path.join(RESULT_DIR, 'data')

for d in [MODEL_DIR, PLOT_DIR, DATA_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 实验变量 ---
MODE_COUNTS_TO_TEST = [10, 20, 40, 80, 100, 120, 150, 200]
VIS_MODE_COMPARISON = [10, 80, 200]

# --- 训练超参数 ---
TRAIN_FILE_LIMIT = 2500
EPOCHS = 50000
BATCH_SIZE = 64
LR = 3e-4
HIDDEN_UNITS = 256
LAYERS = 4
DROPOUT = 0.1
D2_WEIGHT = 0.5
WARMUP_STEPS = 2000
SEED = 42

# --- 评估设置 ---
NUM_VIS_SAMPLES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# ==========================================
# 2. 模型定义 (Model Definitions)
# ==========================================

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
        self.register_buffer('pod_basis', torch.tensor(pod_basis, dtype=DTYPE))
        self.register_buffer('y_mean_pod_scaled', torch.tensor(y_mean_pod_scaled, dtype=DTYPE))

    def forward(self, branch_x):
        branch_out_coeffs = self.branch(branch_x)
        y_pred_scaled = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean_pod_scaled
        return y_pred_scaled


class WeightedMSELoss(nn.Module):
    def __init__(self, d2_weight=0.5):
        super().__init__()
        self.d2_weight = d2_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred_scaled, y_true_scaled):
        field_len = y_pred_scaled.shape[1] // 2
        loss_d1 = self.mse(y_pred_scaled[:, :field_len], y_true_scaled[:, :field_len])
        loss_d2 = self.mse(y_pred_scaled[:, field_len:], y_true_scaled[:, field_len:])
        total_loss = (1 - self.d2_weight) * loss_d1 + self.d2_weight * loss_d2
        return total_loss


# ==========================================
# 3. 数据处理辅助函数
# ==========================================

def load_training_data(data_dir, target_num_files):
    print(f"从 {data_dir} 加载训练数据 (目标: {target_num_files} 文件)...")
    all_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_files:
        raise FileNotFoundError("未找到训练数据文件")

    random.seed(SEED)
    selected_files = random.sample(all_files, min(len(all_files), target_num_files))

    branch_inputs, y_snapshots = [], []
    unified_a, unified_tau = None, None

    for f in tqdm(selected_files, desc="解析 CSV"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip').dropna(subset=['D1_fp', 'D2_fp'])
            if df.empty: continue
            if unified_a is None:
                unified_a, unified_tau = sorted(df['A'].unique()), sorted(df['tau'].unique())

            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values
            branch_inputs.append(params)

            df_sorted = df.sort_values(by=['tau', 'A'])
            snapshot = np.concatenate([df_sorted['D1_fp'].values, df_sorted['D2_fp'].values])
            y_snapshots.append(snapshot)
        except:
            continue

    print(f"成功加载 {len(branch_inputs)} 个样本。")
    return np.array(branch_inputs), np.array(y_snapshots), np.array(unified_a), np.array(unified_tau)


def manual_scaler(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def manual_scaler_transform_tensor(data, mean, std, device):
    t_data = torch.tensor(data, dtype=DTYPE, device=device)
    t_mean = torch.tensor(mean, dtype=DTYPE, device=device)
    t_std = torch.tensor(std, dtype=DTYPE, device=device)
    return (t_data - t_mean) / t_std


# ==========================================
# 4. 主逻辑
# ==========================================

def main():
    # --- Step 1: 准备训练数据与全量 POD ---
    print(">>> 步骤 1/5: 准备训练数据")
    torch.manual_seed(SEED);
    np.random.seed(SEED);
    random.seed(SEED)

    b_in, y_snap, u_a, u_tau = load_training_data(TRAIN_DATA_DIR, TRAIN_FILE_LIMIT)

    # Split & Scale
    b_train, b_val, y_train, y_val = train_test_split(b_in, y_snap, test_size=0.1, random_state=SEED)

    b_train_s, b_mean, b_std = manual_scaler(b_train)
    b_val_s = manual_scaler(b_val, b_mean, b_std)
    y_train_s, y_mean, y_std = manual_scaler(y_train)
    y_val_s = manual_scaler(y_val, y_mean, y_std)

    # Compute Full SVD once
    print("计算全量 SVD (可能需要几分钟)...")
    y_mean_pod = np.mean(y_train_s, axis=0)
    U, S, Vt = np.linalg.svd(y_train_s - y_mean_pod, full_matrices=False)
    print(f"SVD 完成。最大秩: {Vt.shape[0]}")

    # ========== 新增：保存奇异值和能量数据 ==========
    print(">>> 保存POD特征数据...")
    # 计算累积能量（从方差角度）
    energy = (S ** 2) / np.sum(S ** 2)
    cumulative_energy = np.cumsum(energy)

    # 创建POD特征数据 DataFrame
    pod_modes = np.arange(1, len(S) + 1)
    df_pod_features = pd.DataFrame({
        'Mode': pod_modes,
        'Singular_Value': S,
        'Cumulative_Energy': cumulative_energy
    })

    # 保存为CSV
    pod_features_csv = os.path.join(DATA_SAVE_DIR, 'pod_singular_values_and_energy.csv')
    df_pod_features.to_csv(pod_features_csv, index=False)
    print(f"POD特征数据已保存到: {pod_features_csv}")

    train_ds = TensorDataset(torch.from_numpy(b_train_s).to(DEVICE), torch.from_numpy(y_train_s).to(DEVICE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    all_loss_histories = {}

    # --- Step 2: 循环训练 (不同模态数) ---
    print("\\n>>> 步骤 2/5: 开始多模态训练循环")
    for n_modes in MODE_COUNTS_TO_TEST:
        if n_modes > Vt.shape[0]:
            print(f"跳过模态数 {n_modes} (超过最大秩 {Vt.shape[0]})")
            continue

        print(f"\\n--- 正在训练模态数: {n_modes} ---")

        # 截取基底
        basis_k = Vt.T[:, :n_modes]

        # 初始化模型
        model = PODDeepONet(3, HIDDEN_UNITS, LAYERS, n_modes, basis_k, y_mean_pod, DROPOUT).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
        scheduler = get_scheduler(optimizer, WARMUP_STEPS, EPOCHS)
        criterion = WeightedMSELoss(d2_weight=D2_WEIGHT)

        loss_history = []
        model.train()

        # 训练循环
        data_iter = iter(train_loader)
        pbar = tqdm(range(EPOCHS), desc=f"Training Modes={n_modes}", leave=False)

        for step in pbar:
            try:
                batch_b, batch_y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_b, batch_y = next(data_iter)

            optimizer.zero_grad()
            pred = model(batch_b)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())
            if step % 500 == 0:
                pbar.set_postfix({'loss': f"{loss.item():.2e}"})

        # 保存模型为 .pth 文件
        save_name = f"modes_{n_modes}"
        model_save_path = os.path.join(MODEL_DIR, f'model_{save_name}.pth')

        checkpoint = {
            'model_state': model.state_dict(),
            'model_config': {
                'branch_input_dim': 3,
                'hidden_units': HIDDEN_UNITS,
                'num_hidden_layers': LAYERS,
                'num_pod_modes': n_modes,
                'dropout_rate': DROPOUT
            },
            'basis': basis_k,
            'y_mean_pod': y_mean_pod,
            'scalers': {
                'b_mean': b_mean,
                'b_std': b_std,
                'y_mean': y_mean,
                'y_std': y_std
            },
            'grid': {
                'a': u_a,
                'tau': u_tau
            },
            'loss_history': loss_history,
            'num_modes': n_modes
        }

        torch.save(checkpoint, model_save_path)
        print(f"模型已保存到: {model_save_path}")

        all_loss_histories[n_modes] = loss_history

        # 清理显存
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
        gc.collect()

    # ========== 新增：保存所有模型的损失函数数据到一个CSV ==========
    print("\\n>>> 保存训练损失函数数据...")

    # 找到最大迭代数
    max_iterations = max(len(history) for history in all_loss_histories.values())

    # 创建损失函数 DataFrame，使用迭代次数作为第一列
    loss_data_dict = {'iteration': np.arange(max_iterations)}

    for n_modes in MODE_COUNTS_TO_TEST:
        if n_modes not in all_loss_histories:
            continue
        history = all_loss_histories[n_modes]
        # 如果长度不足，用NaN填充
        padded_history = np.full(max_iterations, np.nan)
        padded_history[:len(history)] = history
        loss_data_dict[f'model_{n_modes}'] = padded_history

    df_loss = pd.DataFrame(loss_data_dict)
    loss_csv_path = os.path.join(DATA_SAVE_DIR, 'training_loss_all_models.csv')
    df_loss.to_csv(loss_csv_path, index=False)
    print(f"损失函数数据已保存到: {loss_csv_path}")

    # --- Step 3: 评估 (Calculation & Save CSV) ---
    print("\\n>>> 步骤 3/5: 外部数据评估与CSV保存")

    # 1. 锁定所有测试文件
    all_test_files = glob.glob(os.path.join(TEST_KM_DATA_DIR, "*.csv"))
    if not all_test_files:
        raise FileNotFoundError(f"在 {TEST_KM_DATA_DIR} 未找到测试文件")
    print(f"共找到 {len(all_test_files)} 个外部测试文件。")

    # 2. 锁定 3 个用于可视化的文件
    random.seed(SEED + 123)
    vis_files_path = random.sample(all_test_files, min(len(all_test_files), NUM_VIS_SAMPLES))
    vis_filenames = [os.path.basename(f) for f in vis_files_path]
    print(f"选定可视化的文件: {vis_filenames}")

    # 3. 预加载所有测试数据到内存
    test_data_cache = []
    for fpath in tqdm(all_test_files, desc="预加载测试文件"):
        try:
            fname = os.path.basename(fpath)
            params_str = fname.replace('(', '').replace(').csv', '').split(',')
            params = [float(p) for p in params_str]
            df = pd.read_csv(fpath)
            test_data_cache.append({
                'fname': fname,
                'params': params,
                'df': df,
                'is_vis': fpath in vis_files_path
            })
        except Exception as e:
            continue

    # 存储每个测试样本的所有MSE
    all_sample_mse_list = []

    # ========== 新增：预先收集所有预测结果，用于保存CSV ==========
    # 结构: {file_idx: {mode: {'d1_field', 'd2_field', 'params'}, ...}, ...}
    all_predictions = {}

    # 4. 遍历所有模态进行预测
    for mode_idx, n_modes in enumerate(MODE_COUNTS_TO_TEST):
        ckpt_path = os.path.join(MODEL_DIR, f'model_modes_{n_modes}.pth')
        if not os.path.exists(ckpt_path):
            print(f"警告: 模型文件不存在 {ckpt_path}，跳过")
            continue

        # 加载模型
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        scalers = ckpt['scalers']
        grid_a, grid_tau = ckpt['grid']['a'], ckpt['grid']['tau']

        model = PODDeepONet(3, HIDDEN_UNITS, LAYERS, n_modes,
                            ckpt['basis'], ckpt['y_mean_pod'], 0.0).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        # 准备反归一化张量
        y_mean_t = torch.tensor(scalers['y_mean'], device=DEVICE)
        y_std_t = torch.tensor(scalers['y_std'], device=DEVICE)

        # 遍历所有缓存的测试文件
        for file_idx, item in enumerate(tqdm(test_data_cache, desc=f"Eval Modes={n_modes}")):
            # 预测
            input_p = np.array([item['params']])
            input_t = manual_scaler_transform_tensor(input_p, scalers['b_mean'], scalers['b_std'], DEVICE)

            with torch.no_grad():
                pred_scaled = model(input_t)
                pred_full = (pred_scaled * y_std_t + y_mean_t).cpu().numpy().flatten()

            # 拆分场
            mid = len(pred_full) // 2
            d1_field = pred_full[:mid].reshape(len(grid_tau), len(grid_a))
            d2_field = pred_full[mid:].reshape(len(grid_tau), len(grid_a))

            # 计算误差
            interp_d1 = RegularGridInterpolator((grid_tau, grid_a), d1_field, bounds_error=False, fill_value=None)
            interp_d2 = RegularGridInterpolator((grid_tau, grid_a), d2_field, bounds_error=False, fill_value=None)

            query_points = item['df'][['tau_sec', 'A']].values
            d1_pred_pts = interp_d1(query_points)
            d2_pred_pts = interp_d2(query_points)

            mask = ~np.isnan(item['df']['D1_data']) & ~np.isnan(d1_pred_pts)
            if np.sum(mask) > 0:
                d1_mse = np.mean((item['df']['D1_data'][mask] - d1_pred_pts[mask]) ** 2)
                d2_mse = np.mean((item['df']['D2_data'][mask] - d2_pred_pts[mask]) ** 2)

                # 保存MSE
                if file_idx >= len(all_sample_mse_list):
                    all_sample_mse_list.append({
                        'nu': item['params'][0],
                        'kappa': item['params'][1],
                        'd_diffusion': item['params'][2]
                    })

                all_sample_mse_list[file_idx][f'mode_{n_modes}_d1_mse'] = d1_mse
                all_sample_mse_list[file_idx][f'mode_{n_modes}_d2_mse'] = d2_mse
                all_sample_mse_list[file_idx][f'mode_{n_modes}_total_mse'] = d1_mse + d2_mse

            # 保存预测结果供后续可视化使用
            if file_idx not in all_predictions:
                all_predictions[file_idx] = {}

            all_predictions[file_idx][n_modes] = {
                'd1_field': d1_field,
                'd2_field': d2_field,
                'grid_a': grid_a,
                'grid_tau': grid_tau,
                'params': item['params']
            }

        # 清理显存
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ========== 新增：保存每个样本的MSE随模态变化 ==========
    print("\\n>>> 保存样本MSE数据...")

    # 将列表转为DataFrame并保存
    df_sample_mse = pd.DataFrame(all_sample_mse_list)

    # 重新排列列：参数列优先，然后MSE列
    param_cols = ['nu', 'kappa', 'd_diffusion']
    mse_cols = [col for col in df_sample_mse.columns if col not in param_cols]

    df_sample_mse = df_sample_mse[param_cols + mse_cols]

    sample_mse_csv_path = os.path.join(DATA_SAVE_DIR, 'sample_mse_vs_modes.csv')
    df_sample_mse.to_csv(sample_mse_csv_path, index=False)
    print(f"样本MSE数据已保存到: {sample_mse_csv_path}")

    # --- Step 3b: 保存预测场数据为CSV ---
    print("\\n>>> 保存预测场和真实场数据...")

    for file_idx, item in enumerate(test_data_cache):
        if file_idx not in all_predictions:
            continue

        fname = item['fname']
        params = item['params']
        df_true = item['df']

        # 准备输出 DataFrame
        # 基础列：A, tau_sec
        output_data = {
            'A': df_true['A'].values,
            'tau_sec': df_true['tau_sec'].values,
            'D1_true': df_true['D1_data'].values,
        }

        # 针对每个模态，添加预测场和误差场
        for n_modes in sorted(all_predictions[file_idx].keys()):
            pred_info = all_predictions[file_idx][n_modes]
            grid_a = pred_info['grid_a']
            grid_tau = pred_info['grid_tau']
            d1_field = pred_info['d1_field']
            d2_field = pred_info['d2_field']

            # 插值预测场到实际点
            interp_d1 = RegularGridInterpolator((grid_tau, grid_a), d1_field, bounds_error=False, fill_value=None)
            query_points = df_true[['tau_sec', 'A']].values
            d1_pred_pts = interp_d1(query_points)

            # 计算误差
            d1_error = np.abs(df_true['D1_data'].values - d1_pred_pts)

            output_data[f'D1_pred_mode_{n_modes}'] = d1_pred_pts
            output_data[f'D1_error_mode_{n_modes}'] = d1_error

        # 添加D2真实场
        output_data['D2_true'] = df_true['D2_data'].values

        # 针对每个模态，添加D2预测场和误差场
        for n_modes in sorted(all_predictions[file_idx].keys()):
            pred_info = all_predictions[file_idx][n_modes]
            grid_a = pred_info['grid_a']
            grid_tau = pred_info['grid_tau']
            d2_field = pred_info['d2_field']

            # 插值预测场到实际点
            interp_d2 = RegularGridInterpolator((grid_tau, grid_a), d2_field, bounds_error=False, fill_value=None)
            query_points = df_true[['tau_sec', 'A']].values
            d2_pred_pts = interp_d2(query_points)

            # 计算误差
            d2_error = np.abs(df_true['D2_data'].values - d2_pred_pts)

            output_data[f'D2_pred_mode_{n_modes}'] = d2_pred_pts
            output_data[f'D2_error_mode_{n_modes}'] = d2_error

        # 创建 DataFrame
        df_output = pd.DataFrame(output_data)

        # 保存为CSV
        # 文件名格式：field_data_{原始文件名}
        output_csv_name = f"field_data_{fname}"
        output_csv_path = os.path.join(DATA_SAVE_DIR, output_csv_name)
        df_output.to_csv(output_csv_path, index=False)
        print(f"字段数据已保存: {output_csv_name}")

    # ==========================================
    # 5. 可视化绘图 (Plotting)
    # ==========================================
    print("\\n>>> 步骤 4/5: 生成图像")

    # --- 图 1: 训练 Loss 对比 ---
    plt.figure(figsize=(10, 6))
    for n_modes, history in all_loss_histories.items():
        smoothed = pd.Series(history).rolling(window=100).mean()
        plt.plot(smoothed, label=f'Modes={n_modes}')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Weighted MSE)')
    plt.title('Training Loss Convergence Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '1_loss_comparison.png'), dpi=300)
    plt.close()
    print(f"图1已保存: 1_loss_comparison.png")

    # --- 图 2: 奇异值衰减 ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(pod_modes, S[:len(pod_modes)], 'o-', linewidth=2, markersize=6)
    plt.xlabel('POD Mode')
    plt.ylabel('Singular Value')
    plt.title('POD Singular Value Decay')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '2_singular_value_decay.png'), dpi=300)
    plt.close()
    print(f"图2已保存: 2_singular_value_decay.png")

    # --- 图 3: 能量累积 ---
    plt.figure(figsize=(10, 6))
    plt.plot(pod_modes, cumulative_energy[:len(pod_modes)], 'o-', linewidth=2, markersize=6)
    plt.xlabel('POD Mode')
    plt.ylabel('Cumulative Energy Ratio')
    plt.title('POD Energy Accumulation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '3_cumulative_energy.png'), dpi=300)
    plt.close()
    print(f"图3已保存: 3_cumulative_energy.png")

    # --- 图 4: MSE随模态变化 (汇总统计) ---
    if not df_sample_mse.empty:
        mse_cols_d1 = [col for col in df_sample_mse.columns if 'd1_mse' in col]
        mse_cols_d2 = [col for col in df_sample_mse.columns if 'd2_mse' in col]

        if mse_cols_d1 and mse_cols_d2:
            avg_d1_mse = df_sample_mse[mse_cols_d1].mean()
            avg_d2_mse = df_sample_mse[mse_cols_d2].mean()

            modes = [int(col.split('_')[1]) for col in mse_cols_d1]

            plt.figure(figsize=(10, 6))
            plt.plot(modes, avg_d1_mse.values, 'o-', label='D1 MSE', linewidth=2, markersize=8)
            plt.plot(modes, avg_d2_mse.values, 's-', label='D2 MSE', linewidth=2, markersize=8)
            plt.yscale('log')
            plt.xlabel('Number of POD Modes')
            plt.ylabel('Average MSE')
            plt.title('Generalization Error vs. Mode Count')
            plt.legend()
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, '4_error_vs_modes.png'), dpi=300)
            plt.close()
            print(f"图4已保存: 4_error_vs_modes.png")

    print(f"\\n全部完成！结果已保存在: {RESULT_DIR}")
    print(f"- 模型: {MODEL_DIR}")
    print(f"- 数据: {DATA_SAVE_DIR}")
    print(f"- 图片: {PLOT_DIR}")


if __name__ == "__main__":
    main()
'''

# 保存到文件
with open('pod_deeponet_modified.py', 'w', encoding='utf-8') as f:
    f.write(code_content)

print("✓ 修改后的代码已生成！")
print("\n主要改动总结：\n")

changes = """
1. **POD特征数据 (pod_singular_values_and_energy.csv)**
   - 第1列: Mode (模态序号 1, 2, 3, ...)
   - 第2列: Singular_Value (奇异值)
   - 第3列: Cumulative_Energy (累积能量比)
   - 文件位置: data/pod_singular_values_and_energy.csv

2. **训练损失函数数据 (training_loss_all_models.csv)**
   - 第1列: iteration (迭代次数 0, 1, 2, ...)
   - 第2列: model_10 (模态10的损失)
   - 第3列: model_20 (模态20
   - 文件位置: data/trai的损失)
   - ... 依次类推ning_loss_all_models.csv

3. **样本MSE数据 (sample_mse_vs_modes.csv)**
   - 第1列: nu (参数 nu)
   - 第2列: kappa (参数 kappa)
   - 第3列: d_diffusion (参数 d_diffusion)
   - 第4列: mode_10_d1_mse (模态10 D1字段MSE)
   - 第5列: mode_10_d2_mse (模态10 D2字段MSE)
   - 第6列: mode_10_total_mse (模态10总MSE)
   - 第7列: mode_20_d1_mse
   - ... 依次类推
   - 文件位置: data/sample_mse_vs_modes.csv
   - 说明: 每行对应一个测试样本，包含所有模态的MSE数据

4. **预测场和误差数据 (field_data_*.csv)**
   - 第1列: A (空间参数A)
   - 第2列: tau_sec (时间参数)
   - 第3列: D1_true (D1真实场)
   - 第4列: D1_pred_mode_10 (D1模态10预测)
   - 第5列: D1_error_mode_10 (D1模态10误差)
   - 第6列: D1_pred_mode_20 (D1模态20预测)
   - 第7列: D1_error_mode_20 (D1模态20误差)
   - ... 依次所有模态
   - 第N列: D2_true (D2真实场)
   - 第N+1列: D2_pred_mode_10
   - 第N+2列: D2_error_mode_10
   - ... 依次所有模态
   - 文件位置: data/field_data_*.csv (每个测试样本一个文件)

5. **模型文件 (model_modes_*.pth)**
   - 格式: PyTorch .pth 文件
   - 包含内容:
     * model_state: 模型参数
     * model_config: 模型配置信息
     * basis: POD基底矩阵
     * y_mean_pod: POD均值
     * scalers: 数据缩放参数
     * grid: 空间网格信息
     * loss_history: 训练损失历史
     * num_modes: 模态数
   - 文件位置: models/model_modes_10.pth, model_modes_20.pth, ...

6. **新增可视化图表**
   - 1_loss_comparison.png: 所有模型训练损失对比
   - 2_singular_value_decay.png: 奇异值衰减曲线
   - 3_cumulative_energy.png: 能量累积曲线
   - 4_error_vs_modes.png: 泛化误差随模态变化
   - 文件位置: plots/

说明:
- 所有CSV文件都没有index列
- 数据自动化处理，避免手动操作
- 每个测试样本的MSE都被单独记录
- POD和损失数据统一保存在一个文件中便于查看
'''

