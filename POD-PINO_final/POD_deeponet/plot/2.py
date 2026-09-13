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
from tqdm import tqdm
import random
import traceback
from torch.optim.lr_scheduler import LambdaLR
from scipy.optimize import curve_fit

# --- Configuration ---
# Directories
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\train_result_v3_analysis'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'
os.makedirs(RESULT_DIR, exist_ok=True)

# Run ID
RUN_ID = "pod_mode_sensitivity_analysis"
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, f'model_{RUN_ID}.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, f'scalers_{RUN_ID}.pth')
POD_PARAMS_SAVE_PATH = os.path.join(RESULT_DIR, f'pod_params_{RUN_ID}.pth')

# ### 分析设置：我们将在不训练神经网络的情况下，测试这些模态数的物理还原能力 ###
# 测试列表：我们将计算如果保留 k 个模态，理论上误差是多少
TEST_MODE_COUNTS = [5, 10, 20, 40, 60, 80, 100, 128, 150, 200]
# 可视化列表：我们将为这些模态数绘制“场”的对比图
VISUALIZE_MODE_COUNTS = [10, 50, 100, 150]

# 保存路径
ANALYSIS_SAVE_CSV = os.path.join(RESULT_DIR, f'mode_error_data_{RUN_ID}.csv')
ANALYSIS_PLOT_ERROR = os.path.join(RESULT_DIR, f'mode_error_curve_{RUN_ID}.png')
ANALYSIS_PLOT_FIELDS = os.path.join(RESULT_DIR, f'mode_reconstruction_fields_{RUN_ID}.png')

# Model Hyperparameters (实际训练用的参数)
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 256
NUM_HIDDEN_LAYERS = 4
REQUESTED_NUM_POD_MODES = 100  # 最终选定用于训练的模态数
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6

# Training Hyperparameters
ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 64
ADAM_ITERATIONS = 50000
D2_LOSS_WEIGHT = 0.8
VALIDATION_SPLIT = 0.2
SEED = 42
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 100
WARMUP_STEPS = 2000

# Data Config
TARGET_NUM_FILES = 2000

# Computation Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# --- Helper Functions ---
def load_unified_data(data_dir, target_num_files):
    print(f"从 {data_dir} 加载统一数据...");
    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_available_files: raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件。")

    all_available_files.sort()
    random.seed(SEED)
    selected_files = random.sample(all_available_files, min(len(all_available_files), target_num_files))

    print(f"从 {len(all_available_files)} 个文件中选择了 {len(selected_files)} 个文件进行加载。")
    branch_inputs_list, y_snapshots_list = [], []

    # --- 修复部分开始 ---
    # 这里只有两个变量需要初始化，之前多写了一个 None
    unified_a_grid, unified_tau_grid = None, None
    # --- 修复部分结束 ---

    for f in tqdm(selected_files, desc="加载快照"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip').dropna(subset=['D1_fp', 'D2_fp'])
            if df.empty: continue
            if unified_a_grid is None:
                unified_a_grid, unified_tau_grid = sorted(df['A'].unique()), sorted(df['tau'].unique())
            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values
            branch_inputs_list.append(params)
            df_sorted = df.sort_values(by=['tau', 'A'])
            final_snapshot = np.concatenate([df_sorted['D1_fp'].values, df_sorted['D2_fp'].values])
            y_snapshots_list.append(final_snapshot)
        except Exception as e:
            continue

    branch_inputs_np, y_snapshots_np = np.array(branch_inputs_list), np.array(y_snapshots_list)
    return branch_inputs_np, y_snapshots_np, np.array(unified_a_grid), np.array(unified_tau_grid)


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def compute_pod_matrices(y_data_scaled):
    """
    计算全量 SVD (POD)。这是所有分析的基础。
    """
    print("计算全量 SVD (POD)...")
    y_mean = np.mean(y_data_scaled, axis=0)
    # Full SVD allows us to test any number of modes later
    U, S, Vt = np.linalg.svd(y_data_scaled - y_mean, full_matrices=False)
    return y_mean, U, S, Vt


# --- 核心：模态敏感性分析模块 ---
def fitting_func_power_law(x, a, b, c):
    """拟合函数：幂律衰减 y = a * x^(-b) + c"""
    return a * np.power(x, -b) + c


def analyze_mode_sensitivity(y_data_scaled, y_mean, Vt, S,
                             unified_a_grid, unified_tau_grid,
                             test_counts, viz_counts):
    print("\n" + "=" * 60)
    print("  开始模态敏感性分析 (Theoretical Reconstruction Analysis)")
    print("  注意：此步骤不使用神经网络，而是使用数学投影计算理论最佳重构。")
    print("=" * 60)

    # 1. 计算不同模态数下的误差
    results = []
    total_variance = np.sum(S ** 2)
    max_rank = Vt.shape[0]
    valid_test_counts = [k for k in test_counts if k <= max_rank]

    # 为了速度，我们只用验证集的前500个样本进行评估
    num_eval = min(len(y_data_scaled), 500)
    y_sample = y_data_scaled[:num_eval]

    for k in tqdm(valid_test_counts, desc="评估模态数对误差的影响"):
        # --- 关键步骤：数学投影 ---
        # 我们截取前 k 个基向量
        basis_k = Vt.T[:, :k]  # Shape: (Feature_Dim, k)

        # 1. 投影 (Projection)：计算如果拥有完美网络，系数应该是多少
        # Coefficients = (X - Mean) @ Basis
        coeffs_theoretical = np.dot(y_sample - y_mean, basis_k)

        # 2. 重构 (Reconstruction)：用这些系数还原场
        # X_recon = Coefficients @ Basis.T + Mean
        y_recon = np.dot(coeffs_theoretical, basis_k.T) + y_mean

        # 3. 计算误差
        mse = np.mean((y_sample - y_recon) ** 2)
        mae = np.mean(np.abs(y_sample - y_recon))

        # 相对 L2 误差
        norm_true = np.linalg.norm(y_sample, axis=1)
        norm_diff = np.linalg.norm(y_sample - y_recon, axis=1)
        rel_l2 = np.mean(norm_diff / (norm_true + 1e-10))

        # 能量占比
        energy_ratio = np.sum(S[:k] ** 2) / total_variance

        results.append({
            'num_modes': k,
            'mse': mse,
            'mae': mae,
            'rel_l2': rel_l2,
            'energy_captured': energy_ratio
        })

    # 保存数据
    df_res = pd.DataFrame(results)
    df_res.to_csv(ANALYSIS_SAVE_CSV, index=False)
    print(f"分析数据已保存至: {ANALYSIS_SAVE_CSV}")

    # 2. 绘制误差曲线与拟合
    plt.figure(figsize=(10, 6))
    x_data = df_res['num_modes'].values
    y_data = df_res['mse'].values

    # 绘制实际点
    plt.scatter(x_data, y_data, color='red', label='Actual MSE (Projection)', zorder=5)
    plt.plot(x_data, y_data, 'r--', alpha=0.3)

    # 曲线拟合 (Power Law)
    try:
        # 初始猜测 [scale, decay_rate, offset]
        p0 = [np.max(y_data), 0.5, 0]
        popt, _ = curve_fit(fitting_func_power_law, x_data, y_data, p0=p0, maxfev=5000)

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = fitting_func_power_law(x_fit, *popt)

        fit_eq = f'Fit: $y = {popt[0]:.1e} \cdot x^{{-{popt[1]:.2f}}} + {popt[2]:.1e}$'
        plt.plot(x_fit, y_fit, 'b-', linewidth=2, label=fit_eq)
        print(f"误差曲线拟合参数: a={popt[0]:.2e}, b={popt[1]:.2f}, c={popt[2]:.2e}")
    except Exception as e:
        print(f"曲线拟合失败: {e}")

    plt.title('Theoretical Reconstruction Error vs. Number of Modes')
    plt.xlabel('Number of POD Modes')
    plt.ylabel('Mean Squared Error (Scaled Space)')
    plt.yscale('log')  # 对数坐标轴更清晰
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_ERROR, dpi=300)
    plt.close()

    # 3. 绘制场重构可视化 (Heatmaps)
    # 随机选一个样本
    sample_idx = random.randint(0, len(y_data_scaled) - 1)
    y_true_sample = y_data_scaled[sample_idx]

    valid_viz_counts = [k for k in viz_counts if k <= max_rank]
    num_cols = len(valid_viz_counts) + 1  # +1 for Ground Truth

    # 设置画布
    fig, axes = plt.subplots(2, num_cols, figsize=(3.5 * num_cols, 7))
    num_a, num_tau = len(unified_a_grid), len(unified_tau_grid)
    field_len = num_a * num_tau

    def plot_subplot(ax, data, title, vmin, vmax):
        im = ax.imshow(data.reshape(num_tau, num_a), aspect='auto', origin='lower',
                       extent=[unified_a_grid.min(), unified_a_grid.max(),
                               unified_tau_grid.min(), unified_tau_grid.max()],
                       vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # 准备真值 (Ground Truth)
    d1_true = y_true_sample[:field_len]
    d2_true = y_true_sample[field_len:]
    vmin_d1, vmax_d1 = d1_true.min(), d1_true.max()
    vmin_d2, vmax_d2 = d2_true.min(), d2_true.max()

    # 绘制真值列
    plot_subplot(axes[0, 0], d1_true, "Ground Truth (D1)", vmin_d1, vmax_d1)
    plot_subplot(axes[1, 0], d2_true, "Ground Truth (D2)", vmin_d2, vmax_d2)
    axes[0, 0].set_ylabel('tau')
    axes[1, 0].set_ylabel('tau')

    # 绘制不同模态数的重构列
    for i, k in enumerate(valid_viz_counts):
        col = i + 1
        # --- 再次执行投影重构 ---
        basis_k = Vt.T[:, :k]
        # 这一步就是“假设我们有完美的网络预测出了正确的系数”
        coeff = np.dot(y_true_sample - y_mean, basis_k)
        recon = np.dot(coeff, basis_k.T) + y_mean

        d1_recon = recon[:field_len]
        d2_recon = recon[field_len:]

        # 计算该样本在该模态数下的特定误差
        mse_sample = np.mean((y_true_sample - recon) ** 2)

        plot_subplot(axes[0, col], d1_recon, f'Modes={k}\nMSE={mse_sample:.1e}', vmin_d1, vmax_d1)
        im = plot_subplot(axes[1, col], d2_recon, f'Modes={k}', vmin_d2, vmax_d2)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Scaled Value')

    plt.suptitle(f'Field Reconstruction Quality (Sample {sample_idx}) \n Theoretical Limit via Projection', fontsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
    plt.savefig(ANALYSIS_PLOT_FIELDS, dpi=300)
    plt.close()
    print(f"可视化图像已保存: \n1. 误差曲线: {ANALYSIS_PLOT_ERROR} \n2. 场重构图: {ANALYSIS_PLOT_FIELDS}")


# --- DeepONet Definitions (用于后续训练) ---
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
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, pod_basis, y_mean_pod_scaled,
                 dropout_rate):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, dropout_rate)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod_scaled, dtype=DTYPE), requires_grad=False)

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
        return total_loss, loss_d1.detach(), loss_d2.detach()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# --- Main Execution ---
if __name__ == "__main__":
    # 0. Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 1. Load Data
    branch_inputs_np, y_snapshots_np, unified_a_grid, unified_tau_grid = load_unified_data(DATA_DIR, TARGET_NUM_FILES)

    # 2. Split Data
    branch_train_np, branch_val_np, y_train_np, y_val_np = train_test_split(
        branch_inputs_np, y_snapshots_np, test_size=VALIDATION_SPLIT, random_state=SEED
    )

    # 3. Scale Data
    branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
    branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)
    y_train_scaled, y_mean_scaler, y_std_scaler = manual_scaler(y_train_np)
    y_val_scaled = manual_scaler(y_val_np, y_mean_scaler, y_std_scaler)

    # 4. Compute Full POD Matrices
    # 注意：我们使用训练集计算基底，然后在验证集上进行测试
    y_mean_pod_scaled, U, S, Vt = compute_pod_matrices(y_train_scaled)

    # 5. === 执行模态敏感性分析 (Analysis) ===
    # 这一步不训练网络，而是通过数学方法验证不同模态数对场恢复的影响
    analyze_mode_sensitivity(
        y_val_scaled,  # 使用未见过的验证集数据
        y_mean_pod_scaled,
        Vt,
        S,
        unified_a_grid,
        unified_tau_grid,
        TEST_MODE_COUNTS,
        VISUALIZE_MODE_COUNTS
    )

    # 6. === 执行实际训练 (Training) ===
    print(f"\n" + "=" * 60)
    print(f"  开始 DeepONet 训练 (选定模态数: {REQUESTED_NUM_POD_MODES})")
    print("=" * 60)

    # 截断基底
    actual_num_modes = min(REQUESTED_NUM_POD_MODES, Vt.shape[0])
    pod_basis_train = Vt.T[:, :actual_num_modes]

    # 保存参数
    torch.save({'branch_mean': branch_mean, 'branch_std': branch_std,
                'y_mean_scaler': y_mean_scaler, 'y_std_scaler': y_std_scaler}, SCALER_SAVE_PATH)
    torch.save({'y_mean_pod_scaled': y_mean_pod_scaled, 'pod_basis': pod_basis_train,
                'num_pod_modes': actual_num_modes, 'unified_a_grid': unified_a_grid,
                'unified_tau_grid': unified_tau_grid}, POD_PARAMS_SAVE_PATH)

    train_dataset = TensorDataset(torch.from_numpy(branch_train_scaled).to(DEVICE),
                                  torch.from_numpy(y_train_scaled).to(DEVICE))
    val_dataset = TensorDataset(torch.from_numpy(branch_val_scaled).to(DEVICE),
                                torch.from_numpy(y_val_scaled).to(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=ADAM_BATCH_SIZE, shuffle=True)

    model = PODDeepONet(BRANCH_INPUT_DIM, HIDDEN_UNITS, NUM_HIDDEN_LAYERS,
                        actual_num_modes, pod_basis_train, y_mean_pod_scaled, DROPOUT_RATE).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = WeightedMSELoss(d2_weight=D2_LOSS_WEIGHT)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, ADAM_ITERATIONS)

    logs = {'iter': [], 'train_loss': [], 'val_iter': [], 'val_loss': []}
    min_val_loss = float('inf')
    early_stop_counter = 0

    pbar = tqdm(total=ADAM_ITERATIONS, desc="训练进度")
    global_iter = 0
    done = False

    while not done:
        for batch_branch, batch_y_scaled in train_loader:
            if global_iter >= ADAM_ITERATIONS: done = True; break

            model.train()
            optimizer.zero_grad()
            y_pred_scaled = model(batch_branch)
            loss, _, _ = loss_fn(y_pred_scaled, batch_y_scaled)
            loss.backward()
            optimizer.step()
            scheduler.step()

            logs['iter'].append(global_iter)
            logs['train_loss'].append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.2e}"})

            if global_iter % VALIDATION_FREQUENCY == 0:
                model.eval()
                with torch.no_grad():
                    val_b, val_y = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset))))
                    val_pred = model(val_b)
                    val_loss, _, _ = loss_fn(val_pred, val_y)
                    logs['val_iter'].append(global_iter)
                    logs['val_loss'].append(val_loss.item())

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    early_stop_counter = 0
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                        done = True;
                        break
            global_iter += 1
    pbar.close()
    print(f"训练完成。最优模型已保存至: {MODEL_SAVE_PATH}")