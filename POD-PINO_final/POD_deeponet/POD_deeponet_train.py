# -*- coding: utf-8 -*-
import os
import glob
import time
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
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configuration ---
# Directories and Files
RESULT_DIR = r'D:\PINN\zenodo\AFP\result\POD_AdaptiveLR'  # 新的输出目录
DATA_DIR = r'D:\PINN\zenodo\AFP\DeepONet_Data_DynamicA_DynN\Train_data_2'
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'model_pod_adaptive_lr.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, 'scalers_pod_adaptive_lr.pth')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, 'training_loss_pod_adaptive_lr.png')
LOSS_DATA_SAVE_PATH = os.path.join(RESULT_DIR, 'loss_data_pod_adaptive_lr.csv')
POD_PARAMS_SAVE_PATH = os.path.join(RESULT_DIR, 'pod_params_adaptive_lr.pth')
POD_ANALYSIS_PLOT_PATH = os.path.join(RESULT_DIR, 'pod_singular_values.png')

# Model Hyperparameters
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 3
REQUESTED_NUM_POD_MODES = 100  # 增加模式数以确保POD不是瓶颈
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-5

# Training Hyperparameters
ADAM_LR = 1e-3  # 初始学习率
ADAM_BATCH_SIZE = 256
ADAM_ITERATIONS = 200000
LOSS_SWITCH_ITER = 150000
VALIDATION_SPLIT = 0.2
SEED = 42
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 500  # 适当增加耐心，给自适应学习率调整留出时间

# Data Config
TARGET_NUM_FILES = 5000
UNIFIED_A_POINTS = 21

# Computation Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# --- Helper Functions ---
def load_and_interpolate_data(data_dir, target_num_files, unified_a_points):
    print(f"从 {data_dir} 加载数据...");
    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_available_files: raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件。")
    selected_files = random.sample(all_available_files, min(len(all_available_files), target_num_files))
    print(f"从 {len(all_available_files)} 个文件中选择了 {len(selected_files)} 个文件进行分析和加载。")
    print("\n--- 阶段1: 扫描文件以确定全局A范围 ---");
    global_a_min, global_a_max = float('inf'), float('-inf')
    for f in tqdm(selected_files, desc="扫描A范围"):
        try:
            df = pd.read_csv(f, usecols=['A'], on_bad_lines='skip')
            if not df.empty: global_a_min = min(global_a_min, df['A'].min()); global_a_max = max(global_a_max,
                                                                                                 df['A'].max())
        except Exception:
            pass
    if np.isinf(global_a_min) or np.isinf(global_a_max): raise ValueError("无法确定有效的全局A范围。")
    print(f"\n--- 阶段2: 加载数据并插值到统一A网格 ---");
    print(f"全局A范围: [{global_a_min:.4f}, {global_a_max:.4f}]")
    unified_a_grid = np.linspace(global_a_min, global_a_max, unified_a_points);
    print(f"已创建 {unified_a_points} 个点的统一A网格。")
    branch_inputs_list, y_interpolated_snapshots_list = [], []
    for f in tqdm(selected_files, desc="加载并插值快照"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip');
            df.dropna(subset=['D1_fp', 'D2_fp'], inplace=True)
            if df.empty: continue
            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values
            tau_values = sorted(df['tau'].unique());
            n_tau_points = len(tau_values)
            interpolated_d1_field = np.zeros((n_tau_points, unified_a_points));
            interpolated_d2_field = np.zeros((n_tau_points, unified_a_points))
            for i, tau in enumerate(tau_values):
                tau_slice_df = df[df['tau'] == tau].sort_values(by='A')
                original_a_grid = tau_slice_df['A'].values;
                original_d1_values = tau_slice_df['D1_fp'].values;
                original_d2_values = tau_slice_df['D2_fp'].values
                if len(original_a_grid) < 2: continue
                interp_func_d1 = interp1d(original_a_grid, original_d1_values, kind='linear', bounds_error=False,
                                          fill_value="extrapolate")
                interp_func_d2 = interp1d(original_a_grid, original_d2_values, kind='linear', bounds_error=False,
                                          fill_value="extrapolate")
                interpolated_d1_field[i, :] = interp_func_d1(unified_a_grid);
                interpolated_d2_field[i, :] = interp_func_d2(unified_a_grid)
            snapshot_d1_flat = interpolated_d1_field.flatten();
            snapshot_d2_flat = interpolated_d2_field.flatten()
            final_snapshot = np.concatenate([snapshot_d1_flat, snapshot_d2_flat])
            branch_inputs_list.append(params);
            y_interpolated_snapshots_list.append(final_snapshot)
        except Exception:
            pass
    if not branch_inputs_list: raise ValueError("未能成功加载和插值任何快照。")
    branch_inputs_np = np.array(branch_inputs_list);
    y_snapshots_np = np.array(y_interpolated_snapshots_list)
    print(f"\n成功创建 {len(branch_inputs_np)} 个长度统一的插值后快照。")
    print(f"Branch 输入形状: {branch_inputs_np.shape}");
    print(f"Y 快照形状: {y_snapshots_np.shape}")

    # --- !! 修改：返回 unified_a_grid !! ---
    return branch_inputs_np, y_snapshots_np, unified_a_grid


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0);
        std = np.std(data, axis=0);
        std[std < 1e-10] = 1.0;
        return (
                       data - mean) / std, mean, std
    else:
        return (data - mean) / std


def pod(y_data, requested_num_modes):
    print("在快照矩阵上执行POD...");
    y_mean = np.mean(y_data, axis=0);
    y_centered = y_data - y_mean
    U, S, Vt = np.linalg.svd(y_centered, full_matrices=False);
    max_available_modes = Vt.shape[0]
    actual_num_modes = min(requested_num_modes, max_available_modes)
    if actual_num_modes < requested_num_modes: print(
        f"!!! 警告: 数据集只能提供 {max_available_modes} 个模式。将使用所有可用的 {actual_num_modes} 个模式。")
    pod_basis = Vt.T[:, :actual_num_modes];
    print(f"POD: 原始场维度 {y_data.shape[1]}, 提取了 {actual_num_modes} 个模式。")
    print(f"POD基形状: {pod_basis.shape}");
    return y_mean, pod_basis, S, actual_num_modes


def plot_pod_analysis(S, save_path):
    plt.figure(figsize=(14, 6));
    plt.subplot(1, 2, 1);
    plt.plot(range(1, len(S) + 1), S, 'o-');
    plt.yscale('log')
    plt.title('Singular Value Decay');
    plt.xlabel('Mode Number');
    plt.ylabel('Singular Value');
    plt.grid(True, which='both', linestyle='--')
    plt.subplot(1, 2, 2);
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    plt.plot(range(1, len(cumulative_energy) + 1), cumulative_energy, 'o-');
    plt.ylim([0, 1.05])
    plt.title('Cumulative Energy');
    plt.xlabel('Number of Modes');
    plt.ylabel('Fraction of Total Energy');
    plt.grid(True, linestyle='--')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% Energy');
    plt.axhline(y=0.999, color='g', linestyle='--', label='99.9% Energy')
    plt.legend();
    plt.tight_layout();
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path);
    print(f"POD分析图已保存到 {save_path}");
    plt.show()


# --- Model Definitions ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE), nn.Tanh(), nn.Dropout(p=dropout_rate)]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(hidden_units, hidden_units, dtype=DTYPE), nn.Tanh(), nn.Dropout(p=dropout_rate)])
        layers.append(nn.Linear(hidden_units, output_dim, dtype=DTYPE))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, pod_basis, y_mean,
                 dropout_rate):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, num_pod_modes, dropout_rate)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean = nn.Parameter(torch.tensor(y_mean, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        branch_out_coeffs = self.branch(branch_x);
        y_pred = torch.matmul(branch_out_coeffs, self.pod_basis.T) + self.y_mean
        return y_pred


# --- Loss Functions ---
def loss_mse_n(y_pred_n, y_true_n): return torch.mean((y_pred_n - y_true_n) ** 2)


def loss_phase1(y_pred, y_true):
    field_len = y_pred.shape[1] // 2;
    loss1 = loss_mse_n(y_pred[:, :field_len], y_true[:, :field_len]);
    loss2 = loss_mse_n(y_pred[:, field_len:], y_true[:, field_len:])
    return 0.5 * (loss1 + loss2)


def loss_phase2(y_pred, y_true, epsilon=1e-8):
    field_len = y_pred.shape[1] // 2;
    loss1 = loss_mse_n(y_pred[:, :field_len], y_true[:, :field_len]);
    loss2 = loss_mse_n(y_pred[:, field_len:], y_true[:, field_len:])
    return 0.5 * (loss1 + loss2 * (loss2 / (loss1 + epsilon)))


if __name__ == "__main__":
    torch.manual_seed(SEED);
    np.random.seed(SEED);
    random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    branch_inputs_np, y_snapshots_np, unified_a_grid_for_saving = load_and_interpolate_data(DATA_DIR, TARGET_NUM_FILES,
                                                                                            UNIFIED_A_POINTS)

    SNAPSHOT_FIELD_DIM = y_snapshots_np.shape[1]
    branch_train_np, branch_val_np, y_train_snapshots_np, y_val_snapshots_np = train_test_split(
        branch_inputs_np, y_snapshots_np, test_size=VALIDATION_SPLIT, random_state=SEED)
    print(f"\n训练集快照数量: {len(branch_train_np)}, 验证集快照数量: {len(branch_val_np)}")

    y_mean_pod, pod_basis, singular_values, actual_num_modes = pod(y_train_snapshots_np, REQUESTED_NUM_POD_MODES)
    plot_pod_analysis(singular_values, POD_ANALYSIS_PLOT_PATH)

    branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
    branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)

    scalers = {'branch_mean': torch.tensor(branch_mean, dtype=DTYPE),
               'branch_std': torch.tensor(branch_std, dtype=DTYPE)}

    # --- !! 修改：在 pod_params 中保存 unified_a_grid !! ---
    pod_params = {'y_mean_pod': torch.tensor(y_mean_pod, dtype=DTYPE),
                  'pod_basis': torch.tensor(pod_basis, dtype=DTYPE),
                  'num_pod_modes': actual_num_modes,
                  'snapshot_field_dim': SNAPSHOT_FIELD_DIM,
                  'unified_a_grid': torch.tensor(unified_a_grid_for_saving, dtype=DTYPE)  # <-- 新增这一行
                  }

    torch.save(scalers, SCALER_SAVE_PATH);
    print(f"Scalers 已保存到 {SCALER_SAVE_PATH}")
    torch.save(pod_params, POD_PARAMS_SAVE_PATH);
    print(f"POD参数已保存到 {POD_PARAMS_SAVE_PATH}")

    train_dataset = TensorDataset(torch.tensor(branch_train_scaled, dtype=DTYPE).to(DEVICE),
                                  torch.tensor(y_train_snapshots_np, dtype=DTYPE).to(DEVICE))
    val_dataset = TensorDataset(torch.tensor(branch_val_scaled, dtype=DTYPE).to(DEVICE),
                                torch.tensor(y_val_snapshots_np, dtype=DTYPE).to(DEVICE))
    adam_batch_size = min(ADAM_BATCH_SIZE, len(train_dataset)) if train_dataset else ADAM_BATCH_SIZE
    train_loader_adam = DataLoader(train_dataset, batch_size=adam_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(adam_batch_size * 2, len(val_dataset))) if val_dataset else None

    model = PODDeepONet(BRANCH_INPUT_DIM, HIDDEN_UNITS, NUM_HIDDEN_LAYERS, actual_num_modes, pod_basis, y_mean_pod,
                        DROPOUT_RATE).to(DEVICE)
    print("\n--- 模型架构 ---\n", model,
          f"\n模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n使用设备: {DEVICE}\n")

    optimizer_adam = optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)

    # --- !! 修改: 使用 ReduceLROnPlateau 调度器 !! ---
    scheduler = ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=150, verbose=True)

    iterations_log, loss_log, val_iterations_log, val_loss_log = [], [], [], []
    start_time = time.time();
    global_iter = 0;
    early_stopping_counter = 0;
    min_val_loss = float('inf')
    best_model_state = None;
    early_stopping_triggered = False

    print("--- 开始 Adam 训练 ---")
    if not train_dataset:
        print("!!! 错误: 训练集为空，无法开始训练。")
    else:
        while global_iter < ADAM_ITERATIONS and not early_stopping_triggered:
            model.train()
            for batch_branch, batch_y_snapshot in train_loader_adam:
                if global_iter >= ADAM_ITERATIONS: break
                optimizer_adam.zero_grad()
                y_pred = model(batch_branch)
                loss = loss_phase1(y_pred, batch_y_snapshot) if global_iter < LOSS_SWITCH_ITER else loss_phase2(y_pred,
                                                                                                                batch_y_snapshot)
                loss.backward();
                optimizer_adam.step()
                # --- !! 注意: StepLR 的 scheduler.step() 已被移除 !! ---

                iterations_log.append(global_iter);
                loss_log.append(loss.item())

                if global_iter % VALIDATION_FREQUENCY == 0:
                    model.eval()
                    current_val_losses = []
                    with torch.no_grad():
                        for val_batch_branch, val_batch_y in val_loader:
                            val_y_pred = model(val_batch_branch)
                            val_loss = loss_phase1(val_y_pred,
                                                   val_batch_y) if global_iter < LOSS_SWITCH_ITER else loss_phase2(
                                val_y_pred, val_batch_y)
                            current_val_losses.append(val_loss.item())
                    avg_val_loss = np.mean(current_val_losses)
                    val_iterations_log.append(global_iter);
                    val_loss_log.append(avg_val_loss)

                    print(
                        f"Iter: {global_iter}, Train Loss: {loss.item():.6e}, Val Loss: {avg_val_loss:.6e}, LR: {optimizer_adam.param_groups[0]['lr']:.6e}")

                    if avg_val_loss < min_val_loss:
                        min_val_loss = avg_val_loss;
                        early_stopping_counter = 0;
                        best_model_state = model.state_dict()
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter * VALIDATION_FREQUENCY >= EARLY_STOPPING_PATIENCE * VALIDATION_FREQUENCY:
                        print(f"\n*** 早停触发: 验证损失连续 {EARLY_STOPPING_PATIENCE} 次检查未改善。 ***")
                        early_stopping_triggered = True;
                        break

                    # --- !! 修改: 在验证后调用新的调度器 !! ---
                    scheduler.step(avg_val_loss)

                global_iter += 1
            if early_stopping_triggered: break

        end_time = time.time();
        print(f"\n--- 训练结束 (共 {global_iter} 次迭代) ---\n总训练时间: {end_time - start_time:.2f} 秒")
        if best_model_state:
            torch.save(best_model_state, MODEL_SAVE_PATH);
            print(f"最佳模型参数已保存到 {MODEL_SAVE_PATH}")
        else:
            torch.save(model.state_dict(), MODEL_SAVE_PATH);
            print("警告: 未找到最佳模型状态，将保存当前模型。")

        plt.figure(figsize=(12, 6))
        plt.plot(iterations_log, loss_log, label='Training Loss', alpha=0.8, linewidth=1)
        if val_iterations_log: plt.plot(val_iterations_log, val_loss_log, label='Validation Loss', alpha=0.8,
                                        linewidth=1.5, marker='o', markersize=2)
        plt.axvline(x=LOSS_SWITCH_ITER, color='red', linestyle='--', label=f'Loss Switch')
        if early_stopping_triggered and val_iterations_log:
            stop_iter, stop_loss = val_iterations_log[-1], val_loss_log[-1]
            plt.scatter([stop_iter], [stop_loss], color='red', s=100, zorder=5, label=f'Early Stop')
        plt.xlabel('Iteration');
        plt.ylabel('Loss');
        plt.yscale('log');
        plt.title('Training and Validation Loss Curve (POD, Adaptive LR)')
        plt.legend();
        plt.grid(True, which='both', linestyle='--', linewidth=0.5);
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH);
        print(f"损失曲线图已保存到 {PLOT_SAVE_PATH}");
        plt.show()

        loss_df = pd.DataFrame({'Iteration': iterations_log, 'Training_Loss': loss_log})
        val_loss_df = pd.DataFrame({'Iteration': val_iterations_log, 'Validation_Loss': val_loss_log})
        full_loss_df = pd.merge(loss_df, val_loss_df, on='Iteration', how='outer').sort_values(by='Iteration')
        full_loss_df.to_csv(LOSS_DATA_SAVE_PATH, index=False);
        print(f"损失数据已保存到 {LOSS_DATA_SAVE_PATH}")