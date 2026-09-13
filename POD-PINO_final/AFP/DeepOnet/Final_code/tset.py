# -*- coding: utf-8 -*-
import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.interpolate import interp1d  # 导入插值函数

# --- Configuration ---
# Directories and Files
DATA_DIR = r'D:\PINN\zenodo\AFP\DeepONet_Data_DynamicA_DynN\Train_data_2'  # ! 指向您生成数据的目录 !
MODEL_SAVE_PATH = r'D:\PINN\zenodo\AFP\result\DeepOnet_train_result\model_tset_1000_D1_20_(2,6)_input_corrected.pth'  # 更新模型保存路径
SCALER_SAVE_PATH = r'D:\PINN\zenodo\AFP\result\DeepOnet_train_result\scalers_tset_1000_D1_20_(2,6)_input_corrected.pth'
PLOT_SAVE_PATH = r'D:\PINN\zenodo\AFP\result\DeepOnet_train_result\training_loss_tset_1000_D1_20_(2,6)_input_corrected.png'  # 更新绘图保存路径

# Model Hyperparameters
N_BRANCH_EVAL_POINTS = 21  # 文章中指定的固定 A 点数量
BRANCH_INPUT_DIM = N_BRANCH_EVAL_POINTS  # 分支网络输入是 D1(a) 在固定 A 点的值
TRUNK_INPUT_DIM = 2  # A, tau (这里的 A 是输出点，不是分支网络输入)
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 5
OUTPUT_FEATURES = 128  # Size 'p' for branch/trunk output features

# Training Hyperparameters
ADAM_ITERATIONS = 500000
LBFGS_ITERATIONS = 0
LOSS_SWITCH_ITER = 15000
ADAM_LR = 0.001
LR_DECAY_RATE = 0.98
LR_DECAY_STEP = 1000
ADAM_BATCH_SIZE = 1024
LBFGS_BATCH_SIZE = 2 ** 16
VALIDATION_SPLIT = 0.2
SEED = 42
VALIDATION_FREQUENCY = 100
EARLY_STOPPING_THRESHOLD = 1e-6
EARLY_STOPPING_PATIENCE = 1000

# Data Sampling Configuration (针对文件数量)
TARGET_NUM_FILES = 1000

# Computation Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# --- 定义 DeepONet 分支网络所需的固定 A 点 ---
FIXED_A_FOR_BRANCH_INPUT = np.linspace(2, 6, N_BRANCH_EVAL_POINTS)
print(f"DeepONet 分支网络将使用 {N_BRANCH_EVAL_POINTS} 个固定 A 点的 D1(a) 值作为输入：")
print(FIXED_A_FOR_BRANCH_INPUT)

# --- 理论 KM 系数计算函数 (与数据生成脚本中的相同) ---
def theoretical_D1_d(A, nu, kappa, d_diffusion):
    """Theoretical Drift Coefficient D1(A) using d_diffusion/A form."""
    A = np.asarray(A)
    term_gamma = np.zeros_like(A, dtype=float)
    non_zero_A_mask = np.abs(A) > 1e-15
    term_gamma[non_zero_A_mask] = d_diffusion / A[non_zero_A_mask]
    term_nu = nu * A
    term_kappa = (kappa / 8) * A ** 3
    return term_nu - term_kappa + term_gamma

def theoretical_D2_d(A, nu, kappa, d_diffusion):
    """Theoretical Diffusion Coefficient D2(A) using d_diffusion."""
    A = np.asarray(A)
    return np.full_like(A, d_diffusion)


# --- Helper Functions ---
def load_and_preprocess_data(data_dir, target_num_files, validation_split, seed, fixed_a_points_for_branch):
    """
    加载所有 CSV 数据, 合并, 清理并分割。
    此版本会从所有可用文件中随机选择指定数量的文件进行加载。
    分支网络输入将是 D1(a) 在固定 A 点的理论值。
    """
    print(f"从 {data_dir} 加载数据...")
    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_available_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件。请先运行数据生成脚本。")

    if len(all_available_files) > target_num_files:
        selected_files = random.sample(all_available_files, target_num_files)
        print(f"从 {len(all_available_files)} 个文件中随机选择了 {target_num_files} 个文件进行训练。")
    else:
        selected_files = all_available_files
        print(f"可用文件数 ({len(all_available_files)}) 小于或等于目标文件数 ({target_num_files})，将加载所有文件。")

    processed_data_list = []
    # 使用一个集合来跟踪已经处理过的 (nu, kappa, d_diffusion) 组合，避免重复计算理论D1(a)
    # 理论 D1(a) 只依赖于 nu, kappa, d_diffusion，与 tau 和具体的 A 点无关
    unique_param_sets_processed = {} # Key: (nu, kappa, d_diffusion) tuple, Value: theoretical D1(a) at fixed points

    for f in tqdm(selected_files, desc="处理 CSV 文件"):
        try:
            df = pd.read_csv(f)
            df.dropna(subset=['D1_fp', 'D2_fp'], inplace=True)
            if df.empty:
                print(f"警告: 文件 {f} 在清理 NaN 后为空，跳过。")
                continue

            # 获取当前文件的参数 (nu, kappa, d_diffusion)
            current_nu = df['nu'].iloc[0]
            current_kappa = df['kappa'].iloc[0]
            current_d_diffusion = df['d_diffusion'].iloc[0]
            param_tuple = (current_nu, current_kappa, current_d_diffusion)

            # --- 计算理论 D1(a) 作为分支网络输入 ---
            if param_tuple not in unique_param_sets_processed:
                theoretical_d1_for_branch = theoretical_D1_d(
                    fixed_a_points_for_branch, current_nu, current_kappa, current_d_diffusion
                )
                unique_param_sets_processed[param_tuple] = theoretical_d1_for_branch
            else:
                theoretical_d1_for_branch = unique_param_sets_processed[param_tuple]

            # 遍历每个原始数据行，构建 DeepONet 样本
            # 这里的每一行都是一个 (nu, kappa, d_diffusion, tau, A) -> (D1_fp, D2_fp) 样本
            for _, row in df.iterrows():
                processed_data_list.append({
                    'branch_input_D1_a': theoretical_d1_for_branch, # 21个理论D1(a)值
                    'trunk_input_A': row['A'],
                    'trunk_input_tau': row['tau'],
                    'output_D1_fp': row['D1_fp'],
                    'output_D2_fp': row['D2_fp']
                })

        except Exception as e:
            print(f"警告: 处理文件 {f} 时出错: {e}")
            # traceback.print_exc() # 调试时可以打开

    if not processed_data_list:
        raise ValueError("未能处理任何数据。请检查数据文件内容或路径。")

    # 将处理后的列表转换为 NumPy 数组
    branch_inputs_list = [d['branch_input_D1_a'] for d in processed_data_list]
    trunk_inputs_list = [[d['trunk_input_A'], d['trunk_input_tau']] for d in processed_data_list]
    outputs_list = [[d['output_D1_fp'], d['output_D2_fp']] for d in processed_data_list]

    branch_inputs_np = np.array(branch_inputs_list, dtype=np.float64)
    trunk_inputs_np = np.array(trunk_inputs_list, dtype=np.float64)
    outputs_np = np.array(outputs_list, dtype=np.float64)

    print(f"总共处理了 {len(processed_data_list)} 个 DeepONet 样本。")

    # 分割数据集
    (branch_train, branch_val,
     trunk_train, trunk_val,
     y_train, y_val) = train_test_split(branch_inputs_np, trunk_inputs_np, outputs_np,
                                        test_size=validation_split, random_state=seed)
    print(f"训练集大小: {len(y_train)}, 验证集大小: {len(y_val)}")
    return branch_train, branch_val, trunk_train, trunk_val, y_train, y_val


def manual_scaler(data, mean=None, std=None):
    """手动标准化数据 (计算或应用)"""
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0  # 防止除以零
        scaled_data = (data - mean) / std
        return scaled_data, mean, std
    else:
        # 确保 std 不为零
        std_tensor = torch.as_tensor(std, dtype=DTYPE) if isinstance(std, np.ndarray) else std
        std_tensor[std_tensor < 1e-10] = 1.0
        # 确保 data 是 numpy 数组
        data_np = data if isinstance(data, np.ndarray) else data.numpy()
        scaled_data = (data_np - mean) / std_tensor.numpy()
        return scaled_data


def manual_unscaler(scaled_data, mean, std):
    """手动反标准化数据"""
    # 确保 scaled_data 是 numpy 数组
    scaled_data_np = scaled_data if isinstance(scaled_data, np.ndarray) else scaled_data.numpy()
    return scaled_data_np * std + mean


# --- DeepONet Model (与之前相同) ---
class MLP(nn.Module):
    """简单的多层感知机 (用于 Branch 和 Trunk)"""

    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units, dtype=DTYPE))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units, dtype=DTYPE))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, output_dim, dtype=DTYPE))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepONet(nn.Module):
    """DeepONet 架构"""

    def __init__(self, branch_input_dim, trunk_input_dim, hidden_units, num_hidden_layers, output_features):
        super().__init__()
        self.branch = MLP(branch_input_dim, hidden_units, num_hidden_layers, output_features)
        self.trunk1 = MLP(trunk_input_dim, hidden_units, num_hidden_layers, output_features)
        self.trunk2 = MLP(trunk_input_dim, hidden_units, num_hidden_layers, output_features)
        self.bias1 = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        self.bias2 = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, branch_x, trunk_x):
        branch_out = self.branch(branch_x)
        trunk1_out = self.trunk1(trunk_x)
        trunk2_out = self.trunk2(trunk_x)

        output1 = torch.sum(branch_out * trunk1_out, dim=1, keepdim=True) + self.bias1
        output2 = torch.sum(branch_out * trunk2_out, dim=1, keepdim=True) + self.bias2

        return torch.cat((output1, output2), dim=1)


# --- Loss Functions (与之前相同) ---
def loss_mse_n(y_pred_n, y_true_n):
    """计算第 n 个输出的 MSE 损失 L_MSE^(n)"""
    return torch.mean((y_pred_n - y_true_n) ** 2)


def loss_phase1(y_pred, y_true):
    """第一阶段损失函数: L = 0.5 * (L_MSE^(1) + L_MSE^(2))"""
    loss1 = loss_mse_n(y_pred[:, 0], y_true[:, 0])
    loss2 = loss_mse_n(y_pred[:, 1], y_true[:, 1])
    return 0.5 * (loss1 + loss2)


def loss_phase2(y_pred, y_true, epsilon=1e-8):
    """第二阶段损失函数: L = 0.5 * (L_MSE^(1) + L_MSE^(2) * L_MSE^(2) / (L_MSE^(1) + epsilon))"""
    loss1 = loss_mse_n(y_pred[:, 0], y_true[:, 0])
    loss2 = loss_mse_n(y_pred[:, 1], y_true[:, 1])
    weighted_loss2 = loss2 * (loss2 / (loss1 + epsilon))
    return 0.5 * (loss1 + weighted_loss2)


# --- Main Training Script ---
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 1. 加载和预处理数据 (包含文件抽样和 D1(a) 插值)
    try:
        branch_train_np, branch_val_np, trunk_train_np, trunk_val_np, y_train_np, y_val_np = \
            load_and_preprocess_data(DATA_DIR, TARGET_NUM_FILES, VALIDATION_SPLIT, SEED, FIXED_A_FOR_BRANCH_INPUT)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载错误: {e}")
        exit()

    # 2. 标准化数据
    print("标准化数据...")
    # branch_train_np 现在是 (N_samples, 21)
    branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
    trunk_train_scaled, trunk_mean, trunk_std = manual_scaler(trunk_train_np)
    y_train_scaled, y_mean, y_std = manual_scaler(y_train_np)

    branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)
    trunk_val_scaled = manual_scaler(trunk_val_np, trunk_mean, trunk_std)
    y_val_scaled = manual_scaler(y_val_np, y_mean, y_std)

    scalers = {
        'branch_mean': torch.tensor(branch_mean, dtype=DTYPE), 'branch_std': torch.tensor(branch_std, dtype=DTYPE),
        'trunk_mean': torch.tensor(trunk_mean, dtype=DTYPE), 'trunk_std': torch.tensor(trunk_std, dtype=DTYPE),
        'y_mean': torch.tensor(y_mean, dtype=DTYPE), 'y_std': torch.tensor(y_std, dtype=DTYPE),
        'fixed_a_for_branch_input': torch.tensor(FIXED_A_FOR_BRANCH_INPUT, dtype=DTYPE)  # 保存固定 A 点
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    torch.save(scalers, SCALER_SAVE_PATH)
    print(f"Scalers 已保存到 {SCALER_SAVE_PATH}")

    # 3. 创建 PyTorch 数据集和数据加载器
    branch_train_t = torch.tensor(branch_train_scaled, dtype=DTYPE).to(DEVICE)
    trunk_train_t = torch.tensor(trunk_train_scaled, dtype=DTYPE).to(DEVICE)
    y_train_t = torch.tensor(y_train_scaled, dtype=DTYPE).to(DEVICE)
    branch_val_t = torch.tensor(branch_val_scaled, dtype=DTYPE).to(DEVICE)
    trunk_val_t = torch.tensor(trunk_val_scaled, dtype=DTYPE).to(DEVICE)
    y_val_t = torch.tensor(y_val_scaled, dtype=DTYPE).to(DEVICE)
    train_dataset = TensorDataset(branch_train_t, trunk_train_t, y_train_t)
    val_dataset = TensorDataset(branch_val_t, trunk_val_t, y_val_t)
    train_loader_adam = DataLoader(train_dataset, batch_size=ADAM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(LBFGS_BATCH_SIZE, len(val_dataset)))
    print(f"Adam 批次大小: {ADAM_BATCH_SIZE}")
    print(f"验证批次大小: {min(LBFGS_BATCH_SIZE, len(val_dataset))}")

    # 4. 初始化模型、优化器和调度器
    model = DeepONet(BRANCH_INPUT_DIM, TRUNK_INPUT_DIM, HIDDEN_UNITS, NUM_HIDDEN_LAYERS, OUTPUT_FEATURES).to(DEVICE)
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"使用设备: {DEVICE}")
    optimizer_adam = optim.Adam(model.parameters(), lr=ADAM_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=LR_DECAY_STEP, gamma=LR_DECAY_RATE)
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100,
                                  line_search_fn="strong_wolfe")

    # 5. 训练循环
    iterations_log = []
    loss_log = []
    val_iterations_log = []
    val_loss_log = []
    start_time = time.time()
    global_iter = 0
    early_stopping_triggered = False
    early_stopping_counter = 0
    min_val_loss = float('inf')
    best_model_state = None

    print("\n--- 开始 Adam 训练 ---")
    adam_iter_count = 0
    while global_iter < ADAM_ITERATIONS and not early_stopping_triggered:
        model.train()
        for batch_branch, batch_trunk, batch_y in train_loader_adam:
            if global_iter >= ADAM_ITERATIONS or early_stopping_triggered:
                break

            optimizer_adam.zero_grad()
            y_pred = model(batch_branch, batch_trunk)

            if global_iter < LOSS_SWITCH_ITER:
                loss = loss_phase1(y_pred, batch_y)
            else:
                loss = loss_phase2(y_pred, batch_y)

            loss.backward()
            optimizer_adam.step()
            scheduler.step()

            iterations_log.append(global_iter)
            loss_log.append(loss.item())

            if global_iter % VALIDATION_FREQUENCY == 0:
                model.eval()
                current_val_losses = []
                with torch.no_grad():
                    for val_batch_branch, val_batch_trunk, val_batch_y in val_loader:
                        val_y_pred = model(val_batch_branch, val_batch_trunk)
                        if global_iter < LOSS_SWITCH_ITER:
                            val_loss = loss_phase1(val_y_pred, val_batch_y)
                        else:
                            val_loss = loss_phase2(val_y_pred, val_batch_y)
                        current_val_losses.append(val_loss.item())
                avg_val_loss = np.mean(current_val_losses)
                val_iterations_log.append(global_iter)
                val_loss_log.append(avg_val_loss)
                loss_type = "Phase 1" if global_iter < LOSS_SWITCH_ITER else "Phase 2"
                print(
                    f"Iter: {global_iter}, Train Loss ({loss_type}): {loss.item():.6e}, Val Loss: {avg_val_loss:.6e}, LR: {scheduler.get_last_lr()[0]:.6e}")

                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    best_model_state = model.state_dict()
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"\n*** 早停触发于迭代 {global_iter}，验证损失 {avg_val_loss:.6e}，连续 {EARLY_STOPPING_PATIENCE} 次未改善。 ***")
                    early_stopping_triggered = True
                    break

                model.train()
            elif global_iter % 100 == 0:
                loss_type = "Phase 1" if global_iter < LOSS_SWITCH_ITER else "Phase 2"
                print(
                    f"Iter: {global_iter}, Train Loss ({loss_type}): {loss.item():.6e}, LR: {scheduler.get_last_lr()[0]:.6e}")

            global_iter += 1
            adam_iter_count += 1

    print(f"\n--- Adam 训练结束 (实际迭代 {adam_iter_count} 次) ---")

    if LBFGS_ITERATIONS > 0 and not early_stopping_triggered:
        print("\n--- 开始 L-BFGS 训练 ---")
        lbfgs_start_iter = global_iter
        lbfgs_iter_count = 0
        while global_iter < ADAM_ITERATIONS + LBFGS_ITERATIONS and not early_stopping_triggered:
            model.train()
            for batch_branch, batch_trunk, batch_y in DataLoader(train_dataset,
                                                                 batch_size=min(LBFGS_BATCH_SIZE, len(train_dataset)),
                                                                 shuffle=True):
                if global_iter >= ADAM_ITERATIONS + LBFGS_ITERATIONS or early_stopping_triggered:
                    break


                def closure():
                    optimizer_lbfgs.zero_grad()
                    y_pred = model(batch_branch, batch_trunk)
                    if global_iter < LOSS_SWITCH_ITER:
                        loss = loss_phase1(y_pred, batch_y)
                    else:
                        loss = loss_phase2(y_pred, batch_y)
                    loss.backward()
                    return loss


                loss = optimizer_lbfgs.step(closure)
                current_loss = loss.item()
                iterations_log.append(global_iter)
                loss_log.append(current_loss)

                if global_iter % VALIDATION_FREQUENCY == 0 or global_iter == lbfgs_start_iter:
                    model.eval()
                    current_val_losses = []
                    with torch.no_grad():
                        for val_batch_branch, val_batch_trunk, val_batch_y in val_loader:
                            val_y_pred = model(val_batch_branch, val_batch_trunk)
                            if global_iter < LOSS_SWITCH_ITER:
                                val_loss = loss_phase1(val_y_pred, val_batch_y)
                            else:
                                val_loss = loss_phase2(val_y_pred, val_batch_y)
                            current_val_losses.append(val_loss.item())
                    avg_val_loss = np.mean(current_val_losses)
                    val_iterations_log.append(global_iter)
                    val_loss_log.append(avg_val_loss)
                    loss_type = "Phase 1" if global_iter < LOSS_SWITCH_ITER else "Phase 2"
                    print(
                        f"Iter: {global_iter}, Train Loss ({loss_type}): {current_loss:.6e}, Val Loss: {avg_val_loss:.6e}")

                    if avg_val_loss < min_val_loss:
                        min_val_loss = avg_val_loss
                        early_stopping_counter = 0
                        best_model_state = model.state_dict()
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                        print(
                            f"\n*** 早停触发于迭代 {global_iter}，验证损失 {avg_val_loss:.6e}，连续 {EARLY_STOPPING_PATIENCE} 次未改善。 ***")
                        early_stopping_triggered = True
                        break

                    model.train()
                elif global_iter % 100 == 0:
                    loss_type = "Phase 1" if global_iter < LOSS_SWITCH_ITER else "Phase 2"
                    print(f"Iter: {global_iter}, Train Loss ({loss_type}): {current_loss:.6e}")

                global_iter += 1
                lbfgs_iter_count += 1
                if early_stopping_triggered:
                    break

        print(f"\n--- L-BFGS 训练结束 (实际迭代 {lbfgs_iter_count} 次) ---")
    else:
        if LBFGS_ITERATIONS > 0:
            print("\n--- 由于 Adam 阶段触发早停，跳过 L-BFGS 训练 ---")
        else:
            print("\n--- L-BFGS 迭代次数设置为 0，跳过 L-BFGS 训练 ---")

    end_time = time.time()
    total_iterations = len(iterations_log)
    print(f"\n--- 训练完成 (总共 {total_iterations} 次迭代) ---")
    print(f"总训练时间: {end_time - start_time:.2f} 秒")
    if early_stopping_triggered:
        print(f"训练因早停而提前结束于迭代 {global_iter}.")

    # 6. 保存模型参数 (保存最佳模型)
    if best_model_state:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(best_model_state, MODEL_SAVE_PATH)
        print(f"最佳模型参数已保存到 {MODEL_SAVE_PATH}")
    else:
        print("未找到最佳模型状态，可能训练未开始或验证损失未改善。")

    # 7. 绘制损失曲线 (包含验证损失)
    print("绘制损失曲线...")
    plt.figure(figsize=(12, 6))
    plt.plot(iterations_log, loss_log, label='Training Loss', alpha=0.8, linewidth=1)
    if val_iterations_log:
        plt.plot(val_iterations_log, val_loss_log, label='Validation Loss', alpha=0.8, linewidth=1.5, linestyle='-',
                 marker='o', markersize=3)

    actual_adam_end_iter = min(ADAM_ITERATIONS, total_iterations)
    if ADAM_ITERATIONS > 0 and actual_adam_end_iter > 0:
        plt.axvline(x=actual_adam_end_iter, color='gray', linestyle='--',
                    label=f'Adam End (Iter {actual_adam_end_iter})')

    actual_loss_switch_iter = min(LOSS_SWITCH_ITER, total_iterations)
    if LOSS_SWITCH_ITER > 0 and actual_loss_switch_iter > 0:
        plt.axvline(x=actual_loss_switch_iter, color='red', linestyle='--',
                    label=f'Loss Switch (Iter {actual_loss_switch_iter})')

    if early_stopping_triggered and val_iterations_log:
        stop_iter = val_iterations_log[-1]
        stop_loss = val_loss_log[-1]
        plt.scatter([stop_iter], [stop_loss], color='red', s=100, zorder=5,
                    label=f'Early Stop Trigger ({stop_iter}, {stop_loss:.2e})')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss Curve with Early Stopping (D1(a) as Branch Input)')  # 更新标题
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
    try:
        plt.savefig(PLOT_SAVE_PATH)
        print(f"损失曲线图已保存到 {PLOT_SAVE_PATH}")
    except Exception as e:
        print(f"保存损失曲线图时出错: {e}")
    plt.show()

    # 8. 打印最终验证损失
    if val_loss_log:
        print(f"最终记录的验证损失: {val_loss_log[-1]:.6e} (在迭代 {val_iterations_log[-1]})")
        print(f"最佳验证损失: {min_val_loss:.6e}")
    else:
        print("没有记录验证损失。")