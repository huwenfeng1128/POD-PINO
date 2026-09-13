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
from torch.optim.lr_scheduler import LambdaLR

# --- 1. 配置参数 ---
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\compare\result\POD-DeepONet_v3_Final'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'  # 代码1产生的数据
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 24
TARGET_NUM_FILES = 2500
RUN_ID = f"pod_v3_seed{SEED}_files{TARGET_NUM_FILES}"

# 文件保存路径
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, f'model_{RUN_ID}.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, f'scalers_{RUN_ID}.pth')
POD_PARAMS_SAVE_PATH = os.path.join(RESULT_DIR, f'pod_params_{RUN_ID}.pth')
METRICS_SAVE_PATH = os.path.join(RESULT_DIR, f'metrics_{RUN_ID}.csv')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, f'loss_{RUN_ID}.png')

# 超参数 (对齐代码2)
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
REQUESTED_NUM_POD_MODES = 100
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6

ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 64
ADAM_ITERATIONS = 500000
D2_LOSS_WEIGHT = 0.5
VALIDATION_SPLIT = 0.2
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 500
WARMUP_STEPS = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# --- 2. 辅助函数 ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_unified_data(data_dir, target_num_files, seed):
    print(f"从 {data_dir} 加载统一数据并抽样...")
    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    random.seed(seed)
    selected_files = random.sample(all_available_files, min(len(all_available_files), target_num_files))

    branch_inputs_list, y_snapshots_list = [], []
    unified_a_grid, unified_tau_grid = None, None

    for f in tqdm(selected_files, desc="加载快照"):
        df = pd.read_csv(f).dropna(subset=['D1_fp', 'D2_fp'])
        if df.empty: continue
        if unified_a_grid is None:
            unified_a_grid, unified_tau_grid = sorted(df['A'].unique()), sorted(df['tau'].unique())

        params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values
        branch_inputs_list.append(params)
        df_sorted = df.sort_values(by=['tau', 'A'])
        snapshot = np.concatenate([df_sorted['D1_fp'].values, df_sorted['D2_fp'].values])
        y_snapshots_list.append(snapshot)

    return np.array(branch_inputs_list), np.array(y_snapshots_list), np.array(unified_a_grid), np.array(
        unified_tau_grid)


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def pod(y_data_scaled, requested_num_modes):
    print("执行 POD 分解...")
    y_mean_pod_scaled = np.mean(y_data_scaled, axis=0)
    U, S, Vt = np.linalg.svd(y_data_scaled - y_mean_pod_scaled, full_matrices=False)
    actual_num_modes = min(requested_num_modes, Vt.shape[0])
    pod_basis = Vt.T[:, :actual_num_modes]
    return y_mean_pod_scaled, pod_basis, S, actual_num_modes


# --- 3. 模型定义 (LayerNorm + GELU) ---
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
            nn.Linear(hidden_units, output_dim, dtype=DTYPE)
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(self, branch_dim, hidden, layers, num_modes, pod_basis, y_mean_pod, dropout):
        super().__init__()
        self.branch = MLP(branch_dim, hidden, layers, num_modes, dropout)
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


class WeightedMSELoss(nn.Module):
    def __init__(self, d2_weight=0.5):
        super().__init__()
        self.d2_weight = d2_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        field_len = y_pred.shape[1] // 2
        loss_d1 = self.mse(y_pred[:, :field_len], y_true[:, :field_len])
        loss_d2 = self.mse(y_pred[:, field_len:], y_true[:, field_len:])
        return (1 - self.d2_weight) * loss_d1 + self.d2_weight * loss_d2, loss_d1.detach(), loss_d2.detach()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# --- 4. 训练逻辑 ---
if __name__ == "__main__":
    set_seed(SEED)

    # 数据准备
    b_in, y_snap, a_grid, tau_grid = load_unified_data(DATA_DIR, TARGET_NUM_FILES, SEED)
    b_train, b_val, y_train, y_val = train_test_split(b_in, y_snap, test_size=VALIDATION_SPLIT, random_state=SEED)

    b_train_s, b_mean, b_std = manual_scaler(b_train)
    b_val_s = manual_scaler(b_val, b_mean, b_std)
    y_train_s, y_mean_s, y_std_s = manual_scaler(y_train)
    y_val_s = manual_scaler(y_val, y_mean_s, y_std_s)

    y_mean_pod, pod_basis, S, actual_modes = pod(y_train_s, REQUESTED_NUM_POD_MODES)

    # 保存配置
    torch.save({'b_mean': b_mean, 'b_std': b_std, 'y_mean_s': y_mean_s, 'y_std_s': y_std_s}, SCALER_SAVE_PATH)
    torch.save({'y_mean_pod': y_mean_pod, 'pod_basis': pod_basis, 'a_grid': a_grid, 'tau_grid': tau_grid},
               POD_PARAMS_SAVE_PATH)

    # Dataset
    train_loader = DataLoader(TensorDataset(torch.tensor(b_train_s, dtype=DTYPE).to(DEVICE),
                                            torch.tensor(y_train_s, dtype=DTYPE).to(DEVICE)),
                              batch_size=ADAM_BATCH_SIZE, shuffle=True)
    val_b = torch.tensor(b_val_s, dtype=DTYPE).to(DEVICE)
    val_y = torch.tensor(y_val_s, dtype=DTYPE).to(DEVICE)

    # 初始化
    model = PODDeepONet(BRANCH_INPUT_DIM, HIDDEN_UNITS, NUM_HIDDEN_LAYERS, actual_modes,
                        pod_basis, y_mean_pod, DROPOUT_RATE).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = WeightedMSELoss(d2_weight=D2_LOSS_WEIGHT)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, ADAM_ITERATIONS)

    # 记录容器
    logs = {'iter': [], 'train_loss': [], 'val_loss': [], 'iter_time': []}
    min_val_loss = float('inf')
    early_stop_counter = 0

    print("--- 开始 AdamW 训练 ---")
    total_train_start = time.time()
    global_iter = 0
    done = False
    pbar = tqdm(total=ADAM_ITERATIONS, desc="进度")

    while not done:
        for batch_b, batch_y in train_loader:
            if global_iter >= ADAM_ITERATIONS:
                done = True;
                break

            iter_start = time.perf_counter()

            model.train()
            optimizer.zero_grad()
            y_pred = model(batch_b)
            loss, _, _ = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_stop = time.perf_counter()

            logs['iter'].append(global_iter)
            logs['train_loss'].append(loss.item())
            logs['iter_time'].append(iter_stop - iter_start)

            if global_iter % VALIDATION_FREQUENCY == 0:
                model.eval()
                with torch.no_grad():
                    v_pred = model(val_b)
                    v_loss, _, _ = loss_fn(v_pred, val_y)
                    logs['val_loss'].append(v_loss.item())

                if v_loss < min_val_loss:
                    min_val_loss = v_loss.item()
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n触发早停，迭代次数: {global_iter}")
                    done = True;
                    break

            global_iter += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.2e}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

    total_train_time = time.time() - total_train_start
    pbar.close()

    # --- 5. 结果保存与统计 ---
    avg_iter_time = np.mean(logs['iter_time'])
    print(f"\n训练结束统计:")
    print(f"收敛总迭代次数: {global_iter}")
    print(f"总训练时间: {total_train_time:.2f} s")
    print(f"平均单次迭代时间: {avg_iter_time:.6f} s")

    # 保存统计指标
    metrics_df = pd.DataFrame({
        'metric': ['total_iterations', 'total_train_time_sec', 'avg_iter_time_sec', 'best_val_loss', 'seed',
                   'num_files'],
        'value': [global_iter, total_train_time, avg_iter_time, min_val_loss, SEED, TARGET_NUM_FILES]
    })
    metrics_df.to_csv(METRICS_SAVE_PATH, index=False)

    # 绘制 Loss
    plt.figure(figsize=(10, 5))
    plt.plot(logs['iter'], logs['train_loss'], label='Train Loss', alpha=0.6)
    plt.yscale('log')
    plt.title(f'POD-DeepONet Training Loss (Seed {SEED})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

    print(f"所有结果已保存至: {RESULT_DIR}")