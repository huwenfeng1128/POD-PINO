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
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'  # 代码1生成的数据路径
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\compare\result\Vanilla_DeepONet_v3'
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'best_model.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, 'scalers.pth')
METRICS_SAVE_PATH = os.path.join(RESULT_DIR, 'training_metrics.csv')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, 'loss_curve.png')

# 核心超参数 (对齐代码2)
SEED = 24
TARGET_NUM_FILES = 2500
BRANCH_INPUT_DIM = 3  # nu, kappa, d
TRUNK_INPUT_DIM = 2  # A, tau
HIDDEN_UNITS = 128  # 代码2参数
NUM_HIDDEN_LAYERS = 4  # 代码2参数
OUTPUT_FEATURES = 128  # p (branch和trunk乘积的空间维度)
DROPOUT_RATE = 0.1  # 代码2参数

# 训练超参数 (对齐代码2)
ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 1024
ADAM_ITERATIONS = 500000
WARMUP_STEPS = 5000
D2_LOSS_WEIGHT = 0.5  # 代码2设置的权重
WEIGHT_DECAY = 1e-6
VALIDATION_SPLIT = 0.2
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# --- 2. 辅助函数 ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sampled_data(data_dir, target_num_files, seed):
    print(f"正在从 {data_dir} 加载并抽样数据...")
    all_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    random.seed(seed)
    selected_files = random.sample(all_files, min(len(all_files), target_num_files))

    df_list = []
    for f in tqdm(selected_files, desc="读取CSV"):
        df = pd.read_csv(f).dropna(subset=['D1_fp', 'D2_fp'])
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    branch_in = full_df[['nu', 'kappa', 'd_diffusion']].values
    trunk_in = full_df[['A', 'tau']].values
    targets = full_df[['D1_fp', 'D2_fp']].values
    return branch_in, trunk_in, targets


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


# --- 3. 模型架构 (对齐代码2的策略: LayerNorm + GELU) ---
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


class VanillaDeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden, layers, p, dropout):
        super().__init__()
        self.branch = MLP(branch_dim, hidden, layers, p, dropout)
        self.trunk_d1 = MLP(trunk_dim, hidden, layers, p, dropout)
        self.trunk_d2 = MLP(trunk_dim, hidden, layers, p, dropout)
        self.b1 = nn.Parameter(torch.zeros(1, dtype=DTYPE))
        self.b2 = nn.Parameter(torch.zeros(1, dtype=DTYPE))

    def forward(self, x_b, x_t):
        b_out = self.branch(x_b)
        t1_out = self.trunk_d1(x_t)
        t2_out = self.trunk_d2(x_t)
        # 点积操作实现算子映射
        out1 = torch.sum(b_out * t1_out, dim=1, keepdim=True) + self.b1
        out2 = torch.sum(b_out * t2_out, dim=1, keepdim=True) + self.b2
        return torch.cat([out1, out2], dim=1)


# --- 4. 损失函数与调度器 ---
class WeightedMSELoss(nn.Module):
    def __init__(self, d2_weight=0.5):
        super().__init__()
        self.d2_weight = d2_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss_d1 = self.mse(y_pred[:, 0], y_true[:, 0])
        loss_d2 = self.mse(y_pred[:, 1], y_true[:, 1])
        return (1 - self.d2_weight) * loss_d1 + self.d2_weight * loss_d2, loss_d1.detach(), loss_d2.detach()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# --- 5. 主训练流程 ---
if __name__ == "__main__":
    set_seed(SEED)

    # 加载与标准化
    b_raw, t_raw, y_raw = load_sampled_data(DATA_DIR, TARGET_NUM_FILES, SEED)

    b_train, b_val, t_train, t_val, y_train, y_val = train_test_split(
        b_raw, t_raw, y_raw, test_size=VALIDATION_SPLIT, random_state=SEED)

    b_train_s, b_mean, b_std = manual_scaler(b_train)
    t_train_s, t_mean, t_std = manual_scaler(t_train)
    y_train_s, y_mean, y_std = manual_scaler(y_train)

    b_val_s = manual_scaler(b_val, b_mean, b_std)
    t_val_s = manual_scaler(t_val, t_mean, t_std)
    y_val_s = manual_scaler(y_val, y_mean, y_std)

    torch.save({'b_mean': b_mean, 'b_std': b_std, 't_mean': t_mean, 't_std': t_std,
                'y_mean': y_mean, 'y_std': y_std}, SCALER_SAVE_PATH)

    # DataLoader
    train_ds = TensorDataset(torch.tensor(b_train_s, dtype=DTYPE).to(DEVICE),
                             torch.tensor(t_train_s, dtype=DTYPE).to(DEVICE),
                             torch.tensor(y_train_s, dtype=DTYPE).to(DEVICE))
    train_loader = DataLoader(train_ds, batch_size=ADAM_BATCH_SIZE, shuffle=True)

    val_b_t = torch.tensor(b_val_s, dtype=DTYPE).to(DEVICE)
    val_t_t = torch.tensor(t_val_s, dtype=DTYPE).to(DEVICE)
    val_y_t = torch.tensor(y_val_s, dtype=DTYPE).to(DEVICE)

    # 模型初始化
    model = VanillaDeepONet(BRANCH_INPUT_DIM, TRUNK_INPUT_DIM, HIDDEN_UNITS,
                            NUM_HIDDEN_LAYERS, OUTPUT_FEATURES, DROPOUT_RATE).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = WeightedMSELoss(d2_weight=D2_LOSS_WEIGHT)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, ADAM_ITERATIONS)

    # 训练统计
    logs = {'iter': [], 'train_loss': [], 'val_loss': [], 'iter_time': []}
    min_val_loss = float('inf')
    early_stop_counter = 0
    total_start_time = time.time()

    print(f"--- 开始训练 (目标迭代: {ADAM_ITERATIONS}) ---")
    pbar = tqdm(total=ADAM_ITERATIONS, desc="进度")
    global_iter = 0
    done = False

    while not done:
        for b_batch, t_batch, y_batch in train_loader:
            if global_iter >= ADAM_ITERATIONS:
                done = True;
                break

            iter_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()

            y_pred = model(b_batch, t_batch)
            loss, _, _ = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_stop = time.perf_counter()

            # 记录指标
            logs['iter'].append(global_iter)
            logs['train_loss'].append(loss.item())
            logs['iter_time'].append(iter_stop - iter_start)

            if global_iter % VALIDATION_FREQUENCY == 0:
                model.eval()
                with torch.no_grad():
                    v_pred = model(val_b_t, val_t_t)
                    v_loss, v_d1, v_d2 = loss_fn(v_pred, val_y_t)
                    logs['val_loss'].append(v_loss.item())

                if v_loss < min_val_loss:
                    min_val_loss = v_loss.item()
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                pbar.set_postfix({'V_Loss': f"{v_loss.item():.2e}", 'best': f"{min_val_loss:.2e}"})

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n早停触发于迭代 {global_iter}")
                    done = True;
                    break

            global_iter += 1
            pbar.update(1)

    total_duration = time.time() - total_start_time
    pbar.close()

    # --- 6. 统计与保存 ---
    avg_iter_time = np.mean(logs['iter_time'])
    print(f"\n训练总结:")
    print(f"收敛总时间: {total_duration:.2f} s")
    print(f"平均每次迭代时间: {avg_iter_time:.4f} s")
    print(f"最终迭代次数: {global_iter}")
    print(f"最佳验证损失: {min_val_loss:.6e}")

    # 保存统计数据
    metrics_df = pd.DataFrame({
        'total_time_sec': [total_duration],
        'avg_iter_time_sec': [avg_iter_time],
        'total_iterations': [global_iter],
        'best_val_loss': [min_val_loss]
    })
    metrics_df.to_csv(METRICS_SAVE_PATH, index=False)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(logs['iter'], logs['train_loss'], label='Train Loss', alpha=0.5)
    plt.yscale('log')
    plt.title('Vanilla DeepONet Training Loss (Weighted MSE)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()