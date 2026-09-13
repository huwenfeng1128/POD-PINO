# -*- coding: utf-8 -*-
import os
import glob
import time
import math
import random
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split


# =========================================================
# Configuration
# =========================================================
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\train_result_physsoft'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'
os.makedirs(RESULT_DIR, exist_ok=True)

RUN_ID = "pod_softphysics"

MODEL_SAVE_PATH = os.path.join(RESULT_DIR, f'model_{RUN_ID}.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, f'scalers_{RUN_ID}.pth')
POD_PARAMS_SAVE_PATH = os.path.join(RESULT_DIR, f'pod_params_{RUN_ID}.pth')
LOSS_DATA_SAVE_PATH = os.path.join(RESULT_DIR, f'loss_data_{RUN_ID}.csv')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, f'training_loss_{RUN_ID}.png')
POD_ANALYSIS_PLOT_PATH = os.path.join(RESULT_DIR, f'pod_singular_values_{RUN_ID}.png')
PREDICTION_PLOT_PATH = os.path.join(RESULT_DIR, f'prediction_vs_truth_{RUN_ID}.png')

# Model Hyperparameters
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
REQUESTED_NUM_POD_MODES = 100
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6

# Training Hyperparameters
ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 64
ADAM_ITERATIONS = 300000
D2_LOSS_WEIGHT = 0.5
VALIDATION_SPLIT = 0.2
SEED = 24
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 500
WARMUP_STEPS = 5000

# Data Config
TARGET_NUM_FILES = 2500

# Soft Physics Constraint Hyperparameters
PHYSICS_LOSS_WEIGHT = 1e-4         # 初始建议 1e-4，稳定后可试 1e-3
PHYSICS_D2_WEIGHT = 1.0            # 物理约束中 D2 的权重
PHYSICS_START_ITER = 5000          # 前 5000 iter 不加物理约束
PHYSICS_RAMP_ITERS = 10000         # 再用 10000 iter 线性升权重
PHYSICS_NUM_ANCHORS = 8            # 每次抽多少个 a 锚点
PHYSICS_AVG_FIRST_K_TAU = 3        # 用前 K 个最小 tau 平均近似生成元
PHYSICS_MAX_TAU_POINTS = 5         # 物理损失中最多抽多少个 tau 点
SMOOTHNESS_LOSS_WEIGHT = 1e-8      # 很小的平滑项，稳定用

# PDE Solver Settings
PHYSICS_TIME_STEPS_PER_INTERVAL = 8    # 每个 tau 段内部子步数，越大越稳但越慢
PHYSICS_CLAMP_D2_MIN = 1e-8            # 防止负扩散导致数值炸掉
PHYSICS_USE_NEUMANN_BC = True          # 边界条件：True=近似 Neumann

# Computation Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# =========================================================
# Helper Functions
# =========================================================
def load_unified_data(data_dir, target_num_files):
    print(f"从 {data_dir} 加载统一数据...")
    all_available_files = glob.glob(os.path.join(data_dir, "data_*.csv"))
    if not all_available_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件。")

    selected_files = random.sample(all_available_files, min(len(all_available_files), target_num_files))
    print(f"从 {len(all_available_files)} 个文件中选择了 {len(selected_files)} 个文件进行加载。")

    branch_inputs_list = []
    y_snapshots_list = []
    unified_a_grid, unified_tau_grid, snapshot_field_dim = None, None, None

    for f in tqdm(selected_files, desc="加载快照"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip').dropna(subset=['D1_fp', 'D2_fp'])
            if df.empty:
                continue

            if unified_a_grid is None:
                unified_a_grid = sorted(df['A'].unique())
                unified_tau_grid = sorted(df['tau'].unique())
                snapshot_field_dim = len(unified_a_grid) * len(unified_tau_grid) * 2
                print(f"Grid: A={len(unified_a_grid)}, tau={len(unified_tau_grid)}. Snapshot Dim: {snapshot_field_dim}")

            params = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values
            branch_inputs_list.append(params)

            df_sorted = df.sort_values(by=['tau', 'A'])
            final_snapshot = np.concatenate([df_sorted['D1_fp'].values, df_sorted['D2_fp'].values])
            y_snapshots_list.append(final_snapshot)

        except Exception as e:
            print(f"处理文件 {os.path.basename(f)} 时发生错误: {e}")

    if not branch_inputs_list:
        raise ValueError("未能成功加载任何快照。")

    branch_inputs_np = np.array(branch_inputs_list)
    y_snapshots_np = np.array(y_snapshots_list)

    return branch_inputs_np, y_snapshots_np, np.array(unified_a_grid), np.array(unified_tau_grid)


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def pod(y_data_scaled, requested_num_modes):
    print("在归一化快照矩阵上执行POD...")
    y_mean_pod_scaled = np.mean(y_data_scaled, axis=0)
    U, S, Vt = np.linalg.svd(y_data_scaled - y_mean_pod_scaled, full_matrices=False)
    actual_num_modes = min(requested_num_modes, Vt.shape[0])
    pod_basis = Vt.T[:, :actual_num_modes]
    print(f"POD: 原始场维度 {y_data_scaled.shape[1]}, 提取了 {actual_num_modes} 个模式。")
    return y_mean_pod_scaled, pod_basis, S, actual_num_modes


def plot_pod_analysis(S, save_path):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(S) + 1), S, 'o-')
    plt.yscale('log')
    plt.title('Singular Value Decay')
    plt.grid(True, which='both')

    plt.subplot(1, 2, 2)
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    plt.plot(range(1, len(cumulative_energy) + 1), cumulative_energy, 'o-')
    plt.ylim([0, 1.05])
    plt.title('Cumulative Energy')
    plt.grid(True)
    plt.axhline(y=0.999, color='g', linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Model Definitions
# =========================================================
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


# =========================================================
# Loss Functions
# =========================================================
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


# =========================================================
# Physics Constraint Utilities
# =========================================================
def split_fields(y, num_tau, num_a):
    field_len = y.shape[1] // 2
    d1 = y[:, :field_len].reshape(-1, num_tau, num_a)
    d2 = y[:, field_len:].reshape(-1, num_tau, num_a)
    return d1, d2


def finite_diff_first(u, da):
    du = torch.zeros_like(u)
    du[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * da)
    du[..., 0] = (u[..., 1] - u[..., 0]) / da
    du[..., -1] = (u[..., -1] - u[..., -2]) / da
    return du


def finite_diff_second(u, da):
    d2u = torch.zeros_like(u)
    d2u[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / (da ** 2)
    d2u[..., 0] = (u[..., 2] - 2.0 * u[..., 1] + u[..., 0]) / (da ** 2)
    d2u[..., -1] = (u[..., -1] - 2.0 * u[..., -2] + u[..., -3]) / (da ** 2)
    return d2u


def apply_boundary_condition(P):
    if PHYSICS_USE_NEUMANN_BC:
        P = P.clone()
        P[..., 0] = P[..., 1]
        P[..., -1] = P[..., -2]
    return P


def adjoint_rhs(P, d1_gen, d2_gen, da):
    """
    Adjoint Fokker-Planck operator (soft-constraint version):
        dP/dt = D1(a) * dP/da + D2(a) * d2P/da2

    If your theoretical definition differs (e.g. factor 1/2), modify here.
    """
    P = apply_boundary_condition(P)
    dP_da = finite_diff_first(P, da)
    d2P_da2 = finite_diff_second(P, da)
    rhs = d1_gen * dP_da + d2_gen * d2P_da2
    return rhs


def solve_adjoint_pde_soft(d1_gen, d2_gen, a_grid, tau_grid, anchor_indices, tau_indices):
    """
    d1_gen, d2_gen: (B, A)
    Returns:
        recon_d1: (B, len(tau_indices), len(anchor_indices))
        recon_d2: (B, len(tau_indices), len(anchor_indices))
    """
    device = d1_gen.device
    dtype = d1_gen.dtype

    B, A = d1_gen.shape
    a_grid_t = torch.tensor(a_grid, dtype=dtype, device=device)
    tau_grid_np = np.asarray(tau_grid, dtype=np.float64)

    da = float(a_grid[1] - a_grid[0])

    # 保证扩散非负，减少数值不稳定
    d2_gen = torch.clamp(d2_gen, min=PHYSICS_CLAMP_D2_MIN)

    recon_d1 = torch.zeros((B, len(tau_indices), len(anchor_indices)), dtype=dtype, device=device)
    recon_d2 = torch.zeros((B, len(tau_indices), len(anchor_indices)), dtype=dtype, device=device)

    for j, a_idx in enumerate(anchor_indices):
        a0 = a_grid_t[a_idx]

        # 初值: P1(a',0)=(a'-a), P2(a',0)=(a'-a)^2
        base = a_grid_t[None, :] - a0
        P1 = base.repeat(B, 1)
        P2 = (base ** 2).repeat(B, 1)

        prev_t = 0.0

        for m, tau_idx in enumerate(tau_indices):
            target_t = float(tau_grid_np[tau_idx])
            dt_total = target_t - prev_t

            if dt_total < 0:
                raise ValueError("tau_grid must be nondecreasing.")

            # 每个 tau 段再切分成更细子步
            n_substeps = max(1, PHYSICS_TIME_STEPS_PER_INTERVAL)
            dt = dt_total / n_substeps if dt_total > 0 else 0.0

            for _ in range(n_substeps):
                if dt == 0.0:
                    continue

                rhs1 = adjoint_rhs(P1, d1_gen, d2_gen, da)
                rhs2 = adjoint_rhs(P2, d1_gen, d2_gen, da)

                P1 = P1 + dt * rhs1
                P2 = P2 + dt * rhs2

                P1 = apply_boundary_condition(P1)
                P2 = apply_boundary_condition(P2)

            tau_val = max(target_t, 1e-10)
            recon_d1[:, m, j] = P1[:, a_idx] / tau_val
            recon_d2[:, m, j] = P2[:, a_idx] / (2.0 * tau_val)

            prev_t = target_t

    return recon_d1, recon_d2


def physics_consistency_loss(
    y_pred_scaled,
    num_tau,
    num_a,
    a_grid,
    tau_grid,
    physics_num_anchors=8,
    physics_avg_first_k_tau=3,
    physics_max_tau_points=5,
    physics_d2_weight=1.0,
):
    """
    Soft physics loss:
    1) 从预测场前几个最小 tau 近似生成元
    2) 用伴随方程重构有限时间 KM
    3) 和网络预测做一致性 MSE
    """
    device = y_pred_scaled.device
    d1_pred, d2_pred = split_fields(y_pred_scaled, num_tau, num_a)  # (B,T,A)

    # 用最小若干 tau 平均，近似生成元
    k = min(physics_avg_first_k_tau, num_tau)
    d1_gen = d1_pred[:, :k, :].mean(dim=1)   # (B,A)
    d2_gen = d2_pred[:, :k, :].mean(dim=1)   # (B,A)

    # anchor points: 避开边界
    start_idx = 1
    end_idx = num_a - 1
    candidate_anchor_indices = np.arange(start_idx, end_idx)

    if len(candidate_anchor_indices) == 0:
        raise ValueError("Not enough A grid points for physics anchors.")

    if len(candidate_anchor_indices) <= physics_num_anchors:
        anchor_indices = candidate_anchor_indices
    else:
        anchor_indices = np.linspace(start_idx, end_idx - 1, physics_num_anchors, dtype=int)

    # tau points: 跳过第 0 个 tau，避免 1/tau 奇异
    valid_tau_indices = np.arange(1, num_tau)
    if len(valid_tau_indices) == 0:
        raise ValueError("Need at least two tau points for physics consistency loss.")

    if len(valid_tau_indices) <= physics_max_tau_points:
        tau_indices = valid_tau_indices
    else:
        tau_indices = np.linspace(1, num_tau - 1, physics_max_tau_points, dtype=int)

    recon_d1, recon_d2 = solve_adjoint_pde_soft(
        d1_gen, d2_gen, a_grid, tau_grid, anchor_indices, tau_indices
    )

    pred_d1_sub = d1_pred[:, tau_indices][:, :, anchor_indices]
    pred_d2_sub = d2_pred[:, tau_indices][:, :, anchor_indices]

    loss_d1 = torch.mean((pred_d1_sub - recon_d1) ** 2)
    loss_d2 = torch.mean((pred_d2_sub - recon_d2) ** 2)
    total_phys = loss_d1 + physics_d2_weight * loss_d2

    return total_phys, loss_d1.detach(), loss_d2.detach()


def smoothness_loss(y_pred_scaled, num_tau, num_a):
    d1_pred, d2_pred = split_fields(y_pred_scaled, num_tau, num_a)
    loss_d1 = torch.mean((d1_pred[:, :, 1:] - d1_pred[:, :, :-1]) ** 2)
    loss_d2 = torch.mean((d2_pred[:, :, 1:] - d2_pred[:, :, :-1]) ** 2)
    return loss_d1 + loss_d2


def get_current_physics_weight(global_iter):
    if global_iter < PHYSICS_START_ITER:
        return 0.0
    ramp = min(1.0, (global_iter - PHYSICS_START_ITER) / max(1, PHYSICS_RAMP_ITERS))
    return PHYSICS_LOSS_WEIGHT * ramp


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("=== 加载数据 ===")
    branch_inputs_np, y_snapshots_np, unified_a_grid, unified_tau_grid = load_unified_data(DATA_DIR, TARGET_NUM_FILES)

    if len(unified_a_grid) < 4:
        raise ValueError("A grid 点数太少，无法稳定做有限差分。")
    if len(unified_tau_grid) < 2:
        raise ValueError("tau grid 至少需要 2 个点。")

    branch_train_np, branch_val_np, y_train_np, y_val_np = train_test_split(
        branch_inputs_np, y_snapshots_np, test_size=VALIDATION_SPLIT, random_state=SEED
    )

    branch_train_scaled, branch_mean, branch_std = manual_scaler(branch_train_np)
    branch_val_scaled = manual_scaler(branch_val_np, branch_mean, branch_std)

    y_train_scaled, y_mean_scaler, y_std_scaler = manual_scaler(y_train_np)
    y_val_scaled = manual_scaler(y_val_np, y_mean_scaler, y_std_scaler)

    y_mean_pod_scaled, pod_basis, S, actual_num_modes = pod(y_train_scaled, REQUESTED_NUM_POD_MODES)
    plot_pod_analysis(S, POD_ANALYSIS_PLOT_PATH)

    torch.save({
        'branch_mean': branch_mean,
        'branch_std': branch_std,
        'y_mean_scaler': y_mean_scaler,
        'y_std_scaler': y_std_scaler
    }, SCALER_SAVE_PATH)

    torch.save({
        'y_mean_pod_scaled': y_mean_pod_scaled,
        'pod_basis': pod_basis,
        'num_pod_modes': actual_num_modes,
        'unified_a_grid': unified_a_grid,
        'unified_tau_grid': unified_tau_grid
    }, POD_PARAMS_SAVE_PATH)

    train_dataset = TensorDataset(
        torch.from_numpy(branch_train_scaled).to(DEVICE),
        torch.from_numpy(y_train_scaled).to(DEVICE)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(branch_val_scaled).to(DEVICE),
        torch.from_numpy(y_val_scaled).to(DEVICE)
    )

    train_loader = DataLoader(train_dataset, batch_size=ADAM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = PODDeepONet(
        BRANCH_INPUT_DIM,
        HIDDEN_UNITS,
        NUM_HIDDEN_LAYERS,
        actual_num_modes,
        pod_basis,
        y_mean_pod_scaled,
        DROPOUT_RATE
    ).to(DEVICE)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, ADAM_ITERATIONS)
    data_loss_fn = WeightedMSELoss(d2_weight=D2_LOSS_WEIGHT)

    logs = {
        'iter': [],
        'train_total_loss': [],
        'train_data_loss': [],
        'train_phys_loss': [],
        'train_smooth_loss': [],
        'train_lambda_phys': [],
        'lr': [],

        'val_iter': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_d1': [],
        'val_d2': [],
        'val_phys_loss': [],
        'val_phys_d1': [],
        'val_phys_d2': [],
        'val_smooth_loss': [],
        'val_lambda_phys': [],
    }

    min_val_loss = float('inf')
    early_stop_counter = 0

    print("\n--- 开始训练: Data Loss + Soft Physics Loss ---")
    pbar = tqdm(total=ADAM_ITERATIONS, desc="训练进度")
    global_iter = 0
    done = False

    num_a = len(unified_a_grid)
    num_tau = len(unified_tau_grid)

    while not done:
        for batch_branch, batch_y_scaled in train_loader:
            if global_iter >= ADAM_ITERATIONS:
                done = True
                break

            model.train()
            optimizer.zero_grad()

            y_pred_scaled = model(batch_branch)

            data_loss, _, _ = data_loss_fn(y_pred_scaled, batch_y_scaled)

            lambda_phys_now = get_current_physics_weight(global_iter)

            phys_loss, _, _ = physics_consistency_loss(
                y_pred_scaled=y_pred_scaled,
                num_tau=num_tau,
                num_a=num_a,
                a_grid=unified_a_grid,
                tau_grid=unified_tau_grid,
                physics_num_anchors=PHYSICS_NUM_ANCHORS,
                physics_avg_first_k_tau=PHYSICS_AVG_FIRST_K_TAU,
                physics_max_tau_points=PHYSICS_MAX_TAU_POINTS,
                physics_d2_weight=PHYSICS_D2_WEIGHT,
            )

            smooth_loss = smoothness_loss(y_pred_scaled, num_tau, num_a)

            total_loss = data_loss + lambda_phys_now * phys_loss + SMOOTHNESS_LOSS_WEIGHT * smooth_loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()

            logs['iter'].append(global_iter)
            logs['train_total_loss'].append(total_loss.item())
            logs['train_data_loss'].append(data_loss.item())
            logs['train_phys_loss'].append(phys_loss.item())
            logs['train_smooth_loss'].append(smooth_loss.item())
            logs['train_lambda_phys'].append(lambda_phys_now)
            logs['lr'].append(optimizer.param_groups[0]['lr'])

            pbar.update(1)
            pbar.set_postfix({
                'total': f'{total_loss.item():.3e}',
                'data': f'{data_loss.item():.3e}',
                'phys': f'{phys_loss.item():.3e}',
                'lam_p': f'{lambda_phys_now:.1e}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            if global_iter % VALIDATION_FREQUENCY == 0:
                model.eval()
                with torch.no_grad():
                    val_branch_all, val_y_all_scaled = next(iter(val_loader))
                    val_y_pred_scaled = model(val_branch_all)

                    val_data_loss, val_d1_loss, val_d2_loss = data_loss_fn(val_y_pred_scaled, val_y_all_scaled)
                    val_lambda_phys_now = get_current_physics_weight(global_iter)

                    val_phys_loss, val_phys_d1, val_phys_d2 = physics_consistency_loss(
                        y_pred_scaled=val_y_pred_scaled,
                        num_tau=num_tau,
                        num_a=num_a,
                        a_grid=unified_a_grid,
                        tau_grid=unified_tau_grid,
                        physics_num_anchors=PHYSICS_NUM_ANCHORS,
                        physics_avg_first_k_tau=PHYSICS_AVG_FIRST_K_TAU,
                        physics_max_tau_points=PHYSICS_MAX_TAU_POINTS,
                        physics_d2_weight=PHYSICS_D2_WEIGHT,
                    )

                    val_smooth_loss = smoothness_loss(val_y_pred_scaled, num_tau, num_a)

                    val_total_loss = (
                        val_data_loss
                        + val_lambda_phys_now * val_phys_loss
                        + SMOOTHNESS_LOSS_WEIGHT * val_smooth_loss
                    )

                    logs['val_iter'].append(global_iter)
                    logs['val_total_loss'].append(val_total_loss.item())
                    logs['val_data_loss'].append(val_data_loss.item())
                    logs['val_d1'].append(val_d1_loss.item())
                    logs['val_d2'].append(val_d2_loss.item())
                    logs['val_phys_loss'].append(val_phys_loss.item())
                    logs['val_phys_d1'].append(val_phys_d1.item())
                    logs['val_phys_d2'].append(val_phys_d2.item())
                    logs['val_smooth_loss'].append(val_smooth_loss.item())
                    logs['val_lambda_phys'].append(val_lambda_phys_now)

                    print(
                        f"\nIter: {global_iter}, "
                        f"Val Total: {val_total_loss.item():.6e}, "
                        f"Val Data: {val_data_loss.item():.6e}, "
                        f"Val Phys: {val_phys_loss.item():.6e}, "
                        f"(D1: {val_d1_loss.item():.6e}, D2: {val_d2_loss.item():.6e})"
                    )

                if val_total_loss.item() < min_val_loss:
                    min_val_loss = val_total_loss.item()
                    early_stop_counter = 0
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print("  -> New best model saved.")
                else:
                    early_stop_counter += 1

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n*** Early stopping triggered after {global_iter} iterations. ***")
                    done = True
                    break

            global_iter += 1

    pbar.close()

    print("\n=== 保存 loss 记录 ===")
    loss_df = pd.DataFrame(logs)
    loss_df.to_csv(LOSS_DATA_SAVE_PATH, index=False)

    print("=== 绘制训练曲线 ===")
    plt.figure(figsize=(14, 7))
    plt.plot(logs['iter'], logs['train_total_loss'], label='Train Total Loss', alpha=0.6)
    plt.plot(logs['iter'], logs['train_data_loss'], label='Train Data Loss', alpha=0.5)
    plt.plot(logs['iter'], logs['train_phys_loss'], label='Train Physics Loss', alpha=0.5)

    plt.plot(logs['val_iter'], logs['val_total_loss'], label='Val Total Loss', marker='.')
    plt.plot(logs['val_iter'], logs['val_data_loss'], label='Val Data Loss', linestyle=':')
    plt.plot(logs['val_iter'], logs['val_phys_loss'], label='Val Physics Loss', linestyle=':')

    best_iter = logs['val_iter'][int(np.argmin(logs['val_total_loss']))]
    plt.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', label='Warmup End')
    plt.axvline(x=PHYSICS_START_ITER, color='purple', linestyle='--', label='Physics Start')
    plt.scatter([best_iter], [min_val_loss], color='green', s=100, zorder=5, label=f'Best @ Iter {best_iter}')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss with Soft Physics Constraint')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.show()

    print("\n--- 可视化验证 ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    num_a = len(unified_a_grid)
    num_tau = len(unified_tau_grid)
    vis_count = min(4, len(branch_val_scaled))
    indices = np.random.choice(len(branch_val_scaled), vis_count, replace=False)

    fig, axes = plt.subplots(vis_count, 4, figsize=(20, 5 * vis_count))
    if vis_count == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle('Prediction vs. Ground Truth Comparison', fontsize=16)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            branch_input_scaled = torch.from_numpy(branch_val_scaled[idx]).unsqueeze(0).to(DEVICE)
            y_pred_np = model.predict(branch_input_scaled, y_mean_scaler, y_std_scaler).cpu().numpy().flatten()
            y_true_np = y_val_np[idx]

            field_len_half = len(y_true_np) // 2
            y_pred_d1 = y_pred_np[:field_len_half].reshape(num_tau, num_a)
            y_pred_d2 = y_pred_np[field_len_half:].reshape(num_tau, num_a)
            y_true_d1 = y_true_np[:field_len_half].reshape(num_tau, num_a)
            y_true_d2 = y_true_np[field_len_half:].reshape(num_tau, num_a)

            vmax_d1 = max(y_true_d1.max(), y_pred_d1.max())
            vmin_d1 = min(y_true_d1.min(), y_pred_d1.min())
            vmax_d2 = max(y_true_d2.max(), y_pred_d2.max())
            vmin_d2 = min(y_true_d2.min(), y_pred_d2.min())

            def plot_field(ax, data, title, vmin, vmax):
                im = ax.imshow(
                    data,
                    aspect='auto',
                    origin='lower',
                    extent=[unified_a_grid.min(), unified_a_grid.max(),
                            unified_tau_grid.min(), unified_tau_grid.max()],
                    vmin=vmin,
                    vmax=vmax
                )
                ax.set_title(title)
                ax.set_xlabel('A')
                ax.set_ylabel('tau')
                fig.colorbar(im, ax=ax)

            plot_field(axes[i, 0], y_true_d1, f'Sample {idx} - D1 True', vmin_d1, vmax_d1)
            plot_field(axes[i, 1], y_pred_d1, f'Sample {idx} - D1 Pred', vmin_d1, vmax_d1)
            plot_field(axes[i, 2], y_true_d2, f'Sample {idx} - D2 True', vmin_d2, vmax_d2)
            plot_field(axes[i, 3], y_pred_d2, f'Sample {idx} - D2 Pred', vmin_d2, vmax_d2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.show()

    print("\n=== 训练完成 ===")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    print(f"POD params saved to: {POD_PARAMS_SAVE_PATH}")
    print(f"Loss data saved to: {LOSS_DATA_SAVE_PATH}")