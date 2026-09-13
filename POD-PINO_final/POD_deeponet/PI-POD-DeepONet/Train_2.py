import os
import glob
import math
import random
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split


# =========================================================
# Configuration
# =========================================================
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\train_result_v6_pplus_pde'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA'
os.makedirs(RESULT_DIR, exist_ok=True)

RUN_ID = 'pod_v6_pplus_pde'

MODEL_SAVE_PATH = os.path.join(RESULT_DIR, f'model_{RUN_ID}.pth')
SCALER_SAVE_PATH = os.path.join(RESULT_DIR, f'scalers_{RUN_ID}.pth')
POD_PARAMS_SAVE_PATH = os.path.join(RESULT_DIR, f'pod_params_{RUN_ID}.pth')
LOSS_DATA_SAVE_PATH = os.path.join(RESULT_DIR, f'loss_data_{RUN_ID}.csv')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, f'training_loss_{RUN_ID}.png')
POD_ANALYSIS_PLOT_PATH = os.path.join(RESULT_DIR, f'pod_singular_values_{RUN_ID}.png')
PREDICTION_PLOT_PATH = os.path.join(RESULT_DIR, f'prediction_vs_truth_{RUN_ID}.png')
GENERATOR_PLOT_PATH = os.path.join(RESULT_DIR, f'generator_comparison_{RUN_ID}.png')
VAL_METRICS_SAVE_PATH = os.path.join(RESULT_DIR, f'best_val_metrics_{RUN_ID}.csv')

# -----------------------------
# Model Hyperparameters
# -----------------------------
BRANCH_INPUT_DIM = 3
HIDDEN_UNITS = 128
NUM_HIDDEN_LAYERS = 4
REQUESTED_NUM_POD_MODES = 100
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-6

# -----------------------------
# Training Hyperparameters
# -----------------------------
ADAM_LR = 3e-4
ADAM_BATCH_SIZE = 32
ADAM_ITERATIONS = 50000
D2_LOSS_WEIGHT = 0.5
VALIDATION_SPLIT = 0.2
SEED = 24
VALIDATION_FREQUENCY = 200
EARLY_STOPPING_PATIENCE = 500
WARMUP_STEPS = 5000
TARGET_NUM_FILES = 2500

# -----------------------------
# P+ PDE Physics Loss
# -----------------------------
PPLUS_PDE_LOSS_WEIGHT = 1e-3
PPLUS_PDE_START_ITER = 2000
PPLUS_PDE_RAMP_ITERS = 10000
PPLUS_PDE_AVG_FIRST_K_TAU = 3
PPLUS_PDE_N2_WEIGHT = 1.0
PPLUS_D2_MIN = 1e-12
SMOOTHNESS_LOSS_WEIGHT = 1e-8

# -----------------------------
# Computation Settings
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64
NUM_WORKERS = 0
PIN_MEMORY = False


# =========================================================
# Helpers
# =========================================================
@dataclass
class SampleRecord:
    csv_path: str
    npz_path: str
    branch_params: np.ndarray
    y_snapshot: np.ndarray


def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    A_safe = np.maximum(A, 1e-9)
    return nu * A - (kappa / 8.0) * A ** 3 + d_diffusion / A_safe


def theoretical_D2_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    return np.full_like(A, max(0.0, d_diffusion))


def manual_scaler(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-10] = 1.0
        return (data - mean) / std, mean, std
    return (data - mean) / std


def inverse_scale_output_torch(y_scaled, y_mean_scaler, y_std_scaler, device, dtype):
    y_mean_t = torch.tensor(y_mean_scaler, dtype=dtype, device=device)
    y_std_t = torch.tensor(y_std_scaler, dtype=dtype, device=device)
    return y_scaled * y_std_t + y_mean_t


def pod(y_data_scaled, requested_num_modes):
    print('Running POD on normalized snapshot matrix...')
    y_mean_pod_scaled = np.mean(y_data_scaled, axis=0)
    U, S, Vt = np.linalg.svd(y_data_scaled - y_mean_pod_scaled, full_matrices=False)
    actual_num_modes = min(requested_num_modes, Vt.shape[0])
    pod_basis = Vt.T[:, :actual_num_modes]
    print(f'POD: field dim={y_data_scaled.shape[1]}, modes={actual_num_modes}')
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
    plt.savefig(save_path, dpi=150)
    plt.close()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_current_pplus_weight(global_iter):
    if global_iter < PPLUS_PDE_START_ITER:
        return 0.0
    ramp = min(1.0, (global_iter - PPLUS_PDE_START_ITER) / max(1, PPLUS_PDE_RAMP_ITERS))
    return PPLUS_PDE_LOSS_WEIGHT * ramp


def build_param_id(nu, kappa, d_diffusion):
    return (
        f'nu_{nu:.3f}_kappa_{kappa:.3f}_d_{d_diffusion:.3f}'
        .replace('.', '_')
        .replace('-', 'neg')
    )


def load_unified_data_with_pplus(data_dir, target_num_files):
    print(f'Loading paired CSV/NPZ data from {data_dir} ...')
    all_csv_files = sorted(glob.glob(os.path.join(data_dir, 'data_*.csv')))
    if not all_csv_files:
        raise FileNotFoundError(f'No CSV files found in {data_dir}')

    paired_records = []
    unified_a_grid = None
    unified_tau_grid = None
    pplus_a_targets = None
    pplus_a_grid = None
    pplus_tau_grid = None

    random.shuffle(all_csv_files)
    if target_num_files is not None:
        all_csv_files = all_csv_files[:min(len(all_csv_files), target_num_files)]

    for csv_path in tqdm(all_csv_files, desc='Scanning files'):
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip').dropna(subset=['D1_adj', 'D2_adj'])
            if df.empty:
                continue

            params_row = df[['nu', 'kappa', 'd_diffusion']].iloc[0].values.astype(np.float64)
            param_id = build_param_id(params_row[0], params_row[1], params_row[2])
            npz_path = os.path.join(data_dir, f'pplus_{param_id}.npz')
            if not os.path.exists(npz_path):
                continue

            if unified_a_grid is None:
                unified_a_grid = np.sort(df['A'].unique()).astype(np.float64)
                unified_tau_grid = np.sort(df['tau'].unique()).astype(np.float64)
            else:
                current_a = np.sort(df['A'].unique()).astype(np.float64)
                current_tau = np.sort(df['tau'].unique()).astype(np.float64)
                if len(current_a) != len(unified_a_grid) or not np.allclose(current_a, unified_a_grid):
                    continue
                if len(current_tau) != len(unified_tau_grid) or not np.allclose(current_tau, unified_tau_grid):
                    continue

            with np.load(npz_path) as npz_data:
                cur_tau = np.asarray(npz_data['tau_values'], dtype=np.float64)
                cur_targets = np.asarray(npz_data['A_targets'], dtype=np.float64)
                cur_grid = np.asarray(npz_data['A_grid'], dtype=np.float64)
                if pplus_tau_grid is None:
                    pplus_tau_grid = cur_tau
                    pplus_a_targets = cur_targets
                    pplus_a_grid = cur_grid
                else:
                    if (len(cur_tau) != len(pplus_tau_grid)) or (not np.allclose(cur_tau, pplus_tau_grid)):
                        continue
                    if (len(cur_targets) != len(pplus_a_targets)) or (not np.allclose(cur_targets, pplus_a_targets)):
                        continue
                    if (len(cur_grid) != len(pplus_a_grid)) or (not np.allclose(cur_grid, pplus_a_grid)):
                        continue

            df_sorted = df.sort_values(by=['tau', 'A'])
            y_snapshot = np.concatenate([df_sorted['D1_adj'].values, df_sorted['D2_adj'].values]).astype(np.float64)

            paired_records.append(SampleRecord(
                csv_path=csv_path,
                npz_path=npz_path,
                branch_params=params_row,
                y_snapshot=y_snapshot,
            ))

        except Exception:
            traceback.print_exc()
            continue

    if not paired_records:
        raise ValueError('No valid paired CSV/NPZ samples could be loaded.')

    branch_inputs_np = np.stack([r.branch_params for r in paired_records], axis=0)
    y_snapshots_np = np.stack([r.y_snapshot for r in paired_records], axis=0)

    print(f'Loaded {len(paired_records)} paired samples.')
    print(f'A grid: {len(unified_a_grid)}, tau grid: {len(unified_tau_grid)}, Pplus grid: {len(pplus_a_grid)}')

    return (
        paired_records,
        branch_inputs_np,
        y_snapshots_np,
        unified_a_grid,
        unified_tau_grid,
        pplus_a_targets,
        pplus_a_grid,
        pplus_tau_grid,
    )


# =========================================================
# Dataset
# =========================================================
class PairedAdjointDataset(Dataset):
    def __init__(self, records, branch_scaled, y_scaled):
        self.records = records
        self.branch_scaled = branch_scaled.astype(np.float64)
        self.y_scaled = y_scaled.astype(np.float64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with np.load(rec.npz_path) as npz_data:
            pplus_n1 = np.asarray(npz_data['Pplus_n1'], dtype=np.float64)
            pplus_n2 = np.asarray(npz_data['Pplus_n2'], dtype=np.float64)

        return {
            'branch': torch.from_numpy(self.branch_scaled[idx]).to(DTYPE),
            'y': torch.from_numpy(self.y_scaled[idx]).to(DTYPE),
            'pplus_n1': torch.from_numpy(pplus_n1).to(DTYPE),
            'pplus_n2': torch.from_numpy(pplus_n2).to(DTYPE),
            'params': torch.from_numpy(self.records[idx].branch_params.astype(np.float64)).to(DTYPE),
        }


def collate_dict(batch_list):
    out = {}
    for key in batch_list[0].keys():
        out[key] = torch.stack([item[key] for item in batch_list], dim=0)
    return out


# =========================================================
# Model
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
                nn.Linear(hidden_units, hidden_units, dtype=DTYPE),
            ])
        layers.extend([
            nn.GELU(),
            nn.LayerNorm(hidden_units, dtype=DTYPE),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, output_dim, dtype=DTYPE),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PODDeepONet(nn.Module):
    def __init__(self, branch_input_dim, hidden_units, num_hidden_layers,
                 num_pod_modes, pod_basis, y_mean_pod_scaled, dropout_rate):
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
# Losses
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
        total_loss = (1.0 - self.d2_weight) * loss_d1 + self.d2_weight * loss_d2
        return total_loss, loss_d1.detach(), loss_d2.detach()


def split_fields(y, num_tau, num_a):
    field_len = y.shape[1] // 2
    d1 = y[:, :field_len].reshape(-1, num_tau, num_a)
    d2 = y[:, field_len:].reshape(-1, num_tau, num_a)
    return d1, d2


def smoothness_loss(y_pred_scaled, num_tau, num_a):
    d1_pred, d2_pred = split_fields(y_pred_scaled, num_tau, num_a)
    loss_d1 = torch.mean((d1_pred[:, :, 1:] - d1_pred[:, :, :-1]) ** 2)
    loss_d2 = torch.mean((d2_pred[:, :, 1:] - d2_pred[:, :, :-1]) ** 2)
    return loss_d1 + loss_d2


def interpolate_linear_batch_1d(x_src, y_src, x_tgt):
    # x_src: (S,), y_src: (B, S), x_tgt: (T,)
    x_src = x_src.to(y_src.device, y_src.dtype)
    x_tgt = x_tgt.to(y_src.device, y_src.dtype)

    idx = torch.searchsorted(x_src, x_tgt)
    idx = torch.clamp(idx, 1, len(x_src) - 1)
    x0 = x_src[idx - 1]
    x1 = x_src[idx]

    y0 = y_src[:, idx - 1]
    y1 = y_src[:, idx]

    denom = torch.clamp(x1 - x0, min=1e-12)
    w = (x_tgt - x0) / denom
    return y0 + (y1 - y0) * w.unsqueeze(0)


def pplus_pde_loss(
    y_pred_physical,
    pplus_n1,
    pplus_n2,
    num_tau,
    num_a,
    a_targets,
    pplus_a_grid,
    tau_grid,
    avg_first_k_tau=3,
    n2_weight=1.0,
):
    # y_pred_physical: (B, 2*T*A)
    # pplus_n{1,2}: (B, T, A_target, G)
    d1_pred, d2_pred = split_fields(y_pred_physical, num_tau, num_a)

    k = min(avg_first_k_tau, num_tau)
    d1_gen = d1_pred[:, :k, :].mean(dim=1)
    d2_gen = d2_pred[:, :k, :].mean(dim=1)
    d2_gen = F.softplus(d2_gen) + PPLUS_D2_MIN

    a_targets_t = torch.tensor(a_targets, dtype=y_pred_physical.dtype, device=y_pred_physical.device)
    pplus_a_grid_t = torch.tensor(pplus_a_grid, dtype=y_pred_physical.dtype, device=y_pred_physical.device)
    tau_grid_t = torch.tensor(tau_grid, dtype=y_pred_physical.dtype, device=y_pred_physical.device)

    # Residual only on the region where the network predicts D1/D2 (within A_targets range)
    domain_mask = (pplus_a_grid_t >= a_targets_t[0]) & (pplus_a_grid_t <= a_targets_t[-1])
    domain_idx = torch.nonzero(domain_mask, as_tuple=False).flatten()
    if len(domain_idx) < 3:
        raise ValueError('Insufficient overlap between Pplus A_grid and prediction A_targets.')

    domain_idx = domain_idx[1:-1]
    pplus_a_used = pplus_a_grid_t[domain_idx]

    d1_on_grid = interpolate_linear_batch_1d(a_targets_t, d1_gen, pplus_a_used)
    d2_on_grid = interpolate_linear_batch_1d(a_targets_t, d2_gen, pplus_a_used)

    dt = torch.clamp(tau_grid_t[2:] - tau_grid_t[:-2], min=1e-12)
    da = torch.clamp(pplus_a_grid_t[2:] - pplus_a_grid_t[:-2], min=1e-12)
    da2 = torch.clamp((pplus_a_grid_t[1] - pplus_a_grid_t[0]) ** 2, min=1e-12)

    def compute_residual(P):
        finite_all = torch.isfinite(P)
        P_safe = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        dP_dt = (P_safe[:, 2:, :, :] - P_safe[:, :-2, :, :]) / dt.view(1, -1, 1, 1)
        dP_da = (P_safe[:, :, :, 2:] - P_safe[:, :, :, :-2]) / da.view(1, 1, 1, -1)
        d2P_da2 = (P_safe[:, :, :, 2:] - 2.0 * P_safe[:, :, :, 1:-1] + P_safe[:, :, :, :-2]) / da2

        # align time interior and selected spatial domain
        dP_dt = dP_dt[:, :, :, domain_idx]
        dP_da = dP_da[:, 1:-1, :, :][:, :, :, domain_idx - 1]
        d2P_da2 = d2P_da2[:, 1:-1, :, :][:, :, :, domain_idx - 1]

        valid_t = finite_all[:, 2:, :, :] & finite_all[:, :-2, :, :]
        valid_a_first = finite_all[:, :, :, 2:] & finite_all[:, :, :, :-2]
        valid_a_second = finite_all[:, :, :, 2:] & finite_all[:, :, :, 1:-1] & finite_all[:, :, :, :-2]
        valid_t = valid_t[:, :, :, domain_idx]
        valid_a_first = valid_a_first[:, 1:-1, :, :][:, :, :, domain_idx - 1]
        valid_a_second = valid_a_second[:, 1:-1, :, :][:, :, :, domain_idx - 1]
        valid = valid_t & valid_a_first & valid_a_second

        rhs = d1_on_grid[:, None, None, :] * dP_da + d2_on_grid[:, None, None, :] * d2P_da2
        residual = dP_dt - rhs

        if valid.any():
            loss = (residual[valid] ** 2).mean()
        else:
            loss = torch.zeros((), dtype=y_pred_physical.dtype, device=y_pred_physical.device)
        return loss

    loss_n1 = compute_residual(pplus_n1)
    loss_n2 = compute_residual(pplus_n2)
    total = loss_n1 + n2_weight * loss_n2
    return total, loss_n1.detach(), loss_n2.detach()


def pplus_diagonal_consistency_metric(y_pred_physical, pplus_n1, pplus_n2, num_tau, num_a, a_targets, pplus_a_grid):
    d1_pred, d2_pred = split_fields(y_pred_physical, num_tau, num_a)

    a_targets_t = torch.tensor(a_targets, dtype=y_pred_physical.dtype, device=y_pred_physical.device)
    pplus_a_grid_t = torch.tensor(pplus_a_grid, dtype=y_pred_physical.dtype, device=y_pred_physical.device)

    diag_indices = []
    for a in a_targets_t.detach().cpu().numpy():
        idx = int(np.argmin(np.abs(pplus_a_grid - float(a))))
        diag_indices.append(idx)
    diag_indices_t = torch.tensor(diag_indices, dtype=torch.long, device=y_pred_physical.device)

    pplus1_diag = torch.gather(
        pplus_n1,
        dim=3,
        index=diag_indices_t.view(1, 1, -1, 1).expand(pplus_n1.shape[0], pplus_n1.shape[1], pplus_n1.shape[2], 1)
    ).squeeze(-1)
    pplus2_diag = torch.gather(
        pplus_n2,
        dim=3,
        index=diag_indices_t.view(1, 1, -1, 1).expand(pplus_n2.shape[0], pplus_n2.shape[1], pplus_n2.shape[2], 1)
    ).squeeze(-1)

    return d1_pred, d2_pred, pplus1_diag, pplus2_diag


def pplus_diagonal_consistency_loss(y_pred_physical, pplus_n1, pplus_n2, num_tau, num_a, a_targets, pplus_a_grid, tau_grid):
    d1_pred, d2_pred, pplus1_diag, pplus2_diag = pplus_diagonal_consistency_metric(
        y_pred_physical, pplus_n1, pplus_n2, num_tau, num_a, a_targets, pplus_a_grid
    )
    tau_t = torch.tensor(tau_grid, dtype=y_pred_physical.dtype, device=y_pred_physical.device).view(1, -1, 1)
    target1 = tau_t * d1_pred
    target2 = 2.0 * tau_t * d2_pred

    valid1 = torch.isfinite(pplus1_diag) & torch.isfinite(target1)
    valid2 = torch.isfinite(pplus2_diag) & torch.isfinite(target2)

    loss1 = ((pplus1_diag[valid1] - target1[valid1]) ** 2).mean() if valid1.any() else torch.zeros((), dtype=y_pred_physical.dtype, device=y_pred_physical.device)
    loss2 = ((pplus2_diag[valid2] - target2[valid2]) ** 2).mean() if valid2.any() else torch.zeros((), dtype=y_pred_physical.dtype, device=y_pred_physical.device)
    total = loss1 + loss2
    return total, loss1.detach(), loss2.detach()


# =========================================================
# Training / Validation
# =========================================================
def run_validation(model, val_loader, data_loss_fn, y_mean_scaler, y_std_scaler,
                   num_tau, num_a, a_targets, pplus_a_grid, tau_grid, global_iter):
    model.eval()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    d1_loss_sum = 0.0
    d2_loss_sum = 0.0
    pplus_loss_sum = 0.0
    pplus_n1_sum = 0.0
    pplus_n2_sum = 0.0
    smooth_sum = 0.0
    diag_sum = 0.0
    diag_n1_sum = 0.0
    diag_n2_sum = 0.0
    sample_count = 0

    lambda_pplus = get_current_pplus_weight(global_iter)

    with torch.no_grad():
        for batch in val_loader:
            branch = batch['branch'].to(DEVICE)
            y_scaled = batch['y'].to(DEVICE)
            pplus_n1 = batch['pplus_n1'].to(DEVICE)
            pplus_n2 = batch['pplus_n2'].to(DEVICE)
            bs = branch.shape[0]

            y_pred_scaled = model(branch)
            data_loss, d1_loss, d2_loss = data_loss_fn(y_pred_scaled, y_scaled)
            smooth_loss = smoothness_loss(y_pred_scaled, num_tau, num_a)

            if lambda_pplus > 0.0:
                y_pred_physical = inverse_scale_output_torch(
                    y_pred_scaled, y_mean_scaler, y_std_scaler, DEVICE, DTYPE
                )
                pplus_loss, pplus_n1_loss, pplus_n2_loss = pplus_pde_loss(
                    y_pred_physical=y_pred_physical,
                    pplus_n1=pplus_n1,
                    pplus_n2=pplus_n2,
                    num_tau=num_tau,
                    num_a=num_a,
                    a_targets=a_targets,
                    pplus_a_grid=pplus_a_grid,
                    tau_grid=tau_grid,
                    avg_first_k_tau=PPLUS_PDE_AVG_FIRST_K_TAU,
                    n2_weight=PPLUS_PDE_N2_WEIGHT,
                )
                diag_loss, diag_n1_loss, diag_n2_loss = pplus_diagonal_consistency_loss(
                    y_pred_physical=y_pred_physical,
                    pplus_n1=pplus_n1,
                    pplus_n2=pplus_n2,
                    num_tau=num_tau,
                    num_a=num_a,
                    a_targets=a_targets,
                    pplus_a_grid=pplus_a_grid,
                    tau_grid=tau_grid,
                )
            else:
                pplus_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                pplus_n1_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                pplus_n2_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                diag_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                diag_n1_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                diag_n2_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)

            total = data_loss + lambda_pplus * pplus_loss + SMOOTHNESS_LOSS_WEIGHT * smooth_loss

            total_loss_sum += total.item() * bs
            data_loss_sum += data_loss.item() * bs
            d1_loss_sum += d1_loss.item() * bs
            d2_loss_sum += d2_loss.item() * bs
            pplus_loss_sum += pplus_loss.item() * bs
            pplus_n1_sum += pplus_n1_loss.item() * bs
            pplus_n2_sum += pplus_n2_loss.item() * bs
            smooth_sum += smooth_loss.item() * bs
            diag_sum += diag_loss.item() * bs
            diag_n1_sum += diag_n1_loss.item() * bs
            diag_n2_sum += diag_n2_loss.item() * bs
            sample_count += bs

    metrics = {
        'val_total_loss': total_loss_sum / max(1, sample_count),
        'val_data_loss': data_loss_sum / max(1, sample_count),
        'val_d1': d1_loss_sum / max(1, sample_count),
        'val_d2': d2_loss_sum / max(1, sample_count),
        'val_pplus_loss': pplus_loss_sum / max(1, sample_count),
        'val_pplus_n1': pplus_n1_sum / max(1, sample_count),
        'val_pplus_n2': pplus_n2_sum / max(1, sample_count),
        'val_smooth_loss': smooth_sum / max(1, sample_count),
        'val_diag_loss': diag_sum / max(1, sample_count),
        'val_diag_n1': diag_n1_sum / max(1, sample_count),
        'val_diag_n2': diag_n2_sum / max(1, sample_count),
        'val_lambda_pplus': lambda_pplus,
    }
    return metrics


# =========================================================
# Visualization
# =========================================================
def save_training_plots(logs):
    plt.figure(figsize=(15, 7))
    plt.plot(logs['iter'], logs['train_total_loss'], label='Train Total', alpha=0.7)
    plt.plot(logs['iter'], logs['train_data_loss'], label='Train Data', alpha=0.5)
    plt.plot(logs['iter'], logs['train_pplus_loss'], label='Train Pplus PDE', alpha=0.5)

    plt.plot(logs['val_iter'], logs['val_total_loss'], label='Val Total', marker='.')
    plt.plot(logs['val_iter'], logs['val_data_loss'], label='Val Data', linestyle=':')
    plt.plot(logs['val_iter'], logs['val_pplus_loss'], label='Val Pplus PDE', linestyle=':')
    if 'val_diag_loss' in logs and len(logs['val_diag_loss']) == len(logs['val_iter']):
        plt.plot(logs['val_iter'], logs['val_diag_loss'], label='Val Diag Consistency', linestyle='-.')

    plt.axvline(x=WARMUP_STEPS, color='gray', linestyle='--', label='Warmup End')
    plt.axvline(x=PPLUS_PDE_START_ITER, color='purple', linestyle='--', label='Pplus PDE Start')

    if logs['val_total_loss']:
        best_idx = int(np.argmin(logs['val_total_loss']))
        best_iter = logs['val_iter'][best_idx]
        best_val = logs['val_total_loss'][best_idx]
        plt.scatter([best_iter], [best_val], color='green', s=100, zorder=5, label=f'Best @ {best_iter}')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()


def save_prediction_plots(model, val_records, branch_val_scaled, y_val_np,
                          y_mean_scaler, y_std_scaler, unified_a_grid, unified_tau_grid):
    model.eval()
    vis_count = min(4, len(val_records))
    indices = np.random.choice(len(val_records), vis_count, replace=False)

    num_tau = len(unified_tau_grid)
    num_a = len(unified_a_grid)

    fig, axes = plt.subplots(vis_count, 4, figsize=(20, 5 * vis_count))
    if vis_count == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle('Prediction vs Ground Truth', fontsize=16)

    with torch.no_grad():
        for row_id, idx in enumerate(indices):
            branch_input_scaled = torch.from_numpy(branch_val_scaled[idx]).unsqueeze(0).to(DEVICE).to(DTYPE)
            y_pred_np = model.predict(branch_input_scaled, y_mean_scaler, y_std_scaler).cpu().numpy().flatten()
            y_true_np = y_val_np[idx]

            field_len = len(y_true_np) // 2
            y_pred_d1 = y_pred_np[:field_len].reshape(num_tau, num_a)
            y_pred_d2 = y_pred_np[field_len:].reshape(num_tau, num_a)
            y_true_d1 = y_true_np[:field_len].reshape(num_tau, num_a)
            y_true_d2 = y_true_np[field_len:].reshape(num_tau, num_a)

            vmax_d1 = max(y_true_d1.max(), y_pred_d1.max())
            vmin_d1 = min(y_true_d1.min(), y_pred_d1.min())
            vmax_d2 = max(y_true_d2.max(), y_pred_d2.max())
            vmin_d2 = min(y_true_d2.min(), y_pred_d2.min())

            def plot_field(ax, data, title, vmin, vmax):
                im = ax.imshow(
                    data,
                    aspect='auto',
                    origin='lower',
                    extent=[unified_a_grid.min(), unified_a_grid.max(), unified_tau_grid.min(), unified_tau_grid.max()],
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(title)
                ax.set_xlabel('A')
                ax.set_ylabel('tau')
                fig.colorbar(im, ax=ax)

            plot_field(axes[row_id, 0], y_true_d1, f'Sample {idx} - D1 True', vmin_d1, vmax_d1)
            plot_field(axes[row_id, 1], y_pred_d1, f'Sample {idx} - D1 Pred', vmin_d1, vmax_d1)
            plot_field(axes[row_id, 2], y_true_d2, f'Sample {idx} - D2 True', vmin_d2, vmax_d2)
            plot_field(axes[row_id, 3], y_pred_d2, f'Sample {idx} - D2 Pred', vmin_d2, vmax_d2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.close()


def save_generator_comparison_plot(model, val_records, branch_val_scaled,
                                   y_mean_scaler, y_std_scaler,
                                   unified_a_grid, unified_tau_grid):
    model.eval()
    vis_count = min(4, len(val_records))
    indices = np.random.choice(len(val_records), vis_count, replace=False)
    num_tau = len(unified_tau_grid)
    num_a = len(unified_a_grid)

    fig, axes = plt.subplots(vis_count, 2, figsize=(14, 4 * vis_count))
    if vis_count == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row_id, idx in enumerate(indices):
            branch_input_scaled = torch.from_numpy(branch_val_scaled[idx]).unsqueeze(0).to(DEVICE).to(DTYPE)
            y_pred_np = model.predict(branch_input_scaled, y_mean_scaler, y_std_scaler).cpu().numpy().flatten()
            field_len = len(y_pred_np) // 2
            y_pred_d1 = y_pred_np[:field_len].reshape(num_tau, num_a)
            y_pred_d2 = y_pred_np[field_len:].reshape(num_tau, num_a)

            pred_gen_d1 = y_pred_d1[:PPLUS_PDE_AVG_FIRST_K_TAU].mean(axis=0)
            pred_gen_d2 = y_pred_d2[:PPLUS_PDE_AVG_FIRST_K_TAU].mean(axis=0)

            params = val_records[idx].branch_params
            nu, kappa, d_diffusion = params.tolist()
            true_d1 = theoretical_D1_d(unified_a_grid, nu, kappa, d_diffusion)
            true_d2 = theoretical_D2_d(unified_a_grid, nu, kappa, d_diffusion)

            axes[row_id, 0].plot(unified_a_grid, true_d1, label='Theory')
            axes[row_id, 0].plot(unified_a_grid, pred_gen_d1, '--', label='Pred avg small-tau')
            axes[row_id, 0].set_title(f'Sample {idx} D1 generator')
            axes[row_id, 0].grid(True)
            axes[row_id, 0].legend()

            axes[row_id, 1].plot(unified_a_grid, true_d2, label='Theory')
            axes[row_id, 1].plot(unified_a_grid, pred_gen_d2, '--', label='Pred avg small-tau')
            axes[row_id, 1].set_title(f'Sample {idx} D2 generator')
            axes[row_id, 1].grid(True)
            axes[row_id, 1].legend()

    plt.tight_layout()
    plt.savefig(GENERATOR_PLOT_PATH, dpi=150)
    plt.close()


# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    (
        paired_records,
        branch_inputs_np,
        y_snapshots_np,
        unified_a_grid,
        unified_tau_grid,
        pplus_a_targets,
        pplus_a_grid,
        pplus_tau_grid,
    ) = load_unified_data_with_pplus(DATA_DIR, TARGET_NUM_FILES)

    if len(unified_a_grid) != len(pplus_a_targets) or not np.allclose(unified_a_grid, pplus_a_targets):
        raise ValueError('CSV A grid and NPZ A_targets do not match. Data must be generated on the same fixed A set.')
    if len(unified_tau_grid) != len(pplus_tau_grid) or not np.allclose(unified_tau_grid, pplus_tau_grid):
        raise ValueError('CSV tau grid and NPZ tau_values do not match.')

    indices_all = np.arange(len(paired_records))
    train_idx, val_idx = train_test_split(indices_all, test_size=VALIDATION_SPLIT, random_state=SEED)

    train_records = [paired_records[i] for i in train_idx]
    val_records = [paired_records[i] for i in val_idx]

    branch_train_np = branch_inputs_np[train_idx]
    branch_val_np = branch_inputs_np[val_idx]
    y_train_np = y_snapshots_np[train_idx]
    y_val_np = y_snapshots_np[val_idx]

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
        'y_std_scaler': y_std_scaler,
    }, SCALER_SAVE_PATH)

    torch.save({
        'y_mean_pod_scaled': y_mean_pod_scaled,
        'pod_basis': pod_basis,
        'num_pod_modes': actual_num_modes,
        'unified_a_grid': unified_a_grid,
        'unified_tau_grid': unified_tau_grid,
        'pplus_a_grid': pplus_a_grid,
        'pplus_tau_grid': pplus_tau_grid,
    }, POD_PARAMS_SAVE_PATH)

    train_dataset = PairedAdjointDataset(train_records, branch_train_scaled, y_train_scaled)
    val_dataset = PairedAdjointDataset(val_records, branch_val_scaled, y_val_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=ADAM_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_dict,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ADAM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_dict,
    )

    model = PODDeepONet(
        BRANCH_INPUT_DIM,
        HIDDEN_UNITS,
        NUM_HIDDEN_LAYERS,
        actual_num_modes,
        pod_basis,
        y_mean_pod_scaled,
        DROPOUT_RATE,
    ).to(DEVICE)

    print(f'Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, ADAM_ITERATIONS)
    data_loss_fn = WeightedMSELoss(d2_weight=D2_LOSS_WEIGHT)

    num_a = len(unified_a_grid)
    num_tau = len(unified_tau_grid)

    logs = {
        'iter': [],
        'train_total_loss': [],
        'train_data_loss': [],
        'train_pplus_loss': [],
        'train_pplus_n1': [],
        'train_pplus_n2': [],
        'train_smooth_loss': [],
        'train_lambda_pplus': [],
        'lr': [],
        'val_iter': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_d1': [],
        'val_d2': [],
        'val_pplus_loss': [],
        'val_pplus_n1': [],
        'val_pplus_n2': [],
        'val_smooth_loss': [],
        'val_diag_loss': [],
        'val_diag_n1': [],
        'val_diag_n2': [],
        'val_lambda_pplus': [],
    }

    min_val_loss = float('inf')
    early_stop_counter = 0
    best_val_metrics = None

    print('\n--- Start training: Data Loss + Pplus PDE Loss ---')
    pbar = tqdm(total=ADAM_ITERATIONS, desc='Training')
    global_iter = 0
    done = False

    while not done:
        for batch in train_loader:
            if global_iter >= ADAM_ITERATIONS:
                done = True
                break

            model.train()
            optimizer.zero_grad()

            batch_branch = batch['branch'].to(DEVICE)
            batch_y_scaled = batch['y'].to(DEVICE)
            batch_pplus_n1 = batch['pplus_n1'].to(DEVICE)
            batch_pplus_n2 = batch['pplus_n2'].to(DEVICE)

            y_pred_scaled = model(batch_branch)
            data_loss, _, _ = data_loss_fn(y_pred_scaled, batch_y_scaled)
            smooth_loss = smoothness_loss(y_pred_scaled, num_tau, num_a)
            lambda_pplus_now = get_current_pplus_weight(global_iter)

            if lambda_pplus_now > 0.0:
                y_pred_physical = inverse_scale_output_torch(
                    y_pred_scaled, y_mean_scaler, y_std_scaler, DEVICE, DTYPE
                )
                pplus_loss, pplus_n1_loss, pplus_n2_loss = pplus_pde_loss(
                    y_pred_physical=y_pred_physical,
                    pplus_n1=batch_pplus_n1,
                    pplus_n2=batch_pplus_n2,
                    num_tau=num_tau,
                    num_a=num_a,
                    a_targets=unified_a_grid,
                    pplus_a_grid=pplus_a_grid,
                    tau_grid=unified_tau_grid,
                    avg_first_k_tau=PPLUS_PDE_AVG_FIRST_K_TAU,
                    n2_weight=PPLUS_PDE_N2_WEIGHT,
                )
            else:
                pplus_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                pplus_n1_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
                pplus_n2_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)

            total_loss = data_loss + lambda_pplus_now * pplus_loss + SMOOTHNESS_LOSS_WEIGHT * smooth_loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()

            logs['iter'].append(global_iter)
            logs['train_total_loss'].append(total_loss.item())
            logs['train_data_loss'].append(data_loss.item())
            logs['train_pplus_loss'].append(pplus_loss.item())
            logs['train_pplus_n1'].append(pplus_n1_loss.item())
            logs['train_pplus_n2'].append(pplus_n2_loss.item())
            logs['train_smooth_loss'].append(smooth_loss.item())
            logs['train_lambda_pplus'].append(lambda_pplus_now)
            logs['lr'].append(optimizer.param_groups[0]['lr'])

            pbar.update(1)
            pbar.set_postfix({
                'total': f'{total_loss.item():.3e}',
                'data': f'{data_loss.item():.3e}',
                'pplus': f'{pplus_loss.item():.3e}',
                'lam_p': f'{lambda_pplus_now:.1e}',
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            if global_iter % VALIDATION_FREQUENCY == 0:
                val_metrics = run_validation(
                    model=model,
                    val_loader=val_loader,
                    data_loss_fn=data_loss_fn,
                    y_mean_scaler=y_mean_scaler,
                    y_std_scaler=y_std_scaler,
                    num_tau=num_tau,
                    num_a=num_a,
                    a_targets=unified_a_grid,
                    pplus_a_grid=pplus_a_grid,
                    tau_grid=unified_tau_grid,
                    global_iter=global_iter,
                )

                logs['val_iter'].append(global_iter)
                logs['val_total_loss'].append(val_metrics['val_total_loss'])
                logs['val_data_loss'].append(val_metrics['val_data_loss'])
                logs['val_d1'].append(val_metrics['val_d1'])
                logs['val_d2'].append(val_metrics['val_d2'])
                logs['val_pplus_loss'].append(val_metrics['val_pplus_loss'])
                logs['val_pplus_n1'].append(val_metrics['val_pplus_n1'])
                logs['val_pplus_n2'].append(val_metrics['val_pplus_n2'])
                logs['val_smooth_loss'].append(val_metrics['val_smooth_loss'])
                logs['val_diag_loss'].append(val_metrics['val_diag_loss'])
                logs['val_diag_n1'].append(val_metrics['val_diag_n1'])
                logs['val_diag_n2'].append(val_metrics['val_diag_n2'])
                logs['val_lambda_pplus'].append(val_metrics['val_lambda_pplus'])

                print(
                    f"\nIter {global_iter} | Val Total={val_metrics['val_total_loss']:.6e} | "
                    f"Val Data={val_metrics['val_data_loss']:.6e} | "
                    f"Val Pplus={val_metrics['val_pplus_loss']:.6e} | "
                    f"Val Pplus(n1)={val_metrics['val_pplus_n1']:.6e} | "
                    f"Val Pplus(n2)={val_metrics['val_pplus_n2']:.6e} | "
                    f"Val Diag={val_metrics['val_diag_loss']:.6e} | "
                    f"D1={val_metrics['val_d1']:.6e} | D2={val_metrics['val_d2']:.6e}"
                )

                if val_metrics['val_total_loss'] < min_val_loss:
                    min_val_loss = val_metrics['val_total_loss']
                    early_stop_counter = 0
                    best_val_metrics = {'iter': global_iter, **val_metrics}
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    pd.DataFrame([best_val_metrics]).to_csv(VAL_METRICS_SAVE_PATH, index=False)
                    print('  -> New best model saved.')
                else:
                    early_stop_counter += 1

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    print(f'\n*** Early stopping at iter {global_iter}. ***')
                    done = True
                    break

            global_iter += 1

    pbar.close()

    loss_df = pd.DataFrame(logs)
    loss_df.to_csv(LOSS_DATA_SAVE_PATH, index=False)

    save_training_plots(logs)

    print('\n--- Visualization ---')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    save_prediction_plots(
        model=model,
        val_records=val_records,
        branch_val_scaled=branch_val_scaled,
        y_val_np=y_val_np,
        y_mean_scaler=y_mean_scaler,
        y_std_scaler=y_std_scaler,
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
    )
    save_generator_comparison_plot(
        model=model,
        val_records=val_records,
        branch_val_scaled=branch_val_scaled,
        y_mean_scaler=y_mean_scaler,
        y_std_scaler=y_std_scaler,
        unified_a_grid=unified_a_grid,
        unified_tau_grid=unified_tau_grid,
    )

    print('\n=== Training complete ===')
    print(f'Best model: {MODEL_SAVE_PATH}')
    print(f'Scalers: {SCALER_SAVE_PATH}')
    print(f'POD params: {POD_PARAMS_SAVE_PATH}')
    print(f'Loss csv: {LOSS_DATA_SAVE_PATH}')
    print(f'Training plot: {PLOT_SAVE_PATH}')
    print(f'Prediction plot: {PREDICTION_PLOT_PATH}')
    print(f'Generator plot: {GENERATOR_PLOT_PATH}')
    if best_val_metrics is not None:
        print('Best validation metrics:', best_val_metrics)
