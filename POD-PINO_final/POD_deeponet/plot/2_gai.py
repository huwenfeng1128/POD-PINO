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
# 训练数据路径
TRAIN_DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'
# 外部测试验证数据路径
TEST_KM_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
# 结果输出路径
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\mode_sensitivity_final_v2'

os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(RESULT_DIR, 'models')
PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
DATA_SAVE_DIR = os.path.join(RESULT_DIR, 'field_results_csv')  # 专门存放CSV的文件夹

for d in [MODEL_DIR, PLOT_DIR, DATA_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 实验变量 ---
# 测试的模态数列表
MODE_COUNTS_TO_TEST = [10, 20, 40, 80, 100, 120, 150, 200]

# 用于画图对比的三个模态 (Low, Mid, High)
# 必须包含在 MODE_COUNTS_TO_TEST 中
VIS_MODE_COMPARISON = [10, 80, 200]

# --- 训练超参数 ---
TRAIN_FILE_LIMIT = 2500  # 训练用的文件数量
EPOCHS = 20000  # 固定训练迭代次数
BATCH_SIZE = 64
LR = 3e-4
HIDDEN_UNITS = 256
LAYERS = 4
DROPOUT = 0.1
D2_WEIGHT = 0.8
WARMUP_STEPS = 2000
SEED = 42

# --- 评估设置 ---
NUM_VIS_SAMPLES = 3  # 随机选3个文件画场图

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
        # 注册不可训练参数
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
    # full_matrices=False 是关键，节省内存
    U, S, Vt = np.linalg.svd(y_train_s - y_mean_pod, full_matrices=False)
    print(f"SVD 完成。最大秩: {Vt.shape[0]}")

    train_ds = TensorDataset(torch.from_numpy(b_train_s).to(DEVICE), torch.from_numpy(y_train_s).to(DEVICE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    all_loss_histories = {}

    # --- Step 2: 循环训练 (不同模态数) ---
    print("\n>>> 步骤 2/5: 开始多模态训练循环")
    for n_modes in MODE_COUNTS_TO_TEST:
        if n_modes > Vt.shape[0]:
            print(f"跳过模态数 {n_modes} (超过最大秩 {Vt.shape[0]})")
            continue

        print(f"\n--- 正在训练模态数: {n_modes} ---")

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

        # 保存模型
        save_name = f"modes_{n_modes}"
        torch.save({
            'model_state': model.state_dict(),
            'basis': basis_k,
            'y_mean_pod': y_mean_pod,
            'scalers': {'b_mean': b_mean, 'b_std': b_std, 'y_mean': y_mean, 'y_std': y_std},
            'grid': {'a': u_a, 'tau': u_tau},
            'loss_history': loss_history
        }, os.path.join(MODEL_DIR, f'ckpt_{save_name}.pth'))

        all_loss_histories[n_modes] = loss_history

        # 清理显存
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
        gc.collect()

    # --- Step 3: 评估 (Calculation & Save CSV) ---
    print("\n>>> 步骤 3/5: 外部数据评估与CSV保存")

    # 1. 锁定所有测试文件
    all_test_files = glob.glob(os.path.join(TEST_KM_DATA_DIR, "*.csv"))
    if not all_test_files:
        raise FileNotFoundError(f"在 {TEST_KM_DATA_DIR} 未找到测试文件")
    print(f"共找到 {len(all_test_files)} 个外部测试文件。将全部用于计算误差。")

    # 2. 锁定 3 个用于可视化的文件
    random.seed(SEED + 123)  # 确保每次运行选同样的文件
    vis_files_path = random.sample(all_test_files, min(len(all_test_files), NUM_VIS_SAMPLES))
    vis_filenames = [os.path.basename(f) for f in vis_files_path]
    print(f"选定可视化的文件: {vis_filenames}")

    # 3. 预加载所有测试数据到内存 (加速循环)
    test_data_cache = []
    for fpath in tqdm(all_test_files, desc="预加载测试文件"):
        try:
            fname = os.path.basename(fpath)
            # 解析文件名参数 (x,y,z).csv
            params_str = fname.replace('(', '').replace(').csv', '').split(',')
            params = [float(p) for p in params_str]  # [nu, kappa, d]
            df = pd.read_csv(fpath)
            test_data_cache.append({'fname': fname, 'params': params, 'df': df, 'is_vis': fpath in vis_files_path})

            # 如果是可视化文件，保存其真实值的CSV
            if fpath in vis_files_path:
                save_true_path = os.path.join(DATA_SAVE_DIR, f'vis_TRUE_{fname}')  # 保留.csv后缀
                df[['tau_sec', 'A', 'D1_data', 'D2_data']].to_csv(save_true_path, index=False)

        except Exception as e:
            continue

    error_summary = []  # 记录所有模态的平均误差

    # 4. 遍历所有模态进行预测
    for n_modes in MODE_COUNTS_TO_TEST:
        ckpt_path = os.path.join(MODEL_DIR, f'ckpt_modes_{n_modes}.pth')
        if not os.path.exists(ckpt_path): continue

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

        d1_errs, d2_errs = [], []

        # 遍历所有缓存的测试文件
        for item in tqdm(test_data_cache, desc=f"Eval Modes={n_modes}"):
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

            # 计算误差 (Interpolation)
            interp_d1 = RegularGridInterpolator((grid_tau, grid_a), d1_field, bounds_error=False, fill_value=None)
            interp_d2 = RegularGridInterpolator((grid_tau, grid_a), d2_field, bounds_error=False, fill_value=None)

            query_points = item['df'][['tau_sec', 'A']].values
            d1_pred_pts = interp_d1(query_points)
            d2_pred_pts = interp_d2(query_points)

            mask = ~np.isnan(item['df']['D1_data']) & ~np.isnan(d1_pred_pts)
            if np.sum(mask) > 0:
                d1_mse = np.mean((item['df']['D1_data'][mask] - d1_pred_pts[mask]) ** 2)
                d2_mse = np.mean((item['df']['D2_data'][mask] - d2_pred_pts[mask]) ** 2)
                d1_errs.append(d1_mse)
                d2_errs.append(d2_mse)

            # --- 保存 CSV: 仅针对3个可视化文件 & 选定的3个对比模态 ---
            if item['is_vis'] and (n_modes in VIS_MODE_COMPARISON):
                # 构造 DataFrame 保存预测场 (展开网格)
                # Meshgrid
                GA, GT = np.meshgrid(grid_a, grid_tau)
                flat_a = GA.flatten()
                flat_tau = GT.flatten()
                flat_d1 = d1_field.flatten()
                flat_d2 = d2_field.flatten()

                df_pred = pd.DataFrame({
                    'A': flat_a,
                    'tau': flat_tau,
                    'D1_pred': flat_d1,
                    'D2_pred': flat_d2
                })

                save_csv_name = f"vis_PRED_{item['fname'][:-4]}_mode_{n_modes}.csv"
                df_pred.to_csv(os.path.join(DATA_SAVE_DIR, save_csv_name), index=False)

        # 记录误差
        if d1_errs:
            error_summary.append({
                'num_modes': n_modes,
                'mse_d1': np.mean(d1_errs),
                'mse_d2': np.mean(d2_errs),
                'mse_total': np.mean(d1_errs) + np.mean(d2_errs)
            })

    # 保存总误差表
    df_err_summary = pd.DataFrame(error_summary)
    df_err_summary.to_csv(os.path.join(RESULT_DIR, 'summary_error_vs_modes.csv'), index=False)

    # ==========================================
    # 5. 可视化绘图 (Plotting)
    # ==========================================
    print("\n>>> 步骤 4/5: 生成图像")

    # --- 图 1: 训练 Loss 对比 ---
    plt.figure(figsize=(10, 6))
    for n_modes, history in all_loss_histories.items():
        # 移动平均平滑
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

    # --- 图 2: 误差随模态变化 ---
    if not df_err_summary.empty:
        plt.figure(figsize=(8, 6))
        plt.plot(df_err_summary['num_modes'], df_err_summary['mse_d1'], 'o-', label='D1 MSE')
        plt.plot(df_err_summary['num_modes'], df_err_summary['mse_d2'], 's-', label='D2 MSE')
        plt.plot(df_err_summary['num_modes'], df_err_summary['mse_total'], 'k--', label='Total MSE', alpha=0.5)
        plt.yscale('log')
        plt.xlabel('Number of POD Modes')
        plt.ylabel('Test Set MSE')
        plt.title('Generalization Error vs. Mode Count')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, '2_error_vs_modes.png'), dpi=300)
        plt.close()

    # --- 图 3: 场域恢复对比 (Low, Mid, High) ---
    # 我们需要从刚才保存的 CSV 中读取数据来画图

    valid_modes_for_plot = [m for m in VIS_MODE_COMPARISON if m in df_err_summary['num_modes'].values]
    if len(valid_modes_for_plot) < 3:
        print(f"警告: 无法找到所有对比模态 ({VIS_MODE_COMPARISON}) 的数据，可能某些模态未训练。")

    # 重新加载三个文件的 CSV 数据进行绘图
    for fname in vis_filenames:
        print(f"正在绘制文件: {fname}")
        base_name = fname[:-4]

        # 1. 读取真实值
        true_csv_path = os.path.join(DATA_SAVE_DIR, f'vis_TRUE_{fname}')
        if not os.path.exists(true_csv_path):
            print(f"  找不到真实值文件 {true_csv_path}，跳过。")
            continue
        df_true = pd.read_csv(true_csv_path)

        # 2. 读取 3 个模态的预测值
        pred_dfs = {}
        for m in valid_modes_for_plot:
            p_csv = os.path.join(DATA_SAVE_DIR, f'vis_PRED_{base_name}_mode_{m}.csv')
            if os.path.exists(p_csv):
                pred_dfs[m] = pd.read_csv(p_csv)

        if len(pred_dfs) == 0: continue

        # 准备绘图网格 (从第一个预测文件获取网格结构)
        first_m = list(pred_dfs.keys())[0]
        # 因为保存的是展开的 meshgrid，我们需要 reshape 回去
        # Pandas 读取后顺序可能不变，但为了保险，用 pivot
        # 假设 Grid 是规则的
        grid_df = pred_dfs[first_m]
        unique_a = sorted(grid_df['A'].unique())
        unique_tau = sorted(grid_df['tau'].unique())

        # Pivot table to get matrix
        # 这里只为了获取 extent 和 meshgrid
        GA, GT = np.meshgrid(unique_a, unique_tau)

        # 插值真实值到网格 (为了画图)
        # griddata points: (x,y) -> values
        valid_true = df_true.dropna(subset=['D1_data'])
        d1_true_grid = griddata((valid_true['A'], valid_true['tau_sec']), valid_true['D1_data'], (GA, GT),
                                method='linear')

        # 开始画图
        # 布局: 4列 (True, Low, Mid, High) x 2行 (Field, Error)
        cols = 1 + len(valid_modes_for_plot)
        fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8), constrained_layout=True)

        # 统一 D1 的 Color range
        vmin = np.nanmin(d1_true_grid)
        vmax = np.nanmax(d1_true_grid)

        # --- Column 1: True ---
        im0 = axes[0, 0].pcolormesh(GA, GT, d1_true_grid, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Ground Truth (Interp)")
        fig.colorbar(im0, ax=axes[0, 0], location='bottom', pad=0.1)
        axes[1, 0].axis('off')  # 无 Error 图

        # --- Columns 2+: Modes ---
        global_max_err = 0

        # 第一次循环先找最大误差以便统一 Colorbar
        for i, m in enumerate(valid_modes_for_plot):
            # Reshape pred data
            p_data = pred_dfs[m]
            # 必须确保顺序正确，使用 pivot
            d1_pred_mat = p_data.pivot(index='tau', columns='A', values='D1_pred').values
            # 注意: pivot 可能会重新排序 index/columns，所以要确认顺序。
            # 上面 unique_a 是 sorted 的，pivot 默认也是 sorted，所以应该匹配。

            err = np.abs(d1_true_grid - d1_pred_mat)
            global_max_err = max(global_max_err, np.nanmax(err))

        # 第二次循环画图
        for i, m in enumerate(valid_modes_for_plot):
            col_idx = i + 1
            p_data = pred_dfs[m]
            d1_pred_mat = p_data.pivot(index='tau', columns='A', values='D1_pred').values
            err = np.abs(d1_true_grid - d1_pred_mat)

            # Field
            im = axes[0, col_idx].pcolormesh(GA, GT, d1_pred_mat, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, col_idx].set_title(f"Pred (Modes={m})")
            fig.colorbar(im, ax=axes[0, col_idx], location='bottom', pad=0.1)

            # Error
            im_e = axes[1, col_idx].pcolormesh(GA, GT, err, shading='auto', cmap='inferno', vmin=0, vmax=global_max_err)
            axes[1, col_idx].set_title(f"Abs Error (Modes={m})")
            fig.colorbar(im_e, ax=axes[1, col_idx], location='bottom', pad=0.1)

        plt.suptitle(f"Reconstruction Analysis: {base_name} (D1 Field)", fontsize=16)
        save_plot_path = os.path.join(PLOT_DIR, f'compare_{base_name}.png')
        plt.savefig(save_plot_path, dpi=200)
        plt.close()

    print(f"\n全部完成！结果已保存在: {RESULT_DIR}")
    print(f"- 模型: {MODEL_DIR}")
    print(f"- 图片: {PLOT_DIR}")
    print(f"- CSV数据: {DATA_SAVE_DIR}")


if __name__ == "__main__":
    main()