# -*- coding: utf-8 -*-
import os
import glob
import json
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

# --- 1. 全局配置 ---
BASE_RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\full_sensitivity_analysis_v1'
DATA_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

# 基础参数 (作为控制变量时的默认值)
DEFAULT_CONFIG = {
    "d2_weight": 0.5,
    "layers": 4,
    "hidden_units": 128
}

# 训练参数
POD_MODES = 100
ADAM_LR = 1e-3
# 为了演示，这里设为 3000，实际科研建议设为 10000 或更多
ADAM_ITERATIONS = 50000
BATCH_SIZE = 64
WARMUP_STEPS = 500
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# --- 2. 实验设计生成器 ---
def get_experiment_configs():
    experiments = {}

    # Study 1: 损失权重影响 (固定结构: 4层, 128单元)
    # 目的: 分析对 D2 的关注度如何影响恢复效果
    weight_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    experiments['Study_Weight'] = []
    for w in weight_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['d2_weight'] = w
        cfg['id'] = f"Weight_{w}"
        experiments['Study_Weight'].append(cfg)

    # Study 2: 网络深度影响 (固定: 权重0.5, 宽度128)
    # 目的: 分析深层网络是否能捕捉更复杂的模式
    depth_values = [2, 4, 6, 8]
    experiments['Study_Depth'] = []
    for d in depth_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['layers'] = d
        cfg['id'] = f"Depth_{d}"
        experiments['Study_Depth'].append(cfg)

    # Study 3: 网络宽度影响 (固定: 权重0.5, 深度4)
    # 目的: 分析参数量对拟合能力的影响
    width_values = [32, 64, 128, 256]
    experiments['Study_Width'] = []
    for u in width_values:
        cfg = DEFAULT_CONFIG.copy()
        cfg['hidden_units'] = u
        cfg['id'] = f"Width_{u}"
        experiments['Study_Width'].append(cfg)

    return experiments


# --- 3. 模型定义 (保持不变) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, num_hidden_layers, output_dim, dropout_rate):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units, dtype=DTYPE)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.GELU(),
                nn.LayerNorm(hidden_units, dtype=DTYPE),
                # nn.Dropout(p=dropout_rate), # 在分析实验中暂时关闭Dropout以减少随机性波动
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
        self.pod_basis = nn.Parameter(torch.tensor(pod_basis, dtype=DTYPE), requires_grad=False)
        self.y_mean_pod_scaled = nn.Parameter(torch.tensor(y_mean_pod_scaled, dtype=DTYPE), requires_grad=False)

    def forward(self, branch_x):
        coeffs = self.branch(branch_x)
        return torch.matmul(coeffs, self.pod_basis.T) + self.y_mean_pod_scaled


class WeightedMSELoss(nn.Module):
    def __init__(self, d2_weight=0.5):
        super().__init__()
        self.d2_weight = d2_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mid = y_pred.shape[1] // 2
        loss_d1 = self.mse(y_pred[:, :mid], y_true[:, :mid])
        loss_d2 = self.mse(y_pred[:, mid:], y_true[:, mid:])
        # 注意：为了让不同权重的 total loss 有可比性，这里最好不做加权求和对比，
        # 但为了训练，必须加权。分析时我们主要看 D1 和 D2 的独立 Loss。
        total = (1 - self.d2_weight) * loss_d1 + self.d2_weight * loss_d2
        return total, loss_d1, loss_d2


# --- 4. 数据加载与预处理 ---
def prepare_data():
    print("加载并预处理数据...")
    files = glob.glob(os.path.join(DATA_DIR, "data_*.csv"))
    random.seed(SEED)
    selected = random.sample(files, min(len(files), 2000))  # 样本量

    b_list, y_list = [], []
    ua, ut = None, None
    for f in tqdm(selected, desc="Loading"):
        try:
            df = pd.read_csv(f, on_bad_lines='skip').dropna()
            if df.empty: continue
            if ua is None: ua, ut = sorted(df['A'].unique()), sorted(df['tau'].unique())
            b_list.append(df[['nu', 'kappa', 'd_diffusion']].iloc[0].values)
            df = df.sort_values(['tau', 'A'])
            y_list.append(np.concatenate([df['D1_fp'].values, df['D2_fp'].values]))
        except:
            continue

    b_np, y_np = np.array(b_list), np.array(y_list)

    # 归一化
    b_mean, b_std = b_np.mean(0), b_np.std(0)
    y_mean, y_std = y_np.mean(0), y_np.std(0)
    y_std[y_std < 1e-10] = 1.0
    b_scaled = (b_np - b_mean) / b_std
    y_scaled = (y_np - y_mean) / y_std

    # 划分
    b_train, b_val, y_train, y_val = train_test_split(b_scaled, y_scaled, test_size=0.2, random_state=SEED)

    # POD 计算
    y_mean_pod = y_train.mean(0)
    U, S, Vt = np.linalg.svd(y_train - y_mean_pod, full_matrices=False)
    pod_basis = Vt.T[:, :POD_MODES]

    data_dict = {
        'train_loader': DataLoader(TensorDataset(torch.tensor(b_train, dtype=DTYPE).to(DEVICE),
                                                 torch.tensor(y_train, dtype=DTYPE).to(DEVICE)),
                                   batch_size=BATCH_SIZE, shuffle=True),
        'val_b': torch.tensor(b_val, dtype=DTYPE).to(DEVICE),
        'val_y': torch.tensor(y_val, dtype=DTYPE).to(DEVICE),
        'pod_basis': pod_basis,
        'y_mean_pod': y_mean_pod,
        'grid_shape': (len(ut), len(ua)),
        'grid_vals': (ua, ut)
    }
    return data_dict


# --- 5. 训练执行函数 ---
def run_experiment(study_name, config, data_dict):
    save_dir = os.path.join(BASE_RESULT_DIR, study_name, config['id'])
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    model = PODDeepONet(3, config['hidden_units'], config['layers'], POD_MODES,
                        data_dict['pod_basis'], data_dict['y_mean_pod']).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, s / WARMUP_STEPS))
    loss_fn = WeightedMSELoss(d2_weight=config['d2_weight'])

    history = {'iter': [], 'total_loss': [], 'd1_loss': [], 'd2_loss': []}

    pbar = tqdm(range(ADAM_ITERATIONS), desc=f"{study_name}-{config['id']}", leave=False)
    iter_idx = 0
    while iter_idx < ADAM_ITERATIONS:
        for bx, by in data_dict['train_loader']:
            if iter_idx >= ADAM_ITERATIONS: break

            model.train()
            optimizer.zero_grad()
            pred = model(bx)
            loss, _, _ = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iter_idx % 100 == 0:
                model.eval()
                with torch.no_grad():
                    v_pred = model(data_dict['val_b'])
                    v_tot, v_d1, v_d2 = loss_fn(v_pred, data_dict['val_y'])
                    history['iter'].append(iter_idx)
                    history['total_loss'].append(v_tot.item())
                    history['d1_loss'].append(v_d1.item())
                    history['d2_loss'].append(v_d2.item())

            iter_idx += 1
            pbar.update(1)
    pbar.close()

    # 保存结果
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'loss_history.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    return history


# --- 6. 高级绘图分析模块 ---
def analyze_study(study_name, configs, data_dict):
    print(f"\n正在分析研究组: {study_name} ...")
    study_path = os.path.join(BASE_RESULT_DIR, study_name)

    # 收集最终结果用于绘制趋势图
    summary_data = []

    # 1. 绘制 Loss 趋势对比图
    plt.figure(figsize=(12, 5))
    for cfg in configs:
        csv_path = os.path.join(study_path, cfg['id'], 'loss_history.csv')
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)

        # 记录最后一个点的 Loss
        final_d1 = df['d1_loss'].iloc[-1]
        final_d2 = df['d2_loss'].iloc[-1]

        # 提取变量值 (从 'Weight_0.1' 提取 0.1)
        var_val = float(cfg['id'].split('_')[1])
        summary_data.append({'val': var_val, 'd1': final_d1, 'd2': final_d2})

        plt.plot(df['iter'], df['d1_loss'] + df['d2_loss'], label=f"{cfg['id']} (Sum MSE)")

    plt.yscale('log')
    plt.title(f'{study_name}: Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of MSE (D1+D2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(study_path, 'comparison_loss_curves.png'))
    plt.close()

    # 2. 绘制变量影响趋势图 (Parameter vs Final Error)
    if summary_data:
        sdf = pd.DataFrame(summary_data).sort_values('val')
        plt.figure(figsize=(8, 6))
        plt.plot(sdf['val'], sdf['d1'], 'o-', label='D1 Final Error')
        plt.plot(sdf['val'], sdf['d2'], 's-', label='D2 Final Error')
        plt.xlabel('Parameter Value')
        plt.ylabel('Final Validation MSE')
        plt.title(f'{study_name}: Sensitivity Analysis')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(study_path, 'parameter_sensitivity.png'))
        plt.close()

    # 3. 绘制 3x3 九宫格可视化 (Low, Mid, High)
    # 选取三个代表性配置：第一个(Low)，中间一个(Mid)，最后一个(High)
    indices = [0, len(configs) // 2, len(configs) - 1]
    selected_cfgs = [configs[i] for i in indices]

    # 固定一个样本用于对比
    sample_idx = 0
    input_sample = data_dict['val_b'][sample_idx].unsqueeze(0)
    true_sample = data_dict['val_y'][sample_idx].cpu().numpy()
    nt, na = data_dict['grid_shape']
    half_len = nt * na

    # 准备真值图像数据
    d1_true = true_sample[:half_len].reshape(nt, na)
    d2_true = true_sample[half_len:].reshape(nt, na)

    # 创建画布: 3行 (Low, Mid, High 参数) x 3列 (预测D1, 预测D2, 绝对误差)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 统一 Colorbar 范围以便对比
    vmax_d1, vmin_d1 = d1_true.max(), d1_true.min()
    vmax_d2, vmin_d2 = d2_true.max(), d2_true.min()

    for row_idx, cfg in enumerate(selected_cfgs):
        # 加载模型预测
        model_path = os.path.join(study_path, cfg['id'], 'model.pth')
        model = PODDeepONet(3, cfg['hidden_units'], cfg['layers'], POD_MODES,
                            data_dict['pod_basis'], data_dict['y_mean_pod']).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            pred = model(input_sample).cpu().numpy().flatten()

        d1_pred = pred[:half_len].reshape(nt, na)
        d2_pred = pred[half_len:].reshape(nt, na)

        # 计算误差场 (D1+D2 的绝对误差总和，或者单独展示)
        # 这里为了直观，展示总绝对误差 |True - Pred|
        err_map_d1 = np.abs(d1_true - d1_pred)
        err_map_d2 = np.abs(d2_true - d2_pred)
        total_err_map = err_map_d1 + err_map_d2

        # 绘图 - 列1: D1 预测
        im1 = axes[row_idx, 0].imshow(d1_pred, origin='lower', aspect='auto', vmin=vmin_d1, vmax=vmax_d1,
                                      cmap='viridis')
        axes[row_idx, 0].set_ylabel(f"Param: {cfg['id']}", fontsize=12, fontweight='bold')
        if row_idx == 0: axes[row_idx, 0].set_title("Predicted D1 Field")

        # 绘图 - 列2: D2 预测
        im2 = axes[row_idx, 1].imshow(d2_pred, origin='lower', aspect='auto', vmin=vmin_d2, vmax=vmax_d2,
                                      cmap='viridis')
        if row_idx == 0: axes[row_idx, 1].set_title("Predicted D2 Field")

        # 绘图 - 列3: 误差热力图
        im3 = axes[row_idx, 2].imshow(total_err_map, origin='lower', aspect='auto', cmap='inferno')
        if row_idx == 0: axes[row_idx, 2].set_title("Total Absolute Error |True-Pred|")

        # 移除刻度
        for ax in axes[row_idx]: ax.set_xticks([]); ax.set_yticks([])

    # 另存一张 Ground Truth 用于参照
    plt.tight_layout()
    plt.savefig(os.path.join(study_path, 'visualization_3x3_grid.png'))
    plt.close()

    # 单独画真值
    fig_t, ax_t = plt.subplots(1, 2, figsize=(8, 4))
    ax_t[0].imshow(d1_true, origin='lower', aspect='auto', vmin=vmin_d1, vmax=vmax_d1)
    ax_t[0].set_title("Ground Truth D1")
    ax_t[1].imshow(d2_true, origin='lower', aspect='auto', vmin=vmin_d2, vmax=vmax_d2)
    ax_t[1].set_title("Ground Truth D2")
    plt.savefig(os.path.join(study_path, 'visualization_ground_truth.png'))
    plt.close()


# --- 7. 主程序入口 ---
if __name__ == "__main__":
    # 1. 准备数据 (只做一次，保证所有实验数据一致)
    data_dict = prepare_data()

    # 2. 获取所有实验配置
    all_experiments = get_experiment_configs()

    # 3. 循环执行研究组
    for study_name, config_list in all_experiments.items():
        print(f"\n{'=' * 50}")
        print(f"开始研究组: {study_name} (包含 {len(config_list)} 个实验)")
        print(f"{'=' * 50}")

        # 3.1 训练该组内的所有模型
        for config in config_list:
            # 检查是否已经运行过 (避免重复)
            if os.path.exists(os.path.join(BASE_RESULT_DIR, study_name, config['id'], 'model.pth')):
                print(f"Skipping {config['id']}, already exists.")
                continue
            run_experiment(study_name, config, data_dict)

        # 3.2 分析该研究组
        analyze_study(study_name, config_list, data_dict)

    print(f"\n全参数敏感性分析完成！结果已保存在: {BASE_RESULT_DIR}")