# -*- coding: utf-8 -*-
"""
POD-DeepONet 全场预测程序
调用已训练好的模型参数，进行完整网格上的前向预测，并将真实场、预测场、误差场整合到单个CSV文件中。
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import gc
from scipy.interpolate import griddata

# ==========================================
# 配置参数
# ==========================================

# 路径设置
MODEL_DIR = r'D:\PINN\zenodo\POD_deeponet\mode_sensitivity_final_v3\models'
TEST_DATA_DIR = r'D:\PINN\zenodo\AFP\P(A,t)_data\km_data_4'
RESULT_DIR = r'D:\PINN\zenodo\POD_deeponet\mode_sensitivity_final_v3\fullfield_predictions'  # 输出全场预测结果的目录

os.makedirs(RESULT_DIR, exist_ok=True)

# 预测配置
MODES_TO_PREDICT = [10, 20, 40, 80, 100, 120, 150, 200]  # 要进行预测的模态数，必须与模型文件中保存的模态数一致
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

print(f"设备: {DEVICE}")
print(f"数据类型: {DTYPE}")


# ==========================================
# 模型定义 (与训练代码保持一致)
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


# ==========================================
# 辅助函数
# ==========================================

def manual_scaler_transform_tensor(data, mean, std, device):
    """将数据标准化转换为张量"""
    t_data = torch.tensor(data, dtype=DTYPE, device=device)
    t_mean = torch.tensor(mean, dtype=DTYPE, device=device)
    t_std = torch.tensor(std, dtype=DTYPE, device=device)
    return (t_data - t_mean) / t_std


def load_test_files_and_params(test_dir):
    """加载测试文件，提取参数，并缓存真实场数据"""
    print(f"从 {test_dir} 加载测试数据...")
    all_files = glob.glob(os.path.join(test_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"在 {test_dir} 未找到测试文件")

    test_data = []
    for fpath in tqdm(all_files, desc="缓存测试真实数据"):
        try:
            fname = os.path.basename(fpath)
            # 从文件名提取参数: (nu,kappa,d_diffusion).csv
            params_str = fname.replace('(', '').replace(').csv', '').split(',')
            params = [float(p.strip()) for p in params_str]

            df = pd.read_csv(fpath)
            test_data.append({
                'fname': fname,
                'params': params,
                'df_real': df  # 缓存原始真实数据DataFrame
            })
        except Exception as e:
            print(f"警告: 无法加载或解析 {fname}: {e}")
            continue

    print(f"成功加载 {len(test_data)} 个测试样本")
    return test_data


def load_models(model_dir, modes_list):
    """加载所有已训练的模型"""
    models_dict = {}

    # 确保 MODES_TO_PREDICT 中的模态数都在模型目录中找到
    found_modes = []
    for n_modes in modes_list:
        ckpt_path = os.path.join(model_dir, f'model_modes_{n_modes}.pth')
        if not os.path.exists(ckpt_path):
            print(f"警告: 找不到模态 {n_modes} 的模型文件: {ckpt_path}，将跳过此模态。")
            continue
        found_modes.append(n_modes)

    for n_modes in tqdm(found_modes, desc="加载模型"):
        ckpt_path = os.path.join(model_dir, f'model_modes_{n_modes}.pth')
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model_config = ckpt.get('model_config', {})

            model = PODDeepONet(
                3,  # branch_input_dim
                model_config.get('hidden_units', 256),
                model_config.get('num_hidden_layers', 4),
                n_modes,
                ckpt['basis'],
                ckpt['y_mean_pod'],
                0.0  # 预测时dropout=0
            ).to(DEVICE)

            model.load_state_dict(ckpt['model_state'])
            model.eval()

            models_dict[n_modes] = {
                'model': model,
                'basis': ckpt['basis'],
                'y_mean_pod': ckpt['y_mean_pod'],
                'scalers': ckpt['scalers'],
                'grid': ckpt['grid']
            }
            # print(f"成功加载模态 {n_modes} 的模型")
        except Exception as e:
            print(f"错误: 加载模态 {n_modes} 的模型失败: {e}")
            continue

    return models_dict


# ==========================================
# 主预测函数
# ==========================================

def predict_and_integrate_fullfield(test_data, models_dict):
    """
    对每个测试样本进行全场预测，并将真实场、预测场、误差场整合到一个CSV文件中。
    """

    if not models_dict:
        print("没有加载任何模型，无法进行预测。")
        return

    # 获取参考网格信息 (从任意一个加载的模型中获取，假定所有模型都使用相同的网格)
    first_model_info = list(models_dict.values())[0]
    grid_a = first_model_info['grid']['a']
    grid_tau = first_model_info['grid']['tau']

    # 构建完整网格的所有点
    GA, GT = np.meshgrid(grid_a, grid_tau)
    full_grid_points = np.column_stack([GA.flatten(), GT.flatten()])
    num_grid_points = len(full_grid_points)

    print(f"预测网格点数量: {num_grid_points}")

    for sample_idx, sample in enumerate(tqdm(test_data, desc="处理测试样本")):
        fname = sample['fname']
        params = np.array([sample['params']])
        df_real_original = sample['df_real']  # 原始真实数据

        # 1. 准备基础DataFrame (A, tau_sec)
        result_df = pd.DataFrame({
            'A': full_grid_points[:, 0],
            'tau_sec': full_grid_points[:, 1]
        })

        # 2. 插值真实场数据到完整网格
        valid_real = df_real_original.dropna(subset=['D1_data', 'D2_data'])
        if len(valid_real) > 0:
            real_points_coords = valid_real[['A', 'tau_sec']].values
            real_d1_values = valid_real['D1_data'].values
            real_d2_values = valid_real['D2_data'].values

            d1_true_grid = griddata(real_points_coords, real_d1_values, full_grid_points, method='linear')
            d2_true_grid = griddata(real_points_coords, real_d2_values, full_grid_points, method='linear')
        else:
            d1_true_grid = np.full(num_grid_points, np.nan)
            d2_true_grid = np.full(num_grid_points, np.nan)

        result_df['D1_true'] = d1_true_grid

        # 3. 对每个模态进行D1预测和误差计算
        for n_modes in sorted(models_dict.keys()):
            model_info = models_dict[n_modes]
            model = model_info['model']
            scalers = model_info['scalers']

            # 标准化输入参数
            input_scaled = manual_scaler_transform_tensor(
                params,
                scalers['b_mean'],
                scalers['b_std'],
                DEVICE
            )

            # 前向预测
            with torch.no_grad():
                pred_scaled = model(input_scaled)
                pred_full = (pred_scaled * torch.tensor(scalers['y_std'], device=DEVICE) + \
                             torch.tensor(scalers['y_mean'], device=DEVICE)).cpu().numpy().flatten()

            # 拆分D1和D2
            mid = len(pred_full) // 2
            d1_pred_flat = pred_full[:mid]

            # 将预测值reshape到网格并展平
            d1_pred_grid = d1_pred_flat.reshape(len(grid_tau), len(grid_a)).flatten()

            # 保存预测值和误差
            result_df[f'D1_pred_mode_{n_modes}'] = d1_pred_grid
            result_df[f'D1_error_mode_{n_modes}'] = np.abs(d1_true_grid - d1_pred_grid)

        # 4. 添加D2真实值
        result_df['D2_true'] = d2_true_grid

        # 5. 对每个模态进行D2预测和误差计算
        for n_modes in sorted(models_dict.keys()):
            model_info = models_dict[n_modes]
            model = model_info['model']
            scalers = model_info['scalers']

            # 标准化输入参数
            input_scaled = manual_scaler_transform_tensor(
                params,
                scalers['b_mean'],
                scalers['b_std'],
                DEVICE
            )

            with torch.no_grad():
                pred_scaled = model(input_scaled)
                pred_full = (pred_scaled * torch.tensor(scalers['y_std'], device=DEVICE) + \
                             torch.tensor(scalers['y_mean'], device=DEVICE)).cpu().numpy().flatten()

            # 拆分D2
            mid = len(pred_full) // 2
            d2_pred_flat = pred_full[mid:]

            # 将预测值reshape到网格并展平
            d2_pred_grid = d2_pred_flat.reshape(len(grid_tau), len(grid_a)).flatten()

            # 保存预测值和误差
            result_df[f'D2_pred_mode_{n_modes}'] = d2_pred_grid
            result_df[f'D2_error_mode_{n_modes}'] = np.abs(d2_true_grid - d2_pred_grid)

        # 6. 保存到CSV文件
        output_fname = f'fullfield_data_{fname}'
        output_path = os.path.join(RESULT_DIR, output_fname)
        result_df.to_csv(output_path, index=False)
        # print(f"  已保存: {output_fname} ({len(result_df)} 个网格点)")

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()


# ==========================================
# 主程序
# ==========================================

def main():
    print("=" * 60)
    print("POD-DeepONet 全场预测程序 (详细整合版)")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 步骤1: 加载测试文件和其真实数据
    print("\n>>> 步骤1: 加载测试文件并缓存真实数据")
    test_data_samples = load_test_files_and_params(TEST_DATA_DIR)

    # 步骤2: 加载模型
    print("\n>>> 步骤2: 加载已训练的模型")
    models_dict = load_models(MODEL_DIR, MODES_TO_PREDICT)

    if not models_dict:
        raise RuntimeError("无法加载任何模型！请检查模型路径和文件名，以及 MODES_TO_PREDICT 列表。")

    print(f"成功加载 {len(models_dict)} 个模型，模态数: {sorted(models_dict.keys())}")

    # 步骤3: 执行全场预测并整合数据
    print("\n>>> 步骤3: 执行全场预测并整合真实场、预测场、误差场数据")
    predict_and_integrate_fullfield(test_data_samples, models_dict)

    print("\n" + "=" * 60)
    print(f"全场预测和数据整合完成！结果已保存到: {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

