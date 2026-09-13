
import os
import glob
import time
import numpy as np
import pandas as pd
import torch # Keep for scaler saving if needed later
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import identity, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapezoid as trapz
from tqdm import tqdm
import math
from joblib import Parallel, delayed
import traceback
import random

# --- START: 强制 Joblib 临时文件夹使用纯 ASCII 路径 ---
joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp_deeponet_dynamicA' # 保持不变或更改

print(f"设置 joblib 临时文件夹为: {joblib_temp_folder}")
try:
    os.makedirs(joblib_temp_folder, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    print(f"成功设置 JOBLIB_TEMP_FOLDER 环境变量。")
except Exception as e:
    print(f"错误: 无法创建或设置 joblib 临时文件夹 '{joblib_temp_folder}': {e}")
    exit()
# --- END: 强制 Joblib 临时文件夹 ---


# --- DeepONet 数据生成配置 ---
OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\data\Data'  # 更新输出目录名以反映 A 的变化
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数采样范围
NU_RANGE = [-20, 20]
KAPPA_RANGE = [0.5, 5.25]
D2_RANGE = [1, 20]
TAU_RANGE = [0.01, 0.5]

# 采样设置
N_PARAM_SETS = 5000   # 要采样的 (nu, kappa, d_diffusion) 组合数量 (可增加到 2900)
N_TAU_POINTS = 50    # 要采样的 tau 点数量

# --- 修改: 固定的 A 值设置 ---
A_MIN_FIXED = 0.0
A_MAX_FIXED = 8.0
N_A_POINTS_FIXED = 80 # A 的固定点数

# P_inf 和 A 值选择设置 (这些现在大部分将不再使用，但保留以防万一)
A_EVAL_GRID_SIZE = 2000
A_EVAL_MAX_FACTOR = 5.0
A_EVAL_MIN_GUESS = 1e-4
A_THRESHOLD_RATIO = 0.1
A_MIN_SELECTED = 1e-3
# N_A_POINTS 不再需要，将动态计算

# 并行处理设置
N_JOBS = -1

# Fokker-Planck 求解器设置
FP_N_GRID = 100     # 空间网格点

# --- 辅助函数 (理论 KM 和网格) (与之前相同) ---
def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    A_safe = np.maximum(A, 1e-9)
    term_gamma = d_diffusion / A_safe
    term_nu = nu * A
    term_kappa = (kappa / 8.0) * A**3
    return term_nu - term_kappa + term_gamma

def theoretical_D2_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    return np.full_like(A, max(0.0, d_diffusion))

def create_grid(A_max_sim_for_grid, N_grid=FP_N_GRID):
    A_grid_max = max(A_max_sim_for_grid * 1.5, A_max_sim_for_grid + 1.0)
    A_grid_min = 0
    A_grid = np.linspace(A_grid_min, A_grid_max, N_grid)
    dA = A_grid[1] - A_grid[0] if N_grid > 1 else 0
    return A_grid, dA, A_grid_max

# --- FPSolver 类 (使用动态时间步长) (与之前相同) ---
class FPSolver:
    def __init__(self, A_grid, dA, nu, kappa, d_diffusion):
        self.A_grid = A_grid
        self.dA = dA
        self.N_grid = len(A_grid)
        self.nu = nu
        self.kappa = kappa
        self.d_diffusion = max(1e-15, d_diffusion)
        self.L = None

        if np.isnan(nu) or np.isinf(nu) or \
           np.isnan(kappa) or np.isinf(kappa) or kappa <= 0 or \
           np.isnan(self.d_diffusion) or np.isinf(self.d_diffusion) or self.d_diffusion <= 0:
             raise ValueError(f"FPSolver 参数无效: nu={nu}, k={kappa}, d={self.d_diffusion}")
        if self.dA <= 0:
             raise ValueError(f"无效网格间距 dA={self.dA}")

        try:
            self.D1_values = theoretical_D1_d(self.A_grid, nu, kappa, self.d_diffusion)
            self.D2_values = theoretical_D2_d(self.A_grid, nu, kappa, self.d_diffusion)
            self.D2_values = np.maximum(self.D2_values, 1e-15)
        except Exception as e:
            raise ValueError(f"无法计算有效 D1/D2: {e}") from e

        try:
            self.L = self._build_forward_operator_matrix()
            if self.L is None or not isinstance(self.L, csc_matrix):
                 raise ValueError("算子矩阵 L 未正确构建。")
            if not np.all(np.isfinite(self.L.data)):
                 nan_inf_count = np.sum(~np.isfinite(self.L.data))
                 allowed_bad_values = max(5, self.N_grid * 0.01)
                 if nan_inf_count > allowed_bad_values:
                    raise ValueError(f"在算子矩阵 L 数据中检测到 {nan_inf_count} 个 NaN/Inf。")
                 else:
                    self.L.data[~np.isfinite(self.L.data)] = 0.0
                    self.L.eliminate_zeros()
        except Exception as e:
            raise ValueError(f"无法构建算子矩阵 L: {e}") from e

    def _build_forward_operator_matrix(self):
        N = self.N_grid
        dA = self.dA
        dA2 = dA**2
        L = lil_matrix((N, N), dtype=float)
        safe_D1 = np.nan_to_num(self.D1_values, nan=0.0, posinf=1e10, neginf=-1e10)
        safe_D2 = self.D2_values
        dD1_dA = np.gradient(safe_D1, dA, edge_order=1)
        dD2_dA = np.gradient(safe_D2, dA, edge_order=1)
        d2D2_dA2 = np.gradient(dD2_dA, dA, edge_order=1)
        dD1_dA = np.nan_to_num(dD1_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        dD2_dA = np.nan_to_num(dD2_dA, nan=0.0, posinf=1e10, neginf=-1e10)
        d2D2_dA2 = np.nan_to_num(d2D2_dA2, nan=0.0, posinf=1e10, neginf=-1e10)
        coeff_p = -(dD1_dA - d2D2_dA2)
        coeff_dp_dA = -safe_D1 + 2 * dD2_dA
        coeff_d2p_dA2 = safe_D2
        for i in range(1, N - 1):
            L[i, i-1] = -coeff_dp_dA[i] / (2.0 * dA) + coeff_d2p_dA2[i] / dA2
            L[i, i]   = coeff_p[i] - 2.0 * coeff_d2p_dA2[i] / dA2
            L[i, i+1] = coeff_dp_dA[i] / (2.0 * dA) + coeff_d2p_dA2[i] / dA2
        L[0, :] = 0; L[0, 0] = 0
        L[N-1, :] = 0; L[N-1, N-1] = 0
        return L.tocsc()

    def initial_condition_delta(self, A_target):
        p0 = np.zeros(self.N_grid)
        idx = np.argmin(np.abs(self.A_grid - A_target))
        idx = max(0, min(self.N_grid - 1, idx))
        if self.dA > 1e-15: p0[idx] = 1.0 / self.dA
        else: p0[idx] = 1.0
        return p0

    def solve_forward_cn(self, A_target, tau):
        if self.L is None: return None, "算子 L 未构建"
        if tau <= 0: return None, f"无效的 tau: {tau}"
        try:
            dynamic_dt = 0.5 * (self.dA**2) / self.d_diffusion
            dynamic_dt = max(1e-9, dynamic_dt)
            dynamic_dt = min(dynamic_dt, tau / 2.0) # 至少两步
        except ZeroDivisionError: return None, f"无法计算动态 dt：d_diffusion 为零"
        except AttributeError: return None, "无法计算动态 dt：缺少 dA 或 d_diffusion"
        if dynamic_dt <= 0 or np.isnan(dynamic_dt) or np.isinf(dynamic_dt):
            return None, f"计算出的动态 dt 无效: {dynamic_dt}"

        num_steps = max(1, int(round(tau / dynamic_dt)))
        actual_dt_step = tau / num_steps
        P_current = self.initial_condition_delta(A_target)
        if np.sum(P_current) < 1e-10: return None, f"初始条件在 A={A_target:.3f} 附近为零"

        Id = identity(self.N_grid, format='csc')
        try:
            LHS = Id - 0.5 * actual_dt_step * self.L
            RHS = Id + 0.5 * actual_dt_step * self.L
            if not np.all(np.isfinite(LHS.data)) or not np.all(np.isfinite(RHS.data)):
                raise ValueError("LHS/RHS 矩阵中含 NaN/Inf")
        except Exception as e: return None, f"无法构建 LHS/RHS: {e}"

        try:
            for step in range(num_steps):
                b = RHS.dot(P_current)
                if np.isnan(b).any() or np.isinf(b).any():
                    return None, f"NaN/Inf 在 RHS 'b' 步骤 {step+1}, A={A_target:.3f}, dt={actual_dt_step:.2e}"
                P_next = spsolve(LHS, b)
                if np.isnan(P_next).any() or np.isinf(P_next).any():
                    return None, f"NaN/Inf 在求解器步骤 {step+1}, A={A_target:.3f}, dt={actual_dt_step:.2e}"
                P_current = P_next
        except np.linalg.LinAlgError as e: return None, f"LinAlgError 步骤 {step+1}: {e}, dt={actual_dt_step:.2e}"
        except Exception as e: return None, f"求解步骤 {step+1} 出错: {e}, dt={actual_dt_step:.2e}"

        P_final = np.maximum(P_current, 0)
        integral_P_final = trapz(P_final, self.A_grid)
        if integral_P_final > 1e-10: P_final /= integral_P_final
        else: P_final = np.zeros_like(P_final)
        return P_final, None

    def calculate_km_from_solution(self, P_final, A_target, tau):
        if P_final is None or tau <= 0: return np.nan, np.nan
        if np.sum(P_final) < 1e-10: return 0.0, 0.0
        M1 = trapz((self.A_grid - A_target) * P_final, self.A_grid)
        M2 = trapz((self.A_grid - A_target)**2 * P_final, self.A_grid)
        if tau > 1e-15:
            D1_fp = M1 / (1.0 * tau); D2_fp = M2 / (2.0 * tau)
        else: D1_fp, D2_fp = np.nan, np.nan
        D2_fp = max(0.0, D2_fp) if not np.isnan(D2_fp) else np.nan
        return D1_fp, D2_fp

# --- 数据生成函数 ---

def sample_parameters(n_sets, nu_range, kappa_range, d2_range):
    sampled_params = []
    for _ in range(n_sets):
        nu = random.uniform(nu_range[0], nu_range[1])
        kappa = random.uniform(kappa_range[0], kappa_range[1])
        kappa = max(1e-6, kappa)
        d2 = random.uniform(d2_range[0], d2_range[1])
        d2 = max(1e-6, d2)
        sampled_params.append({'nu': nu, 'kappa': kappa, 'd_diffusion': d2})
    return sampled_params

def get_tau_values(n_points, tau_range):
    return np.linspace(tau_range[0], tau_range[1], n_points)

def calculate_stationary_pdf(a_eval_grid, nu, kappa, d_diffusion):
    if d_diffusion <= 0 or kappa <= 0: raise ValueError("d_diffusion/kappa 必须为正")
    exponent = (nu / (2.0 * d_diffusion)) * a_eval_grid**2 - (kappa / (32.0 * d_diffusion)) * a_eval_grid**4
    max_exponent = np.max(exponent); exponent = np.clip(exponent, max_exponent - 70, max_exponent + 10)
    P_inf_unnormalized = a_eval_grid * np.exp(exponent)
    P_inf_unnormalized = np.nan_to_num(P_inf_unnormalized, nan=0.0, posinf=0.0, neginf=0.0)
    integral_Z = trapz(P_inf_unnormalized, a_eval_grid)
    if integral_Z < 1e-15: return P_inf_unnormalized, integral_Z
    P_inf_normalized = P_inf_unnormalized / integral_Z
    return P_inf_normalized, integral_Z

# --- 移除 get_dynamic_A_values 函数，因为我们现在使用固定的 A 值 ---

# --- 修改: process_parameter_set 不再接收 da_step，而是使用固定的 A 值 ---
def process_parameter_set(params, tau_values_global, a_values_fixed_global, output_dir):
    """
    处理单个参数集。使用固定的 A 值范围，求解 AFP 并保存。
    """
    nu = params['nu']
    kappa = params['kappa']
    d_diffusion = params['d_diffusion']
    param_id_str = f"nu_{nu:.3f}_kappa_{kappa:.3f}_d_{d_diffusion:.3f}".replace('.', '_').replace('-', 'neg')
    output_filename = os.path.join(output_dir, f"data_{param_id_str}.csv")

    if os.path.exists(output_filename):
        return f"跳过 {param_id_str}"

    results_list = []
    solver = None
    status = f"未知错误 {param_id_str}"

    try:
        # --- 1. 使用固定的 A 值 ---
        A_values_to_use = a_values_fixed_global
        num_a_points_actual = len(A_values_to_use)

        # --- 2. 设置 FP 求解器 ---
        # A_max_sim_for_grid 应该基于我们使用的 A_values_to_use 的最大值
        A_max_sim_for_grid = A_values_to_use[-1]
        A_grid, dA, _ = create_grid(A_max_sim_for_grid, N_grid=FP_N_GRID)
        solver = FPSolver(A_grid, dA, nu, kappa, d_diffusion)

        # --- 3. 循环计算 KM 系数 ---
        solve_start_time = time.time()
        calculation_count = 0
        failed_solves = 0

        for tau_sec in tau_values_global:
            for A_target in A_values_to_use:
                try:
                    P_final, error_msg = solver.solve_forward_cn(A_target, tau_sec)
                    if P_final is None:
                        D1_fp, D2_fp = np.nan, np.nan
                        failed_solves += 1
                    else:
                        D1_fp, D2_fp = solver.calculate_km_from_solution(P_final, A_target, tau_sec)

                    results_list.append({
                        'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
                        'tau': tau_sec, 'A': A_target, 'D1_fp': D1_fp, 'D2_fp': D2_fp
                    })
                    calculation_count += 1
                except Exception as e_inner:
                    results_list.append({
                        'nu': nu, 'kappa': kappa, 'd_diffusion': d_diffusion,
                        'tau': tau_sec, 'A': A_target, 'D1_fp': np.nan, 'D2_fp': np.nan
                    })
                    failed_solves += 1
                    calculation_count += 1

        solve_duration = time.time() - solve_start_time

        # --- 4. 保存结果 ---
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(output_filename, index=False)
            status = f"保存 {param_id_str} ({num_a_points_actual} A点, {calculation_count} 总点, {failed_solves} 失败, {solve_duration:.1f}s)"
        else:
            status = f"无结果 {param_id_str}"

    except ValueError as e_outer: status = f"失败 ValueErr {param_id_str}: {e_outer}"
    except Exception as e_outer:
        traceback.print_exc(); status = f"失败 Other {param_id_str}"

    return status


# --- 主执行逻辑 ---
if __name__ == "__main__":
    print("--- DeepONet 数据生成脚本 (固定 A 范围, 动态 DT) ---")

    # 1. 采样参数
    print(f"\n采样 {N_PARAM_SETS} 个参数集...")
    parameter_sets = sample_parameters(N_PARAM_SETS, NU_RANGE, KAPPA_RANGE, D2_RANGE)
    print(f"采样了 {len(parameter_sets)} 个参数集。")

    # 2. 定义 Tau 值
    tau_values = get_tau_values(N_TAU_POINTS, TAU_RANGE)
    print(f"\n生成了 {len(tau_values)} 个 tau 值，范围在 {TAU_RANGE[0]} 到 {TAU_RANGE[1]} 之间。")

    # --- 新增: 定义固定的 A 值 ---
    a_values_fixed = np.linspace(A_MIN_FIXED, A_MAX_FIXED, N_A_POINTS_FIXED)
    print(f"将使用固定的 {N_A_POINTS_FIXED} 个 A 值，范围在 {A_MIN_FIXED} 到 {A_MAX_FIXED} 之间。")
    print(f"FP 网格点数 N_grid = {FP_N_GRID}。")
    print(f"将为每次求解动态计算时间步长 dt = 0.5 * da^2 / D(2)。")

    # 3. 为每个参数集并行运行 AFP 求解器
    print(f"\n开始并行 AFP 计算，使用 {N_JOBS if N_JOBS > 0 else '所有'} 个核心...")
    total_start_time = time.time()

    # --- 修改: 传递固定的 A 值而不是 da_step ---
    results_status = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_parameter_set)(params, tau_values, a_values_fixed, OUTPUT_DIR)
        for params in tqdm(parameter_sets, desc="分发参数集任务")
    )

    total_end_time = time.time()
    print(f"\n--- 并行计算完成 ---")
    print(f"总时间: {total_end_time - total_start_time:.2f} 秒。")

    # 4. 总结结果
    print("\n处理摘要:")
    saved_count = sum(1 for s in results_status if s.startswith("保存"))
    skipped_count = sum(1 for s in results_status if s.startswith("跳过"))
    failed_pinf_count = sum(1 for s in results_status if s.startswith("失败 P_inf")) # 这个计数现在可能不那么相关，因为 P_inf 不再用于 A 的选择
    failed_val_count = sum(1 for s in results_status if s.startswith("失败 ValueErr"))
    failed_oth_count = sum(1 for s in results_status if s.startswith("失败 Other"))
    no_results_count = sum(1 for s in results_status if s.startswith("无结果"))
    print(f"  成功生成并保存: {saved_count}/{len(parameter_sets)}")
    print(f"  跳过 (已存在):      {skipped_count}/{len(parameter_sets)}")
    print(f"  失败 (P_inf 计算):  {failed_pinf_count}/{len(parameter_sets)}")
    print(f"  失败 (参数/值错误): {failed_val_count}/{len(parameter_sets)}")
    print(f"  失败 (其他错误):    {failed_oth_count}/{len(parameter_sets)}")
    print(f"  未生成结果:       {no_results_count}/{len(parameter_sets)}")
    # 打印一些详细状态
    print("\n部分详细状态:")
    count = 0
    for s in results_status:
        if count < 10 or not s.startswith("跳过"):
             print(f"  - {s}")
        if not s.startswith("跳过"): count +=1


    print(f"\n生成的数据保存在: {OUTPUT_DIR}")
    print("脚本完成。")