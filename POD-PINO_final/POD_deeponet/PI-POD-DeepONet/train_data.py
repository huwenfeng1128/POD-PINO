import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import identity, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from joblib import Parallel, delayed
import traceback
import random

# =========================================================
# 0. 强制 Joblib 临时文件夹使用纯 ASCII 路径
# =========================================================
joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp_deeponet_fixedA_adjoint'

print(f"设置 joblib 临时文件夹为: {joblib_temp_folder}")
try:
    os.makedirs(joblib_temp_folder, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    print("成功设置 JOBLIB_TEMP_FOLDER 环境变量。")
except Exception as e:
    print(f"错误: 无法创建或设置 joblib 临时文件夹 '{joblib_temp_folder}': {e}")
    raise SystemExit(1)

# =========================================================
# 1. 输出目录与配置
# =========================================================
OUTPUT_DIR = r'D:\PINN\zenodo\POD_deeponet\PI-POD-DeepONet\Data_AdjointFP_FixedA'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数采样范围
NU_RANGE = [-20, 20]
KAPPA_RANGE = [0.5, 5.25]
D2_RANGE = [1, 20]
TAU_RANGE = [0.01, 0.5]

# 采样设置
N_PARAM_SETS = 5000
N_TAU_POINTS = 50

# 固定 A 取值
A_MIN_FIXED = 0.0
A_MAX_FIXED = 8.0
N_A_POINTS_FIXED = 80

# 并行
N_JOBS = -1

# PDE 空间网格
FP_N_GRID = 100

# 是否保存完整 P+ 场
SAVE_PPLUS_FIELD = True

# =========================================================
# 2. 理论 D1 / D2
# =========================================================
def theoretical_D1_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    A_safe = np.maximum(A, 1e-9)
    return nu * A - (kappa / 8.0) * A**3 + d_diffusion / A_safe

def theoretical_D2_d(A, nu, kappa, d_diffusion):
    A = np.asarray(A)
    return np.full_like(A, max(0.0, d_diffusion))

def create_grid(A_max_sim_for_grid, N_grid=FP_N_GRID):
    A_grid_max = max(A_max_sim_for._grid * 1.5, A_max_sim_for_grid + 1.0)
    A_grid_min = 0.0
    A_grid = np.linspace(A_grid_min, A_grid_max, N_grid)
    dA = A_grid[1] - A_grid[0] if N_grid > 1 else 0.0
    return A_grid, dA, A_grid_max

# =========================================================
# 3. 伴随 Fokker-Planck 求解器
# =========================================================
class AdjointFPSolver:
    """
    求解伴随 Fokker-Planck 方程：

        ∂P⁺/∂t = L⁺ P⁺

    其中
        L⁺ = D1(a') ∂/∂a' + D2(a') ∂²/∂a'²

    初值
        P⁺_{n,a}(a',0) = (a' - a)^n

    最终
        M_n(a, τ) = P⁺_{n,a}(a, τ)
        D1(a,τ) = M1(a,τ)/τ
        D2(a,τ) = M2(a,τ)/(2τ)
    """
    def __init__(self, A_grid, dA, nu, kappa, d_diffusion):
        self.A_grid = A_grid
        self.dA = dA
        self.N_grid = len(A_grid)
        self.nu = nu
        self.kappa = kappa
        self.d_diffusion = max(1e-15, d_diffusion)
        self.L_adj = None

        if np.isnan(nu) or np.isinf(nu):
            raise ValueError(f"无效 nu={nu}")
        if np.isnan(kappa) or np.isinf(kappa) or kappa <= 0:
            raise ValueError(f"无效 kappa={kappa}")
        if np.isnan(self.d_diffusion) or np.isinf(self.d_diffusion) or self.d_diffusion <= 0:
            raise ValueError(f"无效 d_diffusion={self.d_diffusion}")
        if self.dA <= 0:
            raise ValueError(f"无效 dA={self.dA}")

        self.D1_values = theoretical_D1_d(self.A_grid, nu, kappa, self.d_diffusion)
        self.D2_values = theoretical_D2_d(self.A_grid, nu, kappa, self.d_diffusion)

        self.D1_values = np.nan_to_num(self.D1_values, nan=0.0, posinf=1e10, neginf=-1e10)
        self.D2_values = np.maximum(
            np.nan_to_num(self.D2_values, nan=1e-15, posinf=1e10, neginf=1e-15),
            1e-15
        )

        self.L_adj = self._build_adjoint_operator_matrix()

        if self.L_adj is None or not isinstance(self.L_adj, csc_matrix):
            raise ValueError("伴随算子矩阵 L_adj 构建失败。")
        if not np.all(np.isfinite(self.L_adj.data)):
            raise ValueError("L_adj 中存在 NaN/Inf。")

    def _build_adjoint_operator_matrix(self):
        """
        离散:
            L⁺u = D1(a) u_a + D2(a) u_aa

        内点中心差分:
            u_a   ≈ (u_{i+1} - u_{i-1}) / (2 dA)
            u_aa  ≈ (u_{i+1} - 2u_i + u_{i-1}) / dA²

        边界条件:
            齐次 Neumann:
                ∂u/∂a = 0 at a=0 and a=Amax

        用镜像 ghost-point:
            u_{-1} = u_1, u_N = u_{N-2}
        """
        N = self.N_grid
        dA = self.dA
        dA2 = dA**2
        L = lil_matrix((N, N), dtype=float)

        D1 = self.D1_values
        D2 = self.D2_values

        # 内点
        for i in range(1, N - 1):
            L[i, i - 1] = -D1[i] / (2.0 * dA) + D2[i] / dA2
            L[i, i]     = -2.0 * D2[i] / dA2
            L[i, i + 1] =  D1[i] / (2.0 * dA) + D2[i] / dA2

        # 左边界: u_a = 0
        L[0, 0] = -2.0 * D2[0] / dA2
        L[0, 1] =  2.0 * D2[0] / dA2

        # 右边界: u_a = 0
        L[N - 1, N - 2] =  2.0 * D2[N - 1] / dA2
        L[N - 1, N - 1] = -2.0 * D2[N - 1] / dA2

        return L.tocsc()

    def initial_condition_polynomial(self, A_target, order_n):
        """
        初值:
            P⁺_{n,a}(a',0) = (a' - a)^n
        """
        return (self.A_grid - A_target) ** order_n

    def solve_adjoint_cn(self, A_target, tau, order_n):
        """
        Crank-Nicolson:
            ∂U/∂t = L⁺ U
        """
        if self.L_adj is None:
            return None, "L_adj 未构建"
        if tau <= 0:
            return None, f"无效 tau={tau}"
        if order_n < 0:
            return None, f"无效 order_n={order_n}"

        try:
            dynamic_dt = 0.5 * (self.dA ** 2) / self.d_diffusion
            dynamic_dt = max(1e-9, dynamic_dt)
            dynamic_dt = min(dynamic_dt, tau / 2.0)
        except ZeroDivisionError:
            return None, "无法计算动态 dt：d_diffusion 为零"
        except Exception as e:
            return None, f"无法计算动态 dt：{e}"

        if dynamic_dt <= 0 or np.isnan(dynamic_dt) or np.isinf(dynamic_dt):
            return None, f"动态 dt 无效: {dynamic_dt}"

        num_steps = max(1, int(round(tau / dynamic_dt)))
        actual_dt = tau / num_steps

        U_current = self.initial_condition_polynomial(A_target, order_n)
        if np.any(np.isnan(U_current)) or np.any(np.isinf(U_current)):
            return None, "初值存在 NaN/Inf"

        Id = identity(self.N_grid, format='csc')

        try:
            LHS = Id - 0.5 * actual_dt * self.L_adj
            RHS = Id + 0.5 * actual_dt * self.L_adj
            if not np.all(np.isfinite(LHS.data)) or not np.all(np.isfinite(RHS.data)):
                raise ValueError("LHS/RHS 中存在 NaN/Inf")
        except Exception as e:
            return None, f"无法构造 CN 矩阵: {e}"

        try:
            for step in range(num_steps):
                b = RHS.dot(U_current)
                if np.any(np.isnan(b)) or np.any(np.isinf(b)):
                    return None, f"RHS 在 step={step+1} 出现 NaN/Inf"

                U_next = spsolve(LHS, b)

                if np.any(np.isnan(U_next)) or np.any(np.isinf(U_next)):
                    return None, f"求解结果在 step={step+1} 出现 NaN/Inf"

                U_current = U_next
        except Exception as e:
            return None, f"CN 时间推进失败: {e}"

        return U_current, None

    def evaluate_at_target(self, U_final, A_target):
        if U_final is None:
            return np.nan
        return np.interp(A_target, self.A_grid, U_final)

    def calculate_km_from_adjoint_solution(self, U1_final, U2_final, A_target, tau):
        """
        M1 = P⁺_{1,a}(a,τ)
        M2 = P⁺_{2,a}(a,τ)

        D1(a,τ) = M1 / τ
        D2(a,τ) = M2 / (2τ)
        """
        if tau <= 0:
            return np.nan, np.nan

        M1 = self.evaluate_at_target(U1_final, A_target)
        M2 = self.evaluate_at_target(U2_final, A_target)

        if np.isnan(M1) or np.isnan(M2):
            return np.nan, np.nan

        D1_adj = M1 / tau
        D2_adj = M2 / (2.0 * tau)

        if not np.isnan(D2_adj):
            D2_adj = max(0.0, D2_adj)

        return D1_adj, D2_adj

# =========================================================
# 4. 参数与 τ 固定取样
# =========================================================
def sample_parameters(n_sets, nu_range, kappa_range, d2_range):
    sampled_params = []
    for _ in range(n_sets):
        nu = random.uniform(nu_range[0], nu_range[1])
        kappa = random.uniform(kappa_range[0], kappa_range[1])
        kappa = max(1e-6, kappa)
        d2 = random.uniform(d2_range[0], d2_range[1])
        d2 = max(1e-6, d2)
        sampled_params.append({
            'nu': nu,
            'kappa': kappa,
            'd_diffusion': d2
        })
    return sampled_params

def get_tau_values(n_points, tau_range):
    return np.linspace(tau_range[0], tau_range[1], n_points)

# =========================================================
# 5. 单个参数集处理
# =========================================================
def process_parameter_set(params, tau_values_global, a_values_fixed_global, output_dir):
    """
    单个参数集处理：
      - A 和 tau 均为固定采样
      - 求解伴随 FP
      - 保存:
          1) csv: D1_adj, D2_adj
          2) npz: Pplus_n1, Pplus_n2
    """
    nu = params['nu']
    kappa = params['kappa']
    d_diffusion = params['d_diffusion']

    param_id_str = (
        f"nu_{nu:.3f}_kappa_{kappa:.3f}_d_{d_diffusion:.3f}"
        .replace('.', '_')
        .replace('-', 'neg')
    )

    csv_filename = os.path.join(output_dir, f"data_{param_id_str}.csv")
    npz_filename = os.path.join(output_dir, f"pplus_{param_id_str}.npz")

    if os.path.exists(csv_filename) and os.path.exists(npz_filename):
        return f"跳过 {param_id_str}"

    results_list = []
    status = f"未知错误 {param_id_str}"

    try:
        # ---------- 固定 A ----------
        A_values_to_use = np.asarray(a_values_fixed_global, dtype=float)
        tau_values_to_use = np.asarray(tau_values_global, dtype=float)

        num_a_points_actual = len(A_values_to_use)
        num_tau_points_actual = len(tau_values_to_use)

        # ---------- PDE 网格 ----------
        A_max_sim_for_grid = A_values_to_use[-1]
        A_grid, dA, _ = create_grid(A_max_sim_for_grid, N_grid=FP_N_GRID)

        solver = AdjointFPSolver(A_grid, dA, nu, kappa, d_diffusion)

        # ---------- 预分配 P+ ----------
        # shape = (N_tau, N_A_target, N_grid)
        Pplus_n1_all = np.full(
            (num_tau_points_actual, num_a_points_actual, len(A_grid)),
            np.nan,
            dtype=np.float32
        )
        Pplus_n2_all = np.full(
            (num_tau_points_actual, num_a_points_actual, len(A_grid)),
            np.nan,
            dtype=np.float32
        )

        solve_start_time = time.time()
        calculation_count = 0
        failed_solves = 0

        for i_tau, tau_sec in enumerate(tau_values_to_use):
            for i_a, A_target in enumerate(A_values_to_use):
                try:
                    U1_final, err1 = solver.solve_adjoint_cn(A_target, tau_sec, order_n=1)
                    U2_final, err2 = solver.solve_adjoint_cn(A_target, tau_sec, order_n=2)

                    if (U1_final is None) or (U2_final is None):
                        D1_adj, D2_adj = np.nan, np.nan
                        failed_solves += 1
                    else:
                        D1_adj, D2_adj = solver.calculate_km_from_adjoint_solution(
                            U1_final, U2_final, A_target, tau_sec
                        )

                        if SAVE_PPLUS_FIELD:
                            Pplus_n1_all[i_tau, i_a, :] = U1_final.astype(np.float32)
                            Pplus_n2_all[i_tau, i_a, :] = U2_final.astype(np.float32)

                    results_list.append({
                        'nu': nu,
                        'kappa': kappa,
                        'd_diffusion': d_diffusion,
                        'tau': tau_sec,
                        'A': A_target,
                        'D1_adj': D1_adj,
                        'D2_adj': D2_adj
                    })

                    calculation_count += 1

                except Exception:
                    results_list.append({
                        'nu': nu,
                        'kappa': kappa,
                        'd_diffusion': d_diffusion,
                        'tau': tau_sec,
                        'A': A_target,
                        'D1_adj': np.nan,
                        'D2_adj': np.nan
                    })
                    failed_solves += 1
                    calculation_count += 1

        solve_duration = time.time() - solve_start_time

        # ---------- 保存 csv ----------
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(csv_filename, index=False)

            # ---------- 保存 P+ ----------
            if SAVE_PPLUS_FIELD:
                np.savez_compressed(
                    npz_filename,
                    nu=np.array([nu], dtype=np.float64),
                    kappa=np.array([kappa], dtype=np.float64),
                    d_diffusion=np.array([d_diffusion], dtype=np.float64),
                    tau_values=tau_values_to_use.astype(np.float32),
                    A_targets=A_values_to_use.astype(np.float32),
                    A_grid=A_grid.astype(np.float32),
                    Pplus_n1=Pplus_n1_all,
                    Pplus_n2=Pplus_n2_all
                )

            status = (
                f"保存 {param_id_str} "
                f"({num_tau_points_actual} tau点, {num_a_points_actual} A点, "
                f"{calculation_count} 总点, {failed_solves} 失败, {solve_duration:.1f}s)"
            )
        else:
            status = f"无结果 {param_id_str}"

    except ValueError as e:
        status = f"失败 ValueErr {param_id_str}: {e}"
    except Exception:
        traceback.print_exc()
        status = f"失败 Other {param_id_str}"

    return status

# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    print("--- DeepONet 数据生成脚本（伴随 Fokker-Planck，固定 A，固定 tau，动态 dt）---")

    # 1. 参数采样
    print(f"\n采样 {N_PARAM_SETS} 个参数集...")
    parameter_sets = sample_parameters(N_PARAM_SETS, NU_RANGE, KAPPA_RANGE, D2_RANGE)
    print(f"采样了 {len(parameter_sets)} 个参数集。")

    # 2. 固定 tau
    tau_values = get_tau_values(N_TAU_POINTS, TAU_RANGE)
    print(f"\n生成了 {len(tau_values)} 个 tau 值，范围在 {TAU_RANGE[0]} 到 {TAU_RANGE[1]} 之间。")

    # 3. 固定 A
    a_values_fixed = np.linspace(A_MIN_FIXED, A_MAX_FIXED, N_A_POINTS_FIXED)
    print(f"将使用固定的 {N_A_POINTS_FIXED} 个 A 值，范围在 {A_MIN_FIXED} 到 {A_MAX_FIXED} 之间。")
    print(f"FP 网格点数 N_grid = {FP_N_GRID}。")
    print("方程类型：伴随 Fokker-Planck。")
    print("时间推进：Crank–Nicolson。")
    print("边界条件：齐次 Neumann。")
    print("动态时间步长：dt = 0.5 * dA^2 / D^(2)。")
    print(f"保存完整 P+ 场：{SAVE_PPLUS_FIELD}")

    # 4. 并行
    print(f"\n开始并行计算，使用 {N_JOBS if N_JOBS > 0 else '所有'} 个核心...")
    total_start_time = time.time()

    results_status = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_parameter_set)(params, tau_values, a_values_fixed, OUTPUT_DIR)
        for params in tqdm(parameter_sets, desc="分发参数集任务")
    )

    total_end_time = time.time()
    print("\n--- 并行计算完成 ---")
    print(f"总时间: {total_end_time - total_start_time:.2f} 秒。")

    # 5. 汇总
    print("\n处理摘要:")
    saved_count = sum(1 for s in results_status if s.startswith("保存"))
    skipped_count = sum(1 for s in results_status if s.startswith("跳过"))
    failed_val_count = sum(1 for s in results_status if s.startswith("失败 ValueErr"))
    failed_oth_count = sum(1 for s in results_status if s.startswith("失败 Other"))
    no_results_count = sum(1 for s in results_status if s.startswith("无结果"))

    print(f"  成功生成并保存: {saved_count}/{len(parameter_sets)}")
    print(f"  跳过 (已存在):   {skipped_count}/{len(parameter_sets)}")
    print(f"  失败 (参数/值错误): {failed_val_count}/{len(parameter_sets)}")
    print(f"  失败 (其他错误): {failed_oth_count}/{len(parameter_sets)}")
    print(f"  未生成结果:      {no_results_count}/{len(parameter_sets)}")

    print("\n部分详细状态:")
    count = 0
    for s in results_status:
        if count < 10 or not s.startswith("跳过"):
            print(f"  - {s}")
        if not s.startswith("跳过"):
            count += 1

    print(f"\n生成的数据保存在: {OUTPUT_DIR}")
    print("脚本完成。")