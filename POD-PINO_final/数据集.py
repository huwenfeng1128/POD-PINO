import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import identity, csc_matrix, lil_matrix
from scipy.sparse.linalg import splu
from scipy.integrate import trapezoid as trapz
from tqdm import tqdm
import math
from joblib import Parallel, delayed
import traceback
import random


# --- START: 强制 Joblib 临时文件夹使用纯 ASCII 路径 ---
joblib_temp_folder = r'D:\PINN\zenodo\AFP\joblib_temp_deeponet_dynamicA'

print(f"设置 joblib 临时文件夹为: {joblib_temp_folder}")
try:
    os.makedirs(joblib_temp_folder, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_folder
    print("成功设置 JOBLIB_TEMP_FOLDER 环境变量。")
except Exception as e:
    print(f"错误: 无法创建或设置 joblib 临时文件夹 '{joblib_temp_folder}': {e}")
    exit()
# --- END: 强制 Joblib 临时文件夹 ---


# --- DeepONet 数据生成配置 ---
OUTPUT_DIR = r'E:\数据集'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数采样范围
NU_RANGE = [-20000, 20000]
KAPPA_RANGE = [-20001, 20001]
D2_RANGE = [0.0001, 20]
TAU_RANGE = [0.01, 0.5]

# 采样设置
N_PARAM_SETS = 500000
N_TAU_POINTS = 50

# 固定 A 值设置
A_MIN_FIXED = 0.0
A_MAX_FIXED = 10.0
N_A_POINTS_FIXED = 100

# 需要计算的阶数 n
# P_n^\dagger(a, 0) = (a - a')^n
MOMENT_ORDERS = [1, 2]

# 并行处理设置
N_JOBS = 20

# Fokker-Planck / AFP 求解器设置
FP_N_GRID = 100


# --- 辅助函数 ---
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
    A_grid_min = 0.0

    A_grid = np.linspace(A_grid_min, A_grid_max, N_grid)
    dA = A_grid[1] - A_grid[0] if N_grid > 1 else 0.0

    return A_grid, dA, A_grid_max


# --- AFP Solver 类 ---
class FPSolver:
    """
    后向 / adjoint Fokker-Planck 方程：

        ∂P_n^\dagger / ∂τ = D1(a) ∂P_n^\dagger / ∂a
                            + D2(a) ∂²P_n^\dagger / ∂a²

    初始条件：

        P_n^\dagger(a, 0) = (a - a')^n

    其中：
        a  -> A_grid
        a' -> A_target
    """

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
            raise ValueError(
                f"FPSolver 参数无效: nu={nu}, kappa={kappa}, d={self.d_diffusion}"
            )

        if self.dA <= 0:
            raise ValueError(f"无效网格间距 dA={self.dA}")

        try:
            self.D1_values = theoretical_D1_d(
                self.A_grid, nu, kappa, self.d_diffusion
            )
            self.D2_values = theoretical_D2_d(
                self.A_grid, nu, kappa, self.d_diffusion
            )
            self.D2_values = np.maximum(self.D2_values, 1e-15)
        except Exception as e:
            raise ValueError(f"无法计算有效 D1/D2: {e}") from e

        try:
            self.L = self._build_backward_operator_matrix()

            if self.L is None or not isinstance(self.L, csc_matrix):
                raise ValueError("算子矩阵 L 未正确构建。")

            if not np.all(np.isfinite(self.L.data)):
                nan_inf_count = np.sum(~np.isfinite(self.L.data))
                allowed_bad_values = max(5, self.N_grid * 0.01)

                if nan_inf_count > allowed_bad_values:
                    raise ValueError(
                        f"在算子矩阵 L 数据中检测到 {nan_inf_count} 个 NaN/Inf。"
                    )
                else:
                    self.L.data[~np.isfinite(self.L.data)] = 0.0
                    self.L.eliminate_zeros()

        except Exception as e:
            raise ValueError(f"无法构建算子矩阵 L: {e}") from e

    def _build_backward_operator_matrix(self):
        """
        构建后向 / adjoint Fokker-Planck 算子：

            L P = D1(a) * dP/da + D2(a) * d²P/da²

        中心差分：

            dP/da   ≈ (P[i+1] - P[i-1]) / (2 dA)
            d²P/da² ≈ (P[i-1] - 2P[i] + P[i+1]) / dA²
        """

        N = self.N_grid
        dA = self.dA
        dA2 = dA**2

        L = lil_matrix((N, N), dtype=float)

        safe_D1 = np.nan_to_num(
            self.D1_values,
            nan=0.0,
            posinf=1e10,
            neginf=-1e10
        )

        safe_D2 = np.nan_to_num(
            self.D2_values,
            nan=1e-15,
            posinf=1e10,
            neginf=1e-15
        )

        for i in range(1, N - 1):
            D1_i = safe_D1[i]
            D2_i = safe_D2[i]

            L[i, i - 1] = -D1_i / (2.0 * dA) + D2_i / dA2
            L[i, i]     = -2.0 * D2_i / dA2
            L[i, i + 1] =  D1_i / (2.0 * dA) + D2_i / dA2

        # 边界行设为 0，相当于边界值在时间推进中保持初始值
        L[0, :] = 0.0
        L[0, 0] = 0.0

        L[N - 1, :] = 0.0
        L[N - 1, N - 1] = 0.0

        return L.tocsc()

    def build_initial_condition_matrix(self, A_targets, moment_orders):
        """
        批量构造初始条件矩阵。

        对每个 A_target 和 n：

            P_n^\dagger(a, 0) = (a - A_target)^n

        返回矩阵形状：

            (N_grid, len(moment_orders) * len(A_targets))

        列顺序为：

            n=moment_orders[0] 的所有 A_target,
            n=moment_orders[1] 的所有 A_target,
            ...
        """

        A_targets = np.asarray(A_targets)
        initial_blocks = []

        for n in moment_orders:
            block = (self.A_grid[:, None] - A_targets[None, :]) ** n
            initial_blocks.append(block)

        P0 = np.concatenate(initial_blocks, axis=1)

        return P0

    def solve_adjoint_cn_batch(self, A_targets, tau, moment_orders):
        """
        批量求解：

            ∂P_n^\dagger / ∂τ = L P_n^\dagger

        初始条件：

            P_n^\dagger(a, 0) = (a - A_target)^n

        这个函数相比逐个 A_target、逐个 n 求解，不改变数值格式，
        只是把多个右端项合并成矩阵一起求解，并复用同一个 LU 分解。
        """

        if self.L is None:
            return None, "算子 L 未构建"

        if tau <= 0:
            return None, f"无效的 tau: {tau}"

        if any(n < 0 for n in moment_orders):
            return None, f"无效的 moment order: {moment_orders}"

        try:
            dynamic_dt = 0.5 * (self.dA**2) / self.d_diffusion
            dynamic_dt = max(1e-9, dynamic_dt)
            dynamic_dt = min(dynamic_dt, tau / 2.0)
        except ZeroDivisionError:
            return None, "无法计算动态 dt：d_diffusion 为零"
        except AttributeError:
            return None, "无法计算动态 dt：缺少 dA 或 d_diffusion"

        if dynamic_dt <= 0 or np.isnan(dynamic_dt) or np.isinf(dynamic_dt):
            return None, f"计算出的动态 dt 无效: {dynamic_dt}"

        num_steps = max(1, int(round(tau / dynamic_dt)))
        actual_dt_step = tau / num_steps

        P_current = self.build_initial_condition_matrix(A_targets, moment_orders)

        if np.isnan(P_current).any() or np.isinf(P_current).any():
            return None, f"初始条件矩阵含 NaN/Inf, tau={tau:.6f}"

        Id = identity(self.N_grid, format='csc')

        try:
            LHS = Id - 0.5 * actual_dt_step * self.L
            RHS = Id + 0.5 * actual_dt_step * self.L

            if not np.all(np.isfinite(LHS.data)) or not np.all(np.isfinite(RHS.data)):
                raise ValueError("LHS/RHS 矩阵中含 NaN/Inf")

            # 关键加速点：
            # 对同一个 tau、同一个 actual_dt_step，LHS 不变。
            # 因此只做一次 LU 分解，后续每一步复用。
            lu = splu(LHS)

        except Exception as e:
            return None, f"无法构建或分解 LHS/RHS: {e}"

        try:
            for step in range(num_steps):
                B = RHS.dot(P_current)

                if np.isnan(B).any() or np.isinf(B).any():
                    return None, (
                        f"NaN/Inf 在 RHS B 步骤 {step + 1}, "
                        f"tau={tau:.6f}, dt={actual_dt_step:.2e}"
                    )

                P_next = lu.solve(B)

                if np.isnan(P_next).any() or np.isinf(P_next).any():
                    return None, (
                        f"NaN/Inf 在 LU 求解步骤 {step + 1}, "
                        f"tau={tau:.6f}, dt={actual_dt_step:.2e}"
                    )

                P_current = P_next

        except np.linalg.LinAlgError as e:
            return None, (
                f"LinAlgError 步骤 {step + 1}: {e}, "
                f"dt={actual_dt_step:.2e}"
            )
        except Exception as e:
            return None, (
                f"求解步骤 {step + 1} 出错: {e}, "
                f"dt={actual_dt_step:.2e}"
            )

        return P_current, None

    def evaluate_batch_at_targets(self, P_final, A_targets, moment_orders):
        """
        将批量求得的 P_n^\dagger(a, tau) 插值到对应的 a = A_target 处。

        输入：
            P_final.shape = (N_grid, len(moment_orders) * len(A_targets))

        输出：
            result_dict[n] = array, shape = (len(A_targets),)
        """

        if P_final is None:
            return {
                n: np.full(len(A_targets), np.nan)
                for n in moment_orders
            }

        if np.isnan(P_final).any() or np.isinf(P_final).any():
            return {
                n: np.full(len(A_targets), np.nan)
                for n in moment_orders
            }

        A_targets = np.asarray(A_targets)
        N_A = len(A_targets)

        result_dict = {}

        for order_idx, n in enumerate(moment_orders):
            start_col = order_idx * N_A
            end_col = start_col + N_A

            values = np.empty(N_A, dtype=float)

            for j, A_target in enumerate(A_targets):
                col = start_col + j
                values[j] = np.interp(A_target, self.A_grid, P_final[:, col])

            result_dict[n] = values

        return result_dict

    def calculate_km_batch(self, P_final, A_targets, tau, moment_orders):
        """
        批量计算 Kramers-Moyal 系数：

            D_n(a, tau) = P_n^\dagger(a, tau) / (n! * tau)

        其中 P_n^\dagger(a, tau) 在 a = A_target 处取值。
        """

        if P_final is None or tau <= 0:
            return {
                n: np.full(len(A_targets), np.nan)
                for n in moment_orders
            }

        P_values_dict = self.evaluate_batch_at_targets(
            P_final,
            A_targets,
            moment_orders
        )

        D_values_dict = {}

        for n in moment_orders:
            P_values = P_values_dict[n]
            D_values = P_values / (math.factorial(n) * tau)

            D_values = np.where(
                np.isfinite(D_values),
                D_values,
                np.nan
            )

            D_values_dict[n] = D_values

        return D_values_dict


# --- 数据生成函数 ---
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


def calculate_stationary_pdf(a_eval_grid, nu, kappa, d_diffusion):
    if d_diffusion <= 0 or kappa <= 0:
        raise ValueError("d_diffusion/kappa 必须为正")

    exponent = (
        (nu / (2.0 * d_diffusion)) * a_eval_grid**2
        - (kappa / (32.0 * d_diffusion)) * a_eval_grid**4
    )

    max_exponent = np.max(exponent)
    exponent = np.clip(exponent, max_exponent - 70, max_exponent + 10)

    P_inf_unnormalized = a_eval_grid * np.exp(exponent)

    P_inf_unnormalized = np.nan_to_num(
        P_inf_unnormalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    integral_Z = trapz(P_inf_unnormalized, a_eval_grid)

    if integral_Z < 1e-15:
        return P_inf_unnormalized, integral_Z

    P_inf_normalized = P_inf_unnormalized / integral_Z

    return P_inf_normalized, integral_Z


def process_parameter_set(params, tau_values_global, a_values_fixed_global, output_dir):
    """
    处理单个参数集。

    对每个 tau：
        一次性批量求解所有 A_target 和所有 moment order。

    计算：
        1. P_1^\dagger(a,0) = (a - A_target)^1 -> D1_fp
        2. P_2^\dagger(a,0) = (a - A_target)^2 -> D2_fp

    这不会改变数值精度，只是减少重复矩阵分解和重复线性求解开销。
    """

    nu = params['nu']
    kappa = params['kappa']
    d_diffusion = params['d_diffusion']

    param_id_str = (
        f"nu_{nu:.3f}_kappa_{kappa:.3f}_d_{d_diffusion:.3f}"
        .replace('.', '_')
        .replace('-', 'neg')
    )

    output_filename = os.path.join(output_dir, f"data_{param_id_str}.csv")

    if os.path.exists(output_filename):
        return f"跳过 {param_id_str}"

    results_list = []
    solver = None
    status = f"未知错误 {param_id_str}"

    try:
        A_values_to_use = np.asarray(a_values_fixed_global)
        num_a_points_actual = len(A_values_to_use)

        A_max_sim_for_grid = A_values_to_use[-1]
        A_grid, dA, _ = create_grid(A_max_sim_for_grid, N_grid=FP_N_GRID)

        solver = FPSolver(A_grid, dA, nu, kappa, d_diffusion)

        solve_start_time = time.time()
        calculation_count = 0
        failed_solves = 0

        for tau_sec in tau_values_global:
            try:
                P_final_batch, error_msg = solver.solve_adjoint_cn_batch(
                    A_targets=A_values_to_use,
                    tau=tau_sec,
                    moment_orders=MOMENT_ORDERS
                )

                if P_final_batch is None:
                    failed_solves += len(A_values_to_use) * len(MOMENT_ORDERS)

                    for A_target in A_values_to_use:
                        results_list.append({
                            'nu': nu,
                            'kappa': kappa,
                            'd_diffusion': d_diffusion,
                            'tau': tau_sec,
                            'A': A_target,
                            'P1_dagger': np.nan,
                            'P2_dagger': np.nan,
                            'D1_fp': np.nan,
                            'D2_fp': np.nan
                        })

                    calculation_count += len(A_values_to_use)
                    continue

                P_values_dict = solver.evaluate_batch_at_targets(
                    P_final_batch,
                    A_values_to_use,
                    MOMENT_ORDERS
                )

                D_values_dict = solver.calculate_km_batch(
                    P_final_batch,
                    A_values_to_use,
                    tau_sec,
                    MOMENT_ORDERS
                )

                P1_values = P_values_dict.get(
                    1,
                    np.full(len(A_values_to_use), np.nan)
                )
                P2_values = P_values_dict.get(
                    2,
                    np.full(len(A_values_to_use), np.nan)
                )

                D1_values = D_values_dict.get(
                    1,
                    np.full(len(A_values_to_use), np.nan)
                )
                D2_values = D_values_dict.get(
                    2,
                    np.full(len(A_values_to_use), np.nan)
                )

                for idx, A_target in enumerate(A_values_to_use):
                    if not np.isfinite(D1_values[idx]) or not np.isfinite(D2_values[idx]):
                        failed_solves += 1

                    results_list.append({
                        'nu': nu,
                        'kappa': kappa,
                        'd_diffusion': d_diffusion,
                        'tau': tau_sec,
                        'A': A_target,

                        # adjoint 解在 a=A_target 处的值
                        'P1_dagger': P1_values[idx],
                        'P2_dagger': P2_values[idx],

                        # KM 系数
                        'D1_fp': D1_values[idx],
                        'D2_fp': D2_values[idx]
                    })

                calculation_count += len(A_values_to_use)

            except Exception:
                traceback.print_exc()

                for A_target in A_values_to_use:
                    results_list.append({
                        'nu': nu,
                        'kappa': kappa,
                        'd_diffusion': d_diffusion,
                        'tau': tau_sec,
                        'A': A_target,
                        'P1_dagger': np.nan,
                        'P2_dagger': np.nan,
                        'D1_fp': np.nan,
                        'D2_fp': np.nan
                    })

                failed_solves += len(A_values_to_use) * len(MOMENT_ORDERS)
                calculation_count += len(A_values_to_use)

        solve_duration = time.time() - solve_start_time

        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(output_filename, index=False)

            status = (
                f"保存 {param_id_str} "
                f"({num_a_points_actual} A点, "
                f"{calculation_count} 总点, "
                f"{failed_solves} 失败, "
                f"{solve_duration:.1f}s)"
            )
        else:
            status = f"无结果 {param_id_str}"

    except ValueError as e_outer:
        status = f"失败 ValueErr {param_id_str}: {e_outer}"

    except Exception:
        traceback.print_exc()
        status = f"失败 Other {param_id_str}"

    return status


# --- 主执行逻辑 ---
if __name__ == "__main__":
    print("--- DeepONet 数据生成脚本：P_n^dagger(a,0) = (a - a')^n ---")
    print("--- 加速版本：LU 分解复用 + A_target 批量求解 + n=1,2 合并推进 ---")
    print("--- 注意：未改变 dt、网格、采样点、CN 格式，因此不降低计算精度 ---")

    # 1. 采样参数
    print(f"\n采样 {N_PARAM_SETS} 个参数集...")
    parameter_sets = sample_parameters(
        N_PARAM_SETS,
        NU_RANGE,
        KAPPA_RANGE,
        D2_RANGE
    )
    print(f"采样了 {len(parameter_sets)} 个参数集。")

    # 2. 定义 Tau 值
    tau_values = get_tau_values(N_TAU_POINTS, TAU_RANGE)
    print(
        f"\n生成了 {len(tau_values)} 个 tau 值，"
        f"范围在 {TAU_RANGE[0]} 到 {TAU_RANGE[1]} 之间。"
    )

    # 3. 定义固定 A 值
    a_values_fixed = np.linspace(
        A_MIN_FIXED,
        A_MAX_FIXED,
        N_A_POINTS_FIXED
    )

    print(
        f"将使用固定的 {N_A_POINTS_FIXED} 个 A 值，"
        f"范围在 {A_MIN_FIXED} 到 {A_MAX_FIXED} 之间。"
    )

    print(f"FP 网格点数 N_grid = {FP_N_GRID}。")
    print(f"将计算 moment orders: {MOMENT_ORDERS}")
    print("初始条件: P_n^dagger(a, 0) = (a - a')^n")
    print("其中 a' 在代码中对应 A_target。")
    print("仍然使用原始动态时间步长 dt = 0.5 * da^2 / D(2)，并受 tau/2 限制。")
    print("没有引入 MAX_STEPS，没有增大 dt，没有减少网格点或采样点。")

    # 4. 为每个参数集并行运行 AFP 求解器
    print(
        f"\n开始并行 AFP 计算，使用 "
        f"{N_JOBS if N_JOBS > 0 else '所有'} 个核心..."
    )

    total_start_time = time.time()

    results_status = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_parameter_set)(
            params,
            tau_values,
            a_values_fixed,
            OUTPUT_DIR
        )
        for params in tqdm(parameter_sets, desc="分发参数集任务")
    )

    total_end_time = time.time()

    print("\n--- 并行计算完成 ---")
    print(f"总时间: {total_end_time - total_start_time:.2f} 秒。")

    # 5. 总结结果
    print("\n处理摘要:")

    saved_count = sum(1 for s in results_status if s.startswith("保存"))
    skipped_count = sum(1 for s in results_status if s.startswith("跳过"))
    failed_val_count = sum(1 for s in results_status if s.startswith("失败 ValueErr"))
    failed_oth_count = sum(1 for s in results_status if s.startswith("失败 Other"))
    no_results_count = sum(1 for s in results_status if s.startswith("无结果"))

    print(f"  成功生成并保存: {saved_count}/{len(parameter_sets)}")
    print(f"  跳过，已存在:    {skipped_count}/{len(parameter_sets)}")
    print(f"  失败，值错误:    {failed_val_count}/{len(parameter_sets)}")
    print(f"  失败，其他错误:  {failed_oth_count}/{len(parameter_sets)}")
    print(f"  未生成结果:      {no_results_count}/{len(parameter_sets)}")

    # 打印部分详细状态
    print("\n部分详细状态:")

    count = 0
    for s in results_status:
        if count < 10 or not s.startswith("跳过"):
            print(f"  - {s}")

        if not s.startswith("跳过"):
            count += 1

    print(f"\n生成的数据保存在: {OUTPUT_DIR}")
    print("脚本完成。")