#解决数据生成参数和实际数据不对应问题
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import gaussian_kde
import pandas as pd
import multiprocessing as mp
import os
import time
import re
import traceback # <-- 修正：新增这行导入 traceback 模块

# ==============================================
# 1. 定义模拟函数
# ==============================================
def run_simulation(params):
    """
    运行单次模拟并返回结果。
    params: 包含 nu, kappa, D 的字典。
    """
    start_time = time.time() # 记录开始时间

    nu = params['nu']
    kappa = params['kappa']
    D = params['D']

    omega0 = 150 * 2 * np.pi
    Gamma = D * 4 * omega0 ** 2

    dt = 1e-4
    total_time = 500
    num_steps = int(total_time / dt)

    t = np.zeros(num_steps)
    eta = np.zeros(num_steps)
    v = np.zeros(num_steps)

    eta[0] = 0.0
    v[0] = 0.0

    # 使用Runge-Kutta 4阶方法
    for n in range(num_steps - 1):
        Z = np.random.normal(0, 1)
        dW = np.sqrt(Gamma * dt) * Z

        current_eta = eta[n]
        current_v = v[n]

        # RK4 k1
        k1_eta = current_v
        k1_v = 2 * nu * current_v - omega0 ** 2 * current_eta - kappa * current_eta ** 2 * current_v

        # RK4 k2
        eta_temp1 = current_eta + 0.5 * dt * k1_eta
        v_temp1 = current_v + 0.5 * dt * k1_v
        k2_eta = v_temp1
        k2_v = 2 * nu * v_temp1 - omega0 ** 2 * eta_temp1 - kappa * eta_temp1 ** 2 * v_temp1

        # RK4 k3
        eta_temp2 = current_eta + 0.5 * dt * k2_eta
        v_temp2 = current_v + 0.5 * dt * k2_v
        k3_eta = v_temp2
        k3_v = 2 * nu * v_temp2 - omega0 ** 2 * eta_temp2 - kappa * eta_temp2 ** 2 * v_temp2

        # RK4 k4
        eta_temp3 = current_eta + dt * k3_eta
        v_temp3 = current_v + dt * k3_v
        k4_eta = v_temp3
        k4_v = 2 * nu * v_temp3 - omega0 ** 2 * eta_temp3 - kappa * eta_temp3 ** 2 * v_temp3

        # Update eta and v
        eta_det = current_eta + (dt / 6) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        v_det = current_v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        eta[n + 1] = eta_det
        v[n + 1] = v_det + dW # Add stochastic term to velocity

        t[n + 1] = t[n] + dt

    # 包络提取
    # 确保信号足够长，以便镜像延拓不会导致问题
    extension_len = 500
    if num_steps < 2 * extension_len + 1:
        print(f"警告: 信号长度 ({num_steps}) 过短，可能影响镜像延拓。建议增加 total_time。")
        extension_len = num_steps // 3 # 调整延拓长度以避免错误
        if extension_len < 1: extension_len = 1 # 最小为1

    extended_eta = mirror_extension(eta, extension_len)
    analytic_signal = hilbert(extended_eta)
    envelope_extended = np.abs(analytic_signal)
    envelope = envelope_extended[extension_len:-extension_len]

    # 带通滤波
    fs = 1/dt
    nyq = 0.5 * fs
    center_freq = 150
    bandwidth = 100
    lowcut = center_freq - bandwidth/2
    highcut = center_freq + bandwidth/2

    # 确保滤波频率在 Nyquist 频率内
    if lowcut >= nyq or highcut >= nyq or lowcut < 0 or highcut < 0:
        print(f"警告: 滤波频率 {lowcut}-{highcut} Hz 超出 Nyquist 频率 {nyq} Hz 或为负值。请检查 dt 或滤波参数。跳过滤波。")
        eta_bpf = eta # 如果滤波参数有问题，直接返回原始信号
    else:
        order = 4
        # 检查 Wn 是否有效
        Wn_normalized = [lowcut/nyq, highcut/nyq]
        if not (0 < Wn_normalized[0] < Wn_normalized[1] < 1):
             print(f"警告: 归一化滤波频率 {Wn_normalized} 无效 (不在 0-1 之间或顺序错误)。跳过滤波。")
             eta_bpf = eta
        else:
            try:
                b, a = butter(
                    N=order,
                    Wn=Wn_normalized,
                    btype='bandpass'
                )
                eta_bpf = filtfilt(b, a, eta)
            except ValueError as e:
                print(f"滤波失败: {e}. 返回原始信号。")
                eta_bpf = eta
            except Exception as e: # 捕获其他可能的滤波错误
                print(f"未知滤波错误: {e}. 返回原始信号。")
                eta_bpf = eta


    end_time = time.time() # 记录结束时间
    elapsed_time = end_time - start_time # 计算耗时

    # 返回参数字典，以便在主进程中正确匹配
    return params, t, eta, envelope, eta_bpf, elapsed_time

# ==============================================
# 2. 镜像延拓函数定义
# ==============================================
def mirror_extension(signal, extension_len=500):
    """对信号进行镜像延拓以抑制边界效应"""
    if len(signal) < 2 * extension_len + 1: # 确保信号长度足够进行延拓
        # 调整延拓长度，使其至少能延拓一部分
        new_extension_len = max(1, (len(signal) - 1) // 2)
        if new_extension_len < extension_len:
            print(f"警告: 信号长度 ({len(signal)}) 过短，调整镜像延拓长度从 {extension_len} 到 {new_extension_len}。")
        extension_len = new_extension_len
        if len(signal) <= 1: return signal # 信号太短无法延拓

    left_ext = signal[:extension_len][::-1]  # 左镜像
    right_ext = signal[-extension_len:][::-1]  # 右镜像
    return np.concatenate([left_ext, signal, right_ext])

# ==============================================
# 3. 参数生成
# ==============================================
num_simulations = 120
# 确保 nu, kappa, D 的值是浮点数，且小数点后有两位，以匹配文件名格式
nu_values = np.round(np.linspace(20, 40, num_simulations), 2)
kappa_values = np.round(np.linspace(5.25, 10.5, num_simulations), 2)
D_values = np.round(np.linspace(20, 40, num_simulations), 2)

# 生成参数组合列表
param_list = []
for i in range(num_simulations):
    param_list.append({
        'nu': nu_values[i],
        'kappa': kappa_values[i],
        'D': D_values[i]
    })

# ==============================================
# 4. 并行计算
# ==============================================
if __name__ == '__main__':
    # 创建存储结果的文件夹
    output_dir = r"/AFP/P(A,t)_data/sim_data/sim_data_out"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用进程池进行并行计算
    pool = mp.Pool(processes=mp.cpu_count()) # 使用所有可用的CPU核心

    # 使用 pool.imap_unordered 来跟踪进度
    results_iterator = pool.imap_unordered(run_simulation, param_list)

    # 关闭进程池
    pool.close()

    # 跟踪进度并保存结果
    print(f"开始进行 {num_simulations} 组参数的模拟...")
    processed_count = 0
    for result in results_iterator: # 直接迭代结果，不使用enumerate(param_list)
        # 解包结果，现在包含了原始参数
        current_params, t, eta, envelope, eta_bpf, elapsed_time = result

        processed_count += 1
        # 打印当前模拟的进度和耗时，使用返回的 current_params
        print(f"完成第 {processed_count}/{num_simulations} 组模拟 (nu={current_params['nu']:.2f}, kappa={current_params['kappa']:.2f}, D={current_params['D']:.2f}) - 耗时: {elapsed_time:.2f} 秒")

        # ==============================================
        # 5. 结果保存
        # ==============================================
        df_combined = pd.DataFrame({
            "Time (s)": t,
            "Eta": eta,
            "Envelope": envelope,
            "eta_filtered": eta_bpf
        })

        # 保存为CSV文件，文件名包含从结果中获取的参数信息
        # 确保文件名格式与第一个脚本的解析逻辑一致
        file_name = f"({current_params['nu']:.2f},{current_params['kappa']:.2f},{current_params['D']:.2f}).csv"
        file_path = os.path.join(output_dir, file_name)
        try:
            df_combined.to_csv(file_path, index=False)
        except Exception as e:
            print(f"保存文件失败 {file_path}: {e}")

    pool.join() # 等待所有进程完成

    print("所有模拟任务完成！")

    # ==============================================
    # 6. 绘制示例结果 (可选)
    # ==============================================
    if num_simulations > 0:
        first_params_to_plot = param_list[0] # 获取原始参数列表中的第一个
        plot_file_name = f"({first_params_to_plot['nu']:.2f},{first_params_to_plot['kappa']:.2f},{first_params_to_plot['D']:.2f}).csv"
        plot_file_path = os.path.join(output_dir, plot_file_name)

        if os.path.exists(plot_file_path):
            print(f"\n尝试绘制文件: {plot_file_path}")
            try:
                df_plot_result = pd.read_csv(plot_file_path)
                t = df_plot_result["Time (s)"].values
                eta = df_plot_result["Eta"].values
                envelope = df_plot_result["Envelope"].values
                eta_bpf = df_plot_result["eta_filtered"].values

                plt.figure(figsize=(10, 6))
                plt.plot(t, eta, label='η(t)')
                plt.xlabel('Time (s)')
                plt.ylabel('Displacement η')
                plt.title(f'Simulation of η(t) with nu={first_params_to_plot["nu"]:.2f}, kappa={first_params_to_plot["kappa"]:.2f}, D={first_params_to_plot["D"]:.2f}')
                plt.legend()
                plt.grid(True)
                plt.show()

                # 绘制 PDF
                steady_start = int(0.2 * len(eta))
                eta_steady = eta[steady_start:]
                envelope_steady = envelope[steady_start:]

                def plot_pdf(data, label, color):
                    """绘制核密度估计的PDF曲线"""
                    if len(data) == 0:
                        print(f"警告: {label} 数据为空，无法绘制PDF。")
                        return
                    kde = gaussian_kde(data)
                    # 确保 x 范围有效，避免 np.min/max(empty_array) 报错
                    if np.min(data) == np.max(data): # 如果数据只有一个值
                        x = np.array([np.min(data) - 0.1, np.min(data), np.min(data) + 0.1])
                    else:
                        x = np.linspace(np.min(data), np.max(data), 1000)
                    plt.plot(x, kde(x), color=color, label=label, lw=2)

                plt.figure(figsize=(12, 6))
                plot_pdf(eta_steady, 'Original η(t)', 'blue')
                plot_pdf(envelope_steady, 'Envelope', 'red')
                plt.title(f'Steady State PDFs for nu={first_params_to_plot["nu"]:.2f}, kappa={first_params_to_plot["kappa"]:.2f}, D={first_params_to_plot["D"]:.2f}')
                plt.legend()
                plt.grid(True)
                plt.show()

                # 绘制滤波前后对比
                plt.figure(figsize=(12, 6))
                plt.plot(t, eta, label='Original η(t)', alpha=0.6)
                plt.plot(t, eta_bpf, label='Filtered η(t)', color='red', lw=1.5)
                plt.xlim(0, 500)
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title(f'Bandpass Filtered Signal for nu={first_params_to_plot["nu"]:.2f}, kappa={first_params_to_plot["kappa"]:.2f}, D={first_params_to_plot["D"]:.2f}')
                plt.legend()
                plt.grid(True)
                plt.show()

            except Exception as e:
                print(f"绘制示例图时发生错误: {e}")
                traceback.print_exc() # 打印详细错误信息
        else:
            print(f"未找到用于绘制示例图的文件: {plot_file_path}")



