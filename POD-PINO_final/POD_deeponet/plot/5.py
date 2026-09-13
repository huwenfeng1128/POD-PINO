import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 生成模拟压力振荡信号
def generate_pressure_signal(t, f0=100, noise_level=0.2, modulation_freq=5):
    """生成包含噪声和调制的压力信号"""
    # 主振荡频率
    carrier = np.sin(2 * np.pi * f0 * t)

    # 振幅调制
    modulation = 1 + 0.3 * np.sin(2 * np.pi * modulation_freq * t)

    # 添加噪声
    noise = noise_level * np.random.randn(len(t))

    # 组合信号
    pressure = modulation * carrier + noise
    return pressure


# 参数设置
fs = 1000  # 采样率
T = 2.0  # 信号时长
t = np.linspace(0, T, int(fs * T))
f0 = 100  # 主频

# 生成信号
pressure_signal = generate_pressure_signal(t, f0=f0)

# 创建图形
fig = plt.figure(figsize=(20, 16))

# 1. 原始信号展示
gs1 = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs1[0, :])
ax1.plot(t, pressure_signal, 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('压力 (Pa)')
ax1.set_title('(a) 原始压力振荡信号')
ax1.grid(True, alpha=0.3)

# 2. 频谱分析
ax2 = fig.add_subplot(gs1[1, 0])
frequencies, psd = signal.welch(pressure_signal, fs, nperseg=1024)
ax2.semilogy(frequencies, psd, 'r-', linewidth=2)
ax2.axvline(x=f0, color='k', linestyle='--', alpha=0.7, label=f'主频 {f0}Hz')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('功率谱密度')
ax2.set_title('(b) 信号功率谱')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 200)


# 3. 带通滤波过程
def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """带通滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal_data)
    return filtered


# 滤波参数
lowcut = f0 - 20
highcut = f0 + 20
filtered_signal = bandpass_filter(pressure_signal, lowcut, highcut, fs)

ax3 = fig.add_subplot(gs1[1, 1])
ax3.plot(t, pressure_signal, 'b-', alpha=0.3, label='原始信号')
ax3.plot(t, filtered_signal, 'r-', linewidth=1.5, label='滤波后信号')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('压力 (Pa)')
ax3.set_title('(c) 带通滤波过程')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.5, 0.7)  # 放大显示

# 4. 希尔伯特变换提取包络
analytic_signal = hilbert(filtered_signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)

ax4 = fig.add_subplot(gs1[2, 0])
ax4.plot(t, filtered_signal, 'b-', alpha=0.7, label='滤波信号')
ax4.plot(t, amplitude_envelope, 'r-', linewidth=2, label='包络线')
ax4.fill_between(t, -amplitude_envelope, amplitude_envelope, alpha=0.2, color='red')
ax4.set_xlabel('时间 (s)')
ax4.set_ylabel('幅值')
ax4.set_title('(d) 希尔伯特变换提取包络')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0.5, 0.7)

# 5. 概率密度估计
ax5 = fig.add_subplot(gs1[2, 1])
# 核密度估计
kde = gaussian_kde(amplitude_envelope)
x_pdf = np.linspace(0, np.max(amplitude_envelope), 200)
y_pdf = kde(x_pdf)

ax5.hist(amplitude_envelope, bins=50, density=True, alpha=0.7, color='skyblue', label='直方图')
ax5.plot(x_pdf, y_pdf, 'r-', linewidth=2, label='核密度估计')
ax5.set_xlabel('振幅 a')
ax5.set_ylabel('概率密度 P(a)')
ax5.set_title('(e) 振幅概率密度函数估计')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 第二张图：联合概率密度和条件概率
fig2 = plt.figure(figsize=(16, 12))

# 6. 联合概率密度函数可视化
tau_value = 0.01  # 选择时间延迟
delay_samples = int(tau_value * fs)

# 创建延迟数据对
a_t = amplitude_envelope[:-delay_samples]
a_t_tau = amplitude_envelope[delay_samples:]

# 计算联合概率密度
xy = np.vstack([a_t, a_t_tau])
kde_joint = gaussian_kde(xy)

# 创建网格
x_grid = np.linspace(0, np.max(a_t), 50)
y_grid = np.linspace(0, np.max(a_t_tau), 50)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = kde_joint(positions).reshape(X.shape)

ax6 = fig2.add_subplot(2, 2, 1)
contour = ax6.contourf(X, Y, Z, levels=20, cmap='viridis')
ax6.set_xlabel('a(t)')
ax6.set_ylabel(f'a(t+τ), τ={tau_value}s')
ax6.set_title('(f) 联合概率密度函数 P(a(t), a(t+τ))')
plt.colorbar(contour, ax=ax6)

# 7. 条件概率密度可视化
# 选择几个特定的a值来展示条件概率
selected_a_values = [0.5, 1.0, 1.5]  # 选择几个振幅值
colors = ['red', 'blue', 'green']

ax7 = fig2.add_subplot(2, 2, 2)
for a_val, color in zip(selected_a_values, colors):
    # 找到最接近的a(t)值
    idx = np.argmin(np.abs(a_t - a_val))
    actual_a = a_t[idx]

    # 计算条件概率密度：P(a'|a) = P(a',a) / P(a)
    conditional_pdf = kde_joint([np.full_like(y_grid, actual_a), y_grid]) / kde(actual_a)

    ax7.plot(y_grid, conditional_pdf, color=color, linewidth=2,
             label=f'a(t) = {actual_a:.2f}')

ax7.set_xlabel("a' = a(t+τ)")
ax7.set_ylabel("P(a'|a)")
ax7.set_title('(g) 条件概率密度函数 P(a′|a)')
ax7.legend()
ax7.grid(True, alpha=0.3)


# 8. 有限时间条件矩计算
def calculate_conditional_moments(a_t, a_t_tau, a_values, n_max=2):
    """计算条件矩"""
    moments = {n: [] for n in range(1, n_max + 1)}

    for a in a_values:
        # 找到a(t)接近当前a值的点
        indices = np.where(np.abs(a_t - a) < 0.1)[0]
        if len(indices) > 0:
            a_tau_values = a_t_tau[indices]
            for n in range(1, n_max + 1):
                moment = np.mean((a_tau_values - a) ** n)
                moments[n].append(moment)
        else:
            for n in range(1, n_max + 1):
                moments[n].append(0)

    return moments


# 选择一组a值
a_eval_points = np.linspace(0.2, np.max(amplitude_envelope) * 0.8, 50)

# 计算条件矩
moments = calculate_conditional_moments(a_t, a_t_tau, a_eval_points)

ax8 = fig2.add_subplot(2, 2, 3)
ax8.plot(a_eval_points, moments[1], 'ro-', linewidth=2, label='Mτ^(1)(a)')
ax8.set_xlabel('振幅 a')
ax8.set_ylabel('一阶条件矩 Mτ^(1)')
ax8.set_title('(h) 一阶有限时间条件矩')
ax8.legend()
ax8.grid(True, alpha=0.3)

ax9 = fig2.add_subplot(2, 2, 4)
ax9.plot(a_eval_points, moments[2], 'bo-', linewidth=2, label='Mτ^(2)(a)')
ax9.set_xlabel('振幅 a')
ax9.set_ylabel('二阶条件矩 Mτ^(2)')
ax9.set_title('(i) 二阶有限时间条件矩')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 第三张图：KM系数提取
fig3 = plt.figure(figsize=(12, 8))


# 9. 提取有限时间KM系数
def calculate_km_coefficients(moments, tau, n_max=2):
    """计算KM系数"""
    km_coeffs = {}
    for n in range(1, n_max + 1):
        km_coeffs[n] = np.array(moments[n]) / (np.math.factorial(n) * tau)
    return km_coeffs


# 计算KM系数
km_coeffs = calculate_km_coefficients(moments, tau_value)

ax10 = fig3.add_subplot(1, 2, 1)
ax10.plot(a_eval_points, km_coeffs[1], 'ro-', linewidth=2, label='Dτ^(1)(a)')
ax10.set_xlabel('振幅 a')
ax10.set_ylabel('一阶KM系数 Dτ^(1)')
ax10.set_title('(j) 一阶有限时间KM系数')
ax10.legend()
ax10.grid(True, alpha=0.3)

ax11 = fig3.add_subplot(1, 2, 2)
ax11.plot(a_eval_points, km_coeffs[2], 'bo-', linewidth=2, label='Dτ^(2)(a)')
ax11.set_xlabel('振幅 a')
ax11.set_ylabel('二阶KM系数 Dτ^(2)')
ax11.set_title('(k) 二阶有限时间KM系数')
ax11.legend()
ax11.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. 多时间延迟KM系数对比
fig4, (ax12, ax13) = plt.subplots(1, 2, figsize=(16, 6))

# 测试不同时间延迟
tau_values = [0.005, 0.01, 0.02, 0.05]
colors = ['red', 'blue', 'green', 'purple']

for tau, color in zip(tau_values, colors):
    delay_samples = int(tau * fs)
    a_t = amplitude_envelope[:-delay_samples]
    a_t_tau = amplitude_envelope[delay_samples:]

    moments_tau = calculate_conditional_moments(a_t, a_t_tau, a_eval_points)
    km_coeffs_tau = calculate_km_coefficients(moments_tau, tau)

    ax12.plot(a_eval_points, km_coeffs_tau[1], color=color, linewidth=2,
              label=f'τ = {tau}s')
    ax13.plot(a_eval_points, km_coeffs_tau[2], color=color, linewidth=2,
              label=f'τ = {tau}s')

ax12.set_xlabel('振幅 a')
ax12.set_ylabel('一阶KM系数 Dτ^(1)')
ax12.set_title('(l) 不同时间延迟的一阶KM系数')
ax12.legend()
ax12.grid(True, alpha=0.3)

ax13.set_xlabel('振幅 a')
ax13.set_ylabel('二阶KM系数 Dτ^(2)')
ax13.set_title('(m) 不同时间延迟的二阶KM系数')
ax13.legend()
ax13.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("数据提取过程可视化完成！")
print("流程总结：")
print("1. 原始压力信号 → 2. 频谱分析 → 3. 带通滤波 → 4. 希尔伯特变换提取包络")
print("5. 概率密度估计 → 6. 联合概率密度 → 7. 条件概率密度 → 8. 条件矩计算")
print("9. KM系数提取 → 10. 多时间延迟分析")