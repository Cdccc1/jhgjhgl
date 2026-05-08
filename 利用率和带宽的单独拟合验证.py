import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ===================== 评价指标 =====================
def mean_relative_error(y_true, y_pred):
    """计算平均相对误差 (MRE)，若真实值为0则跳过"""
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ===================== 定义待拟合的函数 =====================
def cpu_util_model(f_cpu_scaled, u0, gamma):
    """CPU利用率与频率的关系（输入频率已缩放，f_cpu_max内部由调用者固定）"""
    # 注意：f_cpu_max 在拟合时通过闭包或外部传入，这里采用一个技巧：传入的f_cpu_scaled已经除以了f_max？
    # 为了保持模型形式：u = u0 / (1 + gamma * (f/f_max))，我们需要在 curve_fit 中使用预除的版本。
    # 我们将在 fit_and_plot 中构造 lambda 时除以 f_max。
    pass

# ===================== 数据加载与预处理 =====================
def load_and_prepare(csv_path):
    """读取CSV并返回所需数据，频率不做缩放"""
    df = pd.read_csv(csv_path)
    cols = {
        'fcpu': ['fcpu', 'f_cpu', 'FCPU', 'cpu_freq'],
        'fgpu': ['fgpu', 'f_gpu', 'FGPU', 'gpu_freq'],
        'cpu_uti': ['cpu_uti', 'cpu_util', 'CPU_uti'],
        'gpu_uti': ['gpu_uti', 'gpu_util', 'GPU_uti'],
        'r_bandwidth': ['r_bandwidth', 'read_bandwidth', 'r_bw'],
        'w_bandwidth': ['w_bandwidth', 'write_bandwidth', 'w_bw'],
        'fps': ['fps', 'FPS']
    }
    col_map = {}
    for key, candidates in cols.items():
        for c in candidates:
            if c in df.columns:
                col_map[key] = c
                break
        else:
            raise KeyError(f"未找到 {key} 对应的列，请检查CSV列名。")

    data = {
        'fcpu': df[col_map['fcpu']].values.astype(float),
        'fgpu': df[col_map['fgpu']].values.astype(float),
        'cpu_uti': df[col_map['cpu_uti']].values.astype(float),
        'gpu_uti': df[col_map['gpu_uti']].values.astype(float),
        'bandwidth': df[col_map['r_bandwidth']].values.astype(float) +
                     df[col_map['w_bandwidth']].values.astype(float),
        'fps': df[col_map['fps']].values.astype(float)
    }
    return data

# ===================== 拟合与绘图 =====================
def fit_and_plot(data):
    """对三组关系进行拟合并可视化"""
    # 频率缩放
    fcpu_raw = data['fcpu']
    fgpu_raw = data['fgpu']
    fcpu_scaled = fcpu_raw / 1e6       # 转换为 MHz
    fgpu_scaled = fgpu_raw / 1e9       # 转换为 GHz

    # 获取最大频率（缩放后）
    fcpu_max = np.max(fcpu_scaled)
    fgpu_max = np.max(fgpu_scaled)
    print(f"缩放后 CPU 最大频率: {fcpu_max:.4f} MHz")
    print(f"缩放后 GPU 最大频率: {fgpu_max:.4f} GHz")

    # 利用率数据
    y_cpu = data['cpu_uti']
    y_gpu = data['gpu_uti']

    # 带宽数据
    x_fps = data['fps']
    y_bw = data['bandwidth']

    plt.figure(figsize=(15, 4))

    # ---------- CPU 利用率 vs 频率 ----------
    plt.subplot(1, 3, 1)
    p0_cpu = [np.max(y_cpu), 0.5]
    try:
        popt_cpu, _ = curve_fit(
            lambda f, u0, gamma: cpu_util_model(f, u0, gamma, fcpu_max),  # 这里需要定义模型
            fcpu_scaled, y_cpu, p0=p0_cpu, bounds=(0, [1.2, 5])
        )
        u0_cpu, gamma_cpu = popt_cpu
        # 计算预测值
        y_pred_cpu = cpu_util_model(fcpu_scaled, u0_cpu, gamma_cpu, fcpu_max)
        r2_cpu = r2_score(y_cpu, y_pred_cpu)
        mre_cpu = mean_relative_error(y_cpu, y_pred_cpu)

        f_smooth = np.linspace(np.min(fcpu_scaled)*0.9, np.max(fcpu_scaled)*1.1, 100)
        y_smooth = cpu_util_model(f_smooth, u0_cpu, gamma_cpu, fcpu_max)
        plt.plot(f_smooth, y_smooth, 'r-',
                 label=f'fit (u0={u0_cpu:.2f}, γ={gamma_cpu:.2f})')
        plt.title(f'CPU util vs freq\n$R^2$={r2_cpu:.3f}, MRE={mre_cpu:.3f}')
        print(f"CPU: u0={u0_cpu:.4f}, γ={gamma_cpu:.4f}, R²={r2_cpu:.4f}, MRE={mre_cpu:.4f}")
    except Exception as e:
        print(f"CPU拟合失败: {e}")
    plt.scatter(fcpu_scaled, y_cpu, alpha=0.6)
    plt.xlabel('CPU frequency (MHz)')
    plt.ylabel('Utilization')
    plt.legend()

    # ---------- GPU 利用率 vs 频率 ----------
    plt.subplot(1, 3, 2)
    p0_gpu = [np.max(y_gpu), 0.5]
    try:
        popt_gpu, _ = curve_fit(
            lambda f, u0, gamma: cpu_util_model(f, u0, gamma, fgpu_max),  # 与CPU形式相同
            fgpu_scaled, y_gpu, p0=p0_gpu, bounds=(0, [1.2, 5])
        )
        u0_gpu, gamma_gpu = popt_gpu
        y_pred_gpu = cpu_util_model(fgpu_scaled, u0_gpu, gamma_gpu, fgpu_max)
        r2_gpu = r2_score(y_gpu, y_pred_gpu)
        mre_gpu = mean_relative_error(y_gpu, y_pred_gpu)

        f_smooth = np.linspace(np.min(fgpu_scaled)*0.9, np.max(fgpu_scaled)*1.1, 100)
        y_smooth = cpu_util_model(f_smooth, u0_gpu, gamma_gpu, fgpu_max)
        plt.plot(f_smooth, y_smooth, 'r-',
                 label=f'fit (u0={u0_gpu:.2f}, γ={gamma_gpu:.2f})')
        plt.title(f'GPU util vs freq\n$R^2$={r2_gpu:.3f}, MRE={mre_gpu:.3f}')
        print(f"GPU: u0={u0_gpu:.4f}, γ={gamma_gpu:.4f}, R²={r2_gpu:.4f}, MRE={mre_gpu:.4f}")
    except Exception as e:
        print(f"GPU拟合失败: {e}")
    plt.scatter(fgpu_scaled, y_gpu, alpha=0.6)
    plt.xlabel('GPU frequency (GHz)')
    plt.ylabel('Utilization')
    plt.legend()

    # ---------- 带宽 vs 帧率 ----------
    plt.subplot(1, 3, 3)
    # 强制过原点拟合 B = Qframe * fps
    Qframe = np.sum(x_fps * y_bw) / np.sum(x_fps ** 2)
    y_pred_bw = Qframe * x_fps
    r2_bw = r2_score(y_bw, y_pred_bw)
    mre_bw = mean_relative_error(y_bw, y_pred_bw)

    f_smooth = np.linspace(0, np.max(x_fps)*1.1, 50)
    plt.plot(f_smooth, Qframe * f_smooth, 'r-',
             label=f'B = {Qframe:.2e} * fps')
    plt.title(f'Bandwidth vs FPS\n$R^2$={r2_bw:.3f}, MRE={mre_bw:.3f}')
    plt.scatter(x_fps, y_bw, alpha=0.6)
    plt.xlabel('FPS')
    plt.ylabel('Total bandwidth')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出总结
    print("========== 拟合结果汇总 ==========")
    if 'u0_cpu' in locals():
        print(f"CPU: u0 = {u0_cpu:.4f}, γ = {gamma_cpu:.4f}, R² = {r2_cpu:.4f}, MRE = {mre_cpu:.4f}")
    if 'u0_gpu' in locals():
        print(f"GPU: u0 = {u0_gpu:.4f}, γ = {gamma_gpu:.4f}, R² = {r2_gpu:.4f}, MRE = {mre_gpu:.4f}")
    print(f"带宽: Q_frame = {Qframe:.4e} (每帧单位), R² = {r2_bw:.4f}, MRE = {mre_bw:.4f}")

# ===================== 正确定义带 f_max 的模型函数 =====================
# 需要在 fit_and_plot 中使用时已经固定 f_max，这里重写一个闭包版本
def cpu_util_model(f, u0, gamma, f_max):
    return u0 / (1.0 + gamma * (f / f_max))

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 直接在此处修改 CSV 文件路径
    csv_path = "your_data.csv"   # 请替换为实际路径
    data = load_and_prepare(csv_path)
    fit_and_plot(data)
