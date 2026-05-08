import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse

# ===================== 定义待拟合的函数 =====================
def cpu_util_model(f_cpu, u0, gamma, f_max):
    """CPU利用率与频率的关系"""
    return u0 / (1 + gamma * (f_cpu / f_max))

def gpu_util_model(f_gpu, u0, gamma, f_max):
    """GPU利用率与频率的关系（与CPU形式相同）"""
    return u0 / (1 + gamma * (f_gpu / f_max))

# ===================== 数据加载与预处理 =====================
def load_and_prepare(csv_path):
    """读取CSV并返回所需数据"""
    df = pd.read_csv(csv_path)
    # 确保列名存在（可自动识别大小写/下划线变体）
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
        'fcpu': df[col_map['fcpu']].values,
        'fgpu': df[col_map['fgpu']].values,
        'cpu_uti': df[col_map['cpu_uti']].values,
        'gpu_uti': df[col_map['gpu_uti']].values,
        'bandwidth': df[col_map['r_bandwidth']].values + df[col_map['w_bandwidth']].values,
        'fps': df[col_map['fps']].values
    }
    return data

# ===================== 拟合与绘图 =====================
def fit_and_plot(data, fcpu_max=None, fgpu_max=None):
    """对三组关系进行拟合并可视化"""
    plt.figure(figsize=(15, 4))

    # ---------- CPU 利用率 vs 频率 ----------
    plt.subplot(1, 3, 1)
    x_cpu = data['fcpu']
    y_cpu = data['cpu_uti']
    if fcpu_max is None:
        fcpu_max = np.max(x_cpu)
    # 初始猜测：u0 = max util, gamma = 0.5
    p0_cpu = [np.max(y_cpu), 0.5]
    try:
        popt_cpu, _ = curve_fit(
            lambda f, u0, gamma: cpu_util_model(f, u0, gamma, fcpu_max),
            x_cpu, y_cpu, p0=p0_cpu, bounds=(0, [1.2, 5])
        )
        u0_cpu, gamma_cpu = popt_cpu
        y_pred_cpu = cpu_util_model(x_cpu, u0_cpu, gamma_cpu, fcpu_max)
        r2_cpu = r2_score(y_cpu, y_pred_cpu)
        # 生成平滑曲线用于显示
        f_smooth = np.linspace(np.min(x_cpu)*0.9, np.max(x_cpu)*1.1, 100)
        y_smooth = cpu_util_model(f_smooth, u0_cpu, gamma_cpu, fcpu_max)
        plt.plot(f_smooth, y_smooth, 'r-', label=f'fit (u0={u0_cpu:.2f}, γ={gamma_cpu:.2f})')
        plt.title(f'CPU util vs freq\n$R^2$ = {r2_cpu:.3f}')
    except Exception as e:
        print(f"CPU拟合失败: {e}")
        r2_cpu = None
    plt.scatter(x_cpu, y_cpu, alpha=0.6)
    plt.xlabel('CPU freq (MHz)')
    plt.ylabel('Utilization')
    plt.legend()

    # ---------- GPU 利用率 vs 频率 ----------
    plt.subplot(1, 3, 2)
    x_gpu = data['fgpu']
    y_gpu = data['gpu_uti']
    if fgpu_max is None:
        fgpu_max = np.max(x_gpu)
    p0_gpu = [np.max(y_gpu), 0.5]
    try:
        popt_gpu, _ = curve_fit(
            lambda f, u0, gamma: gpu_util_model(f, u0, gamma, fgpu_max),
            x_gpu, y_gpu, p0=p0_gpu, bounds=(0, [1.2, 5])
        )
        u0_gpu, gamma_gpu = popt_gpu
        y_pred_gpu = gpu_util_model(x_gpu, u0_gpu, gamma_gpu, fgpu_max)
        r2_gpu = r2_score(y_gpu, y_pred_gpu)
        f_smooth = np.linspace(np.min(x_gpu)*0.9, np.max(x_gpu)*1.1, 100)
        y_smooth = gpu_util_model(f_smooth, u0_gpu, gamma_gpu, fgpu_max)
        plt.plot(f_smooth, y_smooth, 'r-', label=f'fit (u0={u0_gpu:.2f}, γ={gamma_gpu:.2f})')
        plt.title(f'GPU util vs freq\n$R^2$ = {r2_gpu:.3f}')
    except Exception as e:
        print(f"GPU拟合失败: {e}")
        r2_gpu = None
    plt.scatter(x_gpu, y_gpu, alpha=0.6)
    plt.xlabel('GPU freq (MHz)')
    plt.ylabel('Utilization')
    plt.legend()

    # ---------- 带宽 vs 帧率 ----------
    plt.subplot(1, 3, 3)
    x_fps = data['fps']
    y_bw = data['bandwidth']
    # 线性回归 B = Qframe * fps
    # 使用 np.polyfit 进行过原点的线性拟合 (y = k*x)
    # 也可用曲线拟合，但简单线性即可
    slope, intercept = np.polyfit(x_fps, y_bw, 1)  # 包含截距，看是否接近0
    # 强制零截距拟合
    Qframe = np.sum(x_fps * y_bw) / np.sum(x_fps ** 2)  # 最小二乘过原点
    y_pred_bw = Qframe * x_fps
    r2_bw = r2_score(y_bw, y_pred_bw)
    f_smooth = np.linspace(0, np.max(x_fps)*1.1, 50)
    plt.plot(f_smooth, Qframe * f_smooth, 'r-', label=f'B = {Qframe:.2e} * fps')
    plt.title(f'Bandwidth vs FPS\n$R^2$ = {r2_bw:.3f}')
    plt.scatter(x_fps, y_bw, alpha=0.6)
    plt.xlabel('FPS')
    plt.ylabel('Total bandwidth')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出拟合结果
    print("========== 拟合结果 ==========")
    if r2_cpu is not None:
        print(f"CPU: u0 = {u0_cpu:.4f}, γ = {gamma_cpu:.4f}, R² = {r2_cpu:.4f}")
    if r2_gpu is not None:
        print(f"GPU: u0 = {u0_gpu:.4f}, γ = {gamma_gpu:.4f}, R² = {r2_gpu:.4f}")
    print(f"带宽: Q_frame = {Qframe:.4e} (字节或计数值/frame), R² = {r2_bw:.4f}")
    if abs(intercept) > 0.1 * np.mean(y_bw):
        print(f"注意: 带宽-FPS 回归截距 = {intercept:.2f}，偏离原点，可能模型 B=Q*fps 不完全准确。")

# ===================== 主程序 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证性能模型中的利用率与带宽关系')
    parser.add_argument('csv_file', type=str, help='包含 fcpu, fgpu, cpu_uti, gpu_uti, r_bandwidth, w_bandwidth, fps 的CSV路径')
    parser.add_argument('--fcpu_max', type=float, default=None, help='CPU最大频率，若不提供则使用数据中最大值')
    parser.add_argument('--fgpu_max', type=float, default=None, help='GPU最大频率，若不提供则使用数据中最大值')
    args = parser.parse_args()

    data = load_and_prepare(args.csv_file)
    fit_and_plot(data, args.fcpu_max, args.fgpu_max)
