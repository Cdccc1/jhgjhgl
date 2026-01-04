import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import curve_fit
import os
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 配置区
# ==============================
DATA_FILE = r"D:\Google download\data\1_color_newbrand.csv"
output_dir = r"D:\Google download\data\latency_vs_min_phi_plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 【关键】定义 phi(fddr) 函数 —— 后续只需修改此处即可更换形式
# 输入：fddr in Hz；输出：标量特征（越小表示带宽越强）
# ==============================
def phi_ddr(fddr_hz):
    """
    可替换的 DDR 特征映射函数。
    当前默认：phi = 1 / fddr  （即延迟与频率成反比）
    其他候选示例见下方注释。
    """
    return 1.0 / fddr_hz

    # 示例替换（取消注释即可切换）：
    # return np.exp(-fddr_hz / 1e9)          # 归一化到 GHz 再指数衰减
    # return 1.0 / np.sqrt(fddr_hz + 1e-6)   # 平方根倒数
    # return 1.0 / (np.log(fddr_hz + 1))     # 对数倒数
    # return np.maximum(1e-9, 1.0 / fddr_hz) # 带下限保护

# ==============================
# 读取数据
# ==============================
df = pd.read_csv(DATA_FILE)

required_cols = ['fcpu', 'fgpu', 'fddr', 'fps']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# 单位统一为 GHz
df['fcpu_hz'] = df['fcpu'] / 1e6
df['fgpu_hz'] = df['fgpu'] / 1e9
df['fddr_hz'] = df['fddr'] / 1e9

# 计算 latency（秒）
df['latency'] = 1.0 / df['fps']

# 构造两个瓶颈项
inv_cpu_gpu = 1.0 / df['fcpu_hz'] + 1.0 / df['fgpu_hz']
phi_ddr_vals = phi_ddr(df['fddr_hz'].values)

# 自变量 x = min(phi(fddr), 1/fcpu + 1/fgpu)
df['x'] = np.minimum(phi_ddr_vals, inv_cpu_gpu)

# 过滤无效值
valid_mask = (df['x'] > 0) & (df['latency'] > 0) & np.isfinite(df['x']) & np.isfinite(df['latency'])
df = df[valid_mask].copy()

if len(df) == 0:
    raise ValueError("No valid data after filtering.")

x_global = df['x'].values
y_global = df['latency'].values

# ==============================
# 候选模型（输入 x，输出 latency）
# ==============================
def linear(x, a, b):
    return a * x + b

def power_law(x, a, b):
    return a * np.power(x + 1e-12, b)

def exp_decay(x, a, b, c):
    return a + b * np.exp(-c * x)

def rational(x, a, b):
    return a / (x + b + 1e-12)

def const_plus_inv(x, a, b):
    return a + b / (x + 1e-12)

models = {
    "Linear": (linear, 2),
    "PowerLaw": (power_law, 2),
    "ExpDecay": (exp_decay, 3),
    "Rational": (rational, 2),
    "ConstPlusInv": (const_plus_inv, 2),
}

# ==============================
# MRE 计算函数
# ==============================
def compute_mre(y_true, y_pred):
    epsilon = 1e-12
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# ==============================
# 全局拟合（按 MRE 选最佳）
# ==============================
best_model_name = None
best_mre = np.inf
best_pred = None
best_params = None
best_func = None
all_results = {}

for name, (func, min_n) in models.items():
    if len(x_global) < min_n:
        continue
    try:
        # 初始猜测
        p0 = None
        if name == "Linear":
            p0 = [y_global.mean() / x_global.mean(), y_global.min()]
        elif name == "PowerLaw":
            p0 = [1.0, 1.0]
        elif name == "ExpDecay":
            p0 = [y_global.min(), y_global.max() - y_global.min(), 1.0]
        elif name == "Rational":
            p0 = [y_global.mean() * x_global.mean(), x_global.mean()]
        elif name == "ConstPlusInv":
            p0 = [y_global.min(), (y_global.max() - y_global.min()) * x_global.mean()]

        popt, _ = curve_fit(func, x_global, y_global, p0=p0, maxfev=10000)
        y_pred = func(x_global, *popt)

        if not np.all(np.isfinite(y_pred)) or np.any(y_pred <= 0):
            continue

        mre = compute_mre(y_global, y_pred)
        all_results[name] = (y_pred, mre, popt)

        if mre < best_mre:
            best_mre = mre
            best_model_name = name
            best_pred = y_pred
            best_params = popt
            best_func = func

    except Exception:
        continue

# ==============================
# 输出结果
# ==============================
if best_func is not None:
    r2 = 1 - np.sum((y_global - best_pred)**2) / np.sum((y_global - y_global.mean())**2)
    mae = mean_absolute_error(y_global, best_pred)
    rmse = np.sqrt(mean_squared_error(y_global, best_pred))

    print(f"\nBest Model (by MRE): {best_model_name}")
    print(f"  MRE:  {best_mre:.2f} %")
    print(f"  MAE:  {mae:.6f} s")
    print(f"  RMSE: {rmse:.6f} s")
    print(f"  R²:   {r2:.4f}")
    print(f"  Params: {best_params}")

    # 保存指标
    metrics = pd.DataFrame([{
        'phi_function': '1/fddr',  # 可改为自动提取函数名（如 inspect）
        'best_model': best_model_name,
        'MRE_%': best_mre,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'params': str(best_params),
        'n_samples': len(y_global)
    }])
    metrics.to_csv(os.path.join(output_dir, "global_metrics.csv"), index=False)

    # 绘制拟合图（latency vs x）
    plt.figure(figsize=(10, 6))
    plt.scatter(x_global, y_global, alpha=0.6, label='Data (Latency)', color='gray')

    x_plot = np.linspace(x_global.min(), x_global.max(), 300)
    y_plot = best_func(x_plot, *best_params)
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label=f'Best: {best_model_name} (MRE={best_mre:.1f}%)')

    plt.xlabel('x = min( φ(f_ddr), 1/f_cpu + 1/f_gpu )')
    plt.ylabel('Latency (seconds)')
    plt.title(f'Latency vs Bottleneck Feature (φ = {phi_ddr.__name__})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "global_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 可选：绘制 FPS 视角（转换后）
    fps_data = 1.0 / y_global
    fps_pred = 1.0 / best_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(x_global, fps_data, alpha=0.6, label='Data (FPS)', color='gray')
    plt.plot(x_plot, 1.0 / y_plot, 'b--', linewidth=2, label=f'Predicted FPS (via latency fit)')
    plt.xlabel('x = min( φ(f_ddr), 1/f_cpu + 1/f_gpu )')
    plt.ylabel('Frame Rate (FPS)')
    plt.title('FPS View of Latency-Based Fit')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "global_fit_fps_view.png"), dpi=150, bbox_inches='tight')
    plt.close()

else:
    print("No valid model found.")

print(f"\nAll outputs saved to: {output_dir}")