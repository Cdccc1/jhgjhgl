import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import os
import warnings
warnings.filterwarnings("ignore")  # 忽略拟合警告（如不收敛）

# 文件路径
DATA_FILE = r"D:\Google download\data\1_color_newbrand.csv"

# 读取数据
df = pd.read_csv(DATA_FILE)

# 必要列检查
required_cols = ['fcpu', 'fgpu', 'fddr', 'fps']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# 计算帧率倒数（即每帧耗时，单位：秒）
df['latency'] = 1.0 / df['fps']

# 单位转换：全部转为 GHz
df['fcpu_ghz'] = df['fcpu'] / 1e6   # kHz -> GHz
df['fgpu_ghz'] = df['fgpu'] / 1e9   # Hz -> GHz
df['fddr_ghz'] = df['fddr'] / 1e9   # Hz -> GHz

# 浮点分组防精度问题
df['fcpu_round'] = df['fcpu_ghz'].round(6)
df['fgpu_round'] = df['fgpu_ghz'].round(6)

grouped = df.groupby(['fcpu_round', 'fgpu_round'])

# 创建输出目录
output_dir = r"D:\Google download\data\ddr_vs_latency_plots"
os.makedirs(output_dir, exist_ok=True)

# ====== 定义候选模型 ======
def poly1(x, a, b):
    return a * x + b

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def inverse_model(x, a, b):
    # latency = a + b / fddr  --> b 应 > 0 才合理（fddr↑ → latency↓）
    return a + b / (x + 1e-12)

def exp_decay(x, a, b, c):
    # latency = a + b * exp(-c * x), 要求 b>0, c>0
    return a + b * np.exp(-c * x)

def log_model(x, a, b):
    # latency = a - b * log(x), 要求 b>0
    return a - b * np.log(x + 1e-12)

# 模型字典：名称 -> (函数, 最少所需参数数量)
models = {
    # "Poly1": (poly1, 2),
    # "Poly2": (poly2, 3),
    # "Poly3": (poly3, 4),
    # "Inverse": (inverse_model, 2),
    "ExpDecay": (exp_decay, 3),
    "Logarithmic": (log_model, 2),
}

# 遍历每个 (fcpu, fgpu) 组合
for (fcpu_val, fgpu_val), group in grouped:
    if len(group) < 2:
        continue

    x = group['fddr_ghz'].values
    y_latency = group['latency'].values  # 保留原始 latency 用于拟合

    # 跳过包含非正 fddr 或非正 latency 的情况
    if np.any(x <= 0) or np.any(y_latency <= 0):
        continue

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_latency_sorted = y_latency[sort_idx]

    # 计算用于绘图的 fps（仅用于可视化）
    y_fps = 1.0 / y_latency

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y_fps, label='Data (FPS)', color='black', zorder=5)

    best_r2 = -np.inf
    best_name = "None"

    # 尝试每种模型（拟合仍用 latency）
    for name, (func, min_params) in models.items():
        if len(x) < min_params:
            continue

        try:
            # 设置初始猜测值（基于 latency）
            p0 = None
            if name == "ExpDecay":
                p0 = [y_latency.min(), y_latency.max() - y_latency.min(), 1.0]
            elif name == "Inverse":
                p0 = [y_latency.min(), (y_latency.max() - y_latency.min()) * x.mean()]
            elif name == "Logarithmic":
                p0 = [y_latency.mean(), 0.1]

            # 拟合：输入 x 和 y_latency
            popt, _ = curve_fit(func, x, y_latency, p0=p0, maxfev=5000)

            # 用拟合结果预测 latency
            y_pred_latency = func(x, *popt)
            r2 = r2_score(y_latency, y_pred_latency)

            # 生成平滑曲线用于绘图（转换为 FPS）
            x_plot = np.linspace(x.min(), x.max(), 200)
            y_plot_latency = func(x_plot, *popt)
            y_plot_fps = 1.0 / y_plot_latency  # 转为 FPS 画图

            plt.plot(x_plot, y_plot_fps, '--', linewidth=1.5, label=f'{name} (R²={r2:.3f})')

            if r2 > best_r2:
                best_r2 = r2
                best_name = name

        except Exception as e:
            # 可选：打印错误调试
            # print(f"Model {name} failed: {e}")
            continue

    plt.title(f'fcpu={fcpu_val:.3f} GHz, fgpu={fgpu_val:.3f} GHz\nBest: {best_name} (R²={best_r2:.3f})')
    plt.xlabel('DDR Frequency (GHz)')
    plt.ylabel('Frame Rate (FPS)')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    filename = f"fcpu_{fcpu_val:.3f}_fgpu_{fgpu_val:.3f}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

    # print(f"[fcpu={fcpu_val:.3f}, fgpu={fgpu_val:.3f}] Best model: {best_name}, R²: {best_r2:.4f}")

print(f"\nAll plots saved to: {output_dir}")
# ==============================
# 全局指数模型拟合与评估指标
# ==============================

print("\n" + "=" * 60)
print("GLOBAL FIT: Fitting latency = a + b * exp(-c * fddr_ghz) on ALL data")
print("=" * 60)

# 提取全局 x, y
x_global = df['fddr_ghz'].values
y_global = df['latency'].values

# 过滤无效值（确保 x > 0）
valid_mask = x_global > 0
x_global = x_global[valid_mask]
y_global = y_global[valid_mask]


# 定义全局指数模型
def global_exp_decay(x, a, b, c):
    return a + b * np.exp(-c * x)


# 初始猜测
p0_global = [y_global.min(), y_global.max() - y_global.min(), 1.0]

try:
    # 拟合
    popt_global, pcov_global = curve_fit(
        global_exp_decay, x_global, y_global, p0=p0_global, maxfev=10000
    )
    a, b, c = popt_global
    # print(f"Fitted parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")

    # 预测
    y_pred_global = global_exp_decay(x_global, *popt_global)

    # 计算评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    r2 = r2_score(y_global, y_pred_global)
    mae = mean_absolute_error(y_global, y_pred_global)
    rmse = np.sqrt(mean_squared_error(y_global, y_pred_global))

    # 平均相对误差（MRE）：避免除零，加小量
    epsilon = 1e-12
    mre = np.mean(np.abs((y_global - y_pred_global) / (y_global + epsilon))) * 100  # 百分比

    # 输出结果
    print(f"\nGlobal Exponential Model Performance:")
    print(f"  R² (Coefficient of Determination): {r2:.6f}")
    print(f"  MAE (Mean Absolute Error):         {mae:.6f} seconds")
    print(f"  RMSE (Root Mean Squared Error):   {rmse:.6f} seconds")
    print(f"  MRE (Mean Relative Error):        {mre:.2f} %")

    # 可选：保存全局拟合图
    plt.figure(figsize=(10, 6))

    # 原始数据：x = fddr_ghz, y = fps（不是 latency！）
    x_data_fps = df.loc[valid_mask, 'fddr_ghz'].values
    y_data_fps = df.loc[valid_mask, 'fps'].values
    plt.scatter(x_data_fps, y_data_fps, alpha=0.6, label='All Data (FPS)', color='gray')

    # 模型预测：先预测 latency，再转为 fps
    x_plot = np.linspace(x_global.min(), x_global.max(), 300)
    latency_pred = global_exp_decay(x_plot, *popt_global)
    fps_pred = 1.0 / latency_pred  # 转换为帧率

    plt.plot(x_plot, fps_pred, 'r-', linewidth=2, label=f'Global ExpFit → FPS (R² on latency={r2:.3f})')

    plt.title('DDR Frequency vs Frame Rate (FPS)')
    plt.xlabel('DDR Frequency (GHz)')
    plt.ylabel('Frame Rate (FPS)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "global_exponential_fit_fps.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ==============================
    # 新增：为 4 个 DDR 频点绘制局部放大图（zoomed-in views）
    # ==============================

    print("\n" + "=" * 60)
    print("Generating ZOOMED-IN plots around each DDR frequency")
    print("=" * 60)

    # 定义 4 个 DDR 频率（Hz → GHz）
    ddr_hz_list = [204_000_000, 665_600_000, 2_133_000_000, 3_199_000_000]
    ddr_ghz_list = [f / 1e9 for f in ddr_hz_list]

    # 设置每个频点的显示窗口宽度（GHz），可根据数据密度调整
    zoom_width = 0.05  # ±0.05 GHz 范围

    for ddr_ghz in ddr_ghz_list:
        # 确定 x 轴范围
        x_min = ddr_ghz - zoom_width
        x_max = ddr_ghz + zoom_width

        # 筛选落在该窗口内的数据点
        mask_zoom = (df['fddr_ghz'] >= x_min) & (df['fddr_ghz'] <= x_max)
        if not mask_zoom.any():
            print(f"Warning: No data in [{x_min:.3f}, {x_max:.3f}] GHz for DDR={ddr_ghz:.3f} GHz")
            continue

        # 提取局部数据
        x_local = df.loc[mask_zoom, 'fddr_ghz'].values
        y_local = df.loc[mask_zoom, 'fps'].values

        # 绘图
        plt.figure(figsize=(6, 4))
        plt.scatter(x_local, y_local, alpha=0.8, color='tab:blue', edgecolor='k', linewidth=0.5, label='Data (FPS)')

        # 在局部区间绘制全局模型预测曲线
        x_plot_local = np.linspace(x_min, x_max, 200)
        latency_pred_local = global_exp_decay(x_plot_local, *popt_global)
        fps_pred_local = 1.0 / latency_pred_local
        plt.plot(x_plot_local, fps_pred_local, 'r-', linewidth=12, label='Global ExpFit')

        # 设置标题和标签
        plt.title(f'Zoomed View: DDR ≈ {ddr_ghz:.3f} GHz')
        plt.xlabel('DDR Frequency (GHz)')
        plt.ylabel('Frame Rate (FPS)')
        plt.xlim(x_min, x_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 保存
        filename = f"zoom_ddr_{ddr_ghz:.3f}GHz.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Zoom] Saved plot for DDR={ddr_ghz:.3f} GHz with {len(x_local)} points")

    print(f"\nZoomed-in plots saved to: {output_dir}")

    # 可选：保存指标到 CSV
    metrics_df = pd.DataFrame([{
        'model': 'Global_Exponential',
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MRE_%': mre,
        'a': a,
        'b': b,
        'c': c,
        'n_samples': len(y_global)
    }])
    metrics_df.to_csv(os.path.join(output_dir, "global_model_metrics.csv"), index=False)
    print(f"\nGlobal metrics saved to: {os.path.join(output_dir, 'global_model_metrics.csv')}")

except Exception as e:
    print(f"Global fit failed: {e}")