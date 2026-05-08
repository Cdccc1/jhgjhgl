import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===================== 评价指标 =====================
def mean_relative_error(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ===================== 数据加载 =====================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # 列名自适应（与之前相同）
    col_map = {}
    candidates = {
        'fcpu': ['fcpu', 'f_cpu', 'FCPU', 'cpu_freq'],
        'fgpu': ['fgpu', 'f_gpu', 'FGPU', 'gpu_freq'],
        'cpu_uti': ['cpu_uti', 'cpu_util', 'CPU_uti'],
        'gpu_uti': ['gpu_uti', 'gpu_util', 'GPU_uti'],
        'r_bw': ['r_bandwidth', 'read_bandwidth', 'r_bw'],
        'w_bw': ['w_bandwidth', 'write_bandwidth', 'w_bw'],
        'fps': ['fps', 'FPS']
    }
    for key, cands in candidates.items():
        for c in cands:
            if c in df.columns:
                col_map[key] = c
                break
        else:
            raise KeyError(f"缺少列：{key}")

    fcpu = df[col_map['fcpu']].values.astype(float) / 1e6   # MHz
    fgpu = df[col_map['fgpu']].values.astype(float) / 1e9   # GHz
    cpu_uti = df[col_map['cpu_uti']].values.astype(float)   # 百分数 (0-100)
    gpu_uti = df[col_map['gpu_uti']].values.astype(float)   # 百分数
    bandwidth = (df[col_map['r_bw']].values.astype(float) +
                 df[col_map['w_bw']].values.astype(float))
    fps = df[col_map['fps']].values.astype(float)

    return {'fcpu': fcpu, 'fgpu': fgpu, 'cpu_uti': cpu_uti,
            'gpu_uti': gpu_uti, 'bandwidth': bandwidth, 'fps': fps}

# ===================== 拟合与绘图 =====================
def fit_and_plot(data):
    fcpu = data['fcpu']
    fgpu = data['fgpu']
    fps = data['fps']
    cpu_uti = data['cpu_uti']
    gpu_uti = data['gpu_uti']
    bw = data['bandwidth']

    plt.figure(figsize=(12, 8))

    # ---------- CPU 利用率：util = a * (fps/fcpu) + b ----------
    plt.subplot(2, 2, 1)
    X_cpu = (fps / fcpu).reshape(-1, 1)
    y_cpu = cpu_uti
    model_cpu = LinearRegression().fit(X_cpu, y_cpu)
    a_cpu = model_cpu.coef_[0]
    b_cpu = model_cpu.intercept_
    y_pred_cpu = model_cpu.predict(X_cpu)
    r2_cpu = r2_score(y_cpu, y_pred_cpu)
    mre_cpu = mean_relative_error(y_cpu, y_pred_cpu)

    plt.scatter(X_cpu, y_cpu, alpha=0.6, label='data')
    x_range = np.linspace(X_cpu.min(), X_cpu.max(), 50).reshape(-1, 1)
    plt.plot(x_range, model_cpu.predict(x_range), 'r-',
             label=f'fit: a={a_cpu:.2f}, b={b_cpu:.2f}')
    plt.xlabel('FPS / f_cpu (MHz⁻¹·frame)')
    plt.ylabel('CPU util (%)')
    plt.title(f'CPU util model\nR² = {r2_cpu:.3f}, MRE = {mre_cpu:.3f}')
    plt.legend()

    # ---------- GPU 利用率：util = a * (fps/fgpu) + b ----------
    plt.subplot(2, 2, 2)
    X_gpu = (fps / fgpu).reshape(-1, 1)
    y_gpu = gpu_uti
    model_gpu = LinearRegression().fit(X_gpu, y_gpu)
    a_gpu = model_gpu.coef_[0]
    b_gpu = model_gpu.intercept_
    y_pred_gpu = model_gpu.predict(X_gpu)
    r2_gpu = r2_score(y_gpu, y_pred_gpu)
    mre_gpu = mean_relative_error(y_gpu, y_pred_gpu)

    plt.scatter(X_gpu, y_gpu, alpha=0.6, label='data')
    x_range = np.linspace(X_gpu.min(), X_gpu.max(), 50).reshape(-1, 1)
    plt.plot(x_range, model_gpu.predict(x_range), 'r-',
             label=f'fit: a={a_gpu:.2f}, b={b_gpu:.2f}')
    plt.xlabel('FPS / f_gpu (GHz⁻¹·frame)')
    plt.ylabel('GPU util (%)')
    plt.title(f'GPU util model\nR² = {r2_gpu:.3f}, MRE = {mre_gpu:.3f}')
    plt.legend()

    # ---------- 带宽：B = η_c * f_cpu + η_g * f_gpu ----------
    plt.subplot(2, 2, 3)
    X_bw = np.column_stack([fcpu, fgpu])
    y_bw = bw
    model_bw = LinearRegression(fit_intercept=True).fit(X_bw, y_bw)  # 可包含截距
    eta_c, eta_g = model_bw.coef_
    intercept_bw = model_bw.intercept_
    y_pred_bw = model_bw.predict(X_bw)
    r2_bw = r2_score(y_bw, y_pred_bw)
    mre_bw = mean_relative_error(y_bw, y_pred_bw)

    # 由于有两个自变量，无法直接画二维散点与回归线，改用预测值 vs 实际值
    plt.scatter(y_bw, y_pred_bw, alpha=0.6)
    plt.plot([y_bw.min(), y_bw.max()], [y_bw.min(), y_bw.max()], 'r--')
    plt.xlabel('True Bandwidth')
    plt.ylabel('Predicted Bandwidth')
    plt.title(f'Bandwidth model\nη_c={eta_c:.2e}, η_g={eta_g:.2e}\nR²={r2_bw:.3f}, MRE={mre_bw:.3f}')

    # 汇总打印
    print("========== 新关系拟合结果 ==========")
    print(f"CPU util: a={a_cpu:.4f}, b={b_cpu:.4f}, R²={r2_cpu:.4f}, MRE={mre_cpu:.4f}")
    print(f"GPU util: a={a_gpu:.4f}, b={b_gpu:.4f}, R²={r2_gpu:.4f}, MRE={mre_gpu:.4f}")
    print(f"Bandwidth: η_cpu={eta_c:.4e}, η_gpu={eta_g:.4e}, intercept={intercept_bw:.4e}")
    print(f"Bandwidth: R²={r2_bw:.4f}, MRE={mre_bw:.4f}")

    plt.tight_layout()
    plt.show()

# ===================== 主程序 =====================
if __name__ == "__main__":
    csv_path = "your_data.csv"   # 请改为实际路径
    data = load_data(csv_path)
    fit_and_plot(data)
