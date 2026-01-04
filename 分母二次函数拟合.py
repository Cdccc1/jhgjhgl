# model_a_rational_ddr.py
# 拟合模型为φ(fddr) = fddr / (a1*fddr^2 + a2*fddr + a3)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import curve_fit
import os
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = r"D:\Google download\data\1_color_newbrand.csv"
output_dir = r"D:\Google download\data\model_a_rational_ddr"
os.makedirs(output_dir, exist_ok=True)

# --- Load and convert to GHz ---
df = pd.read_csv(DATA_FILE)
required_cols = ['fcpu', 'fgpu', 'fddr', 'fps']
assert all(col in df.columns for col in required_cols), f"Missing columns: {required_cols}"

df['fcpu_ghz'] = df['fcpu'] / 1e6  # kHz → GHz
df['fgpu_ghz'] = df['fgpu'] / 1e9  # Hz → GHz
df['fddr_ghz'] = df['fddr'] / 1e9  # Hz → GHz
df['latency'] = 1.0 / df['fps']

mask = (df[['fcpu_ghz', 'fgpu_ghz', 'fddr_ghz', 'latency']] > 0).all(axis=1)
df = df[mask].copy()

fcpu = df['fcpu_ghz'].values
fgpu = df['fgpu_ghz'].values
fddr = df['fddr_ghz'].values
y_true = df['latency'].values


# --- Model A: Rational DDR ---
def model_a(freqs, a1, a2, a3, alpha1, alpha2):
    fcpu_ghz, fgpu_ghz, fddr_ghz = freqs

    # φ(fddr) = fddr / (a1*fddr^2 + a2*fddr + a3)
    denom = a1 * fddr_ghz ** 2 + a2 * fddr_ghz + a3
    denom = np.maximum(denom, 1e-12)
    phi_ddr = fddr_ghz / denom

    # CPU+GPU term
    cpu_gpu_term = alpha1 / (fcpu_ghz + 1e-12) + alpha2 / (fgpu_ghz + 1e-12)

    # Latency = max(phi, cpu_gpu)
    latency_pred = np.maximum(phi_ddr, cpu_gpu_term)
    return latency_pred


# Initial guess
p0 = [0.0, 1.0, 0.0, 1.0, 1.0]  # a1, a2, a3, alpha1, alpha2
bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

try:
    popt, _ = curve_fit(model_a, (fcpu, fgpu, fddr), y_true, p0=p0, bounds=bounds, maxfev=20000)
    a1, a2, a3, alpha1, alpha2 = popt
    y_pred = model_a((fcpu, fgpu, fddr), *popt)

    mre = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    print("✅ Model A (Rational DDR) Fitted!")
    print(f"φ(fddr) = fddr / ({a1:.4e}*fddr² + {a2:.4e}*fddr + {a3:.4e})")
    print(f"α1 (CPU): {alpha1:.4f}, α2 (GPU): {alpha2:.4f}")
    print(f"MRE: {mre:.2f}%, MAE: {mae:.6f}s, RMSE: {rmse:.6f}s, R²: {r2:.4f}")

    # Save metrics
    pd.DataFrame([{
        'a1': a1, 'a2': a2, 'a3': a3,
        'alpha1': alpha1, 'alpha2': alpha2,
        'MRE_%': mre, 'MAE': mae, 'RMSE': rmse, 'R2': r2
    }]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Latency (s)')
    plt.ylabel('Predicted Latency (s)')
    plt.title('Model A: Rational DDR\nLatency = max(φ(fddr), α1/fcpu + α2/fgpu)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

except Exception as e:
    print("❌ Model A fitting failed:", e)

print(f"\nModel A outputs saved to: {output_dir}")