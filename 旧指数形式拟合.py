# model_b_three_param_exponential_ddr.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import curve_fit
import os
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = r"D:\Google download\data\1_color_newbrand.csv"
output_dir = r"D:\Google download\data\model_b_three_param_exponential_ddr"
os.makedirs(output_dir, exist_ok=True)

# --- Load data and convert to GHz ---
df = pd.read_csv(DATA_FILE)
required_cols = ['fcpu', 'fgpu', 'fddr', 'fps']
assert all(col in df.columns for col in required_cols), f"Missing columns: {required_cols}"

df['fcpu_ghz'] = df['fcpu'] / 1e6  # kHz → GHz
df['fgpu_ghz'] = df['fgpu'] / 1e9  # Hz → GHz
df['fddr_ghz'] = df['fddr'] / 1e9  # Hz → GHz
df['latency'] = 1.0 / df['fps']

# Filter valid positive values
mask = (df[['fcpu_ghz', 'fgpu_ghz', 'fddr_ghz', 'latency']] > 0).all(axis=1)
df = df[mask].copy()

fcpu = df['fcpu_ghz'].values
fgpu = df['fgpu_ghz'].values
fddr = df['fddr_ghz'].values
y_true = df['latency'].values


# --- Updated Model B: φ = α1 + α2 * exp(-α3 * fddr) ---
def model_b_updated(freqs, a1, a2, a3, beta1, beta2):
    """
    Parameters:
        a1: baseline DDR latency (>=0)
        a2: amplitude of exponential term (>=0)
        a3: decay rate (>=0)
        beta1, beta2: CPU/GPU weights (>=0)
    """
    fcpu_ghz, fgpu_ghz, fddr_ghz = freqs

    # New DDR model: φ = a1 + a2 * exp(-a3 * fddr)
    phi_ddr = a1 + a2 * np.exp(-a3 * fddr_ghz)

    # CPU+GPU bottleneck term
    cpu_gpu_term = beta1 / (fcpu_ghz + 1e-12) + beta2 / (fgpu_ghz + 1e-12)

    # Total latency = max(DDR delay, CPU+GPU delay)
    latency_pred = np.maximum(phi_ddr, cpu_gpu_term)
    return latency_pred


# Initial guess
# - a1: small baseline, e.g., 0.01
# - a2: ~0.1–1.0 (since latency is in seconds)
# - a3: ~1.0 (GHz scale)
# - beta1, beta2: ~1.0
p0 = [0.01, 0.5, 1.0, 1.0, 1.0]  # a1, a2, a3, beta1, beta2

# Enforce all parameters >= 0
bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

try:
    popt, _ = curve_fit(
        model_b_updated,
        (fcpu, fgpu, fddr),
        y_true,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    a1, a2, a3, beta1, beta2 = popt
    y_pred = model_b_updated((fcpu, fgpu, fddr), *popt)

    # Metrics
    mre = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print("✅ Updated Model B Fitted!")
    print(f"φ(fddr) = {a1:.6f} + {a2:.6f} * exp(-{a3:.4f} * fddr)")
    print(f"β1 (CPU weight): {beta1:.4f}")
    print(f"β2 (GPU weight): {beta2:.4f}")
    print(f"MRE: {mre:.2f} %")
    print(f"MAE: {mae:.6f} s, RMSE: {rmse:.6f} s, R²: {r2:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame([{
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'beta1': beta1,
        'beta2': beta2,
        'MRE_%': mre,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'n_samples': len(y_true)
    }])
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Plot: True vs Predicted Latency
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidth=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
    plt.xlabel('True Latency (s)')
    plt.ylabel('Predicted Latency (s)')
    plt.title('Model B: 3-Param Exponential DDR\nLatency = max(a1+a2·exp(-a3·fddr), β1/fcpu + β2/fgpu)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "latency_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # FPS view
    fps_true = 1.0 / y_true
    fps_pred = 1.0 / y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(fps_true, fps_pred, alpha=0.7)
    plt.plot([fps_true.min(), fps_true.max()], [fps_true.min(), fps_true.max()], 'r--')
    plt.xlabel('True FPS')
    plt.ylabel('Predicted FPS')
    plt.title('FPS View (Updated Model B)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fps_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

except Exception as e:
    print("❌ Updated Model B fitting failed:", e)
    import traceback
    traceback.print_exc()

print(f"\nAll outputs saved to: {output_dir}")