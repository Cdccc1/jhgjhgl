# model_b_exponential_ddr_decay.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import curve_fit
import os
import warnings

warnings.filterwarnings("ignore")

DATA_FILE = r"D:\Google download\data\1_color_newbrand.csv"
output_dir = r"D:\Google download\data\model_b_exponential_ddr_decay"
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


# --- Model B (Decay Exponential DDR): φ = exp(-b1 * fddr) ---
def model_b_decay(freqs, b1, alpha1, alpha2):
    fcpu_ghz, fgpu_ghz, fddr_ghz = freqs

    # φ(fddr) = exp(-b1 * fddr_ghz)  ← decay form (b1 >= 0)
    phi_ddr = np.exp(-b1 * fddr_ghz)

    # CPU+GPU bottleneck term
    cpu_gpu_term = alpha1 / (fcpu_ghz + 1e-12) + alpha2 / (fgpu_ghz + 1e-12)

    # Total latency = max(DDR delay, CPU+GPU delay)
    latency_pred = np.maximum(phi_ddr, cpu_gpu_term)
    return latency_pred


# Initial guess: b1 ~ 1.0 (since fddr in GHz, e^{-1*1.5} ≈ 0.22 is reasonable)
p0 = [1.0, 1.0, 1.0]  # b1, alpha1, alpha2

# Enforce b1 >= 0, alpha1 >= 0, alpha2 >= 0
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

try:
    popt, _ = curve_fit(
        model_b_decay,
        (fcpu, fgpu, fddr),
        y_true,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    b1, alpha1, alpha2 = popt
    y_pred = model_b_decay((fcpu, fgpu, fddr), *popt)

    # Metrics
    mre = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print("✅ Model B (Decay Exponential DDR) Fitted!")
    print(f"φ(fddr) = exp(-{b1:.4f} * fddr)")
    print(f"α1 (CPU weight): {alpha1:.4f}")
    print(f"α2 (GPU weight): {alpha2:.4f}")
    print(f"MRE: {mre:.2f} %")
    print(f"MAE: {mae:.6f} s, RMSE: {rmse:.6f} s, R²: {r2:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame([{
        'b1': b1,
        'alpha1': alpha1,
        'alpha2': alpha2,
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
    plt.title('Model B: Decay Exponential DDR\nLatency = max(exp(-b1·fddr), α1/fcpu + α2/fgpu)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "latency_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Optional: FPS view
    fps_true = 1.0 / y_true
    fps_pred = 1.0 / y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(fps_true, fps_pred, alpha=0.7)
    plt.plot([fps_true.min(), fps_true.max()], [fps_true.min(), fps_true.max()], 'r--')
    plt.xlabel('True FPS')
    plt.ylabel('Predicted FPS')
    plt.title('FPS View (Model B)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fps_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

except Exception as e:
    print("❌ Model B (decay) fitting failed:", e)
    import traceback

    traceback.print_exc()

print(f"\nAll outputs saved to: {output_dir}")