from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = r"D:\Google download\data"
TARGET_COL = "total_time"

# DDR频率固定 => (f_ddr_max / f_ddr) 恒为 1
DDR_GHZ = 1.6


def read_all_csvs(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")  # 自动识别tab/逗号/空格
        df["__file__"] = os.path.basename(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def add_freq_ghz(df: pd.DataFrame, fcpu_col="fcpu", fgpu_col="fgpu") -> pd.DataFrame:
    """
    统一单位到GHz：
      fcpu: kHz -> GHz  ( /1e6 )
      fgpu: Hz  -> GHz  ( /1e9 )
    """
    out = df.copy()
    out["fcpu_ghz"] = pd.to_numeric(out[fcpu_col], errors="coerce") / 1e6
    out["fgpu_ghz"] = pd.to_numeric(out[fgpu_col], errors="coerce") / 1e9
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    return out


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    eps = 1e-12
    rel = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    mre = float(np.mean(rel))

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return mre, mae, rmse, r2


def fit_mre_beta(df: pd.DataFrame):
    """
    由于DDR比值恒为1，原模型可改写为：
      T_hat = beta_cpu*(fcpu_max/fcpu) + beta_gpu*(fgpu_max/fgpu) + beta0
    其中：
      beta_cpu = D_min * alpha_cpu
      beta_gpu = D_min * alpha_gpu
      beta0    = D_min * alpha_ddr   (此处代表“固定项”，不代表DDR频率敏感性)

    约束：beta_cpu,beta_gpu,beta0 >= 0
    目标：最小化 MRE
    """
    need = ["fcpu_ghz", "fgpu_ghz", TARGET_COL, "__file__"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    work = df[need].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["fcpu_ghz"] > 0) & (work["fgpu_ghz"] > 0) & (work[TARGET_COL] > 0)]
    if len(work) < 10:
        raise ValueError("Too few valid rows after cleaning; check column names/units.")

    fcpu = work["fcpu_ghz"].to_numpy(float)
    fgpu = work["fgpu_ghz"].to_numpy(float)
    y = work[TARGET_COL].to_numpy(float)

    fcpu_max = float(np.max(fcpu))
    fgpu_max = float(np.max(fgpu))

    x_cpu = fcpu_max / fcpu
    x_gpu = fgpu_max / fgpu
    x0 = np.ones_like(x_cpu)

    def y_hat(beta: np.ndarray) -> np.ndarray:
        b_cpu, b_gpu, b0 = beta
        return b_cpu * x_cpu + b_gpu * x_gpu + b0 * x0

    def objective(beta: np.ndarray) -> float:
        if np.any(beta < 0):
            return 1e9
        pred = y_hat(beta)
        eps = 1e-12
        rel = np.abs(pred - y) / np.maximum(np.abs(y), eps)
        return float(np.mean(rel))

    # 初值：用量级初始化，避免收敛到奇怪点
    y_min = float(np.min(y))
    beta0_init = y_min * 0.7
    beta_cpu_init = y_min * 0.2
    beta_gpu_init = y_min * 0.1
    x_init = np.array([beta_cpu_init, beta_gpu_init, beta0_init], float)

    res = minimize(
        objective,
        x_init,
        method="SLSQP",
        bounds=[(0, None), (0, None), (0, None)],
        options={"maxiter": 8000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    b_cpu, b_gpu, b0 = map(float, res.x)
    pred = y_hat(res.x)
    mre, mae, rmse, r2 = calc_metrics(y, pred)

    # 还原到原参数（满足 alpha和=1）
    D_min = b_cpu + b_gpu + b0
    alpha_cpu = b_cpu / D_min if D_min > 0 else np.nan
    alpha_gpu = b_gpu / D_min if D_min > 0 else np.nan
    alpha_ddr = b0 / D_min if D_min > 0 else np.nan

    out = work.copy()
    out["T_hat"] = pred
    out["rel_err"] = np.abs(out["T_hat"] - out[TARGET_COL]) / np.maximum(out[TARGET_COL], 1e-12)
    out["x_cpu"] = x_cpu
    out["x_gpu"] = x_gpu

    return {
        "beta_cpu": b_cpu,
        "beta_gpu": b_gpu,
        "beta0": b0,
        "D_min": D_min,
        "alpha_cpu": alpha_cpu,
        "alpha_gpu": alpha_gpu,
        "alpha_ddr": alpha_ddr,
        "fcpu_max_ghz": fcpu_max,
        "fgpu_max_ghz": fgpu_max,
        "MRE": mre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "pred_df": out,
        "opt": res,
    }


def main():
    df = read_all_csvs(DATA_DIR)
    df = add_freq_ghz(df, fcpu_col="fcpu", fgpu_col="fgpu")

    result = fit_mre_beta(df)

    print("===== Files/Rows =====")
    print(f"Total rows used: {len(result['pred_df'])}")
    print("===== Max freq (GHz) used in ratios =====")
    print(f"fcpu_max_ghz = {result['fcpu_max_ghz']}")
    print(f"fgpu_max_ghz = {result['fgpu_max_ghz']}")
    print("===== Fitted beta (recommended to report) =====")
    print(f"beta_cpu = {result['beta_cpu']:.10g}")
    print(f"beta_gpu = {result['beta_gpu']:.10g}")
    print(f"beta0    = {result['beta0']:.10g}")
    print("===== Restored original parameters =====")
    print(f"D_min     = {result['D_min']:.10g}")
    print(f"alpha_cpu = {result['alpha_cpu']:.10g}")
    print(f"alpha_gpu = {result['alpha_gpu']:.10g}")
    print(f"alpha_ddr = {result['alpha_ddr']:.10g}")
    print("===== Fit quality (focus on MRE) =====")
    print(f"MRE  = {result['MRE']:.6%}")
    print(f"MAE  = {result['MAE']:.6g}")
    print(f"RMSE = {result['RMSE']:.6g}")
    print(f"R2   = {result['R2']:.6g}")

    # 导出结果
    pred_path = os.path.join(DATA_DIR, "fit_predictions.csv")
    worst_path = os.path.join(DATA_DIR, "fit_worst10.csv")
    summary_path = os.path.join(DATA_DIR, "fit_summary.txt")

    result["pred_df"].to_csv(pred_path, index=False)
    worst10 = result["pred_df"].sort_values("rel_err", ascending=False).head(10)
    worst10.to_csv(worst_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== Fit summary =====\n")
        f.write(f"DATA_DIR: {DATA_DIR}\n")
        f.write(f"Rows used: {len(result['pred_df'])}\n")
        f.write(f"fcpu_max_ghz: {result['fcpu_max_ghz']}\n")
        f.write(f"fgpu_max_ghz: {result['fgpu_max_ghz']}\n\n")
        f.write("beta_cpu,beta_gpu,beta0:\n")
        f.write(f"{result['beta_cpu']},{result['beta_gpu']},{result['beta0']}\n\n")
        f.write("Restored D_min and alphas:\n")
        f.write(f"D_min={result['D_min']}\n")
        f.write(f"alpha_cpu={result['alpha_cpu']}\n")
        f.write(f"alpha_gpu={result['alpha_gpu']}\n")
        f.write(f"alpha_ddr={result['alpha_ddr']}\n\n")
        f.write("Metrics:\n")
        f.write(f"MRE={result['MRE']}\n")
        f.write(f"MAE={result['MAE']}\n")
        f.write(f"RMSE={result['RMSE']}\n")
        f.write(f"R2={result['R2']}\n")

    print("Saved:")
    print(f"  {pred_path}")
    print(f"  {worst_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
