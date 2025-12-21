from __future__ import annotations

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize


DATA_DIR = r"D:\Google download\data"
TARGET_COL = "total_time"


# 你可以在这里补充更多“可能出现的列名”
ALIASES = {
    "fcpu": {"fcpu", "f_cpu", "cpu_freq", "cpu_frequency", "cpu_khz", "cpu"},
    "fgpu": {"fgpu", "f_gpu", "gpu_freq", "gpu_frequency", "gpu_hz", "gpu"},
    "total_time": {"total_time", "total", "frame_time", "t_total", "totaltime"},
}


def safe_name(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", base)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c = str(c).replace("\ufeff", "")   # 去BOM
        c = c.strip()
        c = c.lower()
        new_cols.append(c)
    df.columns = new_cols
    return df


def rename_by_alias(df: pd.DataFrame) -> pd.DataFrame:
    """
    将各种可能的列名映射为标准列名：fcpu, fgpu, total_time
    """
    df = df.copy()
    colset = set(df.columns)

    rename_map = {}
    for std, candidates in ALIASES.items():
        hit = None
        for cand in candidates:
            if cand in colset:
                hit = cand
                break
        if hit is not None and hit != std:
            rename_map[hit] = std

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df["__file__"] = os.path.basename(path)
    df = normalize_columns(df)
    df = rename_by_alias(df)
    return df


def add_freq_ghz(df: pd.DataFrame, fcpu_col="fcpu", fgpu_col="fgpu") -> pd.DataFrame:
    """
    统一单位到GHz：
      fcpu: kHz -> GHz
      fgpu: Hz  -> GHz
    """
    out = df.copy()

    if fcpu_col not in out.columns or fgpu_col not in out.columns:
        raise KeyError(
            f"Missing freq columns. Need '{fcpu_col}' and '{fgpu_col}'. "
            f"Got columns: {list(out.columns)}"
        )
    if TARGET_COL not in out.columns:
        raise KeyError(
            f"Missing target column '{TARGET_COL}'. Got columns: {list(out.columns)}"
        )

    out["fcpu_ghz"] = pd.to_numeric(out[fcpu_col], errors="coerce") / 1e6
    out["fgpu_ghz"] = pd.to_numeric(out[fgpu_col], errors="coerce") / 1e9
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    return out


def calc_metrics(y_true, y_pred):
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


def fit_one_file(df: pd.DataFrame):
    need = ["fcpu_ghz", "fgpu_ghz", TARGET_COL]
    work = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["fcpu_ghz"] > 0) & (work["fgpu_ghz"] > 0) & (work[TARGET_COL] > 0)]
    if len(work) < 5:
        raise ValueError(f"Too few valid rows after cleaning: {len(work)}")

    fcpu = work["fcpu_ghz"].to_numpy(float)
    fgpu = work["fgpu_ghz"].to_numpy(float)
    y = work[TARGET_COL].to_numpy(float)

    fcpu_max = float(np.max(fcpu))
    fgpu_max = float(np.max(fgpu))

    x_cpu = fcpu_max / fcpu
    x_gpu = fgpu_max / fgpu

    # fgpu若不变，则x_gpu恒为1 -> GPU项与常数项不可辨识
    gpu_identifiable = float(np.std(x_gpu)) > 1e-8

    if not gpu_identifiable:
        # 退化为：T = b_cpu*x_cpu + b0
        def pred2(b):
            b_cpu, b0 = b
            return b_cpu * x_cpu + b0

        def obj2(b):
            if np.any(np.asarray(b) < 0):
                return 1e9
            yh = pred2(b)
            eps = 1e-12
            return float(np.mean(np.abs(yh - y) / np.maximum(np.abs(y), eps)))

        y_min = float(np.min(y))
        init = np.array([y_min * 0.2, y_min * 0.8], float)
        res = minimize(obj2, init, method="SLSQP", bounds=[(0, None), (0, None)],
                       options={"maxiter": 6000, "ftol": 1e-12})
        if not res.success:
            raise RuntimeError(res.message)
        beta_cpu = float(res.x[0])
        beta_gpu = 0.0
        beta0 = float(res.x[1])
        yh = pred2(res.x)

    else:
        # 正常：T = b_cpu*x_cpu + b_gpu*x_gpu + b0
        def pred3(b):
            b_cpu, b_gpu, b0 = b
            return b_cpu * x_cpu + b_gpu * x_gpu + b0

        def obj3(b):
            if np.any(np.asarray(b) < 0):
                return 1e9
            yh = pred3(b)
            eps = 1e-12
            return float(np.mean(np.abs(yh - y) / np.maximum(np.abs(y), eps)))

        y_min = float(np.min(y))
        init = np.array([y_min * 0.2, y_min * 0.1, y_min * 0.7], float)
        res = minimize(obj3, init, method="SLSQP",
                       bounds=[(0, None), (0, None), (0, None)],
                       options={"maxiter": 8000, "ftol": 1e-12})
        if not res.success:
            raise RuntimeError(res.message)
        beta_cpu, beta_gpu, beta0 = map(float, res.x)
        yh = pred3(res.x)

    mre, mae, rmse, r2 = calc_metrics(y, yh)

    D_min = beta_cpu + beta_gpu + beta0
    alpha_cpu = beta_cpu / D_min if D_min > 0 else np.nan
    alpha_gpu = beta_gpu / D_min if D_min > 0 else np.nan
    alpha_ddr = beta0 / D_min if D_min > 0 else np.nan

    out = work.copy()
    out["T_hat"] = yh
    out["rel_err"] = np.abs(out["T_hat"] - out[TARGET_COL]) / np.maximum(out[TARGET_COL], 1e-12)
    out["x_cpu"] = x_cpu
    out["x_gpu"] = x_gpu

    return {
        "rows": int(len(work)),
        "fcpu_max_ghz": fcpu_max,
        "fgpu_max_ghz": fgpu_max,
        "gpu_identifiable": bool(gpu_identifiable),
        "beta_cpu": beta_cpu,
        "beta_gpu": beta_gpu,
        "beta0": beta0,
        "D_min": D_min,
        "alpha_cpu": alpha_cpu,
        "alpha_gpu": alpha_gpu,
        "alpha_ddr": alpha_ddr,
        "MRE": mre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "pred_df": out,
    }


def main():
    files = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        b = os.path.basename(p).lower()
        if b.startswith("fit_") or b.startswith("per_file_"):
            continue
        files.append(p)
    if not files:
        raise FileNotFoundError(f"No csv in {DATA_DIR}")

    summary_rows = []

    for path in files:
        base = os.path.basename(path)
        tag = safe_name(base)

        try:
            df = read_one_csv(path)
            df = add_freq_ghz(df)
            r = fit_one_file(df)

            pred_path = os.path.join(DATA_DIR, f"per_file_predictions_{tag}.csv")
            worst_path = os.path.join(DATA_DIR, f"per_file_worst10_{tag}.csv")

            r["pred_df"].to_csv(pred_path, index=False)
            r["pred_df"].sort_values("rel_err", ascending=False).head(10).to_csv(worst_path, index=False)

            summary_rows.append({
                "file": base,
                "rows": r["rows"],
                "fcpu_max_ghz": r["fcpu_max_ghz"],
                "fgpu_max_ghz": r["fgpu_max_ghz"],
                "gpu_identifiable": r["gpu_identifiable"],
                "beta_cpu": r["beta_cpu"],
                "beta_gpu": r["beta_gpu"],
                "beta0": r["beta0"],
                "D_min": r["D_min"],
                "alpha_cpu": r["alpha_cpu"],
                "alpha_gpu": r["alpha_gpu"],
                "alpha_ddr": r["alpha_ddr"],
                "MRE": r["MRE"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
                "error": ""
            })

        except Exception as e:
            summary_rows.append({
                "file": base,
                "rows": 0,
                "error": str(e)
            })

    summary = pd.DataFrame(summary_rows)
    out_path = os.path.join(DATA_DIR, "per_file_fit_summary.csv")
    summary.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    ok = summary[summary["error"] == ""].copy()
    if len(ok):
        ok = ok.sort_values("MRE")
        print("\nBest 5 by MRE:")
        print(ok[["file", "rows", "gpu_identifiable", "MRE", "MAE", "RMSE", "R2"]].head(5).to_string(index=False))

        print("\nWorst 5 by MRE:")
        print(ok[["file", "rows", "gpu_identifiable", "MRE", "MAE", "RMSE", "R2"]].tail(5).to_string(index=False))

    bad = summary[summary["error"] != ""]
    if len(bad):
        print("\nFailed files:")
        print(bad[["file", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()