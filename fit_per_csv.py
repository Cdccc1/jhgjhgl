from __future__ import annotations

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = r"D:\Google download\data"
TARGET_COL = "fps"  # 修改：目标是帧率，我们拟合帧率的倒数


def safe_name(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", base)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c = str(c).replace("\ufeff", "")  # 去BOM
        c = c.strip()
        c = c.lower()
        new_cols.append(c)
    df.columns = new_cols
    return df


def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df["__file__"] = os.path.basename(path)
    df = normalize_columns(df)
    return df


def add_freq_ghz(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一单位到GHz：
      fcpu: kHz -> GHz (除以1e6)
      fgpu: hz -> GHz (除以1e9)
      ddr: 固定为1.6GHz
    """
    out = df.copy()

    if "fcpu" not in out.columns or "fgpu" not in out.columns:
        raise KeyError(
            f"Missing freq columns. Need 'fcpu' and 'fgpu'. "
            f"Got columns: {list(out.columns)}"
        )
    if TARGET_COL not in out.columns:
        raise KeyError(
            f"Missing target column '{TARGET_COL}'. Got columns: {list(out.columns)}"
        )

    # 转换频率单位
    out["fcpu_ghz"] = pd.to_numeric(out["fcpu"], errors="coerce") / 1e6
    out["fgpu_ghz"] = pd.to_numeric(out["fgpu"], errors="coerce") / 1e9
    out["ddr_ghz"] = 1.6  # 固定DDR频率为1.6GHz

    # 计算帧率的倒数（即每帧时间）
    out["frame_time"] = 1.0 / pd.to_numeric(out[TARGET_COL], errors="coerce")
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
    need = ["fcpu_ghz", "fgpu_ghz", "ddr_ghz", "frame_time"]
    work = df[need].copy().replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["fcpu_ghz"] > 0) & (work["fgpu_ghz"] > 0) & (work["frame_time"] > 0)]
    if len(work) < 5:
        raise ValueError(f"Too few valid rows after cleaning: {len(work)}")

    fcpu = work["fcpu_ghz"].to_numpy(float)
    fgpu = work["fgpu_ghz"].to_numpy(float)
    fddr = work["ddr_ghz"].to_numpy(float)  # DDR频率固定
    y = work["frame_time"].to_numpy(float)  # 目标是帧率的倒数（每帧时间）

    fcpu_max = float(np.max(fcpu))
    fgpu_max = float(np.max(fgpu))
    fddr_max = float(np.max(fddr))  # 对于固定值，最大值就是其本身

    x_cpu = fcpu_max / fcpu
    x_gpu = fgpu_max / fgpu
    x_ddr = fddr_max / fddr  # 对于固定值，x_ddr也是固定值

    # 使用约束优化：alpha_cpu + alpha_gpu + alpha_ddr = 1
    def obj(params):
        # 参数: [alpha_cpu, alpha_gpu, D_min]
        alpha_cpu, alpha_gpu, D_min = params
        alpha_ddr = 1.0 - alpha_cpu - alpha_gpu  # 由约束条件确定

        # 确保所有alpha值非负
        if alpha_cpu < 0 or alpha_gpu < 0 or alpha_ddr < 0 or D_min < 0:
            return 1e9

        # 计算预测值
        y_pred = (
                alpha_cpu * D_min * x_cpu +
                alpha_gpu * D_min * x_gpu +
                alpha_ddr * D_min * x_ddr
        )

        # 使用相对误差作为目标函数
        eps = 1e-12
        rel_err = np.abs(y_pred - y) / np.maximum(np.abs(y), eps)
        return float(np.mean(rel_err))

    # 初始参数: [alpha_cpu, alpha_gpu, D_min]
    # 约束: alpha_cpu + alpha_gpu + alpha_ddr = 1, 且所有alpha >= 0
    y_mean = float(np.mean(y))
    init = np.array([0.3, 0.3, y_mean], float)

    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},  # alpha_cpu >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]},  # alpha_gpu >= 0
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},  # alpha_ddr >= 0
        {'type': 'ineq', 'fun': lambda x: x[2]}  # D_min >= 0
    ]

    bounds = [(0, 1), (0, 1), (0, None)]

    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 8000, "ftol": 1e-12})

    if not res.success:
        raise RuntimeError(res.message)

    alpha_cpu, alpha_gpu, D_min = res.x
    alpha_ddr = 1.0 - alpha_cpu - alpha_gpu

    # 计算预测值
    y_pred = (
            alpha_cpu * D_min * x_cpu +
            alpha_gpu * D_min * x_gpu +
            alpha_ddr * D_min * x_ddr
    )

    mre, mae, rmse, r2 = calc_metrics(y, y_pred)

    out = work.copy()
    out["T_hat"] = y_pred
    out["rel_err"] = np.abs(out["T_hat"] - out["frame_time"]) / np.maximum(out["frame_time"], 1e-12)
    out["x_cpu"] = x_cpu
    out["x_gpu"] = x_gpu
    out["x_ddr"] = x_ddr

    return {
        "rows": int(len(work)),
        "fcpu_max_ghz": fcpu_max,
        "fgpu_max_ghz": fgpu_max,
        "fddr_max_ghz": fddr_max,
        "alpha_cpu": alpha_cpu,
        "alpha_gpu": alpha_gpu,
        "alpha_ddr": alpha_ddr,
        "D_min": D_min,
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
                "fddr_max_ghz": r["fddr_max_ghz"],
                "alpha_cpu": r["alpha_cpu"],
                "alpha_gpu": r["alpha_gpu"],
                "alpha_ddr": r["alpha_ddr"],
                "D_min": r["D_min"],
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
        print(ok[["file", "rows", "MRE", "MAE", "RMSE", "R2", "alpha_cpu", "alpha_gpu", "alpha_ddr"]].head(5).to_string(
            index=False))

        print("\nWorst 5 by MRE:")
        print(ok[["file", "rows", "MRE", "MAE", "RMSE", "R2", "alpha_cpu", "alpha_gpu", "alpha_ddr"]].tail(5).to_string(
            index=False))

    bad = summary[summary["error"] != ""]
    if len(bad):
        print("\nFailed files:")
        print(bad[["file", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()