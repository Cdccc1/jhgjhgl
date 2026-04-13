import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. 定义白盒化性能模型函数 (保持不变)
def performance_model(X, w_c0, w_c1, w_g0, w_g1, theta_r, theta_w, tau_0, gamma, lam):
    util_cpu, util_gpu, cache_hit, BW_r, BW_w, avg_osd_r, f_cpu, f_gpu, f_ddr, f_cpu_max, f_gpu_max = X
    
    # --- 计算受限项 (Compute-bound) ---
    alpha_cpu = w_c0 * util_cpu + w_c1 * util_cpu * (1 - cache_hit)
    alpha_gpu = w_g0 * util_gpu + w_g1 * util_gpu * (1 - cache_hit)
    compute_bound = alpha_cpu * (f_cpu_max / f_cpu) + alpha_gpu * (f_gpu_max / f_gpu)
    
    # --- 访存受限项 (Memory-bound) ---
    T_ddr = (theta_r * BW_r + theta_w * BW_w) * (tau_0 + (1 + gamma * avg_osd_r) * np.exp(-lam * f_ddr))
    
    # --- 最终统一模型 ---
    return np.maximum(compute_bound, T_ddr)

# 2. 【核心修改】自定义 Loss 函数：直接计算并返回 MRE
def mre_loss(params, X, y_true):
    # params 解包传递给模型
    y_pred = performance_model(X, *params)
    
    # 计算 MRE (加入 1e-9 防止分母为 0 的极端情况)
    mre = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100
    return mre

def main():
    csv_file = 'performance_data.csv'
    
    # 读取数据 (包含自动生成模拟数据以防报错的逻辑)
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取数据文件: {csv_file}")
    except FileNotFoundError:
        print(f"未找到 {csv_file}，正在生成模拟数据进行演示...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'util_cpu': np.random.uniform(0.1, 0.9, n),
            'util_gpu': np.random.uniform(0.1, 0.9, n),
            'cache_hit': np.random.uniform(0.8, 0.99, n),
            'BW_r': np.random.uniform(1000, 5000, n),
            'BW_w': np.random.uniform(500, 2000, n),
            'avg_osd_r': np.random.uniform(1, 10, n),
            'f_cpu': np.random.uniform(1000, 3000, n),
            'f_gpu': np.random.uniform(300, 1000, n),
            'f_ddr': np.random.uniform(1600, 3200, n)
        })
        f_cpu_max, f_gpu_max = 3000.0, 1000.0
        X_sim = (df['util_cpu'], df['util_gpu'], df['cache_hit'], df['BW_r'], df['BW_w'], 
                 df['avg_osd_r'], df['f_cpu'], df['f_gpu'], df['f_ddr'], f_cpu_max, f_gpu_max)
        df['T_true'] = performance_model(X_sim, 1.2, 0.8, 1.5, 1.0, 0.05, 0.02, 10.0, 0.1, 0.001) * (1 + np.random.normal(0, 0.05, n))

    f_cpu_max = df['f_cpu'].max()
    f_gpu_max = df['f_gpu'].max()

    X_data = (
        df['util_cpu'].values, df['util_gpu'].values, df['cache_hit'].values, 
        df['BW_r'].values, df['BW_w'].values, df['avg_osd_r'].values, 
        df['f_cpu'].values, df['f_gpu'].values, df['f_ddr'].values,
        f_cpu_max, f_gpu_max
    )
    y_data = df['T_true'].values

    # 3. 【核心修改】参数拟合设置
    # p0 顺序: w_c0, w_c1, w_g0, w_g1, theta_r, theta_w, tau_0, gamma, lam
    initial_guess = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 5.0, 0.1, 0.001]
    
    # minimize 的边界设置格式为 (min, max) 的元组列表
    # 强制所有物理参数为正，lam 不宜过大
    bounds = [
        (0, None), (0, None), (0, None), (0, None), # w_c0, w_c1, w_g0, w_g1
        (0, None), (0, None),                       # theta_r, theta_w
        (0, None), (0, None), (0, 1.0)              # tau_0, gamma, lam
    ]

    print("正在以最小化 MRE 为目标拟合参数，请稍候...\n")
    
    # 使用 L-BFGS-B 算法处理带边界的优化问题
    result = minimize(
        mre_loss, 
        initial_guess, 
        args=(X_data, y_data), 
        method='L-BFGS-B', 
        bounds=bounds
    )

    if not result.success:
        print(f"警告: 优化可能未完全收敛。原因: {result.message}")

    popt = result.x

    # 4. 打印拟合的参数值
    param_names = ['w_c0', 'w_c1', 'w_g0', 'w_g1', 'theta_r', 'theta_w', 'tau_0', 'gamma', 'lambda']
    print("="*40)
    print(" 拟合参数结果:")
    print("="*40)
    for name, val in zip(param_names, popt):
        print(f"{name:8s} = {val:.6f}")

    # 5. 打印带入参数的完整公式
    w_c0, w_c1, w_g0, w_g1, theta_r, theta_w, tau_0, gamma, lam = popt
    print("\n" + "="*40)
    print(" 完整性能公式 (已代入参数):")
    print("="*40)
    formula = (
        f"T_hat = MAX(\n"
        f"    [ {w_c0:.4f} * util_cpu + {w_c1:.4f} * util_cpu * (1 - cache_hit) ] * ({f_cpu_max:.1f} / f_cpu) +\n"
        f"    [ {w_g0:.4f} * util_gpu + {w_g1:.4f} * util_gpu * (1 - cache_hit) ] * ({f_gpu_max:.1f} / f_gpu),\n\n"
        f"    ( {theta_r:.4f} * BW_r + {theta_w:.4f} * BW_w ) * "
        f"[ {tau_0:.4f} + (1 + {gamma:.4f} * avg_osd_r) * EXP(-{lam:.6f} * f_ddr) ]\n"
        f")"
    )
    print(formula)

    # 6. 打印最终的 MRE (即 result.fun，因为我们的 Loss 返回的就是 MRE 百分比)
    print("\n" + "="*40)
    print(f" 模型最终拟合指标 (MRE): {result.fun:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
