import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

# 1. 定义软最大值 (Soft Maximum)
# 替代 np.maximum，解决模型在“计算受限”和“访存受限”切换时的不可导问题
# alpha 越大，越接近真实的 max，但在优化初期建议用较小的值（如 5-10）帮助收敛
def smooth_max(a, b, alpha=10.0):
    return (np.exp(alpha * a) + np.exp(alpha * b)) / alpha - np.log(2) / alpha
    # 或者使用更简单的 LogSumExp 形式:
    # return np.log(np.exp(a) + np.exp(b)) 

def performance_model(X, w_c0, w_c1, w_g0, w_g1, theta_r, theta_w, tau_0, gamma, lam):
    # 你的原始数据解包逻辑保持不变
    util_cpu, util_gpu, cache_hit, BW_r, BW_w, avg_osd_r, f_cpu, f_gpu, f_ddr, f_cpu_max, f_gpu_max = X
    
    # 你的原始公式计算逻辑保持不变
    alpha_cpu = w_c0 * util_cpu + w_c1 * util_cpu * (1 - cache_hit)
    alpha_gpu = w_g0 * util_gpu + w_g1 * util_gpu * (1 - cache_hit)
    
    compute_bound = alpha_cpu * (f_cpu_max / f_cpu) + alpha_gpu * (f_gpu_max / f_gpu)
    
    T_ddr = (theta_r * BW_r + theta_w * BW_w) * (tau_0 + (1 + gamma * avg_osd_r) * np.exp(-lam * f_ddr))
    
    # 【修改点】使用软最大值替代 np.maximum
    # 注意：这里使用 soft_max 仅在拟合阶段辅助收敛，若需严格物理意义，拟合后可换回 hard max 验证
    single_frame_time = smooth_max(compute_bound, T_ddr, alpha=20.0) 
    
    return 1000.0 / single_frame_time

def mre_loss(params, X, y_true):
    y_pred = performance_model(X, *params)
    # 防止除以零，使用极小值保护
    mre = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100
    return mre

def main():
    csv_file = r"user\data.csv"
    df = pd.read_csv(csv_file)
    
    # 你的原始数据处理逻辑保持不变
    f_cpu_max = df['fcpu'].max() / 1e6
    f_gpu_max = df['fgpu'].max() / 1e9
    
    X_data = (
        df['cpu_uti'].values / 100, 
        df['gpu_uti'].values / 100, 
        df['cache_hit'].values / 100, 
        df['r_bandwidth'].values, 
        df['w_bandwidth'].values, 
        df['r_osd'].values, 
        df['fcpu'].values / 1e6, 
        df['fgpu'].values / 1e9, 
        df['fddr'].values / 1e9, # 保持你的单位处理
        f_cpu_max, 
        f_gpu_max
    )
    
    y_data = df['fps'].values
    
    # 【修改点】调整参数边界，特别是 lam
    # 原 lam (0, 5) 可能太小，导致指数项对频率不敏感。扩大范围以捕捉 DDR 变化
    bounds = [
        (0.1, 50.0), (0.0, 100.0), (0.1, 50.0), (0.0, 100.0), # w_c0, w_c1, w_g0, w_g1
        (0.5, 10.0), (0.0, 3.0),                               # theta_r, theta_w
        (0.1, 20.0), (0.0, 10.0), (0.1, 50.0)                  # tau_0, gamma, lam (范围扩大)
    ]
    
    print("开始全局搜索 (Differential Evolution)...")
    # 第一阶段：全局搜索
    result_global = differential_evolution(
        mre_loss,
        bounds,
        args=(X_data, y_data),
        strategy='best1bin',
        popsize=30,      # 增加种群数量以提高搜索质量
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        disp=True,
        maxiter=1000
    )
    
    print(f"全局搜索 MRE: {result_global.fun:.4f}%")
    
    print("开始局部微调 (L-BFGS-B)...")
    # 第二阶段：局部微调 (使用 L-BFGS-B 进行高精度收敛)
    result_local = minimize(
        mre_loss,
        result_global.x,  # 以全局搜索的结果为初值
        args=(X_data, y_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 5000}
    )
    
    # 最终结果
    if result_local.success:
        popt = result_local.x
        final_mre = result_local.fun
        print("局部微调成功！")
    else:
        popt = result_global.x
        final_mre = result_global.fun
        print("局部微调未收敛，使用全局搜索结果。")
        
    print(f"最终拟合 MRE: {final_mre:.4f}%")
    
    # 打印参数
    param_names = ['w_c0', 'w_c1', 'w_g0', 'w_g1', 'theta_r', 'theta_w', 'tau_0', 'gamma', 'lam']
    for name, val in zip(param_names, popt):
        print(f"{name}: {val:.6f}")

if __name__ == "__main__":
    main()
