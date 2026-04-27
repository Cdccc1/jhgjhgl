import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

# 1. 定义软最大值 (Soft Maximum)
def smooth_max(a, b, alpha=20.0):
    # LogSumExp 形式的平滑最大值
    return np.log(np.exp(alpha * a) + np.exp(alpha * b)) / alpha

def performance_model(X, w_c0, w_g0, beta_c1, omega_cmd):
    """
    X: 输入特征元组
    参数: 待拟合系数
    """
    # 解包输入数据
    # 注意：X中的频率单位：f_cpu(MHz), f_gpu(GHz), f_ddr(GHz)
    util_cpu, util_gpu, cache_hit, B, W_mem_total, P_hit, P_miss, avg_osd_r, \
    f_cpu, f_gpu, f_ddr, f_cpu_max, f_gpu_max = X
    
    # --- 常量定义 ---
    Width = 8          # 总线位宽 (Bytes)
    N_banks = 8        # Bank数量
    # 时序常量 30ns -> 0.03us (微秒)，以便与后续计算单位统一
    tau_RCD = 0.03     
    tau_CL = 0.03      
    
    # --- 1. 计算 CPU/GPU 计算时间分量 ---
    
    # alpha_cpu: 注意分母 B (带宽)
    # 为了防止除以0，给 B 加一个极小值
    alpha_cpu = w_c0 * util_cpu + beta_c1 * (util_cpu * (1 - cache_hit)) / (B + 1e-9)
    
    # alpha_gpu: 简化公式
    alpha_gpu = w_g0 * util_gpu
    
    # 计算受限时间 (Compute Bound)
    # f_cpu (MHz), f_gpu (GHz) -> 比率无量纲，只要分子分母单位一致即可
    compute_bound = alpha_cpu * (f_cpu_max / f_cpu) + alpha_gpu * (f_gpu_max / f_gpu)
    
    # --- 2. 计算 DDR 访存时间分量 (T_ddr) ---
    
    # 2.1 总线基础传输时间 (Bus Transfer Time)
    # 公式: W / (2 * Width * f * Omega)
    # f_ddr 输入是 GHz，公式通常用 MHz。转换: 1 GHz = 1000 MHz
    # 假设 W_mem_total 单位为 MBytes，f_ddr 为 MHz，则结果单位为 Microseconds (us)
    f_ddr_mhz = f_ddr * 1000.0
    bus_transfer_time = W_mem_total / (2.0 * Width * f_ddr_mhz * omega_cmd + 1e-9)
    
    # 2.2 电路延迟摊销 (Circuit Delay Amortization)
    # 分子: 平均延迟 = P_hit * tCL + P_miss * (tRCD + tCL)
    avg_latency = P_hit * tau_CL + P_miss * (tau_RCD + tau_CL)
    
    # 分母: 动态并行度 (BLP)，取 avg_osd_r 和 N_banks 的最小值
    # 防止除以0
    blp = np.minimum(avg_osd_r, N_banks)
    blp = np.maximum(blp, 1.0) 
    
    # N_req: 请求数量。公式中未明确定义，通常 N_req = Total_Data / Request_Size
    # 假设 W_mem_total 是总数据量 (MBytes)，每个请求 64Bytes (0.000064 MBytes)
    # 这里假设 N_req 正比于 W_mem_total。若 W_mem_total 本身代表请求数，请修改此处。
    # 假设每个请求传输 64 Bytes (Cache Line 大小)
    bytes_per_req = 64.0 / (1024 * 1024) # 64 Bytes in MBytes
    N_req = W_mem_total / bytes_per_req
    
    circuit_delay = N_req * (avg_latency / blp)
    
    T_ddr = bus_transfer_time + circuit_delay
    
    # --- 3. 总时间与帧率 ---
    # 使用软最大值平滑切换
    single_frame_time = smooth_max(compute_bound, T_ddr, alpha=20.0)
    
    # 返回 FPS
    # single_frame_time 单位是 us (微秒)，FPS = 1,000,000 / Time
    return 1000000.0 / single_frame_time

def mre_loss(params, X, y_true):
    y_pred = performance_model(X, *params)
    # 计算 MRE
    mre = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100
    return mre

def main():
    csv_file = r"user\data.csv"
    df = pd.read_csv(csv_file)
    
    # 数据预处理 (保持你的单位处理逻辑)
    f_cpu_max = df['fcpu'].max() / 1e6  # MHz
    f_gpu_max = df['fgpu'].max() / 1e9   # GHz
    
    # 构造输入矩阵 X
    # 注意：这里假设 df['B'] 是总带宽 (MB/s)，df['W_mem_total'] 是总数据量 (MB)
    X_data = (
        df['cpu_uti'].values / 100.0,          # util_cpu
        df['gpu_uti'].values / 100.0,          # util_gpu
        df['cache_hit'].values / 100.0,        # cache_hit
        df['B'].values,                        # B (总带宽)
        df['W_mem_total'].values,              # W_mem_total (总数据量)
        df['P_hit'].values,                    # P_hit (概率 0-1)
        df['P_miss'].values,                   # P_miss (概率 0-1)
        df['avg_osd_r'].values,                # avg_osd_r
        df['fcpu'].values / 1e6,               # f_cpu (MHz)
        df['fgpu'].values / 1e9,               # f_gpu (GHz)
        df['fddr'].values / 1e9,               # f_ddr (GHz)
        f_cpu_max, 
        f_gpu_max
    )
    
    y_data = df['fps'].values
    
    # 参数边界设置
    # w_c0, w_g0, beta_c1, omega_cmd
    bounds = [
        (0.1, 100.0),  # w_c0
        (0.1, 100.0),  # w_g0
        (0.0, 100.0),  # beta_c1
        (0.1, 5.0)     # omega_cmd (指令瓶颈系数)
    ]
    
    print("开始全局搜索 (Differential Evolution)...")
    result_global = differential_evolution(
        mre_loss,
        bounds,
        args=(X_data, y_data),
        strategy='best1bin',
        popsize=20,
        tol=1e-4,
        seed=42,
        maxiter=1000,
        disp=True
    )
    
    print(f"全局搜索 MRE: {result_global.fun:.4f}%")
    
    print("开始局部微调 (L-BFGS-B)...")
    result_local = minimize(
        mre_loss,
        result_global.x,
        args=(X_data, y_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 5000}
    )
    
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
    param_names = ['w_c0', 'w_g0', 'beta_c1', 'Omega_cmd']
    print("\n拟合参数结果:")
    for name, val in zip(param_names, popt):
        print(f"{name}: {val:.6f}")

if __name__ == "__main__":
    main()
