import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================
# 1. 数据加载与预处理
# ========================
def load_and_preprocess(csv_path, temp_col=None):
    """
    加载CSV数据，预处理帧时间、DDR频率单位等。
    参数:
        csv_path: CSV文件路径
        temp_col: 温度列名，若为None则自动尝试常见名称
    返回:
        X: 特征矩阵 (n_samples, 8) 列顺序: fcpu, fgpu, fddr_GHz, temp, cpu_uti, gpu_uti, ddr_uti, chr
        y: 帧时间 (ms)
        f_max_cpu, f_max_gpu: CPU和GPU的最大频率
    """
    df = pd.read_csv(csv_path)
    
    # 自动检测温度列
    if temp_col is None:
        possible_temp_cols = ['temp_soc', 'temperature', 'T_soc', 'temp', 'soc_temp']
        for col in possible_temp_cols:
            if col in df.columns:
                temp_col = col
                break
        if temp_col is None:
            raise KeyError("未找到温度列，请通过 temp_col 参数指定温度列名。")
    
    # 计算帧时间 (ms)
    if 'fps' not in df.columns:
        raise KeyError("CSV中缺少 'fps' 列")
    df['T_ms'] = 1000.0 / df['fps']
    
    # DDR频率转换为 GHz
    if 'fddr' not in df.columns:
        raise KeyError("CSV中缺少 'fddr' 列")
    df['fddr_GHz'] = df['fddr'] / 1000.0
    
    # 提取最大频率（用于频率缩放因子）
    f_max_cpu = df['fcpu'].max()
    f_max_gpu = df['fgpu'].max()
    
    # 构建资源矩阵 X: 列顺序 [fcpu, fgpu, fddr_GHz, temp, cpu_uti, gpu_uti, ddr_uti, cache_hit]
    required_cols = ['fcpu', 'fgpu', 'fddr_GHz', temp_col, 'cpu_uti', 'gpu_uti', 'ddr_uti', 'cache_hit']
    X = df[required_cols].values
    y = df['T_ms'].values
    
    return X, y, f_max_cpu, f_max_gpu

# ========================
# 2. 模型定义
# ========================
def predict_T(params, X, f_max_cpu, f_max_gpu):
    """
    params: 待拟合系数数组，顺序如下:
        0-3:   a_cpu0, a_cpuU, a_cpu_miss, a_cpuT
        4-7:   a_gpu0, a_gpuU, a_gpu_miss, a_gpuT
        8-11:  tau00, theta0_T, theta0_U, theta0_miss
        12-15: tau10, theta1_T, theta1_U, theta1_miss
        16-18: lambda0, theta_lambda_U, theta_lambda_miss
    """
    # 解析参数
    a_cpu0, a_cpuU, a_cpu_miss, a_cpuT = params[0:4]
    a_gpu0, a_gpuU, a_gpu_miss, a_gpuT = params[4:8]
    tau00, theta0_T, theta0_U, theta0_miss = params[8:12]
    tau10, theta1_T, theta1_U, theta1_miss = params[12:16]
    lambda0, theta_lambda_U, theta_lambda_miss = params[16:19]
    
    # 提取资源向量列
    f_cpu   = X[:, 0]
    f_gpu   = X[:, 1]
    f_ddr   = X[:, 2]          # GHz
    temp    = X[:, 3]
    u_cpu   = X[:, 4]
    u_gpu   = X[:, 5]
    u_ddr   = X[:, 6]
    chr_    = X[:, 7]          # cache hit rate
    miss = 1.0 - chr_          # cache miss rate
    
    # ---------------- CPU-GPU 部分 ----------------
    alpha_cpu = a_cpu0 + a_cpuU * u_cpu + a_cpu_miss * miss + a_cpuT * temp
    alpha_gpu = a_gpu0 + a_gpuU * u_gpu + a_gpu_miss * miss + a_gpuT * temp
    T_cpugpu = alpha_cpu * (f_max_cpu / f_cpu) + alpha_gpu * (f_max_gpu / f_gpu)
    
    # ---------------- DDR 部分 (指数饱和) ----------------
    tau0 = tau00 + theta0_T * temp + theta0_U * u_ddr + theta0_miss * miss
    tau1 = tau10 + theta1_T * temp + theta1_U * u_ddr + theta1_miss * miss
    lam  = lambda0 + theta_lambda_U * u_ddr + theta_lambda_miss * miss
    # 保证 tau1 >= 0, lam >= 0 (边界约束已处理，但计算中仍取绝对值或clip)
    tau1 = np.maximum(tau1, 0)
    lam  = np.maximum(lam, 0)
    T_ddr = tau0 + tau1 * np.exp(-lam * f_ddr)
    
    # 整体帧时间 = max(CPU-GPU, DDR)
    T_pred = np.maximum(T_cpugpu, T_ddr)
    return T_pred

# ========================
# 3. 损失函数
# ========================
def loss(params, X, y, f_max_cpu, f_max_gpu):
    y_pred = predict_T(params, X, f_max_cpu, f_max_gpu)
    return np.mean((y - y_pred) ** 2)

# ========================
# 4. 拟合主函数
# ========================
def fit_model(csv_path, temp_col=None, method='L-BFGS-B', max_iter=5000):
    """
    拟合模型参数。
    参数:
        csv_path: CSV文件路径
        temp_col: 温度列名，若不指定则自动检测
        method: 优化方法，默认 'L-BFGS-B'
        max_iter: 最大迭代次数
    返回:
        fitted_params: 拟合后的参数数组
        y_pred: 训练集预测值
    """
    X, y, f_max_cpu, f_max_gpu = load_and_preprocess(csv_path, temp_col)
    
    # 初始参数猜测 (根据物理量级设定)
    init_params = np.array([
        5.0,  0.5,  2.0,  0.1,   # a_cpu0, a_cpuU, a_cpu_miss, a_cpuT
        5.0,  0.5,  2.0,  0.1,   # a_gpu0, a_gpuU, a_gpu_miss, a_gpuT
        5.0,  0.1,  0.5,  2.0,   # tau00, theta0_T, theta0_U, theta0_miss
        20.0, 0.2, -0.5,  2.0,   # tau10, theta1_T, theta1_U, theta1_miss
        2.0, -0.3, -0.5          # lambda0, theta_lambda_U, theta_lambda_miss
    ])
    
    # 参数边界 (基于物理符号预期)
    lower_bounds = []
    upper_bounds = []
    bounds_info = [
        ('a_cpu0', 0, 50), ('a_cpuU', 0, 10), ('a_cpu_miss', 0, 20), ('a_cpuT', 0, 5),
        ('a_gpu0', 0, 50), ('a_gpuU', 0, 10), ('a_gpu_miss', 0, 20), ('a_gpuT', 0, 5),
        ('tau00', 0, 50), ('theta0_T', 0, 5), ('theta0_U', 0, 10), ('theta0_miss', 0, 20),
        ('tau10', 0, 100), ('theta1_T', 0, 10), ('theta1_U', -10, 0), ('theta1_miss', 0, 20),
        ('lambda0', 0, 10), ('theta_lambda_U', -10, 0), ('theta_lambda_miss', -10, 0)
    ]
    for _, lb, ub in bounds_info:
        lower_bounds.append(lb)
        upper_bounds.append(ub)
    
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # 优化
    result = minimize(
        loss, init_params, args=(X, y, f_max_cpu, f_max_gpu),
        method=method, bounds=bounds, options={'maxiter': max_iter, 'disp': True}
    )
    
    if result.success:
        print("优化成功！")
    else:
        print("优化失败:", result.message)
    
    # 输出拟合参数
    param_names = [name for name, _, _ in bounds_info]
    fitted_params = result.x
    print("\n拟合参数:")
    for name, val in zip(param_names, fitted_params):
        print(f"  {name:20} = {val:.6f}")
    
    # 预测并评估
    y_pred = predict_T(result.x, X, f_max_cpu, f_max_gpu)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n评估指标 (训练集):")
    print(f"  RMSE = {rmse:.4f} ms")
    print(f"  MAE  = {mae:.4f} ms")
    print(f"  R²   = {r2:.4f}")
    
    return result.x, y_pred

# ========================
# 5. 示例运行
# ========================
if __name__ == "__main__":
    # 请将 'your_data.csv' 替换为实际文件路径
    # 若温度列名不是常见名称，通过 temp_col 指定，例如 temp_col='soc_temperature'
    fitted_params, predictions = fit_model('your_data.csv')
