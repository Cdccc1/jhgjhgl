import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ========================
# 1. 数据加载与预处理（保持不变，但返回完整数据以便划分）
# ========================
def load_data(csv_path, temp_col=None):
    """
    加载CSV，返回特征矩阵X和帧时间y，以及最大频率、特征名称。
    X列顺序固定为: fcpu, fgpu, fddr_GHz, U_cpu, U_gpu, U_ddr, miss, temp
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
            raise KeyError("未找到温度列，请通过 temp_col 参数指定。")
    
    # 帧时间
    if 'fps' not in df.columns:
        raise KeyError("CSV中缺少 'fps' 列")
    df['T_ms'] = 1000.0 / df['fps']
    
    # DDR频率转GHz
    if 'fddr' not in df.columns:
        raise KeyError("缺少 fddr 列")
    df['fddr_GHz'] = df['fddr'] / 1000.0
    
    # 最大频率（在训练集上计算，但为了统一，使用全部数据）
    f_max_cpu = df['fcpu'].max()
    f_max_gpu = df['fgpu'].max()
    
    # 缓存缺失率
    df['miss'] = 1.0 - df['cache_hit']
    
    # 构建特征矩阵，列顺序固定，便于后续配置
    feature_columns = ['fcpu', 'fgpu', 'fddr_GHz', 'cpu_uti', 'gpu_uti', 'ddr_uti', 'miss', temp_col]
    # 确保列存在
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")
    X = df[feature_columns].values
    y = df['T_ms'].values
    
    # 特征名称映射（与列顺序一致）
    feature_names = ['fcpu', 'fgpu', 'fddr_GHz', 'U_cpu', 'U_gpu', 'U_ddr', 'miss', 'temp']
    return X, y, f_max_cpu, f_max_gpu, feature_names

# ========================
# 2. 可配置模型类（与之前相同，无修改）
# ========================
class ConfigurablePerformanceModel:
    """
    可配置的性能模型，支持为每个线性项指定使用的特征子集。
    配置格式:
        config = {
            'alpha_cpu': ['U_cpu', 'miss', 'temp'],   # 可任意子集，顺序无关
            'alpha_gpu': ['U_gpu', 'miss', 'temp'],
            'tau0':      ['temp', 'U_ddr', 'miss'],
            'tau1':      ['temp', 'U_ddr', 'miss'],
            'lambda':    ['U_ddr', 'miss'],
        }
    """
    def __init__(self, config, feature_names, f_max_cpu, f_max_gpu):
        self.config = config
        self.feature_names = feature_names
        self.f_max_cpu = f_max_cpu
        self.f_max_gpu = f_max_gpu
        
        # 建立特征名到列索引的映射
        self.feat2idx = {name: i for i, name in enumerate(feature_names)}
        
        # 构建参数列表及其边界
        self.param_names = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.param_slices = {}  # 记录每个线性项的起止索引
        
        # 定义每个线性项的默认边界（根据物理符号预期）
        # 格式: (参数前缀, 边界下限, 边界上限)
        default_bounds = {
            'alpha_cpu': [('intercept', 0, 50), ('U_cpu', 0, 10), ('miss', 0, 20), ('temp', 0, 5)],
            'alpha_gpu': [('intercept', 0, 50), ('U_gpu', 0, 10), ('miss', 0, 20), ('temp', 0, 5)],
            'tau0':      [('intercept', 0, 50), ('temp', 0, 5), ('U_ddr', 0, 10), ('miss', 0, 20)],
            'tau1':      [('intercept', 0, 100), ('temp', 0, 10), ('U_ddr', -10, 0), ('miss', 0, 20)],
            'lambda':    [('intercept', 0, 10), ('U_ddr', -10, 0), ('miss', -10, 0)]
        }
        
        for key, vars_in_use in config.items():
            # 获取该项的默认边界模板
            bounds_template = default_bounds[key]
            # 根据vars_in_use筛选参数：intercept总是保留，其余特征只在vars_in_use中才添加
            start = len(self.param_names)
            for param_desc in bounds_template:
                feat_name = param_desc[0]
                if feat_name == 'intercept':
                    # 总是包含截距
                    param_name = f"{key}_intercept"
                    lb, ub = param_desc[1], param_desc[2]
                    self.param_names.append(param_name)
                    self.lower_bounds.append(lb)
                    self.upper_bounds.append(ub)
                elif feat_name in vars_in_use:
                    param_name = f"{key}_{feat_name}"
                    lb, ub = param_desc[1], param_desc[2]
                    self.param_names.append(param_name)
                    self.lower_bounds.append(lb)
                    self.upper_bounds.append(ub)
            self.param_slices[key] = slice(start, len(self.param_names))
        
        self.n_params = len(self.param_names)
        self.bounds = Bounds(self.lower_bounds, self.upper_bounds)
    
    def _linear_combination(self, params, X, key):
        """计算给定线性项的值，X是原始特征矩阵（包含所有特征）"""
        sl = self.param_slices[key]
        coeffs = params[sl]
        # 构建设计矩阵：截距 + 对应特征
        # 获取该项使用的特征列表（不含intercept）
        used_feats = [f for f in self.config[key] if f != 'intercept']
        # 构造设计矩阵，第一列为1（截距）
        n_samples = X.shape[0]
        X_design = np.ones((n_samples, 1))
        for f in used_feats:
            col_idx = self.feat2idx[f]
            X_design = np.hstack([X_design, X[:, col_idx:col_idx+1]])
        return X_design @ coeffs
    
    def predict(self, params, X):
        """根据参数和特征矩阵X预测帧时间"""
        # X列顺序必须与feature_names一致
        # 提取频率等
        f_cpu = X[:, self.feat2idx['fcpu']]
        f_gpu = X[:, self.feat2idx['fgpu']]
        f_ddr = X[:, self.feat2idx['fddr_GHz']]
        
        # CPU-GPU部分
        alpha_cpu = self._linear_combination(params, X, 'alpha_cpu')
        alpha_gpu = self._linear_combination(params, X, 'alpha_gpu')
        T_cpugpu = alpha_cpu * (self.f_max_cpu / f_cpu) + alpha_gpu * (self.f_max_gpu / f_gpu)
        
        # DDR部分
        tau0 = self._linear_combination(params, X, 'tau0')
        tau1 = self._linear_combination(params, X, 'tau1')
        lam = self._linear_combination(params, X, 'lambda')
        tau1 = np.maximum(tau1, 0)   # 非负约束
        lam = np.maximum(lam, 0)
        T_ddr = tau0 + tau1 * np.exp(-lam * f_ddr)
        
        return np.maximum(T_cpugpu, T_ddr)
    
    def loss(self, params, X, y):
        y_pred = self.predict(params, X)
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, init_params=None, method='L-BFGS-B', max_iter=5000, verbose=True):
        if init_params is None:
            # 默认初始猜测：截距为合理值，其他系数为0.1（注意正负区分）
            init_params = np.zeros(self.n_params)
            # 设置截距项的初值：alpha约5, tau0约5, tau1约20, lambda约2
            for key, sl in self.param_slices.items():
                intercept_idx = None
                for i, name in enumerate(self.param_names[sl]):
                    if name.endswith('_intercept'):
                        intercept_idx = sl.start + i
                        break
                if intercept_idx is not None:
                    if key in ['alpha_cpu', 'alpha_gpu']:
                        init_params[intercept_idx] = 5.0
                    elif key == 'tau0':
                        init_params[intercept_idx] = 5.0
                    elif key == 'tau1':
                        init_params[intercept_idx] = 20.0
                    elif key == 'lambda':
                        init_params[intercept_idx] = 2.0
            # 其他系数初始化为0.1（正边界）或 -0.1（负边界）
            for i, (lb, ub) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
                if init_params[i] == 0:
                    if ub > 0 and lb >= 0:
                        init_params[i] = 0.1
                    elif ub <= 0 and lb < 0:
                        init_params[i] = -0.1
                    else:
                        init_params[i] = 0.0
        
        result = minimize(
            self.loss, init_params, args=(X, y),
            method=method, bounds=self.bounds, options={'maxiter': max_iter, 'disp': verbose}
        )
        if verbose:
            if result.success:
                print("优化成功")
            else:
                print("优化失败:", result.message)
        return result.x, result.success
    
    def evaluate(self, params, X, y):
        y_pred = self.predict(params, X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return rmse, mae, r2, y_pred

# ========================
# 3. 带训练/验证划分的实验函数
# ========================
def run_experiments_with_validation(csv_path, configs, temp_col=None, 
                                    test_size=1/3, random_state=42, 
                                    verbose=True):
    """
    对每个配置：
        - 划分训练集（2/3）和验证集（1/3）
        - 在训练集上拟合模型
        - 在验证集上预测并计算指标
    返回每个配置的结果字典，包含训练集指标、验证集指标、验证集预测值和真实值。
    """
    # 加载完整数据
    X_all, y_all, f_max_cpu, f_max_gpu, feature_names = load_data(csv_path, temp_col)
    
    # 划分训练集和验证集（随机打乱）
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    if verbose:
        print(f"数据总量: {len(y_all)}")
        print(f"训练集: {len(y_train)} 样本, 验证集: {len(y_val)} 样本")
    
    results = []
    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"配置 {i+1}: {cfg}")
            print('='*60)
        
        # 创建模型对象（注意：最大频率使用全部数据，这样更稳定）
        model = ConfigurablePerformanceModel(cfg, feature_names, f_max_cpu, f_max_gpu)
        
        # 在训练集上拟合
        params, success = model.fit(X_train, y_train, verbose=verbose)
        
        # 训练集评估（用于观察过拟合）
        rmse_train, mae_train, r2_train, y_train_pred = model.evaluate(params, X_train, y_train)
        
        # 验证集评估
        rmse_val, mae_val, r2_val, y_val_pred = model.evaluate(params, X_val, y_val)
        
        if verbose:
            print(f"训练集: RMSE={rmse_train:.4f}, MAE={mae_train:.4f}, R²={r2_train:.4f}")
            print(f"验证集: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R²={r2_val:.4f}")
        
        # 收集参数
        param_dict = {name: params[i] for i, name in enumerate(model.param_names)}
        
        results.append({
            'config': cfg,
            'success': success,
            'params': param_dict,
            'train_rmse': rmse_train,
            'train_mae': mae_train,
            'train_r2': r2_train,
            'val_rmse': rmse_val,
            'val_mae': mae_val,
            'val_r2': r2_val,
            'y_val_true': y_val,
            'y_val_pred': y_val_pred,
            'y_train_true': y_train,
            'y_train_pred': y_train_pred
        })
    
    return results

# ========================
# 4. 示例：定义多个对比配置并运行
# ========================
if __name__ == "__main__":
    # 定义配置（与之前相同）
    config_full = {
        'alpha_cpu': ['U_cpu', 'miss', 'temp'],
        'alpha_gpu': ['U_gpu', 'miss', 'temp'],
        'tau0':      ['temp', 'U_ddr', 'miss'],
        'tau1':      ['temp', 'U_ddr', 'miss'],
        'lambda':    ['U_ddr', 'miss'],
    }
    
    config_no_temp = {
        'alpha_cpu': ['U_cpu', 'miss'],
        'alpha_gpu': ['U_gpu', 'miss'],
        'tau0':      ['U_ddr', 'miss'],
        'tau1':      ['U_ddr', 'miss'],
        'lambda':    ['U_ddr', 'miss'],
    }
    
    config_only_util = {
        'alpha_cpu': ['U_cpu'],
        'alpha_gpu': ['U_gpu'],
        'tau0':      ['U_ddr'],
        'tau1':      ['U_ddr'],
        'lambda':    ['U_ddr'],
    }
    
    config_only_miss = {
        'alpha_cpu': ['miss'],
        'alpha_gpu': ['miss'],
        'tau0':      ['miss'],
        'tau1':      ['miss'],
        'lambda':    ['miss'],
    }
    
    config_constant = {
        'alpha_cpu': [],
        'alpha_gpu': [],
        'tau0':      [],
        'tau1':      [],
        'lambda':    [],
    }
    
    configs_to_run = [config_full, config_no_temp, config_only_util, config_only_miss, config_constant]
    
    # 运行实验（请替换为实际CSV路径，并指定温度列名）
    results = run_experiments_with_validation('your_data.csv', configs_to_run, temp_col='temp_soc')
    
    # 打印汇总比较表
    print("\n\n========== 汇总比较 ==========")
    print("配置\t\t训练RMSE\t验证RMSE\t训练R²\t验证R²")
    for i, res in enumerate(results):
        cfg_name = f"cfg{i+1}"
        print(f"{cfg_name}\t{res['train_rmse']:.4f}\t{res['val_rmse']:.4f}\t{res['train_r2']:.4f}\t{res['val_r2']:.4f}")
