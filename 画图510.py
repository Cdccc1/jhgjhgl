import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import io

# 创建示例CSV数据
sample_data = """fcpu,fgpu,fddr,fps_pred,fps
2400,1800,2133,45.2,47.0
2600,1900,2400,52.1,50.8
2800,2000,2666,58.3,60.1
2400,1800,2133,42.7,44.5
2500,1850,2200,48.9,47.2
2700,1950,2500,55.6,53.8
2900,2100,2800,62.4,61.0
2300,1750,2000,40.3,41.8
2550,1875,2300,50.7,49.2
2750,2025,2700,59.1,60.5
2450,1825,2150,46.8,45.1
2650,1925,2550,53.4,54.9
2850,2050,2750,60.2,58.7
2350,1775,2050,41.6,43.2
2575,1887,2350,51.3,52.8
2775,2037,2650,57.8,56.4
2950,2125,2850,63.5,62.1
2250,1700,1900,38.9,40.5
2475,1837,2250,47.5,46.0
2675,1962,2600,54.9,55.3"""

# 将示例数据写入临时CSV文件
with open('performance_data.csv', 'w') as f:
    f.write(sample_data)

# 读取数据
df = pd.read_csv('performance_data.csv')

# 计算相对误差百分比
relative_errors = np.abs(df['fps_pred'] - df['fps']) / df['fps'] * 100

# 判断哪些样本的MRE <= 10%
mre_threshold = 10
accurate_mask = relative_errors <= mre_threshold
inaccurate_mask = relative_errors > mre_threshold

# 计算准确率
accuracy_rate = np.sum(accurate_mask) / len(df) * 100

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('帧率预测模型可视化评估', fontsize=14)

# 设置全局样式
plt.rcParams.update({
    'font.size': 12,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# 上图：散点图
ax1.scatter(df.loc[accurate_mask, 'fps'], df.loc[accurate_mask, 'fps_pred'], 
           c='green', s=50, alpha=0.7, label=f'MRE ≤ {mre_threshold}%', edgecolors='black')
ax1.scatter(df.loc[inaccurate_mask, 'fps'], df.loc[inaccurate_mask, 'fps_pred'], 
           c='none', s=50, alpha=0.7, label=f'MRE > {mre_threshold}%', edgecolors='red')

# 添加y=x参考线
min_val = min(min(df['fps']), min(df['fps_pred']))
max_val = max(max(df['fps']), max(df['fps_pred']))
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='y=x')

ax1.set_xlabel('真实帧率 (FPS)')
ax1.set_ylabel('预测帧率 (FPS)')
ax1.set_title('预测值 vs 真实值')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# 下图：柱状图
x_pos = np.arange(len(df))
bars = ax2.bar(x_pos, relative_errors, color=['green' if e else 'red' for e in accurate_mask])
ax2.set_xlabel('样本序号')
ax2.set_ylabel('相对误差 (%)')
ax2.set_title(f'相对误差分布 (准确率: {accuracy_rate:.2f}%)')

# 在图顶部居中显示准确率
ax2.text(0.5, 1.05, f'准确率: {accuracy_rate:.2f}%', 
         transform=ax2.transAxes, ha='center', va='top', fontsize=14, weight='bold')

# 添加网格
ax2.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

print(f"数据集包含 {len(df)} 个样本")
print(f"MRE <= {mre_threshold}% 的样本数量: {np.sum(accurate_mask)}")
print(f"MRE > {mre_threshold}% 的样本数量: {np.sum(inaccurate_mask)}")
print(f"准确率: {accuracy_rate:.2f}%")

# 如果需要从实际CSV文件读取，取消下面的注释
"""
# 读取实际CSV文件
df = pd.read_csv('your_file.csv')

# 其余代码保持不变...
"""
