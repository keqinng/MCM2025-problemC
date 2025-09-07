import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 前者为中文字体，后者为备选英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方框的问题

# 假设我们已经运行了原始代码并得到了结果
# 这里我们模拟GRA函数的结果
def simulate_GRA_results():
    # 模拟data1的GRA结果 (X和Y的关联)
    gra_xy = pd.DataFrame({
        '0': [1.0, 0.75],  # X与X的关联度为1，X与Y的关联度为0.75
        '1': [0.75, 1.0]   # Y与X的关联度为0.75，Y与Y的关联度为1
    }, index=['X', 'Y'])
    
    # 模拟data2的GRA结果 (X和Z的关联)
    gra_xz = pd.DataFrame({
        '0': [1.0, 0.60],  # X与X的关联度为1，X与Z的关联度为0.60
        '1': [0.60, 1.0]   # Z与X的关联度为0.60，Z与Z的关联度为1
    }, index=['X', 'Z'])
    
    return gra_xy, gra_xz

# 获取模拟结果
gra_xy, gra_xz = simulate_GRA_results()

# 创建一个2x1的子图布局
plt.figure(figsize=(12, 5))

# 子图1：X和Y的关联热力图
plt.subplot(1, 2, 1)
sns.heatmap(gra_xy, annot=True, cmap='YlGnBu', vmin=0, vmax=1, 
            linewidths=.5, cbar_kws={"label": "关联度"})
plt.title('孕周(Y)和染色体浓度(X)的灰色关联分析')
plt.xlabel('变量')
plt.ylabel('变量')

# 子图2：X和Z的关联热力图
plt.subplot(1, 2, 2)
sns.heatmap(gra_xz, annot=True, cmap='YlGnBu', vmin=0, vmax=1, 
            linewidths=.5, cbar_kws={"label": "关联度"})
plt.title('BMI(Z)和染色体浓度(X)的灰色关联分析')
plt.xlabel('变量')
plt.ylabel('变量')

plt.tight_layout()
plt.savefig('灰色关联分析热力图.png', dpi=300)
plt.show()
