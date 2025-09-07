import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#导入数据
def dataset(x):
    df = pd.read_excel("cleaneddata.xlsx",engine='openpyxl',header=None,usecols=[x],skiprows=1,nrows=174)
    return df[x].tolist()
data = pd.DataFrame({'BMI':dataset(3),'孕周':dataset(2),'Y染色体浓度':dataset(4),'年龄':dataset(1),'GC含量':dataset(5)})

#将BMI按照四分位数分组
data['BMI_quartile'] = pd.qcut(data['BMI'],q=4,labels = ['Q1','Q2','Q3','Q4'])

# 为每个BMI分组寻找最佳孕周阈值
thresholds = {}
roc_curves = {}  # 存储ROC曲线数据

# 创建图形
plt.figure(figsize=(15, 10))

# 1. 绘制所有BMI分组的ROC曲线
plt.subplot(2, 2, 1)
for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    group_data = data[data['BMI_quartile'] == group]
    
    # 创建二分类标签：Y染色体浓度是否大于0.04
    y_true = (group_data['Y染色体浓度'] > 0.04).astype(int)
    
    # 使用孕周作为预测变量
    y_score = group_data['年龄']
    
    # 计算ROC曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 存储ROC曲线数据
    roc_curves[group] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=f'{group} (AUC = {roc_auc:.3f})')
    
    # 计算约登指数（Youden's J statistic）
    youden_j = tpr - fpr
    
    # 找到最大约登指数对应的阈值
    best_idx = np.argmax(youden_j)
    best_threshold = roc_thresholds[best_idx]
    
    # 存储结果
    thresholds[group] = {
        'threshold': best_threshold,
        'tpr': tpr[best_idx],
        'fpr': fpr[best_idx],
        'youden_j': youden_j[best_idx],
        'auc': roc_auc
    }
    
    # 在ROC曲线上标记最佳阈值点
    plt.plot(fpr[best_idx], tpr[best_idx], 'o')
    
    print(f"BMI分组 {group}: 潜在风险最小的孕龄 = {best_threshold:.2f}, 真阳性率 = {tpr[best_idx]:.2f}, 假阳性率 = {fpr[best_idx]:.2f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('各BMI分组的ROC曲线')
plt.legend(loc="lower right")

# 2. 绘制各BMI分组的最佳阈值比较
plt.subplot(2, 2, 2)
groups = list(thresholds.keys())
best_thresholds = [thresholds[group]['threshold'] for group in groups]
bars = plt.bar(groups, best_thresholds, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('BMI分组')
plt.ylabel('潜在风险最小的孕龄')
plt.title('各BMI分组的最佳孕龄比较')

# 在柱状图上添加数值标签
for bar, threshold in zip(bars, best_thresholds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{threshold:.2f}', ha='center', va='bottom')

# 3. 绘制约登指数比较
plt.subplot(2, 2, 3)
youden_indices = [thresholds[group]['youden_j'] for group in groups]
bars = plt.bar(groups, youden_indices, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('BMI分组')
plt.ylabel('约登指数')
plt.title('各BMI分组的约登指数比较')

# 在柱状图上添加数值标签
for bar, youden in zip(bars, youden_indices):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{youden:.3f}', ha='center', va='bottom')

# 4. 绘制AUC值比较
plt.subplot(2, 2, 4)
auc_values = [thresholds[group]['auc'] for group in groups]
bars = plt.bar(groups, auc_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel('BMI分组')
plt.ylabel('AUC值')
plt.title('各BMI分组的AUC值比较')

# 在柱状图上添加数值标签
for bar, auc_val in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{auc_val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('BMI_groups_analysis.png', dpi=300)
plt.show()

# 5. 绘制箱线图，展示各BMI分组的Y染色体浓度分布
plt.figure(figsize=(10, 6))
sns.boxplot(x='BMI_quartile', y='Y染色体浓度', data=data)
plt.axhline(y=0.04, color='r', linestyle='--', label='Y染色体浓度阈值 (0.04)')
plt.xlabel('BMI分组')
plt.ylabel('Y染色体浓度')
plt.title('各BMI分组的Y染色体浓度分布')
plt.legend()
plt.savefig('Y_chromosome_distribution.png', dpi=300)
plt.show()

# 6. 绘制散点图，展示孕周与Y染色体浓度的关系，并标记最佳阈值线
plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, group in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    group_data = data[data['BMI_quartile'] == group]
    plt.scatter(group_data['年龄'], group_data['Y染色体浓度'], 
                alpha=0.6, label=f'{group} (阈值: {thresholds[group]["threshold"]:.1f})',
                color=colors[i])
    
    # 为每个BMI分组添加最佳阈值垂直线
    plt.axvline(x=thresholds[group]['threshold'], color=colors[i], 
                linestyle='--', alpha=0.7)

plt.axhline(y=0.04, color='black', linestyle='-', alpha=0.3, label='Y染色体浓度阈值 (0.04)')
plt.xlabel('年龄')
plt.ylabel('Y染色体浓度')
plt.title('年龄与Y染色体浓度的关系及各BMI分组的潜在风险最小的孕龄')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gestational_age_vs_Y_chromosome.png', dpi=300)
plt.show()

# 输出所有阈值
print("\n各BMI分组的最佳孕龄:")
for group, result in thresholds.items():
    print(f"{group}: {result['threshold']:.2f} (AUC = {result['auc']:.3f}, 约登指数 = {result['youden_j']:.3f})")
