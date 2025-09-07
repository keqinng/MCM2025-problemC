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

# 对其他变量也进行分组
data['GC含量_quartile'] = pd.qcut(data['GC含量'], q=4, labels=['低', '较低', '较高', '高'])
data['BMI_quartile'] = pd.qcut(data['BMI'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 创建多变量交叉分组
data['多因素分组'] = data['BMI_quartile'].astype(str) + "_" + data['GC含量_quartile'].astype(str)
# 创建综合评分
from sklearn.preprocessing import StandardScaler

# 标准化各变量
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data[['BMI', 'GC含量']]), 
                           columns=['BMI_scaled', 'GC含量_scaled'])

# 计算综合评分（可根据实际情况调整权重）
data['综合评分'] = 0.7 * data_scaled['BMI_scaled'] + 0.3 * data_scaled['GC含量_scaled'] 

# 根据综合评分分组
data['综合评分_quartile'] = pd.qcut(data['综合评分'], q=4, labels=['低风险', '中低风险', '中高风险', '高风险'])

# 为每个多因素分组寻找最佳孕周阈值
thresholds = {}
roc_curves = {}

# 创建图形
plt.figure(figsize=(15, 10))

# 绘制所有分组的ROC曲线
plt.subplot(2, 2, 1)
# 获取所有分组
groups = data['多因素分组'].unique()  

for group in groups:
    group_data = data[data['多因素分组'] == group]  
    
    # 创建二分类标签：Y染色体浓度是否大于0.04
    y_true = (group_data['Y染色体浓度'] > 0.04).astype(int)
    
    # 使用孕周作为预测变量
    y_score = group_data['孕周']
    
    # 计算ROC曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 存储ROC曲线数据
    roc_curves[group] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=f'{group} (AUC = {roc_auc:.3f})')
    
    # 计算约登指数
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
    
    print(f"分组 {group}: 最佳孕周阈值 = {best_threshold:.2f}, 真阳性率 = {tpr[best_idx]:.2f}, 假阳性率 = {fpr[best_idx]:.2f}")


plt.figure(figsize=(12, 6))
sns.boxplot(x='多因素分组', y='Y染色体浓度', data=data)  # 或 x='综合评分_quartile'
plt.axhline(y=0.04, color='r', linestyle='--', label='Y染色体浓度阈值 (0.04)')
plt.xlabel('多因素分组')  # 或 '综合评分分组'
plt.ylabel('Y染色体浓度')
plt.title('各分组的Y染色体浓度分布')
plt.xticks(rotation=45)  # 如果分组名称较长，可以旋转标签
plt.legend()
plt.tight_layout()  # 确保所有标签都可见
plt.savefig('Y_chromosome_distribution_by_groups.png', dpi=300)
plt.show()

#  绘制散点图，展示多变量关系
plt.figure(figsize=(14, 10))
# 使用不同颜色和形状表示不同分组
groups = data['多因素分组'].unique()  
colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

for i, group in enumerate(groups):
    group_data = data[data['多因素分组'] == group]  # 或 data[data['综合评分_quartile'] == group]
    plt.scatter(group_data['孕周'], group_data['Y染色体浓度'], 
                alpha=0.6, label=f'{group} (阈值: {thresholds.get(group, {"threshold": float("nan")})["threshold"]:.1f})',
                color=colors[i])
    
    # 为每个分组添加最佳阈值垂直线
    if group in thresholds:
        plt.axvline(x=thresholds[group]['threshold'], color=colors[i], 
                    linestyle='--', alpha=0.7)

plt.axhline(y=0.04, color='black', linestyle='-', alpha=0.3, label='Y染色体浓度阈值 (0.04)')
plt.xlabel('孕周')
plt.ylabel('Y染色体浓度')
plt.title('孕周与Y染色体浓度的关系及各分组的最佳孕周阈值')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图表外
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gestational_age_vs_Y_chromosome_by_groups.png', dpi=300, bbox_inches='tight')
plt.show()

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 创建交互式散点图
fig = px.scatter(data, x='孕周', y='Y染色体浓度', color='多因素分组', 
                 hover_data=['BMI', 'GC含量'],
                 title='孕周与Y染色体浓度的关系（按多因素分组）')

# 添加Y染色体浓度阈值线
fig.add_hline(y=0.04, line_dash="dash", line_color="red", 
              annotation_text="Y染色体浓度阈值 (0.04)")

# 为每个分组添加最佳孕周阈值线
for group in groups:
    if group in thresholds:
        fig.add_vline(x=thresholds[group]['threshold'], line_dash="dash", 
                     opacity=0.5, annotation_text=f"{group}阈值")

fig.update_layout(
    legend_title_text="分组",
    xaxis_title="孕周",
    yaxis_title="Y染色体浓度"
)

fig.show()
fig.write_html("interactive_gestational_age_vs_Y_chromosome.html")