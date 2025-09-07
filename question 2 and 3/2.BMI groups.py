import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
#导入数据
def dataset(x):
    df = pd.read_excel("cleaneddata.xlsx",engine='openpyxl',header=None,usecols=[x],skiprows=1,nrows=612)
    return df[x].tolist()
data = pd.DataFrame({'BMI':dataset(3),'孕周':dataset(2),'Y染色体浓度':dataset(4)})
#将BMI按照四分位数分组
data['BMI_quartile'] = pd.qcut(data['BMI'],q=4,labels = ['Q1','Q2','Q3','Q4'])
# 为每个BMI分组寻找最佳孕周阈值
thresholds = {}
for group in ['Q1', 'Q2', 'Q3', 'Q4']:
    group_data = data[data['BMI_quartile'] == group]
    
    # 创建二分类标签：Y染色体浓度是否大于0.04
    y_true = (group_data['Y染色体浓度'] > 0.04).astype(int)
    
    # 使用孕周作为预测变量
    y_score = group_data['孕周']
    
    # 计算ROC曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    
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
        'youden_j': youden_j[best_idx]
    }
    
    print(f"BMI分组 {group}: 最佳孕周阈值 = {best_threshold:.2f}, 真阳性率 = {tpr[best_idx]:.2f}, 假阳性率 = {fpr[best_idx]:.2f}")

# 输出所有阈值
print("\n各BMI分组的最佳孕周阈值:")
for group, result in thresholds.items():
    print(f"{group}: {result['threshold']:.2f}")