import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
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
    X = group_data[['孕周']]
    y = (group_data['Y染色体浓度'] > 0.04).astype(int)
    
    # 使用逻辑回归
    model = LogisticRegression()
    model.fit(X, y)
    
    # 预测概率
    y_prob = model.predict_proba(X)[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)
    
    # 计算约登指数
    youden_j = tpr - fpr
    
    # 找到最大约登指数对应的阈值
    best_idx = np.argmax(youden_j)
    
    # 将概率阈值转换为孕周阈值
    # 逻辑回归的决策函数是: logit(p) = b0 + b1*x
    # 我们需要找到x使得p等于最佳概率阈值
    b0 = model.intercept_[0]
    b1 = model.coef_[0][0]
    prob_threshold = roc_thresholds[best_idx]
    
    # 避免log(0)的情况
    if prob_threshold <= 0:
        prob_threshold = 1e-10
    elif prob_threshold >= 1:
        prob_threshold = 1 - 1e-10
    
    # 计算对应的孕周阈值
    logit_threshold = np.log(prob_threshold / (1 - prob_threshold))
    week_threshold = (logit_threshold - b0) / b1
    
    # 存储结果
    thresholds[group] = {
        'threshold': week_threshold,
        'tpr': tpr[best_idx],
        'fpr': fpr[best_idx],
        'youden_j': youden_j[best_idx]
    }
    
    print(f"BMI分组 {group}: 最佳孕周阈值 = {week_threshold:.2f}, 真阳性率 = {tpr[best_idx]:.2f}, 假阳性率 = {fpr[best_idx]:.2f}")

# 输出所有阈值
print("\n各BMI分组的最佳孕周阈值:")
for group, result in thresholds.items():
    print(f"{group}: {result['threshold']:.2f}")
