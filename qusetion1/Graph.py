import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据准备
try:
    # 读取Excel文件
    df = pd.read_excel('NIPT清洗后.xlsx', 
                      sheet_name='Sheet1', 
                      engine='openpyxl',
                      header=None,  # 不使用第一行作为列名
                      usecols=[2,3,4],  # 读取2,3,4列
                      skiprows=1,  # 跳过第一行
                      nrows=612)  
    
    # 重命名列
    df.columns = ['Y', 'Z', 'X']  # 按照C、D、E的顺序对应Y、Z、X
    
    # 检查数据是否读取成功
    print("数据预览：")
    print(df.head())
    print("\n数据形状：", df.shape)
    
    # 检查并处理空值
    if df.isnull().any().any():
        print("\n警告：数据中存在空值，将进行填充处理")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 确保数据是数值型
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 描述性统计
    print("\n描述性统计：")
    print(df.describe())
    
except FileNotFoundError:
    print("错误：找不到文件 'NIPT清洗后数据.xlsx'")
    exit()
except Exception as e:
    print(f"发生错误：{str(e)}")
    exit()

# 2. 数据探索与可视化
# 相关性热力图
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=.5, cbar_kws={"label": "相关系数"})
plt.title('变量相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300)
plt.show()

# 散点图矩阵
sns.pairplot(df)
plt.suptitle('变量间关系散点图', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('散点图矩阵.png', dpi=300)
plt.show()

# 3. 建立多元线性回归模型
# 准备自变量和因变量
X = df[['Y', 'Z']]  # 孕周和BMI作为自变量
y = df['X']         # Y染色体浓度作为因变量

# 添加常数项（截距）
X = sm.add_constant(X)

# 建立并拟合模型
model = sm.OLS(y, X).fit()

# 输出模型结果
print("\n多元线性回归模型结果：")
print(model.summary())

# 4. 模型诊断
# 检查多重共线性
vif_data = pd.DataFrame()
vif_data["变量"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n方差膨胀因子(VIF)：")
print(vif_data)

# 5. 模型预测与评估
# 预测值
y_pred = model.predict(X)

# 计算评估指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\n模型评估指标：")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 6. 可视化结果
# 实际值vs预测值
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('Y染色体浓度实际值vs预测值')
plt.grid(True)
plt.tight_layout()
plt.savefig('实际值vs预测值.png', dpi=300)
plt.show()

# 残差图
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True)
plt.tight_layout()
plt.savefig('残差图.png', dpi=300)
plt.show()

# 残差的正态性检验
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='45', fit=True)
plt.title('残差Q-Q图')
plt.tight_layout()
plt.savefig('残差QQ图.png', dpi=300)
plt.show()

# 7. 结果解释
print("\n模型解释：")
print(f"回归方程: X = {model.params['const']:.4f} + {model.params['Y']:.4f}*Y + {model.params['Z']:.4f}*Z")
print(f"其中：")
print(f"- X: Y染色体浓度")
print(f"- Y: 孕周")
print(f"- Z: BMI")

# 解释系数
print("\n系数解释：")
print(f"- 截距(const): {model.params['const']:.4f}")
print(f"  当孕周和BMI都为0时，Y染色体浓度的估计值。")
print(f"- 孕周系数: {model.params['Y']:.4f}")
print(f"  当BMI保持不变时，孕周每增加一个单位，Y染色体浓度平均变化{model.params['Y']:.4f}个单位。")
print(f"- BMI系数: {model.params['Z']:.4f}")
print(f"  当孕周保持不变时，BMI每增加一个单位，Y染色体浓度平均变化{model.params['Z']:.4f}个单位。")

# 解释模型拟合优度
print(f"\n模型拟合优度：")
print(f"决定系数(R²) = {r2:.4f}")
print(f"这意味着模型中的自变量（孕周和BMI）可以解释Y染色体浓度变异的{r2*100:.2f}%。")

# 检查模型显著性
print("\n模型显著性检验：")
print(f"F统计量 = {model.fvalue:.4f}")
print(f"P值 = {model.f_pvalue:.4f}")
if model.f_pvalue < 0.05:
    print("P值小于0.05，模型整体上是显著的。")
else:
    print("P值大于0.05，模型整体上不显著。")

# 检查系数显著性
print("\n系数显著性检验：")
for i, col in enumerate(X.columns):
    p_value = model.pvalues[col]
    print(f"{col}的P值 = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  {col}的系数在0.05水平上显著。")
    else:
        print(f"  {col}的系数在0.05水平上不显著。")


