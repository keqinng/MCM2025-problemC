import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from numpy import zeros, amax, amin, mean
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 前者为中文字体，后者为备选英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方框的问题
# 无量纲化
def dimensionlessProcessing(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        MEAN = d.mean()
        newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()
    return newDataFrame

def GRA_ONE(gray, m=0):
    # 读取为df格式
    gray = dimensionlessProcessing(gray)
    # 标准化
    std = gray.iloc[:, m]  # 为标准要素
    gray.drop(str(m),axis=1,inplace=True)
    ce = gray.iloc[:, 0:]  # 为比较要素
    shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c, d = amax(a), amin(a)

    # 计算值
    result = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值,并返回
    result_list = [mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m,1)
    return pd.DataFrame(result_list)


def GRA(DataFrame):
    df = DataFrame.copy()
    list_columns = [
        str(s) for s in range(len(df.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    df.columns=list_columns
    for i in range(len(df.columns)):
        df_local.iloc[:, i] = GRA_ONE(df, m=i)[0]
    return df_local


def GRA_multi_factors(data, reference_col='V'):
    """
    多因素灰色关联分析
    data: 包含所有因素的DataFrame
    reference_col: 参考序列（因变量）的列名
    """
    results = {}
    reference = data[reference_col]
    
    # 计算每个因素与参考序列的灰色关联度
    for column in data.columns:
        if column != reference_col:
            factor_data = data[[reference_col, column]]
            factor_data = factor_data.dropna()
            gra_result = GRA(factor_data)
            results[column] = gra_result
    
    return results

try:
    # 读取Excel文件
    df = pd.read_excel('cleaneddata.xlsx', 
                      sheet_name='Sheet1', 
                      engine='openpyxl',
                      header=None,
                      usecols=[4,5,7,11,13,16,21],  # 扩展读取更多列
                      skiprows=1,
                      nrows=66)
    
    # 重命名列
    df.columns = [ 'C','D', 'H', 'L', 'N','Q','V']  # 根据实际数据调整列名
    
    print("数据预览：")
    print(df.head())
    
    
    # 检查并处理空值
    if df.isnull().any().any():
        print("\n警告：数据中存在空值，将进行填充处理")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 确保数据是数值型
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 执行多因素灰色关联分析
    results = GRA_multi_factors(df, reference_col='V')
    
    # 打印结果
    print("\n各因素与染色体浓度的灰色关联度：")
    for factor, gra_value in results.items():
        factor_name = {
            'C': 'BMI',
            'D': '原始读段数',
            'H': '重复读段比例',  
            'L': '18号染色体的Z值',
            'N': 'X染色体的Z值' ,
            'Q': '18号染色体的GC含量'
        }.get(factor, factor)
        print(f'{factor_name}和18号染色体是否异常的关联: {str(gra_value)}')
    

    
except FileNotFoundError:
    print("错误：找不到文件 'cleanedata.xlsx'")
except Exception as e:
    print(f"发生错误：{str(e)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_gra_results_bar(results_dict):
    """
    将灰色关联分析结果可视化为柱状图
    
    参数:
    results_dict: 包含各因素关联度的字典
    """
    # 创建列表存储因素名称和对应的关联度
    factors = []
    correlation_values = []
    
    # 从结果字典中提取数据
    for factor, matrix in results_dict.items():
        factors.append(factor)
        # 取矩阵的非对角线元素作为关联度值
        correlation_values.append(matrix.iloc[0, 1])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Factor': factors,
        'Correlation': correlation_values
    })
    
    # 按关联度值排序
    df = df.sort_values('Correlation', ascending=False)
    
    # 设置图形大小
    plt.figure(figsize=(12, 6))
    
    # 创建柱状图
    bars = plt.bar(df['Factor'], df['Correlation'], color='skyblue', alpha=0.7)
    
    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 设置标题和标签
    plt.title('各因素与18号染色体异常的灰色关联度', pad=20, fontsize=14)
    plt.xlabel('影响因素', fontsize=12)
    plt.ylabel('灰色关联度', fontsize=12)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 设置y轴范围
    plt.ylim(0, max(df['Correlation']) * 1.1)
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
   
    plt.close()

# 示例数据（使用你提供的结果）
results = {
    'BMI': pd.DataFrame([[1.000000, 0.583774], [0.583774, 1.0]]),
    '原始读段数': pd.DataFrame([[1.00000, 0.60101], [0.60101, 1.0]]),
    '重复读段比例': pd.DataFrame([[1.000000, 0.721026], [0.721026, 1.0]]),
    '18号染色体Z值': pd.DataFrame([[1.000000, 0.593052], [0.593052, 1.0]]),
    'X染色体Z值': pd.DataFrame([[1.000000, 0.588585], [0.588585, 1.0]]),
    '18号染色体GC含量': pd.DataFrame([[1.000000, 0.671983], [0.671983, 1.0]])
}

# 可视化结果
visualize_gra_results_bar(results)