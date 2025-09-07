import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

try:
    # 读取Excel文件，指定行范围和列
    df = pd.read_excel('NIPT清洗后数据.xlsx', 
                      sheet_name='Sheet1', 
                      engine='openpyxl',
                      header=None,  # 不使用第一行作为列名
                      usecols=[2,3,4],  # 读取2,3,4列
                      skiprows=1,  # 跳过第一行
                      nrows=400)  
    
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
    
    # 创建两个数据集
    data1 = df[['X', 'Y']]  # X和Y的数据
    data2 = df[['X', 'Z']]  # X和Z的数据
    
    # 确保没有空值
    data1 = data1.dropna()
    data2 = data2.dropna()
    
    print("\n数据1（X和Y）的统计信息：")
    print(data1.describe())
    
    print("\n数据2（X和Z）的统计信息：")
    print(data2.describe())

    corr_matrix1 = data1.corr()
    corr_matrix2 = data2.corr()

    print("\n数据1（X和Y）的相关矩阵：")
    print(corr_matrix1)

    print("\n数据2（X和Z）的相关矩阵：")
    print(corr_matrix2)

    corr_matrix3 = data1.corr(method='spearman')
    corr_matrix4 = data2.corr(method='spearman')

    print("\n数据1（X和Y）的Spearman相关矩阵：")
    print(corr_matrix3)

    print("\n数据2（X和Z）的Spearman相关矩阵：")
    print(corr_matrix4)




except FileNotFoundError:
    print("错误：找不到文件 'NIPT清洗后数据.xlsx'")
except Exception as e:
    print(f"发生错误：{str(e)}")