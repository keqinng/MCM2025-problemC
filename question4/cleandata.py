import pandas as pd
import re
import numpy as np

def convert_gestational_age(file_path,column_name):
  
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    def parse_gestational_age(age_str):
        """
        解析孕周字符串，返回带小数的周数
        例如：'12w+3' -> 12.43周 (12 + 3/7)
        """
        try:
            # 使用正则表达式提取数字
            match = re.match(r'(\d+)w\+(\d+)', str(age_str))
            if match:
                weeks = int(match.group(1))
                days = int(match.group(2))
                # 计算小数周数（天数/7）
                decimal_weeks = weeks + (days / 7)
                # 保留两位小数
                return round(decimal_weeks, 2)
            else:
                # 如果格式不匹配，尝试直接转换为数字
                return float(age_str)
        except:
            # 如果转换失败，返回NaN
            return np.nan
    
    # 转换孕周数据
    df[column_name] = df[column_name].apply(parse_gestational_age)
    
    # 保存处理后的数据
    df.to_excel(file_path, index=False)
    print(f"数据已处理并更新到原文件: {file_path}")
   

# 使用示例
# convert_gestational_age('原始数据.xlsx', '处理后数据.xlsx', '孕周')
convert_gestational_age('attached1.xlsx','检测孕周')
