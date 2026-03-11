import pandas as pd
import os
from pathlib import Path


# 读取已有因子库
file_path = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\factor_set.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")
factor_set = pd.read_csv(file_path)

# print("提取成功！前5行数据：")
# print(factor_set.head())


target_factor = ['01']
base_path = rf'/WorldQuant_Alpha101'

# 1. 读取并添加因子标识
all_factors = []
for factor in target_factor:
    print(f'正在处理因子{factor}')
    fp = Path(base_path) / f'alpha_{factor}.csv'
    df = pd.read_csv(fp)
    df = df.iloc[:, [0, 1, -1]]  # 这里可以用位置索引
    df.columns = ['ts_code', 'trade_date', 'factor_value']
    df['factor_name'] = f'alpha_{factor}'
    print(df)
    all_factors.append(df)

# 2. 合并所有因子数据
combined_df = pd.concat(all_factors, ignore_index=True)

# 3. pivot转宽表，自动保证trade_date横向对齐
factor_wide = combined_df.pivot_table(
    index=['ts_code', 'trade_date'],
    columns='factor_name',
    values='factor_value',
    aggfunc='first'
).reset_index()

print(factor_wide)

# 4. 与history_df合并
history_df = pd.merge(factor_set, factor_wide, on=['ts_code', 'trade_date'], how='left')
history_df.to_csv('factor_set.csv', index=False)
print("提取成功！前5行数据：")
print(history_df)

