import pandas as pd
import glob
import os
import numpy as np

'''
1. 批量读取：自动遍历指定文件夹下所有的 CSV 文件，仅提取 ts_code、ann_date、end_date、roe 四列。
2. 格式清洗：日期转纯数字字符串，代码转字符串保留前导零。
3. 数据合并：垂直拼接并去重。
4. 条件过滤：保留 ann_date <= 20251231，剔除 ts_code 以 "A" 开头的数据。
5. 缺失值处理（新增）：
   - 按股票分组，按时间排序。
   - 若 ROE 为 NaN，且前一季度有值，则用前一季度值填充。
   - 若连续多个季度为 NaN，仅填充第一个，后续保持 NaN（防止错误延续）。
6. 结果输出：保存CSV并打印验证信息。
'''

# --- 自定义填充函数 ---
def fill_single_nan(group):
    """
    对单个股票的 ROE 序列进行处理：
    只填充连续 NaN 中的第一个，后续的 NaN 保持不变。
    """
    roe_values = group['roe'].values
    
    cleaned_roe = []
    for i, val in enumerate(roe_values):
        if pd.isna(val):
            # 当前是 NaN
            if i > 0 and not pd.isna(roe_values[i-1]):
                # 前一个原始数据不是 NaN -> 填充
                cleaned_roe.append(roe_values[i-1])
            else:
                # 前一个原始数据也是 NaN -> 不填充，保持 NaN
                cleaned_roe.append(np.nan)
        else:
            # 当前不是 NaN，直接保留
            cleaned_roe.append(val)
            
    group['roe'] = cleaned_roe
    return group

# ---------------- 主程序 ----------------

folder_path = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务指标数据'
file_pattern = os.path.join(folder_path, '*.csv')
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print("未找到文件。")
else:
    target_columns = ['ts_code', 'ann_date', 'end_date', 'roe']
    dtype_map = {'ts_code': str} 
    
    df_list = []
    
    for file in file_list:
        try:
            df_temp = pd.read_csv(file, usecols=target_columns, dtype=dtype_map)
            
            for col in ['ann_date', 'end_date']:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].apply(lambda x: str(int(float(x))) if pd.notna(x) and x != '' else x)

            df_list.append(df_temp)
            print(f"已读取: {os.path.basename(file)}")

        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")

    if df_list:
        # 1. 合并
        final_df = pd.concat(df_list, ignore_index=True)
        
        # 2. 去重
        final_df = final_df.drop_duplicates()
        
        # 3. 选列
        final_df = final_df[['ts_code', 'ann_date', 'end_date', 'roe']]

        # 4. 基础过滤
        final_df = final_df[final_df['ann_date'] <= '20251231']
        final_df = final_df[~final_df['ts_code'].str.startswith('A')]

        print("\n正在执行 ROE 缺失值填充逻辑...")
     
        final_df = final_df.sort_values(by=['ts_code', 'ann_date'], ascending=[True, True])
        
  
        final_df = final_df.groupby('ts_code', group_keys=False).apply(fill_single_nan)

        final_df = final_df.drop_duplicates()

        final_df = final_df.reset_index(drop=True)

        print("-" * 30)
        print(f"处理完成！剩余行数: {len(final_df)}")
        
        # 统计填充情况
        total_nan = final_df['roe'].isna().sum()
        print(f"最终 ROE 为空的行数: {total_nan}")

        save_path = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务指标数据\roe_data_clean.csv'
        final_df.to_csv(save_path, index=False)
        print(f"已保存为: {save_path}")
        
        print("\n前 5 行预览:")
        print(final_df.head())
        
        if len(final_df) > 0:
            print(f"\n验证: 最大日期是 {final_df['ann_date'].max()}")
            if final_df['ts_code'].str.startswith('A').any():
                print("警告: 发现还有以 A 开头的代码！")
            else:
                print("验证通过: 没有以 A 开头的代码。")
                
            # 简单展示一个填充成功的案例（如果有）
            # 查找那些 ROE 不为空，但上一行（同股票）原本可能是空的逻辑较难直接展示
            # 这里只确认没有报错即可
            print("ROE 填充逻辑执行完毕。")
            
    else:
        print("没有成功读取到任何数据。")