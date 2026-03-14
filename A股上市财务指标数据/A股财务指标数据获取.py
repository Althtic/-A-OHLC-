import tushare as ts
import pandas as pd
import numpy as np
import os  # 1. 导入 os 模块用于处理路径

pro = ts.pro_api()

# target_period = ['20160331', '20160630', '20160930', '20161231',
#                  '20170331', '20170630', '20170930', '20171231',
#                  '20180331', '20180630', '20180930', '20181231',
#                  '20190331', '20190630', '20190930', '20191231']

target_period = ['20200331', '20200630', '20200930', '20201231',
                 '20210331', '20210630', '20210930', '20211231',
                 '20220331', '20220630', '20220930', '20221231',
                 '20230331', '20230630', '20230930', '20231231',
                 '20240331', '20240630', '20240930', '20241231',
                 '20250331', '20250630', '20250930']               

# 2. 定义你要保存的目标文件夹路径 (请修改为你实际想要保存的路径)
# 注意：Windows 路径建议在字符串前加 r 以避免转义字符问题，例如 r'C:\MyData'
save_dir = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务指标数据'

# 3. 【重要】检查文件夹是否存在，不存在则自动创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"文件夹不存在，已自动创建: {save_dir}")
else:
    print(f"已确认存储路径: {save_dir}")

for period in target_period:
    try:
        df = pro.fina_indicator_vip(period=period, fields='ts_code,f_ann_date,end_date,total_hldr_eqy_exc_min_int')
        df = df.drop_duplicates()

        def fill_ann_date(row):
            if pd.notna(row['ann_date']):
                return row['ann_date']
            if pd.isna(row['end_date']):
                return np.nan
            ed = str(int(float(row['end_date']))).zfill(8)[:8]
            if len(ed) != 8:
                return np.nan
            suffix = ed[-4:]
            y = int(ed[:4])
            if suffix == '0331':
                return f'{y}0430'
            elif suffix == '0930':
                return f'{y}1031'
            elif suffix == '0630':
                return f'{y}0831'
            elif suffix == '1231':
                return f'{y+1}0430'
            return np.nan

        mask = df['ann_date'].isna() & df['end_date'].notna()
        df.loc[mask, 'ann_date'] = df.loc[mask].apply(fill_ann_date, axis=1)
        

        # (可选) 检查是否还有空值
        if df['ann_date'].isnull().any():
            print("警告：填补后 ann_date 仍有空值，可能是因为对应的 end_date 也是空的。")
        else:
            print("成功：所有 ann_date 的空值已填补完毕。")

        df = df.dropna(subset=['ann_date','end_date'])
        # 构建完整的文件路径：文件夹路径 + 文件名
        # 例如：C:\...\data\zcfz20160331.csv
        file_name = f"{period}.csv"
        full_path = os.path.join(save_dir, file_name)

        if int(len(df)) < 15000:
            print(f'没有出现过量调用，正在保存至: {full_path}')
            # 4. 使用完整路径保存
            df.to_csv(full_path, index=False)
     
        else:
            print(f'数据量较大 ({len(df)})，保存失败')

    except Exception as e:
        print(f"处理期间 {period} 出错: {e}")

print("所有数据处理完成。")