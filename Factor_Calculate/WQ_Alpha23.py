import logging
import time
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_mad import mad_outlier_remove
from factor_winsorize import winsorize_factor
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot

'''      
Alpha_23 =  -1 * delta(high,2)    if mean(high,20) < high
            0    otherwise
'''
# --- 配置日志 ---
# level=logging.INFO 表示记录 INFO 及以上级别的信息
# format 定义了日志消息的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_23(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True]).reset_index(drop=True)

    data_c['mean_high_20d'] = data_c.groupby('ts_code')['high'].transform(
        lambda x: x.rolling(window=20).mean()
    )

    data_c['delta_high_ret2D'] = data_c.groupby('ts_code')['high'].transform(
        lambda x: (x - x.shift(2)) / x.shift(2)
    )
    data_c['alpha_23'] = np.where(
        data_c['mean_high_20d'].notna() & (data_c['mean_high_20d'] < data_c['high']), # 显式处理NaN
        -1 * data_c['delta_high_ret2D'],
        0
    )
    factor_name = data_c.columns[-1]

    try:
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=20)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_23']
    available_columns = [col for col in return_columns if col in data_c.columns]
    return_data = data_c[available_columns]
    return return_data


if __name__ == "__main__":
    print("--- 开始执行 Alpha#23 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
            'industry_name',
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
                                   usecols=columns_needed)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_23(history_data)
        logger.info(processed_data[processed_data['ts_code'] == '920992.BJ'])

        logger.info(f"Alpha#23 特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_23.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
