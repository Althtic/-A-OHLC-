import logging
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot

'''
Fama-French CMA 因子
CMA = 1/2 * (S/C + B/C) - 1/2 * (S/A + B/A)
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
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)  # 列宽无限制（防止单元格内容被截断）


def calculate_alpha(data) -> pd.DataFrame:

    data = data.copy()
    data = data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
    data_eqy = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\环比购买固定资产支出增长率(TTM).csv',
                           usecols=['ts_code', 'trade_date', 'qoq'],
                           low_memory=False
                           ) 


    data_c = data.merge(data_eqy, how='left', on=['ts_code', 'trade_date'])

    data_c['rank_qoq'] = data_c.groupby('trade_date')['qoq'].rank(pct=True)
    data_c['rank_circ_mv'] = data_c.groupby('trade_date')['circ_mv'].rank(pct=True)
    

    def spread_ret_calc(group, ret_col='dret'):
        # 流通市值中位数分Big/Small; BM按30%/70%分High/Middle/Low
        big = group['rank_circ_mv'] > 0.5
        small = group['rank_circ_mv'] <= 0.5
        aggresive = group['rank_qoq'] > 0.70
        conservative = group['rank_qoq'] <= 0.30

        sc = group.loc[small & conservative, ret_col]
        sa = group.loc[small & aggresive, ret_col]
        bc = group.loc[big & conservative, ret_col]
        ba = group.loc[big & aggresive, ret_col]

        if sc.empty or sa.empty or bc.empty or ba.empty:
            return np.nan

        ret_conversative = (sc.mean() + bc.mean()) / 2
        ret_aggresive = (sa.mean() + ba.mean()) / 2
        return (ret_conversative - ret_aggresive).round(7)
    daily_spread_series = data_c.groupby('trade_date').apply(
        lambda x: spread_ret_calc(x)
    )
    factor_name = 'cma'
    return_data = daily_spread_series.reset_index()

    return_data.columns = ['trade_date', factor_name]

    try:
        return_data = return_data.dropna(subset=factor_name)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e
    return return_data



if __name__ == "__main__":
    logger.info("--- 开始执行 CMA 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'dret',
            'industry_name',
            'circ_mv',
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv',
            usecols=columns_needed,
            low_memory=False)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
        logger.info(f"开始计算特征")
        # print(history_data.head())
        processed_data = calculate_alpha(history_data)
        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "cma.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")