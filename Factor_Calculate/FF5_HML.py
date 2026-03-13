import logging
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot


'''
Fama-French HML 因子
账面市值比 BM
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
    data_eqy = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_balance_sheet.csv',
                           usecols=['ts_code', 'trade_date', 'total_hldr_eqy_exc_min_int'],
                           low_memory=False
                           )  # total_hldr_eqy_exc_min_int 归母股东权益
    data_c = data.merge(data_eqy, how='left', on=['ts_code', 'trade_date'])
    # BM = 归母股东收益（去除少数股东） / 流通总市值
    data_c['BM'] = data_c['total_hldr_eqy_exc_min_int'] / data_c['circ_mv']
    data_c['rank_BM'] = data_c.groupby('trade_date')['BM'].rank(pct=True)
    

    '''参照Fama-French原文中的方法,计算多空收益'''
    def spread_ret_calc(group, rank_col='rank_BM', ret_col='dret'):
        # 70%-90% - 10%-30%
        short_mask = (group[rank_col] > 0.10) & (group[rank_col] <= 0.30)
        short_group = group.loc[short_mask, ret_col]
        long_mask = (group[rank_col] > 0.70) & (group[rank_col] <= 0.90)
        long_group = group.loc[long_mask, ret_col]

        if long_group.empty or short_group.empty:
            return np.nan

        mean_long = long_group.mean()
        mean_short = short_group.mean()
        spread = mean_long - mean_short
        spread = spread.round(7)
        return spread
    daily_spread_series = data_c.groupby('trade_date').apply(
        lambda x: spread_ret_calc(x)
    )
    factor_name = 'hml'
    return_data = daily_spread_series.reset_index()
    print(return_data)
    return
    return_data.columns = ['trade_date', factor_name]

    try:
        return_data = return_data.dropna(subset=factor_name)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e
    return return_data



if __name__ == "__main__":
    logger.info("--- 开始执行 HML 计算流程 ---")
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
        save_data(processed_data, "hml.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")