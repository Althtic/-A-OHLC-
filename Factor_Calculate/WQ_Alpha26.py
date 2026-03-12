import logging
import time
import warnings
import numpy as np
import pandas as pd
from save_csv import save_data
from factor_mad import mad_outlier_remove
from factor_winsorize import winsorize_factor
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot
from calculate_ts_rank import calc_ts_rank
from calculate_rolling_corr import rolling_corr_numba

'''      
分析成交量和最高价排名相关性识别市场异常
Alpha_26 = -1 * ts_max(correlation(ts_rank(vol,5),ts_rank(high,5),5),3)
'''
# --- 配置日志 ---
# level=logging.INFO 表示记录 INFO 及以上级别的信息
# format 定义了日志消息的格式
warnings.filterwarnings("ignore", category=FutureWarning)
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



def calculate_alpha(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)


    try: # part 1 计算成交量和最高价排名 
        codes = data_c['ts_code'].values
        ts_rank_vol_5 = calc_ts_rank(data_c['vol'], codes, 5)
        ts_rank_high_5 = calc_ts_rank(data_c['high'], codes, 5)
        data_c['ts_rank_vol_5'] = ts_rank_vol_5
        data_c['ts_rank_high_5'] = ts_rank_high_5

    except Exception as e:
        logger.info(f"Error in Part 1: {e}")
        raise e
    
    try: # part 2 计算成交量和最高价排名相关性
        corr_ts_rank_vol_high_5 = rolling_corr_numba(ts_rank_vol_5, ts_rank_high_5, 5)
        data_c['corr_ts_rank_vol_high_5'] = corr_ts_rank_vol_high_5
        ts_max_corr = data_c.groupby('trade_date')['corr_ts_rank_vol_high_5'].rolling(window=5).max().reset_index(level=0, drop=True)
        data_c['alpha_26'] = -ts_max_corr
        factor_name = data_c.columns[-1]
    except Exception as e:
        logger.info(f"Error in Part 2: {e}")
        raise e
    
    try:
        data_c[factor_name] = data_c[factor_name].replace([np.inf, -np.inf], np.nan)
        data_c = data_c.dropna(subset=[factor_name])
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=5)
        # 去除极端值（MAD）
        #data_c = mad_outlier_remove(data_c)
        # 去除极端值（分位数截断法）
        data_c = winsorize_factor(data_c)
        # 中性化处理因子值（市值 & 行业：回归残差法）
        data_c = neutralize_factor(data_c, target_factor=factor_name)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
        # 截面 z-score 标准化
        data_c = zscore_transform(data_c)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    # ==================== 返回指定列（修复列名） ====================
    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_26']
    return_data = data_c[return_columns]

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印耗时信息
    logger.info(f"因子值计算完成！")
    logger.info(f"   - 返回数据行数：{len(return_data)}")
    logger.info(f"   - 总耗时：{elapsed_time:.4f} 秒 ({elapsed_time * 1000:.2f} 毫秒)")

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
            'vol',
            'industry_name',
            'suspend_type',
            'circ_mv'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv',
                                   usecols=columns_needed,
                                   low_memory=False)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha(history_data)

        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_26.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
