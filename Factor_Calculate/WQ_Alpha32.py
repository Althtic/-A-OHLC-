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
from calculate_rolling_corr import rolling_corr_numba
'''     
长短期结合 
Alpha_32 =  scale(((sum(close,7)/7)-close)) + (20 * scale(correlation(vwap,delay(close,5),252))))
修改为10倍的相关系数
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

def calculate_alpha_32(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True]).reset_index(drop=True)


    # part A: 短期价格动量
    try:
        data_c['vwap'] = 10 * data_c['amount'] / data_c['vol']
        data_c['delta_close_mean_7d_1d'] = data_c.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(window=7).mean() - x
        )
    except Exception as e:
        logger.info(f"Error in Part A: {e}")
        raise e

    # part B: 长期价格动量 
    try:
        def compute_corr(group):
            vwaps = group['vwap'].values
            closes = group['close_delay_5d'].values
            
            n = len(vwaps)
            if n < 10: return pd.Series([np.nan] * n, index=group.index)
            # 调用相关系数计算函数
            corr_res = rolling_corr_numba(vwaps, closes, 252) 
            # 必须返回一个 Series，且 index 要与 group 的 index 一致，这样 apply 才能自动对齐
            return pd.Series(corr_res, index=group.index)
        data_c['close_delay_5d'] = data_c.groupby('ts_code')['close'].shift(5)
        data_c['corr_252d_vwap_delay_close_5d'] = data_c.groupby('ts_code', group_keys=False).apply(compute_corr)
           
    except Exception as e:
        logger.info(f"Error in Part B: {e}")
        raise e
    
    # part C: 截面标准化
    try:
        data_c['delta_close_mean_7d_1d_scaled'] = data_c.groupby('trade_date')['delta_close_mean_7d_1d'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        data_c['corr_252d_vwap_delay_close_5d_scaled'] = data_c.groupby('trade_date')['corr_252d_vwap_delay_close_5d'].transform(
            lambda x: 10 * (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    except Exception as e:
        logger.info(f"Error in Part C: {e}")
        raise e


    data_c['alpha_32'] = data_c['corr_252d_vwap_delay_close_5d_scaled'] + data_c['delta_close_mean_7d_1d_scaled']
    factor_name = data_c.columns[-1]

    try:
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=252)
        # 去除极端值（MAD）
        data_c = mad_outlier_remove(data_c)
        # 去除极端值（分位数截断法）
        # data_c = winsorize_factor(data_c)
        # 中性化处理因子值（市值 & 行业：回归残差法）
        data_c = neutralize_factor(data_c, target_factor=factor_name)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
        # 截面 z-score 标准化
        data_c = zscore_transform(data_c)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e
    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_32']
    available_columns = [col for col in return_columns if col in data_c.columns]
    return_data = data_c[available_columns]

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印耗时信息
    logger.info(f"因子值计算完成！")
    logger.info(f"   - 返回数据行数：{len(return_data)}")
    logger.info(f"   - 总耗时：{elapsed_time:.4f} 秒 ({elapsed_time*1000:.2f} 毫秒)")

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#32 计算流程 ---")
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
            'amount',
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
        processed_data = calculate_alpha_32(history_data)
        logger.info(processed_data[processed_data['ts_code'] == '000001.SZ'])

        logger.info(f"Alpha#32 特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_32.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
