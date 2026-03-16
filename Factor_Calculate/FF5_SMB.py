import logging
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from sklearn.linear_model import LinearRegression

'''
Fama-French SMB 因子
SMB = 1/3 * (SMB_bm + SMB_roe + SMB_inv)
    SMB_bm = 1/3 * (S/H + S/M + S/L) - 1/3 * (B/H + B/M + B/L)
    SMB_roe = 1/3 * (S/R + S/N + S/W) - 1/3 * (B/R + B/N + B/W)
    SMB_inv = 1/3 * (S/C + S/N + S/A) - 1/3 * (B/C + B/N + B/A)
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

    try:
        data = data.copy()
        data = data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
    except Exception as e:
        logger.error(f"[1-数据排序] 错误: {e}")
        raise

    try:
        data_eqy = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\归母股东权益.csv',
                               usecols=['ts_code', 'trade_date', 'total_hldr_eqy_exc_min_int'],
                               low_memory=False)
        data_c = data.merge(data_eqy, how='left', on=['ts_code', 'trade_date'])
        data_c['BM'] = data_c['total_hldr_eqy_exc_min_int'] / data_c['circ_mv']
        data_c['rank_BM'] = data_c.groupby('trade_date')['BM'].rank(pct=True)
    except Exception as e:
        logger.error(f"[2-BM计算] 错误: {e}")
        raise

    try:
        data_roe = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\roe.csv',
                               usecols=['ts_code', 'trade_date', 'roe'],
                               low_memory=False)
        data_c = data_c.merge(data_roe, how='left', on=['ts_code', 'trade_date'])
        data_c['rank_roe'] = data_c.groupby('trade_date')['roe'].rank(pct=True)
    except Exception as e:
        logger.error(f"[3-ROE计算] 错误: {e}")
        raise

    try:
        data_inv = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\环比购买固定资产支出增长率(TTM).csv',
                               usecols=['ts_code', 'trade_date', 'qoq'],
                               low_memory=False)
        data_c = data_c.merge(data_inv, how='left', on=['ts_code', 'trade_date'])
        data_c['rank_inv'] = data_c.groupby('trade_date')['qoq'].rank(pct=True)
    except Exception as e:
        logger.error(f"[4-INV计算] 错误: {e}")
        raise

    data_c['rank_circ_mv'] = data_c.groupby('trade_date')['circ_mv'].rank(pct=True)
    print(data_c.head())

    def spread_ret_calc(group, ret_col='dret'):
        big = group['rank_circ_mv'] > 0.5
        small = group['rank_circ_mv'] <= 0.5
        high_bm = group['rank_BM'] > 0.70
        middle_bm = (group['rank_BM'] > 0.30) & (group['rank_BM'] <= 0.70)
        low_bm = group['rank_BM'] <= 0.30
        robust_roe = group['rank_roe'] > 0.70
        neutral_roe = (group['rank_roe'] > 0.30) & (group['rank_roe'] <= 0.70)
        weak_roe = group['rank_roe'] <= 0.30
        aggressive_inv = group['rank_roe'] > 0.70
        neutral_inv = (group['rank_inv'] > 0.30) & (group['rank_inv'] <= 0.70)
        conservative_inv = group['rank_inv'] <= 0.30

        bh = group.loc[big & high_bm, ret_col]
        sh = group.loc[small & high_bm, ret_col]
        bl = group.loc[big & low_bm, ret_col]
        sl = group.loc[small & low_bm, ret_col]
        bm = group.loc[big & middle_bm, ret_col]
        sm = group.loc[small & middle_bm, ret_col]

        br = group.loc[big & robust_roe, ret_col]
        sr = group.loc[small & robust_roe, ret_col]
        bw = group.loc[big & weak_roe, ret_col]
        sw = group.loc[small & weak_roe, ret_col]
        bn = group.loc[big & neutral_roe, ret_col]
        sn = group.loc[small & neutral_roe, ret_col]

        ba = group.loc[big & aggressive_inv, ret_col]
        sa = group.loc[small & aggressive_inv, ret_col]
        bc = group.loc[big & conservative_inv, ret_col]
        sc = group.loc[small & conservative_inv, ret_col]
        bnn = group.loc[big & neutral_inv, ret_col]
        snn = group.loc[small & neutral_inv, ret_col]

        if bh.empty or sh.empty or bl.empty or sl.empty or br.empty or sr.empty or bw.empty or sw.empty or bn.empty or sn.empty or ba.empty or sa.empty or bc.empty or sc.empty or bnn.empty or snn.empty:
            return np.nan
        
        smb_bm = 1/3 * (sh.mean() + sm.mean() + sl.mean()) - 1/3 * (bh.mean() + bm.mean() + bl.mean())
        smb_roe = 1/3 * (sr.mean() + sn.mean() + sw.mean()) - 1/3 * (br.mean() + bn.mean() + bl.mean())
        smb_inv = 1/3 * (sc.mean() + snn.mean() + sa.mean()) - 1/3 * (bc.mean() + bnn.mean() + ba.mean())
        return (1/3 * (smb_bm + smb_roe + smb_inv)).round(7)

    try:
        daily_spread_series = data_c.groupby('trade_date').apply(
            lambda x: spread_ret_calc(x)
        )
    except Exception as e:
        logger.error(f"[5-SMB价差计算] 错误: {e}")
        raise

    try:
        factor_name = 'smb'
        return_data = daily_spread_series.reset_index()
        return_data.columns = ['trade_date', factor_name]
        return_data = return_data.dropna(subset=factor_name)
    except Exception as e:
        logger.error(f"[6-结果整理] 错误: {e}")
        raise

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 SMB 计算流程 ---")
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
    try:
        logger.info("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv',
            usecols=columns_needed,
            low_memory=False)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
    except Exception as e:
        logger.error(f"[main-读取数据] 错误: {e}")
        raise

    try:
        logger.info("开始计算特征")
        processed_data = calculate_alpha(history_data)
        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
    except Exception as e:
        logger.error(f"[main-计算特征] 错误: {e}")
        raise

    try:
        logger.info("正在生成特征分布直方图。")
        distribution_plot(processed_data)
    except Exception as e:
        logger.error(f"[main-分布图] 错误: {e}")
        raise

    try:
        logger.info("正在保存数据...")
        save_data(processed_data, "smb.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.error(f"[main-保存数据] 错误: {e}")
        raise
