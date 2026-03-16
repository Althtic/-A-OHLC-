import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
from statsmodels.regression.linear_model import OLS
from config_loader import traget_factor, test_window_start, test_window_end

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────
def data_loading(traget_factor):
    logger.info(f"目标检测因子: {traget_factor}")
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    return df_loading
# ─────────────────────────────────────────────────────────────────────────────
# 筛选时间窗口
# ─────────────────────────────────────────────────────────────────────────────
def cut_time_window(df, test_window_start, test_window_end):
    try:
        if df['trade_date'].dtype in ['int64', 'int32']:
            df_date_for_check = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        elif df['trade_date'].dtype == 'object':
            df_date_for_check = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        else:
            df_date_for_check = df['trade_date']

        original_min_date = df_date_for_check.min()
        original_max_date = df_date_for_check.max()
        logger.info(f"原始数据的时间范围: {original_min_date.strftime('%Y%m%d')} 到 {original_max_date.strftime('%Y%m%d')}")

        input_start_dt = pd.to_datetime(str(test_window_start), format='%Y%m%d')
        input_end_dt = pd.to_datetime(str(test_window_end), format='%Y%m%d')

        if input_start_dt > original_max_date or input_end_dt < original_min_date:
            logger.info(f"错误: 请求的时间窗口 [{test_window_start}, {test_window_end}] 完全超出了原始数据的日期范围 [{original_min_date.strftime('%Y%m%d')}, {original_max_date.strftime('%Y%m%d')}]。")
            logger.info("--- 程序终止 ---")
            return None 

        if input_start_dt < original_min_date or input_end_dt > original_max_date:
             effective_start = max(input_start_dt, original_min_date)
             effective_end = min(input_end_dt, original_max_date)
             logger.info(f"警告: 请求的时间窗口 [{test_window_start}, {test_window_end}] 部分超出了原始数据范围。")
             logger.info(f"     建议使用有效范围: [{effective_start.strftime('%Y%m%d')}, {effective_end.strftime('%Y%m%d')}]")
             return None 

        df_copy = df.copy()
        if df_copy['trade_date'].dtype in ['int64', 'int32']:
            df_copy['trade_date'] = df_copy['trade_date'].astype(str)

        mask = (df_copy['trade_date'] >= test_window_start) & (df_copy['trade_date'] <= test_window_end)
        df_in_window = df_copy.loc[mask]
        df_in_window = df_in_window.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        return df_in_window
    except Exception as e:
        logger.info(f"处理时间窗口时发生错误: {e}")
        return None
# ─────────────────────────────────────────────────────────────────────────────
# 窗口期数据合并（无风险利率数据 & Fama-French 五因子日度数据）
# ─────────────────────────────────────────────────────────────────────────────
def data_preprocessing(df, target_factor, test_window_start, test_window_end):
    df = df.copy()
    df = cut_time_window(df, test_window_start, test_window_end)
    
    try: # 读取无风险利率数据（SHIBOR 3M）
        rf = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\rf.csv')
        rf_filtered = rf[['trade_date', 'rf']].copy()
        rf_filtered['trade_date'] = rf_filtered['trade_date'].astype(str)
        df_rf = df.merge(rf_filtered, on=['trade_date'], how='left')
        df_rf['excess_dret'] = df_rf['dret'] - df_rf['rf']
    except Exception as e:
        logger.info(f"读取无风险利率数据时发生错误: {e}")
        return None

    try: # 读取 Fama-French 五因子日度数据
        df_ff5 = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\FF5.csv')
        df_ff5['trade_date'] = df_ff5['trade_date'].astype(str) # 将 trade_date 转换为字符串类型（原来是int）
        df_ff5_rf = df_rf.merge(df_ff5, on='trade_date', how='left')
        # 日度收益率滞后一期（对未来一天的收益率进行预测与解释）
        df_ff5_rf['excess_dret_shift1'] = df_ff5_rf.groupby('ts_code')['excess_dret'].shift(1)
        df_ff5_rf = df_ff5_rf.dropna(subset=['excess_dret_shift1'])
    except Exception as e:
        logger.info(f"合并五因子数据与无风险利率数据时发生错误: {e}")
        return None

    return_columns = ['ts_code', 'trade_date', 'excess_dret_shift1', 'mkt', 'smb', 'hml', 'rmw', 'cma', target_factor]
    df_ff5_rf = df_ff5_rf[return_columns]
    return df_ff5_rf
# ─────────────────────────────────────────────────────────────────────────────
# 截面回归分析（Fama-MacBeth Regression）
# ─────────────────────────────────────────────────────────────────────────────
def regression_analysis_by_date(df, factor_col=None, target_col='excess_dret_shift1', ff5_factors=None):
    logger.info("截面回归分析(Fama-MacBeth Regression)...")
    if ff5_factors is None:
        FF5_FACTORS = ['mkt', 'smb', 'hml', 'rmw', 'cma']
    else:
        FF5_FACTORS = ff5_factors

    if factor_col is None:
        target_factors = []
    elif isinstance(factor_col, str):
        target_factors = [factor_col]
    else:
        target_factors = list(factor_col)

    # 构建回归变量列表 (X 的列)
    all_regression_factors = target_factors + [f for f in FF5_FACTORS if f not in target_factors]
    
    if not all_regression_factors:
        logger.error("没有指定任何回归变量 (既无目标因子也无 FF5 因子)")
        return pd.DataFrame()

    results = []
    grouped = df.groupby('trade_date')
    total_groups = len(grouped)
    
    logger.info(f"开始截面回归分析 | Y: {target_col} | X: {all_regression_factors}")

    for date, group in grouped:
        cols_needed = [target_col] + all_regression_factors
        missing_cols = [c for c in cols_needed if c not in group.columns]
        if missing_cols:
            continue
        subset = group.dropna(subset=cols_needed)
        n_samples = len(subset)
        if n_samples < 10:
            continue
        y = subset[target_col].values
        X = subset[all_regression_factors].values
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True)
        
        try:
            model.fit(X, y)
            alpha_val = float(model.intercept_)
            r_squared_val = float(model.score(X, y))
            # 映射系数到因子名
            coefs_dict = {}
            for i, factor_name in enumerate(all_regression_factors):
                coefs_dict[f'beta_{factor_name}'] = float(model.coef_[i])
            
            res_row = {
                'trade_date': date,
                'alpha': alpha_val,
                'r_squared': r_squared_val,
                'n_samples': n_samples,
                **coefs_dict
            }
            results.append(res_row)
        except Exception as e:
            logger.warning(f"日期 {date} 回归失败: {e}")
            continue

    if not results:
        logger.warning("回归分析未产生任何有效结果。")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # 按日期排序
    if 'trade_date' in result_df.columns:
        result_df = result_df.sort_values('trade_date').reset_index(drop=True)
    logger.info(f"回归分析完成。有效交易日: {len(result_df)} / {total_groups}")
    return result_df
# ─────────────────────────────────────────────────────────────────────────────
# 因子独立性检验（基于Fama-French五因子）
# ─────────────────────────────────────────────────────────────────────────────
def factor_purity_test(df, target_factor, ff5_factors=None):
    """
    用FF5对目标因子做时间序列回归，检验目标因子是否仅为FF5的某种暴露。
    Y = 目标因子截面均值(日度), X = FF5
    R²高=因子主要为FF5暴露(不纯净), R²低=因子含独立信息(纯净)
    """
    logger.info("因子独立性检验(基于Fama-French五因子)...")
    if ff5_factors is None:
        FF5_FACTORS = ['mkt', 'smb', 'hml', 'rmw', 'cma']
    else:
        FF5_FACTORS = ff5_factors
    cols = ['trade_date', target_factor] + FF5_FACTORS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error(f"纯净度检验缺少列: {missing}")
        return None
    agg = df.groupby('trade_date').agg({target_factor: 'mean', **{f: 'first' for f in FF5_FACTORS}}).reset_index()
    agg = agg.dropna(subset=[target_factor] + FF5_FACTORS)
    if len(agg) < 22:
        logger.warning("纯净度检验样本不足")
        return None
    y = agg[target_factor].values
    X = agg[FF5_FACTORS].values
    X_const = np.column_stack([np.ones(len(X)), X])
    model = OLS(y, X_const).fit()
    r2 = float(model.rsquared)
    resid = model.resid
    coefs = {f: float(model.params[i + 1]) for i, f in enumerate(FF5_FACTORS)}
    pvals = {f: float(model.pvalues[i + 1]) for i, f in enumerate(FF5_FACTORS)}
    tvals = {f: float(model.tvalues[i + 1]) for i, f in enumerate(FF5_FACTORS)}
    purity = {
        'r2_ff5': r2,
        'resid_std': float(np.std(resid)),
        'factor_std': float(np.std(y)),
        'independent_ratio': 1 - r2,
        'coefs': coefs,
        'pvals': pvals,
        'tvals': tvals,
        'intercept': float(model.params[0]),
    }
    print_purity_results(purity)
    return purity
# ─────────────────────────────────────────────────────────────────────────────
# 因子独立性检验结果
# ─────────────────────────────────────────────────────────────────────────────
def print_purity_results(purity):
    if purity is None:
        return
    print("\n======== 因子纯净度检验 (目标因子 vs FF5) ========")
    print(f"R²(FF5解释):     {purity['r2_ff5']:.4f}  (高=主要为FF5暴露)")
    print(f"独立成分比例:   {purity['independent_ratio']:.4f}  (1-R²)")
    print(f"残差标准差:     {purity['resid_std']:.6f}")
    print(f"因子标准差:     {purity['factor_std']:.6f}")
    print("FF5回归系数 (coef / t值 / p值):")
    for k in purity['coefs']:
        c, t, p = purity['coefs'][k], purity['tvals'][k], purity['pvals'][k]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {k}: {c:.6f}  t={t:.3f}  p={p:.4f} {sig}")
    print("================================================\n")
# ─────────────────────────────────────────────────────────────────────────────
# 绘制beta稳定性图
# ─────────────────────────────────────────────────────────────────────────────
def plot_beta_stability(result_df, beta_col, target_factor):
    result_df = result_df.copy()
    result_df['trade_date_dt'] = pd.to_datetime(result_df['trade_date'].astype(str), format='%Y%m%d')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result_df['trade_date_dt'], result_df[beta_col], label='Daily Beta', alpha=0.6, color='gray', linewidth=1)
    ax.plot(result_df['trade_date_dt'], result_df['beta_ma21'], label='21-Day MA', linewidth=2, color='red')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Beta', fontsize=10)
    ax.set_title(f'Factor Stability: {target_factor}\n(Daily Beta vs. 21-Day Moving Average)', fontsize=12, pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
""" =======================主执行函数======================= """
def regression_analysis():
    df = data_loading(traget_factor)
    df = data_preprocessing(df, traget_factor, test_window_start, test_window_end)
    
    factor_purity_test(df, traget_factor)

    result_df = regression_analysis_by_date(df, factor_col=traget_factor)

    beta_col = f'beta_{traget_factor}'
    if beta_col not in result_df.columns:
        logger.error(f"目标因子beta列缺失: {beta_col}")
        return
    result_df['beta_ma21'] = result_df[beta_col].rolling(window=21, min_periods=14).mean()

    plot_beta_stability(result_df, beta_col, traget_factor)

if __name__ == "__main__":
    regression_analysis()
