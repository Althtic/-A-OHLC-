import logging
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from config_loader import traget_factor, test_window_start, test_window_end, test_period, ic_ma_period

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', None)         # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)  # 列宽无限制（防止单元格内容被截断）

# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────
def data_loading(traget_factor):
    logger.info(f"目标检测因子：{traget_factor}")
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    return df_loading
# ─────────────────────────────────────────────────────────────────────────────
# 截取时间窗
# ─────────────────────────────────────────────────────────────────────────────
def cut_time_window(df, start_time, end_time):
    try:
        # 预处理原始数据的日期列
        if df['trade_date'].dtype in ['int64', 'int32']:
            df_date_for_check = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        elif df['trade_date'].dtype == 'object':
            df_date_for_check = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        else:
            df_date_for_check = df['trade_date']

        # 获取原始数据的日期范围
        original_min_date = df_date_for_check.min()
        original_max_date = df_date_for_check.max()
        logger.info(f"原始数据的时间范围: {original_min_date.strftime('%Y%m%d')} 到 {original_max_date.strftime('%Y%m%d')}")

        # 将输入的时间也转换为 datetime 进行比较
        input_start_dt = pd.to_datetime(str(start_time), format='%Y%m%d')
        input_end_dt = pd.to_datetime(str(end_time), format='%Y%m%d')

        # 判断输入窗口是否超出原始数据范围 ---
        if input_start_dt > original_max_date or input_end_dt < original_min_date:
            logger.info(f"错误: 请求的时间窗口 [{start_time}, {end_time}] 完全超出了原始数据的日期范围 [{original_min_date.strftime('%Y%m%d')}, {original_max_date.strftime('%Y%m%d')}]。")
            logger.info("--- 程序终止 ---")

            return None # 返回 None 表示操作失败

        if input_start_dt < original_min_date or input_end_dt > original_max_date:
             # 检查是否有部分超出，如果有，也视为超出范围
             effective_start = max(input_start_dt, original_min_date)
             effective_end = min(input_end_dt, original_max_date)
             logger.info(f"警告: 请求的时间窗口 [{start_time}, {end_time}] 部分超出了原始数据范围。")
             logger.info(f"     建议使用有效范围: [{effective_start.strftime('%Y%m%d')}, {effective_end.strftime('%Y%m%d')}]")

             return None # 返回 None 表示操作失败

        # 如果时间窗口在范围内，进行筛选
        # 原始数据 'trade_date' 列转化为字符串 string (用于后续筛选)
        df_copy = df.copy()
        if df_copy['trade_date'].dtype in ['int64', 'int32']:
            df_copy['trade_date'] = df_copy['trade_date'].astype(str)

        # 创建布尔掩码并筛选
        mask = (df_copy['trade_date'] >= start_time) & (df_copy['trade_date'] <= end_time)
        df_in_window = df_copy.loc[mask]
        df_in_window = df_in_window.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        return df_in_window

    except Exception as e:
        logger.info(f"处理时间窗口时发生错误: {e}")

        return None
# ─────────────────────────────────────────────────────────────────────────────
# 持有期对数收益率计算, 后续将分别按照t期因子值与t+1期收益率进行排序
# ─────────────────────────────────────────────────────────────────────────────
def data_preprocessing(df,test_window_start,test_window_end):
    df = df.copy()
    # 截取回测期间的历史数据
    df_preprocess = cut_time_window(df, test_window_start, test_window_end)
    # 今日对数收益率计算
    df_preprocess['lndret'] = np.log(df_preprocess['close'] / df_preprocess['pre_close'])
    # 持有至第二个交易日的对数收益率holding1D_lndret
    df_preprocess['holding1D_lndret'] = df_preprocess.groupby('ts_code')['lndret'].shift(-1)
    df_preprocess = df_preprocess.dropna()
    return df_preprocess
# ─────────────────────────────────────────────────────────────────────────────
# 因子与未来累计收益排序
# ─────────────────────────────────────────────────────────────────────────────
def factor_cumuret_rank(df, traget_factor, test_period):
    # 根据因子值大小排序(升序)，用于计算斯皮尔曼相关系数(RankIC)
    df_factor_ranking = df
    # 按照过滤后得到的中性化因子值进行截面排序
    df_factor_ranking['factor_rank'] = df_factor_ranking.groupby('trade_date')[traget_factor].rank(pct=True)
    # 未来累计收益排序（升序），用于计算斯皮尔曼相关系数（RankIC）
    # rolling()的计算只能面向“过去”，所以需要提前翻转数据（日期翻转），使得rolling()函数可以直接计算未来收益
    df_factor_ranking = df_factor_ranking.sort_values(by=['trade_date', 'ts_code'], ascending=[False, True])
    # 计算未来持有test_period个交易日的收益累计和
    df_factor_ranking['period_cumu_lndret'] = df_factor_ranking.groupby('ts_code')['lndret'].transform(
        lambda x: x.shift(1).rolling(window = test_period).sum()
    )
  
    df_cumu_dret_rank = df_factor_ranking.dropna(subset=['period_cumu_lndret'])
    # 数据顺序恢复原状(升序)
    df_cumu_dret_rank = df_cumu_dret_rank.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
    # 针对累计对数收益率与日度收益率进行截面rank排序
    df_cumu_dret_rank['cumu_lndret_rank'] = df_cumu_dret_rank.groupby('trade_date')['period_cumu_lndret'].rank(pct=True)
    df_cumu_dret_rank['daily_cumu_lndret_rank'] = df_cumu_dret_rank.groupby('trade_date')['holding1D_lndret'].rank(pct=True)
    df_factor_cumuret_rank = df_cumu_dret_rank
  
    return df_factor_cumuret_rank
# ─────────────────────────────────────────────────────────────────────────────
# RankIC, ICmean, ICIR及相关评价指标计算
# ─────────────────────────────────────────────────────────────────────────────
def IC_calculate(df, ic_ma_period, test_period):
    df_ic_processing = df.copy()
    df_ic_processing['trade_date'] = pd.to_datetime(df_ic_processing['trade_date'])
    # 持有期rank_IC（适配调仓频率）
    rank_ic_series = df_ic_processing.groupby('trade_date').apply(
        lambda x: x['factor_rank'].corr(x['cumu_lndret_rank']), include_groups=False
    )

    # 因子有效性评估
    ic_ttest_sample(rank_ic_series, threshold=0.05)
    # 计算累计IC及最大回撤 
    cumulative_ic_series = rank_ic_series.cumsum()
    cumu_ic_running_max = cumulative_ic_series.cummax()
    cumu_ic_drawdown = cumu_ic_running_max - cumulative_ic_series
    cumu_ic_max_dd = cumu_ic_drawdown.max()
    max_dd_date = cumu_ic_drawdown.idxmax()
    cumu_ic_drawdown_ratio = cumu_ic_max_dd / cumu_ic_running_max.loc[max_dd_date]
    # 计算Rank_IC的均值和标准差
    rank_ic_series_mean = rank_ic_series.mean()
    rank_ic_series_std = rank_ic_series.std()
    ICIR_value = rank_ic_series_mean / rank_ic_series_std
    # 计算IC胜率
    ic_win_rate = (rank_ic_series > 0).mean()
    # 计算IC_MA及均值
    min_period = int(int(test_period) * 0.6) # 保证窗口计算稳定性(60%)
    ic_ma_series = rank_ic_series.rolling(window=ic_ma_period, min_periods=min_period).mean()  # 持有期IC的滚动均值
    # IC衰减：各持有期限(1~test_period日)的截面IC均值
    df_desc = df_ic_processing.sort_values(by=['trade_date', 'ts_code'], ascending=[False, True])
    ic_decay = []
    for h in range(1, test_period + 1):
        cumu_h = df_desc.groupby('ts_code')['lndret'].transform(
            lambda x, w=h: x.shift(1).rolling(window=w).sum()
        )
        df_h = df_desc.assign(_cumu=cumu_h).dropna(subset=['_cumu'])
        df_h = df_h.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        df_h['_rank'] = df_h.groupby('trade_date')['_cumu'].rank(pct=True)
        ic_s = df_h.groupby('trade_date').apply(
            lambda g: g['factor_rank'].corr(g['_rank']), include_groups=False
        )
        ic_decay.append(ic_s.mean())

    temp_df = pd.DataFrame({
        'trade_date': rank_ic_series.index,
        'Rank_IC': rank_ic_series.values,
        'Rank_IC_Std': [rank_ic_series_std] * len(rank_ic_series),
        'Cumulative_IC': cumulative_ic_series.values,
        'Cumulative_IC_MaxDD': [cumu_ic_max_dd] * len(rank_ic_series),
        'MaxDD_Occur_Date': [max_dd_date] * len(rank_ic_series), 
        'Cumulative_IC_MaxDD_Ratio': [cumu_ic_drawdown_ratio] * len(rank_ic_series),
        'IC_MA': ic_ma_series.values,
        'ICIR': [ICIR_value] * len(rank_ic_series),
        'IC_Win_Rate': [ic_win_rate] * len(rank_ic_series),
    })

    # 计算每月的 Rank_IC 均值
    temp_df['year_month'] = temp_df['trade_date'].dt.to_period('M')  # e.g., '2024-08'
    monthly_means = temp_df.groupby('year_month')['Rank_IC'].mean()
    monthly_ICIR = temp_df.groupby('year_month').apply(
        lambda x: x['Rank_IC'].mean()/x['Rank_IC'].std(), include_groups=False
    )
    # 使用 map 函数，将每一行的 'year_month' 映射到对应的月度均值
    temp_df['Monthly_Rank_IC_Mean'] = temp_df['year_month'].map(monthly_means)
    temp_df['Monthly_ICIR'] = temp_df['year_month'].map(monthly_ICIR)

    # 计算每年的 Rank_IC 均值，原理同上
    temp_df['year'] = temp_df['trade_date'].dt.to_period('Y')  # e.g., '2024'
    yearly_means = temp_df.groupby('year')['Rank_IC'].mean()
    yearly_ICIR = temp_df.groupby('year').apply(
        lambda x: x['Rank_IC'].mean() / x['Rank_IC'].std(), include_groups=False
    )
    temp_df['Yearly_Rank_IC_Mean'] = temp_df['year'].map(yearly_means)
    temp_df['Yearly_ICIR'] = temp_df['year'].map(yearly_ICIR)
    return temp_df, ic_decay
# ─────────────────────────────────────────────────────────────────────────────
# IC序列的t检验（双尾检验), 需同样进行Newey-West调整自相关
# ─────────────────────────────────────────────────────────────────────────────
def ic_ttest_sample(rank_ic_series, threshold=0.05, alpha=0.05):
    # 数据预处理
    if hasattr(rank_ic_series, 'values'):
        rank_ic_series = rank_ic_series.values
    rank_ic_series = np.array(rank_ic_series, dtype=float)
    rank_ic_series = rank_ic_series[~np.isnan(rank_ic_series)]
    n = len(rank_ic_series)
    
    if n < 5:
        print("样本量不足 (N<5)，无法进行可靠检验。")
        return {"t_stat": np.nan, "status": "invalid", "mean": np.nan, "se": np.nan}

    # 构建截距模型 (Y = mean + error)
    X = np.ones((n, 1))
    model = sm.OLS(rank_ic_series, X)
    results = model.fit()
    
    lags_calc = int(np.floor(4 * (n / 100)**(2/9)))
    # 安全约束: 
    # 1. 至少为 1 (只要 n>1)
    # 2. 不能超过 n-2 (保证自由度)，通常建议不超过 n/4 以避免过度平滑
    lags_used = max(1, min(lags_calc, n - 2))

    # 计算 HAC (Newey-West) 协方差矩阵
    try:
        cov_matrix = cov_hac(results, nlags=lags_used)
        se_nw = np.sqrt(cov_matrix[0, 0])
    except Exception:
        # 如果 NW 计算失败，回退到普通标准误
        se_nw = results.bse[0]
        lags_used = 0

    sample_mean = results.params[0]
    
    if se_nw < 1e-12:
        t_stat = 0.0
    else:
        t_stat = sample_mean / se_nw
    
    # 判断逻辑
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    is_significant = abs(t_stat) > t_critical
    
    status = "invalid"
    
    if not is_significant:
        status = "invalid"
    else:
        if se_nw >= 1e-12:
            # 检验是否显著大于 threshold
            t_stat_upper = (sample_mean - threshold) / se_nw
            p_upper = 1 - stats.t.cdf(t_stat_upper, df=n - 1)
            is_strong_positive = p_upper < alpha
            
            # 检验是否显著小于 -threshold
            t_stat_lower = (sample_mean + threshold) / se_nw
            p_lower = stats.t.cdf(t_stat_lower, df=n - 1)
            is_strong_negative = p_lower < alpha
        else:
            is_strong_positive = False
            is_strong_negative = False
            
        if is_strong_positive:
            status = "strong_positive"
        elif is_strong_negative:
            status = "strong_negative"
        else:
            status = "normal"

    print(f"T-Stat (NW): {t_stat:.4f} | Mean: {sample_mean:.4f} | SE: {se_nw:.4f} | Status: {status}")
    
    if status == "strong_positive":
        print("结论：强有效正向因子")
    elif status == "strong_negative":
        print("结论：强有效反向因子")
    elif status == "normal":
        print("结论：普通有效因子（显著异于 0 但未突破阈值）")
    else:
        print("结论：无效因子（不显著）")
    return {
        "t_stat": t_stat,
        "status": status,
        "mean": sample_mean,
        "se": se_nw
    }
# ─────────────────────────────────────────────────────────────────────────────
# 嵌套字典，各月份对应的各年度指标，单独独立出来的，便于可视化的前置处理
# ─────────────────────────────────────────────────────────────────────────────
def monthly_processing(df):
    df = df.copy()
    # 需要处理的列
    target_columns = ['year_month', 'year', 'Monthly_Rank_IC_Mean','Monthly_ICIR']
    df = df[target_columns]
    # 确保 'year_month' 列是字符串类型
    df['year_month'] = df['year_month'].astype(str)
    df['month'] = df['year_month'].str.split('-').str[1].astype(int)

    # 构建嵌套字典
    result_nested_dict_mean = {}
    # 构建 Monthly_ICIR 的嵌套字典
    result_nested_dict_icir = {}

    for _, row in df.iterrows():
        month = row['month']  # month 已经是 int
        year = row['year']    # year 已经是 int
        mean_value = row['Monthly_Rank_IC_Mean']
        icir_value = row['Monthly_ICIR']

        # 为 Monthly_Rank_IC_Mean 构建字典
        if month not in result_nested_dict_mean:
            result_nested_dict_mean[month] = {}
        result_nested_dict_mean[month][year] = mean_value

        # 为 Monthly_ICIR 构建字典
        if month not in result_nested_dict_icir:
            result_nested_dict_icir[month] = {}
        result_nested_dict_icir[month][year] = icir_value

    return result_nested_dict_mean, result_nested_dict_icir
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅰ: 持有期IC+均值 | 累计IC+柱状 | IC衰减
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_analysis(df, ic_decay, test_period):
   
    rank_ic_series = df.set_index('trade_date')['Rank_IC']
    cumulative_ic_series = df.set_index('trade_date')['Cumulative_IC']
    ic_ma_vals = df.set_index('trade_date')['IC_MA']
    # 全局指标获取
    icir = df['ICIR'].iloc[0]
    max_dd_date_val = df['MaxDD_Occur_Date'].iloc[0]
    cumu_ic_drawdown_ratio_val = df['Cumulative_IC_MaxDD_Ratio'].iloc[0]
    ic_win_rate_val = df['IC_Win_Rate'].iloc[0]
    rank_ic_series_std_val = df['Rank_IC_Std'].iloc[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f'IC Analysis (ICIR: {icir:.2f})', fontsize=12, y=1.02)

    ax1.plot(rank_ic_series.index, rank_ic_series.values, label=f'Rank IC({test_period}D)', color='blue', linewidth=0.8)
    ax1.plot(rank_ic_series.index, ic_ma_vals, label='Rank IC MA', color='red', linewidth=1.2)
    ax1.set_title('Rank IC & Mean')
    text_content = f'IC Win Rate: {ic_win_rate_val:.4f} ({ic_win_rate_val*100:.2f}%)\nRank IC Std: {rank_ic_series_std_val:.4f}'
    ax1.text(0.5, 0.98, text_content, 
         transform=ax1.transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='center')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()


    ax2.bar(cumulative_ic_series.index, rank_ic_series.values, alpha=0.35, color='steelblue', width=1.5, label='Rank IC')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(cumulative_ic_series.index, cumulative_ic_series.values, label='Cumulative IC', color='green', linewidth=1.0)
    ax2.set_title('Cumulative IC')
    ax2.set_ylabel('Rank IC')
    ax2_twin.set_ylabel('Cumulative IC')


    text_content = f'MaxDD: {cumu_ic_drawdown_ratio_val:.4f} ({cumu_ic_drawdown_ratio_val*100:.2f}%)\nDate: {max_dd_date_val.strftime("%Y-%m-%d")}'
    ax2.text(0.5, 0.98, text_content, 
         transform=ax2.transAxes, 
         fontsize=10, 
         verticalalignment='top', 
         horizontalalignment='center')

    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    horizons = list(range(1, len(ic_decay) + 1))
    ax3.plot(horizons, ic_decay, 'o-', color='darkorange', linewidth=1.2, markersize=4)
    ax3.set_title('IC Daily Decay')
    ax3.set_xlabel('Holding Days')
    ax3.set_ylabel('Rank IC Mean')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, linestyle='--', alpha=0.6)
    step = 5 if len(horizons) > 10 else 1
    ax3.set_xticks(horizons[::step])

    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅱ:Yearly IC_mean & ICIR
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_yearly_series_bar(df):
    yearly_data = df.sort_values(by='year').copy()
    # 提取 'year' 列的唯一值，并转换为字符串列表
    years = yearly_data['year'].unique().astype(str).tolist()
    # 提取 Yearly_ICIR 和 Rank_IC_Mean 值
    icir_values = yearly_data['Yearly_ICIR'].unique().tolist()
    rank_ic_mean_values = yearly_data['Yearly_Rank_IC_Mean'].unique().tolist()
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 左图：ICIR ---
    bars0 = axs[0].bar(years, icir_values)
    axs[0].set_title('ICIR')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('ICIR')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', alpha=0.5)
    # 添加数值标签 (保留2位小数)
    for bar in bars0:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=10)

    # --- 右图：IC Mean ---
    bars1 = axs[1].bar(years, rank_ic_mean_values)
    axs[1].set_title('IC Mean')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('IC Mean')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y', alpha=0.5)
    # 添加数值标签 (保留4位小数，因为IC通常较小)
    for bar in bars1:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅲ:Monthly & IC_mean & ICIR
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_monthly_series_bar(dict1, dict2):
    result_nested_dict_mean = dict1
    result_nested_dict_icir = dict2
    months = list(range(1, 13))
    years = set()

    for month_year_values in [result_nested_dict_mean, result_nested_dict_icir]:
        for month, year_values in month_year_values.items():
            years.update(year_values.keys())

    years = sorted(years)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for idx, (title, data_dict) in enumerate(
            [("Rank IC Mean", result_nested_dict_mean), ("ICIR", result_nested_dict_icir)]):
        ax = axs[idx]
        for i, year in enumerate(years):
            values = [data_dict.get(month, {}).get(year, 0) for month in months]
            ax.bar([m + i * 0.1 for m in months], values, width=0.1, label=str(year))
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks([m + 0.1 * (len(years) - 1) / 2 for m in months])
        ax.set_xticklabels(months)
        ax.legend()
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 主运行函数
# ─────────────────────────────────────────────────────────────────────────────
def run():
    df = data_loading(traget_factor)
    '''数据加载与预处理'''
    df_initial_preprocess = data_preprocessing(df, test_window_start, test_window_end)
    '''计算因子与未来累计收益排序'''
    df_factor_rank_processed = factor_cumuret_rank(df_initial_preprocess, traget_factor, test_period)
    '''因子有效性相关评价指标'''
    df_validation_features, ic_decay = IC_calculate(df_factor_rank_processed, ic_ma_period, test_period)
    # print(df_validation_features.head(3))
    '''返回嵌套字典，Rank IC mean & ICIR mean (monthly)可视化的前置操作，单独分出来'''
    result_nested_dict_mean, result_nested_dict_icir = monthly_processing(df_validation_features)
    '''可视化：持有期IC、累计IC、IC衰减'''
    plot_validation_analysis(df_validation_features, ic_decay, test_period)
    '''可视化：Rank IC mean & ICIR mean (yearly)'''
    plot_validation_yearly_series_bar(df_validation_features)
    '''可视化：Rank IC mean & ICIR mean (monthly)'''
    plot_validation_monthly_series_bar(result_nested_dict_mean, result_nested_dict_icir)

if __name__ == "__main__":
    '''
            IC_Mean 衡量的是因子的预测强度。但是，单独看 IC_Mean 是有误导性的，必须结合 ICIR 看。
            绝对值 > 0.05：通常被认为是有意义的。
            绝对值 > 0.1：非常强的预测能力（这种因子很少见，通常很快会被市场消化）。
            ICIR 的绝对值大于 0.5 通常被视为具备一定预测能力，而大于 2 则被视为统计显著的优质因子
    '''
    run()











