import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
from sklearn.linear_model import LinearRegression
from factor_validation_config_loader import traget_factor, layers, holding_period, test_window_start, test_window_end

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────
def data_loading(traget_factor):
    print(f"开始分析因子: {traget_factor}")
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    # 读取市值数据，以便后续市值中性化
    cv_df = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
                        usecols=['trade_date', 'ts_code', 'circ_mv'])
    df_merge = df_loading.merge(cv_df,on=['trade_date','ts_code'],how='left')
    return df_merge
# ─────────────────────────────────────────────────────────────────────────────
# 筛选时间窗口
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
        print(f"原始数据的时间范围: {original_min_date.strftime('%Y%m%d')} 到 {original_max_date.strftime('%Y%m%d')}")

        # 将输入的时间也转换为 datetime 进行比较
        input_start_dt = pd.to_datetime(str(start_time), format='%Y%m%d')
        input_end_dt = pd.to_datetime(str(end_time), format='%Y%m%d')

        # 判断输入窗口是否超出原始数据范围 ---
        if input_start_dt > original_max_date or input_end_dt < original_min_date:
            print(f"错误: 请求的时间窗口 [{start_time}, {end_time}] 完全超出了原始数据的日期范围 [{original_min_date.strftime('%Y%m%d')}, {original_max_date.strftime('%Y%m%d')}]。")
            print("--- 程序终止 ---")

            return None # 返回 None 表示操作失败

        if input_start_dt < original_min_date or input_end_dt > original_max_date:
             # 检查是否有部分超出，如果有，也视为超出范围
             effective_start = max(input_start_dt, original_min_date)
             effective_end = min(input_end_dt, original_max_date)
             print(f"警告: 请求的时间窗口 [{start_time}, {end_time}] 部分超出了原始数据范围。")
             print(f"     建议使用有效范围: [{effective_start.strftime('%Y%m%d')}, {effective_end.strftime('%Y%m%d')}]")

             return None # 返回 None 表示操作失败

        # 如果时间窗口在范围内，进行筛选
        # 原始数据 'trade_date' 列转化为字符串 string (用于后续筛选)
        df_copy = df.copy()
        if df_copy['trade_date'].dtype in ['int64', 'int32']:
            df_copy['trade_date'] = df_copy['trade_date'].astype(str)

        # 创建布尔掩码并筛选
        mask = (df_copy['trade_date'] >= start_time) & (df_copy['trade_date'] <= end_time)
        df_in_window = df_copy.loc[mask]

        return df_in_window

    except Exception as e:
        print(f"处理时间窗口时发生错误: {e}")
        print("--- 程序终止 ---")
        return None # 返回 None 表示操作失败
# ─────────────────────────────────────────────────────────────────────────────
# 持有期内累计对数收益率计算
# ─────────────────────────────────────────────────────────────────────────────
def data_preprocessing(df,test_window_start,test_window_end, holding_period):
    df = df.copy()
    # 截取回测期间的历史数据
    df_preprocess = cut_time_window(df, test_window_start, test_window_end)
    # 今日对数收益率计算
    df_preprocess['lndret'] = np.log(df_preprocess['close'] / df_preprocess['pre_close'])

    df_preprocess['holding_lndret'] = (
        df_preprocess.groupby('ts_code')['close']
        .transform(lambda x: np.log(x.shift(-holding_period) / x))
    )
    df_preprocess = df_preprocess.dropna(subset=['holding_lndret'])
    return df_preprocess
# ─────────────────────────────────────────────────────────────────────────────
# 市值/行业中性化处理
# ─────────────────────────────────────────────────────────────────────────────
def neutralize_factor_by_date(group, factor_col='alpha_15', cap_col='circ_mv', industry_col='industry_name'):
    """
    Factor ~ ln(MarketCap) + Industry_Dummies，返回残差作为中性化后的因子值
    """
    group = group.replace([np.inf, -np.inf], np.nan)
    valid_mask = group[factor_col].notna() & group[cap_col].notna() & group[industry_col].notna()

    if valid_mask.sum() < 10:
        group['factor_neutralized'] = np.nan
        return group
    data_valid = group[valid_mask].copy()

    y = data_valid[factor_col].values
    log_cap = np.log(data_valid[cap_col]).values.reshape(-1, 1)

    # 处理自变量 X2: 行业哑变量
    industry_series = data_valid[industry_col].fillna('Unknown')
    # 生成哑变量矩阵（True & False）
    industry_dummies = pd.get_dummies(industry_series, prefix='ind', dummy_na=False)
    # 合并 X (市值 + 行业)将 log_cap 和 dummies 的 values 拼在一起,使用 np.hstack 效率更高，
    X = np.hstack([log_cap, industry_dummies.values])

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    # 提取残差
    group['factor_neutralized'] = np.nan
    group.loc[valid_mask, 'factor_neutralized'] = residuals
    return group
# ─────────────────────────────────────────────────────────────────────────────
# 计算每层收益均值
# ─────────────────────────────────────────────────────────────────────────────
def process_group_by_date(group, layers):
    # 按中性化之后的因子值对当前分组进行排序(从小到大升序排列，负向信号在前group_0,正向信号在后group_5)
    sorted_group = group.sort_values(by='factor_neutralized', ascending=False).reset_index(drop=True)

    n = len(sorted_group)
    # 计算基础组大小和余数（按索引分组，防止.qcut()面对大量重复值时的无效分组问题）
    base_size = n // layers  # 每组的基本大小
    remainder = n % layers  # 无法整除后剩下的样本数

    quantiles = np.empty(n, dtype=int)
    current_idx = 0
    for i in range(layers):
        size_for_this_group = base_size + (1 if i < remainder else 0)
        quantiles[current_idx: current_idx + size_for_this_group] = i
        current_idx += size_for_this_group

    # 计算出的分组结果赋值给 DataFrame
    sorted_group['quantile'] = quantiles
    # 计算每组的平均收益率
    sorted_group['mean_lndret'] = sorted_group.groupby('quantile')['holding_lndret'].transform('mean')

    return sorted_group
# ─────────────────────────────────────────────────────────────────────────────
# group累计对数收益计算
# ─────────────────────────────────────────────────────────────────────────────
def spread_ret_cumsum_calculate(data, layers):
    # pivot_table数据透视
    Spread_ret = data.pivot_table(
        index='trade_date',
        columns='quantile',
        values='mean_lndret',
        aggfunc='mean' # 数值都一样，取第一个就好
    ).reset_index()
    # 多空收益计算
    Spread_ret['L-S'] = Spread_ret.iloc[:, 1] - Spread_ret.iloc[:, -1]  # group_0 - group_last
    # L-S多空收益率序列,用于后续的t检验
    t_test_series = np.array(Spread_ret['L-S'])
    # 计算各组累计收益 (只计算存在的列)
    quantiles = list(range(layers)) # e.g  layers=5,quantiles=[0,1,2,3,4]
    for q in quantiles:
        if q in Spread_ret.columns:  # 检查列是否存在
            Spread_ret[f'sum_ret_{q}'] = Spread_ret[q].cumsum()
            # print(f"已计算分位数 {q} 的累计收益")
        else:
            print(f"警告: 数据中缺少分位数 {q} 的数据")
    # 分层累计收益
    Spread_ret['sum_ret_L-S'] = Spread_ret['L-S'].cumsum()
    # 提取各组最后一天的总累计收益，存入字典
    final_cumulative_returns = {}
    for q in quantiles:
        sum_ret_col_name = f'sum_ret_{q}'
        if sum_ret_col_name in Spread_ret.columns:
            # 取该列的最后一行的值，即最后一天的累计收益
            final_cumulative_returns[q] = Spread_ret[sum_ret_col_name].iloc[-1]
        else:
            print(f"警告: 未能找到累计收益列 '{sum_ret_col_name}' 来提取最终收益。")

    return Spread_ret, t_test_series, final_cumulative_returns
# ─────────────────────────────────────────────────────────────────────────────
# L-S收益率的t检验（Newey-West调整）
# ─────────────────────────────────────────────────────────────────────────────
def t_test_spread_ret(Series):
    """
    对多空收益序列进行 Newey-West 调整的 t 检验 (单尾)
    使用 statsmodels 计算 HAC 标准误以确保精度
    """
    # 数据预处理
    if hasattr(Series, 'values'):
        rets = Series.values
    else:
        rets = np.array(Series)
    # 去除 NaN 值
    rets = rets[~np.isnan(rets)]
    T = len(rets)
    if T < 5:
        print("样本量过小，无法进行 Newey-West 调整。")
        return False

    mean_ret = np.mean(rets)
    
    # 构建截距模型: Y = alpha + error
    # 这里的 alpha 就是均值 mean_ret
    X = np.ones((T, 1)) 
    model = sm.OLS(rets, X)
    results = model.fit()
    
    # 调用 cov_hac 计算 Newey-West 协方差矩阵，nlags=None 表示让库函数自动根据公式选择滞后阶数
    # 库函数内部会自动应用小样本修正因子 (n / (n-k))
    cov_matrix = cov_hac(results, nlags=None)
    
    # 提取标准误 (协方差矩阵对角线元素的平方根)
    # 因为是单变量回归，cov_matrix 是 1x1
    se_nw = np.sqrt(cov_matrix[0, 0])
    
    # 查看库函数自动选择的滞后阶数，cov_hac 不直接返回使用的 lags，但逻辑与 floor(4*(T/100)^(2/9)) 一致
    # lags_used = int(np.floor(4 * (T / 100)**(2/9)))
    # lags_used = max(1, min(lags_used, T - 2))

    # 计算统计量
    if se_nw < 1e-10:
        t_stat = 0.0
    else:
        t_stat = mean_ret / se_nw
    # 计算双尾 p 值
    two_tailed_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=T-1))
    # 单尾检验逻辑
    if t_stat > 0:
        one_tailed_p_value = two_tailed_p_value / 2
    else:
        one_tailed_p_value = 1.0

    # print(f"[NW Adjusted] Lags used: {lags_used}, T: {T}")
    print(f"[NW Adjusted] T-statistic: {t_stat:.4f}")
    print(f"[NW Adjusted] P-value (one-tailed): {one_tailed_p_value:.4f}")
    
    alpha = 0.05
    if one_tailed_p_value < alpha and t_stat > 0:
        conclusion = "多空组合收益率序列显著为正 (Newey-West 调整)"
    else:
        conclusion = "多空组合收益率序列不显著为正 (Newey-West 调整)"
    print(f"结论: {conclusion}")
    
    return one_tailed_p_value < 0.05
# ─────────────────────────────────────────────────────────────────────────────
# 分层回测结果绘图
# ───────────────────────────────────────────────────────────────────────────── 
def plot_multiple_return_metrics(dataframe, cumulative_returns, layers, target_factor):
    # 设置中文字体以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 创建2*1的布局，即上下两部分
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    # 格式转换YYYY-MM-DD
    if not np.issubdtype(dataframe['trade_date'].dtype, np.datetime64):
        date_col = pd.to_datetime(dataframe['trade_date'].astype(str), format='%Y%m%d')
    else:
        date_col = dataframe['trade_date']

    colors = ['#228B22', '#32CD32', '#7CFC00', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6A6A','#DC143C']
    # 上方柱状图
    labels = []
    bars = []
    for idx, (key, value) in enumerate(cumulative_returns.items()):
        if key == 'L-S':  # 跳过多空组合
            continue
        labels.append(f'group_{key}')
        color_idx = int(key) % len(colors)
        bar = ax1.bar(labels[-1], value, color=colors[color_idx])
        bars.append(bar)
        # 在柱状图上方添加文本注释显示累计收益值
        ax1.text(bar[0].get_x() + bar[0].get_width() / 2,
                 bar[0].get_height(),
                 f'{value:.4f}',  # 格式化为小数点后两位
                 ha='center', va='bottom')  # 水平居中对齐，垂直底部对齐
    ax1.set_title('各组最终累计收益', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)  # 添加横向grid

    # 下方折线图
    n = len(dataframe) - 1  # 减1是为了作为索引
    indices = np.linspace(0, n, 8, dtype=int)
    xtick_labels = date_col.iloc[indices].dt.strftime('%Y%m%d')
    xtick_positions = indices  # x轴位置对应这些索引
    # 动态绘制每条线
    for q in range(layers):
        col_name = f'sum_ret_{q}'
        if col_name in dataframe.columns:
            color_idx = q % len(colors)
            ax2.plot(dataframe.index, dataframe[col_name], label=f'group_{q}', color=colors[color_idx], linewidth=1)
        else:
            print(f"警告: 数据中不存在列 '{col_name}'")
    # 单独画多空收益曲线
    excess_col = 'sum_ret_L-S'
    if excess_col in dataframe.columns:
        ax2.plot(dataframe.index, dataframe[excess_col], label='L-S', color='blue', linestyle='--', linewidth=2)
    else:
        print(f"警告: 数据中不存在列 '{excess_col}'")

    ax2.set_title(f"{target_factor}因子分组多空回测结果图", fontsize=14, fontweight='bold')  # 添加标题
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=45)

    plt.tight_layout()
    plt.show()

""" =======================主执行函数======================= """
def run():
    df = data_loading(traget_factor)
    df = data_preprocessing(df, test_window_start, test_window_end, holding_period)
    print('数据预处理完毕')
    '''中性化处理（市值 & 行业）：回归残差法'''
    df = df.groupby('trade_date', group_keys=False).apply(neutralize_factor_by_date, factor_col=traget_factor)
    print('中性化处理完毕（行业 + 市值）')
    '''按日期分层处理'''
    df_layers_processed = df.groupby('trade_date')[['trade_date', 'ts_code', 'factor_neutralized', 'holding_lndret']].apply(
        lambda group: process_group_by_date(group, layers)
    ).reset_index(drop=True)
    '''分层累计收益计算'''
    print('正在计算分层收益')
    df_layers_cumulative_ret, spread_ret_series, final_cumulative_returns = spread_ret_cumsum_calculate(
        df_layers_processed, layers)
    print('分层收益计算完毕')
    print('正在计算Newey—West调整后的 t 统计量')
    '''多空收益的t-检验'''
    t_test_spread_ret(spread_ret_series)
    print('Newey—West调整后的 t 统计量计算完成')
    '''结果绘图'''
    plot_multiple_return_metrics(df_layers_cumulative_ret, final_cumulative_returns, layers, traget_factor)

if __name__ == "__main__":
    run()










