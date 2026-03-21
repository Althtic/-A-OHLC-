import os
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac
from config_loader import traget_factor, layers, holding_period, test_window_start, test_window_end

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)
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

        return df_in_window

    except Exception as e:
        logger.info(f"处理时间窗口时发生错误: {e}")
        logger.info("--- 程序终止 ---")
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

    df_preprocess['holding_period_lndret'] = (
        df_preprocess.groupby('ts_code')['close']
        .transform(lambda x: np.log(x.shift(-holding_period) / x))
    )
    # 注意：这一步需要将持有期收益转化为日度持有期收益率，以防后续按日累加出错，导致高估收益率的情况
    df_preprocess['holding_lndret'] = df_preprocess['holding_period_lndret'] / holding_period
    df_preprocess = df_preprocess.dropna(subset=['holding_lndret'])
    return df_preprocess
# ─────────────────────────────────────────────────────────────────────────────
# 计算每层收益均值
# ─────────────────────────────────────────────────────────────────────────────
def process_group_by_date(group, layers):
    # 按中性化之后的因子值对当前分组进行排序(从小到大升序排列，负向信号在前group_0,正向信号在后group_5)
    sorted_group = group.sort_values(by=traget_factor, ascending=False).reset_index(drop=True)

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
# 计算换手率
# ─────────────────────────────────────────────────────────────────────────────
def calculate_turnover_rate(df_processed, layers, holding_period):
    trade_dates = sorted(df_processed['trade_date'].unique())
    rebalance_dates = trade_dates[::holding_period]
    if len(rebalance_dates) < 2:
        turnover_df = pd.DataFrame()
        nan = float('nan')
        group_turnovers = {q: nan for q in range(layers)}
        return turnover_df, nan, nan, nan, group_turnovers
    # 只保留所需的目标信息
    sub = df_processed.loc[
        df_processed['trade_date'].isin(rebalance_dates), ['trade_date', 'ts_code', 'quantile']
    ]
    # 将数据按日期拆解成字典：{日期: 该日期的 [代码, 分层] DataFrame}
    date_to_sub = {d: g[['ts_code', 'quantile']] for d, g in sub.groupby('trade_date', sort=False)}

    rows = []
    layer_turnover_sum = np.zeros(layers, dtype=float)
    n_pairs = 0
    q_idx = np.arange(layers, dtype=np.int16)

    for i in range(1, len(rebalance_dates)):
        curr_date = rebalance_dates[i]
        prev_date = rebalance_dates[i - 1]

         # 获取前后两天的持仓数据，如果某天缺失数据（极端情况），则初始化为空表
        sub_prev = date_to_sub.get(prev_date)
        sub_curr = date_to_sub.get(curr_date)

        if sub_prev is None:
            sub_prev = pd.DataFrame(columns=['ts_code', 'quantile'])
        if sub_curr is None:
            sub_curr = pd.DataFrame(columns=['ts_code', 'quantile'])
        
        # 【核心步骤】外连接 (Outer Join)
        # 将前一天的持仓和后一天的持仓通过 'ts_code' 合并。
        # 这样可以看到：
        # 1. 继续持有的股票 (两边都有)
        # 2. 新买入的股票 (只有当前有，前一边是 NaN)
        # 3. 卖出的股票 (只有前一天有，后一边是 NaN)
        merged = sub_prev.merge(sub_curr, on='ts_code', how='outer', suffixes=('_prev', '_curr'))
        pv = merged['quantile_prev']
        cv = merged['quantile_curr']
        n = len(merged)
        if n == 0:
            rows.append({'date': curr_date, 'turnover': 0.0, 'changes': 0, 'n_stocks': 0})
            n_pairs += 1
            continue
        # 分层改变 (pv != cv) OR 新买入 (pv is NA) OR 卖出 (cv is NA)
        changes = int(((pv != cv) | pv.isna() | cv.isna()).sum())
        turnover = changes / n
        rows.append({'date': curr_date, 'turnover': turnover, 'changes': changes, 'n_stocks': n})

        pv_arr = pv.astype('float64', copy=False).to_numpy()
        cv_arr = cv.astype('float64', copy=False).to_numpy()

        # 构建布尔矩阵：
        # prev_in[i, q] 为 True 表示第 i 只股票在昨天属于第 q 层
        # curr_in[i, q] 为 True 表示第 i 只股票在今天属于第 q 层
        # 利用广播机制 (np.newaxis) 进行比较       
        prev_in = pv_arr[:, np.newaxis] == q_idx
        curr_in = cv_arr[:, np.newaxis] == q_idx
        union_mask = prev_in | curr_in
        uni_counts = union_mask.sum(axis=0)
        sym_counts = np.logical_xor(prev_in, curr_in).sum(axis=0)
        valid = uni_counts > 0
        layer_turnover_sum += np.where(valid, sym_counts.astype(np.float64) / uni_counts, 0.0)
        n_pairs += 1

    turnover_df = pd.DataFrame(rows)
    if turnover_df.empty:
        nan = float('nan')
        group_turnovers = {q: nan for q in range(layers)}
        return turnover_df, nan, nan, nan, group_turnovers

    avg_turnover = turnover_df['turnover'].mean()
    std_turnover = turnover_df['turnover'].std()
    annual_turnover = avg_turnover * (252 / holding_period) if holding_period else float('nan')
    group_turnovers = {
        q: (layer_turnover_sum[q] / n_pairs) if n_pairs else float('nan')
        for q in range(layers)
    }

    return turnover_df, avg_turnover, std_turnover, annual_turnover, group_turnovers
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
        return {"t_stat": np.nan, "p_value": np.nan, "conclusion": "样本量过小"}
 
    mean_ret = np.mean(rets)

    lags = int(np.floor(4 * (T / 100)**(2/9)))
    lags = max(1, min(lags, T - 2)) 
    
    # 构建截距模型: Y = alpha + error, 这里的 alpha 就是均值 mean_ret
    X = np.ones((T, 1)) 
    model = sm.OLS(rets, X)
    results = model.fit()
    
    try:  # 显式传入 lags
        cov_matrix = cov_hac(results, nlags=lags)
    except Exception as e:
        logger.warning(f"HAC 计算失败，回退到普通标准误: {e}")
        se_nw = results.bse[0]
        lags = 0
    se_nw = np.sqrt(cov_matrix[0, 0]) if 'cov_matrix' in locals() else results.bse[0]
    
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
        conclusion = "多空组合收益率序列显著为正"
    else:
        conclusion = "多空组合收益率序列不显著为正"
    print(f"结论: {conclusion}")
    
    return {"t_stat": t_stat, "p_value": one_tailed_p_value, "conclusion": conclusion}
# ─────────────────────────────────────────────────────────────────────────────
# 分层回测结果绘图
# ───────────────────────────────────────────────────────────────────────────── 
def plot_multiple_return_metrics(dataframe, cumulative_returns, layers, target_factor, save_dir=None, t_test_result=None):
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
            logger.info(f"警告: 数据中不存在列 '{col_name}'")
    # 单独画多空收益曲线
    excess_col = 'sum_ret_L-S'
    if excess_col in dataframe.columns:
        ax2.plot(dataframe.index, dataframe[excess_col], label='L-S', color='blue', linestyle='--', linewidth=2)
    else:
        logger.info(f"警告: 数据中不存在列 '{excess_col}'")

    ax2.set_title(f"{target_factor}因子分组多空回测结果图", fontsize=14, fontweight='bold')  # 添加标题
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=45)

    if t_test_result:
        t_val = t_test_result['t_stat']
        t_str = f"{t_val:.4f}" if not np.isnan(t_val) else "N/A"
        txt = f"[NW Adjusted] T-statistic: {t_str}\nConclusion: {t_test_result['conclusion']}"
        fig.text(0.02, 0.02, txt, fontsize=12, verticalalignment='bottom', transform=fig.transFigure,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Quantile_Spread_Test.png'), dpi=150, bbox_inches='tight')
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 换手率结果绘图
# ───────────────────────────────────────────────────────────────────────────── 
def plot_turnover(turnover_df, group_turnovers, target_factor, layers, save_dir=None):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    if turnover_df is None or turnover_df.empty:
        axes[0].text(0.5, 0.5, '换手率样本不足（再平衡次数<2）', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].set_visible(False)
        plt.tight_layout()
        plt.show()
        return

    # 图1：换手率时序
    axes[0].plot(pd.to_datetime(turnover_df['date']), turnover_df['turnover'],
                 marker='o', linestyle='-', linewidth=1, markersize=4)
    m = turnover_df['turnover'].mean()
    axes[0].axhline(y=m, color='r',
                    linestyle='--', label=f"均值: {m:.3f}")
    axes[0].set_title(f'{target_factor} 因子组合换手率时序')
    axes[0].set_ylabel('换手率')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2：各分组换手率对比
    groups = list(group_turnovers.keys())
    turnovers = list(group_turnovers.values())
    colors = ['green' if i in [0, layers-1] else 'gray' for i in range(len(groups))]
    axes[1].bar(groups, turnovers, color=colors, alpha=0.7)
    axes[1].set_title('各分组平均换手率对比')
    axes[1].set_ylabel('换手率')
    axes[1].set_xlabel('分组')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Turnover_Rate_result_plot.png'), dpi=150, bbox_inches='tight')
    plt.show()

""" =======================主执行函数======================= """
def quantile_spread_test():
    save_dir = os.path.join(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\因子检验结果', traget_factor)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = data_loading(traget_factor)
    df = data_preprocessing(df, test_window_start, test_window_end, holding_period)
    logger.info('数据预处理完毕')

    '''按日期分层处理'''
    df_layers_processed = df.groupby('trade_date')[['trade_date', 'ts_code', traget_factor, 'holding_lndret']].apply(
        lambda group: process_group_by_date(group, layers)
    ).reset_index(drop=True)
    
    print(df_layers_processed)

 
    
    '''分层累计收益计算'''
    df_layers_cumulative_ret, spread_ret_series, final_cumulative_returns = spread_ret_cumsum_calculate(df_layers_processed, layers)

    logger.info('开始换手率分析...')
    
    # 1. 整体换手率
    turnover_df, avg_turnover, std_turnover, annual_turnover, group_turnovers = calculate_turnover_rate(df_layers_processed, layers, holding_period)
    
    if pd.notna(avg_turnover):
        logger.info(f'平均换手率: {avg_turnover:.4f} ({avg_turnover*100:.2f}%)')
        logger.info(f'换手率标准差: {std_turnover:.4f}')
    if pd.notna(annual_turnover):
        logger.info(f'年化换手率: {annual_turnover:.2f} 倍')


    logger.info('分层收益计算完毕')
    '''多空收益的t-检验'''
    t_test_result = t_test_spread_ret(spread_ret_series)
    '''结果绘图'''
    plot_multiple_return_metrics(df_layers_cumulative_ret, final_cumulative_returns, layers, traget_factor, save_dir, t_test_result)
    plot_turnover(turnover_df, group_turnovers, traget_factor, layers, save_dir)
if __name__ == "__main__":
    quantile_spread_test()










