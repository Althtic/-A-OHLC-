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
def _vw_mean_by_quantile(sorted_group):
    vw_map = {}
    for q, g in sorted_group.groupby('quantile', observed=True):
        w = g['circ_mv'].to_numpy(dtype=np.float64)
        r = g['holding_lndret'].to_numpy(dtype=np.float64)
        # 只保留市值和收益率都非空且市值大于0的股票
        valid = np.isfinite(w) & (w > 0) & np.isfinite(r)
        # 计算加权平均收益率
        vw_map[q] = np.nan if not valid.any() else float(np.dot(w[valid], r[valid])) / np.sum(w[valid])
    return sorted_group['quantile'].map(vw_map)

def process_group_by_date(group, layers):
    sorted_group = group.sort_values(by=traget_factor, ascending=False).reset_index(drop=True)

    n = len(sorted_group)
    base_size = n // layers
    remainder = n % layers

    quantiles = np.empty(n, dtype=int)
    current_idx = 0
    for i in range(layers):
        size_for_this_group = base_size + (1 if i < remainder else 0)
        quantiles[current_idx: current_idx + size_for_this_group] = i
        current_idx += size_for_this_group

    sorted_group['quantile'] = quantiles
    sorted_group['mean_lndret'] = sorted_group.groupby('quantile')['holding_lndret'].transform('mean')
    sorted_group['mean_lndret_vw'] = _vw_mean_by_quantile(sorted_group)

    return sorted_group
# ─────────────────────────────────────────────────────────────────────────────
# group累计对数收益计算
# ─────────────────────────────────────────────────────────────────────────────
def spread_ret_cumsum_calculate(data, layers, ret_col='mean_lndret', col_suffix='_ew'):
    Spread_ret = data.pivot_table(
        index='trade_date',
        columns='quantile',
        values=ret_col,
        aggfunc='mean'
    ).reset_index()
    hi, lo = 0, layers - 1
    if hi not in Spread_ret.columns or lo not in Spread_ret.columns:
        raise KeyError(f'分层列缺失: 需要 quantile {hi} 与 {lo}')
    Spread_ret['L-S'] = Spread_ret[hi] - Spread_ret[lo]
    t_test_series = np.array(Spread_ret['L-S'])
    quantiles = list(range(layers))
    suf = col_suffix
    for q in quantiles:
        if q in Spread_ret.columns:
            Spread_ret[f'sum_ret_{q}{suf}'] = Spread_ret[q].cumsum()
        else:
            print(f"警告: 数据中缺少分位数 {q} 的数据")
    Spread_ret[f'sum_ret_L-S{suf}'] = Spread_ret['L-S'].cumsum()
    final_cumulative_returns = {}
    for q in quantiles:
        sum_ret_col_name = f'sum_ret_{q}{suf}'
        if sum_ret_col_name in Spread_ret.columns:
            final_cumulative_returns[q] = Spread_ret[sum_ret_col_name].iloc[-1]
        else:
            print(f"警告: 未能找到累计收益列 '{sum_ret_col_name}' 来提取最终收益。")
    if 'L-S' not in final_cumulative_returns:
        final_cumulative_returns['L-S'] = Spread_ret[f'sum_ret_L-S{suf}'].iloc[-1]

    return Spread_ret, t_test_series, final_cumulative_returns
# ─────────────────────────────────────────────────────────────────────────────
# 单边换手率：0.5 * Σ|w_t - w_{t-1}|（多头暴露合计+0.5、空头暴露合计-0.5）
# ─────────────────────────────────────────────────────────────────────────────
def _one_way_turnover(w_prev: dict, w_curr: dict) -> float:
    # 双边金额 Σ|Δw| 中买、卖各算一遍；单边换手率取一半，便于直接 × 单边费率估成本
    # 合并调仓前与调仓后所有涉及的股票
    keys = set(w_prev.keys()) | set(w_curr.keys())
    if not keys:
        return 0.0
    s = 0.0
    for k in keys:
        # 计算每只股票的权重变化绝对值后累加
        s += abs(w_curr.get(k, 0.0) - w_prev.get(k, 0.0))
    return 0.5 * s # 单边换手率取一半，便于直接 × 单边费率估成本


def _ls_weights_ew(sub: pd.DataFrame, layers: int) -> dict:
    q0 = sub.loc[sub['quantile'] == 0, 'ts_code']
    qs = sub.loc[sub['quantile'] == layers - 1, 'ts_code']
    n0, ns = len(q0), len(qs)
    if n0 == 0 or ns == 0:
        return {}
    w = {}
    # 多头组每只股票头寸大小（总多头暴露: +0.5）
    hl = 0.5 / n0
    # 空头组每只股票头寸大小（总空头暴露: -0.5）
    hs = 0.5 / ns
    # 多头组合正权重（每只股票 +hl）
    for c in q0:
        w[str(c)] = w.get(str(c), 0.0) + hl
    # 空头组合负权重（每只股票 -hs）
    for c in qs:
        w[str(c)] = w.get(str(c), 0.0) - hs
    return w


def _ls_weights_vw(sub: pd.DataFrame, layers: int) -> dict:
    q0 = sub[sub['quantile'] == 0]
    qs = sub[sub['quantile'] == layers - 1]
    mv0 = q0['circ_mv'].to_numpy(dtype=np.float64)
    mvs = qs['circ_mv'].to_numpy(dtype=np.float64)
    c0 = q0['ts_code'].astype(str).to_numpy()
    cs = qs['ts_code'].astype(str).to_numpy()
    # 布尔掩码：市值非空且大于0
    m0 = np.isfinite(mv0) & (mv0 > 0)
    ms = np.isfinite(mvs) & (mvs > 0)
    # 计算总市值（多头组和空头组）
    s0 = mv0[m0].sum()
    ss = mvs[ms].sum()
    if s0 <= 0 or ss <= 0:
        return {}
    w = {}
    # 多头/空头各自在层内按 circ_mv 归一化后再 ×0.5，总暴露仍为 +0.5 / -0.5
    for i in np.flatnonzero(m0): # np.flatnonzero() 只返回市值非空且大于0的股票的索引
        # 最终头寸 = 当前头寸 + 0.5 * 当前市值 / 总市值
        w[c0[i]] = w.get(c0[i], 0.0) + 0.5 * float(mv0[i]) / s0
    for i in np.flatnonzero(ms):
        w[cs[i]] = w.get(cs[i], 0.0) - 0.5 * float(mvs[i]) / ss
    return w

# 独立分析每层的换手情况
def _layer_weights_ew(sub: pd.DataFrame, q: int) -> dict:
    # 仅用于「单层、纯多头、等权」口径下的单边换手（柱状图）；权重和为 1
    g = sub[sub['quantile'] == q]
    n = len(g)
    if n == 0:
        return {}
    hl = 1.0 / n # 计算等权权重
    return {str(r['ts_code']): hl for _, r in g.iterrows()}


def calculate_turnover_rate(df_processed, layers, holding_period):
    trade_dates = sorted(df_processed['trade_date'].unique())
    # 每隔 holding_period 个交易日调仓一次，只在调仓日用当日截面算权重
    rebalance_dates = trade_dates[::holding_period]
    nan = float('nan')
    if len(rebalance_dates) < 2:
        turnover_df = pd.DataFrame()
        group_turnovers = {q: nan for q in range(layers)}
        return turnover_df, nan, nan, nan, nan, nan, nan, nan, group_turnovers

    cols = ['trade_date', 'ts_code', 'quantile', 'circ_mv']
    sub = df_processed.loc[df_processed['trade_date'].isin(rebalance_dates), cols]
    # 将调仓日数据按日期分组，方便后续计算单边换手率
    date_to_sub = {d: g for d, g in sub.groupby('trade_date', sort=False)}


    rows = []
    layer_turnover_sum = np.zeros(layers, dtype=float)
    n_pairs = 0

    w_prev_ew = {}
    w_prev_vw = {}
    layer_prev = {q: {} for q in range(layers)}

    for i in range(1, len(rebalance_dates)):
        curr_date = rebalance_dates[i]
        prev_date = rebalance_dates[i - 1]
        sub_prev = date_to_sub.get(prev_date)
        sub_curr = date_to_sub.get(curr_date)
        if sub_prev is None:
            sub_prev = pd.DataFrame(columns=cols)
        if sub_curr is None:
            sub_curr = pd.DataFrame(columns=cols)

        # 当前调仓日截面构建组合，与上一调仓日权重比单边换手（首轮上一档为空 → 相当于建仓换手）
        w_curr_ew = _ls_weights_ew(sub_curr, layers)
        w_curr_vw = _ls_weights_vw(sub_curr, layers)

        # 计算单边换手率
        tow_ew = _one_way_turnover(w_prev_ew, w_curr_ew)
        tow_vw = _one_way_turnover(w_prev_vw, w_curr_vw)
        w_prev_ew, w_prev_vw = w_curr_ew, w_curr_vw

        for q in range(layers):
            lp = layer_prev[q]
            lc = _layer_weights_ew(sub_curr, q)
            layer_turnover_sum[q] += _one_way_turnover(lp, lc)  # 各层独立累计，最后再对调仓次数取平均
            layer_prev[q] = lc

        rows.append({
            'date': curr_date,
            'turnover_ls_ew': tow_ew,
            'turnover_ls_vw': tow_vw,
        })
        n_pairs += 1

    turnover_df = pd.DataFrame(rows)
    if turnover_df.empty:
        group_turnovers = {q: nan for q in range(layers)}
        return turnover_df, nan, nan, nan, nan, nan, nan, nan, group_turnovers

    avg_ew = turnover_df['turnover_ls_ew'].mean()
    std_ew = turnover_df['turnover_ls_ew'].std()
    avg_vw = turnover_df['turnover_ls_vw'].mean()
    std_vw = turnover_df['turnover_ls_vw'].std()
    # 每年约 252 个交易日，每 H 日调仓一次 → 年化换手指「每期均单边换手 × 每年期数」
    ann_ew = avg_ew * (252 / holding_period) if holding_period else nan
    ann_vw = avg_vw * (252 / holding_period) if holding_period else nan
    group_turnovers = {
        q: (layer_turnover_sum[q] / n_pairs) if n_pairs else nan
        for q in range(layers)
    }
    if pd.notna(avg_ew):
        logger.info(f'多空-等权 平均单边换手率: {avg_ew:.4f}  std: {std_ew:.4f}  年化(×252/H): {ann_ew:.2f}')
    if pd.notna(avg_vw):
        logger.info(f'多空-市值加权 平均单边换手率: {avg_vw:.4f}  std: {std_vw:.4f}  年化(×252/H): {ann_vw:.2f}')

    return turnover_df, avg_ew, std_ew, ann_ew, avg_vw, std_vw, ann_vw, group_turnovers
# ─────────────────────────────────────────────────────────────────────────────
# L-S收益率的t检验（Newey-West调整）
# ─────────────────────────────────────────────────────────────────────────────
def t_test_spread_ret(Series, run_label=''):
    """
    对多空收益序列进行 Newey-West 调整的 t 检验 (单尾)
    使用 statsmodels 计算 HAC 标准误以确保精度
    """
    prefix = f'[{run_label}] ' if run_label else ''
    # 数据预处理
    if hasattr(Series, 'values'):
        rets = Series.values
    else:
        rets = np.array(Series)
    # 去除 NaN 值
    rets = rets[~np.isnan(rets)]
    T = len(rets)
    if T < 5:
        print(f"{prefix}样本量过小，无法进行 Newey-West 调整。")
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
    print(f"{prefix}[NW Adjusted] T-statistic: {t_stat:.4f}")
    print(f"{prefix}[NW Adjusted] P-value (one-tailed): {one_tailed_p_value:.4f}")
    
    alpha = 0.05
    if one_tailed_p_value < alpha and t_stat > 0:
        conclusion = "多空组合收益率序列显著为正"
    else:
        conclusion = "多空组合收益率序列不显著为正"
    print(f"{prefix}结论: {conclusion}")
    
    return {"t_stat": t_stat, "p_value": one_tailed_p_value, "conclusion": conclusion}
# ─────────────────────────────────────────────────────────────────────────────
# 分层回测结果绘图（上：等权，下：市值加权）
# ─────────────────────────────────────────────────────────────────────────────
def plot_multiple_return_metrics(
    df_ew, cum_ew, df_vw, cum_vw, layers, target_factor,
    save_dir=None, t_test_ew=None, t_test_vw=None, suffix_ew='_ew', suffix_vw='_vw'):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    ax1, ax2, ax3, ax4 = axes

    ref = df_ew
    if not np.issubdtype(ref['trade_date'].dtype, np.datetime64):
        date_col = pd.to_datetime(ref['trade_date'].astype(str), format='%Y%m%d')
    else:
        date_col = ref['trade_date']

    colors = ['#228B22', '#32CD32', '#7CFC00', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6A6A', '#DC143C']
    n = max(len(ref) - 1, 0)
    k = min(8, max(len(ref), 1))
    indices = np.linspace(0, n, k, dtype=int)
    xtick_labels = date_col.iloc[indices].dt.strftime('%Y%m%d')
    xtick_positions = indices

    def _bars(ax, cumulative_returns):
        for key, value in cumulative_returns.items():
            if key == 'L-S':
                continue
            color_idx = int(key) % len(colors)
            bar = ax.bar(f'group_{key}', value, color=colors[color_idx])
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), f'{value:.4f}', ha='center', va='bottom')
        ax.set_ylabel('Cumulative Return', fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, axis='y', alpha=0.3)

    def _lines(ax, dataframe, suf, title):
        for q in range(layers):
            col_name = f'sum_ret_{q}{suf}'
            if col_name in dataframe.columns:
                ax.plot(dataframe.index, dataframe[col_name], label=f'group_{q}', color=colors[q % len(colors)], linewidth=1)
        excess_col = f'sum_ret_L-S{suf}'
        if excess_col in dataframe.columns:
            ax.plot(dataframe.index, dataframe[excess_col], label='L-S', color='blue', linestyle='--', linewidth=2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Return', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45)

    _bars(ax1, cum_ew)
    ax1.set_title('等权重：各组最终累计收益', fontsize=13, fontweight='bold')
    _lines(ax2, df_ew, suffix_ew, f'{target_factor} 等权重分层多空累计收益')

    _bars(ax3, cum_vw)
    ax3.set_title('市值加权：各组最终累计收益', fontsize=13, fontweight='bold')
    _lines(ax4, df_vw, suffix_vw, f'{target_factor} 市值加权分层多空累计收益')

    bbox_kw = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    if t_test_ew:
        te = t_test_ew['t_stat']
        te_str = f'{te:.4f}' if not np.isnan(te) else 'N/A'
        ax2.text(
            0.02, 0.02, f'等权 L-S [NW] T={te_str}\n{t_test_ew["conclusion"]}',
            transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', bbox=bbox_kw)
    if t_test_vw:
        tv = t_test_vw['t_stat']
        tv_str = f'{tv:.4f}' if not np.isnan(tv) else 'N/A'
        ax4.text(
            0.02, 0.02, f'市值加权 L-S [NW] T={tv_str}\n{t_test_vw["conclusion"]}',
            transform=ax4.transAxes, fontsize=10, verticalalignment='bottom', bbox=bbox_kw)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Quantile_Spread_result_plot.png'), dpi=150, bbox_inches='tight')
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

    dt = pd.to_datetime(turnover_df['date'])
    axes[0].plot(dt, turnover_df['turnover_ls_ew'], marker='o', linestyle='-', linewidth=1, markersize=3, label='多空-等权')
    axes[0].plot(dt, turnover_df['turnover_ls_vw'], marker='s', linestyle='-', linewidth=1, markersize=3, label='多空-市值加权')
    m_ew = turnover_df['turnover_ls_ew'].mean()
    m_vw = turnover_df['turnover_ls_vw'].mean()
    axes[0].axhline(y=m_ew, color='r', linestyle='--', alpha=0.7, label=f'等权均值: {m_ew:.3f}')
    axes[0].axhline(y=m_vw, color='b', linestyle='--', alpha=0.7, label=f'市值加权均值: {m_vw:.3f}')
    axes[0].set_title(f'{target_factor} 多空组合单边换手率时序 (0.5×Σ|Δw|)')
    axes[0].set_ylabel('单边换手率')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    groups = list(group_turnovers.keys())
    turnovers = list(group_turnovers.values())
    colors = ['green' if i in [0, layers-1] else 'gray' for i in range(len(groups))]
    axes[1].bar(groups, turnovers, color=colors, alpha=0.7)
    axes[1].set_title('各分组单边换手率（层内等权、全多头口径）')
    axes[1].set_ylabel('单边换手率')
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
    if 'circ_mv' not in df.columns:
        raise ValueError("因子数据文件需包含列 circ_mv（当日流通市值）")

    df = data_preprocessing(df, test_window_start, test_window_end, holding_period)
    logger.info('数据预处理完毕')
    layer_cols = ['trade_date', 'ts_code', traget_factor, 'holding_lndret', 'circ_mv']
    df_layers_processed = (
        df[layer_cols]
        .groupby('trade_date', group_keys=False)
        .apply(lambda group: process_group_by_date(group, layers))
        .reset_index(drop=True)
    )
    df_ew, spread_ew, cum_ew = spread_ret_cumsum_calculate(df_layers_processed, layers, 'mean_lndret', '_ew')
    df_vw, spread_vw, cum_vw = spread_ret_cumsum_calculate(df_layers_processed, layers, 'mean_lndret_vw', '_vw')

    turnover_df, avg_ew, std_ew, ann_ew, avg_vw, std_vw, ann_vw, group_turnovers = calculate_turnover_rate(df_layers_processed, layers, holding_period)

    logger.info('分层收益计算完毕')
    t_ew = t_test_spread_ret(spread_ew, run_label='等权L-S')
    t_vw = t_test_spread_ret(spread_vw, run_label='市值加权L-S')
    plot_multiple_return_metrics(df_ew, cum_ew, df_vw, cum_vw, layers, traget_factor, save_dir, t_ew, t_vw, '_ew', '_vw')
    plot_turnover(turnover_df, group_turnovers, traget_factor, layers, save_dir)
    
if __name__ == "__main__":
    quantile_spread_test()










