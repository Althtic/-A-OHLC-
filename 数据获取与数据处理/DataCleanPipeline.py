import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any
from datetime import datetime
# ─────────────────────────────────────────────────────────────────────────────
# Pandas 显示设置
# ─────────────────────────────────────────────────────────────────────────────
# pd.set_option('display.max_rows', None)   # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)        # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None) # 列宽无限制（防止单元格内容被截断）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ─────────────────────────────────────────────────────────────────────────────
# 前复权处理
# ─────────────────────────────────────────────────────────────────────────────
def price_back_adj(hist_data, adj_data):
    adj_data = adj_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True])
    is_adjusting_data = hist_data.merge(adj_data, how='left', on=['ts_code', 'trade_date'])
    latest_factors = is_adjusting_data.groupby('ts_code')['adj_factor'].transform('last')
    # 调整后的前复权因子forward_factor
    is_adjusting_data['forward_factor'] = is_adjusting_data['adj_factor'] / latest_factors
    # 前复权：当前价格 * (最新因子/当前因子)
    is_adjusting_data['adj_close'] = is_adjusting_data['close'] * is_adjusting_data['forward_factor']
    is_adjusting_data['adj_open'] = is_adjusting_data['open'] * is_adjusting_data['forward_factor']
    is_adjusting_data['adj_low'] = is_adjusting_data['low'] * is_adjusting_data['forward_factor']
    is_adjusting_data['adj_high'] = is_adjusting_data['high'] * is_adjusting_data['forward_factor']
    is_adjusting_data['adj_pre_close'] = is_adjusting_data['pre_close'] * is_adjusting_data['forward_factor']
    # 成交额不做变动（交易的绝对金额没有改变），成交量vol应该与close价格做反向变动
    is_adjusting_data['adj_vol'] = is_adjusting_data['vol'] / is_adjusting_data['forward_factor']

    return_columns = ['ts_code', 'trade_date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_pre_close', 'adj_vol', 'adj_factor', 'amount', 'industry_name']
    available_columns = [col for col in return_columns if col in is_adjusting_data.columns]
    return_data = is_adjusting_data[available_columns]

    cols_to_round = ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_pre_close', 'adj_vol']
    cols_to_round = [col for col in cols_to_round if col in return_data.columns]
    return_data[cols_to_round] = return_data[cols_to_round].round(2) # 保留两位小数
    print(return_data)
    mapping = {
        'adj_open': 'open',
        'adj_high': 'high',
        'adj_low': 'low',
        'adj_close': 'close',
        'adj_pre_close': 'pre_close',
        'adj_vol': 'vol'
    }

    cols_to_rename = [col for col in mapping.keys() if col in return_data.columns]
    final_mapping = {col: mapping[col] for col in cols_to_rename}

    return_data.rename(columns=final_mapping, inplace=True)
    print(return_data)
    return return_data
# ─────────────────────────────────────────────────────────────────────────────
# 剔除风险警示板股票(*ST&ST)
# ─────────────────────────────────────────────────────────────────────────────
def ST_stock_id(hist_data, st_data):
    st_data = st_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True])
    st_data = st_data[['ts_code', 'trade_date', 'type']]
    ST_processing_data = hist_data.merge(st_data, how='left', on=['ts_code', 'trade_date'])
    ST_processed_data = ST_processing_data[~(ST_processing_data['type'] == 'ST')] # 保留不在风险警示板上的股票数据
    del ST_processed_data['type']
    return ST_processed_data
# ─────────────────────────────────────────────────────────────────────────────
# 剔除次新股上市数据（period=120）
# ─────────────────────────────────────────────────────────────────────────────
def remove_new_stock_initial_days(hist_data):
    df = hist_data.copy()
    df['trade_date'] = df['trade_date'].astype(int)
    start_time = df['trade_date'].iloc[0]
    end_time = df['trade_date'].iloc[-1]
    # 找出每只股票的首个交易日
    first_trade = df.groupby('ts_code')['trade_date'].min().reset_index()
    first_trade.columns = ['ts_code', 'first_date']
    df = df.merge(first_trade, on='ts_code', how='left')
    # 标记是否为需要清洗的新股
    df['is_new_stock'] = (df['first_date'] > start_time) & (df['first_date'] <= end_time)
    # 标记每只股票的第几个交易日
    df['trade_day_num'] = df.groupby('ts_code').cumcount() + 1
    # 执行删除逻辑:老股票（is_new_stock=False）全部保留,新股（is_new_stock=True）：只保留第120个交易日之后
    df_clean = df[(df['is_new_stock'] == False) | (df['trade_day_num'] > 120)].copy()
    df_clean = df_clean.drop(columns=['first_date', 'is_new_stock', 'trade_day_num']).reset_index(drop=True)
    print(f"    删除数据量: {len(df) - len(df_clean):,}")
    print(f"    删除比例: {(len(df) - len(df_clean)) / len(df) * 100:.2f}%")
    return df_clean
# ─────────────────────────────────────────────────────────────────────────────
# 日度收益率计算
# ─────────────────────────────────────────────────────────────────────────────
def calculate_dret(hist_data):
    hist_data['dret'] = hist_data.groupby('ts_code')['close'].transform(
        lambda x: ((x - x.shift(1)) / x.shift(1)).round(5)
    )
    # 剔除第一行因为没有前一天close导致dret为nan的数据
    ret_data = hist_data.dropna(subset=['dret'])
    return ret_data
# ─────────────────────────────────────────────────────────────────────────────
# 可成交日状态标记（停牌日、复牌日无成交、全天无成交）
# ─────────────────────────────────────────────────────────────────────────────
def is_trading_processing(hist_data, sus_data):
    # tushare 初始数据源中存在重复数据，需要提前去除后合并
    sus_data = sus_data.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
    is_trade_processing_data = hist_data.merge(sus_data, how='left', on=['ts_code', 'trade_date'])
    # 条件1：停牌（suspend_timing为NaN 且 suspend_type=='S'）注：suspend_timing不为NaN表示盘中停牌，正常状态下不妨碍正常交易
    is_suspended = (
            (is_trade_processing_data['suspend_timing'].isna()) &
            (is_trade_processing_data['suspend_type'] == 'S')
    )
    # 条件2：一字涨跌停(OHLC交易数据相等)
    is_one_word = (
            (is_trade_processing_data['dret'].notna()) &
            (is_trade_processing_data['open'] == is_trade_processing_data['close']) &
            (is_trade_processing_data['high'] == is_trade_processing_data['low']) &
            (is_trade_processing_data['open'] == is_trade_processing_data['high'])
    )
    conditions = [
        is_suspended,
        is_one_word
    ]
    choices = [-1, -2]
    # 可交易日赋值为 1
    is_trade_processing_data['is_trading'] = np.select(conditions, choices, default=1)
    print("    交易日赋值情况描述：停牌('-1'), 一字涨跌停('-2'), 正常可交易('1')")
    return is_trade_processing_data
# ─────────────────────────────────────────────────────────────────────────────
# 异常收益率处理
# ─────────────────────────────────────────────────────────────────────────────
def return_adjustment_trading_suspensions(hist_data):
    is_adjusting_abnormal_dret = hist_data.copy()
    chg_limits = {
        '北交所': 0.30,
        '科创板': 0.20,
        '创业板': 0.20,
        '沪深主板': 0.10,
        '其他': 0.30  # 默认30%
    }
    # 市场状态识别函数
    def identify_market(ts_code):
        ts_code = str(ts_code)
        if ts_code.startswith('8') or ts_code.startswith('9'):
            return '北交所'
        elif ts_code.startswith('688') or ts_code.startswith('689'):
            return '科创板'
        elif ts_code.startswith('300') or ts_code.startswith('301') or ts_code.startswith('302'):
            return '创业板'
        elif ts_code.startswith('60') or ts_code.startswith('000') or ts_code.startswith('001') or ts_code.startswith(
                '002') or ts_code.startswith('003'):
            return '沪深主板'
        else:
            return '其他'
    # 异常值赋NaN值，后续继续做填充处理
    # 关于对于复牌日股票直接赋值limit可能带来的虚假收益与虚假亏损(-40% -> -10%)，策略判断逻辑是对于'S'股票将会第一时间卖出，因此不涉及复牌后高低估的问题
    def truncate_chg(row):
        market = row['market']
        chg = row['dret']
        limit = chg_limits.get(market, 0.30)
        if abs(chg) > limit:
            return limit
        return chg
    is_adjusting_abnormal_dret['market'] = is_adjusting_abnormal_dret['ts_code'].astype(str).apply(identify_market)
    is_adjusting_abnormal_dret['dret'] = is_adjusting_abnormal_dret.apply(truncate_chg, axis=1)
    return is_adjusting_abnormal_dret
# ─────────────────────────────────────────────────────────────────────────────
# 股票按申万一级行业分类
# ─────────────────────────────────────────────────────────────────────────────
def industry_classification(hist_data, ind_data):
    stock_daily_data = hist_data.copy()
    ind_data.columns = ['trade_date', 'ts_code', 'industry_name']
    # 按交易日期映射
    trade_date_list = stock_daily_data['trade_date'].unique()
    all_result = []
    for date in trade_date_list:
        intrady_stock_data = stock_daily_data[stock_daily_data['trade_date'] == date]
        intrady_sorted_result = ind_data[ind_data['trade_date'] == date]
        industry_mapping = intrady_sorted_result[['ts_code', 'industry_name']]
        intrady_stock_data = intrady_stock_data.merge(
            industry_mapping,
            on='ts_code',
            how='left'  # 保留所有股票，未匹配的行业设为NaN
        )
        intrady_stock_data['industry_name'] = intrady_stock_data['industry_name'].fillna('未分类')
        all_result.append(intrady_stock_data)
    industry_classified_data = pd.concat(all_result).sort_values(by=['trade_date'])
    return industry_classified_data
# ─────────────────────────────────────────────────────────────────────────────
# 每日指标数据处理与合并(向前赋值ffill)
# ─────────────────────────────────────────────────────────────────────────────
def merge_daily_basic(hist_data, daily_data):
    daily_data = daily_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
    # 删除daily_basic中自带多余的close列
    daily_data.drop(columns=['close'], inplace=True)
    # print("向前填充前每列缺失值数量:")
    # print(daily_data.isnull().sum())
    cols_to_fill = daily_data.columns.difference(['ts_code', 'trade_date'])  # 排除非数值列，或者直接对全表操作
    daily_data[cols_to_fill] = daily_data.groupby('ts_code')[cols_to_fill].ffill()
    # print("填充后每列缺失值数量:")
    # print(daily_data.isnull().sum())
    # 仍然会存在NaN值，这是因为部分股票本身就不提供相关指标的数据
    daily_processed_data = hist_data.merge(daily_data, how='left', on=['ts_code', 'trade_date'])
    daily_processed_data = daily_processed_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
    return daily_processed_data
# ─────────────────────────────────────────────────────────────────────────────
# 数据排序
# ─────────────────────────────────────────────────────────────────────────────
def sort_value_data(hist_data):
    data = hist_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True])
    return data
# ─────────────────────────────────────────────────────────────────────────────
# 收益率直方图，检查是否存在离群值
# ─────────────────────────────────────────────────────────────────────────────
def distribution_plot(df, value_col='dret', market_col='market', save_path='markets_dret_distribution_plot.png'):
    markets = df[market_col].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, market in enumerate(markets):
        # 提取当前市场数据，替换无穷值为NaN
        data = df[df[market_col] == market][value_col].replace([np.inf, -np.inf], np.nan)
        axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[i].set_title(f'{market}\n(n={len(data):,})', fontsize=12)
        axes[i].set_xlabel(value_col)
        axes[i].set_ylabel('frequency')
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1)

    for j in range(len(markets), 4):
        axes[j].set_visible(False)

    plt.suptitle(f'{value_col} distribution - by markets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存至: {os.path.abspath(save_path)}")
    plt.close()
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 类封装
# ─────────────────────────────────────────────────────────────────────────────
class DataPipeline:

    def __init__(self, initial_data: pd.DataFrame, name: str = "StockDataPipeline"):
        self.data = initial_data.copy()
        self.name = name
        self.steps: List[Dict] = []
        self.start_time = datetime.now()

    def add_step(self, func: Callable, name: str = None, verbose: bool = True, **kwargs) -> 'DataPipeline':
        """添加处理步骤"""
        step_name = name or func.__name__
        prev_rows = len(self.data)

        try:
            self.data = func(self.data, **kwargs)
            curr_rows = len(self.data)

            self.steps.append({
                'step': step_name,
                'status': 'success',
                'prev_rows': prev_rows,
                'curr_rows': curr_rows,
                'delta': curr_rows - prev_rows
            })

            if verbose:
                delta_str = f"{curr_rows - prev_rows:+,}"
                print(f"✅ {step_name}: {prev_rows:,} → {curr_rows:,} ({delta_str})")

        except Exception as e:
            self.steps.append({
                'step': step_name,
                'status': 'failed',
                'error': str(e)
            })
            print(f"❌ {step_name} 失败：{e}")
            raise

        return self

    def summary(self) -> None:
        """打印执行摘要"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print(f"📊 Pipeline 执行摘要：{self.name}")
        print("=" * 70)
        print(f"{'步骤':<25} {'状态':<8} {'行数变化':>15}")
        print("-" * 70)

        for step in self.steps:
            if step['status'] == 'success':
                delta_str = f"{step['delta']:+,}"
                print(f"✅ {step['step']:<22} {'成功':<8} {delta_str:>15}")
            else:
                print(f"❌ {step['step']:<22} {'失败':<8} {step.get('error', ''):>15}")

        print("=" * 70)
        print(f"最终数据量：{len(self.data):,} 行")
        print(f"总耗时：{duration:.2f} 秒")
        print("=" * 70 + "\n")

    def get_data(self) -> pd.DataFrame:
        """获取处理后的数据"""
        return self.data

    def save(self, path: str, **kwargs) -> 'DataPipeline':
        """保存数据"""
        self.data.to_csv(path, index=False, **kwargs)
        print(f"💾 数据已保存至：{os.path.abspath(path)}")
        return self
# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 加载数据
    print("=" * 70)
    print("🚀 开始数据处理 Pipeline")
    print("=" * 70)

    history_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_ori.csv')
    adj_factor_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\adj_factor_ori.csv')
    st_stock_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\ST-Stocks_ori.csv')
    suspension_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\Suspension_data17-25_ori.csv')
    industry_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\SWlevel1_sorted_ori.csv')
    daily_basic_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\daily_basic_ori.csv')

    print(f"📁 历史数据：{len(history_data):,} 行")
    print(f"📁 复权因子数据：{len(adj_factor_data):,} 行")
    print(f"📁 ST 数据：{len(st_stock_data):,} 行")
    print(f"📁 停牌数据：{len(suspension_data):,} 行")
    print(f"📁 行业分类数据：{len(industry_data):,} 行")
    print(f"📁 每日指标数据：{len(daily_basic_data):,} 行")

    print("=" * 70 + "\n")

    # 构建 Pipeline
    pipeline = (DataPipeline(history_data, name="股票数据清洗 Pipeline")
                .add_step(price_back_adj, "OHLC 前复权处理",adj_data=adj_factor_data)
                .add_step(ST_stock_id, "ST 股票剔除", st_data=st_stock_data)
                .add_step(remove_new_stock_initial_days, "次新股交易数据剔除(120Days)")
                .add_step(calculate_dret, "日度收益率计算")
                .add_step(is_trading_processing, "停牌数据合并与可交易状态标记", sus_data=suspension_data)
                .add_step(return_adjustment_trading_suspensions, "异常收益率调整")
                .add_step(industry_classification, "股票行业分类(SW-Level1)", ind_data=industry_data)
                .add_step(merge_daily_basic, "每日指标数据处理与合并", daily_data=daily_basic_data)
                .add_step(sort_value_data, "数据排序后保存")
                )

    # 执行摘要
    pipeline.summary()

    # 可视化
    print("📈 生成收益率分布图...")
    distribution_plot(pipeline.get_data())

    # 保存结果
    pipeline.save('20170930-20251231_pipe.csv')

    # 最终验证
    df_final = pipeline.get_data()
    print("\n📋 最终数据概览:")
    print(f"   总行数：{len(df_final):,}")
    print(f"   总列数：{len(df_final.columns)}")
    print(f"   列名：{df_final.columns.tolist()}")



