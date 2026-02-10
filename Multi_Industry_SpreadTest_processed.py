import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）
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
def data_preprocessing(traget_factor,test_window_start,test_window_end):
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df = pd.read_csv(Input_path)

    df, Stock_industry = calculate_lndret_exchange_identity(df)
    df_preprocess = cut_time_window(df, test_window_start, test_window_end)
    return df_preprocess, Stock_industry
def calculate_lndret_exchange_identity(orginal_data):
    # 对数收益率计算
    orginal_data['lndret'] = np.log(orginal_data['close'] / orginal_data['pre_close'])
    orginal_data['holding1D_lndret'] = orginal_data.groupby('ts_code')['lndret'].shift(-1)
    orginal_data = orginal_data.dropna()
    # 提取股票所属行业信息
    Stock_industry = orginal_data['industry_name'].unique()
    return orginal_data, Stock_industry
def process_group_by_date(group, target_factor, layers):
    # 先按目标因子值对当前分组进行排序(从小到大升序排列，负向信号在前group_0,正向信号在后group_5)
    sorted_group = group.sort_values(by=target_factor, ascending=False).reset_index(drop=True)
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
    sorted_group['mean_lndret'] = sorted_group.groupby('quantile')['holding1D_lndret'].transform('mean')
    return sorted_group
def spread_ret_cumsum_calculate(data, layers):
    Spread_ret = data.pivot_table(
        index='trade_date',
        columns='quantile',
        values='mean_lndret',
        aggfunc='first'
    ).reset_index()
    Spread_ret['L-S'] = Spread_ret.iloc[:, 1] - Spread_ret.iloc[:, -1]  # group_0 - group_last
    quantiles = list(range(layers)) # e.g  layers=5,quantiles=[0,1,2,3,4]
    for q in quantiles:
        if q in Spread_ret.columns:  # 检查列是否存在
            Spread_ret[f'sum_ret_{q}'] = Spread_ret[q].cumsum()
        else:
            print(f"警告: 数据中缺少分位数 {q} 的数据")
    Spread_ret['sum_ret_L-S'] = Spread_ret['L-S'].cumsum()
    return Spread_ret
def plot_multiple_return_metrics(industry_dict, layers):
    # 设置中文字体以支持中文行业名
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 定义颜色
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    max_industries_per_row = 5
    max_rows = 6
    total_plots = len(industry_dict)
    rows_needed = int(np.ceil(total_plots / max_industries_per_row))

    fig, axes = plt.subplots(rows_needed, max_industries_per_row,
                             figsize=(max_industries_per_row * 4, rows_needed * 3))
    fig.suptitle(f'全行业（申万一级）分层回测结果', fontsize=16, fontweight='bold')
    # 确保 axes 是二维数组，即使只有一个子图
    if total_plots == 1:
        axes = [axes]
    elif rows_needed == 1:
        axes = axes.reshape(1, -1)
    elif max_industries_per_row == 1:
        axes = axes.reshape(-1, 1)
    # 扁平化 axes 数组以便于遍历
    flat_axes = axes.flatten()
    # 遍历字典中的每个行业及其数据
    for idx, (industry_name, dataframe) in enumerate(industry_dict.items()):
        if idx >= len(flat_axes):  # 防止字典中的行业数超过预设的子图数量
            break
        ax = flat_axes[idx]
        # 处理日期列
        if not np.issubdtype(dataframe['trade_date'].dtype, np.datetime64):
            date_col = pd.to_datetime(dataframe['trade_date'].astype(str), format='%Y%m%d')
        else:
            date_col = dataframe['trade_date']
        # 生成 x 轴刻度标签
        n = len(dataframe) - 1
        indices = np.linspace(0, n, 6, dtype=int)
        xtick_labels = date_col.iloc[indices].dt.strftime('%Y%m%d')
        xtick_positions = indices
        # 绘制各层曲线
        for q in range(layers):
            col_name = f'sum_ret_{q}'
            if col_name in dataframe.columns:
                color_idx = q % len(colors)
                ax.plot(dataframe.index, dataframe[col_name], label=f'group_{q}', color=colors[color_idx],
                        linewidth=1)
            else:
                print(f"警告: 行业 '{industry_name}' 的数据中不存在列 '{col_name}'")

        ax.set_title(f'{industry_name}', fontsize=8)
        # ax.set_ylabel('Cumulative Return', fontsize=8) # 如果需要，可以取消注释
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(fontsize=6, loc='upper left')  # 图例字体较小
        ax.grid(True, alpha=0.3)

        # 设置 x 轴刻度
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
    # 如果子图数量多于实际需要的行业数，隐藏多余的子图
    for j in range(idx + 1, len(flat_axes)):
        flat_axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect 参数为 [left, bottom, right, top]，为总标题留出空间
    plt.show()

# --- 主程序执行 ---
if __name__ == "__main__":
    traget_factor = 'alpha_09'  # 分层多空检验的目标因子名称
    layers = 5 # 分层层数（默认为5）max = 10
    test_window_start = '20220701'
    test_window_end = '20240731'
    multi_industry_results_dict = {} # 储存每组日度的平均收益
    multi_industry_results_analysis = {} # 储存最后的累计收益结果sum_ret列

    df, Stock_industry = data_preprocessing(traget_factor, test_window_start, test_window_end)
    for item in Stock_industry:
        df_se = df[df['industry_name'] == item].copy()
        df_industry_result = df_se.groupby('trade_date')[
            ['trade_date', 'ts_code', traget_factor, 'holding1D_lndret']].apply(
            lambda group: process_group_by_date(group, traget_factor, layers)
        ).reset_index(drop=True)
        # print(df_industry_result.head(3))
        # item行业的分组收益率计算存入字典
        multi_industry_results_dict[item] = df_industry_result
        multi_industry_results_analysis[item] = spread_ret_cumsum_calculate(df_industry_result, layers)
        # print(f'{item} 行业结果形状:', df_result.shape)
    # 行业分层回测图
    plot_multiple_return_metrics(multi_industry_results_analysis, layers)









