import pandas as pd
import numpy as np

'''
功能模块：行情与财务数据高精度对齐 (Merge Market & Financial Data)

1. 核心目标：
   解决高频行情数据（日频）与低频财务数据（季频）的匹配问题，构建用于量化回测的标准数据集。

2. 关键逻辑 - 严格防止未来函数 (No Look-ahead Bias)：
   - 双重时间校验：仅当【财报公告日 (ann_date)】与【报告期截止日 (end_date)】均 <= 【交易日期 (trade_date)】时，该财务数据才被视为“已知”。
   - 确保在回测中，任何一天的交易决策只能使用当时市场上已经公开披露的最新财务信息，杜绝数据泄露。

3. 匹配策略 - 最新可用原则：
   - 对于每一个交易日，自动筛选出所有满足时间条件的财报中，公告时间最晚（即信息最新）的那一条记录。
   - 若某交易日尚无最新财报公布，则对应财务字段自动填充为 NA，保持数据真实性。

4. 性能优化：
   - 向量化广播计算：利用 Numpy 广播机制构建 (交易日数 × 财报数) 的匹配矩阵，替代低效的逐行循环，大幅提升全市场数千只股票的处理速度。
   - 分股并行处理：按股票代码 (ts_code) 分组独立计算，避免内存溢出并逻辑隔离。

5. 数据预处理与输出：
   - 自动取交集：仅保留同时存在于行情表和财务表中的股票代码。
   - 类型兼容：日期列统一转换为支持缺失值的 Int64 格式，方便后续存储与分析。
   - 输出结果：生成包含每日行情及当时最新财务指标 (如 ROE) 的宽表，直接可用于因子挖掘或策略回测。
'''


def merge_market_with_financial(market_df, financial_df):
    print("开始快速合并...")

    # 1. 预处理：排序
    fin_sorted = financial_df.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)
    mkt_sorted = market_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    results = []
    stocks = mkt_sorted['ts_code'].unique()
    total_stocks = len(stocks)
    print(f"共需处理 {total_stocks} 只股票...")

    for i, stock in enumerate(stocks):
        if (i + 1) % 100 == 0:
            print(f"进度：{i + 1}/{total_stocks}")

        # 提取单只股票数据
        f_mask = fin_sorted['ts_code'] == stock
        m_mask = mkt_sorted['ts_code'] == stock
        current_fin = fin_sorted[f_mask]
        current_mkt = mkt_sorted[m_mask]

        if current_fin.empty or current_mkt.empty:
            continue

        # 转为 Numpy 数组以加速计算
        mkt_dates = current_mkt['trade_date'].values
        fin_ann = current_fin['ann_date'].values
        fin_end = current_fin['end_date'].values

        M, N = len(mkt_dates), len(fin_ann)

        # 向量化广播计算 (M, 1) vs (1, N)
        # 掩码：同时满足公告日<=交易日 且 财报截止日<=交易日
        valid_mask = (fin_ann.reshape(1, -1) <= mkt_dates.reshape(-1, 1)) & \
                     (fin_end.reshape(1, -1) <= mkt_dates.reshape(-1, 1))

        # 若无任何匹配，直接生成全 NA 的财务表
        if not np.any(valid_mask):
            empty_fin = pd.DataFrame(pd.NA, index=range(M), columns=current_fin.columns)
        else:
            # 将无效位置的 ann_date 设为 -1，以便 argmax 找到有效最大值
            ann_values = np.where(valid_mask, fin_ann, -1)
            best_idx = np.argmax(ann_values, axis=1)
            has_match = np.max(ann_values, axis=1) > -1

            # 选取对应的财务行
            selected_rows = current_fin.iloc[best_idx].reset_index(drop=True)

            # 构建最终财务表：先复制选中的行，再将无匹配的行设为 NA
            final_fin_data = selected_rows.copy()
            if not np.all(has_match):
                cols_to_nan = [c for c in final_fin_data.columns if c != 'ts_code']
                final_fin_data.loc[~has_match, cols_to_nan] = pd.NA

        # 清理重叠列 (避免 concat 冲突)，保留行情表的 ts_code 和 trade_date
        clean_fin = final_fin_data.drop(columns=['ts_code', 'trade_date'], errors='ignore')
        clean_fin = clean_fin.reset_index(drop=True)
        reset_mkt = current_mkt.reset_index(drop=True)

        # 关键优化：使用 pd.concat 一次性拼接，避免 DataFrame 碎片化
        combined = pd.concat([reset_mkt, clean_fin], axis=1)
        results.append(combined)

    print("合并完成，正在拼接总表...")
    return pd.concat(results, ignore_index=True)


# ================= 主程序 =================
if __name__ == '__main__':
    # 读取数据
    market_df = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv')
    financial_df = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市财务指标数据\roe_data_clean.csv')

    print(f"原始行情股票数: {len(market_df['ts_code'].unique())}")
    print(f"原始财务股票数: {len(financial_df['ts_code'].unique())}")

    # 获取交集
    common_codes = pd.Index(market_df['ts_code'].unique()).intersection(pd.Index(financial_df['ts_code'].unique()))
    # 如需全量运行，请注释掉下一行的切片操作
    # common_codes = common_codes[:100]

    filtered_mkt = market_df[market_df['ts_code'].isin(common_codes)]
    filtered_fin = financial_df[financial_df['ts_code'].isin(common_codes)]

    # 执行合并
    merged_df = merge_market_with_financial(filtered_mkt, filtered_fin)

    # 日期列类型转换 (兼容 NA 的 Int64)
    date_cols = ['ann_date', 'end_date', 'trade_date']
    for col in date_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype('Int64')

    # 保存结果
    output_file = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\roe.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"成功保存至 {output_file}")