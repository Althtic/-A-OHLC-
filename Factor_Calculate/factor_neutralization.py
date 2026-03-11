import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def neutralize_factor(df,
                      target_factor,
                      cap_col='circ_mv', # 流通总市值
                      industry_col='industry_name', # 行业名称
                      date_col='trade_date',
                      id_col='ts_code'):

    # 检查合并后是否有大量缺失市值
    missing_cap = df[cap_col].isna().sum()
    if missing_cap > 0:
        logger.info(f"Warning: {missing_cap} 行数据缺失市值 ({cap_col})，这些行在中性化时将返回 NaN")

    # 截面中性化逻辑函数
    def process_single_date(group):
        # 初始化结果列为 NaN
        group['factor_neutralized'] = np.nan
        group_clean = group.copy()
        # 找出所有数值类型的列
        numeric_cols = group_clean.select_dtypes(include=[np.number]).columns
        # 数据清洗：替换 inf 为 NaN
        # 使用 mask 将 inf/-inf 替换为 nan (不会触发 Downcasting 警告)
        if len(numeric_cols) > 0:
            group_clean[numeric_cols] = group_clean[numeric_cols].mask(
                np.isinf(group_clean[numeric_cols]),
                np.nan
            )

        valid_mask = (
                group_clean[target_factor].notna() &
                group_clean[cap_col].notna() &
                group_clean[industry_col].notna()
        )
        if valid_mask.sum() < 10:
            return group

        data_valid = group_clean[valid_mask].copy()

        # 准备 Y (因子值)
        y = data_valid[target_factor].values
        # 准备 X1: ln(市值)
        # 确保市值为正数再取 log，否则会有 warning 或 nan
        log_cap_vals = data_valid[cap_col].clip(lower=1e-9)  # 防止 log(0) 或负数
        log_cap = np.log(log_cap_vals).values.reshape(-1, 1)
        # 准备 X2: 行业哑变量
        # fillna('Unknown') 防止行业列为空导致 get_dummies 出错
        industry_series = data_valid[industry_col].fillna('Unknown')
        industry_dummies = pd.get_dummies(industry_series, prefix='ind', drop_first=False)
        # 合并 X (市值 + 行业)
        # 注意：get_dummies 返回的是 DataFrame，需要取 .values 转为 numpy 数组以便 hstack
        X = np.hstack([log_cap, industry_dummies.values])


        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        residuals = y - predictions
        group.loc[valid_mask, 'factor_neutralized'] = residuals
        return group

    # --- 步骤 3: 按日期分组应用 ---
    logger.info(f"正在进行截面中性化处理...")

    # 注意：apply 可能会改变索引顺序，通常建议最后 reset_index 或者保持原样
    result_df = df.groupby(date_col, group_keys=False).apply(process_single_date)
    # 保留初始因子值，全行业分层回测时仍需使用(raw_factor: 未经过中性化的因字值)
    result_df.rename(columns={target_factor: 'raw_factor'}, inplace=True)
    result_df['raw_factor'] = result_df['raw_factor'].round(6)
    # 保留残差作为中性化后的因子值使用
    result_df.rename(columns={'factor_neutralized': target_factor}, inplace=True)
    return result_df

# ==========================================
# 使用示例，以WorldQuant#45号因子为例
# ==========================================
if __name__ == "__main__":
    data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101\alpha_45.csv')
    # 调用函数
    final_data = neutralize_factor(data, target_factor='alpha_45')
    # 查看结果
    print(final_data.head())
    pass
