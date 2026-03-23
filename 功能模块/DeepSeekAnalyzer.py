import os
import json
import requests

API_KEY = "sk-3dd3717d2b914b95802e7949062f76fd"
API_URL = "https://api.deepseek.com/v1/chat/completions"

def load_validation_result(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_rank_ic_mean_matrix(rank_ic_mean):
    """
    将 rank_ic_mean 数据转换为热力图矩阵
    格式: rank_ic_mean_matrix[year_index][month_index] = IC均值
    year_index: 0=2019, 1=2020, ..., 6=2025
    month_index: 0=1月, 1=2月, ..., 11=12月
    rank_ic_mean 结构: {"1": {"2019": val, ...}, "2": {...}, ...}
    """
    years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    # 初始化 7行(年) x 12列(月) 的矩阵
    matrix = [[None for _ in range(12)] for _ in range(7)]

    for group_str, year_values in rank_ic_mean.items():
        group = int(group_str)  # "1" -> 1, "12" -> 12
        for year_str, ic_value in year_values.items():
            year_idx = years.index(year_str) if year_str in years else None
            if year_idx is not None and 1 <= group <= 12:
                matrix[year_idx][group - 1] = ic_value

    return matrix


def build_prompt(data):
    target_factor = data.get("target_factor", "unknown")
    ic_analysis = data.get("ic_analysis", {})
    yearly = data.get("yearly_ic_icir", {})
    monthly = data.get("monthly_ic_icir", {})
    ttest = yearly.get("ttest_result", {})
    # rank_ic 的数据字典{月份：{年份：'value'}}}
    rank_ic_mean = monthly.get("rank_ic_mean", {})
    # icir 的数据字典{月份：{年份：'value'}}}
    ic_ir = monthly.get("icir", {})


    rank_ic_mean_matrix = build_rank_ic_mean_matrix(rank_ic_mean)
    ic_ir_matrix = build_rank_ic_mean_matrix(ic_ir)

    prompt = f"""你是量化分析师。请分析因子 **{target_factor}** 的有效性（要凸显专业性，不能放过任何一个你读取到的有效信息，要发现其中的关联以及可能蕴含的经济原理）。

【IC分析】
- Rank_IC胜率: {ic_analysis.get('ic_win_rate', 'N/A')}
- ICIR: {ic_analysis.get('icir', 'N/A')}
- Rank_IC在持有期内的日度衰减: {ic_analysis.get('ic_decay', 'N/A')}
- Rank_IC在持有期内的累计: {ic_analysis.get('cumulative_ic', 'N/A')}
- Rank_IC在持有期内的累计最大回撤: {ic_analysis.get('max_dd_ratio', 'N/A')}
- Rank_IC在持有期内的累计最大回撤日期: {ic_analysis.get('max_dd_date', 'N/A')}

【年度IC】
- 年份: {yearly.get('years', [])}
- ICIR序列: {yearly.get('icir_values', [])}
- RankIC均值序列: {yearly.get('rank_ic_mean_values', [])}

【月度IC热力图】
- ICIR对应的热力图矩阵(例如rank_ic_mean_matrix[0][11] = 0.756 表示 2019年12月 的 IC 均值): {rank_ic_mean_matrix}
- rankic均值对应的热力图矩阵(规则与ICIR对应的热力图矩阵相同)): {ic_ir_matrix}

【T检验】
- (NW-adjusted)t统计量: {ttest.get('t_stat', 'N/A')}
- 均值: {ttest.get('mean', 'N/A')}
- 显著性: {ttest.get('status', 'N/A')}
- 结论: {ttest.get('conclusion', 'N/A')}

请输出简洁的分析报告，包含：
1. IC有效性评估
2. 稳定性判断
3. 调仓建议
"""

    return prompt

def analyze_with_deepseek(prompt, model="deepseek-chat"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    json_path = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\因子检验结果\alpha_60\validation_test_result.json"

    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
    else:
        data = load_validation_result(json_path)
        prompt = build_prompt(data)
        analysis = analyze_with_deepseek(prompt)

        report_dir = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\因子分析报告"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        target_factor = data.get("target_factor", "unknown")
        report_path = os.path.join(report_dir, f"{target_factor}_validation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 因子 {target_factor} 有效性分析报告\n\n")
            f.write(analysis)

        print(f"报告已保存至: {report_path}")
        print("\n--- 分析结果 ---\n")
        print(analysis)
