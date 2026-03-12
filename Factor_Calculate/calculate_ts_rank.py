import numpy as np
from numba import njit


def calc_ts_rank(series, codes, window):
    values = np.asarray(series, dtype=np.float64)
    change = np.where(codes[:-1] != codes[1:])[0] + 1
    group_starts = np.concatenate(([0], change))
    group_lengths = np.diff(np.concatenate((group_starts, [len(codes)])))
    return calc_ts_rank_numba(values, group_starts, group_lengths, window)


@njit
def calc_ts_rank_numba(values, group_starts, group_lengths, window):

    n_total = len(values)
    ranks = np.full(n_total, np.nan)

    for g_idx in range(len(group_starts)):
        start = group_starts[g_idx]
        length = group_lengths[g_idx]
        end = start + length
        # 如果数据长度不足窗口，跳过（保持 NaN）
        if length < window:
            continue
        # 处理当前股票的时间序列
        for i in range(start + window - 1, end):
            current_val = values[i]
            count = 0
            # 窗口范围 [i-window+1, i]
            w_start = i - window + 1
            # 内部循环被 Numba 编译为机器码，极快
            for j in range(w_start, i + 1):
                if values[j] <= current_val:
                    count += 1

            ranks[i] = count / window

    return ranks
