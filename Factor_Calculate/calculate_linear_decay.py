import numpy as np
from numba import njit

def linear_decay_peaks(series, codes, window):
    values = np.asarray(series, dtype=np.float64)
    change = np.where(codes[:-1] != codes[1:])[0] + 1
    group_starts = np.concatenate(([0], change))
    group_lengths = np.diff(np.concatenate((group_starts, [len(codes)])))
    return calc_grouped_linear_decay_peaks(values, group_starts, group_lengths, window)

@njit
def _is_peak(values, j, start, end):
    if j <= start:
        return values[j] >= values[j + 1] if j + 1 <= end else True
    if j >= end:
        return values[j] >= values[j - 1] if j - 1 >= start else True
    return values[j] >= values[j - 1] and values[j] >= values[j + 1]

@njit
def calc_grouped_linear_decay_peaks(values, group_starts, group_lengths, window):
    n_total = len(values)
    out = np.full(n_total, np.nan)
    for g_idx in range(len(group_starts)):
        start = group_starts[g_idx]
        length = group_lengths[g_idx]
        end = start + length
        if length < window:
            continue
        for i in range(start + window - 1, end):
            w_start = i - window + 1
            s = 0.0
            for k in range(window):
                j = w_start + k
                if not _is_peak(values, j, w_start, i):
                    continue
                w = (k + 1) / window
                s += values[j] * w
            out[i] = s
    return out
