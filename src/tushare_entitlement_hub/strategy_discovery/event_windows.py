"""市场日矩阵的因果均线与事件时钟，不跨停牌压缩时间。"""

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray


def shift_matrix(values: FloatArray, sessions: int = 1) -> FloatArray:
    """按市场日滞后二维矩阵，开头保留未知；例如[[1],[2]]滞后1日为[[nan],[1]]。

    输入必须为二维，sessions必须为正整数；空矩阵返回同形空矩阵。
    """
    if values.ndim != 2 or type(sessions) is not int or sessions < 1:
        raise ValueError("invalid_matrix_shift")
    result = np.full(values.shape, np.nan)
    result[sessions:] = values[:-sessions]
    return result


def moving_average(values: FloatArray, window: int) -> FloatArray:
    """仅对完整有限窗口求均值；例如[[1],[3],[nan],[7]]与2输出[[nan],[2],[nan],[nan]]。

    使用逐日累计和与有限值计数，缺失不填零；窗口长于样本时全部未知。
    """
    if values.ndim != 2 or type(window) is not int or window < 1:
        raise ValueError("invalid_matrix_average")
    result = np.full(values.shape, np.nan)
    totals = np.zeros(values.shape[1])
    counts = np.zeros(values.shape[1], dtype=np.int64)
    for day, row in enumerate(values):
        known = np.isfinite(row)
        totals += np.where(known, row, 0)
        counts += known
        if day >= window:
            previous = values[day - window]
            previous_known = np.isfinite(previous)
            totals -= np.where(previous_known, previous, 0)
            counts -= previous_known
        result[day] = np.where(counts == window, totals / window, np.nan)
    return result


def recent_event(events: NDArray[np.bool_], window: int) -> NDArray[np.bool_]:
    """标记含当日的最近window日是否有事件；例如[真,假,假]与2输出[真,真,假]。

    输入为日×证券布尔矩阵，窗口须为正整数；不借用未来事件反填历史。
    """
    if events.ndim != 2 or events.dtype != np.bool_ or type(window) is not int or window < 1:
        raise ValueError("invalid_recent_event")
    result = np.zeros(events.shape, dtype=bool)
    last = np.full(events.shape[1], -window, dtype=np.int64)
    for day, row in enumerate(events):
        last = np.where(row, day, last)
        result[day] = day - last < window
    return result


def support_formation(short: FloatArray, middle: FloatArray, long: FloatArray) -> NDArray[np.bool_]:
    """识别十市场日内依次完成的三均线金叉，仅在最后一叉当日发事件。

    例如5日线上穿10日线在t-4、上穿20日线在t-2、10日线上穿20日线在t，
    当前5>10>20且20日线不低于t-3，则t为True。三叉可同日，顺序倒置不通过。
    均线未知时重置事件记忆；1e-10相对容差防止累计误差把平线变成金叉。
    """
    if short.ndim != 2 or short.shape != middle.shape or short.shape != long.shape:
        raise ValueError("invalid_support_averages")
    result = np.zeros(short.shape, dtype=bool)
    last_cross = np.full((3, short.shape[1]), -100, dtype=np.int64)
    above = np.stack(
        (short > middle * (1 + 1e-10), short > long * (1 + 1e-10), middle > long * (1 + 1e-10))
    )
    known = np.isfinite(short) & np.isfinite(middle) & np.isfinite(long)
    for day in range(1, len(short)):
        valid = known[day] & known[day - 1]
        last_cross[:, ~valid] = -100
        crossed = above[:, day] & ~above[:, day - 1] & valid
        last_cross = np.where(crossed, day, last_cross)
        if day >= 3:
            result[day] = (
                valid
                & above[:, day].all(axis=0)
                & (last_cross[0] >= day - 9)
                & (last_cross[1] >= last_cross[0])
                & (last_cross[2] >= last_cross[1])
                & (last_cross[2] == day)
                & (long[day] >= long[day - 3] * (1 - 1e-10))
            )
    return result
