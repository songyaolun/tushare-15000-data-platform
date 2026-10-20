"""同源筹码成本分布特征；不假设供应商价格复权基准或历史不可修订。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def candidate_chip_features(
    costs: NDArray[np.float64],
    candidate_days: NDArray[np.int64],
    candidate_securities: NDArray[np.int64],
    *,
    change_sessions: int = 5,
) -> NDArray[np.float64]:
    """把日×证券×四成本映射为候选的偏离、宽度及向后窗口宽度变化。

    四成本依次为5分位、50分位、95分位、加权平均。所有比值仅使用同一
    响应内的成本，因此共同乘以任何正复权因子都不改变结果。偏离为
    (50分位-加权均值)/加权均值；宽度为(95分位-5分位)/50分位；变化
    为当前宽度减去指定交易日前宽度。整个窗口必须有效，未知保留NaN。
    这不是实际账户持仓或严格历史PIT认证，不接收未来数据填补缺失。

    示例：单行成本[8,10,12,10]及候选索引[0],[0]返回[[0,0.4,NaN]]。
    输入维度、索引或窗口无效抛ValueError；输出顺序与候选输入一致。
    """
    if (
        costs.ndim != 3
        or costs.shape[2] != 4
        or candidate_days.ndim != 1
        or candidate_securities.shape != candidate_days.shape
        or not np.issubdtype(candidate_days.dtype, np.integer)
        or not np.issubdtype(candidate_securities.dtype, np.integer)
        or type(change_sessions) is not int
        or change_sessions < 1
        or np.any(candidate_days < 0)
        or np.any(candidate_days >= costs.shape[0])
        or np.any(candidate_securities < 0)
        or np.any(candidate_securities >= costs.shape[1])
    ):
        raise ValueError("invalid_chip_feature_axis")
    lower, median, upper, weighted = np.moveaxis(costs, 2, 0)
    valid = (
        np.isfinite(costs).all(axis=2)
        & (costs > 0).all(axis=2)
        & (lower <= median)
        & (median <= upper)
    )
    width = np.full(valid.shape, np.nan)
    offset = np.full(valid.shape, np.nan)
    np.divide(upper - lower, median, out=width, where=valid)
    np.divide(median - weighted, weighted, out=offset, where=valid)
    change = np.full(valid.shape, np.nan)
    if change_sessions < len(costs):
        complete = np.ones(valid[change_sessions:].shape, dtype=bool)
        for lag in range(change_sessions + 1):
            complete &= valid[lag:len(costs) - change_sessions + lag]
        difference = width[change_sessions:] - width[:-change_sessions]
        change[change_sessions:] = np.where(complete, difference, np.nan)
    return np.column_stack((
        offset[candidate_days, candidate_securities],
        width[candidate_days, candidate_securities],
        change[candidate_days, candidate_securities],
    ))
