"""跨证券收益比较的市场日窗口完整性，不能用跳过停牌的观察数替代。"""

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray


def complete_history(valid: NDArray[np.bool_] | FloatArray, lookback: int) -> NDArray[np.bool_]:
    """要求当日及此前lookback个市场日全部有效，返回同形状布尔矩阵。

    输入单列[1,1,1,0,1,1,1]和lookback=2，输出[假,假,真,假,假,假,真]。
    仅True或数值1算有效，NaN、无穷、0均中断连续性；不足窗口不前填。
    输入必须日×证券且lookback为正整数；例如lookback=0抛ValueError。
    按日期维护每证券连续计数，仅需证券数规模的临时空间，且不读取未来。
    """
    if valid.ndim != 2 or type(lookback) is not int or lookback < 1:
        raise ValueError("invalid_complete_history_contract")
    result = np.zeros(valid.shape, dtype=bool)
    consecutive = np.zeros(valid.shape[1], dtype=np.int64)
    for day, observations in enumerate(valid):
        consecutive = np.where(observations == 1, consecutive + 1, 0)
        result[day] = consecutive > lookback
    return result
