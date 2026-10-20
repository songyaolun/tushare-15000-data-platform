"""非线性研究模型的固定输入契约，不接受日期、证券身份或未来目标。"""

import numpy as np

from ..strategy_optimization.features import FloatArray
from .economic_features import FEATURE_BUCKETS

NONLINEAR_FEATURE_NAMES = tuple(
    name for name, (group, _) in FEATURE_BUCKETS.items() if group != "market"
)


def nonlinear_input_features(
    raw: dict[str, FloatArray],
    signal_hits: FloatArray,
    signal_names: tuple[str, ...],
) -> tuple[FloatArray, tuple[str, ...]]:
    """拼接固定技术、行业原始值和信号命中，缺失值原样保留供树模型处理。

    raw须恰含NONLINEAR_FEATURE_NAMES中的16个字段，signal_hits为同行数二值
    矩阵，signal_names与列一一对应。例两行原始特征加一列[0,1]信号，输出
    2×17矩阵及有序名称，最后一列名为signal:<信号名>。日期和股票编号不能
    增补为输入列；市场主效应不在本契约中，但这不构成消除市场影响的保证。
    允许NaN代表未知，拒绝无穷大，数组不得作为历史可知权重的证据。
    """
    if (
        set(raw) != set(NONLINEAR_FEATURE_NAMES)
        or signal_hits.ndim != 2
        or signal_hits.shape[1] != len(signal_names)
        or not signal_names
        or len(set(signal_names)) != len(signal_names)
        or any(not name or name.strip() != name for name in signal_names)
        or not np.all(np.isin(signal_hits, [0, 1]))
        or any(value.shape != (len(signal_hits),) for value in raw.values())
        or any(np.any(np.isinf(value)) for value in raw.values())
    ):
        raise ValueError("invalid_nonlinear_input_features")
    names = NONLINEAR_FEATURE_NAMES + tuple("signal:" + name for name in signal_names)
    return np.column_stack([raw[name] for name in NONLINEAR_FEATURE_NAMES] + [signal_hits]), names
