"""按冻结证据权重评分并统一为高分优先；不在运行时读取收益标签。"""

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray


def priority_scores(values: FloatArray, eligible: NDArray[np.bool_]) -> FloatArray:
    """将当日合格池内原始效用转为0到100的平均秩分，保持原排序及并列。

    输入日×证券效用与资格，输出同形状分数，未合格或缺值为NaN。
    例效用[1,2,2,4]得到[0,50,50,100]；仅一只或全并列时为50。
    分数表示当日相对排序，不是胜率，不可跨日当成绝对收益阈值。
    """
    if values.ndim != 2 or eligible.shape != values.shape or eligible.dtype != np.bool_:
        raise ValueError("invalid_priority_scores")
    result = np.full(values.shape, np.nan)
    for day in range(len(values)):
        selected = np.flatnonzero(eligible[day] & np.isfinite(values[day]))
        if not len(selected):
            continue
        unique, inverse, counts = np.unique(
            values[day, selected], return_inverse=True, return_counts=True
        )
        if len(unique) == 1:
            result[day, selected] = 50.0
        else:
            starts = np.cumsum(counts) - counts
            average_ranks = starts + (counts - 1) / 2
            result[day, selected] = 100 * average_ranks[inverse] / (len(selected) - 1)
    return result


def signal_evidence_scores(
    patterns: NDArray[np.uint32], weights: FloatArray, *, average_hits: bool = True
) -> FloatArray:
    """将每个命中信号的冻结权重相加或平均，返回可解释的原始评分。

    patterns为日×证券位图；weights为信号轴或日×信号矩阵，后者允许用当日
    已知市场状态选权重。权重由外部离线研究冻结，本函数不拟合或读取未来收益。
    例位图[1,2,3]、权重[.02,-.01]，平均模式输出[.02,-.01,.005]。
    未命中为NaN；不把重复命中次数自动当成信心，默认取平均。
    离线全样本拟合后的回测仍然是样本内，不能因推理无标签就宣称样本外。
    """
    if (
        patterns.ndim != 2
        or patterns.dtype != np.uint32
        or weights.ndim not in (1, 2)
        or not 1 <= weights.shape[-1] <= 32
        or (weights.ndim == 2 and weights.shape[0] != len(patterns))
        or not np.all(np.isfinite(weights))
        or type(average_hits) is not bool
    ):
        raise ValueError("invalid_signal_evidence_scores")
    if np.any(patterns.astype(np.uint64) >= (1 << weights.shape[-1])):
        raise ValueError("unknown_signal_bit")
    result = np.zeros(patterns.shape)
    counts = np.zeros(patterns.shape, dtype=np.uint8)
    for index in range(weights.shape[-1]):
        hit = (patterns & np.uint32(1 << index)) != 0
        weight = weights[index] if weights.ndim == 1 else weights[:, index, None]
        result += hit * weight
        counts += hit
    if average_hits:
        np.divide(result, counts, out=result, where=counts > 0)
    result[counts == 0] = np.nan
    return result
