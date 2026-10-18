"""全候选的条件增益诊断；收益标签只用于研究，禁止回流当日交易特征。"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray


def forward_open_returns(
    opening: FloatArray,
    adjustment: FloatArray,
    upper_limit: FloatArray,
    lower_limit: FloatArray,
    closing: FloatArray,
    holding: int,
    maximum_gap: float = 0.02,
) -> FloatArray:
    """计算信号后次开盘买、holding日后开盘卖的复权毛收益标签。

    输入日×证券原始价和因子；例如次开盘10、五日后11，holding=5返回.1。
    买入涨停、卖出跌停、缺报价、越过样本尾部或高开超限均返回NaN，并须
    在报告中计入缺失，不能当零收益。不是账户回测：没有资金约束、费用或
    除权分红现金时点模型；停牌/跌停导致的延期退出由后续真实账本验证。
    """
    arrays = (opening, adjustment, upper_limit, lower_limit, closing)
    if (
        opening.ndim != 2
        or any(value.shape != opening.shape for value in arrays)
        or type(holding) is not int
        or holding < 1
        or not np.isfinite(maximum_gap)
        or maximum_gap < 0
    ):
        raise ValueError("invalid_forward_return_inputs")
    result = np.full(opening.shape, np.nan)
    count = len(opening) - holding - 1
    if count <= 0:
        return result
    entry = opening[1 : count + 1]
    exit_price = opening[holding + 1 :]
    entry_factor = adjustment[1 : count + 1]
    exit_factor = adjustment[holding + 1 :]
    reference = closing[:count] * adjustment[:count]
    upper = upper_limit[1 : count + 1]
    lower = lower_limit[holding + 1 :]
    known = np.ones(entry.shape, dtype=bool)
    for value in (entry, exit_price, entry_factor, exit_factor, reference, upper, lower):
        known &= np.isfinite(value) & (value > 0)
    known &= (entry < upper - 1e-8) & (exit_price > lower + 1e-8)
    known &= entry * entry_factor <= reference * (1 + maximum_gap)
    np.divide(exit_price * exit_factor, entry * entry_factor, out=result[:count], where=known)
    result[:count] -= 1
    return result


def liquidity_buckets(amount: FloatArray, universe: NDArray[np.bool_]) -> NDArray[np.int8]:
    """按当日合格证券成交额划三组；例如同额证券始终同组，未知记为-1。

    输出日×证券的0/1/2标签，只用当日可知成交额；不使用未来收益定组。
    """
    if amount.ndim != 2 or amount.shape != universe.shape or universe.dtype != np.bool_:
        raise ValueError("invalid_liquidity_bucket_inputs")
    result = np.full(amount.shape, -1, dtype=np.int8)
    for day in range(len(amount)):
        valid = universe[day] & np.isfinite(amount[day]) & (amount[day] > 0)
        if not valid.any():
            continue
        cuts = np.quantile(amount[day, valid], [1 / 3, 2 / 3])
        result[day, valid] = np.searchsorted(cuts, amount[day, valid], side="left")
    return result


def matched_daily_effect(
    outcome: FloatArray,
    universe: NDArray[np.bool_],
    hit: NDArray[np.bool_],
    industry: FloatArray,
    buckets: NDArray[np.int8],
    minimum_peers: int = 5,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """在同日、同行业、同流动性组，以未命中股票作对照并按命中股票平均。

    例某组命中收益.03、五只未命中均值.01，输出当日增益.02和配对数1。
    未命中对照不足minimum_peers则不配对；两边都只使用有效标签。返回
    每日增益和配对事件数，后续按日期聚合，避免把同日股票当独立时间样本。
    """
    if (
        outcome.ndim != 2
        or any(value.shape != outcome.shape for value in (universe, hit, industry, buckets))
        or universe.dtype != np.bool_
        or hit.dtype != np.bool_
        or type(minimum_peers) is not int
        or minimum_peers < 1
    ):
        raise ValueError("invalid_matched_effect_inputs")
    result = np.full(len(outcome), np.nan)
    counts = np.zeros(len(outcome), dtype=np.int64)
    for day in range(len(outcome)):
        known = (
            universe[day]
            & np.isfinite(outcome[day])
            & np.isfinite(industry[day])
            & (industry[day] >= 0)
            & (buckets[day] >= 0)
        )
        selected = known & hit[day]
        control = known & ~hit[day]
        keys = np.where(known, industry[day] * 3 + buckets[day], 0).astype(int)
        control_counts = np.bincount(keys[control])
        control_sums = np.bincount(keys[control], weights=outcome[day, control])
        selected_indices = np.flatnonzero(selected)
        if not len(selected_indices) or not len(control_counts):
            continue
        selected_keys = keys[selected_indices]
        valid = selected_keys < len(control_counts)
        selected_indices = selected_indices[valid]
        selected_keys = selected_keys[valid]
        valid = control_counts[selected_keys] >= minimum_peers
        selected_indices, selected_keys = selected_indices[valid], selected_keys[valid]
        if len(selected_indices):
            differences = outcome[day, selected_indices] - (
                control_sums[selected_keys] / control_counts[selected_keys]
            )
            result[day] = np.mean(differences)
            counts[day] = len(differences)
    return result, counts


def event_summary(outcome: FloatArray, selected: NDArray[np.bool_]) -> dict[str, Any]:
    """统计事件覆盖和收益，同时保留日等权均值；例全缺失报告coverage=0。

    selected包含未能成交及样本尾部事件，不能先按未来可成交情况缩小分母。
    返回事件数、可观察数、毛胜率、均值及日等权均值；不宣称为账户收益。
    """
    if outcome.ndim != 2 or outcome.shape != selected.shape or selected.dtype != np.bool_:
        raise ValueError("invalid_event_summary_inputs")
    valid = selected & np.isfinite(outcome)
    observations = outcome[valid]
    counts = valid.sum(axis=1)
    totals = np.where(valid, outcome, 0).sum(axis=1)
    daily = totals[counts > 0] / counts[counts > 0]
    return {
        "events": int(selected.sum()),
        "observed": len(observations),
        "coverage": float(valid.sum() / selected.sum()) if selected.any() else 0.0,
        "mean_gross_return": float(np.mean(observations)) if len(observations) else None,
        "gross_win_rate": float(np.mean(observations > 0)) if len(observations) else None,
        "daily_equal_mean": float(np.mean(daily)) if len(daily) else None,
        "signal_days": int(np.sum(selected.any(axis=1))),
    }


def score_groups(
    scores: FloatArray, universe: NDArray[np.bool_], groups: int = 5
) -> NDArray[np.int8]:
    """每日按当前分数划分分位组，低分0、高分groups-1；未知为-1。

    例分数完全相同时都进0组，不按证券代码伪造高低分；因此组数可能不足。
    """
    if (
        scores.ndim != 2
        or scores.shape != universe.shape
        or universe.dtype != np.bool_
        or type(groups) is not int
        or not 2 <= groups <= 100
    ):
        raise ValueError("invalid_score_groups_inputs")
    result = np.full(scores.shape, -1, dtype=np.int8)
    for day in range(len(scores)):
        known = universe[day] & np.isfinite(scores[day])
        if known.any():
            boundaries = np.quantile(scores[day, known], np.arange(1, groups) / groups)
            result[day, known] = np.searchsorted(boundaries, scores[day, known], side="left")
    return result
