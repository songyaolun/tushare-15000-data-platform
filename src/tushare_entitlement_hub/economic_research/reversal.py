"""固定规则的大中盘短期相对反转，包含真实延迟与同日同业匹配对照。"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_discovery.windows import complete_history
from ..strategy_optimization.features import FloatArray

REQUIRED_FIELDS = (
    "valid",
    "close",
    "beta",
    "capitalization",
    "raw_amount",
    "raw_up_limit",
    "raw_down_limit",
    "sector_index",
)


@dataclass(frozen=True)
class ReversalFeatures:
    """保留逐日股票池与排序证据；例如universe[t,i]说明证券i在t日进入前500流动性池。

    capitalization取Panel当时已知的月末total_mv向后填充值，非当前成分股名单。
    同业均值至少3个含自身成员，覆盖分母为当日同业池；基础资格已要求21日完整，
    因此正常构造下60%覆盖约束冗余，但仍显式检查，不声称历史ST或分类已认证。
    """

    eligibility: NDArray[np.bool_]
    large_cap_pool: NDArray[np.bool_]
    universe: NDArray[np.bool_]
    average_amount20: FloatArray
    return5: FloatArray
    relative_return5: FloatArray
    entry_mask: NDArray[np.bool_]
    scores: FloatArray


@dataclass(frozen=True)
class ReversalStrategy:
    """绑定冻结候选、原始信号日及单次次日开盘有效的执行要求。

    例delayed.source_days[22]=21且entry_mask[22]来自21日，最早23日开盘成交。
    matched_pairs元素为(原信号日,处理股列号,对照股列号)，对照继承处理股优先分数；
    市值/流动性近邻不是严格风险相同。adapter须从真实买入开始计5日持有，未成交
    订单在唯一允许开盘之后撤销，不得沿用公共strategy_candidates重算延迟日资格。
    """

    name: str
    entry_mask: NDArray[np.bool_]
    scores: FloatArray
    source_days: NDArray[np.int64]
    policy: PortfolioPolicy
    maximum_pending_sessions: int = 1
    matched_pairs: tuple[tuple[int, int, int], ...] = ()


def _average_amount20(amount: FloatArray) -> FloatArray:
    """包含当日计算20个市场日完整正成交额均值；例如20天均为20000，最后输出20000。

    任一天缺失、非有限或非正则该窗口未知；滚动累加仅需证券数规模临时空间。
    """
    result = np.full(amount.shape, np.nan)
    total = np.zeros(amount.shape[1])
    count = np.zeros(amount.shape[1], dtype=int)
    for day, current in enumerate(amount):
        known = np.isfinite(current) & (current > 0)
        total += np.where(known, current, 0)
        count += known
        if day >= 20:
            previous = amount[day - 20]
            previous_known = np.isfinite(previous) & (previous > 0)
            total -= np.where(previous_known, previous, 0)
            count -= previous_known
        result[day] = np.where(count == 20, total / 20, np.nan)
    return result


def _relative_returns(
    returns: FloatArray,
    universe: NDArray[np.bool_],
    membership: FloatArray,
) -> FloatArray:
    """减去同日同池同历史行业的5日收益均值，至少3成员含自身且有效占比>=60%。

    例同业收益[-.04,0,.01]得到相对收益[-.03,.01,.02]；3/5有效可算，3/6则未知。
    未知/负数/非整数行业ID不分组；按当日归属比较，不额外要求21日行业身份不变。
    """
    result = np.full(returns.shape, np.nan)
    for day, active in enumerate(universe):
        known_sector = np.isfinite(membership[day]) & (membership[day] >= 0)
        known_sector &= membership[day] == np.floor(membership[day])
        for sector in np.unique(membership[day, active & known_sector]):
            members = active & known_sector & (membership[day] == sector)
            observed = members & np.isfinite(returns[day])
            count = int(np.sum(observed))
            if count >= 3 and count >= 0.6 * int(np.sum(members)):
                result[day, observed] = returns[day, observed] - np.mean(returns[day, observed])
    return result


def build_features(panel: Panel) -> ReversalFeatures:
    """按每日历史资格先取市值前1000，再按20日均额取前500，选负相对收益最差20%。

    例如同池10股、负相对收益3股，按相对收益升序选择前2股；并列按证券列号。
    名额为floor(当日池大小*.2)，不补正相对收益。资格要求21个连续市场日valid=1、
    复权close正有限、当前beta有限、历史月末前填total_mv正有限、当日成交额>=20000
    千元及官方上下限价带>7%。20日均额须完整，行业ID必须有限、非负整数；未知行业不占池名额。
    5日收益直接用当日/五市场日前复权close，不使用供应商跨停牌观察日收益字段。
    限价带过滤只是沿用资格约束，不声称排尽历史ST；不读取KDJ/BOLL或未来价格。
    """
    shape = len(panel.calendar), len(panel.codes)
    if (
        not all(shape)
        or panel.calendar != sorted(set(panel.calendar))
        or len(panel.codes) != len(set(panel.codes))
        or any(
            name not in panel.values or panel.values[name].shape != shape
            for name in REQUIRED_FIELDS
        )
    ):
        raise ValueError("reversal_panel_fields_or_axes_invalid")
    values = panel.values
    close, amount, capital = values["close"], values["raw_amount"], values["capitalization"]
    tradable = (values["valid"] == 1) & np.isfinite(close) & (close > 0)
    average = _average_amount20(amount)
    upper, lower = values["raw_up_limit"], values["raw_down_limit"]
    band = np.divide(
        upper - lower,
        upper + lower,
        out=np.full(shape, np.nan),
        where=np.isfinite(upper) & np.isfinite(lower) & (upper > lower) & (lower > 0),
    )
    mainland = np.array([bool(re.fullmatch(r"\d{6}\.(SH|SZ)", code)) for code in panel.codes])
    eligibility = (
        mainland[None, :]
        & complete_history(tradable, 20)
        & np.isfinite(values["beta"])
        & np.isfinite(values["sector_index"])
        & (values["sector_index"] >= 0)
        & (values["sector_index"] == np.floor(values["sector_index"]))
        & np.isfinite(capital)
        & (capital > 0)
        & np.isfinite(amount)
        & (amount >= 20000)
        & np.isfinite(average)
        & (average > 0)
        & (band > 0.07)
    )
    large_cap_pool, universe = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for day in range(shape[0]):
        columns = np.flatnonzero(eligibility[day])
        largest = columns[np.lexsort((columns, -capital[day, columns]))[:1000]]
        large_cap_pool[day, largest] = True
        liquid = largest[np.lexsort((largest, -average[day, largest]))[:500]]
        universe[day, liquid] = True
    returns = np.full(shape, np.nan)
    returns[5:] = (
        np.divide(
            close[5:],
            close[:-5],
            out=np.full(close[5:].shape, np.nan),
            where=np.isfinite(close[5:])
            & (close[5:] > 0)
            & np.isfinite(close[:-5])
            & (close[:-5] > 0),
        )
        - 1
    )
    relative = _relative_returns(returns, universe, values["sector_index"])
    entry_mask = np.zeros(shape, dtype=bool)
    for day in range(shape[0]):
        negative = np.flatnonzero(universe[day] & np.isfinite(relative[day]) & (relative[day] < 0))
        ranked = negative[np.lexsort((negative, relative[day, negative]))]
        quota = int(np.sum(universe[day])) // 5
        entry_mask[day, ranked[:quota]] = True
    scores = -relative
    return ReversalFeatures(
        eligibility, large_cap_pool, universe, average, returns, relative, entry_mask, scores
    )


def _matched_control(
    panel: Panel,
    features: ReversalFeatures,
) -> tuple[NDArray[np.bool_], FloatArray, tuple[tuple[int, int, int], ...]]:
    """按处理股分数降序逐一匹配同日同池同行业的非处理股，不复用且不跨行业补缺。

    距离为abs(log市值差)+abs(log20日均额差)，并列按列号；例如处理股A最接近B，
    则B分数沿用A。匹配只使用市值、流动性及行业，不读取对照股收益或未来数据。
    """
    mask = np.zeros(features.entry_mask.shape, dtype=bool)
    scores = np.full(mask.shape, np.nan)
    pairs = []
    capital, membership = panel.values["capitalization"], panel.values["sector_index"]
    for day in range(len(panel.calendar)):
        treatments = np.flatnonzero(features.entry_mask[day])
        ranked = treatments[np.lexsort((treatments, -features.scores[day, treatments]))]
        available = features.universe[day] & ~features.entry_mask[day]
        for treatment in ranked:
            choices = np.flatnonzero(available & (membership[day] == membership[day, treatment]))
            if not len(choices):
                continue
            distances = np.abs(
                np.log(capital[day, choices]) - np.log(capital[day, treatment])
            ) + np.abs(
                np.log(features.average_amount20[day, choices])
                - np.log(features.average_amount20[day, treatment])
            )
            control = int(choices[np.lexsort((choices, distances))[0]])
            available[control] = False
            mask[day, control] = True
            scores[day, control] = features.scores[day, treatment]
            pairs.append((day, int(treatment), control))
    return mask, scores, tuple(pairs)


def build_strategies(
    panel: Panel,
    features: ReversalFeatures | None = None,
) -> list[ReversalStrategy]:
    """返回reversal/delayed/cost_stress/matched_control四个固定配置，不扫描阈值。

    例基础滑点.0005、费用倍数1；压力组滑点.002、费用倍数2且候选完全相同。
    延迟仅把原mask和scores整体后移一日，首日空，source_days保留原日；不再筛延迟日。
    全部5日持有、单票5%/行业25%/总90%、关闭固定价格退出，无额外均线或摆动退出。
    """
    if features is None:
        features = build_features(panel)
    expected_shape = len(panel.calendar), len(panel.codes)
    if any(
        getattr(features, field).shape != expected_shape for field in features.__dataclass_fields__
    ):
        raise ValueError("reversal_feature_shape_mismatch")
    policy = PortfolioPolicy(
        name="reversal",
        switching=False,
        stage_allocation=False,
        force_rebalance=False,
        maximum_holding=5,
        position_weight=0.05,
        sector_weight=0.25,
        fixed_price_exits=False,
        slippage_rate=0.0005,
        non_stage_allocation=0.9,
        fee_multiplier=1,
    )
    source_days = np.arange(len(panel.calendar), dtype=np.int64)
    base = ReversalStrategy("reversal", features.entry_mask, features.scores, source_days, policy)
    delayed_mask = np.zeros(features.entry_mask.shape, dtype=bool)
    delayed_scores = np.full(features.scores.shape, np.nan)
    delayed_mask[1:] = features.entry_mask[:-1]
    delayed_scores[1:] = features.scores[:-1]
    delayed = ReversalStrategy(
        "delayed", delayed_mask, delayed_scores, source_days - 1, replace(policy, name="delayed")
    )
    stress = replace(
        base,
        name="cost_stress",
        policy=replace(policy, name="cost_stress", slippage_rate=0.002, fee_multiplier=2),
    )
    matched_mask, matched_scores, pairs = _matched_control(panel, features)
    control = ReversalStrategy(
        "matched_control",
        matched_mask,
        matched_scores,
        source_days,
        replace(policy, name="matched_control"),
        matched_pairs=pairs,
    )
    return [base, delayed, stress, control]


def candidate_orders(strategy: ReversalStrategy, day: int) -> list[tuple[int, str, float]]:
    """按冻结分数返回候选，完全不查询发单日Panel，保留延迟策略的原信号日资格。

    例如原日证券0分数.04、延迟日资格改变，延迟候选仍为(0,'short_reversal',.04)。
    所有候选只允许其发单后的唯一一次开盘，adapter另负责现金/交易限制与挂单撤销。
    """
    if type(day) is not int or not 0 <= day < len(strategy.source_days):
        raise ValueError("reversal_candidate_day_invalid")
    columns = np.flatnonzero(strategy.entry_mask[day] & np.isfinite(strategy.scores[day]))
    ranked = columns[np.lexsort((columns, -strategy.scores[day, columns]))]
    return [
        (int(column), "short_reversal", float(strategy.scores[day, column])) for column in ranked
    ]
