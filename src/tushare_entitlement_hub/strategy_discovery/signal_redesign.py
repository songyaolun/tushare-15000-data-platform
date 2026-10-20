"""将信号、市场准入、确认时点和退出规则拆开的有限研究变体。"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray
from .contracts import DEFAULT_POLICY, StrategyDefinition
from .profit_protection import ProfitProtection


def top_candidates(
    eligible: NDArray[np.bool_], scores: FloatArray, limit: int = 20
) -> NDArray[np.bool_]:
    """按当日分数选最多limit个合格候选，同分按证券轴顺序；缺分不入选。

    例合格分数[1,3,2]、limit=2输出[False,True,True]；无候选日保持全假。
    """
    if (
        eligible.ndim != 2
        or scores.shape != eligible.shape
        or eligible.dtype != np.bool_
        or type(limit) is not int
        or limit < 1
    ):
        raise ValueError("invalid_top_candidate_inputs")
    result = np.zeros(eligible.shape, dtype=bool)
    for day in range(len(eligible)):
        columns = np.flatnonzero(eligible[day] & np.isfinite(scores[day]))
        order = np.argsort(-scores[day, columns], kind="stable")
        result[day, columns[order[:limit]]] = True
    return result


def confirmation_entries(
    source: NDArray[np.bool_],
    closing: FloatArray,
    high: FloatArray,
    low: FloatArray,
    volume: FloatArray,
    mode: str,
    wait: int = 5,
) -> NDArray[np.bool_]:
    """将原始信号延后为首次回踩/再突破确认，确认收盘后才允许次开盘交易。

    例信号收盘10、高10.2、低9.8；下一日低9.95、收10.05且缩量，retest
    当日确认；followthrough需收盘严格突破信号最高。回踩容差1%、缩量20%，
    两模式均不能跌破源日低点；缺报价或超过wait个市场日取消，已有事件
    不被重复信号覆盖。只生成首次确认，不假定在日内最低价成交。
    """
    if (
        source.ndim != 2
        or source.dtype != np.bool_
        or any(value.shape != source.shape for value in (closing, high, low, volume))
        or mode not in {"retest", "followthrough"}
        or type(wait) is not int
        or wait < 1
    ):
        raise ValueError("invalid_confirmation_inputs")
    result = np.zeros(source.shape, dtype=bool)
    active = np.full(source.shape[1], -1, dtype=int)
    columns = np.arange(source.shape[1])
    for day in range(len(source)):
        known = np.ones(source.shape[1], dtype=bool)
        for value in (closing, high, low, volume):
            known &= np.isfinite(value[day]) & (value[day] > 0)
        known &= (low[day] <= closing[day]) & (closing[day] <= high[day])
        occupied = active >= 0
        origin = np.maximum(active, 0)
        original_close = closing[origin, columns]
        valid = occupied & known & (day - active <= wait)
        valid &= low[day] >= low[origin, columns]
        if mode == "retest":
            confirmed = (
                valid
                & (low[day] <= original_close * 1.01)
                & (closing[day] >= original_close)
                & (volume[day] <= volume[origin, columns] * 0.8)
            )
        else:
            confirmed = valid & (closing[day] > high[origin, columns])
        result[day] = confirmed
        active[occupied & (~valid | confirmed)] = -1
        new_source = source[day] & known & ~occupied
        active[new_source] = day
    return result


def exit_variants(
    definition: StrategyDefinition, structural_failure: NDArray[np.bool_]
) -> list[StrategyDefinition]:
    """为固定入场生成六种退出，不变更入场资格、排名与追价。

    例base返回8/8控制、4/8、12/6、持10日、结构失败+6%止损、激活后跟踪。
    所有固定退出为收盘触发次开盘，不承诺止损价成交；结构退出使用独立
    entry_invalidation_mask全假，避免误把新增卖出条件反过来删掉入场。
    """
    if (
        structural_failure.shape != definition.entry_mask.shape
        or structural_failure.dtype != np.bool_
    ):
        raise ValueError("invalid_structural_failure")
    base = replace(
        definition,
        exit_mask=None,
        trailing_stop=0,
        profit_protection=None,
        entry_invalidation_mask=np.zeros(definition.entry_mask.shape, dtype=bool),
        policy=replace(
            definition.policy,
            take_profit=0.08,
            stop_loss=0.08,
            maximum_holding=30,
            fixed_price_exits=True,
            take_profit_enabled=True,
            early_exit=0,
        ),
    )
    variants = [replace(base, name=definition.name + "__exit_8_8")]
    for label, target, stop, holding in [("4_8", 0.04, 0.08, 20), ("12_6", 0.12, 0.06, 30)]:
        variants.append(
            replace(
                base,
                name=definition.name + "__exit_" + label,
                policy=replace(
                    base.policy, take_profit=target, stop_loss=stop, maximum_holding=holding
                ),
            )
        )
    variants.append(
        replace(
            base,
            name=definition.name + "__exit_time10",
            policy=replace(base.policy, fixed_price_exits=False, maximum_holding=10),
        )
    )
    variants.append(
        replace(
            base,
            name=definition.name + "__exit_structure",
            exit_mask=structural_failure,
            policy=replace(base.policy, stop_loss=0.06, take_profit_enabled=False),
        )
    )
    variants.append(
        replace(
            base,
            name=definition.name + "__exit_protect",
            policy=replace(base.policy, stop_loss=0.06, take_profit_enabled=False),
            profit_protection=ProfitProtection(0.04, trailing_drawdown=0.03),
        )
    )
    return variants


def low_score_candidate(
    eligible: NDArray[np.bool_],
    technical_pass: NDArray[np.bool_],
    production_scores: FloatArray,
    *,
    target: float = 0.02,
    stop: float = 0.08,
) -> StrategyDefinition:
    """构造研究用低分候选账户，与旧高分排序保持相同资格及价量风控。

    输入当日发布资格、技术风控通过矩阵和旧分数，输出Top20独立策略定义。
    例如分数[10,20,30]倾向10分，次开盘高开超过2%取消；默认2%止盈、8%
    止损、20市场日到期均为收盘触发次开盘。这是样本内发现后的研究候选，
    不是校准后的上涨概率或生产推荐；不读取任何未来收益标签或账户结果。
    """
    if (
        technical_pass.shape != eligible.shape
        or technical_pass.dtype != np.bool_
        or eligible.dtype != np.bool_
        or isinstance(target, bool)
        or isinstance(stop, bool)
        or not 0 < target < 1
        or not 0 < stop < 1
    ):
        raise ValueError("invalid_low_score_candidate")
    scores = -np.asarray(production_scores, dtype=float)
    mask = top_candidates(eligible & technical_pass, scores)
    return StrategyDefinition(
        "stockops_low_score_research",
        "reverse_rank",
        mask,
        scores,
        replace(DEFAULT_POLICY, take_profit=target, stop_loss=stop, maximum_holding=20),
        maximum_entry_gap=0.02,
        maximum_pending_sessions=1,
    )
