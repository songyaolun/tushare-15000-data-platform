"""冻结的广度仓位研究候选：收盘计算状态，成交账本在次开盘读取。"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray
from .contracts import DEFAULT_POLICY, StrategyDefinition
from .evidence_scoring import priority_scores
from .signal_redesign import low_score_candidate, top_candidates


def breadth_low_score_candidate(
    eligible: NDArray[np.bool_],
    technical_pass: NDArray[np.bool_],
    production_scores: FloatArray,
    breadth: FloatArray,
    *,
    candidate_count: int = 10,
    breadth_threshold: float = 0.4,
    weak_allocation: float = 0.15,
    strong_allocation: float = 0.9,
) -> tuple[StrategyDefinition, NDArray[np.int8], PortfolioPolicy]:
    """返回低分策略、当日广度状态和共享账户仓位规则，不读取未来收益。

    输入同轴发布资格、技术风控、旧分数及每日站上20日均线的股票比例。
    默认每天按旧分升序取10只，2%止盈、8%止损、20日到期。广度达到40%
    时新买入总仓上限90%，否则15%；不主动减仓，缺广度按弱市处理。
    例breadth=[.39,.4,nan]输出状态[0,1,0]；账户次开盘使用前日状态，
    此处不额外移位。返回值用于simulate_composite，不可把状态当原五阶段。
    这是全样本选择后的研究候选，不代表已验证的上涨概率或实盘收益。
    """
    if (
        breadth.ndim != 1
        or eligible.ndim != 2
        or len(breadth) != len(eligible)
        or any(
            isinstance(value, bool) or not np.isfinite(value)
            for value in (breadth_threshold, weak_allocation, strong_allocation)
        )
        or not 0 < breadth_threshold < 1
        or not 0 <= weak_allocation <= strong_allocation <= 1
        or np.any(np.isfinite(breadth) & ((breadth < 0) | (breadth > 1)))
    ):
        raise ValueError("invalid_breadth_candidate")
    definition = low_score_candidate(eligible, technical_pass, production_scores)
    definition = replace(
        definition,
        name=f"low_top{candidate_count}",
        entry_mask=top_candidates(eligible & technical_pass, definition.scores, candidate_count),
    )
    states = (np.isfinite(breadth) & (breadth >= breadth_threshold)).astype(np.int8)
    policy = replace(
        DEFAULT_POLICY,
        stage_allocation=True,
        force_rebalance=False,
        allocations=(weak_allocation,) + (strong_allocation,) * 4,
    )
    return definition, states, policy


def high_priority_breadth_candidate(
    eligible: NDArray[np.bool_],
    technical_pass: NDArray[np.bool_],
    production_scores: FloatArray,
    breadth: FloatArray,
) -> tuple[StrategyDefinition, NDArray[np.int8], PortfolioPolicy]:
    """把冻结广度候选输出为0到100的高分优先接口，保持全部交易规则。

    输入日×证券发布资格、价量风控、旧分数及每日广度，输出策略/状态/仓位。
    例合格旧分[10,20,30]对应新优先级[100,50,0]，从高到低选Top10。
    并列仍并列；不合格或缺旧分为NaN。它只是现有反向排序的等价表达，
    没有重新拟合权重或产生额外收益，100分也不代表100%胜率。固定买卖、
    广度阈值与仓位使用breadth_low_score_candidate的已归档默认值。
    """
    definition, states, policy = breadth_low_score_candidate(
        eligible, technical_pass, production_scores, breadth
    )
    return (
        replace(
            definition,
            name="stockops_high_priority_research",
            scores=priority_scores(definition.scores, eligible & technical_pass),
        ),
        states,
        policy,
    )
