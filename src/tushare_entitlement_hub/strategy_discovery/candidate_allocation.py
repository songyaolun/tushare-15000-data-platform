"""候选分配的独立对照：共同资格前置、排除持仓前置、卖单条件预备。"""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution, Holding
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from .contracts import StrategyDefinition
from .entry_timing import OpeningGapRule, opening_gap_allowed
from .execution import (
    additional_exit_reason,
    common_candidate_eligibility,
    entry_gap_allowed,
)
from .signal_redesign import top_candidates

AllocationMode = Literal["baseline", "common_first", "holdings_first", "exit_reserve"]


def simulate_candidate_allocation(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    definition: StrategyDefinition,
    account_policy: PortfolioPolicy,
    *,
    mode: AllocationMode,
    count: int = 10,
    capital: float = 1_000_000,
    start: str = "20211018",
    opening_gap_rule: OpeningGapRule | None = None,
) -> dict[str, Any]:
    """用同一完整候选池和固定评分，独立改变候选生成的一个环节。

    definition.entry_mask必须是未截TopK的发布资格与技术风控交集。例common_first
    先剔除当日共同资格失败者，再选10只；holdings_first只前置持仓/待单排除，
    共同资格仍在Top10后检查；exit_reserve保留原Top10，只为已有卖单准备候选。
    预备单依赖实际卖出笔数激活，不能提前释放现金，挂单仅次日有效。
    返回逐日实际候选选择和原账户结果，方便核对信号、成交、费用与名额。
    仅支持本实验的单日挂单、收盘固定退出策略，拒绝盘中模型，避免隐式改变买卖。
    opening_gap_rule是附加的次开盘价格区间，例如(-.03,.01)排除大低开和追涨；
    与definition.maximum_entry_gap取交集，默认None精确保留旧行为。
    """
    if (
        mode not in {"baseline", "common_first", "holdings_first", "exit_reserve"}
        or type(count) is not int
        or count < 1
        or definition.entry_mask.shape != (len(panel.calendar), len(panel.codes))
        or definition.maximum_pending_sessions != 1
        or definition.intraday_exits
    ):
        raise ValueError("invalid_candidate_allocation")
    policy = replace(
        definition.policy,
        name=definition.name,
        stage_allocation=account_policy.stage_allocation,
        force_rebalance=account_policy.force_rebalance,
        allocations=account_policy.allocations,
        non_stage_allocation=account_policy.non_stage_allocation,
        position_weight=account_policy.position_weight,
        sector_weight=account_policy.sector_weight,
        fee_multiplier=account_policy.fee_multiplier,
        slippage_rate=account_policy.slippage_rate,
        trailing=0,
    )
    selections: list[dict[str, Any]] = []

    def candidates(
        current: Panel,
        day: int,
        stage: int,
        current_policy: PortfolioPolicy,
        holdings: frozenset[int],
        pending: frozenset[int],
    ) -> list[tuple[int, str, float]]:
        """只用当前收盘事实选股并留痕；例holdings_first将已持仓证券剔除后取Top10。"""
        eligible = definition.entry_mask[day].copy()
        common = common_candidate_eligibility(current, day)
        if mode == "common_first":
            eligible &= common
        if mode == "holdings_first":
            eligible[list(holdings | pending)] = False
        selected = top_candidates(eligible[None, :], definition.scores[day : day + 1], count)[0]
        invalidation = (
            definition.entry_invalidation_mask
            if definition.entry_invalidation_mask is not None
            else definition.exit_mask
        )
        qualified = selected & common
        if invalidation is not None:
            qualified &= ~invalidation[day]
        columns = np.flatnonzero(qualified)
        columns = columns[np.argsort(-definition.scores[day, columns], kind="stable")]
        selections.append(
            {
                "day": day,
                "selected": np.flatnonzero(selected).tolist(),
                "qualified": columns.tolist(),
                "held_selected": len(set(np.flatnonzero(selected)) & holdings),
                "common_rejected": int((selected & ~common).sum()),
            }
        )
        return [
            (int(column), definition.name, float(definition.scores[day, column]))
            for column in columns
        ]

    def entry_filter(current: Panel, confirmation: int, day: int, security: int) -> bool:
        """次日开盘检查追价，不读取日后结果；例超过信号价2%则取消该订单。"""
        return (
            day == confirmation + 1
            and entry_gap_allowed(
                current, confirmation, day, security, definition.maximum_entry_gap
            )
            and (
                opening_gap_rule is None
                or opening_gap_allowed(current, confirmation, day, security, opening_gap_rule)
            )
        )

    def cancel_pending(current: Panel, confirmation: int, day: int, security: int) -> bool:
        """次日收盘撤销未成交单，阻塞预备单也不延期；例day=confirmation+1返回真。"""
        return day >= confirmation + 1

    def close_exit(
        current: Panel,
        holding: Holding,
        day: int,
        current_policy: PortfolioPolicy,
    ) -> str:
        """固定价格退出由原账本处理，保留策略追加退出；例无结构条件时为空字符串。"""
        return additional_exit_reason(current, holding, day, definition)

    result = simulate_portfolio(
        panel,
        states,
        events,
        policy,
        capital=capital,
        start=start,
        context_candidate_provider=candidates,
        reserve_exit_candidates=mode == "exit_reserve",
        entry_filter=entry_filter,
        cancel_pending_provider=cancel_pending,
        exit_provider=close_exit,
    )
    result["candidate_selections"] = selections
    result["candidate_allocation"] = mode
    return result
