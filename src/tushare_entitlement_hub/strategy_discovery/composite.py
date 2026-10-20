"""多策略共享一份现金账本，持仓退出规则绑定入场来源，不能事后切换。"""

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution, Holding
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, exit_reason, simulate_portfolio
from .contracts import DEFAULT_POLICY, StrategyDefinition
from .early_failure import EarlyFailure
from .execution import (
    additional_exit_reason,
    entry_gap_allowed,
    intraday_fixed_exit,
    strategy_candidates,
)


def simulate_composite(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    definitions: tuple[StrategyDefinition, ...],
    *,
    name: str,
    policy: PortfolioPolicy = DEFAULT_POLICY,
    daily_quota: int = 20,
    capital: float = 1_000_000,
    start: str = "20211018",
    failure_policies: dict[str, EarlyFailure] | None = None,
) -> dict[str, Any]:
    """按definitions优先级合并候选，以同一现金/行业/总仓限制成交。

    例(A,B)同日同时选中证券X时只归A，退出持续使用A的止损、止盈和结构规则；
    不因第二天B信号出现就改持仓身份。daily_quota是每策略当日候选数量上限，
    不是每策略资金配额；策略间通过总仓与行业限制竞争剩余现金。持仓不重复。
    policy只拥有共享仓位和费用；固定价格/持有期限退出由入场策略hook处理，
    避免组合默认的8%止盈覆盖另一策略的2%。阶段仓位可用policy启用，按前日
    状态执行，force_rebalance=False时只限制新买入。原账本不作任何修改。
    """
    shape = len(panel.calendar), len(panel.codes)
    if (
        not definitions
        or not name
        or len({item.name for item in definitions}) != len(definitions)
        or any(item.entry_mask.shape != shape for item in definitions)
        or type(daily_quota) is not int
        or daily_quota < 1
    ):
        raise ValueError("invalid_composite_definition")
    by_name = {item.name: item for item in definitions}
    early_policies = failure_policies or {}
    if set(early_policies) - set(by_name):
        raise ValueError("unknown_early_failure_strategy")
    routes: dict[tuple[int, int], StrategyDefinition] = {}
    shared = replace(
        policy,
        name=name,
        fixed_price_exits=False,
        maximum_holding=len(panel.calendar) + 1,
        trailing=0,
        early_exit=0,
    )

    def candidates(
        current: Panel, day: int, stage: int, current_policy: PortfolioPolicy
    ) -> list[tuple[int, str, float]]:
        """收盘按预先给定顺序合并并冻结来源；例同股仅保留第一策略的一笔订单。"""
        result: list[tuple[int, str, float]] = []
        occupied: set[int] = set()
        for definition in definitions:
            for security, _, _ in strategy_candidates(current, day, definition)[:daily_quota]:
                if security not in occupied:
                    routes[day, security] = definition
                    occupied.add(security)
                    result.append((security, definition.name, float(-len(result))))
        return result

    def entry_filter(current: Panel, confirmation: int, day: int, security: int) -> bool:
        """按原入场策略有效期和开盘追价判断；例A允许3日，B只允许1日。"""
        definition = routes[confirmation, security]
        return day <= confirmation + definition.maximum_pending_sessions and entry_gap_allowed(
            current, confirmation, day, security, definition.maximum_entry_gap
        )

    def cancel_pending(current: Panel, confirmation: int, day: int, security: int) -> bool:
        """保留入场策略的收盘取消规则；例到第1日仍未买到则取消单日期订单。"""
        definition = routes[confirmation, security]
        invalidation = (
            definition.entry_invalidation_mask
            if definition.entry_invalidation_mask is not None
            else definition.exit_mask
        )
        return bool(
            (
                definition.maximum_pending_sessions < 3
                and day >= confirmation + definition.maximum_pending_sessions
            )
            or (invalidation is not None and invalidation[day, security])
        )

    def close_exit(
        current: Panel, holding: Holding, day: int, current_policy: PortfolioPolicy
    ) -> str:
        """按冻结来源计算收盘固定/结构退出；例高胜率支路保持2%止盈。"""
        definition = by_name[holding.family]
        adjusted = (
            current.values["raw_close"][day, holding.security]
            * current.values["raw_adj_factor"][day, holding.security]
        )
        reason = exit_reason(
            holding,
            day,
            adjusted,
            replace(definition.policy, trailing=0),
            fixed_exits=not definition.intraday_exits,
        )
        failure = early_policies.get(definition.name)
        if (
            not reason
            and failure is not None
            and failure.triggered(day - holding.entry + 1, holding.entry_adjusted, float(adjusted))
        ):
            return "early_failure"
        return reason or additional_exit_reason(current, holding, day, definition)

    def intraday_exit(
        current: Panel, holding: Holding, day: int, current_policy: PortfolioPolicy
    ) -> tuple[float, str] | None:
        """仅原策略显式启用盘中模型时委托；例收盘策略返回None保持原时点。"""
        definition = by_name[holding.family]
        return (
            intraday_fixed_exit(current, holding, day, definition.policy)
            if definition.intraday_exits
            else None
        )

    result = simulate_portfolio(
        panel,
        states,
        events,
        shared,
        capital=capital,
        start=start,
        candidate_provider=candidates,
        entry_filter=entry_filter,
        cancel_pending_provider=cancel_pending,
        exit_provider=close_exit,
        intraday_exit_provider=intraday_exit,
    )
    result["strategy_routing"] = {
        "priority": [item.name for item in definitions],
        "daily_quota_per_strategy": daily_quota,
        "exit_rule": "frozen_entry_source",
        "shared_cash": True,
        "intraday_model": "daily_bar_proxy" if any(x.intraday_exits for x in definitions) else None,
    }
    return result
