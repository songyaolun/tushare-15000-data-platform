"""独立候选交易标签，复用实际退出、分红和费用，禁止作为当日特征。"""

from typing import Any

import numpy as np

from ..regime_research.accounting import (
    Distribution,
    DistributionBook,
    Holding,
    affordable_shares,
    transaction_fee,
)
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, exit_reason
from .execution import entry_gap_allowed


def independent_trade_label(
    panel: Panel,
    confirmation: int,
    security: int,
    events: list[Distribution],
    policy: PortfolioPolicy,
    *,
    budget: float = 50_000,
    maximum_gap: float = 0.02,
) -> dict[str, Any]:
    """回放一个已通过信号日资格检查的候选，返回净收益、期限及成交状态。

    输入面板、信号日、证券列、该证券分红和固定退出政策；例信号0、次开盘10、
    次收盘10.3触发2%止盈，日2开盘11卖出，输出status='closed'及费用后收益。
    假定单笔预算5万元、独立资金，不模拟同股占用、行业和组合仓位竞争。
    不可买入返回unfilled；样本结束未卖出返回censored，收益均为None，不能补零。
    收盘触发后下一可成交开盘卖，未上市送股延迟卖出；现金红利预扣20%。
    标签使用未来数据，只能用于离线研究，不可回流当日评分特征。
    """
    if (
        not 0 <= confirmation < len(panel.calendar)
        or not 0 <= security < len(panel.codes)
        or not np.isfinite(budget)
        or budget <= 0
        or not np.isfinite(maximum_gap)
        or maximum_gap < 0
        or any(event.security != security for event in events)
        or policy.trailing != 0
        or policy.early_exit != 0
    ):
        raise ValueError("invalid_trade_label")
    entry = confirmation + 1
    empty = {"return": None, "entry": entry, "exit": None, "holding_sessions": None}
    if entry >= len(panel.calendar):
        return {**empty, "status": "censored"}
    values = panel.values
    opening = float(values["raw_open"][entry, security])
    upper = float(values["raw_up_limit"][entry, security])
    if (
        not np.isfinite(opening)
        or opening <= 0
        or not np.isfinite(upper)
        or opening >= upper
        or not entry_gap_allowed(panel, confirmation, entry, security, maximum_gap)
    ):
        return {**empty, "status": "unfilled"}
    price = min(opening * (1 + policy.slippage_rate), upper)
    shares = affordable_shares(
        budget,
        price,
        panel.codes[security],
        panel.calendar[entry],
        fee_multiplier=policy.fee_multiplier,
    )
    if not shares:
        return {**empty, "status": "unfilled"}
    notional = shares * price
    cost = (
        notional + transaction_fee(notional, panel.calendar[entry], False) * policy.fee_multiplier
    )
    adjusted = price * values["raw_adj_factor"][entry, security]
    holding = Holding(
        security,
        confirmation,
        entry,
        shares,
        cost,
        price,
        adjusted,
        adjusted,
        "label",
        0,
        0,
        observed=entry,
    )
    book = DistributionBook(events)
    for day in range(entry, len(panel.calendar)):
        book.advance(day)
        opening = float(values["raw_open"][day, security])
        lower = float(values["raw_down_limit"][day, security])
        if (
            day > entry
            and holding.sell_reason
            and np.isfinite(opening)
            and opening > 0
            and np.isfinite(lower)
            and opening > lower
            and holding.locked < 1e-8
        ):
            proceeds = max(opening * (1 - policy.slippage_rate), lower) * holding.shares
            proceeds -= transaction_fee(proceeds, panel.calendar[day], True) * policy.fee_multiplier
            return {
                "status": "closed",
                "entry": entry,
                "exit": day,
                "return": (proceeds + holding.dividend) / cost - 1,
                "holding_sessions": day - entry + 1,
                "reason": holding.sell_reason,
                "cost": cost,
                "proceeds": proceeds,
                "dividend": holding.dividend,
            }
        if not holding.sell_reason:
            closing = values["raw_close"][day, security] * values["raw_adj_factor"][day, security]
            holding.sell_reason = exit_reason(holding, day, closing, policy)
        book.record(day, {security: holding})
    return {**empty, "status": "censored"}
