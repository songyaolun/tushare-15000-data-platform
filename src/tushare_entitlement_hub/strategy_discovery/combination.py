"""按初始资金固定分配独立子账户，保留真实佣金、整手和各自交易时序。"""

from math import fsum, isclose
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution
from ..regime_research.panel import Panel
from .contracts import StrategyDefinition
from .execution import simulate_strategy


def _combine_daily(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按资产金额合并同一天的子账本，收盘权重与开盘仓位分别使用各自资产基数。

    例如子账户收盘资产60/40、同股权重.2/.5，组合该股权重为.32；
    开盘资产不同则另用open_nav加权，不把收盘盈亏带回开盘仓位。
    """
    reference = rows[0]
    if any(
        (row["day"], row["date"], row["stage"])
        != (reference["day"], reference["date"], reference["stage"])
        for row in rows
    ):
        raise ValueError("sleeve_daily_calendar_mismatch")
    if any(
        not np.isfinite(row[field]) or row[field] <= 0
        for row in rows
        for field in ("nav", "open_nav")
    ):
        raise ValueError("sleeve_daily_assets_must_be_positive_finite")
    combined = {field: reference[field] for field in ("day", "date", "stage")}
    for field in ("nav", "open_nav", "cash", "receivable", "cost"):
        combined[field] = fsum(row[field] for row in rows)
    combined["positions"] = sum(row["positions"] for row in rows)
    for field in ("exposure", "beta"):
        combined[field] = fsum(row[field] * row["nav"] for row in rows) / combined["nav"]
    combined["open_exposure"] = (
        fsum(row["open_exposure"] * row["open_nav"] for row in rows) / combined["open_nav"]
    )
    securities = sorted({security for row in rows for security in row["weights"]})
    combined["weights"] = {
        security: fsum(row["weights"].get(security, 0) * row["nav"] for row in rows)
        / combined["nav"]
        for security in securities
    }
    return combined


def simulate_sleeves(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    definitions: list[StrategyDefinition],
    weights: list[float],
    *,
    capital: float = 1_000_000,
    start: str = "20211018",
) -> dict[str, Any]:
    """按正权重分配初始资金并独立重跑每个策略，输出与simulate_strategy兼容的总账。

    例capital=100000、weights=[.6,.4]分别以60000/40000启动两个账户，
    最低佣金和整手数量各自计算；期间不转移现金、不重平衡子账户资金比例。
    同股市值按实际子NAV相加，positions表示独立持仓批次数；成交增加唯一sleeve。
    weights必须与非空definitions等长、有限且严格为正，总和须在1e-12内等于1。
    """
    if (
        not definitions
        or len(definitions) != len(weights)
        or any(not isinstance(definition, StrategyDefinition) for definition in definitions)
        or not np.isfinite(capital)
        or capital <= 0
    ):
        raise ValueError("invalid_sleeve_configuration")
    allocations = np.asarray(weights, dtype=float)
    if (
        allocations.shape != (len(definitions),)
        or not np.all(np.isfinite(allocations) & (allocations > 0))
        or not isclose(fsum(allocations), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("sleeve_weights_must_be_positive_finite_and_sum_to_one")
    accounts = [
        simulate_strategy(
            panel, states, events, definition, capital=capital * float(weight), start=start
        )
        for definition, weight in zip(definitions, allocations, strict=True)
    ]
    daily = [
        _combine_daily(list(rows))
        for rows in zip(*(account["daily"] for account in accounts), strict=True)
    ]
    sleeves = [
        {
            "sleeve": f"{index}:{definition.name}",
            "strategy": definition.name,
            "weight": float(weight),
            "capital": account["capital"],
        }
        for index, (definition, weight, account) in enumerate(
            zip(definitions, allocations, accounts, strict=True)
        )
    ]
    trades = [
        {**trade, "sleeve": sleeve["sleeve"]}
        for sleeve, account in zip(sleeves, accounts, strict=True)
        for trade in account["trades"]
    ]
    trades.sort(
        key=lambda trade: (trade["entry"], trade["exit"], trade["sleeve"], trade["security"])
    )
    result = {
        "daily": daily,
        "trades": trades,
        "entry_days": sorted(day for account in accounts for day in account["entry_days"]),
        "capital": capital,
        "policy": "fixed_capital_sleeves",
        "sleeves": sleeves,
    }
    for field in (
        "stale_value",
        "fees",
        "turnover",
        "reconciliation_error",
        "open_cost",
        "open_market_value",
    ):
        result[field] = fsum(account[field] for account in accounts)
    for field in ("open_positions", "rejected_budget"):
        result[field] = sum(account[field] for account in accounts)
    return result
