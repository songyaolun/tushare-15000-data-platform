"""收盘决策、下一合格开盘成交的资金受限组合模拟。"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .accounting import (
    Distribution,
    DistributionBook,
    Holding,
    affordable_shares,
    transaction_fee,
)
from .panel import Panel


@dataclass(frozen=True)
class PortfolioPolicy:
    """策略与仓位配置；例默认上涨90%、修复60%、过渡30%、下跌10%、风险0%。"""

    name: str = "switch"
    switching: bool = True
    mixture: bool = False
    stage_allocation: bool = True
    force_rebalance: bool = True
    allow_weak_entries: bool = False
    ranking: str = "original"
    advance_family: str = ""
    allocations: tuple[float, ...] = (0.3, 0.9, 0.6, 0.1, 0.0)
    position_weight: float = 0.05
    sector_weight: float = 0.25
    take_profit: float = 0.04
    stop_loss: float = 0.10
    maximum_holding: int = 20
    early_exit: int = 0
    trailing: float = 0
    beta_tilt: float = 0
    recovery_family: str = "volume_reversal__sector_group"

    def __post_init__(self) -> None:
        """拒绝杠杆、负权重和不合法退出配置；例如position_weight=0抛ValueError。"""
        if not (
            len(self.allocations) == 5
            and all(0 <= value <= 1 for value in self.allocations)
            and 0 < self.position_weight <= self.sector_weight <= 1
            and 0 < self.take_profit < 1
            and 0 < self.stop_loss < 1
            and self.maximum_holding >= 1
            and self.early_exit >= 0
            and 0 <= self.trailing < 1
            and self.ranking in {"original", "oversold", "liquid"}
        ):
            raise ValueError("invalid_portfolio_policy")


def entry_candidates(
    panel: Panel, day: int, stage: int, policy: PortfolioPolicy
) -> list[tuple[int, str, float]]:
    """按信号日信息生成稳定排序，不读取未来报价；输出(证券列号,策略族,评分)。

    例上涨阶段启用趋势回调与突破；风险/下跌切换策略不新买，原策略对照仍独立运行。
    入场要求已知行业、历史Beta和市值，成交额至少2000万元并排除约5%限价带。
    """
    values = panel.values
    families: tuple[str, ...]
    if policy.mixture:
        families = (
            "volume_reversal__sector_group",
            "boll_reentry__sector_group",
            "trend_pullback__sector_group",
            "trend_breakout__sector_hot_group",
        )
    elif policy.switching:
        if stage in (3, 4) and not policy.allow_weak_entries:
            return []
        families = (
            ("trend_pullback__sector_group", "trend_breakout__sector_hot_group")
            if stage == 1
            else (policy.recovery_family,)
            if stage == 2
            else ("boll_reentry__sector_group",)
        )
        if stage == 1 and policy.advance_family:
            families = (policy.advance_family,)
        if stage in (3, 4):
            families = (policy.recovery_family,)
    else:
        families = (policy.recovery_family,)
    mask = np.logical_or.reduce([panel.signals[name][day] for name in families])
    known = np.isfinite(values["beta"][day]) & np.isfinite(values["capitalization"][day])
    band = (values["raw_up_limit"][day] - values["raw_down_limit"][day]) / (
        values["raw_up_limit"][day] + values["raw_down_limit"][day]
    )
    eligible = (
        mask
        & known
        & (values["sector_index"][day] >= 0)
        & (values["raw_amount"][day] >= 20000)
        & (band > 0.07)
    )
    columns = np.flatnonzero(eligible)
    scores = (
        np.nan_to_num(values["sector_heat"][day], nan=0)
        + np.nan_to_num(values["sector_breadth"][day], nan=0)
        - np.nan_to_num(values["kdj_j"][day], nan=100) / 200
        + (policy.beta_tilt * values["beta"][day] if stage == 1 else 0)
    )
    if policy.ranking == "oversold":
        scores = (
            -np.nan_to_num(values["return20"][day], nan=0)
            + np.nan_to_num(values["sector_breadth"][day], nan=0) * 0.2
            - np.nan_to_num(values["kdj_j"][day], nan=100) / 500
        )
    elif policy.ranking == "liquid":
        scores = np.log(np.maximum(1, values["raw_amount"][day]))
    ranked = columns[np.lexsort((columns, -scores[columns]))]
    return [
        (
            int(column),
            next(name for name in families if panel.signals[name][day, column]),
            float(scores[column]),
        )
        for column in ranked
    ]


def exit_reason(holding: Holding, day: int, adjusted_close: float, policy: PortfolioPolicy) -> str:
    """按入场时策略和当前已知收盘判断退出，不因阶段切换放宽止损。

    例成本复权10、收盘8.9、stop_loss=.1输出stop；停牌无价格仍可触发持有到期。
    """
    held = day - holding.entry + 1
    if np.isfinite(adjusted_close):
        multiple = adjusted_close / holding.entry_adjusted
        holding.peak_adjusted = max(holding.peak_adjusted, adjusted_close)
        if multiple <= 1 - policy.stop_loss:
            return "stop"
        if policy.trailing and holding.stage == 1:
            if adjusted_close / holding.peak_adjusted <= 1 - policy.trailing:
                return "trailing"
        elif multiple >= 1 + policy.take_profit:
            return "profit"
        if policy.early_exit and held >= policy.early_exit and multiple <= 1:
            return "failed_rebound"
    return "time" if held >= policy.maximum_holding else ""


def simulate_portfolio(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    policy: PortfolioPolicy,
    *,
    capital: float = 1_000_000,
    start: str = "20211018",
) -> dict[str, Any]:
    """完整逐日资金回放，返回净值、交易、敞口、成交日和陈旧报价诊断。

    例100万元账户、5%单股权重，买单预算不超过5万元且受现金和行业上限限制。
    三日订单有效期；T+1、限价和停牌限制；卖出收益留在现金，不以未来净值定仓。
    原始股数处理现金/送股，不支持配股认购；零交易不伪造胜率。
    """
    if capital <= 0 or len(states) != len(panel.calendar):
        raise ValueError("invalid_simulation_inputs")
    values, calendar = panel.values, panel.calendar
    begin = next((i for i, day in enumerate(calendar) if day >= start), len(calendar))
    cash = capital
    holdings: dict[int, Holding] = {}
    pending: dict[int, tuple[int, str, float, int, float]] = {}
    book = DistributionBook(events)
    daily, trades, entry_days = [], [], []
    turnover, fees, rejected = 0.0, 0.0, 0
    for day in range(begin, len(calendar)):
        previous_fees = fees
        slippage = 0.0
        cash += book.advance(day)
        # 先卖后买，所有卖出指令来自之前收盘；未上市送股存在时保守延后整笔退出。
        for security, holding in list(holdings.items()):
            opening = values["raw_open"][day, security]
            lower = values["raw_down_limit"][day, security]
            can_sell = (
                day > holding.entry
                and np.isfinite(opening)
                and opening > 0
                and np.isfinite(lower)
                and opening > lower
                and holding.locked < 1e-8
            )
            if holding.sell_reason and can_sell:
                price = max(opening * 0.9995, lower)
                notional = price * holding.shares
                slippage += (opening - price) * holding.shares
                fee = transaction_fee(notional, calendar[day], True)
                cash += notional - fee
                fees += fee
                turnover += notional
                trades.append(
                    {
                        "security": security,
                        "ts_code": panel.codes[security],
                        "confirmation": holding.confirmation,
                        "entry": holding.entry,
                        "exit": day,
                        "family": holding.family,
                        "stage": holding.stage,
                        "cost": holding.cost,
                        "proceeds": notional - fee,
                        "dividend": holding.dividend,
                        "return": (notional - fee + holding.dividend) / holding.cost - 1,
                        "reason": holding.sell_reason,
                        "shares": holding.shares,
                    }
                )
                del holdings[security]
        for security, (confirmation, family, score, stage, budget) in list(pending.items()):
            if day > confirmation + 3:
                del pending[security]
                continue
            opening, upper = (
                values["raw_open"][day, security],
                values["raw_up_limit"][day, security],
            )
            if (
                not np.isfinite(opening)
                or opening <= 0
                or not np.isfinite(upper)
                or opening >= upper
            ):
                continue
            price = min(opening * 1.0005, upper)
            # 日内只使用开盘与此前收盘计算可投资资产和行业已有暴露。
            marked = {
                column: position.shares
                * (
                    values["raw_open"][day, column]
                    if np.isfinite(values["raw_open"][day, column])
                    else position.price
                )
                for column, position in holdings.items()
            }
            assets = cash + sum(marked.values()) + book.outstanding()
            current_stage = int(states[day - 1])
            target = policy.allocations[current_stage] if policy.stage_allocation else 0.9
            sector = values["sector_index"][confirmation, security]
            sector_value = sum(
                amount
                for column, amount in marked.items()
                if values["sector_index"][day - 1, column] == sector
            )
            allowance = min(
                budget,
                cash,
                target * assets - sum(marked.values()),
                policy.sector_weight * assets - sector_value,
            )
            shares = affordable_shares(allowance, price, panel.codes[security], calendar[day])
            if shares:
                notional = shares * price
                slippage += (price - opening) * shares
                fee = transaction_fee(notional, calendar[day], False)
                cash -= notional + fee
                fees += fee
                turnover += notional
                adjusted = price * values["raw_adj_factor"][day, security]
                holdings[security] = Holding(
                    security,
                    confirmation,
                    day,
                    shares,
                    notional + fee,
                    price,
                    adjusted,
                    adjusted,
                    family,
                    stage,
                    score,
                    observed=day,
                )
                entry_days.append(day)
            else:
                rejected += 1
            del pending[security]
        opening_value = sum(
            position.shares
            * (
                values["raw_open"][day, column]
                if np.isfinite(values["raw_open"][day, column])
                else position.price
            )
            for column, position in holdings.items()
        )
        opening_assets = cash + opening_value + book.outstanding()
        for security, holding in holdings.items():
            closing = values["raw_close"][day, security]
            if np.isfinite(closing) and closing > 0:
                holding.price, holding.observed = float(closing), day
            if not holding.sell_reason:
                adjusted = closing * values["raw_adj_factor"][day, security]
                holding.sell_reason = exit_reason(holding, day, adjusted, policy)
        invested = sum(holding.shares * holding.price for holding in holdings.values())
        assets = cash + invested + book.outstanding()
        if cash < -1e-6 or not np.isfinite(assets) or assets <= 0:
            raise ValueError("accounting_invariant_failed")
        stage = int(states[day])
        target = policy.allocations[stage] if policy.stage_allocation else 0.9
        # 只在收盘生成减仓订单，不能把当日风险信号用于当天开盘。
        if policy.stage_allocation and policy.force_rebalance:
            projected = invested - sum(
                position.shares * position.price
                for position in holdings.values()
                if position.sell_reason
            )
            for holding in sorted(holdings.values(), key=lambda position: position.score):
                if projected <= target * assets + 0.01 * assets:
                    break
                if not holding.sell_reason:
                    holding.sell_reason = "allocation"
                    projected -= holding.shares * holding.price
            if target == 0:
                pending.clear()
        book.record(day, holdings)
        beta = (
            sum(
                position.shares
                * position.price
                * (values["beta"][day, column] if np.isfinite(values["beta"][day, column]) else 1.0)
                for column, position in holdings.items()
            )
            / assets
        )
        weights = {
            column: position.shares * position.price / assets
            for column, position in holdings.items()
        }
        daily.append(
            {
                "day": day,
                "date": calendar[day],
                "nav": assets,
                "cash": cash,
                "receivable": book.outstanding(),
                "exposure": invested / assets,
                "beta": beta,
                "stage": stage,
                "positions": len(holdings),
                "weights": weights,
            }
        )
        daily[-1].update(
            {
                "open_exposure": opening_value / opening_assets,
                "cost": fees - previous_fees + slippage,
            }
        )
        available_slots = max(
            0, int(target / policy.position_weight + 1e-8) - len(holdings) - len(pending)
        )
        for security, family, score in entry_candidates(panel, day, stage, policy):
            if available_slots <= 0:
                break
            if security not in holdings and security not in pending:
                pending[security] = (day, family, score, stage, assets * policy.position_weight)
                available_slots -= 1
    stale_value = sum(
        position.shares * position.price
        for position in holdings.values()
        if len(calendar) - 1 - position.observed > 20
    )
    realized = sum(trade["proceeds"] + trade["dividend"] - trade["cost"] for trade in trades)
    unrealized = sum(
        position.shares * position.price + position.dividend - position.cost
        for position in holdings.values()
    )
    reconstructed = capital + realized + unrealized
    actual_assets = daily[-1]["nav"] if daily else capital
    return {
        "daily": daily,
        "trades": trades,
        "entry_days": entry_days,
        "open_positions": len(holdings),
        "stale_value": stale_value,
        "fees": fees,
        "turnover": turnover,
        "rejected_budget": rejected,
        "capital": capital,
        "policy": policy.name,
        "reconciliation_error": actual_assets - reconstructed,
        "open_cost": sum(position.cost for position in holdings.values()),
        "open_market_value": sum(
            position.shares * position.price for position in holdings.values()
        ),
    }
