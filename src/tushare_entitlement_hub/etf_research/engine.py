"""无杠杆ETF三账户回放，按月末信号、次日开盘及现金分红执行。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Configuration:
    """固定实验费用与时间；默认100万元、100份一手，佣金与滑点是研究假设。"""

    start: str = "20211018"
    end: str = "20261016"
    capital: float = 1_000_000
    lookback: int = 126
    lot: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5
    slippage_rate: float = 0.0005

    def __post_init__(self) -> None:
        """拒绝无效时间和费用；例如capital=-1抛ValueError，不允许负现金融资。"""
        for value in (self.start, self.end):
            datetime.strptime(value, "%Y%m%d")
        if self.start > self.end or self.capital <= 0 or self.lookback < 1 or self.lot < 1:
            raise ValueError("invalid_etf_configuration")
        values = (self.capital, self.commission_rate, self.minimum_commission, self.slippage_rate)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid_etf_configuration")
        if self.commission_rate >= 1 or self.slippage_rate >= 1:
            raise ValueError("invalid_etf_configuration")


@dataclass(frozen=True)
class Bar:
    """原始元价格、手成交量与同日限制；factor仅供信号，不参与现金成交。"""

    opening: float
    high: float
    low: float
    close: float
    volume: float
    factor: float
    upper: float | None
    lower: float | None


@dataclass(frozen=True)
class Dividend:
    """已实施每份现金派息；例如100份、cash_per_share=.1在支付日到账10元。"""

    identity: str
    record_date: str
    ex_date: str
    pay_date: str
    cash_per_share: float


def commission(notional: float, configuration: Configuration) -> float:
    """计算ETF双边佣金、无印花税和过户费；例：1000元成交、最低5元返回5。"""
    return max(configuration.minimum_commission, notional * configuration.commission_rate)


def trend_weight(
    decision: str, calendar: list[str], bars: dict[str, Bar], lookback: int
) -> float:
    """仅比较决策日与126交易日前的复权收盘；例：10涨至11返回1，持平返回0。"""
    position = calendar.index(decision)
    if position < lookback:
        raise ValueError("etf_signal_warmup_missing")
    first, last = bars.get(calendar[position - lookback]), bars.get(decision)
    if first is None or last is None or first.factor <= 0 or last.factor <= 0:
        raise ValueError("etf_signal_prices_missing")
    return float(last.close * last.factor > first.close * first.factor)


def opening_trade(
    day: str, bar: Bar | None, cash: float, shares: int, receivable: float,
    target: float, configuration: Configuration,
) -> tuple[float, int, dict[str, Any]]:
    """按开盘目标比例买卖整手，保留足额现金并保守拒绝封板/缺限价。

    输入1000元、股价1元、目标100%，零费用时买1000份；受限时不改变持仓。
    日总成交量仅作成交状态检查，不被用于证明开盘订单容量。
    """
    order: dict[str, Any] = {"date": day, "target": target, "status": "blocked"}
    if bar is None or bar.volume <= 0 or bar.opening <= 0:
        order["reason"] = "missing_or_suspended"
        return cash, shares, order
    if bar.upper is None or bar.lower is None:
        order["reason"] = "missing_limits"
        return cash, shares, order
    if bar.opening >= bar.upper - 1e-9 or bar.opening <= bar.lower + 1e-9:
        order["reason"] = "opening_at_price_limit"
        return cash, shares, order
    assets = cash + shares * bar.opening + receivable
    desired = math.floor(target * assets / bar.opening / configuration.lot) * configuration.lot
    change = desired - shares
    side = "buy" if change > 0 else "sell"
    price = bar.opening * (1 + configuration.slippage_rate * (1 if change > 0 else -1))
    if price >= bar.upper or price <= bar.lower:
        order["reason"] = "slippage_exceeds_price_limit"
        return cash, shares, order
    quantity = abs(change)
    if change > 0:
        affordable = math.floor(cash / price / configuration.lot) * configuration.lot
        quantity = min(quantity, affordable)
        while quantity > 0 and (
            quantity * price + commission(quantity * price, configuration) > cash
        ):
            quantity -= configuration.lot
    if quantity == 0:
        order.update(status="no_trade", reason="target_or_cash_rounding")
        return cash, shares, order
    notional = quantity * price
    fee = commission(notional, configuration)
    if change > 0:
        cash, shares = cash - notional - fee, shares + quantity
    else:
        cash, shares = cash + notional - fee, shares - quantity
    order.update(status="filled", side=side, shares=quantity, price=price,
                 opening=bar.opening, notional=notional, commission=fee,
                 stamp_tax=0.0, transfer_fee=0.0,
                 slippage_cost=quantity * abs(price - bar.opening))
    return cash, shares, order


def summarize(daily: list[dict[str, Any]], capital: float) -> dict[str, Any]:
    """按交易日净值算复合年化和回撤，年度以期初实际资产衔接。

    输入两个净值100、90、起始100，输出累计收益-.1、最大回撤-.1；年化按250日。
    """
    if not daily:
        raise ValueError("empty_etf_account")
    peak, maximum_drawdown = capital, 0.0
    annual: dict[str, dict[str, float]] = {}
    previous = capital
    for row in daily:
        value = float(row["assets"])
        peak = max(peak, value)
        maximum_drawdown = min(maximum_drawdown, value / peak - 1)
        year = str(row["date"])[:4]
        annual.setdefault(year, {"opening_assets": previous})
        annual[year]["closing_assets"] = value
        annual[year]["return"] = value / annual[year]["opening_assets"] - 1
        previous = value
    return {"ending_assets": previous, "total_return": previous / capital - 1,
            "annualized_return": (previous / capital) ** (250 / len(daily)) - 1,
            "maximum_drawdown": maximum_drawdown, "annual_returns": annual,
            "sessions": len(daily), "annualization_basis": "250 exchange sessions"}


def run_account(
    strategy: str, calendar: list[str], bars: dict[str, Bar], dividends: list[Dividend],
    configuration: Configuration | None = None,
) -> dict[str, Any]:
    """回放单个ETF策略；月末收盘决策、下一交易日开盘执行，失败订单延后重试。

    例：monthly_half首日买约一半资产，此后仅月末恢复比例；buy_hold不再投入分红。
    权益登记在收盘后，应收在除息日出现，支付日之后首个交易日才可购买；末日不强卖。
    """
    configuration = configuration or Configuration()
    if strategy not in {"buy_hold", "monthly_half", "monthly_trend_126"}:
        raise ValueError("unknown_etf_strategy")
    if calendar != sorted(set(calendar)):
        raise ValueError("invalid_etf_calendar")
    days = [day for day in calendar if configuration.start <= day <= configuration.end]
    if not days or days[0] != configuration.start or days[-1] != configuration.end:
        raise ValueError("etf_target_calendar_missing")
    first_position = calendar.index(days[0])
    prior_months = [day for day in calendar[:first_position] if day[:6] < days[0][:6]]
    if not prior_months:
        raise ValueError("etf_initial_month_end_missing")
    initial_decision = prior_months[-1]
    weight = 0.5 if strategy == "monthly_half" else 1.0
    if strategy == "monthly_trend_126":
        weight = trend_weight(initial_decision, calendar, bars, configuration.lookback)
    pending: tuple[str, float] | None = (initial_decision, weight)
    cash, shares = configuration.capital, 0
    last_price = bars[initial_decision].close
    entitlements: dict[str, float] = {}
    outstanding: dict[str, float] = {}
    processed_ex: set[str] = set()
    processed_pay: set[str] = set()
    orders: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    decisions = [{"decision_date": initial_decision, "execution_not_before": days[0],
                  "target": weight}]
    for day in days:
        for event in dividends:
            amount = entitlements.get(event.identity, 0.0)
            if event.ex_date <= day and event.identity not in processed_ex:
                outstanding[event.identity] = amount
                processed_ex.add(event.identity)
                if amount:
                    events.append({"date": day, "event": "ex_dividend", "id": event.identity,
                                   "cash_amount": amount})
            if event.pay_date < day and event.identity not in processed_pay:
                paid = outstanding.pop(event.identity, 0.0)
                cash += paid
                processed_pay.add(event.identity)
                if paid:
                    events.append({"date": day, "event": "dividend_paid", "id": event.identity,
                                   "cash_amount": paid})
        bar = bars.get(day)
        if pending is not None:
            cash, shares, order = opening_trade(
                day, bar, cash, shares, sum(outstanding.values()), pending[1], configuration
            )
            order["decision_date"] = pending[0]
            orders.append(order)
            if order["status"] != "blocked":
                pending = None
        if bar is not None and bar.close > 0:
            last_price = bar.close
        for event in dividends:
            if event.record_date == day:
                entitlements[event.identity] = shares * event.cash_per_share
        receivable = sum(outstanding.values())
        assets = cash + shares * last_price + receivable
        if cash < -1e-6 or shares < 0 or not math.isfinite(assets) or assets <= 0:
            raise ValueError("etf_account_insolvent")
        daily.append({"date": day, "cash": cash, "shares": shares, "close": last_price,
                      "market_value": shares * last_price, "receivable": receivable,
                      "assets": assets, "exposure": shares * last_price / assets,
                      "stale_mark": bar is None, "pending_target": pending[1] if pending else None})
        position = calendar.index(day)
        next_day = calendar[position + 1] if position + 1 < len(calendar) else None
        if strategy != "buy_hold" and next_day is not None and next_day[:6] != day[:6]:
            weight = 0.5 if strategy == "monthly_half" else trend_weight(
                day, calendar, bars, configuration.lookback
            )
            pending = day, weight
            decisions.append({"decision_date": day, "execution_not_before": next_day,
                              "target": weight})
    trades = [order for order in orders if order["status"] == "filled"]
    summary = summarize(daily, configuration.capital)
    buy_positions = [days.index(order["date"]) for order in trades if order["side"] == "buy"]
    boundaries = [0, *buy_positions, len(days) - 1]
    gaps = [right - left for left, right in zip(boundaries, boundaries[1:], strict=False)]
    summary.update(strategy=strategy, trades=len(trades),
                   commission=sum(order["commission"] for order in trades),
                   slippage_cost=sum(order["slippage_cost"] for order in trades),
                   paid_dividends=sum(e["cash_amount"] for e in events
                                      if e["event"] == "dividend_paid"),
                   mean_exposure=sum(row["exposure"] for row in daily) / len(daily),
                   roundtrip_win_rate=None,
                   roundtrip_win_rate_reason="partial inventory trades not defined as roundtrips",
                   maximum_buy_gap_sessions=max(gaps),
                   buy_gap_basis="exchange-session index distance, including start/end boundaries",
                   buy_count=len(buy_positions),
                   execution_assumptions={
                       "sizing_and_fill": "observed raw opening price with assumed slippage; "
                       "daily bars do not prove auction order capacity or executable exact sizing",
                       "volume": "full-day volume only checks trading status, not opening capacity",
                       "dividend_cash": "available at first exchange opening after pay_date",
                       "buy_hold_cash_dividends": "not automatically reinvested",
                       "cash_interest": 0,
                       "costs": "commission/minimum/slippage are research assumptions; "
                       "ETF stamp tax and transfer fee zero",
                       "settlement": "at most one order per session; newly bought shares never "
                       "sold same day; month-end close decision fills next session opening",
                       "end": "remaining inventory marked at close, no forced liquidation",
                   })
    return {"configuration": asdict(configuration), "summary": summary, "daily": daily,
            "orders": orders, "trades": trades, "dividends": events, "decisions": decisions}
