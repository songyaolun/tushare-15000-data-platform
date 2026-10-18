"""原始股数、现金、分红应收与交易成本；不使用复权单位替代现金账本。"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..strategy_optimization.dataset import read_envelope


@dataclass(frozen=True)
class Distribution:
    """实施分红的一次股权登记及支付；例每股税前1元、送股.2对应cash=1,stock=.2。"""

    security: int
    record: int
    ex: int
    pay: int
    listing: int
    cash: float
    stock: float


@dataclass
class Holding:
    """独立持仓及退出状态；例100股成本1000元、收盘11元，市值1100元。"""

    security: int
    confirmation: int
    entry: int
    shares: float
    cost: float
    price: float
    entry_adjusted: float
    peak_adjusted: float
    family: str
    stage: int
    score: float
    dividend: float = 0
    locked: float = 0
    sell_reason: str = ""
    observed: int = 0


def transaction_fee(notional: float, day: str, selling: bool) -> float:
    """计算最低5元佣金、分时期过户费和卖出印花税；不含价格滑点。

    例2024年买入1000元，输出5.01元；卖出再增加.5元印花税。
    """
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional_must_be_positive")
    transfer = 0.00001 if day >= "20220429" else 0.00002
    stamp = (0.0005 if day >= "20230828" else 0.001) if selling else 0
    return max(5.0, notional * 0.0003) + notional * (transfer + stamp)


def affordable_shares(budget: float, price: float, code: str, day: str) -> int:
    """按沪深整手规则向下取股数并预留费用；科创板至少200股、其余100股整数倍。

    例预算1005.1元、价格10、主板代码，返回100股；预算不足首手返回0。
    """
    if not np.isfinite(price) or price <= 0 or budget <= 5:
        return 0
    minimum, step = (200, 1) if code.startswith(("688", "689")) else (100, 100)
    shares = int(budget / price) // step * step
    while shares >= minimum:
        notional = shares * price
        if notional + transaction_fee(notional, day, False) <= budget:
            return shares
        shares -= step
    return 0


def load_distributions(raw: Path, calendar: list[str], codes: list[str]) -> list[Distribution]:
    """读取实施分红，按登记日取得资格、除权日确认权益、支付/上市日解除限制。

    输入分红现金1元且ex_date有效，输出一条Distribution；未知支付日期保留为样本末以后。
    缺登记日、重复冲突或非有限权益拒绝，不把缺失分红日期解释为零分红。
    """
    dates = {day: index for index, day in enumerate(calendar)}
    securities = {code: index for index, code in enumerate(codes)}
    records: dict[tuple[int, int, str], Distribution] = {}
    resolution_path = raw / "distribution-resolutions.json"
    resolutions = json.loads(resolution_path.read_text()) if resolution_path.exists() else {}
    for path in sorted(raw.glob("dividend-*.json")):
        if path.name == "dividend-probe.json":
            continue
        for row in read_envelope(path):
            if row.get("div_proc") != "实施" or row.get("ts_code") not in securities:
                continue
            if row.get("ex_date") not in dates:
                continue
            if row.get("record_date") not in dates:
                raise ValueError("dividend_missing_record_session")
            cash, stock = float(row.get("cash_div_tax") or 0), float(row.get("stk_div") or 0)
            if not np.isfinite(cash + stock) or min(cash, stock) < 0:
                raise ValueError("invalid_distribution_amount")
            event = Distribution(
                securities[row["ts_code"]],
                dates[row["record_date"]],
                dates[row["ex_date"]],
                dates.get(str(row.get("pay_date")), len(calendar)),
                dates.get(str(row.get("div_listdate")), len(calendar)) if stock else 0,
                cash,
                stock,
            )
            if (
                event.ex <= event.record
                or (cash and event.pay < event.ex)
                or (stock and event.listing < event.ex)
            ):
                raise ValueError("invalid_distribution_timeline")
            key = (event.security, event.ex, str(row.get("end_date")))
            resolution = resolutions.get(
                "|".join((row["ts_code"], row["ex_date"], str(row.get("end_date"))))
            )
            if resolution is not None:
                corrected_cash = float(resolution["cash_div_tax"])
                if (
                    not str(resolution.get("source_url", "")).startswith("https://")
                    or not np.isfinite(corrected_cash)
                    or corrected_cash < 0
                ):
                    raise ValueError("invalid_distribution_resolution")
                event = replace(event, cash=corrected_cash)
            if key in records and records[key] != event:
                raise ValueError("conflicting_distribution")
            records[key] = event
    return list(records.values())


class DistributionBook:
    """保存登记权益、应收现金和未上市送股；现金红利统一预扣20%作为保守税费假设。"""

    def __init__(self, events: list[Distribution]) -> None:
        """建立登记日事件索引及空账本；输入[]输出无应收款账本。"""
        self.records: dict[int, list[Distribution]] = {}
        for event in events:
            self.records.setdefault(event.record, []).append(event)
        self.entitlements: list[tuple[Distribution, Holding, float]] = []
        self.receivables: list[tuple[int, float]] = []
        self.locked_shares: list[tuple[int, Holding, float]] = []

    def record(self, day: int, holdings: dict[int, Holding]) -> None:
        """登记日收盘记录现有股数；例如100股、每股1元，次日确认80元税后权益。"""
        for event in self.records.get(day, []):
            holding = holdings.get(event.security)
            if holding is not None:
                self.entitlements.append((event, holding, holding.shares))

    def advance(self, day: int) -> float:
        """开盘前确认除权权益并支付到期现金、解锁送股；返回今日到账现金。

        例100股每股1元、同日除权支付，返回80；支付日尚未来临则应收增加80。
        """
        remaining = []
        for event, holding, shares in self.entitlements:
            if event.ex == day:
                cash = shares * event.cash * 0.8
                stock = shares * event.stock
                holding.dividend += cash
                holding.shares += stock
                holding.locked += stock
                if cash:
                    self.receivables.append((event.pay, cash))
                if stock:
                    self.locked_shares.append((event.listing, holding, stock))
            else:
                remaining.append((event, holding, shares))
        self.entitlements = remaining
        paid = sum(amount for due, amount in self.receivables if due <= day)
        self.receivables = [(due, amount) for due, amount in self.receivables if due > day]
        locked = []
        for due, holding, shares in self.locked_shares:
            if due <= day:
                holding.locked = max(0.0, holding.locked - shares)
            else:
                locked.append((due, holding, shares))
        self.locked_shares = locked
        return paid

    def outstanding(self) -> float:
        """返回未到账现金权益总额；例如两笔80和20应收输出100，不能用于买入。"""
        return sum(amount for _, amount in self.receivables)
