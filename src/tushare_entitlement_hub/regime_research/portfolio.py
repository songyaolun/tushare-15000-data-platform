"""收盘决策、下一合格开盘成交的资金受限组合模拟。"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Real
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
from .execution_evidence import ExecutionEvidence
from .panel import Panel

ContextCandidates = Callable[
    [Panel, int, int, "PortfolioPolicy", frozenset[int], frozenset[int]],
    list[tuple[int, str, float]],
]
ReplacementProvider = Callable[
    [
        Panel,
        int,
        "PortfolioPolicy",
        tuple[Holding, ...],
        tuple[tuple[int, str, float], ...],
        frozenset[int],
    ],
    list[tuple[int, int]],
]

PositionWeightProvider = Callable[[Panel, int, int, float, int, int, "PortfolioPolicy"], float]


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
    fixed_price_exits: bool = True
    slippage_rate: float = 0.0005
    non_stage_allocation: float = 0.9
    fee_multiplier: float = 1.0
    take_profit_enabled: bool = True

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
            and isinstance(self.fixed_price_exits, bool)
            and isinstance(self.take_profit_enabled, bool)
            and 0 <= self.slippage_rate < 1
            and 0 <= self.non_stage_allocation <= 1
            and np.isfinite(self.fee_multiplier)
            and self.fee_multiplier >= 0
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


def exit_reason(
    holding: Holding,
    day: int,
    adjusted_close: float,
    policy: PortfolioPolicy,
    *,
    fixed_exits: bool = True,
) -> str:
    """按入场时策略和当前已知收盘判断退出，不因阶段切换放宽止损。

    例成本复权10、收盘8.9、stop_loss=.1输出stop；停牌无价格仍可触发持有到期。
    policy.fixed_price_exits=False关闭固定止盈止损，期限、跟踪与反弹失败规则保留；
    例如同一持仓收盘8.9且未到期时输出空字符串。fixed_exits=False也可由盘中模型
    关闭收盘固定退出；该参数不能重新开启策略明确关闭的固定退出。
    take_profit_enabled=False仅关闭固定止盈，止损保持，例如上涨9%不止盈、跌9%仍止损。
    """
    held = day - holding.entry + 1
    fixed_exits = fixed_exits and policy.fixed_price_exits
    if np.isfinite(adjusted_close):
        multiple = adjusted_close / holding.entry_adjusted
        holding.peak_adjusted = max(holding.peak_adjusted, adjusted_close)
        if fixed_exits and multiple <= 1 - policy.stop_loss:
            return "stop"
        if policy.trailing and holding.stage == 1:
            if adjusted_close / holding.peak_adjusted <= 1 - policy.trailing:
                return "trailing"
        elif fixed_exits and policy.take_profit_enabled and multiple >= 1 + policy.take_profit:
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
    candidate_provider: Callable[[Panel, int, int, PortfolioPolicy], list[tuple[int, str, float]]]
    | None = None,
    exit_provider: Callable[[Panel, Holding, int, PortfolioPolicy], str] | None = None,
    entry_filter: Callable[[Panel, int, int, int], bool] | None = None,
    cancel_pending_provider: Callable[[Panel, int, int, int], bool] | None = None,
    intraday_exit_provider: Callable[
        [Panel, Holding, int, PortfolioPolicy], tuple[float, str] | None
    ]
    | None = None,
    context_candidate_provider: ContextCandidates | None = None,
    reserve_exit_candidates: bool = False,
    position_weight_provider: PositionWeightProvider | None = None,
    replacement_provider: ReplacementProvider | None = None,
    capture_execution_evidence: bool = False,
) -> dict[str, Any]:
    """完整逐日资金回放，返回净值、交易、敞口、成交日和陈旧报价诊断。

    例100万元账户、5%单股权重，买单预算不超过5万元且受现金和行业上限限制。
    三日订单有效期；T+1、限价和停牌限制；卖出收益留在现金，不以未来净值定仓。
    原始股数处理现金/送股，不支持配股认购；零交易不伪造胜率。
    可选candidate_provider(panel,day,stage,policy)只在收盘提供候选；
    exit_provider(panel,holding,day,policy)补充原退出规则，无权覆盖已启用的固定止损；
    entry_filter(panel,confirmation,day,security)只可读取信号日历史及成交日开盘信息，
    False取消待买订单；cancel_pending_provider(panel,confirmation,day,security)
    在收盘为True时撤销尚未成交的买单，不影响该日已成交订单。
    intraday_exit_provider返回(原始基准价格,原因)，只在开盘买单后、收盘估值前调用；
    提供方负责完整OHLC和限价可成交检查，账本仍强制T+1与送股锁定。
    例如所有hook为None时使用内置策略；成交按前一日行业及当前资产核算仓位上限。
    slippage_rate同时影响买卖成交价格、预算和日成本，fee_multiplier乘完整买卖费用；
    例如fee_multiplier=2时买单先预留两倍手续费，再决定可买股数。
    非阶段仓位使用non_stage_allocation，例如0.5表示最多投入当前资产的50%。
    context_candidate_provider附加不可变的当前持仓和待单证券集合，仅在收盘调用。
    reserve_exit_candidates允许已发卖单支持条件预备名额；第k个预备单须等该批
    卖单至少“当前超额名额+k”笔实际成交才能参与买入，仍受真实现金/仓位/行业限制，
    默认关闭。超额名额按收盘新挂单前的持仓与待单计算，避免弱市收紧后提前补仓。
    例满仓两只均待卖时可准备两单；只卖出一只最多启用第一单，不假定第二只可卖。
    position_weight_provider在收盘为新订单提供预算比例，接收面板、日、证券、分数、
    候选排名、候选数及策略；排名在持仓排除前按零起计算。输出必须为正且不超过
    policy.position_weight，例如返回.03将该笔信号预算冻结为收盘资产的3%。
    replacement_provider只在收盘返回(卖出证券,买入证券)配对，收到持仓副本和已合格候选。
    仅在启用条件预备、正常退出后仍满仓且没有超额持仓时使用；卖出实际成交后才准买入
    配对候选，不允许借另一笔卖出替代。例[(0,2)]表示卖0后才可买2。
    名额仍按最大单票比例计算，开盘仍受原仓位、现金与行业限制，不强制填满剩余现金。
    capture_execution_evidence=True追加成交、分红事件与逐日持仓CSV数据；默认False
    不追加返回字段。例如100股以10元成交，证据记录原始股数100和金额1000元。
    """
    if capital <= 0 or len(states) != len(panel.calendar):
        raise ValueError("invalid_simulation_inputs")
    values, calendar = panel.values, panel.calendar
    begin = next((i for i, day in enumerate(calendar) if day >= start), len(calendar))
    cash = capital
    holdings: dict[int, Holding] = {}
    pending: dict[int, tuple[int, str, float, int, float]] = {}
    reserve_dependencies: dict[tuple[int, int], tuple[frozenset[int], int]] = {}
    reserve_created = reserve_waits = reserve_filled = 0
    paired_sales: dict[tuple[int, int], int] = {}
    replacement_decisions: list[dict[str, int]] = []
    evidence = ExecutionEvidence(panel) if capture_execution_evidence else None
    book = DistributionBook(events, evidence)
    daily: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    entry_days: list[int] = []
    turnover, fees, rejected = 0.0, 0.0, 0
    slippage = 0.0

    def sell_position(
        security: int,
        holding: Holding,
        day: int,
        reference_price: float,
        lower_limit: float,
        reason: str,
    ) -> None:
        """用同一手续费、滑点及分红账本记卖出；例基准价10以max(9.995,跌停价)卖出。"""
        nonlocal cash, fees, turnover, slippage
        price = max(reference_price * (1 - policy.slippage_rate), lower_limit)
        notional = price * holding.shares
        slippage += (reference_price - price) * holding.shares
        fee = transaction_fee(notional, calendar[day], True) * policy.fee_multiplier
        cash += notional - fee
        fees += fee
        turnover += notional
        if evidence is not None:
            evidence.fill(holding, day, "sell", price, reference_price, fee, cash, reason)
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
                "reason": reason,
                "shares": holding.shares,
            }
        )
        del holdings[security]

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
                sell_position(security, holding, day, opening, lower, holding.sell_reason)
        for security, (confirmation, family, score, stage, budget) in list(pending.items()):
            if day > confirmation + 3:
                del pending[security]
                continue
            paired_sale = paired_sales.get((confirmation, security))
            if paired_sale is not None and paired_sale in holdings:
                reserve_waits += 1
                continue
            dependency = reserve_dependencies.get((confirmation, security))
            if dependency is not None:
                outgoing, required_sales = dependency
                if len(outgoing - holdings.keys()) < required_sales:
                    reserve_waits += 1
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
            if entry_filter is not None and not entry_filter(panel, confirmation, day, security):
                del pending[security]
                continue
            sector = values["sector_index"][day - 1, security]
            if not np.isfinite(sector) or sector < 0:
                del pending[security]
                continue
            price = min(opening * (1 + policy.slippage_rate), upper)
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
            target = (
                policy.allocations[current_stage]
                if policy.stage_allocation
                else policy.non_stage_allocation
            )
            sector_value = sum(
                amount
                for column, amount in marked.items()
                if values["sector_index"][day - 1, column] == sector
            )
            allowance = min(
                budget,
                policy.position_weight * assets,
                cash,
                target * assets - sum(marked.values()),
                policy.sector_weight * assets - sector_value,
            )
            shares = affordable_shares(
                allowance,
                price,
                panel.codes[security],
                calendar[day],
                fee_multiplier=policy.fee_multiplier,
            )
            if shares:
                notional = shares * price
                slippage += (price - opening) * shares
                fee = transaction_fee(notional, calendar[day], False) * policy.fee_multiplier
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
                if evidence is not None:
                    evidence.fill(
                        holdings[security], day, "buy", price, opening, fee, cash, "entry"
                    )
                if dependency is not None:
                    reserve_filled += 1
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
        # 开盘资产先冻结；盘中卖出现金不能用于已经结束的开盘买单。
        if intraday_exit_provider is not None:
            for security, holding in list(holdings.items()):
                if day <= holding.entry or holding.locked >= 1e-8:
                    continue
                instruction = intraday_exit_provider(panel, holding, day, policy)
                if instruction is not None:
                    reference_price, reason = instruction
                    sell_position(
                        security,
                        holding,
                        day,
                        reference_price,
                        float(values["raw_down_limit"][day, security]),
                        reason,
                    )
        for security, holding in holdings.items():
            closing = values["raw_close"][day, security]
            if np.isfinite(closing) and closing > 0:
                holding.price, holding.observed = float(closing), day
            if not holding.sell_reason:
                adjusted = closing * values["raw_adj_factor"][day, security]
                holding.sell_reason = exit_reason(
                    holding, day, adjusted, policy, fixed_exits=intraday_exit_provider is None
                )
                if not holding.sell_reason and exit_provider is not None:
                    holding.sell_reason = exit_provider(panel, holding, day, policy)
        invested = sum(holding.shares * holding.price for holding in holdings.values())
        assets = cash + invested + book.outstanding()
        if cash < -1e-6 or not np.isfinite(assets) or assets <= 0:
            raise ValueError("accounting_invariant_failed")
        stage = int(states[day])
        target = (
            policy.allocations[stage] if policy.stage_allocation else policy.non_stage_allocation
        )
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
        if evidence is not None:
            evidence.snapshot(day, holdings)
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
                "open_nav": opening_assets,
                "cost": fees - previous_fees + slippage,
            }
        )
        if cancel_pending_provider is not None:
            for security, (confirmation, _, _, _, _) in list(pending.items()):
                if cancel_pending_provider(panel, confirmation, day, security):
                    del pending[security]
        candidates = candidate_provider if candidate_provider is not None else entry_candidates
        candidate_rows = (
            context_candidate_provider(
                panel, day, stage, policy, frozenset(holdings), frozenset(pending)
            )
            if context_candidate_provider is not None
            else candidates(panel, day, stage, policy)
        )
        pairs: list[tuple[int, int]] = []
        capacity = int(target / policy.position_weight + 1e-8)
        normal_outgoing = {column for column, position in holdings.items() if position.sell_reason}
        if (
            replacement_provider is not None
            and reserve_exit_candidates
            and len(holdings) + len(pending) == capacity
            and not normal_outgoing
        ):
            pairs = replacement_provider(
                panel,
                day,
                policy,
                tuple(replace(position) for position in holdings.values()),
                tuple(candidate_rows),
                frozenset(pending),
            )
            sellers = [outgoing_security for outgoing_security, _ in pairs]
            buyers = [incoming_security for _, incoming_security in pairs]
            candidate_codes = {column for column, _, _ in candidate_rows}
            if (
                len(set(sellers)) != len(sellers)
                or len(set(buyers)) != len(buyers)
                or any(column not in holdings or holdings[column].sell_reason for column in sellers)
                or any(
                    column not in candidate_codes or column in holdings or column in pending
                    for column in buyers
                )
            ):
                raise ValueError("invalid_replacement_pairs")
            for outgoing_security, incoming_security in pairs:
                holdings[outgoing_security].sell_reason = "replacement"
                paired_sales[day, incoming_security] = outgoing_security
                replacement_decisions.append(
                    {"day": day, "sell": outgoing_security, "buy": incoming_security}
                )
            if pairs:
                candidate_rows = [row for row in candidate_rows if row[0] in set(buyers)]
        available_slots = max(
            0, int(target / policy.position_weight + 1e-8) - len(holdings) - len(pending)
        )
        regular_slots = available_slots
        slot_deficit = max(
            0, len(holdings) + len(pending) - int(target / policy.position_weight + 1e-8)
        )
        outgoing = frozenset(
            security for security, holding in holdings.items() if holding.sell_reason
        )
        if reserve_exit_candidates:
            available_slots = max(
                available_slots,
                int(target / policy.position_weight + 1e-8)
                - len(holdings)
                + len(outgoing)
                - len(pending),
            )
        scheduled = 0
        for rank, (security, family, score) in enumerate(candidate_rows):
            if available_slots <= 0:
                break
            if security not in holdings and security not in pending:
                weight = (
                    policy.position_weight
                    if position_weight_provider is None
                    else position_weight_provider(
                        panel, day, security, score, rank, len(candidate_rows), policy
                    )
                )
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, Real)
                    or not np.isfinite(weight)
                    or not 0 < weight <= policy.position_weight
                ):
                    raise ValueError("invalid_candidate_position_weight")
                pending[security] = (day, family, score, stage, assets * weight)
                scheduled += 1
                if scheduled > regular_slots:
                    reserve_dependencies[day, security] = (
                        (frozenset({paired_sales[day, security]}), 1)
                        if (day, security) in paired_sales
                        else (outgoing, slot_deficit + scheduled - regular_slots)
                    )
                    reserve_created += 1
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
        **({"execution_evidence": evidence.result()} if evidence is not None else {}),
        "trades": trades,
        **(
            {"replacement_decisions": replacement_decisions}
            if replacement_provider is not None
            else {}
        ),
        "entry_days": entry_days,
        "open_positions": len(holdings),
        "stale_value": stale_value,
        "fees": fees,
        "turnover": turnover,
        "rejected_budget": rejected,
        **(
            {
                "reserve_orders": {
                    "created": reserve_created,
                    "blocked_open_attempts": reserve_waits,
                    "filled": reserve_filled,
                }
            }
            if reserve_exit_candidates
            else {}
        ),
        "capital": capital,
        "policy": policy.name,
        "reconciliation_error": actual_assets - reconstructed,
        "open_cost": sum(position.cost for position in holdings.values()),
        "open_market_value": sum(
            position.shares * position.price for position in holdings.values()
        ),
    }
