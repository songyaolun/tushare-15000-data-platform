"""从有来源证据的价格窗口识别回调后突破、回踩及入场风险。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from statistics import median
from typing import Any

from ..product_foundation.contracts import EvidenceRef
from .common import unique_evidence
from .models import ResearchInputs, ResearchRow
from .right_side_policy import RightSidePolicy


@dataclass(frozen=True)
class PriceBar:
    """以当前价格为锚的供应商收益链接价格；例：拆股前20元可归一到当前10元。"""

    day: str
    close: float
    high: float
    low: float
    amount: float


@dataclass(frozen=True)
class StructureObservation:
    """保存结构事实、引用及不可确认原因；例：缺窗口返回空特征与明确reason。"""

    values: dict[str, float]
    details: dict[str, Any]
    references: tuple[EvidenceRef, ...]
    reason: str | None


def price_window(
    inputs: ResearchInputs, identity: str
) -> tuple[tuple[PriceBar, ...], tuple[EvidenceRef, ...], str | None]:
    """核验唯一日期、CNY价格及收益，反向链接到最新收盘价以消除裸价跳变。

    例如前日close=20、当日close=10且供应商pct_chg=0，前日也链接为10；
    缺价格/收益认证、同日冲突或少于61条返回原因，不直接混用未复权高低价。
    该序列只使用供应商价格收益，不称为含分红的总收益或历史可交易价格。
    """
    grouped: dict[str, list[ResearchRow]] = {}
    for row in inputs.rows("daily", identity):
        day = row.availability.data_as_of
        if day is None:
            return (), (), "right_side_date_missing"
        grouped.setdefault(day, []).append(row)
    if any(len(rows) != 1 for rows in grouped.values()):
        return (), (), "right_side_date_conflict"
    ordered = [grouped[day][0] for day in sorted(grouped)]
    selected = ordered[-inputs.config.right_side.history_observations :]
    references = unique_evidence(
        replace(row.evidence(*names)[0], fields=names)
        for row in selected
        if (
            names := tuple(
                name
                for name in ("close", "high", "low", "pct_chg", "amount", "trade_date")
                if name in row.raw
            )
        )
    )
    if len(selected) < inputs.config.right_side.history_observations:
        return (), references, "right_side_history_incomplete"
    raw_values = []
    for row in selected:
        price_facts = tuple(row.numeric.get(name) for name in ("close", "high", "low"))
        if any(fact is None or fact.dimension != "price" for fact in price_facts):
            return (), references, "right_side_price_unverified"
        prices = tuple(row.number(name, "元") for name in ("close", "high", "low"))
        amount = row.number("amount", "元")
        amount_fact = row.numeric.get("amount")
        if amount_fact is None or amount_fact.dimension != "money":
            return (), references, "right_side_amount_unverified"
        change = row.number("pct_chg")
        fact = row.numeric.get("pct_chg")
        if any(value is None or not isfinite(value) or value <= 0 for value in prices):
            return (), references, "right_side_price_unverified"
        close, high, low = (float(value) for value in prices if value is not None)
        if not low <= close <= high:
            return (), references, "right_side_ohlc_invalid"
        if amount is None or not isfinite(amount) or amount <= 0:
            return (), references, "right_side_amount_unverified"
        if (
            change is None
            or not isfinite(change)
            or change <= -1
            or fact is None
            or fact.dimension != "ratio"
            or fact.unit not in {"ratio", "fraction"}
        ):
            return (), references, "right_side_return_unverified"
        raw_values.append((close, high, low, amount, change))
    linked_close = raw_values[-1][0]
    reversed_bars = []
    for index in range(len(selected) - 1, -1, -1):
        close, high, low, amount, change = raw_values[index]
        linked_high, linked_low = linked_close * (high / close), linked_close * (low / close)
        if not all(
            isfinite(value) and value > 0 for value in (linked_close, linked_high, linked_low)
        ):
            return (), references, "right_side_linked_price_invalid"
        reversed_bars.append(
            PriceBar(
                selected[index].availability.data_as_of or "",
                linked_close,
                linked_high,
                linked_low,
                amount,
            )
        )
        linked_close = linked_close / (1 + change)
    return tuple(reversed(reversed_bars)), references, None


def structure_at(
    bars: tuple[PriceBar, ...], breakout_index: int, policy: RightSidePolicy
) -> StructureObservation:
    """评价指定突破日及当前回踩，返回可解释的结构值和首个失败条件。

    例如整理高点10、现价10.12、失效价9.85、历史阻力12，输出突破确认及
    约6.96倍结构空间/风险；该阻力是历史价格，不是预测目标或上涨概率。
    """
    base_start = breakout_index - policy.consolidation_sessions
    base = bars[base_start:breakout_index]
    peak_zone = bars[: base_start - policy.consolidation_sessions]
    peak_index = max(range(len(peak_zone)), key=lambda index: peak_zone[index].high)
    resistance = peak_zone[peak_index].high
    trough = min(bar.low for bar in bars[peak_index + 1 : base_start])
    pivot = max(bar.high for bar in base)
    support = min(bar.low for bar in base)
    breakout = bars[breakout_index]
    current = bars[-1]
    amount_baseline = median(bar.amount for bar in bars[breakout_index - 20 : breakout_index])
    invalidation = support * (1 - policy.invalidation_buffer)
    distance = current.close - invalidation
    reward_risk = (resistance - current.close) / distance if distance > 0 else 0.0
    current_average = sum(bar.close for bar in bars[-20:]) / 20
    previous_average = sum(bar.close for bar in bars[-21:-1]) / 20
    values = {
        "entry_price": current.close,
        "confirmation_price": pivot,
        "invalidation_price": invalidation,
        "historical_resistance": resistance,
        "pullback_depth": 1 - trough / resistance,
        "higher_low_ratio": support / trough - 1,
        "base_range": pivot / support - 1,
        "breakout_extension": current.close / pivot - 1,
        "entry_risk": distance / current.close,
        "structural_reward_risk": reward_risk,
        "amount_confirmation_ratio": breakout.amount / amount_baseline,
    }
    if not all(isfinite(value) for value in values.values()):
        return StructureObservation(
            {}, {"pattern": "unknown"}, (), "right_side_calculation_invalid"
        )
    checks: tuple[tuple[bool, str], ...] = (
        (values["pullback_depth"] >= policy.minimum_pullback, "right_side_pullback_missing"),
        (values["higher_low_ratio"] >= policy.minimum_higher_low, "right_side_higher_low_missing"),
        (values["base_range"] <= policy.maximum_base_range, "right_side_base_too_wide"),
        (
            current.close > current_average and current_average >= previous_average,
            "right_side_trend_unconfirmed",
        ),
        (breakout.close > pivot, "right_side_breakout_unconfirmed"),
        (
            values["amount_confirmation_ratio"] >= policy.minimum_amount_ratio,
            "right_side_amount_confirmation_missing",
        ),
    )
    if breakout_index != len(bars) - 1:
        checks += (
            (
                current.low <= pivot * (1 + policy.retest_tolerance)
                and all(
                    bar.low >= pivot * (1 - policy.retest_tolerance) and bar.close >= pivot
                    for bar in bars[breakout_index + 1 :]
                )
                and current.close > bars[-2].close
                and current.amount <= breakout.amount,
                "right_side_retest_unconfirmed",
            ),
        )
    checks += (
        (
            0 <= values["breakout_extension"] <= policy.maximum_breakout_extension,
            "right_side_entry_extended",
        ),
        (0 < values["entry_risk"] <= policy.maximum_entry_risk, "right_side_entry_risk_excessive"),
        (reward_risk >= policy.minimum_reward_risk, "right_side_reward_risk_insufficient"),
    )
    reason = next((reason for passed, reason in checks if not passed), None)
    return StructureObservation(
        {key: round(value, 12) for key, value in values.items()},
        {
            "pattern": "breakout" if breakout_index == len(bars) - 1 else "retest",
            "breakout_date": breakout.day,
            "base_start_date": base[0].day,
            "base_end_date": base[-1].day,
            "data_as_of": current.day,
            "structure_checks": {reason: passed for passed, reason in checks},
            "resistance_basis": "historical_linked_price_not_forecast",
        },
        (),
        reason,
    )


def observe_structure(inputs: ResearchInputs, identity: str) -> StructureObservation:
    """检查当天突破或最近五日突破后的回踩，始终保留失败和窗口血缘。

    输入完整且不过度延伸的回调突破窗口返回reason=None；左侧下跌返回明确原因。
    多个可行结构按突破日由近到远确定选择，不使用未来价格寻找最佳收益。
    """
    bars, references, reason = price_window(inputs, identity)
    if reason is not None:
        return StructureObservation({}, {"pattern": "unknown"}, references, reason)
    observations = tuple(
        structure_at(bars, index, inputs.config.right_side)
        for index in range(
            len(bars) - 1, len(bars) - inputs.config.right_side.confirmation_sessions - 1, -1
        )
    )
    selected = next((item for item in observations if item.reason is None), observations[0])
    return StructureObservation(selected.values, selected.details, references, selected.reason)
