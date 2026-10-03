"""已采 A 股截面的市场压力与独立宏观状态，不推导仓位或收益。"""

from __future__ import annotations

from dataclasses import replace
from math import ceil, fsum, isfinite
from typing import Any, cast

from ..product_foundation.contracts import EvidenceRef, FeatureContribution, Truth
from ..product_foundation.numeric import price_gate
from ..product_foundation.temporal import availability_gate
from ..standard import normalize_date, normalize_period
from .common import gate, latest, make_result, metric
from .models import MARKET_ENTITY, ResearchInputs, ResearchResult, ResearchRow


def _day(row: ResearchRow) -> str | None:
    """读取已声明且一致的交易日；例：trade_date=20260930得到2026-09-30。

    缺失、非法日期或与时间适配结果冲突时返回None，避免把不同日期相连。
    """
    try:
        day = normalize_date(row.text("trade_date"))
    except (ValueError, TypeError):
        return None
    return day if row.availability.data_as_of == day else None


def _month(row: ResearchRow) -> int | None:
    """将真实月份转换为相邻性序号；例：202601比202512大1。

    接受YYYYMM或YYYY-MM；非法月份和缺字段返回None，不从观察日期猜月份。
    """
    try:
        value = normalize_period(row.text("month"), "month").replace("-", "")
    except (ValueError, TypeError):
        return None
    return int(value[:4]) * 12 + int(value[4:]) - 1


def _quarter(row: ResearchRow) -> int | None:
    """将真实季度转换为相邻序号，并核对已声明报告期；例：2026Q1紧邻2025Q4。

    仅接受合法季度，缺字段、非法季度或报告期适配冲突返回None，绝不按交易日连接。
    """
    try:
        value = normalize_period(row.text("quarter"), "quarter")
    except (ValueError, TypeError):
        return None
    if row.availability.report_period is not None and row.availability.report_period != value:
        return None
    return int(value[:4]) * 4 + int(value[-1]) - 1


def _macro(inputs: ResearchInputs, api: str) -> tuple[dict[str, Any], tuple[ResearchRow, ...]]:
    """生成独立月频或季频状态和相邻变化；例：GDP仅有2026Q2时只显示当季同比。

    PMI采用认证index点数，增速采用ratio，社融当月量采用CNY元。宏观不进入E0。
    同期间多版本须能唯一选择，未来期间与尚未可见记录不能使用；公开可用时间单列。
    """
    fields, unit = {
        "cn_pmi": (("pmi010000",), "index"),
        "cn_ppi": (("ppi_yoy",), "ratio"),
        "cn_m": (("m2_yoy", "m1_yoy"), "ratio"),
        "sf_month": (("inc_month",), "元"),
        "cn_gdp": (("gdp_yoy",), "ratio"),
        "cn_cpi": (("nt_yoy",), "ratio"),
    }[api]
    quarterly = api == "cn_gdp"
    period_field = "quarter" if quarterly else "month"
    period_number = _quarter if quarterly else _month
    year, month = int(inputs.analysis_as_of[:4]), int(inputs.analysis_as_of[5:7])
    cutoff = year * 4 + (month - 1) // 3 if quarterly else year * 12 + month - 1
    periods: dict[int, list[ResearchRow]] = {}
    for row in inputs.rows(api):
        period = period_number(row)
        if period is not None and period <= cutoff:
            periods.setdefault(period, []).append(row)
    state: dict[str, Any] = {
        "source_api": api,
        "unit": unit,
        "value": None,
        "change": None,
        "status": "unknown",
        "reason_codes": list(inputs.source_reasons.get(api, ())),
        "mode": "observed_current_only",
        "frequency": period_field,
        "period": None,
        "available_at": None,
    }
    if not periods:
        state["reason_codes"].append("macro_period_unavailable")
        return state, ()
    period = max(periods)
    current = latest(tuple(periods[period]))
    previous = latest(tuple(periods.get(period - 1, ())))
    if current is None or current.raw.get("business_variant_key"):
        state["reason_codes"].append("macro_version_ambiguous")
        return state, tuple(periods[period])
    state["period"] = normalize_period(current.text(period_field), period_field)
    state["available_at"] = current.availability.available_at
    state["public_available_at"] = current.availability.public_available_at
    observations = (current,) if previous is None else (previous, current)
    values = [current.number(field, unit) for field in fields]
    if any(value is None for value in values) or (
        api == "sf_month" and current.numeric["inc_month"].currency != "CNY"
    ):
        state["reason_codes"].append("macro_unit_or_value_unverified")
        return state, observations
    numbers = [cast(float, number) for number in values]
    value = numbers[0] - numbers[1] if len(numbers) == 2 else numbers[0]
    if not isfinite(value):
        state["reason_codes"].append("nonfinite_macro_calculation")
        return state, observations
    state.update(value=value, status="current_only")
    if api == "cn_pmi":
        state["direction"] = (
            "expansion" if value > 50 else "contraction" if value < 50 else "neutral"
        )
    if api in {"cn_ppi", "cn_m", "cn_gdp", "cn_cpi"}:
        old = [] if previous is None else [previous.number(field, unit) for field in fields]
        if (
            previous is not None
            and old
            and all(number is not None for number in old)
            and not previous.raw.get("business_variant_key")
        ):
            old_numbers = [cast(float, number) for number in old]
            previous_value = (
                old_numbers[0] - old_numbers[1] if len(old_numbers) == 2 else old_numbers[0]
            )
            change = value - previous_value
            if isfinite(change):
                state.update(
                    change=change,
                    status="adjacent_change",
                    previous_period=normalize_period(previous.text(period_field), period_field),
                    previous_available_at=previous.availability.available_at,
                )
            else:
                state["reason_codes"].append("nonfinite_macro_calculation")
        else:
            state["reason_codes"].append("adjacent_period_unavailable")
    return state, observations


def _suspension(
    inputs: ResearchInputs,
    identity: str,
    trade_day: str | None,
) -> tuple[Truth, tuple[ResearchRow, ...], list[dict[str, Any]]]:
    """按证券和同交易日连接停复牌，明确S优先保守排除，R只说明当日复牌记录。

    例：日线有成交但同日S记录仍输出TRUE；日内时段单列，不把日内停牌当全天无成交。
    变体或未知类型只输出UNKNOWN，昨日记录不推演成今日状态。
    同日行被质量或时间门禁拒绝时保留UNKNOWN；无记录仅在来源完整认证后才返回FALSE。
    """
    rows = tuple(
        row
        for row in inputs.rows("suspend_d", identity)
        if trade_day is not None and _day(row) == trade_day
    )
    accepted = tuple(
        row
        for row in rows
        if row.raw.get("quality_tier") in {"verified", "usable_with_annotation"}
        and row.raw.get("is_quarantined") is not True
        and availability_gate(row.availability, inputs.analysis_as_of).passed
    )
    rejected = len(accepted) != len(rows)
    rows = accepted
    observations = []
    for row in rows:
        timing = row.text("suspend_timing")
        scope = (
            "intraday"
            if timing is not None
            else "full_day"
            if ("suspend_timing" in row.raw and row.raw["suspend_timing"] in (None, ""))
            else "unknown"
        )
        observations.append(
            {
                "entity_id": identity,
                "trade_date": trade_day,
                "source_record_id": row.availability.source_record_id,
                "suspend_type": row.text("suspend_type"),
                "suspend_timing": timing,
                "scope": scope,
                "available_at": row.availability.available_at,
            }
        )
    if any(
        row.text("suspend_type") == "S" and not row.raw.get("business_variant_key") for row in rows
    ):
        return Truth.TRUE, rows, observations
    uncertain = (
        rejected
        or bool(inputs.reasons("suspend_d", identity))
        or any(
            row.text("suspend_type") not in {"S", "R"} or row.raw.get("business_variant_key")
            for row in rows
        )
        or (not rows and (trade_day is None or "suspend_d" not in inputs.config.complete_scopes))
    )
    return Truth.UNKNOWN if uncertain else Truth.FALSE, rows, observations


def build_market(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """输出唯一市场研究记录，所有分母限于显式绑定、同日且有有效成交的证券。

    例如两只合格股票一涨一跌、金额为100/300且一只跌停，得到广度0.5、金额集中度
    0.75、跌停占比0.5和E0=58.333…；前10%数量向上取整。任何一项缺失均不发布E0。
    无成交、未知价格、停牌和旧日记录分别记录，空输入也返回带缺口的单一结果。
    """
    visible = {
        api: rows
        if api == "suspend_d"
        else tuple(
            row for row in rows if availability_gate(row.availability, inputs.analysis_as_of).passed
        )
        for api, rows in inputs.rows_by_api.items()
    }
    inputs = replace(inputs, rows_by_api=visible)
    identities = set(inputs.entity_ids)
    daily = tuple(row for row in inputs.rows("daily") if row.entity_id in identities)
    daily_by_entity: dict[str | None, list[ResearchRow]] = {}
    limits_by_entity: dict[str | None, list[ResearchRow]] = {}
    for row in daily:
        daily_by_entity.setdefault(row.entity_id, []).append(row)
    for row in inputs.rows("stk_limit"):
        limits_by_entity.setdefault(row.entity_id, []).append(row)
    valid_days = [
        day for row in daily if (day := _day(row)) is not None and day <= inputs.analysis_as_of[:10]
    ]
    trade_day = max(valid_days, default=None)
    eligible: list[ResearchRow] = []
    excluded: list[ResearchRow] = []
    counts = {
        "excluded_no_trade_count": 0,
        "excluded_price_unknown_count": 0,
        "excluded_stale_count": 0,
        "excluded_suspended_count": 0,
        "excluded_suspension_unknown_count": 0,
    }
    suspension_observations = []
    suspension_refs: list[EvidenceRef] = []
    suspension_contrary: list[EvidenceRef] = []
    for identity in inputs.entity_ids:
        rows = tuple(daily_by_entity.get(identity, ()))
        selected = latest(
            tuple(row for row in rows if trade_day is not None and _day(row) == trade_day)
        )
        suspension, suspension_rows, event_details = _suspension(inputs, identity, trade_day)
        suspension_observations.extend(event_details)
        state_refs = tuple(
            ref
            for row in suspension_rows
            for ref in row.evidence("trade_date", "suspend_type", "suspend_timing")
        )
        suspension_refs.extend(state_refs)
        if suspension is not Truth.FALSE:
            counts[
                "excluded_suspended_count"
                if suspension is Truth.TRUE
                else "excluded_suspension_unknown_count"
            ] += 1
            excluded.extend((selected,) if selected is not None else rows)
            suspension_contrary.extend(state_refs)
            continue
        if selected is None:
            counts["excluded_price_unknown_count"] += 1
            counts["excluded_stale_count"] += int(
                bool(rows) and all(_day(row) != trade_day for row in rows)
            )
            excluded.extend(rows)
            continue
        if selected.raw.get("market_activity_status") == "suspended":
            counts["excluded_suspended_count"] += 1
        if (
            selected.raw.get("market_activity_status") == "suspended"
            or not price_gate(selected.raw, "return").passed
        ):
            known_no_trade = selected.raw.get("market_activity_status") in {"no_trade", "suspended"}
            known_no_trade |= any(selected.raw.get(field) == 0 for field in ("vol", "amount"))
            counts[
                "excluded_no_trade_count" if known_no_trade else "excluded_price_unknown_count"
            ] += 1
            excluded.append(selected)
            continue
        values = [
            selected.number("close", "元"),
            selected.number("amount", "元"),
            selected.number("vol"),
            selected.number("pct_chg"),
        ]
        if (
            any(value is None for value in values)
            or any(selected.numeric[field].currency != "CNY" for field in ("close", "amount"))
            or any(cast(float, value) <= 0 for value in values[:3])
            or selected.numeric["close"].dimension != "price"
            or selected.numeric["amount"].dimension != "money"
            or selected.numeric["pct_chg"].dimension != "ratio"
            or selected.raw.get("business_variant_key")
        ):
            counts["excluded_price_unknown_count"] += 1
            excluded.append(selected)
            continue
        eligible.append(selected)
    denominator = len(eligible)
    price_refs = tuple(
        ref
        for row in eligible
        for ref in row.evidence("trade_date", "close", "vol", "amount", "pct_chg")
    )
    breadth = (
        sum(cast(float, row.number("pct_chg")) > 0 for row in eligible) / denominator
        if denominator
        else None
    )
    amounts = sorted((cast(float, row.number("amount", "元")) for row in eligible), reverse=True)
    # 缩放后求和，避免极端但有限的已认证金额使合计溢出。
    concentration = (
        (
            fsum(value / amounts[0] for value in amounts[: ceil(denominator / 10)])
            / fsum(value / amounts[0] for value in amounts)
        )
        if denominator
        else None
    )
    down_count = 0
    limit_refs: list[EvidenceRef] = []
    limit_known = 0
    for row in eligible:
        limit = latest(
            tuple(
                item for item in limits_by_entity.get(row.entity_id, ()) if _day(item) == trade_day
            )
        )
        value = None if limit is None else limit.number("down_limit", "元")
        if (
            limit is not None
            and value is not None
            and value > 0
            and limit.numeric["down_limit"].currency == "CNY"
            and not limit.raw.get("business_variant_key")
        ):
            limit_known += 1
            down_count += int(cast(float, row.number("close", "元")) <= value)
            limit_refs.extend(limit.evidence("down_limit", "trade_date"))
    down_share = down_count / denominator if denominator and limit_known == denominator else None
    features = (
        metric(
            "breadth", breadth, "fraction", price_refs, () if denominator else ("no_valid_returns",)
        ),
        metric("liquidity_concentration", concentration, "fraction", price_refs),
        metric(
            "down_limit_share",
            down_share,
            "fraction",
            tuple(limit_refs) + price_refs,
            () if down_share is not None else ("down_limit_coverage_incomplete",),
        ),
    )
    complete = all(item.value is not None for item in features)
    components = tuple(
        FeatureContribution(
            feature_id=item.name,
            raw_value=item.value,
            unit=item.unit,
            transformed_value=(1 - item.value if item.name == "breadth" else item.value)
            if item.value is not None
            else None,
            weight=1 / 3,
            contribution=(100 / 3 * (1 - item.value if item.name == "breadth" else item.value))
            if complete and item.value is not None
            else None,
            transform="one_minus" if item.name == "breadth" else "identity",
            missing_reason=None if complete else "market_component_unavailable",
            evidence_refs=item.evidence_refs,
        )
        for item in features
    )
    score = sum(cast(float, item.contribution) for item in components) if complete else None
    states = []
    macro_refs: list[EvidenceRef] = []
    for api in ("cn_pmi", "cn_ppi", "cn_m", "sf_month", "cn_gdp", "cn_cpi"):
        state, observations = _macro(inputs, api)
        states.append(state)
        macro_refs.extend(
            ref
            for row in observations
            for ref in row.evidence(
                "month",
                "quarter",
                "pmi010000",
                "ppi_yoy",
                "m1_yoy",
                "m2_yoy",
                "inc_month",
                "gdp_yoy",
                "nt_yoy",
            )
        )
    result = make_result(
        inputs,
        "market_context/v1",
        MARKET_ENTITY,
        metrics=features
        + (metric("stress_score", score, "score", price_refs + tuple(limit_refs)),),
        gates=(
            gate(
                "valid_market_sample",
                Truth.TRUE if denominator else Truth.UNKNOWN,
                "market_sample_unavailable",
            ),
        ),
        profile="market_observed_daily",
        contributions=components,
        references=macro_refs + suspension_refs,
        contrary=tuple(ref for row in excluded for ref in row.evidence("close", "vol", "amount"))
        + tuple(suspension_contrary),
        details={
            "trade_date": trade_day,
            "denominator_count": denominator,
            "observed_scope_count": len(inputs.entity_ids),
            **counts,
            "down_limit_known_count": limit_known,
            "macro_states": states,
            "suspension_observations": suspension_observations,
            "research_topics": ["优先复核流动性"]
            if breadth is not None
            and breadth < 0.35
            and down_share is not None
            and down_share > 0.02
            else [],
            "thresholds_calibrated": False,
        },
    )
    result = replace(
        result,
        envelope=replace(result.envelope, data_as_of=trade_day or inputs.analysis_as_of[:10]),
    )
    return (result,)
