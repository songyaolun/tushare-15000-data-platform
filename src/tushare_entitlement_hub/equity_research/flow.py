"""资金支持和拥挤风险采用独立坐标，交易可行性由三值门禁约束。"""

from __future__ import annotations

from datetime import date
from math import isclose
from statistics import mean, median
from typing import Any, cast

from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef, QualityGateResult, Truth
from .common import gate, latest, make_result, metric, percentiles
from .flow_consumers import auxiliary_observation
from .governance import _ambiguous, _number
from .models import Metric, ResearchInputs, ResearchResult, ResearchRow

FLOW_FIELDS = {
    "moneyflow": ("buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"),
    "moneyflow_dc": ("net_amount",),
    "moneyflow_ths": ("net_amount",),
}
AUXILIARY_FIELDS = {
    "top_inst": ("exalter", "side", "buy", "sell", "net_buy", "reason"),
    "top_list": ("name", "reason", "l_buy", "l_sell", "net_amount"),
    "hm_detail": ("hm_name", "hm_orgs", "buy_amount", "sell_amount", "net_amount"),
    "hm_list": ("name", "desc", "orgs"),
    "hsgt_top10": ("name", "rank", "market_type", "net_amount", "amount"),
    "stock_hsgt": ("type", "type_name"),
    "margin_secs": ("exchange",),
    "slb_sec": ("lent_qnt", "end_bal"),
    "slb_sec_detail": ("tenor", "fee_rate", "lent_qnt"),
    "limit_list_d": ("limit", "open_times", "fd_amount", "amount", "limit_times"),
    "limit_list_ths": ("limit_type", "open_num", "lu_desc"),
    "limit_step": ("nums",),
    "stk_shock": ("reason", "period"),
    "cyq_perf": ("cost_5pct", "cost_50pct", "cost_95pct", "weight_avg", "winner_rate"),
    "rt_min": ("time", "trade_time", "freq", "close", "vol", "amount"),
}


def _auxiliary_rows(inputs: ResearchInputs, api: str, entity_id: str) -> tuple[ResearchRow, ...]:
    """字典只随本证券实际席位标签进入局部证据；例：A的甲标签不会带入B的乙字典。"""
    if api == "hm_list":
        names = {row.text("hm_name") for row in inputs.rows("hm_detail", entity_id)} - {None}
        return tuple(row for row in inputs.rows(api) if row.text("name") in names)
    return inputs.rows(api, entity_id)


def _aligned(row: ResearchRow | None, daily: ResearchRow | None) -> bool:
    """同证券来源必须有相同业务日期；例：昨日融资与今日行情返回False。"""
    return (
        row is not None
        and daily is not None
        and row.availability.data_as_of is not None
        and row.availability.data_as_of == daily.availability.data_as_of
    )


def _positive(row: ResearchRow | None, field: str, unit: str) -> float | None:
    """只返回合格正值，零分母保留未知；例：已认证amount=0返回None。"""
    value = _number(row, field, unit)
    return value if value is not None and value > 0 else None


def _source_flow(
    inputs: ResearchInputs,
    entity_id: str,
    api: str,
    daily: ResearchRow | None,
) -> Metric:
    """各来源金额只除同日成交额，不跨供应商加总；例：净额20/成交额100得到0.2。"""
    row = latest(inputs.rows(api, entity_id))
    fields = FLOW_FIELDS[api]
    name = "large_flow_ratio" if api == "moneyflow" else api + "_ratio"
    denominator = _positive(daily, "amount", "元")
    amounts = [_number(row, field, "元") for field in fields]
    reasons = list(inputs.reasons(api, entity_id))
    if not _aligned(row, daily):
        reasons.append("flow_date_unaligned")
    if denominator is None or any(value is None for value in amounts):
        reasons.append("flow_amount_or_unit_unverified")
    if _activity(daily)[0] is not Truth.TRUE:
        reasons.append("flow_trading_activity_unverified")
    references = row.evidence(*fields) if row else ()
    if daily is not None:
        references += daily.evidence("amount")
    net = sum(
        float(value) * (1 if index < 2 else -1)
        for index, value in enumerate(amounts)
        if value is not None
    )
    return metric(
        name, None if reasons else net / cast(float, denominator), "ratio", references, reasons
    )


def _leverage(inputs: ResearchInputs, entity_id: str, daily: ResearchRow | None) -> list[Metric]:
    """同日融资余额与流通市值比较，换手率单列；例：10元/100元得到0.1。"""
    basic = latest(inputs.rows("daily_basic", entity_id))
    margin = latest(inputs.rows("margin_detail", entity_id))
    capitalization = _positive(basic, "circ_mv", "元")
    balance = _number(margin, "rzye", "元")
    matched = _aligned(basic, daily) and _aligned(margin, daily)
    ratio = (
        balance / capitalization
        if (matched and balance is not None and balance >= 0 and capitalization is not None)
        else None
    )
    references = margin.evidence("rzye") if margin else ()
    if basic is not None:
        references += basic.evidence("circ_mv")
    turnover = _number(basic, "turnover_rate", "fraction")
    return [
        metric(
            "margin_intensity",
            ratio,
            "ratio",
            references,
            inputs.reasons("margin_detail", entity_id),
        ),
        metric(
            "turnover_rate",
            turnover if _aligned(basic, daily) and turnover is not None and turnover >= 0 else None,
            "ratio",
            basic.evidence("turnover_rate") if basic else (),
            inputs.reasons("daily_basic", entity_id),
        ),
    ]


def _overhang(inputs: ResearchInputs, entity_id: str, daily: ResearchRow | None) -> Metric:
    """只从认证完整且权重和为1的成本分布计算套牢占比；例：高于收盘权重0.3得到0.3。"""
    rows = inputs.rows("cyq_chips", entity_id)
    price = _positive(daily, "close", "元")
    weights = [_number(row, "percent", "fraction") for row in rows]
    prices = [_positive(row, "price", "元") for row in rows]
    reasons = list(inputs.reasons("cyq_chips", entity_id))
    if "cyq_chips" not in inputs.config.complete_scopes or not rows:
        reasons.append("chip_distribution_scope_unknown")
    if (
        price is None
        or any(value is None or value < 0 or value > 1 for value in weights)
        or any(value is None for value in prices)
        or _ambiguous(rows)
        or any(not _aligned(row, daily) for row in rows)
    ):
        reasons.append("chip_distribution_unverified")
    if not isclose(sum(value for value in weights if value is not None), 1, abs_tol=1e-8):
        reasons.append("chip_weights_not_complete")
    value = (
        None
        if reasons
        else sum(
            cast(float, weight)
            for weight, quote in zip(weights, prices, strict=True)
            if cast(float, quote) > cast(float, price)
        )
    )
    references = tuple(ref for row in rows for ref in row.evidence("price", "percent"))
    if daily is not None:
        references += daily.evidence("close")
    return metric("overhang", value, "fraction", references, reasons)


def _activity(daily: ResearchRow | None) -> tuple[Truth, Truth, str]:
    """价格可用与量额一致性分别判定；例：量0额10使量额门禁失败且价格不可用。"""
    close, volume, amount = (
        _number(daily, field, unit)
        for field, unit in (
            ("close", "元"),
            ("vol", "股"),
            ("amount", "元"),
        )
    )
    if volume is None or amount is None:
        consistency, activity = Truth.UNKNOWN, "unknown"
    elif volume < 0 or amount < 0 or (volume > 0) != (amount > 0):
        consistency, activity = Truth.FALSE, "contradictory"
    else:
        consistency, activity = Truth.TRUE, "traded" if volume > 0 else "no_trade"
    no_trade = daily is not None and (
        daily.text("market_activity_status") == "no_trade"
        or daily.text("price_type") in {"reference", "reference_close", "settlement"}
        or any("no_traded_price" in fact.reason_codes for fact in daily.numeric.values())
    )
    if no_trade or activity in {"no_trade", "contradictory"} or (close is not None and close <= 0):
        price = Truth.FALSE
    elif close is None or activity == "unknown":
        price = Truth.UNKNOWN
    else:
        price = Truth.TRUE
    return price, consistency, activity


def _suspension(inputs: ResearchInputs, entity_id: str, daily: ResearchRow | None) -> Truth:
    """按同日停复牌及明确时段判定，未知时段不作全天停牌；例：S且空时段为FALSE。"""
    rows = tuple(row for row in inputs.rows("suspend_d", entity_id) if _aligned(row, daily))
    if any(row.text("suspend_type") == "S" and "suspend_timing" in row.raw for row in rows):
        return Truth.FALSE
    if any(row.text("suspend_type") != "R" for row in rows):
        return Truth.UNKNOWN
    if (rows or "suspend_d" in inputs.config.complete_scopes) and not inputs.reasons(
        "suspend_d", entity_id
    ):
        return Truth.TRUE
    return Truth.UNKNOWN


def _limits(
    inputs: ResearchInputs,
    entity_id: str,
    daily: ResearchRow | None,
) -> tuple[Truth, Metric]:
    """比较同日上下限价，边界内才通过；例：收盘10、上限10得到FALSE和距离0。"""
    limits = latest(inputs.rows("stk_limit", entity_id))
    close = _positive(daily, "close", "元")
    upper, lower = _positive(limits, "up_limit", "元"), _positive(limits, "down_limit", "元")
    references = limits.evidence("up_limit", "down_limit") if limits else ()
    if daily is not None:
        references += daily.evidence("close")
    if (
        close is None
        or upper is None
        or lower is None
        or upper <= lower
        or not _aligned(limits, daily)
    ):
        return Truth.UNKNOWN, metric(
            "limit_distance", None, "ratio", references, ("limit_price_unverified",)
        )
    distance = min(upper - close, close - lower) / close
    return (
        Truth.TRUE if lower < close < upper else Truth.FALSE,
        metric("limit_distance", distance, "ratio", references),
    )


def _freshness(inputs: ResearchInputs, daily: ResearchRow | None) -> Truth:
    """业务日期与最近采集年龄独立检查；例：缺预计交易日返回UNKNOWN，旧于预计日为FALSE。"""
    if daily is None:
        return Truth.UNKNOWN
    expected = inputs.config.foundation.expected_data_as_of.get("daily")
    actual = daily.availability.data_as_of
    if expected is None or actual is None:
        return Truth.UNKNOWN
    age = (
        time_boundary(inputs.analysis_as_of) - time_boundary(daily.availability.observed_at)
    ).total_seconds() / 86400
    if actual < expected or age > inputs.config.thresholds.maximum_observation_age_days:
        return Truth.FALSE
    return Truth.TRUE if actual == expected else Truth.UNKNOWN


def _shock(inputs: ResearchInputs, entity_id: str) -> Truth:
    """严重异动记录不按文本period猜结束日；例：已发生且无确认终点的监管记录阻断。"""
    rows = inputs.rows("stk_high_shock", entity_id)
    active = tuple(
        row
        for row in rows
        if row.availability.effective_to is None
        or time_boundary(row.availability.effective_to) > time_boundary(inputs.analysis_as_of)
    )
    if active:
        return Truth.FALSE
    if "stk_high_shock" in inputs.config.complete_scopes and not inputs.reasons(
        "stk_high_shock", entity_id
    ):
        return Truth.TRUE
    return Truth.UNKNOWN


def _gates(
    inputs: ResearchInputs,
    entity_id: str,
    daily: ResearchRow | None,
) -> tuple[list[QualityGateResult], Metric, str]:
    """生成六项独立门禁并保留活动状态；例：过期限价未知不覆盖已确认停牌失败。"""
    price, volume, activity = _activity(daily)
    limit, distance = _limits(inputs, entity_id, daily)
    states = {
        "suspension": _suspension(inputs, entity_id, daily),
        "price": price,
        "limit": limit,
        "severe_shock": _shock(inputs, entity_id),
        "volume_amount": volume,
        "freshness": _freshness(inputs, daily),
    }
    references = tuple(
        ref
        for api in ("daily", "suspend_d", "stk_limit", "stk_high_shock")
        for row in inputs.rows(api, entity_id)
        for ref in row.evidence(
            "close",
            "vol",
            "amount",
            "suspend_type",
            "suspend_timing",
            "up_limit",
            "down_limit",
            "period",
        )
    )
    return (
        [
            gate("flow_" + name, state, name + "_failed_or_unknown", references=references)
            for name, state in states.items()
        ],
        distance,
        activity,
    )


def _block_trades(inputs: ResearchInputs, entity_id: str, daily: ResearchRow | None) -> list[Any]:
    """每笔大宗交易单列同日折价，不合并匿名席位；例：成交9/收盘10得到折价0.1。"""
    output = []
    close = _positive(daily, "close", "元")
    for row in inputs.rows("block_trade", entity_id):
        price = _positive(row, "price", "元")
        discount = (
            1 - price / close
            if price is not None and close is not None and _aligned(row, daily)
            else None
        )
        output.append(
            {
                "source_record_id": row.raw["source_record_id"],
                "block_discount": discount,
                "reason": None if discount is not None else "block_price_or_date_unverified",
            }
        )
    return output


def build_flow(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """输出每只显式股票的资金支持、拥挤风险及交易门禁，不生成交易指令。

    例：三个来源比例为[-0.1,0,0.1]时方向为0、一致性为0；没有两个来源则一致性未知。
    只对齐同日人民币口径，匿名席位、分钟和前十样本保留局部证据。
    """
    features: dict[str, list[Metric]] = {}
    daily_rows = {
        identity: latest(inputs.rows("daily", identity)) for identity in inputs.entity_ids
    }
    for entity_id, daily in daily_rows.items():
        flows = [_source_flow(inputs, entity_id, api, daily) for api in FLOW_FIELDS]
        values = [item.value for item in flows if item.value is not None]
        other = [item.value for item in flows[1:] if item.value is not None]
        references = tuple(ref for item in flows for ref in item.evidence_refs)
        signs = [(value > 0) - (value < 0) for value in values]
        center = median(values) if values else None
        direction = float((center > 0) - (center < 0)) if center is not None else None
        features[entity_id] = flows + [
            metric(
                "other_flow_ratio_median", median(other) if other else None, "ratio", references
            ),
            metric("flow_source_median", center, "ratio", references),
            metric("flow_direction", direction, "direction", references),
            metric(
                "sign_agreement",
                abs(sum(signs)) / len(signs) if len(signs) >= 2 else None,
                "fraction",
                references,
                () if len(signs) >= 2 else ("flow_sources_insufficient",),
            ),
            *_leverage(inputs, entity_id, daily),
            _overhang(inputs, entity_id, daily),
        ]
    axes = {
        "flow_support": ("large_flow_ratio", "other_flow_ratio_median"),
        "crowding": ("turnover_rate", "margin_intensity", "overhang"),
    }
    profiles = {
        identity: "flow_sources:"
        + ":".join(item.name for item in items[:3] if item.value is not None)
        for identity, items in features.items()
    }
    rank_cache = {
        (profile, axis, name): percentiles(
            inputs,
            {
                identity: next(item.value for item in features[identity] if item.name == name)
                for identity in features
                if profiles[identity] == profile
            },
            profile + ":" + axis + ":" + name,
        )
        for profile in set(profiles.values())
        for axis, names in axes.items()
        for name in names
    }
    results = []
    for entity_id, items in features.items():
        components: dict[str, Any] = {}
        for axis, names in axes.items():
            normalized = []
            axis_references: list[EvidenceRef] = []
            components[axis] = []
            for name in names:
                rank = rank_cache[profiles[entity_id], axis, name][entity_id]
                normalized.append(rank.percentile)
                item = next(item for item in items if item.name == name)
                axis_references.extend(item.evidence_refs)
                components[axis].append(
                    {
                        "feature": name,
                        "raw_value": item.value,
                        "percentile": rank.percentile,
                        "peer_count": rank.peer_count,
                        "peer_level": rank.peer_level,
                        "weight": 1 / len(names),
                        "contribution": None
                        if rank.percentile is None
                        else 100 * rank.percentile / len(names),
                    }
                )
            score = (
                100 * mean(cast(float, value) for value in normalized)
                if all(value is not None for value in normalized)
                else None
            )
            items.append(
                metric(
                    axis,
                    score,
                    "score",
                    axis_references,
                    () if score is not None else (axis + "_feature_or_peer_unknown",),
                )
            )
        daily = daily_rows[entity_id]
        gates, distance, activity = _gates(inputs, entity_id, daily)
        auxiliary = [
            auxiliary_observation(inputs, row, entity_id, fields)
            for api, fields in AUXILIARY_FIELDS.items()
            for row in _auxiliary_rows(inputs, api, entity_id)
        ]
        references = tuple(
            ref
            for api, fields in AUXILIARY_FIELDS.items()
            for row in _auxiliary_rows(inputs, api, entity_id)
            for ref in row.evidence(*fields)
        )
        references += tuple(
            ref
            for row in inputs.rows("block_trade", entity_id)
            for ref in row.evidence("price", "trade_date")
        )
        results.append(
            make_result(
                inputs,
                "flow_feasibility/v1",
                entity_id,
                metrics=items + [distance],
                gates=gates,
                profile=profiles[entity_id],
                references=references,
                details={
                    "score_components": components,
                    "market_activity": activity,
                    "source_disagreement": {item.name: item.value for item in items[:3]},
                    "block_trades": _block_trades(inputs, entity_id, daily),
                    "auxiliary": auxiliary,
                    "scope_notes": [
                        "anonymous_seats_not_aggregated",
                        "top10_is_selected_sample",
                        "minutes_do_not_extend_observed_scope",
                        "chips_are_vendor_estimates",
                    ],
                    "analysis_date": date.fromisoformat(inputs.analysis_as_of[:10]).isoformat(),
                },
            )
        )
    return tuple(results)
