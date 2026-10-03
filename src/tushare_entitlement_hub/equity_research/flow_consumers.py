"""资金来源只形成有边界的局部证据；资格、供应商标签均不代替交易门禁。"""

from __future__ import annotations

import re
from math import isclose, isfinite
from typing import Any, cast

from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.temporal import availability_gate
from .common import gate, latest, unique_evidence
from .models import ResearchInputs, ResearchRow

FLOW_CONSUMER_FIELDS = {
    "cyq_perf": (
        "ts_code",
        "trade_date",
        "cost_5pct",
        "cost_50pct",
        "cost_95pct",
        "weight_avg",
        "winner_rate",
    ),
    "hm_detail": (
        "ts_code",
        "trade_date",
        "buy_amount",
        "sell_amount",
        "net_amount",
        "hm_name",
        "hm_orgs",
    ),
    "hm_list": ("name", "desc", "orgs"),
    "hsgt_top10": ("ts_code", "trade_date", "market_type", "net_amount", "amount", "name", "rank"),
    "limit_step": ("ts_code", "trade_date", "nums"),
    "margin_secs": ("ts_code", "trade_date", "exchange"),
    "slb_sec": ("ts_code", "trade_date", "lent_qnt", "end_bal"),
    "slb_sec_detail": ("ts_code", "trade_date", "tenor", "fee_rate", "lent_qnt"),
    "stock_hsgt": ("ts_code", "trade_date", "type", "type_name"),
    "top_list": ("ts_code", "trade_date", "reason", "l_buy", "l_sell", "net_amount", "name"),
}
NUMERIC_UNITS = {
    "cost_5pct": ("元",),
    "cost_50pct": ("元",),
    "cost_95pct": ("元",),
    "weight_avg": ("元",),
    "winner_rate": ("fraction",),
    "buy_amount": ("元",),
    "sell_amount": ("元",),
    "net_amount": ("元",),
    "amount": ("元",),
    "rank": ("count", "名"),
    "nums": ("count", "次"),
    "lent_qnt": ("股",),
    "end_bal": ("元",),
    "tenor": ("天", "day", "days"),
    "fee_rate": ("fraction",),
    "l_buy": ("元",),
    "l_sell": ("元",),
    "buy": ("元",),
    "sell": ("元",),
    "net_buy": ("元",),
    "close": ("元",),
    "price": ("元",),
    "vol": ("股",),
    "percent": ("fraction",),
    "open_times": ("count", "次"),
    "open_num": ("count", "次"),
    "fd_amount": ("元",),
    "limit_times": ("count", "次"),
}
NORTHBOUND_TYPES = {"HK_SH": "沪股通(港>沪)", "HK_SZ": "深股通(港>深)"}
SOUTHBOUND_TYPES = {"SH_HK": "港股通(沪>港)", "SZ_HK": "港股通(深>港)"}


def _value(row: ResearchRow, field: str) -> float | None:
    """仅取当前行已认证、有限且单位合格的数值；例：原始费率2但无证书返回None。"""
    fact = row.numeric.get(field)
    if (
        field not in row.raw
        or fact is None
        or fact.quality_tier != "verified"
        or (
            fact.entity_id != row.entity_id
            or fact.availability.source_record_id != row.availability.source_record_id
            or fact.availability.data_as_of != row.availability.data_as_of
        )
    ):
        return None
    for unit in NUMERIC_UNITS.get(field, ()):
        value = row.number(field, unit)
        if value is not None and isfinite(value) and (unit != "元" or fact.currency == "CNY"):
            return value
    return None


def _aliases(value: Any) -> tuple[str, ...]:
    """按明确分隔符读取供应商席位标签，不做模糊实体合并；例：'甲、乙'得到两项。"""
    if isinstance(value, str):
        return tuple(part.strip() for part in re.split(r"[、,，;；\n]", value) if part.strip())
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(item.strip() for item in value if item.strip())
    return ()


def _row_reasons(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
    *,
    metadata: bool = False,
    retained_previous: bool = False,
) -> tuple[str, ...]:
    """重验可见性、隔离和实体范围；例：股票B传给A保持未知，相邻旧观察仍须已公开。"""
    reasons = list(inputs.reasons(row.source_api, entity_id))
    if row.raw.get("quality_tier") == "quarantine" or row.raw.get("is_quarantined") is True:
        reasons.append("quarantined_record")
    if row.raw.get("business_variant_key"):
        reasons.append("business_variant_unresolved")
    if not metadata and (entity_id not in inputs.entity_ids or row.entity_id != entity_id):
        reasons.append("entity_binding_unknown")
    if availability_gate(row.availability, inputs.analysis_as_of).result is not Truth.TRUE:
        reasons.append("not_yet_available")
    if not metadata:
        day = row.availability.data_as_of
        expected = inputs.config.foundation.expected_data_as_of.get(row.source_api)
        expected = expected or inputs.config.foundation.expected_data_as_of.get("daily")
        if (
            day is None
            or day > inputs.analysis_as_of[:10]
            or (not retained_previous and expected and day != expected)
        ):
            reasons.append("business_date_unverified")
    return tuple(sorted(set(reasons)))


def _fields(row: ResearchRow, names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, str]]:
    """保存实际消费字段及质量，原始数字不冒充计算值；例：未认证买入额输出None。"""
    values: dict[str, Any] = {}
    quality = {}
    for name in names:
        if name not in row.raw or row.raw[name] is None:
            values[name], quality[name] = None, "missing"
        elif name in NUMERIC_UNITS:
            values[name] = _value(row, name)
            quality[name] = "verified" if values[name] is not None else "unit_or_value_unverified"
        elif name in {"orgs", "hm_orgs"}:
            values[name] = list(_aliases(row.raw[name]))
            quality[name] = "observed_label" if values[name] else "label_unverified"
        else:
            values[name] = row.text(name)
            quality[name] = "observed_label" if values[name] is not None else "label_unverified"
    return values, quality


def _same_day(left: ResearchRow, right: ResearchRow) -> bool:
    """两个来源须具有同一证券和业务日期；例：周一金额不与周二报价核对。"""
    return (
        left.entity_id == right.entity_id
        and left.availability.data_as_of is not None
        and left.availability.data_as_of == right.availability.data_as_of
    )


def _amount_check(row: ResearchRow, buy: str, sell: str, net: str) -> Truth:
    """核对已认证人民币买卖与净额，不跨行加总；例：10减3等于7时为TRUE。"""
    bought, sold, balance = (_value(row, name) for name in (buy, sell, net))
    if bought is None or sold is None or balance is None:
        return Truth.UNKNOWN
    return (
        Truth.TRUE
        if (
            bought >= 0
            and sold >= 0
            and isclose(bought - sold, balance, rel_tol=1e-8, abs_tol=1e-8)
        )
        else Truth.FALSE
    )


def _chip_check(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """以完整同日成本分布核对分位和均价；例：半数8元、半数12元的均价为10元。

    使用累计权重首次达到分位点的离散口径；同时要求认证成交报价。winner_rate只保存
    供应商筹码获利比例标签，不生成策略胜率或新增独立经济证据。
    """
    daily = latest(inputs.rows("daily", entity_id))
    chips = inputs.rows("cyq_chips", entity_id)
    chips = tuple(chip for chip in chips if _same_day(row, chip))
    references = tuple(
        ref
        for chip in chips
        if not _row_reasons(inputs, chip, entity_id)
        for ref in chip.evidence("price", "percent")
    )
    if daily is not None and not _row_reasons(inputs, daily, entity_id):
        references += daily.evidence("trade_date", "close", "vol", "amount")
    names = ("cost_5pct", "cost_50pct", "cost_95pct", "weight_avg")
    costs = [_value(row, name) for name in names]
    winner = _value(row, "winner_rate")
    quote = _value(daily, "close") if daily else None
    quote_valid = daily is not None and not _row_reasons(inputs, daily, entity_id)
    quote_valid = quote_valid and _same_day(row, daily) if daily else False
    quote_valid = bool(
        quote_valid
        and quote is not None
        and quote > 0
        and daily is not None
        and all((_value(daily, name) or 0) > 0 for name in ("vol", "amount"))
        and daily.raw.get("price_type")
        not in {
            "reference",
            "reference_close",
            "reference_price",
            "settlement",
        }
        and daily.raw.get("market_activity_status") != "no_trade"
        and daily.raw.get("price_observation_status")
        not in {
            "reference_close_without_trade",
            "no_price_observed",
        }
    )
    output: dict[str, Any] = {
        "costs_cny": dict(zip(names, costs, strict=True)),
        "winner_rate": winner if winner is not None and 0 <= winner <= 1 else None,
        "winner_rate_semantics": "vendor_chip_profit_fraction_not_strategy_win_rate",
        "quote_cny": quote if quote_valid else None,
        "quote_vs_average_cost": None,
        "distribution_comparison": "unknown",
        "quantile_method": "first_price_at_cumulative_weight",
        "recomputed_costs_cny": None,
    }
    if any(value is None or value <= 0 for value in costs) or not quote_valid:
        return output, Truth.UNKNOWN, references
    low, middle, high, average = (float(value) for value in costs if value is not None)
    relative_cost = cast(float, quote) / average - 1
    output["quote_vs_average_cost"] = relative_cost if isfinite(relative_cost) else None
    if not low <= middle <= high or (winner is not None and not 0 <= winner <= 1):
        output["distribution_comparison"] = "conflict"
        return output, Truth.FALSE, references
    points = [(_value(chip, "price"), _value(chip, "percent")) for chip in chips]
    if (
        "cyq_chips" not in inputs.config.complete_scopes
        or not points
        or any(_row_reasons(inputs, chip, entity_id) for chip in chips)
        or any(
            price is None or price <= 0 or weight is None or not 0 <= weight <= 1
            for price, weight in points
        )
        or not isclose(sum(weight or 0 for _, weight in points), 1, abs_tol=1e-8)
        or len({chip.raw.get("business_group_key") for chip in chips}) != len(chips)
    ):
        return output, Truth.UNKNOWN, references
    valid_points = sorted(
        (float(p), float(w)) for p, w in points if p is not None and w is not None
    )
    cumulative = 0.0
    recomputed: dict[str, float] = {}
    for price, weight in valid_points:
        cumulative += weight
        for name, quantile in zip(names[:3], (0.05, 0.5, 0.95), strict=True):
            if name not in recomputed and cumulative + 1e-12 >= quantile:
                recomputed[name] = price
    recomputed["weight_avg"] = sum(price * weight for price, weight in valid_points)
    output["recomputed_costs_cny"] = recomputed
    matched = all(
        isclose(float(cost), recomputed[name], rel_tol=1e-6, abs_tol=1e-8)
        for name, cost in zip(names, costs, strict=True)
        if cost is not None
    )
    output["distribution_comparison"] = "matched" if matched else "conflict"
    return (
        output,
        Truth.TRUE
        if matched and winner is not None
        else (Truth.FALSE if not matched else Truth.UNKNOWN),
        references,
    )


def _seat_labels(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """用精确游资名称连接字典，再核对席位标签；例：甲的'营业部A'只匹配甲的同名别名。"""
    candidates = tuple(
        item
        for item in inputs.rows("hm_list")
        if item.text("name") == row.text("hm_name") and row.text("hm_name") is not None
    )
    usable = tuple(
        item
        for item in candidates
        if not _row_reasons(
            inputs,
            item,
            entity_id,
            metadata=True,
        )
    )
    references = tuple(ref for item in usable for ref in item.evidence("name", "desc", "orgs"))
    details: dict[str, Any] = {
        "tag_link": "unknown",
        "tag_description": None,
        "alias_match": "unknown",
        "beneficial_owner": "unknown",
        "institutional_investor_evidence": False,
        "anonymous_seats_aggregated": False,
        "net_amount_consistency": _amount_check(
            row, "buy_amount", "sell_amount", "net_amount"
        ).value,
    }
    if len(usable) != 1 or len(candidates) != 1:
        state = _amount_check(row, "buy_amount", "sell_amount", "net_amount")
        return details, Truth.FALSE if state is Truth.FALSE else Truth.UNKNOWN, references
    label = usable[0]
    aliases = set(_aliases(label.raw.get("orgs")))
    seats = set(_aliases(row.raw.get("hm_orgs")))
    matched = bool(seats and aliases and seats <= aliases)
    details.update(
        {
            "tag_link": "exact_name",
            "tag_description": label.text("desc"),
            "alias_match": "matched" if matched else "unknown",
            "alias_names": sorted(aliases),
        }
    )
    state = _amount_check(row, "buy_amount", "sell_amount", "net_amount")
    details["net_amount_consistency"] = state.value
    if state is not Truth.FALSE and (not matched or label.text("desc") is None):
        state = Truth.UNKNOWN
    return details, state, references


def _stock_totals(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """按同日同上榜原因核对股票总量与各席位上界，匿名席位不求和；例：单席位买11大于总10冲突。"""
    state = _amount_check(row, "l_buy", "l_sell", "net_amount")
    checks = []
    references: tuple[EvidenceRef, ...] = ()
    for seat in inputs.rows("top_inst", entity_id):
        if (
            not _same_day(row, seat)
            or not row.text("reason")
            or seat.text("reason") != row.text("reason")
        ):
            continue
        if not _row_reasons(inputs, seat, entity_id):
            references += seat.evidence(
                "trade_date", "reason", "exalter", "side", "buy", "sell", "net_buy"
            )
        values = [
            _value(seat, "buy"),
            _value(seat, "sell"),
            _value(row, "l_buy"),
            _value(row, "l_sell"),
        ]
        check = Truth.UNKNOWN
        if not _row_reasons(inputs, seat, entity_id) and all(value is not None for value in values):
            bought, sold, total_buy, total_sell = (
                float(value) for value in values if value is not None
            )
            check = (
                Truth.TRUE if 0 <= bought <= total_buy and 0 <= sold <= total_sell else Truth.FALSE
            )
        checks.append(
            {
                "source_record_id": seat.availability.source_record_id,
                "within_stock_total": check.value,
            }
        )
        if check is Truth.FALSE:
            state = Truth.FALSE
    if not row.text("reason") and state is not Truth.FALSE:
        state = Truth.UNKNOWN
    return (
        {
            "net_amount_consistency": _amount_check(row, "l_buy", "l_sell", "net_amount").value,
            "seat_checks": checks,
            "seat_totals_aggregated": False,
            "selection_bias": "disclosed_top_list_only",
            "market_flow_inference": "not_supported",
        },
        state,
        references,
    )


def _eligibility(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """资格是当前名单成员证据，缺席不证明不合格；例：HK_SH仅说明沪股通资格观察。"""
    details: dict[str, Any] = {
        "eligibility": "unknown",
        "tradability": "unknown",
        "holdings": "unknown",
        "absence_implies_ineligible": False,
        "absence_implies_no_holdings": False,
        "membership_change": "unknown",
    }
    references: tuple[EvidenceRef, ...] = ()
    if row.source_api == "margin_secs":
        exchange = row.text("exchange")
        valid = exchange in {"SSE", "SZSE", "BSE"}
        details.update(
            {
                "eligibility": "listed" if valid else "unknown",
                "exchange": exchange,
                "denominator_scope": "explicit_collected_members_only",
                "scope_certified_complete": "margin_secs" in inputs.config.complete_scopes,
            }
        )
        return details, Truth.TRUE if valid else Truth.UNKNOWN, references
    kind, label = row.text("type"), row.text("type_name")
    if kind in SOUTHBOUND_TYPES:
        details["channel"] = "southbound_not_a_share_northbound"
        return details, Truth.UNKNOWN, references
    valid = kind in NORTHBOUND_TYPES
    expected = NORTHBOUND_TYPES.get(kind or "")
    details.update(
        {
            "channel": kind,
            "channel_label": label,
            "eligibility": "listed" if valid and label == expected else "unknown",
        }
    )
    previous = latest(
        tuple(
            item
            for item in inputs.rows("stock_hsgt", entity_id)
            if item.availability.data_as_of
            and row.availability.data_as_of
            and item.availability.data_as_of < row.availability.data_as_of
        )
    )
    if previous is not None and not _row_reasons(
        inputs,
        previous,
        entity_id,
        retained_previous=True,
    ):
        previous_kind = previous.text("type")
        previous_label = previous.text("type_name")
        references += previous.evidence("trade_date", "type", "type_name")
        if (
            valid
            and label == expected
            and previous_kind in NORTHBOUND_TYPES
            and (previous_label == NORTHBOUND_TYPES[previous_kind])
        ):
            details["channel_observation_change"] = {
                "from_type": previous_kind,
                "to_type": kind,
                "changed": previous_kind != kind,
                "period": [previous.availability.data_as_of, row.availability.data_as_of],
                "membership_change": "unknown",
            }
    return details, Truth.TRUE if valid and label == expected else Truth.UNKNOWN, references


def _lending(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """按证券及期限费率保留借券观察，余额变化只对相邻认证日期计算；例：2日至3日余额10到12为2。"""
    quantity = _value(row, "lent_qnt")
    details: dict[str, Any] = {
        "lent_shares": quantity,
        "borrowing_availability": "unknown",
        "tradability": "unknown",
        "total_short_interest": "not_supported",
        "historical_trend": "not_enabled_due_to_history",
    }
    references: tuple[EvidenceRef, ...] = ()
    if row.source_api == "slb_sec_detail":
        tenor, rate = _value(row, "tenor"), _value(row, "fee_rate")
        details.update(
            {
                "tenor_days": tenor,
                "fee_rate_fraction": rate,
                "strata_aggregated": False,
                "annualized_rate": "unknown",
            }
        )
        if tenor is None or rate is None or quantity is None:
            return details, Truth.UNKNOWN, references
        valid = tenor > 0 and float(tenor).is_integer() and rate >= 0 and quantity >= 0
        return details, Truth.TRUE if valid else Truth.FALSE, references
    balance = _value(row, "end_bal")
    details.update(
        {
            "ending_balance_cny": balance,
            "balance_change_cny": None,
            "lent_change_shares": None,
            "change_period": None,
        }
    )
    earlier = tuple(
        item
        for item in inputs.rows("slb_sec", entity_id)
        if item.availability.data_as_of
        and row.availability.data_as_of
        and item.availability.data_as_of < row.availability.data_as_of
    )
    previous = latest(earlier)
    if previous is not None:
        previous_reasons = _row_reasons(inputs, previous, entity_id, retained_previous=True)
        if not previous_reasons:
            references += previous.evidence("trade_date", "lent_qnt", "end_bal")
        previous_balance, previous_quantity = (
            _value(previous, "end_bal"),
            _value(previous, "lent_qnt"),
        )
        if not previous_reasons and all(
            value is not None and value >= 0
            for value in (
                balance,
                quantity,
                previous_balance,
                previous_quantity,
            )
        ):
            details.update(
                {
                    "balance_change_cny": cast(float, balance) - cast(float, previous_balance),
                    "lent_change_shares": cast(float, quantity) - cast(float, previous_quantity),
                    "change_period": [
                        previous.availability.data_as_of,
                        row.availability.data_as_of,
                    ],
                    "change_semantics": "between_retained_observations_not_daily_trend",
                }
            )
    if balance is None or quantity is None:
        return details, Truth.UNKNOWN, references
    return details, Truth.TRUE if balance >= 0 and quantity >= 0 else Truth.FALSE, references


def _local_context(row: ResearchRow) -> tuple[dict[str, Any], Truth]:
    """前十和连板只解释当前局部风险；例：三连板得到脆弱性背景，不产生未来跌幅。"""
    if row.source_api == "limit_step":
        count = _value(row, "nums")
        valid = count is not None and count >= 1 and float(count).is_integer()
        return {
            "consecutive_limits": count,
            "fragility_background": "consecutive_limit_ladder",
            "crowding_signal": cast(float, count) >= 2 if valid else None,
            "return_forecast": "not_supported",
        }, Truth.UNKNOWN if count is None else Truth.TRUE if valid else Truth.FALSE
    amount, net = _value(row, "amount"), _value(row, "net_amount")
    details: dict[str, Any] = {
        "sample_scope": "northbound_top10_only",
        "market_type": row.text("market_type"),
        "net_to_sample_turnover": None,
        "full_market_net_flow": "not_supported",
        "holdings_inference": "not_supported",
    }
    if amount is None or net is None or row.text("market_type") not in {"1", "3"}:
        return details, Truth.UNKNOWN
    valid = amount >= 0 and abs(net) <= amount
    if valid and amount > 0:
        details["net_to_sample_turnover"] = net / amount
    return details, Truth.TRUE if valid else Truth.FALSE


def _dictionary(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
) -> tuple[dict[str, Any], Truth, tuple[EvidenceRef, ...]]:
    """全局字典与股票标签关联分别记录；例：无股票时仍观察名录，A只引用A的席位。"""
    linked = tuple(
        detail
        for detail in inputs.rows("hm_detail", entity_id)
        if detail.text("hm_name") == row.text("name")
        and row.text("name") is not None
        and not _row_reasons(inputs, detail, entity_id)
    )
    references = tuple(ref for item in linked for ref in item.evidence("hm_name", "hm_orgs"))
    global_scope = entity_id not in inputs.entity_ids
    state = (
        Truth.TRUE
        if row.text("name") and row.text("desc") and _aliases(row.raw.get("orgs"))
        else Truth.UNKNOWN
    )
    if (not global_scope and not linked) or sum(
        item.text("name") == row.text("name") for item in inputs.rows("hm_list")
    ) > 1:
        state = Truth.UNKNOWN
    return (
        {
            "metadata_scope": "global_dictionary" if global_scope else "linked_local_seat_labels",
            "equity_attribution": "unknown"
            if global_scope
            else "label_only"
            if linked
            else "unknown",
            "linked_seat_records": [item.availability.source_record_id for item in linked],
            "beneficial_owner": "unknown",
            "institutional_investor_evidence": False,
        },
        state,
        references,
    )


def consume_flow(
    inputs: ResearchInputs,
    product: str,
    api: str,
    rows: tuple[ResearchRow, ...],
    entity_id: str,
) -> dict[str, Any]:
    """执行十个新增资金消费者并返回字段质量、真实引用及软三值门禁。

    例：P07的无单位slb_sec记录输出unknown；已认证HK_SH名单输出listed，仍不改变
    股票的停牌和交易价格门禁。隔离行不产生数值或业务引用，未来行不参与当前观察。
    """
    if product != "P07" or api not in FLOW_CONSUMER_FIELDS:
        raise ValueError("unsupported_flow_consumer")
    observations: list[dict[str, Any]] = []
    states: list[Truth] = []
    reasons: list[str] = []
    references: list[EvidenceRef] = []
    for row in rows:
        row_reasons = _row_reasons(inputs, row, entity_id, metadata=api == "hm_list")
        if row.source_api != api:
            row_reasons += ("source_api_mismatch",)
        if row_reasons:
            reasons.extend(row_reasons)
            observations.append(
                {
                    "source_record_id": row.availability.source_record_id,
                    "status": "unknown",
                    "reason_codes": list(row_reasons),
                }
            )
            states.append(Truth.UNKNOWN)
            continue
        fields, quality = _fields(row, FLOW_CONSUMER_FIELDS[api])
        extra: tuple[EvidenceRef, ...] = ()
        if api == "cyq_perf":
            details, state, extra = _chip_check(inputs, row, entity_id)
        elif api == "hm_detail":
            details, state, extra = _seat_labels(inputs, row, entity_id)
        elif api == "hm_list":
            details, state, extra = _dictionary(inputs, row, entity_id)
        elif api == "top_list":
            details, state, extra = _stock_totals(inputs, row, entity_id)
        elif api in {"margin_secs", "stock_hsgt"}:
            details, state, extra = _eligibility(inputs, row, entity_id)
        elif api in {"slb_sec", "slb_sec_detail"}:
            details, state, extra = _lending(inputs, row, entity_id)
        else:
            details, state = _local_context(row)
        states.append(state)
        references.extend(row.evidence(*FLOW_CONSUMER_FIELDS[api]))
        references.extend(extra)
        observations.append(
            {
                "source_record_id": row.availability.source_record_id,
                "data_as_of": row.availability.data_as_of,
                "observed_at": row.availability.observed_at,
                "fields": fields,
                "field_quality": quality,
                "interpretation": details,
                "status": "observed"
                if state is Truth.TRUE
                else "conflict"
                if state is Truth.FALSE
                else "unknown",
            }
        )
    state = (
        Truth.FALSE
        if Truth.FALSE in states
        else (
            Truth.TRUE if states and all(value is Truth.TRUE for value in states) else Truth.UNKNOWN
        )
    )
    if state is not Truth.TRUE:
        reasons.append(api + ("_conflict" if state is Truth.FALSE else "_evidence_unknown"))
    if not rows:
        reasons.append("no_current_rows")
    result_references = unique_evidence(references)
    return {
        "status": "observed"
        if state is Truth.TRUE
        else "conflict"
        if state is Truth.FALSE
        else "unknown",
        "reason_codes": tuple(sorted(set(reasons))),
        "facts": {
            "observations": observations,
            "local_only": True,
            "independent_evidence": False,
            "calculation_scope": "current_observation",
            "historical_trend": "not_enabled_due_to_history",
            "affects_tradability": False,
            "affects_core_score": False,
        },
        "evidence_refs": result_references,
        "gates": (
            gate(
                "flow_local_" + api,
                state,
                api + "_unverified_or_conflicting",
                hard=False,
                references=result_references,
            ),
        ),
    }


def auxiliary_observation(
    inputs: ResearchInputs,
    row: ResearchRow,
    entity_id: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """给既有局部来源附字段质量和边界，保持每个席位单列；例：匿名top_inst只作席位线索。"""
    reasons = _row_reasons(inputs, row, entity_id, metadata=row.source_api == "hm_list")
    values, quality = _fields(row, fields)
    if reasons:
        values = {name: None for name in values}
    return {
        "source_api": row.source_api,
        "source_record_id": row.availability.source_record_id,
        "business_variant_key": row.raw.get("business_variant_key"),
        "source_fields": values,
        "field_quality": quality,
        "status": "unknown"
        if reasons or any(state not in {"verified", "observed_label"} for state in quality.values())
        else "observed",
        "reason_codes": list(reasons),
        "local_only": True,
        "independent_evidence": False,
        "institutional_investor_evidence": False,
        "beneficial_owner": "unknown",
        "full_market_flow_inference": "not_supported",
    }
