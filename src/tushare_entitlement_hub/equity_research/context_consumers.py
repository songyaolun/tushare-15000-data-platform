"""市场背景和技术旁证的领域消费，不将供应商衍生值计为独立经济证据。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from math import fsum, isclose, isfinite
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.numeric import price_gate
from ..product_foundation.temporal import availability_gate
from ..standard import normalize_date
from .common import gate, latest, unique_evidence
from .models import ResearchInputs, ResearchRow

CONTEXT_FIELDS = {
    "daily_info": ("trade_date", "ts_code", "com_count", "total_mv", "amount", "exchange"),
    "eco_cal": ("date", "time", "country", "event", "value", "pre_value", "fore_value"),
    "idx_factor_pro": ("ts_code", "trade_date", "close", "vol", "amount"),
    "index_basic": ("ts_code", "market", "publisher", "index_type", "weight_rule"),
    "index_daily": ("ts_code", "trade_date", "close", "pre_close", "pct_chg", "amount"),
    "index_dailybasic": ("ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate"),
    "index_monthly": ("ts_code", "trade_date", "close", "vol", "amount"),
    "index_weekly": ("ts_code", "trade_date", "close", "vol", "amount"),
    "moneyflow_mkt_dc": ("trade_date", "net_amount", "net_amount_rate"),
    "shibor_lpr": ("date", "1y", "5y"),
    "stk_account": ("date", "weekly_new", "total", "weekly_hold", "weekly_trade"),
    "sz_daily_info": ("trade_date", "ts_code", "count", "amount", "total_mv"),
    "trade_cal": ("exchange", "cal_date", "is_open", "pretrade_date"),
    "adj_factor": ("ts_code", "trade_date", "adj_factor"),
    "bak_daily": ("ts_code", "trade_date", "close", "vol", "amount"),
    "bse_mapping": ("o_code", "n_code", "list_date"),
    "factor_list": ("factor_name", "asset_type", "factor_type", "factor_desc"),
    "factor_value": ("factor_name", "ts_code", "trade_date", "factor_value"),
    "monthly": ("ts_code", "trade_date", "close", "vol", "amount"),
    "pro_bar": ("ts_code", "trade_date", "close", "vol", "amount"),
    "stk_factor": ("ts_code", "trade_date", "adj_factor", "close_qfq", "macd", "rsi_6"),
    "stk_factor_pro": ("ts_code", "trade_date", "adj_factor", "close_qfq", "pe_ttm", "circ_mv"),
    "stk_nineturn": ("ts_code", "trade_date", "freq", "nine_up_turn", "nine_down_turn"),
    "stk_week_month_adj": ("ts_code", "trade_date", "end_date", "freq", "close_qfq", "close_hfq"),
    "stk_weekly_monthly": ("ts_code", "trade_date", "end_date", "freq", "close", "vol"),
    "weekly": ("ts_code", "trade_date", "close", "vol", "amount"),
}
PERIOD_APIS = {
    "index_weekly",
    "index_monthly",
    "weekly",
    "monthly",
    "stk_weekly_monthly",
    "stk_week_month_adj",
}
INDEX_APIS = {
    "index_basic",
    "index_daily",
    "index_dailybasic",
    "index_monthly",
    "index_weekly",
    "idx_factor_pro",
}


def _visible(inputs: ResearchInputs, rows: tuple[ResearchRow, ...]) -> tuple[ResearchRow, ...]:
    """只保留可见的合格非变体记录；例：未来公开或隔离行均不能进入当前旁证。

    包装器已检查accepted来源，本层再次检查质量和时点，保证直接调用也不会使用隔离值。
    """
    return tuple(
        row
        for row in rows
        if row.raw.get("quality_tier") in {"verified", "usable_with_annotation"}
        and row.raw.get("is_quarantined") is not True
        and not row.raw.get("business_variant_key")
        and availability_gate(row.availability, inputs.analysis_as_of).passed
    )


def _day(row: ResearchRow, field: str = "trade_date", *, aligned: bool = True) -> str | None:
    """读取真实日期并按需核对适配日期；例：20260930输出2026-09-30，非法日期输出None。

    周月end_date是统计终点，trade_date是标签，所以调用者可关闭标签与data_as_of相等检查。
    """
    try:
        source = row.text(field)
        if (
            row.source_api == "stk_nineturn"
            and field == "trade_date"
            and source
            and len(source) > 10
        ):
            timestamp = datetime.fromisoformat(source)
            timezone = row.availability.market_timezone
            if timestamp.tzinfo is None or timezone is None:
                return None
            value = timestamp.astimezone(ZoneInfo(timezone)).date().isoformat()
        else:
            value = normalize_date(source)
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        return None
    return value if not aligned or row.availability.data_as_of == value else None


def _refs(row: ResearchRow, *fields: str) -> tuple[EvidenceRef, ...]:
    """合并实际字段和单位证书引用；例：amount核对保留金额行及元换算的认证依据。

    缺字段不创建血缘，证书引用去重后稳定排序，便于存储复验。
    """
    return unique_evidence(
        row.evidence(*fields)
        + tuple(
            reference
            for name in fields
            if name in row.numeric
            for certification in (row.numeric[name].certification,)
            if certification is not None
            for reference in certification.evidence_refs
        )
    )


def _number(row: ResearchRow, field: str, dimension: str | None = None) -> float | None:
    """仅取有限且量纲合格的认证数值；例：金额字段被误认证为ratio时返回None。

    未认证原始数字不能成为可计算值，CNY币种检查继续由ResearchRow.number执行。
    """
    value = row.number(field)
    fact = row.numeric.get(field)
    return (
        value
        if value is not None
        and isfinite(value)
        and (dimension is None or (fact is not None and fact.dimension == dimension))
        else None
    )


def _values(row: ResearchRow, fields: tuple[str, ...]) -> dict[str, Any]:
    """生成有单位状态的指定领域数值；例：未认证1y输出value=None，不用原始值兜底。

    输入明确的领域字段而非遍历raw，输出可序列化的数值、目标单位与缺口。
    """
    return {
        field: {
            "value": _number(row, field),
            "unit": row.numeric[field].unit if field in row.numeric else None,
            "status": "observed" if _number(row, field) is not None else "unknown",
        }
        for field in fields
    }


def _same_units(left: ResearchRow, left_field: str, right: ResearchRow, right_field: str) -> bool:
    """要求数值、目标单位、量纲和币种同时可比；例：元和万元未统一前不能比较。

    认证已经转换至同一单位的来源允许核对，任一侧缺证书或数值时返回False。
    """
    if _number(left, left_field) is None or _number(right, right_field) is None:
        return False
    a, b = left.numeric[left_field], right.numeric[right_field]
    return (a.unit, a.dimension, a.currency) == (b.unit, b.dimension, b.currency)


def _comparison(
    left: ResearchRow, left_field: str, right: ResearchRow | None, right_field: str
) -> dict[str, Any]:
    """核对已完成实体日期连接的两字段；例：同单位10和11输出conflict及有限差值。

    仅使用小量浮点误差容限，不允许未知单位或缺对照来源被解释为匹配。
    """
    if right is None or not _same_units(left, left_field, right, right_field):
        return {"status": "unknown", "reason": "comparison_unit_or_source_unavailable"}
    left_value, right_value = _number(left, left_field), _number(right, right_field)
    assert left_value is not None and right_value is not None
    delta = left_value - right_value
    if not isfinite(delta):
        return {"status": "unknown", "reason": "nonfinite_comparison"}
    return {
        "status": "matched"
        if isclose(left_value, right_value, rel_tol=1e-8, abs_tol=1e-8)
        else "conflict",
        "left": left_value,
        "right": right_value,
        "difference": delta,
        "unit": left.numeric[left_field].unit,
    }


def _kind(inputs: ResearchInputs, row: ResearchRow) -> str | None:
    """从显式实体登记读取类别；例：INDEX返回index，未绑定代码返回None。

    不靠证券代码后缀猜测指数、股票或市场身份，保证指数不会混入股票候选。
    """
    return next(
        (
            item.identity.kind.value
            for item in inputs.config.foundation.entities
            if item.identity.entity_id == row.entity_id
        ),
        None,
    )


def _peer(
    inputs: ResearchInputs, row: ResearchRow, api: str, day: str | None = None
) -> ResearchRow | None:
    """按认证同实体和真实交易日取唯一对照；例：股票A周线只能连接A统计终点的日线。

    日期未知、身份未绑定或同日多版本冲突返回None；相同业务哈希不构成连接依据。
    """
    target_day = day or _day(row)
    if row.entity_id is None or target_day is None:
        return None
    return latest(
        tuple(
            item
            for item in _visible(inputs, inputs.rows(api, row.entity_id))
            if _day(item) == target_day
        )
    )


def _result(
    status: str, details: dict[str, Any], references: tuple[EvidenceRef, ...], *reasons: str
) -> dict[str, Any]:
    """构造单个领域观察及软门禁；例：核对冲突保留FALSE软门禁，不改变候选核心分。

    所有技术与市场上下文都声明独立证据数为零，历史不足仍可携带真实当前观察。
    """
    return {
        "status": status,
        "reason_codes": tuple(sorted(set(reasons))),
        "facts": details,
        "evidence_refs": unique_evidence(references),
        "gates": (
            gate(
                "context_consistency",
                Truth.FALSE
                if status == "conflict"
                else Truth.TRUE
                if status in {"observed", "not_applicable"}
                else Truth.UNKNOWN,
                reasons[0] if reasons else "context_observation_unavailable",
                hard=False,
                references=references,
            ),
        ),
    }


def _index(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """解释指数身份、收益和估值状态；例：点位102对昨收100核得当前收益0.02。

    基准必须显式登记为index。元数据不发明上市状态，行情只用已认证点位及比例。
    """
    api = row.source_api
    refs = _refs(row, *CONTEXT_FIELDS[api])
    facts: dict[str, Any] = {"entity_id": row.entity_id, "index_code": row.text("ts_code")}
    if _kind(inputs, row) != "index":
        return _result(
            "unknown", facts | {"identity_status": "unverified"}, refs, "index_entity_unverified"
        )
    facts["identity_status"] = "certified_index"
    if api == "index_basic":
        fields: tuple[str, ...] = ("market", "publisher", "index_type", "weight_rule")
        facts.update({field: row.text(field) for field in fields})
        return _result(
            "observed" if all(facts[field] for field in fields) else "unknown",
            facts,
            refs,
            *(() if all(facts[field] for field in fields) else ("index_metadata_incomplete",)),
        )
    day = _day(row)
    facts["trade_date"] = day
    if day is None or day > inputs.analysis_as_of[:10]:
        return _result("unknown", facts, refs, "trade_date_unverified")
    if api == "index_dailybasic":
        fields = ("pe_ttm", "pb", "turnover_rate")
        facts["valuation_context"] = _values(row, fields)
        facts["valuation_state"] = (
            "nonpositive_multiple"
            if any(
                (value := _number(row, field)) is not None and value <= 0
                for field in ("pe_ttm", "pb")
            )
            else "current_observation"
        )
        return _result(
            "observed" if any(_number(row, field) is not None for field in fields) else "unknown",
            facts,
            refs,
        )
    close, previous = _number(row, "close", "index"), _number(row, "pre_close", "index")
    facts["current_index"] = _values(row, ("close", "pre_close", "pct_chg", "amount"))
    computed = (
        close / previous - 1
        if close is not None
        and close > 0
        and previous is not None
        and previous > 0
        and _same_units(row, "close", row, "pre_close")
        else None
    )
    computed = computed if computed is not None and isfinite(computed) else None
    facts["recomputed_return"] = computed
    reported = (
        _number(row, "pct_chg", "ratio")
        if row.numeric.get("pct_chg") and (row.numeric["pct_chg"].unit in {"ratio", "fraction"})
        else None
    )
    facts["return_check"] = (
        "unknown"
        if computed is None or reported is None
        else ("matched" if isclose(computed, reported, abs_tol=0.00005) else "conflict")
    )
    status = (
        "conflict"
        if facts["return_check"] == "conflict"
        else ("observed" if computed is not None else "unknown")
    )
    return _result(
        status,
        facts,
        refs,
        *(
            ("index_return_conflict",)
            if status == "conflict"
            else ("index_return_unverified",)
            if status == "unknown"
            else ()
        ),
    )


def _summary(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """按同日认证市场范围对账市场汇总；例：两来源同一深市范围金额不同输出conflict。

    已采股票成交合计只注明local_sample，不能冒充交易所全集；代码相同也不绕过市场绑定。
    """
    api = row.source_api
    count_field = "com_count" if api == "daily_info" else "count"
    day = _day(row)
    facts: dict[str, Any] = {
        "market_code": row.text("ts_code"),
        "trade_date": day,
        "market_entity_id": row.entity_id,
        "reported_summary": _values(row, (count_field, "amount", "total_mv")),
        "scope_basis": "certified_market_entity",
        "local_sample_is_exchange_total": False,
    }
    refs = _refs(row, *CONTEXT_FIELDS[api])
    sample_exchange = (
        "SZSE"
        if api == "sz_daily_info"
        else {"SZ": "SZSE", "SH": "SSE", "SSE": "SSE", "SZSE": "SZSE"}.get(
            row.text("exchange") or ""
        )
    )
    sample_amounts: list[float] = []
    sample_count = 0
    for identity in inputs.entity_ids:
        metadata = latest(_visible(inputs, inputs.rows("stock_basic", identity)))
        if (
            sample_exchange is None
            or metadata is None
            or metadata.text("exchange") != sample_exchange
        ):
            continue
        daily = latest(
            tuple(
                item
                for item in _visible(inputs, inputs.rows("daily", identity))
                if day is not None and _day(item) == day
            )
        )
        if daily is None:
            continue
        sample_count += 1
        refs += _refs(metadata, "ts_code", "exchange") + _refs(
            daily, "ts_code", "trade_date", "amount"
        )
        amount = _number(daily, "amount", "money")
        if amount is not None and amount >= 0 and daily.numeric["amount"].unit == "元":
            sample_amounts.append(amount)
    try:
        sample_total = (
            fsum(sample_amounts) if sample_count and len(sample_amounts) == sample_count else None
        )
    except OverflowError:
        sample_total = None
    facts["local_exchange_sample"] = {
        "exchange": sample_exchange,
        "observed_stock_count": sample_count,
        "amount_sum_cny": sample_total,
        "scope_basis": "explicit_stock_bindings",
        "comparable_to_reported_scope": False,
    }
    peer_api = "sz_daily_info" if api == "daily_info" else "daily_info"
    peer = _peer(inputs, row, peer_api)
    comparable = day is not None and _kind(inputs, row) == "market" and peer is not None
    if comparable and peer is not None:
        comparable = (row if api == "daily_info" else peer).text("exchange") in {"SZ", "SZSE"}
    facts["scope_comparable"] = comparable
    checks = {
        field: _comparison(
            row,
            field,
            peer if comparable else None,
            "count" if field == "com_count" else "com_count" if field == "count" else field,
        )
        for field in (count_field, "amount", "total_mv")
    }
    facts["summary_checks"] = checks
    if peer is not None:
        refs += _refs(peer, *CONTEXT_FIELDS[peer_api])
    state = (
        "conflict"
        if any(check["status"] == "conflict" for check in checks.values())
        else (
            "observed"
            if all(check["status"] == "matched" for check in checks.values())
            else "unknown"
        )
    )
    return _result(
        state,
        facts,
        refs,
        *(
            ()
            if state == "observed"
            else ("market_summary_conflict",)
            if state == "conflict"
            else ("market_scope_or_units_unverified",)
        ),
    )


def _macro_context(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """生成LPR、账户参与、市场资金和经济日程状态；例：LPR5y减1y给认证期限差。

    账户数不能解释成资金，市场净额仅代表单一供应商；日程预测版本未认证时禁止预期差。
    """
    api = row.source_api
    refs = _refs(row, *CONTEXT_FIELDS[api])
    day_field = "trade_date" if api == "moneyflow_mkt_dc" else "date"
    day = _day(row, day_field)
    facts: dict[str, Any] = {
        "observation_date": day,
        "available_at": row.availability.available_at,
        "public_available_at": row.availability.public_available_at,
    }
    if api == "eco_cal":
        facts.update({field: row.text(field) for field in ("date", "time", "country", "event")})
        facts["release_values"] = _values(row, ("value", "pre_value", "fore_value"))
        facts["forecast_surprise"] = None
        facts["forecast_version_status"] = "unverified"
        facts["schedule_state"] = (
            "scheduled"
            if day and day > inputs.analysis_as_of[:10]
            else (
                "release_observed"
                if day and _number(row, "value") is not None
                else "release_unconfirmed"
            )
        )
        current = _number(row, "value")
        prior = _number(row, "pre_value")
        change = (
            current - prior
            if current is not None
            and prior is not None
            and (_same_units(row, "value", row, "pre_value"))
            else None
        )
        facts["previous_value_change"] = change if change is not None and isfinite(change) else None
        valid = day is not None and all(row.text(field) for field in ("time", "country", "event"))
        return _result(
            "observed" if valid else "unknown", facts, refs, "forecast_time_version_unverified"
        )
    if day is None or day > inputs.analysis_as_of[:10]:
        return _result("unknown", facts, refs, "observation_date_unverified")
    if api == "shibor_lpr":
        facts["policy_rates"] = _values(row, ("1y", "5y"))
        one, five = _number(row, "1y", "ratio"), _number(row, "5y", "ratio")
        spread = (
            five - one
            if one is not None and five is not None and (_same_units(row, "1y", row, "5y"))
            else None
        )
        facts["term_spread"] = spread if spread is not None and isfinite(spread) else None
        facts["term_spread_unit"] = row.numeric["1y"].unit if spread is not None else None
        return _result(
            "observed" if facts["term_spread"] is not None else "unknown",
            facts,
            refs,
            *(() if facts["term_spread"] is not None else ("lpr_units_unverified",)),
        )
    if api == "moneyflow_mkt_dc":
        value = _number(row, "net_amount", "money")
        facts.update(
            supplier="eastmoney",
            scope="provider_market_aggregate",
            market_flow=_values(row, ("net_amount", "net_amount_rate")),
            direction="unknown"
            if value is None
            else "inflow"
            if value > 0
            else "outflow"
            if value < 0
            else "balanced",
        )
        return _result(
            "observed" if value is not None else "unknown",
            facts,
            refs,
            "single_supplier_definition",
        )
    fields = ("weekly_new", "total", "weekly_hold", "weekly_trade")
    facts.update(
        account_participation=_values(row, fields),
        frequency="weekly",
        monetary_flow_interpretation=False,
    )
    total = _number(row, "total", "quantity")
    inconsistent = False
    for field in ("weekly_hold", "weekly_trade"):
        value = _number(row, field, "quantity")
        ratio = None
        if (
            total is not None
            and total > 0
            and value is not None
            and (_same_units(row, field, row, "total"))
        ):
            ratio = value / total
            inconsistent |= value < 0 or value > total
        facts[field + "_share"] = ratio if ratio is not None and isfinite(ratio) else None
    observed = total is not None and total > 0
    return _result(
        "conflict" if inconsistent else "observed" if observed else "unknown",
        facts,
        refs,
        *(
            ("account_count_conflict",)
            if inconsistent
            else ()
            if observed
            else ("account_units_unverified",)
        ),
    )


def _calendar(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """解释原始交易日历，不推定北交所开市；例：SSE/is_open=0输出exchange_closed。

    前交易日必须早于日历日，非法代码或日期保留未知，交易所休市不是个股停牌。
    """
    day, previous = _day(row, "cal_date"), _day(row, "pretrade_date", aligned=False)
    exchange, opened = row.text("exchange"), row.text("is_open")
    valid = (
        day is not None
        and previous is not None
        and previous < day
        and (exchange in {"SSE", "SZSE"} and opened in {"0", "1"})
    )
    return _result(
        "observed" if valid else "unknown",
        {
            "exchange": exchange,
            "calendar_date": day,
            "previous_trade_date": previous,
            "exchange_state": "exchange_open"
            if valid and opened == "1"
            else "exchange_closed"
            if valid
            else "unknown",
            "individual_suspension": None,
            "bse_derived_calendar_status": "not_provided",
            "current_as_of": day == inputs.analysis_as_of[:10],
        },
        _refs(row, *CONTEXT_FIELDS["trade_cal"]),
        *(() if valid else ("calendar_state_unverified",)),
    )


def _mapping(inputs: ResearchInputs, row: ResearchRow, entity_id: str) -> dict[str, Any]:
    """用同一认证命名空间的历史别名验证北交所代码映射；例：旧新代码同指A才确认。

    旧别名可已过期但必须在分析时可见，两边分别唯一且属于同一股票生命周期。上市日期
    只作来源事实，不伪装成代码变更日期，也不新增证券或自动写入别名配置。
    """
    mapping = next(
        (
            item
            for item in inputs.config.foundation.entity_mappings
            if item.source_api == "bse_mapping"
        ),
        None,
    )
    aliases = tuple(
        item
        for item in inputs.config.foundation.aliases
        if mapping is not None
        and item.namespace == mapping.namespace
        and item.visible_at(inputs.analysis_as_of)
    )
    old, new = row.text("o_code"), row.text("n_code")
    old_aliases = tuple(item for item in aliases if item.alias == old)
    new_aliases = tuple(item for item in aliases if item.alias == new)
    old_ids, new_ids = (
        {item.entity_id for item in old_aliases},
        {item.entity_id for item in new_aliases},
    )
    same = len(old_ids) == len(new_ids) == 1 and old_ids == new_ids
    resolved = next(iter(old_ids)) if same else None
    same = same and resolved in inputs.entity_ids
    relevant = entity_id not in inputs.entity_ids or resolved == entity_id
    refs = _refs(row, *CONTEXT_FIELDS["bse_mapping"]) + tuple(
        ref for alias in old_aliases + new_aliases for ref in alias.evidence_refs
    )
    known_different = len(old_ids) == len(new_ids) == 1 and old_ids != new_ids
    status = "observed" if same and relevant else "conflict" if known_different else "unknown"
    return _result(
        status,
        {
            "old_code": old,
            "new_code": new,
            "listing_date": _day(row, "list_date", aligned=False),
            "same_entity_confirmed": same,
            "resolved_entity_id": resolved,
            "creates_additional_security": False,
        },
        refs,
        *(
            ()
            if status == "observed"
            else ("mapping_entity_conflict",)
            if known_different
            else ("mapping_aliases_unverified",)
        ),
    )


def _factor_dictionary(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """消费供应商因子字典的资产范围和定义；例：asset_type=IDX明确不适用于股票。

    描述字段只保留供应商定义，不能据此宣称公式已复算或回看窗口已通过历史认证。
    """
    fields = CONTEXT_FIELDS["factor_list"]
    facts: dict[str, Any] = {field: row.text(field) for field in fields}
    stock = facts["asset_type"] == "STK"
    facts.update(stock_applicable=stock, formula_validated=False, historical_window_validated=False)
    complete = all(facts[field] for field in fields)
    return _result(
        "not_applicable" if complete and not stock else "observed" if complete else "unknown",
        facts,
        _refs(row, *fields),
        *(
            ()
            if complete and stock
            else ("factor_asset_not_stock",)
            if complete
            else ("factor_definition_incomplete",)
        ),
    )


def _factor_value(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """按factor_name和STK资产类型连接因子定义；例：同名IDX字典不能认证股票因子。

    当前供应商值可展示，公式和历史窗口尚未验证时明确不启用历史特征或核心评分。
    """
    name = row.text("factor_name")
    dictionary = latest(
        tuple(
            item
            for item in _visible(inputs, inputs.rows("factor_list"))
            if name is not None
            and item.text("factor_name") == name
            and item.text("asset_type") == "STK"
        )
    )
    refs = _refs(row, *CONTEXT_FIELDS["factor_value"])
    facts: dict[str, Any] = {
        "factor_name": name,
        "trade_date": _day(row),
        "supplier_value": _values(row, ("factor_value",)),
        "formula_validated": False,
        "historical_window_validated": False,
    }
    if dictionary is not None:
        refs += _refs(dictionary, *CONTEXT_FIELDS["factor_list"])
        facts["definition"] = {
            field: dictionary.text(field) for field in CONTEXT_FIELDS["factor_list"]
        }
    valid = (
        dictionary is not None
        and all(dictionary.text(field) for field in ("factor_type", "factor_desc"))
        and _kind(inputs, row) == "security"
        and (_day(row) is not None and _number(row, "factor_value") is not None)
    )
    return _result(
        "not_enabled_due_to_history" if valid else "unknown",
        facts,
        refs,
        "factor_formula_and_history_unverified"
        if valid
        else "factor_dictionary_or_value_unverified",
    )


def _adjustment(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """给出带真实来源记录的当前复权锚点和相邻变化；例：2到2.2显示因子变化1.1。

    当前锚点仅说明本地观察，不能推定供应商前复权归一基准，也不自动认证历史企业行动。
    """
    day, value = _day(row), _number(row, "adj_factor")
    refs = _refs(row, *CONTEXT_FIELDS["adj_factor"])
    earlier = tuple(
        item
        for item in _visible(inputs, inputs.rows("adj_factor", row.entity_id))
        if day is not None and (previous_day := _day(item)) is not None and previous_day < day
    )
    previous = latest(earlier) if row.entity_id is not None else None
    change = None
    if previous is not None:
        refs += _refs(previous, *CONTEXT_FIELDS["adj_factor"])
        old = _number(previous, "adj_factor")
        if (
            old is not None
            and old > 0
            and value is not None
            and value > 0
            and (_same_units(row, "adj_factor", previous, "adj_factor"))
        ):
            candidate = value / old
            change = candidate if isfinite(candidate) else None
    valid = day is not None and value is not None and value > 0 and _kind(inputs, row) == "security"
    return _result(
        "not_enabled_due_to_history" if valid else "unknown",
        {
            "anchor_date": day,
            "anchor_factor": value if valid else None,
            "anchor_record_id": row.availability.source_record_id if valid else None,
            "anchor_basis": "current_accepted_observation" if valid else "unverified",
            "previous_observation_date": _day(previous) if previous is not None else None,
            "factor_change_ratio": change,
            "corporate_action_confirmed": False,
            "supplier_normalization_anchor": None,
            "historical_adjustment_enabled": False,
        },
        refs,
        "historical_adjustment_anchor_unverified" if valid else "adjustment_factor_unverified",
    )


def _traded(row: ResearchRow) -> bool:
    """要求明确成交的认证价格与量额；例：close=10而vol=0不能成为日行情核对依据。

    参考价、停牌标记及无成交状态都拒绝，但本函数失败不等于认定个股停牌。
    """
    return (
        price_gate(row.raw, "return").passed
        and row.raw.get("market_activity_status") != "suspended"
        and all(
            (value := _number(row, field)) is not None and value > 0
            for field in ("close", "vol", "amount")
        )
    )


def _quote_check(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """比较同证券同日备用或组合日线；例：pro_bar与daily相同只产生一次旁证核对。

    无成交0价保留未知且不补入主行情，指数技术来源只能与认证指数日线比较。
    """
    index = row.source_api == "idx_factor_pro"
    peer_api = "index_daily" if index else "daily"
    peer = _peer(inputs, row, peer_api)
    valid = _kind(inputs, row) == ("index" if index else "security") and _traded(row)
    if peer is not None and not index:
        valid = valid and _traded(peer)
    checks = {
        field: _comparison(row, field, peer if valid else None, field)
        for field in ("close", "vol", "amount")
    }
    refs = _refs(row, *CONTEXT_FIELDS[row.source_api])
    if peer is not None:
        refs += _refs(peer, "ts_code", "trade_date", "close", "vol", "amount")
    status = (
        "conflict"
        if any(check["status"] == "conflict" for check in checks.values())
        else (
            "observed"
            if all(check["status"] == "matched" for check in checks.values())
            else "unknown"
        )
    )
    return _result(
        status,
        {
            "trade_date": _day(row),
            "quote_checks": checks,
            "current_quote": _values(row, ("close", "vol", "amount")),
            "composite_daily_source": row.source_api == "pro_bar",
            "score_eligible": False,
        },
        refs,
        *(
            ()
            if status == "observed"
            else ("quote_conflict",)
            if status == "conflict"
            else ("quote_date_activity_or_units_unverified",)
        ),
    )


def _exchange(inputs: ResearchInputs, row: ResearchRow) -> str | None:
    """只从同实体认证元数据读取交易所；例：stock_basic.exchange=SZSE得到SZSE。

    不从代码后缀猜日历；指数发布市场只有明确SSE/SZSE映射时可复用对应日历。
    """
    if row.entity_id is None:
        return None
    api = "index_basic" if _kind(inputs, row) == "index" else "stock_basic"
    metadata = latest(_visible(inputs, inputs.rows(api, row.entity_id)))
    value = metadata.text("market" if api == "index_basic" else "exchange") if metadata else None
    return {"SH": "SSE", "SSE": "SSE", "SZ": "SZSE", "SZSE": "SZSE"}.get(value or "")


def _period(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """核对周月统计终点、当前收盘及完整期间量额；例：标签周五、end_date周三只比周三。

    完整聚合要求同交易所逐日可见日历和全部开市日日线；缺少历史时仍保留当前收盘核对。
    复权周月没有供应商固定锚点证据时禁止将close_qfq与未复权close直接比较。
    """
    api = row.source_api
    combined = api in {"stk_weekly_monthly", "stk_week_month_adj"}
    frequency = row.text("freq") if combined else "week" if "weekly" in api else "month"
    label = _day(row, aligned=False)
    end = _day(row, "end_date", aligned=False) if combined else _day(row)
    facts: dict[str, Any] = {
        "label_date": label,
        "statistical_end_date": end,
        "frequency": frequency,
        "full_period_check": "unknown",
    }
    refs = _refs(row, *CONTEXT_FIELDS[api])
    required_kind = "index" if api.startswith("index_") else "security"
    if _kind(inputs, row) != required_kind:
        return _result("unknown", facts, refs, "period_entity_unverified")
    if (
        label is None
        or end is None
        or frequency not in {"week", "month"}
        or end > label
        or (end > inputs.analysis_as_of[:10])
        or row.availability.data_as_of not in {label, end}
    ):
        return _result("unknown", facts, refs, "period_date_or_frequency_unverified")
    end_date, label_date = date.fromisoformat(end), date.fromisoformat(label)
    start = (
        end_date - timedelta(days=end_date.weekday())
        if frequency == "week"
        else end_date.replace(day=1)
    )
    natural_end = (
        start + timedelta(days=6)
        if frequency == "week"
        else (end_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    )
    valid_label = (
        label_date.isocalendar()[:2] == end_date.isocalendar()[:2]
        if frequency == "week"
        else (label[:7] == end[:7])
    )
    if not valid_label:
        return _result("unknown", facts, refs, "period_label_mismatch")
    facts["period_start"] = start.isoformat()
    facts["period_phase"] = "partial_or_unverified"
    peer_api = "index_daily" if api.startswith("index_") else "daily"
    peer = _peer(inputs, row, peer_api, end)
    if peer is not None:
        refs += _refs(peer, "ts_code", "trade_date", "close", "vol", "amount")
    if api == "stk_week_month_adj":
        facts["current_adjusted_quotes"] = _values(row, ("close_qfq", "close_hfq"))
        factor = _peer(inputs, row, "adj_factor", end)
        if factor is not None:
            refs += _refs(factor, "trade_date", "adj_factor")
            facts["observed_factor"] = _values(factor, ("adj_factor",))
        facts["supplier_normalization_anchor"] = None
        adjusted_observed = any(
            _number(row, field, "price") is not None for field in ("close_qfq", "close_hfq")
        )
        return _result(
            "not_enabled_due_to_history" if peer is not None and adjusted_observed else "unknown",
            facts,
            refs,
            "fixed_adjustment_anchor_unverified",
        )
    close = _number(row, "close")
    peer_close = _number(peer, "close") if peer is not None else None
    positive = close is not None and close > 0
    positive = positive and peer is not None and peer_close is not None and peer_close > 0
    if required_kind == "security":
        positive = positive and peer is not None and _traded(peer)
    current = _comparison(row, "close", peer if positive else None, "close")
    facts["current_close_check"] = current
    facts["current_period_quote"] = _values(
        row, ("close", "vol", "amount") if api != "stk_weekly_monthly" else ("close", "vol")
    )
    if current["status"] == "conflict":
        return _result("conflict", facts, refs, "period_close_conflict")
    exchange = _exchange(inputs, row)
    metadata_api = "index_basic" if api.startswith("index_") else "stock_basic"
    metadata = (
        latest(_visible(inputs, inputs.rows(metadata_api, row.entity_id)))
        if row.entity_id
        else None
    )
    if metadata is not None:
        refs += _refs(metadata, "ts_code", "market", "exchange")
    calendar = tuple(
        item
        for item in _visible(inputs, inputs.rows("trade_cal"))
        if exchange is not None
        and item.text("exchange") == exchange
        and (calendar_day := _day(item, "cal_date")) is not None
        and start.isoformat() <= calendar_day <= natural_end.isoformat()
    )
    refs += tuple(ref for item in calendar for ref in _refs(item, *CONTEXT_FIELDS["trade_cal"]))
    expected_days = {
        (start + timedelta(days=offset)).isoformat()
        for offset in range((natural_end - start).days + 1)
    }
    calendar_days = {_day(item, "cal_date") for item in calendar}
    complete = (
        "trade_cal" in inputs.config.complete_scopes
        and calendar_days == expected_days
        and natural_end.isoformat() <= inputs.analysis_as_of[:10]
    )
    complete = (
        complete
        and len(calendar_days) == len(calendar)
        and all(item.text("is_open") in {"0", "1"} for item in calendar)
    )
    open_days = {
        calendar_day
        for item in calendar
        if item.text("is_open") == "1" and (calendar_day := _day(item, "cal_date")) is not None
    }
    complete = complete and bool(open_days) and max(open_days) == end
    history = tuple(
        item
        for item in _visible(inputs, inputs.rows(peer_api, row.entity_id))
        if row.entity_id is not None and _day(item) in open_days
    )
    history_by_day: dict[str | None, list[ResearchRow]] = defaultdict(list)
    for item in history:
        history_by_day[_day(item)].append(item)
    daily = tuple(latest(tuple(items)) for items in history_by_day.values())
    complete = (
        complete
        and bool(open_days)
        and set(history_by_day) == open_days
        and all(item is not None for item in daily)
    )
    facts["observed_daily_count"] = len(daily)
    facts["expected_trading_days"] = len(open_days) if complete else None
    if not complete or current["status"] != "matched":
        return _result(
            "not_enabled_due_to_history" if current["status"] == "matched" else "unknown",
            facts,
            refs,
            "complete_period_history_unavailable",
        )
    facts["period_phase"] = "complete"
    checks: dict[str, Any] = {}
    for field in ("vol",) if api == "stk_weekly_monthly" else ("vol", "amount"):
        comparable = all(
            item is not None and _same_units(row, field, item, field) for item in daily
        )
        if not comparable:
            checks[field] = {"status": "unknown", "reason": "period_units_unverified"}
            continue
        daily_values = [
            number
            for item in daily
            if item is not None and (number := _number(item, field)) is not None
        ]
        # 非空daily已逐行通过_same_units，周期值与各日值均已认证且非空。
        value = cast(float, _number(row, field))
        try:
            total = fsum(daily_values)
        except OverflowError:
            total = float("inf")
        checks[field] = {
            "status": "unknown"
            if not isfinite(total)
            else "matched"
            if isclose(value, total, rel_tol=1e-8, abs_tol=1e-8)
            else "conflict",
            "daily_sum": total if isfinite(total) else None,
            "reported": value,
        }
    refs += tuple(
        ref
        for item in daily
        if item is not None
        for ref in _refs(item, "trade_date", "close", "vol", "amount")
    )
    status = (
        "conflict"
        if any(item["status"] == "conflict" for item in checks.values())
        else (
            "observed"
            if all(item["status"] == "matched" for item in checks.values())
            else "unknown"
        )
    )
    facts.update(full_period_check=status, aggregation_checks=checks)
    return _result(
        status, facts, refs, *(() if status == "observed" else ("period_aggregation_unverified",))
    )


def _technical(inputs: ResearchInputs, row: ResearchRow) -> dict[str, Any]:
    """消费供应商技术值、复权因子和估值旁证；例：stk_factor_pro的PE仅对同日日指标。

    MACD/RSI公式及前复权固定锚点没有独立认证，保留当前字段而不将它们加入技术分。
    九转正负9是供应商事件标签，只确认已声明daily频率，不能解释为买卖指令。
    """
    api, day = row.source_api, _day(row)
    refs = _refs(row, *CONTEXT_FIELDS[api])
    facts: dict[str, Any] = {
        "trade_date": day,
        "supplier_derived": True,
        "formula_validated": False,
        "normalization_anchor_validated": False,
    }
    if day is None or day > inputs.analysis_as_of[:10] or _kind(inputs, row) != "security":
        return _result("unknown", facts, refs, "technical_identity_or_date_unverified")
    if api == "stk_nineturn":
        frequency = row.text("freq")
        up, down = row.text("nine_up_turn"), row.text("nine_down_turn")
        facts.update(
            frequency=frequency,
            up_event=up in {"9", "9.0", "+9"},
            down_event=down in {"-9", "-9.0"},
            provider_labels={"nine_up_turn": up, "nine_down_turn": down},
        )
        valid = (
            frequency == "daily"
            and up in {"9", "9.0", "+9", "0", "0.0"}
            and (down in {"-9", "-9.0", "0", "0.0"})
        )
        return _result(
            "not_enabled_due_to_history" if valid else "unknown",
            facts,
            refs,
            "technical_event_formula_unverified" if valid else "technical_event_label_unverified",
        )
    numeric_fields = (
        ("adj_factor", "close_qfq", "macd", "rsi_6")
        if api == "stk_factor"
        else ("adj_factor", "close_qfq", "pe_ttm", "circ_mv")
    )
    facts["supplier_observation"] = _values(row, numeric_fields)
    factor = _peer(inputs, row, "adj_factor")
    checks = {"adj_factor": _comparison(row, "adj_factor", factor, "adj_factor")}
    if factor is not None:
        refs += _refs(factor, "ts_code", "trade_date", "adj_factor")
    if api == "stk_factor_pro":
        valuation = _peer(inputs, row, "daily_basic")
        for field in ("pe_ttm", "circ_mv"):
            checks[field] = _comparison(row, field, valuation, field)
        if valuation is not None:
            refs += _refs(valuation, "ts_code", "trade_date", "pe_ttm", "circ_mv")
    facts["same_day_checks"] = checks
    conflict = any(item["status"] == "conflict" for item in checks.values())
    observed = any(_number(row, field) is not None for field in numeric_fields)
    return _result(
        "conflict" if conflict else "not_enabled_due_to_history" if observed else "unknown",
        facts,
        refs,
        "technical_source_conflict" if conflict else "technical_formula_and_anchor_unverified",
    )


def consume_context(
    inputs: ResearchInputs, product: str, api: str, rows: tuple[ResearchRow, ...], entity_id: str
) -> dict[str, Any]:
    """按P01/P03分配消费有限合格行并合并领域观察；例：空index_daily输出unknown及缺口。

    每组采用唯一最新版本，跨因子、频率、市场、事件分别保留。返回软门禁、真实血缘和
    当前状态，技术/组合来源的independent_evidence始终False，core_score_contribution恒为0。
    """
    if api not in CONTEXT_FIELDS or product not in {"P01", "P03"}:
        raise ValueError("unsupported_context_consumer")
    usable = _visible(inputs, tuple(row for row in rows if row.source_api == api))
    groups: dict[tuple[Any, ...], list[ResearchRow]] = defaultdict(list)
    keys = {
        "factor_list": ("factor_name", "asset_type"),
        "factor_value": ("factor_name",),
        "trade_cal": ("exchange",),
        "bse_mapping": ("o_code", "n_code"),
        "eco_cal": ("date", "time", "country", "event"),
        "stk_nineturn": ("freq",),
        "stk_weekly_monthly": ("freq",),
        "stk_week_month_adj": ("freq",),
    }.get(api, ())
    for row in usable:
        key = (row.entity_id, row.text("ts_code")) + tuple(row.text(field) for field in keys)
        groups[key].append(row)
    observations = []
    for group in groups.values():
        candidates = tuple(group)
        if api in {"stk_weekly_monthly", "stk_week_month_adj"}:
            end_dates = [_day(row, "end_date", aligned=False) for row in group]
            newest_end = max((day for day in end_dates if day is not None), default=None)
            candidates = tuple(
                row for row in group if _day(row, "end_date", aligned=False) == newest_end
            )
        selected = latest(candidates)
        if selected is None:
            observations.append(
                _result(
                    "unknown",
                    {"version_status": "ambiguous"},
                    tuple(ref for row in group for ref in _refs(row, *CONTEXT_FIELDS[api])),
                    "context_current_version_ambiguous",
                )
            )
            continue
        if api in PERIOD_APIS:
            observation = _period(inputs, selected)
        elif api in {"index_basic", "index_daily", "index_dailybasic"}:
            observation = _index(inputs, selected)
        elif api in {"daily_info", "sz_daily_info"}:
            observation = _summary(inputs, selected)
        elif api in {"shibor_lpr", "eco_cal", "stk_account", "moneyflow_mkt_dc"}:
            observation = _macro_context(inputs, selected)
        elif api == "trade_cal":
            observation = _calendar(inputs, selected)
        elif api == "bse_mapping":
            observation = _mapping(inputs, selected, entity_id)
        elif api == "factor_list":
            observation = _factor_dictionary(inputs, selected)
        elif api == "factor_value":
            observation = _factor_value(inputs, selected)
        elif api == "adj_factor":
            observation = _adjustment(inputs, selected)
        elif api in {"bak_daily", "pro_bar", "idx_factor_pro"}:
            observation = _quote_check(inputs, selected)
        else:
            observation = _technical(inputs, selected)
        observations.append(observation)
    states = {item["status"] for item in observations}
    status = (
        "conflict"
        if "conflict" in states
        else "observed"
        if "observed" in states
        else (
            "not_enabled_due_to_history"
            if "not_enabled_due_to_history" in states
            else "not_applicable"
            if states == {"not_applicable"}
            else "unknown"
        )
    )
    reasons = {reason for item in observations for reason in item["reason_codes"]}
    if not observations:
        reasons.add("context_accepted_rows_unavailable")
    if len(usable) != len(rows):
        reasons.add("context_rows_rejected_by_quality_or_time")
    return {
        "status": status,
        "reason_codes": tuple(sorted(reasons)),
        "facts": {
            "independent_evidence": False,
            "independent_evidence_count": 0,
            "core_score_contribution": 0,
            "calculation_scope": "current_observation",
            "historical_pit_certified": False,
            "observations": [
                item["facts"]
                | {"status": item["status"], "reason_codes": list(item["reason_codes"])}
                for item in observations
            ],
        },
        "evidence_refs": unique_evidence(
            ref for item in observations for ref in item["evidence_refs"]
        ),
        "gates": tuple(check for item in observations for check in item["gates"]),
    }
