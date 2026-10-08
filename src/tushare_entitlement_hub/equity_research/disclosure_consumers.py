"""财务与治理的当前披露旁证；口径、来源版本和认证单位不满足时明确保留未知。"""

from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite
from typing import Any, cast

from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.entities import EntityRelation
from ..product_foundation.numeric import price_gate
from ..product_foundation.temporal import availability_gate
from ..standard import normalize_date
from .common import gate, unique_evidence
from .fundamental import _context, _money, _version
from .models import MARKET_ENTITY, ResearchInputs, ResearchRow

DISCLOSURE_FIELDS = {
    "bak_basic": ("pe", "pb", "float_share", "total_assets"),
    "fina_indicator": ("ann_date", "end_date", "update_flag", "roe", "debt_to_assets"),
    "fina_indicator_vip": (
        "ann_date",
        "end_date",
        "update_flag",
        "roe",
        "ocf_to_profit",
        "debt_to_assets",
    ),
    "cb_daily": ("trade_date", "close", "cb_value", "cb_over_rate", "vol"),
    "cb_factor_pro": ("trade_date", "close", "vol", "amount"),
    "cb_issue": ("ann_date", "plan_issue_size", "issue_size", "res_ann_date"),
    "cb_rate": ("rate_start_date", "rate_end_date", "coupon_rate", "rate_freq"),
    "cb_rating": ("ann_date", "rating_date", "rating_com_name", "rating", "rating_outlook"),
    "new_share": ("ipo_date", "issue_date", "amount", "price"),
    "stk_holdernumber": ("ann_date", "end_date", "holder_num"),
    "stk_managers": ("ann_date", "name", "title", "begin_date", "end_date"),
    "stk_rewards": ("end_date", "name", "title", "reward", "hold_vol"),
    "top10_cb_holders": ("end_date", "holder_rank", "holder_name", "hold_ratio"),
    "top10_floatholders": (
        "ann_date",
        "end_date",
        "holder_name",
        "hold_amount",
        "hold_float_ratio",
    ),
    "top10_holders": ("ann_date", "end_date", "holder_name", "hold_ratio", "hold_change"),
}
FINANCIAL_APIS = frozenset({"bak_basic", "fina_indicator", "fina_indicator_vip"})
BOND_APIS = frozenset(api for api in DISCLOSURE_FIELDS if api.startswith("cb_")) | {
    "top10_cb_holders"
}


def _date(row: ResearchRow, field: str) -> str | None:
    """解析来源实际日期，不以观察时间填补业务日期；例：20260630得到2026-06-30。"""
    try:
        return normalize_date(row.text(field))
    except (ValueError, TypeError):
        return None


def _visible(inputs: ResearchInputs, row: ResearchRow) -> bool:
    """重验可见性与隔离标志，防止直接调用绕过适配层；例：未来观察或quarantine返回False。"""
    business_field = (
        "trade_date"
        if row.source_api in {"bak_basic", "cb_daily", "cb_factor_pro", "daily_basic", "daily"}
        else "end_date"
        if row.source_api
        in FINANCIAL_APIS
        | {
            "stk_rewards",
            "stk_holdernumber",
            "top10_holders",
            "top10_floatholders",
            "top10_cb_holders",
        }
        else "rating_date"
        if row.source_api == "cb_rating"
        else None
    )
    business_date = _date(row, business_field) if business_field is not None else None
    return (
        row.raw.get("is_quarantined") is not True
        and row.raw.get("quality_tier") not in {"quarantine", "unknown"}
        and availability_gate(row.availability, inputs.analysis_as_of).passed
        and (business_date is None or business_date <= inputs.analysis_as_of[:10])
        and all(
            row.text(field) is None
            or ((day := _date(row, field)) is not None and day <= inputs.analysis_as_of[:10])
            for field in ("ann_date", "f_ann_date", "pub_date")
        )
    )


def _value(row: ResearchRow | None, field: str, unit: str) -> float | None:
    """只读认证后单位，拒绝原始数值和伪人民币量纲；例：亿股未认证为股时返回None。"""
    if row is None:
        return None
    value = row.number(field, unit)
    fact = row.numeric.get(field)
    if unit == "元" and (
        fact is None or fact.currency != "CNY" or fact.dimension not in {"money", "price"}
    ):
        return None
    return value


def _ratio_value(row: ResearchRow | None, field: str) -> float | None:
    """把已认证比例统一为小数，保留负比率；例：40%的明确证书得到0.4，原始40不换算。"""
    if row is None or field not in row.numeric:
        return None
    fact = row.numeric[field]
    if fact.dimension != "ratio":
        return None
    value = row.number(field)
    if value is None or fact.unit not in {"fraction", "ratio", "%"}:
        return None
    return value / 100 if fact.unit == "%" else value


def _current(rows: tuple[ResearchRow, ...], field: str) -> ResearchRow | None:
    """选择最新业务期内唯一可见版本，缺日期或同刻冲突不猜选；例：旧期与新期只取新期。"""
    periods = [_date(row, field) for row in rows]
    if not rows or None in periods:
        return None
    period = max(value for value in periods if value is not None)
    return _version(tuple(row for row in rows if _date(row, field) == period))[0]


def _other_rows(inputs: ResearchInputs, api: str, source: ResearchRow) -> tuple[ResearchRow, ...]:
    """按实体索引或无实体时的代码索引取旁证，再逐候选重验原可见性门禁。

    例：5000条日线中只有2条属于证券A，仅检查这2条；未映射债券仍按ts_code分开核价。
    有实体时不按同代码回退，缺实体且缺代码时返回空；候选顺序及版本歧义原样保留。
    """
    if source.entity_id is not None:
        candidates = inputs.rows(api, source.entity_id)
    else:
        code = source.text("ts_code")
        candidates = inputs.code_rows.get((api, code), ()) if code is not None else ()
    return tuple(row for row in candidates if _visible(inputs, row))


def _underlying_relations(
    inputs: ResearchInputs, source: ResearchRow
) -> tuple[EntityRelation, ...]:
    """按债券实体索引转股关系，再保留已确认且在当前截止时点可用的全部候选。

    例：千条关系中债券A有两条冲突关系，仍返回这两条供下游判歧义；无实体返回空。
    不缓存时点判定、不去重或任选目标，保持关系证据的原始顺序。
    """
    candidates = (
        inputs.config.relations_by_source.get((source.entity_id, "cb_underlying"), ())
        if source.entity_id is not None
        else ()
    )
    return tuple(
        relation
        for relation in candidates
        if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
    )


def _result(
    api: str,
    facts: dict[str, Any],
    references: tuple[EvidenceRef, ...],
    reasons: tuple[str, ...] = (),
    *,
    conflict: bool = False,
) -> dict[str, Any]:
    """输出旁证软门禁和有限JSON事实，任何缺口不转成零；例：单位缺失生成unknown软门禁。"""
    status = "conflict" if conflict else "unknown" if reasons else "observed"
    evidence = unique_evidence(references)
    codes = tuple(sorted(set(reasons)))
    return {
        "status": status,
        "reason_codes": codes,
        "facts": {
            "independent_evidence": False,
            "calculation_scope": "current_observation",
            "historical_status": "not_enabled_due_to_history",
            **facts,
        },
        "evidence_refs": evidence,
        "gates": (
            gate(
                "disclosure_" + api,
                Truth.FALSE if conflict else Truth.UNKNOWN if reasons else Truth.TRUE,
                codes[0] if codes else "disclosure_check_unavailable",
                hard=False,
                references=evidence,
            ),
        ),
    }


def _comparison(left: float | None, right: float | None) -> dict[str, Any]:
    """比较同口径有限数值，不对缺值判一致；例：0.4与0.40000000001在数值误差内一致。"""
    return {
        "left": left,
        "right": right,
        "agreement": None
        if left is None or right is None
        else isclose(left, right, rel_tol=1e-8, abs_tol=1e-10),
    }


def _backup(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """核对同日股本与市净率，动态PE及无报告期资产保持不同口径；例：亿股认证后才与股比较。"""
    source = _current(rows, "trade_date")
    refs = tuple(ref for row in rows for ref in row.evidence("trade_date", *DISCLOSURE_FIELDS[api]))
    if source is None:
        return _result(api, {}, refs, ("snapshot_date_or_version_unverified",))
    other = _current(_other_rows(inputs, "daily_basic", source), "trade_date")
    same_day = other is not None and _date(other, "trade_date") == _date(source, "trade_date")
    comparisons = {
        name: _comparison(
            _value(source, name, unit), _value(other, name, unit) if same_day else None
        )
        for name, unit in (("pb", "ratio"), ("float_share", "股"))
    }
    if other is not None:
        refs += other.evidence("pb", "float_share", "trade_date")
    reasons = [] if same_day else ["valuation_snapshot_date_mismatch"]
    if any(item["agreement"] is None for item in comparisons.values()):
        reasons.append("valuation_unit_or_value_unverified")
    conflict = any(item["agreement"] is False for item in comparisons.values())
    if conflict:
        reasons.append("valuation_snapshot_conflict")
    return _result(
        api,
        {
            "trade_date": _date(source, "trade_date"),
            "comparisons": comparisons,
            "provider_pe": _value(source, "pe", "ratio"),
            "pe_comparison_status": "unknown",
            "pe_comparison_reason": "dynamic_pe_earnings_basis_not_certified",
            "reported_total_assets": _value(source, "total_assets", "元"),
            "assets_comparison_status": "unknown",
            "assets_comparison_reason": "backup_report_period_and_scope_missing",
            "share_conversion_policy": "certificate_only_no_implicit_billion_multiplier",
        },
        refs,
        tuple(reasons),
        conflict=conflict,
    )


def _statement_for(
    inputs: ResearchInputs, source: ResearchRow, family: str, fields: tuple[str, ...]
) -> tuple[ResearchRow | None, tuple[EvidenceRef, ...]]:
    """选择与指标完全同口径的财务表，普通/VIP冲突不任选；例：不同公司类型不连接。"""
    context = _context(source)
    candidates = tuple(
        row
        for api in (family + "_vip", family)
        for row in _other_rows(inputs, api, source)
        if context is not None and _context(row) == context
    )
    refs = tuple(
        ref
        for row in candidates
        for ref in row.evidence("end_date", "report_type", "comp_type", *fields)
    )
    chosen = {
        api: _version(tuple(row for row in candidates if row.source_api == api))
        for api in (family + "_vip", family)
    }
    if any(reason not in {None, "statement_missing"} for _, reason in chosen.values()):
        return None, refs
    vip, ordinary = (chosen[api][0] for api in (family + "_vip", family))
    if (
        vip is not None
        and ordinary is not None
        and any(
            _money(vip, field) is None or _money(vip, field) != _money(ordinary, field)
            for field in fields
        )
    ):
        return None, refs
    return vip or ordinary, refs


def _indicators(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """保留供应商指标并核对同报告期、范围、行业及版本，普通/VIP不加分。

    例：已认证资产负债率40%和同口径负债40/资产100一致；官方指标缺report_type或comp_type
    时只展示当前指标并说明schema gap，ROE不以期末净资产替代平均净资产。
    """
    refs = tuple(
        ref
        for row in rows
        for ref in row.evidence(*DISCLOSURE_FIELDS[api], "report_type", "comp_type")
    )
    source = _current(rows, "end_date")
    if source is None:
        return _result(api, {}, refs, ("indicator_period_or_revision_unverified",))
    other_api = "fina_indicator" if api.endswith("_vip") else "fina_indicator_vip"
    other = _current(_other_rows(inputs, other_api, source), "end_date")
    context = _context(source)
    same_scope = context is not None and other is not None and _context(other) == context
    same_version = (
        other is not None
        and _date(source, "ann_date") is not None
        and _date(source, "ann_date") == _date(other, "ann_date")
        and source.text("update_flag") is not None
        and source.text("update_flag") == other.text("update_flag")
    )
    ordinary_checks = {
        field: _comparison(
            _ratio_value(source, field),
            _ratio_value(other, field) if same_scope and same_version else None,
        )
        for field in ("roe", "debt_to_assets")
    }
    if other is not None:
        refs += other.evidence(
            "ann_date",
            "end_date",
            "update_flag",
            "report_type",
            "comp_type",
            "roe",
            "debt_to_assets",
        )
    reasons = []
    if context is None:
        reasons.append("indicator_report_scope_schema_gap")
    if not same_scope or not same_version:
        reasons.append("indicator_duplicate_scope_or_version_unverified")
    nonfinancial = context is not None and context[2] == "1"
    certified = context is not None and context[1] in inputs.config.certified_report_types
    version_verified = (
        source.text("update_flag") is not None and _date(source, "ann_date") is not None
    )
    local: dict[str, Any] = {}
    balance, balance_refs = _statement_for(
        inputs, source, "balancesheet", ("total_liab", "total_assets")
    )
    income, income_refs = _statement_for(inputs, source, "income", ("operate_profit",))
    cashflow, cash_refs = _statement_for(inputs, source, "cashflow", ("n_cashflow_act",))
    refs += balance_refs + income_refs + cash_refs
    for field, top_row, top_field, bottom_row, bottom_field in (
        ("debt_to_assets", balance, "total_liab", balance, "total_assets"),
        ("ocf_to_profit", cashflow, "n_cashflow_act", income, "operate_profit"),
    ):
        if field not in DISCLOSURE_FIELDS[api]:
            continue
        top, bottom = _money(top_row, top_field), _money(bottom_row, bottom_field)
        computed = None
        if (
            nonfinancial
            and certified
            and version_verified
            and top is not None
            and bottom is not None
            and bottom > 0
        ):
            candidate = top / bottom
            computed = candidate if isfinite(candidate) else None
        local[field] = _comparison(_ratio_value(source, field), computed)
    if any(check["agreement"] is None for check in (*ordinary_checks.values(), *local.values())):
        reasons.append("indicator_unit_or_local_basis_unverified")
    conflict = any(
        check["agreement"] is False for check in (*ordinary_checks.values(), *local.values())
    )
    if conflict:
        reasons.append("indicator_cross_check_conflict")
    return _result(
        api,
        {
            "report_period": _date(source, "end_date"),
            "report_type": source.text("report_type"),
            "comp_type": source.text("comp_type"),
            "version": {
                "ann_date": _date(source, "ann_date"),
                "update_flag": source.text("update_flag"),
            },
            "provider_ratios": {
                field: _ratio_value(source, field)
                for field in DISCLOSURE_FIELDS[api]
                if field in {"roe", "debt_to_assets", "ocf_to_profit"}
            },
            "ordinary_vip_checks": ordinary_checks,
            "local_ratio_checks": local,
            "roe_local_comparison_reason": "average_equity_basis_unavailable",
            "industry_scope": "nonfinancial"
            if nonfinancial
            else "financial"
            if context and context[2] in {"2", "3", "4", "7"}
            else "unknown",
            "scoring_policy": "corroboration_only_no_duplicate_score",
        },
        refs,
        tuple(reasons),
        conflict=conflict,
    )


def _issuance(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """区分发行计划、发行结果和上市阶段，不把预告当完成；例：未来结果公告只保留计划金额。"""
    source = _current(rows, "ann_date" if api == "cb_issue" else "ipo_date")
    refs = tuple(ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api]))
    if source is None:
        return _result(api, {}, refs, ("issuance_date_or_version_unverified",))
    today = inputs.analysis_as_of[:10]
    reasons = []
    if api == "cb_issue":
        planned, actual = (
            _value(source, field, "元") for field in ("plan_issue_size", "issue_size")
        )
        result_day = _date(source, "res_ann_date")
        implemented = result_day is not None and result_day <= today
        stage = "implemented" if implemented else "planned"
        announcement_day = _date(source, "ann_date")
        if (source.text("res_ann_date") is not None and result_day is None) or (
            result_day is not None
            and announcement_day is not None
            and result_day < announcement_day
        ):
            reasons.append("issuance_result_date_unverified")
            implemented = False
            stage = "unknown"
        if planned is None or planned < 0 or (implemented and (actual is None or actual < 0)):
            reasons.append("issuance_amount_unverified")
        facts = {
            "stage": stage,
            "ann_date": _date(source, "ann_date"),
            "result_announcement_date": result_day,
            "planned_amount": planned if planned is not None and planned >= 0 else None,
            "implemented_amount": actual
            if implemented and actual is not None and actual >= 0
            else None,
            "unit": "元",
            "aggregation_policy": "latest_announcement_no_snapshot_sum",
        }
    else:
        ipo, listing = _date(source, "ipo_date"), _date(source, "issue_date")
        amount, price = _value(source, "amount", "股"), _value(source, "price", "元")
        stage = (
            "listed"
            if listing is not None and listing <= today
            else "subscription_started"
            if ipo is not None and ipo <= today
            else "planned"
        )
        if amount is None or amount < 0 or price is None or price <= 0:
            reasons.append("ipo_amount_or_price_unverified")
        if listing is not None and ipo is not None and listing < ipo:
            reasons.append("ipo_date_order_invalid")
        facts = {
            "stage": stage,
            "subscription_date": ipo,
            "listing_date": listing,
            "reported_issue_shares": amount if amount is not None and amount >= 0 else None,
            "issue_price": price if price is not None and price > 0 else None,
            "listed_supply_shares": amount if stage == "listed" and not reasons else None,
            "equity_dilution_ratio": None,
            "dilution_reason": "new_listing_supply_not_existing_share_dilution",
        }
    return _result(api, facts, refs, tuple(reasons))


def _coupons(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """按付息区间构造余额不变的票息情景，不估算应计利息；例：1000元×2%÷2次得到10元。"""
    refs = tuple(ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api]))
    today = inputs.analysis_as_of[:10]
    periods = []
    reasons = []
    active_count = 0
    seen = set()
    for row in rows:
        identity = (row.text("rate_start_date"), row.text("rate_freq"))
        if identity in seen or row.raw.get("business_variant_key"):
            reasons.append("coupon_period_variant_unresolved")
        seen.add(identity)
        start, end = _date(row, "rate_start_date"), _date(row, "rate_end_date")
        rate, frequency = _ratio_value(row, "coupon_rate"), _value(row, "rate_freq", "次/年")
        valid = start is not None and end is not None and start <= end
        state = (
            "unknown"
            if not valid
            else "planned"
            if today < cast(str, start)
            else "completed"
            if today > cast(str, end)
            else "current"
        )
        active_count += state == "current"
        basic, _ = _version(_other_rows(inputs, "cb_basic", row))
        principal = _value(basic, "remain_size", "元")
        if basic is not None:
            refs += basic.evidence("remain_size", "cb_type")
        expense = None
        if (
            valid
            and rate is not None
            and rate >= 0
            and frequency is not None
            and frequency > 0
            and float(frequency).is_integer()
            and principal is not None
            and principal >= 0
        ):
            candidate = principal * rate / frequency
            expense = candidate if isfinite(candidate) else None
        if expense is None:
            reasons.append("coupon_period_unit_or_principal_unverified")
        periods.append(
            {
                "start_date": start,
                "end_date": end,
                "period_state": state,
                "annual_coupon_rate": rate,
                "payments_per_year": frequency,
                "principal_basis": principal,
                "payment_scenario_amount": expense,
            }
        )
    if active_count > 1:
        reasons.append("overlapping_current_coupon_periods")
    if "coupon_period_variant_unresolved" in reasons or active_count > 1:
        for period in periods:
            period["payment_scenario_amount"] = None
    return _result(
        api,
        {
            "periods": periods,
            "scenario": "remaining_principal_constant_per_payment",
            "accrued_interest": None,
            "accrual_reason": "day_count_and_paid_status_unverified",
        },
        refs,
        tuple(reasons),
    )


def _ratings(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """在同机构、同评级类型内比较相邻可见评级，禁止跨方法混排；例：AAA到AA+只记变化。"""
    refs = tuple(
        ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api], "rating_type")
    )
    grouped: dict[tuple[str | None, str | None], list[ResearchRow]] = defaultdict(list)
    for row in rows:
        grouped[row.text("rating_com_name"), row.text("rating_type")].append(row)
    observations, reasons = [], []
    for (agency, kind), items in sorted(grouped.items(), key=lambda item: str(item[0])):
        current = _current(tuple(items), "rating_date")
        prior = (
            None
            if current is None
            else _current(
                tuple(
                    row
                    for row in items
                    if _date(row, "rating_date") is not None
                    and (_date(row, "rating_date") or "") < (_date(current, "rating_date") or "")
                ),
                "rating_date",
            )
        )
        if agency is None or kind is None or current is None or current.text("rating") is None:
            reasons.append("rating_agency_method_or_version_unverified")
        if (
            current is not None
            and (_date(current, "rating_date") or "") > inputs.analysis_as_of[:10]
        ):
            reasons.append("rating_date_in_future")
            current = None
        comparable = (
            agency is not None
            and kind is not None
            and current is not None
            and prior is not None
            and prior.text("rating") is not None
        )
        observations.append(
            {
                "agency": agency,
                "rating_type": kind,
                "rating_date": None if current is None else _date(current, "rating_date"),
                "rating": None if current is None else current.text("rating"),
                "outlook": None if current is None else current.text("rating_outlook"),
                "previous_rating": None
                if not comparable or prior is None
                else prior.text("rating"),
                "rating_changed": None
                if not comparable or current is None or prior is None
                else current.text("rating") != prior.text("rating"),
                "outlook_changed": None
                if not comparable
                or current is None
                or prior is None
                or current.text("rating_outlook") is None
                or prior.text("rating_outlook") is None
                else current.text("rating_outlook") != prior.text("rating_outlook"),
                "change_status": "observed" if comparable else "not_enabled_due_to_history",
            }
        )
    return _result(
        api,
        {"agency_ratings": observations, "ranking_policy": "no_cross_agency_or_method_ordering"},
        refs,
        tuple(reasons),
    )


def _holders(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """计算当前报告期已披露持有人集中度，保持股票、流通股、债券分母边界。

    例：两名股东认证比例20%和10%得到已观察集中度0.3；两条同刻冲突版本或比例和超1
    不发布集中度，十人名单不能推出全体持有人分布。
    """
    refs = tuple(ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api], "ann_date"))
    periods = {_date(row, "end_date") for row in rows}
    if None in periods:
        return _result(api, {}, refs, ("holder_report_period_unverified",))
    period = max(value for value in periods if value is not None)
    grouped: dict[str | None, list[ResearchRow]] = defaultdict(list)
    for row in rows:
        if _date(row, "end_date") == period:
            grouped[row.text("holder_name")].append(row)
    ratio_field = "hold_float_ratio" if api == "top10_floatholders" else "hold_ratio"
    holders: list[dict[str, Any]] = []
    reasons = []
    for name, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        current, reason = _version(tuple(items))
        ratio = _ratio_value(current, ratio_field)
        if name is None or reason is not None or ratio is None or not 0 <= ratio <= 1:
            reasons.append("holder_identity_ratio_or_version_unverified")
        holders.append(
            {
                "holder_name": name,
                "reported_rank": None if current is None else current.text("holder_rank"),
                "ratio": ratio if ratio is not None and 0 <= ratio <= 1 else None,
                "holding_shares": _value(current, "hold_amount", "股")
                if api != "top10_cb_holders"
                else None,
                "reported_change_shares": _value(current, "hold_change", "股")
                if api == "top10_holders"
                else None,
            }
        )
    total = None if reasons else sum(item["ratio"] for item in holders)
    conflict = total is not None and (not isfinite(total) or total > 1 + 1e-10)
    if conflict:
        reasons.append("holder_ratio_sum_out_of_range")
        total = None
    if len(holders) > 10:
        reasons.append("top10_scope_exceeded")
        total = None
    return _result(
        api,
        {
            "report_period": period,
            "holders": holders,
            "observed_holder_count": len(holders),
            "observed_concentration_ratio": total,
            "denominator_scope": "bond_outstanding"
            if api == "top10_cb_holders"
            else "floating_shares"
            if api == "top10_floatholders"
            else "total_shares",
            "full_holder_distribution": None,
            "coverage_boundary": "reported_top10_only",
            "stock_holder_attribution": api != "top10_cb_holders",
        },
        refs,
        tuple(reasons),
        conflict=conflict,
    )


def _holder_counts(
    inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]
) -> dict[str, Any]:
    """保留股东户数当前状态，认证相邻两期后才算变化；例：100户到90户得到-0.1。"""
    refs = tuple(ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api]))
    current = _current(rows, "end_date")
    if current is None:
        return _result(api, {}, refs, ("holder_count_period_or_version_unverified",))
    prior = _current(
        tuple(
            row
            for row in rows
            if _date(row, "end_date") is not None
            and (_date(row, "end_date") or "") < (_date(current, "end_date") or "")
        ),
        "end_date",
    )
    value, previous = _value(current, "holder_num", "户"), _value(prior, "holder_num", "户")
    valid = value is not None and value >= 0 and float(value).is_integer()
    comparable = valid and previous is not None and previous > 0 and float(previous).is_integer()
    change = cast(float, value) / cast(float, previous) - 1 if comparable else None
    return _result(
        api,
        {
            "report_period": _date(current, "end_date"),
            "holder_count": value if valid else None,
            "previous_report_period": None if prior is None else _date(prior, "end_date"),
            "previous_holder_count": previous,
            "change_ratio": change if change is not None and isfinite(change) else None,
            "change_status": "observed"
            if comparable
            else "not_enabled_due_to_history"
            if prior is None
            else "unknown",
        },
        refs,
        () if valid else ("holder_count_unit_or_value_unverified",),
    )


def _management(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """解释任职区间或薪酬持股事实，不据此评价能力或重复汇总多人多职。

    例：未来begin_date保留planned；reward=100元、hold_vol=10股分别保留事实，不换算能力分数。
    """
    refs = tuple(ref for row in rows for ref in row.evidence(*DISCLOSURE_FIELDS[api], "ann_date"))
    reasons, observations = [], []
    selected = rows
    if api == "stk_rewards":
        periods = {_date(row, "end_date") for row in rows}
        if None in periods:
            return _result(api, {}, refs, ("management_report_period_unverified",))
        period = max(value for value in periods if value is not None)
        selected = tuple(row for row in rows if _date(row, "end_date") == period)
    grouped: dict[tuple[str | None, ...], list[ResearchRow]] = defaultdict(list)
    for row in selected:
        key = (row.text("name"), row.text("title")) + (
            (row.text("begin_date"),) if api == "stk_managers" else ()
        )
        grouped[key].append(row)
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        current, reason = _version(tuple(items))
        if current is None or key[0] is None:
            reasons.append(reason or "manager_identity_unverified")
            continue
        fact: dict[str, Any] = {"name": current.text("name"), "title": current.text("title")}
        if api == "stk_managers":
            start, end = _date(current, "begin_date"), _date(current, "end_date")
            today = inputs.analysis_as_of[:10]
            state = (
                "unknown"
                if start is None
                or (current.text("end_date") is not None and end is None)
                or (end is not None and end < start)
                else "planned"
                if start > today
                else "ended"
                if end is not None and end <= today
                else "active"
                if end is not None
                else "open_end_observed"
            )
            if state == "unknown":
                reasons.append("manager_tenure_interval_unverified")
            fact.update(begin_date=start, end_date=end, tenure_state=state)
        else:
            reward, shares = _value(current, "reward", "元"), _value(current, "hold_vol", "股")
            if reward is None or reward < 0 or shares is None or shares < 0:
                reasons.append("management_compensation_unit_or_value_unverified")
            fact.update(
                report_period=_date(current, "end_date"),
                reward=reward if reward is not None and reward >= 0 else None,
                holding_shares=shares if shares is not None and shares >= 0 else None,
            )
        observations.append(fact)
    return _result(
        api,
        {
            "management_facts": observations,
            "management_ability_inference": "not_supported",
            "aggregation_policy": "distinct_person_title_facts_no_sum",
        },
        refs,
        tuple(reasons),
    )


def _conversion_value(
    inputs: ResearchInputs, source: ResearchRow
) -> tuple[float | None, tuple[EvidenceRef, ...], str | None]:
    """用已认证面值、转股价与同日正股成交价核对转股价值，关系必须唯一有效。

    例：CB面值100元、转股价10元和同日正股价12元得到120元；EB、区间外或关系未知返回None。
    """
    relations = _underlying_relations(inputs, source)
    targets = {relation.to_entity_id for relation in relations}
    refs = tuple(ref for relation in relations for ref in relation.evidence_refs)
    if len(targets) != 1:
        return None, refs, "cb_relation_unverified"
    target = next(iter(targets))
    basic, _ = _version(_other_rows(inputs, "cb_basic", source))
    stock = _current(
        tuple(row for row in inputs.rows("daily", target) if _visible(inputs, row)), "trade_date"
    )
    if basic is not None:
        refs += basic.evidence("par", "conv_price", "cb_type", "conv_start_date", "conv_end_date")
    if stock is not None:
        refs += stock.evidence(
            "close",
            "trade_date",
            "vol",
            "amount",
            "price_type",
            "market_activity_status",
            "price_observation_status",
        )
    if basic is not None and basic.text("cb_type") == "EB":
        return None, refs, "exchangeable_bond_relationship_requires_separate_basis"
    quote_day = _date(source, "trade_date")
    start, end = (
        (None, None)
        if basic is None
        else (_date(basic, "conv_start_date"), _date(basic, "conv_end_date"))
    )
    if (
        basic is None
        or basic.text("cb_type") != "CB"
        or quote_day is None
        or start is None
        or end is None
        or not start <= quote_day <= end
        or stock is None
        or _date(stock, "trade_date") != quote_day
        or not price_gate(stock.raw, "premium").passed
    ):
        return None, refs, "conversion_period_or_underlying_quote_unverified"
    par, price, close = (
        _value(basic, "par", "元"),
        _value(basic, "conv_price", "元"),
        _value(stock, "close", "元"),
    )
    if par is None or price is None or close is None or min(par, price, close) <= 0:
        return None, refs, "conversion_value_units_unverified"
    value = par / price * close
    return (value, refs, None) if isfinite(value) else (None, refs, "nonfinite_conversion_value")


def _bond_quotes(inputs: ResearchInputs, api: str, rows: tuple[ResearchRow, ...]) -> dict[str, Any]:
    """核对同债券同日成交报价和供应商转股溢价，不把零成交参考价用于计算。

    例：认证close=110元、cb_value=100元、cb_over_rate=10%得到一致；无正成交量额则未知。
    本地转股价值另需认证面值、正股关系、转股期间与同日正股成交价。
    """
    refs = tuple(
        ref
        for row in rows
        for ref in row.evidence(
            *DISCLOSURE_FIELDS[api],
            "amount",
            "price_type",
            "market_activity_status",
            "price_observation_status",
        )
    )
    source = _current(rows, "trade_date")
    if source is None:
        return _result(api, {}, refs, ("bond_quote_date_or_version_unverified",))
    other_api = "cb_factor_pro" if api == "cb_daily" else "cb_daily"
    other = _current(_other_rows(inputs, other_api, source), "trade_date")
    if other is not None:
        refs += other.evidence(
            "trade_date",
            "close",
            "vol",
            "amount",
            "cb_value",
            "cb_over_rate",
            "price_type",
            "market_activity_status",
            "price_observation_status",
        )
    traded = price_gate(source.raw, "premium").passed
    same_day = other is not None and _date(other, "trade_date") == _date(source, "trade_date")
    other_traded = other is not None and price_gate(other.raw, "premium").passed
    close = _value(source, "close", "元") if traded else None
    comparisons = {
        "close": _comparison(
            close, _value(other, "close", "元") if same_day and other_traded else None
        )
    }
    for name, unit in (("vol", "手"), ("amount", "元")):
        comparisons[name] = _comparison(
            _value(source, name, unit) if traded else None,
            _value(other, name, unit) if same_day and other_traded else None,
        )
    daily = source if api == "cb_daily" else other if same_day else None
    value = _value(daily, "cb_value", "元")
    price = (
        _value(daily, "close", "元")
        if daily is not None and price_gate(daily.raw, "premium").passed
        else None
    )
    premium = None
    if value is not None and value > 0 and price is not None and price > 0:
        candidate = price / value - 1
        premium = candidate if isfinite(candidate) else None
    premium_check = _comparison(_ratio_value(daily, "cb_over_rate"), premium)
    computed_value, conversion_refs, conversion_reason = _conversion_value(inputs, source)
    refs += conversion_refs
    conversion_check = _comparison(value, computed_value)
    reasons = []
    if not traded:
        reasons.append("bond_no_usable_traded_price")
    if any(item["agreement"] is None for item in comparisons.values()):
        reasons.append("bond_quote_peer_date_or_unit_unverified")
    if premium_check["agreement"] is None:
        reasons.append("bond_conversion_value_or_premium_unverified")
    if conversion_reason is not None:
        reasons.append(conversion_reason)
    conflict = any(
        item["agreement"] is False
        for item in (*comparisons.values(), premium_check, conversion_check)
    )
    if conflict:
        reasons.append("bond_quote_or_premium_conflict")
    return _result(
        api,
        {
            "trade_date": _date(source, "trade_date"),
            "traded_price_usable": traded and close is not None,
            "quote_checks": comparisons,
            "conversion_premium_check": premium_check,
            "underlying_conversion_value_check": conversion_check,
            "conversion_value_basis": "provider_reported_same_bond_quote",
            "technical_factor_score_increment": 0,
        },
        refs,
        tuple(reasons),
        conflict=conflict,
    )


def consume_disclosure(
    inputs: ResearchInputs, product: str, api: str, rows: tuple[ResearchRow, ...], entity_id: str
) -> dict[str, Any]:
    """消费P04/P06的十五个新增来源，按证券分别保留事实和旁证，不改变原核心分数。

    例：无认证关系的cb_rating仍在MARKET_ENTITY独立结果保留评级事实和unknown关系；
    已确认关系的EB保留债券事件，equity_dilution_applicable=False。未来及隔离行不进入证据。
    """
    if api not in DISCLOSURE_FIELDS or product != ("P04" if api in FINANCIAL_APIS else "P06"):
        return _result(api, {}, (), ("disclosure_consumer_binding_invalid",))
    visible = tuple(row for row in rows if row.source_api == api and _visible(inputs, row))
    reasons = []
    if len(visible) != len(rows):
        reasons.append("disclosure_unavailable_rows_rejected")
    if not visible:
        return _result(api, {"observations": []}, (), tuple(reasons) + ("source_rows_missing",))
    grouped: dict[str, list[ResearchRow]] = defaultdict(list)
    for row in visible:
        grouped[row.entity_id or row.text("ts_code") or row.availability.source_record_id].append(
            row
        )
    handlers = {
        "bak_basic": _backup,
        "fina_indicator": _indicators,
        "fina_indicator_vip": _indicators,
        "cb_daily": _bond_quotes,
        "cb_factor_pro": _bond_quotes,
        "cb_issue": _issuance,
        "new_share": _issuance,
        "cb_rate": _coupons,
        "cb_rating": _ratings,
        "stk_holdernumber": _holder_counts,
        "stk_managers": _management,
        "stk_rewards": _management,
        "top10_cb_holders": _holders,
        "top10_floatholders": _holders,
        "top10_holders": _holders,
    }
    observations, conflict = [], False
    refs: tuple[EvidenceRef, ...] = ()
    for _identity, items in sorted(grouped.items()):
        source = items[0]
        observation = handlers[api](inputs, api, tuple(items))
        current_facts = observation["facts"] | {
            "source_entity_id": source.entity_id,
            "source_security_code": source.text("ts_code"),
        }
        if api in BOND_APIS:
            relations = _underlying_relations(inputs, source)
            targets = {relation.to_entity_id for relation in relations}
            related = len(targets) == 1 and (entity_id == MARKET_ENTITY or entity_id in targets)
            if not related:
                reasons.append("cb_relation_unverified")
            refs += tuple(ref for relation in relations for ref in relation.evidence_refs)
            basic, _ = _version(_other_rows(inputs, "cb_basic", source))
            if basic is not None:
                refs += basic.evidence("cb_type", "stk_code")
            kind = None if basic is None else basic.text("cb_type")
            current_facts.update(
                underlying_relation_status="confirmed" if related else "unknown",
                underlying_entity_id=next(iter(targets)) if related else None,
                bond_type=kind,
                equity_dilution_applicable=False if kind == "EB" else None,
                equity_dilution_reason="exchangeable_bond_not_issuer_dilution"
                if kind == "EB"
                else "auxiliary_evidence_not_dilution_calculation",
            )
        elif source.entity_id is None or (
            entity_id != MARKET_ENTITY and source.entity_id != entity_id
        ):
            reasons.append("disclosure_entity_binding_unverified")
        reasons.extend(observation["reason_codes"])
        refs += observation["evidence_refs"]
        conflict = conflict or observation["status"] == "conflict"
        observations.append(current_facts)
    return _result(api, {"observations": observations}, refs, tuple(reasons), conflict=conflict)
