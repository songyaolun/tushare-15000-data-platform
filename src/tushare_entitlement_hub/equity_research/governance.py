"""股权供给与治理风险：事件证据、潜在供给及监管门禁分别输出。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from statistics import mean
from typing import Any, cast

from ..pit import time_boundary
from ..product_foundation.contracts import (
    EvidenceRef,
    FeatureContribution,
    QualityGateResult,
    Truth,
)
from ..standard import normalize_date
from .common import gate, latest, make_result, metric, percentiles, unique_evidence
from .models import Metric, ResearchInputs, ResearchResult, ResearchRow

EVENT_APIS = (
    "share_float",
    "stk_holdertrade",
    "repurchase",
    "dividend",
    "stk_seasoned",
    "new_share",
    "pledge_stat",
    "pledge_detail",
    "stk_holdernumber",
    "top10_holders",
    "top10_floatholders",
    "st",
    "stock_st",
    "stk_alert",
    "stk_managers",
    "stk_rewards",
    "cb_basic",
    "cb_call",
    "cb_issue",
    "cb_share",
    "cb_rate",
    "cb_rating",
    "top10_cb_holders",
    "cb_daily",
    "cb_factor_pro",
)
EVENT_FIELDS = (
    "ann_date",
    "pub_date",
    "publish_date",
    "imp_date",
    "float_date",
    "end_date",
    "start_date",
    "trade_date",
    "first_ann_date",
    "proc",
    "cur_stage",
    "div_proc",
    "st_type",
    "type",
    "holder_name",
    "share_type",
    "in_de",
    "change_reason",
    "cb_type",
    "call_type",
    "call_desc",
)
ST_ACTIVE = {"ST", "*ST", "实施ST", "实施*ST", "风险警示", "退市风险警示"}
ST_CLEAR = {"撤销ST", "撤销*ST", "撤销风险警示", "撤销退市风险警示"}


def _day(value: str | None) -> date | None:
    """仅解析真实日期，非法或缺失值返回未知；例：20261003得到2026-10-03。"""
    try:
        return date.fromisoformat(normalize_date(value))
    except (ValueError, TypeError):
        return None


def _number(row: ResearchRow | None, field: str, unit: str) -> float | None:
    """读取单位已认证数值，金额另要求人民币；例：美元amount不参与元金额运算。"""
    if row is None:
        return None
    value = row.number(field, unit)
    if value is not None and unit == "元" and row.numeric[field].currency != "CNY":
        return None
    return value


def _ambiguous(rows: tuple[ResearchRow, ...]) -> bool:
    """保留无法判定独立性的来源变体；例：同业务组两条不同来源记录返回True。"""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.raw.get("business_variant_key"):
            return True
        groups[row.source_api + "/" + str(row.raw.get("business_group_key"))].add(
            str(row.raw["source_record_id"])
        )
        if row.source_api == "share_float":
            batch = "share_float/batch/" + repr(
                (row.text("float_date"), row.text("holder_name"), row.text("share_type"))
            )
            groups[batch].add(str(row.raw["source_record_id"]))
    return any(len(records) > 1 for records in groups.values())


def _event(row: ResearchRow, today: date) -> dict[str, Any]:
    """提取可追溯事件阶段，不把未知阶段解释成实施；例：董事会预案得到planned。"""
    stage_value = row.text("cur_stage") or row.text("proc") or row.text("div_proc")
    stages = {
        "意向": "announced",
        "董事会预案": "planned",
        "股东大会通过": "planned",
        "监管审批": "planned",
        "预案": "planned",
        "实施": "implemented",
        "完成": "implemented",
        "终止": "terminated",
    }
    stage = stages.get(stage_value or "", "unknown")
    event_day = _day(row.text("float_date") or row.text("imp_date"))
    if event_day is not None:
        stage = "planned" if event_day > today else "effective"
    amount = _number(row, "amount", "元")
    return {
        "source_api": row.source_api,
        "source_record_id": row.raw["source_record_id"],
        "business_group_key": row.raw["business_group_key"],
        "business_variant_key": row.raw.get("business_variant_key"),
        "event_stage": stage,
        "event_date": event_day.isoformat() if event_day else None,
        "calendar_days_to_event": (event_day - today).days if event_day else None,
        "trading_days_to_event": None,
        "trading_distance_reason": "trading_calendar_not_certified",
        "planned_amount": amount if stage == "planned" else None,
        "executed_amount": amount if stage == "implemented" else None,
        "source_fields": {field: row.raw[field] for field in EVENT_FIELDS if field in row.raw},
    }


def _absent(name: str, api: str, inputs: ResearchInputs, entity_id: str) -> Metric:
    """完整观察范围无事件标不适用，其余未知；例：已完整采集且无CB不输出零稀释。"""
    complete = api in inputs.config.complete_scopes and not inputs.reasons(api, entity_id)
    return metric(
        name,
        None,
        "ratio",
        reasons=("no_confirmed_related_event" if complete else "event_scope_unknown",)
        + inputs.reasons(api, entity_id),
        not_applicable=complete,
    )


def _supply_metric(
    inputs: ResearchInputs,
    entity_id: str,
    api: str,
    name: str,
    rows: tuple[ResearchRow, ...],
    field: str,
    denominator: str,
) -> Metric:
    """对可辨独立事件求股本比率，任一缺失即不发布总量；例：100股/1000股得到0.1。"""
    if not rows:
        return _absent(name, api, inputs, entity_id)
    basic = latest(inputs.rows("daily_basic", entity_id))
    base = _number(basic, denominator, "股")
    values = [_number(row, field, "股") for row in rows]
    references = tuple(ref for row in rows for ref in row.evidence(field))
    if basic is not None:
        references += basic.evidence(denominator)
    reasons = list(inputs.reasons(api, entity_id))
    if _ambiguous(rows):
        reasons.append("business_variant_unresolved")
    if base is None or base <= 0 or any(value is None or value < 0 for value in values):
        reasons.append("share_amount_or_denominator_unverified")
    value = (
        None if reasons else sum(value for value in values if value is not None) / cast(float, base)
    )
    return metric(name, value, "ratio", references, reasons)


def _unlock(inputs: ResearchInputs, entity_id: str, today: date) -> Metric:
    """按未来30自然日选择已公告解禁；例：第30日计入，第31日不计入。"""
    rows = inputs.rows("share_float", entity_id)
    selected = []
    invalid = []
    for row in rows:
        announcement = _day(row.text("ann_date"))
        event_day = _day(row.text("float_date"))
        if announcement is None or event_day is None or announcement > today:
            invalid.append(row)
        elif today < event_day <= today + timedelta(days=30):
            selected.append(row)
    if invalid:
        return metric(
            "unlock_30d",
            None,
            "ratio",
            reasons=("event_date_unverified",),
            references=(ref for row in invalid for ref in row.evidence("ann_date")),
        )
    return _supply_metric(
        inputs,
        entity_id,
        "share_float",
        "unlock_30d",
        tuple(selected),
        "float_share",
        "float_share",
    )


def _placement(inputs: ResearchInputs, entity_id: str, *, executed: bool = False) -> Metric:
    """计划与实施使用不同字段，不把意向计作确认发行；例：实施仅读取fo_vol_act。"""
    rows = inputs.rows("stk_seasoned", entity_id)
    name = "placement_executed_dilution" if executed else "placement_dilution"
    known = {"董事会预案", "监管审批", "实施", "终止"}
    if any(row.text("cur_stage") not in known for row in rows):
        return metric(name, None, "ratio", reasons=("placement_stage_unknown",))
    lineages: dict[tuple[str | None, str | None], list[ResearchRow]] = defaultdict(list)
    for row in rows:
        lineages[row.text("first_ann_date"), row.text("fo_type")].append(row)
    revisions = []
    for identity, candidates in lineages.items():
        if len(candidates) == 1:
            revisions.extend(candidates)
            continue
        current = latest(tuple(candidates))
        if (
            None in identity
            or current is None
            or current.text("update_flag") != "1"
            or any(
                time_boundary(other.availability.available_at)
                >= time_boundary(current.availability.available_at)
                for other in candidates
                if other is not current
            )
        ):
            return metric(name, None, "ratio", reasons=("placement_revision_unresolved",))
        revisions.append(current)
    selected = tuple(
        row
        for row in revisions
        if row.text("cur_stage") in ({"实施"} if executed else {"董事会预案", "监管审批"})
    )
    return _supply_metric(
        inputs,
        entity_id,
        "stk_seasoned",
        name,
        selected,
        "fo_vol_act" if executed else "fo_vol_high",
        "total_share",
    )


def _convertible(inputs: ResearchInputs, entity_id: str, today: date) -> Metric:
    """只计算已认证CB正股关系及转股期间内的上界；例：100元/10元/100股得到0.1。"""
    relations = tuple(
        relation
        for relation in inputs.config.confirmed_relations
        if relation.to_entity_id == entity_id
        and relation.confirmed
        and relation.relation_type == "cb_underlying"
        and relation.usable_at(inputs.analysis_as_of)
    )
    bonds = {relation.from_entity_id for relation in relations}
    rows = tuple(row for bond in sorted(bonds) for row in inputs.rows("cb_basic", bond))
    reasons = list(inputs.reasons("cb_basic", entity_id))
    reasons.extend(reason for bond in bonds for reason in inputs.reasons("cb_basic", bond))
    if any(not inputs.rows("cb_basic", bond) for bond in bonds):
        reasons.append("cb_reference_missing")
    if not rows:
        stock = next(
            entity
            for entity in inputs.config.foundation.entities
            if entity.identity.entity_id == entity_id
        )
        unresolved = any(
            row.text("stk_code") in {None, stock.identity.local_id}
            for row in inputs.rows("cb_basic")
        )
        if unresolved or reasons:
            return metric(
                "cb_potential_dilution",
                None,
                "ratio",
                reasons=tuple(reasons) + (("cb_relation_unverified",) if unresolved else ()),
            )
        return _absent("cb_potential_dilution", "cb_basic", inputs, entity_id)
    if all(row.text("cb_type") == "EB" for row in rows):
        return metric(
            "cb_potential_dilution",
            None,
            "ratio",
            not_applicable=not reasons,
            reasons=tuple(reasons) or ("exchangeable_bond_not_issuer_dilution",),
        )
    basic = latest(inputs.rows("daily_basic", entity_id))
    total = _number(basic, "total_share", "股")
    shares = []
    references = tuple(ref for relation in relations for ref in relation.evidence_refs)
    if basic is not None:
        references += basic.evidence("total_share")
    for row in rows:
        references += row.evidence(
            "remain_size", "conv_price", "cb_type", "conv_start_date", "conv_end_date"
        )
        if row.text("cb_type") == "EB":
            continue
        start, stop = _day(row.text("conv_start_date")), _day(row.text("conv_end_date"))
        amount, price = _number(row, "remain_size", "元"), _number(row, "conv_price", "元")
        if (
            row.text("cb_type") != "CB"
            or start is None
            or stop is None
            or not start <= today <= stop
            or amount is None
            or amount < 0
            or price is None
            or price <= 0
        ):
            reasons.append("cb_terms_unverified")
        else:
            shares.append(amount / price)
    if _ambiguous(rows):
        reasons.append("business_variant_unresolved")
    if total is None or total <= 0:
        reasons.append("share_denominator_unverified")
    return metric(
        "cb_potential_dilution",
        None if reasons else sum(shares) / cast(float, total),
        "ratio",
        references,
        reasons,
    )


def _regulatory(inputs: ResearchInputs, entity_id: str, today: date) -> list[QualityGateResult]:
    """独立判定ST实施、重点监控与冲突；例：未来实施ST保留计划，当前ST硬失败。"""
    state_rows = inputs.rows("st", entity_id)
    states: list[Truth] = []
    references: list[EvidenceRef] = []
    for row in state_rows:
        references.extend(row.evidence("st_type", "imp_date"))
        effective = _day(row.text("imp_date"))
        label = row.text("st_type")
        if effective is None or label not in ST_ACTIVE | ST_CLEAR:
            states.append(Truth.UNKNOWN)
        elif effective <= today:
            states.append(Truth.FALSE if label in ST_ACTIVE else Truth.TRUE)
    dated_states = [(row, _day(row.text("imp_date"))) for row in state_rows]
    effective_dates = [day for _, day in dated_states if day is not None and day <= today]
    if effective_dates:
        current = max(effective_dates)
        current_labels = {row.text("st_type") for row, day in dated_states if day == current}
        if current_labels <= ST_CLEAR:
            states = [Truth.TRUE]
        elif current_labels <= ST_ACTIVE:
            states = [Truth.FALSE]
    if any(
        _day(row.text("imp_date")) is None or row.text("st_type") not in ST_ACTIVE | ST_CLEAR
        for row in state_rows
    ):
        states.append(Truth.UNKNOWN)
    if inputs.reasons("st", entity_id) or inputs.reasons("stock_st", entity_id):
        states.append(Truth.UNKNOWN)
    stock_rows = inputs.rows("stock_st", entity_id)
    expected = inputs.config.foundation.expected_data_as_of.get("stock_st")
    if stock_rows and expected is None:
        states.append(Truth.UNKNOWN)
    current_stock = tuple(
        row
        for row in stock_rows
        if expected is not None and row.availability.data_as_of == expected
    )
    if current_stock:
        references.extend(
            ref for row in current_stock for ref in row.evidence("type", "trade_date")
        )
        states.append(Truth.FALSE)
    conflict = Truth.TRUE in states and Truth.FALSE in states
    if Truth.FALSE in states:
        state = Truth.FALSE
    elif Truth.UNKNOWN in states:
        state = Truth.UNKNOWN
    elif states or ({"st", "stock_st"} <= set(inputs.config.complete_scopes)):
        state = Truth.TRUE
    else:
        state = Truth.UNKNOWN
    monitoring = inputs.rows("stk_alert", entity_id)
    active, uncertain = False, bool(inputs.reasons("stk_alert", entity_id))
    for row in monitoring:
        start = _day(row.text("start_date"))
        if start is None:
            uncertain = True
        elif start <= today and not (
            row.availability.effective_to is not None
            and time_boundary(row.availability.effective_to) <= time_boundary(inputs.analysis_as_of)
        ):
            active = True
    monitor_state = (
        Truth.FALSE
        if active
        else Truth.UNKNOWN
        if uncertain
        else (Truth.TRUE if "stk_alert" in inputs.config.complete_scopes else Truth.UNKNOWN)
    )
    all_rows = tuple(row for api in EVENT_APIS for row in inputs.rows(api, entity_id))
    conflict = conflict or any(row.raw.get("state_conflict") is True for row in all_rows)
    plan_states: dict[tuple[str, str, str], set[str | None]] = defaultdict(set)
    for row in inputs.rows("stk_seasoned", entity_id):
        first, kind = row.text("first_ann_date"), row.text("fo_type")
        if first is not None and kind is not None:
            plan_states[first, kind, row.availability.available_at].add(row.text("cur_stage"))
            references.extend(row.evidence("first_ann_date", "fo_type", "cur_stage"))
    conflict = conflict or any({"实施", "终止"} <= stages for stages in plan_states.values())
    return [
        gate("governance_st", state, "st_active_or_unverified", references=references),
        gate(
            "governance_monitoring",
            monitor_state,
            "monitoring_active_or_unverified",
            references=(
                ref for row in monitoring for ref in row.evidence("start_date", "end_date")
            ),
        ),
        gate(
            "governance_conflict",
            Truth.FALSE if conflict else Truth.TRUE,
            "major_state_conflict",
            references=references,
        ),
    ]


def build_governance(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """按显式股票实体生成供给风险及事件清单，空认证不产生无风险结论。

    例：完整输入100股解禁、1000股流通股本得到unlock_30d=0.1；缺单位得到unknown。
    供给评分只使用同一特征档的同行分位，ST硬失败独立于分数。
    """
    today = date.fromisoformat(inputs.analysis_as_of[:10])
    features: dict[str, list[Metric]] = {}
    for entity_id in inputs.entity_ids:
        pledge = latest(inputs.rows("pledge_stat", entity_id))
        ratio = _number(pledge, "pledge_ratio", "fraction")
        features[entity_id] = [
            _unlock(inputs, entity_id, today),
            _placement(inputs, entity_id),
            _placement(inputs, entity_id, executed=True),
            _convertible(inputs, entity_id, today),
            metric(
                "pledge_ratio",
                ratio if ratio is not None and 0 <= ratio <= 1 else None,
                "fraction",
                pledge.evidence("pledge_ratio") if pledge else (),
                inputs.reasons("pledge_stat", entity_id),
            ),
        ]
    ranking: dict[str, dict[str, Any]] = {}
    for entity_id, items in features.items():
        lookup = {item.name: item for item in items}
        financing = [lookup[name] for name in ("placement_dilution", "cb_potential_dilution")]
        known = [item for item in financing if item.status == "available"]
        status = (
            "unknown"
            if any(item.status == "unknown" for item in financing)
            else ("available" if known else "not_applicable")
        )
        combined = metric(
            "financing_dilution",
            sum(cast(float, item.value) for item in known) if status == "available" else None,
            "ratio",
            (ref for item in known for ref in item.evidence_refs),
            not_applicable=status == "not_applicable",
        )
        items.append(combined)
        selected = [lookup["unlock_30d"], lookup["pledge_ratio"], combined]
        profile = "supply:" + ":".join(
            item.name for item in selected if item.status != "not_applicable"
        )
        ranking[entity_id] = {"profile": profile, "selected": selected}
    results = []
    for profile in sorted({row["profile"] for row in ranking.values()}):
        identities = [identity for identity, row in ranking.items() if row["profile"] == profile]
        ranks = {
            name: percentiles(
                inputs,
                {
                    identity: next(
                        item.value for item in ranking[identity]["selected"] if item.name == name
                    )
                    for identity in identities
                },
                profile + ":" + name,
            )
            for name in ("unlock_30d", "pledge_ratio", "financing_dilution")
        }
        for entity_id in identities:
            selected = [
                item for item in ranking[entity_id]["selected"] if item.status != "not_applicable"
            ]
            normalized = [ranks[item.name][entity_id].percentile for item in selected]
            score = (
                100 * mean(cast(float, value) for value in normalized)
                if (selected and all(value is not None for value in normalized))
                else None
            )
            contributions = tuple(
                FeatureContribution(
                    feature_id=item.name,
                    raw_value=item.value,
                    unit=item.unit,
                    transformed_value=value,
                    transform="peer_percentile",
                    peer_level=ranks[item.name][entity_id].peer_level,
                    weight=1 / len(selected),
                    contribution=100 * value / len(selected) if value is not None else None,
                    evidence_refs=item.evidence_refs,
                )
                for item, value in zip(selected, normalized, strict=True)
            )
            related = {
                relation.from_entity_id
                for relation in inputs.config.confirmed_relations
                if relation.to_entity_id == entity_id
                and relation.confirmed
                and relation.relation_type == "cb_underlying"
                and relation.usable_at(inputs.analysis_as_of)
            } | {entity_id}
            rows = tuple(
                row for api in EVENT_APIS for row in inputs.rows(api) if row.entity_id in related
            )
            events = [_event(row, today) for row in rows]
            evidence = unique_evidence(ref for row in rows for ref in row.evidence(*EVENT_FIELDS))
            scores = metric(
                "R_supply",
                score,
                "score",
                (ref for item in selected for ref in item.evidence_refs),
                () if score is not None else ("supply_peer_or_feature_unavailable",),
            )
            results.append(
                make_result(
                    inputs,
                    "capital_governance_risks/v1",
                    entity_id,
                    metrics=features[entity_id] + [scores],
                    gates=_regulatory(inputs, entity_id, today),
                    contributions=contributions,
                    references=evidence,
                    profile=profile,
                    details={
                        "events": events,
                        "variant_status": "unresolved"
                        if _ambiguous(rows)
                        else "distinct_observed_records",
                        "unlock_window_calendar_days": 30,
                        "cb_scenario": "all_remaining_cb_convert_upper_bound",
                        "repurchase_snapshot_policy": "preserve_separate_no_sum",
                        "source_scope": "observed_research",
                    },
                )
            )
    return tuple(sorted(results, key=lambda result: result.envelope.entity_id))
