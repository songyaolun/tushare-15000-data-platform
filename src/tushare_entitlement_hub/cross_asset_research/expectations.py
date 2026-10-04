"""P05 公告可见的业绩预期、机构修正和兑现差异，保留口径及来源链。"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any, cast

from ..equity_research.common import gate, make_result, metric
from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..product_foundation.contracts import EvidenceRef, Truth
from .thematic_common import (
    DETAIL_SAMPLE,
    accepted_groups,
    comparable,
    date_text,
    finite,
    number,
    observation_summary,
    row_key,
    sampled_refs,
    text_values,
)

API_FIELDS = {
    "express": ("ann_date", "end_date", "n_income", "is_audit"),
    "express_vip": ("ann_date", "end_date", "revenue", "n_income", "is_audit"),
    "forecast": ("ann_date", "end_date", "type", "net_profit_min", "net_profit_max"),
    "forecast_vip": (
        "ann_date",
        "end_date",
        "type",
        "net_profit_min",
        "net_profit_max",
        "change_reason",
    ),
    "report_rc": (
        "ts_code",
        "report_date",
        "org_name",
        "author_name",
        "quarter",
        "np",
        "eps",
        "create_time",
    ),
    "broker_recommend": ("month", "broker", "ts_code"),
    "stk_surv": ("ts_code", "surv_date", "rece_org", "rece_mode", "content"),
}
SCOPE_FIELDS = ("profit_scope", "report_type", "accounting_standard")


def _publication_key(row: ResearchRow) -> tuple[str, str]:
    """按公告可用时间排序，报告期不参与公告先后判断。

    输入报告期2026年且公告日2027年，排序键仍以2027年公开时间为首。
    """
    return row.availability.public_available_at or "", date_text(row.text("ann_date")) or ""


def _scope(row: ResearchRow) -> tuple[str | None, ...]:
    """提取显式会计口径，不把空口径解释成合并报表或归母利润。

    输入profit_scope='parent'得到('parent',None,None)，空记录得到三个None。
    """
    return tuple(row.text(field) for field in SCOPE_FIELDS)


def _pair_events(
    sources: dict[str, tuple[ResearchRow, ...]],
    ordinary: str,
    vip: str,
    fields: Sequence[str],
) -> tuple[tuple[ResearchRow, ...], dict[str, Any]]:
    """按同一报告期、公告日及事件类型核对普通版和VIP版，保留单一经济事件。

    输入同一预告普通版与VIP版各一条同值，返回一个VIP事件和duplicate_count=1；
    若认证利润值冲突，该事件不用于计算，conflict_count=1。未公开记录不以报告期回填。
    """
    groups: dict[tuple[Any, ...], list[ResearchRow]] = {}
    rejected = 0
    for api in (ordinary, vip):
        for row in sources.get(api, ()):
            period, announcement = date_text(row.text("end_date")), date_text(row.text("ann_date"))
            if (
                period is None
                or announcement is None
                or row.availability.public_available_at is None
            ):
                rejected += 1
                continue
            key = (
                period,
                announcement,
                row.text("type") if ordinary == "forecast" else "express",
                _scope(row),
            )
            groups.setdefault(key, []).append(row)
    selected, duplicates, conflicts = [], 0, 0
    for key in sorted(groups, key=str):
        rows = groups[key]
        variants = set()
        for row in rows:
            signature: list[Any] = []
            for field in fields:
                fact = row.numeric.get(field)
                signature.append(
                    (
                        number(row, field),
                        None if fact is None else (fact.unit, fact.dimension, fact.currency),
                    )
                )
            if ordinary == "express":
                signature.append(row.text("is_audit"))
            variants.add(tuple(signature))
        if len(variants) != 1:
            conflicts += 1
            continue
        duplicates += len(rows) - 1
        # 固定选择只影响来源证据，不改变经济事件分母。
        selected.append(sorted(rows, key=lambda row: (row.source_api != vip, row_key(row)))[0])
    return tuple(sorted(selected, key=_publication_key)), {
        "economic_event_count": len(groups),
        "usable_event_count": len(selected),
        "duplicate_count": duplicates,
        "conflict_count": conflicts,
        "public_time_or_period_unknown_count": rejected,
    }


def _latest_event(rows: Sequence[ResearchRow], period: str | None) -> ResearchRow | None:
    """取某报告期唯一的最新公开事件，最新同时存在多种口径时返回未知。

    输入同期间两次不同公告返回后公告；最新同日两条事件返回None，避免任意挑选。
    """
    candidates = [row for row in rows if date_text(row.text("end_date")) == period]
    if not candidates:
        return None
    key = max(_publication_key(row) for row in candidates)
    latest = [row for row in candidates if _publication_key(row) == key]
    return latest[0] if len(latest) == 1 else None


def _interval(row: ResearchRow | None) -> tuple[float, float] | None:
    """读取合法认证利润区间，倒置范围、缺单位或币种未认证均关闭。

    输入min=-20元、max=10元返回(-20,10)；输入min=20、max=10返回None。
    """
    low, high = number(row, "net_profit_min", "元"), number(row, "net_profit_max", "元")
    if low is None or high is None or low > high:
        return None
    return low, high


def _forecast_metrics(
    events: Sequence[ResearchRow],
    selected: ResearchRow | None,
    period: str | None,
) -> list[Metric]:
    """计算同报告期、同口径预告区间中点与修正值，禁止将跨期增长当预期修正。

    输入上一公告区间80—100、新公告100—120，返回中点110、修正20元。
    新旧会计口径不同或只有一次公告时，revision保持unknown。
    """
    interval = _interval(selected)
    refs = () if selected is None else selected.evidence(*API_FIELDS[selected.source_api])
    midpoint = None if interval is None else finite(interval[0] / 2 + interval[1] / 2)
    metrics = [
        metric(
            "forecast_profit_midpoint",
            midpoint,
            "元",
            refs,
            () if midpoint is not None else ("forecast_interval_unverified",),
        ),
        metric(
            "forecast_profit_range",
            None if interval is None else finite(interval[1] - interval[0]),
            "元",
            refs,
            () if interval is not None else ("forecast_interval_unverified",),
        ),
    ]
    previous = [
        row
        for row in events
        if selected is not None
        and date_text(row.text("end_date")) == period
        and _publication_key(row) < _publication_key(selected)
        and _scope(row) == _scope(selected)
    ]
    older = _latest_event(previous, period)
    old_interval = _interval(older)
    delta = (
        finite(midpoint - (old_interval[0] / 2 + old_interval[1] / 2))
        if midpoint is not None and old_interval is not None
        else None
    )
    metrics.append(
        metric(
            "forecast_midpoint_revision",
            delta,
            "元",
            refs
            + (
                ()
                if older is None
                else older.evidence("net_profit_min", "net_profit_max", "ann_date")
            ),
            () if delta is not None else ("comparable_prior_forecast_unavailable",),
        )
    )
    return metrics


def _realization(
    forecast: ResearchRow | None,
    actual: ResearchRow | None,
) -> tuple[list[Metric], dict[str, Any]]:
    """以先公开的预告与后公开的同期间快报比较，明确要求利润归属口径一致。

    输入归母口径预告100—120、后续同期间快报130，返回区间外差额10元、中点差20元。
    缺profit_scope或快报先于预告时返回unknown，避免将事后修正当提前预测。
    """
    interval, actual_value = _interval(forecast), number(actual, "n_income", "元")
    refs = (
        ()
        if forecast is None
        else forecast.evidence(
            "net_profit_min", "net_profit_max", "ann_date", "end_date", *SCOPE_FIELDS
        )
    )
    refs += (
        ()
        if actual is None
        else actual.evidence("n_income", "ann_date", "end_date", "is_audit", *SCOPE_FIELDS)
    )
    scope_known = (
        forecast is not None
        and actual is not None
        and forecast.text("profit_scope") is not None
        and _scope(forecast) == _scope(actual)
    )
    ordered = (
        forecast is not None
        and actual is not None
        and _publication_key(forecast) < _publication_key(actual)
    )
    same_period = (
        forecast is not None
        and actual is not None
        and date_text(forecast.text("end_date")) is not None
        and date_text(forecast.text("end_date")) == date_text(actual.text("end_date"))
    )
    valid = (
        scope_known
        and ordered
        and same_period
        and interval is not None
        and actual_value is not None
    )
    midpoint_gap = None
    outside_gap = None
    if valid and interval is not None and actual_value is not None:
        midpoint_gap = finite(actual_value - (interval[0] / 2 + interval[1] / 2))
        outside_gap = finite(actual_value - min(max(actual_value, interval[0]), interval[1]))
    reason = (
        "profit_scope_unverified"
        if not scope_known
        else "forecast_after_realization"
        if not ordered
        else "realization_period_mismatch"
        if not same_period
        else "realization_value_unverified"
    )
    metrics = [
        metric(
            "express_net_income",
            actual_value,
            "元",
            () if actual is None else actual.evidence("n_income"),
            () if actual_value is not None else ("express_value_unverified",),
        ),
        metric(
            "realization_midpoint_gap",
            midpoint_gap,
            "元",
            refs,
            () if midpoint_gap is not None else (reason,),
        ),
        metric(
            "realization_outside_range_gap",
            outside_gap,
            "元",
            refs,
            () if outside_gap is not None else (reason,),
        ),
    ]
    revenue = number(actual, "revenue", "元")
    metrics.append(
        metric(
            "express_revenue",
            revenue,
            "元",
            () if actual is None else actual.evidence("revenue"),
            () if revenue is not None else ("express_revenue_unverified",),
        )
    )
    return metrics, {
        "audit_status": None if actual is None else actual.text("is_audit"),
        "profit_scope": None if actual is None else actual.text("profit_scope"),
        "comparison_status": "available" if valid else "unknown",
        "reason_codes": [] if valid else [reason],
    }


def _broker_revisions(rows: Sequence[ResearchRow]) -> tuple[list[Metric], dict[str, Any]]:
    """按机构、预测期间与会计口径保留独立预测轨迹，不用不同机构数值互算修正。

    输入机构A两次np=100、120及机构B一次200，A修正20元，B修正未知。
    同机构同公开时点多作者冲突关闭该时点；quarter只保留显式期间文本。
    """
    cohorts: dict[tuple[Any, ...], list[ResearchRow]] = {}
    unknown = 0
    for row in rows:
        organization, quarter = row.text("org_name"), row.text("quarter")
        if organization is None or quarter is None or row.availability.public_available_at is None:
            unknown += 1
            continue
        cohorts.setdefault((organization, quarter, _scope(row)), []).append(row)
    metrics: list[Metric] = []
    summaries: list[dict[str, Any]] = []
    for key in sorted(cohorts, key=str):
        cohort_metrics = []
        ordered = sorted(
            cohorts[key], key=lambda row: (row.availability.public_available_at or "", row_key(row))
        )
        by_publication: dict[str, list[ResearchRow]] = {}
        for row in ordered:
            by_publication.setdefault(row.availability.public_available_at or "", []).append(row)
        publications = sorted(by_publication)
        last = by_publication[publications[-1]]
        selected = last[0] if len(last) == 1 else None
        previous = by_publication[publications[-2]] if len(publications) >= 2 else []
        older = previous[0] if len(previous) == 1 else None
        suffix = sha256(str(key).encode("utf-8")).hexdigest()[:16]
        for field in ("np", "eps"):
            estimate = number(selected, field)
            estimate_unit = (
                selected.numeric[field].unit
                if estimate is not None and selected is not None
                else "元"
            )
            cohort_metrics.append(
                metric(
                    "institution_" + suffix + "_" + field + "_estimate",
                    estimate,
                    estimate_unit or "unknown",
                    ()
                    if selected is None
                    else selected.evidence(field, "org_name", "quarter", "report_date"),
                    () if estimate is not None else ("institution_latest_estimate_unverified",),
                )
            )
            valid = (
                selected is not None
                and older is not None
                and comparable(selected, field, older, field)
            )
            value = (
                finite(cast(float, number(selected, field)) - cast(float, number(older, field)))
                if valid
                else None
            )
            refs = (
                ()
                if selected is None
                else selected.evidence(field, "org_name", "quarter", "report_date", "create_time")
            )
            refs += (
                () if older is None else older.evidence(field, "org_name", "quarter", "report_date")
            )
            unit = selected.numeric[field].unit if valid and selected is not None else "元"
            cohort_metrics.append(
                metric(
                    "institution_" + suffix + "_" + field + "_revision",
                    value,
                    unit or "unknown",
                    refs,
                    () if value is not None else ("comparable_institution_prior_unavailable",),
                )
            )
        if len(summaries) < DETAIL_SAMPLE:
            metrics.extend(cohort_metrics)
        summaries.append(
            {
                "institution": key[0],
                "forecast_period": key[1],
                "scope": list(key[2]),
                "metric_key": suffix,
                "publication_count": len(publications),
                "latest_publication_conflict": len(last) != 1,
                "revisions": {
                    item.name.rsplit("_", 2)[-2]: item.value
                    for item in cohort_metrics
                    if item.name.endswith("_revision")
                },
                "estimates": {
                    item.name.rsplit("_", 2)[-2]: item.value
                    for item in cohort_metrics
                    if item.name.endswith("_estimate")
                },
            }
        )
    refs = sampled_refs(rows, API_FIELDS["report_rc"])
    metrics.append(
        metric(
            "institution_forecast_cohort_count",
            len(cohorts) if rows else None,
            "count",
            refs,
            () if rows else ("institution_forecasts_unavailable",),
        )
    )
    return metrics, {
        "cohort_count": len(cohorts),
        "unknown_identity_or_period_count": unknown,
        "cohort_samples": summaries[:DETAIL_SAMPLE],
        "metric_cohort_sample_limit": DETAIL_SAMPLE,
        "cohort_results_digest": sha256(str(summaries).encode("utf-8")).hexdigest(),
        "authors": text_values(rows, "author_name"),
        "cross_institution_averaging": False,
    }


def _attention(api: str, rows: Sequence[ResearchRow]) -> tuple[Metric, dict[str, Any]]:
    """汇总推荐覆盖或调研事件，同键重复只计一次且不解释为机构持仓。

    输入同月同机构两条推荐返回一次覆盖；调研正文只汇总有内容事件数及摘要证据。
    """
    fields = (
        ("month", "broker", "ts_code")
        if api == "broker_recommend"
        else ("ts_code", "surv_date", "rece_org", "rece_mode")
    )
    keys = {
        tuple(row.text(field) for field in fields)
        for row in rows
        if all(row.text(field) is not None for field in fields)
    }
    value = float(len(keys)) if rows else None
    details = {"unique_observed_events": len(keys), "scope": "observed_attention_not_positions"}
    if api == "broker_recommend":
        details["months"] = text_values(rows, "month")
        details["institutions"] = text_values(rows, "broker")
    else:
        details["reception_modes"] = text_values(rows, "rece_mode")
        details["receiving_organizations"] = text_values(rows, "rece_org")
        details["events_with_content"] = sum(row.text("content") is not None for row in rows)
    return metric(
        api + "_observed_coverage",
        value,
        "count",
        sampled_refs(rows, API_FIELDS[api]),
        () if value is not None else ("attention_source_unavailable",),
    ), details


def build_expectation_revision(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """将七个预期接口加工成公告可见的单实体证据链及严格口径修正。

    输入同期间预告、快报、研报及关注事件，输出预告修正、独立机构修正及可认证兑现差；
    空输入返回()。VIP事件不双计；公开时间未知不会使用end_date代替。
    报告期、公开时点和采集时点分别保留，所有结果仅用于人工观察。
    """
    results = []
    for entity, sources in sorted(accepted_groups(inputs, API_FIELDS).items()):
        forecast_events, forecast_audit = _pair_events(
            sources, "forecast", "forecast_vip", ("net_profit_min", "net_profit_max")
        )
        express_events, express_audit = _pair_events(
            sources, "express", "express_vip", ("n_income",)
        )
        periods = sorted(
            {
                period
                for row in forecast_events + express_events
                if (period := date_text(row.text("end_date"))) is not None
            }
        )
        period = periods[-1] if periods else None
        forecast, actual = (
            _latest_event(forecast_events, period),
            _latest_event(express_events, period),
        )
        # 快报兑现必须比较先前可见预告，即使后来还有预告更新。
        prior_forecasts = [
            row
            for row in forecast_events
            if actual is not None and _publication_key(row) < _publication_key(actual)
        ]
        realization_forecast = _latest_event(prior_forecasts, period)
        metrics = _forecast_metrics(forecast_events, forecast, period)
        realization_metrics, realization_details = _realization(realization_forecast, actual)
        metrics.extend(realization_metrics)
        broker_metrics, broker_details = _broker_revisions(sources.get("report_rc", ()))
        metrics.extend(broker_metrics)
        references: list[EvidenceRef] = []
        source_details: dict[str, Any] = {}
        for api, rows in sorted(sources.items()):
            references.extend(sampled_refs(rows, API_FIELDS[api]))
            source_details[api] = observation_summary(rows, API_FIELDS[api])
            source_details[api]["reason_codes"] = list(inputs.reasons(api, entity))
        attention = {}
        for api in ("broker_recommend", "stk_surv"):
            value, summary = _attention(api, sources.get(api, ()))
            metrics.append(value)
            attention[api] = summary
        for family, audit, apis in (
            ("forecast", forecast_audit, ("forecast", "forecast_vip")),
            ("express", express_audit, ("express", "express_vip")),
        ):
            rows = tuple(row for api in apis for row in sources.get(api, ()))
            refs = tuple(
                ref for api in apis for ref in sampled_refs(sources.get(api, ()), API_FIELDS[api])
            )
            metrics.append(
                metric(
                    family + "_economic_events",
                    float(audit["usable_event_count"]) if rows else None,
                    "count",
                    refs,
                    () if rows else ("disclosure_events_unavailable",),
                )
            )
        conflicts = forecast_audit["conflict_count"] + express_audit["conflict_count"]
        details = {
            "sources": source_details,
            "report_period": period,
            "forecast_reconciliation": forecast_audit,
            "express_reconciliation": express_audit,
            "realization": realization_details,
            "institution_revisions": broker_details,
            "attention": attention,
            "forecast_change_reasons": text_values(
                sources.get("forecast_vip", ()), "change_reason"
            ),
            "forecast_types": text_values(forecast_events, "type"),
            "historical_pit_certified": False,
        }
        results.append(
            make_result(
                inputs,
                "expectation_revision/v1",
                entity,
                metrics=metrics,
                references=references,
                gates=(
                    gate(
                        "disclosure_versions_agree",
                        Truth.FALSE if conflicts else Truth.TRUE if references else Truth.UNKNOWN,
                        "disclosure_version_conflict"
                        if conflicts
                        else "disclosure_evidence_unavailable",
                    ),
                ),
                details=details,
                profile="announcement_visible_expectation/v1",
            )
        )
    return tuple(results)
