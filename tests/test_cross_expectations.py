"""P05公告可见、机构分组及VIP经济事件去重的离线场景。"""

from dataclasses import replace

import pytest
from test_research_governance import inputs, row

from tushare_entitlement_hub.cross_asset_research.expectations import (
    API_FIELDS,
    _attention,
    _broker_revisions,
    _forecast_metrics,
    _interval,
    _latest_event,
    _pair_events,
    _realization,
    build_expectation_revision,
)


def disclosure(
    api="forecast_vip",
    record="f",
    *,
    low=100,
    high=120,
    amount=130,
    day="2026-09-20",
    scope="parent",
    period="20260930",
):
    """场景：认证归母利润预告或快报；预期：报告期和公开时间分别存在，可构造修正和兑现。"""
    values = (
        {"net_profit_min": (low, "元"), "net_profit_max": (high, "元")}
        if api.startswith("forecast")
        else {"n_income": (amount, "元"), "revenue": (300, "元")}
    )
    return row(
        api,
        record,
        day=day,
        raw={
            "ann_date": day,
            "end_date": period,
            "type": "增长",
            "is_audit": "1",
            "profit_scope": scope,
            "change_reason": "成本变化",
        },
        values=values,
        public_available_at=day + "T16:00:00Z",
    )


def forecast_report(
    record, organization="机构A", quarter="2026Q4", value=100, day="2026-09-20", **raw
):
    """场景：带机构及预测期间的研报；预期：同机构同期间才比较前后预测。"""
    return row(
        "report_rc",
        record,
        day=day,
        raw={
            "ts_code": "A",
            "org_name": organization,
            "quarter": quarter,
            "report_date": day,
            "create_time": day + " 10:00:00",
            "author_name": "研究员",
            **raw,
        },
        values={"np": (value, "元"), "eps": (value / 100, "元")},
        public_available_at=day + "T16:00:00Z",
    )


def test_complete_expectation_economic_events_revisions_and_realization():
    """场景：两轮预告、普通版重复、后续快报及独立机构修正；预期：同一事件不双计且差值正确。"""
    sources = [
        disclosure("forecast", "normal"),
        disclosure(),
        disclosure(record="older", low=80, high=100, day="2026-09-10"),
        disclosure("express", "e", day="2026-09-30"),
        disclosure("express_vip", "ev", day="2026-09-30"),
        forecast_report("r1"),
        forecast_report("r2", value=120, day="2026-09-25"),
        row("broker_recommend", raw={"month": "202609", "broker": "机构A", "ts_code": "A"}),
        row(
            "stk_surv",
            raw={
                "ts_code": "A",
                "surv_date": "20260920",
                "rece_org": "机构A",
                "rece_mode": "现场",
                "content": "经营讨论",
            },
        ),
    ]
    result = build_expectation_revision(inputs(*sources))[0]
    assert len(API_FIELDS) == 7
    assert {ref.source_api for ref in result.envelope.evidence_refs} == set(API_FIELDS)
    assert result.metric("forecast_economic_events") == 2
    assert result.metric("express_economic_events") == 1
    assert result.metric("forecast_profit_midpoint") == 110
    assert result.metric("forecast_midpoint_revision") == 20
    assert result.metric("realization_midpoint_gap") == 20
    assert result.metric("realization_outside_range_gap") == 10
    assert result.details["forecast_reconciliation"]["duplicate_count"] == 1
    assert result.details["express_reconciliation"]["duplicate_count"] == 1
    revisions = [item.value for item in result.metrics if item.name.endswith("_np_revision")]
    assert revisions == [20]
    assert result.details["attention"]["stk_surv"]["events_with_content"] == 1
    assert not result.envelope.decision_allowed


def test_vip_disagreement_blocks_economic_event_without_choosing_larger_value():
    """场景：普通预告与VIP同事件利润冲突；预期：事件关闭并给出硬冲突门禁。"""
    normal = disclosure("forecast", "normal", low=80)
    vip = disclosure()
    result = build_expectation_revision(inputs(normal, vip))[0]
    assert result.metric("forecast_profit_midpoint") is None
    assert result.details["forecast_reconciliation"]["conflict_count"] == 1
    assert "disclosure_version_conflict" in result.envelope.gate_codes


def test_report_period_and_observation_time_never_replace_publication_time():
    """场景：预告有报告期却无公开时间，另有未来公告；预期：两者都不能形成可用预测。"""
    candidate = disclosure()
    unknown = replace(
        candidate, availability=replace(candidate.availability, public_available_at=None)
    )
    future = replace(
        candidate,
        availability=replace(candidate.availability, public_available_at="2027-01-01T00:00:00Z"),
    )
    result = build_expectation_revision(inputs(unknown, future))[0]
    assert result.metric("forecast_profit_midpoint") is None
    assert result.details["forecast_reconciliation"]["public_time_or_period_unknown_count"] == 1
    malformed = replace(candidate, raw=candidate.raw | {"end_date": "2026Q3", "ann_date": "bad"})
    assert (
        _pair_events(
            {"forecast_vip": (malformed,)},
            "forecast",
            "forecast_vip",
            ("net_profit_min", "net_profit_max"),
        )[1]["public_time_or_period_unknown_count"]
        == 1
    )


@pytest.mark.parametrize(
    "low,high,expected", [(-20, 10, (-20, 10)), (120, 100, None), (None, 100, None)]
)
def test_forecast_interval_rejects_inverted_or_unknown_values(low, high, expected):
    """场景：亏损区间、倒置区间与缺值；预期：合法负利润可计算，其余保持未知。"""
    assert _interval(disclosure(low=low, high=high)) == expected
    assert _interval(None) is None


def test_same_time_or_scope_conflicts_do_not_make_a_revision():
    """场景：同公开时点两种利润口径；预期：最新事件未知，旧不同口径不被当作修正基线。"""
    normal, other = disclosure(), disclosure(record="other", scope="total")
    assert _latest_event((normal, other), "2026-09-30") is None
    assert _latest_event((), "2026-09-30") is None
    older = disclosure(record="old", day="2026-09-10", scope="total")
    assert _forecast_metrics((older, normal), normal, "2026-09-30")[-1].value is None


@pytest.mark.parametrize(
    "forecast_changes,actual_day,reason",
    [
        ({"scope": None}, "2026-09-30", "profit_scope_unverified"),
        ({}, "2026-09-10", "forecast_after_realization"),
        ({"low": None}, "2026-09-30", "realization_value_unverified"),
    ],
)
def test_realization_requires_ex_ante_same_profit_scope(forecast_changes, actual_day, reason):
    """场景：缺归属口径、快报更早或预告值缺失；预期：兑现差未知并给出对应原因。"""
    metrics, details = _realization(
        disclosure(**forecast_changes), disclosure("express_vip", "e", day=actual_day)
    )
    assert metrics[1].value is None
    assert details["reason_codes"] == [reason]


def test_realization_uses_latest_forecast_before_actual_even_with_later_update():
    """场景：快报后出现更高预告修正；预期：兑现差仍使用快报前可见的110元中点。"""
    result = build_expectation_revision(
        inputs(
            disclosure(),
            disclosure(record="late", low=200, high=220, day="2026-10-02"),
            disclosure("express_vip", "e", day="2026-09-30"),
        )
    )[0]
    assert result.metric("forecast_profit_midpoint") == 210
    assert result.metric("realization_midpoint_gap") == 20


def test_institutions_periods_and_author_conflicts_remain_independent():
    """场景：不同机构、不同预测期和同时间作者冲突；预期：分成独立群组并关闭冲突预测。"""
    reports = (
        forecast_report("a"),
        forecast_report("b", organization="机构B", value=200),
        forecast_report("q", quarter="2027Q4"),
        forecast_report("conflict", author_name="另一人"),
    )
    metrics, details = _broker_revisions(reports)
    assert details["cohort_count"] == 3
    assert all(item.value is None for item in metrics if item.name.endswith("_revision"))
    assert any(item["latest_publication_conflict"] for item in details["cohort_samples"])
    missing = replace(reports[0], raw=reports[0].raw | {"quarter": None})
    assert _broker_revisions((missing,))[1]["unknown_identity_or_period_count"] == 1
    assert _broker_revisions(())[0][-1].value is None


def test_prior_institution_conflict_and_unit_changes_block_revision():
    """场景：先前公开版本冲突或最新单位不一致；预期：机构修正不跨过冲突填值。"""
    older = forecast_report("old")
    duplicate = forecast_report("duplicate")
    newer = forecast_report("new", day="2026-09-25", value=120)
    assert all(
        item.value is None
        for item in _broker_revisions((older, duplicate, newer))[0]
        if item.name.endswith("_revision")
    )
    incompatible = replace(newer, numeric={})
    assert all(
        item.value is None
        for item in _broker_revisions((older, incompatible))[0]
        if item.name.endswith("_revision")
    )


def test_attention_deduplicates_exact_observed_keys_and_handles_empty_sources():
    """场景：同月同机构重复推荐及缺键记录；预期：有效覆盖只计一次，不推导机构仓位。"""
    candidate = row("broker_recommend", raw={"month": "202609", "broker": "机构A", "ts_code": "A"})
    value, details = _attention(
        "broker_recommend", (candidate, candidate, row("broker_recommend", "missing"))
    )
    assert value.value == 1
    assert details["scope"] == "observed_attention_not_positions"
    assert _attention("stk_surv", ())[0].value is None
    assert build_expectation_revision(inputs()) == ()
    rejected = build_expectation_revision(
        inputs(candidate, reasons={"broker_recommend": ("capacity_exceeded",)})
    )[0]
    assert "disclosure_evidence_unavailable" in rejected.envelope.gate_codes


def test_cross_period_realization_is_unknown_even_with_identical_profit_scope():
    """场景：预告和快报利润归属相同但报告期不同；预期：不把跨期差异算作兑现差。"""
    metrics, details = _realization(
        disclosure(), disclosure("express_vip", "e", day="2026-09-30", period="20260630")
    )
    assert metrics[1].value is None
    assert details["reason_codes"] == ["realization_period_mismatch"]


def test_institution_cohort_display_is_bounded_without_truncating_counts():
    """场景：100个独立机构群组；预期：完整群组计数100，仅展示32组指标且保留全量摘要。"""
    reports = tuple(
        forecast_report(str(index), organization=f"机构{index}") for index in range(100)
    )
    metrics, details = _broker_revisions(reports)
    assert metrics[-1].value == 100
    assert len(metrics) == 129
    assert len(details["cohort_samples"]) == 32
    assert (
        details["cohort_results_digest"]
        == _broker_revisions(reports[::-1])[1]["cohort_results_digest"]
    )
