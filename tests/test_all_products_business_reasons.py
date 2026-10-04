"""业务缺口按真实指标及门禁引用归属接口，分别验证观察与未知原因。"""

from dataclasses import replace

import pytest
from test_all_products_audit import sample_row
from test_research_shared import inputs, source

from tushare_entitlement_hub.cross_asset_research.audit import ConsumptionAudit
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.consumption import attach_consumption
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.product_foundation.contracts import Truth


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("metric_unknown", {"funding_scope_unverified": 1}),
        ("metric_not_applicable", {}),
        ("metric_available", {}),
        ("gate_unknown", {"funding_scope_unverified": 1}),
        ("gate_conflict", {"funding_scope_conflict": 1}),
        ("gate_false", {}),
        ("gate_true", {}),
    ],
)
def test_business_unknown_reason_preserves_observed_and_marks_only_real_gaps(kind, expected):
    """场景：同一行已有可用指标，同时有未知或已知否定；预期：仅未知业务原因使该API未解决。"""
    row = sample_row()
    references = row.evidence("amount", "trade_date")
    values = [metric("funding_amount", 10, "元", references)]
    checks = ()
    if kind.startswith("metric"):
        values.append(
            metric(
                "funding_comparison",
                10 if kind == "metric_available" else None,
                "元",
                references,
                ("funding_scope_unverified",),
                not_applicable=kind == "metric_not_applicable",
            )
        )
    else:
        truth = (
            Truth.FALSE
            if kind == "gate_false"
            else Truth.TRUE
            if kind == "gate_true"
            else Truth.UNKNOWN
        )
        reason = "funding_scope_conflict" if kind == "gate_conflict" else "funding_scope_unverified"
        checks = (gate("funding_comparison", truth, reason, references=references),)
    context = inputs((row,))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        row.entity_id,
        metrics=tuple(values),
        gates=checks,
        references=references,
    )
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        record = audit.summary()["by_api"]["repo_daily"]
    assert record["observed"] is True
    assert record["unresolved"] is bool(expected), record
    assert record.get("consumption_reason_counts", {}) == expected
    assert record["reason_counts"] == expected


@pytest.mark.parametrize("scope", ["no_reference", "different_product"])
def test_product_level_unknown_without_owned_reference_is_not_assigned_to_api(scope):
    """场景：未知指标无引用或仅引用其他产品来源；预期：不把产品级缺口归咎于已观察的API。"""
    row = sample_row()
    foreign = source(api="daily")
    unknown_refs = () if scope == "no_reference" else foreign.evidence("amount")
    context = inputs((row, foreign))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        row.entity_id,
        metrics=(
            metric("funding_amount", 10, "元", row.evidence("amount")),
            metric("global_comparison", None, "元", unknown_refs, ("global_scope_unverified",)),
        ),
    )
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        summary = audit.summary()
    record = summary["by_api"]["repo_daily"]
    assert record["observed"] and not record["unresolved"]
    assert all(not item.get("consumption_reason_counts") for item in summary["by_api"].values())


def test_reason_deduplicates_api_per_item_but_counts_independent_diagnostics():
    """场景：每项指标重复引用同源两个字段，两个指标都有同一缺口；预期：诊断计2次而非引用条数。"""
    row = sample_row()
    refs = row.evidence("amount", "trade_date")
    context = inputs((row,))
    values = (metric("amount", 10, "元", refs),) + tuple(
        metric(f"comparison_{index}", None, "元", refs + refs, ("scope_unverified",))
        for index in range(2)
    )
    assert len(values[1].evidence_refs) == 2
    assert {ref.source_api for ref in values[1].evidence_refs} == {"repo_daily"}
    result = make_result(context, "financing_tail_risk/v1", row.entity_id, metrics=values)
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        record = audit.summary()["by_api"]["repo_daily"]
    assert record["observed_rows"] == record["retained_rows"] == 1
    assert record.get("consumption_reason_counts", {}) == {"scope_unverified": 2}
    assert record["reason_counts"] == {"scope_unverified": 2}


def test_unknown_reason_is_assigned_to_each_referenced_current_product_api():
    """场景：未知比较真实引用回购与大宗成交两个来源；预期：仅这两个API各计一次业务缺口。"""
    row = sample_row()
    other = source(api="bond_blk")
    other = replace(other, raw=other.raw | {"quality_tier": "verified", "trade_date": "2026-09-30"})
    first_refs, second_refs = row.evidence("amount"), other.evidence("amount")
    context = inputs((row, other))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        row.entity_id,
        metrics=(
            metric("repo_amount", 10, "元", first_refs),
            metric("block_amount", 10, "元", second_refs),
            metric(
                "comparison", None, "元", first_refs + second_refs, ("market_scopes_incomparable",)
            ),
        ),
    )
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        summary = audit.summary()
    for api in ("repo_daily", "bond_blk"):
        record = summary["by_api"][api]
        assert record["observed"] and record["unresolved"]
        assert record.get("consumption_reason_counts", {}) == {"market_scopes_incomparable": 1}
    assert all(
        not item.get("consumption_reason_counts")
        for api, item in summary["by_api"].items()
        if api not in {"repo_daily", "bond_blk"}
    )


def test_legacy_consumption_reason_breakdown_does_not_double_total_counts():
    """场景：旧P01无行情行并产生两个真实消费原因；预期：分项展示，原总原因仍各计一次。"""
    context = inputs()
    product = PRODUCT_ALIASES["P01"]
    result = attach_consumption(
        context, product, (make_result(context, product, MARKET_ENTITY),)
    )[0]
    old_record = next(
        record for record in result.details["consumption"] if record["api"] == "index_daily"
    )
    assert old_record["reason_codes"] == [
        "consumption_rows_unavailable", "context_accepted_rows_unavailable"
    ]
    with ConsumptionAudit(("P01",), {}) as audit:
        audit.inputs(context)
        audit.result("P01", result)
        record = audit.summary()["by_api"]["index_daily"]
    assert not record["observed"] and record["unresolved"]
    assert record["reason_counts"] == {
        "consumption_rows_unavailable": 1,
        "context_accepted_rows_unavailable": 1,
        "no_certified_business_observation": 1,
    }
    assert record.get("consumption_reason_counts", {}) == {
        "consumption_rows_unavailable": 1,
        "context_accepted_rows_unavailable": 1,
    }


def test_one_metric_repeated_reason_code_counts_as_one_diagnostic():
    """场景：同一未知指标重复携带相同原因码；预期：该指标在本API只形成一次诊断计数。"""
    row = sample_row()
    refs = row.evidence("amount", "trade_date")
    duplicated = replace(
        metric("comparison", None, "元", refs, ("scope_unverified",)),
        reason_codes=("scope_unverified", "scope_unverified"),
    )
    context = inputs((row,))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        row.entity_id,
        metrics=(metric("funding_amount", 10, "元", refs), duplicated),
    )
    # metric辅助函数会去重，因此用合法模型显式保留重复输入，并确认到达审计边界。
    assert result.metrics[-1].reason_codes == ("scope_unverified", "scope_unverified")
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        record = audit.summary()["by_api"]["repo_daily"]
    assert record["observed"] and record["unresolved"]
    assert record.get("consumption_reason_counts", {}) == {"scope_unverified": 1}
    assert record["reason_counts"] == {"scope_unverified": 1}
