"""消费审计只认可已认证、已引用的本来源业务字段。"""

from dataclasses import replace

import pytest
from test_research_shared import NOW, inputs, source

from tushare_entitlement_hub.cross_asset_research.audit import ConsumptionAudit, certified_fields
from tushare_entitlement_hub.cross_asset_research.registry import registrations
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.product_foundation.contracts import Truth


def sample_row(record="used"):
    """生成P11回购认证行；例：默认amount=10元，并含公开来源与已验证质量。"""
    row = source(record, api="repo_daily")
    return replace(
        row, raw=row.raw | {"quality_tier": "verified", "ts_code": "R1", "trade_date": "2026-09-30"}
    )


def registration():
    """取得P11回购真实登记；例：返回repo_daily的矩阵字段和处理器。"""
    return next(item for item in registrations() if item.spec.api == "repo_daily")


@pytest.mark.parametrize("variant", ["entity", "source", "quarantine", "quality", "future"])
def test_certified_fields_rejects_unusable_identity_quality_and_time(variant):
    """场景：实体、来源、质量、隔离或可见时间非法；预期：整行无法认证字段。"""
    row = sample_row()
    if variant == "entity":
        row = replace(row, entity_id=None)
    elif variant == "source":
        row = replace(row, raw=row.raw | {"source_api": "daily"})
    elif variant == "quarantine":
        row = replace(row, raw=row.raw | {"is_quarantined": True})
    elif variant == "quality":
        row = replace(row, raw=row.raw | {"quality_tier": "quarantined"})
    else:
        row = replace(
            row, availability=replace(row.availability, observed_at="2027-01-01T00:00:00Z")
        )
    assert certified_fields(row, registration(), NOW) == ()


@pytest.mark.parametrize("variant", ["valid", "missing", "unknown", "reasons", "annotation"])
def test_certified_fields_requires_numeric_fact_but_preserves_identity_text(variant):
    """场景：认证数值、缺数值或单位缺口；预期：缺认证不计amount，身份文本仍可认证。"""
    row = sample_row()
    if variant == "missing":
        row = replace(row, numeric={})
    elif variant == "unknown":
        row = replace(
            row,
            numeric={
                "amount": replace(row.numeric["amount"], value=None, reason_codes=("unknown",))
            },
        )
    elif variant == "reasons":
        row = replace(
            row, numeric={"amount": replace(row.numeric["amount"], reason_codes=("unit_missing",))}
        )
    elif variant == "annotation":
        row = replace(row, raw=row.raw | {"quality_tier": "usable_with_annotation"})
    fields = certified_fields(row, registration(), NOW)
    assert "trade_date" in fields
    assert ("amount" in fields) == (variant in {"valid", "annotation"})


def test_audit_counts_unique_actual_references_and_independent_unresolved_states():
    """场景：两条可用行仅引用一条且重复引用；预期：observed_rows=1，缺口独立保留。"""
    used, unused = sample_row(), sample_row("unused")
    context = replace(
        inputs((used, unused)),
        source_reasons={"repo_daily": ("partial_scope",)},
        entity_source_reasons={
            "repo_daily/" + used.entity_id: ("entity_gap",),
            "other/entity": ("ignored",),
        },
    )
    binding = {
        "sources": {
            "repo_daily": {
                "readable": True,
                "configured": True,
                "reason_counts": {"time_gap": 1},
                "unit_gaps": {"close": {"unit_missing": 2}},
            }
        }
    }
    result = make_result(
        context,
        "financing_tail_risk/v1",
        used.entity_id,
        references=used.evidence("amount"),
        metrics=(metric("funding_amount", 10.0, "元", used.evidence("amount")),),
    )
    with ConsumptionAudit(("P11",), binding) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        audit.result("P11", result)
        summary = audit.summary()
    assert summary["total_registered"] == 206
    assert summary["registered"] == summary["assigned"] == summary["executed"] == 14
    assert summary["observed"] == 1
    record = summary["by_api"]["repo_daily"]
    assert (
        record["observed"] and record["unresolved"] and record["readable"] and record["configured"]
    )
    assert record["retained_rows"] == 2 and record["observed_rows"] == 1
    assert record["reason_counts"] == {
        "entity_gap": 1,
        "partial_scope": 1,
        "retained_rows_without_certified_product_reference": 1,
        "time_gap": 1,
        "unit_missing": 2,
    }
    assert not summary["by_api"]["margin"]["readable"]


def test_audit_does_not_count_unrelated_or_unusable_source_evidence():
    """场景：引用其他产品来源及不合格本来源行；预期：执行不等于观察，observed仍为0。"""
    row = replace(sample_row(), entity_id=None)
    other = source("outside", api="daily")
    context = inputs((row, other))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        other.entity_id,
        references=row.evidence("amount") + other.evidence("amount"),
        metrics=(
            metric(
                "unusable_amount", 10.0, "元", row.evidence("amount") + other.evidence("amount")
            ),
        ),
    )
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        record = audit.summary()["by_api"]["repo_daily"]
    assert record["executed"] and not record["observed"]
    assert record["reason_counts"]["record_entity_time_or_unit_unresolved"] == 1
    assert record["reason_counts"]["no_certified_business_observation"] == 1


def test_audit_rejects_returned_product_identity_mismatch():
    """场景：P11处理器却返回P05产品标识；预期：不能登记为正确执行。"""
    result = make_result(inputs(), "expectation_revision/v1", "cn:security:A:listed")
    with ConsumptionAudit(("P11",), {}) as audit:
        with pytest.raises(ValueError, match="all_products_result_product_mismatch"):
            audit.result("P11", result)


@pytest.mark.parametrize(
    "kind", ["explanation", "unknown_metric", "unknown_gate", "available_metric", "true_gate"]
)
def test_observed_requires_available_metric_or_true_gate_reference(kind):
    """场景：相同来源仅作解释或形成可用指标/真门禁；预期：前者不计观察，后者才计观察。"""
    row = sample_row()
    refs = row.evidence("amount")
    values = ()
    checks = ()
    if kind in {"unknown_metric", "available_metric"}:
        values = (
            metric("funding_amount", 10.0 if kind == "available_metric" else None, "元", refs),
        )
    if kind in {"unknown_gate", "true_gate"}:
        checks = (
            gate(
                "funding_evidence",
                Truth.TRUE if kind == "true_gate" else Truth.UNKNOWN,
                "funding_unknown",
                references=refs,
            ),
        )
    context = inputs((row,))
    result = make_result(
        context,
        "financing_tail_risk/v1",
        row.entity_id,
        metrics=values,
        gates=checks,
        references=refs,
    )
    with ConsumptionAudit(("P11",), {}) as audit:
        audit.inputs(context)
        audit.result("P11", result)
        record = audit.summary()["by_api"]["repo_daily"]
    expected = kind in {"available_metric", "true_gate"}
    assert record["observed"] is expected
    assert record["observed_rows"] == int(expected)


def test_numeric_business_key_still_requires_certified_units():
    """场景：数字字段即使属于业务键但无认证；预期：不能因其键身份绕过单位门禁。"""
    row = replace(sample_row(), numeric={})
    item = registration()
    numeric_key = replace(item, spec=replace(item.spec, business_key=("amount",)))
    assert "amount" not in certified_fields(row, numeric_key, NOW)


def test_legacy_market_series_can_use_complete_business_key_without_security_entity():
    """场景：季度宏观序列无证券实体；预期：完整季度键可认证，缺季度键则整行拒绝。"""
    item = next(item for item in registrations() if item.spec.api == "cn_gdp")
    row = replace(source(api="cn_gdp"), entity_id=None, numeric={})
    row = replace(row, raw=row.raw | {"quarter": "2026-Q2", "quality_tier": "verified"})
    assert "quarter" in certified_fields(row, item, NOW)
    assert certified_fields(replace(row, raw=row.raw | {"quarter": None}), item, NOW) == ()
