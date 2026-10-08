"""基金披露降级时保留有界来源证据，未知权重与坏父结果不能生成观察值。"""

from dataclasses import replace

import pytest
from test_cross_portfolio import fund_holding, parent_result, portfolio_inputs

from tushare_entitlement_hub.cross_asset_research import portfolio
from tushare_entitlement_hub.cross_asset_research.audit import ConsumptionAudit
from tushare_entitlement_hub.product_foundation.contracts import Truth


def unverified_weight(record="unknown-weight"):
    """生成可见且保留原始权重字段的披露，但去掉权重证书；不能按市值或原数字猜权重。"""
    held = fund_holding(record=record)
    return replace(
        held, numeric={key: value for key, value in held.numeric.items() if key != "stk_mkv_ratio"}
    )


def required_gate(result):
    """取得唯一的持仓输入未知门禁，例：未知持仓必须保留position_input而非伪造可用数值。"""
    gates = [item for item in result.envelope.quality_gates if item.gate_id == "position_input"]
    assert len(gates) == 1
    assert gates[0].result is Truth.UNKNOWN
    return gates[0]


@pytest.mark.parametrize("missing", ["weight", "holding_entity"])
def test_visible_unknown_fund_disclosure_keeps_own_evidence_without_observed(missing):
    """场景：可见披露缺权重证书或持仓证券身份未知；预期：保留本来源证据但不产生指标和观察。"""
    held = (
        unverified_weight()
        if missing == "weight"
        else fund_holding(record="unknown-entity", symbol="UNMAPPED")
    )
    context = portfolio_inputs(held)
    result = portfolio.build_portfolio(context)[0]
    assert result.details["status"] == "input_required"
    assert result.details["reason_codes"] == ["fund_holding_weight_or_entity_unverified"]
    assert result.metrics == ()
    with ConsumptionAudit(("P12",), {}) as audit:
        audit.inputs(context)
        audit.result("P12", result)
        record = audit.summary()["by_api"]["fund_portfolio"]
    assert record["observed"] is False
    assert record["unresolved"] is True
    gate = required_gate(result)
    assert {ref.source_record_id for ref in result.envelope.evidence_refs} == {
        held.raw["source_record_id"]
    }
    assert {ref.source_api for ref in result.envelope.evidence_refs} == {"fund_portfolio"}
    assert {field for ref in gate.evidence_refs for field in ref.fields} == {
        "ann_date", "end_date", "symbol", "mkv", "stk_mkv_ratio"
    }
    assert gate.evidence_refs == result.envelope.evidence_refs


@pytest.mark.parametrize("bad_parent", ["wrong_run", "future_metric_evidence"])
def test_invalid_parent_retains_fund_disclosure_but_never_parent_evidence(bad_parent):
    """场景：父运行不匹配或父指标证据来自未来；预期：只保留基金原始披露，拒收坏父指标与血缘。"""
    held = fund_holding(record="known-fund")
    context = portfolio_inputs(held)
    parent = parent_result(context)
    if bad_parent == "wrong_run":
        parent = replace(parent, envelope=replace(parent.envelope, run_id="unrelated-run"))
    else:
        value = parent.metrics[0]
        reference = replace(value.evidence_refs[0], observed_at="2027-01-01T00:00:00Z")
        parent = replace(parent, metrics=(replace(value, evidence_refs=(reference,)),))
    result = portfolio.build_portfolio(context, (parent,))[0]
    assert result.details["reason_codes"] == ["portfolio_parent_context_or_metric_invalid"]
    assert result.metrics == ()
    assert result.parent_result_ids == ()
    gate = required_gate(result)
    assert {ref.source_record_id for ref in gate.evidence_refs} == {held.raw["source_record_id"]}
    assert {ref.source_api for ref in result.envelope.evidence_refs} == {"fund_portfolio"}
    assert gate.evidence_refs == result.envelope.evidence_refs


def test_unknown_fund_evidence_samples_eight_records_deterministically():
    """场景：17条可见未知权重披露按三种顺序输入；预期：只引用最小8个记录ID且结果字节一致。"""
    rows = tuple(unverified_weight(f"record-{index:02}") for index in range(17))
    variants = (rows, rows[::-1], rows[::2] + rows[1::2])
    results = tuple(
        portfolio.build_portfolio(portfolio_inputs(*variant))[0] for variant in variants
    )
    assert len({result.to_json() for result in results}) == 1
    expected = {row.raw["source_record_id"] for row in rows[:8]}
    for result in results:
        assert result.metrics == ()
        refs = required_gate(result).evidence_refs
        assert {ref.source_record_id for ref in refs} == expected
        assert len({ref.source_record_id for ref in refs}) == 8
        assert len(refs) <= 8 * 5
        assert {ref.source_api for ref in refs} == {"fund_portfolio"}


@pytest.mark.parametrize(
    "unavailable", ["future", "quality_quarantine", "flag_quarantine", "empty"]
)
def test_unavailable_or_absent_fund_disclosure_does_not_acquire_evidence(unavailable):
    """场景：披露尚未公开、隔离状态、隔离标志或空输入；预期：不因降级保留不合法来源引用。"""
    if unavailable == "empty":
        rows = ()
    elif unavailable == "future":
        rows = (fund_holding(public="2027-01-01T00:00:00Z"),)
    else:
        patch = (
            {"quality_tier": "quarantine"}
            if unavailable == "quality_quarantine"
            else {"is_quarantined": True}
        )
        rows = (fund_holding(raw=patch),)
    result = portfolio.build_portfolio(portfolio_inputs(*rows))[0]
    assert result.details["reason_codes"] == ["portfolio_holdings_missing"]
    assert result.metrics == ()
    assert result.envelope.evidence_refs == ()
    assert required_gate(result).evidence_refs == ()
