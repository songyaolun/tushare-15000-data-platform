"""市场证据抽样边界、宏观歧义与未展示输入的合成回归。"""

from dataclasses import replace

import pytest
from test_research_market import market_inputs, row

from tushare_entitlement_hub.equity_research import market_evidence as evidence
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    consume_core,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.market import build_market
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY, PRODUCT_IDS
from tushare_entitlement_hub.store import encode


@pytest.mark.parametrize("count", [0, 1, 8, 9, 65])
def test_sample_boundaries_and_duplicate_multiplicity(count):
    """空输入、8行边界及大量重复行均精确计数；样本有界且重复次数改变完整指纹。"""
    aggregate = evidence.MarketEvidence("daily", evidence.DAILY_FIELDS)
    source = row()
    empty = aggregate.summary()["full_fingerprint"]
    for _ in range(count):
        aggregate.add(source, "eligible")
    summary = aggregate.summary()
    assert summary["total_rows"] == count
    assert summary["candidate_reference_count"] == count * 5
    assert summary["sample_count"] == min(count, 1)
    assert summary["truncated"] == (count > 1)
    assert (summary["full_fingerprint"] == empty) == (count == 0)
    expected = (
        tuple(sorted(source.evidence(*evidence.DAILY_FIELDS), key=lambda ref: ref.to_json()))
        if count
        else ()
    )
    assert aggregate.references() == expected


def test_hash_only_and_empty_fields_preserve_rows_without_references():
    """未参与计算的输入与无实际字段行仍进入摘要；不捏造引用或占用展示样本。"""
    for sample in (False, True):
        aggregate = evidence.MarketEvidence("unused", ("absent",), sample=sample)
        aggregate.add(row(), "retained")
        summary = aggregate.summary()
        assert summary["total_rows"] == 1
        assert summary["sample_count"] == summary["candidate_reference_count"] == 0
        assert aggregate.references() == ()
        assert aggregate.source_as_of == {}


def test_sampling_state_and_source_dates_are_bounded_and_order_independent():
    """97条不同来源逐条加入时最多8个样本；反序相同，全部来源日期独立累计。"""
    sources = tuple(row(record=str(index)) for index in range(97))
    forward = evidence.MarketEvidence("daily", evidence.DAILY_FIELDS)
    backward = evidence.MarketEvidence("daily", evidence.DAILY_FIELDS)
    for source in sources:
        forward.add(source, "eligible")
        assert len(forward.samples) <= 8
    for source in reversed(sources):
        backward.add(source, "eligible")
    assert forward.summary() == backward.summary()
    assert forward.references() == backward.references()
    assert forward.source_as_of == {"daily": "2026-09-30"}
    undated = replace(sources[0], availability=replace(sources[0].availability, data_as_of=None))
    collector = evidence.MarketEvidence("undated", evidence.DAILY_FIELDS)
    collector.add(undated, "unknown")
    assert collector.source_as_of == {}
    assert collector.references()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_reference_utf8_capacity_is_exact_and_fail_closed(offset, monkeypatch):
    """引用按UTF8字节校验恰好上限及前后边界；超限抛稳定原因，不截断定位信息。"""
    source = row()
    reference = replace(source.evidence("close")[0], locator="synthetic:合成来源")

    def unicode_reference(self, *fields):
        """提供含合法中文定位符的证据；例：close仍为原字段，只有locator使用UTF8文本。"""
        return (reference,)

    monkeypatch.setattr(type(source), "evidence", unicode_reference)
    size = len(reference.to_json().encode("utf-8"))
    monkeypatch.setattr(evidence, "MAXIMUM_REFERENCE_BYTES", size + offset)
    collector = evidence.MarketEvidence("daily", ("close",))
    if offset < 0:
        with pytest.raises(ValueError, match="^research_market_evidence_capacity_exceeded$"):
            collector.add(source, "eligible")
    else:
        collector.add(source, "eligible")
        assert collector.references() == source.evidence("close")


def test_unsampled_candidate_identity_is_still_validated():
    """候选来源身份错配即失败，不能因最终不抽样而绕过原证据身份校验。"""
    source = row()
    collector = evidence.MarketEvidence("daily", ("close",))
    with pytest.raises(ValueError, match="^source_identity_mismatch$"):
        collector.add(replace(source, raw=source.raw | {"source_record_id": "other"}), "eligible")


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_suspension_observation_capacity_has_explicit_error(offset, monkeypatch):
    """停复牌样本明细按UTF8执行精确上限；超限固定失败，不让另一条展示路径无限放大。"""
    collector = evidence.MarketEvidence("suspension", evidence.SUSPENSION_FIELDS)
    collector.add(
        row(
            "suspend_d",
            fields={"trade_date": "20260930", "suspend_type": "S", "suspend_timing": "上午"},
        ),
        "true",
    )
    expected = collector.suspension_observations()
    size = len(encode(expected[0]).encode("utf-8"))
    monkeypatch.setattr(evidence, "MAXIMUM_OBSERVATION_BYTES", size + offset)
    if offset < 0:
        with pytest.raises(ValueError, match="^research_market_observation_capacity_exceeded$"):
            collector.suspension_observations()
    else:
        assert collector.suspension_observations() == expected


def test_one_security_many_suspension_events_never_expands_details():
    """同一证券65条停牌事件仍全部判定，明细与引用有界，反转事件顺序结果一致。"""
    events = tuple(
        row("suspend_d", record=str(index), fields={"trade_date": "20260930", "suspend_type": "S"})
        for index in range(65)
    )
    result = build_market(market_inputs(1, (row(), *events)))[0]
    assert result.details["excluded_suspended_count"] == 1
    assert result.details["market_evidence_audit"]["suspension_all"]["total_rows"] == 65
    assert result.details["market_evidence_audit"]["suspension_used"]["total_rows"] == 0
    assert len(result.details["suspension_observations"]) == 8
    assert not result.envelope.evidence_refs
    assert len(result.envelope.contrary_evidence) == 19
    reverse = build_market(market_inputs(1, (row(), *reversed(events))))[0]
    assert result.to_json() == reverse.to_json()


def test_rejected_and_unused_inputs_are_not_lost_from_full_fingerprint():
    """时间拒绝的停牌与未选中的边界仍计入输入摘要；改变它们不改指标但改变全量指纹。"""
    future = row(
        "suspend_d",
        record="future",
        public="2027-01-01T00:00:00Z",
        fields={"trade_date": "20260930", "suspend_type": "S"},
    )
    unused = row("stk_limit", index=99, fields={"trade_date": "20260930", "down_limit": 10.0})
    inputs = market_inputs(1, (row(), future, unused))
    before = build_market(inputs)[0]
    changed = market_inputs(
        1,
        (
            row(),
            replace(future, raw=future.raw | {"suspend_type": "R"}),
            replace(unused, numeric={}),
        ),
    )
    after = build_market(changed)[0]
    assert before.metrics == after.metrics
    for role in ("limit_input", "suspension_input"):
        first, second = (
            result.details["market_evidence_audit"][role] for result in (before, after)
        )
        assert first["total_rows"] == second["total_rows"] == 1
        assert first["full_fingerprint"] != second["full_fingerprint"]


@pytest.mark.parametrize("count", [2, 64, 65, 257])
def test_ambiguous_macro_base_and_consumption_are_both_bounded(count):
    """宏观同期间多版本保持unknown，主体及大规模消费同时有界，认证数覆盖所有当期行。"""
    rows = tuple(
        row("cn_ppi", record=str(index), fields={"month": "202609", "ppi_yoy": 0.02})
        for index in range(count)
    )
    inputs = market_inputs(0, rows)
    result = attach_consumption(inputs, PRODUCT_IDS[0], build_market(inputs))[0]
    validate_result_consumption(result)
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_ppi")
    assert state["status"] == "unknown" and "macro_version_ambiguous" in state["reason_codes"]
    base = result.details["market_evidence_audit"]["macro_cn_ppi"]
    assert base["total_rows"] == count and base["sample_count"] == min(count, 8)
    record = next(item for item in result.details["consumption"] if item["api"] == "cn_ppi")
    assert record["retained_row_count"] == record["facts"]["current_record_count"] == count
    assert len(result.to_json().encode()) < 512 * 1024
    if count > 8:
        facts = record["facts"]
        assert facts["certified_values_scope"] == "evidence_sample_only"
        assert facts["certified_field_counts"]["ppi_yoy"] == count
        assert len(facts["certified_values"]["ppi_yoy"]) <= 8
        assert facts["market_evidence_audit"]["total_rows"] == count
    reverse = market_inputs(0, tuple(reversed(rows)))
    assert (
        attach_consumption(reverse, PRODUCT_IDS[0], build_market(reverse))[0].to_json()
        == result.to_json()
    )


@pytest.mark.parametrize("has_field", [False, True])
def test_macro_core_sampling_preserves_observed_and_unknown(has_field):
    """65条缺字段行加可选一条实际字段；空字段不得挤掉有效证据，完全无字段仍unknown。"""
    sources = tuple(
        row("cn_ppi", record=str(index), fields={"ppi_yoy": None}) for index in range(65)
    )
    if has_field:
        sources += (row("cn_ppi", record="actual", fields={"ppi_yoy": 0.02}),)
    outcome = consume_core(market_inputs(0, sources), "P01", "cn_ppi", sources, MARKET_ENTITY)
    assert outcome["status"] == ("observed" if has_field else "unknown")
    assert bool(outcome["reason_codes"]) == (not has_field)
    assert len(outcome["evidence_refs"]) == int(has_field)
    assert outcome["facts"]["market_evidence_audit"]["total_rows"] == len(sources)
