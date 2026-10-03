"""全量指纹覆盖行语义与未抽样证据变化，聚合门禁保留全部未知/冲突原因。"""

from dataclasses import replace

import pytest
from test_research_context_consumers import INDEX, context_row

from tushare_entitlement_hub.equity_research.context_aggregation import (
    ContextAggregate,
    fingerprint_leaf,
    row_fingerprint,
)
from tushare_entitlement_hub.product_foundation.contracts import Truth


@pytest.mark.parametrize(
    "field",
    [
        "price_type",
        "market_activity_status",
        "price_observation_status",
        "source_units",
        "standard_units",
        "conversion_version",
    ],
)
def test_row_fingerprint_covers_price_and_unit_semantics(field):
    """关键价格/单位语义改变，即使当前数值或整体未知状态未变，行指纹也必须改变。"""
    row = context_row("idx_factor_pro", {"ts_code": "I", "close": 10.0}, entity=INDEX)
    changed = replace(row, raw=row.raw | {field: "changed"})
    assert row_fingerprint(row, ("ts_code", "close")) != row_fingerprint(
        changed,
        ("ts_code", "close"),
    )


def aggregate_observations(rows, observations):
    """使用真实聚合器累加合成领域判断；例如两组冲突与未知保留两个原因码及全量摘要。"""
    with ContextAggregate(
        "index_basic", rows, rows, ("ts_code",), rows[0].availability.observed_at
    ) as aggregate:
        for index, observation in enumerate(observations):
            aggregate.add((index,), 1, observation)
        return aggregate.finish()


def test_full_fingerprint_covers_unsampled_comparison_references():
    """没有改变源行和核对数字，只改变未抽样的对照证据契约时，全量摘要仍必须变化。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    comparisons = tuple(
        context_row("index_daily", {"ts_code": "I"}, entity=INDEX, record=str(i)) for i in range(16)
    )
    observation = {
        "status": "unknown",
        "reason_codes": ("comparison_unverified",),
        "facts": {},
        "evidence_refs": row.evidence("ts_code")
        + tuple(ref for peer in comparisons for ref in peer.evidence("ts_code")),
    }
    before = aggregate_observations((row,), (observation,))
    sampled = {
        ref.source_record_id for ref in before["evidence_refs"] if ref.source_api == "index_daily"
    }
    altered = tuple(
        replace(ref, contract_hash="b" * 64)
        if ref.source_api == "index_daily" and ref.source_record_id not in sampled
        else ref
        for ref in observation["evidence_refs"]
    )
    after = aggregate_observations((row,), (observation | {"evidence_refs": altered},))
    original, changed = (result["facts"]["aggregation"] for result in (before, after))
    assert original["rows_fingerprint"] == changed["rows_fingerprint"]
    assert original["groups_fingerprint"] == changed["groups_fingerprint"]
    assert original["evidence_fingerprint"] != changed["evidence_fingerprint"]
    assert original["full_fingerprint"] != changed["full_fingerprint"]


def test_aggregate_gate_preserves_union_of_group_reasons():
    """混合冲突和未知时，一个软门禁仍保留原因并集，避免信封诊断只剩第一个原因。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    observations = tuple(
        {
            "status": status,
            "reason_codes": (reason,),
            "facts": {},
            "evidence_refs": row.evidence("ts_code"),
        }
        for status, reason in (("conflict", "quote_conflict"), ("unknown", "units_unverified"))
    )
    result = aggregate_observations((row, row), observations)
    assert result["status"] == "conflict"
    assert result["reason_codes"] == ("quote_conflict", "units_unverified")
    assert len(result["gates"]) == 1
    gate = result["gates"][0]
    assert gate.result is Truth.FALSE
    assert gate.reason_codes == result["reason_codes"]
    assert gate.hard is False


def test_sample_keeps_local_fields_when_same_api_certification_witnesses_dominate():
    """同API的单位证书见证行不属于当前消费行时单列支持证据，样本必须保留当前本源字段。

    将当前行选为引用哈希最大的候选，旧的只按API分层会抽走全部本源引用；修复后保留
    当前行及至多4条支持引用，observed消费仍有实际本源证据可验证。
    """
    rows = tuple(
        context_row("index_basic", {"ts_code": "I"}, entity=INDEX, record=str(i)) for i in range(32)
    )
    # fingerprint_leaf与引用的规范JSON采用相同排序/编码，排序用于确定性构造最坏场景。
    rows = tuple(
        sorted(rows, key=lambda row: fingerprint_leaf(row.evidence("ts_code")[0].to_dict()))
    )
    current = rows[-1]
    observation = {
        "status": "observed",
        "reason_codes": (),
        "facts": {},
        "evidence_refs": tuple(ref for row in rows for ref in row.evidence("ts_code")),
    }
    result = aggregate_observations((current,), (observation,))
    assert current.evidence("ts_code")[0] in result["evidence_refs"]
    assert result["facts"]["aggregation"]["sample_count"] == 5


def test_unknown_aggregate_cannot_omit_reason_code():
    """上游领域观察缺原因时聚合仍保守unknown并给固定缺口，不产生无法保存的空原因。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    result = aggregate_observations(
        (row,),
        (
            {
                "status": "unknown",
                "reason_codes": (),
                "facts": {},
                "evidence_refs": row.evidence("ts_code"),
            },
        ),
    )
    assert result["status"] == "unknown"
    assert result["reason_codes"] == ("context_aggregate_unavailable",)
    assert result["gates"][0].reason_codes == result["reason_codes"]
