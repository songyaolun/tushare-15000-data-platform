"""定点复现市场证据审查的四项缺陷，不访问正式参考数据。"""

import json
from collections import Counter
from dataclasses import replace

import duckdb
import pytest
from test_research_market import market_inputs, row

from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.context_aggregation import (
    fingerprint_leaf,
    row_fingerprint,
)
from tushare_entitlement_hub.equity_research.market import build_market
from tushare_entitlement_hub.equity_research.market_evidence import (
    SUSPENSION_FIELDS,
    MarketEvidence,
)
from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS
from tushare_entitlement_hub.equity_research.storage import (
    fingerprints,
    initialize_tables,
    record_rows,
    save_results,
    stored_rows,
)


@pytest.mark.parametrize("count", [0, 1, 8, 9, 32, 64, 65])
def test_p01_core_uses_eight_rows_at_every_size(count):
    """0/1/8/9/32/64/65条宏观记录全量计数，消费值与门禁最多引用8行，超限披露审计。"""
    sources = tuple(
        row("cn_ppi", record=str(index), fields={"month": "202609", "ppi_yoy": 0.02})
        for index in range(count)
    )
    inputs = market_inputs(0, sources)
    result = attach_consumption(inputs, PRODUCT_IDS[0], build_market(inputs))[0]
    validate_result_consumption(result)
    record = next(item for item in result.details["consumption"] if item["api"] == "cn_ppi")
    facts = record["facts"]
    assert record["retained_row_count"] == facts["current_record_count"] == count
    assert len(facts["certified_values"]["ppi_yoy"]) == min(count, 8)
    assert len({ref["source_record_id"] for ref in record["evidence_refs"]}) == min(count, 8)
    assert all(len(gate["evidence_refs"]) <= 16 for gate in record["gates"])
    if count > 8:
        audit = facts["market_evidence_audit"]
        assert audit["total_rows"] == count and audit["sample_count"] == 8
        assert audit["candidate_reference_count"] == 2 * count
        assert audit["truncated"] is True
        assert facts["certified_field_counts"]["ppi_yoy"] == count
        assert facts["certified_values_scope"] == "evidence_sample_only"
    reverse = market_inputs(0, tuple(reversed(sources)))
    assert (
        attach_consumption(reverse, PRODUCT_IDS[0], build_market(reverse))[0].to_json()
        == result.to_json()
    )


@pytest.mark.parametrize(
    "api,period_field,field,current_period,previous_period",
    [
        ("cn_ppi", "month", "ppi_yoy", "202609", "202608"),
        ("cn_gdp", "quarter", "gdp_yoy", "2026Q2", "2026Q1"),
    ],
)
@pytest.mark.parametrize("previous_count", [2, 12])
def test_ambiguous_adjacent_macro_versions_all_enter_fingerprint(
    api, period_field, field, current_period, previous_period, previous_count
):
    """当前期唯一、相邻期2/12个歧义版本；全部版本入指纹，未抽样旧值变化可检出且不发布变化率。"""
    current = row(api, record="current", fields={period_field: current_period, field: 0.02})
    previous = tuple(
        row(api, record=f"previous-{index}", fields={period_field: previous_period, field: 0.01})
        for index in range(previous_count)
    )
    inputs = market_inputs(0, (current, *previous))
    result = build_market(inputs)[0]
    state = next(item for item in result.details["macro_states"] if item["source_api"] == api)
    assert state["value"] == 0.02 and state["change"] is None
    assert "adjacent_period_unavailable" in state["reason_codes"]
    audit = result.details["market_evidence_audit"]["macro_" + api]
    assert audit["total_rows"] == previous_count + 1
    assert audit["candidate_reference_count"] == 2 * (previous_count + 1)
    assert audit["sample_count"] == min(previous_count + 1, 8)
    sampled = {sample["source_record_id"] for sample in audit["samples"]}
    chosen = next(
        (item for item in previous if item.raw["source_record_id"] not in sampled), previous[0]
    )
    changed = replace(
        chosen,
        raw=chosen.raw | {field: 0.03},
        numeric={
            field: replace(chosen.numeric[field], value=0.03),
        },
    )
    replaced_rows = tuple(changed if item is chosen else item for item in (current, *previous))
    after = build_market(market_inputs(0, replaced_rows))[0]
    assert after.details["macro_states"] == result.details["macro_states"]
    assert (
        after.details["market_evidence_audit"]["macro_" + api]["full_fingerprint"]
        != audit["full_fingerprint"]
    )
    assert (
        build_market(market_inputs(0, tuple(reversed((current, *previous)))))[0].to_json()
        == result.to_json()
    )


def test_superseded_current_macro_version_is_audited_without_changing_value():
    """当期早版被较晚公开版本取代；早版仍参与候选判定的指纹，指标只取唯一新版本。"""
    old = row(
        "cn_ppi",
        record="old",
        fields={"month": "202609", "ppi_yoy": 0.01},
        public="2026-09-29T12:00:00Z",
    )
    new = row("cn_ppi", record="new", fields={"month": "202609", "ppi_yoy": 0.02})
    result = build_market(market_inputs(0, (old, new)))[0]
    audit = result.details["market_evidence_audit"]["macro_cn_ppi"]
    assert audit["total_rows"] == audit["sample_count"] == 2
    assert (
        next(state for state in result.details["macro_states"] if state["source_api"] == "cn_ppi")[
            "value"
        ]
        == 0.02
    )


def test_suspension_evidence_roles_are_disjoint_through_storage():
    """复牌仅为used、停牌与未知仅为contrary；全状态审计保留三行，512MB落库后血缘互斥。"""
    sources = tuple(row(index=index, record=f"daily-{index}") for index in range(3)) + tuple(
        row(
            "suspend_d",
            index=index,
            record=record,
            fields={
                "trade_date": "20260930",
                "suspend_type": kind,
                "suspend_timing": None,
            },
        )
        for index, (record, kind) in enumerate((("resume", "R"), ("stop", "S"), ("uncertain", "X")))
    )
    inputs = market_inputs(3, sources)
    result = attach_consumption(inputs, PRODUCT_IDS[0], build_market(inputs))[0]
    validate_result_consumption(result)
    used = result.envelope.evidence_refs
    contrary = result.envelope.contrary_evidence
    assert not {ref.to_json() for ref in used} & {ref.to_json() for ref in contrary}
    assert {ref.source_record_id for ref in used if ref.source_api == "suspend_d"} == {"resume"}
    assert {ref.source_record_id for ref in contrary if ref.source_api == "suspend_d"} == {
        "stop",
        "uncertain",
    }
    audit = result.details["market_evidence_audit"]
    assert audit["suspension_all"]["status_counts"] == {"false": 1, "true": 1, "unknown": 1}
    assert audit["suspension_used"]["status_counts"] == {"false": 1}
    assert audit["suspension_contrary"]["status_counts"] == {"true": 1, "unknown": 1}
    assert len(result.details["suspension_observations"]) == 3
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        initialize_tables(connection)
        save_results(connection, (result,), 128)
        assert fingerprints(record_rows(result)) == fingerprints(
            stored_rows(connection, inputs.run_id, 128)
        )
        relationships = {}
        for (payload,) in connection.execute("SELECT payload FROM research.lineage").fetchall():
            edge = json.loads(payload)
            relationships.setdefault(edge["evidence_id"], set()).add(edge["relationship"])
        assert all(len(kinds) == 1 for kinds in relationships.values())


@pytest.fixture(scope="module")
def rare_status_suspensions():
    """构造真实摘要排序不利于稀有状态的合成输入：至少24条false的摘要小于true和unknown。

    例：旧全局最小8行会全部选到false；新分层策略必须包含稀有停牌和未知状态。
    """
    role = "suspension_all"
    rare = tuple(
        (
            status,
            row(
                "suspend_d",
                index=index,
                record="rare-" + status,
                fields={"trade_date": "20260930", "suspend_type": kind},
            ),
        )
        for index, (status, kind) in enumerate((("true", "S"), ("unknown", "X")), start=256)
    )
    boundary = min(
        fingerprint_leaf([role, status, row_fingerprint(source, SUSPENSION_FIELDS)])
        for status, source in rare
    )
    candidates = tuple(
        row(
            "suspend_d",
            index=index,
            record=f"common-{index}",
            fields={"trade_date": "20260930", "suspend_type": "R"},
        )
        for index in range(256)
    )
    common = tuple(
        ("false", source)
        for source in candidates
        if fingerprint_leaf([role, "false", row_fingerprint(source, SUSPENSION_FIELDS)]) < boundary
    )
    assert len(common) >= 24
    return common[:24] + rare


def test_status_stratification_matches_full_oracle_with_fixed_eight_row_state(
    rare_status_suspensions,
):
    """稀有状态被全局哈希挤出的输入按状态保留，余量严格取全局最小摘要，重排及重复均确定。"""
    sources = rare_status_suspensions + rare_status_suspensions[:3]
    leaves = {
        fingerprint_leaf(
            ["suspension_all", status, row_fingerprint(source, SUSPENSION_FIELDS)]
        ): status
        for status, source in sources
    }
    assert {leaves[key] for key in sorted(leaves)[:8]} == {"false"}
    required = {
        min(key for key in leaves if leaves[key] == status) for status in set(leaves.values())
    }
    expected = required | set(sorted(leaves.keys() - required)[: 8 - len(required)])
    summaries = []
    for ordered in (sources, tuple(reversed(sources)), sources[7:] + sources[:7]):
        collector = MarketEvidence("suspension_all", SUSPENSION_FIELDS)
        for status, source in ordered:
            collector.add(source, status)
            assert len(collector.samples) == len(collector.sample_statuses) <= 8
        assert set(collector.samples) == expected
        summary = collector.summary()
        assert summary["total_rows"] == len(sources)
        assert summary["status_counts"] == dict(Counter(status for status, _ in sources))
        assert summary["sampled_status_counts"] == {"false": 6, "true": 1, "unknown": 1}
        assert summary["sample_count"] == 8 and summary["truncated"] is True
        assert summary["sampling_policy"]["method"] == "status_stratified_lowest_row_sha256"
        assert summary["sampling_policy"]["minimum_rows_per_status"] == 1
        assert summary["sampling_policy"]["remaining_slots"] == "lowest_row_sha256"
        for sample in summary["samples"]:
            assert sample["status"] == leaves[sample["row_fingerprint"]]
        summaries.append(summary)
    assert summaries[0] == summaries[1] == summaries[2]


def test_market_suspension_details_keep_rare_states_without_cross_role_references(
    rare_status_suspensions,
):
    """真实build_market中大量复牌和稀有停牌/未知并存；8条明细覆盖三态，正反引用仍互斥。"""
    sources = tuple(row(index=index, record=f"daily-{index}") for index in range(258)) + tuple(
        source for _, source in rare_status_suspensions
    )
    result = build_market(market_inputs(258, sources))[0]
    observations = result.details["suspension_observations"]
    assert len(observations) == 8
    assert {item["suspend_type"] for item in observations} == {"R", "S", "X"}
    assert result.details["market_evidence_audit"]["suspension_all"]["total_rows"] == 26
    assert result.details["denominator_count"] == 256
    used = {ref.to_json() for ref in result.envelope.evidence_refs}
    contrary = {ref.to_json() for ref in result.envelope.contrary_evidence}
    assert not used & contrary
    assert (
        build_market(market_inputs(258, tuple(reversed(sources))))[0].to_json() == result.to_json()
    )


def test_status_sample_budget_exhaustion_fails_closed():
    """8个可展示状态恰好可覆盖，第9个状态以固定容量错误拒绝，不能静默删掉某一状态。"""
    collector = MarketEvidence("synthetic", ("close",))
    for index in range(8):
        collector.add(row(record=str(index)), f"state{index}")
    assert collector.summary()["sample_count"] == 8
    with pytest.raises(ValueError, match="^research_market_sample_status_capacity_exceeded$"):
        collector.add(row(record="ninth"), "state8")


def test_status_without_source_fields_is_counted_but_cannot_invent_a_sample():
    """没有候选引用的状态仍计入全量摘要，抽样策略明确只为有引用的行分配名额。"""
    collector = MarketEvidence("synthetic", ("absent",))
    collector.add(row(), "unknown")
    summary = collector.summary()
    assert summary["status_counts"] == {"unknown": 1}
    assert summary["sampled_status_counts"] == {} and summary["sample_count"] == 0
    assert summary["sampling_policy"]["eligible_rows"] == "rows_with_candidate_references"
