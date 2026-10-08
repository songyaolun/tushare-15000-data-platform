"""债券聚合审计在JSON和持久化边界上的篡改拒绝回归。"""

from dataclasses import replace

import duckdb
import pytest
from test_research_bond_market import COUNTS, bond_market_inputs

from tushare_entitlement_hub.equity_research.bond_aggregation import (
    bond_audit_fingerprint,
    bond_full_fingerprint,
)
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.equity_research.storage import initialize_tables, save_results
from tushare_entitlement_hub.product_foundation.contracts import QualityGateResult
from tushare_entitlement_hub.store import encode


@pytest.fixture(scope="module")
def bond_result():
    """创建九来源各9行的真实聚合结果供独立JSON副本使用；例：cb_daily含9行与两个样本。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    return attach_consumption(context, PRODUCT_ALIASES["P06"], ())[0]


@pytest.mark.parametrize(
    "damage", ["missing_aggregation", "usable_rows", "rows_fingerprint", "sample_strata", "gates"]
)
def test_rebuilt_bond_result_rejects_missing_or_tampered_audit(bond_result, damage):
    """删审计、改计数/摘要/样本或清空门禁后，即便旧引用及字段质量不变，JSON重建校验仍须拒绝。"""
    details = bond_result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_daily")
    summary = record["facts"]["aggregation"]
    if damage == "missing_aggregation":
        del record["facts"]["aggregation"]
    elif damage == "usable_rows":
        summary["usable_rows"] -= 1
    elif damage == "rows_fingerprint":
        summary["rows_fingerprint"] = "0" * 64
    elif damage == "sample_strata":
        summary["sample_strata"] = []
    else:
        record["gates"] = []
    rebuilt = ResearchResult.from_json(replace(bond_result, details_json=encode(details)).to_json())
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(rebuilt)


@pytest.mark.parametrize(
    "path,value",
    [
        (("version",), "bond-market-aggregate/v1"),
        (("fingerprint_algorithm",), "unverified"),
        (("audited_fields",), []),
        (("rows_fingerprint",), True),
        (("rows_fingerprint",), "bad"),
        (("group_count",), True),
        (("candidate_reference_count",), -1),
        (("total_rows",), 8),
        (("usable_rows",), 8),
        (("rejected_rows",), 1),
        (("status_counts",), []),
        (("rows_by_group_status",), []),
        (("status_counts",), {}),
        (("status_counts",), {"invalid": 9}),
        (("rows_by_group_status",), {"unknown": 9}),
        (("status_counts",), {"observed": True}),
        (("status_counts",), {"observed": 0}),
        (("group_count",), 8),
        (("rows_by_group_status",), {"observed": 8}),
        (("status_counts",), {"observed": 10}),
        (("field_coverage_scope",), "sample"),
        (("field_coverage", "close", "certified_rows"), 1),
        (("sampling_policy", "per_stratum_limit"), True),
        (("sample_count",), 9),
        (("candidate_reference_count",), 0),
        (("truncated",), 1),
        (("truncated",), False),
        (("sample_strata",), {}),
        (("sample_strata", 0), {}),
        (("sample_strata", 0, "status"), "unknown"),
        (("sample_strata", 0, "reference_fingerprint"), "0" * 64),
        (("sample_strata", 0, "source_role"), "invalid"),
        (("source_as_of",), []),
        (("source_as_of", "cb_basic"), None),
        (("source_as_of", "cb_basic"), "not-a-period"),
        (("source_as_of", "cb_basic"), "20261002"),
        (("source_as_of", "cb_basic"), "2026-10-04"),
        (("source_as_of", "unexpected"), "2026-10-02"),
    ],
)
def test_recomputed_checksum_cannot_bypass_audit_structure(bond_result, path, value):
    """即使重新计算两层摘要，非法版本、计数、覆盖分母、样本或来源期间仍按结构契约拒绝。"""
    details = bond_result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_daily")
    summary = record["facts"]["aggregation"]
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    summary["full_fingerprint"] = bond_full_fingerprint(
        "cb_daily", bond_result.envelope.analysis_as_of, summary
    )
    summary["audit_fingerprint"] = bond_audit_fingerprint(summary)
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(replace(bond_result, details_json=encode(details)))


@pytest.mark.parametrize("state", ["observed", "unknown", "conflict", "rejected", "low"])
def test_valid_aggregate_and_low_cardinality_results_still_rebuild(state, tmp_path):
    """真实observed/unknown/conflict、全部拒绝行及低基数精确观察均可在512MB保存重建。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 8 if state == "low" else 9))
    sources = dict(context.rows_by_api)
    if state == "unknown":
        for api in ("cb_daily", "cb_factor_pro"):
            sources[api] = tuple(replace(row, numeric={}) for row in sources[api])
    elif state == "conflict":
        rows = list(sources["cb_daily"])
        rows[0] = replace(
            rows[0],
            numeric=rows[0].numeric | {"close": replace(rows[0].numeric["close"], value=120)},
        )
        sources["cb_daily"] = tuple(rows)
    elif state == "rejected":
        sources["cb_call"] = tuple(
            replace(row, availability=replace(row.availability, observed_at="2026-10-04T12:00:00Z"))
            for row in sources["cb_call"]
        )
    result = attach_consumption(replace(context, rows_by_api=sources), PRODUCT_ALIASES["P06"], ())[
        0
    ]
    api = "cb_call" if state == "rejected" else "cb_daily"
    record = next(item for item in result.details["consumption"] if item["api"] == api)
    expected = "observed" if state == "low" else "unknown" if state == "rejected" else state
    assert record["status"] == expected
    validate_result_consumption(result)
    with duckdb.connect(
        str(tmp_path / "audit.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        assert save_results(connection, (result,), 128)["result"] == 1
        rebuilt = ResearchResult.from_json(
            connection.execute("SELECT payload FROM research.result").fetchone()[0]
        )
    assert rebuilt.to_json() == result.to_json()
    validate_result_consumption(rebuilt)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_record",
        "duplicate_envelope",
        "duplicate_execution",
        "hard",
        "state",
        "references",
        "reason",
    ],
)
def test_aggregate_gates_must_match_record_and_envelope(bond_result, damage):
    """同步篡改记录与信封的门禁仍须拒绝，不能只检查门禁对象是否出现在信封集合中。"""
    details = bond_result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_daily")
    checks = record["gates"]
    if damage == "duplicate_record":
        checks.append(dict(checks[0]))
    elif damage == "hard":
        checks[0]["hard"] = True
    elif damage == "state":
        checks[0].update(result="unknown", reason_codes=["changed"])
    elif damage == "references":
        checks[0]["evidence_refs"] = []
    elif damage == "reason":
        checks[1]["reason_codes"] = ["changed"]
    changed = {check["gate_id"]: QualityGateResult.from_dict(check) for check in checks}
    envelope_checks = tuple(
        changed.get(check.gate_id, check) for check in bond_result.envelope.quality_gates
    )
    if damage in {"duplicate_envelope", "duplicate_execution"}:
        envelope_checks += (
            QualityGateResult.from_dict(checks[int(damage == "duplicate_execution")]),
        )
    broken = replace(
        bond_result,
        details_json=encode(details),
        envelope=replace(bond_result.envelope, quality_gates=envelope_checks),
    )
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(broken)


@pytest.mark.parametrize(
    "damage", ["missing_sample", "duplicate_stratum", "reversed_strata", "oversized_group"]
)
def test_resealed_sample_and_group_counts_remain_bounded(bond_result, damage):
    """重新封装摘要后，丢样本、重复/乱序抽样层或单组行数超上限仍拒绝。"""
    details = bond_result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_daily")
    summary = record["facts"]["aggregation"]
    if damage == "missing_sample":
        summary["sample_strata"].pop()
    elif damage == "duplicate_stratum":
        summary["sample_strata"].append(dict(summary["sample_strata"][0]))
    elif damage == "reversed_strata":
        summary["sample_strata"].reverse()
    else:
        summary.update(
            total_rows=5000,
            rejected_rows=4991,
            group_count=1,
            status_counts={"observed": 1},
            rows_by_group_status={"observed": 5000},
        )
    summary["audit_fingerprint"] = bond_audit_fingerprint(summary)
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(replace(bond_result, details_json=encode(details)))


def test_persisted_audit_removal_cannot_pass_reconstruction(bond_result, tmp_path):
    """落库后仅删聚合审计，保留行数、引用与原门禁，重新读取的结果必须在消费校验失败。"""
    details = bond_result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_daily")
    del record["facts"]["aggregation"]
    damaged = replace(bond_result, details_json=encode(details))
    with duckdb.connect(
        str(tmp_path / "damaged.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        save_results(connection, (bond_result,), 128)
        connection.execute(
            "UPDATE research.result SET payload=CAST(? AS JSON)", [damaged.to_json()]
        )
        restored = ResearchResult.from_json(
            connection.execute("SELECT payload FROM research.result").fetchone()[0]
        )
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(restored)


def test_rejected_rows_still_require_audit_when_usable_count_is_zero():
    """九行均被门禁拒绝后usable为0，删审计和记录门禁仍须被信封聚合门禁识别，不能退回低基数。"""
    context = bond_market_inputs({"cb_call": 9})
    rows = tuple(
        replace(row, availability=replace(row.availability, observed_at="2026-10-04T12:00:00Z"))
        for row in context.rows("cb_call")
    )
    context = replace(context, rows_by_api=context.rows_by_api | {"cb_call": rows})
    result = attach_consumption(context, PRODUCT_ALIASES["P06"], ())[0]
    validate_result_consumption(result)
    details = result.details
    record = next(item for item in details["consumption"] if item["api"] == "cb_call")
    assert record["retained_row_count"] == 0
    del record["facts"]["aggregation"]
    record["gates"] = []
    with pytest.raises(ValueError, match="^research_bond_aggregation_audit_invalid$"):
        validate_result_consumption(replace(result, details_json=encode(details)))
