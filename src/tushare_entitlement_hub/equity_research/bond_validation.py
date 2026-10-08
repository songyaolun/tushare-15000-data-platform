"""P06市场债券聚合的持久化审计契约；只检查有界摘要和样本，不重新扫描来源。"""

from __future__ import annotations

import re
from typing import Any

from ..product_foundation.contracts import DecisionEnvelope, EvidenceRef
from ..standard import normalize_period
from ..store import digest, encode
from .bond_aggregation import (
    AUDIT_FIELDS,
    BOND_AGGREGATION_VERSION,
    MAXIMUM_DETAILED_BOND_ROWS,
    MAXIMUM_GROUP_ROWS,
    NUMERIC_FIELDS,
    RAW_DIMENSIONS,
    bond_audit_fingerprint,
    bond_full_fingerprint,
)
from .consumption_registry import ConsumptionRegistration
from .context_aggregation import MAXIMUM_REFERENCE_BYTES
from .context_digest import DIGEST_ALGORITHM

AUDIT_KEYS = frozenset(
    "version fingerprint_algorithm total_rows usable_rows rejected_rows group_count "
    "status_counts rows_by_group_status field_coverage_scope field_coverage "
    "metadata_availability quote_check_counts rows_fingerprint groups_fingerprint "
    "evidence_fingerprint full_fingerprint sampling_policy candidate_reference_count "
    "sample_count truncated business_statistics group_reason_codes audited_fields "
    "source_as_of sample_strata audit_fingerprint".split()
)
SAMPLING_POLICY = {
    "version": "sha256-min-per-status-and-source-role-v1",
    "per_stratum_limit": 1,
    "maximum_samples": 10,
    "maximum_reference_bytes": MAXIMUM_REFERENCE_BYTES,
    "statistics_scope": "all_rows_and_all_latest_groups",
    "source_role": "retained_source_fields",
    "supporting_role": "comparison_or_certification_witness",
}
BOND_STATUSES = frozenset({"observed", "unknown", "conflict"})


def _require(valid: bool) -> None:
    """将聚合契约失败统一为安全原因码；例：False抛ValueError，True返回None。"""
    if not valid:
        raise ValueError("research_bond_aggregation_audit_invalid")


def _counts(summary: dict[str, Any], record: dict[str, Any]) -> None:
    """核对行数、组状态和原字段质量的完整分母，拒绝布尔冒充计数及抽样分母。

    例：total=9、usable=8、rejected=1且记录行数8可通过；usable被改为7时拒绝。
    """
    names = (
        "total_rows",
        "usable_rows",
        "rejected_rows",
        "group_count",
        "candidate_reference_count",
        "sample_count",
    )
    _require(all(type(summary[name]) is int and summary[name] >= 0 for name in names))
    _require(summary["total_rows"] > MAXIMUM_DETAILED_BOND_ROWS)
    _require(summary["usable_rows"] == record["retained_row_count"])
    _require(summary["total_rows"] == summary["usable_rows"] + summary["rejected_rows"])
    states, rows = summary["status_counts"], summary["rows_by_group_status"]
    _require(type(states) is dict and type(rows) is dict)
    _require(bool(states) and set(states) <= BOND_STATUSES and set(states) == set(rows))
    _require(all(type(value) is int and value > 0 for value in (*states.values(), *rows.values())))
    _require(sum(states.values()) == summary["group_count"])
    _require(sum(rows.values()) == summary["total_rows"])
    _require(all(states[key] <= rows[key] <= MAXIMUM_GROUP_ROWS * states[key] for key in states))
    expected = {
        field: {key: values[key] for key in ("present_rows", "certified_rows")}
        for field, values in record["field_quality"].items()
    }
    _require(summary["field_coverage_scope"] == "usable_rows")
    _require(encode(summary["field_coverage"]) == encode(expected))


def _samples(
    summary: dict[str, Any], record: dict[str, Any], refs: tuple[EvidenceRef, ...]
) -> None:
    """把每个抽样层绑定到记录中的真实引用，检查层唯一、状态、来源角色、数量和容量。

    例：两个层引用同一旁证时sample_count仍为1；删除一个引用或伪造层摘要均被拒绝。
    """
    _require(encode(summary["sampling_policy"]) == encode(SAMPLING_POLICY))
    references = {digest(ref.to_dict()): ref for ref in refs}
    _require(len(references) == len(refs) == summary["sample_count"] <= 10)
    _require(all(len(ref.to_json().encode("utf-8")) <= MAXIMUM_REFERENCE_BYTES for ref in refs))
    _require(summary["candidate_reference_count"] >= len(refs))
    _require(type(summary["truncated"]) is bool)
    _require(summary["truncated"] == (summary["candidate_reference_count"] > len(refs)))
    strata = summary["sample_strata"]
    _require(type(strata) is list and len(strata) <= 10)
    keys, sampled = [], set()
    for sample in strata:
        _require(
            type(sample) is dict
            and set(sample) == {"status", "source_role", "reference_fingerprint"}
        )
        status, role, fingerprint = (
            sample["status"],
            sample["source_role"],
            sample["reference_fingerprint"],
        )
        _require(type(status) is str and status in summary["status_counts"])
        _require(type(fingerprint) is str and fingerprint in references)
        reference = references[fingerprint]
        local = reference.source_api == record["api"] and any(
            field in record["field_quality"] and record["field_quality"][field]["present_rows"] > 0
            for field in reference.fields
        )
        _require(role == ("source" if local else "supporting"))
        keys.append((status, role))
        sampled.add(fingerprint)
    _require(keys == sorted(set(keys)))
    _require(sampled == set(references))


def validate_bond_aggregation(
    record: dict[str, Any],
    registration: ConsumptionRegistration,
    envelope: DecisionEnvelope,
    refs: tuple[EvidenceRef, ...],
) -> None:
    """校验市场债券高基数记录的完整审计及两条软门禁，低基数精确路径保持不变。

    例：9行记录删除aggregation或清空gates会拒绝；9行全被门禁拒绝时仍从信封门禁
    识别必需聚合。摘要只验证持久化自洽性，不用抽样引用冒充参考库全量行验证。
    """
    api = registration.spec.api
    gate_id = f"consumption/{api}/bond_market_aggregate"
    envelope_checks = [check for check in envelope.quality_gates if check.gate_id == gate_id]
    facts: Any = record.get("facts")
    _require(type(facts) is dict)
    if (
        record["retained_row_count"] <= MAXIMUM_DETAILED_BOND_ROWS
        and "aggregation" not in facts
        and not envelope_checks
    ):
        return
    summary = facts.get("aggregation")
    _require(type(summary) is dict and set(summary) == AUDIT_KEYS)
    _require(summary["version"] == BOND_AGGREGATION_VERSION)
    _require(summary["fingerprint_algorithm"] == DIGEST_ALGORITHM)
    fields = sorted(
        set(
            registration.spec.selected_fields
            + RAW_DIMENSIONS[api]
            + NUMERIC_FIELDS[api]
            + AUDIT_FIELDS
        )
    )
    _require(summary["audited_fields"] == fields)
    for name in (
        "rows_fingerprint",
        "groups_fingerprint",
        "evidence_fingerprint",
        "full_fingerprint",
        "audit_fingerprint",
    ):
        _require(
            type(summary[name]) is str and re.fullmatch(r"[0-9a-f]{64}", summary[name]) is not None
        )
    _require(
        summary["full_fingerprint"] == bond_full_fingerprint(api, envelope.analysis_as_of, summary)
    )
    _require(summary["audit_fingerprint"] == bond_audit_fingerprint(summary))
    _counts(summary, record)
    _samples(summary, record, refs)
    periods = summary["source_as_of"]
    _require(type(periods) is dict)
    for source, period in periods.items():
        _require(type(period) is str)
        try:
            normalized = normalize_period(period)
        except ValueError:
            raise ValueError("research_bond_aggregation_audit_invalid") from None
        _require(normalized == period)
        _require(source in envelope.source_as_of and period <= envelope.source_as_of[source])
    checks = record["gates"]
    expected_ids = {gate_id, f"consumption/{api}/execution_status"}
    _require(len(checks) == 2 and {check["gate_id"] for check in checks} == expected_ids)
    _require(len(envelope_checks) == 1)
    _require(sum(check.gate_id in expected_ids for check in envelope.quality_gates) == 2)
    truth = (
        "true"
        if record["status"] == "observed"
        else "false"
        if record["status"] == "conflict"
        else "unknown"
    )
    expected_refs = sorted(encode(ref.to_dict()) for ref in refs)
    for check in checks:
        _require(check["hard"] is False and check["result"] == truth)
        _require(sorted(encode(ref) for ref in check["evidence_refs"]) == expected_refs)
        if check["gate_id"] != gate_id:
            _require(
                check["reason_codes"]
                == ([] if truth == "true" else ["consumption_" + record["status"]])
            )
        else:
            _require(all(reason in record["reason_codes"] for reason in check["reason_codes"]))
