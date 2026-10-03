"""高基数上下文的全量统计、指纹及有界证据抽样；不保留逐组结果副本。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

from ..product_foundation.contracts import EvidenceRef, QualityGateResult, Truth
from ..store import digest, encode
from .common import unique_evidence
from .context_digest import DIGEST_ALGORITHM, DiskKeySet, MultisetDigest
from .models import ResearchRow

AGGREGATION_VERSION = "context-aggregate-v2"
MAXIMUM_DETAILED_ROWS = 64
SAMPLES_PER_STRATUM = 4
MAXIMUM_REFERENCE_BYTES = 4096
MAXIMUM_DIMENSION_VALUES = 16
MAXIMUM_DIMENSION_BYTES = 128
METADATA_FIELDS = ("market", "publisher", "index_type", "weight_rule")
QUOTE_FIELDS = ("close", "vol", "amount")


def fingerprint_leaf(value: Any) -> str:
    """只散列明确投影的上下文，不输出原值；非有限源数字也有稳定指纹。

    输入{'close': float('nan')}返回64位SHA256；NaN与字符串'NaN'的指纹不同。
    JSON只用于散列，允许记录被数值门禁拒绝的NaN/Infinity，不将它们持久化为事实。
    """
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def row_fingerprint(row: ResearchRow, fields: tuple[str, ...]) -> str:
    """绑定参与行的身份、时间、关键原值、质量及数值证书，不复制原始大字段到输出。

    输入同一来源行，修改未入样本的pb、证书或公开时间均改变指纹；重复行不去重。
    """
    identity = (
        "source_api",
        "source_record_id",
        "business_group_key",
        "business_variant_key",
        "contract_hash",
        "quality_tier",
        "is_quarantined",
        "source_occurrences",
        "price_type",
        "market_activity_status",
        "price_observation_status",
        "source_units",
        "standard_units",
        "conversion_version",
    )
    return fingerprint_leaf(
        {
            "entity_id": row.entity_id,
            "api": row.source_api,
            "raw": {field: row.raw.get(field) for field in (*identity, *fields)},
            "present_fields": sorted(field for field in fields if field in row.raw),
            "availability": row.availability.to_dict(),
            "numeric": {
                field: row.numeric[field].to_dict() for field in fields if field in row.numeric
            },
        }
    )


def context_status(states: set[str]) -> str:
    """沿用上下文的保守状态优先级；例：observed和unknown合并仍为unknown。

    冲突优先于未知，其次历史受限、已观察、不适用；没有组时返回unknown。
    """
    return next(
        (
            status
            for status in (
                "conflict",
                "unknown",
                "not_enabled_due_to_history",
                "observed",
                "not_applicable",
            )
            if status in states
        ),
        "unknown",
    )


class ContextAggregate(AbstractContextManager["ContextAggregate"]):
    """逐组累积全量审计，最多5状态×2来源角色×4个证据和一个软门禁。"""

    def __init__(
        self,
        api: str,
        rows: tuple[ResearchRow, ...],
        usable: tuple[ResearchRow, ...],
        fields: tuple[str, ...],
        analysis_as_of: str,
    ) -> None:
        """扫描所有输入行并计算固定字段覆盖，不先抽样；例如12000行全部进入行指纹。

        rows为消费收到的保留行，usable为质量/时点过滤后行；两者计数分别披露。
        摘要使用固定大小累计状态；本源字段、精确去重及分类基数使用可溢写临时集合。
        调用者通过with管理生命周期，数值覆盖统计仅针对usable行。
        """
        self.api = api
        self.analysis_as_of = analysis_as_of
        self.total_rows = len(rows)
        self.usable_rows = len(usable)
        row_digest = MultisetDigest()
        for row in rows:
            row_digest.add(row_fingerprint(row, fields))
        self.rows_fingerprint = row_digest.hexdigest()
        self.field_coverage = {
            field: {
                "present_rows": sum(row.raw.get(field) is not None for row in usable),
                "certified_rows": sum(row.number(field) is not None for row in usable),
            }
            for field in fields
        }
        self.status_counts: Counter[str] = Counter()
        self.group_row_counts: Counter[str] = Counter()
        self.reasons: set[str] = set()
        self.group_digest = MultisetDigest()
        self.reference_digest = MultisetDigest()
        self.samples: dict[tuple[str, str], dict[tuple[str, str], EvidenceRef]] = {}
        self.metadata: dict[str, Counter[str]] = {field: Counter() for field in METADATA_FIELDS}
        self.metadata_total: Counter[str] = Counter()
        self.metadata_distinct: Counter[str] = Counter()
        self.metadata_missing: Counter[str] = Counter()
        self.metadata_oversized: Counter[str] = Counter()
        self.quote_checks: dict[str, Counter[str]] = {field: Counter() for field in QUOTE_FIELDS}
        self._keys = DiskKeySet()
        try:
            for row in usable:
                for field in fields:
                    if row.raw.get(field) is not None:
                        self._keys.add(
                            "local", encode([str(row.raw["source_record_id"]), field]).encode()
                        )
        except BaseException:
            self._keys.close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """正常返回或异常离开with时均释放临时集合，异常继续向上传播；例如超限失败不留临时库。"""
        self._keys.close()

    def _metadata_value(self, field: str, value: str) -> None:
        """只保留字典序最小的16个分类计数，精确总量和基数在有界计数器及磁盘集合中维护。

        例如先遇到Z再遇到A，最终仍显示最小分类的完整频次；其他分类计入other_groups。
        被淘汰的值不会重新进入单调降低的前16阈值，因此不需要保留其完整频次表。
        """
        self.metadata_total[field] += 1
        self.metadata_distinct[field] += self._keys.add("metadata/" + field, value.encode("utf-8"))
        counts = self.metadata[field]
        if value in counts:
            counts[value] += 1
        elif len(counts) < MAXIMUM_DIMENSION_VALUES or value < max(counts):
            if len(counts) == MAXIMUM_DIMENSION_VALUES:
                del counts[max(counts)]
            counts[value] = 1

    def add(self, key: tuple[Any, ...], row_count: int, observation: dict[str, Any]) -> None:
        """吸收一个最新组的全部业务判断，再释放其观察、门禁和全量引用。

        输入idx_factor_pro的冲突组，增加冲突计数和全量组摘要；样本按状态及本源/对照
        分层选择最小SHA256，遍历顺序不改变选择，非样本组仍进入统计及指纹。
        """
        status = observation["status"]
        self.status_counts[status] += 1
        self.group_row_counts[status] += row_count
        self.reasons.update(observation["reason_codes"])
        self.group_digest.add(
            fingerprint_leaf(
                {
                    "key": key,
                    "rows": row_count,
                    "status": status,
                    "reasons": sorted(observation["reason_codes"]),
                    "facts": observation["facts"],
                }
            )
        )
        if self.api == "index_basic":
            for field in METADATA_FIELDS:
                value = observation["facts"].get(field)
                if not value:
                    self.metadata_missing[field] += 1
                elif len(value.encode("utf-8")) > MAXIMUM_DIMENSION_BYTES:
                    self.metadata_oversized[field] += 1
                else:
                    self._metadata_value(field, value)
        if self.api == "idx_factor_pro":
            checks = observation["facts"].get("quote_checks", {})
            for field in QUOTE_FIELDS:
                self.quote_checks[field][checks.get(field, {}).get("status", "unknown")] += 1
        for reference in observation["evidence_refs"]:
            serialized = reference.to_json()
            if len(serialized.encode("utf-8")) > MAXIMUM_REFERENCE_BYTES:
                raise ValueError("research_context_evidence_capacity_exceeded")
            fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            if self._keys.add("reference", serialized.encode("utf-8")):
                self.reference_digest.add(fingerprint)
            local = reference.source_api == self.api and any(
                self._keys.contains("local", encode([reference.source_record_id, field]).encode())
                for field in reference.fields
            )
            role = "source" if local else "supporting"
            sample = self.samples.setdefault((status, role), {})
            sample[(fingerprint, serialized)] = reference
            if len(sample) > SAMPLES_PER_STRATUM:
                del sample[max(sample)]

    def finish(self) -> dict[str, Any]:
        """输出固定维度统计、全量指纹、有界样本和单个聚合软门禁。

        例如12000个完整index_basic组得到observed、group_count=12000和至多4条本源
        样本；truncated明确说明引用抽样，统计和指纹从未抽样。类别超过16项进入other。
        """
        status = context_status(set(self.status_counts))
        reasons = tuple(sorted(self.reasons))
        if not reasons and status not in {"observed", "not_applicable"}:
            reasons = (
                "context_aggregate_unavailable"
                if self.status_counts
                else "context_accepted_rows_unavailable",
            )
        references = unique_evidence(
            ref for sample in self.samples.values() for ref in sample.values()
        )
        metadata = {}
        if self.api == "index_basic":
            for field, counts in self.metadata.items():
                selected = sorted(counts)[:MAXIMUM_DIMENSION_VALUES]
                metadata[field] = {
                    "available_groups": self.metadata_total[field] + self.metadata_oversized[field],
                    "missing_groups": self.metadata_missing[field],
                    "oversized_value_groups": self.metadata_oversized[field],
                    "distinct_values": self.metadata_distinct[field],
                    "values": [{"value": value, "groups": counts[value]} for value in selected],
                    "other_groups": self.metadata_total[field] - sum(counts.values()),
                    "value_limit": MAXIMUM_DIMENSION_VALUES,
                    "value_byte_limit": MAXIMUM_DIMENSION_BYTES,
                }
        group_fingerprint = self.group_digest.hexdigest()
        evidence_fingerprint = self.reference_digest.hexdigest()
        summary = {
            "version": AGGREGATION_VERSION,
            "fingerprint_algorithm": DIGEST_ALGORITHM,
            "total_rows": self.total_rows,
            "usable_rows": self.usable_rows,
            "rejected_rows": self.total_rows - self.usable_rows,
            "group_count": sum(self.status_counts.values()),
            "status_counts": dict(sorted(self.status_counts.items())),
            "rows_by_group_status": dict(sorted(self.group_row_counts.items())),
            "field_coverage_scope": "usable_rows",
            "field_coverage": self.field_coverage,
            "metadata_availability": metadata,
            "quote_check_counts": {
                field: dict(sorted(counts.items())) for field, counts in self.quote_checks.items()
            }
            if self.api == "idx_factor_pro"
            else {},
            "rows_fingerprint": self.rows_fingerprint,
            "groups_fingerprint": group_fingerprint,
            "evidence_fingerprint": evidence_fingerprint,
            "full_fingerprint": digest(
                [
                    AGGREGATION_VERSION,
                    self.api,
                    self.analysis_as_of,
                    self.total_rows,
                    self.rows_fingerprint,
                    group_fingerprint,
                    evidence_fingerprint,
                ]
            ),
            "sampling_policy": {
                "version": "sha256-min-per-status-and-source-role-v1",
                "per_stratum_limit": SAMPLES_PER_STRATUM,
                "maximum_samples": 5 * 2 * SAMPLES_PER_STRATUM,
                "maximum_reference_bytes": MAXIMUM_REFERENCE_BYTES,
                "statistics_scope": "all_rows_and_all_latest_groups",
                "source_role": "retained_source_fields",
                "supporting_role": "comparison_or_certification_witness",
            },
            "candidate_reference_count": self.reference_digest.count,
            "sample_count": len(references),
            "truncated": self.reference_digest.count > len(references),
        }
        return {
            "status": status,
            "reason_codes": reasons,
            "facts": {"aggregation": summary},
            "evidence_refs": references,
            "gates": (
                QualityGateResult(
                    gate_id="context_aggregate",
                    result=Truth.FALSE
                    if status == "conflict"
                    else Truth.TRUE
                    if status in {"observed", "not_applicable"}
                    else Truth.UNKNOWN,
                    reason_codes=reasons if status not in {"observed", "not_applicable"} else (),
                    hard=False,
                    evidence_refs=references,
                ),
            ),
        }
