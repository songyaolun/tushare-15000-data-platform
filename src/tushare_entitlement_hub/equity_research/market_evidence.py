"""市场全量计算的常量空间审计与有界代表证据，不截断指标输入。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..product_foundation.contracts import EvidenceRef
from ..store import encode
from .common import unique_evidence
from .context_aggregation import fingerprint_leaf, row_fingerprint
from .context_digest import DIGEST_ALGORITHM, MultisetDigest
from .models import ResearchRow

MARKET_EVIDENCE_VERSION = "market-evidence/v1"
MAXIMUM_SAMPLE_ROWS = 8
MAXIMUM_REFERENCE_BYTES = 4096
MAXIMUM_OBSERVATION_BYTES = 4096
DAILY_FIELDS = ("trade_date", "close", "vol", "amount", "pct_chg")
LIMIT_FIELDS = ("down_limit", "trade_date")
SUSPENSION_FIELDS = ("trade_date", "suspend_type", "suspend_timing")


class MarketEvidence:
    """逐角色累积全部行及状态，仅保留最多8个确定性样本，不保存全量哈希集合。"""

    def __init__(
        self,
        role: str,
        fields: tuple[str, ...],
        *,
        sample: bool = True,
        non_null_fields: bool = False,
    ) -> None:
        """指定角色、关键字段和是否展示引用，初始化固定大小的多重集摘要。

        例如MarketEvidence('daily_input', DAILY_FIELDS, sample=False)只审计原始输入；
        默认展示按行摘要排序的8个不同样本，重复行仍逐次参与总数与摘要。
        non_null_fields=True沿用消费层仅引用非空字段的规则，无字段行仍计入摘要。
        """
        self.role = role
        self.fields = fields
        self.non_null_fields = non_null_fields
        self.maximum_samples = MAXIMUM_SAMPLE_ROWS if sample else 0
        self.rows_digest = MultisetDigest()
        self.status_counts: Counter[str] = Counter()
        self.candidate_reference_count = 0
        self.source_as_of: dict[str, str] = {}
        self.samples: dict[str, tuple[ResearchRow, tuple[EvidenceRef, ...]]] = {}

    def add(self, row: ResearchRow, status: str) -> None:
        """散列一条参与行的来源、时间、原值、证书及状态，并更新全量引用出现次数。

        例如加入第9条后total_rows为9，样本最多8条；未抽样行变化也改变摘要。
        引用计数按出现次数而非去重数；所有候选引用均校验身份及4096字节上限。
        sample=False时不生成引用，用于保留全部输入（含未被指标选择的行）的审计。
        """
        fingerprint = fingerprint_leaf([self.role, status, row_fingerprint(row, self.fields)])
        self.rows_digest.add(fingerprint)
        self.status_counts[status] += 1
        if not self.maximum_samples:
            return
        references = row.evidence(
            *(
                field
                for field in self.fields
                if not self.non_null_fields or row.raw.get(field) is not None
            )
        )
        self.candidate_reference_count += len(references)
        for reference in references:
            if len(reference.to_json().encode("utf-8")) > MAXIMUM_REFERENCE_BYTES:
                raise ValueError("research_market_evidence_capacity_exceeded")
        if not references:
            return
        if row.availability.data_as_of is not None:
            self.source_as_of[row.source_api] = max(
                row.availability.data_as_of, self.source_as_of.get(row.source_api, "")
            )
        self.samples[fingerprint] = row, references
        if len(self.samples) > self.maximum_samples:
            del self.samples[max(self.samples)]

    def references(self) -> tuple[EvidenceRef, ...]:
        """返回已排序、去重的有界引用，保留原始定位字段，不改写或缩短来源身份。

        例如8条日线最多40个字段引用；同一来源重复出现仍只返回一次引用。
        """
        return unique_evidence(ref for _, refs in self.samples.values() for ref in refs)

    def summary(self) -> dict[str, Any]:
        """输出全量计数、状态、摘要算法及可回查样本清单，不输出原始大字段。

        例如1000条输入输出total_rows=1000、sample_count<=8、truncated=true；
        full_fingerprint覆盖未入样行和重复次数，statistics_scope明确为全部角色行。
        """
        return {
            "version": MARKET_EVIDENCE_VERSION,
            "total_rows": self.rows_digest.count,
            "candidate_reference_count": self.candidate_reference_count,
            "candidate_reference_count_basis": "occurrences_before_deduplication",
            "full_fingerprint": self.rows_digest.hexdigest(),
            "fingerprint_algorithm": DIGEST_ALGORITHM,
            "status_counts": dict(sorted(self.status_counts.items())),
            "sampling_policy": {
                "method": "lowest_row_sha256" if self.maximum_samples else "fingerprint_only",
                "maximum_rows": self.maximum_samples,
                "maximum_references": self.maximum_samples * len(self.fields),
                "maximum_reference_bytes": MAXIMUM_REFERENCE_BYTES,
                "statistics_scope": "all_role_rows",
                "fields": self.fields,
                "non_null_fields_only": self.non_null_fields,
            },
            "sample_count": len(self.samples),
            "sample_reference_count": len(self.references()),
            "truncated": self.rows_digest.count > len(self.samples),
            "samples": [
                {
                    "source_api": row.source_api,
                    "source_record_id": row.availability.source_record_id,
                    "row_fingerprint": fingerprint,
                }
                for fingerprint, (row, _) in sorted(self.samples.items())
            ],
        }

    def suspension_observations(self) -> list[dict[str, Any]]:
        """仅为已选停复牌样本生成原有展示字段，逐条执行4096字节容量门禁。

        例如1000条停复牌仅展示最多8条；suspend_timing为空仍表示full_day，缺失为unknown。
        完整事件数量和状态在summary中保留，超长原值不被静默截断。
        """
        observations = []
        for _, (row, _) in sorted(self.samples.items()):
            timing = row.text("suspend_timing")
            scope = (
                "intraday"
                if timing is not None
                else "full_day"
                if ("suspend_timing" in row.raw and row.raw["suspend_timing"] in (None, ""))
                else "unknown"
            )
            observation = {
                "entity_id": row.entity_id,
                "trade_date": row.availability.data_as_of,
                "source_record_id": row.availability.source_record_id,
                "suspend_type": row.text("suspend_type"),
                "suspend_timing": timing,
                "scope": scope,
                "available_at": row.availability.available_at,
            }
            if len(encode(observation).encode("utf-8")) > MAXIMUM_OBSERVATION_BYTES:
                raise ValueError("research_market_observation_capacity_exceeded")
            observations.append(observation)
        return observations
