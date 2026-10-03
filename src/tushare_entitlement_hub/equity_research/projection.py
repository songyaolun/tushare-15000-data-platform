"""从一次完整公开序列化产生不可变投影文本，内部复用，不放宽外部JSON入口。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..store import encode
from .models import ResearchResult

RESEARCH_TABLES = ("result", "feature", "contribution", "evidence", "gate", "signal", "lineage")


def _snapshot_json(value: Any) -> str:
    """仅编码已通过Model.to_json检查的本地快照及固定投影字段，不接受外部输入。

    例：快照中的一个已校验指标编码为与原Metric.to_json相同的字节，无需再次assert_public。
    调用方必须先完整校验根；公开insert_rows和数据库读取仍分别使用encode检查外部数据。
    """
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _snapshot_digest(value: Any) -> str:
    """为已校验快照派生稳定主键；例：[result_id,'gate','known']与原digest结果完全相同。"""
    return hashlib.sha256(_snapshot_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncodedRecord:
    """内部记录使用不可变的规范JSON文本；不把可变字典或未经校验的用户文本标为可信。"""

    table: str
    record_id: str
    run_id: str
    result_id: str
    payload: str

    def decoded(self) -> dict[str, Any]:
        """返回独立的公开记录副本；例：修改返回的payload不会污染后续投影。"""
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "result_id": self.result_id,
            "payload": json.loads(self.payload),
        }

    def fingerprint(self) -> int:
        """增量散列规范完整行，不重新解析或复制大payload，元数据仍经公开安全检查。

        例：结果等于int(digest(self.decoded()),16)，包括四个字段且不忽略物理主键变化。
        payload只能来自根公开序列化、公开encode，或读取边界的encode(json.loads(...))。
        """
        metadata = encode(
            {"record_id": self.record_id, "result_id": self.result_id, "run_id": self.run_id}
        )
        hashed = hashlib.sha256(b'{"payload":')
        hashed.update(self.payload.encode("utf-8"))
        hashed.update(b",")
        hashed.update(metadata[1:].encode("utf-8"))
        return int(hashed.hexdigest(), 16)


def project_result(result: ResearchResult) -> Iterator[EncodedRecord]:
    """每结果只调用一次公开to_json，从独立快照流式产生全部七表记录，不缓存全结果集合。

    例：正反证共用引用时仍产生两条lineage和一条evidence；旧主键、顺序和载荷均保留。
    根文本已递归检查敏感字段及非有限值；子投影只引用该快照和本模块固定字段。
    """
    serialized = result.to_json()
    payload = json.loads(serialized)
    envelope = payload["envelope"]
    identity, run_id = result.envelope.result_id, envelope["run_id"]

    def entry(table: str, key: Any, value: dict[str, Any]) -> EncodedRecord:
        """包装一个已校验子投影；例：gate/known生成与旧版相同的结果专属主键。"""
        return EncodedRecord(
            table, _snapshot_digest([identity, table, key]), run_id, identity, _snapshot_json(value)
        )

    yield EncodedRecord("result", identity, run_id, identity, serialized)
    for table, items, key in (
        ("feature", payload["metrics"], "name"),
        ("contribution", envelope["contributions"], "feature_id"),
        ("gate", envelope["quality_gates"], "gate_id"),
    ):
        for item in items:
            yield entry(table, item[key], item)
    for name in result.predicate_results:
        yield entry(
            "signal", name, {"predicate_id": name, "result": payload["predicate_results"][name]}
        )
    references = {}
    for relationship, evidence in (
        ("used", envelope["evidence_refs"]),
        ("contrary", envelope["contrary_evidence"]),
    ):
        for reference in evidence:
            evidence_id = _snapshot_digest(reference)
            references[evidence_id] = reference
            yield entry(
                "lineage",
                [relationship, evidence_id],
                {
                    "relationship": relationship,
                    "evidence_id": _snapshot_digest([identity, "evidence", evidence_id]),
                    "parent_result_id": None,
                },
            )
    for key, reference in references.items():
        yield entry("evidence", key, reference)
    for parent in payload["parent_result_ids"]:
        yield entry(
            "lineage",
            ["parent", parent],
            {"relationship": "parent", "evidence_id": None, "parent_result_id": parent},
        )


def accumulate_records(
    rows: Iterable[EncodedRecord], totals: dict[str, list[int]] | None = None
) -> dict[str, list[int]]:
    """复用规范文本累计与旧fingerprints相同的按表计数/模2**256摘要，内存与表数相关。

    例：先累计物理result，再补六个子表，结果与一次累计全部七表相同；顺序不影响结果。
    """
    totals = {table: [0, 0] for table in RESEARCH_TABLES} if totals is None else totals
    for row in rows:
        totals[row.table][0] += 1
        totals[row.table][1] = (totals[row.table][1] + row.fingerprint()) % (2**256)
    return totals
