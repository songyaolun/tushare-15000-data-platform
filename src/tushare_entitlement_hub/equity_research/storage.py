"""独立产品库research表，按批写入并重建派生记录校验，不修改reference。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import duckdb

from ..product_foundation.config import READ_BATCH_SIZE, validate_batch_size
from ..product_foundation.contracts import EvidenceRef
from ..reference_storage import file_checksum
from ..store import digest, encode
from .consumption import count_consumption, summarize_consumption, validate_result_consumption
from .consumption_registry import (
    CONSUMPTION_VERSION,
    registry_fingerprint,
    validate_consumption_registry,
)
from .models import PRODUCT_IDS, RULE_VERSION, ResearchConfig, ResearchResult
from .source_validation import verify_source_evidence

RESEARCH_TABLES = ("result", "feature", "contribution", "evidence", "gate", "signal", "lineage")
MAXIMUM_INSERT_ROWS = 128
MAXIMUM_INSERT_JSON_BYTES = 512 * 1024


def run_identity(
    reference_hash: str, cutoff: str, config: ResearchConfig, selection: tuple[str, ...]
) -> str:
    """将输入指纹、截止、完整配置和产品选择绑定为运行主键。

    例：相同输入重跑得到同一个64位摘要；阈值改变得到新批次。
    """
    return digest(
        [
            RULE_VERSION,
            CONSUMPTION_VERSION,
            registry_fingerprint(),
            reference_hash,
            cutoff,
            config.to_dict(),
            sorted(selection),
        ]
    )


def record_rows(result: ResearchResult) -> Iterator[tuple[str, dict[str, Any]]]:
    """将一个结果拆成关联记录，来源证据和上游结果分别保存血缘。

    例：一个指标、一个谓词得到feature和signal记录，unknown按原字符串保存。
    所有派生表键由结果与完整语义生成，重复运行不会产生第二份记录。
    """
    identity = result.envelope.result_id
    run_id = result.envelope.run_id
    yield (
        "result",
        {
            "record_id": identity,
            "run_id": run_id,
            "result_id": identity,
            "payload": result.to_dict(),
        },
    )
    payloads: list[tuple[str, Any, dict[str, Any]]] = []
    payloads.extend(("feature", item.name, item.to_dict()) for item in result.metrics)
    payloads.extend(
        ("contribution", item.feature_id, item.to_dict()) for item in result.envelope.contributions
    )
    payloads.extend(
        ("gate", item.gate_id, item.to_dict()) for item in result.envelope.quality_gates
    )
    payloads.extend(
        ("signal", name, {"predicate_id": name, "result": value.value})
        for name, value in result.predicate_results.items()
    )
    references: dict[str, EvidenceRef] = {}
    for relationship, evidence in (
        ("used", result.envelope.evidence_refs),
        ("contrary", result.envelope.contrary_evidence),
    ):
        for reference in evidence:
            evidence_id = digest(reference.to_dict())
            references[evidence_id] = reference
            payloads.append(
                (
                    "lineage",
                    [relationship, evidence_id],
                    {
                        "relationship": relationship,
                        "evidence_id": digest([identity, "evidence", evidence_id]),
                        "parent_result_id": None,
                    },
                )
            )
    payloads.extend(("evidence", key, reference.to_dict()) for key, reference in references.items())
    payloads.extend(
        (
            "lineage",
            ["parent", parent],
            {
                "relationship": "parent",
                "evidence_id": None,
                "parent_result_id": parent,
            },
        )
        for parent in result.parent_result_ids
    )
    for table, key, payload in payloads:
        yield (
            table,
            {
                "record_id": digest([identity, table, key]),
                "run_id": run_id,
                "result_id": identity,
                "payload": payload,
            },
        )


def fingerprints(rows: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, list[int]]:
    """按表累积密码摘要和行数，顺序无关且内存固定，供完整派生重建对账。

    例：交换两行顺序不变，修改一个unknown值会改变摘要。主键唯一性另由SQL检查。
    """
    totals = {table: [0, 0] for table in RESEARCH_TABLES}
    for table, row in rows:
        totals[table][0] += 1
        totals[table][1] = (totals[table][1] + int(digest(row), 16)) % (2**256)
    return totals


def initialize_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """幂等创建独立research表，不触碰product基础表；例：新库得到7张数据表和run表。

    所有子表持有显式result_id和run_id，完整校验执行反向引用检查。
    """
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("CREATE SCHEMA IF NOT EXISTS research")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS research.run "
        "(run_id VARCHAR PRIMARY KEY, payload JSON NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS research.metadata "
        "(key VARCHAR PRIMARY KEY, payload JSON NOT NULL)"
    )
    for table in RESEARCH_TABLES:
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS research."{table}" '
            "(record_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, "
            "result_id VARCHAR NOT NULL, payload JSON NOT NULL)"
        )


def insert_rows(
    connection: duckdb.DuckDBPyConnection, table: str, rows: list[dict[str, Any]]
) -> None:
    """按行数和JSON字节双重限批，以参数列写入，避免整批json_each反复解析。

    例：512条小记录至少分4次写入；两条各400KiB的JSON分别写入，单条超限记录独占
    一批且不截断。表名受白名单控制，主键与载荷参数分离，冲突拒绝而非覆盖。
    """
    if table not in RESEARCH_TABLES:
        raise ValueError("unknown_research_table")
    columns: list[list[str]] = [[], [], [], []]
    pending_bytes = 0

    def flush() -> None:
        """写入一个有界参数批并立即释放缓冲；例：两行对应四个等长参数列。"""
        connection.execute(
            f'INSERT INTO research."{table}" '
            "SELECT unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), "
            "unnest(?::VARCHAR[]), CAST(unnest(?::VARCHAR[]) AS JSON)",
            columns,
        )
        for values in columns:
            values.clear()

    for row in rows:
        payload = encode(row["payload"])
        payload_bytes = len(payload.encode("utf-8"))
        if columns[0] and (
            len(columns[0]) >= MAXIMUM_INSERT_ROWS
            or pending_bytes + payload_bytes > MAXIMUM_INSERT_JSON_BYTES
        ):
            flush()
            pending_bytes = 0
        for values, field in zip(columns[:3], ("record_id", "run_id", "result_id"), strict=True):
            values.append(row[field])
        columns[3].append(payload)
        pending_bytes += payload_bytes
    if columns[0]:
        flush()


def save_results(
    connection: duckdb.DuckDBPyConnection, results: Iterable[ResearchResult], batch_size: int
) -> dict[str, int]:
    """将业务结果逐批拆表写入并计数；例：重复主键会失败并由上层事务回滚。

    结果载荷按单条写入，其他表最多缓冲min(batch_size,128)行；实际JSON插入另有
    512KiB字节预算。完整消费与血缘全部保留，不构造全库JSON数组。
    """
    validate_batch_size(batch_size)
    pending: dict[str, list[dict[str, Any]]] = {table: [] for table in RESEARCH_TABLES}
    counts: Counter[str] = Counter()
    for result in results:
        for table, row in record_rows(result):
            pending[table].append(row)
            counts[table] += 1
            write_size = 1 if table == "result" else min(batch_size, MAXIMUM_INSERT_ROWS)
            if len(pending[table]) >= write_size:
                insert_rows(connection, table, pending[table])
                pending[table].clear()
    for table, rows in pending.items():
        if rows:
            insert_rows(connection, table, rows)
    return {table: counts[table] for table in RESEARCH_TABLES}


def stored_rows(
    connection: duckdb.DuckDBPyConnection, run_id: str, batch_size: int
) -> Iterator[tuple[str, dict[str, Any]]]:
    """流式读取一个批次的全部物理表；例：超过512行时分多个fetchmany完成。

    同连接读取期间不运行其他SQL，防止游标被覆盖而漏验尾部。
    """
    validate_batch_size(batch_size)
    read_size = min(batch_size, READ_BATCH_SIZE)
    for table in RESEARCH_TABLES:
        cursor = connection.execute(
            f'SELECT record_id,run_id,result_id,payload FROM research."{table}" WHERE run_id=?',
            [run_id],
        )
        while batch := cursor.fetchmany(read_size):
            for record_id, stored_run, result_id, payload in batch:
                yield (
                    table,
                    {
                        "record_id": record_id,
                        "run_id": stored_run,
                        "result_id": result_id,
                        "payload": json.loads(payload),
                    },
                )


def verify_run(
    connection: duckdb.DuckDBPyConnection, metadata: dict[str, Any], *, batch_size: int = 512
) -> dict[str, Any]:
    """重建全部结果及派生表，校验主键、引用、范围、三值状态和规则版本。

    例：把signal的unknown改为false、删证据或改贡献均导致验证失败。
    返回原始运行摘要，valid仅代表完整性，不表示历史可交易或数据齐备。
    """
    validate_batch_size(batch_size)
    read_size = min(batch_size, READ_BATCH_SIZE)
    validate_consumption_registry()
    config = ResearchConfig.from_dict(metadata["config"])
    from .binding_builder import binding_summary, verify_binding_audit

    verify_binding_audit(config, metadata["reference_sha256"], metadata["analysis_as_of"])
    run_id = metadata["run_id"]
    selection = metadata["selection"]
    required_products = set(selection)
    if required_products & set(PRODUCT_IDS[4:]):
        required_products.update(PRODUCT_IDS[1:4])
    if (
        not selection
        or selection != sorted(set(selection))
        or not set(selection) <= set(PRODUCT_IDS)
        or metadata["products"] != [name for name in PRODUCT_IDS if name in required_products]
    ):
        raise ValueError("research_product_selection_invalid")
    if (
        metadata.get("consumption_version") != CONSUMPTION_VERSION
        or metadata.get("consumption_registry_fingerprint") != registry_fingerprint()
        or metadata["rule_version"] != RULE_VERSION
    ) or run_id != run_identity(
        metadata["reference_sha256"],
        metadata["analysis_as_of"],
        config,
        tuple(metadata["selection"]),
    ):
        raise ValueError("research_run_identity_mismatch")
    for table in RESEARCH_TABLES:
        duplicate = connection.execute(
            f'SELECT count(*)-count(DISTINCT record_id) FROM research."{table}" WHERE run_id=?',
            [run_id],
        ).fetchone()
        if duplicate is None or duplicate[0]:
            raise ValueError("research_primary_key_invalid")
        orphan = connection.execute(
            f'SELECT count(*) FROM research."{table}" child '
            "LEFT JOIN research.result parent ON child.result_id=parent.record_id "
            "AND child.run_id=parent.run_id WHERE child.run_id=? "
            "AND parent.record_id IS NULL",
            [run_id],
        ).fetchone()
        if orphan is None or orphan[0]:
            raise ValueError("research_result_reference_missing")
    parents = connection.execute(
        "SELECT count(*) FROM research.lineage edge LEFT JOIN research.result parent "
        "ON (edge.payload->>'parent_result_id')=parent.record_id AND edge.run_id=parent.run_id "
        "WHERE edge.run_id=? AND (edge.payload->>'relationship')='parent' "
        "AND parent.record_id IS NULL",
        [run_id],
    ).fetchone()
    if parents is None or parents[0]:
        raise ValueError("research_parent_reference_missing")

    product_states: dict[str, Counter[str]] = {}
    consumption_counts: dict[str, Counter[str]] = {}

    def rebuilt_rows() -> Iterator[tuple[str, dict[str, Any]]]:
        """按有限批次反序列化结果并产生规范派生行；例：非法100分范围在此拒绝。

        只读取result表，避免物理子表被错误修改后成为校验真值来源。
        """
        cursor = connection.execute("SELECT payload FROM research.result WHERE run_id=?", [run_id])
        while batch := cursor.fetchmany(read_size):
            for (payload,) in batch:
                result = ResearchResult.from_json(payload)
                validate_result_consumption(result)
                count_consumption(consumption_counts, result)
                envelope = result.envelope
                if (
                    envelope.run_id != run_id
                    or envelope.model_version != RULE_VERSION
                    or envelope.product_id not in metadata["products"]
                    or envelope.analysis_as_of != metadata["analysis_as_of"]
                ):
                    raise ValueError("research_result_context_mismatch")
                product_states.setdefault(envelope.product_id, Counter())[envelope.state.value] += 1
                yield from record_rows(result)

    expected = fingerprints(rebuilt_rows())
    actual = fingerprints(stored_rows(connection, run_id, batch_size))
    if expected != actual or metadata["summary"]["counts"] != {
        table: values[0] for table, values in actual.items()
    }:
        raise ValueError("research_projection_mismatch")
    summary = metadata["summary"]
    reconstructed = {
        "schema_version": RULE_VERSION,
        "run_id": run_id,
        "analysis_as_of": metadata["analysis_as_of"],
        "mode": "observed_research",
        "historical_pit_certified": False,
        "requested_products": selection,
        "products": metadata["products"],
        "result_count": actual["result"][0],
        "product_states": {
            name: dict(sorted(states.items())) for name, states in product_states.items()
        },
        "consumption": summarize_consumption(consumption_counts, metadata["products"]),
        "status": "incomplete"
        if any(
            set(states) & {"blocked", "incomplete", "unsupported"}
            for states in product_states.values()
        )
        else "complete",
    }
    if config.binding_audit:
        reconstructed["bindings"] = binding_summary(config, reconstructed["consumption"])
    processing = summary["processing"]
    if (
        set(product_states) != required_products
        or any(summary.get(key) != value for key, value in reconstructed.items())
        or any(type(value) is not int or value < 0 for value in processing.values())
        or processing["scanned_rows"]
        != sum(processing[name] for name in ("retained_rows", "excluded_rows", "discarded_rows"))
    ):
        raise ValueError("research_summary_mismatch")
    return dict(metadata["summary"])


def validate_research(reference_path: Path, product_path: Path) -> dict[str, Any]:
    """只读验证研究库所有批次与当前参考指纹；例：篡改引用得到valid=false。

    接口只返回固定错误码，不回显来源行或敏感连接信息。
    """
    try:
        if reference_path.resolve() == product_path.resolve():
            raise ValueError("input_output_path_collision")
        reference_hash = file_checksum(reference_path)
        with duckdb.connect(
            str(product_path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            runs = connection.execute(
                "SELECT run_id,payload FROM research.run ORDER BY run_id"
            ).fetchall()
            if not runs:
                raise ValueError("research_run_missing")
            summaries = []
            current_row = connection.execute(
                "SELECT payload FROM research.metadata WHERE key='current_run'"
            ).fetchone()
            current = json.loads(current_row[0]) if current_row else None
            if current not in {run_id for run_id, _ in runs}:
                raise ValueError("research_current_run_missing")
            for run_id, payload in runs:
                metadata = json.loads(payload)
                if run_id != metadata["run_id"] or (
                    run_id == current and metadata["reference_sha256"] != reference_hash
                ):
                    raise ValueError("research_reference_changed")
                summaries.append(verify_run(connection, metadata))
                if run_id == current:
                    verify_source_evidence(
                        reference_path,
                        connection,
                        run_id,
                        ResearchConfig.from_dict(metadata["config"]),
                        analysis_as_of=metadata["analysis_as_of"],
                    )
            for table in RESEARCH_TABLES:
                orphan = connection.execute(
                    f'SELECT count(*) FROM research."{table}" child '
                    "LEFT JOIN research.run parent USING(run_id) "
                    "WHERE parent.run_id IS NULL"
                ).fetchone()
                if orphan is None or orphan[0]:
                    raise ValueError("research_run_reference_missing")
        return {
            "valid": True,
            "rule_version": RULE_VERSION,
            "runs": summaries,
            "historical_pit_certified": False,
        }
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError):
        return {
            "valid": False,
            "reason_codes": ["research_validation_failed"],
            "historical_pit_certified": False,
        }
