"""独立产品库research表，按批写入并重建派生记录校验，不修改reference。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import duckdb

from ..product_foundation.config import validate_batch_size
from ..product_foundation.progress import ProgressCallback
from ..reference_storage import file_checksum
from ..store import digest, encode
from .consumption import count_consumption, summarize_consumption, validate_result_consumption
from .consumption_registry import (
    CONSUMPTION_VERSION,
    registry_fingerprint,
    validate_consumption_registry,
)
from .models import (
    ALL_EQUITY_PRODUCT_IDS,
    CANDIDATE_RANKING_PRODUCT_IDS,
    PRODUCT_IDS,
    RULE_VERSION,
    ResearchConfig,
    ResearchResult,
)
from .partitions import (
    PartitionWriter,
    partition_catalog,
    validate_partition_layout,
    verify_partition_keys,
)
from .persistence_runtime import (
    RESULT_PAYLOAD_CAPACITY_ERROR as RESULT_PAYLOAD_CAPACITY_ERROR,
)
from .persistence_runtime import (
    PersistenceProgress,
    PersistenceStepError,
    ResultPayloadCapacityError,
)
from .projection import RESEARCH_TABLES, EncodedRecord, accumulate_records, project_result
from .source_validation import verify_source_evidence

MAXIMUM_INSERT_ROWS = 128
MAXIMUM_INSERT_JSON_BYTES = 512 * 1024
MAXIMUM_RESULT_JSON_BYTES = 8 * 1024 * 1024
MAXIMUM_STAGE_TRANSACTION_BYTES = 16 * 1024 * 1024
MAXIMUM_STAGE_TRANSACTION_ROWS = 16384


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
    for row in project_result(result):
        yield row.table, row.decoded()


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
    一批且不截断。单result最多8MiB UTF8，超限在其SQL调用前抛固定容量错误。
    表名受白名单控制，主键与载荷参数分离，冲突拒绝而非覆盖。
    """
    if table not in RESEARCH_TABLES:
        raise ValueError("unknown_research_table")
    buffer = _InsertBuffer(connection, table, MAXIMUM_INSERT_ROWS)
    for row in rows:
        buffer.add(
            EncodedRecord(
                table, row["record_id"], row["run_id"], row["result_id"], encode(row["payload"])
            )
        )
    buffer.flush()


class _InsertBuffer:
    """共享行数/字节限批器，内部规范JSON直接复用；不放宽公开字典写入边界。"""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        table: str,
        row_limit: int,
        trace: PersistenceProgress | None = None,
        after_flush: Callable[[int, int], None] | None = None,
        partitions: PartitionWriter | None = None,
    ) -> None:
        """初始化四列缓冲；例：row_limit=128仍受512KiB约束，大结果独占一批。"""
        self.connection, self.table, self.row_limit, self.trace = (
            connection,
            table,
            row_limit,
            trace,
        )
        self.after_flush = after_flush
        self.partitions = partitions
        self.columns: list[list[str]] = [[], [], [], []]
        self.pending_bytes = 0

    def add(self, row: EncodedRecord) -> None:
        """接收一个已通过公开边界的规范快照，超8MiB结果在SQL前失败；例：两条400KiB分批写入。"""
        size = len(row.payload.encode("utf-8"))
        if self.trace is not None:
            self.trace.table = self.table
        if self.table == "result" and size > MAXIMUM_RESULT_JSON_BYTES:
            raise ResultPayloadCapacityError(json.loads(row.payload), size)
        if self.columns[0] and self.pending_bytes + size > MAXIMUM_INSERT_JSON_BYTES:
            self.flush()
        for column, value in zip(
            self.columns, (row.record_id, row.run_id, row.result_id, row.payload), strict=True
        ):
            column.append(value)
        self.pending_bytes += size
        if (
            len(self.columns[0]) >= self.row_limit
            or self.pending_bytes >= MAXIMUM_INSERT_JSON_BYTES
        ):
            self.flush()

    def flush(self) -> None:
        """把非空批写入事务后清空，报告实际写入量；例：空缓冲不执行SQL。"""
        if not self.columns[0]:
            return
        if self.partitions is not None:
            self.partitions.insert(self.table, self.columns, self.pending_bytes)
        else:
            self.connection.execute(
                f'INSERT INTO research."{self.table}" '
                "SELECT unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), "
                "unnest(?::VARCHAR[]), CAST(unnest(?::VARCHAR[]) AS JSON)",
                self.columns,
            )
        if self.trace is not None:
            self.trace.advance(self.table, len(self.columns[0]), self.pending_bytes)
        if self.after_flush is not None:
            self.after_flush(len(self.columns[0]), self.pending_bytes)
        for values in self.columns:
            values.clear()
        self.pending_bytes = 0


class _StageTransactions:
    """仅在不可见私有副本上限制事务规模；外部原子性由验证后的文件发布保证。"""

    def __init__(self, connection: duckdb.DuckDBPyConnection, trace: PersistenceProgress) -> None:
        """初始化零预算；例：每16MiB载荷或16384条物理记录提交并checkpoint一次。"""
        self.connection, self.trace = connection, trace
        self.rows = self.payload_bytes = 0
        self.trace.transaction_state = "active"

    def advance(self, rows: int, payload_bytes: int) -> None:
        """接收已写批大小，达到任一阈值提交并开启下一事务；例：两批9MiB触发一次分段提交。

        每个SQL批仍受128行/512KiB限制，合法宽结果独占；不截断结果、不减少任何投影。
        此连接必须由私有快照独占，失败后副本整体删除，正式库不会看见部分提交。
        """
        self.rows += rows
        self.payload_bytes += payload_bytes
        if (
            self.rows >= MAXIMUM_STAGE_TRANSACTION_ROWS
            or self.payload_bytes >= MAXIMUM_STAGE_TRANSACTION_BYTES
        ):
            for substep, command in (
                ("commit", "COMMIT"),
                ("checkpoint", "CHECKPOINT"),
                ("reopen", "BEGIN TRANSACTION"),
            ):
                self.trace.substep = substep
                self.trace.step("commit_batch")
                try:
                    self.connection.execute(command)
                except Exception as error:
                    self.trace.transaction_state = "failed"
                    raise PersistenceStepError(substep, error) from error
                self.trace.transaction_state = "active" if substep == "reopen" else "inactive"
            self.trace.substep = None
            self.rows = self.payload_bytes = 0
            self.trace.operation = "write"


def save_results(
    connection: duckdb.DuckDBPyConnection,
    results: Iterable[ResearchResult],
    batch_size: int,
    *,
    progress: ProgressCallback | None = None,
    run_id: str | None = None,
    staged: bool = False,
) -> dict[str, int]:
    """每结果只序列化根一次，七表各自按行数/字节限批，完成表示事务内写入，不代表提交。

    例：128个小结果可合批，超512KiB的合法结果独占一批；失败撤销事务或删除私有副本。
    不减少任何消费记录、正反证或血缘，不更改物理记录键及载荷。
    staged=True仅供私有副本使用，按字节/行数分段提交，公开直接调用默认仍由外层事务控制。
    """
    validate_batch_size(batch_size)
    trace = PersistenceProgress("save", progress, run_id)
    partitions = None
    try:
        trace.emit("start-save")
        trace.operation = "write"
        transactions = _StageTransactions(connection, trace) if staged else None
        partitions = PartitionWriter(connection) if staged else None
        buffers = {
            table: _InsertBuffer(
                connection,
                table,
                min(batch_size, MAXIMUM_INSERT_ROWS),
                trace,
                transactions.advance if transactions is not None else None,
                partitions,
            )
            for table in RESEARCH_TABLES
        }
        counts: Counter[str] = Counter()
        for result in results:
            for row in project_result(result):
                buffers[row.table].add(row)
                counts[row.table] += 1
            trace.result_completed()
        for buffer in buffers.values():
            buffer.flush()
        trace.emit("save-complete")
        return {table: counts[table] for table in RESEARCH_TABLES}
    except Exception as error:
        # 独立调用方仍可按DuckDB约束异常处理；阶段信息通过进度提供。
        trace.failure(error)
        raise
    finally:
        if partitions is not None:
            partitions.close()


def stored_rows(
    connection: duckdb.DuckDBPyConnection, run_id: str, batch_size: int
) -> Iterator[tuple[str, dict[str, Any]]]:
    """流式读取一个批次的全部物理表；例：超过512行时分多个fetchmany完成。

    同连接读取期间不运行其他SQL，防止游标被覆盖而漏验尾部。
    """
    validate_batch_size(batch_size)
    for row in _stored_records(connection, run_id, RESEARCH_TABLES):
        yield row.table, row.decoded()


def _stored_records(
    connection: duckdb.DuckDBPyConnection, run_id: str, tables: tuple[str, ...]
) -> Iterator[EncodedRecord]:
    """逐条取出物理JSON并重新检查公开边界，避免512个8MiB结果同时进入Python。

    例：读取两个大结果每次fetchmany(1)，不随配置批大小提高在途载荷数。
    SQL表名仅来自本模块固定表集合，输入值仍绑定参数。
    """
    parts = partition_catalog(connection)
    for table in tables:
        for physical in parts.get(table, (table,)):
            cursor = connection.execute(
                "SELECT record_id,run_id,result_id,payload "
                f'FROM research."{physical}" WHERE run_id=?',
                [run_id],
            )
            while batch := cursor.fetchmany(1):
                record_id, stored_run, result_id, payload = batch[0]
                if len(payload.encode("utf-8")) > MAXIMUM_RESULT_JSON_BYTES:
                    raise ValueError("research_projection_payload_capacity_exceeded")
                yield EncodedRecord(
                    table, record_id, stored_run, result_id, encode(json.loads(payload))
                )


def _verify_keys(
    connection: duckdb.DuckDBPyConnection, run_id: str, trace: PersistenceProgress
) -> None:
    """按真实主键约束或完整去重检查验证七表键和反向引用，不跳过缺约束表。

    例：正常库不重复建立COUNT DISTINCT哈希表；有人重建无主键表时仍扫描并拒绝重复键。
    分段表先核对行键一一对应，再以窄键连接并逐段检查血缘；缺失计数仍拒绝。
    """
    trace.step("primary_keys")
    parts = validate_partition_layout(connection)
    parent_table = "_keys_result" if "result" in parts else "result"
    constrained = {
        table
        for table, columns in connection.execute(
            "SELECT table_name,constraint_column_names FROM duckdb_constraints() "
            "WHERE database_name=current_database() AND schema_name='research' "
            "AND constraint_type='PRIMARY KEY'",
            [],
        ).fetchall()
        if columns == ["record_id"]
    }
    for table in RESEARCH_TABLES:
        trace.table = table
        if table in parts:
            verify_partition_keys(connection, table, run_id)
        elif table not in constrained:
            duplicate = connection.execute(
                f'SELECT count(*)-count(DISTINCT record_id) FROM research."{table}" WHERE run_id=?',
                [run_id],
            ).fetchone()
            if duplicate is None or duplicate[0]:
                raise ValueError("research_primary_key_invalid")
        reference_table = "_keys_" + table if table in parts else table
        orphan = connection.execute(
            f'SELECT count(*) FROM research."{reference_table}" child '
            f"LEFT JOIN research.{parent_table} parent ON child.result_id=parent.record_id "
            "AND child.run_id=parent.run_id WHERE child.run_id=? "
            "AND parent.record_id IS NULL",
            [run_id],
        ).fetchone()
        if orphan is None or orphan[0]:
            raise ValueError("research_result_reference_missing")
    for physical in parts.get("lineage", ("lineage",)):
        parents = connection.execute(
            f"SELECT count(*) FROM research.{physical} edge "
            f"LEFT JOIN research.{parent_table} parent "
            "ON (edge.payload->>'parent_result_id')=parent.record_id AND edge.run_id=parent.run_id "
            "WHERE edge.run_id=? AND (edge.payload->>'relationship')='parent' "
            "AND parent.record_id IS NULL",
            [run_id],
        ).fetchone()
        if parents is None or parents[0]:
            raise ValueError("research_parent_reference_missing")


def verify_run(
    connection: duckdb.DuckDBPyConnection,
    metadata: dict[str, Any],
    *,
    batch_size: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """完整重建全部结果和物理投影，并报告安全子阶段；例：缺少子表记录仍抛projection_mismatch。

    开始/结束及阶段事件不写入摘要，重跑保持同一运行键和JSON；失败不输出数据库异常原文。
    """
    validate_batch_size(batch_size)
    trace = PersistenceProgress("verify", progress, metadata.get("run_id"))
    try:
        trace.emit("verify-start")
        summary = _verify_run(connection, metadata, batch_size=batch_size, trace=trace)
        trace.emit("verify-complete")
        return summary
    except Exception as error:
        trace.failure(error)
        raise


def _verify_run(
    connection: duckdb.DuckDBPyConnection,
    metadata: dict[str, Any],
    *,
    batch_size: int,
    trace: PersistenceProgress,
) -> dict[str, Any]:
    """重建全部结果及派生表，校验主键、引用、范围、三值状态和规则版本。

    例：把signal的unknown改为false、删证据或改贡献均导致验证失败。
    返回原始运行摘要，valid仅代表完整性，不表示历史可交易或数据齐备。
    """
    validate_batch_size(batch_size)
    validate_consumption_registry()
    config = ResearchConfig.from_dict(metadata["config"])
    from .binding_builder import binding_summary, verify_binding_audit

    verify_binding_audit(config, metadata["reference_sha256"], metadata["analysis_as_of"])
    run_id = metadata["run_id"]
    selection = metadata["selection"]
    required_products = set(selection)
    if required_products & {PRODUCT_IDS[4], PRODUCT_IDS[5], CANDIDATE_RANKING_PRODUCT_IDS[0]}:
        required_products.update(PRODUCT_IDS[1:4])
    if required_products & set(CANDIDATE_RANKING_PRODUCT_IDS[1:]):
        required_products.update(PRODUCT_IDS[1:3])
    if (
        not selection
        or selection != sorted(set(selection))
        or not set(selection) <= set(ALL_EQUITY_PRODUCT_IDS)
        or metadata["products"]
        != [name for name in ALL_EQUITY_PRODUCT_IDS if name in required_products]
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
    _verify_keys(connection, run_id, trace)
    trace.step("rebuild")
    actual = {table: [0, 0] for table in RESEARCH_TABLES}
    product_states: dict[str, Counter[str]] = {}
    consumption_counts: dict[str, Counter[str]] = {}

    def rebuilt_rows() -> Iterator[EncodedRecord]:
        """按有限批次反序列化结果并产生规范派生行；例：非法100分范围在此拒绝。

        只读取result表，避免物理子表被错误修改后成为校验真值来源。
        """
        for stored in _stored_records(connection, run_id, ("result",)):
            # 同一物理行只读一次：原始规范载荷与模型规范化后的投影分别散列，仍能发现差异。
            accumulate_records((stored,), actual)
            result = ResearchResult.from_json(stored.payload)
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
            for projected in project_result(result):
                trace.advance(projected.table, 1, len(projected.payload.encode("utf-8")))
                yield projected
            trace.result_completed()

    expected = accumulate_records(rebuilt_rows())
    trace.step("physical_projections")
    for row in _stored_records(connection, run_id, RESEARCH_TABLES[1:]):
        accumulate_records((row,), actual)
        trace.advance(row.table, 1, len(row.payload.encode("utf-8")))
    if expected != actual or metadata["summary"]["counts"] != {
        table: values[0] for table, values in actual.items()
    }:
        raise ValueError("research_projection_mismatch")
    trace.step("summary")
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
            checked_tables = RESEARCH_TABLES + tuple(
                "_keys_" + table for table in partition_catalog(connection)
            )
            for table in checked_tables:
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
