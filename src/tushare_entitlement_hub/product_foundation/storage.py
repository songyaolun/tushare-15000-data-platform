"""产品独立 DuckDB：严格模型、只增事实、事务、字段投影及证据血缘。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from ..pit import time_boundary
from ..reference_storage import file_checksum
from ..store import digest, encode
from .batch_io import (
    batches,
    check_conflicts,
    check_entities,
    check_table_entities,
    check_table_lineage,
    insert_records,
    stage_records,
)
from .config import DEFAULT_FACT_BATCH_SIZE, READ_BATCH_SIZE, validate_batch_size
from .entities import Entity, EntityAlias, EntityRelation
from .features import FeatureValue
from .model import Model
from .numeric import TypedFact
from .registry import load_registry, validate_registry
from .signals import SignalFact
from .storage_records import CoverageSlice, GateRecord, LineageEdge, ScoreComponent
from .temporal import AvailabilityFact

STORAGE_VERSION = "product-foundation/v1"
TABLES: dict[str, tuple[type[Model], tuple[str, ...], dict[str, str]]] = {
    "entity": (
        Entity,
        ("entity_id",),
        {"entity_id": "VARCHAR", "kind": "VARCHAR", "namespace": "VARCHAR"},
    ),
    "entity_alias": (
        EntityAlias,
        ("namespace", "alias", "entity_id", "effective_from", "version", "observed_at"),
        {
            "entity_id": "VARCHAR",
            "namespace": "VARCHAR",
            "alias": "VARCHAR",
            "effective_from": "TIMESTAMPTZ",
            "effective_to": "TIMESTAMPTZ",
        },
    ),
    "entity_relation": (
        EntityRelation,
        (
            "from_entity_id",
            "to_entity_id",
            "relation_type",
            "effective_from",
            "version",
            "observed_at",
        ),
        {
            "from_entity_id": "VARCHAR",
            "to_entity_id": "VARCHAR",
            "relation_type": "VARCHAR",
            "confirmed": "BOOLEAN",
        },
    ),
    "availability_fact": (
        AvailabilityFact,
        ("source_api", "source_record_id", "certification_version"),
        {
            "source_api": "VARCHAR",
            "source_record_id": "VARCHAR",
            "observed_at": "TIMESTAMPTZ",
            "public_available_at": "TIMESTAMPTZ",
            "available_at": "TIMESTAMPTZ",
            "data_as_of": "VARCHAR",
            "historical_pit_certified": "BOOLEAN",
        },
    ),
    "typed_fact": (
        TypedFact,
        ("fact_id",),
        {
            "entity_id": "VARCHAR",
            "source_api": "VARCHAR",
            "field": "VARCHAR",
            "value": "DOUBLE",
            "unit": "VARCHAR",
            "quality_tier": "VARCHAR",
        },
    ),
    "feature_value": (
        FeatureValue,
        (
            "entity_id",
            "feature_id",
            "analysis_as_of",
            "scope_id",
            "window",
            "feature_profile",
            "version",
        ),
        {
            "entity_id": "VARCHAR",
            "feature_id": "VARCHAR",
            "analysis_as_of": "TIMESTAMPTZ",
            "feature_profile": "VARCHAR",
            "value": "DOUBLE",
            "unit": "VARCHAR",
        },
    ),
    "signal_fact": (
        SignalFact,
        ("entity_id", "rule_id", "rule_version", "triggered_at"),
        {
            "entity_id": "VARCHAR",
            "rule_id": "VARCHAR",
            "result": "VARCHAR",
            "next_check_at": "TIMESTAMPTZ",
        },
    ),
    "score_component": (
        ScoreComponent,
        ("result_id", "feature_id"),
        {"result_id": "VARCHAR", "feature_id": "VARCHAR", "contribution": "DOUBLE"},
    ),
    "quality_gate_result": (
        GateRecord,
        ("result_id", "rule_version", "gate_id"),
        {"result_id": "VARCHAR", "gate_id": "VARCHAR", "result": "VARCHAR"},
    ),
    "coverage_slice": (
        CoverageSlice,
        ("source_api", "scope_id", "partition", "run_id"),
        {
            "source_api": "VARCHAR",
            "run_id": "VARCHAR",
            "accepted_rows": "BIGINT",
            "quarantined_rows": "BIGINT",
            "data_as_of": "VARCHAR",
        },
    ),
    "lineage_edge": (
        LineageEdge,
        ("output_id", "source_api", "source_record_id", "fields", "relationship"),
        {
            "output_id": "VARCHAR",
            "source_api": "VARCHAR",
            "source_record_id": "VARCHAR",
            "relationship": "VARCHAR",
        },
    ),
}


def _projection(record: Model, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """提取便于 SQL 使用的稳定字段；输入 TypedFact，输出额外 source_api 投影。

    嵌套完整内容仍保存在 payload，不丢失有效区间、证据或单位认证。
    """
    value = record.to_dict() if payload is None else dict(payload)
    if isinstance(record, Entity):
        value.update(
            entity_id=record.identity.entity_id,
            kind=record.identity.kind.value,
            namespace=record.identity.namespace,
        )
    elif isinstance(record, AvailabilityFact):
        value["available_at"] = record.available_at
    elif isinstance(record, TypedFact):
        value["source_api"] = record.availability.source_api
    elif isinstance(record, ScoreComponent):
        value.update(
            feature_id=record.component.feature_id, contribution=record.component.contribution
        )
    elif isinstance(record, GateRecord):
        value.update(gate_id=record.gate.gate_id, result=record.gate.result.value)
    return value


def _evidence_objects(value: Any, excluded: bool = False) -> list[tuple[dict[str, Any], bool]]:
    """递归提取结构化证据用于自动血缘；输入嵌套模型字典，输出证据字典列表。

    例如 {'evidence_refs':[ref]} → [(ref,False)]；排除证据标为 True，普通数值不产生血缘。
    """
    if isinstance(value, dict):
        if {"source_api", "source_record_id", "contract_hash", "fields", "locator"} <= set(value):
            return [(value, excluded)]
        return [
            item
            for key, child in value.items()
            for item in _evidence_objects(child, excluded or key == "excluded_evidence")
        ]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _evidence_objects(child, excluded)]
    return []


def _prepare_records(
    table: str, records: Iterable[Model]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """重验输入并按身份去重，不允许批内同键异值，返回 SQL 行和原始顺序身份。

    输入两个相同实体，输出一条待写行及两个相同身份；非法表模型立即拒绝。
    """
    rows: dict[str, dict[str, Any]] = {}
    identities = []
    model, keys, columns = TABLES[table]
    for record in records:
        if type(record) is not model:
            raise ValueError("product_table_model_mismatch")
        record = model.from_dict(record.to_dict())
        payload = asdict(record)
        value = _projection(record, payload)
        identity = digest([value[key] for key in keys])
        if identity in rows and rows[identity]["payload"] != payload:
            raise ValueError("immutable_record_conflict")
        rows[identity] = {
            "record_id": identity,
            "payload": payload,
            **{name: value.get(name) for name in columns},
        }
        identities.append(identity)
    return rows, identities


def _entity_references(table: str, rows: Iterable[dict[str, Any]]) -> set[str]:
    """提取批次中的显式实体引用，实体表自身不需要外键检查。

    输入 typed_fact 行，输出 entity_id 集合；输入 entity 行输出空集合。
    """
    return {
        row[key]
        for row in rows
        for key in ("entity_id", "from_entity_id", "to_entity_id")
        if table != "entity" and key in row
    }


def _lineage_records(rows: Iterable[dict[str, Any]]) -> list[LineageEdge]:
    """从已校验的批次 payload 生成 used/excluded 血缘，身份仍遵循共享表契约。

    输入包含一条排除证据的分量行，输出一条 relationship=excluded 的 LineageEdge。
    """
    return [
        LineageEdge(
            output_id=row["record_id"],
            source_api=evidence["source_api"],
            source_record_id=evidence["source_record_id"],
            fields=tuple(evidence["fields"]),
            contract_hash=evidence["contract_hash"],
            relationship="excluded" if excluded else "used",
        )
        for row in rows
        for evidence, excluded in _evidence_objects(row["payload"])
    ]


class ProductStore:
    """只写产品库，默认 1 线程及 512MB DuckDB 内存；上下文负责事务收口。"""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        """初始化独立产品表或只读打开；输入新产品路径，输出可用数据库连接。

        例如 ProductStore(Path('/tmp/products.duckdb')).counts()['entity'] → 0。
        """
        if read_only and not path.is_file():
            raise ValueError("product_library_missing")
        existing = path.exists()
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.connection = duckdb.connect(
            str(path), read_only=read_only, config={"threads": 1, "memory_limit": "512MB"}
        )
        try:
            if existing and set(self.tables()) != set(TABLES):
                raise ValueError("product_tables_missing")
            if not read_only:
                self.connection.execute("CREATE SCHEMA IF NOT EXISTS product")
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS product_metadata"
                    "(key VARCHAR PRIMARY KEY, payload JSON)"
                )
                for name, (_, _, columns) in TABLES.items():
                    projection = ", ".join(f'"{column}" {kind}' for column, kind in columns.items())
                    self.connection.execute(
                        f'CREATE TABLE IF NOT EXISTS product."{name}" '
                        f"(record_id VARCHAR PRIMARY KEY, payload JSON NOT NULL, {projection})"
                    )
                self.connection.execute(
                    "INSERT INTO product_metadata VALUES ('schema_version', ?) "
                    "ON CONFLICT DO NOTHING",
                    [encode(STORAGE_VERSION)],
                )
            if self.metadata("schema_version") != STORAGE_VERSION:
                raise ValueError("product_schema_version_mismatch")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self) -> Self:
        """开始一次原子写批次；输入 with ProductStore(...)，输出自身。

        只读连接不执行事务写操作，便于校验时保持文件不变。
        """
        if not self.read_only:
            self.connection.execute("BEGIN TRANSACTION")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """成功提交、失败回滚并释放连接；批次中异常 → 旧数据仍在。

        输入上下文退出状态，输出 None，原异常继续由调用者处理。
        """
        try:
            if not self.read_only:
                self.connection.execute("ROLLBACK" if kind else "COMMIT")
        finally:
            self.close()

    def close(self) -> None:
        """释放 DuckDB 连接；store.close() → None，随后可再次打开同一文件。"""
        self.connection.close()

    def tables(self) -> tuple[str, ...]:
        """返回实际产品 schema 的表名；例如新库输出 11 张共享表，不包含控制元数据。"""
        return tuple(
            row[0]
            for row in self.connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='product' ORDER BY table_name"
            ).fetchall()
        )

    def counts(self) -> dict[str, int]:
        """返回固定共享表的物理行数；例如空库 entity=0，不代表上游接口无数据。"""
        return {
            name: int(
                self.connection.execute(f'SELECT count(*) FROM product."{name}"').fetchall()[0][0]
            )
            for name in TABLES
        }

    def metadata(self, key: str) -> Any:
        """读取产品运行元数据，未登记键返回 None；schema_version → 固定版本。"""
        row = self.connection.execute(
            "SELECT payload FROM product_metadata WHERE key=?", [key]
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_metadata(self, key: str, value: Any) -> None:
        """在当前事务保存公开运行摘要；输入 summary 字典，输出 None。

        凭据字段由共用 encode 拒绝，未知 SQL 键值均用绑定参数处理。
        """
        self.connection.execute(
            "INSERT OR REPLACE INTO product_metadata VALUES (?, ?)", [key, encode(value)]
        )

    def put(self, table: str, record: Model) -> str:
        """用批量写入的同一校验路径写单条，保留原有返回值与不可变冲突语义。

        输入 entity 模型，输出稳定 record_id；同键同内容幂等，同键异值抛错。
        """
        return self.put_many(table, (record,))[0]

    def put_many(
        self, table: str, records: Iterable[Model], *, batch_size: int = DEFAULT_FACT_BATCH_SIZE
    ) -> list[str]:
        """分批校验、集合式插入事实与血缘，绝不为每行先 SELECT 再 INSERT。

        输入一千条事实及 batch_size=512，输出按输入顺序排列的一千个身份；
        同批重复也幂等，任何不可变冲突抛错。用 with ProductStore 包裹整个调用，
        即可在后续批次失败时回滚所有批次；单条 put 同样使用此路径。
        批大小仅允许1到512；输入百万级大小在消费records或执行SQL之前拒绝。
        """
        validate_batch_size(batch_size)
        if table not in TABLES:
            raise ValueError("product_table_model_mismatch")
        identities = []
        for batch in batches(records, batch_size):
            rows, batch_ids = _prepare_records(table, batch)
            stage = stage_records(self.connection, table, TABLES[table][2], list(rows.values()))
            check_conflicts(
                self.connection,
                table,
                stage,
                {key: encode(value["payload"]) for key, value in rows.items()},
            )
            check_entities(self.connection, _entity_references(table, rows.values()))
            edges = _lineage_records(rows.values()) if table != "lineage_edge" else []
            edge_stage = None
            if edges:
                edge_rows, _ = _prepare_records("lineage_edge", edges)
                edge_stage = stage_records(
                    self.connection,
                    "lineage_edge",
                    TABLES["lineage_edge"][2],
                    list(edge_rows.values()),
                )
                check_conflicts(
                    self.connection,
                    "lineage_edge",
                    edge_stage,
                    {key: encode(value["payload"]) for key, value in edge_rows.items()},
                )
            insert_records(self.connection, table, stage)
            if edge_stage is not None:
                insert_records(self.connection, "lineage_edge", edge_stage)
            identities.extend(batch_ids)
        return identities

    def verify_contents(self, *, batch_size: int = READ_BATCH_SIZE) -> None:
        """流式重验模型、身份和 SQL 投影，按批次集合式检查实体引用及证据边。

        输入完整库返回 None；改动投影、身份、实体或血缘均抛固定原因码。
        先集合式检查引用与血缘，再遍历当前事务；遍历期间不在同连接执行其他 SQL，
        最后核对实际遍历行数，避免结果被引擎中断时将部分扫描误认成完整验证。
        输入合法batch_size=512时实际每次最多读取128行；超限大小在任何SQL之前拒绝。
        """
        validate_batch_size(batch_size)
        read_size = min(batch_size, READ_BATCH_SIZE)
        if set(self.tables()) != set(TABLES):
            raise ValueError("product_tables_missing")
        for table, (model, keys, columns) in TABLES.items():
            check_table_entities(self.connection, table, columns)
            if table != "lineage_edge":
                check_table_lineage(self.connection, table)
            expected_rows = self.connection.execute(
                f'SELECT count(*) FROM product."{table}"'
            ).fetchall()[0][0]
            checked_rows = 0
            result = self.connection.sql(f'SELECT to_json(stored) FROM product."{table}" stored')
            while batch := result.fetchmany(read_size):
                checked_rows += len(batch)
                for raw in batch:
                    stored = json.loads(raw[0])
                    record = model.from_dict(stored["payload"])
                    value = _projection(record, asdict(record))
                    identity = digest([value[key] for key in keys])
                    if stored["record_id"] != identity:
                        raise ValueError("product_record_identity_mismatch")
                    for name, kind in columns.items():
                        actual, expected = stored[name], value.get(name)
                        if kind == "TIMESTAMPTZ" and actual is not None and expected is not None:
                            actual, expected = time_boundary(actual), time_boundary(expected)
                        if actual != expected:
                            raise ValueError("product_projection_mismatch")
            if checked_rows != expected_rows:
                raise ValueError("product_scan_incomplete")

    def records(self, table: str) -> list[Model]:
        """按稳定身份读取并重验模型；输入 entity → Entity 列表，任意表名拒绝。"""
        if table not in TABLES:
            raise ValueError("unknown_product_table")
        model = TABLES[table][0]
        return [
            model.from_json(row[0])
            for row in self.connection.execute(
                f'SELECT payload FROM product."{table}" ORDER BY record_id'
            ).fetchall()
        ]

    def verify_snapshot(self, registered_apis: set[str]) -> dict[str, Any]:
        """对账初始化快照的摘要、当前批次覆盖集合和逐项来源就绪门禁。

        调用前须以 verify_contents 校验模型及 SQL 投影；普通手工事实库无需调用。
        输入当前注册表的206个API，完整快照输出已对账摘要；缺一项覆盖则抛ValueError。
        """
        summary = self.metadata("summary")
        if not isinstance(summary, dict):
            raise ValueError("product_summary_missing")
        run_id = self.metadata("run_id")
        counts = summary.get("counts")
        if (
            not isinstance(run_id, str)
            or not run_id
            or summary.get("run_id") != run_id
            or summary.get("schema_version") != self.metadata("schema_version")
            or type(summary.get("registered_apis")) is not int
            or summary.get("registered_apis") != len(registered_apis)
            or not isinstance(counts, dict)
            or any(type(value) is not int for value in counts.values())
            or counts != self.counts()
        ):
            raise ValueError("product_summary_mismatch")
        coverage = self.connection.execute(
            "SELECT record_id, source_api, run_id FROM product.coverage_slice"
        ).fetchall()
        if (
            len(coverage) != len(registered_apis)
            or {row[1] for row in coverage} != registered_apis
            or any(row[2] != run_id for row in coverage)
        ):
            raise ValueError("product_coverage_mismatch")
        readiness_ids = {
            row[0]
            for row in self.connection.execute(
                "SELECT result_id FROM product.quality_gate_result WHERE gate_id='source_readiness'"
            ).fetchall()
        }
        if readiness_ids != {row[0] for row in coverage}:
            raise ValueError("product_source_readiness_mismatch")
        return summary


def validate_library(reference_path: Path, product_path: Path) -> dict[str, Any]:
    """只读验证注册表、输入指纹、持久化模型，以及当前覆盖和摘要对账。

    输入未变参考库及产品库，输出 valid=true；修改参考库后输出 valid=false。
    valid 仅表示库的完整性，不能替代 coverage 或历史 PIT 资格。
    """
    try:
        entries = load_registry()
        validate_registry(entries)
        if reference_path.resolve() == product_path.resolve() or reference_path.samefile(
            product_path
        ):
            raise ValueError("input_output_path_collision")
        with ProductStore(product_path, read_only=True) as store:
            if store.metadata("reference_sha256") != file_checksum(reference_path):
                raise ValueError("reference_changed")
            store.verify_contents()
            summary = store.verify_snapshot({entry.api for entry in entries})
            return {
                "valid": True,
                "status": summary["status"],
                "counts": store.counts(),
                "historical_pit_certified": False,
            }
    except (ValueError, OSError, duckdb.Error, KeyError):
        return {
            "valid": False,
            "status": "failed",
            "reason_codes": ["product_validation_failed"],
            "historical_pit_certified": False,
        }
