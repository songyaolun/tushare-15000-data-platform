"""产品库的集合式 SQL 操作；表名和列定义仅由存储契约提供。"""

import json
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any, TypeVar

import duckdb

from ..store import encode
from .config import validate_batch_size

Item = TypeVar("Item")


def batches(values: Iterable[Item], size: int) -> Iterator[list[Item]]:
    """按允许范围内的正整数惰性分批，不一次展开全部输入。

    输入 range(5)、size=2，依次输出 [0,1]、[2,3]、[4]；非法大小抛固定错误。
    size=513在开始迭代输入前拒绝，任意一批最多保留512个输入对象。
    """
    validate_batch_size(size)
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def stage_records(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: dict[str, str],
    records: list[dict[str, Any]],
) -> str:
    """用一个 JSON 参数建立有类型的临时批次，空值保持 SQL NULL。

    输入经过模型校验的 entity 批次，输出临时表名；一次 SQL 导入整个批次。
    table/columns 必须来自内部 TABLES，外部数据均通过绑定参数传入。
    """
    name = "product_batch_" + table
    projected = ", ".join(
        f"CAST(json_extract_string(value, '$.{column}') AS {kind}) AS \"{column}\""
        for column, kind in columns.items()
    )
    connection.execute(
        f'CREATE OR REPLACE TEMP TABLE "{name}" AS SELECT '
        "json_extract_string(value, '$.record_id') AS record_id, "
        "json_extract(value, '$.payload') AS payload, "
        f"{projected} FROM json_each(?::JSON)",
        [encode(records)],
    )
    return name


def check_conflicts(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    stage: str,
    payloads: dict[str, str],
) -> None:
    """一次 JOIN 取出批次内已存在的事实，规范化 JSON 后检查不可变冲突。

    输入同键同内容批次返回 None；同键不同内容抛 immutable_record_conflict。
    """
    for identity, payload in connection.execute(
        f'SELECT stored.record_id, stored.payload FROM product."{table}" stored '
        f'JOIN "{stage}" incoming USING (record_id)'
    ).fetchall():
        if encode(json.loads(payload)) != payloads[identity]:
            raise ValueError("immutable_record_conflict")


def check_entities(connection: duckdb.DuckDBPyConnection, identities: set[str]) -> None:
    """集合式检查一个批次中的所有实体外键，不为每条事实查询一次。

    输入已登记实体集合返回 None；任一实体未登记则抛 referenced_entity_missing。
    """
    if (
        identities
        and connection.execute(
            "SELECT entity_id FROM unnest(?::VARCHAR[]) refs(entity_id) "
            "ANTI JOIN product.entity USING (entity_id) LIMIT 1",
            [sorted(identities)],
        ).fetchone()
    ):
        raise ValueError("referenced_entity_missing")


def insert_records(connection: duckdb.DuckDBPyConnection, table: str, stage: str) -> None:
    """用单个 INSERT SELECT 插入不存在的记录，现有不可变记录不覆盖。

    输入已检查冲突的批次，输出 None；再次插入相同批次时表行数保持不变。
    """
    connection.execute(
        f'INSERT INTO product."{table}" SELECT incoming.* FROM "{stage}" incoming '
        f'ANTI JOIN product."{table}" stored USING (record_id)'
    )


def check_table_entities(
    connection: duckdb.DuckDBPyConnection, table: str, columns: dict[str, str]
) -> None:
    """在遍历开始前一次检查整表实体引用，避免辅助查询中断同连接流式结果。

    输入 typed_fact 表定义，所有引用存在时返回 None；任一缺失则拒绝校验。
    """
    references = [
        name
        for name in ("entity_id", "from_entity_id", "to_entity_id")
        if name in columns and table != "entity"
    ]
    if not references:
        return
    query = " UNION ALL ".join(
        f'SELECT "{name}" AS entity_id FROM product."{table}"' for name in references
    )
    if connection.execute(
        f"SELECT 1 FROM ({query}) refs ANTI JOIN product.entity USING (entity_id) LIMIT 1"
    ).fetchone():
        raise ValueError("referenced_entity_missing")


def check_table_lineage(connection: duckdb.DuckDBPyConnection, table: str) -> None:
    """集合式展开整表证据并核对完整血缘，不在流式读取中插入新查询。

    输入具有完整 used/excluded 边的表返回 None；任意位置缺边或内容不同均拒绝。
    json_tree 仅用于定位证据，行模型、身份和投影仍由上层逐批严格验证。
    """
    required_fields = ("source_api", "source_record_id", "contract_hash", "fields", "locator")
    evidence_filter = " AND ".join(
        f"json_exists(nodes.value, '$.{field}')" for field in required_fields
    )
    if connection.execute(
        f"SELECT 1 FROM (SELECT source.record_id AS output_id, nodes.value AS evidence, "
        "CASE WHEN contains(nodes.fullkey, '.excluded_evidence[') THEN 'excluded' "
        "ELSE 'used' END AS relationship "
        f'FROM product."{table}" source, json_tree(source.payload) nodes '
        f"WHERE nodes.type='OBJECT' AND {evidence_filter}) refs "
        "ANTI JOIN product.lineage_edge stored ON stored.output_id=refs.output_id AND "
        "stored.payload=json_object("
        "'contract_hash', json_extract_string(refs.evidence, '$.contract_hash'), "
        "'fields', json_extract(refs.evidence, '$.fields'), "
        "'output_id', refs.output_id, 'parent_result_id', NULL, 'reason_code', NULL, "
        "'relationship', refs.relationship, "
        "'source_api', json_extract_string(refs.evidence, '$.source_api'), "
        "'source_record_id', json_extract_string(refs.evidence, '$.source_record_id')) LIMIT 1"
    ).fetchone():
        raise ValueError("product_lineage_missing")
