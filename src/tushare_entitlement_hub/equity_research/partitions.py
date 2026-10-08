"""宽JSON的有界物理分段，保留七个逻辑投影的列、值和跨段唯一键。"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .key_index import DiskKeyIndex
from .projection import RESEARCH_TABLES

MAXIMUM_PARTITION_BYTES = 16 * 1024 * 1024
MAXIMUM_PARTITIONS = 4096
MAXIMUM_PARTITION_ROWS = 16384
ROW_PAYLOAD_BOUND = 8 * 1024 * 1024
_COLUMNS = "record_id,run_id,result_id,payload"
_KEY_LAYOUT = '"disk_btree_v1"'


def _disk_key_layout(connection: duckdb.DuckDBPyConnection) -> bool:
    """读取物理键存储版本，只接受缺省旧ART或明确磁盘唯一性协议；例：未知版本固定拒绝。"""
    row = connection.execute(
        "SELECT payload FROM research.metadata WHERE key='partition_keys'", []
    ).fetchone()
    if row is not None and row[0] != _KEY_LAYOUT:
        raise ValueError("research_partition_layout_invalid")
    return row is not None


def partition_name(table: str, number: int) -> str:
    """生成仅由内部表白名单和整数构成的段名；例：result/0得到_part_result_000000。"""
    if table not in RESEARCH_TABLES or not 0 <= number < MAXIMUM_PARTITIONS:
        raise ValueError("research_partition_capacity_exceeded")
    return f"_part_{table}_{number:06d}"


def partition_catalog(connection: duckdb.DuckDBPyConnection) -> dict[str, tuple[str, ...]]:
    """读取当前库保留前缀的段目录并拒绝非法名字，不信任其他附加库；例：普通旧库返回空字典。"""
    grouped: dict[str, list[str]] = {}
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() "
        "AND table_schema='research' AND starts_with(table_name,'_part_') ORDER BY table_name",
        [],
    ).fetchall()
    for (name,) in rows:
        match = re.fullmatch(r"_part_([a-z]+)_([0-9]{6})", name)
        if match is None or match[1] not in RESEARCH_TABLES:
            raise ValueError("research_partition_layout_invalid")
        grouped.setdefault(match[1], []).append(name)
    return {table: tuple(names) for table, names in grouped.items()}


def view_select(parts: tuple[str, ...] | list[str]) -> str:
    """生成保持四列原值的UNION ALL，不合并或过滤来源行；例：两段输出两支显式四列SELECT。"""
    branches = [f'SELECT {_COLUMNS} FROM research."{name}"' for name in parts]
    while len(branches) > 1:
        branches = [
            f"({branches[index]}) UNION ALL ({branches[index + 1]})"
            if index + 1 < len(branches)
            else branches[index]
            for index in range(0, len(branches), 2)
        ]
    return branches[0] if branches else ""


def _view_parts(sql: str) -> tuple[str, ...]:
    """识别唯一允许的四列UNION ALL语法，接受持久化后等价的括号平衡。

    例：'(SELECT四列FROM段0) UNION ALL (SELECT四列FROM段1)'返回两个段名；
    函数调用、过滤、列改写或其他SQL均拒绝，不靠删除任意括号来放宽语义。
    """
    if not isinstance(sql, str) or " AS " not in sql:
        raise ValueError("research_partition_layout_invalid")
    body = sql.split(" AS ", 1)[1].removesuffix(";")
    pattern = re.compile(
        r"SELECT\s+record_id\s*,\s*run_id\s*,\s*result_id\s*,\s*payload\s+"
        r"FROM\s+research\.(_part_[a-z]+_[0-9]{6})"
    )
    names = tuple(match[1] for match in pattern.finditer(body))
    remainder = pattern.sub("X", body)
    operand, depth = True, 0
    for token in re.findall(r"UNION\s+ALL|X|[()]|[^\s()]+", remainder):
        if token == "(" and operand:
            depth += 1
        elif token == "X" and operand:
            operand = False
        elif re.fullmatch(r"UNION\s+ALL", token) and not operand:
            operand = True
        elif token == ")" and not operand and depth:
            depth -= 1
        else:
            raise ValueError("research_partition_layout_invalid")
    if operand or depth or not names:
        raise ValueError("research_partition_layout_invalid")
    return names


class PartitionWriter:
    """事务内按字节和行数滚动分段，精确磁盘索引拒绝跨段/跨run重复，窄表不再使用全局ART。"""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        """验证历史键并迁移旧ART，分批登记磁盘索引；例：新增run仍拒绝历史record_id，旧JSON不重写。"""
        self.connection = connection
        self.parts = {
            table: list(names) for table, names in validate_partition_layout(connection).items()
        }
        self.targets: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        self.rows: dict[str, int] = {}
        self.index = DiskKeyIndex()
        try:
            for table in RESEARCH_TABLES:
                if table in self.parts:
                    verify_partition_keys(connection, table, None)
                source = "_keys_" + table if table in self.parts else table
                cursor = connection.execute(f'SELECT record_id FROM research."{source}"')
                while batch := cursor.fetchmany(1024):
                    self.index.add(table, (row[0] for row in batch))
            if not _disk_key_layout(connection):
                for table in self.parts:
                    replacement = "_migrate_keys_" + table
                    connection.execute(
                        f'CREATE TABLE research."{replacement}" (record_id VARCHAR NOT NULL,'
                        "run_id VARCHAR NOT NULL,result_id VARCHAR NOT NULL)"
                    )
                    connection.execute(
                        f'INSERT INTO research."{replacement}" SELECT record_id,run_id,result_id '
                        f'FROM research."_keys_{table}"'
                    )
                    connection.execute(f'DROP TABLE research."_keys_{table}"')
                    connection.execute(
                        f'ALTER TABLE research."{replacement}" RENAME TO "_keys_{table}"'
                    )
                connection.execute(
                    "INSERT INTO research.metadata VALUES ('partition_keys',?)", [_KEY_LAYOUT]
                )
        except BaseException:
            self.index.close()
            raise

    def close(self) -> None:
        """释放私有磁盘索引，主库只保留可独立重建的窄键行；例：保存失败后不留下SQLite侧车。"""
        self.index.close()

    def __enter__(self) -> PartitionWriter:
        """限定索引生命周期；例：with PartitionWriter(connection) as writer退出即清理。"""
        return self

    def __exit__(self, *_: object) -> None:
        """保留原异常并清理私有索引；例：重复键抛错后调用者仍可回滚DuckDB事务。"""
        self.close()

    def _target(self, table: str, size: int, rows: int) -> str:
        """按字节和行数双界限封存旧表/段；例：达到16MiB或16384行后另开段，旧值不重写。"""
        if table not in self.targets:
            if table in self.parts:
                self.targets[table] = ""
                self.sizes[table] = MAXIMUM_PARTITION_BYTES
                self.rows[table] = MAXIMUM_PARTITION_ROWS
            else:
                counted = self.connection.execute(
                    f'SELECT count(*) FROM research."{table}"'
                ).fetchone()
                if counted is None:
                    raise ValueError("research_partition_layout_invalid")
                count = counted[0]
                self.targets[table] = table
                self.sizes[table] = count * ROW_PAYLOAD_BOUND
                self.rows[table] = count
        if (
            self.sizes[table] + size <= MAXIMUM_PARTITION_BYTES
            and self.rows[table] + rows <= MAXIMUM_PARTITION_ROWS
        ):
            return self.targets[table]
        if table not in self.parts:
            first = partition_name(table, 0)
            self.connection.execute(f'ALTER TABLE research."{table}" RENAME TO "{first}"')
            self.connection.execute(
                f'CREATE TABLE research."_keys_{table}" '
                "(record_id VARCHAR NOT NULL,run_id VARCHAR NOT NULL,result_id VARCHAR NOT NULL)"
            )
            self.connection.execute(
                f'INSERT INTO research."_keys_{table}" SELECT record_id,run_id,result_id '
                f'FROM research."{first}"'
            )
            self.parts[table] = [first]
        name = partition_name(table, len(self.parts[table]))
        self.connection.execute(
            f'CREATE TABLE research."{name}" (record_id VARCHAR NOT NULL,run_id VARCHAR NOT NULL,'
            "result_id VARCHAR NOT NULL,payload JSON NOT NULL)"
        )
        self.parts[table].append(name)
        self.connection.execute(
            f'CREATE OR REPLACE VIEW research."{table}" AS {view_select(self.parts[table])}'
        )
        self.targets[table], self.sizes[table] = name, 0
        self.rows[table] = 0
        return name

    def insert(self, table: str, columns: list[list[str]], payload_bytes: int) -> None:
        """先登记全局唯一键再写相同四列，全部受调用者事务保护；例：跨段重复键仍抛ConstraintException。

        输入仅来自_InsertBuffer已公开校验的快照；超段预算拒绝，不通过截断或少写投影绕过。
        """
        if table not in RESEARCH_TABLES:
            raise ValueError("unknown_research_table")
        if payload_bytes > MAXIMUM_PARTITION_BYTES:
            raise ValueError("research_partition_payload_capacity_exceeded")
        if len(columns[0]) > MAXIMUM_PARTITION_ROWS:
            raise ValueError("research_partition_row_capacity_exceeded")
        target = self._target(table, payload_bytes, len(columns[0]))
        self.index.add(table, columns[0])
        if table in self.parts:
            self.connection.execute(
                f'INSERT INTO research."_keys_{table}" SELECT unnest(?::VARCHAR[]),'
                "unnest(?::VARCHAR[]),unnest(?::VARCHAR[])",
                columns[:3],
            )
        self.connection.execute(
            f'INSERT INTO research."{target}" SELECT unnest(?::VARCHAR[]),unnest(?::VARCHAR[]),'
            "unnest(?::VARCHAR[]),CAST(unnest(?::VARCHAR[]) AS JSON)",
            columns,
        )
        self.sizes[table] += payload_bytes
        self.rows[table] += len(columns[0])


def validate_partition_layout(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, tuple[str, ...]]:
    """核对所有保留段、逻辑视图与全局键约束，防止隐藏/改写段内数据；例：视图少一段即拒绝。

    仅接受固定四列SELECT与UNION ALL组成的视图，允许DuckDB重开时平衡联合树的括号。
    普通旧版基础表没有分段，不需要迁移或额外元数据。
    """
    parts = partition_catalog(connection)
    disk_keys = _disk_key_layout(connection)
    rows = connection.execute(
        "SELECT table_name,table_type FROM information_schema.tables "
        "WHERE table_catalog=current_database() AND table_schema='research'",
        [],
    ).fetchall()
    kinds = dict(rows)
    keys = {name.removeprefix("_keys_") for name in kinds if name.startswith("_keys_")}
    if keys != set(parts):
        raise ValueError("research_partition_layout_invalid")
    constraints = {
        name
        for name, columns in connection.execute(
            "SELECT table_name,constraint_column_names FROM duckdb_constraints() "
            "WHERE database_name=current_database() AND schema_name='research' "
            "AND constraint_type='PRIMARY KEY'",
            [],
        ).fetchall()
        if columns == ["record_id"]
    }
    indexed = {
        row[0] for row in connection.execute(
            "SELECT table_name FROM duckdb_indexes() WHERE database_name=current_database() "
            "AND schema_name='research' UNION SELECT table_name FROM duckdb_constraints() "
            "WHERE database_name=current_database() AND schema_name='research' "
            "AND constraint_type IN ('PRIMARY KEY','UNIQUE','FOREIGN KEY')", []
        ).fetchall()
    }
    for table, names in parts.items():
        if (
            kinds.get(table) != "VIEW"
            or kinds.get("_keys_" + table) != "BASE TABLE"
            or (("_keys_" + table in constraints) == disk_keys)
            or (disk_keys and "_keys_" + table in indexed)
            or names != tuple(partition_name(table, number) for number in range(len(names)))
            or any(kinds[name] != "BASE TABLE" for name in names)
        ):
            raise ValueError("research_partition_layout_invalid")
        columns = connection.execute(
            "SELECT column_name,data_type,is_nullable FROM information_schema.columns "
            "WHERE table_catalog=current_database() AND table_schema='research' "
            "AND table_name=? ORDER BY ordinal_position", ["_keys_" + table],
        ).fetchall()
        if columns != [(name, "VARCHAR", "NO") for name in ("record_id", "run_id", "result_id")]:
            raise ValueError("research_partition_layout_invalid")
        definitions = connection.execute(
            "SELECT sql FROM duckdb_views() WHERE database_name=current_database() "
            "AND schema_name='research' AND view_name=?",
            [table],
        ).fetchall()
        if len(definitions) != 1 or _view_parts(definitions[0][0]) != names:
            raise ValueError("research_partition_layout_invalid")
    return parts


def verify_partition_keys(
    connection: duckdb.DuckDBPyConnection, table: str, run_id: str | None
) -> None:
    """先全局精确去重，再由DuckDB核对物理行与窄键一一对应。

    输入 ``table='result', run_id='run'`` 时只比较该批次的三列键；``run_id=None``
    覆盖全部历史行。计数相等且窄键全局唯一时，任一窄键缺少完全相同的物理三元组，
    都能同时检出物理缺行、多行、重复行或元数据变化，而不再逐行搬入SQLite删除。
    """
    duplicate = connection.execute(
        f'SELECT record_id FROM research."_keys_{table}" '
        "GROUP BY record_id HAVING count(*)>1 LIMIT 1", []
    ).fetchone()
    if duplicate is not None:
        raise ValueError("research_primary_key_invalid")
    row = connection.execute(
        f'SELECT (SELECT count(*) FROM research."{table}" WHERE (? IS NULL OR run_id=?)) - '
        f'(SELECT count(*) FROM research."_keys_{table}" WHERE (? IS NULL OR run_id=?))',
        [run_id, run_id, run_id, run_id],
    ).fetchone()
    if row is None or row[0]:
        raise ValueError("research_partition_keys_invalid")
    with TemporaryDirectory(prefix="research-key-buckets-") as temporary:
        roots = []
        for label, source in (("expected", "_keys_" + table), ("stored", table)):
            root = Path(temporary) / label
            connection.execute(
                f'COPY (SELECT record_id,run_id,result_id,hash(record_id)%256 AS bucket '
                f'FROM research."{source}" WHERE ($run IS NULL OR run_id=$run)) '
                "TO $destination (FORMAT CSV, HEADER FALSE, PARTITION_BY (bucket), "
                "OVERWRITE_OR_IGNORE TRUE)",
                {"run": run_id, "destination": str(root)},
            )
            roots.append(root)
        buckets = {item.name for root in roots for item in root.iterdir() if item.is_dir()}
        for bucket in buckets:
            sides = []
            for root in roots:
                records: list[tuple[str, ...]] = []
                directory = root / bucket
                if directory.is_dir():
                    for path in directory.glob("*.csv"):
                        with path.open(encoding="utf-8", newline="") as stream:
                            records.extend(
                                tuple(record) for record in csv.reader(stream, strict=True)
                            )
                sides.append(sorted(records))
            if sides[0] != sides[1]:
                raise ValueError("research_partition_keys_invalid")
