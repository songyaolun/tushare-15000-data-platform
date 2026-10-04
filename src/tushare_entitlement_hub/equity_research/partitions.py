"""宽JSON的有界物理分段，保留七个逻辑投影的列、值和跨段唯一键。"""

from __future__ import annotations

import re

import duckdb

from .projection import RESEARCH_TABLES

MAXIMUM_PARTITION_BYTES = 16 * 1024 * 1024
MAXIMUM_PARTITIONS = 4096
ROW_PAYLOAD_BOUND = 8 * 1024 * 1024
_COLUMNS = "record_id,run_id,result_id,payload"


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
    """事务内按实际JSON字节滚动分段，已有段封存；键另存窄表防止跨段重复。"""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        """加载段目录，不扫描历史JSON；旧表按每行8MiB上界保守计预算，例：旧大表在首次追加前封存。"""
        self.connection = connection
        self.parts = {
            table: list(names) for table, names in validate_partition_layout(connection).items()
        }
        self.targets: dict[str, str] = {}
        self.sizes: dict[str, int] = {}

    def _target(self, table: str, size: int) -> str:
        """在超预算前封存旧表/段并创建下一段，旧行不复制、不重写；例：16MiB段不会再接收下一行。"""
        if table not in self.targets:
            if table in self.parts:
                self.targets[table] = ""
                self.sizes[table] = MAXIMUM_PARTITION_BYTES
            else:
                counted = self.connection.execute(
                    f'SELECT count(*) FROM research."{table}"'
                ).fetchone()
                if counted is None:
                    raise ValueError("research_partition_layout_invalid")
                count = counted[0]
                self.targets[table] = table
                self.sizes[table] = count * ROW_PAYLOAD_BOUND
        if self.sizes[table] + size <= MAXIMUM_PARTITION_BYTES:
            return self.targets[table]
        if table not in self.parts:
            first = partition_name(table, 0)
            self.connection.execute(f'ALTER TABLE research."{table}" RENAME TO "{first}"')
            self.connection.execute(
                f'CREATE TABLE research."_keys_{table}" '
                "(record_id VARCHAR PRIMARY KEY,run_id VARCHAR NOT NULL,result_id VARCHAR NOT NULL)"
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
        return name

    def insert(self, table: str, columns: list[list[str]], payload_bytes: int) -> None:
        """先登记全局唯一键再写相同四列，全部受调用者事务保护；例：跨段重复键仍抛ConstraintException。

        输入仅来自_InsertBuffer已公开校验的快照；超段预算拒绝，不通过截断或少写投影绕过。
        """
        if table not in RESEARCH_TABLES:
            raise ValueError("unknown_research_table")
        if payload_bytes > MAXIMUM_PARTITION_BYTES:
            raise ValueError("research_partition_payload_capacity_exceeded")
        target = self._target(table, payload_bytes)
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


def validate_partition_layout(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, tuple[str, ...]]:
    """核对所有保留段、逻辑视图与全局键约束，防止隐藏/改写段内数据；例：视图少一段即拒绝。

    仅接受固定四列SELECT与UNION ALL组成的视图，允许DuckDB重开时平衡联合树的括号。
    普通旧版基础表没有分段，不需要迁移或额外元数据。
    """
    parts = partition_catalog(connection)
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
    for table, names in parts.items():
        if (
            kinds.get(table) != "VIEW"
            or "_keys_" + table not in constraints
            or names != tuple(partition_name(table, number) for number in range(len(names)))
            or any(kinds[name] != "BASE TABLE" for name in names)
        ):
            raise ValueError("research_partition_layout_invalid")
        definitions = connection.execute(
            "SELECT sql FROM duckdb_views() WHERE database_name=current_database() "
            "AND schema_name='research' AND view_name=?",
            [table],
        ).fetchall()
        if len(definitions) != 1 or _view_parts(definitions[0][0]) != names:
            raise ValueError("research_partition_layout_invalid")
    return parts


def verify_partition_keys(connection: duckdb.DuckDBPyConnection, table: str, run_id: str) -> None:
    """用窄元数据核对物理行与全局主键的一一对应，不扫描大payload；例：多行共用一个键或键缺行均拒绝。"""
    row = connection.execute(
        f'SELECT (SELECT count(*) FROM research."{table}" WHERE run_id=?) - '
        f'(SELECT count(*) FROM research."_keys_{table}" WHERE run_id=?)',
        [run_id, run_id],
    ).fetchone()
    if row is None or row[0]:
        raise ValueError("research_partition_keys_invalid")
    row = connection.execute(
        "SELECT count(*) FROM "
        f'(SELECT record_id,run_id,result_id FROM research."{table}" WHERE run_id=?) data '
        "FULL JOIN "
        f'(SELECT record_id,run_id,result_id FROM research."_keys_{table}" WHERE run_id=?) keys '
        "USING(record_id,run_id,result_id) "
        "WHERE data.record_id IS NULL OR keys.record_id IS NULL",
        [run_id, run_id],
    ).fetchone()
    if row is None or row[0]:
        raise ValueError("research_partition_keys_invalid")
