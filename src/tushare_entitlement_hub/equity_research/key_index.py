"""私有写入阶段的精确磁盘唯一键登记，不把全量ART索引留在DuckDB预算中。"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import duckdb

from .projection import RESEARCH_TABLES

KEY_CACHE_KIB = 8192


class DiskKeyIndex:
    """用SQLite磁盘B-tree维护精确的(table, record_id)唯一性，私有目录退出时删除。"""

    def __init__(self) -> None:
        """创建0700临时目录及8MiB页缓存，关闭mmap；例：百万键只扩大磁盘文件，不扩大页缓存。"""
        self.temporary = TemporaryDirectory(prefix="research-unique-keys-")
        try:
            self.connection = sqlite3.connect(Path(self.temporary.name) / "keys.sqlite")
            self.connection.execute(f"PRAGMA cache_size=-{KEY_CACHE_KIB}")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute(
                "CREATE TABLE keys(source INTEGER NOT NULL, record_id TEXT NOT NULL, "
                "PRIMARY KEY(source,record_id)) WITHOUT ROWID"
            )
            self.connection.execute("BEGIN")
        except BaseException:
            self.close()
            raise

    def add(self, table: str, identities: Iterable[str]) -> None:
        """用保存点精确登记一批键，不逐批COMMIT；例：另一run重用record_id时原子撤销该批并拒绝。"""
        source = RESEARCH_TABLES.index(table)
        self.connection.execute("SAVEPOINT key_batch")
        try:
            self.connection.executemany(
                "INSERT INTO keys VALUES (?,?)", ((source, identity) for identity in identities)
            )
        except sqlite3.IntegrityError:
            self.connection.execute("ROLLBACK TO key_batch")
            self.connection.execute("RELEASE key_batch")
            raise duckdb.ConstraintException("research_primary_key_invalid") from None
        self.connection.execute("RELEASE key_batch")

    def close(self) -> None:
        """关闭连接回滚私有登记并删除索引，允许重复清理；例：DuckDB提交失败不保留该侧车。"""
        connection = getattr(self, "connection", None)
        try:
            if connection is not None:
                connection.close()
        finally:
            self.temporary.cleanup()

    def __enter__(self) -> Self:
        """提供显式资源范围；例：with DiskKeyIndex() as index在退出后不留下磁盘键库。"""
        return self

    def __exit__(self, *_: object) -> None:
        """无论正常还是异常都清理，绝不抑制原异常；例：重复键错误传播给保存阶段。"""
        self.close()


class DiskKeyMatcher(DiskKeyIndex):
    """使用相同有界缓存逐条核销三列元数据，避免全量UNION与窄键FULL JOIN。"""

    def __init__(self) -> None:
        """创建精确三列磁盘登记；例：每条物理行必须消耗一条完全相同的登记行。"""
        super().__init__()
        try:
            self.connection.execute(
                "CREATE TABLE records(record_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,"
                "result_id TEXT NOT NULL) WITHOUT ROWID"
            )
        except BaseException:
            self.close()
            raise

    def load_csv(self, path: Path) -> None:
        """分批载入DuckDB流式导出的三列CSV，字段逐字保留；例：跨run同record_id仍拒绝，格式损坏失败。"""
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                while batch := list(islice(reader, 1024)):
                    self.connection.executemany("INSERT INTO records VALUES (?,?,?)", batch)
        except sqlite3.IntegrityError:
            raise ValueError("research_primary_key_invalid") from None
        except (csv.Error, sqlite3.ProgrammingError):
            raise ValueError("research_partition_keys_invalid") from None

    def match(self, rows: list[tuple[str, str, str]]) -> None:
        """每条物理元数据精确删除一条登记，重复或任一字段不符均失败；例：重复物理行第二次匹配为零。"""
        result = self.connection.executemany(
            "DELETE FROM records WHERE record_id=? AND run_id=? AND result_id=?", rows
        )
        if result.rowcount != len(rows):
            raise ValueError("research_partition_keys_invalid")

    def finish(self) -> None:
        """要求所有登记均已被唯一匹配，缺计数也拒绝；例：窄键表比物理表多一行时失败。"""
        row = self.connection.execute("SELECT count(*) FROM records").fetchone()
        if row is None or row[0]:
            raise ValueError("research_partition_keys_invalid")
