"""磁盘唯一键登记的精确性、批回滚、固定缓存及私有资源清理。"""

import csv
import sqlite3
from pathlib import Path

import duckdb
import pytest

from tushare_entitlement_hub.equity_research import key_index
from tushare_entitlement_hub.equity_research.persistence_runtime import (
    PersistenceStepError,
    database_error_code,
)


def test_disk_keys_are_exact_across_batches_and_namespaces():
    """跨批同表重复必须失败，批内先插入的新键需撤销；不同表可复用键，缓存固定且不逐批COMMIT。"""
    with key_index.DiskKeyIndex() as index:
        root = Path(index.temporary.name)
        commands = []
        index.connection.set_trace_callback(commands.append)
        index.add("result", ["same"])
        with pytest.raises(duckdb.ConstraintException, match="research_primary_key_invalid"):
            index.add("result", ["new", "same"])
        index.add("result", ["new"])
        index.add("feature", ["same"])
        assert index.connection.execute("SELECT count(*) FROM keys").fetchone() == (3,)
        assert index.connection.execute("PRAGMA cache_size").fetchone() == (-8192,)
        assert index.connection.execute("PRAGMA mmap_size").fetchone() == (0,)
        assert index.connection.execute("PRAGMA temp_store").fetchone() == (1,)
        assert index.connection.in_transaction
        assert not any(command == "COMMIT" for command in commands)
    assert not root.exists()
    index.close()


@pytest.mark.parametrize("stage", ["connect", "configure", "interrupt"])
def test_disk_index_initialization_failure_always_removes_private_directory(monkeypatch, stage):
    """连接失败、配置失败及初始化中断均清理私有目录；已打开的SQLite连接也必须关闭。"""
    directories, closed = [], []
    temporary, connect = key_index.TemporaryDirectory, sqlite3.connect

    def make_directory(*args, **kwargs):
        """记录本次私有目录；例：失败后该路径必须不存在，不枚举或清理其他任务的目录。"""
        result = temporary(*args, **kwargs)
        directories.append(Path(result.name))
        return result

    class BrokenConfiguration:
        """提供真实连接的关闭能力，只在第一个配置语句注入故障。"""

        def __init__(self, path):
            """打开真实私有数据库；例：配置失败时不能遗留连接或文件。"""
            self.connection = connect(path)

        def execute(self, query):
            """模拟普通错误或键盘中断；例：两种异常都必须触发构造器清理。"""
            if stage == "interrupt":
                raise KeyboardInterrupt()
            raise sqlite3.OperationalError("PRIVATE")

        def close(self):
            """关闭底层连接并记录；例：一次失败应恰好关闭已经打开的连接。"""
            self.connection.close()
            closed.append(True)

    def opening(path):
        """连接前或配置时失败；例：connect失败时无需关闭尚不存在的连接。"""
        if stage == "connect":
            raise sqlite3.OperationalError("PRIVATE")
        return BrokenConfiguration(path)

    monkeypatch.setattr(key_index, "TemporaryDirectory", make_directory)
    monkeypatch.setattr(sqlite3, "connect", opening)
    with pytest.raises(KeyboardInterrupt if stage == "interrupt" else sqlite3.OperationalError):
        key_index.DiskKeyIndex()
    assert directories and all(not path.exists() for path in directories)
    assert len(closed) == int(stage != "connect")


@pytest.mark.parametrize(
    "code,expected,category",
    [
        (sqlite3.SQLITE_FULL, "sqlite_full", "io"),
        (sqlite3.SQLITE_IOERR | (1 << 8), "sqlite_io", "io"),
        (sqlite3.SQLITE_CANTOPEN, "sqlite_io", "io"),
        (sqlite3.SQLITE_NOMEM, "sqlite_memory", "memory"),
        (sqlite3.SQLITE_CORRUPT, "sqlite_error", "database"),
    ],
)
def test_sqlite_errors_use_numeric_codes_without_exposing_messages(code, expected, category):
    """磁盘索引出错只暴露原生数值码的固定分类，扩展IO码也能识别，私有路径不会进入错误文本。"""
    error = sqlite3.OperationalError("PRIVATE_PATH_AND_VALUES")
    error.sqlite_errorcode = code
    assert database_error_code(error) == expected
    failure = PersistenceStepError("commit", error)
    assert failure.category == category
    assert "PRIVATE" not in str(failure)


@pytest.mark.parametrize("damage", [None, "duplicate", "metadata", "extra", "csv", "columns"])
def test_disk_metadata_matching_is_exact_and_bounded(tmp_path, damage):
    """元数据通过CSV及磁盘B-tree逐字匹配；重复、错字段、额外登记、坏CSV或列数均拒绝。"""
    path = tmp_path / "metadata.csv"
    rows = [("id,quoted", 'run\n"quoted"', "parent"), ("other", "run", "parent")]
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows + [rows[0]] if damage == "duplicate" else rows)
    if damage == "csv":
        path.write_text('"unfinished', encoding="utf-8")
    elif damage == "columns":
        path.write_text("two,columns\n", encoding="utf-8")
    with key_index.DiskKeyMatcher() as matcher:
        if damage in {"duplicate", "csv", "columns"}:
            with pytest.raises(ValueError, match="research_(primary_key|partition_keys)_invalid"):
                matcher.load_csv(path)
            return
        matcher.load_csv(path)
        if damage == "metadata":
            with pytest.raises(ValueError, match="research_partition_keys_invalid"):
                matcher.match([(rows[0][0], "wrong_run", rows[0][2])])
            return
        matcher.match(rows[:1] if damage == "extra" else rows)
        if damage == "extra":
            with pytest.raises(ValueError, match="research_partition_keys_invalid"):
                matcher.finish()
        else:
            matcher.finish()


def test_disk_matcher_constructor_failure_cleans_up(monkeypatch):
    """元数据表创建失败发生在基类初始化之后，仍必须关闭连接并删除整个私有目录。"""
    initialize = key_index.DiskKeyIndex.__init__
    paths = []

    def conflict(self):
        """预先创建同名元数据表触发真实SQLite错误；例：子类初始化不能泄漏已建立的磁盘索引。"""
        initialize(self)
        paths.append(Path(self.temporary.name))
        self.connection.execute("CREATE TABLE records(i INT)")

    monkeypatch.setattr(key_index.DiskKeyIndex, "__init__", conflict)
    with pytest.raises(sqlite3.OperationalError):
        key_index.DiskKeyMatcher()
    assert paths and all(not path.exists() for path in paths)


def test_disk_matcher_missing_final_count_fails_closed():
    """最终核销统计未返回时不能默认为零，仍报告固定物理键错误。"""
    with key_index.DiskKeyMatcher() as matcher:
        original = matcher.connection

        class MissingCount:
            """只替换最终统计，不改动实际磁盘资源的生命周期。"""

            def execute(self, query):
                """返回缺失统计游标；例：finish必须拒绝未知计数。"""
                return self

            def fetchone(self):
                """模拟未取得计数；例：None不能当作所有键已经核销。"""
                return None

        matcher.connection = MissingCount()
        try:
            with pytest.raises(ValueError, match="research_partition_keys_invalid"):
                matcher.finish()
        finally:
            matcher.connection = original
