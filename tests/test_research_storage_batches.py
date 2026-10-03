"""双预算写入的真实DuckDB边界验证，限制线程1和内存512MB。"""

import json
from copy import deepcopy

import duckdb
import pytest
from research_fixture import DATABASE_CONFIG

from tushare_entitlement_hub.equity_research import storage
from tushare_entitlement_hub.store import encode


class RecordingConnection:
    """记录真实插入批次并委托DuckDB执行；例：两行插入记录为四列各两个参数。"""

    def __init__(self, connection):
        """绑定现有临时数据库并初始化插入记录；例：新对象的batches为空。"""
        self.connection = connection
        self.batches = []

    def execute(self, query, parameters):
        """复制即将消费的参数并执行真实SQL；例：flush清空原列表后仍能检查原始批次。"""
        self.batches.append(deepcopy(parameters))
        return self.connection.execute(query, parameters)


def storage_rows(texts):
    """创建带稳定主键的不同载荷；例：['甲','乙']生成row-0、row-1，文字不改写。"""
    return [
        {
            "record_id": f"row-{index}",
            "run_id": "synthetic-run",
            "result_id": "synthetic-result",
            "payload": {"text": text},
        }
        for index, text in enumerate(texts)
    ]


def verify_written_rows(connection, expected):
    """核对数据库中每个主键及完整JSON载荷；例：重复、丢行或截断任一发生均失败。"""
    actual = connection.execute(
        'SELECT record_id,run_id,result_id,payload FROM research."result" ORDER BY record_id'
    ).fetchall()
    assert actual == [
        (item["record_id"], item["run_id"], item["result_id"], encode(item["payload"]))
        for item in sorted(expected, key=lambda item: item["record_id"])
    ]
    assert [json.loads(item[3]) for item in actual] == [
        item["payload"] for item in sorted(expected, key=lambda item: item["record_id"])
    ]


@pytest.mark.parametrize(
    "texts,expected_batches",
    [
        ([], []),
        (["x"], [1]),
        (["x", "x"], [2]),
        (["x", "x", "x"], [2, 1]),
        (["超长字符" * 20], [1]),
        (["x", "超长字符" * 20, "x"], [1, 1, 1]),
        (["中文", "中文"], [1, 1]),
    ],
)
def test_json_byte_budget_exact_boundary_overflow_and_utf8(texts, expected_batches, monkeypatch):
    """场景：空批、恰好字节预算、跨预算、超长单行和UTF8多字节；预期：分批准确且载荷完整。"""
    budget = 2 * len(encode({"text": "x"}).encode("utf-8"))
    monkeypatch.setattr(storage, "MAXIMUM_INSERT_JSON_BYTES", budget)
    expected = storage_rows(texts)
    with duckdb.connect(":memory:", config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        recording = RecordingConnection(connection)
        storage.insert_rows(recording, "result", expected)
        assert [len(batch[0]) for batch in recording.batches] == expected_batches
        assert all(len({len(column) for column in batch}) == 1 for batch in recording.batches)
        assert all(
            sum(len(payload.encode("utf-8")) for payload in batch[3]) <= budget
            or len(batch[0]) == 1
            for batch in recording.batches
        )
        verify_written_rows(connection, expected)
        assert connection.execute(
            "SELECT current_setting('preserve_insertion_order')"
        ).fetchone() == (False,)


@pytest.mark.parametrize("count", [0, 1, 127, 128, 129, 256, 257, 512])
def test_row_budget_preserves_exact_boundaries_and_final_tail(count):
    """场景：128行上限前后及512行输入；预期：每批最多128行，尾批准确且全部记录仅写一次。"""
    expected = storage_rows(["small"] * count)
    with duckdb.connect(":memory:", config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        recording = RecordingConnection(connection)
        storage.insert_rows(recording, "result", expected)
        full, tail = divmod(count, 128)
        assert [len(batch[0]) for batch in recording.batches] == [128] * full + (
            [tail] if tail else []
        )
        verify_written_rows(connection, expected)


def test_insert_failure_rolls_back_previously_flushed_batches(monkeypatch):
    """场景：跨批后最后一批主键重复；预期：外层事务回滚全部已刷写记录，不残留半成品。"""
    monkeypatch.setattr(storage, "MAXIMUM_INSERT_ROWS", 1)
    expected = storage_rows(["first", "second"])
    with duckdb.connect(":memory:", config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        with pytest.raises(duckdb.ConstraintException, match="duplicate key"):
            storage.insert_rows(connection, "result", expected + [expected[0]])
        connection.execute("ROLLBACK")
        assert connection.execute('SELECT COUNT(*) FROM research."result"').fetchone() == (0,)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_result_payload_eight_mib_boundary_before_sql(offset):
    """单结果8MiB边界前后按UTF8精确计量；超限时SQL从未执行，合法载荷完整持久化。"""
    assert storage.MAXIMUM_RESULT_JSON_BYTES == 8 * 1024 * 1024
    overhead = len(encode({"text": ""}).encode())
    rows = storage_rows(["x" * (storage.MAXIMUM_RESULT_JSON_BYTES - overhead + offset)])
    with duckdb.connect(":memory:", config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        recording = RecordingConnection(connection)
        if offset > 0:
            with pytest.raises(ValueError, match="^research_result_payload_capacity_exceeded$"):
                storage.insert_rows(recording, "result", rows)
            assert recording.batches == []
            assert connection.execute("SELECT count(*) FROM research.result").fetchone() == (0,)
        else:
            storage.insert_rows(recording, "result", rows)
            verify_written_rows(connection, rows)


def test_result_capacity_counts_utf8_and_preserves_outer_transaction(monkeypatch):
    """中文按UTF8字节而非字符计量；超限后的外层回滚清除先前已写结果，不截断任何载荷。"""
    monkeypatch.setattr(storage, "MAXIMUM_RESULT_JSON_BYTES", len(encode({"text": "中"}).encode()))
    with duckdb.connect(":memory:", config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        storage.insert_rows(connection, "result", storage_rows(["中"]))
        with pytest.raises(ValueError, match="^research_result_payload_capacity_exceeded$"):
            storage.insert_rows(connection, "result", storage_rows(["中文"]))
        connection.execute("ROLLBACK")
        assert connection.execute("SELECT count(*) FROM research.result").fetchone() == (0,)


def test_cli_exposes_only_fixed_result_capacity_error(monkeypatch, tmp_path):
    """存储容量错误沿research run返回固定原因码和退出1，不回显超大载荷或底层异常。"""
    from typer.testing import CliRunner

    from tushare_entitlement_hub.cli import app
    from tushare_entitlement_hub.equity_research import cli

    def oversized_result(*args, **kwargs):
        """模拟存储容量门禁；任意运行参数均抛固定容量错误，不产生DuckDB或报告。"""
        raise ValueError(storage.RESULT_PAYLOAD_CAPACITY_ERROR)

    monkeypatch.setattr(cli, "run_research", oversized_result)
    result = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            "2026-10-03T12:00:00Z",
            "--product",
            "P01",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": ["research_result_payload_capacity_exceeded"],
    }


def test_unknown_storage_table_rejected_without_sql():
    """表名白名单在任何SQL或容量处理前拒绝未知表；无需建立连接即可验证边界。"""
    with pytest.raises(ValueError, match="^unknown_research_table$"):
        storage.insert_rows(None, "result; DROP SCHEMA research", [])
