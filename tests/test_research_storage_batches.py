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
