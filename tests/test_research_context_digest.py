"""定点验证摘要不随数据量保留哈希、精确集合溢写，以及临时资源异常清理。"""

import random
import sqlite3
import tracemalloc
from collections import Counter
from contextlib import closing
from hashlib import sha256

import pytest
from test_research_context_consumers import INDEX, context_inputs, context_row

from tushare_entitlement_hub.equity_research import context_aggregation as aggregation
from tushare_entitlement_hub.equity_research import context_digest
from tushare_entitlement_hub.equity_research.context_consumers import consume_context
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY


def multiset_fingerprint(values):
    """流式输入测试叶摘要并返回最终值；例如[A,B,A]与[A,A,B]结果相同。"""
    result = context_digest.MultisetDigest()
    for value in values:
        result.add(value)
    return result.hexdigest()


def test_multiset_hash_is_order_independent_and_preserves_duplicate_count():
    """排列不影响摘要，增加重复项、移除尾项或修改叶值必须改变摘要，空集合也有稳定值。"""
    values = [sha256(str(i).encode()).hexdigest() for i in range(100)]
    values += values[:10]
    shuffled = list(values)
    random.Random(42).shuffle(shuffled)
    assert multiset_fingerprint(values) == multiset_fingerprint(reversed(values))
    assert multiset_fingerprint(values) == multiset_fingerprint(shuffled)
    assert multiset_fingerprint(values) != multiset_fingerprint(values[:-1])
    assert multiset_fingerprint(values) != multiset_fingerprint(values + [values[0]])
    assert multiset_fingerprint(values) != multiset_fingerprint(["0" * 64] + values[1:])
    assert multiset_fingerprint([]) == multiset_fingerprint(iter(()))


def test_million_leaf_digest_keeps_constant_python_state():
    """百万个不同叶摘要全部累计，Python峰值小于1MiB且最终仅保留三个整数，不存叶列表。"""
    value = context_digest.MultisetDigest()
    tracemalloc.start()
    try:
        for index in range(1_000_000):
            value.add(f"{index:064x}")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert value.count == 1_000_000
    assert set(vars(value)) == {"count", "total", "squares"}
    assert value.total.bit_length() <= 256 and value.squares.bit_length() <= 256
    assert peak < 1024 * 1024
    print(f"百万叶摘要Python峰值：{peak}字节")


def test_exact_set_spills_beyond_cache_and_deduplicates_full_bytes():
    """十万个不同引用键超过2MiB缓存后保留精确计数，重复键不增加记录，命名空间不混合。"""
    with closing(context_digest.DiskKeySet()) as keys:
        tracemalloc.start()
        try:
            for index in range(100_000):
                key = index.to_bytes(8) + b"reference" * 24
                assert keys.add("reference", key)
                if index % 1000 == 0:
                    assert not keys.add("reference", key)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        first = bytes(8) + b"reference" * 24
        assert keys.contains("reference", first)
        assert not keys.contains("local", first)
        assert keys.add("local", first)
        assert keys.connection.execute(
            "SELECT count(*) FROM keys WHERE namespace='reference'"
        ).fetchone() == (100_000,)
        assert keys.connection.execute("PRAGMA cache_size").fetchone() == (-2048,)
        assert keys.connection.execute("PRAGMA temp_store").fetchone() == (1,)
        pages = keys.connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = keys.connection.execute("PRAGMA page_size").fetchone()[0]
        assert pages * page_size > context_digest.SET_CACHE_KIB * 1024
        assert keys.pending < context_digest.SET_COMMIT_INTERVAL
        assert peak < 1024 * 1024
        print(f"十万精确键：临时页{pages * page_size}字节，Python峰值{peak}字节")
    keys.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        keys.connection.execute("SELECT 1")


@pytest.mark.parametrize("stage", ["open", "setup", "query"])
def test_spill_failures_are_fixed_errors_and_setup_closes_connection(monkeypatch, stage):
    """打开或使用临时库失败时给固定错误；初始化中途失败会关闭连接，不泄露键内容。"""
    if stage == "open":

        def cannot_open(*args, **kwargs):
            """模拟SQLite临时库打开失败；输入任意连接参数，输出固定测试异常。"""
            raise sqlite3.OperationalError("synthetic open failure")

        monkeypatch.setattr(context_digest.sqlite3, "connect", cannot_open)
        with pytest.raises(ValueError, match="^research_context_spill_failed$"):
            context_digest.DiskKeySet()
    elif stage == "setup":

        class BrokenSetup:
            """模拟已打开但配置失败的连接，并记录close是否执行。"""

            closed = False

            def execute(self, *args):
                """任何配置SQL均失败，用于验证初始化清理和固定错误转换。"""
                raise sqlite3.OperationalError("synthetic setup failure")

            def close(self):
                """关闭标记由False变为True，不执行真实数据库操作。"""
                self.closed = True

        connection = BrokenSetup()
        monkeypatch.setattr(context_digest.sqlite3, "connect", lambda *args, **kwargs: connection)
        with pytest.raises(ValueError, match="^research_context_spill_failed$"):
            context_digest.DiskKeySet()
        assert connection.closed
    else:
        keys = context_digest.DiskKeySet()
        keys.close()
        with pytest.raises(ValueError, match="^research_context_spill_failed$"):
            keys.contains("reference", b"synthetic")


@pytest.mark.parametrize("stage", ["normal", "construction", "add", "finish"])
def test_consumer_releases_spill_store_on_every_exit(monkeypatch, stage):
    """上下文消费者构建、逐组处理、完成阶段的异常及正常返回都关闭私有临时库。"""
    stores = []
    original = aggregation.DiskKeySet

    def tracked_store():
        """创建真实临时集合并保留连接用于关闭后断言，不改变集合行为。"""
        store = original()
        stores.append(store)
        if stage == "construction":
            monkeypatch.setattr(store, "add", interrupted)
        return store

    def interrupted(*args, **kwargs):
        """模拟本阶段被中断；调用者必须清理资源后原样传播测试异常。"""
        raise ValueError("synthetic interruption")

    monkeypatch.setattr(aggregation, "DiskKeySet", tracked_store)
    if stage in {"add", "finish"}:
        monkeypatch.setattr(aggregation.ContextAggregate, stage, interrupted)
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    inputs = context_inputs(row)
    if stage == "normal":
        result = consume_context(inputs, "P01", "index_basic", (row,) * 65, MARKET_ENTITY)
        assert result["facts"]["aggregation"]["total_rows"] == 65
    else:
        with pytest.raises(ValueError, match="^synthetic interruption$"):
            consume_context(inputs, "P01", "index_basic", (row,) * 65, MARKET_ENTITY)
    assert len(stores) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        stores[0].connection.execute("SELECT 1")


def test_row_fingerprints_do_not_materialize_all_leaves(monkeypatch):
    """五万条输入逐条进入指纹，聚合器构造Python峰值小于2MiB，不物化或排序全量叶哈希。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    rows = (row,) * 50_000
    processed = 0
    original = aggregation.row_fingerprint

    def counted(item, fields):
        """透传真实叶摘要函数并计数；例如每调用一次processed加1。"""
        nonlocal processed
        processed += 1
        return original(item, fields)

    monkeypatch.setattr(aggregation, "row_fingerprint", counted)
    tracemalloc.start()
    try:
        with aggregation.ContextAggregate(
            "index_basic", rows, rows, ("ts_code",), row.availability.observed_at
        ) as result:
            assert result.total_rows == 50_000
            assert not hasattr(result, "group_hashes") and not hasattr(result, "reference_hashes")
            assert not hasattr(result, "local_fields")
            _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert processed == 50_000
    assert peak < 2 * 1024 * 1024
    print(f"五万行构造Python峰值：{peak}字节")


@pytest.mark.parametrize("reverse", [False, True])
def test_bounded_metadata_counts_match_exact_frequency_table(reverse):
    """只留16个展示计数仍保留精确基数及other，逆序输入的完整摘要与正序相同。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    values = [f"publisher-{index:03}" for index in range(50) for _ in range(index % 5 + 1)]
    expected = Counter(values)
    summary = []
    for incoming in (values, list(reversed(values)) if reverse else values[::2] + values[1::2]):
        with aggregation.ContextAggregate(
            "index_basic", (row,), (row,), ("ts_code",), row.availability.observed_at
        ) as result:
            for value in incoming:
                result.add(
                    (value,),
                    1,
                    {
                        "status": "observed",
                        "reason_codes": (),
                        "facts": {"publisher": value},
                        "evidence_refs": row.evidence("ts_code"),
                    },
                )
                assert len(result.metadata["publisher"]) <= 16
            summary.append(result.finish()["facts"]["aggregation"])
    assert summary[0] == summary[1]
    counts = summary[0]["metadata_availability"]["publisher"]
    selected = sorted(expected)[:16]
    assert counts["distinct_values"] == 50
    assert counts["available_groups"] == len(values)
    assert counts["values"] == [{"value": key, "groups": expected[key]} for key in selected]
    assert counts["other_groups"] == sum(expected[key] for key in expected if key not in selected)


def test_group_and_reference_state_remains_bounded_for_many_unique_groups():
    """五万个不同业务组与元数据分类全部计数，Python聚合状态保持有界，重复证据只计一次。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    references = row.evidence("ts_code")
    with aggregation.ContextAggregate(
        "index_basic", (row,), (row,), ("ts_code",), row.availability.observed_at
    ) as result:
        tracemalloc.start()
        try:
            for index in range(50_000):
                result.add(
                    (index,),
                    1,
                    {
                        "status": "observed",
                        "reason_codes": (),
                        "facts": {"publisher": f"P{index:05}"},
                        "evidence_refs": references,
                    },
                )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        summary = result.finish()["facts"]["aggregation"]
        assert summary["group_count"] == 50_000
        assert result.group_digest.count == 50_000
        assert summary["candidate_reference_count"] == summary["sample_count"] == 1
        assert summary["metadata_availability"]["publisher"]["distinct_values"] == 50_000
        assert len(result.metadata["publisher"]) == 16
        assert peak < 2 * 1024 * 1024
        print(f"五万组累计Python峰值：{peak}字节")
