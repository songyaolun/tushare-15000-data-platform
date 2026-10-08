"""批大小硬上限、读写隔离和事实缓冲的离线回归，测试先于修复编写。"""

import sys
from collections.abc import Iterator
from dataclasses import replace

import duckdb
import pytest
from test_product_batching import bulk_reference
from test_product_foundation import NOW
from test_product_storage import bindings

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.product_foundation.batch_io import batches
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.reference_scan import consume_rows, reference_rows
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.product_foundation.storage import ProductStore
from tushare_entitlement_hub.reference_contract import load_reference_contracts

INVALID_BATCH_SIZES = (0, -1, True, False, 1.5, 513, 1_000_000, sys.maxsize + 1)


class UnreadableRows:
    """只用于证明非法批大小不会启动输入迭代。"""

    def __iter__(self) -> Iterator:
        """拒绝任何消费；输入 iter(rows)，输出断言失败，合法实现应提前报批大小错误。"""
        raise AssertionError("非法批大小不应开始消费输入")


class UnavailableConnection:
    """只用于证明非法批大小不会执行数据库操作。"""

    def execute(self, *args, **kwargs):
        """拒绝任何 SQL；输入 SELECT，输出断言失败，非法批大小应更早被拒绝。"""
        raise AssertionError("非法批大小不应执行 SQL")


class FetchRecorder:
    """透传真实连接及关系结果，记录实际传给 fetchmany 的大小。"""

    def __init__(self, target, sizes):
        """保存真实对象及共享列表；输入连接和空列表，输出可透传的记录器。"""
        self.target = target
        self.sizes = sizes

    def execute(self, query, *args, **kwargs):
        """真实执行连接查询并继续包装游标；输入 SELECT，输出自身以记录后续读取。"""
        self.target.execute(query, *args, **kwargs)
        return self

    def sql(self, query, *args, **kwargs):
        """包装真实惰性关系；输入 SELECT，输出记录 fetchmany 的关系包装器。"""
        return FetchRecorder(self.target.sql(query, *args, **kwargs), self.sizes)

    def fetchmany(self, size):
        """记录并透传本次读取大小；输入128，输出真实的至多128行且列表增加128。"""
        self.sizes.append(size)
        return self.target.fetchmany(size)

    def __getattr__(self, name):
        """将未包装能力交还底层；输入 close，输出真实关闭方法。"""
        return getattr(self.target, name)


@pytest.fixture(scope="module")
def daily_context():
    """读取离线登记与契约，使用两个数值字段触发一行多事实。

    输入无参数，输出 daily 消费者及契约；不读取数据载荷或访问网络。
    """
    entry = next(item for item in load_registry() if item.api == "daily")
    contract = load_reference_contracts(load_policies())["daily"]
    return replace(entry, selected_fields=("amount", "close")), contract


@pytest.mark.parametrize("size", INVALID_BATCH_SIZES)
def test_policy_rejects_invalid_and_oversized_batches(size):
    """场景：配置为非正整数或超过512；预期：配置构造与JSON载入都拒绝。

    输入513、百万或超出系统整数范围的批大小，输出ValueError，不分配对应大小对象。
    """
    with pytest.raises(ValueError):
        FoundationPolicy(version="batch-limits-v1", fact_batch_size=size)
    with pytest.raises(ValueError):
        FoundationPolicy.from_dict({"version": "batch-limits-v1", "fact_batch_size": size})


@pytest.mark.parametrize("size", INVALID_BATCH_SIZES)
def test_batches_reject_invalid_size_before_consuming_input(size):
    """场景：直接调用分批器并传入非法大小；预期：在输入生成器开始前拒绝。

    输入不可迭代探针和百万级大小，输出invalid_batch_size，不触发探针断言。
    """
    with pytest.raises(ValueError, match="^invalid_batch_size$"):
        list(batches(UnreadableRows(), size))


@pytest.mark.parametrize("size", INVALID_BATCH_SIZES)
def test_reference_rows_reject_invalid_size_before_sql(size):
    """场景：绕过配置直接读来源；预期：非法大小在SQL和fetchmany之前拒绝。

    输入513或超大整数，输出invalid_batch_size，连接探针从不执行。
    """
    with pytest.raises(ValueError, match="^invalid_batch_size$"):
        list(reference_rows(UnavailableConnection(), "daily", batch_size=size))


@pytest.mark.parametrize("size", INVALID_BATCH_SIZES)
def test_store_entrypoints_reject_invalid_size_before_work(size):
    """场景：绕过配置直接批写或校验库；预期：消费输入和访问数据库前拒绝。

    输入非法大小及探针连接，两个入口均输出invalid_batch_size而不触发探针。
    """
    store = object.__new__(ProductStore)
    store.connection = UnavailableConnection()
    with pytest.raises(ValueError, match="^invalid_batch_size$"):
        store.put_many("entity", UnreadableRows(), batch_size=size)
    with pytest.raises(ValueError, match="^invalid_batch_size$"):
        store.verify_contents(batch_size=size)


@pytest.mark.parametrize("size", INVALID_BATCH_SIZES)
@pytest.mark.parametrize("bound", [False, True])
def test_consumer_rejects_invalid_size_even_without_mapping(daily_context, size, bound):
    """场景：直接消费入口带非法大小且可能没有实体映射；预期：始终首先拒绝。

    输入百万级大小和有或无映射绑定，输出invalid_batch_size，不扫描或写入。
    """
    entry, contract = daily_context
    configured = bindings() if bound else FoundationBindings()
    with pytest.raises(ValueError, match="^invalid_batch_size$"):
        consume_rows(
            UnavailableConnection(),
            None,
            entry,
            contract,
            configured,
            NOW,
            "limits",
            batch_size=size,
        )


@pytest.mark.parametrize("size", [1, 512])
def test_accepted_batch_boundaries_preserve_order_and_idempotency(tmp_path, size):
    """场景：最小和最大合法批大小；预期：配置可回读、按序分批且存储幂等。

    输入513个整数和重复实体，输出完整有序整数及相同身份，物理实体仅一条。
    """
    policy = FoundationPolicy(version="batch-limits-v1", fact_batch_size=size)
    assert FoundationPolicy.from_json(policy.to_json()) == policy
    chunks = list(batches(range(513), size))
    assert [value for chunk in chunks for value in chunk] == list(range(513))
    assert max(map(len, chunks)) == size
    with ProductStore(tmp_path / "products.duckdb") as store:
        entity = bindings().entities[0]
        identities = store.put_many("entity", (entity, entity), batch_size=size)
        assert identities[0] == identities[1]
        assert store.put_many("entity", [entity], batch_size=size) == identities[:1]
        assert store.counts()["entity"] == 1
        store.verify_contents(batch_size=size)
    assert FoundationPolicy(version="defaults").fact_batch_size == 512


@pytest.mark.parametrize("size", [None, 1, 128, 512])
def test_reference_fetch_has_independent_read_limit(tmp_path, size):
    """场景：默认读取或显式传入合法写入上限；预期：真实fetch每次最多128行。

    输入257条合成来源，输出全部257条且读取大小为128，显式1则每次读取1条。
    """
    source = bulk_reference(tmp_path, 257)
    sizes = []
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "512MB"}) as conn:
        recorded = FetchRecorder(conn, sizes)
        rows = list(
            reference_rows(recorded, "daily", **({} if size is None else {"batch_size": size}))
        )
    assert len(rows) == 257
    assert len({row["source_record_id"] for row in rows}) == 257
    assert set(sizes) == {1 if size == 1 else 128}


def test_verification_caps_real_fetch_size_independently(tmp_path):
    """场景：写入513条后按最大合法批大小校验；预期：读取仍最多128行且全表通过。

    输入跨写入批次的合成实体，输出完整513条并成功校验，真实fetch参数均为128。
    """
    first = bindings().entities[0]
    sizes = []
    with ProductStore(tmp_path / "products.duckdb") as store:
        records = (
            replace(first, identity=replace(first.identity, local_id=f"{i}.SZ")) for i in range(513)
        )
        assert len(store.put_many("entity", records, batch_size=512)) == 513
        store.connection = FetchRecorder(store.connection, sizes)
        store.verify_contents(batch_size=512)
        assert store.counts()["entity"] == 513
    assert sizes and set(sizes) == {128}


def test_consumer_keeps_read_and_fact_buffers_bounded(tmp_path, monkeypatch, daily_context):
    """场景：513条来源各生成两条数值事实；预期：读批128、写缓冲不超过512。

    输入1026条派生数值事实，输出完整事实与血缘，进度仍按512条来源分批汇报。
    """
    source = bulk_reference(tmp_path, 513)
    entry, contract = daily_context
    sizes, writes, progress = [], [], []
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "512MB"}) as conn:
        with ProductStore(tmp_path / "products.duckdb") as store:
            store.put_many("entity", bindings().entities)
            put_many = store.put_many

            def bounded_put_many(table, records, *, batch_size=512):
                """透传真实写入并检查调用前缓冲；输入事实列表，输出原始写入身份列表。"""
                if isinstance(records, list):
                    writes.append((table, len(records)))
                    assert len(records) <= batch_size
                return put_many(table, records, batch_size=batch_size)

            monkeypatch.setattr(store, "put_many", bounded_put_many)
            reasons, latest = consume_rows(
                FetchRecorder(conn, sizes),
                store,
                entry,
                contract,
                bindings(),
                NOW,
                "limits",
                batch_size=512,
                progress=progress.append,
            )
            assert store.counts()["availability_fact"] == 513
            assert store.counts()["typed_fact"] == 1026
            assert store.counts()["lineage_edge"] >= 1026
            store.verify_contents()
    assert sizes and set(sizes) == {128}
    assert progress == [512, 1]
    assert writes and max(size for _, size in writes) == 512
    assert "row_adaptation_failed" not in reasons
    assert latest == "2026-09-30"


def test_buffer_write_failure_propagates_and_rolls_back(tmp_path, monkeypatch, daily_context):
    """场景：第二次数值缓冲写入失败；预期：异常穿过适配层并回滚此前写入。

    输入513条合成来源，第二批抛immutable_record_conflict，重新打开库后所有事实为空。
    """
    source = bulk_reference(tmp_path, 513)
    output = tmp_path / "products.duckdb"
    entry, contract = daily_context
    completed_writes = []
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "512MB"}) as conn:
        with (
            pytest.raises(ValueError, match="^immutable_record_conflict$"),
            ProductStore(output) as store,
        ):
            store.put_many("entity", bindings().entities)
            put_many = store.put_many

            def fail_second_typed_write(table, records, *, batch_size=512):
                """第一次透传真实数值写入，第二次抛冲突；输入数值批次，输出身份或异常。"""
                if table == "typed_fact" and records:
                    if completed_writes:
                        raise ValueError("immutable_record_conflict")
                    result = put_many(table, records, batch_size=batch_size)
                    completed_writes.append(len(result))
                    return result
                return put_many(table, records, batch_size=batch_size)

            monkeypatch.setattr(store, "put_many", fail_second_typed_write)
            consume_rows(conn, store, entry, contract, bindings(), NOW, "limits", batch_size=512)
    assert completed_writes == [512]
    with ProductStore(output, read_only=True) as store:
        assert all(count == 0 for count in store.counts().values())
