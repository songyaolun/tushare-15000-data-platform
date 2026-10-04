"""增量保留桶的业务语义、容量失败关闭和线性访问次数回归。"""

from dataclasses import replace
from weakref import ref

import pytest
from test_cross_asset_bindings import CONTRACTS, cross_config
from test_research_adapter import reference_database, source
from test_research_binding_fixture import AS_OF
from test_research_governance import row

from tushare_entitlement_hub.cross_asset_research import adapter
from tushare_entitlement_hub.cross_asset_research.adapter import load_inputs, retain_rows
from tushare_entitlement_hub.cross_asset_research.retention import RetentionGroup
from tushare_entitlement_hub.store import encode


@pytest.mark.parametrize("api", ["fx_daily", "forecast"])
@pytest.mark.parametrize(
    "periods",
    [
        ("2026-09-01", "2026-09-03", "2026-09-02", "2026-09-01", "2026-09-04"),
        (None, None, "2026-09-01", "2026-09-02", None, "2026-09-01"),
        ("2026-09-03", "2026-09-03", "2026-09-01", "2026-09-02", "2026-09-02"),
    ],
)
def test_incremental_groups_preserve_complete_periods_and_all_versions(api, periods):
    """乱序日期、同期成员及无日期快照逐行输入；预期与完整期纯函数逐步一致。

    例如 01、03、02 日到达后保留 02 和 03 日的全部成员，forecast 保留全部版本。
    """
    group = RetentionGroup(keep_versions=api in adapter.VERSION_APIS)
    expected = []
    sizes = {}
    assert list(group.iter_rows()) == []
    for index, period in enumerate(periods):
        candidate = row(api, str(index), day=period)
        size = index + 10
        sizes[str(index)] = size
        old_count, old_bytes = group.row_count, group.byte_count
        delta = group.append(candidate, size)
        expected = retain_rows(expected, candidate)
        actual = list(group.iter_rows())
        assert sorted(item.raw["source_record_id"] for item in actual) == sorted(
            item.raw["source_record_id"] for item in expected
        )
        assert group.row_count == len(expected)
        assert group.byte_count == sum(sizes[item.raw["source_record_id"]] for item in expected)
        assert delta == (group.row_count - old_count, group.byte_count - old_bytes)
        assert len(group.buckets) <= (1 if api in adapter.VERSION_APIS else 2)


class RejectIterationList(list):
    """禁止追加期间重扫已有成员；例如任何隐式列表复制都会立即使测试失败。"""

    def __iter__(self):
        """对重复全组遍历立即报错；例如 list(saved_rows) 不能经过本守卫。"""
        raise AssertionError("retained_members_rescanned_during_append")


@pytest.mark.parametrize("keep_versions", [False, True])
def test_same_period_append_never_iterates_existing_large_group(keep_versions):
    """同实体累计 20001 行；预期追加不遍历旧成员，日期桶与累计容量保持常数更新。"""
    group = RetentionGroup(keep_versions=keep_versions)
    candidate = row("forecast" if keep_versions else "fx_daily", "same")
    assert group.append(candidate, 7) == (1, 7)
    bucket = next(iter(group.buckets.values()))
    bucket.rows = RejectIterationList([candidate])
    for _ in range(20_000):
        assert group.append(candidate, 7) == (1, 7)
    assert group.row_count == 20_001
    assert group.byte_count == 140_007
    assert len(group.buckets) == 1


def test_old_period_release_does_not_scan_surviving_members():
    """旧日期桶被替换或旧期迟到；预期不遍历保留成员，释放按桶累计字节精确扣减。"""
    group = RetentionGroup()
    first = row("fx_daily", "first", day="2026-09-01")
    second = row("fx_daily", "second", day="2026-09-03")
    group.append(first, 20)
    group.append(second, 30)
    group.buckets["2026-09-03"].rows = RejectIterationList([second])
    assert group.append(row("fx_daily", "late", day="2026-08-01"), 999) == (0, 0)
    assert group.append(row("fx_daily", "middle", day="2026-09-02"), 40) == (0, 20)
    assert group.append(row("fx_daily", "newest", day="2026-09-04"), 50) == (0, 10)
    assert group.row_count == 2 and group.byte_count == 80
    assert set(group.buckets) == {"2026-09-03", "2026-09-04"}


def test_high_cardinality_has_independent_bounded_period_buckets():
    """20000 个实体各经历三期；预期每个实体只保留两期，追加不查找其他实体。"""
    groups = {str(index): RetentionGroup() for index in range(20_000)}
    for index in range(3):
        candidate = row("fx_daily", str(index), day=f"2026-09-0{index + 1}")
        for group in groups.values():
            group.append(candidate, 10)
    assert all(group.row_count == 2 and group.byte_count == 20 for group in groups.values())
    assert all(len(group.buckets) == 2 for group in groups.values())


def forbid_legacy_retention(*args, **kwargs):
    """生产扫描调用旧纯函数即失败；例如同一期第二行不能触发完整列表复制。"""
    raise AssertionError("bulk_reader_called_legacy_retention")


def test_bulk_reader_uses_incremental_groups_and_preserves_final_record_order(
    tmp_path, monkeypatch
):
    """乱序期间跨批输入且禁止旧函数；预期最近两完整期保留，并按来源记录编号排序。"""
    records = [
        source("fx_daily", record, trade_date=day)
        for record, day in (
            ("z", "20260901"),
            ("d", "20260903"),
            ("b", "20260902"),
            ("a", "20260903"),
            ("y", "20260901"),
            ("c", "20260904"),
        )
    ]
    path = reference_database(tmp_path, {"fx_daily": records})
    config = cross_config("fx_daily")
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    monkeypatch.setattr(adapter, "retain_rows", forbid_legacy_retention)
    result, counts = load_inputs(path, AS_OF, "retention", config)
    assert [item.raw["source_record_id"] for item in result.rows("fx_daily")] == ["a", "c", "d"]
    assert counts == {
        "scanned_rows": 6,
        "retained_rows": 3,
        "excluded_rows": 0,
        "batches": 3,
        "discarded_rows": 3,
    }


@pytest.mark.parametrize("budget", ["entity", "source", "total", "source_bytes", "total_bytes"])
def test_large_complete_period_rejects_whole_group_at_capacity(tmp_path, monkeypatch, budget):
    """同实体完整期超行数或字节容量；预期清空整组并扫描后续行，不交付部分成员。"""
    records = [source("fx_daily", f"{index:04}") for index in range(128)]
    path = reference_database(tmp_path, {"fx_daily": records})
    changes = {
        "entity": {"maximum_rows_per_entity_source": 32},
        "source": {"maximum_source_rows": 32},
        "total": {"maximum_total_rows": 32},
    }.get(budget, {})
    config = cross_config("fx_daily", **changes)
    if budget in {"source_bytes", "total_bytes"}:
        candidate = adapter.adapt_row(records[0], CONTRACTS["fx_daily"], config, AS_OF)
        byte_budget = 32 * len(encode(candidate.raw).encode("utf-8"))
        monkeypatch.setattr(
            adapter,
            "MAX_SOURCE_BYTES" if budget == "source_bytes" else "MAX_RETAINED_BYTES",
            byte_budget,
        )
    monkeypatch.setattr(adapter, "retain_rows", forbid_legacy_retention)
    result, counts = load_inputs(path, AS_OF, "retention", config)
    assert result.rows("fx_daily") == ()
    assert counts["scanned_rows"] == counts["discarded_rows"] == 128
    assert counts["excluded_rows"] == 0
    if budget == "entity":
        assert set(result.entity_source_reasons.values()) == {("entity_source_capacity_exceeded",)}
    else:
        assert (
            "source_byte_capacity_exceeded"
            if budget.endswith("bytes")
            else "source_capacity_exceeded"
        ) in result.source_reasons["fx_daily"]


def test_bulk_history_eviction_releases_byte_budget_before_capacity_check(tmp_path, monkeypatch):
    """三个大日期桶依次到达；预期第三期替换第一期后按两期字节判断，不累计历史量。"""
    records = [
        source("fx_daily", str(index), trade_date=f"2026090{index + 1}") for index in range(3)
    ]
    config = cross_config("fx_daily")
    sizes = [
        len(
            encode(adapter.adapt_row(item, CONTRACTS["fx_daily"], config, AS_OF).raw).encode(
                "utf-8"
            )
        )
        for item in records
    ]
    monkeypatch.setattr(adapter, "MAX_SOURCE_BYTES", sum(sorted(sizes)[-2:]))
    monkeypatch.setattr(adapter, "retain_rows", forbid_legacy_retention)
    path = reference_database(tmp_path, {"fx_daily": records})
    result, counts = load_inputs(path, AS_OF, "retention", config)
    assert [item.raw["source_record_id"] for item in result.rows("fx_daily")] == ["1", "2"]
    assert counts["retained_rows"] == 2 and counts["discarded_rows"] == 1
    assert "source_byte_capacity_exceeded" not in result.source_reasons["fx_daily"]


@pytest.mark.parametrize("capacity", [2, 3])
def test_bulk_version_source_keeps_all_dates_or_rejects_entire_entity(
    tmp_path, monkeypatch, capacity
):
    """版本接口有三个不同报告日期；预期容量够时保留三期，超限则整实体拒绝。"""
    records = [source("forecast", str(index), end_date=f"2026090{index + 1}") for index in range(3)]
    config = cross_config("forecast", maximum_rows_per_entity_source=capacity)
    config = replace(
        config,
        foundation=replace(
            config.foundation,
            entity_mappings=tuple(
                replace(mapping, namespace="synthetic")
                for mapping in config.foundation.entity_mappings
            ),
        ),
    )
    path = reference_database(tmp_path, {"forecast": records})
    monkeypatch.setattr(adapter, "retain_rows", forbid_legacy_retention)
    result, counts = load_inputs(path, AS_OF, "versions", config)
    assert counts["excluded_rows"] == 0
    assert counts["retained_rows"] == (3 if capacity == 3 else 0)
    if capacity == 3:
        assert len({item.availability.data_as_of for item in result.rows("forecast")}) == 3
        assert result.entity_source_reasons == {}
    else:
        assert result.rows("forecast") == ()
        assert set(result.entity_source_reasons.values()) == {("entity_source_capacity_exceeded",)}


@pytest.mark.parametrize("budget", ["entity", "source"])
def test_capacity_rejection_releases_group_objects_before_next_batch(tmp_path, monkeypatch, budget):
    """实体或整源容量拒绝后继续扫描；预期下一批到来前无临时变量持有已拒绝日期桶。"""
    records = [
        source("fx_daily", str(index), ts_code="A" if budget == "entity" else f"A{index}")
        for index in range(4)
    ]
    changes = (
        {"maximum_rows_per_entity_source": 1} if budget == "entity" else {"maximum_source_rows": 1}
    )
    config = cross_config("fx_daily", **changes)
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    original = adapter.RetentionGroup
    references = []

    def create_group(**kwargs):
        """仅保留桶对象弱引用；例如对象释放后 references[0]() 返回 None。"""
        group = original(**kwargs)
        references.append(ref(group))
        return group

    def check_released(event):
        """首批已触发容量拒绝；例如批进展回调时所有被拒绝桶都应释放。"""
        assert all(reference() is None for reference in references)

    monkeypatch.setattr(adapter, "RetentionGroup", create_group)
    path = reference_database(tmp_path, {"fx_daily": records})
    result, counts = load_inputs(path, AS_OF, "release", config, progress=check_released)
    assert result.rows("fx_daily") == () and counts["discarded_rows"] == 4
    assert references
