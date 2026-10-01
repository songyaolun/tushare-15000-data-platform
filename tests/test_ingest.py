"""政策计划、探活分类、失败恢复与严格水位。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.client import Client, FakeTransport, ProviderError
from tushare_entitlement_hub.ingest import (
    IngestionError,
    plan_tasks,
    probe,
    require_policy,
    run_task,
)
from tushare_entitlement_hub.raw import RawStore

OBS = "2024-01-03T00:00:00Z"


def response(rows):
    """生成确定性指数响应；输入[['A.SH','20240102',2]]，输出公开字段和数据行。"""
    return {"code": 0, "data": {"fields": ["ts_code", "trade_date", "close"], "items": rows}}


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"lifecycle": "unclassified"}, "interface_not_enabled"),
        ({"permission_mode": "independent"}, "interface_not_enabled"),
        ({"minimum_points": None}, "entitlement_needs_review"),
        ({"minimum_points": 15001}, "entitlement_needs_review"),
        ({"needs_review": True}, "entitlement_needs_review"),
        ({"policy": None}, "incomplete_policy"),
    ],
)
def test_require_policy_rejects_ineligible(entry, changes, code):
    """场景：无权限或不完整政策；预期：固定门禁错误。

    示例：独立权限 → interface_not_enabled。"""
    with pytest.raises(IngestionError, match=code):
        require_policy(replace(entry, **changes))
    assert require_policy(replace(entry, lifecycle="blocked"), for_probe=True) == entry.policy


def test_plan_modes_parameters_and_shards(entry, policy):
    """场景：时间、证券、报告期分片；预期：稳定任务且清除探测参数。

    示例：两天 → 两个任务。"""
    policy.minimal_params = {"ts_code": "PROBE.SH", "limit": 1, "exchange": "SSE"}
    tasks = plan_tasks(entry, "2024-01-01", "2024-01-02", "initialize", OBS)
    assert len(tasks) == 2
    assert tasks == plan_tasks(entry, "2024-01-01", "2024-01-02", "initialize", OBS)
    assert tasks[0]["params"] == {"exchange": "SSE", "trade_date": "20240101"}
    filtered = plan_tasks(
        entry,
        "2024-01-01",
        "2024-01-03",
        "backfill",
        OBS,
        parameters={"trade_dates": ["2024-01-02"]},
    )
    assert [item["start"] for item in filtered] == ["2024-01-02"]
    securities = plan_tasks(
        entry, "2024-01-01", "2024-01-01", "event", OBS, parameters={"ts_codes": ["A.SH", "B.SH"]}
    )
    assert [task["params"]["ts_code"] for task in securities] == ["A.SH", "B.SH"]
    with pytest.raises(IngestionError, match="security_universe_required"):
        plan_tasks(entry, "2024-01-01", "2024-01-01", "event", OBS, parameters={"ts_codes": []})
    for mode, start, end, error in [
        ("bad", "2024-01-01", "2024-01-02", "invalid_mode"),
        ("event", "2024-01-02", "2024-01-01", "invalid_range"),
    ]:
        with pytest.raises(IngestionError, match=error):
            plan_tasks(entry, start, end, mode, OBS)
    for field, shard, expected in [
        ("month", "range", {"start_m": "202401", "end_m": "202404"}),
        ("quarter", "range", {"start_q": "2024Q1", "end_q": "2024Q2"}),
        ("end_date", "period", {"period": "20240331"}),
        ("trade_date", "ts_code", {"start_date": "20240101", "end_date": "20240401"}),
        ("", "snapshot", {}),
    ]:
        modified = replace(entry, policy=replace(policy, date_field=field, shard_by=shard))
        parameters = (
            {"ts_code": "A.SH", "period": "20240331"} if shard == "period" else {"ts_code": "A.SH"}
        )
        task = plan_tasks(
            modified, "2024-01-01", "2024-04-01", "initialize", OBS, parameters=parameters
        )[0]
        assert expected.items() <= task["params"].items()
    security_entry = replace(entry, policy=replace(policy, shard_by="ts_code"))
    with pytest.raises(IngestionError, match="security_shard_required"):
        plan_tasks(security_entry, "2024-01-01", "2024-01-01", "event", OBS)
    period_entry = replace(entry, policy=replace(policy, shard_by="period", date_field="end_date"))
    assert plan_tasks(period_entry, "2024-01-01", "2024-02-01", "event", OBS) == []
    year = plan_tasks(period_entry, "2024-01-01", "2024-12-31", "backfill", OBS)
    assert [task["params"]["period"] for task in year] == [
        "20240331",
        "20240630",
        "20240930",
        "20241231",
    ]
    assert (
        plan_tasks(period_entry, "2024-03-31", "2024-03-31", "event", OBS)[0]["params"]["period"]
        == "20240331"
    )


@pytest.mark.parametrize(
    "kind",
    ["available", "empty", "unauthorized", "unsupported", "transient_failure", "contract_failure"],
)
def test_probe_preserves_classification(entry, policy, store, tmp_path, kind):
    """场景：六类最小探活结果；预期：如实分类且fixture不算live。

    示例：成功零行 → empty。"""
    policy.supports_pagination = True
    payload = response([["A.SH", "20240102", 2]]) if kind == "available" else response([])
    item = payload if kind in {"available", "empty"} else ProviderError(kind, "模拟失败")
    transport = FakeTransport([item])
    client = Client(transport, RawStore(tmp_path / "raw"), retries=0)
    result = probe(entry, client, store, OBS)
    assert result["status"] == kind
    assert result["source"] == "fixture"
    assert transport.calls[0]["params"]["limit"] == 1
    assert store.records("probes")[0]["status"] == kind


def test_run_success_idempotency_and_incremental(entry, policy, store, tmp_path):
    """场景：完成、重跑、增量回看；预期：不重复写入且水位受回看控制。

    示例：重复执行 → already_committed。"""
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    transport = FakeTransport([response([["A.SH", "20240102", 2]])])
    client = Client(transport, RawStore(tmp_path / "raw"))
    result = run_task(task, entry, client, store, OBS, expected_dates=["20240102"])
    assert result["state"] == "committed" and result["rows"] == 1
    assert run_task(task, entry, client, store, OBS)["state"] == "already_committed"
    assert len(transport.calls) == 1
    assert store.watermark(entry.api_name, policy.date_field) == "2024-01-02"
    assert plan_tasks(entry, "2024-01-01", "2024-01-01", "incremental", OBS, store) == []
    policy.lookback_days = 1
    assert (
        plan_tasks(entry, "2024-01-01", "2024-01-02", "lookback", OBS, store)[0]["start"]
        == "2024-01-01"
    )


@pytest.mark.parametrize(
    "payload",
    [response([]), response([["A.SH", "20240202", 2]]), response([["A.SH", "20240102", None]])],
)
def test_run_quality_failure_keeps_raw_without_watermark(entry, store, tmp_path, payload):
    """场景：空页、越界日期、缺字段值；预期：raw保留但任务失败。

    示例：未来月份响应 → 无水位。"""
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    client = Client(FakeTransport([payload]), RawStore(tmp_path / "raw"))
    with pytest.raises(IngestionError, match="quality_or_storage_failed"):
        run_task(task, entry, client, store, OBS)
    assert store.records("tasks")[0]["state"] == "failed"
    assert store.rows(entry.api_name) == []
    assert store.watermark(entry.api_name, "trade_date") is None
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 1


def test_run_page_failure_recovery_and_plan_validation(entry, policy, store, tmp_path):
    """场景：第二页失败后恢复；预期：前页raw不丢、只完整成功才提交。

    示例：瞬时失败 → 重跑2行。"""
    policy.supports_pagination = True
    policy.page_size = 1
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    first = response([["A.SH", "20240102", 2]])
    client = Client(
        FakeTransport([first, ProviderError("transient_failure", "模拟")]),
        RawStore(tmp_path / "raw"),
        retries=0,
    )
    with pytest.raises(IngestionError, match="transient_failure"):
        run_task(task, entry, client, store, OBS)
    assert store.watermark(entry.api_name, "trade_date") is None
    retry = Client(FakeTransport([first, response([])]), RawStore(tmp_path / "raw"))
    assert run_task(task, entry, retry, store, OBS)["rows"] == 1
    for changes, error in [
        ({"api_name": "wrong"}, "plan_policy_changed"),
        ({"source": "unknown"}, "invalid_source"),
    ]:
        with pytest.raises(IngestionError, match=error):
            run_task(dict(task, **changes), entry, retry, store, OBS)


@pytest.mark.parametrize("month,success", [("202401", True), ("202403", False)])
def test_monthly_response_respects_planned_period(entry, policy, store, tmp_path, month, success):
    """场景：宏观月数据返回计划外月份；预期：只有范围内月份可提交。

    示例：计划2024年1月、响应202403 → 失败且水位仍为空。
    """
    month_policy = replace(
        policy,
        api_name="cn_cpi",
        category="macro",
        date_field="month",
        shard_by="range",
        primary_key=["month"],
        required_fields=["month", "value"],
        units={},
    )
    current = replace(entry, api_name="cn_cpi", policy=month_policy)
    task = plan_tasks(current, "2024-01-01", "2024-01-31", "initialize", OBS)[0]
    payload = {"code": 0, "data": {"fields": ["month", "value"], "items": [[month, 1.2]]}}
    client = Client(FakeTransport([payload]), RawStore(tmp_path / "raw"))
    if success:
        assert run_task(task, current, client, store, OBS)["state"] == "committed"
    else:
        with pytest.raises(IngestionError, match="quality_or_storage_failed"):
            run_task(task, current, client, store, OBS)
        assert store.watermark("cn_cpi", "month") is None
