"""控制面事务、恢复、水位及版本审计。"""

from dataclasses import asdict, replace

import pytest

from tushare_entitlement_hub.ingest import plan_tasks
from tushare_entitlement_hub.models import CatalogEntry, entry_from_dict, policy_from_dict
from tushare_entitlement_hub.standard import normalize_rows
from tushare_entitlement_hub.store import digest, encode

OBS = "2024-01-03T00:00:00Z"


def test_model_serialization(entry, policy):
    """场景：嵌套政策往返；预期：对象相等且不改输入。例：asdict(entry) → entry。"""
    payload = asdict(entry)
    assert entry_from_dict(payload) == entry
    assert isinstance(payload["policy"], dict)
    assert policy_from_dict(asdict(policy)) == policy
    assert entry_from_dict({"api_name": "unknown"}) == CatalogEntry("unknown")
    assert encode({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    with pytest.raises(ValueError, match="凭据"):
        digest({"token": "<REDACTED>"})


def test_catalog_merge_and_probe_lifecycle(store, entry):
    """场景：重复发现与探活结果；预期：不丢证据，fixture不变权限。例：live拒绝 → blocked。"""
    entry.evidence = [{"source": "old"}]
    store.save_catalog([entry])
    store.record_probe(entry.api_name, OBS, "fixture", "unauthorized", {})
    assert store.catalog()[0].lifecycle == "production"
    store.record_probe(entry.api_name, OBS, "live", "unauthorized", {})
    assert store.catalog()[0].lifecycle == "blocked"
    newer = replace(entry, evidence=[{"source": "new"}], minimum_points=3000)
    store.save_catalog([newer])
    saved = store.catalog()[0]
    assert saved.lifecycle == "blocked" and saved.needs_review
    assert saved.evidence == [{"source": "new"}, {"source": "old"}]
    store.record_probe(entry.api_name, OBS, "live", "available", {})
    assert store.catalog()[0].lifecycle == "production"
    store.record_probe("missing", OBS, "live", "empty", {})
    store.record_probe(entry.api_name, OBS, "live", "transient_failure", {})
    assert store.catalog()[0].lifecycle == "production"


def test_transactions_versions_watermarks_and_export(store, entry, policy, tmp_path):
    """场景：乱序提交及回看修订；预期：水位不越过待办，历史只增。例：先第2日 → 无水位。"""
    tasks = plan_tasks(entry, "2024-01-01", "2024-01-02", "initialize", OBS)
    store.save_plan(tasks, OBS)
    assert store.watermark(entry.api_name, policy.date_field) is None
    assert store.previous_fields(entry.api_name) is None
    fields = ["ts_code", "trade_date", "close"]
    for task in reversed(tasks):
        assert store.begin_task(task, OBS, "fixture")
        rows = normalize_rows(
            [{"ts_code": "A.SH", "trade_date": task["end"], "close": 2}],
            policy,
            OBS,
            "public-response",
        )
        store.commit_task(task, rows, policy.primary_key, OBS, "fixture", fields)
        if task == tasks[1]:
            assert store.watermark(entry.api_name, policy.date_field) is None
    assert store.watermark(entry.api_name, policy.date_field) == "2024-01-02"
    assert not store.begin_task(tasks[0], OBS, "fixture")
    store.save_plan(tasks, OBS)
    assert all(record["state"] == "committed" for record in store.records("tasks"))
    assert store.previous_fields(entry.api_name) == fields
    assert store.previous_fields(entry.api_name, "live") is None
    assert len(store.rows(entry.api_name)) == 2
    assert store.rows(entry.api_name, "live") == []
    assert store.export_parquet(tmp_path / "export" / "rows.parquet") == 2
    assert (
        store.connection.read_parquet(str(tmp_path / "export" / "rows.parquet"))
        .count("*")
        .fetchone()[0]
        == 2
    )
    for index, close in enumerate([3, 2], start=4):
        observed = f"2024-01-0{index}T00:00:00Z"
        task = plan_tasks(entry, "2024-01-01", "2024-01-01", "lookback", observed)[0]
        store.begin_task(task, observed, "fixture")
        row = normalize_rows(
            [{"ts_code": "A.SH", "trade_date": "20240101", "close": close}],
            policy,
            observed,
            "response",
        )
        store.commit_task(task, row, policy.primary_key, observed, "fixture", fields)
    assert len(store.rows(entry.api_name)) == 4
    assert store.watermark(entry.api_name, policy.date_field) == "2024-01-02"


def test_transaction_rollback_and_recovery(store, entry, policy):
    """场景：第二条版本编码失败；预期：第一条同样回滚且重试成功。例：含token行 → 零标准行。"""
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "backfill", OBS)[0]
    store.begin_task(task, OBS, "fixture")
    row = normalize_rows(
        [{"ts_code": "A.SH", "trade_date": "20240102", "close": 2}], policy, OBS, "response"
    )[0]
    with pytest.raises(ValueError, match="凭据"):
        store.commit_task(
            task,
            [row, dict(row, token="<REDACTED>")],
            policy.primary_key,
            OBS,
            "fixture",
            list(row["source_fields"]),
        )
    assert store.rows(entry.api_name) == []
    assert store.watermark(entry.api_name, policy.date_field) is None
    assert store.records("quality") == []
    store.fail_task(task["task_id"], "quality_failed", [{"code": "bad", "severity": "error"}])
    assert store.records("tasks")[0]["state"] == "failed"
    assert store.begin_task(task, OBS, "fixture")
    store.commit_task(task, [row], policy.primary_key, OBS, "fixture", list(row["source_fields"]))
    assert store.records("tasks")[0]["state"] == "committed"
    with pytest.raises(ValueError, match="未知"):
        store.records("standard; DROP TABLE standard")
    for product, count in [(" ", 1), ("market", 0)]:
        with pytest.raises(ValueError, match="消费"):
            store.consume(product, entry.api_name, "fixture", OBS, count, "hash")
    store.consume("market", entry.api_name, "fixture", OBS, 1, "hash")
    assert store.records("consumption")[0]["row_count"] == 1


def test_new_plan_supersedes_failed_and_retains_running(store, entry, policy, tmp_path):
    """场景：同范围政策修订后重新计划；预期：失败旧任务不永久阻塞。

    示例：旧planned任务+新同范围计划 → 旧任务superseded，新完成后水位可推进。
    """
    original = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    store.save_plan([original], OBS)
    replacement = plan_tasks(
        entry, "2024-01-02", "2024-01-02", "initialize", "2024-01-04T00:00:00Z"
    )[0]
    store.save_plan([replacement], OBS)
    states = {row["task_id"]: row["state"] for row in store.records("tasks")}
    assert states[original["task_id"]] == "superseded"
    assert store.begin_task(replacement, OBS, "fixture")
    rows = normalize_rows(
        [{"ts_code": "A.SH", "trade_date": "20240102", "close": 2}], policy, OBS, "response"
    )
    store.commit_task(
        replacement, rows, policy.primary_key, OBS, "fixture", list(rows[0]["source_fields"])
    )
    assert store.watermark(entry.api_name, policy.date_field) == "2024-01-02"
    assert store.export_parquet(tmp_path / "live-empty.parquet", "live") == 0
    relation = store.connection.read_parquet(str(tmp_path / "live-empty.parquet"))
    assert relation.count("*").fetchone()[0] == 0
