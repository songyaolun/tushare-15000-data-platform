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
    """场景：文档同步和账户探测；预期：运行结果不改文档生命周期，历史证据保留。"""
    entry.evidence = [{"source": "old"}]
    store.save_catalog([entry])
    store.record_probe(entry.api_name, OBS, "fixture", "unauthorized", {})
    assert store.catalog()[0].lifecycle == "production"
    store.record_probe(entry.api_name, OBS, "live", "unauthorized", {})
    assert store.catalog()[0].lifecycle == "production"
    newer = replace(entry, evidence=[{"source": "new"}], minimum_points=3000)
    store.save_catalog([newer])
    saved = store.catalog()[0]
    assert saved.lifecycle == "production" and saved.needs_review
    assert saved.evidence == [{"source": "new"}, {"source": "old"}]
    store.record_probe(entry.api_name, OBS, "live", "available", {})
    assert store.catalog()[0].lifecycle == "production"


def test_explicit_scope_block_survives_catalog_refresh(store, entry):
    """场景：既有明确范围阻断后同步研究政策；预期：不会静默解除原始范围决策。"""
    store.save_catalog([replace(entry, lifecycle="blocked")])
    store.save_catalog([entry])
    assert store.catalog()[0].lifecycle == "blocked"
    store.record_probe("missing", OBS, "live", "empty", {})
    store.record_probe(entry.api_name, OBS, "live", "transient_failure", {})
    assert store.catalog()[0].lifecycle == "blocked"


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
    # 不同执行模式才表示新的任务；仅 planned_at 变化不再制造新 ID。
    for index, close in enumerate([3, 2], start=4):
        observed = f"2024-01-0{index}T00:00:00Z"
        task = plan_tasks(
            entry, "2024-01-01", "2024-01-01", "lookback" if index == 4 else "event", observed
        )[0]
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
    # 政策变化才表示替代任务；保留原有 superseded 与水位断言。
    replacement = plan_tasks(
        replace(entry, policy=replace(policy, lookback_days=1)),
        "2024-01-02",
        "2024-01-02",
        "initialize",
        "2024-01-04T00:00:00Z",
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


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "forged"},
        {"start": "2024-01-01"},
        {"end": "2024-01-03"},
        {"params": {"trade_date": "20240103"}},
        {"extra": "unexpected"},
    ],
)
def test_begin_task_rejects_tampering_without_mutating_plan(store, entry, change):
    """场景：已有任务 ID 或执行载荷被篡改；预期：拒绝启动且数据库载荷原样保留。

    示例：保持 task_id 但更改 params → ValueError，任务仍 planned。
    """
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    store.save_plan([task], OBS)
    before = store.records("tasks")
    with pytest.raises(ValueError):
        store.begin_task(dict(task, **change), OBS, "fixture")
    assert store.records("tasks") == before
    assert store.rows(entry.api_name) == []
    assert store.records("checkpoints") == []


def test_begin_task_allows_retry_but_rejects_running_and_superseded(store, entry, policy):
    """场景：有效任务经过计划、运行、失败和替代状态；预期：只允许明确可重试状态启动。

    示例：running 再启动和 superseded 复活均失败；failed 重试成功。
    """
    original = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    store.save_plan([original], OBS)
    assert store.begin_task(original, OBS, "fixture")
    with pytest.raises(ValueError):
        store.begin_task(original, OBS, "fixture")
    replacement = plan_tasks(
        replace(entry, policy=replace(policy, lookback_days=1)),
        "2024-01-02",
        "2024-01-02",
        "initialize",
        OBS,
    )[0]
    store.save_plan([replacement], OBS)
    assert (
        next(row for row in store.records("tasks") if row["task_id"] == original["task_id"])[
            "state"
        ]
        == "running"
    )
    store.fail_task(original["task_id"], "retryable", [])
    assert store.begin_task(original, OBS, "fixture")
    store.fail_task(original["task_id"], "retryable", [])
    replacement = plan_tasks(
        replace(entry, policy=replace(policy, lookback_days=2)),
        "2024-01-02",
        "2024-01-02",
        "initialize",
        OBS,
    )[0]
    store.save_plan([replacement], OBS)
    before = store.records("tasks")
    with pytest.raises(ValueError):
        store.begin_task(original, OBS, "fixture")
    assert store.records("tasks") == before


def test_watermark_stops_at_unplanned_gap_and_advances_when_filled(store, entry, policy):
    """场景：仅完成 1 日和 3 日；预期：水位停在 1 日，补齐 2 日才推进至 3 日。

    示例：任务完成顺序 1、3、2 → 水位依次 1、1、3。
    """
    for day, expected in [(1, "2024-01-01"), (3, "2024-01-01"), (2, "2024-01-03")]:
        date = f"2024-01-{day:02d}"
        task = plan_tasks(entry, date, date, "initialize", OBS)[0]
        assert store.begin_task(task, OBS, "fixture")
        rows = normalize_rows(
            [{"ts_code": "A.SH", "trade_date": date, "close": 2}], policy, OBS, "response"
        )
        store.commit_task(
            task, rows, policy.primary_key, OBS, "fixture", list(rows[0]["source_fields"])
        )
        assert store.watermark(entry.api_name, "trade_date") == expected
    assert len(store.rows(entry.api_name)) == 3


@pytest.mark.parametrize(
    "field,start,end,params,expected",
    [
        ("trade_date", "2024-01-15", "2024-01-20", {}, ("2024-01-15", "2024-01-20")),
        ("end_date", "2024-01-15", "2024-01-20", {}, ("2024-01-15", "2024-01-20")),
        ("month", "2024-02-15", "2024-02-20", {}, ("2024-02-01", "2024-02-29")),
        ("quarter", "2024-02-15", "2024-05-20", {}, ("2024-01-01", "2024-06-30")),
        ("year", "2024-02-15", "2024-05-20", {}, ("2024-01-01", "2024-12-31")),
        (
            "end_date",
            "2024-03-31",
            "2024-03-31",
            {"period": "20240331"},
            ("2024-01-01", "2024-03-31"),
        ),
        ("year", "2024-12-31", "2024-12-31", {"period": "20241231"}, ("2024-01-01", "2024-12-31")),
    ],
)
def test_coverage_interval_respects_period_semantics(field, start, end, params, expected):
    """场景：自然日与明确周期任务；预期：只扩展具有完整周期语义的覆盖区间。

    示例：2024 年 2 月任务覆盖闰年 29 日；普通 end_date 保留原范围。
    """
    from tushare_entitlement_hub.store import coverage_interval

    actual = coverage_interval({"date_field": field, "start": start, "end": end, "params": params})
    assert tuple(value.isoformat() for value in actual) == expected


@pytest.mark.parametrize(
    "start,end,period",
    [
        ("2024-02-01", "2024-01-01", None),
        ("2024-01-01", "2024-12-31", "20240131"),
        ("2024-01-01", "2024-12-31", "20240330"),
        ("2024-01-01", "2024-02-01", "20240331"),
    ],
)
def test_coverage_interval_rejects_invalid_range_or_period(start, end, period):
    """场景：倒置范围、非季末或范围外报告期；预期：拒绝不可信覆盖区间。

    示例：period=20240330 → task_period_invalid。
    """
    from tushare_entitlement_hub.store import coverage_interval

    with pytest.raises(ValueError, match="task_(range|period)_invalid"):
        coverage_interval(
            {"date_field": "end_date", "start": start, "end": end, "params": {"period": period}}
        )


def test_task_identity_and_commit_state_validation(store, entry, policy):
    """场景：缺字段、来源冲突和未启动直接提交；预期：拒绝且控制表不写入伪记录。

    示例：planned 任务直接 commit → task_not_running，仍为 planned。
    """
    from tushare_entitlement_hub.store import task_identity

    with pytest.raises(ValueError, match="task_payload_incomplete"):
        task_identity({})
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    for source in ["live", "unknown"]:
        with pytest.raises(ValueError, match="task_source_mismatch"):
            store.begin_task(task, OBS, source)
    with pytest.raises(ValueError, match="task_identity_mismatch"):
        store.save_plan([dict(task, task_id="forged")], OBS)
    store.save_plan([task], OBS)
    with pytest.raises(ValueError, match="task_not_running"):
        store.commit_task(task, [], policy.primary_key, OBS, "fixture", [])
    assert store.records("tasks")[0]["state"] == "planned"
    assert store.rows(entry.api_name) == []


@pytest.mark.parametrize(
    "field,first,middle,last",
    [
        ("month", "2024-01-15", "2024-02-15", "2024-03-15"),
        ("quarter", "2024-01-15", "2024-04-15", "2024-07-15"),
        ("year", "2022-06-15", "2023-06-15", "2024-06-15"),
    ],
)
def test_period_watermarks_require_contiguous_completed_intervals(
    store, entry, policy, field, first, middle, last
):
    """场景：月、季、年任务乱序补齐；预期：不会跳过中间周期，完整后推进到周期末。

    示例：一月、三月先完成 → 水位一月末；二月完成后到三月末。
    """
    from tushare_entitlement_hub.store import coverage_interval

    current = replace(entry, policy=replace(policy, date_field=field, shard_by="range"))
    first_watermark = None
    for index, day in enumerate([first, last, middle]):
        task = plan_tasks(current, day, day, "initialize", OBS)[0]
        store.begin_task(task, OBS, "fixture")
        store.commit_task(task, [], policy.primary_key, OBS, "fixture", [field])
        if index == 0:
            first_watermark = coverage_interval(task)[1].isoformat()
        if index < 2:
            assert store.watermark(entry.api_name, field) == first_watermark
        else:
            final = plan_tasks(current, last, last, "initialize", OBS)[0]
            assert store.watermark(entry.api_name, field) == coverage_interval(final)[1].isoformat()


def test_watermark_pending_overlap_blocks_only_unfinished_range(store, entry, policy):
    """场景：宽区间完成但内部证券分片待办；预期：仅推进至待办前一天，补齐后继续。

    示例：1 至 3 日主分片完成、2 日另一证券待办 → 水位 1 日。
    """
    current = replace(entry, policy=replace(policy, shard_by="range"))
    wide = plan_tasks(current, "2024-01-01", "2024-01-03", "initialize", OBS)[0]
    pending = plan_tasks(
        current, "2024-01-02", "2024-01-02", "initialize", OBS, parameters={"ts_code": "B.SH"}
    )[0]
    store.save_plan([wide, pending], OBS)
    store.begin_task(wide, OBS, "fixture")
    store.commit_task(wide, [], policy.primary_key, OBS, "fixture", [])
    assert store.watermark(entry.api_name, "trade_date") == "2024-01-01"
    store.begin_task(pending, OBS, "fixture")
    store.commit_task(pending, [], policy.primary_key, OBS, "fixture", [])
    assert store.watermark(entry.api_name, "trade_date") == "2024-01-03"


def test_watermark_waits_for_all_security_shards_at_same_start(store, entry, policy):
    """场景：同一天两个证券分片只完成一个；预期：水位为空，全部完成后才推进。

    示例：A.SH 已完成而 B.SH planned → 无水位；B.SH 完成 → 2024-01-02。
    """
    tasks = plan_tasks(
        entry,
        "2024-01-02",
        "2024-01-02",
        "initialize",
        OBS,
        parameters={"ts_codes": ["A.SH", "B.SH"]},
    )
    store.save_plan(tasks, OBS)
    for index, task in enumerate(tasks):
        store.begin_task(task, OBS, "fixture")
        store.commit_task(task, [], policy.primary_key, OBS, "fixture", [])
        assert store.watermark(entry.api_name, "trade_date") == (
            None if index == 0 else "2024-01-02"
        )


def test_watermark_at_maximum_date_does_not_overflow_on_new_revision(store, entry, policy):
    """场景：手工合法计划覆盖日期上界且后续修订；预期：保持上界检查点，不计算越界次日。

    示例：9999-12-31 首次提交和另一模式修订均成功，水位仍为 9999-12-31。
    """
    from tushare_entitlement_hub.store import task_identity

    template = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
    for mode in ["initialize", "lookback"]:
        task = dict(
            template,
            start="9999-12-31",
            end="9999-12-31",
            mode=mode,
            params={"trade_date": "99991231"},
        )
        task["task_id"] = task_identity(task)
        store.begin_task(task, OBS, "fixture")
        store.commit_task(task, [], policy.primary_key, OBS, "fixture", [])
        assert store.watermark(entry.api_name, "trade_date") == "9999-12-31"
    assert all(row["state"] == "committed" for row in store.records("tasks"))
