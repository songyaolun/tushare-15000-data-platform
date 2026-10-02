"""覆盖债务分母、消费证据和十类离线产品。"""

import json
from dataclasses import replace

import pytest

from tushare_entitlement_hub.coverage import coverage_report, policy_debt, ratio
from tushare_entitlement_hub.ingest import plan_tasks
from tushare_entitlement_hub.models import PRODUCTS, CatalogEntry
from tushare_entitlement_hub.products import build_products, product_content, summarize
from tushare_entitlement_hub.standard import normalize_rows

OBS = "2024-01-03T00:00:00Z"


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"category": " "}, "missing_category"),
        ({"lifecycle": "invalid"}, "invalid_policy_lifecycle"),
        ({"primary_key": [" "]}, "missing_primary_key"),
        ({"required_fields": []}, "missing_field_contract"),
        ({"purpose": " "}, "missing_collection_policy"),
        ({"date_field": " "}, "missing_watermark_dimension"),
        ({"page_size": 0}, "invalid_collection_bounds"),
        ({"units": {"close": " "}}, "invalid_units_or_freshness"),
        ({"consumers": ["archive: "]}, "missing_consumer"),
        ({"consumers": []}, "missing_consumer"),
    ],
)
def test_policy_debt_cannot_be_bypassed(policy, changes, code):
    """场景：政策要素缺失；预期：明确债务拒绝空白绕过。

    示例：purpose=' ' → missing_collection_policy。"""
    assert policy_debt(policy) == []
    assert code in policy_debt(replace(policy, **changes))
    assert policy_debt(replace(policy, consumers=["research:主题", "archive:历史"])) == []


def test_coverage_empty_and_unclassified(store):
    """场景：空库和未知接口；预期：零分母不是100%，未知产生债务。

    示例：空库 → gate_passed=False。"""
    assert ratio(0, 0) == {"numerator": 0, "denominator": 0, "rate": None}
    empty = coverage_report(store, "2024-01-03")
    assert not empty["gate_passed"] and empty["discovered"] == 0
    store.save_catalog([CatalogEntry("new_api")])
    report = coverage_report(store, "2024-01-03")
    assert report["metrics"]["registration"]["rate"] == 0
    assert {item["code"] for item in report["debt"]} >= {
        "unclassified",
        "missing_documentation",
        "unknown_permission",
        "missing_update_schedule",
    }


def test_coverage_registration_variants(store, entry):
    """场景：未知权益和政策冲突；预期：每种缺失可追踪。

    示例：points=None → points_not_verified。"""
    entries = [
        replace(entry, api_name="conflict", needs_review=True, minimum_points=None),
        replace(entry, api_name="blocked", lifecycle="blocked"),
        replace(entry, api_name="no_policy", policy=None),
    ]
    store.save_catalog(entries)
    report = coverage_report(store, "2024-01-03", "live")
    codes = {item["code"] for item in report["debt"]}
    assert {
        "documentation_conflict",
        "blocked_interface",
        "points_not_verified",
        "missing_policy",
    } <= codes
    assert report["source"] == "live"


def test_products_all_views_lineage_and_consumption(store, entry, policy, tmp_path):
    """场景：有效离线数据驱动产品；预期：十类产物有来源且消费计数真实。

    示例：一行 → market available。"""
    policy.consumers = list(PRODUCTS) + ["research:测试", "archive:快照"]
    store.save_catalog([entry, CatalogEntry("disabled")])
    task = plan_tasks(entry, "2024-01-02", "2024-01-02", "backfill", OBS)[0]
    store.begin_task(task, OBS, "fixture")
    fields = {"ts_code": "A.SH", "trade_date": "20240102", "close": 2, "pct_chg": 5}
    rows = normalize_rows([fields], policy, OBS, "public-hash")
    store.commit_task(
        task,
        rows,
        policy.primary_key,
        OBS,
        "fixture",
        list(fields),
        [{"code": "schema_drift_not_checked", "severity": "not_checked"}],
    )
    store.record_probe(entry.api_name, OBS, "fixture", "available", {})
    paths = build_products(store, tmp_path / "products", "2024-01-03")
    assert len(paths) == 24
    payload = json.loads((tmp_path / "products" / "market.json").read_text())
    assert payload["status"] == "available" and payload["fixture"]
    assert payload["lineage"] == ["public-hash"]
    assert payload["content"]["statistics"][entry.api_name]["close"]["mean"] == 2
    report = coverage_report(store, "2024-01-03")
    for name in (
        "capability_probe",
        "collection",
        "consumption",
        "freshness",
        "historical_backfill",
    ):
        assert report["metrics"][name]["rate"] == 1
    assert report["live_probe"]["numerator"] == 0
    assert len(store.records("consumption")) == 11
    live_paths = build_products(store, tmp_path / "live", "2024-01-03", "live")
    assert len(live_paths) == 20
    assert "账户采集数据" in (tmp_path / "live" / "市场总览.md").read_text()
    assert json.loads((tmp_path / "live" / "market.json").read_text())["status"] == "missing_inputs"


def test_summary_and_product_views():
    """场景：空值、无穷、布尔及产品布局；预期：只统计有限数值。

    示例：[2,4,None] → mean=3。"""
    assert summarize([{"close": x} for x in [2, 4, None, True, float("inf")]])["close"] == {
        "count": 2,
        "min": 2,
        "max": 4,
        "mean": 3,
    }
    rows = [
        {"ts_code": "A.SH", "response_hash": "hash", "ann_date": "2024-01-02", "pct_chg": 2},
        {"response_hash": "other", "pct_chg": None},
    ]
    assert set(product_content("security", {"x": rows})["profiles"]) == {"A.SH", "未提供证券代码"}
    assert product_content("events", {"x": rows})["events"][0]["dates"] == {
        "ann_date": "2024-01-02"
    }
    for name in ["rotation", "sentiment"]:
        assert product_content(name, {"x": rows})["ranking"][0]["ts_code"] == "A.SH"


def test_freshness_snapshot_invalid_dates_and_quality_debt(store, entry, policy):
    """场景：快照时间与非法日期；预期：快照按观察时间，坏日期记未知债务。

    示例：bad → freshness_unknown。"""
    snapshot = replace(
        entry,
        api_name="snap",
        policy=replace(policy, api_name="snap", shard_by="snapshot", collection_mode="snapshot"),
    )
    store.save_catalog([entry, snapshot])
    for current, value in [(entry, "bad"), (snapshot, "2024-01-01")]:
        task = plan_tasks(current, "2024-01-02", "2024-01-02", "initialize", OBS)[0]
        store.begin_task(task, OBS, "fixture")
        row = {
            "ts_code": "A.SH",
            "trade_date": value,
            "source_fields": {"id": 1},
            "observed_at": OBS,
        }
        store.commit_task(
            task,
            [row],
            policy.primary_key,
            OBS,
            "fixture",
            ["id"],
            [{"code": "calendar_coverage_not_checked", "severity": "not_checked"}],
        )
    report = coverage_report(store, "2024-01-03")
    assert report["metrics"]["freshness"]["numerator"] == 1
    assert {item["code"] for item in report["debt"]} >= {
        "freshness_unknown",
        "calendar_coverage_not_checked",
    }


@pytest.mark.parametrize("placeholder", ["TODO", "TBD", "default", "unknown", "未知", "默认"])
def test_policy_placeholder_debt(policy, placeholder):
    """场景：用占位描述冒充用途；预期：用途和消费者仍有债务。

    示例：purpose='TODO'和consumer='archive:TODO' → 两项missing债务。
    """
    invalid = replace(policy, purpose=placeholder, consumers=[f"archive:{placeholder}"])
    assert {"missing_collection_policy", "missing_consumer"} <= set(policy_debt(invalid))


@pytest.mark.parametrize(
    "status,count",
    [
        ("available", 1),
        ("empty", 1),
        ("unauthorized", 1),
        ("unsupported", 1),
        ("transient_failure", 0),
        ("contract_failure", 0),
    ],
)
def test_probe_coverage_counts_only_conclusive_results(store, entry, status, count):
    """场景：权限探活结果分母；预期：瞬时与契约失败不冒充已确认权益。

    示例：transient_failure → capability_probe.numerator=0，并保留失败观测。
    """
    store.save_catalog([entry])
    store.record_probe(entry.api_name, OBS, "fixture", status, {})
    report = coverage_report(store, "2024-01-03")
    assert report["metrics"]["capability_probe"]["numerator"] == count
    assert report["live_probe"]["numerator"] == 0
