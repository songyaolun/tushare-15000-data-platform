"""真实权限使用离线 HTTP/fixture 重放验证，不需要真实凭据或网络。"""

import json
from dataclasses import asdict, replace

import pytest

from tushare_entitlement_hub.access import (
    bind_account,
    latest_probe,
    readable_policy,
    verified_policy,
)
from tushare_entitlement_hub.client import Client, FakeTransport, ProviderError
from tushare_entitlement_hub.coverage import coverage_report
from tushare_entitlement_hub.credentials import configured_credential
from tushare_entitlement_hub.ingest import (
    IngestionError,
    plan_tasks,
    probe,
    require_policy,
    run_task,
)
from tushare_entitlement_hub.models import CatalogEntry
from tushare_entitlement_hub.products import build_products
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.store import digest
from tushare_entitlement_hub.survey import survey

OBS = "2024-01-03T00:00:00Z"


def payload(rows=None):
    """构造公开行情响应；默认一条有效记录，rows=[] 返回合法空响应。"""
    return {
        "code": 0,
        "data": {
            "fields": ["ts_code", "trade_date", "close"],
            "items": [["A.SH", "20240102", 2]] if rows is None else rows,
        },
    }


def client_for(tmp_path, responses, account="local"):
    """创建不发网络请求的客户端；例如 [payload()] 可重放一个页面。"""
    return Client(FakeTransport(responses), RawStore(tmp_path / "raw"), account=account)


@pytest.mark.parametrize(
    "lifecycle,permission",
    [
        ("research", "unknown"),
        ("independent_permission", "independent"),
        ("deprecated", "points"),
        ("blocked", "unknown"),
    ],
)
def test_probe_overrides_documentary_restrictions(entry, store, tmp_path, lifecycle, permission):
    """场景：四种文档限制；预期：只读探测成功后可采集，原始文档分类不改。"""
    entry = replace(
        entry,
        lifecycle=lifecycle,
        permission_mode=permission,
        needs_review=True,
        policy=replace(entry.policy, lifecycle=lifecycle),
    )
    store.save_catalog([entry])
    client = client_for(tmp_path, [payload(), payload()])
    result = probe(entry, client, store, OBS, "live", {"trade_date": "20240102"})
    assert result["status"] == "available"
    assert store.catalog()[0].lifecycle == lifecycle
    tasks = plan_tasks(entry, "2024-01-02", "2024-01-02", "initialize", OBS, store, "live")
    outcome = run_task(
        tasks[0],
        entry,
        client,
        store,
        OBS,
        previous_fields=result["evidence"]["fields"],
        expected_dates=["2024-01-02"],
    )
    assert outcome["state"] == "committed"
    build_products(store, tmp_path / "earlier", "2024-01-01", "live")
    assert json.loads((tmp_path / "earlier/market.json").read_text())["status"] == "missing_inputs"
    paths = build_products(store, tmp_path / "products", OBS, "live")
    assert (
        paths
        and json.loads((tmp_path / "products/market.json").read_text())["status"] == "available"
    )


def test_access_evidence_scope_latest_and_policy_changes(entry, store, tmp_path):
    """场景：跨账户、fixture、过期政策和最新拒绝；预期：均不能借用历史成功放行。"""
    store.save_catalog([entry])
    with pytest.raises(IngestionError, match="live_evidence_store_required"):
        require_policy(entry, source="live")
    with pytest.raises(ValueError, match="successful_live_probe"):
        verified_policy(store, entry, "local")
    with pytest.raises(IngestionError, match="successful_live_probe"):
        require_policy(entry, store=store, source="live")
    evidence = {"account": "other", "request_policy": digest(asdict(entry.policy))}
    store.record_probe(entry.api_name, OBS, "live", "available", evidence)
    assert latest_probe(store, entry, "local") is None
    store.record_probe("other_api", OBS, "live", "available", evidence)
    store.record_probe(entry.api_name, OBS, "fixture", "available", evidence)
    with pytest.raises(ValueError, match="live_account_mismatch"):
        bind_account(store, "other")
    with pytest.raises(ValueError, match="invalid_account"):
        bind_account(store, "bad alias")
    probe(entry, client_for(tmp_path, [payload()]), store, OBS, "live")
    assert verified_policy(store, entry, "local") == entry.policy
    changed = replace(entry, policy=replace(entry.policy, lookback_days=9))
    assert latest_probe(store, changed, "local") is None
    evidence["account"] = "local"
    store.record_probe(entry.api_name, "2024-01-04T00:00:00Z", "live", "unauthorized", evidence)
    with pytest.raises(ValueError, match="successful_live_probe"):
        verified_policy(store, entry, "local")
    assert latest_probe(store, CatalogEntry("unknown"), "local") is None


def test_live_coverage_excludes_account_operations_even_with_points_label(entry, store):
    """场景：账户接口误标为积分研究接口；预期：仍不进入公共只读 live 覆盖分母。"""
    store.save_catalog([
        entry, replace(entry, api_name="p_save"), replace(entry, api_name="p_get"),
    ])
    report = coverage_report(store, "2024-01-03", "live")
    assert report["metrics"]["collection"]["denominator"] == 1


def test_missing_contract_does_not_count_as_live_collectable(entry, store):
    """场景：积分研究目录缺少采集政策；预期：离线报告保留政策债务，live 不算可采集。"""
    store.save_catalog([replace(entry, policy=None)])
    fixture_report = coverage_report(store, "2024-01-03")
    assert fixture_report["metrics"]["freshness"]["numerator"] == 0
    assert {"api_name": entry.api_name, "code": "missing_policy"} in fixture_report["debt"]
    assert coverage_report(store, "2024-01-03", "live")["enabled"] == 0


def test_structural_contract_write_and_account_guards(entry, store, tmp_path):
    """场景：写操作、无政策和不完整主键；预期：真实成功不能解除数据结构要求。"""
    with pytest.raises(IngestionError, match="account_write"):
        require_policy(replace(entry, api_name="p_delete"), for_probe=True)
    with pytest.raises(ValueError, match="missing_api_contract"):
        readable_policy(CatalogEntry("unknown"))
    private = replace(entry, api_name="p_get", policy=replace(entry.policy, lifecycle="blocked"))
    assert readable_policy(private).lifecycle == "archive"
    incomplete = replace(entry, policy=replace(entry.policy, primary_key=[]))
    store.save_catalog([incomplete])
    probe(incomplete, client_for(tmp_path, [payload()]), store, OBS, "live")
    with pytest.raises(ValueError, match="incomplete_policy"):
        verified_policy(store, incomplete, "local")
    task = {"source": "live", "account": "other"}
    with pytest.raises(IngestionError, match="live_account_mismatch"):
        run_task(task, entry, client_for(tmp_path, []), store, OBS)
    store.save_catalog([replace(entry, api_name="p_delete")])
    build_products(store, tmp_path / "products", OBS, "live")


def test_credential_sources_never_execute_configuration(tmp_path, monkeypatch):
    """场景：环境、引号文件、空值及不可读文件；预期：仅读取值，无命令执行或内容回显。"""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    assert configured_credential() is None
    monkeypatch.setenv("TUSHARE_TOKEN", "fixture-secret")
    assert configured_credential() == "fixture-secret"
    path = tmp_path / "credentials.env"
    path.write_text("IGNORE=no\nexport TUSHARE_TOKEN='fixture-file'\n")
    assert configured_credential(path) == "fixture-file"
    path.write_text("TUSHARE_TOKEN=fixture-unquoted\n")
    assert configured_credential(path) == "fixture-unquoted"
    path.write_text("TUSHARE_TOKEN=\n")
    with pytest.raises(ValueError, match="credential_file"):
        configured_credential(path)
    with pytest.raises(ValueError, match="credential_file"):
        configured_credential(tmp_path / "absent")


def test_survey_resume_samples_and_exclusions(entry, store, tmp_path):
    """场景：成功、拒绝、私有和写接口混合；预期：继续扫描，保留样本，重跑不发成功请求。"""
    entries = [
        entry,
        replace(entry, api_name="news"),
        replace(entry, api_name="p_save"),
        replace(entry, api_name="p_get"),
        CatalogEntry("unresolved_doc_1"),
    ]
    store.save_catalog(entries)
    client = client_for(tmp_path, [payload(), ProviderError("unauthorized", "denied")])
    result = survey(store, client, tmp_path / "survey", lambda: OBS, progress=lambda text: None)
    assert result["counts"] == {"available": 1, "unauthorized": 1, "excluded": 3}
    sample = json.loads((tmp_path / "survey/samples/index_daily.json").read_text())
    assert sample["scope"] == "probe_sample_only" and sample["rows"][0]["close"] == 2
    assert not store.records("tasks") and not store.rows(entry.api_name, "live")
    result = survey(
        store,
        client_for(tmp_path, []),
        tmp_path / "survey",
        lambda: OBS,
        progress=lambda text: None,
    )
    assert result["counts"]["available"] == 1
    result = survey(
        store,
        client_for(tmp_path, [payload([])]),
        tmp_path / "survey",
        lambda: OBS,
        api_names=[entry.api_name],
        refresh=True,
        progress=lambda text: None,
    )
    assert result["counts"] == {"empty": 1}
    with pytest.raises(ValueError, match="invalid_source"):
        survey(store, client, tmp_path, lambda: OBS, source="wrong")


def test_survey_bad_rows_remain_auditable(entry, store, tmp_path):
    """场景：成功响应缺少主键值；预期：权限成功独立记录，标准化失败不丢原始样本。"""
    store.save_catalog([entry])
    result = survey(
        store,
        client_for(tmp_path, [payload([[None, "20240102", 2]])]),
        tmp_path / "survey",
        lambda: OBS,
        source="fixture",
        progress=lambda text: None,
    )
    assert result["results"][0]["quality"][0]["code"] == "normalization_failed"


def test_provider_missing_interface_is_not_missing_permission(tmp_path):
    """场景：提供方明确要求正确接口名；预期：分类为不支持，不误报无权限。"""
    client = client_for(tmp_path, [{"code": 40101, "msg": "请指定正确的接口名"}])
    with pytest.raises(ProviderError) as failure:
        client.fetch("stk_account_old", {}, OBS)
    assert failure.value.kind == "unsupported"
