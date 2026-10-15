"""通用数据参考的离线验收，覆盖字段、分页、隔离、发布、恢复和账户边界。"""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import duckdb
import httpx
import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.client import Client, FakeTransport, HttpTransport, ProviderError
from tushare_entitlement_hub.models import CatalogEntry, Policy
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.reference import (
    _manifest_dates,
    _records_from_raw_pages,
    build_reference,
    compose_reference,
    reference_collection_policy,
)
from tushare_entitlement_hub.reference_collect import collect_reference_pages
from tushare_entitlement_hub.reference_contract import (
    ReferenceContract,
    canonical_fields,
    load_reference_contracts,
    normalize_reference_row,
    reference_value,
)
from tushare_entitlement_hub.reference_quality import evaluate_records, row_findings
from tushare_entitlement_hub.reference_scope import (
    build_scope_certificate,
    certified_dates,
    merge_scope_certificates,
)
from tushare_entitlement_hub.reference_storage import (
    atomic_json,
    file_checksum,
    merge_dataset_files,
    publish_library,
    reference_fingerprint,
    verified_files,
    write_dataset_files,
)

OBS = "2026-10-02T00:00:00Z"


def contract_for(name="daily", **rules):
    """创建实际官方字段契约并覆盖测试规则；例如daily返回证券日期业务键。"""
    contract = load_reference_contracts(load_policies())[name]
    return replace(contract, rules={**contract.rules, **rules})


def response(items=None, more=False):
    """返回合成行情页；默认一条close=2的数据，more=True声明后续还有页面。"""
    return {
        "code": 0,
        "data": {
            "fields": ["ts_code", "trade_date", "close"],
            "items": [["A.SZ", "20260930", 2]] if items is None else items,
            "has_more": more,
        },
    }


def test_scope_certificates_separate_complete_requests_from_quality_failures():
    """场景：完整日期分区、失败分区和旧证书合并；预期：只认证完整且质量通过的日期。"""
    parameters = [{"trade_date": "20261008"}, {"trade_date": "2026-10-09"}]
    complete = build_scope_certificate(
        "daily",
        parameters,
        [
            {"parameters": parameters[0], "status": "complete_provider_response"},
            {"parameters": parameters[1], "status": "complete_contract_short_page"},
        ],
        "complete_partitions",
        OBS,
        {"status": "passed", "accepted_rows": 3, "quarantined_rows": 0},
    )
    failed = build_scope_certificate(
        "daily",
        parameters,
        [
            {"parameters": parameters[0], "status": "complete_provider_response"},
            {"parameters": parameters[1], "status": "transient_failure"},
        ],
        "incomplete_partitions",
        OBS,
        {"status": "partial", "accepted_rows": 1, "quarantined_rows": 1},
    )
    merged = merge_scope_certificates([complete], [complete, failed])
    assert complete["status"] == "certified" and failed["status"] == "incomplete"
    assert len(merged) == 2 and failed["failed_partitions"][0]["status"] == "transient_failure"
    assert certified_dates(merged, "daily") == ("2026-10-08", "2026-10-09")
    assert certified_dates(merged, "daily", "ann_date") == ()
    invalid_date = build_scope_certificate(
        "daily",
        {"trade_date": "not-a-date"},
        [{"parameters": {"trade_date": "not-a-date"}, "status": "complete_provider_response"}],
        "complete_partitions",
        OBS,
        {"status": "empty"},
    )
    assert invalid_date["parameter_dates"] == {}


def fake_client(tmp_path, responses):
    """创建完全离线且保存合成原始响应的客户端；响应序列按请求顺序消费。"""
    return Client(FakeTransport(responses), RawStore(tmp_path / "raw"), account="local")


def test_explicit_fields_use_protocol_top_level(tmp_path):
    """场景：补取默认隐藏字段；预期：fields在协议顶层，参数不混入fields。"""

    def handle(request):
        """校验离线POST结构并返回合成数据；不发送网络请求。"""
        payload = json.loads(request.content)
        assert payload["fields"] == "ts_code,close"
        assert payload["params"] == {"trade_date": "20260930"}
        return httpx.Response(200, json=response())

    transport = HttpTransport(
        lambda: "fixture-secret", live=True, http_transport=httpx.MockTransport(handle)
    )
    assert transport.request("daily", {"trade_date": "20260930", "fields": "ts_code,close"})
    with pytest.raises(ProviderError, match="输出字段"):
        transport.request("daily", {"fields": ["ts_code"]})


def test_contract_fields_aliases_nullable_and_lineage():
    """场景：大写和报价别名、合法可空维度；预期：保留源字段，业务键与原始身份分离。"""
    contract = contract_for("bc_otcqt")
    assert canonical_fields({"BID_BANK": "bank"}, contract) == {"bank": "bank"}
    with pytest.raises(ValueError, match="collision"):
        canonical_fields({"MONTH": 1, "month": 2}, contract)
    contract = contract_for("repurchase")
    source = {"ts_code": "a.sz", "ann_date": "20260930", "end_date": None, "proc": "实施"}
    row = normalize_reference_row(source, contract, OBS, "source-hash", 4)
    assert row["end_date"] is None and row["ts_code"] == "A.SZ"
    assert row["source_fields"] == source and row["source_record_id"] != row["business_key"]
    assert row["availability_basis"] == "disclosed_and_observed"
    assert not row["historical_pit_certified"]
    assert contract.fingerprint and "end_date" in contract.requested_fields


@pytest.mark.parametrize(
    "source,code",
    [
        ({"trade_date": "20260930"}, "business_key_field_missing"),
        ({"ts_code": None, "trade_date": "20260930"}, "business_key_null"),
        ({"ts_code": "A", "trade_date": "broken"}, "invalid_field"),
        ({"ts_code": "A", "trade_date": "20260930", "close": "broken"}, "invalid_field"),
    ],
)
def test_invalid_reference_rows_are_not_accepted(source, code):
    """场景：缺键、空键、坏日期和非数值价格；预期：拒绝并输出不包含数据值的问题码。"""
    with pytest.raises(ValueError, match=code):
        normalize_reference_row(source, contract_for(), OBS, "hash", 0)


def test_non_scalar_business_key_is_rejected():
    """场景：自定义文本键返回对象；预期：不能将对象编码成伪业务键。"""
    contract = ReferenceContract(Policy("test", primary_key=["identity"]), {}, {})
    with pytest.raises(ValueError, match="not_scalar"):
        normalize_reference_row({"identity": {}}, contract, OBS, "hash", 0)


@pytest.mark.parametrize(
    "value,field,kind,name,expected",
    [
        ("-", "close", "float", "daily", None),
        ("202609", "month", "str", "cn_pmi", "2026-09"),
        ("2026-09-30 00:00:00", "end_date", "str", "us_income", "2026-09-30"),
        ("9-26", "financial_date", "str", "us_fina_indicator", "9-26"),
        ("09:00-15:00", "trade_time", "str", "sge_basic", "09:00-15:00"),
        ("2026-09-30 09:31:00", "trade_time", "str", "stk_mins", "2026-09-30T01:31:00Z"),
        ("2026-09-30T01:31:00Z", "trade_time", "str", "stk_mins", "2026-09-30T01:31:00Z"),
        ("20260928-20260930", "period", "str", "stk_shock", "20260928-20260930"),
        ("2", "vol", "int", "daily", 2),
    ],
)
def test_field_semantics_do_not_confuse_periods_and_timestamps(value, field, kind, name, expected):
    """场景：空占位、期间、日期、分钟和时段文本；预期：按声明语义转换且不臆造日期。"""
    assert reference_value(value, field, kind, contract_for(name)) == expected


@pytest.mark.parametrize("value", [True, "nan", "inf", "1.5"])
def test_invalid_integer_values(value):
    """场景：布尔、非有限值和小数整数；预期：拒绝不合法值，nan文本作为缺失处理。"""
    if value == "nan":
        assert reference_value(value, "vol", "int", contract_for()) is None
    else:
        with pytest.raises(ValueError):
            reference_value(value, "vol", "int", contract_for())


def test_quality_counts_conflicts_duplicates_and_bad_values():
    """场景：重复、同键异值、非法价格混合；预期：精确去重留来源，冲突全隔离并计数守恒。"""
    rows = [
        {"ts_code": "A", "trade_date": "20260930", "close": 2},
        {"ts_code": "A", "trade_date": "20260930", "close": 2},
        {"ts_code": "B", "trade_date": "20260930", "close": 2},
        {"ts_code": "B", "trade_date": "20260930", "close": 3},
        {"ts_code": "C", "trade_date": "20260930", "close": "bad"},
    ]
    accepted, rejected, quality = evaluate_records(
        [(row, "hash", index, "request") for index, row in enumerate(rows)], contract_for(), OBS
    )
    assert len(accepted) == 1 and len(accepted[0]["source_occurrences"]) == 2
    assert len(rejected) == 3 and quality["exact_duplicate_rows"] == 1
    assert quality["status"] == "partial"
    assert row_findings({"open": 2, "high": 1, "low": 0, "close": 1}, contract_for())
    assert row_findings({"open": None, "high": 1, "low": 0, "close": 1, "vol": 2}, contract_for())
    assert row_findings({"amount": -1}, contract_for())
    assert row_findings({"adj_factor": 0}, contract_for("adj_factor"))
    assert evaluate_records([], contract_for(), OBS)[2]["status"] == "empty"
    nullable = {"ts_code": "A", "trade_date": "20260930", "close": None}
    result = evaluate_records([(nullable, "h", 0, "request")], contract_for(), OBS)
    assert result[2]["null_counts"]["close"] == 1


def test_provider_more_controls_complete_pagination(tmp_path):
    """场景：文档未写分页但提供方声明后页；预期：取完第二页，保留每页原始证据。"""
    contract = contract_for()
    contract.policy.supports_pagination = False
    client = fake_client(tmp_path, [response(more=True), response([["B", "20260930", 3]])])
    entry = CatalogEntry("daily", policy=contract.policy)
    pages, rows, state = collect_reference_pages(entry, contract, client, {}, OBS)
    assert len(pages) == 2 and len(rows) == 2 and state == "complete_provider_response"
    assert pages[1]["parameters"]["offset"] == 1


@pytest.mark.parametrize(
    "responses,max_pages,expected",
    [
        ([response(more=True), response(more=True)], 3, "incomplete_repeated_page"),
        ([response([], more=True)], 3, "incomplete_no_progress"),
        ([response(more=True)], 1, "incomplete_page_budget"),
        ([response(more=None)], 3, "unknown_provider_limit"),
    ],
)
def test_incomplete_pagination_never_claims_complete(tmp_path, responses, max_pages, expected):
    """场景：重复页、空后页、超预算或无边界；预期：明确不完整且保留已有原始数据。"""
    contract = contract_for()
    contract.policy.supports_pagination = False
    entry = CatalogEntry("daily", policy=contract.policy)
    assert (
        collect_reference_pages(
            entry, contract, fake_client(tmp_path, responses), {}, OBS, max_pages
        )[2]
        == expected
    )


def test_pagination_invalid_metadata_and_contract_fallback(tmp_path):
    """场景：无效分页元数据、短页和组合结果；预期：拒绝异常并分别识别有效终止依据。"""
    contract = contract_for()
    entry = CatalogEntry("daily", policy=contract.policy)
    with pytest.raises(ValueError):
        collect_reference_pages(entry, contract, fake_client(tmp_path, []), {}, OBS, 0)
    with pytest.raises(ProviderError):
        collect_reference_pages(
            entry, contract, fake_client(tmp_path, [response(more="bad")]), {}, OBS
        )
    contract.policy.supports_pagination = True
    assert (
        collect_reference_pages(
            entry, contract, fake_client(tmp_path, [response(more=None)]), {}, OBS
        )[2]
        == "complete_contract_short_page"
    )
    contract.policy.supports_pagination = False
    entry.api_name = "pro_bar"
    client = fake_client(tmp_path, [])
    page = fake_client(tmp_path, [response(more=None)]).fetch("daily", {}, OBS)
    client.fetch = lambda *args: page
    assert collect_reference_pages(entry, contract, client, {}, OBS)[2] == "complete_composed_range"


def test_published_files_checksum_and_duckdb_queries(tmp_path):
    """场景：正式数据与隔离数据发布；预期：SQL可查、校验可复用、损坏后拒绝复用。"""
    contract = contract_for()
    row = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "close": 2}, contract, OBS, "hash", 0
    )
    files = write_dataset_files(tmp_path / "dataset", [row], [], contract)
    manifest = {"api_name": "daily", "status": "accepted", **files}
    assert verified_files(manifest)
    path = publish_library(tmp_path, [manifest])
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM accepted.daily").fetchone()[0] == 1
    parquet = Path(files["files"]["accepted"]["parquet"])
    parquet.write_bytes(b"broken")
    assert not verified_files(manifest)
    assert not verified_files({})
    with pytest.raises(ValueError):
        publish_library(tmp_path, [{"api_name": "a;bad", "status": "bad"}])


def test_parquet_fragments_are_merged_and_latest_business_version_is_exposed(tmp_path):
    """场景：同一业务键先后发布两批；预期：保留两片明细，查询视图只暴露新观察版本。"""
    contract = contract_for()
    old = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "close": 2},
        contract,
        OBS,
        "old",
        0,
    )
    new = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "close": 3},
        contract,
        "2026-10-03T00:00:00Z",
        "new",
        0,
    )
    first = write_dataset_files(tmp_path / "first", [old], [], contract)
    second = write_dataset_files(tmp_path / "second", [new], [], contract)
    files = merge_dataset_files(first["files"], second["files"])
    manifest = {"api_name": "daily", "status": "accepted", "files": files}
    assert files["accepted"]["rows"] == 2 and len(files["accepted"]["fragments"]) == 2
    database = publish_library(tmp_path, [manifest])
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT close FROM accepted.daily").fetchone()[0] == 3
    assert reference_fingerprint(database) != ""
    Path(files["accepted"]["fragments"][0]["parquet"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="reference_artifact_verification_failed"):
        reference_fingerprint(database)


def test_reference_fingerprint_validates_sidecar_and_legacy_catalog_fallback(tmp_path):
    """场景：索引结构损坏、无索引旧库和非对象 manifest；预期：拒绝坏索引并兼容旧库。"""
    contract = contract_for()
    row = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "close": 2},
        contract,
        OBS,
        "hash",
        0,
    )
    files = write_dataset_files(tmp_path / "dataset", [row], [], contract)
    database = publish_library(
        tmp_path,
        [{"api_name": "daily", "status": "accepted", "files": files["files"]}],
    )
    index = tmp_path / "reference-artifacts.json"
    atomic_json(index, {"artifacts": {}})
    with pytest.raises(ValueError, match="reference_artifact_index_invalid"):
        reference_fingerprint(database)
    index.unlink()
    assert reference_fingerprint(database) == file_checksum(database)


def test_verified_files_supports_legacy_jsonl_and_detects_its_corruption(tmp_path):
    """场景：旧 manifest 同时登记 JSONL 和 Parquet；预期：兼容校验并检出 JSONL 改写。"""
    contract = contract_for()
    row = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "close": 2},
        contract,
        OBS,
        "hash",
        0,
    )
    files = write_dataset_files(tmp_path / "dataset", [row], [], contract)
    jsonl = tmp_path / "accepted.jsonl"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    item = files["files"]["accepted"] | {
        "jsonl": str(jsonl),
        "sha256": file_checksum(jsonl),
    }
    manifest = {"files": {"accepted": item}}
    assert verified_files(manifest)
    jsonl.write_text("changed", encoding="utf-8")
    assert not verified_files(manifest)
    assert not verified_files(
        {
            "files": {
                "accepted": {
                    "fragments": [
                        {
                            "parquet": str(tmp_path / "missing.parquet"),
                            "parquet_sha256": "missing",
                            "rows": 1,
                        }
                    ]
                }
            }
        }
    )
    existing = tmp_path / "wrong-hash.parquet"
    existing.write_bytes(b"present")
    assert not verified_files(
        {
            "files": {
                "accepted": {
                    "fragments": [
                        {
                            "parquet": str(existing),
                            "parquet_sha256": "wrong",
                            "rows": 1,
                        }
                    ]
                }
            }
        }
    )


def test_build_reference_resume_scope_and_failures(store, tmp_path):
    """场景：完整、无权限、非API目录；预期：逐接口报告，恢复不重请求，篡改后重新采集。"""
    policies = load_policies()
    store.save_catalog(
        [
            CatalogEntry("daily", policy=policies["daily"]),
            CatalogEntry("news", policy=policies["news"]),
            CatalogEntry("non_api"),
        ]
    )
    root = tmp_path / "reference"
    client = fake_client(tmp_path, [response(), ProviderError("unauthorized", "denied")])
    result = build_reference(
        store, client, root, lambda: OBS, source="fixture", progress=lambda _: None
    )
    assert result["counts"] == {"accepted": 1, "unauthorized": 1, "excluded": 1}
    assert (root / "datasets/news/source_dictionary.json").is_file()
    result = build_reference(
        store,
        fake_client(tmp_path, []),
        root,
        lambda: OBS,
        api_names=["daily"],
        source="fixture",
        progress=lambda _: None,
    )
    assert result["counts"]["accepted"] == 1
    with pytest.raises(ValueError, match="invalid_source"):
        build_reference(store, client, root, lambda: OBS, source="invalid")
    with pytest.raises(ValueError, match="reference_account_mismatch"):
        build_reference(store, client, root, lambda: OBS, source="live")
    atomic_json(root / "account.json", {"account": "local", "source": "live"})
    result = build_reference(
        store,
        fake_client(tmp_path, [response([])]),
        root,
        lambda: OBS,
        api_names=["daily"],
        refresh=True,
        progress=lambda _: None,
    )
    assert result["counts"]["empty"] == 1


def test_reference_cli_live_opt_in_and_catalog_required(tmp_path):
    """场景：未显式联网或空目录；预期：拒绝执行，不读取真实凭据或发送请求。"""
    runner = CliRunner()
    assert runner.invoke(app, ["reference", "build"]).exit_code != 0
    assert (
        runner.invoke(
            app, ["--db", str(tmp_path / "hub.duckdb"), "reference", "build", "--live"]
        ).exit_code
        != 0
    )


def test_mid_collection_failure_preserves_successful_pages(tmp_path):
    """场景：第一页成功、第二页服务拒绝；预期：原始页和已有记录保留且不声明完成。"""
    contract = contract_for()
    pages, rows, state = collect_reference_pages(
        CatalogEntry("daily", policy=contract.policy),
        contract,
        fake_client(tmp_path, [response(more=True), ProviderError("unauthorized", "denied")]),
        {},
        OBS,
    )
    assert len(pages) == len(rows) == 1
    assert state == "incomplete_unauthorized" and pages[0]["processed_rows"] == 1


def test_stable_column_types_and_source_identity():
    """场景：文本日期、分钟、整数及同响应多次观察；预期：类型稳定、来源身份随观察变化。"""
    from tushare_entitlement_hub.reference_storage import reference_columns

    assert reference_columns(contract_for("stk_mins"), [])["trade_time"] == "TIMESTAMPTZ"
    assert reference_columns(contract_for("us_fina_indicator"), [])["financial_date"] == "VARCHAR"
    assert reference_columns(contract_for("trade_cal"), [])["is_open"] == "VARCHAR"
    source = {"ts_code": "A", "trade_date": "20260930"}
    first = normalize_reference_row(source, contract_for(), OBS, "h", 0)
    second = normalize_reference_row(source, contract_for(), "2026-10-03T00:00:00Z", "h", 0)
    assert first["source_record_id"] != second["source_record_id"]


@pytest.mark.parametrize("corruption", ["none", "raw", "identity", "response", "dictionary"])
def test_offline_rebuild_validates_raw_and_preserves_observation(store, tmp_path, corruption):
    """场景：规则重建或原始页/字典损坏；预期：无网络重建保留观察时点，坏原始页禁止发布。"""
    from tushare_entitlement_hub.reference import rebuild_reference
    from tushare_entitlement_hub.reference_storage import file_checksum

    store.save_catalog(
        [CatalogEntry("daily", policy=contract_for().policy), CatalogEntry("non_api")]
    )
    root = tmp_path / "reference"
    build_reference(
        store,
        fake_client(tmp_path, [response()]),
        root,
        lambda: OBS,
        source="fixture",
        progress=lambda _: None,
    )
    manifest_path = root / "datasets/daily/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw_path = Path(manifest["pages"][0]["raw_path"])
    if corruption == "raw":
        raw_path.write_text("broken")
        assert not verified_files(manifest)
    elif corruption in {"identity", "response"}:
        envelope = json.loads(raw_path.read_text())
        if corruption == "identity":
            envelope["api_name"] = "news"
        else:
            envelope["response"]["data"]["items"][0][2] = 3
        atomic_json(raw_path, envelope)
        manifest["pages"][0]["file_sha256"] = file_checksum(raw_path)
        atomic_json(manifest_path, manifest)
    elif corruption == "dictionary":
        Path(manifest["dictionary"]).write_text("broken")
        assert not verified_files(manifest)
    result = rebuild_reference(root, progress=lambda _: None)
    daily = next(item for item in result["datasets"] if item["api_name"] == "daily")
    assert daily["observed_at"] == OBS
    if corruption in {"raw", "identity", "response"}:
        assert daily["status"] == "rebuild_failed" and not daily["files"]
    else:
        assert daily["status"] == "accepted" and verified_files(daily)
        with duckdb.connect(result["database"], read_only=True) as connection:
            assert connection.execute("SELECT close FROM accepted.daily").fetchone()[0] == 2


def test_skipped_corrupt_dataset_not_published(store, tmp_path):
    """场景：仅选择其他接口但旧产物损坏；预期：剔除损坏产物并标记校验失败。"""
    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    root = tmp_path / "reference"
    result = build_reference(
        store,
        fake_client(tmp_path, [response()]),
        root,
        lambda: OBS,
        source="fixture",
        progress=lambda _: None,
    )
    Path(result["datasets"][0]["files"]["accepted"]["parquet"]).write_text("broken")
    result = build_reference(
        store,
        fake_client(tmp_path, []),
        root,
        lambda: OBS,
        api_names=[],
        source="fixture",
        progress=lambda _: None,
    )
    assert result["counts"] == {"artifact_verification_failed": 1}


def test_invalid_publication_schema_rejected(tmp_path):
    """场景：清单伪造发布schema；预期：拒绝执行动态DDL。"""
    with pytest.raises(ValueError, match="invalid_dataset_schema"):
        publish_library(
            tmp_path,
            [
                {
                    "api_name": "daily",
                    "status": "accepted",
                    "files": {"wrong": {"parquet": "unused"}},
                }
            ],
        )


def test_reference_cli_build_and_rebuild_offline(store, tmp_path, monkeypatch):
    """场景：CLI正式采集及重建；预期：假客户端完成导出，重建不需要任何客户端。"""
    from contextlib import contextmanager

    import tushare_entitlement_hub.cli as cli

    @contextmanager
    def client_context(*args):
        """为CLI提供离线客户端上下文；输入任意构造参数，输出合成客户端。"""
        yield fake_client(tmp_path, [response()])

    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    monkeypatch.setattr(cli, "open_client", client_context)
    monkeypatch.setattr(cli, "open_store", lambda ctx: nullcontext(store))
    root = tmp_path / "reference"
    parameters = tmp_path / "params.json"
    atomic_json(parameters, {"daily": {"trade_date": "20260930"}})
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "reference",
            "build",
            "--live",
            "--apis",
            "daily",
            "--output",
            str(root),
            "--parameters",
            str(parameters),
        ],
    )
    assert result.exit_code == 0, result.exception
    result = runner.invoke(app, ["reference", "rebuild", "--output", str(root)])
    assert result.exit_code == 0, result.exception


def test_reference_cli_unknown_api(store, tmp_path, monkeypatch):
    """场景：CLI指定不存在接口；预期：在创建真实客户端前拒绝。"""
    import tushare_entitlement_hub.cli as cli

    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    monkeypatch.setattr(cli, "open_store", lambda ctx: nullcontext(store))
    result = CliRunner().invoke(app, ["reference", "build", "--live", "--apis", "missing"])
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "unknown_api"


def test_revision_date_list_preserves_every_revision(tmp_path):
    """场景：披露日期修正记录含多个日期；预期：解析为完整日期数组，坏日期仍拒绝。"""
    from tushare_entitlement_hub.reference_storage import reference_columns

    contract = contract_for("disclosure_date")
    assert reference_value("20020201,20020202", "modify_date", "str", contract) == [
        "2002-02-01",
        "2002-02-02",
    ]
    assert reference_columns(contract, [])["modify_date"] == "DATE[]"
    with pytest.raises(ValueError):
        reference_value("20020201,broken", "modify_date", "str", contract)


def test_partition_collection_continues_after_failure(tmp_path):
    """场景：分区成功、拒绝、成功混合；预期：保留两份成功数据及失败分区，不声明全范围完成。"""
    from tushare_entitlement_hub.reference_collect import collect_reference_partitions

    contract = contract_for()
    entry = CatalogEntry("daily", policy=contract.policy)
    pages, rows, state, partitions = collect_reference_partitions(
        entry,
        contract,
        fake_client(
            tmp_path,
            [response(), ProviderError("unauthorized", "x"), response([["B", "20260930", 2]])],
        ),
        [{"ts_code": "A"}, {"ts_code": "denied"}, {"ts_code": "B"}],
        OBS,
        lambda _: None,
    )
    assert len(rows) == len(pages) == 2 and state == "incomplete_partitions"
    assert partitions[1]["status"] == "unauthorized" and pages[1]["partition_index"] == 2
    with pytest.raises(ValueError, match="invalid_collection_partitions"):
        collect_reference_partitions(
            entry, contract, fake_client(tmp_path, []), [], OBS, lambda _: None
        )


def test_documented_revision_and_warehouse_dimensions_remain_distinct():
    """场景：财务更新版本及同品种仓库/厂库；预期：声明维度分别保存，不误判为冲突。"""
    rows = [
        {
            "ts_code": "A",
            "ann_date": "20260801",
            "end_date": "20260630",
            "update_flag": flag,
            "eps": value,
        }
        for flag, value in [("0", 1), ("1", 2)]
    ]
    accepted, rejected, _ = evaluate_records(
        [(row, "hash", index, "request") for index, row in enumerate(rows)],
        contract_for("fina_indicator_vip"),
        OBS,
    )
    assert len(accepted) == 2 and not rejected
    rows = [
        {"trade_date": "20260930", "exchange": "SHFE", "fut_name": name}
        for name in ["氧化铝(仓库)", "氧化铝(厂库)"]
    ]
    accepted, rejected, _ = evaluate_records(
        [(row, "hash", index, "request") for index, row in enumerate(rows)],
        contract_for("fut_rcpt_mat"),
        OBS,
    )
    assert len(accepted) == 2 and not rejected


def test_source_identity_distinguishes_identical_responses_for_different_requests():
    """场景：两个分区恰好返回相同内容；预期：保留两个不同的来源请求身份。"""
    source = {"ts_code": "A", "trade_date": "20260930", "close": 1}
    accepted, rejected, quality = evaluate_records(
        [(source, "same-response", 0, "request-a"), (source, "same-response", 0, "request-b")],
        contract_for(),
        OBS,
    )
    assert not rejected and quality["exact_duplicate_rows"] == 1
    assert len(set(accepted[0]["source_occurrences"])) == 2


def test_selective_rebuild_verifies_unselected_artifacts(store, tmp_path):
    """场景：局部重建时未选中表的旧文件损坏；预期：仍识别损坏，禁止混入重新发布的库。"""
    from tushare_entitlement_hub.reference import rebuild_reference

    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    root = tmp_path / "reference"
    result = build_reference(
        store,
        fake_client(tmp_path, [response()]),
        root,
        lambda: OBS,
        source="fixture",
        progress=lambda _: None,
    )
    Path(result["datasets"][0]["files"]["accepted"]["parquet"]).write_text("broken")
    result = rebuild_reference(root, api_names=[], progress=lambda _: None)
    assert result["counts"] == {"artifact_verification_failed": 1}


def test_quarantine_parquet_keeps_invalid_source_values(tmp_path):
    """场景：日期无法解析的隔离记录导出Parquet；预期：导出成功且原值与问题码完整保留。"""
    contract = contract_for()
    accepted, rejected, quality = evaluate_records(
        [({"ts_code": "A", "trade_date": "broken"}, "response", 0, "request")],
        contract,
        OBS,
    )
    assert quality["status"] == "failed"
    manifest = write_dataset_files(tmp_path / "dataset", accepted, rejected, contract)
    with duckdb.connect() as connection:
        row = connection.execute(
            "SELECT source_fields->>'trade_date', issues->>0 FROM read_parquet(?)",
            [manifest["files"]["quarantine"]["parquet"]],
        ).fetchone()
    assert row == ("broken", "invalid_field:trade_date")
    assert verified_files(manifest)


def test_incomplete_collection_is_retried_without_refresh(store, tmp_path):
    """场景：中途失败留下部分合格记录后重新运行；预期：继续尝试采集，不复用不完整结果。"""
    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    root = tmp_path / "reference"
    first = build_reference(
        store,
        fake_client(tmp_path, [response(more=True), ProviderError("unauthorized", "x")]),
        root,
        lambda: OBS,
        source="fixture",
        progress=lambda _: None,
    )
    assert first["counts"] == {"partial": 1}
    second = build_reference(
        store,
        fake_client(tmp_path, [response([["A", "20260930", 2], ["B", "20260930", 3]])]),
        root,
        lambda: OBS,
        source="fixture",
        progress=lambda _: None,
    )
    assert second["counts"] == {"accepted": 1}
    assert second["datasets"][0]["quality"]["source_rows"] == 2


def test_incremental_releases_keep_old_partitions_and_failed_attempt_keeps_release(store, tmp_path):
    """场景：连续两日成功后第三次网络失败；预期：两日可查，失败不覆盖活动 release。"""
    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    root = tmp_path / "reference"
    first = build_reference(
        store,
        fake_client(tmp_path, [response([["A", "20260929", 2]])]),
        root,
        lambda: "2026-10-02T00:00:00Z",
        parameters={"daily": {"trade_date": "20260929"}},
        source="fixture",
        progress=lambda _: None,
    )
    second = build_reference(
        store,
        fake_client(tmp_path, [response([["A", "20260930", 3]])]),
        root,
        lambda: "2026-10-03T00:00:00Z",
        parameters={"daily": {"trade_date": "20260930"}},
        source="fixture",
        progress=lambda _: None,
    )
    manifest = second["datasets"][0]
    assert len(manifest["released_runs"]) == 2
    assert certified_dates(manifest["scope_certificates"], "daily") == ("2026-09-29", "2026-09-30")
    with duckdb.connect(second["database"], read_only=True) as connection:
        assert connection.execute("SELECT count(*),sum(close) FROM accepted.daily").fetchone() == (
            2,
            5,
        )
    failed = build_reference(
        store,
        fake_client(tmp_path, [ProviderError("transient_failure", "x")] * 3),
        root,
        lambda: "2026-10-04T00:00:00Z",
        parameters={"daily": {"trade_date": "20261001"}},
        source="fixture",
        progress=lambda _: None,
    )
    retained = failed["datasets"][0]
    assert retained["release_status"] == "retained_after_failed_attempt"
    assert retained["latest_attempt"]["status"] == "transient_failure"
    assert retained["release_quality"]["accepted_rows"] == 2
    assert first["datasets"][0]["release_quality"]["accepted_rows"] == 1


def test_compose_reference_reuses_parquet_and_backfills_old_scope_dates(store, tmp_path):
    """场景：两个独立日批次零复制组合；预期：统一视图含两日并补出精确日期证书。"""
    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    sources = []
    for index, day in enumerate(("20260929", "20260930")):
        source = tmp_path / f"source-{index}"
        build_reference(
            store,
            fake_client(tmp_path / f"raw-{index}", [response([["A", day, index + 1]])]),
            source,
            lambda index=index: f"2026-10-0{index + 2}T00:00:00Z",
            parameters={"daily": {"trade_date": day}},
            source="fixture",
            progress=lambda _: None,
        )
        sources.append(source)
    first_manifest_path = sources[0] / "datasets/daily/manifest.json"
    first_manifest = json.loads(first_manifest_path.read_text())
    first_manifest.pop("scope_certificates")
    atomic_json(first_manifest_path, first_manifest)
    with duckdb.connect(str(sources[0] / "reference.duckdb")) as connection:
        connection.execute(
            "UPDATE catalog SET manifest=? WHERE api_name='daily'", [json.dumps(first_manifest)]
        )
    output = tmp_path / "composed"
    result = compose_reference(output, sources, progress=lambda _: None)
    manifest = result["datasets"][0]
    assert manifest["release_status"] == "composed"
    assert certified_dates(manifest["scope_certificates"], "daily") == ("2026-09-29", "2026-09-30")
    with duckdb.connect(result["database"], read_only=True) as connection:
        assert connection.execute("SELECT count(*),sum(close) FROM accepted.daily").fetchone() == (
            2,
            3,
        )
    assert not (output / "raw").exists()
    assert _manifest_dates(sources[0] / "reference.duckdb", "daily") == [
        {"trade_date": "2026-09-29"}
    ]
    with pytest.raises(ValueError, match="reference_compose_output_collision"):
        compose_reference(sources[0], sources, progress=lambda _: None)
    with pytest.raises(ValueError, match="reference_sources_required"):
        compose_reference(tmp_path / "absent", [], progress=lambda _: None)
    cli_output = tmp_path / "composed-cli"
    cli_result = CliRunner().invoke(
        app,
        [
            "reference",
            "compose",
            "--source",
            str(sources[0]),
            "--output",
            str(cli_output),
        ],
    )
    assert cli_result.exit_code == 0
    assert json.loads(cli_result.output.splitlines()[-1])["database"].endswith("reference.duckdb")


def test_compose_reference_rejects_account_contract_and_artifact_conflicts(store, tmp_path):
    """场景：来源账户、契约或 Parquet 冲突；预期：组合在发布前分别失败关闭。"""
    store.save_catalog([CatalogEntry("daily", policy=contract_for().policy)])
    sources = []
    for index in range(2):
        source = tmp_path / f"conflict-{index}"
        build_reference(
            store,
            fake_client(tmp_path / f"conflict-raw-{index}", [response()]),
            source,
            lambda index=index: f"2026-10-0{index + 2}T00:00:00Z",
            source="fixture",
            progress=lambda _: None,
        )
        sources.append(source)
    atomic_json(sources[1] / "account.json", {"account": "other", "source": "fixture"})
    with pytest.raises(ValueError, match="reference_account_mismatch"):
        compose_reference(tmp_path / "account-output", sources, progress=lambda _: None)
    atomic_json(sources[1] / "account.json", {"account": "local", "source": "fixture"})
    manifest_path = sources[1] / "datasets/daily/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract_hash"] = "different"
    atomic_json(manifest_path, manifest)
    with duckdb.connect(str(sources[1] / "reference.duckdb")) as connection:
        connection.execute(
            "UPDATE catalog SET manifest=? WHERE api_name='daily'", [json.dumps(manifest)]
        )
    with pytest.raises(ValueError, match="reference_contract_mismatch"):
        compose_reference(tmp_path / "contract-output", sources, progress=lambda _: None)
    manifest["contract_hash"] = contract_for().fingerprint
    atomic_json(manifest_path, manifest)
    with duckdb.connect(str(sources[1] / "reference.duckdb")) as connection:
        connection.execute(
            "UPDATE catalog SET manifest=? WHERE api_name='daily'", [json.dumps(manifest)]
        )
    Path(manifest["files"]["accepted"]["parquet"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="reference_source_artifact_verification_failed"):
        compose_reference(tmp_path / "artifact-output", sources, progress=lambda _: None)


def test_raw_page_recovery_rejects_identity_mismatch(tmp_path):
    """场景：原始页校验和更新但接口身份被改写；预期：离线恢复仍拒绝错误身份。"""
    client = fake_client(tmp_path, [response()])
    page = client.fetch("daily", {}, OBS)
    envelope = json.loads(page.raw.path.read_text())
    envelope["api_name"] = "news"
    atomic_json(page.raw.path, envelope)
    evidence = [
        {
            "raw_path": str(page.raw.path),
            "file_sha256": file_checksum(page.raw.path),
            "response_hash": page.raw.response_hash,
            "parameters": {},
            "processed_rows": 1,
        }
    ]
    with pytest.raises(ValueError, match="raw_identity_mismatch"):
        _records_from_raw_pages("daily", OBS, evidence)


def test_zero_volume_reference_close_is_usable_with_explicit_annotations():
    """场景：零成交时提供方用零开高低及非零收盘；预期：价格缺观测且保留参考收盘。"""
    contract = contract_for(zero_volume_reference_close=True)
    source = {
        "ts_code": "A",
        "trade_date": "20260930",
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 12.3,
        "vol": 0,
        "amount": 0,
    }
    accepted, rejected, quality = evaluate_records(
        [(source, "response", 0, "request")],
        contract,
        OBS,
    )
    assert not rejected and quality["usable_with_annotation_rows"] == 1
    assert accepted[0]["market_activity_status"] == "no_trade"
    assert accepted[0]["price_observation_status"] == "reference_close_without_trade"
    assert [accepted[0][field] for field in ("open", "high", "low")] == [None, None, None]
    assert accepted[0]["close"] == 12.3 and accepted[0]["source_fields"]["open"] == 0


def test_positive_volume_zero_prices_remain_quarantined():
    """场景：有成交量却返回零开高低和非零收盘；预期：继续隔离，不能套用无成交口径。"""
    contract = contract_for(zero_volume_reference_close=True)
    source = {
        "ts_code": "A",
        "trade_date": "20260930",
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 12.3,
        "vol": 1,
        "amount": 12.3,
    }
    accepted, rejected, quality = evaluate_records(
        [(source, "response", 0, "request")],
        contract,
        OBS,
    )
    assert not accepted and rejected[0]["issues"] == ["invalid_ohlc_range"]
    assert quality["quarantined_rows"] == 1


def test_zero_volume_all_zero_prices_become_missing_observation():
    """场景：零成交且所有价格为0；预期：标准价格全部为空，不能把0当作真实市场价格。"""
    contract = contract_for(zero_volume_reference_close=True)
    source = {
        "ts_code": "A",
        "trade_date": "20260930",
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
        "vol": 0,
        "amount": 0,
    }
    accepted, rejected, _ = evaluate_records(
        [(source, "response", 0, "request")],
        contract,
        OBS,
    )
    assert not rejected
    assert [accepted[0][field] for field in ("open", "high", "low", "close")] == [
        None,
        None,
        None,
        None,
    ]
    assert accepted[0]["price_observation_status"] == "no_price_observed"


def test_provider_date_sentinel_becomes_annotated_null():
    """场景：提供方用字符串0表示上市日未确定；预期：标准日期为空并记录空值原因。"""
    contract = contract_for(null_sentinels={"list_date": ["0"]})
    row = normalize_reference_row(
        {"ts_code": "A", "trade_date": "20260930", "list_date": "0"},
        contract,
        OBS,
        "response",
        0,
        "request",
    )
    assert row["list_date"] is None
    assert row["field_quality"] == {"list_date": "provider_sentinel"}


def test_business_group_variants_remain_usable_without_overwriting():
    """场景：同一业务事件有两条不同版本及一条重复；预期：保留两个变体并合并完全重复来源。"""
    contract = contract_for(identity_policy="preserve_source_variants", variant_fields=["close"])
    records = [
        ({"ts_code": "A", "trade_date": "20260930", "close": 2}, "r1", 0, "q1"),
        ({"ts_code": "A", "trade_date": "20260930", "close": 3}, "r2", 0, "q2"),
        ({"ts_code": "A", "trade_date": "20260930", "close": 3}, "r3", 0, "q3"),
    ]
    accepted, rejected, quality = evaluate_records(records, contract, OBS)
    assert not rejected and len(accepted) == 2
    assert len({row["business_group_key"] for row in accepted}) == 1
    assert len({row["business_key"] for row in accepted}) == 2
    assert {row["business_variant_dimensions"]["close"] for row in accepted} == {2.0, 3.0}
    assert {row["identity_resolution"] for row in accepted} == {"business_group_source_variant"}
    assert quality["business_variant_rows"] == 2
    assert quality["usable_with_annotation_rows"] == 2
    assert quality["exact_duplicate_rows"] == 1 and quality["status"] == "partial"


def test_nullable_descriptive_business_dimension_is_preserved():
    """场景：仓单品种名称缺失但代码、交易所和仓库齐全；预期：保留空描述字段及业务身份。"""
    contract = contract_for("fut_wsr")
    source = {
        "trade_date": "20240102",
        "symbol": "PTA",
        "warehouse": "华润化学",
        "year": None,
        "grade": None,
        "brand": None,
        "place": None,
        "is_ct": None,
        "exchange": "CZCE",
        "fut_name": None,
        "area": None,
    }
    row = normalize_reference_row(source, contract, OBS, "response", 0, "request")
    assert row["fut_name"] is None
    assert row["business_dimensions"]["symbol"] == "PTA"


def test_rate_limited_collection_policy_records_retry_window_and_partition():
    """场景：小时限频接口发生临时失败；预期：记录下一可重试时间及节省请求的分区策略。"""
    policy = reference_collection_policy(contract_for("hk_daily"), OBS, "transient_failure")
    assert policy == {
        "rate_per_minute": 1 / 60,
        "partition_strategy": "trade_date_full_market",
        "retry_eligible_at": "2026-10-02T01:00:00Z",
    }
    assert reference_collection_policy(contract_for(), OBS) == {
        "rate_per_minute": 60,
        "partition_strategy": "declared_request",
    }


def test_offline_rebuild_adds_retry_policy_to_rate_limited_failure(tmp_path):
    """场景：旧清单只有限频失败状态；预期：离线重建补上重试时间，不发起网络请求。"""
    from tushare_entitlement_hub.reference import rebuild_reference

    root = tmp_path / "reference"
    atomic_json(root / "account.json", {"account": "local", "source": "live"})
    atomic_json(
        root / "datasets/hk_daily/manifest.json",
        {
            "api_name": "hk_daily",
            "status": "transient_failure",
            "observed_at": OBS,
        },
    )
    result = rebuild_reference(root, progress=lambda _: None)
    manifest = result["datasets"][0]
    assert manifest["collection_policy"]["partition_strategy"] == "trade_date_full_market"
    assert manifest["collection_policy"]["retry_eligible_at"] == "2026-10-02T01:00:00Z"
