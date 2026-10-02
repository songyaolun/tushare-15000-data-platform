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
from tushare_entitlement_hub.reference import build_reference
from tushare_entitlement_hub.reference_collect import collect_reference_pages
from tushare_entitlement_hub.reference_contract import (
    ReferenceContract,
    canonical_fields,
    load_reference_contracts,
    normalize_reference_row,
    reference_value,
)
from tushare_entitlement_hub.reference_quality import evaluate_records, row_findings
from tushare_entitlement_hub.reference_storage import (
    atomic_json,
    publish_library,
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
    return {"code": 0, "data": {"fields": ["ts_code", "trade_date", "close"],
                              "items": [["A.SZ", "20260930", 2]] if items is None else items,
                              "has_more": more}}


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

    transport = HttpTransport(lambda: "fixture-secret", live=True,
                              http_transport=httpx.MockTransport(handle))
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


@pytest.mark.parametrize("source,code", [
    ({"trade_date": "20260930"}, "business_key_field_missing"),
    ({"ts_code": None, "trade_date": "20260930"}, "business_key_null"),
    ({"ts_code": "A", "trade_date": "broken"}, "invalid_field"),
    ({"ts_code": "A", "trade_date": "20260930", "close": "broken"}, "invalid_field"),
])
def test_invalid_reference_rows_are_not_accepted(source, code):
    """场景：缺键、空键、坏日期和非数值价格；预期：拒绝并输出不包含数据值的问题码。"""
    with pytest.raises(ValueError, match=code):
        normalize_reference_row(source, contract_for(), OBS, "hash", 0)


def test_non_scalar_business_key_is_rejected():
    """场景：自定义文本键返回对象；预期：不能将对象编码成伪业务键。"""
    contract = ReferenceContract(Policy("test", primary_key=["identity"]), {}, {})
    with pytest.raises(ValueError, match="not_scalar"):
        normalize_reference_row({"identity": {}}, contract, OBS, "hash", 0)


@pytest.mark.parametrize("value,field,kind,name,expected", [
    ("-", "close", "float", "daily", None),
    ("202609", "month", "str", "cn_pmi", "2026-09"),
    ("2026-09-30 00:00:00", "end_date", "str", "us_income", "2026-09-30"),
    ("9-26", "financial_date", "str", "us_fina_indicator", "9-26"),
    ("09:00-15:00", "trade_time", "str", "sge_basic", "09:00-15:00"),
    ("2026-09-30 09:31:00", "trade_time", "str", "stk_mins", "2026-09-30T01:31:00Z"),
    ("2026-09-30T01:31:00Z", "trade_time", "str", "stk_mins", "2026-09-30T01:31:00Z"),
    ("20260928-20260930", "period", "str", "stk_shock", "20260928-20260930"),
    ("2", "vol", "int", "daily", 2),
])
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


@pytest.mark.parametrize("responses,max_pages,expected", [
    ([response(more=True), response(more=True)], 3, "incomplete_repeated_page"),
    ([response([], more=True)], 3, "incomplete_no_progress"),
    ([response(more=True)], 1, "incomplete_page_budget"),
    ([response(more=None)], 3, "unknown_provider_limit"),
])
def test_incomplete_pagination_never_claims_complete(tmp_path, responses, max_pages, expected):
    """场景：重复页、空后页、超预算或无边界；预期：明确不完整且保留已有原始数据。"""
    contract = contract_for()
    contract.policy.supports_pagination = False
    entry = CatalogEntry("daily", policy=contract.policy)
    assert collect_reference_pages(entry, contract, fake_client(tmp_path, responses), {}, OBS,
                                   max_pages)[2] == expected


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
    assert collect_reference_pages(entry, contract, fake_client(tmp_path, [response(more=None)]),
                                   {}, OBS)[2] == "complete_contract_short_page"
    contract.policy.supports_pagination = False
    entry.api_name = "pro_bar"
    client = fake_client(tmp_path, [])
    page = fake_client(tmp_path, [response(more=None)]).fetch("daily", {}, OBS)
    client.fetch = lambda *args: page
    assert collect_reference_pages(entry, contract, client, {}, OBS)[2] == "complete_composed_range"


def test_published_files_checksum_and_duckdb_queries(tmp_path):
    """场景：正式数据与隔离数据发布；预期：SQL可查、校验可复用、损坏后拒绝复用。"""
    contract = contract_for()
    row = normalize_reference_row({"ts_code": "A", "trade_date": "20260930", "close": 2},
                                  contract, OBS, "hash", 0)
    files = write_dataset_files(tmp_path / "dataset", [row], [], contract)
    manifest = {"api_name": "daily", "status": "accepted", **files}
    assert verified_files(manifest)
    path = publish_library(tmp_path, [manifest])
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM accepted.daily").fetchone()[0] == 1
    parquet = Path(files["files"]["accepted"]["parquet"])
    parquet.write_bytes(b"broken")
    assert not verified_files(manifest)
    Path(files["files"]["accepted"]["jsonl"]).unlink()
    assert not verified_files(manifest)
    assert not verified_files({})
    with pytest.raises(ValueError):
        publish_library(tmp_path, [{"api_name": "a;bad", "status": "bad"}])


def test_build_reference_resume_scope_and_failures(store, tmp_path):
    """场景：完整、无权限、非API目录；预期：逐接口报告，恢复不重请求，篡改后重新采集。"""
    policies = load_policies()
    store.save_catalog([CatalogEntry("daily", policy=policies["daily"]),
                        CatalogEntry("news", policy=policies["news"]), CatalogEntry("non_api")])
    root = tmp_path / "reference"
    client = fake_client(tmp_path, [response(), ProviderError("unauthorized", "denied")])
    result = build_reference(
        store, client, root, lambda: OBS, source="fixture", progress=lambda _: None
    )
    assert result["counts"] == {"accepted": 1, "unauthorized": 1, "excluded": 1}
    assert (root / "datasets/news/source_dictionary.json").is_file()
    result = build_reference(store, fake_client(tmp_path, []), root, lambda: OBS,
                             api_names=["daily"], source="fixture", progress=lambda _: None)
    assert result["counts"]["accepted"] == 1
    with pytest.raises(ValueError, match="invalid_source"):
        build_reference(store, client, root, lambda: OBS, source="invalid")
    with pytest.raises(ValueError, match="reference_account_mismatch"):
        build_reference(store, client, root, lambda: OBS, source="live")
    atomic_json(root / "account.json", {"account": "local", "source": "live"})
    result = build_reference(store, fake_client(tmp_path, [response([])]), root, lambda: OBS,
                             api_names=["daily"], refresh=True, progress=lambda _: None)
    assert result["counts"]["empty"] == 1


def test_reference_cli_live_opt_in_and_catalog_required(tmp_path):
    """场景：未显式联网或空目录；预期：拒绝执行，不读取真实凭据或发送请求。"""
    runner = CliRunner()
    assert runner.invoke(app, ["reference", "build"]).exit_code != 0
    assert runner.invoke(app, ["--db", str(tmp_path / "hub.duckdb"),
                               "reference", "build", "--live"]).exit_code != 0


def test_mid_collection_failure_preserves_successful_pages(tmp_path):
    """场景：第一页成功、第二页服务拒绝；预期：原始页和已有记录保留且不声明完成。"""
    contract = contract_for()
    pages, rows, state = collect_reference_pages(
        CatalogEntry("daily", policy=contract.policy), contract,
        fake_client(tmp_path, [response(more=True), ProviderError("unauthorized", "denied")]),
        {}, OBS,
    )
    assert len(pages) == len(rows) == 1
    assert state == "incomplete_unauthorized" and pages[0]["processed_rows"] == 1


def test_stable_column_types_and_source_identity():
    """场景：文本日期、分钟、整数及同响应多次观察；预期：类型稳定、来源身份随观察变化。"""
    from tushare_entitlement_hub.reference_storage import reference_columns

    assert reference_columns(contract_for("stk_mins"), [])['trade_time'] == 'TIMESTAMPTZ'
    assert reference_columns(contract_for("us_fina_indicator"), [])['financial_date'] == 'VARCHAR'
    assert reference_columns(contract_for("trade_cal"), [])['is_open'] == 'VARCHAR'
    source = {'ts_code': 'A', 'trade_date': '20260930'}
    first = normalize_reference_row(source, contract_for(), OBS, 'h', 0)
    second = normalize_reference_row(source, contract_for(), '2026-10-03T00:00:00Z', 'h', 0)
    assert first['source_record_id'] != second['source_record_id']


@pytest.mark.parametrize('corruption', ['none', 'raw', 'identity', 'response', 'dictionary'])
def test_offline_rebuild_validates_raw_and_preserves_observation(store, tmp_path, corruption):
    """场景：规则重建或原始页/字典损坏；预期：无网络重建保留观察时点，坏原始页禁止发布。"""
    from tushare_entitlement_hub.reference import rebuild_reference
    from tushare_entitlement_hub.reference_storage import file_checksum

    store.save_catalog([CatalogEntry('daily', policy=contract_for().policy),
                        CatalogEntry('non_api')])
    root = tmp_path / 'reference'
    build_reference(store, fake_client(tmp_path, [response()]), root, lambda: OBS,
                    source='fixture', progress=lambda _: None)
    manifest_path = root / 'datasets/daily/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    raw_path = Path(manifest['pages'][0]['raw_path'])
    if corruption == 'raw':
        raw_path.write_text('broken')
        assert not verified_files(manifest)
    elif corruption in {'identity', 'response'}:
        envelope = json.loads(raw_path.read_text())
        if corruption == 'identity':
            envelope['api_name'] = 'news'
        else:
            envelope['response']['data']['items'][0][2] = 3
        atomic_json(raw_path, envelope)
        manifest['pages'][0]['file_sha256'] = file_checksum(raw_path)
        atomic_json(manifest_path, manifest)
    elif corruption == 'dictionary':
        Path(manifest['dictionary']).write_text('broken')
        assert not verified_files(manifest)
    result = rebuild_reference(root, progress=lambda _: None)
    daily = next(item for item in result['datasets'] if item['api_name'] == 'daily')
    assert daily['observed_at'] == OBS
    if corruption in {'raw', 'identity', 'response'}:
        assert daily['status'] == 'rebuild_failed' and not daily['files']
    else:
        assert daily['status'] == 'accepted' and verified_files(daily)
        with duckdb.connect(result['database'], read_only=True) as connection:
            assert connection.execute('SELECT close FROM accepted.daily').fetchone()[0] == 2


def test_skipped_corrupt_dataset_not_published(store, tmp_path):
    """场景：仅选择其他接口但旧产物损坏；预期：剔除损坏产物并标记校验失败。"""
    store.save_catalog([CatalogEntry('daily', policy=contract_for().policy)])
    root = tmp_path / 'reference'
    result = build_reference(store, fake_client(tmp_path, [response()]), root, lambda: OBS,
                             source='fixture', progress=lambda _: None)
    Path(result['datasets'][0]['files']['accepted']['jsonl']).write_text('broken')
    result = build_reference(store, fake_client(tmp_path, []), root, lambda: OBS,
                             api_names=[], source='fixture', progress=lambda _: None)
    assert result['counts'] == {'artifact_verification_failed': 1}


def test_invalid_publication_schema_rejected(tmp_path):
    """场景：清单伪造发布schema；预期：拒绝执行动态DDL。"""
    with pytest.raises(ValueError, match='invalid_dataset_schema'):
        publish_library(tmp_path, [{'api_name': 'daily', 'status': 'accepted',
                                    'files': {'wrong': {'parquet': 'unused'}}}])


def test_reference_cli_build_and_rebuild_offline(store, tmp_path, monkeypatch):
    """场景：CLI正式采集及重建；预期：假客户端完成导出，重建不需要任何客户端。"""
    from contextlib import contextmanager

    import tushare_entitlement_hub.cli as cli

    @contextmanager
    def client_context(*args):
        """为CLI提供离线客户端上下文；输入任意构造参数，输出合成客户端。"""
        yield fake_client(tmp_path, [response()])

    store.save_catalog([CatalogEntry('daily', policy=contract_for().policy)])
    monkeypatch.setattr(cli, 'open_client', client_context)
    monkeypatch.setattr(cli, 'open_store', lambda ctx: nullcontext(store))
    root = tmp_path / 'reference'
    parameters = tmp_path / 'params.json'
    atomic_json(parameters, {'daily': {'trade_date': '20260930'}})
    runner = CliRunner()
    result = runner.invoke(app, ['reference', 'build', '--live', '--apis', 'daily',
                                 '--output', str(root), '--parameters', str(parameters)])
    assert result.exit_code == 0, result.exception
    result = runner.invoke(app, ['reference', 'rebuild', '--output', str(root)])
    assert result.exit_code == 0, result.exception


def test_reference_cli_unknown_api(store, tmp_path, monkeypatch):
    """场景：CLI指定不存在接口；预期：在创建真实客户端前拒绝。"""
    import tushare_entitlement_hub.cli as cli

    store.save_catalog([CatalogEntry('daily', policy=contract_for().policy)])
    monkeypatch.setattr(cli, 'open_store', lambda ctx: nullcontext(store))
    result = CliRunner().invoke(app, ['reference', 'build', '--live', '--apis', 'missing'])
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == 'unknown_api'


def test_revision_date_list_preserves_every_revision(tmp_path):
    """场景：披露日期修正记录含多个日期；预期：解析为完整日期数组，坏日期仍拒绝。"""
    from tushare_entitlement_hub.reference_storage import reference_columns

    contract = contract_for('disclosure_date')
    assert reference_value('20020201,20020202', 'modify_date', 'str', contract) == [
        '2002-02-01', '2002-02-02']
    assert reference_columns(contract, [])['modify_date'] == 'DATE[]'
    with pytest.raises(ValueError):
        reference_value('20020201,broken', 'modify_date', 'str', contract)


def test_partition_collection_continues_after_failure(tmp_path):
    """场景：分区成功、拒绝、成功混合；预期：保留两份成功数据及失败分区，不声明全范围完成。"""
    from tushare_entitlement_hub.reference_collect import collect_reference_partitions

    contract = contract_for()
    entry = CatalogEntry('daily', policy=contract.policy)
    pages, rows, state, partitions = collect_reference_partitions(
        entry, contract, fake_client(tmp_path, [response(), ProviderError('unauthorized', 'x'),
                                               response([['B', '20260930', 2]])]),
        [{'ts_code': 'A'}, {'ts_code': 'denied'}, {'ts_code': 'B'}], OBS, lambda _: None,
    )
    assert len(rows) == len(pages) == 2 and state == 'incomplete_partitions'
    assert partitions[1]['status'] == 'unauthorized' and pages[1]['partition_index'] == 2
    with pytest.raises(ValueError, match='invalid_collection_partitions'):
        collect_reference_partitions(entry, contract, fake_client(tmp_path, []), [],
                                     OBS, lambda _: None)


def test_documented_revision_and_warehouse_dimensions_remain_distinct():
    """场景：财务更新版本及同品种仓库/厂库；预期：声明维度分别保存，不误判为冲突。"""
    rows = [{'ts_code': 'A', 'ann_date': '20260801', 'end_date': '20260630',
             'update_flag': flag, 'eps': value} for flag, value in [('0', 1), ('1', 2)]]
    accepted, rejected, _ = evaluate_records(
        [(row, 'hash', index, 'request') for index, row in enumerate(rows)],
        contract_for('fina_indicator_vip'), OBS,
    )
    assert len(accepted) == 2 and not rejected
    rows = [{'trade_date': '20260930', 'exchange': 'SHFE', 'fut_name': name}
            for name in ['氧化铝(仓库)', '氧化铝(厂库)']]
    accepted, rejected, _ = evaluate_records(
        [(row, 'hash', index, 'request') for index, row in enumerate(rows)],
        contract_for('fut_rcpt_mat'), OBS,
    )
    assert len(accepted) == 2 and not rejected


def test_source_identity_distinguishes_identical_responses_for_different_requests():
    """场景：两个分区恰好返回相同内容；预期：保留两个不同的来源请求身份。"""
    source = {'ts_code': 'A', 'trade_date': '20260930', 'close': 1}
    accepted, rejected, quality = evaluate_records(
        [(source, 'same-response', 0, 'request-a'), (source, 'same-response', 0, 'request-b')],
        contract_for(), OBS,
    )
    assert not rejected and quality['exact_duplicate_rows'] == 1
    assert len(set(accepted[0]['source_occurrences'])) == 2


def test_selective_rebuild_verifies_unselected_artifacts(store, tmp_path):
    """场景：局部重建时未选中表的旧文件损坏；预期：仍识别损坏，禁止混入重新发布的库。"""
    from tushare_entitlement_hub.reference import rebuild_reference

    store.save_catalog([CatalogEntry('daily', policy=contract_for().policy)])
    root = tmp_path / 'reference'
    result = build_reference(store, fake_client(tmp_path, [response()]), root, lambda: OBS,
                             source='fixture', progress=lambda _: None)
    Path(result['datasets'][0]['files']['accepted']['jsonl']).write_text('broken')
    result = rebuild_reference(root, api_names=[], progress=lambda _: None)
    assert result['counts'] == {'artifact_verification_failed': 1}
