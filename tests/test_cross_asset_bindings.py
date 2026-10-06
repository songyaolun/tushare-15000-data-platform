"""全接口绑定与新增输入适配的离线契约测试，所有来源均为可复算合成数据。"""

from dataclasses import replace
from unittest.mock import Mock

import duckdb
import pytest
from test_research_adapter import configured, reference_database, source
from test_research_binding_fixture import AS_OF, DATABASE_CONFIG, write_reference
from test_research_governance import row

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cross_asset_research import adapter, binding_rules, bindings
from tushare_entitlement_hub.cross_asset_research.adapter import adapt_row, load_inputs, retain_rows
from tushare_entitlement_hub.cross_asset_research.binding_rules import (
    cross_registry,
    entity_mapping,
    rules_fingerprint,
    time_registry,
)
from tushare_entitlement_hub.cross_asset_research.bindings import (
    build_config,
    equity_config,
    verify_binding_audit,
)
from tushare_entitlement_hub.equity_research.adapter import adapter_reason
from tushare_entitlement_hub.equity_research.binding_entities import EntityBuild
from tushare_entitlement_hub.equity_research.binding_io import BuildLimits
from tushare_entitlement_hub.equity_research.binding_units import currency_evidence, unit_definition
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.reference_contract import (
    load_reference_contracts,
    normalize_reference_row,
)
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import digest, encode

CONTRACTS = load_reference_contracts(load_policies())
MAPPINGS = {item.source_api: item for item in time_registry(CONTRACTS)}


def write_all_reference(directory, datasets=None):
    """扩展旧离线夹具至206项清单；例如fund_share一行时其余新增接口明确缺表。

    默认保留两只合成股票，所有来源清单均使用当前真实契约指纹。
    """
    reference = write_reference(directory, datasets)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        present = {
            item[0] for item in connection.execute("SELECT api_name FROM catalog").fetchall()
        }
        for entry in cross_registry():
            if entry.api in present:
                continue
            manifest = {
                "contract_hash": CONTRACTS[entry.api].fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": 0,
                    "accepted_rows": 0,
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?, ?)", [entry.api, encode(manifest)])
    return reference


def cross_config(*apis, **changes):
    """构造仅显式声明新来源映射的配置；例如fx_daily无证券别名，只允许来源内序列。

    其他限额由changes覆盖，旧股票别名由独立configured夹具提供。
    """
    config = configured(*apis)
    return replace(
        config,
        foundation=replace(
            config.foundation,
            time_mappings=tuple(MAPPINGS[api] for api in apis),
            entity_mappings=tuple(entity_mapping(api) for api in apis),
        ),
        **changes,
    )


def test_rules_cover_exact_107_and_keep_market_namespaces_separate():
    """场景：全七产品注册；预期：107项显式时间规则，港美汇率身份互不混用。"""
    assert len(cross_registry()) == len(MAPPINGS) == 107
    assert MAPPINGS["forecast"].report_period_field == "end_date"
    assert MAPPINGS["forecast"].optional_public_fields == ("ann_date", "first_ann_date")
    assert MAPPINGS["us_daily"].market_timezone == "America/New_York"
    assert MAPPINGS["hk_income"].optional_public_fields == ()
    assert entity_mapping("index_weight").key_field == "index_code"
    assert entity_mapping("ths_member").key_field == "ts_code"
    assert len({entity_mapping(api).namespace for api in ("hk_basic", "us_basic", "fx_daily")}) == 3
    assert len(rules_fingerprint(CONTRACTS)) == 64


@pytest.mark.parametrize("variant", ["field", "missing", "duplicate"])
def test_explicit_time_rule_drift_fails_closed(monkeypatch, variant):
    """场景：规则使用不存在字段、遗漏API或重复API；预期：拒绝生成推断规则。"""
    original = binding_rules.TIME_RULES
    if variant == "field":
        replacement = original.replace("fx_daily|trade_date", "fx_daily|missing_field")
    elif variant == "missing":
        replacement = original.replace("fx_daily|trade_date||||\n", "")
    else:
        replacement = original + "fx_daily|trade_date||||\n"
    monkeypatch.setattr(binding_rules, "TIME_RULES", replacement)
    with pytest.raises(ValueError, match="binding_time_(contract|registry)_mismatch"):
        time_registry(CONTRACTS)


def test_build_config_certifies_fund_shares_once_and_keeps_legacy_99(tmp_path):
    """场景：基金份额标准化加海外主数据；预期：206配置、一次份额换算、旧范围仍为99。"""
    reference = write_all_reference(
        tmp_path,
        {
            "fund_basic": [{"ts_code": "510001.SH", "list_date": "20000101", "market": "E"}],
            "fund_share": [{"ts_code": "510001.SH", "trade_date": "20261002", "fd_share": 2}],
            "etf_share_size": [
                {"ts_code": "510001.SH", "trade_date": "20261002", "total_share": 2}
            ],
            "hk_basic": [{"ts_code": "ABC", "list_date": "20000101", "curr_type": "HKD"}],
            "us_basic": [{"ts_code": "ABC", "list_date": "20000101"}],
            "forecast": [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20261001",
                    "first_ann_date": "20260901",
                    "end_date": "20261231",
                    "type": "预增",
                    "net_profit_min": 1,
                    "net_profit_max": 2,
                }
            ],
        },
    )
    checksum = file_checksum(reference)
    result = build_config(reference, AS_OF)
    assert result.report is result.audit
    assert len(result.config.foundation.time_mappings) == 206
    assert result.audit["summary"]["observed_apis"] == 0
    assert result.audit["sources"]["fund_share"]["readable"] is True
    assert result.audit["sources"]["fx_daily"]["configured"] is True
    certificate = next(
        item
        for item in result.config.foundation.certifications
        if (item.source_api, item.field) == ("etf_share_size", "total_share")
    )
    assert certificate.target_unit == "份" and certificate.multiplier == 1
    inputs, counts = load_inputs(reference, AS_OF, "run", result.config)
    assert inputs.rows("fund_share")[0].number("fd_share", "份") is None
    assert inputs.rows("etf_share_size")[0].number("total_share", "份") == 20000
    assert inputs.rows("forecast")[0].number("net_profit_min", "元") == 10000
    assert (
        counts["scanned_rows"]
        == counts["retained_rows"] + counts["excluded_rows"] + counts["discarded_rows"]
    )
    old = equity_config(result.config)
    assert len(old.foundation.time_mappings) == 99 and old.binding_audit == {}
    assert all(
        item.identity.namespace not in {"tushare_fund", "tushare_hk", "tushare_us"}
        for item in old.foundation.entities
    )
    assert verify_binding_audit(result.config, checksum, AS_OF) is None
    assert build_config(reference, AS_OF).config.to_json() == result.config.to_json()
    assert file_checksum(reference) == checksum


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("missing", "reference_manifest_missing"),
        ("duplicate", "reference_catalog_duplicate"),
        ("contract", "source_contract_mismatch"),
    ],
)
def test_current_source_manifest_is_required_unique_and_contract_bound(tmp_path, scenario, reason):
    """场景：目标清单缺失、重复或契约改变；预期：全配置构建拒绝，无选择性清单回退。"""
    reference = write_all_reference(tmp_path)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        if scenario == "missing":
            connection.execute("DELETE FROM catalog WHERE api_name='fx_daily'")
        elif scenario == "duplicate":
            connection.execute(
                "INSERT INTO catalog SELECT * FROM catalog WHERE api_name='fx_daily'"
            )
        else:
            connection.execute(
                "UPDATE catalog SET manifest=? WHERE api_name='fx_daily'",
                [encode({"contract_hash": "changed"})],
            )
    with pytest.raises(ValueError, match=reason):
        build_config(reference, AS_OF)


def test_extra_catalog_items_are_not_scanned_or_parsed(tmp_path):
    """场景：目录包含重复且载荷非法的额外项；预期：206目标照常构建，不读取额外载荷。"""
    reference = write_all_reference(tmp_path)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("INSERT INTO catalog VALUES ('extra', 'invalid'), ('extra', 'invalid')")
    result = build_config(reference, AS_OF)
    assert "extra" not in result.audit["sources"]
    inputs, counts = load_inputs(reference, AS_OF, "run", ResearchConfig())
    assert counts["scanned_rows"] == 0
    assert "research_bindings_missing" in inputs.source_reasons["fx_daily"]


def test_product_source_subset_scans_only_requested_interfaces(tmp_path):
    """场景：执行阶段只选择一个产品来源子集；预期：不读取或报告其他产品接口。"""
    reference = write_all_reference(
        tmp_path,
        {
            "fx_daily": [{"ts_code": "USD/CNY", "trade_date": "20261002"}],
            "fund_basic": [{"ts_code": "FUND", "list_date": "20200101"}],
        },
    )
    config = build_config(reference, AS_OF).config
    inputs, counts = load_inputs(
        reference,
        AS_OF,
        "run",
        config,
        source_apis=frozenset({"fx_daily"}),
    )
    assert set(inputs.rows_by_api) == set(inputs.source_reasons) == {"fx_daily"}
    assert counts["scanned_rows"] == 1
    with pytest.raises(ValueError, match="input_source_selection_invalid"):
        load_inputs(reference, AS_OF, "run", config, source_apis=frozenset({"unknown"}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_total_rows", 8),
        ("maximum_source_rows", 2),
        ("maximum_entities", 2),
    ],
)
def test_build_capacity_rejects_whole_config(tmp_path, field, value):
    """场景：新增来源使全量、单源或实体预算超限；预期：整体拒绝，不交付部分证书。"""
    datasets = {
        "fund_basic": [{"ts_code": f"F{index}", "list_date": "20000101"} for index in range(3)]
    }
    reference = write_all_reference(tmp_path, datasets)
    with pytest.raises(ValueError, match="binding_(scan|entity)_capacity_exceeded"):
        build_config(reference, AS_OF, limits=BuildLimits(**{field: value}))


@pytest.mark.parametrize("kind", ["audit", "reference", "cutoff", "config", "version", "rules"])
def test_unified_audit_guards_each_binding_boundary(tmp_path, kind):
    """场景：分别篡改审计、来源、时点、配置或版本；预期：统一校验给固定拒绝码。"""
    result = build_config(write_all_reference(tmp_path), AS_OF)
    config, checksum, cutoff = result.config, result.audit["reference_sha256"], AS_OF
    reason = "binding_configuration_mismatch"
    if kind == "reference":
        checksum, reason = "different", "binding_reference_mismatch"
    elif kind == "cutoff":
        cutoff, reason = "2026-10-04T12:00:00Z", "binding_cutoff_mismatch"
    elif kind == "config":
        config = replace(config, maximum_total_rows=2)
    else:
        audit = config.binding_audit
        if kind == "audit":
            audit["summary"] = {}
            reason = "binding_audit_mismatch"
        else:
            audit["version" if kind == "version" else "rules_fingerprint"] = "changed"
            audit.pop("audit_sha256")
            audit["audit_sha256"] = digest(audit)
        config = replace(config, binding_audit_json=encode(audit))
    with pytest.raises(ValueError, match=reason):
        verify_binding_audit(config, checksum, cutoff)
    assert verify_binding_audit(ResearchConfig(), "manual", AS_OF) is None


def test_source_local_entities_never_claim_security_aliases():
    """场景：两种非证券来源同代码；预期：来源内ID隔离，证券缺alias仍拒绝。"""
    config = cross_config("fx_daily", "sge_daily", "us_daily")
    foreign = adapt_row(source("fx_daily", ts_code="ABC"), CONTRACTS["fx_daily"], config, AS_OF)
    spot = adapt_row(source("sge_daily", ts_code="ABC"), CONTRACTS["sge_daily"], config, AS_OF)
    assert foreign.entity_id != spot.entity_id
    assert foreign.raw["entity_basis"] == "source_local"
    with pytest.raises(ValueError, match="entity_unresolved"):
        adapt_row(source("us_daily", ts_code="ABC"), CONTRACTS["us_daily"], config, AS_OF)


@pytest.mark.parametrize(
    "value,reason", [(None, "source_local_key_missing"), ("x" * 257, "source_local_key_invalid")]
)
def test_source_local_keys_require_bounded_explicit_identity(value, reason):
    """场景：来源代码缺失或过大；预期：来源内身份也拒绝，不能把缺代码行合并。"""
    with pytest.raises(ValueError, match=reason):
        adapt_row(
            source("fx_daily", ts_code=value),
            CONTRACTS["fx_daily"],
            cross_config("fx_daily"),
            AS_OF,
        )


def test_market_series_and_complex_local_code_are_deterministic():
    """场景：市场总序列无代码、商品代码含中文；预期：市场明确分组，复杂代码确定性摘要。"""
    market = adapt_row(
        source("shibor", date="20261002", ts_code=None),
        CONTRACTS["shibor"],
        cross_config("shibor"),
        AS_OF,
    )
    complex_row = adapt_row(
        source("sge_daily", ts_code="合成品种"),
        CONTRACTS["sge_daily"],
        cross_config("sge_daily"),
        AS_OF,
    )
    assert market.entity_id == "source:series:shibor:market"
    assert complex_row.entity_id == "source:series:sge_daily:" + digest("合成品种")


@pytest.mark.parametrize(
    "patch,reason",
    [
        ({"observed_at": "2026-10-04T12:00:00Z"}, "row_unavailable"),
        ({"quality_tier": "quarantine"}, "quarantined_or_unknown_quality"),
        ({"trade_date": "20261004"}, "future_business_date"),
        ({"ann_date": "20261004"}, "public_field_mapping_missing"),
    ],
)
def test_new_source_exclusions_have_specific_fixed_reasons(tmp_path, patch, reason):
    """场景：未来、隔离或隐藏公告行；预期：准确排除原因及计数守恒，不进入保留行。"""
    records = [source("fx_daily", "ok"), source("fx_daily", "bad", **patch)]
    path = reference_database(tmp_path, {"fx_daily": records})
    result, counts = load_inputs(path, AS_OF, "run", cross_config("fx_daily"))
    assert counts["scanned_rows"] == 2 and counts["excluded_rows"] == 1
    assert counts["retained_rows"] == 1 and reason in result.source_reasons["fx_daily"]
    assert counts["discarded_rows"] == 0


def test_history_retention_preserves_two_complete_periods_and_event_versions():
    """场景：三期行情及公告版本；预期：释放最旧行情，事件版本保留用于修正分析。"""
    rows = []
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        rows = retain_rows(rows, row("fx_daily", day, day=day))
    assert len(rows) == 2
    assert rows[0].availability.data_as_of == "2026-09-02"
    versions = []
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        versions = retain_rows(versions, row("forecast", day, day=day))
    assert len(versions) == 3


@pytest.mark.parametrize("limit", ["entity", "source", "remaining", "bytes"])
def test_input_overflow_rejects_whole_group_and_keeps_scanning(tmp_path, monkeypatch, limit):
    """场景：实体、来源、剩余全局行数或字节超限；预期：整组拒绝且继续全量计数。"""
    records = [
        source("fx_daily", str(index), ts_code="A" if limit == "entity" else f"A{index}")
        for index in range(4)
    ]
    path = reference_database(tmp_path, {"fx_daily": records})
    config = cross_config(
        "fx_daily",
        **(
            {"maximum_rows_per_entity_source": 1}
            if limit == "entity"
            else {"maximum_source_rows": 1}
            if limit == "source"
            else {}
        ),
    )
    if limit == "bytes":
        monkeypatch.setattr(adapter, "MAX_SOURCE_BYTES", 1)
    progress = []
    result, counts = load_inputs(
        path,
        AS_OF,
        "run",
        config,
        progress=progress.append,
        remaining_rows=0 if limit == "remaining" else None,
    )
    assert counts["scanned_rows"] == counts["discarded_rows"] == 4
    assert counts["retained_rows"] == 0 and result.rows("fx_daily") == ()
    assert progress
    if limit == "entity":
        assert result.entity_source_reasons
    else:
        assert (
            "source_byte_capacity_exceeded" if limit == "bytes" else "source_capacity_exceeded"
        ) in result.source_reasons["fx_daily"]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_remaining_capacity_is_rejected_before_open(tmp_path, value):
    """场景：调用方传非法剩余预算；预期：拒绝输入容量，不能把布尔值解释为1行。"""
    path = write_all_reference(tmp_path)
    with pytest.raises(ValueError, match="input_capacity_invalid"):
        load_inputs(path, AS_OF, "run", ResearchConfig(), remaining_rows=value)


def test_binding_scan_excludes_invalid_rows_and_snapshot_has_no_freshness(tmp_path):
    """场景：新增主数据未来观察、行情缺业务日；预期：缺口明确且不认证错误的新鲜度。"""
    reference = write_all_reference(
        tmp_path,
        {
            "fund_basic": [{"ts_code": "F", "list_date": "20000101"}],
            "fund_share": [{"ts_code": "F", "trade_date": "20261002", "fd_share": 1}],
        },
    )
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("UPDATE accepted.fund_basic SET observed_at='2026-12-01T00:00:00Z'")
        connection.execute("UPDATE accepted.fund_share SET trade_date=NULL")
    result = build_config(reference, AS_OF)
    assert "binding_future_observation" in result.audit["sources"]["fund_basic"]["reason_counts"]
    assert "fund_share" not in result.config.foundation.expected_data_as_of


def test_changed_reference_and_undeclared_public_field_fail_build(tmp_path, monkeypatch):
    """场景：扫描期间参考库变化或遗漏公开字段；预期：禁止发布对应配置。"""
    reference = write_all_reference(
        tmp_path, {"fund_share": [{"ts_code": "F", "trade_date": "20261002", "fd_share": 1}]}
    )
    monkeypatch.setattr(bindings, "file_checksum", Mock(return_value="changed"))
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        build_config(reference, AS_OF)
    monkeypatch.undo()
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("ALTER TABLE accepted.fund_share ADD COLUMN pub_date VARCHAR")
        connection.execute("UPDATE accepted.fund_share SET pub_date='20261001'")
    with pytest.raises(ValueError, match="public_field_mapping_missing"):
        build_config(reference, AS_OF)


def test_explicit_foreign_currency_does_not_inherit_stock_currency():
    """场景：显式美元与错误股币种继承；预期：可仅认证本行美元，冲突声明始终拒绝。"""
    record = source("us_daily", currency="USD")
    availability = row("us_daily", raw=record).availability
    assert (
        currency_evidence(
            record,
            availability,
            EntityBuild(5),
            allowed_currencies=frozenset({"USD"}),
            inherit_stock_currency=False,
        )[0]
        == "USD"
    )
    assert currency_evidence(
        record | {"curr_type": "CNY"},
        availability,
        EntityBuild(5),
        allowed_currencies=frozenset({"USD", "CNY"}),
        inherit_stock_currency=False,
    ) == (None, ())
    assert adapter_reason(RuntimeError("untrusted payload")) == "row_adaptation_failed"
    assert adapter_reason(ValueError("entity_unresolved")) == "entity_unresolved"


def test_fund_unit_normalization_and_binding_share_one_conversion_table():
    """场景：万份规范化后再绑定；预期：20000份只认证倍率1，缺标准单位拒绝二次放大。"""
    contract = CONTRACTS["etf_share_size"]
    normalized = normalize_reference_row(
        {"ts_code": "F", "trade_date": "20261002", "total_share": 2},
        contract,
        AS_OF,
        "synthetic",
        0,
    )
    assert normalized["total_share"] == 20000
    assert unit_definition(normalized, "total_share", contract) == ("份", "份", "quantity", 1.0)
    with pytest.raises(ValueError, match="unit_standard_missing"):
        unit_definition(normalized | {"standard_units": {}}, "total_share", contract)


@pytest.mark.parametrize(
    "patch",
    [
        {"business_variant_key": "uncertain"},
        {"identity_resolution": "business_group_source_variant"},
    ],
)
def test_source_variants_never_become_local_identity(patch):
    """场景：来源变体尚未消歧；预期：先拒绝变体，不能用来源内series身份绕过门禁。"""
    with pytest.raises(ValueError, match="business_variant_unresolved"):
        adapt_row(
            source("fx_daily", **patch), CONTRACTS["fx_daily"], cross_config("fx_daily"), AS_OF
        )


def test_missing_catalog_is_rejected_by_cross_input_reader(tmp_path):
    """场景：输入库没有catalog；预期：读取器直接拒绝，不推断空接口集合。"""
    path = reference_database(tmp_path, {"fx_daily": [source("fx_daily")]})
    with duckdb.connect(str(path), config=DATABASE_CONFIG) as connection:
        connection.execute("DROP TABLE catalog")
    with pytest.raises(ValueError, match="reference_catalog_missing"):
        load_inputs(path, AS_OF, "run", cross_config("fx_daily"))


def test_empty_master_and_empty_new_source_preserve_unknown_state(tmp_path):
    """场景：合法空主数据及空新来源；预期：配置无实体，读取无批次但保留已配置接口。"""
    reference = write_all_reference(tmp_path, {"stock_basic": [], "fund_share": [], "fx_daily": []})
    result = build_config(reference, AS_OF)
    assert not result.config.foundation.entities
    assert result.audit["sources"]["fx_daily"]["reason_counts"]["market_timezone_uncertified"] == 1
    inputs, counts = load_inputs(reference, AS_OF, "run", result.config)
    assert inputs.rows("fx_daily") == () and counts["scanned_rows"] == 0


def test_cross_input_stream_releases_history_bytes_and_rejects_variants(tmp_path):
    """场景：跨三批新旧历史与变体混排；预期：释放被替换行字节，变体计排除而非保留。"""
    records = [
        source("fx_daily", str(index), trade_date=day)
        for index, day in enumerate(("20260901", "20260903", "20260902", "20260901", "20260904"))
    ]
    records.append(source("fx_daily", "variant", business_variant_key="uncertain"))
    path = reference_database(tmp_path, {"fx_daily": records})
    config = cross_config("fx_daily", maximum_rows_per_entity_source=2)
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    result, counts = load_inputs(path, AS_OF, "run", config)
    assert counts == {
        "scanned_rows": 6,
        "retained_rows": 2,
        "excluded_rows": 1,
        "batches": 3,
        "discarded_rows": 3,
    }
    assert {item.availability.data_as_of for item in result.rows("fx_daily")} == {
        "2026-09-03",
        "2026-09-04",
    }


@pytest.mark.parametrize("budget", ["source", "global"])
def test_rejected_entity_keys_are_themselves_bounded(tmp_path, budget):
    """场景：大量小实体各自超限；预期：拒绝键集合也受预算限制，最终整源失败关闭。"""
    records = [
        source("fx_daily", f"{code}-{index}", ts_code=code)
        for code in ("A", "B", "C")
        for index in range(2)
    ]
    path = reference_database(tmp_path, {"fx_daily": records})
    config = cross_config(
        "fx_daily",
        maximum_rows_per_entity_source=1,
        maximum_source_rows=2 if budget == "source" else 10,
        maximum_total_rows=1 if budget == "global" else 10,
    )
    result, counts = load_inputs(path, AS_OF, "run", config)
    assert counts["scanned_rows"] == counts["discarded_rows"] == 6
    assert "source_capacity_exceeded" in result.source_reasons["fx_daily"]


def test_combined_entity_capacity_includes_legacy_index_entities(tmp_path):
    """场景：旧股票和指数与新增基金分别未超限但合计超限；预期：拒绝合并配置。"""
    reference = write_all_reference(
        tmp_path,
        {
            "index_basic": [{"ts_code": "000001.SH", "list_date": "20000101"}],
            "fund_basic": [{"ts_code": "F", "list_date": "20000101"}],
        },
    )
    with pytest.raises(ValueError, match="binding_entity_capacity_exceeded"):
        build_config(reference, AS_OF, limits=BuildLimits(maximum_entities=3))


def test_current_master_catalog_excludes_future_and_expired_lifecycles(tmp_path):
    """场景：期权目录同时含当前、未来和已到期合约；预期：配置及运行只保留截止时点有效合约。"""
    reference = write_all_reference(
        tmp_path,
        {
            "opt_basic": [
                {
                    "ts_code": "CURRENT",
                    "list_date": "20260101",
                    "delist_date": "20261231",
                },
                {
                    "ts_code": "EXPIRED",
                    "list_date": "20200101",
                    "delist_date": "20260101",
                },
                {
                    "ts_code": "FUTURE",
                    "list_date": "20270101",
                    "delist_date": "20271231",
                },
            ]
        },
    )
    result = build_config(reference, AS_OF)
    aliases = [
        alias.alias for alias in result.config.foundation.aliases if alias.source == "opt_basic"
    ]
    assert aliases == ["CURRENT"]
    reasons = result.audit["sources"]["opt_basic"]["reason_counts"]
    assert reasons["no_longer_effective"] == reasons["not_yet_effective"] == 1
    inputs, counts = load_inputs(reference, AS_OF, "run", result.config)
    assert [row.text("ts_code") for row in inputs.rows("opt_basic")] == ["CURRENT"]
    assert counts["excluded_rows"] >= 2


def test_equity_projection_filters_foreign_peers_and_relation_endpoints():
    """场景：外币实体同行与关系混入全配置；预期：旧入口只保留两端均在旧范围的关系。"""
    from test_research_governance import A, B, inputs, relation

    from tushare_entitlement_hub.product_foundation.entities import (
        Entity,
        EntityIdentity,
        EntityKind,
    )

    foreign = Entity(
        identity=EntityIdentity(
            namespace="tushare_us", kind=EntityKind.SECURITY, local_id="A", lifecycle_id="life"
        ),
        name="A",
    )
    config = inputs(entities=(A, B)).config
    foreign_id = foreign.identity.entity_id
    config = replace(
        config,
        foundation=replace(config.foundation, entities=(*config.foundation.entities, foreign)),
        peer_paths={A: ("stock",), foreign_id: ("foreign",)},
        confirmed_relations=(
            relation(from_entity_id=A, to_entity_id=B),
            relation(from_entity_id=foreign_id, to_entity_id=A),
            relation(from_entity_id=A, to_entity_id=foreign_id),
        ),
    )
    result = equity_config(config)
    assert set(result.peer_paths) == {A} and len(result.confirmed_relations) == 1


def test_alias_preserving_missing_business_time_still_fails_for_security():
    """场景：有海外证券别名但当前来源没有业务时间；预期：沿用时间门禁，不按观察日补齐。"""
    from tushare_entitlement_hub.equity_research.adapter import adapt_row as shared

    config = configured("us_daily")
    with pytest.raises(ValueError, match="entity_business_time_unknown"):
        shared(source("us_daily", trade_date=None), CONTRACTS["us_daily"], config, AS_OF)


def test_empty_bindings_never_issue_accepted_row_count_queries(tmp_path, monkeypatch):
    """场景：缺绑定但来源表真实存在；预期：只查目录和字段，连accepted行数扫描也不执行。"""
    from unittest.mock import MagicMock

    path = reference_database(tmp_path, {"fx_daily": [source("fx_daily")]})
    connection = duckdb.connect(str(path), read_only=True, config=DATABASE_CONFIG)
    proxy = MagicMock(wraps=connection)
    proxy.__enter__.return_value = proxy
    proxy.__exit__.side_effect = lambda *arguments: connection.close()
    monkeypatch.setattr(adapter.duckdb, "connect", Mock(return_value=proxy))
    result, counts = load_inputs(path, AS_OF, "run", ResearchConfig())
    queries = [call.args[0] for call in proxy.execute.call_args_list]
    assert not any("count(*) FROM accepted" in query for query in queries)
    assert "research_bindings_missing" in result.source_reasons["fx_daily"]
    assert counts["scanned_rows"] == 0


def test_duplicate_config_digest_is_also_verified(tmp_path):
    """场景：配置摘要兼容字段被单独篡改并重签审计；预期：仍拒绝不一致配置。"""
    result = build_config(write_all_reference(tmp_path), AS_OF)
    audit = result.config.binding_audit
    audit["config_sha256"] = "different"
    audit.pop("audit_sha256")
    audit["audit_sha256"] = digest(audit)
    with pytest.raises(ValueError, match="binding_configuration_mismatch"):
        verify_binding_audit(
            replace(result.config, binding_audit_json=encode(audit)),
            result.audit["reference_sha256"],
            AS_OF,
        )


def test_theme_directory_links_members_and_prices_without_historical_identity(tmp_path):
    """场景：DC和TDX目录明确提供同代码主题；预期：各自目录、成员、行情汇合，命名空间隔离。"""
    datasets = {}
    for provider in ("dc", "tdx"):
        datasets[provider + "_index"] = [
            {"ts_code": "THEME", "trade_date": "20261002", "name": "合成主题"}
        ]
        datasets[provider + "_daily"] = [
            {"ts_code": "THEME", "trade_date": "20261002", "close": 10, "vol": 1, "amount": 1}
        ]
        datasets[provider + "_member"] = [
            {"ts_code": "THEME", "trade_date": "20261002", "con_code": "000001.SZ"}
        ]
    reference = write_all_reference(tmp_path, datasets)
    config = build_config(reference, AS_OF).config
    result, _ = load_inputs(reference, AS_OF, "run", config)
    identities = {}
    for provider in ("dc", "tdx"):
        records = [result.rows(provider + suffix)[0] for suffix in ("_index", "_daily", "_member")]
        assert len({record.entity_id for record in records}) == 1
        assert {record.raw["entity_basis"] for record in records} == {"observed_directory"}
        assert not any(record.availability.historical_pit_certified for record in records)
        identities[provider] = records[0].entity_id
    assert identities["dc"] != identities["tdx"]
    assert all(
        alias.version == "research-directory/v1" and alias.effective_from == AS_OF
        for alias in config.foundation.aliases
        if alias.source in {"dc_index", "tdx_index"}
    )


def test_latest_directory_snapshot_is_order_independent_and_duplicates_fail_closed(tmp_path):
    """场景：目录包含旧日期和当前冲突重复；预期：旧日期不覆盖最新，同期冲突不随意选身份。"""
    reference = write_all_reference(
        tmp_path,
        {
            "dc_index": [
                {"ts_code": "GOOD", "trade_date": "20261001", "name": "旧主题"},
                {"ts_code": "GOOD", "trade_date": "20261002", "name": "新主题"},
                {"ts_code": "GOOD", "trade_date": "20260930", "name": "更旧主题"},
                {"ts_code": "BAD", "trade_date": "20261002", "name": "主题甲"},
                {"ts_code": "BAD", "trade_date": "20261002", "name": "主题乙"},
            ],
            "dc_daily": [
                {"ts_code": code, "trade_date": "20261002", "close": 10} for code in ("GOOD", "BAD")
            ],
        },
    )
    config = build_config(reference, AS_OF).config
    result, _ = load_inputs(reference, AS_OF, "run", config)
    records = {record.text("ts_code"): record for record in result.rows("dc_daily")}
    assert records["GOOD"].raw["entity_basis"] == "observed_directory"
    assert records["BAD"].raw["entity_basis"] == "source_local"


def test_ths_members_without_business_dates_use_visible_current_directory(tmp_path):
    """场景：同花顺成员快照无in_date；预期：只在当前目录有明确代码证据时绑定父主题。"""
    reference = write_all_reference(
        tmp_path,
        {
            "ths_index": [{"ts_code": "TH1", "list_date": "20000101", "name": "合成主题"}],
            "ths_member": [{"ts_code": "TH1", "con_code": "000001.SZ", "con_name": "合成股票"}],
        },
    )
    config = build_config(reference, AS_OF).config
    result, _ = load_inputs(reference, AS_OF, "run", config)
    member = result.rows("ths_member")[0]
    assert member.entity_id == result.rows("ths_index")[0].entity_id
    assert member.raw["entity_basis"] == "observed_directory"
    assert member.availability.effective_from is None


def test_declared_index_key_and_foreign_exchange_directory_are_observation_only(tmp_path):
    """场景：行业目录用index_code、外汇目录无上市日；预期：从真实目录键建当前series别名。"""
    reference = write_all_reference(
        tmp_path,
        {
            "index_classify": [{"index_code": "SW1", "src": "SW2021", "industry_name": "合成行业"}],
            "fx_obasic": [{"ts_code": "EURUSD", "name": "合成汇率"}],
            "fx_daily": [{"ts_code": "EURUSD", "trade_date": "20261002", "bid_close": 1.1}],
            "sw_daily": [{"ts_code": "SW1", "trade_date": "20261002", "close": 10}],
        },
    )
    config = build_config(reference, AS_OF).config
    result, _ = load_inputs(reference, AS_OF, "run", config)
    assert result.rows("sw_daily")[0].entity_id == result.rows("index_classify")[0].entity_id
    assert result.rows("fx_daily")[0].entity_id == result.rows("fx_obasic")[0].entity_id
    assert all(
        alias.evidence_refs[0].fields == ("index_code",)
        for alias in config.foundation.aliases
        if alias.source == "index_classify"
    )


def test_ci_member_directory_links_current_industry_and_member_research(tmp_path):
    """场景：两个成员明确给出同一CI三级代码与名称；预期：行情、目录及成员结果在一个实体加工。"""
    from tushare_entitlement_hub.cross_asset_research.themes import build_theme_opportunities

    reference = write_all_reference(
        tmp_path,
        {
            "ci_index_member": [
                {
                    "l1_code": "L1",
                    "l2_code": "L2",
                    "l3_code": "CI1",
                    "l3_name": "合成行业",
                    "ts_code": code,
                    "in_date": "20200101",
                }
                for code in ("000001.SZ", "000002.SZ")
            ],
            "ci_daily": [{"ts_code": "CI1", "trade_date": "20261002", "close": 10}],
        },
    )
    config = build_config(reference, AS_OF).config
    inputs, _ = load_inputs(reference, AS_OF, "run", config)
    members = inputs.rows("ci_index_member")
    assert len(members) == 2
    identity = inputs.rows("ci_daily")[0].entity_id
    assert {member.entity_id for member in members} == {identity}
    aliases = [alias for alias in config.foundation.aliases if alias.source == "ci_index_member"]
    assert len(aliases) == 1 and aliases[0].effective_from == AS_OF
    assert aliases[0].evidence_refs[0].fields == ("l3_code", "l3_name")
    results = build_theme_opportunities(inputs)
    result = next(result for result in results if result.envelope.entity_id == identity)
    assert result.metric("ci_index_member_observed_members") == 2
    assert {"ci_daily", "ci_index_member"} <= set(result.details["sources"])
    assert result.details["historical_pit_certified"] is False


@pytest.mark.parametrize("name", [None, "冲突行业"])
def test_ci_missing_or_conflicting_directory_names_do_not_merge_prices(tmp_path, name):
    """场景：同CI代码的成员名称缺失或冲突；预期：拒绝整个目录代码，行情保持来源内身份。"""
    reference = write_all_reference(
        tmp_path,
        {
            "ci_index_member": [
                {
                    "l3_code": "CI1",
                    "l3_name": "合成行业",
                    "ts_code": "000001.SZ",
                    "in_date": "20200101",
                },
                {"l3_code": "CI1", "l3_name": name, "ts_code": "000002.SZ", "in_date": "20200101"},
            ],
            "ci_daily": [{"ts_code": "CI1", "trade_date": "20261002", "close": 10}],
        },
    )
    result = build_config(reference, AS_OF)
    assert not any(alias.source == "ci_index_member" for alias in result.config.foundation.aliases)
    reason = "directory_identity_incomplete" if name is None else "directory_identity_conflict"
    assert result.audit["sources"]["ci_index_member"]["reason_counts"][reason] == 1
    inputs, _ = load_inputs(reference, AS_OF, "run", result.config)
    assert inputs.rows("ci_daily")[0].raw["entity_basis"] == "source_local"


def test_ci_directory_representative_is_order_independent_and_keys_are_bounded():
    """场景：相同代码名称的成员倒序到达、随后新代码超限；预期：代表证据一致，容量失败关闭。"""
    from tushare_entitlement_hub.cross_asset_research.binding_rules import (
        DIRECTORY_IDENTITY_FIELDS,
        MASTER_KINDS,
        MASTER_NAMESPACES,
        OBSERVED_MASTERS,
    )

    records = [
        row("ci_index_member", record, raw={"l3_code": "CI1", "l3_name": "合成行业"})
        for record in ("a", "z")
    ]
    builders = []
    for order in (records, list(reversed(records))):
        build = EntityBuild(
            1,
            extra_namespaces=MASTER_NAMESPACES,
            extra_kinds=MASTER_KINDS,
            extra_key_fields={"ci_index_member": "l3_code"},
            observation_only=OBSERVED_MASTERS,
            directory_identity_fields=DIRECTORY_IDENTITY_FIELDS,
        )
        for item in order:
            build.add(item.raw, item.availability)
        build.materialize()
        builders.append(build)
    assert builders[0].aliases == builders[1].aliases
    assert builders[0].aliases[0].evidence_refs[0].source_record_id == "z"
    extra = row("ci_index_member", "extra", raw={"l3_code": "CI2", "l3_name": "另一行业"})
    with pytest.raises(ValueError, match="binding_entity_capacity_exceeded"):
        builders[0].add(extra.raw, extra.availability)


def test_signed_row_currency_policy_keeps_foreign_facts_separate(tmp_path):
    """场景：同公司同期间主营金额分别明确USD/HKD；预期：逐行发本币证书，禁止跨币相加。"""
    from tushare_entitlement_hub.product_foundation.numeric import arithmetic

    reference = write_all_reference(
        tmp_path,
        {
            "fina_mainbz": [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20260930",
                    "bz_item": "合成业务",
                    "curr_type": currency,
                    "bz_sales": amount,
                }
                for currency, amount in (("USD", 10), ("HKD", 20))
            ]
        },
    )
    result = build_config(reference, AS_OF)
    assert "bz_sales" in result.audit["row_currency_fields"]["fina_mainbz"]
    assert result.audit["sources"]["fina_mainbz"]["unit_gaps"]["bz_sales"] == {
        "currency_varies_by_row": 1
    }
    assert not any(
        certificate.source_api == "fina_mainbz" and certificate.field == "bz_sales"
        for certificate in result.config.foundation.certifications
    )
    inputs, _ = load_inputs(reference, AS_OF, "run", result.config)
    facts = sorted(
        (row.numeric["bz_sales"] for row in inputs.rows("fina_mainbz")),
        key=lambda fact: fact.currency,
    )
    assert [(fact.currency, fact.value) for fact in facts] == [("HKD", 20), ("USD", 10)]
    assert all(fact.certification.evidence_refs[1].fields == ("curr_type",) for fact in facts)
    with pytest.raises(ValueError, match="currency_mismatch"):
        arithmetic(facts[0], facts[1], "add")
    manual = replace(result.config, binding_audit_json="{}")
    manual_inputs, _ = load_inputs(reference, AS_OF, "run", manual)
    assert all("bz_sales" not in record.numeric for record in manual_inputs.rows("fina_mainbz"))
    with pytest.raises(ValueError, match="binding_cutoff_mismatch"):
        load_inputs(reference, "2026-10-04T12:00:00Z", "run", result.config)


def test_row_currency_fallback_requires_currency_and_keeps_unknown_units_unknown(tmp_path):
    """场景：混币快报含无币种行、未知单位字段；预期：仅有明确本币及单位的行获得事实。"""
    reference = write_all_reference(
        tmp_path,
        {
            "express": [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": date,
                    "end_date": "20260930",
                    "currency": currency,
                    "n_income": 10,
                    "diluted_eps": 1,
                }
                for date, currency in (("20261001", "USD"), ("20261002", "HKD"), ("20260930", None))
            ]
        },
    )
    result = build_config(reference, AS_OF)
    inputs, _ = load_inputs(reference, AS_OF, "run", result.config)
    records = {record.text("currency"): record for record in inputs.rows("express")}
    assert records["USD"].numeric["n_income"].currency == "USD"
    assert records["HKD"].numeric["n_income"].currency == "HKD"
    assert "n_income" not in records[None].numeric
    assert all("diluted_eps" not in record.numeric for record in records.values())
    changed = result.config.binding_audit
    changed["row_unit_version"] = "different"
    changed.pop("audit_sha256")
    changed["audit_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="binding_configuration_mismatch"):
        load_inputs(
            reference, AS_OF, "run", replace(result.config, binding_audit_json=encode(changed))
        )


@pytest.mark.parametrize("api", ["fut_daily", "opt_daily"])
def test_derivative_lots_remain_original_contract_quantity(tmp_path, api):
    """场景：期货/期权成交和持仓量的源单位、标准单位均为手；预期：保持手数，可同合约求比。"""
    from tushare_entitlement_hub.product_foundation.numeric import arithmetic

    reference = write_all_reference(
        tmp_path, {api: [{"ts_code": "F1", "trade_date": "20261002", "vol": 2, "oi": 5}]}
    )
    result = build_config(reference, AS_OF)
    inputs, _ = load_inputs(reference, AS_OF, "run", result.config)
    record = inputs.rows(api)[0]
    assert record.number("vol", "手") == 2 and record.number("oi", "手") == 5
    assert record.numeric["vol"].dimension == "quantity"
    assert arithmetic(record.numeric["oi"], record.numeric["vol"], "divide").value == 2.5
    assert "close" not in record.numeric
    with pytest.raises(ValueError, match="unit_unknown"):
        unit_definition(record.raw, "vol", CONTRACTS[api])


@pytest.mark.parametrize(
    "changes", [{"source_units": {}}, {"standard_units": {}}, {"standard_units": {"vol": "股"}}]
)
def test_derivative_lots_need_proof_that_original_units_were_preserved(changes):
    """场景：期货手数缺源/标准单位或已经变成股；预期：新可选认证也拒绝推断转换。"""
    record = source("fut_daily", source_units={"vol": "手"}, standard_units={"vol": "手"}) | changes
    with pytest.raises(ValueError, match="contract_lot_unit_unverified"):
        unit_definition(record, "vol", CONTRACTS["fut_daily"], preserve_contract_lots=True)


def test_retained_row_currency_helper_never_certifies_quantity_as_money():
    """场景：授权字段包含数量、未知单位与冲突币种；预期：只补money/price，数量不走币种通道。"""
    from tushare_entitlement_hub.cross_asset_research.adapter import prepare_row_numeric

    record = row(
        "fut_daily",
        raw=source(
            "fut_daily", currency="USD", source_units={"vol": "手"}, standard_units={"vol": "手"}
        ),
    )
    assert not prepare_row_numeric(
        record, ResearchConfig(), CONTRACTS["fut_daily"], (), frozenset({"vol"})
    ).numeric
    unknown = row("fx_daily", raw=source("fx_daily", currency="USD", bid_open=1))
    assert not prepare_row_numeric(
        unknown, ResearchConfig(), CONTRACTS["fx_daily"], (), frozenset({"bid_open"})
    ).numeric


def test_shared_adapter_preserves_partial_fields_and_reports_unresolved_stock(tmp_path):
    """场景：旧daily缺可选消费字段且股票代码无法解析；预期：继续审计并同时保留字段和实体原因。"""
    from tushare_entitlement_hub.equity_research.adapter import load_inputs as old_load

    path = reference_database(tmp_path, {"daily": [source(ts_code="unmapped")]})
    with duckdb.connect(str(path), config=DATABASE_CONFIG) as connection:
        connection.execute("ALTER TABLE accepted.daily DROP COLUMN close")
    inputs, counts = old_load(path, AS_OF, "run", configured("daily"))
    assert counts["excluded_rows"] == 1
    assert {"consumption_fields_missing", "stock_entity_unresolved", "entity_unresolved"} <= set(
        inputs.source_reasons["daily"]
    )


def test_shared_retention_keeps_explicit_profiles_and_invalid_period_unknown():
    """场景：旧评级、周线与因子有不同分组维度；预期：保留各自口径，非法业务期不伪造日期。"""
    from tushare_entitlement_hub.equity_research.adapter import retain_current, retention_period

    for api, fields in (
        ("cb_rating", {"rating_com_name": "机构", "ann_date": "20261001"}),
        ("stk_weekly_monthly", {"freq": "week", "end_date": "20261002"}),
        ("factor_value", {"factor_name": "factor", "asset_type": "stock"}),
    ):
        groups = {}
        assert not retain_current(groups, row(api, raw=fields), ResearchConfig())
        assert len(groups) == 1
    assert retention_period(row("stk_weekly_monthly", raw={"end_date": "invalid"})) == ""
