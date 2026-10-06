"""研究适配层的离线契约测试，参考库仅含确定性合成记录。"""

import json
from collections import defaultdict
from dataclasses import replace

import duckdb
import pytest
from test_research_governance import DAY, NOW, A, B, inputs, row

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.equity_research import adapter
from tushare_entitlement_hub.equity_research.adapter import (
    adapt_row,
    load_inputs,
    prepare_numeric,
    quoted_numeric,
    retain_current,
)
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.product_foundation.bindings import EntityMapping, TimeMapping
from tushare_entitlement_hub.product_foundation.entities import EntityAlias
from tushare_entitlement_hub.product_foundation.numeric import UnitCertification
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.reference_contract import load_reference_contracts
from tushare_entitlement_hub.reference_storage import file_checksum

CONTRACTS = load_reference_contracts(load_policies())
REGISTRY = {entry.api: entry for entry in load_registry()}
PUBLIC_FIELDS = ("ann_date", "f_ann_date", "pub_date")


def source(api="daily", record="one", **changes):
    """生成含当前合同指纹的accepted合成行；例：默认daily价10元且量额为正。"""
    fields = set(REGISTRY[api].selected_fields + REGISTRY[api].business_key)
    values = {field: None for field in fields}
    values.update(
        {
            "ts_code": "A",
            "trade_date": DAY,
            "source_api": api,
            "source_record_id": record,
            "business_group_key": record,
            "contract_hash": CONTRACTS[api].fingerprint,
            "observed_at": NOW,
            "quality_tier": "verified",
            "historical_pit_certified": False,
            "standard_units": {"amount": "元", "close": "元", "vol": "股"},
            "close": 10.0,
            "amount": 100.0,
            "vol": 10.0,
        }
    )
    values.update({field: "2026-10-02" for field in PUBLIC_FIELDS if field in fields})
    if api.startswith(("income", "balancesheet", "cashflow")):
        values["end_date"] = "2026-09-30"
    return values | changes


def time_mapping(api):
    """按真实来源区分公告和报告期；例：income按end_date留期间，并映射两种公告日期。"""
    financial = api.startswith(("income", "balancesheet", "cashflow"))
    data_field = (
        "end_date"
        if financial
        else (CONTRACTS[api].policy.date_field or None if api in CONTRACTS else "trade_date")
    )
    fields = REGISTRY[api].selected_fields + REGISTRY[api].business_key if api in REGISTRY else ()
    return TimeMapping(
        source_api=api,
        data_field=data_field,
        public_fields=tuple(field for field in PUBLIC_FIELDS if field in fields),
        report_period_field="end_date" if financial else None,
        market_timezone="Asia/Shanghai",
    )


def configured(*apis, certificates=(), **changes):
    """为两个合成实体登记别名与业务时间；例：daily绑定只识别A/B，不猜其他股票代码。"""
    context = inputs(entities=(A, B)).config
    reference = row("daily").evidence("trade_date")
    aliases = tuple(
        EntityAlias(
            namespace="synthetic",
            alias=identity.split(":")[2],
            entity_id=identity,
            effective_from="2000-01-01T00:00:00Z",
            effective_to=None,
            observed_at=NOW,
            source="synthetic",
            version="v1",
            evidence_refs=reference,
            confidence=100,
        )
        for identity in (A, B)
    )
    foundation = replace(
        context.foundation,
        aliases=aliases,
        certifications=certificates,
        entity_mappings=tuple(
            EntityMapping(source_api=api, namespace="synthetic", key_field="ts_code")
            for api in apis
            if api not in adapter.UNRESOLVED_APIS
        ),
        time_mappings=tuple(time_mapping(api) for api in apis),
    )
    return replace(context, foundation=foundation, **changes)


def certification(api="daily", field="amount", **changes):
    """创建明确人民币金额证书；例：千元到元、倍率1000，测试不重复转换标准值。"""
    values = dict(
        version="synthetic/v1",
        source_api=api,
        field=field,
        source_unit="千元",
        target_unit="元",
        dimension="money",
        currency="CNY",
        multiplier=1000,
        evidence_refs=row("daily").evidence("trade_date"),
    )
    return UnitCertification(**(values | changes))


def reference_database(tmp_path, datasets, *, quarantine=None):
    """用正式catalog/accepted结构构造扫描夹具；例：daily一行配精确质量账本。

    行字段显式覆盖当前登记字段与来源元数据；前置清洗另有测试，此处允许注入坏行验证拒绝。
    """
    reference = tmp_path / "synthetic-reference.duckdb"
    with duckdb.connect(
        str(reference), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        connection.execute("CREATE TABLE catalog(api_name VARCHAR, manifest VARCHAR)")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        for api, rows in datasets.items():
            columns = sorted(set().union(*(item.keys() for item in rows)))
            types = {}
            for name in columns:
                values = [item.get(name) for item in rows if item.get(name) is not None]
                sample = values[0] if values else None
                types[name] = (
                    "BOOLEAN"
                    if isinstance(sample, bool)
                    else "DOUBLE"
                    if isinstance(sample, (int, float))
                    else "VARCHAR"
                )
            schema = ",".join('"' + name + '" ' + types[name] for name in columns)
            connection.execute(f'CREATE TABLE accepted."{api}" ({schema})')
            placeholders = ",".join("?" for _ in columns)
            records = [
                [
                    json.dumps(item.get(name), ensure_ascii=False)
                    if isinstance(item.get(name), (dict, list))
                    else item.get(name)
                    for name in columns
                ]
                for item in rows
            ]
            connection.executemany(f'INSERT INTO accepted."{api}" VALUES ({placeholders})', records)
            quarantined = (quarantine or {}).get(api, 0)
            if quarantined:
                connection.execute(
                    f'CREATE TABLE quarantine."{api}" AS SELECT * FROM accepted."{api}" LIMIT 1'
                )
            manifest = {
                "contract_hash": CONTRACTS[api].fingerprint,
                "observed_at": NOW,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": len(rows) + quarantined,
                    "accepted_rows": len(rows),
                    "quarantined_rows": quarantined,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?,?)", [api, json.dumps(manifest)])
    return reference


def test_empty_bindings_inspect_sources_without_scanning(tmp_path):
    """场景：参考库合法但未配置实体/时间；预期：不扫描原始行，返回绑定缺口。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    before = file_checksum(path)
    result, counts = load_inputs(path, NOW, "run", ResearchConfig())
    assert all(value == 0 for value in counts.values())
    assert "research_bindings_missing" in result.source_reasons["daily"]
    assert "missing_interface" in result.source_reasons["moneyflow"]
    assert before == file_checksum(path)


def test_bound_scan_normalizes_once_and_never_changes_reference(tmp_path):
    """场景：同批标准元和源千元金额；预期：分别为100元和2000元，扫描只读且可复现。"""
    path = reference_database(
        tmp_path,
        {
            "daily": [
                source(),
                source("daily", "b", ts_code="B", amount=2, standard_units={"amount": "千元"}),
            ]
        },
    )
    config = configured("daily", certificates=(certification(),))
    progress = []
    checksum = file_checksum(path)
    result, counts = load_inputs(path, NOW, "run", config, progress=progress.append)
    assert counts["scanned_rows"] == counts["retained_rows"] == 2
    assert counts["excluded_rows"] == 0
    assert [item.number("amount", "元") for item in result.rows("daily")] == [100, 2000]
    assert progress[0]["stage"] == "scan_reference"
    assert file_checksum(path) == checksum
    replay, second_counts = load_inputs(path, NOW, "run", config)
    assert replay == result and second_counts == counts


@pytest.mark.parametrize(
    "scenario,code",
    [("missing", "reference_catalog_missing"), ("duplicate", "reference_catalog_duplicate")],
)
def test_catalog_missing_or_duplicate_is_rejected(tmp_path, scenario, code):
    """场景：目录表缺失或一个API多份清单；预期：整体拒绝，不猜选清单。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    with duckdb.connect(str(path), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        if scenario == "missing":
            connection.execute("DROP TABLE catalog")
        else:
            connection.execute("INSERT INTO catalog SELECT * FROM catalog")
    with pytest.raises(ValueError, match=code):
        load_inputs(path, NOW, "run", configured("daily"))


def test_unregistered_source_is_reported_without_access(tmp_path, monkeypatch):
    """场景：声明来源未登记；预期：明确source_not_registered，不从名称猜合同。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    monkeypatch.setattr(adapter, "SOURCE_APIS", adapter.SOURCE_APIS | {"unknown_api"})
    result, _ = load_inputs(path, NOW, "run", configured("daily"))
    assert result.source_reasons["unknown_api"] == ("source_not_registered",)


@pytest.mark.parametrize(
    "patch,code",
    [
        ({"source_api": "other"}, "source_contract_mismatch"),
        ({"contract_hash": "other"}, "source_contract_mismatch"),
        ({"quality_tier": "quarantine"}, "quarantined_or_unknown_quality"),
        ({"quality_tier": "unknown"}, "quarantined_or_unknown_quality"),
        ({"is_quarantined": True}, "quarantined_or_unknown_quality"),
        ({"observed_at": "2026-10-04T00:00:00Z"}, "row_unavailable"),
        ({"trade_date": None}, "entity_business_time_unknown"),
        ({"ts_code": "unmapped"}, "entity_unresolved"),
    ],
)
def test_row_identity_quality_and_time_are_fail_closed(patch, code):
    """场景：来源/质量/实体/时点不合格；预期：在数值适配之前给固定原因拒绝。"""
    with pytest.raises(ValueError, match=code):
        adapt_row(source(**patch), CONTRACTS["daily"], configured("daily"), NOW)


def test_missing_time_or_entity_binding_does_not_guess():
    """场景：省略时间或实体映射；预期：时间映射必需，普通API必须唯一实体。"""
    with pytest.raises(ValueError, match="time_mapping_missing"):
        adapt_row(source(), CONTRACTS["daily"], ResearchConfig(), NOW)
    config = configured("daily")
    config = replace(config, foundation=replace(config.foundation, entity_mappings=()))
    with pytest.raises(ValueError, match="entity_unresolved"):
        adapt_row(source(), CONTRACTS["daily"], config, NOW)
    config = configured("major_news")
    result = adapt_row(source("major_news"), CONTRACTS["major_news"], config, NOW)
    assert result.entity_id is None


def test_future_announcement_is_excluded_even_when_observed_now(tmp_path):
    """场景：当前采集记录含未来公告；预期：可用时间过滤拒绝，不只检查observed_at。"""
    path = reference_database(tmp_path, {"daily": [source(ann_date="2026-10-04")]})
    config = configured("daily")
    mapping = replace(config.foundation.time_mappings[0], public_fields=("ann_date",))
    config = replace(config, foundation=replace(config.foundation, time_mappings=(mapping,)))
    result, counts = load_inputs(path, NOW, "run", config)
    assert counts["excluded_rows"] == 1
    assert result.rows("daily") == ()
    assert "rows_excluded_by_adapter" in result.source_reasons["daily"]


@pytest.mark.parametrize("kind", ["observed", "public"])
def test_future_unit_certificate_does_not_certify_current_rows(tmp_path, kind):
    """场景：单位认证证据未来才观测或公开；预期：移除证书，原值仍保留但不能计算。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    unit = certification()
    evidence = unit.evidence_refs[0]
    future = "2026-10-04T00:00:00Z"
    evidence = (
        replace(evidence, observed_at=future)
        if kind == "observed"
        else replace(evidence, public_available_at=future)
    )
    unit = replace(unit, evidence_refs=(evidence,))
    result, counts = load_inputs(path, NOW, "run", configured("daily", certificates=(unit,)))
    assert counts["retained_rows"] == 1
    assert result.rows("daily")[0].raw["amount"] == 100
    assert result.rows("daily")[0].number("amount") is None
    assert result.config.foundation.certifications == ()


def test_quarantine_table_and_quarantined_accepted_row_never_enter_features(tmp_path):
    """场景：隔离表有记录且accepted误混隔离状态；预期：隔离表不扫描，错放行被拒绝。"""
    path = reference_database(
        tmp_path,
        {"daily": [source(), source("daily", "bad", quality_tier="quarantine")]},
        quarantine={"daily": 1},
    )
    result, counts = load_inputs(path, NOW, "run", configured("daily"))
    assert counts["scanned_rows"] == 2 and counts["excluded_rows"] == 1
    assert counts["retained_rows"] == 1
    assert "quarantined_rows_excluded" in result.source_reasons["daily"]


def test_prepare_numeric_keeps_absent_certificates_and_foreign_api_fields_unknown():
    """场景：未认证股本、缺字段及其他API证书；预期：不猜单位、不误跨API套证书。"""
    record = row("daily", raw=source())
    config = configured(
        "daily",
        certificates=(
            certification("cb_basic"),
            certification(field="total_share"),
            certification(),
        ),
    )
    result = prepare_numeric(record, config)
    assert set(result.numeric) == {"amount"}
    assert result.number("amount") == 100
    assert result.number("close") is None
    assert result.number("total_share") is None


def test_daily_price_requires_trade_but_cb_price_does_not():
    """场景：零成交日线和有效转股报价；预期：日线价拒绝，CB转股价按独立语义认证。"""
    price = certification(field="close", source_unit="元", dimension="price", multiplier=1)
    record = row("daily", raw=source(vol=0))
    assert (
        prepare_numeric(record, configured("daily", certificates=(price,))).number("close") is None
    )
    quote = certification(
        "cb_basic", "conv_price", source_unit="元", dimension="price", multiplier=1
    )
    record = row(
        "cb_basic",
        entity=None,
        raw=source("cb_basic", conv_price=10, standard_units={"conv_price": "元"}, vol=0, amount=0),
    )
    result = prepare_numeric(record, configured("cb_basic", certificates=(quote,)))
    assert result.number("conv_price", "元") == 10
    assert result.numeric["conv_price"].entity_id == "cn:market:A:all"


@pytest.mark.parametrize(
    "patch,certificate_patch,expected",
    [
        ({"standard_units": {"conv_price": "分"}}, {"source_unit": "分", "multiplier": 0.01}, 0.1),
        ({"standard_units": {}, "source_units": {"conv_price": "元"}}, {}, 10),
        ({"standard_units": {"conv_price": "未知单位"}}, {}, None),
        ({"quality_tier": "usable_with_annotation"}, {}, None),
        ({"is_quarantined": True}, {}, None),
        ({"business_variant_key": "uncertain"}, {}, None),
        ({}, {"dimension": "money"}, None),
        ({}, {"currency": "USD"}, None),
        ({"conv_price": 0}, {}, None),
        ({"conv_price": None}, {}, None),
        ({"conv_price": float("inf")}, {}, None),
        (
            {"conv_price": 1e308, "standard_units": {"conv_price": "千元"}},
            {"source_unit": "千元", "multiplier": 1000},
            None,
        ),
    ],
)
def test_quote_quality_units_currency_and_finite_conversion(patch, certificate_patch, expected):
    """场景：非成交报价换算、来源变体或非法值；预期：仅合法人民币报价可用。"""
    record = row(
        "cb_basic",
        raw=source("cb_basic", conv_price=10, standard_units={"conv_price": "元"}) | patch,
    )
    unit = certification(
        "cb_basic",
        "conv_price",
        **({"source_unit": "元", "dimension": "price", "multiplier": 1} | certificate_patch),
    )
    assert quoted_numeric(record, "conv_price", unit).value == expected


def test_current_retention_releases_old_days_and_keeps_financial_profiles():
    """场景：日线、宏观、披露日程与报表口径历史；预期：按各自当前窗口保留。"""
    groups = defaultdict(list)
    config = configured("daily")
    assert not retain_current(groups, row("daily", "old", day="2026-10-01"), config)
    assert not retain_current(groups, row("daily", "new", day=DAY), config)
    assert [item.raw["source_record_id"] for item in groups[(A, "", "")]] == ["new"]
    macro = defaultdict(list)
    for period in ("2026-07", "2026-09", "2026-08"):
        retain_current(macro, row("cn_cpi", period, day=period), config)
    assert {item.availability.data_as_of for item in macro[(A, "", "")]} == {"2026-08", "2026-09"}
    disclosures = defaultdict(list)
    for period in ("2026-03-31", "2026-09-30", "2026-06-30"):
        retain_current(disclosures, row("disclosure_date", period, day=period), config)
    assert {item.availability.data_as_of for item in disclosures[(A, "", "")]} == {
        "2026-06-30",
        "2026-09-30",
    }
    finance = defaultdict(list)
    for report_type in ("1", "2"):
        retain_current(
            finance,
            row("income", report_type, raw={"report_type": report_type, "comp_type": "1"}),
            config,
        )
    assert len(finance) == 2
    unresolved = defaultdict(list)
    retain_current(unresolved, row("major_news", entity=None), config)
    assert ("unresolved", "", "") in unresolved


def test_batch_boundaries_and_current_history_release(tmp_path):
    """场景：多批同实体旧历史后遇最新日；预期：不因历史总量超限，最终仅保留当前日。"""
    records = [
        source("daily", str(index), trade_date="2026-09-" + str(index + 10)) for index in range(3)
    ] + [source("daily", "current")]
    path = reference_database(tmp_path, {"daily": records})
    config = configured("daily", maximum_rows_per_entity_source=1)
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    result, counts = load_inputs(path, NOW, "run", config)
    assert counts["batches"] == 2 and counts["scanned_rows"] == 4
    assert counts["retained_rows"] == 1
    assert result.rows("daily")[0].raw["source_record_id"] == "current"


def test_entity_capacity_rejects_whole_event_group_without_affecting_other_stock(tmp_path):
    """场景：同实体事件超过容量且后续仍有行；预期：整组拒绝，其他实体正常保留。"""
    records = [source("share_float", str(index), float_date="20261101") for index in range(3)]
    records.append(source("share_float", "other", ts_code="B", float_date="20261101"))
    path = reference_database(tmp_path, {"share_float": records})
    config = configured("share_float", maximum_rows_per_entity_source=1)
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    result, counts = load_inputs(path, NOW, "run", config)
    assert counts["retained_rows"] == 1 and result.rows("share_float", A) == ()
    assert result.rows("share_float", B)[0].raw["source_record_id"] == "other"
    assert result.entity_source_reasons["share_float/" + A] == ("entity_source_capacity_exceeded",)


def test_source_capacity_rejects_all_batches_and_keeps_scanning_accounting(tmp_path):
    """场景：多个实体使来源总容量超限；预期：清空整个来源并继续记录扫描总数。"""
    records = [
        source("share_float", str(index), ts_code="A" if index % 2 == 0 else "B")
        for index in range(4)
    ]
    path = reference_database(tmp_path, {"share_float": records})
    config = configured("share_float", maximum_source_rows=1)
    config = replace(config, thresholds=replace(config.thresholds, fact_batch_size=2))
    result, counts = load_inputs(path, NOW, "run", config)
    assert counts["scanned_rows"] == 4 and counts["retained_rows"] == 0
    assert result.rows("share_float") == ()
    assert "source_capacity_exceeded" in result.source_reasons["share_float"]


def test_total_capacity_does_not_publish_truncated_later_source(tmp_path):
    """场景：先前来源占用总预算；预期：后续来源整体降级，不输出截断精确值。"""
    path = reference_database(
        tmp_path, {"daily": [source()], "share_float": [source("share_float")]}
    )
    result, counts = load_inputs(
        path, NOW, "run", configured("daily", "share_float", maximum_total_rows=1)
    )
    assert counts["retained_rows"] == 1 and len(result.rows("daily")) == 1
    assert result.rows("share_float") == ()
    assert "source_capacity_exceeded" in result.source_reasons["share_float"]


def test_equity_source_subset_scans_only_requested_interfaces(tmp_path):
    """场景：调用方只请求日线来源；预期：其他已配置接口不扫描也不进入来源结果。"""
    path = reference_database(
        tmp_path, {"daily": [source()], "share_float": [source("share_float")]}
    )
    config = configured("daily", "share_float")
    result, counts = load_inputs(path, NOW, "run", config, source_apis=frozenset({"daily"}))
    assert set(result.rows_by_api) == set(result.source_reasons) == {"daily"}
    assert counts["scanned_rows"] == counts["retained_rows"] == 1
    with pytest.raises(ValueError, match="input_source_selection_invalid"):
        load_inputs(path, NOW, "run", config, source_apis=frozenset({"unknown"}))


def test_entity_capacity_spans_financial_report_profiles(tmp_path):
    """场景：同一实体的两种报表口径分别未超组限但合计超限；预期：该实体全部拒绝。"""
    records = [
        source("income", "one", report_type="1", comp_type="1"),
        source("income", "two", report_type="2", comp_type="1"),
        source("income", "other", ts_code="B", report_type="1", comp_type="1"),
    ]
    path = reference_database(tmp_path, {"income": records})
    config = configured("income", maximum_rows_per_entity_source=1)
    result, counts = load_inputs(path, NOW, "run", config)
    assert result.rows("income", A) == ()
    assert len(result.rows("income", B)) == 1
    assert result.entity_source_reasons["income/" + A] == ("entity_source_capacity_exceeded",)
    assert counts["discarded_rows"] == 2
    assert (
        counts["scanned_rows"]
        == counts["retained_rows"] + counts["excluded_rows"] + counts["discarded_rows"]
    )


def test_discarded_history_and_rejected_rows_have_separate_accounting(tmp_path):
    """场景：旧历史被释放且未来行被拒绝；预期：丢弃和拒绝分列，扫描总数可逐项对账。"""
    records = [
        source("daily", "old", trade_date="2026-10-01"),
        source(),
        source("daily", "future", observed_at="2026-10-04T00:00:00Z"),
    ]
    path = reference_database(tmp_path, {"daily": records})
    _, counts = load_inputs(path, NOW, "run", configured("daily"))
    assert counts["scanned_rows"] == 3
    assert counts["retained_rows"] == counts["excluded_rows"] == counts["discarded_rows"] == 1


@pytest.mark.parametrize(
    "api,public_field", [("income", "ann_date"), ("income", "f_ann_date"), ("st", "pub_date")]
)
def test_known_public_date_cannot_be_hidden_by_mapping(api, public_field):
    """场景：真实公告字段非空但映射故意省略；预期：在数值与实体适配前拒绝隐藏公开时间。"""
    record = source(api, **{public_field: "2026-10-04"})
    config = configured(api)
    mapping = replace(
        config.foundation.time_mappings[0],
        public_fields=tuple(
            field
            for field in config.foundation.time_mappings[0].public_fields
            if field != public_field
        ),
    )
    config = replace(config, foundation=replace(config.foundation, time_mappings=(mapping,)))
    with pytest.raises(ValueError, match="public_field_mapping_missing"):
        adapt_row(record, CONTRACTS[api], config, NOW)


def test_mapped_future_financial_publication_is_still_unavailable():
    """场景：实际公告日期未来且已正确映射，普通公告较早；预期：取更晚公开边界并拒绝。"""
    record = source("income", ann_date="2026-10-01", f_ann_date="2026-10-04")
    with pytest.raises(ValueError, match="row_unavailable"):
        adapt_row(record, CONTRACTS["income"], configured("income"), NOW)


def test_non_event_future_daily_business_date_is_rejected(tmp_path):
    """场景：采集时间当前但日线业务日未来；预期：逐行拒绝，不能冒充当日行情。"""
    record = source(trade_date="2026-10-04")
    with pytest.raises(ValueError, match="future_business_date"):
        adapt_row(record, CONTRACTS["daily"], configured("daily"), NOW)
    path = reference_database(tmp_path, {"daily": [record]})
    result, counts = load_inputs(path, NOW, "run", configured("daily"))
    assert result.rows("daily") == () and counts["excluded_rows"] == 1
    assert "rows_excluded_by_adapter" in result.source_reasons["daily"]


def test_announced_future_event_date_remains_available():
    """场景：已公告的解禁计划发生在未来；预期：允许保留事件，公告时间与实施日期分离。"""
    config = configured("share_float")
    mapping = replace(config.foundation.time_mappings[0], data_field="float_date")
    config = replace(config, foundation=replace(config.foundation, time_mappings=(mapping,)))
    record = source("share_float", ann_date="2026-10-01", float_date="2026-11-01")
    result = adapt_row(record, CONTRACTS["share_float"], config, NOW)
    assert result.availability.data_as_of == "2026-11-01"
    assert result.availability.public_available_at == "2026-10-01T16:00:00Z"
    assert result.entity_id == A
