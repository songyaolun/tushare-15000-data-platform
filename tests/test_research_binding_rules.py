"""99个消费来源的显式时间与可选公开时间规则。"""

from dataclasses import replace

import pytest
from test_research_binding_fixture import AS_OF

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.equity_research import binding_rules
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.product_foundation.bindings import TimeMapping
from tushare_entitlement_hub.product_foundation.reference_scan import map_availability
from tushare_entitlement_hub.reference_contract import load_reference_contracts


def test_time_registry_covers_all_consumers_and_fingerprints_contracts():
    """完整正式契约生成99条唯一、排序的规则；daily合同变更必须改变规则指纹。

    输入当前contracts及修改版daily fingerprint，输出两个不同的64位SHA256。
    """
    contracts = load_reference_contracts(load_policies())
    mappings = binding_rules.time_registry(contracts)
    names = [item.source_api for item in mappings]
    assert len(names) == len(set(names)) == 99
    assert names == sorted(names)
    first = binding_rules.rules_fingerprint(contracts)
    assert len(first) == 64
    contracts["daily"] = replace(
        contracts["daily"], rules={**contracts["daily"].rules, "test_rule": True}
    )
    assert binding_rules.rules_fingerprint(contracts) != first


@pytest.mark.parametrize("failure", ["missing_field", "missing_api", "extra_api"])
def test_time_registry_rejects_contract_or_consumer_mismatch(monkeypatch, failure):
    """规则引用未声明字段、缺少API或重复API时拒绝，不输出不完整注册。

    例如删除daily.trade_date产生binding_time_contract_mismatch；删除或复制规则
    产生binding_time_registry_mismatch。
    """
    contracts = load_reference_contracts(load_policies())
    if failure == "missing_field":
        documentation = {**contracts["daily"].documentation, "output_fields": []}
        contracts["daily"] = replace(contracts["daily"], documentation=documentation)
        expected = "binding_time_contract_mismatch"
    else:
        lines = binding_rules.TIME_RULES.strip().splitlines()
        monkeypatch.setattr(
            binding_rules,
            "TIME_RULES",
            "\n".join(lines[:-1] if failure == "missing_api" else lines + [lines[0]]),
        )
        expected = "binding_time_registry_mismatch"
    with pytest.raises(ValueError, match=expected):
        binding_rules.time_registry(contracts)


@pytest.mark.parametrize(
    ("announcement", "revision", "expected"),
    [
        ("2026-08-01", None, "2026-08-01T16:00:00Z"),
        ("2026-08-01", "2026-09-01", "2026-09-01T16:00:00Z"),
        (None, "2026-09-01", "2026-09-01T16:00:00Z"),
        (None, None, None),
    ],
)
def test_optional_public_dates_take_latest_nonempty_value(announcement, revision, expected):
    """非空公告与修订日期取最晚公开边界；可选字段全空则只保留观察时点。

    例如ann_date=08-01、f_ann_date=09-01输出09-01上海日终，不使用报告期替代。
    """
    contracts = load_reference_contracts(load_policies())
    mapping = next(
        item for item in binding_rules.time_registry(contracts) if item.source_api == "income"
    )
    row = {
        "source_api": "income",
        "source_record_id": "synthetic-income",
        "observed_at": AS_OF,
        "historical_pit_certified": False,
        "end_date": "2026-06-30",
        "ann_date": announcement,
        "f_ann_date": revision,
    }
    result = map_availability(row, contracts["income"], mapping)
    assert result.public_available_at == expected
    assert result.report_period == "2026-06-30"
    assert result.observed_at == AS_OF


def test_required_public_date_still_fails_closed():
    """可选时间扩展不放宽既有必填规则；输入空ann_date时抛public_time_missing。"""
    contracts = load_reference_contracts(load_policies())
    mapping = TimeMapping(
        source_api="income",
        public_fields=("ann_date",),
        optional_public_fields=("f_ann_date",),
        market_timezone="Asia/Shanghai",
    )
    with pytest.raises(ValueError, match="public_time_missing"):
        map_availability({}, contracts["income"], mapping)


@pytest.mark.parametrize("encoded", ["[]", "null", "3", '"text"'])
def test_research_config_requires_object_audit(encoded):
    """审计必须是JSON对象；例如binding_audit_json='[]'拒绝invalid_binding_audit。"""
    with pytest.raises(ValueError, match="invalid_binding_audit"):
        ResearchConfig(binding_audit_json=encoded)


def test_scope_certificates_require_json_list_and_match_exact_dates():
    """场景：证书 JSON 非列表、通过日期及旧接口级范围；预期：拒绝坏结构并精确匹配。"""
    with pytest.raises(ValueError, match="invalid_binding_audit"):
        ResearchConfig(scope_certificates_json="{}")
    config = ResearchConfig(
        scope_certificates_json=(
            '[{"api_name":"daily","parameter_dates":{"trade_date":["2026-10-09"]},'
            '"status":"certified"}]'
        ),
        complete_scopes=("trade_cal",),
    )
    assert config.scope_complete("daily")
    assert config.scope_complete("daily", data_as_of="2026-10-09")
    assert not config.scope_complete("daily", data_as_of="2026-10-08")
    assert config.scope_complete("trade_cal")
    assert not config.scope_complete("trade_cal", data_as_of="2026-10-09")


def test_index_listing_date_is_public_time_evidence():
    """指数上市日兼作公开日期证据，防止尚未上市指数提前使用。

    输入index_basic.list_date=2026-10-04，输出上海当日日终公开时间而非当前观察时间。
    """
    contracts = load_reference_contracts(load_policies())
    mapping = next(
        item for item in binding_rules.time_registry(contracts) if item.source_api == "index_basic"
    )
    assert "list_date" in mapping.optional_public_fields
    fact = map_availability(
        {
            "source_api": "index_basic",
            "source_record_id": "synthetic",
            "list_date": "2026-10-04",
            "observed_at": AS_OF,
            "historical_pit_certified": False,
        },
        contracts["index_basic"],
        mapping,
    )
    assert fact.public_available_at == "2026-10-04T16:00:00Z"


@pytest.mark.parametrize(
    ("api", "public_field", "date_fields"),
    [
        ("stk_seasoned", "board_approval_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "sh_approval_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "sasac_approval_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "csrc_approval_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "plan_chg_ann_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "prospectus_pub_dt", {"ann_date": "2026-10-01"}),
        ("stk_seasoned", "offer_intent_pub_dt", {"ann_date": "2026-10-01"}),
        ("disclosure_date", "actual_date", {"ann_date": "2026-10-01", "end_date": "2026-06-30"}),
    ],
)
def test_secondary_publication_dates_cannot_bypass_cutoff(api, public_field, date_fields):
    """增发各阶段公告与实际披露日期必须参与公开时间最大值，不因字段名特殊而遗漏。

    例如ann_date=10月1日但plan_chg_ann_dt=10月4日，10月3日分析时抛未来公开时间。
    """
    from tushare_entitlement_hub.equity_research.binding_builder import binding_availability

    contracts = load_reference_contracts(load_policies())
    mapping = next(
        item for item in binding_rules.time_registry(contracts) if item.source_api == api
    )
    assert public_field in mapping.optional_public_fields
    assert public_field in binding_rules.PUBLIC_FIELDS
    row = {
        "source_api": api,
        "contract_hash": contracts[api].fingerprint,
        "source_record_id": "synthetic",
        "quality_tier": "verified",
        "historical_pit_certified": False,
        "observed_at": AS_OF,
        **date_fields,
        public_field: "2026-10-04",
    }
    with pytest.raises(ValueError, match="binding_future_public_time"):
        binding_availability(row, contracts[api], mapping, AS_OF)


@pytest.mark.parametrize("api", ["stk_shock", "stk_high_shock"])
def test_shock_trade_date_is_day_precision_publication_time(api):
    """异动公告的trade_date按公告日期处理，上海当日结束前不可见；普通行情不套此规则。

    输入trade_date=10月3日，在UTC12时仍未来公开，在UTC16时达到可见边界。
    """
    from tushare_entitlement_hub.equity_research.binding_builder import binding_availability

    contracts = load_reference_contracts(load_policies())
    mapping = next(
        item for item in binding_rules.time_registry(contracts) if item.source_api == api
    )
    assert mapping.optional_public_fields == ("trade_date",)
    assert "trade_date" not in binding_rules.PUBLIC_FIELDS
    row = {
        "source_api": api,
        "contract_hash": contracts[api].fingerprint,
        "source_record_id": "synthetic",
        "quality_tier": "verified",
        "historical_pit_certified": False,
        "observed_at": AS_OF,
        "trade_date": "2026-10-03",
    }
    with pytest.raises(ValueError, match="binding_future_public_time"):
        binding_availability(row, contracts[api], mapping, AS_OF)
    fact = binding_availability(row, contracts[api], mapping, "2026-10-03T16:00:00Z")
    assert fact.public_available_at == "2026-10-03T16:00:00Z"


@pytest.mark.parametrize("timestamp", ["2026-10-03T20:00:00Z", "2026-10-03 20:00:00+00:00"])
def test_business_timestamp_strings_use_market_local_date(timestamp):
    """带T或空格的业务时间先解析真实时点，再取市场本地日，不截取UTC日期。

    输入UTC10月3日20时的字符串，上海映射输出data_as_of=2026-10-04。
    """
    contracts = load_reference_contracts(load_policies())
    mapping = TimeMapping(
        source_api="daily", data_field="trade_date", market_timezone="Asia/Shanghai"
    )
    fact = map_availability(
        {
            "source_api": "daily",
            "source_record_id": "synthetic",
            "observed_at": AS_OF,
            "historical_pit_certified": False,
            "trade_date": timestamp,
        },
        contracts["daily"],
        mapping,
    )
    assert fact.data_as_of == "2026-10-04"
    assert fact.observed_at == AS_OF
