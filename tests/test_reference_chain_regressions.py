"""真实参考链的标准数值恢复、来源内快照与实体键投影回归。"""

from __future__ import annotations

from dataclasses import replace

import duckdb
import pytest
from test_cross_asset_bindings import (
    AS_OF,
    CONTRACTS,
    DATABASE_CONFIG,
    cross_config,
    write_all_reference,
)
from test_research_adapter import source
from test_research_governance import row

from tushare_entitlement_hub.cross_asset_research.adapter import adapt_row, load_inputs
from tushare_entitlement_hub.cross_asset_research.audit import certified_fields
from tushare_entitlement_hub.cross_asset_research.binding_rules import time_registry
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.pipeline import run
from tushare_entitlement_hub.cross_asset_research.registry import registrations
from tushare_entitlement_hub.cross_asset_research.storage import validate
from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.reference_storage import file_checksum


def persisted(output, api):
    """返回已持久化且自身声明来源的结果；例：fut_trade_param只返回自身约束结果。"""
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        rows = connection.execute("SELECT payload FROM research.result").fetchall()
    results = [ResearchResult.from_json(payload) for (payload,) in rows]
    return [result for result in results if result.details.get("source_api") == api]


@pytest.mark.parametrize("legacy_varchar", [False, True])
def test_known_percent_string_survives_complete_reference_chain_without_double_scaling(
    tmp_path, legacy_varchar
):
    """场景：官方str类型的10%参数走新DOUBLE或旧VARCHAR参考库；预期：持久化0.1并重算一致。"""
    reference = write_all_reference(
        tmp_path,
        {
            "fut_trade_param": [
                {
                    "trade_date": "20261002",
                    "exchange": "SHFE",
                    "fut_code": "CU",
                    "price_limit": "10",
                }
            ]
        },
    )
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        assert connection.execute(
            "SELECT typeof(price_limit),price_limit FROM accepted.fut_trade_param"
        ).fetchone() == ("DOUBLE", 0.1)
        if legacy_varchar:
            connection.execute(
                "ALTER TABLE accepted.fut_trade_param "
                "ALTER price_limit TYPE VARCHAR USING price_limit::VARCHAR"
            )
            assert connection.execute(
                "SELECT price_limit FROM accepted.fut_trade_param"
            ).fetchone() == ("0.1",)
    checksum = file_checksum(reference)
    built = build_config(reference, AS_OF)
    inputs, _ = load_inputs(reference, AS_OF, "reference-regression", built.config)
    fact = inputs.rows("fut_trade_param")[0].numeric["price_limit"]
    assert fact.value == pytest.approx(0.1) and fact.certification.multiplier == 1
    if legacy_varchar:
        assert inputs.rows("fut_trade_param")[0].raw["source_numeric_text"] == {
            "price_limit": "0.1"
        }
    output = tmp_path / "products.duckdb"
    summary = run(reference, output, AS_OF, built.config, ("P09",))
    assert summary["consumption"]["by_api"]["fut_trade_param"]["observed"] is True
    assert persisted(output, "fut_trade_param")[0].metric(
        "price_limit_constraint"
    ) == pytest.approx(0.1)
    assert validate(reference, output)["valid"]
    assert file_checksum(reference) == checksum


@pytest.mark.parametrize(
    "changes",
    [
        {"source_units": {}},
        {"standard_units": {}},
        {"source_units": {"price_limit": "元"}},
        {"standard_units": {"price_limit": "%"}},
        {"price_limit": "NaN"},
        {"price_limit": "inf"},
        {"price_limit": "not-a-number"},
    ],
)
def test_numeric_text_recovery_requires_complete_unit_evidence(changes):
    """场景：单位缺失、冲突或非法字符串；预期：不恢复数值，不把来源不明文本变成已认证值。"""
    from tushare_entitlement_hub.standard import restore_standard_numeric_text

    original = {
        "price_limit": "0.1",
        "source_units": {"price_limit": "%"},
        "standard_units": {"price_limit": "比例"},
    } | changes
    restored = restore_standard_numeric_text(
        original, {"price_limit": "str"}, CONTRACTS["fut_trade_param"].policy
    )
    assert restored["price_limit"] == original["price_limit"]
    assert "source_numeric_text" not in restored


def test_unknown_unit_text_and_source_payload_remain_unchanged():
    """场景：最小价位未知单位、已标准化百分比与原始来源共存；预期：只恢复有证据的数值表示。"""
    from tushare_entitlement_hub.standard import (
        is_standard_numeric_field,
        restore_standard_numeric_text,
    )

    policy = CONTRACTS["fut_trade_param"].policy
    original = {
        "price_limit": "0.1",
        "tick_size": "1",
        "source_fields": {"price_limit": "10"},
        "source_units": {"price_limit": "%"},
        "standard_units": {"price_limit": "比例"},
    }
    restored = restore_standard_numeric_text(
        original, {"price_limit": "str", "tick_size": "str"}, policy
    )
    assert is_standard_numeric_field("price_limit", "str", policy)
    assert not is_standard_numeric_field("tick_size", "str", policy)
    assert restored["price_limit"] == 0.1 and restored["tick_size"] == "1"
    assert original["price_limit"] == "0.1" and restored["source_fields"] == {"price_limit": "10"}


@pytest.mark.parametrize("api", ["rt_hk_k", "stock_company"])
def test_snapshot_metadata_keeps_source_evidence_without_inventing_business_time(tmp_path, api):
    """场景：正式契约明确为无业务时间的快照；预期：仅来源内观测证据进入产品，不认证历史日期。"""
    payload = (
        {"ts_code": "00700.HK", "pre_close": 9, "close": 10, "vol": 100, "amount": 1000}
        if api == "rt_hk_k"
        else {
            "ts_code": "000001.SZ",
            "com_id": "SYNTHETIC_COMPANY_ID",
            "main_business": "合成制造业务",
            "business_scope": "合成设备生产",
        }
    )
    reference = write_all_reference(tmp_path, {api: [payload]})
    built = build_config(reference, AS_OF)
    inputs, _ = load_inputs(reference, AS_OF, "snapshot-run", built.config)
    record = inputs.rows(api)[0]
    assert record.entity_id.startswith("source:series:" + api + ":")
    assert record.raw["entity_basis"] == "source_local"
    assert record.availability.data_as_of is None and record.availability.report_period is None
    assert record.availability.historical_pit_certified is False
    output = tmp_path / "products.duckdb"
    run(reference, output, AS_OF, built.config, ("P10",))
    results = persisted(output, api)
    assert results and any(
        ref.source_api == api for result in results for ref in result.envelope.evidence_refs
    )
    if api == "rt_hk_k":
        assert all(result.metric("observed_market_return") is None for result in results)
    assert validate(reference, output)["valid"]


def test_snapshot_observation_gate_and_contract_drift_stay_fail_closed():
    """场景：未来观测或快照契约变成增量接口；预期：仍拒绝未来记录及未经声明的语义变化。"""
    with pytest.raises(ValueError, match="row_unavailable"):
        adapt_row(
            source("stock_company", observed_at="2026-10-04T12:00:00Z"),
            CONTRACTS["stock_company"],
            cross_config("stock_company"),
            AS_OF,
        )
    changed = dict(CONTRACTS)
    changed["stock_company"] = replace(
        changed["stock_company"],
        policy=replace(changed["stock_company"].policy, collection_mode="incremental"),
    )
    with pytest.raises(ValueError, match="binding_snapshot_contract_mismatch"):
        time_registry(changed)


def test_audit_requires_typed_fact_for_semantically_numeric_strings():
    """场景：规范数值仍为字符串且没有TypedFact；预期：审计不能把它当文本认证，由根代理audit改动保障。"""
    item = next(entry for entry in registrations() if entry.spec.api == "fut_trade_param")
    record = row(
        "fut_trade_param",
        raw=source("fut_trade_param", price_limit="0.1", exchange="SHFE", fut_code="CU"),
    )
    assert "price_limit" not in certified_fields(record, item, AS_OF)


def test_declared_entity_key_survives_reference_projection(tmp_path):
    """场景：fut_code合法但不在矩阵selected_fields或主键中；预期：显式实体映射仍保留它进入产品。"""
    reference = write_all_reference(
        tmp_path,
        {
            "fut_rcpt_mat": [
                {
                    "trade_date": "20261002",
                    "exchange": "SHFE",
                    "fut_name": "铜",
                    "fut_code": "CU",
                    "unit": "吨",
                    "cur_month": 10,
                    "next_month": 20,
                }
            ]
        },
    )
    built = build_config(reference, AS_OF)
    inputs, _ = load_inputs(reference, AS_OF, "receipt-run", built.config)
    assert len(inputs.rows("fut_rcpt_mat")) == 1
    assert inputs.rows("fut_rcpt_mat")[0].raw["fut_code"] == "CU"
    output = tmp_path / "products.duckdb"
    run(reference, output, AS_OF, built.config, ("P09",))
    results = persisted(output, "fut_rcpt_mat")
    assert results and any(
        ref.source_api == "fut_rcpt_mat" for ref in results[0].envelope.evidence_refs
    )
    assert results[0].metric("current_next_maturity_ratio") is None
    assert validate(reference, output)["valid"]


@pytest.mark.parametrize(
    "metadata_field, malformed",
    [("source_units", None), ("standard_units", [])],
)
def test_numeric_text_recovery_rejects_malformed_unit_metadata(metadata_field, malformed):
    """场景：来源单位为空或标准单位错误地为数组；预期：保留原文与输入，不恢复数值。"""
    from copy import deepcopy

    from tushare_entitlement_hub.standard import restore_standard_numeric_text

    original = {
        "price_limit": "0.1",
        "source_units": {"price_limit": "%"},
        "standard_units": {"price_limit": "比例"},
        metadata_field: malformed,
    }
    expected = deepcopy(original)
    restored = restore_standard_numeric_text(
        original, {"price_limit": "str"}, CONTRACTS["fut_trade_param"].policy
    )
    assert restored == expected
    assert original == expected
    assert "source_numeric_text" not in restored


def test_numeric_text_recovery_rejects_unsupported_conversion_target(monkeypatch):
    """场景：注入不受支持的单位转换目标；预期：原数值文本不被认证，也不修改来源数据。"""
    from copy import deepcopy
    from decimal import Decimal

    from tushare_entitlement_hub.standard import UNIT_CONVERSIONS, restore_standard_numeric_text

    monkeypatch.setitem(
        UNIT_CONVERSIONS, "%", ("unsupported-test-standard-unit", Decimal("0.01"))
    )
    original = {
        "price_limit": "0.1",
        "source_units": {"price_limit": "%"},
        "standard_units": {"price_limit": "unsupported-test-standard-unit"},
        "source_fields": {"price_limit": "10"},
    }
    expected = deepcopy(original)
    restored = restore_standard_numeric_text(
        original, {"price_limit": "str"}, CONTRACTS["fut_trade_param"].policy
    )
    assert restored == expected
    assert original == expected
    assert "source_numeric_text" not in restored
