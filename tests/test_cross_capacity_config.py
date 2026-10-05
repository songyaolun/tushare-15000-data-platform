"""构建预算的持久化、指纹与运行传递回归，全部使用独立离线合成库。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_cross_asset_bindings import write_all_reference
from test_research_binding_fixture import AS_OF
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.cross_asset_research.adapter import load_inputs
from tushare_entitlement_hub.cross_asset_research.bindings import (
    build_config,
    equity_config,
    verify_binding_audit,
)
from tushare_entitlement_hub.equity_research.adapter import load_inputs as load_equity_inputs
from tushare_entitlement_hub.equity_research.binding_builder import configuration_hash
from tushare_entitlement_hub.equity_research.binding_io import BuildLimits
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import encode


@pytest.mark.parametrize(
    "source_budget,total_budget",
    [
        pytest.param(2_000_000, 5_000_000, id="documented_defaults"),
        pytest.param(1_000, 9_000, id="smaller_explicit_budgets"),
        pytest.param(90_000, 300_000, id="above_old_runtime_defaults"),
    ],
)
def test_build_config_preserves_budgets_across_serialization(
    tmp_path: Path, source_budget: int, total_budget: int
) -> None:
    """场景：默认或自定义构建预算；预期：运行配置、JSON 与签名审计保留相同值。

    例如单源 2000000、总计 5000000 往返后保持原值，来源文件字节不变。
    """
    reference = write_all_reference(tmp_path)
    checksum = file_checksum(reference)
    result = build_config(
        reference,
        AS_OF,
        limits=BuildLimits(maximum_source_rows=source_budget, maximum_total_rows=total_budget),
    )
    assert result.config.maximum_source_rows == source_budget
    assert result.config.maximum_total_rows == total_budget
    serialized = result.config.to_json()
    restored = ResearchConfig.from_json(serialized)
    assert restored.maximum_source_rows == source_budget
    assert restored.maximum_total_rows == total_budget
    assert restored.to_json() == serialized
    assert result.audit["configuration_sha256"] == configuration_hash(restored)
    assert result.audit["config_sha256"] == configuration_hash(restored)
    assert verify_binding_audit(restored, checksum, AS_OF) is None
    assert file_checksum(reference) == checksum


@pytest.mark.parametrize("field", ["maximum_source_rows", "maximum_total_rows"])
def test_each_persisted_budget_changes_fingerprint_and_rejects_unsigned_edits(
    tmp_path: Path, field: str
) -> None:
    """场景：只变更一个合法预算，或手工改动已签名配置；预期：重建指纹变化，篡改拒绝。

    例如总预算 9000 改为 9001 后应产生新指纹，沿用旧审计则校验失败。
    """
    reference = write_all_reference(tmp_path)
    limits = BuildLimits(maximum_source_rows=1_000, maximum_total_rows=9_000)
    initial = build_config(reference, AS_OF, limits=limits)
    modified = build_config(
        reference, AS_OF, limits=replace(limits, **{field: getattr(limits, field) + 1})
    )
    assert configuration_hash(initial.config) != configuration_hash(modified.config)
    assert initial.audit["reference_sha256"] == modified.audit["reference_sha256"]
    assert verify_binding_audit(modified.config, modified.audit["reference_sha256"], AS_OF) is None
    with pytest.raises(ValueError, match="binding_configuration_mismatch"):
        verify_binding_audit(
            replace(initial.config, **{field: getattr(initial.config, field) + 1}),
            initial.audit["reference_sha256"],
            AS_OF,
        )


@pytest.mark.parametrize("explicit", [False, True], ids=["defaults", "custom"])
def test_build_config_cli_publishes_the_budgets_used_for_scanning(
    tmp_path: Path, explicit: bool
) -> None:
    """场景：通过正式 CLI 生成配置；预期：落盘预算、输出审计与配置指纹一致。

    例如显式 1000/9000 写入配置文件，省略参数时写入 2000000/5000000。
    """
    reference = write_all_reference(tmp_path)
    output = tmp_path / "config.json"
    audit = tmp_path / "audit.json"
    source_budget, total_budget = (1_000, 9_000) if explicit else (2_000_000, 5_000_000)
    arguments = [
        "research-all",
        "build-config",
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--audit",
        str(audit),
        "--as-of",
        AS_OF,
    ]
    if explicit:
        arguments.extend(
            [
                "--maximum-source-rows",
                str(source_budget),
                "--maximum-total-rows",
                str(total_budget),
            ]
        )
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0
    restored = ResearchConfig.from_json(output.read_text())
    assert restored.maximum_source_rows == source_budget
    assert restored.maximum_total_rows == total_budget
    report = json.loads(audit.read_text())
    assert report == json.loads(result.stdout)
    assert report["configuration_sha256"] == configuration_hash(restored)
    assert verify_binding_audit(restored, file_checksum(reference), AS_OF) is None


def test_runtime_remaining_budget_uses_the_persisted_total(tmp_path: Path) -> None:
    """场景：使用受控的旧层已保留计数扣减预算；预期：483 条新来源记录全部保留。

    例如旧层计数为 130590 时，5000000 的总预算仍充足。本例仅验证预算传递，
    不将受控计数当作执行过 130590 条旧来源记录的完整负载证据。
    """
    reference = write_all_reference(
        tmp_path,
        {
            "fx_daily": [
                {"ts_code": f"FX{index}", "trade_date": "20261002", "bid_close": 1.0}
                for index in range(483)
            ]
        },
    )
    built = build_config(reference, AS_OF)
    restored = ResearchConfig.from_json(built.config.to_json())
    legacy_retained_count = 130_590
    remaining = max(0, restored.maximum_total_rows - legacy_retained_count)
    inputs, counts = load_inputs(
        reference, AS_OF, "capacity-budget-handoff", restored, remaining_rows=remaining
    )
    assert counts["scanned_rows"] == 483
    assert counts["retained_rows"] == 483
    assert counts["excluded_rows"] == counts["discarded_rows"] == 0
    assert len({record.raw["source_record_id"] for record in inputs.rows("fx_daily")}) == 483
    assert remaining == BuildLimits().maximum_total_rows - legacy_retained_count
    assert "source_capacity_exceeded" not in inputs.source_reasons["fx_daily"]
    assert (
        sum(len(encode(record.raw).encode()) for record in inputs.rows("fx_daily")) < 16 * 1024**2
    )


def test_configured_audit_matches_actual_legacy_loader_binding_requirements(tmp_path):
    """场景：旧因子有时间规则但缺股票实体映射；预期：审计和实际加载都明确未配置。

    例如 factor_value 为 false、daily 为 true，206 注册项的 configured 汇总取实际值。
    不用虚假的实体或单位认证把因子缺口改写为已覆盖。
    """
    reference = write_all_reference(
        tmp_path,
        {
            "factor_value": [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20261002",
                    "factor_name": "synthetic",
                    "factor_value": 1,
                },
            ]
        },
    )
    built = build_config(reference, AS_OF)
    sources = built.audit["sources"]
    assert len(sources) == 206
    assert sources["factor_value"]["configured"] is False
    assert sources["factor_value"]["reason_counts"]["research_bindings_missing"] == 1
    assert sources["daily"]["configured"] is True
    assert (
        built.audit["summary"]["configured_apis"]
        == sum(source["configured"] for source in sources.values())
        == 205
    )
    inputs, _ = load_equity_inputs(reference, AS_OF, "binding-truth", equity_config(built.config))
    assert inputs.rows("factor_value") == ()
    assert "research_bindings_missing" in inputs.source_reasons["factor_value"]
    assert inputs.rows("daily")
