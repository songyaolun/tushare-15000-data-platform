"""全产品命令行安全错误、JSON契约和有界结果读取。"""

import json

import duckdb
import pytest
from all_products_fixture import AS_OF, DATABASE_CONFIG, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.cross_asset_research import cli
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.pipeline import run
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.equity_research.storage import initialize_tables


@pytest.mark.parametrize("contents", ["{}", "[1]", "invalid-json"])
def test_run_rejects_nonarray_or_nondict_holdings_without_echoing_contents(tmp_path, contents):
    """场景：持仓JSON不是对象数组；预期：退出1、只给固定码，不回显输入载荷。"""
    path = tmp_path / "holdings.json"
    path.write_text(contents)
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(tmp_path / "config"),
            "--as-of",
            AS_OF,
            "--holdings",
            str(path),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": ["all_products_run_failed"],
    }


def test_run_complete_branch_preserves_explicit_holdings_and_product_selection(
    tmp_path, monkeypatch
):
    """场景：底层真实执行摘要完整；预期：CLI退出0并原样转交显式持仓、产品和容量。"""
    configuration, holdings = tmp_path / "config.json", tmp_path / "holdings.json"
    configuration.write_text(ResearchConfig().to_json())
    holdings.write_text('[{"entity_id":"cn:security:A:listed","weight":0.25}]')
    received = []

    def complete(*args, **kwargs):
        """记录CLI传参后模拟完整执行摘要；例：P05及0.25权重原样保留。"""
        received.append((args, kwargs))
        return {"status": "complete", "result_count": 1}

    monkeypatch.setattr(cli, "run_products", complete)
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(configuration),
            "--as-of",
            AS_OF,
            "--holdings",
            str(holdings),
            "--product",
            "P05",
            "--maximum-results",
            "10",
        ],
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["status"] == "complete"
    assert received[0][0][-1] == ("P05",)
    assert received[0][1]["holdings"] == ({"entity_id": "cn:security:A:listed", "weight": 0.25},)
    assert received[0][1]["limits"].maximum_results == 10


@pytest.mark.parametrize("command", ["build-config", "run", "validate", "results", "audit"])
def test_all_commands_return_safe_json_errors_for_missing_files(tmp_path, command):
    """场景：数据库或配置不存在；预期：每个入口退出1并输出固定结构错误。"""
    arguments = ["research-all", command, "--output", str(tmp_path / "absent")]
    if command in {"build-config", "run", "validate"}:
        arguments += ["--reference", str(tmp_path / "reference")]
    if command in {"build-config", "run"}:
        arguments += ["--as-of", AS_OF]
    if command == "run":
        arguments += ["--config", str(tmp_path / "config")]
    if command == "results":
        arguments += ["--product", "P05"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    parsed = json.loads(result.stdout)
    assert parsed["reason_codes"] and str(tmp_path) not in result.stdout


@pytest.mark.parametrize("command", ["results", "audit"])
def test_read_commands_reject_absent_current_run(tmp_path, command):
    """场景：库结构存在但无current_run；预期：失败而非将空库当有效研究。"""
    output = tmp_path / "empty.duckdb"
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        initialize_tables(connection)
    arguments = ["research-all", command, "--output", str(output)]
    if command == "results":
        arguments += ["--product", "P05"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason_codes"] == ["all_products_read_failed"]


def test_results_rejects_unknown_product_and_invalid_limit(tmp_path):
    """场景：未知产品或读取上限越界；预期：未知产品退出1，0和513由CLI拒绝。"""
    runner = CliRunner()
    common = ["research-all", "results", "--output", str(tmp_path / "missing")]
    assert runner.invoke(app, common + ["--product", "P99"]).exit_code == 1
    for value in ("0", "513"):
        assert runner.invoke(app, common + ["--product", "P05", "--limit", value]).exit_code == 2
    assert runner.invoke(app, ["research-all", "--help"]).exit_code == 0


def test_results_limit_alias_and_audit_report_actual_current_run(tmp_path):
    """场景：当前批次有主题和预期；预期：别名等价、JSONL按limit截断、审计可精确读取。"""
    reference = write_reference(tmp_path)
    output = tmp_path / "products.duckdb"
    summary = run(reference, output, AS_OF, build_config(reference, AS_OF).config, ("P02", "P05"))
    runner = CliRunner()
    common = ["research-all", "results", "--output", str(output)]
    all_themes = runner.invoke(app, common + ["--product", "P02"])
    assert all_themes.exit_code == 0
    assert len(all_themes.stdout.splitlines()) == 2
    limited = runner.invoke(app, common + ["--product", "theme_opportunities/v1", "--limit", "1"])
    assert limited.exit_code == 0 and len(limited.stdout.splitlines()) == 1
    assert json.loads(limited.stdout)["envelope"]["product_id"] == "theme_opportunities/v1"
    absent = runner.invoke(app, common + ["--product", "P11"])
    assert absent.exit_code == 0 and absent.stdout == ""
    audit = runner.invoke(app, ["research-all", "audit", "--output", str(output)])
    assert audit.exit_code == 0 and json.loads(audit.stdout) == summary["consumption"]


def test_build_config_explicit_audit_path_and_capacity_failure_preserve_output(tmp_path):
    """场景：指定独立审计路径并随后超扫描预算；预期：两份产物建立，失败不覆盖旧配置。"""
    reference = write_reference(tmp_path)
    output, audit = tmp_path / "config.json", tmp_path / "audit.json"
    args = [
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
    runner = CliRunner()
    built = runner.invoke(app, args)
    assert built.exit_code == 0 and json.loads(audit.read_text()) == json.loads(built.stdout)
    original = output.read_bytes()
    failed = runner.invoke(app, args + ["--maximum-total-rows", "1"])
    assert failed.exit_code == 1 and output.read_bytes() == original


def test_holdings_input_file_over_eight_mib_is_rejected_before_reading(tmp_path):
    """场景：持仓文件大于8MiB；预期：按文件大小预检拒绝，不能先读入全部持仓。"""
    holdings = tmp_path / "holdings.json"
    with holdings.open("wb") as handle:
        handle.seek(8 * 1024 * 1024)
        handle.write(b" ")
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            AS_OF,
            "--config",
            str(tmp_path / "config"),
            "--holdings",
            str(holdings),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["reason_codes"] == ["all_products_run_failed"]


def test_persistence_error_retains_safe_phase_without_raw_exception_text(tmp_path, monkeypatch):
    """场景：持久化回滚内存失败；预期：CLI给固定阶段和错误类别，绝不输出内部错误原文。"""
    from tushare_entitlement_hub.equity_research.persistence_runtime import PersistenceError

    config = tmp_path / "config.json"
    config.write_text(ResearchConfig().to_json())

    def failed(*args, **kwargs):
        """模拟回滚内存失败；例：原异常文本仅保留在链内，不进入标准输出。"""
        raise PersistenceError("rollback", MemoryError("synthetic-private-database-payload"))

    monkeypatch.setattr(cli, "run_products", failed)
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            AS_OF,
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": ["research_rollback_memory_failed"],
        "phase": "rollback",
        "category": "memory",
        "database_error_code": None,
    }
    assert "synthetic-private" not in result.stdout
