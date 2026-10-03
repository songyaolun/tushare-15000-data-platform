"""研究绑定命令行的离线完整链路、确定性及失败关闭验收。"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import duckdb
import pytest
from test_research_binding_fixture import AS_OF, DATABASE_CONFIG, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.reference_storage import file_checksum


def build_arguments(reference: Path, output: Path, as_of: str = AS_OF) -> list[str]:
    """构造离线绑定生成命令，所有输入路径由测试临时目录提供。

    输入 reference.duckdb 和 config.json，输出含 research build-config 与固定时点的参数。
    """
    return [
        "research",
        "build-config",
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--as-of",
        as_of,
    ]


def run_arguments(
    reference: Path, output: Path, configuration: Path | None, as_of: str = AS_OF
) -> list[str]:
    """构造六产品研究命令，并在指定配置时显式附加配置路径。

    输入 configuration=None，输出无 --config 的旧入口参数，用于零扫描回归。
    """
    arguments = [
        "research",
        "run",
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--as-of",
        as_of,
    ]
    if configuration is not None:
        arguments.extend(["--config", str(configuration)])
    return arguments


def stored_snapshot(output: Path) -> dict[str, list[tuple[Any, ...]]]:
    """只读获取研究库各表的稳定逻辑快照，以比较幂等运行前后的全部记录。

    输入包含一条 research.run 的数据库，输出表名到排序后元组记录的映射。
    """
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='research' ORDER BY table_name"
        ).fetchall()
        return {
            table: sorted(connection.execute(f'SELECT * FROM research."{table}"').fetchall())
            for (table,) in tables
        }


def test_binding_cli_build_run_validate_is_offline_deterministic_and_idempotent(
    tmp_path, monkeypatch
):
    """场景：合成 accepted 库生成配置并运行六产品两次；预期：有消费且产物不重复。

    输入两只合成股票与行情，输出字节一致的配置、审计和摘要；validate 为真，
    scanned/observed 大于零，完整数据库逻辑快照相同，reference 字节保持不变。
    """
    reference = write_reference(tmp_path)
    original_hash = file_checksum(reference)
    configuration = tmp_path / "research-config.json"
    output = tmp_path / "products.duckdb"

    def forbidden(*args, **kwargs):
        """拒绝研究路径访问网络或凭据；例：create_connection 调用立即使测试失败。"""
        raise AssertionError("研究绑定验收不得访问网络或真实凭据")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("tushare_entitlement_hub.cli.configured_credential", forbidden)
    runner = CliRunner()
    first_build = runner.invoke(app, build_arguments(reference, configuration))
    assert first_build.exit_code == 0, first_build.output
    first_audit = json.loads(first_build.stdout)
    config_bytes = configuration.read_bytes()
    audit_path = configuration.with_suffix(".audit.json")
    audit_bytes = audit_path.read_bytes()
    configured = ResearchConfig.from_json(config_bytes.decode())
    assert configured.binding_audit["reference_sha256"] == original_hash
    assert configured.binding_audit["summary"]["observed_apis"] == 0
    assert configured.foundation.entities
    second_build = runner.invoke(app, build_arguments(reference, configuration))
    assert second_build.exit_code == 0, second_build.output
    assert configuration.read_bytes() == config_bytes
    assert audit_path.read_bytes() == audit_bytes
    assert second_build.stdout == first_build.stdout
    assert json.loads(second_build.stdout) == first_audit

    arguments = run_arguments(reference, output, configuration)
    first_run = runner.invoke(app, arguments)
    assert first_run.exit_code in {0, 2}, first_run.output
    summary = json.loads(first_run.stdout)
    assert summary["processing"]["scanned_rows"] > 0
    assert summary["consumption"]["observed_apis"] > 0
    assert summary["bindings"]["observed_apis"] > 0
    before_rerun = stored_snapshot(output)
    second_run = runner.invoke(app, arguments)
    assert second_run.exit_code == first_run.exit_code, second_run.output
    assert second_run.stdout == first_run.stdout
    assert stored_snapshot(output) == before_rerun
    assert len(before_rerun["run"]) == 1
    events = [json.loads(line) for line in second_run.stderr.splitlines()]
    assert events[-1]["stage"] == "reused"

    validation = runner.invoke(
        app,
        ["research", "validate", "--reference", str(reference), "--output", str(output)],
    )
    assert validation.exit_code == 0, validation.output
    assert json.loads(validation.stdout)["valid"] is True
    assert file_checksum(reference) == original_hash


def test_binding_cli_honors_explicit_audit_output(tmp_path):
    """场景：显式指定机器审计文件；预期：独立文件生成且不回写默认审计路径。

    输入 --audit reports/bindings.json，输出该文件中的时点及来源摘要，与配置审计一致。
    """
    reference = write_reference(tmp_path)
    configuration = tmp_path / "config.json"
    audit_path = tmp_path / "reports" / "bindings.json"
    result = CliRunner().invoke(
        app, build_arguments(reference, configuration) + ["--audit", str(audit_path)]
    )
    assert result.exit_code == 0, result.output
    configured = ResearchConfig.from_json(configuration.read_text())
    assert json.loads(audit_path.read_text()) == configured.binding_audit
    assert not configuration.with_suffix(".audit.json").exists()


@pytest.mark.parametrize("scenario", ["missing_reference", "invalid_as_of", "reference_output"])
def test_binding_cli_invalid_input_preserves_input_and_returns_safe_failure(tmp_path, scenario):
    """场景：输入库不存在、时点非法或配置覆盖来源；预期：退出 1 且不写半成品。

    输入非法日期文本时，输出固定 reason_codes，消息不包含原始输入；原库哈希不变。
    """
    reference = write_reference(tmp_path)
    original_hash = file_checksum(reference)
    configuration = reference if scenario == "reference_output" else tmp_path / "config.json"
    arguments = build_arguments(
        tmp_path / "absent.duckdb" if scenario == "missing_reference" else reference,
        configuration,
        "invalid-private-date" if scenario == "invalid_as_of" else AS_OF,
    )
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert {key: summary[key] for key in ("status", "reason_codes")} == {
        "status": "failed",
        "reason_codes": ["research_build_config_failed"],
    }
    assert summary["detail_code"] == {
        "missing_reference": "reference_library_missing",
        "invalid_as_of": "invalid_timestamp_or_timezone",
        "reference_output": "input_output_path_collision",
    }[scenario]
    assert file_checksum(reference) == original_hash
    if configuration != reference:
        assert not configuration.exists()


@pytest.mark.parametrize("scenario", ["as_of", "reference", "configuration"])
def test_binding_cli_rejects_config_reuse_after_bound_input_changes(tmp_path, scenario):
    """场景：绑定生成后改变截止、来源字节或配置内容；预期：run 关闭消费且不建产物。

    输入生成配置与下一天时点时，输出 research_run_failed，避免把旧证据用于新快照。
    """
    reference = write_reference(tmp_path)
    configuration = tmp_path / "config.json"
    output = tmp_path / "products.duckdb"
    runner = CliRunner()
    generated = runner.invoke(app, build_arguments(reference, configuration))
    assert generated.exit_code == 0, generated.output
    if scenario == "reference":
        with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
            connection.execute("CREATE TABLE synthetic_change(value INTEGER)")
    elif scenario == "configuration":
        payload = json.loads(configuration.read_text())
        payload["maximum_total_rows"] += 1
        configuration.write_text(json.dumps(payload))
    result = runner.invoke(
        app,
        run_arguments(
            reference,
            output,
            configuration,
            "2026-10-04T12:00:00Z" if scenario == "as_of" else AS_OF,
        ),
    )
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert {key: summary[key] for key in ("status", "reason_codes")} == {
        "status": "failed",
        "reason_codes": ["research_run_failed"],
    }
    assert not output.exists()


def test_binding_cli_legacy_unconfigured_run_keeps_zero_scan_degradation(tmp_path):
    """场景：合成来源存在但未显式提供配置；预期：旧入口仍零扫描、零观测降级。

    输入不带 --config 的研究命令，输出 incomplete、退出 2，不隐式构建绑定。
    """
    reference = write_reference(tmp_path)
    output = tmp_path / "products.duckdb"
    result = CliRunner().invoke(app, run_arguments(reference, output, None))
    assert result.exit_code == 2, result.output
    summary = json.loads(result.stdout)
    assert summary["status"] == "incomplete"
    assert summary["processing"]["scanned_rows"] == 0
    assert summary["consumption"]["observed_apis"] == 0


def test_binding_cli_help_documents_explicit_build_requirements(tmp_path):
    """场景：查看帮助或省略必需时点；预期：构建命令可发现，缺 --as-of 被参数层拒绝。

    输入 research build-config --help，输出成功帮助；无 as-of 构建退出 2 且无文件。
    """
    runner = CliRunner()
    help_result = runner.invoke(app, ["research", "build-config", "--help"])
    assert help_result.exit_code == 0
    assert "--reference" in help_result.stdout
    assert "--output" in help_result.stdout
    assert "--as-of" in help_result.stdout
    configuration = tmp_path / "config.json"
    incomplete = runner.invoke(
        app,
        [
            "research",
            "build-config",
            "--reference",
            str(tmp_path / "reference.duckdb"),
            "--output",
            str(configuration),
        ],
    )
    assert incomplete.exit_code == 2
    assert not configuration.exists()
