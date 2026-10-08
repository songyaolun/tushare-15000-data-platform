"""离线研究命令行的JSON输出、退出码及真实数据库验收。"""

import json
import socket

import pytest
from research_fixture import AS_OF, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app


@pytest.mark.parametrize("command", ["initialize", "run"])
def test_cli_complete_market_preserves_json_and_progress(tmp_path, monkeypatch, command):
    """场景：合成行情与限价完整；预期：两种入口成功、stdout为单JSON、进度仅stderr。"""
    reference, config = write_reference(tmp_path, 2)
    output = tmp_path / "products.duckdb"
    config_path = tmp_path / "configuration.json"
    config_path.write_text(config.to_json())

    def forbidden(*args, **kwargs):
        """拒绝任何凭据或网络调用；例：研究路径调用socket即让用例失败。"""
        raise AssertionError("研究命令不得联网或读取凭据")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("tushare_entitlement_hub.cli.configured_credential", forbidden)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research",
            command,
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--as-of",
            AS_OF,
            "--config",
            str(config_path),
            "--product",
            "P01",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["status"] == "complete"
    assert summary["products"] == ["market_context/v1"]
    assert summary["result_count"] == 1
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert events[0]["stage"] == "start" and events[-1]["stage"] == "complete"
    verified = runner.invoke(
        app, ["research", "validate", "--reference", str(reference), "--output", str(output)]
    )
    assert verified.exit_code == 0
    assert json.loads(verified.stdout)["valid"]


def test_cli_empty_bindings_and_dependency_selection_are_explicit(tmp_path):
    """场景：无配置运行P03；预期：三个依赖同批执行、缺绑定退出2且JSON披露incomplete。"""
    reference, _ = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    result = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--as-of",
            AS_OF,
            "--product",
            "P03",
        ],
    )
    assert result.exit_code == 2
    summary = json.loads(result.stdout)
    assert summary["status"] == "incomplete"
    assert summary["result_count"] == 4
    assert summary["products"] == [
        "fundamental_dossier/v1",
        "capital_governance_risks/v1",
        "flow_feasibility/v1",
        "equity_research_candidates/v1",
    ]


@pytest.mark.parametrize(
    "scenario", ["reference", "time", "product", "config", "json", "collision"]
)
def test_cli_invalid_input_has_safe_machine_readable_failure(tmp_path, scenario):
    """场景：库、日期、产品或配置非法；预期：退出1，只给固定错误码，不输出输入内容。"""
    reference, _ = write_reference(tmp_path, 0)
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{invalid-json")
    arguments = [
        "research",
        "run",
        "--reference",
        str(reference.with_name("absent.duckdb") if scenario == "reference" else reference),
        "--output",
        str(reference if scenario == "collision" else tmp_path / "products.duckdb"),
        "--as-of",
        "not-a-date" if scenario == "time" else AS_OF,
    ]
    if scenario == "product":
        arguments += ["--product", "P99"]
    if scenario in {"config", "json"}:
        arguments += [
            "--config",
            str(configuration if scenario == "json" else configuration.with_name("absent.json")),
        ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": ["research_run_failed"],
    }


def test_cli_validate_missing_and_help_requirements(tmp_path):
    """场景：缺产品库、帮助与缺as-of；预期：校验退出1，帮助成功，缺参数被CLI拒绝。"""
    reference, _ = write_reference(tmp_path, 0)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research",
            "validate",
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "absent.duckdb"),
        ],
    )
    assert result.exit_code == 1 and not json.loads(result.stdout)["valid"]
    assert runner.invoke(app, ["research", "--help"]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "research",
                "run",
                "--reference",
                str(reference),
                "--output",
                str(tmp_path / "new.duckdb"),
            ],
        ).exit_code
        == 2
    )
