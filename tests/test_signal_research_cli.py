"""离线CLI边界：流程依赖替身，确认参数及产物路径传递。"""

from pathlib import Path

from typer.testing import CliRunner

from tushare_entitlement_hub.signal_research import cli


def test_run_command_passes_protocol_and_writes_report(monkeypatch, tmp_path: Path) -> None:
    """场景：已认证微型缓存；预期CLI串联离线加载、研究和报告且不联网。"""
    data = {"trading_dates": ["20260101"]}
    monkeypatch.setattr(cli, "load_history_database", lambda *_: data)
    monkeypatch.setattr(cli, "assert_complete_history", lambda *_: None)
    monkeypatch.setattr(cli, "validate_research_window", lambda *_: None)
    monkeypatch.setattr(cli, "iter_security_history", lambda *_: iter(()))
    observed = []

    def fake_run(securities, calendar, output, configuration):
        """输入研究协议；输出空运行清单并提供报告消费文件。"""
        observed.append(configuration)
        (output / "signal_records.jsonl").write_text("", encoding="utf-8")
        return {"counts": {}}

    def fake_report(summary, manifest, output):
        """输入汇总及manifest；输出报告路径用于CLI打印。"""
        assert manifest["data"] == data
        return output / "report.md", output / "report.json"

    monkeypatch.setattr(cli, "run_research", fake_run)
    monkeypatch.setattr(cli, "write_research_report", fake_report)
    monkeypatch.setattr(cli, "enrich_signal_percentiles", lambda *_: None)
    # 真实DuckDB上下文使用空私有文件；未触发任何网络transport。
    database = tmp_path / "result" / "history.duckdb"
    database.parent.mkdir()
    with cli.duckdb.connect(str(database)):
        pass
    result = CliRunner().invoke(
        cli.signal_research_app,
        [
            "run",
            "--history-root",
            str(tmp_path),
            "--output-directory",
            str(database.parent),
            "--start-date",
            "20260101",
            "--end-date",
            "20260102",
            "--chart-samples",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "report.md" in result.output
    assert observed[0].start_date == "20260101"
    assert observed[0].chart_samples_per_year_direction == 0


def test_review_command_passes_paths(monkeypatch, tmp_path: Path) -> None:
    """场景：显式review子命令；预期：仅串联只读复核，打印CSV路径。"""
    observed = []

    def review(output, benchmark):
        """输入两个CLI路径，输出复核CSV；捕获参数验证离线调用。"""
        observed.append((output, benchmark))
        return output / "baseline_trade_review.csv"

    monkeypatch.setattr(cli, "build_retrospective_review", review)
    result = CliRunner().invoke(
        cli.signal_research_app,
        [
            "review",
            "--output-directory",
            str(tmp_path / "output"),
            "--benchmark-root",
            str(tmp_path / "benchmark"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert observed == [(tmp_path / "output", tmp_path / "benchmark")]
    assert "baseline_trade_review.csv" in result.output


def test_validate_research_window_rejects_short_or_missing_window() -> None:
    """场景：短缓存或缺请求终日；预期拒绝，不发布误标的五年结果。"""
    import pytest

    configuration = cli.ResearchConfiguration("20260101", "20260102")
    with pytest.raises(ValueError, match="预热"):
        cli.validate_research_window(["20260101", "20260102"], configuration)
    with pytest.raises(ValueError, match="预热"):
        cli.validate_research_window(["20260101"], configuration)
    calendar = [str(index) for index in range(250)] + ["20260101", "20260102"]
    assert cli.validate_research_window(calendar, configuration) is None
