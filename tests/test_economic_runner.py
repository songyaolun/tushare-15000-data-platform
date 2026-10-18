"""短反四账户离线编排、冻结配置、来源摘要与拒绝复用验证。"""

import csv
import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from test_economic_execution import frozen_strategy
from test_portfolio_economic import economic_panel as economic_panel
from test_regime_pipeline import save_panel, write_partition

from tushare_entitlement_hub.economic_research import runner


@pytest.fixture
def experiment_paths(tmp_path, economic_panel, monkeypatch):
    """写入60日公开合成面板和空分红；返回panel/raw/output及四个冻结策略，不访问网络。"""
    panel_path, raw, output = (tmp_path / name for name in ("panel", "raw", "output"))
    save_panel(panel_path, economic_panel)
    raw.mkdir()
    for day in economic_panel.calendar:
        write_partition(raw / f"dividend-{day}.json", "dividend", {"ex_date": day}, [])
    strategies = [
        frozen_strategy(economic_panel, name, index % 2)
        for index, name in enumerate(("reversal", "delayed", "cost_stress", "matched_control"))
    ]
    strategies[-1] = replace(strategies[-1], matched_pairs=((0, 0, 1),))
    monkeypatch.setattr(runner, "build_strategies", Mock(return_value=strategies))
    return panel_path, raw, output, strategies


def test_four_account_experiment_exports_auditable_clocks_and_refuses_reuse(experiment_paths):
    """场景：四组完整离线回放；预期独立账户、配置与source摘要、逐笔/逐日CSV及报告齐全，重复执行拒绝。"""
    panel_path, raw, output, strategies = experiment_paths
    completed = runner.run_reversal_experiment(panel_path, raw, output)
    assert completed == output / "report.json"
    report = json.loads(completed.read_text())
    assert report["completed"] and report["primary_control"] == "matched_control"
    assert [row["name"] for row in report["strategies"]] == [
        strategy.name for strategy in strategies
    ]
    contract = report["experiment"]
    assert contract == json.loads((output / "experiment.json").read_text())
    assert contract["rules"]["historical_pit_certified"] is False
    assert len(contract["input_hash"]) == len(contract["source_hash"]) == 64
    for strategy, signature in zip(strategies, contract["configurations"], strict=True):
        path = output / strategy.name
        saved = json.loads((path / "metrics.json").read_text())
        assert saved["configuration"] == signature
        assert len(signature["source_days_hash"]) == len(signature["configuration_hash"]) == 64
        assert saved["metrics"]["closed_trades"] == 1
        assert abs(saved["metrics"]["reconciliation_error"]) < 1e-7
        assert saved["configuration"]["policy"]["fixed_price_exits"] is False
        with (path / "trades.csv").open() as stream:
            trade = list(csv.DictReader(stream))[0]
        delay = int(strategy.name == "delayed")
        assert int(trade["source_signal"]) == 0
        assert int(trade["confirmation"]) == delay
        assert int(trade["entry"]) == 1 + delay
        assert int(trade["exit"]) == 6 + delay
        assert (path / "entries.csv").exists() and (path / "daily.csv").exists()
        assert not (path / "matched_trades.csv").exists()
    with (output / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and "both_targets" in rows[0]
    matching = report["matching"]
    assert matching["base_candidate_count"] == matching["matched_count"] == 1
    assert matching["unmatched_count"] == 0 and matching["matching_rate"] == 1
    with (output / "matched_control" / "matched_pairs.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert pairs[0]["source_signal"] == "0"
    assert pairs[0]["treatment"] == "0" and pairs[0]["control"] == "1"
    comparison = report["matched_account_comparison"]
    assert len(comparison["daily_return_spread"]) == 60
    assert comparison["descriptive_spread_interval"]["selection_adjusted"] is False
    base, control = report["strategies"][0], report["strategies"][-1]
    assert comparison["total_return_difference"] == pytest.approx(
        base["total_return"] - control["total_return"]
    )
    assert comparison["annualized_return_difference"] == pytest.approx(
        base["annualized_return"] - control["annualized_return"]
    )
    assert base["mean_holding_sessions"] == 5
    assert base["mean_touched_calendar_sessions"] == 6
    assert (output / "matched_account_spread.csv").exists()
    with pytest.raises(ValueError, match="reversal_output_must_be_empty"):
        runner.run_reversal_experiment(panel_path, raw, output)


def test_empty_output_is_supported_and_input_is_hashed_only_once(experiment_paths, monkeypatch):
    """场景：预建空输出目录；预期可执行，完整数据只摘要一次，源码在开始和完成各检查一次。"""
    panel_path, raw, output, _ = experiment_paths
    output.mkdir()
    inputs = Mock(wraps=runner.data_fingerprint)
    source = Mock(wraps=runner.source_fingerprint)
    monkeypatch.setattr(runner, "data_fingerprint", inputs)
    monkeypatch.setattr(runner, "source_fingerprint", source)
    runner.run_reversal_experiment(panel_path, raw, output, start="20240101")
    inputs.assert_called_once_with(panel_path, raw)
    assert source.call_count == 2


def test_source_drift_retains_partial_evidence_without_completion_marker(
    experiment_paths, monkeypatch
):
    """场景：运行期间源码摘要发生变化；预期拒绝完成报告，保留可核对半成品并禁止原目录重跑。"""
    panel_path, raw, output, _ = experiment_paths
    monkeypatch.setattr(runner, "source_fingerprint", Mock(side_effect=["a" * 64, "b" * 64]))
    with pytest.raises(ValueError, match="reversal_source_changed_during_run"):
        runner.run_reversal_experiment(panel_path, raw, output)
    assert (output / "reversal" / "metrics.json").exists()
    assert not (output / "report.json").exists()
    with pytest.raises(ValueError, match="reversal_output_must_be_empty"):
        runner.run_reversal_experiment(panel_path, raw, output)


@pytest.mark.parametrize("start", ["bad", "20241301", "20250101", "2024011"])
def test_invalid_start_rejected_before_creating_output(experiment_paths, start):
    """场景：开始日期非法、非规范或超过面板末日；预期运行前拒绝且没有输出副作用。"""
    panel_path, raw, output, _ = experiment_paths
    with pytest.raises(ValueError, match="invalid_reversal_start_or_calendar"):
        runner.run_reversal_experiment(panel_path, raw, output, start=start)
    assert not output.exists()


@pytest.mark.parametrize(
    "case", ["empty_calendar", "duplicate_calendar", "output_file", "missing_dividend"]
)
def test_input_contract_failures_are_explicit(experiment_paths, economic_panel, case):
    """场景：日历空/重、输出是文件或目标分红缺失；预期明确失败，不把缺失分红当零。"""
    _, raw, output, _ = experiment_paths
    if case == "empty_calendar":
        economic_panel.calendar = []
    elif case == "duplicate_calendar":
        economic_panel.calendar[1] = economic_panel.calendar[0]
    elif case == "output_file":
        output.write_text("existing")
    else:
        (raw / f"dividend-{economic_panel.calendar[-1]}.json").unlink()
    with pytest.raises(ValueError):
        runner._validate_inputs(economic_panel, raw, "20240101", output)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_strategy_groups_rejected(experiment_paths, monkeypatch, duplicate):
    """场景：构建器返回缺组或四个同名配置；预期报告生成前拒绝伪造四臂比较。"""
    panel_path, raw, output, strategies = experiment_paths
    monkeypatch.setattr(
        runner,
        "build_strategies",
        Mock(return_value=[strategies[0]] * 4 if duplicate else strategies[:3]),
    )
    with pytest.raises(ValueError, match="reversal_requires_four_unique_strategies"):
        runner.run_reversal_experiment(panel_path, raw, output)
    assert not output.exists()


def test_configuration_hashes_capture_sources_policy_and_array_contents(economic_panel):
    """场景：修改原信号日/政策/分数，或仅更换矩阵布局；预期内容变更改摘要，C/F布局不改变。"""
    strategy = frozen_strategy(economic_panel)
    baseline = runner.configuration(strategy)
    layout = replace(
        strategy,
        entry_mask=np.asfortranarray(strategy.entry_mask),
        scores=np.asfortranarray(strategy.scores),
    )
    assert runner.configuration(layout) == baseline
    assert (
        runner.configuration(replace(strategy, source_days=strategy.source_days + 1))[
            "source_days_hash"
        ]
        != baseline["source_days_hash"]
    )
    assert (
        runner.configuration(replace(strategy, scores=strategy.scores * 2))["scores_hash"]
        != baseline["scores_hash"]
    )
    assert (
        runner.configuration(replace(strategy, policy=replace(strategy.policy, fee_multiplier=2)))[
            "configuration_hash"
        ]
        != baseline["configuration_hash"]
    )
    assert (
        runner.configuration(replace(strategy, matched_pairs=((0, 0, 1),)))["matched_pairs_hash"]
        != baseline["matched_pairs_hash"]
    )
    assert runner._array_fingerprint(np.zeros((0, 2))) != runner._array_fingerprint(
        np.zeros((0, 3))
    )


def test_data_hash_tracks_actual_loader_inputs_and_ignores_unrelated_files(experiment_paths):
    """场景：更改面板/分红/分红纠正清单；预期全部改摘要，无关文件和probe不参与。"""
    panel_path, raw, _, _ = experiment_paths
    baseline = runner.data_fingerprint(panel_path, raw)
    (raw / "unrelated.json").write_text("unrelated")
    (raw / "dividend-probe.json").write_text("probe")
    assert runner.data_fingerprint(panel_path, raw) == baseline
    paths = [
        panel_path / "panel.json",
        panel_path / "values-raw_close.npy",
        sorted(raw.glob("dividend-2024*.json"))[0],
    ]
    for path in paths:
        with path.open("ab") as stream:
            stream.write(b"\n")
        changed = runner.data_fingerprint(panel_path, raw)
        assert changed != baseline
        baseline = changed
    (raw / "distribution-resolutions.json").write_text("{}")
    assert runner.data_fingerprint(panel_path, raw) != baseline


def test_source_hash_includes_all_research_and_transitive_python_modules(tmp_path, monkeypatch):
    """场景：四研究包及共同解析模块任一源码改动；预期来源摘要变化，非源码文件不参与。"""
    package = tmp_path / "package"
    for name in (
        "economic_research",
        "regime_research",
        "strategy_discovery",
        "strategy_optimization",
    ):
        (package / name).mkdir(parents=True)
    monkeypatch.setattr(runner, "__file__", str(package / "economic_research" / "runner.py"))
    previous = runner.source_fingerprint()
    for relative in (
        "economic_research/reversal.py",
        "regime_research/portfolio.py",
        "strategy_discovery/runner.py",
        "strategy_optimization/features.py",
        "common.py",
    ):
        (package / relative).write_text("VERSION = 1\n")
        current = runner.source_fingerprint()
        assert current != previous
        previous = current
    (package / "notes.txt").write_text("irrelevant")
    assert runner.source_fingerprint() == previous


def test_matched_comparison_rejects_misaligned_dates():
    """场景：基础与对照日期不同；预期拒绝位置相减，不能把不同交易日收益当配对差。"""
    with pytest.raises(ValueError, match="reversal_comparison_dates_mismatch"):
        runner.matched_account_comparison({"dates": ["20240101"]}, {"dates": ["20240102"]})


def test_no_signal_experiment_reports_unavailable_holding_and_matching(
    experiment_paths, monkeypatch
):
    """场景：四账户均无合格信号；预期持有统计/匹配率缺省，实际现金对照日差为零且不达双目标。"""
    panel_path, raw, output, strategies = experiment_paths
    empty = [
        replace(strategy, entry_mask=np.zeros_like(strategy.entry_mask), matched_pairs=())
        for strategy in strategies
    ]
    monkeypatch.setattr(runner, "build_strategies", Mock(return_value=empty))
    completed = runner.run_reversal_experiment(panel_path, raw, output)
    report = json.loads(completed.read_text())
    assert report["matching"]["matching_rate"] is None
    assert report["matching"]["base_candidate_count"] == 0
    for account in report["strategies"]:
        assert account["mean_holding_sessions"] is None
        assert account["mean_touched_calendar_sessions"] is None
        assert account["both_targets"] is False
    assert all(
        row["spread"] == 0 for row in report["matched_account_comparison"]["daily_return_spread"]
    )


def test_real_reversal_builder_runs_four_accounts_and_independent_audit(tmp_path, economic_panel):
    """场景：真实股票池/信号/匹配构建到四账本；预期基础t+1、延迟t+2成交且独立费用审计通过。"""
    from test_economic_reversal import _panel

    from tushare_entitlement_hub.economic_research.audit import audit_execution

    panel = _panel()
    count = len(panel.calendar)
    panel.market = {name: values[:count].copy() for name, values in economic_panel.market.items()}
    for name in ("raw_open", "raw_close", "raw_high", "raw_low"):
        panel.values[name] = panel.values["close"].copy()
    for name, period in (("return1", 1), ("return5", 5), ("return20", 20)):
        panel.values[name][:] = 0
        panel.values[name][period:] = (
            panel.values["close"][period:] / panel.values["close"][:-period] - 1
        )
    panel_path, raw, output = (tmp_path / name for name in ("real-panel", "raw", "real-output"))
    save_panel(panel_path, panel)
    raw.mkdir()
    for day in panel.calendar:
        write_partition(raw / f"dividend-{day}.json", "dividend", {"ex_date": day}, [])
    completed = runner.run_reversal_experiment(panel_path, raw, output, start=panel.calendar[0])
    report = json.loads(completed.read_text())
    assert report["matching"]["base_candidate_count"] > 0
    assert report["matching"]["matched_count"] > 0
    for name in ("reversal", "delayed", "cost_stress", "matched_control"):
        checked = audit_execution(output / name, panel, delay=int(name == "delayed"))
        assert checked["checked_trades"] > 0 and checked["fees_fully_reconciled"]
    with (output / "matched_control" / "matched_pairs.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == report["matching"]["matched_count"]
