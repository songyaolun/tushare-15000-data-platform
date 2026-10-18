"""独立策略登记、可恢复离线编排及命令的确定性验证。"""

import csv
import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import numpy as np
import pytest
from test_regime_pipeline import save_panel, write_partition
from test_regime_research import panel as panel
from typer.testing import CliRunner

from tushare_entitlement_hub.strategy_discovery import (
    cli,
    combination,
    defensive,
    registry,
    reversion,
    runner,
    sector,
    trend,
)
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition


@pytest.fixture
def definition(panel):
    """构造全证券独立信号；输入合成面板，输出与其对齐且退出关闭的策略定义。"""
    return StrategyDefinition(
        "fixture",
        "independent",
        np.ones_like(panel.values["close"], dtype=bool),
        panel.values["return20"].copy(),
    )


@pytest.fixture
def discovery_paths(tmp_path, monkeypatch, panel):
    """保存面板和每日空分红；输入临时目录，输出四个本地研究目录。"""
    panel_path, features, raw, output = (
        tmp_path / name for name in ("panel", "features", "raw", "output")
    )
    save_panel(panel_path, panel)
    features.mkdir()
    raw.mkdir()
    for day in panel.calendar:
        write_partition(raw / f"dividend-{day}.json", "dividend", {"ex_date": day}, [])
    monkeypatch.setattr(runner, "load_features", Mock(return_value={}))
    return panel_path, features, raw, output


def test_registry_groups_unknown_names_and_duplicate_rejection(panel, definition, monkeypatch):
    """各组只调用对应构建器，all汇总五组；未知组和重名策略明确拒绝。"""
    builders = {}
    for name, module in (
        ("trend", trend),
        ("sector", sector),
        ("reversion", reversion),
        ("defensive", defensive),
    ):
        builders[name] = Mock(return_value=[replace(definition, name=name)])
        monkeypatch.setattr(module, "build_strategies", builders[name])
    builders["control"] = Mock(return_value=[replace(definition, name="control")])
    monkeypatch.setattr(registry, "control_candidates", builders["control"])
    features = {"fixture": panel.values["close"]}
    assert [item.name for item in registry.build_candidates(panel, features, "all")] == [
        "trend",
        "sector",
        "reversion",
        "defensive",
        "control",
    ]
    for name, builder in builders.items():
        builder.assert_called_once_with(panel, features)
        builder.reset_mock()
        assert registry.build_candidates(panel, features, name)[0].name == name
        builder.assert_called_once_with(panel, features)
    with pytest.raises(ValueError, match="unknown_discovery_group"):
        registry.build_candidates(panel, features, "missing")
    builders["sector"].return_value = builders["trend"].return_value
    with pytest.raises(ValueError, match="duplicate_strategy_name"):
        registry.build_candidates(panel, features, "all")


def test_neighborhood_changes_one_parameter_and_deduplicates(definition):
    """邻域包含原点，每次只改一个参数；默认值和一天持有期产生的重复点被去重。"""
    original = replace(definition, maximum_entry_gap=0.03, trailing_stop=0.08)
    variants = registry.neighborhood(original)
    assert len(variants) == 11 and variants[0] is original
    assert len({variant.name for variant in variants}) == len(variants)
    reference = asdict(original.policy) | {
        "maximum_entry_gap": original.maximum_entry_gap,
        "trailing_stop": original.trailing_stop,
    }
    for variant in variants[1:]:
        changed = asdict(variant.policy) | {
            "maximum_entry_gap": variant.maximum_entry_gap,
            "trailing_stop": variant.trailing_stop,
        }
        assert sum(value != reference[key] for key, value in changed.items() if key != "name") == 1
        assert variant.policy.name == variant.name
        assert variant.entry_mask is original.entry_mask
        assert variant.scores is original.scores
    assert len(registry.neighborhood(definition)) == 9
    short = replace(definition, policy=replace(definition.policy, maximum_holding=1))
    assert len(registry.neighborhood(short)) == 7
    large = replace(original, policy=replace(original.policy, stop_loss=0.9, take_profit=0.9))
    capped = {variant.name.rsplit("__", 1)[-1]: variant for variant in registry.neighborhood(large)}
    assert capped["stop_wider"].policy.stop_loss == 0.95
    assert capped["profit_farther"].policy.take_profit == 0.95


def test_registry_all_builds_sixty_three_unique_declared_candidates(panel):
    """五类真实构建器合计产生63个唯一初始配置，不依赖mock或实盘行情。"""
    names = {
        name
        for module in (trend, sector, reversion, defensive)
        for name in module.REQUIRED_FEATURES
    }
    features = {
        name: panel.values[name] if name in panel.values else np.ones_like(panel.values["close"])
        for name in names
    }
    candidates = registry.build_candidates(panel, features, "all")
    assert len(candidates) == len({candidate.name for candidate in candidates}) == 63
    assert sum(candidate.family == "defensive_rotation" for candidate in candidates) == 6
    assert candidates[-1].name == "control_volume"


def test_configuration_signal_hashes_and_legacy_baseline(definition):
    """配置保留执行参数且仅存信号摘要，数组布局不改变摘要，信号内容变化必须改变摘要。"""
    baseline = runner.configuration(None)
    assert baseline["baseline"] is True
    assert baseline["policy"] == asdict(runner.baseline_policy())
    assert baseline["policy"]["force_rebalance"] is True
    assert baseline["policy"]["switching"] is baseline["policy"]["stage_allocation"] is False
    signature = runner.configuration(definition)
    assert signature["exit_mask"] is None
    assert signature["intraday_exits"] is False
    intraday = runner.configuration(replace(definition, intraday_exits=True))
    assert intraday == signature | {"intraday_exits": True}
    assert len(signature["entry_mask"]) == len(signature["scores"]) == 64
    same = replace(definition, scores=np.asfortranarray(definition.scores))
    assert runner.configuration(same) == signature
    changed_scores = definition.scores.copy()
    changed_scores[0, 0] = np.nan
    changed = replace(definition, scores=changed_scores, exit_mask=definition.entry_mask.copy())
    updated = runner.configuration(changed)
    assert updated["scores"] != signature["scores"]
    assert updated["exit_mask"] == signature["entry_mask"]


def test_fingerprint_tracks_feature_bytes_and_ignores_unrelated_files(discovery_paths):
    """面板、原文和特征内容决定研究指纹；同内容稳定，无关文本不进入特征指纹。"""
    panel_path, features, raw, _ = discovery_paths
    initial = runner.discovery_fingerprint(panel_path, raw, features)
    assert len(initial) == 64
    assert runner.discovery_fingerprint(panel_path, raw, features) == initial
    (features / "notes.txt").write_text("unrelated")
    assert runner.discovery_fingerprint(panel_path, raw, features) == initial
    (features / "features.json").write_text("{}")
    manifested = runner.discovery_fingerprint(panel_path, raw, features)
    assert manifested != initial
    np.save(features / "sample.npy", np.zeros(2))
    array_fingerprint = runner.discovery_fingerprint(panel_path, raw, features)
    assert array_fingerprint != manifested
    np.save(features / "sample.npy", np.ones(2))
    assert runner.discovery_fingerprint(panel_path, raw, features) != array_fingerprint


def test_additional_metrics_include_cash_dividends_losses_and_empty_book():
    """现金分红纳入已实现盈亏，零收益不算亏损；没有交易时不制造胜率或持有期。"""
    result = {
        "trades": [
            {"return": 0.1, "proceeds": 108, "dividend": 2, "cost": 100, "entry": 1, "exit": 2},
            {"return": -0.05, "proceeds": 94, "dividend": 1, "cost": 100, "entry": 3, "exit": 5},
            {"return": 0.0, "proceeds": 100, "dividend": 0, "cost": 100, "entry": 6, "exit": 9},
        ],
        "entry_days": [1, 3, 6, 10],
        "open_cost": 20,
        "open_market_value": 21,
    }
    metrics = runner.additional_metrics(result)
    assert metrics == {
        "mean_winning_return": 0.1,
        "mean_losing_return": -0.05,
        "profit_factor": 2.0,
        "losing_trades": 1,
        "mean_holding_sessions": 3.0,
        "realized_profit": 5.0,
        "entry_count": 4,
        "open_cost": 20,
        "open_market_value": 21,
    }
    empty = runner.additional_metrics(result | {"trades": [], "entry_days": []})
    assert empty["mean_winning_return"] is empty["mean_losing_return"] is None
    assert empty["profit_factor"] is empty["mean_holding_sessions"] is None
    assert empty["realized_profit"] == empty["entry_count"] == empty["losing_trades"] == 0


def test_summary_ranks_targets_separately_and_handles_empty_attempts():
    """收益最高但未达双目标的策略不混入达标排名；空尝试不产生最佳配置。"""
    attempts = [
        {"name": "high", "both_targets": False, "annualized_return": 0.5},
        {"name": "passing", "both_targets": True, "annualized_return": 0.1},
        {"name": "loss", "both_targets": True, "annualized_return": -0.1},
    ]
    report = runner.summarize(attempts)
    assert report["attempts"] == 3 and report["both_target_pass"] == 2
    assert report["positive_return_meeting_targets"] == 1
    assert report["highest_return"] == attempts[0]
    assert report["highest_return_meeting_targets"] == attempts[1]
    empty = runner.summarize([])
    assert empty["attempts"] == empty["both_target_pass"] == 0
    assert empty["highest_return"] is empty["highest_return_meeting_targets"] is None


def test_baseline_full_run_csv_artifacts_resume_and_partial_recovery(discovery_paths, monkeypatch):
    """旧策略基线输出完整账本和限制说明；完成结果直接恢复，无完成标记的目录重新执行。"""
    report_path = runner.run_discovery(*discovery_paths, group="baseline")
    report = json.loads(report_path.read_text())
    assert report["complete_grid"] and report["attempts"] == 1
    assert report["all_data_used_for_optimization"] is True
    assert report["historical_pit_certified"] is False
    assert report["benchmark"] == "000985.CSI / H00985.CSI" and report["limitations"]
    output = discovery_paths[-1]
    strategy_path = output / "baseline_volume"
    for name in ("trades", "entries", "daily", "matched_trades"):
        assert (strategy_path / f"{name}.csv").exists()
    with (strategy_path / "daily.csv").open() as stream:
        daily = list(csv.DictReader(stream))
    assert daily and "weights" not in daily[0]
    assert {"return", "timed_market_return", "cash", "nav"} <= daily[0].keys()
    saved = json.loads((strategy_path / "metrics.json").read_text())
    assert saved["metrics"]["closed_trades"] > 0
    assert saved["metrics"]["entry_count"] > 0
    assert "matched_peers" in saved["metrics"]
    assert not (strategy_path / "metrics.tmp").exists()
    with (output / "attempts.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["family"] == "baseline"
    simulator = Mock(wraps=runner.simulate_portfolio)
    monkeypatch.setattr(runner, "simulate_portfolio", simulator)
    assert runner.run_discovery(*discovery_paths, group="baseline") == report_path
    simulator.assert_not_called()
    (strategy_path / "metrics.json").unlink()
    assert runner.run_discovery(*discovery_paths, group="baseline") == report_path
    simulator.assert_called_once()


def test_selected_strategy_neighbors_limit_and_configuration_changes(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """具名选择支持原点或邻域，limit标注烟测；同名参数和信号变化均拒绝旧结果。"""
    definitions = [definition, replace(definition, name="other")]
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=definitions))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"names": ["fixture"], "neighbors": False}))
    report_path = runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    report = json.loads(report_path.read_text())
    assert report["attempts"] == 1 and report["highest_return"]["family"] == "independent"
    assert report["experiment"]["selection"] == {"names": ["fixture"], "neighbors": False}
    for modified in (
        replace(definition, trailing_stop=0.06),
        replace(definition, entry_mask=np.zeros_like(definition.entry_mask)),
    ):
        monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[modified]))
        with pytest.raises(ValueError, match="discovery_configuration_changed"):
            runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=definitions))
    selection.write_text(json.dumps({"names": ["fixture"], "neighbors": True}))
    neighbor_output = tmp_path / "neighbors"
    neighbor_report = runner.run_discovery(
        *discovery_paths[:3], neighbor_output, group="trend", selection=selection, limit=2
    )
    saved = json.loads(neighbor_report.read_text())
    assert saved["attempts"] == 2 and saved["complete_grid"] is False
    assert (neighbor_output / "fixture__stop_tighter" / "metrics.json").exists()
    assert not (neighbor_output / "other").exists()


def test_runner_rejects_invalid_limit_missing_dividends_and_unknown_selection(
    discovery_paths, panel, definition, monkeypatch, tmp_path
):
    """无效限制、缺失分红和未知策略选择必须失败，不留下伪完成报告。"""
    for limit in (0, -1):
        with pytest.raises(ValueError, match="discovery_limit_must_be_positive"):
            runner.run_discovery(*discovery_paths, limit=limit)
    partition = discovery_paths[2] / f"dividend-{panel.calendar[-1]}.json"
    contents = partition.read_text()
    partition.unlink()
    with pytest.raises(ValueError, match="incomplete_dividend_partitions"):
        runner.run_discovery(*discovery_paths, group="baseline")
    partition.write_text(contents)
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"names": ["missing"]}))
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    assert not discovery_paths[-1].exists()


def test_runner_rejects_changed_inputs_selection_and_unversioned_output(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """输入内容、实验组或选优来源变化不可复用输出，缺少实验契约的旧目录也拒绝。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"names": ["fixture"]}))
    runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    with pytest.raises(ValueError, match="discovery_inputs_or_selection_changed"):
        runner.run_discovery(*discovery_paths, group="sector", selection=selection)
    selection.write_text(json.dumps({"names": ["fixture"], "neighbors": False}))
    with pytest.raises(ValueError, match="discovery_inputs_or_selection_changed"):
        runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    selection.write_text(json.dumps({"names": ["fixture"]}))
    (discovery_paths[1] / "features.json").write_text("{}")
    with pytest.raises(ValueError, match="discovery_inputs_or_selection_changed"):
        runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    (discovery_paths[-1] / "experiment.json").unlink()
    with pytest.raises(ValueError, match="unversioned_discovery_output"):
        runner.run_discovery(*discovery_paths, group="trend", selection=selection)


def test_runner_empty_candidate_group_has_no_fabricated_winner(discovery_paths, monkeypatch):
    """没有候选时报告零次尝试和空最佳策略，并保留实验指纹用于审计。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[]))
    report_path = runner.run_discovery(*discovery_paths, group="trend")
    report = json.loads(report_path.read_text())
    assert report["complete_grid"] and report["attempts"] == 0
    assert report["highest_return"] is report["highest_return_meeting_targets"] is None
    assert (discovery_paths[-1] / "experiment.json").exists()
    assert (discovery_paths[-1] / "attempts.csv").read_text() == ""


def test_cli_passes_prepare_and_run_paths_and_selection_options(tmp_path, monkeypatch):
    """CLI逐项传递本地路径、研究组、选优文件和限制，并打印实际生成的产物路径。"""
    source, panel_path, features, raw, output, selection = (
        tmp_path / name for name in ("source", "panel", "features", "raw", "output", "select.json")
    )
    prepare = Mock(return_value=output / "features.json")
    run = Mock(return_value=output / "report.json")
    monkeypatch.setattr(cli, "prepare_features", prepare)
    monkeypatch.setattr(cli, "run_discovery", run)
    command = CliRunner()
    prepared = command.invoke(
        cli.discovery_app, ["prepare", str(source), str(panel_path), str(output)]
    )
    assert prepared.exit_code == 0, prepared.output
    assert prepared.output.strip() == str(output / "features.json")
    prepare.assert_called_once_with(source, panel_path, output)
    arguments = ["run", str(panel_path), str(features), str(raw), str(output)]
    default = command.invoke(cli.discovery_app, arguments)
    assert default.exit_code == 0, default.output
    run.assert_called_once_with(
        panel_path, features, raw, output, group="all", selection=None, limit=None
    )
    run.reset_mock()
    selected = command.invoke(
        cli.discovery_app,
        arguments + ["--group", "trend", "--selection", str(selection), "--limit", "2"],
    )
    assert selected.exit_code == 0, selected.output
    assert selected.output.strip() == str(output / "report.json")
    run.assert_called_once_with(
        panel_path, features, raw, output, group="trend", selection=selection, limit=2
    )
    combine = Mock(return_value=output / "metrics.json")
    monkeypatch.setattr(cli, "run_combination", combine)
    combined = command.invoke(
        cli.discovery_app,
        ["combine", str(panel_path), str(features), str(raw), str(output), str(selection)],
    )
    assert combined.exit_code == 0, combined.output
    assert combined.output.strip() == str(output / "metrics.json")
    combine.assert_called_once_with(panel_path, features, raw, output, selection)


def test_control_candidates_preserve_signal_ranking_and_baseline_cash(
    discovery_paths, panel, tmp_path
):
    """control复用原信号与评分，默认收盘执行的每日资金和盈亏指标与旧基线一致。"""
    control = registry.control_candidates(panel, {})[0]
    assert control.name == "control_volume" and control.family == "baseline_volume"
    assert control.entry_mask is panel.signals["volume_reversal__sector_group"]
    assert not control.intraday_exits and control.exit_mask is None
    np.testing.assert_allclose(control.scores, 0.8 + 0.8 - 30 / 200)
    panel.values["sector_heat"][0, 0] = np.nan
    panel.values["sector_breadth"][0, 0] = np.nan
    panel.values["kdj_j"][0, 0] = np.nan
    assert registry.control_candidates(panel, {})[0].scores[0, 0] == -0.5
    runner.run_discovery(*discovery_paths, group="baseline")
    control_output = tmp_path / "control"
    runner.run_discovery(*discovery_paths[:3], control_output, group="control")
    baseline_path = discovery_paths[-1] / "baseline_volume"
    control_path = control_output / "control_volume"
    assert (control_path / "daily.csv").read_bytes() == (baseline_path / "daily.csv").read_bytes()
    baseline_metrics = json.loads((baseline_path / "metrics.json").read_text())["metrics"]
    control_metrics = json.loads((control_path / "metrics.json").read_text())["metrics"]
    assert control_metrics == baseline_metrics


def test_registry_resolves_initial_neighbor_structural_and_intraday_names(panel, definition):
    """初始名、已声明邻域、结构名和盘中后缀恢复对应定义，未知名及伪造后缀拒绝。"""
    reentry = replace(definition, name="reentry", family="boll_reentry")
    features = panel.values | {"boll_position": np.full_like(definition.scores, 0.7)}
    definitions = [definition, reentry]
    assert registry.resolve_definition(definitions, "fixture", features) is definition
    neighbor = registry.resolve_definition(definitions, "fixture__stop_tighter", features)
    assert neighbor.policy.stop_loss == definition.policy.stop_loss * 0.75
    structural = registry.resolve_definition(definitions, "reentry__struct_signal_medium", features)
    assert structural.exit_mask.all()
    intraday = registry.resolve_definition(
        definitions, "reentry__struct_signal_medium__intraday", features
    )
    assert intraday.intraday_exits and intraday.name == intraday.policy.name
    np.testing.assert_array_equal(intraday.exit_mask, structural.exit_mask)
    assert registry.resolve_definition(definitions, "fixture__intraday", features).intraday_exits
    for name in ("missing", "fixture_extra", "fixture__unknown", "fixture__intraday__intraday"):
        with pytest.raises(ValueError, match="unknown_selected_strategy"):
            registry.resolve_definition(definitions, name, features)


def test_selected_structural_and_intraday_variants_record_execution_contract(
    discovery_paths, panel, definition, monkeypatch, tmp_path
):
    """结构选择生成五个明确变体，重复名称只执行一次；盘中选项写入配置和执行限制。"""
    reentry = replace(definition, name="reentry", family="boll_reentry")
    features = panel.values | {"boll_position": np.full_like(definition.scores, 0.7)}
    monkeypatch.setattr(runner, "load_features", Mock(return_value=features))
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[reentry]))
    selection = tmp_path / "structural.json"
    selection.write_text(json.dumps({"names": ["reentry", "reentry"], "structural": True}))
    report_path = runner.run_discovery(*discovery_paths, group="reversion", selection=selection)
    assert json.loads(report_path.read_text())["attempts"] == 5
    selected_name = "reentry__struct_signal_off__intraday"
    selection.write_text(json.dumps({"names": [selected_name]}))
    output = tmp_path / "intraday"
    runner.run_discovery(*discovery_paths[:3], output, group="reversion", selection=selection)
    saved = json.loads((output / selected_name / "metrics.json").read_text())
    assert saved["configuration"]["intraday_exits"] is True
    assert saved["configuration"]["exit_mask"] is None
    assert saved["metrics"]["execution_model"]["fixed_exits"] == "daily_ohlc_t_plus_one"


def test_combination_recalculates_half_capital_cash_csv_and_resumes(
    discovery_paths, panel, definition, monkeypatch, tmp_path
):
    """各半资金账户重算整手及费用，CSV真实合并两个账本；完成后恢复且半成品可重跑。"""
    definitions = [definition, replace(definition, name="other")]
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=definitions))
    selection = tmp_path / "combination.json"
    selection.write_text(
        json.dumps({"name": "halves", "names": ["fixture", "other"], "weights": [0.5, 0.5]})
    )
    simulator = Mock(wraps=combination.simulate_strategy)
    monkeypatch.setattr(combination, "simulate_strategy", simulator)
    completed = runner.run_combination(*discovery_paths, selection)
    assert completed == discovery_paths[-1] / "metrics.json"
    assert [call.kwargs["capital"] for call in simulator.call_args_list] == [500000, 500000]
    states = runner.classify_regimes(
        *(panel.market[key] for key in ("index", "breadth", "advancing", "amount", "limit_down"))
    )
    half = runner.simulate_strategy(panel, states, [], definition, capital=500000)
    full = runner.simulate_strategy(panel, states, [], definition)
    with (discovery_paths[-1] / "daily.csv").open() as stream:
        daily = list(csv.DictReader(stream))
    np.testing.assert_allclose(
        [float(row["nav"]) for row in daily], [row["nav"] * 2 for row in half["daily"]]
    )
    assert float(daily[-1]["nav"]) != pytest.approx(full["daily"][-1]["nav"])
    with (discovery_paths[-1] / "trades.csv").open() as stream:
        trades = list(csv.DictReader(stream))
    assert {trade["sleeve"] for trade in trades} == {"0:fixture", "1:other"}
    assert len(trades) == 2 * len(half["trades"])
    by_execution = {
        tuple(int(trade[key]) for key in ("entry", "exit", "security")): int(trade["shares"])
        for trade in trades
        if trade["sleeve"] == "0:fixture"
    }
    assert by_execution == {
        tuple(trade[key] for key in ("entry", "exit", "security")): trade["shares"]
        for trade in half["trades"]
    }
    assert half["trades"][0]["shares"] * 2 != full["trades"][0]["shares"]
    saved = json.loads(completed.read_text())
    assert saved["configuration"]["weights"] == [0.5, 0.5]
    assert saved["metrics"]["execution_model"] == [False, False]
    assert saved["metrics"]["fees"] == pytest.approx(half["fees"] * 2)
    assert abs(saved["metrics"]["reconciliation_error"]) < 1e-6
    assert (discovery_paths[-1] / "entries.csv").exists()
    assert (discovery_paths[-1] / "matched_trades.csv").exists()
    simulator.reset_mock()
    assert runner.run_combination(*discovery_paths, selection) == completed
    simulator.assert_not_called()
    completed.unlink()
    assert runner.run_combination(*discovery_paths, selection) == completed
    assert simulator.call_count == 2


def test_combination_rejects_changed_contract_corrupt_metrics_and_unversioned_output(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """组合输入或权重变化、指标中的配置损坏和没有实验版本的结果均拒绝恢复。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    selection = tmp_path / "combination.json"
    selected = {"name": "single", "names": ["fixture"], "weights": [1.0]}
    selection.write_text(json.dumps(selected))
    completed = runner.run_combination(*discovery_paths, selection)
    selection.write_text(json.dumps(selected | {"weights": [0.5]}))
    with pytest.raises(ValueError, match="combination_inputs_changed_use_new_output"):
        runner.run_combination(*discovery_paths, selection)
    selection.write_text(json.dumps(selected))
    (discovery_paths[1] / "features.json").write_text("{}")
    with pytest.raises(ValueError, match="combination_inputs_changed_use_new_output"):
        runner.run_combination(*discovery_paths, selection)
    (discovery_paths[1] / "features.json").unlink()
    saved = json.loads(completed.read_text())
    saved["configuration"]["name"] = "corrupt"
    completed.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="combination_configuration_changed"):
        runner.run_combination(*discovery_paths, selection)
    (discovery_paths[-1] / "experiment.json").unlink()
    with pytest.raises(ValueError, match="unversioned_combination_output"):
        runner.run_combination(*discovery_paths, selection)


def test_combination_rejects_missing_dividends_unknown_names_and_invalid_weights(
    discovery_paths, panel, definition, monkeypatch, tmp_path
):
    """组合也要求完整分红分区，未知名称或非法权重不能写出完成指标。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    selection = tmp_path / "combination.json"
    selected = {"name": "single", "names": ["fixture"], "weights": [1.0]}
    selection.write_text(json.dumps(selected))
    partition = discovery_paths[2] / f"dividend-{panel.calendar[-1]}.json"
    contents = partition.read_text()
    partition.unlink()
    with pytest.raises(ValueError, match="incomplete_dividend_partitions"):
        runner.run_combination(*discovery_paths, selection)
    partition.write_text(contents)
    selection.write_text(json.dumps(selected | {"names": ["missing"]}))
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        runner.run_combination(*discovery_paths, selection)
    for weights in ([0.5], [-1.0], [], [1.0, 1.0]):
        selection.write_text(json.dumps(selected | {"weights": weights}))
        output = tmp_path / f"invalid-{weights}"
        with pytest.raises(ValueError, match="invalid_sleeve_configuration|sleeve_weights"):
            runner.run_combination(*discovery_paths[:3], output, selection)
        assert not (output / "metrics.json").exists()


def test_runner_rejects_ambiguous_refinement_flags(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """三个展开标志任意两项或全部启用都拒绝，不创建实验输出或调用策略执行器。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    simulator = Mock()
    monkeypatch.setattr(runner, "simulate_strategy", simulator)
    selection = tmp_path / "ambiguous.json"
    for enabled in (
        ("structural", "defensive_exits"),
        ("structural", "neighbors"),
        ("defensive_exits", "neighbors"),
        ("structural", "defensive_exits", "neighbors"),
    ):
        selection.write_text(
            json.dumps({"names": [definition.name]} | dict.fromkeys(enabled, True))
        )
        with pytest.raises(ValueError, match="ambiguous_discovery_refinement"):
            runner.run_discovery(*discovery_paths, group="defensive", selection=selection)
    simulator.assert_not_called()
    assert not discovery_paths[-1].exists()


def test_registry_resolves_defensive_exit_and_intraday_variants(definition):
    """防御退出网格名称恢复指定止盈止损及期限，盘中后缀保留参数，未声明档位拒绝。"""
    defensive_definition = replace(definition, name="defensive", family="defensive_rotation")
    selected_name = "defensive__defexit_tp0.02_sl0.08_h20"
    selected = registry.resolve_definition([defensive_definition], selected_name, {})
    assert selected.name == selected.policy.name == selected_name
    assert (
        selected.policy.take_profit,
        selected.policy.stop_loss,
        selected.policy.maximum_holding,
    ) == (0.02, 0.08, 20)
    assert selected.entry_mask is defensive_definition.entry_mask
    assert selected.scores is defensive_definition.scores
    intraday = registry.resolve_definition([defensive_definition], selected_name + "__intraday", {})
    assert intraday.intraday_exits and intraday.name == intraday.policy.name
    assert replace(intraday.policy, name=selected_name) == selected.policy
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        registry.resolve_definition(
            [defensive_definition], "defensive__defexit_tp0.03_sl0.08_h20", {}
        )


def test_runner_expands_twelve_defensive_exits_and_saves_selected_policy(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """仅启用defensive_exits时执行十二组退出参数，保存可恢复的配置与完整尝试表。"""
    defensive_definition = replace(definition, name="defensive", family="defensive_rotation")
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[defensive_definition]))
    selection = tmp_path / "defensive-exits.json"
    selection_data = {
        "names": ["defensive"],
        "defensive_exits": True,
        "structural": False,
        "neighbors": False,
    }
    selection.write_text(json.dumps(selection_data))
    report_path = runner.run_discovery(*discovery_paths, group="defensive", selection=selection)
    report = json.loads(report_path.read_text())
    assert report["attempts"] == 12 and report["complete_grid"]
    assert report["experiment"]["selection"] == selection_data
    selected_name = "defensive__defexit_tp0.02_sl0.08_h20"
    completed = discovery_paths[-1] / selected_name / "metrics.json"
    saved = json.loads(completed.read_text())
    assert saved["configuration"]["policy"]["take_profit"] == 0.02
    assert saved["configuration"]["policy"]["stop_loss"] == 0.08
    assert saved["configuration"]["policy"]["maximum_holding"] == 20
    assert saved["configuration"]["intraday_exits"] is False
    with (discovery_paths[-1] / "attempts.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    assert len(attempts) == len({row["name"] for row in attempts}) == 12
    assert {row["family"] for row in attempts} == {"defensive_rotation"}
