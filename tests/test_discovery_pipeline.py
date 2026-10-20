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
    breakout_retest,
    cli,
    combination,
    defensive,
    hierarchy,
    hierarchy_signals,
    limit_events,
    moving_support,
    onset_management,
    onset_protection,
    onset_quality,
    onset_structure,
    registry,
    relay_signals,
    reversion,
    runner,
    sector,
    sector_onset,
    selection_timing,
    trend,
)
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.profit_protection import ProfitProtection


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
    """各组只调用对应构建器，all汇总十六组；未知组和重名策略明确拒绝。"""
    builders = {}
    for name, module in (
        ("trend", trend),
        ("sector", sector),
        ("reversion", reversion),
        ("defensive", defensive),
        ("selection_timing", selection_timing),
        ("sector_onset", sector_onset),
        ("onset_quality", onset_quality),
        ("onset_management", onset_management),
        ("onset_structure", onset_structure),
        ("onset_protection", onset_protection),
        ("hierarchy_signals", hierarchy_signals),
        ("relay_signals", relay_signals),
        ("breakout_retest", breakout_retest),
        ("limit_events", limit_events),
        ("moving_support", moving_support),
    ):
        builders[name] = Mock(return_value=[replace(definition, name=name)])
        monkeypatch.setattr(module, "build_strategies", builders[name])
    builders["control"] = Mock(return_value=[replace(definition, name="control")])
    monkeypatch.setattr(registry, "control_candidates", builders["control"])
    features = {
        "fixture": panel.values["close"],
        "sector_level2_index": panel.values["sector_index"],
    }
    assert [item.name for item in registry.build_candidates(panel, features, "all")] == [
        "trend",
        "sector",
        "reversion",
        "defensive",
        "selection_timing",
        "sector_onset",
        "onset_quality",
        "onset_management",
        "onset_structure",
        "onset_protection",
        "hierarchy_signals",
        "relay_signals",
        "breakout_retest",
        "limit_events",
        "moving_support",
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


def test_registry_all_preserves_original_sixty_three_and_adds_new_groups(panel):
    """保留63/96/121项旧网格，新增结构9项和保护8项，无L2时157项、有L2时161项初始配置唯一。"""
    names = {
        name
        for module in (
            trend,
            sector,
            reversion,
            defensive,
            selection_timing,
            sector_onset,
            onset_quality,
            onset_management,
            onset_structure,
            onset_protection,
            relay_signals,
            breakout_retest,
            limit_events,
            moving_support,
        )
        for name in module.REQUIRED_FEATURES
    }
    features = {
        name: panel.values[name] if name in panel.values else np.ones_like(panel.values["close"])
        for name in names
    }
    candidates = registry.build_candidates(panel, features, "all")
    original = [
        candidate
        for group in ("trend", "sector", "reversion", "defensive", "control")
        for candidate in registry.build_candidates(panel, features, group)
    ]
    selected = registry.build_candidates(panel, features, "selection_timing")
    onset = registry.build_candidates(panel, features, "sector_onset")
    quality = registry.build_candidates(panel, features, "onset_quality")
    management = registry.build_candidates(panel, features, "onset_management")
    structure = registry.build_candidates(panel, features, "onset_structure")
    protection = registry.build_candidates(panel, features, "onset_protection")
    relay = registry.build_candidates(panel, features, "relay_signals")
    breakout = registry.build_candidates(panel, features, "breakout_retest")
    assert len(relay) == 4 and len(breakout) == 3
    assert len(original) == 63
    assert len(selected) == 17
    assert len(onset) == 16
    assert len(original) + len(selected) + len(onset) == 96
    assert len(quality) == 13 and len(management) == 12
    assert len(original) + len(selected) + len(onset) + len(quality) + len(management) == 121
    assert len(structure) == 9 and len(protection) == 8
    assert (
        len(candidates)
        == len({candidate.name for candidate in candidates})
        == 121 + 9 + 8 + 4 + 3 + 12
    )
    assert {candidate.name for candidate in candidates} == {
        candidate.name
        for candidate in (
            *original,
            *selected,
            *onset,
            *quality,
            *management,
            *structure,
            *protection,
            *relay,
            *breakout,
            *registry.build_candidates(panel, features, "limit_events"),
            *registry.build_candidates(panel, features, "moving_support"),
        )
    }
    assert sum(candidate.family == "defensive_rotation" for candidate in candidates) == 6
    assert candidates[-1].name == "control_volume"
    with pytest.raises(ValueError, match="missing"):
        registry.build_candidates(panel, features, "hierarchy_signals")
    with_level2 = features | {"sector_level2_index": panel.values["sector_index"].copy()}
    all_level2 = registry.build_candidates(panel, with_level2, "all")
    level2 = registry.build_candidates(panel, with_level2, "hierarchy_signals")
    assert len(level2) == 4 and len(all_level2) == 161
    assert {item.name for item in all_level2} == {item.name for item in candidates + level2}


def test_configuration_signal_hashes_and_legacy_baseline(definition):
    """配置保留执行参数且仅存信号摘要，数组布局不改变摘要，信号内容变化必须改变摘要。"""
    baseline = runner.configuration(None)
    assert baseline["baseline"] is True
    assert baseline["policy"] == asdict(runner.baseline_policy())
    assert baseline["policy"]["force_rebalance"] is True
    assert baseline["policy"]["switching"] is baseline["policy"]["stage_allocation"] is False
    signature = runner.configuration(definition)
    assert signature["exit_mask"] is None
    assert signature["entry_invalidation_mask"] is None
    assert signature["profit_protection"] is None
    protection = ProfitProtection(0.03, minimum_return=0.01)
    protected = runner.configuration(replace(definition, profit_protection=protection))
    assert protected == signature | {"profit_protection": asdict(protection)}
    changed_protection = ProfitProtection(0.05, trailing_drawdown=0.02)
    assert runner.configuration(replace(definition, profit_protection=changed_protection)) == (
        signature | {"profit_protection": asdict(changed_protection)}
    )
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
    invalidation = np.zeros_like(definition.entry_mask)
    separate = runner.configuration(replace(definition, entry_invalidation_mask=invalidation))
    assert separate["entry_invalidation_mask"] is not None
    assert {key: value for key, value in separate.items() if key != "entry_invalidation_mask"} == {
        key: value for key, value in signature.items() if key != "entry_invalidation_mask"
    }
    assert (
        runner.configuration(
            replace(definition, entry_invalidation_mask=np.asfortranarray(invalidation))
        )
        == separate
    )
    invalidation[0, 0] = True
    changed_invalidation = runner.configuration(
        replace(definition, entry_invalidation_mask=invalidation)
    )
    assert changed_invalidation["entry_invalidation_mask"] != separate["entry_invalidation_mask"]
    assert changed_invalidation["exit_mask"] is None


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


def test_runner_records_independent_entry_invalidation_and_rejects_changed_content(
    discovery_paths, definition, monkeypatch
):
    """旧exit兼管候选时不入场；独立全False否决允许入场但仍退出，单bit修改拒绝恢复。"""
    exit_mask = np.ones_like(definition.entry_mask)
    invalidation = np.zeros_like(definition.entry_mask)
    legacy = replace(definition, name="legacy_exit", exit_mask=exit_mask)
    separated = replace(
        definition,
        name="separate_exit",
        exit_mask=exit_mask,
        entry_invalidation_mask=invalidation,
    )
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[legacy, separated]))
    runner.run_discovery(*discovery_paths, group="onset_management")
    output = discovery_paths[-1]
    previous = json.loads((output / "legacy_exit" / "metrics.json").read_text())
    saved = json.loads((output / "separate_exit" / "metrics.json").read_text())
    assert previous["metrics"]["entry_count"] == 0
    assert previous["configuration"]["entry_invalidation_mask"] is None
    assert saved["metrics"]["entry_count"] > 0 and saved["metrics"]["closed_trades"] > 0
    assert saved["configuration"]["exit_mask"] == previous["configuration"]["exit_mask"]
    assert saved["configuration"]["entry_invalidation_mask"] != saved["configuration"]["exit_mask"]
    with (output / "separate_exit" / "trades.csv").open() as stream:
        trades = list(csv.DictReader(stream))
    assert trades and all(trade["reason"] == "signal" for trade in trades)
    invalidation[0, 0] = True
    with pytest.raises(ValueError, match="discovery_configuration_changed"):
        runner.run_discovery(*discovery_paths, group="onset_management")


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
        replace(definition, entry_invalidation_mask=np.zeros_like(definition.entry_mask)),
        replace(definition, profit_protection=ProfitProtection(0.03, minimum_return=0.01)),
    ):
        monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[modified]))
        with pytest.raises(ValueError, match="discovery_configuration_changed"):
            runner.run_discovery(*discovery_paths, group="trend", selection=selection)
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=definitions))
    selection.write_text(json.dumps({"names": ["fixture"], "neighbors": True}))
    neighbor_output = tmp_path / "neighbors"
    neighbor_report = runner.run_discovery(
        *discovery_paths[:3],
        neighbor_output,
        group="trend",
        selection=selection,
        limit=2,
        hierarchy_path=None,
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
        panel_path,
        features,
        raw,
        output,
        group="all",
        selection=None,
        limit=None,
        hierarchy_path=None,
    )
    run.reset_mock()
    selected = command.invoke(
        cli.discovery_app,
        arguments + ["--group", "trend", "--selection", str(selection), "--limit", "2"],
    )
    assert selected.exit_code == 0, selected.output
    assert selected.output.strip() == str(output / "report.json")
    run.assert_called_once_with(
        panel_path,
        features,
        raw,
        output,
        group="trend",
        selection=selection,
        limit=2,
        hierarchy_path=None,
    )
    for group in (
        "selection_timing",
        "sector_onset",
        "onset_quality",
        "onset_management",
        "onset_structure",
        "onset_protection",
    ):
        run.reset_mock()
        result = command.invoke(cli.discovery_app, arguments + ["--group", group])
        assert result.exit_code == 0, result.output
        run.assert_called_once_with(
            panel_path,
            features,
            raw,
            output,
            group=group,
            selection=None,
            limit=None,
            hierarchy_path=None,
        )
    combine = Mock(return_value=output / "metrics.json")
    monkeypatch.setattr(cli, "run_combination", combine)
    combined = command.invoke(
        cli.discovery_app,
        ["combine", str(panel_path), str(features), str(raw), str(output), str(selection)],
    )
    assert combined.exit_code == 0, combined.output
    assert combined.output.strip() == str(output / "metrics.json")
    combine.assert_called_once_with(
        panel_path, features, raw, output, selection, hierarchy_path=None
    )


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


def test_registry_lazily_resolves_management_crosses_and_rejects_unknown_labels(
    panel, definition, monkeypatch
):
    """普通及未知名称不构建管理矩阵，合法质量管理交叉保留原否决并支持盘中后缀。"""
    original_exit = np.zeros_like(definition.entry_mask)
    original_exit[10] = True
    base = replace(
        definition,
        name="onset_quality_market_breadth_50",
        family="sector_breadth_onset",
        exit_mask=original_exit,
    )
    builder = Mock(wraps=onset_management.build_variants)
    monkeypatch.setattr(onset_management, "build_variants", builder)
    for name in (base.name, base.name + "__intraday", base.name + "__stop_tighter"):
        assert registry.resolve_definition([base], name, {}).name == name
    for name in (
        base.name + "__manage_unknown",
        base.name + "_manage_momentum20_only",
        base.name + "__manage_momentum20_only_extra",
        base.name + "__manage_unknown__intraday",
    ):
        with pytest.raises(ValueError, match="unknown_selected_strategy"):
            registry.resolve_definition([base], name, {})
    builder.assert_not_called()
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        registry.resolve_definition([definition], definition.name + "__manage_momentum20_only", {})
    builder.assert_not_called()
    crossed_name = base.name + "__manage_momentum20_only"
    selected = registry.resolve_definition([base], crossed_name, panel.values)
    builder.assert_called_once_with(base, panel.values)
    assert selected.name == selected.policy.name == crossed_name
    assert selected.entry_invalidation_mask is original_exit
    np.testing.assert_array_equal(selected.entry_mask, base.entry_mask & ~original_exit)
    assert not selected.exit_mask.any()
    signature = runner.configuration(selected)
    assert signature["exit_mask"] != signature["entry_invalidation_mask"]
    intraday = registry.resolve_definition([base], crossed_name + "__intraday", panel.values)
    assert intraday.intraday_exits and intraday.name == intraday.policy.name
    assert intraday.entry_invalidation_mask is original_exit
    actual_labels = {
        variant.name.removeprefix(base.name + "__manage_")
        for variant in onset_management.build_variants(base, panel.values)
    }
    assert actual_labels == registry._MANAGEMENT_LABELS


def test_registry_lazily_resolves_protection_crosses_and_preserves_disabled_control(
    definition, monkeypatch
):
    """关闭保护的原名不构建变体；八种合法保护按需解析，未知标签拒绝并保留盘中后缀。"""
    original_exit = np.zeros_like(definition.entry_mask)
    invalidation = np.ones_like(original_exit)
    base = replace(
        definition,
        name="onset_structure_control",
        family="sector_breadth_onset",
        exit_mask=original_exit,
        entry_invalidation_mask=invalidation,
    )
    builder = Mock(wraps=onset_protection.build_variants)
    monkeypatch.setattr(onset_protection, "build_variants", builder)
    for name in (base.name, base.name + "__intraday", base.name + "__stop_tighter"):
        assert registry.resolve_definition([base], name, {}).profit_protection is None
    for name in (
        base.name + "__protect_unknown",
        base.name + "_protect_floor_arm3_lock1",
        base.name + "__protect_floor_arm3_lock1_extra",
        base.name + "__protect_unknown__intraday",
    ):
        with pytest.raises(ValueError, match="unknown_selected_strategy"):
            registry.resolve_definition([base], name, {})
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        registry.resolve_definition(
            [definition], definition.name + "__protect_floor_arm3_lock1", {}
        )
    builder.assert_not_called()
    for label in registry._PROTECTION_LABELS:
        name = base.name + "__protect_" + label
        selected = registry.resolve_definition([base], name, {})
        assert selected.name == selected.policy.name == name
        assert selected.profit_protection is not None
        assert selected.entry_mask is base.entry_mask and selected.scores is base.scores
        assert (
            selected.exit_mask is original_exit and selected.entry_invalidation_mask is invalidation
        )
    assert builder.call_count == 8
    selected = registry.resolve_definition(
        [base], base.name + "__protect_floor_arm3_lock1__intraday", {}
    )
    assert selected.intraday_exits and selected.profit_protection == ProfitProtection(
        0.03, minimum_return=0.01
    )
    assert {
        item.name.removeprefix(base.name + "__protect_")
        for item in onset_protection.build_variants(base)
    } == registry._PROTECTION_LABELS


def test_profit_protection_none_control_and_changed_parameters_reject_resume(
    discovery_paths, definition, monkeypatch
):
    """无保护与显式None的资金结果一致；同名策略的激活或保底参数变化不能恢复旧输出。"""
    disabled = replace(definition, name="disabled", profit_protection=None)
    protected = replace(
        definition, name="protected", profit_protection=ProfitProtection(0.03, minimum_return=0.01)
    )
    definitions = [definition, disabled, protected]
    builder = Mock(return_value=definitions)
    monkeypatch.setattr(runner, "build_candidates", builder)
    runner.run_discovery(*discovery_paths, group="onset_protection")
    output = discovery_paths[-1]
    baseline = json.loads((output / definition.name / "metrics.json").read_text())
    control = json.loads((output / disabled.name / "metrics.json").read_text())
    assert control["metrics"] == baseline["metrics"]
    assert control["configuration"]["profit_protection"] is None
    assert (output / disabled.name / "daily.csv").read_bytes() == (
        output / definition.name / "daily.csv"
    ).read_bytes()
    saved = json.loads((output / protected.name / "metrics.json").read_text())
    assert saved["configuration"]["profit_protection"] == asdict(protected.profit_protection)
    for protection in (
        ProfitProtection(0.05, minimum_return=0.01),
        ProfitProtection(0.03, minimum_return=0),
        None,
    ):
        builder.return_value = [replace(protected, profit_protection=protection)]
        with pytest.raises(ValueError, match="discovery_configuration_changed"):
            runner.run_discovery(*discovery_paths, group="onset_protection")


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


def test_runner_rejects_onset_exits_combined_with_other_refinement_flags(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """onset_exits与任意非空其他展开标志组合均拒绝，原三标志的互斥场景独立保留。"""
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[definition]))
    simulator = Mock()
    monkeypatch.setattr(runner, "simulate_strategy", simulator)
    selection = tmp_path / "ambiguous-onset.json"
    for enabled in (
        ("structural",),
        ("defensive_exits",),
        ("neighbors",),
        ("structural", "defensive_exits"),
        ("structural", "neighbors"),
        ("defensive_exits", "neighbors"),
        ("structural", "defensive_exits", "neighbors"),
    ):
        selection.write_text(
            json.dumps(
                {"names": [definition.name], "onset_exits": True} | dict.fromkeys(enabled, True)
            )
        )
        with pytest.raises(ValueError, match="ambiguous_discovery_refinement"):
            runner.run_discovery(*discovery_paths, group="sector_onset", selection=selection)
    simulator.assert_not_called()
    assert not discovery_paths[-1].exists()


def test_registry_resolves_eight_onset_exit_variants_and_switches_intraday_mode(definition):
    """八个具名退出变体恢复四档止盈，close关闭盘中开关，intraday打开且保留结构退出。"""
    exit_mask = np.zeros_like(definition.entry_mask)
    exit_mask[20] = True
    onset = replace(
        definition,
        name="onset",
        family="sector_breadth_onset",
        exit_mask=exit_mask,
        policy=replace(definition.policy, stop_loss=0.08, maximum_holding=30),
        intraday_exits=True,
    )
    names = set()
    for take_profit in (0.01, 0.02, 0.04, 0.08):
        for mode in ("close", "intraday"):
            name = f"onset__onsetexit_tp{take_profit:g}_{mode}"
            selected = registry.resolve_definition([onset], name, {})
            names.add(selected.name)
            assert selected.name == selected.policy.name == name
            assert selected.policy.take_profit == take_profit
            assert selected.policy.stop_loss == 0.08 and selected.policy.maximum_holding == 30
            assert selected.intraday_exits is (mode == "intraday")
            assert selected.entry_mask is onset.entry_mask
            assert selected.scores is onset.scores and selected.exit_mask is exit_mask
    assert len(names) == 8
    assert registry.resolve_definition(
        [onset], "onset__onsetexit_tp0.01_close__intraday", {}
    ).intraday_exits
    with pytest.raises(ValueError, match="unknown_selected_strategy"):
        registry.resolve_definition([onset], "onset__onsetexit_tp0.03_close", {})


def test_runner_expands_eight_onset_exits_and_saves_execution_metadata(
    discovery_paths, definition, monkeypatch, tmp_path
):
    """八组真实合成账本保存结构退出及SL8/H30，盘中模式实际更早止盈并记录执行限制。"""
    entry_mask = np.zeros_like(definition.entry_mask)
    entry_mask[0, 0] = True
    exit_mask = np.zeros_like(entry_mask)
    exit_mask[20, 0] = True
    onset = replace(
        definition,
        name="onset",
        family="sector_breadth_onset",
        entry_mask=entry_mask,
        exit_mask=exit_mask,
        policy=replace(definition.policy, take_profit=0.08, stop_loss=0.08, maximum_holding=30),
    )
    monkeypatch.setattr(runner, "build_candidates", Mock(return_value=[onset]))
    simulator = Mock(wraps=runner.simulate_strategy)
    monkeypatch.setattr(runner, "simulate_strategy", simulator)
    selection = tmp_path / "onset-exits.json"
    selection_data = {"names": ["onset"], "onset_exits": True}
    selection.write_text(json.dumps(selection_data))
    report_path = runner.run_discovery(*discovery_paths, group="sector_onset", selection=selection)
    report = json.loads(report_path.read_text())
    assert report["attempts"] == 8 and report["complete_grid"]
    assert report["experiment"]["selection"] == selection_data
    assert simulator.call_count == 8
    original_signature = runner.configuration(onset)
    first_exits = {}
    for call in simulator.call_args_list:
        selected = call.args[3]
        assert selected.exit_mask is exit_mask and selected.entry_mask is entry_mask
        strategy_path = discovery_paths[-1] / selected.name
        saved = json.loads((strategy_path / "metrics.json").read_text())
        configuration = saved["configuration"]
        assert configuration["entry_mask"] == original_signature["entry_mask"]
        assert configuration["scores"] == original_signature["scores"]
        assert configuration["exit_mask"] == original_signature["exit_mask"]
        assert configuration["policy"]["stop_loss"] == 0.08
        assert configuration["policy"]["maximum_holding"] == 30
        assert configuration["intraday_exits"] == selected.intraday_exits
        if selected.intraday_exits:
            assert saved["metrics"]["execution_model"]["fixed_exits"] == "daily_ohlc_t_plus_one"
        else:
            assert saved["metrics"]["execution_model"] == "close_next_open"
        with (strategy_path / "trades.csv").open() as stream:
            trades = list(csv.DictReader(stream))
        assert len(trades) == 1
        first_exits[selected.name] = int(trades[0]["exit"])
    assert (
        first_exits["onset__onsetexit_tp0.01_intraday"]
        < first_exits["onset__onsetexit_tp0.01_close"]
    )
    with (discovery_paths[-1] / "attempts.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    assert len(attempts) == len({row["name"] for row in attempts}) == 8


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


@pytest.mark.parametrize(
    ("group", "selected_name", "expected_confirmation"),
    [
        ("selection_timing", "volume_timing_delay1", 31),
        ("sector_onset", "sector_onset_broad_reclaim_mid_tp4_sl10_h20", 30),
        ("onset_quality", "onset_quality_confirm_breadth_1d", 31),
        ("onset_quality", "onset_quality_market_breadth_50__manage_momentum20_only", 30),
        (
            "onset_management",
            "sector_onset_selective_quiet_risk_tp8_sl8_h30__manage_industry_off",
            30,
        ),
        ("onset_structure", "onset_structure_control", 30),
        (
            "onset_protection",
            "onset_quality_market_breadth_50__manage_momentum20_only__protect_floor_arm3_lock1",
            30,
        ),
        ("onset_structure", "onset_structure_market_breadth_50_75__protect_floor_arm3_lock1", 30),
        ("hierarchy_signals", "hierarchy_matched_control", 30),
        ("hierarchy_signals", "hierarchy_l1_peer_relative20", 30),
        ("hierarchy_signals", "hierarchy_l2_peer_relative20", 30),
        ("hierarchy_signals", "hierarchy_l2_onset", 30),
        ("relay_signals", "relay_matched_control", 30),
        ("relay_signals", "relay_previous_leaders", 30),
        ("relay_signals", "relay_new_participants", 30),
        ("relay_signals", "relay_current_strength", 30),
    ],
)
def test_new_groups_cli_runs_real_signals_with_existing_cash_and_timing_contract(
    discovery_paths, panel, monkeypatch, tmp_path, group, selected_name, expected_confirmation
):
    """新组通过CLI真实成交，延期、独立管理和激活后利润保护均沿用下一开盘及资金账本。"""
    enhanced_groups = (
        "onset_quality",
        "onset_management",
        "onset_structure",
        "onset_protection",
        "hierarchy_signals",
        "relay_signals",
    )
    daily_growth = 1.004 if group in enhanced_groups else 1.002
    price = np.broadcast_to(
        (10 * daily_growth ** np.arange(len(panel.calendar)))[:, None], panel.values["close"].shape
    ).copy()
    protected_case = "__protect_" in selected_name
    if protected_case:
        price[46:] = price[31] * 1.005
    for name in ("raw_open", "raw_close", "raw_low", "raw_high", "close"):
        panel.values[name] = price.copy()
    panel.values["raw_up_limit"] = price * 1.1
    panel.values["raw_down_limit"] = price * 0.9
    panel.values["return1"][:] = daily_growth - 1
    panel.values["return5"][:] = daily_growth**5 - 1
    panel.values["return20"][:] = daily_growth**20 - 1
    if protected_case:
        for name, period in (("return1", 1), ("return5", 5), ("return20", 20)):
            panel.values[name][period:] = price[period:] / price[:-period] - 1
    panel.values["ma20"] = price * 1.01
    for sector_identifier in (0, 1):
        columns = np.flatnonzero(panel.values["sector_index"][0] == sector_identifier)
        panel.values["ma20"][:, columns[:4]] = price[:, columns[:4]] * 0.99
        panel.values["ma20"][30:, columns[:8]] = price[30:, columns[:8]] * 0.99
        if group == "onset_quality" and "__manage_" in selected_name:
            panel.values["ma20"][40:, columns] = price[40:, columns] * 1.01
            panel.values["ma20"][40:, columns[:2]] = price[40:, columns[:2]] * 0.99
    for mask in panel.signals.values():
        mask[:] = False
    if group == "selection_timing":
        panel.signals["volume_reversal__sector_group"][30] = True
    features = panel.values | {
        "high": price.copy(),
        "turnover": np.full_like(price, 2),
        "boll_position": np.full_like(price, 0.3),
    }
    save_panel(discovery_paths[0], panel)
    monkeypatch.setattr(runner, "load_features", Mock(return_value=features))
    builder = Mock(wraps=registry.build_candidates)
    monkeypatch.setattr(runner, "build_candidates", builder)
    simulator = Mock(wraps=runner.simulate_strategy)
    monkeypatch.setattr(runner, "simulate_strategy", simulator)
    sidecar = None
    if group == "hierarchy_signals":
        sidecar = prepare_test_hierarchy(
            discovery_paths[0], panel, tmp_path / "members", tmp_path / "sidecar"
        )
    selection = tmp_path / "new-group-selection.json"
    selection.write_text(json.dumps({"names": [selected_name]}))
    command = CliRunner().invoke(
        cli.discovery_app,
        [
            "run",
            *(str(path) for path in discovery_paths),
            "--group",
            group,
            "--selection",
            str(selection),
            *(["--hierarchy", str(sidecar)] if sidecar is not None else []),
        ],
    )
    assert command.exit_code == 0, command.output
    assert builder.call_args.args[2] == group
    report = json.loads((discovery_paths[-1] / "report.json").read_text())
    assert report["attempts"] == 1 and report["highest_return"]["name"] == selected_name
    strategy_path = discovery_paths[-1] / selected_name
    with (strategy_path / "trades.csv").open() as stream:
        trades = list(csv.DictReader(stream))
    assert trades
    assert min(int(trade["confirmation"]) for trade in trades) == expected_confirmation
    assert min(int(trade["entry"]) for trade in trades) == expected_confirmation + 1
    assert all(int(trade["exit"]) > int(trade["entry"]) for trade in trades)
    assert all(int(trade["shares"]) > 0 and int(trade["shares"]) % 100 == 0 for trade in trades)
    with (strategy_path / "daily.csv").open() as stream:
        daily = list(csv.DictReader(stream))
    assert min(float(row["cash"]) for row in daily) >= 0
    assert max(float(row["exposure"]) for row in daily) <= 0.91
    for row in daily:
        assets, cash, receivable = (float(row[key]) for key in ("nav", "cash", "receivable"))
        assert assets == pytest.approx(cash + receivable + assets * float(row["exposure"]))
    saved = json.loads((strategy_path / "metrics.json").read_text())
    assert saved["metrics"]["fees"] > 0
    assert abs(saved["metrics"]["reconciliation_error"]) < 1e-6
    policy = saved["configuration"]["policy"]
    expected_policy = (0.08, 0.08, 30) if group in enhanced_groups else (0.04, 0.10, 20)
    assert (
        policy["take_profit"],
        policy["stop_loss"],
        policy["maximum_holding"],
    ) == expected_policy
    assert saved["configuration"]["intraday_exits"] is False
    if protected_case:
        assert saved["configuration"]["profit_protection"] == asdict(
            ProfitProtection(0.03, minimum_return=0.01)
        )
        assert {trade["reason"] for trade in trades} == {"profit_protection"}
        assert {int(trade["exit"]) for trade in trades} == {47}
    else:
        assert saved["configuration"]["profit_protection"] is None
    if group == "onset_management":
        base = next(
            item
            for item in sector_onset.build_strategies(panel, features)
            if item.name == onset_management.BASELINE_NAME
        )
        selected = simulator.call_args.args[3]
        np.testing.assert_array_equal(selected.entry_mask, base.entry_mask & ~base.exit_mask)
        np.testing.assert_array_equal(selected.scores, base.scores)
        np.testing.assert_array_equal(selected.entry_invalidation_mask, base.exit_mask)
        assert selected.exit_mask is None
        assert saved["configuration"]["exit_mask"] is None
        assert (
            saved["configuration"]["entry_invalidation_mask"]
            == runner.configuration(base)["exit_mask"]
        )
    if group == "onset_quality" and "__manage_" in selected_name:
        quality = next(
            item
            for item in onset_quality.build_strategies(panel, features)
            if item.name == "onset_quality_market_breadth_50"
        )
        selected = simulator.call_args.args[3]
        np.testing.assert_array_equal(selected.entry_mask, quality.entry_mask & ~quality.exit_mask)
        np.testing.assert_array_equal(selected.entry_invalidation_mask, quality.exit_mask)
        assert quality.exit_mask[40:].any() and not selected.exit_mask[40:].any()
        assert min(int(trade["exit"]) for trade in trades) > 40
        assert (
            saved["configuration"]["exit_mask"] != saved["configuration"]["entry_invalidation_mask"]
        )
    if (
        (group == "onset_quality" and "__manage_" in selected_name)
        or (group == "onset_structure" and protected_case)
        or group in ("hierarchy_signals", "relay_signals")
    ):
        required_names = {
            name
            for module in (
                trend,
                sector,
                reversion,
                defensive,
                selection_timing,
                sector_onset,
                onset_quality,
                onset_management,
                onset_structure,
                onset_protection,
                relay_signals,
                breakout_retest,
                limit_events,
                moving_support,
            )
            for name in module.REQUIRED_FEATURES
        }
        for name in required_names:
            features.setdefault(name, np.ones_like(price))
        combination_selection = tmp_path / "crossed-combination.json"
        combination_names = [selected_name, selected_name + "__intraday"]
        combination_selection.write_text(
            json.dumps(
                {
                    "name": "crossed_modes",
                    "names": combination_names,
                    "weights": [0.5, 0.5],
                }
            )
        )
        combination_output = tmp_path / "crossed-combination"
        completed = runner.run_combination(
            *discovery_paths[:3], combination_output, combination_selection, hierarchy_path=sidecar
        )
        combined = json.loads(completed.read_text())
        sleeves = combined["configuration"]["sleeves"]
        assert [sleeve["name"] for sleeve in sleeves] == combination_names
        assert [sleeve["intraday_exits"] for sleeve in sleeves] == [False, True]
        assert combined["metrics"]["closed_trades"] > 0
        for sleeve in sleeves:
            assert sleeve["profit_protection"] == saved["configuration"]["profit_protection"]
            assert sleeve["exit_mask"] == saved["configuration"]["exit_mask"]
            assert (
                sleeve["entry_invalidation_mask"]
                == saved["configuration"]["entry_invalidation_mask"]
            )
        with (combination_output / "trades.csv").open() as stream:
            combined_trades = list(csv.DictReader(stream))
        assert {trade["sleeve"] for trade in combined_trades} == {
            f"{index}:{name}" for index, name in enumerate(combination_names)
        }


def prepare_test_hierarchy(panel_path, panel, source, output):
    """给合成面板添加公开来源摘要并生成真实sidecar；例如输出OUT/hierarchy.json。

    每十只同业共享L2，日期覆盖完整面板；不改变面板数组或任何旧测试的输入。
    """
    manifest_path = panel_path / "panel.json"
    manifest = json.loads(manifest_path.read_text()) | {"source_fingerprint": "a" * 64}
    manifest_path.write_text(json.dumps(manifest))
    source.mkdir()
    rows = [
        {
            "ts_code": code,
            "l1_code": f"{801000 + index % 2:06d}.SI",
            "l2_code": f"{801010 + index % 2:06d}.SI",
            "in_date": panel.calendar[0],
            "out_date": None,
            "is_new": "Y",
        }
        for index, code in enumerate(panel.codes)
    ]
    write_partition(source / "members-fixture.json", "index_member_all", {}, rows)
    return hierarchy.prepare_hierarchy(source, panel_path, output)


def test_cli_hierarchy_creation_and_optional_run_combine_paths(tmp_path, monkeypatch):
    """CLI层级命令传递源目录，run/combine的显式sidecar参数准确传到runner。"""
    source, panel_path, features, raw, output, selection, sidecar = (
        tmp_path / name
        for name in ("source", "panel", "features", "raw", "output", "selection.json", "sidecar")
    )
    prepare = Mock(return_value=sidecar / "hierarchy.json")
    run, combine = (
        Mock(return_value=output / "report.json"),
        Mock(return_value=output / "metrics.json"),
    )
    monkeypatch.setattr(cli, "prepare_hierarchy", prepare)
    monkeypatch.setattr(cli, "run_discovery", run)
    monkeypatch.setattr(cli, "run_combination", combine)
    command = CliRunner()
    prepared = command.invoke(
        cli.discovery_app, ["hierarchy", str(source), str(panel_path), str(sidecar)]
    )
    assert prepared.exit_code == 0, prepared.output
    assert prepared.output.strip() == str(sidecar / "hierarchy.json")
    prepare.assert_called_once_with(source, panel_path, sidecar)
    executed = command.invoke(
        cli.discovery_app,
        [
            "run",
            str(panel_path),
            str(features),
            str(raw),
            str(output),
            "--hierarchy",
            str(sidecar),
            "--group",
            "hierarchy_signals",
        ],
    )
    assert executed.exit_code == 0, executed.output
    run.assert_called_once_with(
        panel_path,
        features,
        raw,
        output,
        group="hierarchy_signals",
        selection=None,
        limit=None,
        hierarchy_path=sidecar,
    )
    combined = command.invoke(
        cli.discovery_app,
        [
            "combine",
            str(panel_path),
            str(features),
            str(raw),
            str(output),
            str(selection),
            "--hierarchy",
            str(sidecar / "hierarchy.json"),
        ],
    )
    assert combined.exit_code == 0, combined.output
    combine.assert_called_once_with(
        panel_path, features, raw, output, selection, hierarchy_path=sidecar / "hierarchy.json"
    )


def test_fingerprint_tracks_explicit_sidecar_and_external_structure_dependencies(
    discovery_paths, panel, tmp_path, monkeypatch
):
    """sidecar两文件与两外部结构源码改变指纹，目录和清单等价，无关文件不进入摘要。"""
    panel_path, features, raw, _ = discovery_paths
    sidecar = prepare_test_hierarchy(panel_path, panel, tmp_path / "members", tmp_path / "sidecar")
    without = runner.discovery_fingerprint(panel_path, raw, features)
    initial = runner.discovery_fingerprint(panel_path, raw, features, sidecar)
    assert initial != without
    assert runner.discovery_fingerprint(panel_path, raw, features, sidecar.parent) == initial
    (sidecar.parent / "unrelated.txt").write_text("unrelated")
    assert runner.discovery_fingerprint(panel_path, raw, features, sidecar) == initial
    sidecar.write_text(sidecar.read_text() + "\n")
    changed = runner.discovery_fingerprint(panel_path, raw, features, sidecar)
    assert changed != initial
    matrix = np.load(sidecar.parent / "values-sector_level2_index.npy", mmap_mode="r+")
    matrix[0, 0] = np.nan
    matrix.flush()
    assert runner.discovery_fingerprint(panel_path, raw, features, sidecar) != changed
    assert runner.discovery_fingerprint(panel_path, raw, features) == without
    package = tmp_path / "package"
    (package / "strategy_discovery").mkdir(parents=True)
    dependency = package / "equity_research"
    dependency.mkdir()
    monkeypatch.setattr(runner, "__file__", str(package / "strategy_discovery" / "runner.py"))
    previous = runner.discovery_fingerprint(panel_path, raw, features)
    for filename in ("right_side_structure.py", "right_side_policy.py"):
        (dependency / filename).write_text("STRUCTURE_VERSION = 1\n")
        current = runner.discovery_fingerprint(panel_path, raw, features)
        assert current != previous
        (dependency / filename).write_text("STRUCTURE_VERSION = 2\n")
        previous = runner.discovery_fingerprint(panel_path, raw, features)
        assert previous != current
    (dependency / "unrelated.py").write_text("UNUSED = True\n")
    assert runner.discovery_fingerprint(panel_path, raw, features) == previous


@pytest.mark.parametrize("combined", [False, True])
def test_runner_loads_verified_hierarchy_and_rejects_sidecar_drift(
    discovery_paths, panel, definition, tmp_path, monkeypatch, combined
):
    """独立/组合runner加载只读L2；原样恢复，合法清单字节漂移拒绝混用旧结果。"""
    sidecar = prepare_test_hierarchy(
        discovery_paths[0], panel, tmp_path / "members", tmp_path / "sidecar"
    )
    builder = Mock(return_value=[definition])
    monkeypatch.setattr(runner, "build_candidates", builder)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"name": "sidecar", "names": [definition.name], "weights": [1.0]})
    )

    def invoke():
        """重放当前参数化入口；例如combined=True返回组合metrics.json路径。"""
        if combined:
            return runner.run_combination(*discovery_paths, selection, hierarchy_path=sidecar)
        return runner.run_discovery(*discovery_paths, hierarchy_path=sidecar.parent)

    completed = invoke()
    matrix = builder.call_args.args[1]["sector_level2_index"]
    assert isinstance(matrix, np.memmap) and not matrix.flags.writeable
    np.testing.assert_array_equal(matrix, panel.values["sector_index"])
    assert invoke() == completed
    sidecar.write_text(sidecar.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs.*changed_use_new_output"):
        invoke()


@pytest.mark.parametrize("combined", [False, True])
def test_runner_rejects_invalid_hierarchy_before_output_creation(
    discovery_paths, panel, tmp_path, combined
):
    """独立/组合运行先核验L2矩阵完整性，损坏数据不能创建实验输出目录。"""
    sidecar = prepare_test_hierarchy(
        discovery_paths[0], panel, tmp_path / "members", tmp_path / "sidecar"
    )
    matrix = np.load(sidecar.parent / "values-sector_level2_index.npy", mmap_mode="r+")
    matrix[0, 0] = 99
    matrix.flush()
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"name": "invalid", "names": ["fixture"], "weights": [1.0]}))
    with pytest.raises(ValueError, match="hierarchy_matrix_integrity_mismatch"):
        if combined:
            runner.run_combination(*discovery_paths, selection, hierarchy_path=sidecar)
        else:
            runner.run_discovery(*discovery_paths, hierarchy_path=sidecar)
    assert not discovery_paths[-1].exists()


@pytest.mark.parametrize(
    ("selected_name", "expected_confirmation"),
    [("breakout_direct", 60), ("first_retest", 61), ("repeat_retest", 61)],
)
def test_breakout_three_arms_cli_reaches_real_cash_ledger(
    panel, tmp_path, monkeypatch, selected_name, expected_confirmation
):
    """61日真实结构穿过CLI、登记器、执行器与审计账本；三臂按各自确认日下一开盘成交。"""
    from datetime import date, timedelta

    from test_discovery_breakout_retest import breakout_fixture

    source_panel, features = breakout_fixture(days=80)
    days = len(source_panel.calendar)
    panel.calendar = [
        (date(2024, 1, 1) + timedelta(days=day)).strftime("%Y%m%d") for day in range(days)
    ]
    panel.codes = source_panel.codes
    panel.values = {name: np.full((days, 1), values[0, 0]) for name, values in panel.values.items()}
    panel.signals = {name: np.zeros((days, 1), dtype=bool) for name in panel.signals}
    panel.market = {name: np.full(days, values[0]) for name, values in panel.market.items()}
    for name in ("index", "total_index", "index_open"):
        panel.market[name] = 100 * 1.001 ** np.arange(days)
    features["close"][70:], features["high"][70:], features["low"][70:] = 11.2, 11.3, 11.1
    features["raw_amount"] *= 1000
    features["ma20"] = features["close"] * 0.99
    features["return1"][1:] = features["close"][1:] / features["close"][:-1] - 1
    panel.values.update(features)
    for name, source in (
        ("raw_open", "close"),
        ("raw_close", "close"),
        ("raw_high", "high"),
        ("raw_low", "low"),
    ):
        panel.values[name] = features[source].copy()
    panel.values["raw_up_limit"] = features["close"] * 1.1
    panel.values["raw_down_limit"] = features["close"] * 0.9
    for name, period in (("return5", 5), ("return20", 20)):
        panel.values[name][period:] = features["close"][period:] / features["close"][:-period] - 1
    panel_path, feature_path, raw, output = (
        tmp_path / name for name in ("panel", "features", "raw", "output")
    )
    save_panel(panel_path, panel)
    feature_path.mkdir()
    raw.mkdir()
    for day in panel.calendar:
        write_partition(raw / f"dividend-{day}.json", "dividend", {"ex_date": day}, [])
    monkeypatch.setattr(runner, "load_features", Mock(return_value=features))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"names": [selected_name]}))
    command = CliRunner().invoke(
        cli.discovery_app,
        [
            "run",
            str(panel_path),
            str(feature_path),
            str(raw),
            str(output),
            "--group",
            "breakout_retest",
            "--selection",
            str(selection),
        ],
    )
    assert command.exit_code == 0, command.output
    saved = json.loads((output / selected_name / "metrics.json").read_text())
    assert saved["configuration"]["name"] == selected_name
    assert saved["configuration"]["policy"]["maximum_holding"] == 30
    assert saved["configuration"]["maximum_entry_gap"] == 0.02
    assert saved["metrics"]["fees"] > 0
    assert abs(saved["metrics"]["reconciliation_error"]) < 1e-6
    with (output / selected_name / "trades.csv").open() as stream:
        trades = list(csv.DictReader(stream))
    assert len(trades) == 1
    trade = trades[0]
    assert int(trade["confirmation"]) == expected_confirmation
    assert int(trade["entry"]) == expected_confirmation + 1
    assert int(trade["exit"]) > int(trade["entry"])
    assert int(trade["shares"]) > 0 and int(trade["shares"]) % 100 == 0
    assert trade["reason"] == "profit"
    with (output / selected_name / "daily.csv").open() as stream:
        daily = list(csv.DictReader(stream))
    assert min(float(row["cash"]) for row in daily) >= 0
    assert max(float(row["exposure"]) for row in daily) <= 0.06
    for row in daily:
        assets, cash, receivable = (float(row[key]) for key in ("nav", "cash", "receivable"))
        assert assets == pytest.approx(cash + receivable + assets * float(row["exposure"]))
