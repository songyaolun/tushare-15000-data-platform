"""公开缓存、面板、命令和可恢复研究流程的无网络验证。"""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_regime_research import envelope
from test_regime_research import panel as panel
from typer.testing import CliRunner

from tushare_entitlement_hub.client import HttpTransport
from tushare_entitlement_hub.regime_research import cli, runner
from tushare_entitlement_hub.regime_research import panel as panel_module
from tushare_entitlement_hub.regime_research.collection import collect_partition, collection_jobs
from tushare_entitlement_hub.regime_research.panel import load_panel, prepare_panel


def write_partition(path, api, parameters, rows):
    """写合成原文缓存供离线重放；输入空行输出成功空表包络。"""
    payload = json.loads(envelope(rows))
    payload.update(api_name=api, parameters=parameters)
    path.write_text(json.dumps(payload))


def save_panel(path, source):
    """保存测试面板；输入Panel写npy及清单，用于真实离线读取路径。"""
    path.mkdir(exist_ok=True)
    manifest = {"calendar": source.calendar, "codes": source.codes}
    for category in ("values", "signals", "market"):
        arrays = getattr(source, category)
        manifest[category] = list(arrays)
        for name, array in arrays.items():
            np.save(path / f"{category}-{name}.npy", array)
    (path / "panel.json").write_text(json.dumps(manifest))


def test_collection_jobs_and_strict_cached_partitions(tmp_path):
    """采集计划按月末/除权日分区，缓存命中不联网，错误代码或日期拒绝。"""
    calendar = ["20211018", "20211019", "20211101"]
    jobs = collection_jobs(calendar)
    assert len(jobs) == 7
    assert jobs[2][2]["trade_date"] == "20211019"
    with pytest.raises(ValueError):
        collection_jobs([])
    with HttpTransport(lambda: "") as transport:
        name, api, params, cap = jobs[-1]
        write_partition(tmp_path / name, api, params, [])
        assert collect_partition(transport, tmp_path, jobs[-1]) == 0
        write_partition(tmp_path / name, api, params, [{"ex_date": "20200101"}])
        with pytest.raises(ValueError, match="date_mismatch"):
            collect_partition(transport, tmp_path, jobs[-1])
        name, api, params, cap = jobs[0]
        write_partition(tmp_path / name, api, params, [{"ts_code": "wrong"}])
        with pytest.raises(ValueError, match="code_mismatch"):
            collect_partition(transport, tmp_path, jobs[0])


def test_prepare_panel_source_contract_and_forward_only_capitalization(
    tmp_path, monkeypatch, panel
):
    """面板读取原securities字段，月末市值只向后延用，基准缺日拒绝。"""
    dataset, raw, output = (tmp_path / name for name in ("features", "raw", "panel"))
    dataset.mkdir()
    raw.mkdir()
    source = {
        "calendar": panel.calendar,
        "securities": panel.codes,
        "source_fingerprint": "fixture",
    }
    (dataset / "dataset.json").write_text(json.dumps(source))
    for code, name in [("000985.CSI", "index"), ("H00985.CSI", "total_index")]:
        rows = [
            dict(
                ts_code=code,
                trade_date=day,
                close=float(panel.market[name][i]),
                open=float(panel.market[name][i]),
            )
            for i, day in enumerate(panel.calendar)
        ]
        write_partition(raw / f"index-{code}.json", "index_daily", {}, rows)
    for code in panel.codes:
        np.savez(
            dataset / f"{code}.npz", **{name: array[:, 0] for name, array in panel.values.items()}
        )
    masks = {name: np.ones(len(panel.calendar), dtype=bool) for name in panel.signals}
    monkeypatch.setattr(panel_module, "strategy_masks", lambda features: masks)
    with pytest.raises(ValueError, match="capitalization"):
        prepare_panel(dataset, raw, output)
    for day in (panel.calendar[30], panel.calendar[-1]):
        write_partition(
            raw / f"capitalization-{day}.json",
            "daily_basic",
            {},
            [
                dict(ts_code=panel.codes[0], trade_date=day, total_mv=100),
                dict(ts_code="unknown", trade_date=day, total_mv=200),
                dict(ts_code=panel.codes[1], trade_date=day, total_mv=None),
            ],
        )
    result = prepare_panel(dataset, raw, output)
    assert prepare_panel(dataset, raw, output) == result
    loaded = load_panel(output)
    assert np.isnan(loaded.values["capitalization"][:30, 0]).all()
    assert loaded.values["capitalization"][30:, 0].min() == 100
    assert loaded.codes == panel.codes
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "file").write_text("partial")
    with pytest.raises(ValueError, match="incomplete_panel"):
        prepare_panel(dataset, raw, partial)
    write_partition(raw / "index-000985.CSI.json", "index_daily", {}, [])
    with pytest.raises(ValueError, match="benchmark"):
        prepare_panel(dataset, raw, tmp_path / "bad-index")


def test_runner_resume_full_output_and_missing_distribution(tmp_path, monkeypatch, panel):
    """资金流程可恢复但参数不匹配拒绝，缺分红日不能当作没有分红。"""
    panel_path = tmp_path / "panel"
    save_panel(panel_path, panel)
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="dividend_partitions"):
        runner.run_research(panel_path, raw, output, limit=1)
    for day in panel.calendar:
        write_partition(raw / f"dividend-{day}.json", "dividend", {"ex_date": day}, [])
    grid = runner.research_grid()
    assert len(grid) == 118 and len({policy.name for _, policy in grid}) == 118
    # 此处只运行两套配置，验证编排语义；真实网格数量单独断言。
    monkeypatch.setattr(runner, "research_grid", lambda: grid[:2])
    report = runner.run_research(panel_path, raw, output)
    saved = json.loads(report.read_text())
    assert saved["complete_grid"] and saved["attempts"] == 2
    assert runner.run_research(panel_path, raw, output) == report
    cfg, policy = grid[0]
    monkeypatch.setattr(runner, "research_grid", lambda: [(cfg, replace(policy, take_profit=0.2))])
    with pytest.raises(ValueError, match="configuration_mismatch"):
        runner.run_research(panel_path, raw, output)
    (output / "input-fingerprint.txt").write_text("different")
    with pytest.raises(ValueError, match="inputs_or_code_changed"):
        runner.run_research(panel_path, raw, output)
    (output / "input-fingerprint.txt").unlink()
    with pytest.raises(ValueError, match="unversioned"):
        runner.run_research(panel_path, raw, output)
    runner.write_csv(tmp_path / "empty.csv", [])
    assert (tmp_path / "empty.csv").read_text() == ""


def test_cli_explicit_live_and_offline_routes(tmp_path, monkeypatch):
    """命令默认不联网；显式live才构建传输，prepare/run交给独立模块。"""
    test = CliRunner()
    assert (
        test.invoke(cli.regime_app, ["collect", "missing", "missing", str(tmp_path)]).exit_code != 0
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(json.dumps({"calendar": ["20211018"]}))
    monkeypatch.setattr(cli, "configured_credential", lambda path: None)
    monkeypatch.setattr(
        cli, "collect_partition", lambda transport, output, job: transport._token_provider()
    )
    live = test.invoke(
        cli.regime_app, ["collect", "missing", str(dataset), str(tmp_path), "--live"]
    )
    assert live.exit_code == 0, live.output
    monkeypatch.setattr(cli, "prepare_panel", lambda *args: tmp_path / "panel.json")
    monkeypatch.setattr(cli, "run_research", lambda *args, **kwargs: tmp_path / "report.json")
    assert (
        test.invoke(
            cli.regime_app, ["prepare", str(dataset), str(tmp_path), str(tmp_path)]
        ).exit_code
        == 0
    )
    assert (
        test.invoke(cli.regime_app, ["run", str(dataset), str(tmp_path), str(tmp_path)]).exit_code
        == 0
    )
