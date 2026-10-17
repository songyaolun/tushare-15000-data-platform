"""离线编排测试验证独立确认、共享证据、研究口径和边界输入。"""

import csv
import gzip
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from tushare_entitlement_hub.signal_research import runner
from tushare_entitlement_hub.signal_research.artifacts import write_signal_artifacts
from tushare_entitlement_hub.signal_research.indicators import (
    IndicatorPoint,
    StockCharacteristics,
    compute_indicators,
)
from tushare_entitlement_hub.signal_research.signals import ExperimentSwitches


def quote(day: str, **changes: object) -> dict:
    """创建有效源行情；输入日期，输出含原始单位的10元行情。

    示例 quote('20230101') 的amount=1代表1千元。
    """
    return dict(
        trade_date=day,
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.0,
        adj_factor=1.0,
        vol=100.0,
        amount=1.0,
        price_limits_exempt=True,
        circ_mv=2.0,
        **changes,
    )


def point(day: str, histogram: float | None = 0.003, **changes: object) -> IndicatorPoint:
    """创建技术条件通过的日点；输入日期，输出可确认的指标点。

    示例 point('20230101', .001) 表示阈值穿越前的日点。
    """
    return replace(
        IndicatorPoint(
            day,
            10.0,
            9.0,
            8.0,
            8.0,
            7.0,
            1.0,
            0.9,
            0.03,
            histogram,
            1.0,
            StockCharacteristics(*([None] * 8)),
        ),
        **changes,
    )


def records(directory: Path) -> list[dict]:
    """读取逐信号证据；输入输出目录，输出JSONL记录列表。

    示例空signal_records.jsonl输出[]。
    """
    return [
        json.loads(line) for line in (directory / "signal_records.jsonl").read_text().splitlines()
    ]


def test_helpers_and_calendar_boundaries() -> None:
    """场景：单位转换、未知日、空日点、非法数值；预期：不伪造可用指标。"""
    source = quote("20230101")
    assert runner._price_bar(source).amount == 1000
    assert runner._price_bar(source).market_cap == 20000
    assert runner._price_bar({**source, "amount": None, "circ_mv": None}).amount is None
    assert runner._valid_quote(source)
    for value in (0, None, True, float("inf"), float("nan")):
        assert not runner._valid_quote({**source, "adj_factor": value})
        assert not runner._valid_quote({**source, "vol": value})
        assert not runner._valid_quote({**source, "amount": value})
    assert not runner._valid_quote({**source, "suspended": True})
    assert not runner._valid_quote({**source, "tradable": False})
    assert not runner._valid_quote({**source, "high": 9})
    assert not runner._valid_quote({**source, "low": 11})
    assert runner._calendar_points((), ["20230101"]) == ()
    points = [point("20230102")]
    expanded = runner._calendar_points(points, ["20230101", "20230102", "20230103"])
    assert expanded[1].normalized_histogram is None
    assert expanded[1].characteristics.market_cap is None
    assert runner._calendar_points(points, ["20230103"])[0].normalized_histogram is None
    experiments = runner.research_experiments()
    assert len({experiment.name for experiment in experiments}) == len(experiments)
    assert any(not experiment.event_filter for experiment in experiments)


def test_forward_return_adjustment_and_missing_quotes() -> None:
    """场景：除权报价、长窗口缺数据；预期：因子收益不按裸价错误计算。"""
    calendar = [f"202301{index:02}" for index in range(1, 7)]
    rows = {
        calendar[0]: quote(calendar[0]),
        calendar[5]: {
            **quote(calendar[5]),
            "open": 5,
            "high": 6,
            "low": 4,
            "close": 5,
            "adj_factor": 2,
        },
    }
    assert runner._forward_returns(calendar[0], rows, calendar)["return_5d"] == 0
    assert runner._forward_returns(calendar[0], rows, calendar)["return_10d"] is None
    rows[calendar[5]]["close"] = 0
    assert runner._forward_returns(calendar[0], rows, calendar)["return_5d"] is None


def test_independent_experiments_calendar_window_and_shared_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """场景：DIF与MA各自首次通过，中间停牌；预期：独立确认且全部保留共享引用。"""
    calendar = ["20230101", "20230102", "20230103", "20230104", "20230105", "20230106"]
    points = (
        point(calendar[0], 0.001),
        point(calendar[1], dif=-1),
        point(calendar[3]),
        point(calendar[4]),
        point(calendar[5]),
    )
    observed_inputs = []

    def indicators(bars, observation_date):
        """捕获真实送入指标的行情；输入bars，输出控制日点以检验确认窗口。"""
        observed_inputs.extend(bar.trade_date for bar in bars)
        return points

    monkeypatch.setattr(runner, "compute_indicators", indicators)
    monkeypatch.setattr(
        runner,
        "research_experiments",
        lambda: (
            ExperimentSwitches("dif", ("dif_positive",)),
            ExperimentSwitches("ma", ("close_above_ma20",)),
            ExperimentSwitches("daily", (), False),
        ),
    )
    source = [quote(day) for day in calendar if day != calendar[2]]
    manifest = runner.run_research(
        [("000001.SZ", source)],
        calendar,
        tmp_path,
        runner.ResearchConfiguration("20230101", "20230106"),
    )
    data = records(tmp_path)
    assert (
        next(item for item in data if item["experiment_id"] == "dif")["confirmation_date"]
        == "2023-01-04"
    )
    assert (
        next(item for item in data if item["experiment_id"] == "ma")["confirmation_date"]
        == "2023-01-02"
    )
    assert sum(item["experiment_id"] == "daily" for item in data) == 5
    assert len({item["source_series"] for item in data}) == 1
    assert any(item["status"] == "skipped" for item in data)
    assert any(item["status"] == "open" for item in data)
    with gzip.open(tmp_path / data[0]["source_series"], "rt") as stream:
        matrix = list(csv.DictReader(stream))
    assert len(matrix) == 6
    assert matrix[2]["normalized_histogram"] == ""
    assert matrix[2]["dif_positive"] == "UNKNOWN"
    assert "efficiency_60" in matrix[0]
    assert matrix[2]["market_cap"] == ""
    assert data[0]["pre_signal_metrics"]["close_ma20_extension_pct"] == pytest.approx(100 / 9)
    assert data[0]["pre_signal_metrics"]["atr14_close_ratio"] == 0.1
    assert data[0]["pre_signal_metrics"]["previous_ma20"] == 8
    assert data[0]["pre_signal_metrics"]["dif"] == 1
    assert manifest["chart_count"] == 0
    assert data[0]["source_series_metadata"]["indicator_anchor_date"] == "20230101"
    # 停牌占T+1，缩短至T+1窗口时DIF在T+2通过已经过期。
    short = runner.run_research(
        [("000001.SZ", source)],
        calendar,
        tmp_path / "short",
        runner.ResearchConfiguration("20230101", "20230106", confirmation_days=1),
    )
    assert short["counts"].get("signals:dif", 0) == 0
    observed_inputs.clear()
    suspended_positive = [*source, {**quote(calendar[2]), "suspended": True}]
    runner.run_research(
        [("000001.SZ", suspended_positive)],
        calendar,
        tmp_path / "suspended",
        runner.ResearchConfiguration("20230101", "20230106"),
    )
    assert calendar[2] not in observed_inputs
    observed_inputs.clear()
    invalid = [*source, {**quote(calendar[2]), "high": 9}]
    invalid_result = runner.run_research(
        [("000001.SZ", invalid)],
        calendar,
        tmp_path / "invalid",
        runner.ResearchConfiguration("20230101", "20230106"),
    )
    assert calendar[2] not in observed_inputs
    assert invalid_result["counts"]["invalid_or_inactive_source_quotes"] == 1


def test_chart_layers_preserve_open_and_all_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """场景：正负零未知四层且尚未退出；预期：每层限图，JSONL不丢任何信号。"""
    calendar = ["20230101", "20230102", "20230103", "20230104"]
    points = tuple(point(day, 0.001 if index == 0 else 0.003) for index, day in enumerate(calendar))
    points = (replace(points[0], ma20=None, atr14=None), *points[1:])
    monkeypatch.setattr(runner, "compute_indicators", lambda *args: points)
    monkeypatch.setattr(
        runner,
        "research_experiments",
        lambda: (ExperimentSwitches("technical_baseline", (), False),),
    )
    returns = iter([0.1, -0.1, 0.0, None] * 2)
    monkeypatch.setattr(runner, "_forward_returns", lambda *args: {"return_20d": next(returns)})
    captured = []

    def write(artifact, directory, **kwargs):
        """替代仅PNG渲染，保存真实JSON/CSV；输入artifact输出真实索引行。"""
        captured.append(artifact)
        return write_signal_artifacts(artifact, directory, render_charts=False)

    monkeypatch.setattr(runner, "write_signal_artifacts", write)
    securities = [(code, [quote(day) for day in calendar]) for code in ("a", "b")]
    result = runner.run_research(
        securities,
        calendar,
        tmp_path,
        runner.ResearchConfiguration("20230101", "20230104", chart_samples_per_year_direction=1),
    )
    assert result["chart_count"] == 4
    assert len(records(tmp_path)) == 8
    assert any(item["status"] == "open" for item in records(tmp_path))
    for artifact in captured:
        assert (
            artifact.pre_signal_metrics["snapshot_price_anchor_date"] == artifact.confirmation_date
        )
    assert (tmp_path / "charts/index.html").exists()


def test_future_anchor_scaling_and_snapshot_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """场景：增加未来因子锚点；预期：H/C及确认快照不变，源指标明确缩放。"""
    calendar = [
        (date(2023, 1, 1) + timedelta(days=index)).strftime("%Y%m%d") for index in range(85)
    ]
    rows = {
        day: {
            **quote(day),
            "open": 10 + index * 0.1,
            "close": 10 + index * 0.1,
            "high": 11 + index * 0.1,
            "low": 9 + index * 0.1,
            "adj_factor": 2 if index >= 80 else 1,
        }
        for index, day in enumerate(calendar)
    }
    bars = [runner._price_bar(row) for row in rows.values()]
    earlier = compute_indicators(bars, calendar[79])
    full = compute_indicators(bars, calendar[-1])
    assert full[79].normalized_histogram == pytest.approx(earlier[79].normalized_histogram)
    assert full[79].ma20 == pytest.approx(earlier[79].ma20)
    record = dict(signal_id="one", pre_signal_metrics={}, filter_results={}, post_signal_returns={})
    artifact = runner._chart_artifact("a", {"confirmation_date": calendar[79]}, full, rows, record)
    position = artifact.dated_values["date"].index(calendar[79])
    assert artifact.dated_values["ma20"][position] == pytest.approx(earlier[79].ma20)
    assert artifact.dated_values["close"][position] == earlier[79].close
    entry = write_signal_artifacts(artifact, tmp_path, render_charts=False)
    evidence = json.loads((tmp_path / entry.evidence_json).read_text())
    assert all(row["date"] <= "2023-03-21" for row in evidence["snapshot"]["observations"])
    assert len(evidence["post_signal_review"]["observations"]) == 5
    monkeypatch.setattr(
        runner, "research_experiments", lambda: (ExperimentSwitches("daily", (), False),)
    )
    runner.run_research(
        [("a", list(rows.values()))],
        calendar,
        tmp_path / "run",
        runner.ResearchConfiguration(calendar[0], calendar[-1], chart_samples_per_year_direction=0),
    )
    confirmation_record = next(
        record
        for record in records(tmp_path / "run")
        if record["confirmation_date"].replace("-", "") == calendar[79]
    )
    for name in (
        "close",
        "ma20",
        "ma60",
        "previous_ma20",
        "previous_ma60",
        "dif",
        "dea",
        "histogram",
        "atr14",
    ):
        assert confirmation_record["pre_signal_metrics"][name] == pytest.approx(
            getattr(earlier[79], name)
        )


@pytest.mark.parametrize("calendar", [[], ["20230102", "20230101"], ["20230101", "20230101"]])
def test_invalid_calendar(tmp_path: Path, calendar: list[str]) -> None:
    """场景：空、乱序、重复日历；预期：拒绝不能证明交易日顺序的输入。"""
    with pytest.raises(ValueError, match="日历"):
        runner.run_research([], calendar, tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_date": "bad"},
        {"end_date": "20230230"},
        {"histogram_threshold": float("inf")},
        {"histogram_threshold": -1},
        {"histogram_threshold": True},
        {"confirmation_days": -1},
        {"confirmation_days": 1.5},
        {"chart_samples_per_year_direction": -1},
    ],
)
def test_invalid_research_configuration(changes: dict) -> None:
    """场景：参数日期或阈值不合法；预期：创建协议时拒绝。"""
    with pytest.raises(ValueError):
        runner.ResearchConfiguration(**changes)


def test_empty_duplicate_and_no_calendar_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """场景：无行情、重复行情、首行情晚于日历；预期：明确计数或拒绝。"""
    calendar = ["20230101"]
    with pytest.raises(ValueError, match="起始"):
        runner.run_research(
            [], calendar, tmp_path, runner.ResearchConfiguration("20230102", "20230101")
        )
    with pytest.raises(ValueError, match="重复"):
        runner.run_research(
            [("a", [quote(calendar[0]), quote(calendar[0])])], calendar, tmp_path / "duplicate"
        )
    result = runner.run_research([("a", [])], calendar, tmp_path / "empty")
    assert result["counts"]["securities_without_valid_factor_quotes"] == 1
    monkeypatch.setattr(runner, "compute_indicators", lambda *args: (point("20230102"),))
    result = runner.run_research([("a", [quote("20230102")])], calendar, tmp_path / "outside")
    assert result["counts"]["securities_without_calendar_points"] == 1


def test_progress_and_end_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """场景：100证券及未来行情；预期：流式进度计数正确且未来不参与。"""
    monkeypatch.setattr(runner, "compute_indicators", lambda *args: (point("20230101", 0.001),))
    securities = [(str(index), [quote("20230101"), quote("20230102")]) for index in range(100)]
    result = runner.run_research(
        securities,
        ["20230101", "20230102"],
        tmp_path,
        runner.ResearchConfiguration("20230101", "20230101"),
    )
    assert result["counts"]["securities_processed"] == 100
    assert "100只证券" in capsys.readouterr().out


@pytest.mark.parametrize("existing", ["signal_records.jsonl", "daily_condition_matrix/old.csv.gz"])
def test_existing_research_artifacts_are_preserved_and_rejected(tmp_path, existing):
    """场景：已有信号或旧市场矩阵；预期：拒绝覆盖，原资料保持原样。"""
    path = tmp_path / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old evidence")
    with pytest.raises(ValueError, match="空产物目录"):
        runner.run_research([], ["20230101"], tmp_path)
    assert path.read_text() == "old evidence"


def test_empty_matrix_directory_and_history_database_are_allowed(tmp_path):
    """场景：CLI预建数据库和空矩阵目录；预期：正常运行且保留数据库。"""
    (tmp_path / "daily_condition_matrix").mkdir()
    database_path = tmp_path / "history.duckdb"
    database_path.write_text("existing history")
    runner.run_research([], ["20230101"], tmp_path)
    assert database_path.read_text() == "existing history"
