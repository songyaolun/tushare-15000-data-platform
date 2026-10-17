"""事后复核使用合成行情和本地日历，无联网或信号写入。"""

import csv
import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from tushare_entitlement_hub.signal_research.retrospective import (
    _benchmark_rows,
    _calendar,
    _day,
    _index_return,
    _number,
    _path_metrics,
    build_retrospective_review,
)


def fixture(directory: Path, years: tuple[int, ...] = (2023,)) -> tuple[list[str], Path]:
    """生成基线已退出/未退出记录；输入目录年份输出日历及指数根。

    示例fixture(tmp_path)生成45天行情，固定窗口MAE=-.2、MFE=.3。
    """
    directory.mkdir(exist_ok=True)
    benchmark = directory / "benchmark"
    benchmark.mkdir()
    calendar = [
        (date(years[0], 1, 1) + timedelta(days=index)).strftime("%Y%m%d") for index in range(45)
    ]
    (benchmark / "trading_dates.json").write_text(json.dumps(calendar))
    matrix = directory / "shared.csv.gz"
    with gzip.open(matrix, "wt", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["trade_date", "low", "high", "close", "adj_factor"]
        )
        writer.writeheader()
        for index, day in enumerate(calendar):
            writer.writerow(
                dict(
                    trade_date=day,
                    low=8 if index else 10,
                    high=13 if index else 10,
                    close=12 if index == 1 else 10,
                    adj_factor=1,
                )
            )
    signals = []
    for year in years:
        for name, status, loss in [
            ("a", "closed", -0.1),
            ("b", "closed", -0.2),
            ("c", "open", None),
            ("other", "closed", -0.9),
        ]:
            signals.append(
                dict(
                    signal_id=f"{year}-{name}",
                    ts_code="a",
                    confirmation_date=f"{year}-01-01",
                    status=status,
                    experiment_id="other" if name == "other" else "technical_baseline",
                    source_series="shared.csv.gz",
                    entry_date=f"{year}-01-02",
                    exit_date=f"{year}-01-04" if status == "closed" else None,
                    entry_price=12.0,
                    exit_price=10.0 if status == "closed" else None,
                    fee_adjusted_research_return=loss,
                    post_signal_returns={"return_20d": 0.0, "return_40d": 0.0},
                    pre_signal_metrics={"efficiency_60": 0.7},
                )
            )
    (directory / "signal_records.jsonl").write_text(
        "\n".join(json.dumps(signal) for signal in signals)
    )
    (benchmark / "benchmarks").mkdir()
    for code in ("000001.SH", "399001.SZ", "399006.SZ"):
        (benchmark / "benchmarks" / f"{code}.json").write_text(
            json.dumps(
                {
                    "response": {
                        "fields": ["ts_code", "trade_date", "close"],
                        "items": [[code, day, 100 + index] for index, day in enumerate(calendar)],
                    }
                }
            )
        )
    return calendar, benchmark


def test_numbers_dates_and_index_returns() -> None:
    """场景：未知数值、缺指数日；预期：None保留且有效价格比正确。"""
    for value in (None, True, "", object(), float("nan")):
        assert _number(value) is None
    assert _number("2") == 2
    assert _day("2023-01-01") == "20230101"
    assert _index_return({"a": 10, "b": 11}, "a", "b") == pytest.approx(0.1)
    assert _index_return({}, None, "b") is None
    assert _index_return({}, "a", "b") is None


def test_path_metrics_distinguish_mae_and_drawdown() -> None:
    """场景：先上涨再回落；预期：MAE与峰谷回撤分别计算，缺数据不填0。"""
    rows = {
        "a": dict(low=9, high=12, close=12, adj_factor=1),
        "b": dict(low=8, high=11, close=10, adj_factor=1),
    }
    result = _path_metrics(rows, ["a", "b"], 10, 1)
    assert result["mae"] == pytest.approx(-0.2)
    assert result["mfe"] == pytest.approx(0.2)
    assert result["close_peak_trough_drawdown"] == pytest.approx(-1 / 6)
    for price, factor, days in [
        (None, 1, ["a"]),
        (0, 1, ["a"]),
        (10, 0, ["a"]),
        (10, None, ["a"]),
        (10, 1, []),
        (10, 1, ["missing"]),
    ]:
        assert _path_metrics(rows, days, price, factor)["mae"] is None


def test_calendar_sources_and_index_formats(tmp_path: Path) -> None:
    """场景：JSON/数据库/日期并集及指数格式；预期：明确日历来源和可读格式。"""
    assert _calendar(tmp_path, ["20230102", "20230101"])[0] == ["20230101", "20230102"]
    with duckdb.connect(str(tmp_path / "history.duckdb")) as connection:
        connection.execute("CREATE TABLE trading_calendar(date VARCHAR)")
        connection.execute("INSERT INTO trading_calendar VALUES ('20230101')")
    assert _calendar(tmp_path, [])[1] == "history_database_trading_calendar"
    assert _calendar(tmp_path / "history.duckdb", [])[0] == ["20230101"]
    (tmp_path / "trading_dates.json").write_text(json.dumps({"trading_dates": ["20230102"]}))
    assert _calendar(tmp_path, [])[0] == ["20230102"]
    source = tmp_path / "index_daily"
    source.mkdir()
    (source / "a.json").write_text(
        json.dumps(
            {
                "records": [
                    dict(ts_code="000001.SH", trade_date="20230101", close=10),
                    dict(ts_code="unknown", trade_date="20230101", close=10),
                    dict(ts_code="399001.SZ", trade_date="20230101", close=0),
                ]
            }
        )
    )
    (source / "b.json").write_text(
        json.dumps([dict(ts_code="399001.SZ", trade_date="20230101", close=11)])
    )
    (source / "c.json").write_text(
        json.dumps(
            {
                "data": {
                    "fields": ["ts_code", "trade_date", "close"],
                    "items": [["399006.SZ", "20230101", 12]],
                }
            }
        )
    )
    assert _benchmark_rows(tmp_path)["000001.SH"]["20230101"] == 10


def test_review_windows_holding_benchmarks_and_extremes(tmp_path: Path) -> None:
    """场景：完整基线复核及六年极端样本；预期：基线限定、各年最差且未退出保留。"""
    _, benchmark = fixture(tmp_path, tuple(range(2020, 2027)))
    path = build_retrospective_review(tmp_path, benchmark)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21
    first = rows[0]
    assert float(first["mae20"]) == pytest.approx(-0.2)
    assert float(first["mfe40"]) == pytest.approx(0.3)
    assert float(first["market20_000001.SH"]) == pytest.approx(0.2)
    assert float(first["holding_market_return_000001.SH"]) == pytest.approx(103 / 101 - 1)
    assert float(first["holding_mae"]) == pytest.approx(8 / 12 - 1)
    assert rows[2]["holding_mae"] == ""
    assert rows[3]["mae20"] == ""
    payload = json.loads((tmp_path / "baseline_loss_samples.json").read_text())
    assert len(payload["samples"]) == 6
    assert all(row["signal_id"].endswith("-b") for row in payload["samples"])
    assert "not_representative" in payload["interpretation"]
    with gzip.open(tmp_path / "shared.csv.gz", "rt") as stream:
        source_rows = list(csv.DictReader(stream))
    source_rows[3].update(low=0.01, high=1000, close=500)
    with gzip.open(tmp_path / "shared.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    with build_retrospective_review(tmp_path, benchmark).open() as stream:
        updated = list(csv.DictReader(stream))[0]
    assert updated["holding_mae"] == first["holding_mae"]
    assert updated["holding_mfe"] == first["holding_mfe"]
    assert (
        updated["holding_close_peak_trough_drawdown"] == first["holding_close_peak_trough_drawdown"]
    )
    assert float(updated["mfe20"]) > float(first["mfe20"])


def test_empty_and_unsafe_series_reference(tmp_path: Path) -> None:
    """场景：空基线和路径穿越；预期：空产物有效，越界引用拒绝。"""
    _, benchmark = fixture(tmp_path)
    (tmp_path / "signal_records.jsonl").write_text("")
    path = build_retrospective_review(tmp_path, benchmark)
    assert path.read_text().startswith("signal_id")
    assert json.loads((tmp_path / "baseline_loss_samples.json").read_text())["samples"] == []
    (tmp_path / "signal_records.jsonl").write_text(
        json.dumps(dict(experiment_id="technical_baseline", source_series="../secret.csv.gz"))
    )
    with pytest.raises(ValueError, match="相对路径"):
        build_retrospective_review(tmp_path, benchmark)


def test_missing_calendar_fallback_and_missing_exit_fill(tmp_path: Path) -> None:
    """场景：无认证日历、无退出成交价；预期：注明日期并集，持仓指标未知。"""
    _, benchmark = fixture(tmp_path)
    (benchmark / "trading_dates.json").unlink()
    path = tmp_path / "signal_records.jsonl"
    signals = [json.loads(line) for line in path.read_text().splitlines()]
    for signal in signals:
        signal["exit_price"] = None
    path.write_text("\n".join(json.dumps(signal) for signal in signals))
    with build_retrospective_review(tmp_path, benchmark).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["holding_mae"] == ""
    assert "not_certified_market_calendar" in rows[0]["calendar_source"]
