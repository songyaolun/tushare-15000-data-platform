"""验证离线研究汇总不隐藏尾端信号，不制造组合收益或缺失数值。"""

import csv
import gzip
import json

import pytest

from tushare_entitlement_hub.signal_research import reporting
from tushare_entitlement_hub.signal_research.reporting import (
    CHARACTERISTIC_COLUMNS,
    enrich_signal_percentiles,
    summarize_trades,
    write_research_report,
)


def write_percentile_fixture(directory, *, explicit_code=False):
    """创建三证券两日矩阵与少量信号；输入临时目录，输出全市场排名离线样本。

    示例：第一日效率 [10,20,30]、只选择中间证券，预期其分位 0.5 而非 0。
    """
    matrix_directory = directory / "daily_condition_matrix"
    matrix_directory.mkdir()
    for position, code in enumerate(("A.SZ", "B.SZ", "C.SZ")):
        with gzip.open(matrix_directory / f"{code}.csv.gz", "wt", newline="") as stream:
            columns = ["trade_date", *CHARACTERISTIC_COLUMNS]
            if explicit_code:
                columns.append("ts_code")
            if position == 0:
                columns.append("extra")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for day in ("20240103", "20240104"):
                row = {column: (position + 1) * 10 for column in CHARACTERISTIC_COLUMNS}
                row["trade_date"] = day
                row["wick_ratio_median_60"] = "UNKNOWN" if position == 0 else 5
                row["market_cap"] = "nan" if position == 1 else (position + 1) * 10
                row["amount_median_20"] = "inf" if position == 0 else (position + 1) * 10
                if explicit_code:
                    row["ts_code"] = code
                writer.writerow(row)
    records = [
        {
            "ts_code": code,
            "confirmation_date": day,
            "signal_id": identifier,
            "experiment_id": experiment,
        }
        for code, day, identifier, experiment in (
            ("B.SZ", "2024-01-03", "s1", "base"),
            ("B.SZ", "20240103", "s2", "ablation"),
            ("A.SZ", "2024-01-04", "s3", "base"),
            ("missing", "2024-01-03", "s4", "base"),
        )
    ]
    if explicit_code:
        for record in records:
            record.update(
                status="closed",
                entry_date="2024-01-04",
                exit_date="2024-01-05",
                fee_adjusted_research_return=0.12,
                post_signal_returns={"return_20d": 0.2},
            )
    (directory / "signal_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )


@pytest.mark.parametrize("explicit_code", [False, True])
def test_signal_percentiles_use_all_market_values_and_preserve_unknown(tmp_path, explicit_code):
    """场景：全市场三证券、只选少数信号且有 UNKNOWN；预期：逐特征全市场排名并保留空值。"""
    write_percentile_fixture(tmp_path, explicit_code=explicit_code)
    path = enrich_signal_percentiles(tmp_path, memory_limit="64MB")
    with path.open() as stream:
        rows = {row["signal_id"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 4
    assert rows["s1"]["efficiency_60"] == "20.0"
    assert rows["s1"]["efficiency_60_percentile"] == "0.5"
    assert rows["s2"]["efficiency_60_percentile"] == "0.5"
    assert rows["s1"]["market_cap"] == "" and rows["s1"]["market_cap_percentile"] == ""
    assert rows["s1"]["wick_ratio_median_60_percentile"] == "0.0"
    assert rows["s3"]["wick_ratio_median_60"] == ""
    assert rows["s4"]["efficiency_60_percentile"] == ""
    assert rows["s1"]["status"] == ("closed" if explicit_code else "")
    assert rows["s1"]["return_20d"] == ("0.2" if explicit_code else "")
    assert not list(tmp_path.glob("signal-percentiles-*"))


@pytest.mark.parametrize("failure", ["missing_matrix", "missing_column", "duplicate", "bad_key"])
def test_signal_percentiles_reject_incomplete_or_duplicate_market(tmp_path, failure):
    """场景：市场缺失、字段未齐、重复证券日或无效日期；预期：拒绝伪造分位并清理临时库。"""
    write_percentile_fixture(tmp_path)
    matrix_directory = tmp_path / "daily_condition_matrix"
    if failure == "missing_matrix":
        for path in matrix_directory.iterdir():
            path.unlink()
    elif failure == "missing_column":
        for path in matrix_directory.iterdir():
            with gzip.open(path, "rt") as stream:
                text = stream.read()
            with gzip.open(path, "wt") as stream:
                stream.write(text.replace("efficiency_60", "other"))
    else:
        path = matrix_directory / "A.SZ.csv.gz"
        with gzip.open(path, "rt") as stream:
            lines = stream.readlines()
        with gzip.open(path, "wt") as stream:
            if failure == "duplicate":
                stream.writelines([*lines, lines[1]])
            else:
                stream.writelines([lines[0], lines[1].replace("20240103", "bad"), lines[2]])
    with pytest.raises(ValueError):
        enrich_signal_percentiles(tmp_path)
    assert not list(tmp_path.glob("signal-percentiles-*"))


def test_signal_percentiles_empty_signal_has_only_header(tmp_path):
    """场景：无信号；预期：无需市场数据，生成空表而非虚构数字。"""
    (tmp_path / "signal_records.jsonl").write_text("\n  \n")
    path = enrich_signal_percentiles(tmp_path)
    with path.open() as stream:
        assert list(csv.DictReader(stream)) == []


def test_signal_projection_preserves_nested_results_and_pure_iso_dates(tmp_path):
    """纯ISO日期被推断DATE仍可排名，未消费嵌套字段不改变所需收益与缺失分位输出。"""
    write_percentile_fixture(tmp_path, explicit_code=True)
    path = tmp_path / "signal_records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        day = record["confirmation_date"].replace("-", "")
        record["confirmation_date"] = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        record["unused_nested_evidence"] = {"notes": ["audit"] * 1000, "numbers": [1, 2, 3]}
    path.write_text("\n".join(json.dumps(record) for record in records))
    output = enrich_signal_percentiles(tmp_path, memory_limit="64MB")
    with output.open() as stream:
        rows = {row["signal_id"]: row for row in csv.DictReader(stream)}
    assert rows["s1"]["confirmation_date"] == "2024-01-03"
    assert rows["s1"]["efficiency_60_percentile"] == "0.5"
    assert rows["s1"]["return_20d"] == "0.2"
    assert rows["s4"]["efficiency_60_percentile"] == ""
    assert "unused_nested_evidence" not in rows["s1"]
    assert not list(tmp_path.glob("signal-percentiles-*"))


def test_explicit_csv_reader_preserves_default_reader_output(tmp_path, monkeypatch):
    """场景：显式CSV参数替代默认嗅探；预期：NULL、独立分位与完整CSV逐字节一致。

    输入异构列、UNKNOWN/nan/inf与收益嵌套样本，输出与原默认reader完全相同。
    """
    write_percentile_fixture(tmp_path, explicit_code=True)
    expected = enrich_signal_percentiles(tmp_path, memory_limit="64MB").read_bytes()
    original_connect = reporting.duckdb.connect

    class OriginalReaderConnection:
        """代理临时数据库，仅把显式CSV读取改回原有默认嗅探以形成对照。"""

        def __init__(self, connection):
            """输入真实DuckDB连接，输出保留在代理对象内的连接引用。"""
            self.connection = connection

        def execute(self, query, *arguments):
            """输入相同查询，去除新增CSV选项后返回真实DuckDB结果。"""
            if query == "SET memory_limit = ?":
                assert arguments[0] == ["1GB"]
            original_query = query.replace(
                ", sample_size=1, header=true, delim=',', quote='\"', escape='\"'", ""
            )
            return self.connection.execute(original_query, *arguments)

        def close(self):
            """输入无，输出None并关闭真实临时数据库连接。"""
            self.connection.close()

    def connect(*arguments, **keywords):
        """输入连接参数，输出使用原始reader查询的临时数据库代理。"""
        return OriginalReaderConnection(original_connect(*arguments, **keywords))

    monkeypatch.setattr(reporting.duckdb, "connect", connect)
    assert enrich_signal_percentiles(tmp_path).read_bytes() == expected


def trade(status, research_return=None, holding_sessions=None, year=2024, experiment="base"):
    """构造一行公开确定性实验记录；输入状态/收益，输出 simulation 风格字典。

    示例：trade('closed', 0.1, 10) → 2024 年基线闭合交易，研究收益 0.1。
    """
    return {
        "status": status,
        "experiment_id": experiment,
        "confirmation_date": f"{year}-01-03",
        "fee_adjusted_research_return": research_return,
        "holding_sessions": holding_sessions,
    }


def test_group_statistics_preserve_every_state_and_count_only_observable_returns():
    """场景：正负零收益、收益缺失和尾端状态混合；预期：分母明确且不隐藏未退出。"""
    trades = [
        trade("closed", 0.2, 10),
        trade("closed", -0.1, 20),
        trade("closed", 0, 0),
        trade("closed", None, None),
        trade("closed", True, -1),
        trade("closed", float("nan"), float("inf")),
        trade("closed", "0.2", True),
        trade("open", 999, 100),
        trade("censored"),
        trade("skipped"),
        trade("closed", 0.3, 4, year=2023),
        trade("closed", -0.4, 3, experiment="other"),
    ]
    summary = summarize_trades(iter(trades), {"base": "基线", "empty": "空实验"})
    baseline, empty, other = summary["experiments"]
    assert baseline["signals"] == 11
    assert [baseline[key] for key in ("closed", "open", "censored", "skipped")] == [8, 1, 1, 1]
    assert baseline["closed_return_observations"] == 4
    assert baseline["closed_return_unavailable"] == 4
    assert baseline["mean_fee_adjusted_research_return"] == pytest.approx(0.1)
    assert baseline["median_fee_adjusted_research_return"] == pytest.approx(0.1)
    assert baseline["win_rate"] == 0.5
    assert baseline["profit_factor"] == 5
    assert baseline["closed_holding_observations"] == 4
    assert baseline["average_holding_sessions"] == 8.5
    assert empty["signals"] == 0 and empty["win_rate"] is None
    assert other["experiment_name"] == "other" and other["profit_factor"] == 0
    assert [row["confirmation_year"] for row in summary["by_confirmation_year"]] == [
        2023,
        2024,
        2024,
    ]
    assert summary["by_confirmation_year"][0]["profit_factor"] is None
    assert summary["portfolio_return_available"] is False
    assert summary["portfolio_drawdown_available"] is False


def test_empty_or_nonclosed_returns_are_not_fabricated():
    """场景：无交易及仅有未退出交易；预期：零计数、空收益和空 profit_factor。"""
    empty = summarize_trades([], ["empty"])
    assert empty["experiments"][0]["signals"] == 0
    assert empty["by_confirmation_year"] == []
    open_trade = trade("open", 10, 2)
    del open_trade["experiment_id"]
    summary = summarize_trades([open_trade], [])
    row = summary["experiments"][0]
    assert row["experiment_id"] == "default"
    assert row["closed_return_observations"] == 0
    assert row["mean_fee_adjusted_research_return"] is None
    assert row["median_fee_adjusted_research_return"] is None
    assert row["average_holding_sessions"] is None
    assert row["profit_factor"] is None


def test_unknown_status_and_invalid_confirmation_date_are_rejected():
    """场景：未知交易状态或无效确认日期；预期：报错而不把记录默默排除。"""
    with pytest.raises(ValueError, match="交易状态"):
        summarize_trades([trade("mystery")], ["base"])
    row = trade("closed")
    row["confirmation_date"] = "bad"
    with pytest.raises(ValueError):
        summarize_trades([row], ["base"])


def test_report_contains_actual_statistics_limits_and_safe_manifest(tmp_path):
    """场景：真实汇总与含 HTML 的运行参数；预期：原值留 JSON，中文报告安全展示且限制齐全。"""
    summary = summarize_trades(
        [trade("closed", 0.123, 7), trade("open")], {"base": "<script>|基线\n"}
    )
    manifest = {"source": "<script>bad</script>", "assumptions": {"commission_rate": 0.0003}}
    markdown_path, json_path = write_research_report(summary, manifest, tmp_path / "explicit")
    report = markdown_path.read_text()
    document = json.loads(json_path.read_text())
    assert document["summary"] == summary and document["manifest"] == manifest
    assert "0.123" in report
    assert "<script>" not in report and "&lt;script&gt;" in report and "&#124;" in report
    for required in (
        "不等于组合收益",
        "仓位和资本",
        "回撤",
        "未退出",
        "基本面和板块",
        "历史行业",
        "量价筛选未启用",
        "H &gt; 0",
        "截尾",
    ):
        assert required in report or required.replace("&gt;", ">") in report
    assert "无法核验" in report
    unsafe_summary = summarize_trades([], {"base": "[click](javascript:alert(1))"})
    markdown_path, _ = write_research_report(unsafe_summary, {}, tmp_path / "links")
    assert r"\[click\]\(javascript:alert\(1\)\)" in markdown_path.read_text()


def test_empty_report_and_nonserializable_manifest(tmp_path):
    """场景：无交易报告及含非有限参数；预期：空表不造数，无效输入不创建产物。"""
    summary = summarize_trades([], [])
    markdown_path, _ = write_research_report(summary, {}, tmp_path / "empty")
    assert "逐确认年统计" in markdown_path.read_text()
    with pytest.raises(ValueError):
        write_research_report(summary, {"x": float("nan")}, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("calendar", [[], ["20200101", "20200102", "20200103"]])
def test_report_explicit_coverage_without_dumping_full_calendar(tmp_path, calendar):
    """场景：真实请求窗口、接口缺失及未知领域；预期：中文概览可读，完整日历只留JSON。"""
    manifest = {
        "protocol": {"start_date": "20200102", "end_date": "20200103"},
        "data": {
            "trading_dates": calendar,
            "historical_fundamentals": "unknown",
            "historical_sector": "unknown",
            "partition_coverage": {
                "daily_basic": {
                    "status": "incomplete",
                    "passed_dates": ["20200101"],
                    "missing_dates": ["20200102", "20200103"],
                }
            },
        },
        "counts": {"securities_processed": 5},
        "long_metadata": "<script>" + "x" * 2500,
    }
    markdown_path, json_path = write_research_report(summarize_trades([], []), manifest, tmp_path)
    report = markdown_path.read_text()
    assert "请求确认窗口：20200102 至 20200103" in report
    assert r"daily\_basic | incomplete | 1 | 2" in report
    assert "历史基本面：unknown" in report
    assert "完成证券数：5" in report
    assert "（节选）" in report and "<script>" not in report
    assert "trading_dates" not in report
    assert json.loads(json_path.read_text())["manifest"] == manifest
