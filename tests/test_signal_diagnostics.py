"""完整矩阵分母与独立实验确认差异诊断。"""

import csv
import gzip
import json
from pathlib import Path

import pytest

from tushare_entitlement_hub.signal_research.diagnostics import build_condition_diagnostics
from tushare_entitlement_hub.signal_research.filters import DEFAULT_FILTERS


def fixture(root: Path, *, signals: bool = True) -> None:
    """生成5日矩阵与三实验信号；输入目录输出研究窗内3个有效日、一个无效日。"""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": {"start_date": "20260102", "end_date": "20260105"},
                "experiments": [
                    {"name": "technical_baseline", "event_filter": True},
                    {"name": "without_dif_positive", "event_filter": True},
                    {"name": "without_macd_event", "event_filter": False},
                ]}
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    directory = root / "daily_condition_matrix"
    directory.mkdir()
    rows = []
    for index in range(1, 6):
        row = {"trade_date": f"2026010{index}", "open": 10, "high": 11, "low": 9,
               "close": 10, "vol": 1, "amount": 1, "adj_factor": 1,
               **{rule.name: "TRUE" for rule in DEFAULT_FILTERS}}
        if index == 3:
            row["dif_positive"] = "FALSE"
        if index == 4:
            row["dif_positive"] = "UNKNOWN"
        if index == 5:
            row["close"] = ""
        rows.append(row)
    with gzip.open(directory / "A.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = [
        {"ts_code": "A", "event_id": "e1", "confirmation_date": "20260102",
         "experiment_id": "technical_baseline"},
        {"ts_code": "A", "event_id": "e2", "confirmation_date": "20260104",
         "experiment_id": "technical_baseline"},
        {"ts_code": "A", "event_id": "e1", "confirmation_date": "20260103",
         "experiment_id": "without_dif_positive"},
        {"ts_code": "A", "event_id": "e3", "confirmation_date": "20260104",
         "experiment_id": "without_dif_positive"},
        {"ts_code": "A", "event_id": None, "confirmation_date": "20260102",
         "experiment_id": "without_macd_event"},
        {"ts_code": "A", "event_id": None, "confirmation_date": "20260103",
         "experiment_id": "without_macd_event"},
    ]
    (root / "signal_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records) if signals else ""
    )


def test_full_matrix_counts_overlap_and_event_changes(tmp_path: Path) -> None:
    """全部矩阵日参与而非基线幸存者；单规则失败、未知、事件增删和改期分别计数。"""
    fixture(tmp_path)
    result = json.loads(build_condition_diagnostics(tmp_path).read_text())
    assert result["eligible_days"] == 3
    assert result["excluded_days"] == 1
    assert result["rule_counts"]["dif_positive"] == {
        "TRUE": 1, "FALSE": 1, "UNKNOWN": 1, "pass_rate_all": 1 / 3,
        "pass_rate_known": .5, "only_failed_among_technical": 1,
    }
    overlap = next(row for row in result["pairwise_true_overlap"]
                   if row["left"] == "dif_positive" and row["right"] == "histogram_positive")
    assert overlap["jaccard"] == 1 / 3
    assert overlap["right_true_given_left"] == 1
    event = result["experiment_comparisons"]["without_dif_positive"]
    assert (event["added_count"], event["removed_count"],
            event["changed_confirmation_date_count"]) == (1, 1, 1)
    calendar = result["experiment_comparisons"]["without_macd_event"]
    assert calendar["direct_event_comparison"] is False
    assert calendar["changed_confirmation_date_count"] is None
    assert calendar["added_count"] == 1 and calendar["removed_count"] == 1
    assert (tmp_path / "condition_diagnostics.csv").exists()
    assert (tmp_path / "condition_diagnostics.duckdb").exists()


def test_iso_signal_dates_are_normalized_before_comparison(tmp_path: Path) -> None:
    """模拟器ISO确认日期被DuckDB推断为DATE时仍统一为YYYYMMDD后比较。"""
    fixture(tmp_path)
    path = tmp_path / "signal_records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        date = record["confirmation_date"]
        record["confirmation_date"] = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = json.loads(build_condition_diagnostics(tmp_path).read_text())
    comparison = result["experiment_comparisons"]["without_dif_positive"]
    assert comparison["changed_confirmation_date_count"] == 1
    assert comparison["changed_confirmation_dates"][0][2:] == ["20260102", "20260103"]


def test_empty_signals_and_unknown_zero_denominators(tmp_path: Path) -> None:
    """没有信号仍完整输出；全部UNKNOWN时已知通过率与TRUE重合Jaccard未知。"""
    fixture(tmp_path, signals=False)
    path = tmp_path / "daily_condition_matrix" / "A.csv.gz"
    with gzip.open(path, "rt") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for rule in DEFAULT_FILTERS:
            row[rule.name] = "UNKNOWN"
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = json.loads(build_condition_diagnostics(tmp_path).read_text())
    assert result["rule_counts"]["dif_positive"]["pass_rate_known"] is None
    assert result["pairwise_true_overlap"][0]["jaccard"] is None
    assert result["experiment_comparisons"]["without_dif_positive"]["baseline_count"] == 0


@pytest.mark.parametrize("failure", ["missing", "fields", "state", "duplicate", "event", "name"])
def test_invalid_matrix_or_signal_contract(tmp_path: Path, failure: str) -> None:
    """缺矩阵/字段、非法三态、重复信号、缺事件标识和畸形实验名均失败。"""
    fixture(tmp_path)
    path = tmp_path / "daily_condition_matrix" / "A.csv.gz"
    if failure == "missing":
        path.unlink()
    elif failure in {"fields", "state"}:
        with gzip.open(path, "rt") as stream:
            rows = list(csv.DictReader(stream))
        if failure == "fields":
            for row in rows:
                del row["dif_positive"]
        else:
            rows[1]["dif_positive"] = "MAYBE"
        with gzip.open(path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif failure in {"duplicate", "event"}:
        path = tmp_path / "signal_records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "duplicate":
            records.append(records[0])
        else:
            records[0]["event_id"] = None
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    else:
        path = tmp_path / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["experiments"][1]["name"] = "without_bad'"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_condition_diagnostics(tmp_path)


def test_bounded_diagnostics_memory_and_private_spill_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式低内存限制仍生成诊断，私有溢写目录最终清理，不保存研究窗外矩阵。"""
    import duckdb

    fixture(tmp_path)
    private_home = tmp_path / "private-home"
    monkeypatch.setattr(Path, "home", lambda: private_home)
    destination = build_condition_diagnostics(tmp_path, memory_limit="64MB")
    assert destination.exists()
    spill_root = private_home / ".cache" / "tushare-hub"
    assert not list(spill_root.glob("condition-diagnostics-spill-*"))
    database = tmp_path / "condition_diagnostics.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT min(trade_date) FROM matrix").fetchone() == ("20260102",)
        tables = connection.execute("SELECT table_name FROM information_schema.tables "
                                    "WHERE table_type='BASE TABLE'").fetchall()
    assert ("eligible",) not in tables and ("research_matrix",) not in tables
