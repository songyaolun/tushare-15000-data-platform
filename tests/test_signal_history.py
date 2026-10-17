"""按公开合成原始分区验证历史数据库隔离、质量门和有界加载。"""

import json
import stat
from pathlib import Path

import duckdb
import pytest

from tushare_entitlement_hub.research_history.collection import FIELDS, partition_quality
from tushare_entitlement_hub.signal_research.history import (
    HISTORY_APIS,
    assert_complete_history,
    create_security_panel,
    iter_security_history,
    load_history_database,
)


def write_partition(
    root: Path, api_name: str, *, name: str = "one", date: str = "20260105",
    codes: tuple[str, ...] = ("000001.SZ",), failed: bool = False,
) -> Path:
    """生成标准collector包络；输入daily输出含合格质量摘要的公开合成JSON路径。"""
    fields = FIELDS[api_name].split(",")
    response = {"code": 0, "data": {"fields": fields, "items": [
        [code if field == "ts_code" else date if field == "trade_date" else 1.0
         for field in fields] for code in codes
    ]}}
    envelope = {"api_name": api_name, "parameters": {"trade_date": date},
                "observed_at": "2026-10-17", "response": response,
                "quality": partition_quality(api_name, {"trade_date": date}, response)}
    if failed:
        envelope["quality"]["status"] = "failed"
    path = root / "raw" / api_name / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope))
    return path


def write_calendar(root: Path, dates: list[str] | None = None) -> None:
    """写公开交易日历；输入None输出默认两个唯一YYYYMMDD日期。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "trading_dates.json").write_text(json.dumps(
        ["20260105", "20260106"] if dates is None else dates
    ))


def test_load_all_security_history_and_source_units(tmp_path: Path) -> None:
    """所有源daily证券包括退市代码保留，缺失关联字段未知，源单位与人民币派生值分开。"""
    write_calendar(tmp_path)
    for api_name in HISTORY_APIS:
        write_partition(tmp_path, api_name)
    write_partition(tmp_path, "daily", name="second", date="20260106",
                    codes=("000001.SZ", "DELISTED.SZ"))
    write_partition(tmp_path, "daily", name="failed", date="20260106", failed=True)
    path = tmp_path / "cache.duckdb"
    manifest = load_history_database(tmp_path, path)
    assert manifest["raw_file_count"] == 6 and manifest["skipped_file_count"] == 1
    assert manifest["api_counts"]["daily"] == 3
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with duckdb.connect(str(path), read_only=True) as connection:
        rows = list(iter_security_history(connection))
        stored_manifest = connection.execute("SELECT manifest FROM cache_manifest").fetchone()
    assert stored_manifest is not None
    assert json.loads(stored_manifest[0]) == manifest
    assert [code for code, _ in rows] == ["000001.SZ", "DELISTED.SZ"]
    first = rows[0][1][0]
    assert first["amount"] == 1 and first["amount_yuan"] == 1000
    assert first["circ_mv"] == 1 and first["floating_market_value_yuan"] == 10000
    assert first["sector"] is None and first["fundamentals"] is None
    assert rows[1][1][0]["adj_factor"] is None
    assert rows[1][1][0]["floating_market_value_yuan"] is None
    assert manifest == load_history_database(tmp_path, path)


def test_bulk_partition_load_and_empty_tables(tmp_path: Path) -> None:
    """单页超过插入批次时完整载入，不需要其他接口有页，不物化全市场历史。"""
    write_calendar(tmp_path)
    write_partition(tmp_path, "daily", codes=tuple(f"{index:06d}.SZ" for index in range(1001)))
    path = tmp_path / "cache.duckdb"
    assert load_history_database(tmp_path, path)["api_counts"]["daily"] == 1001
    with duckdb.connect(str(path), read_only=True) as connection:
        generator = iter_security_history(connection)
        assert next(generator)[0] == "000000.SZ"
        generator.close()
    empty_root = tmp_path / "empty"
    write_calendar(empty_root)
    empty_path = empty_root / "cache.duckdb"
    load_history_database(empty_root, empty_path)
    with duckdb.connect(str(empty_path)) as connection:
        assert list(iter_security_history(connection)) == []


def test_duplicate_keys_fail_and_keep_old_database(tmp_path: Path) -> None:
    """跨分区重复业务键直接失败，已发布缓存完整保留，临时数据库清理。"""
    write_calendar(tmp_path)
    write_partition(tmp_path, "daily")
    path = tmp_path / "cache.duckdb"
    load_history_database(tmp_path, path)
    original = path.read_bytes()
    write_partition(tmp_path, "daily", name="duplicate")
    with pytest.raises(ValueError, match="insert_contract"):
        load_history_database(tmp_path, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("signal-history-*"))
    assert not list(tmp_path.glob("signal-partition-*"))


@pytest.mark.parametrize("mutation,expected", [
    ("api", "identity"), ("response", "identity"), ("quality", "quality_mismatch"),
    ("fields", "quality_mismatch"), ("key", "key_invalid"),
    ("duplicate", "quality_mismatch"), ("outside", "outside_calendar"),
    ("nondict", "envelope"), ("malformed", "public_json"), ("secret", "凭据"),
])
def test_corrupt_partitions_fail_closed(tmp_path: Path, mutation: str, expected: str) -> None:
    """伪造通过质量、坏字段、重复键、日历外日期和敏感字段均不得发布缓存。"""
    write_calendar(tmp_path)
    path = write_partition(tmp_path, "daily")
    value = json.loads(path.read_text())
    if mutation == "api":
        value["api_name"] = "adj_factor"
    elif mutation == "response":
        value["response"] = []
    elif mutation == "quality":
        value["quality"]["rows"] = 99
    elif mutation == "fields":
        value["response"]["data"]["fields"].pop()
        value["response"]["data"]["items"][0].pop()
    elif mutation == "key":
        value["response"]["data"]["items"][0][0] = None
        value["quality"] = partition_quality("daily", value["parameters"], value["response"])
    elif mutation == "duplicate":
        value["response"]["data"]["items"] *= 2
    elif mutation == "outside":
        value["response"]["data"]["items"][0][1] = "20260107"
        value["parameters"]["trade_date"] = "20260107"
        value["quality"] = partition_quality("daily", value["parameters"], value["response"])
    elif mutation == "nondict":
        value = []
    elif mutation == "secret":
        value["token"] = "SYNTHETIC_SECRET"
    path.write_text("{" if mutation == "malformed" else json.dumps(value))
    with pytest.raises(ValueError, match=expected):
        load_history_database(tmp_path, tmp_path / "cache.duckdb")
    assert not (tmp_path / "cache.duckdb").exists()


@pytest.mark.parametrize("value", [[], ["bad"], ["20260105", "20260105"], {}, [1]])
def test_calendar_contract_failures(tmp_path: Path, value: object) -> None:
    """空、非日期、重复和畸形交易日历不能证明确认窗口，载入应拒绝。"""
    (tmp_path / "trading_dates.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="calendar"):
        load_history_database(tmp_path, tmp_path / "cache.duckdb")


def test_calendar_object_and_missing_file(tmp_path: Path) -> None:
    """支持trading_dates对象，缺失日历用固定异常且不暴露任意输入。"""
    with pytest.raises(ValueError, match="public_json"):
        load_history_database(tmp_path, tmp_path / "cache.duckdb")
    (tmp_path / "trading_dates.json").write_text(json.dumps({"trading_dates": ["20260105"]}))
    manifest = load_history_database(tmp_path, tmp_path / "cache.duckdb")
    assert manifest["trading_dates"] == ["20260105"]


def test_core_partition_completeness(tmp_path: Path) -> None:
    """正式研究必须逐日有核心分区，daily_basic缺失显式展示但不阻塞核心校验。"""
    write_calendar(tmp_path)
    for api_name in ("daily", "adj_factor", "stk_limit"):
        for date in ("20260105", "20260106"):
            write_partition(tmp_path, api_name, name=date, date=date)
    path = tmp_path / "cache.duckdb"
    manifest = load_history_database(tmp_path, path)
    assert_complete_history(manifest)
    assert manifest["partition_coverage"]["daily_basic"]["status"] == "incomplete"
    assert manifest["partition_coverage"]["daily_basic"]["missing_dates"] == [
        "20260105", "20260106",
    ]
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_history(manifest, ("daily_basic",))
    with pytest.raises(ValueError, match="invalid"):
        assert_complete_history(manifest, ("other",))
    with pytest.raises(ValueError, match="invalid"):
        assert_complete_history({"trading_dates": ["20260105"]})
    (tmp_path / "raw" / "adj_factor" / "20260106.json").unlink()
    manifest = load_history_database(tmp_path, path)
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_history(manifest)
    assert_complete_history(manifest, ("daily",))


def test_loading_progress_every_500_partitions_and_api_completion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """500个分区输出即时处理进度，接口完成包含跳过计数，空接口也报告完成。"""
    write_calendar(tmp_path)
    for index in range(500):
        write_partition(tmp_path, "daily", name=f"{index:03d}", failed=True)
    manifest = load_history_database(tmp_path, tmp_path / "cache.duckdb")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "[history] daily: processed_partitions=500 loaded_rows=0 skipped_partitions=500",
        "[history] daily complete: processed_partitions=500 loaded_rows=0 skipped_partitions=500",
        "[history] adj_factor complete: processed_partitions=0 loaded_rows=0 skipped_partitions=0",
        "[history] daily_basic complete: processed_partitions=0 loaded_rows=0 skipped_partitions=0",
        "[history] stk_limit complete: processed_partitions=0 loaded_rows=0 skipped_partitions=0",
    ]
    assert manifest["skipped_file_count"] == 500


def test_private_csv_copy_preserves_nulls_dates_and_cleans_up(tmp_path: Path) -> None:
    """CSV COPY保留证券代码/日期文本和数值NULL；成功后临时文件全部清理。"""
    write_calendar(tmp_path)
    write_partition(tmp_path, "daily")
    path = write_partition(tmp_path, "daily_basic")
    envelope = json.loads(path.read_text())
    fields = envelope["response"]["data"]["fields"]
    for field in ("turnover_rate", "circ_mv"):
        envelope["response"]["data"]["items"][0][fields.index(field)] = None
    envelope["quality"] = partition_quality(
        "daily_basic", envelope["parameters"], envelope["response"],
    )
    path.write_text(json.dumps(envelope))
    database = tmp_path / "cache.duckdb"
    load_history_database(tmp_path, database)
    with duckdb.connect(str(database)) as connection:
        code, rows = next(iter_security_history(connection))
    assert code == "000001.SZ"
    assert rows[0]["trade_date"] == "20260105"
    assert rows[0]["turnover_rate"] is None
    assert rows[0]["circ_mv"] is None
    assert rows[0]["floating_market_value_yuan"] is None
    assert not list(tmp_path.glob("signal-partition-*"))


def test_sorted_panel_matches_legacy_history_and_cleans_spill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """排序面板保留全部源daily和缺失关联NULL，迭代输出与旧四表路径完全一致。"""
    write_calendar(tmp_path)
    for api_name in HISTORY_APIS:
        write_partition(tmp_path, api_name)
    write_partition(tmp_path, "daily", name="delisted", date="20260106", codes=("DELISTED.SZ",))
    path = tmp_path / "cache.duckdb"
    load_history_database(tmp_path, path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "private-home")
    with duckdb.connect(str(path)) as connection:
        before = list(iter_security_history(connection))
        assert connection.execute("SELECT current_setting('index_scan_max_count')").fetchone() \
            == (2048,)
        manifest = create_security_panel(connection, memory_limit="64MB")
        assert manifest["rows"] == 2 and manifest["securities"] == 2
        assert before == list(iter_security_history(connection))
        assert connection.execute("SELECT current_setting('index_scan_max_count')").fetchone() \
            == (0,)
        assert connection.execute("SELECT current_setting('index_scan_percentage')").fetchone() \
            == (0.0,)
        assert manifest == create_security_panel(connection, memory_limit="64MB")
    assert not list((tmp_path / "private-home" / ".cache" / "tushare-hub").glob(
        "security-panel-spill-*"
    ))


def test_panel_duplicate_join_failure_rolls_back(tmp_path: Path) -> None:
    """非法源表重复键导致连接行数增加时，面板失败回滚且旧面板仍可读取。"""
    with duckdb.connect() as connection:
        for api_name in HISTORY_APIS:
            columns = ",".join(
                f'"{field}" {"VARCHAR" if field in {"ts_code", "trade_date"} else "DOUBLE"}'
                for field in FIELDS[api_name].split(",")
            )
            connection.execute(f"CREATE TABLE {api_name} ({columns})")
        connection.execute("INSERT INTO daily VALUES ('A','20260105',1,1,1,1,1,1,1,1,1)")
        create_security_panel(connection)
        connection.execute("INSERT INTO adj_factor VALUES ('A','20260105',1),('A','20260105',2)")
        with pytest.raises(ValueError, match="row_count"):
            create_security_panel(connection)
        assert connection.execute("SELECT count(*) FROM security_panel").fetchone() == (1,)
