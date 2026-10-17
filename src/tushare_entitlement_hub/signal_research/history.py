"""将合格历史分区逐批载入私有DuckDB，不以内存全量加载市场历史。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb

from ..raw import assert_public, canonical_json
from ..research_history.collection import FIELDS, partition_quality, response_records

HISTORY_APIS = ("daily", "adj_factor", "daily_basic", "stk_limit")


def _read_public_json(path: Path) -> Any:
    """读取公开JSON且拒绝凭据后才允许摘要；输入普通对象返回对象，畸形抛固定异常。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("history_public_json_invalid") from None
    assert_public(value)
    return value


def _calendar_dates(value: Any) -> list[str]:
    """核验唯一YYYYMMDD交易日；输入['20260101']输出同列表，也支持trading_dates对象。"""
    dates = value.get("trading_dates") if isinstance(value, dict) else value
    if (not isinstance(dates, list) or not dates
            or any(not isinstance(date, str) or not re.fullmatch(r"\d{8}", date)
                   for date in dates) or len(set(dates)) != len(dates)):
        raise ValueError("history_calendar_invalid")
    return sorted(dates)


def _load_partition(
    connection: duckdb.DuckDBPyConnection, api_name: str, envelope: Any,
    temporary_directory: Path,
) -> int | None:
    """重新核验通过分区，以私有临时CSV整页COPY；失败质量页输出None，重复键抛异常。

    输入collector标准envelope和数据库所在临时目录，输出实际入库行数。
    CSV权限0600、每分区限提供方行上限、finally删除；None按空字段映射SQL NULL。
    源单位不在载入时转换；例：adj_factor=None仍保留NULL而不是0。
    """
    if not isinstance(envelope, dict):
        raise ValueError("history_envelope_invalid")
    quality = envelope.get("quality")
    if not isinstance(quality, dict) or quality.get("status") != "passed":
        return None
    if (envelope.get("api_name") != api_name
            or not isinstance(envelope.get("parameters"), dict)
            or not isinstance(envelope.get("response"), dict)):
        raise ValueError("history_partition_identity_invalid")
    calculated = partition_quality(api_name, envelope["parameters"], envelope["response"])
    if calculated != quality or calculated["status"] != "passed":
        raise ValueError("history_partition_quality_mismatch")
    records = response_records(envelope["response"])
    fields = FIELDS[api_name].split(",")
    for record in records:
        if (not isinstance(record["ts_code"], str) or not record["ts_code"]
                or not isinstance(record["trade_date"], str)
                or not re.fullmatch(r"\d{8}", record["trade_date"])):
            raise ValueError("history_partition_key_invalid")
    temporary_csv: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix="signal-partition-", dir=temporary_directory, delete=False,
        ) as stream:
            temporary_csv = Path(stream.name)
            csv.writer(stream).writerows([record[field] for field in fields] for record in records)
        columns = ",".join(f'"{field}"' for field in fields)
        connection.execute(
            f"COPY {api_name} ({columns}) FROM ? "
            "(FORMAT CSV, HEADER FALSE, NULLSTR '', AUTO_DETECT FALSE)",
            [str(temporary_csv)],
        )
    except duckdb.Error:
        raise ValueError("history_partition_insert_contract_failed") from None
    finally:
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)
    return len(records)


def load_history_database(root: Path, database_path: Path) -> dict[str, Any]:
    """逐文件核验root/raw四接口JSON并原子发布权限0600的私有DuckDB缓存。

    输入采集根与目标路径；输出raw_file_count、raw_fingerprint、api_counts、
    trading_dates及database_path清单。摘要只在assert_public通过后计算，不读取凭据。
    例：一份daily页和日历载入后api_counts['daily']=1；所有源daily证券均保留。
    同一证券日期跨分区重复、字段质量损坏或日历外日期均失败，旧缓存保持不变。
    每次重建并原子替换；一次Python内存只包含一个源分区，通过私有CSV整页COPY。
    每500个已处理分区及每接口完成时输出公开计数进度，不打印源内容或文件路径。
    """
    root, database_path = Path(root), Path(database_path)
    calendar_path = root / "trading_dates.json"
    calendar_value = _read_public_json(calendar_path)
    dates = _calendar_dates(calendar_value)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="signal-history-", suffix=".duckdb", dir=database_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    digest = hashlib.sha256(canonical_json(calendar_value).encode())
    counts = dict.fromkeys(HISTORY_APIS, 0)
    partition_dates: dict[str, set[str]] = {api_name: set() for api_name in HISTORY_APIS}
    raw_file_count = skipped_file_count = 0
    try:
        with duckdb.connect(str(temporary)) as connection:
            os.chmod(temporary, 0o600)
            connection.execute("SET threads=1")
            connection.execute("BEGIN TRANSACTION")
            connection.execute("CREATE TABLE trading_calendar(date VARCHAR PRIMARY KEY)")
            connection.executemany("INSERT INTO trading_calendar VALUES (?)",
                                   [(date,) for date in dates])
            for api_name in HISTORY_APIS:
                processed_partitions = 0
                skipped_partitions = 0
                columns = ",".join(
                    f'"{field}" {"VARCHAR" if field in {"ts_code", "trade_date"} else "DOUBLE"}'
                    for field in FIELDS[api_name].split(",")
                )
                connection.execute(f"CREATE TABLE {api_name} ({columns}, "
                                   "PRIMARY KEY(ts_code, trade_date))")
                for path in sorted((root / "raw" / api_name).glob("*.json")):
                    envelope = _read_public_json(path)
                    digest.update(canonical_json({"api": api_name, "file": path.name,
                                                  "envelope": envelope}).encode())
                    raw_file_count += 1
                    rows = _load_partition(connection, api_name, envelope, database_path.parent)
                    if rows is None:
                        skipped_file_count += 1
                        skipped_partitions += 1
                    else:
                        counts[api_name] += rows
                        partition_date = envelope["parameters"].get("trade_date")
                        if partition_date in dates and rows:
                            partition_dates[api_name].add(partition_date)
                    processed_partitions += 1
                    if processed_partitions % 500 == 0:
                        print(
                            f"[history] {api_name}: processed_partitions={processed_partitions} "
                            f"loaded_rows={counts[api_name]} "
                            f"skipped_partitions={skipped_partitions}",
                            flush=True,
                        )
                outside_calendar = connection.execute(
                    f"SELECT count(*) FROM {api_name} WHERE trade_date NOT IN "
                    "(SELECT date FROM trading_calendar)"
                ).fetchone()
                if outside_calendar and outside_calendar[0]:
                    raise ValueError("history_partition_date_outside_calendar")
                print(
                    f"[history] {api_name} complete: "
                    f"processed_partitions={processed_partitions} loaded_rows={counts[api_name]} "
                    f"skipped_partitions={skipped_partitions}",
                    flush=True,
                )
            manifest = {
                "version": "signal_history_v1", "database_path": str(database_path),
                "raw_file_count": raw_file_count, "skipped_file_count": skipped_file_count,
                "raw_fingerprint": digest.hexdigest(), "api_counts": counts,
                "trading_dates": dates,
                "source_units": {"amount": "千元", "circ_mv": "万元", "vol": "手"},
                "historical_fundamentals": "unknown", "historical_sector": "unknown",
                "partition_coverage": {
                    api_name: {
                        "passed_dates": sorted(partition_dates[api_name]),
                        "missing_dates": sorted(set(dates) - partition_dates[api_name]),
                        "status": "complete" if set(dates) == partition_dates[api_name]
                        else "incomplete",
                    } for api_name in HISTORY_APIS
                },
            }
            connection.execute("CREATE TABLE cache_manifest(manifest JSON)")
            connection.execute("INSERT INTO cache_manifest VALUES (?)", [canonical_json(manifest)])
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        temporary.replace(database_path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + ".wal").unlink(missing_ok=True)
    return manifest


def assert_complete_history(
    manifest: dict[str, Any],
    required_apis: tuple[str, ...] = ("daily", "adj_factor", "stk_limit"),
) -> None:
    """核验核心接口每个日历日都有非空且质量通过的独立日分区，缺失则拒绝正式研究。

    输入load_history_database清单；全部日分区完整输出None，任一缺失抛固定ValueError。
    示例：只有daily时默认检查失败；required_apis=('daily',)且daily日分区完整时通过。
    daily_basic默认不作为核心阻塞项，但清单始终列明其逐日缺失，不能声称覆盖完整。
    """
    dates = _calendar_dates(manifest.get("trading_dates"))
    coverage = manifest.get("partition_coverage")
    if not isinstance(coverage, dict) or any(api not in HISTORY_APIS for api in required_apis):
        raise ValueError("history_core_partition_coverage_invalid")
    for api_name in required_apis:
        entry = coverage.get(api_name)
        if (not isinstance(entry, dict) or entry.get("status") != "complete"
                or entry.get("missing_dates") != []
                or entry.get("passed_dates") != dates):
            raise ValueError("history_core_partition_coverage_incomplete")


def iter_security_history(
    connection: duckdb.DuckDBPyConnection,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """按源daily全部证券逐只读取四表左连接历史，不使用当前上市名单和行业回填。

    输入load_history_database产生的DuckDB连接；输出(code,按日期排序记录列表)。
    例：daily有已退市A且adj缺失，仍输出A、adj_factor=None、fundamentals=None。
    amount/circ_mv保留源单位，amount_yuan/floating_market_value_yuan分别乘1000/10000。
    单次仅物化一只证券历史；股票代码清单在DuckDB独立游标逐条迭代。
    """
    panel_exists = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' "
        "AND table_name='security_panel'"
    ).fetchone()
    table_name = "security_panel" if panel_exists and panel_exists[0] else "daily"
    code_cursor = connection.cursor()
    try:
        code_cursor.execute(f"SELECT DISTINCT ts_code FROM {table_name} ORDER BY ts_code")
        while (entry := code_cursor.fetchone()) is not None:
            code = str(entry[0])
            query = "SELECT * FROM security_panel WHERE ts_code=? ORDER BY trade_date" \
                if table_name == "security_panel" else (
                "SELECT d.*, a.adj_factor, b.turnover_rate, b.turnover_rate_f, "
                "b.circ_mv, b.total_mv, l.up_limit, l.down_limit, "
                "d.amount * 1000 AS amount_yuan, "
                "b.circ_mv * 10000 AS floating_market_value_yuan "
                "FROM daily d LEFT JOIN adj_factor a USING(ts_code, trade_date) "
                "LEFT JOIN daily_basic b USING(ts_code, trade_date) "
                "LEFT JOIN stk_limit l USING(ts_code, trade_date) "
                "WHERE d.ts_code=? ORDER BY d.trade_date"
            )
            result = connection.execute(query, [code])
            columns = [column[0] for column in result.description]
            rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
            for row in rows:
                row.update(fundamentals=None, sector=None)
            yield code, rows
    finally:
        code_cursor.close()


def create_security_panel(
    connection: duckdb.DuckDBPyConnection, *, memory_limit: str = "512MB",
) -> dict[str, Any]:
    """一次性连接历史四表并按证券日期排序，避免6000证券各自重复扫描全市场源表。

    输入可写私有历史数据库连接；输出面板行数、证券数、来源语义和内存限制清单。
    例：daily有A两日、B一日，输出rows=3、securities=2，缺失adj/basic仍保留NULL。
    全源daily证券保留，源单位与人民币派生字段同时保存，不回填基本面和历史行业。
    memory_limit默认512MB，使用独立私有缓存溢写目录并在结束清理；调用方不得有活动reader。
    一次交易内替换面板、构建单列证券索引并CHECKPOINT，失败回滚旧面板。
    """
    spill_root = Path.home() / ".cache" / "tushare-hub"
    spill_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="security-panel-spill-", dir=spill_root) as spill:
        connection.execute("SET memory_limit=?", [memory_limit])
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET temp_directory=?", [spill])
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("DROP TABLE IF EXISTS security_panel")
            connection.execute(
                "CREATE TABLE security_panel AS SELECT d.*,a.adj_factor,b.turnover_rate,"
                "b.turnover_rate_f,b.circ_mv,b.total_mv,l.up_limit,l.down_limit,"
                "d.amount*1000 AS amount_yuan,"
                "b.circ_mv*10000 AS floating_market_value_yuan "
                "FROM daily d LEFT JOIN adj_factor a USING(ts_code,trade_date) "
                "LEFT JOIN daily_basic b USING(ts_code,trade_date) "
                "LEFT JOIN stk_limit l USING(ts_code,trade_date) ORDER BY d.ts_code,d.trade_date"
            )
            connection.execute("CREATE INDEX security_panel_code_index ON security_panel(ts_code)")
            counts = connection.execute(
                "SELECT count(*),count(DISTINCT ts_code) FROM security_panel"
            ).fetchone()
            assert counts is not None
            source_count = connection.execute("SELECT count(*) FROM daily").fetchone()
            assert source_count is not None
            if counts[0] != source_count[0]:
                raise ValueError("security_panel_source_row_count_mismatch")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("CHECKPOINT")
    return {"rows": counts[0], "securities": counts[1], "memory_limit": memory_limit,
            "ordering": ["ts_code", "trade_date"], "historical_fundamentals": "unknown",
            "historical_sector": "unknown"}
