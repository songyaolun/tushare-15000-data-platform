"""对完整日矩阵和全部实验信号做无权重条件诊断。"""

from __future__ import annotations

import csv
import json
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import duckdb

from ..raw import assert_public, canonical_json
from .filters import DEFAULT_FILTERS, TECHNICAL_FILTER_NAMES


def _ratio(numerator: int, denominator: int) -> float | None:
    """计算明确分母的比率；输入1,2输出0.5，分母0输出None。"""
    return numerator / denominator if denominator else None


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    """读取整数聚合结果；输入SELECT count(*)输出实际行数。"""
    row = connection.execute(query).fetchone()
    assert row is not None
    return int(row[0])


def _signal_comparison(
    connection: duckdb.DuckDBPyConnection, variant: str, event_filter: bool,
) -> dict[str, Any]:
    """按事件或全日历确认键比较基线与变体，不推断不同定义之间的因果关系。

    输入without_dif_positive事件实验输出added/removed/changed_date计数与明细。
    event_filter=False只按(code,date)做差集并明确不能按事件匹配。
    """
    connection.execute("CREATE OR REPLACE TEMP VIEW baseline AS SELECT * FROM signals "
                       "WHERE experiment_id='technical_baseline'")
    connection.execute("CREATE OR REPLACE TEMP TABLE variant AS SELECT * FROM signals "
                       "WHERE experiment_id=?", [variant])
    baseline_count = _scalar(connection, "SELECT count(*) FROM baseline")
    variant_count = _scalar(connection, "SELECT count(*) FROM variant")
    if event_filter:
        if _scalar(connection, "SELECT count(*) FROM (SELECT * FROM baseline UNION ALL "
                   "SELECT * FROM variant) WHERE event_id IS NULL OR event_id=''"):
            raise ValueError("diagnostics_event_id_missing")
        keys = "ts_code,event_id"
    else:
        keys = "ts_code,confirmation_date"
    added = connection.execute(
        f"SELECT {keys}, confirmation_date FROM variant ANTI JOIN baseline USING({keys})"
        if event_filter else
        f"SELECT {keys} FROM variant ANTI JOIN baseline USING({keys})"
    ).fetchall()
    removed = connection.execute(
        f"SELECT {keys}, confirmation_date FROM baseline ANTI JOIN variant USING({keys})"
        if event_filter else
        f"SELECT {keys} FROM baseline ANTI JOIN variant USING({keys})"
    ).fetchall()
    changed = connection.execute(
        "SELECT b.ts_code,b.event_id,b.confirmation_date,v.confirmation_date "
        "FROM baseline b JOIN variant v USING(ts_code,event_id) "
        "WHERE b.confirmation_date<>v.confirmation_date"
    ).fetchall() if event_filter else []
    return {
        "baseline_count": baseline_count, "variant_count": variant_count,
        "matching": "security_event" if event_filter else "security_confirmation_date",
        "direct_event_comparison": event_filter,
        "added_count": len(added), "removed_count": len(removed),
        "changed_confirmation_date_count": len(changed) if event_filter else None,
        "added": added, "removed": removed, "changed_confirmation_dates": changed,
        "interpretation": "descriptive_confirmation_difference_no_causality"
        if event_filter else "different_calendar_definition_not_directly_event_matchable",
    }


def build_condition_diagnostics(
    output_directory: Path, *, memory_limit: str = "512MB",
) -> Path:
    """磁盘DuckDB读取本次全部gzip矩阵，生成研究窗口三态/重合/消融确认诊断。

    输入含run_manifest.json、daily_condition_matrix/*.csv.gz、signal_records.jsonl的目录。
    输出condition_diagnostics.json路径，同时保存简短CSV与磁盘审计DuckDB。
    例：TRUE/FALSE/UNKNOWN各1时pass_rate_all=1/3、pass_rate_known=1/2。
    只统计源OHLC合法且close/vol/amount/adj_factor正有限的真实可计算日，保留未知预热。
    分母来自完整矩阵而非基线幸存者；不使用未来收益或自拟权重。
    memory_limit默认512MB；独立私有缓存溢写目录结束时清理，研究窗只物化一次。
    """
    output_directory = Path(output_directory)
    manifest = json.loads((output_directory / "run_manifest.json").read_text())
    assert_public(manifest)
    protocol = manifest["protocol"]
    start, end = protocol["start_date"], protocol["end_date"]
    rules = tuple(definition.name for definition in DEFAULT_FILTERS)
    files = sorted((output_directory / "daily_condition_matrix").glob("*.csv.gz"))
    if not files:
        raise ValueError("diagnostics_matrix_missing")
    database = output_directory / "condition_diagnostics.duckdb"
    database.unlink(missing_ok=True)
    spill_root = Path.home() / ".cache" / "tushare-hub"
    spill_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="condition-diagnostics-spill-", dir=spill_root) \
            as spill_directory, duckdb.connect(str(database)) as connection:
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit=?", [memory_limit])
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET temp_directory=?", [spill_directory])
        connection.execute(
            "CREATE TABLE matrix AS SELECT * FROM read_csv(?, header=true, "
            "all_varchar=true, union_by_name=true, filename=true, buffer_size=1048576) "
            "WHERE trade_date BETWEEN ? AND ?",
            [[str(path) for path in files], start, end],
        )
        columns = {row[0] for row in connection.execute("DESCRIBE matrix").fetchall()}
        required = {"trade_date", "open", "high", "low", "close", "vol", "amount",
                    "adj_factor", *rules}
        if not required <= columns:
            raise ValueError("diagnostics_matrix_fields_missing")
        for rule in rules:
            if _scalar(connection, f"SELECT count(*) FROM matrix WHERE \"{rule}\" IS NULL "
                       f"OR \"{rule}\" NOT IN ('TRUE','FALSE','UNKNOWN')"):
                raise ValueError("diagnostics_rule_state_invalid")
        connection.execute("CREATE VIEW research_matrix AS SELECT * FROM matrix")
        positive = " AND ".join(
            f"try_cast(\"{field}\" AS DOUBLE)>0 AND "
            f"isfinite(try_cast(\"{field}\" AS DOUBLE))"
            for field in ("open", "high", "low", "close", "vol", "amount", "adj_factor")
        )
        # Parameterized values cannot be captured in a persistent DuckDB view.
        connection.execute("CREATE VIEW eligible AS SELECT * FROM research_matrix WHERE "
                           + positive + " AND try_cast(low AS DOUBLE)<= "
                           "least(try_cast(open AS DOUBLE),try_cast(close AS DOUBLE)) "
                           "AND try_cast(high AS DOUBLE)>= "
                           "greatest(try_cast(open AS DOUBLE),try_cast(close AS DOUBLE))")
        eligible_count = _scalar(connection, "SELECT count(*) FROM eligible")
        rule_counts = {}
        for rule in rules:
            counts = dict(connection.execute(
                f'SELECT "{rule}",count(*) FROM eligible GROUP BY "{rule}"'
            ).fetchall())
            true, false, unknown = (int(counts.get(state, 0))
                                    for state in ("TRUE", "FALSE", "UNKNOWN"))
            only_failed = None
            if rule in TECHNICAL_FILTER_NAMES:
                others_true = " AND ".join(f'"{other}"=\'TRUE\''
                                           for other in TECHNICAL_FILTER_NAMES if other != rule)
                only_failed = _scalar(connection, f'SELECT count(*) FROM eligible '
                                      f'WHERE "{rule}"=\'FALSE\' AND {others_true}')
            rule_counts[rule] = {"TRUE": true, "FALSE": false, "UNKNOWN": unknown,
                                 "pass_rate_all": _ratio(true, eligible_count),
                                 "pass_rate_known": _ratio(true, true + false),
                                 "only_failed_among_technical": only_failed}
        overlaps = []
        for left, right in combinations(rules, 2):
            overlap_counts = connection.execute(
                f'SELECT count(*) FILTER(WHERE "{left}"=\'TRUE\' AND "{right}"=\'TRUE\'),'
                f'count(*) FILTER(WHERE "{left}"=\'TRUE\' OR "{right}"=\'TRUE\'),'
                f'count(*) FILTER(WHERE "{left}"=\'TRUE\'),'
                f'count(*) FILTER(WHERE "{right}"=\'TRUE\') FROM eligible'
            ).fetchone()
            assert overlap_counts is not None
            intersection, union, left_count, right_count = overlap_counts
            overlaps.append({"left": left, "right": right, "both_true": intersection,
                             "either_true": union, "jaccard": _ratio(intersection, union),
                             "both_true_rate_all": _ratio(intersection, eligible_count),
                             "right_true_given_left": _ratio(intersection, left_count),
                             "left_true_given_right": _ratio(intersection, right_count)})
        signal_path = output_directory / "signal_records.jsonl"
        if signal_path.stat().st_size:
            connection.execute(
                "CREATE TABLE signals AS SELECT ts_code,experiment_id,event_id,"
                "replace(cast(confirmation_date AS VARCHAR),'-','') AS confirmation_date "
                "FROM read_json_auto(?,format='newline_delimited') "
                "WHERE replace(cast(confirmation_date AS VARCHAR),'-','') BETWEEN ? AND ?",
                [str(signal_path), start, end],
            )
            duplicates = _scalar(
                connection, "SELECT count(*) FROM (SELECT ts_code,experiment_id,"
                "confirmation_date,count(*) n FROM signals GROUP BY ALL HAVING n>1)",
            )
            if duplicates:
                raise ValueError("diagnostics_duplicate_signal")
        else:
            connection.execute("CREATE TABLE signals(ts_code VARCHAR,experiment_id VARCHAR,"
                               "event_id VARCHAR,confirmation_date VARCHAR)")
        comparisons = {}
        for experiment in manifest["experiments"]:
            name = experiment["name"]
            if name.startswith("without_"):
                # Names come from the local manifest; reject SQL identifier injection.
                if not name.replace("_", "").isalnum():
                    raise ValueError("diagnostics_experiment_name_invalid")
                comparisons[name] = _signal_comparison(
                    connection, name, experiment["event_filter"],
                )
        result = {
            "protocol": {"start_date": start, "end_date": end},
            "matrix_file_count": len(files), "eligible_days": eligible_count,
            "research_matrix_days": _scalar(connection, "SELECT count(*) FROM research_matrix"),
            "excluded_days": _scalar(connection, "SELECT count(*) FROM research_matrix")
            - eligible_count,
            "denominator": "all_source_calculable_days_in_research_window_including_warmup_unknown",
            "eligibility": "positive_finite_ohlc_volume_amount_factor_and_valid_price_range",
            "rule_counts": rule_counts, "pairwise_true_overlap": overlaps,
            "experiment_comparisons": comparisons,
        }
        connection.execute("CHECKPOINT")
    destination = output_directory / "condition_diagnostics.json"
    destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    with (output_directory / "condition_diagnostics.csv").open(
        "w", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("rule", *next(iter(rule_counts.values()))))
        writer.writeheader()
        writer.writerows({"rule": rule, **counts} for rule, counts in rule_counts.items())
    return destination
