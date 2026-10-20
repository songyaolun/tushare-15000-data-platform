"""导出ETF每日账户、逐笔交易、信号与年度收益，不覆盖既有运行。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .data import load_data
from .engine import Configuration, run_account


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """导出列名并集与全部记录；例如一条date/price记录生成带表头CSV，空输入生成空文件。"""
    with path.open("w", newline="", encoding="utf-8") as output:
        if rows:
            names = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(output, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)


def verify_calendar(
    calendar: list[str], path: Path, configuration: Configuration
) -> dict[str, Any]:
    """对照独立交易日历逐日核查，防止行情/因子/新日历共同缺日。

    例：独立日历有20211019而来源均缺该日，抛etf_independent_calendar_mismatch。
    """
    raw = path.read_bytes()
    expected = [day for day in json.loads(raw) if calendar[0] <= day <= configuration.end]
    if calendar != expected:
        raise ValueError("etf_independent_calendar_mismatch")
    count = sum(configuration.start <= day <= configuration.end for day in calendar)
    if configuration.start == "20211018" and configuration.end == "20261016" and count != 1212:
        raise ValueError("etf_target_session_count_mismatch")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "target_sessions": count, "history_sessions": len(calendar)}


def run_research(reference: Path, output: Path, calendar_path: Path) -> dict[str, Any]:
    """验证来源后将100万元三账户输出至新目录；例：来源缺历史则不生成summary.json。"""
    if output.exists():
        raise ValueError("etf_output_already_exists")
    calendar, bars, dividends, audit = load_data(reference)
    configuration = Configuration()
    audit["independent_calendar"] = verify_calendar(calendar, calendar_path, configuration)
    results = {strategy: run_account(strategy, calendar, bars, dividends, configuration)
               for strategy in ("buy_hold", "monthly_half", "monthly_trend_126")}
    output.mkdir(parents=True)
    (output / "data-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    for strategy, result in results.items():
        folder = output / strategy
        folder.mkdir()
        for name in ("daily", "orders", "trades", "dividends", "decisions"):
            write_csv(folder / f"{name}.csv", result[name])
        write_csv(folder / "annual.csv", [dict(year=year, **values)
                  for year, values in result["summary"]["annual_returns"].items()])
        (folder / "summary.json").write_text(json.dumps(
            {"configuration": result["configuration"], **result["summary"]},
            ensure_ascii=False, indent=2) + "\n")
    summary = {strategy: result["summary"] for strategy, result in results.items()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
