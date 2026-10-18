"""仅从导出CSV复算资金与交易统计，不调用模拟器或组合评价代码。"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

INITIAL_CAPITAL = 1_000_000.0
TOLERANCE = 1e-8


def _require(condition: bool, path: Path, field: str) -> None:
    """为失败提供配置与字段定位；例False、配置A、cash抛出A:cash审计错误。"""
    if not condition:
        raise ValueError(f"audit_failed:{path.name}:{field}")


def _number(value: Any, path: Path, field: str) -> float:
    """转换有限数值并保留错误定位；例'1.25'返回1.25，NaN或空文本被拒绝。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"audit_failed:{path.name}:{field}:not_numeric") from error
    _require(math.isfinite(number) and not isinstance(value, bool), path, field)
    return number


def _equal(actual: Any, expected: float | None, path: Path, field: str) -> None:
    """以1e-8相对/绝对容差比较指标；例None只匹配None，1匹配1+1e-10。"""
    if expected is None:
        _require(actual is None, path, field)
    else:
        number = _number(actual, path, field)
        _require(math.isclose(number, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE), path, field)


def _rows(
    path: Path, filename: str, required: set[str], *, allow_empty: bool = False
) -> list[dict[str, str]]:
    """读取有完整表头的CSV；例day表头返回空列表，allow_empty允许零计数的零字节导出。"""
    source = path / filename
    _require(source.is_file(), path, filename + ":missing")
    if allow_empty and source.stat().st_size == 0:
        return []
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        _require(
            required <= set(fields) and len(fields) == len(set(fields)), path, filename + ":columns"
        )
        rows = list(reader)
    _require(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        path,
        filename + ":ragged_row",
    )
    return rows


def _performance(returns: list[float]) -> dict[str, float]:
    """独立逐日复利并保留起始资金峰值；例[-.1, .1]返回累计-.01、回撤-.1。"""
    curve = peak = 1.0
    drawdown = 0.0
    for change in returns:
        curve *= 1 + change
        peak = max(peak, curve)
        drawdown = min(drawdown, curve / peak - 1)
    return {
        "total_return": curve - 1,
        "annualized_return": curve ** (250 / len(returns)) - 1,
        "maximum_drawdown": drawdown,
    }


def audit_result(path: Path) -> dict[str, Any]:
    """独立审计一个配置的metrics、daily、trades和entries并返回可复核摘要。

    例audit_result(Path('run/reversion_a'))返回配置名、交易数、末端NAV和重算收益；
    daily.return与NAV不符时抛出含reversion_a及daily.return的ValueError。
    初始资本固定100万元，年化250日，入场空窗含起止边界；费用后交易收益
    从(proceeds+dividend-cost)/cost重算。未平仓股息没有导出，末端恒等式只检查
    已导出的reconciliation_error，不以不完整字段重建持仓损益。
    """
    try:
        saved = json.loads((path / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"audit_failed:{path.name}:metrics.json:unreadable") from error
    _require(
        isinstance(saved, dict) and isinstance(saved.get("metrics"), dict),
        path,
        "metrics.json:structure",
    )
    metrics = saved["metrics"]
    required_metrics = {
        "total_return",
        "annualized_return",
        "maximum_drawdown",
        "annual",
        "closed_trades",
        "win_rate",
        "mean_trade_return",
        "maximum_entry_gap",
        "mean_exposure",
        "entry_count",
        "realized_profit",
        "reconciliation_error",
        "terminal_nav_if_stale_zero",
        "stale_value",
    }
    _require(required_metrics <= metrics.keys(), path, "metrics:missing_fields")
    daily = _rows(path, "daily.csv", {"day", "date", "nav", "cash", "exposure", "return"})
    trades = _rows(
        path,
        "trades.csv",
        {"entry", "exit", "cost", "proceeds", "dividend", "return"},
        allow_empty=metrics["closed_trades"] == 0,
    )
    entries = _rows(path, "entries.csv", {"day"}, allow_empty=metrics["entry_count"] == 0)
    _require(bool(daily), path, "daily:empty")
    returns: list[float] = []
    annual_returns: dict[str, list[float]] = {}
    day_years: dict[int, str] = {}
    cash_values, exposures = [], []
    previous_nav = INITIAL_CAPITAL
    previous_date = ""
    previous_day: int | None = None
    for index, row in enumerate(daily):
        field = f"daily[{index}]"
        day_number = _number(row["day"], path, field + ".day")
        _require(day_number.is_integer() and day_number >= 0, path, field + ".day")
        day = int(day_number)
        _require(previous_day is None or day == previous_day + 1, path, "daily:day_sequence")
        date = row["date"]
        try:
            parsed = datetime.strptime(date, "%Y%m%d")
        except ValueError as error:
            raise ValueError(f"audit_failed:{path.name}:daily.date") from error
        _require(parsed.strftime("%Y%m%d") == date and date > previous_date, path, "daily.date")
        nav = _number(row["nav"], path, field + ".nav")
        cash = _number(row["cash"], path, field + ".cash")
        exposure = _number(row["exposure"], path, field + ".exposure")
        _require(nav > 0, path, field + ".nav")
        _require(cash >= -TOLERANCE, path, field + ".cash")
        _require(-TOLERANCE <= exposure <= 1 + TOLERANCE, path, field + ".exposure")
        change = nav / previous_nav - 1
        _equal(row["return"], change, path, field + ".return")
        returns.append(change)
        annual_returns.setdefault(date[:4], []).append(change)
        day_years[day] = date[:4]
        cash_values.append(cash)
        exposures.append(exposure)
        previous_nav, previous_date, previous_day = nav, date, day
    performance = _performance(returns)
    for field, value in performance.items():
        _equal(metrics[field], value, path, field)
    _equal(metrics["mean_exposure"], math.fsum(exposures) / len(exposures), path, "mean_exposure")
    stale_value = _number(metrics["stale_value"], path, "stale_value")
    _require(0 <= stale_value <= previous_nav + TOLERANCE, path, "stale_value")
    _equal(
        metrics["terminal_nav_if_stale_zero"],
        previous_nav - stale_value,
        path,
        "terminal_nav_if_stale_zero",
    )
    _equal(metrics["reconciliation_error"], 0.0, path, "reconciliation_error")
    trade_returns, profits = [], []
    annual_trades: dict[str, list[float]] = {year: [] for year in annual_returns}
    for index, row in enumerate(trades):
        field = f"trades[{index}]"
        entry = _number(row["entry"], path, field + ".entry")
        exit_day = _number(row["exit"], path, field + ".exit")
        _require(
            entry in day_years and exit_day in day_years and exit_day > entry,
            path,
            field + ".dates",
        )
        cost = _number(row["cost"], path, field + ".cost")
        proceeds = _number(row["proceeds"], path, field + ".proceeds")
        dividend = _number(row["dividend"], path, field + ".dividend")
        _require(cost > 0 and proceeds >= 0, path, field + ".cashflows")
        profit = proceeds + dividend - cost
        change = profit / cost
        _equal(row["return"], change, path, field + ".return")
        trade_returns.append(change)
        profits.append(profit)
        annual_trades[day_years[int(exit_day)]].append(change)
    closed = len(trade_returns)
    win_rate = sum(change > 0 for change in trade_returns) / closed if closed else None
    mean_trade_return = math.fsum(trade_returns) / closed if closed else None
    for field, expected in (
        ("closed_trades", closed),
        ("win_rate", win_rate),
        ("mean_trade_return", mean_trade_return),
        ("realized_profit", math.fsum(profits)),
    ):
        _equal(metrics[field], expected, path, field)
    entry_days = [_number(row["day"], path, "entries.day") for row in entries]
    _require(all(day in day_years for day in entry_days), path, "entries.day")
    _equal(metrics["entry_count"], len(entry_days), path, "entry_count")
    start, end = min(day_years), max(day_years)
    boundaries = [start, *sorted(set(entry_days)), end]
    maximum_gap = (
        max(right - left for left, right in zip(boundaries[:-1], boundaries[1:], strict=True))
        if entry_days
        else end - start + 1
    )
    _equal(metrics["maximum_entry_gap"], maximum_gap, path, "maximum_entry_gap")
    _require(
        isinstance(metrics["annual"], dict) and set(metrics["annual"]) == set(annual_returns),
        path,
        "annual:years",
    )
    annual: dict[str, Any] = {}
    for year, changes in annual_returns.items():
        annual[year] = _performance(changes)
        saved_annual = metrics["annual"][year]
        year_trades = annual_trades[year]
        year_count = len(year_trades)
        annual[year].update(
            {
                "trades_closed": year_count,
                "win_rate": sum(change > 0 for change in year_trades) / year_count
                if year_count
                else None,
                "entries": sum(day_years[int(day)] == year for day in entry_days),
            }
        )
        _require(
            isinstance(saved_annual, dict) and annual[year].keys() <= saved_annual.keys(),
            path,
            f"annual.{year}:fields",
        )
        for field, expected in annual[year].items():
            _equal(saved_annual[field], expected, path, f"annual.{year}.{field}")
    return {
        "name": path.name,
        "path": str(path),
        "sessions": len(daily),
        "terminal_nav": previous_nav,
        **performance,
        "closed_trades": closed,
        "win_rate": win_rate,
        "mean_trade_return": mean_trade_return,
        "realized_profit": math.fsum(profits),
        "entry_count": len(entry_days),
        "maximum_entry_gap": maximum_gap,
        "minimum_cash": min(cash_values),
        "maximum_exposure": max(exposures),
        "annual": annual,
        "reconciliation_error": 0.0,
    }


def audit_experiments(roots: list[Path], output: Path) -> Path:
    """审计配置目录或含配置子目录的实验根目录，全部通过才写摘要JSON。

    例audit_experiments([Path('run')], Path('audit.json'))输出audit.json，含合格配置数、
    总平仓交易数和逐配置摘要；缺文件、空根目录或重复配置路径均明确失败。
    """
    _require(bool(roots), output, "roots:empty")
    configurations: list[Path] = []
    for root in roots:
        _require(root.is_dir(), root, "root:missing")
        if (
            (root / "metrics.json").exists()
            or any(root.glob("*.csv"))
            and not (root / "experiment.json").exists()
        ):
            configurations.append(root)
        else:
            children = sorted(child for child in root.iterdir() if child.is_dir())
            _require(bool(children), root, "root:no_configurations")
            configurations.extend(children)
    _require(
        len({path.resolve() for path in configurations}) == len(configurations),
        output,
        "roots:duplicate_configurations",
    )
    summaries = [audit_result(path) for path in configurations]
    report = {
        "passed_configurations": len(summaries),
        "closed_trades": sum(summary["closed_trades"] for summary in summaries),
        "configurations": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return output
