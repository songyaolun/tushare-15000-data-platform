"""只读信号证据的事后路径复核，不构造或修改交易信号。"""

import csv
import gzip
import json
from collections.abc import Mapping, Sequence
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

import duckdb

BENCHMARK_CODES = ("000001.SH", "399001.SZ", "399006.SZ")


def _number(value: Any) -> float | None:
    """读取有限数值，未知保留None；输入''输出None，输入'2'输出2.0。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if isfinite(number) else None


def _day(value: str) -> str:
    """统一日期为YYYYMMDD；输入2023-01-01输出20230101，非法日期拒绝。"""
    return date.fromisoformat(value).strftime("%Y%m%d")


def _calendar(root: Path, observed_dates: Sequence[str]) -> tuple[list[str], str]:
    """优先完整市场日历，缺源时注明共享日期并集；输入root输出(日历,来源)。

    支持trading_dates.json或history.duckdb中的trading_calendar表；无日历和行情输出[]。
    """
    calendar_path = root / "trading_dates.json"
    if calendar_path.exists():
        source = json.loads(calendar_path.read_text())
        values = source.get("trading_dates", []) if isinstance(source, dict) else source
        return sorted({_day(value) for value in values}), "trading_dates.json"
    database = root if root.is_file() and root.suffix == ".duckdb" else root / "history.duckdb"
    if database.exists():
        with duckdb.connect(str(database), read_only=True) as connection:
            values = connection.execute(
                "SELECT date FROM trading_calendar ORDER BY date"
            ).fetchall()
        return [_day(value[0]) for value in values], "history_database_trading_calendar"
    return sorted(set(observed_dates)), "shared_series_date_union_not_certified_market_calendar"


def _benchmark_rows(root: Path) -> dict[str, dict[str, float]]:
    """加载三指数源收盘，支持records或提供方fields/items；输入root输出按代码日期索引。

    示例raw/index_daily/x.json含ts_code/trade_date/close记录，输出对应浮点收盘。
    指数价格只算源价格比，不套用股票复权因子。
    """
    result: dict[str, dict[str, float]] = {code: {} for code in BENCHMARK_CODES}
    paths = sorted(
        set((root / "raw" / "index_daily").glob("*.json"))
        | set((root / "index_daily").glob("*.json"))
        | {
            root / "benchmarks" / f"{code}.json"
            for code in BENCHMARK_CODES
            if (root / "benchmarks" / f"{code}.json").exists()
        }
    )
    for path in paths:
        envelope = json.loads(path.read_text())
        response = envelope.get("response", envelope) if isinstance(envelope, dict) else envelope
        if isinstance(response, dict) and "data" in response:
            response = response["data"]
        if isinstance(response, dict) and "fields" in response:
            rows = [dict(zip(response["fields"], item, strict=True)) for item in response["items"]]
        else:
            rows = response.get("records", []) if isinstance(response, dict) else response
        for row in rows:
            code = row.get("ts_code")
            close = _number(row.get("close"))
            if code in result and close is not None and close > 0:
                result[code][_day(row["trade_date"])] = close
    return result


def _path_metrics(
    rows: Mapping[str, Mapping[str, Any]],
    days: Sequence[str],
    reference_price: float | None,
    reference_factor: float | None,
) -> dict[str, float | None]:
    """计算因子价格路径的MAE/MFE及独立收盘峰谷回撤；输入路径输出三个指标。

    示例基准10、最低8最高12输出MAE=-.2、MFE=.2；drawdown使用有序收盘峰值，
    不能当作MAE。任一窗口日缺有效OHLC或因子，固定窗口指标全部未知。
    """
    unknown: dict[str, float | None] = dict(mae=None, mfe=None, close_peak_trough_drawdown=None)
    if not days or reference_price is None or reference_factor is None:
        return unknown
    if reference_price <= 0 or reference_factor <= 0:
        return unknown
    multiples = []
    for day in days:
        source = rows.get(day, {})
        factor = _number(source.get("adj_factor"))
        values = [_number(source.get(name)) for name in ("low", "high", "close")]
        if factor is None or factor <= 0 or any(value is None or value <= 0 for value in values):
            return unknown
        multiples.append(
            [float(str(value)) * factor / (reference_price * reference_factor) for value in values]
        )
    peak = 1.0
    drawdown = 0.0
    for _, _, close in multiples:
        peak = max(peak, close)
        drawdown = min(drawdown, close / peak - 1)
    return {
        "mae": min(0.0, min(item[0] for item in multiples) - 1),
        "mfe": max(0.0, max(item[1] for item in multiples) - 1),
        "close_peak_trough_drawdown": drawdown,
    }


def _index_return(rows: Mapping[str, float], start: str | None, end: str | None) -> float | None:
    """计算两日期源指数收盘比；输入10到11输出.1，缺日输出None。"""
    if start is None or end is None or start not in rows or end not in rows:
        return None
    return rows[end] / rows[start] - 1


def build_retrospective_review(output_directory: Path, benchmark_root: Path) -> Path:
    """生成基线全信号复核CSV和六年亏损极端样本JSON，不回流任何信号。

    输入已完成runner目录与指数/日历根，输出baseline_trade_review.csv路径。
    固定20/40市场session从确认日下一session开始，以确认收盘为基准；持仓
    从entry至exit以entry成交价格为基准，退出日仅包含开盘成交价而不读取
    退出之后当日最高最低价，未退出或缺执行价格持仓保持未知。
    MAE/MFE使用每日最低/最高，峰谷回撤单独使用收盘，不声称盘中执行路径。
    每确认年选费后收益最差的closed负收益，最多六条；极端样本不代表总体，
    不推断事件因果或精确现金红利收益。示例无基线信号仍输出CSV表头与空samples。
    """
    output_directory, benchmark_root = Path(output_directory), Path(benchmark_root)
    with (output_directory / "signal_records.jsonl").open(encoding="utf-8") as stream:
        signals = [
            record
            for line in stream
            if (record := json.loads(line)).get("experiment_id") == "technical_baseline"
        ]
    series_paths: dict[str, Path] = {}
    for signal in signals:
        relative = str(signal["source_series"])
        if relative not in series_paths:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("共享行情引用必须为安全相对路径")
            series_paths[relative] = output_directory / path
    calendar, calendar_source = _calendar(benchmark_root, [])
    if not calendar:
        observed_dates: set[str] = set()
        for path in series_paths.values():
            with gzip.open(path, "rt", newline="") as stream:
                observed_dates.update(row["trade_date"] for row in csv.DictReader(stream))
        calendar = sorted(observed_dates)
    indices = {day: index for index, day in enumerate(calendar)}
    benchmarks = _benchmark_rows(benchmark_root)
    review_rows: list[dict[str, Any]] = []
    current_series = None
    rows: dict[str, dict[str, Any]] = {}
    for signal in sorted(signals, key=lambda signal: str(signal["source_series"])):
        relative = str(signal["source_series"])
        if relative != current_series:
            with gzip.open(series_paths[relative], "rt", newline="") as stream:
                rows = {row["trade_date"]: row for row in csv.DictReader(stream)}
            current_series = relative
        day = _day(signal["confirmation_date"])
        confirmation = rows.get(day, {})
        entry_day = _day(signal["entry_date"]) if signal.get("entry_date") else None
        exit_day = _day(signal["exit_date"]) if signal.get("exit_date") else None
        result: dict[str, Any] = {
            "signal_id": signal["signal_id"],
            "ts_code": signal["ts_code"],
            "confirmation_date": day,
            "status": signal["status"],
            "entry_date": entry_day,
            "exit_date": exit_day,
            "fee_adjusted_research_return": signal.get("fee_adjusted_research_return"),
            "forward20": signal.get("post_signal_returns", {}).get("return_20d"),
            "forward40": signal.get("post_signal_returns", {}).get("return_40d"),
            "calendar_source": calendar_source,
            **{
                name: signal.get("pre_signal_metrics", {}).get(name)
                for name in (
                    "close_ma20_extension_pct",
                    "atr14_close_ratio",
                    "direction_60",
                    "efficiency_60",
                    "wick_ratio_median_60",
                    "wick_ratio_p90_60",
                    "gap_ratio_60",
                    "turnover_median_20",
                    "amount_median_20",
                    "market_cap",
                    "zero_range_count_60",
                )
            },
        }
        for horizon in (20, 40):
            index = indices.get(day)
            window = calendar[index + 1 : index + horizon + 1] if index is not None else []
            complete = len(window) == horizon
            metrics = _path_metrics(
                rows,
                window if complete else [],
                _number(confirmation.get("close")),
                _number(confirmation.get("adj_factor")),
            )
            result.update({f"{name}{horizon}": value for name, value in metrics.items()})
            for code in BENCHMARK_CODES:
                result[f"market{horizon}_{code}"] = _index_return(
                    benchmarks[code], day, window[-1] if complete else None
                )
        actual_days = (
            calendar[indices[entry_day] : indices[exit_day] + 1]
            if (entry_day in indices and exit_day in indices)
            else []
        )
        exit_price = _number(signal.get("exit_price"))
        holding_rows = dict(rows)
        if exit_day is not None and exit_price is not None:
            holding_rows[exit_day] = {
                **rows.get(exit_day, {}),
                "low": exit_price,
                "high": exit_price,
                "close": exit_price,
            }
        metrics = _path_metrics(
            holding_rows,
            actual_days if exit_price is not None else [],
            _number(signal.get("entry_price")),
            _number(rows.get(entry_day or "", {}).get("adj_factor")),
        )
        result.update({f"holding_{name}": value for name, value in metrics.items()})
        for code in BENCHMARK_CODES:
            result[f"holding_market_return_{code}"] = _index_return(
                benchmarks[code], entry_day, exit_day
            )
        review_rows.append(result)
    columns = (
        list(review_rows[0])
        if review_rows
        else ["signal_id", "ts_code", "confirmation_date", "status"]
    )
    path = output_directory / "baseline_trade_review.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(review_rows)
    yearly: dict[str, dict[str, Any]] = {}
    for row in review_rows:
        loss = _number(row["fee_adjusted_research_return"])
        year = row["confirmation_date"][:4]
        if row["status"] == "closed" and loss is not None and loss < 0:
            previous = yearly.get(year)
            if previous is None or (loss, row["signal_id"]) < (
                previous["fee_adjusted_research_return"],
                previous["signal_id"],
            ):
                yearly[year] = row
    sample_payload = {
        "selection_rule": "worst_closed_negative_fee_adjusted_return_per_confirmation_year_max_six",
        "interpretation": "extreme_loss_examples_not_representative_no_causal_inference",
        "return_semantics": "provider_adjusted_price_research_not_exact_dividend_cash_pnl",
        "drawdown_semantics": "ordered_close_peak_trough_distinct_from_low_based_mae",
        "holding_path_semantics": "entry_day_to_exit_open_fill_excludes_exit_day_later_prices",
        "benchmark_holding_semantics": "source_index_close_on_entry_and_exit_dates_not_open_fill",
        "samples": [yearly[year] for year in sorted(yearly)[:6]],
    }
    (output_directory / "baseline_loss_samples.json").write_text(
        json.dumps(sample_payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return path
