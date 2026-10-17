"""离线五年研究编排；行情共享存储，每条信号保留观察日条件和逐笔结果。"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

from .artifacts import SignalArtifact, write_artifact_index, write_signal_artifacts
from .filters import TECHNICAL_FILTER_NAMES, evaluate_filter_series
from .indicators import IndicatorPoint, PriceBar, StockCharacteristics, compute_indicators
from .signals import ExperimentSwitches, confirm_signals
from .simulation import SimulationConfiguration, simulate_signal_trades


@dataclass(frozen=True)
class ResearchConfiguration:
    """冻结研究协议，避免看结果后改规则。

    输入 ResearchConfiguration()，输出20211018至20261016、阈值.002、确认3天的协议。
    基本面、行业、量价未定规则不自动启用；技术基线不是完整选股策略。
    """

    start_date: str = "20211018"
    end_date: str = "20261016"
    histogram_threshold: float = 0.002
    confirmation_days: int = 3
    chart_samples_per_year_direction: int = 2

    def __post_init__(self) -> None:
        """验证冻结参数；输入非法日期、负截图数或非有限阈值时抛ValueError。

        示例 ResearchConfiguration(histogram_threshold=float('nan')) 拒绝运行；
        日期严格使用YYYYMMDD，确认窗口和截图数允许零但必须为非负整数。
        """
        for value in (self.start_date, self.end_date):
            if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
                raise ValueError("研究日期必须为合法YYYYMMDD")
            date.fromisoformat(value)
        if self.start_date > self.end_date:
            raise ValueError("回测起始日不得晚于结束日")
        if (
            type(self.histogram_threshold) not in (int, float)
            or not isfinite(self.histogram_threshold)
            or self.histogram_threshold < 0
        ):
            raise ValueError("研究阈值必须为非负有限数")
        for integer_parameter in (self.confirmation_days, self.chart_samples_per_year_direction):
            if type(integer_parameter) is not int or integer_parameter < 0:
                raise ValueError("确认窗口和每层截图数必须为非负整数")


def research_experiments() -> tuple[ExperimentSwitches, ...]:
    """创建技术基线及单条件删除实验；输入无，输出独立开关元组。

    event关闭实验评估全日历，不复用基线候选；H正值与显著红柱可能逻辑冗余。
    """
    baseline = ExperimentSwitches("technical_baseline", TECHNICAL_FILTER_NAMES)
    deletions = tuple(
        ExperimentSwitches(
            f"without_{name}", tuple(other for other in TECHNICAL_FILTER_NAMES if other != name)
        )
        for name in TECHNICAL_FILTER_NAMES
    )
    return (
        baseline,
        *deletions,
        ExperimentSwitches("without_macd_event", TECHNICAL_FILTER_NAMES, False),
        ExperimentSwitches("macd_event_only", ()),
    )


def _valid_quote(row: Mapping[str, Any]) -> bool:
    """仅允许完整正价和正因子的指标输入；输入停牌0价行，输出False。

    原始停牌行仍留在模拟日历，不能用0价格计算均线。
    """
    return (
        not row.get("suspended")
        and row.get("tradable", True) is True
        and all(
            type(row.get(name)) in (int, float) and isfinite(row[name]) and row[name] > 0
            for name in ("open", "high", "low", "close", "adj_factor", "vol", "amount")
        )
        and row["low"] <= min(row["open"], row["close"])
        and max(row["open"], row["close"]) <= row["high"]
    )


def _price_bar(row: Mapping[str, Any]) -> PriceBar:
    """将源行情转换指标输入并审计单位；输入amount=1千元，输出amount=1000元。

    daily_basic circ_mv万元转元；源值另存在共享行情文件中，非硬筛选门槛。
    """
    return PriceBar(
        str(row["trade_date"]),
        float(row["open"]),
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
        float(row["adj_factor"]),
        row.get("turnover_rate"),
        float(row["amount"]) * 1000 if row.get("amount") is not None else None,
        float(row["circ_mv"]) * 10000 if row.get("circ_mv") is not None else None,
        row.get("vol"),
    )


def _calendar_points(
    points: Sequence[IndicatorPoint],
    calendar: Sequence[str],
) -> tuple[IndicatorPoint, ...]:
    """在首根已知行情之后补市场日历未知点；输入停牌缺日，输出该日H/C未知。

    日历点用于确认3天窗口；停牌不产生信号，也不压缩确认时间。
    """
    if not points:
        return ()
    by_date = {point.trade_date: point for point in points}
    result: list[IndicatorPoint] = []
    for day in calendar:
        if day < points[0].trade_date:
            continue
        point = by_date.get(day)
        if point is None:
            previous = (
                result[-1]
                if result
                else max(
                    (known for known in points if known.trade_date < day),
                    key=lambda known: known.trade_date,
                )
            )
            point = IndicatorPoint(
                day,
                previous.close,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                StockCharacteristics(None, None, None, None, None, None, None, None, None),
            )
        result.append(point)
    return tuple(result)


def _forward_returns(
    confirmation_date: str,
    rows: Mapping[str, Mapping[str, Any]],
    calendar: Sequence[str],
) -> dict[str, float | None]:
    """计算确认收盘至未来5/10/20/40市场交易日收盘的复权价格收益。

    输入当日10元、20日后11元且因子相同，输出return_20d=.1；缺报价为None。
    此收益不是成交收益，用于人工复核信号后价格变化。
    """
    result: dict[str, float | None] = {}
    start = rows.get(confirmation_date)
    index = calendar.index(confirmation_date)
    for horizon in (5, 10, 20, 40):
        future = rows.get(calendar[index + horizon]) if index + horizon < len(calendar) else None
        result[f"return_{horizon}d"] = (
            float(
                future["close"] * future["adj_factor"] / (start["close"] * start["adj_factor"]) - 1
            )
            if start and future and _valid_quote(start) and _valid_quote(future)
            else None
        )
    return result


def _chart_artifact(
    code: str,
    confirmation: Mapping[str, Any],
    points: Sequence[IndicatorPoint],
    rows: Mapping[str, Mapping[str, Any]],
    record: Mapping[str, Any],
) -> SignalArtifact:
    """提取确认前250根及后60根可交易行情制作复盘截图；输入一信号，输出Artifact。

    价格以确认日因子统一缩放；未来指标仅用于橙色区事后复核，不回流选股。
    """
    day = str(confirmation["confirmation_date"])
    valid_points = [
        point
        for point in points
        if point.trade_date in rows and _valid_quote(rows[point.trade_date])
    ]
    position = next(index for index, point in enumerate(valid_points) if point.trade_date == day)
    window = valid_points[max(0, position - 250) : position + 61]
    # compute_indicators采用固定历史首日锚点；展示以确认日锚点统一重标，H/C保持不变。
    first_factor = rows[valid_points[0].trade_date]["adj_factor"]
    confirmation_factor = rows[day]["adj_factor"]
    scale = first_factor / confirmation_factor
    dated: dict[str, list[Any]] = {
        name: []
        for name in (
            "date",
            "open",
            "high",
            "low",
            "close",
            "ma20",
            "ma60",
            "dif",
            "dea",
            "hist",
            "volume",
        )
    }
    for point in window:
        source = rows[point.trade_date]
        dated["date"].append(point.trade_date)
        for name in ("open", "high", "low", "close"):
            dated[name].append(source[name] * source["adj_factor"] / confirmation_factor)
        for name in ("ma20", "ma60", "dif", "dea", "hist"):
            value = getattr(point, "histogram" if name == "hist" else name)
            dated[name].append(value * scale if value is not None else None)
        dated["volume"].append(source.get("vol") or 0)
    snapshot_metrics = {
        **record["pre_signal_metrics"],
        "snapshot_price_anchor_date": day,
        "snapshot_price_anchor_factor": confirmation_factor,
        "snapshot_price_semantics": "raw_price_times_daily_factor_divided_by_confirmation_factor",
        "snapshot_indicators_rescaled_from_series_anchor": True,
    }
    return SignalArtifact(
        str(record["signal_id"]),
        code,
        day,
        dated,
        snapshot_metrics,
        record["filter_results"],
        record["post_signal_returns"],
    )


def run_research(
    securities: Iterable[tuple[str, list[dict[str, Any]]]],
    calendar: Sequence[str],
    output_directory: Path,
    configuration: ResearchConfiguration | None = None,
) -> dict[str, Any]:
    """逐证券执行技术基线及消融，保存全日条件矩阵和逐信号数据。

    输入证券源行情、完整市场日历和输出目录，输出运行计数与协议；无网络动作。
    日矩阵按证券gzip共享；signal_records.jsonl逐信号引用文件，避免重复数万份行情。
    截图按基线信号确认年及20日后涨跌抽样，不按结果择优删信号。
    """
    protocol = configuration or ResearchConfiguration()
    calendar = tuple(day for day in calendar if day <= protocol.end_date)
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError("完整交易日历必须唯一递增且非空")
    output_directory.mkdir(parents=True, exist_ok=True)
    series_directory = output_directory / "daily_condition_matrix"
    if (output_directory / "signal_records.jsonl").exists() or (
        series_directory.exists() and any(series_directory.iterdir())
    ):
        raise ValueError("研究产物目录已有信号或日矩阵，请使用新的空产物目录")
    series_directory.mkdir(exist_ok=True)
    experiments = research_experiments()
    simulation = SimulationConfiguration(green_histogram_threshold=protocol.histogram_threshold)
    counts: Counter[str] = Counter()
    chart_counts: Counter[tuple[str, str]] = Counter()
    chart_entries = []
    with (output_directory / "signal_records.jsonl").open("w", encoding="utf-8") as signals_file:
        for code, source_rows in securities:
            rows = {
                str(row["trade_date"]): row
                for row in source_rows
                if str(row["trade_date"]) <= protocol.end_date
            }
            if len(rows) != len(
                [row for row in source_rows if str(row["trade_date"]) <= protocol.end_date]
            ):
                raise ValueError("证券行情日期重复")
            counts["invalid_or_inactive_source_quotes"] += sum(
                not _valid_quote(row) for row in rows.values()
            )
            price_bars = [_price_bar(rows[day]) for day in sorted(rows) if _valid_quote(rows[day])]
            points = compute_indicators(price_bars, protocol.end_date)
            if not points:
                counts["securities_without_valid_factor_quotes"] += 1
                continue
            full_points = _calendar_points(points, calendar)
            if not full_points:
                counts["securities_without_calendar_points"] += 1
                continue
            points_by_date = {point.trade_date: point for point in full_points}
            results_by_date = evaluate_filter_series(full_points)
            relative_series = f"daily_condition_matrix/{code}.csv.gz"
            with gzip.open(
                output_directory / relative_series, "wt", encoding="utf-8", newline="",
                compresslevel=1,
            ) as file:
                fieldnames = [
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                    "adj_factor",
                    "turnover_rate",
                    "circ_mv",
                    "ma20",
                    "ma60",
                    "dif",
                    "dea",
                    "histogram",
                    "normalized_histogram",
                    *results_by_date[full_points[0].trade_date],
                    *asdict(full_points[0].characteristics),
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for point in full_points:
                    raw = rows.get(point.trade_date, {})
                    values = {name: raw.get(name) for name in fieldnames[:10]}
                    values.update(trade_date=point.trade_date)
                    values.update({name: getattr(point, name) for name in fieldnames[10:16]})
                    values.update(
                        {name: str(value)
                         for name, value in results_by_date[point.trade_date].items()}
                    )
                    values.update(asdict(point.characteristics))
                    writer.writerow(values)
            confirmations = confirm_signals(
                full_points,
                experiments,
                ts_code=code,
                threshold=protocol.histogram_threshold,
                confirmation_days=protocol.confirmation_days,
                precomputed_results_by_date=results_by_date,
            )
            signal_dicts = [
                asdict(signal)
                for signal in confirmations
                if protocol.start_date <= signal.confirmation_date <= protocol.end_date
                and signal.confirmation_date in rows
                and _valid_quote(rows[signal.confirmation_date])
            ]
            simulation_bars = []
            for day in calendar:
                source = dict(rows.get(day, {}))
                source.update(
                    ts_code=code,
                    trade_date=day,
                    normalized_histogram=points_by_date[day].normalized_histogram
                    if day in points_by_date
                    else None,
                )
                if day not in rows:
                    source["suspended"] = True
                simulation_bars.append(source)
            trades = simulate_signal_trades(simulation_bars, signal_dicts, simulation)
            for trade in trades:
                day = str(trade["confirmation_date"]).replace("-", "")
                point = points_by_date[day]
                snapshot_scale = rows[points[0].trade_date]["adj_factor"] / rows[day]["adj_factor"]
                identifier = f"{code}:{day}:{trade['experiment_id']}"
                record: dict[str, Any] = {
                    **trade,
                    "signal_id": identifier,
                    "source_series": relative_series,
                    "source_series_asof_cutoff": day,
                    "source_series_metadata": {
                        "raw_ohlc_semantics": "unadjusted_provider_quotes",
                        "indicator_price_semantics": (
                            "daily_factor_divided_by_fixed_first_history_anchor"
                        ),
                        "indicator_anchor_date": points[0].trade_date,
                        "indicator_anchor_factor": rows[points[0].trade_date]["adj_factor"],
                        "confirmation_anchor_factor": rows[day]["adj_factor"],
                        "snapshot_scale_from_series": (
                            rows[points[0].trade_date]["adj_factor"] / rows[day]["adj_factor"]
                        ),
                        "future_anchor_used": False,
                    },
                    "pre_signal_metrics": {
                        **asdict(point.characteristics),
                        **{
                            name: getattr(point, name) * snapshot_scale
                            if getattr(point, name) is not None
                            else None
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
                            )
                        },
                        "snapshot_price_anchor_date": day,
                        "snapshot_price_anchor_factor": rows[day]["adj_factor"],
                        "snapshot_price_semantics": (
                            "raw_price_times_daily_factor_divided_by_confirmation_factor"
                        ),
                        "normalized_histogram": point.normalized_histogram,
                        "close_ma20_extension_pct": (
                            (point.close / point.ma20 - 1) * 100
                            if point.ma20 is not None and point.ma20 > 0
                            else None
                        ),
                        "atr14_close_ratio": (
                            point.atr14 / point.close if point.atr14 is not None else None
                        ),
                        "price_units": "CNY, source raw plus adjustment_factor",
                        "amount_units": "CNY",
                        "market_cap_units": "CNY",
                        "vol_units": "100 shares",
                        "fundamentals_status": "UNKNOWN",
                        "sector_status": "UNKNOWN",
                    },
                    "post_signal_returns": _forward_returns(day, rows, calendar),
                    "event_review_status": "unreviewed_causality_unproven",
                }
                signals_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                counts[f"signals:{trade['experiment_id']}"] += 1
                counts[f"status:{trade['experiment_id']}:{trade['status']}"] += 1
                if trade["experiment_id"] == "technical_baseline":
                    forward_return = record["post_signal_returns"]["return_20d"]
                    direction = (
                        "unknown"
                        if forward_return is None
                        else (
                            "positive"
                            if forward_return > 0
                            else "negative"
                            if forward_return < 0
                            else "zero"
                        )
                    )
                    stratum = day[:4], direction
                    if chart_counts[stratum] < protocol.chart_samples_per_year_direction:
                        artifact = _chart_artifact(
                            code, {"confirmation_date": day}, points, rows, record
                        )
                        chart_entries.append(
                            write_signal_artifacts(
                                artifact, output_directory / "charts", render_charts=True
                            )
                        )
                        chart_counts[stratum] += 1
            counts["securities_processed"] += 1
            if counts["securities_processed"] % 100 == 0:
                print(f"研究进度：{counts['securities_processed']}只证券", flush=True)
    write_artifact_index(chart_entries, output_directory / "charts")
    manifest = {
        "protocol": asdict(protocol),
        "experiments": [asdict(item) for item in experiments],
        "counts": dict(counts),
        "chart_count": len(chart_entries),
        "security_scope": "all_supplied_securities_no_implicit_board_exclusion",
        "asof_evidence": (
            "signal pre_signal_metrics and daily source cutoff; future separately marked"
        ),
        "limitations": [
            "technical-only, fundamentals and historical sector not validated",
            "histogram threshold .002 is an unoptimized research assumption",
            "not portfolio NAV or exact dividend cash PnL",
            "pending limit orders have no configured expiration",
            "historical event evidence not automatically complete",
        ],
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest
