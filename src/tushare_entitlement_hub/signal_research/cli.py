"""MACD研究离线入口；取数与研究运行分离。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from .history import assert_complete_history, iter_security_history, load_history_database
from .reporting import enrich_signal_percentiles, summarize_trades, write_research_report
from .runner import ResearchConfiguration, research_experiments, run_research

signal_research_app = typer.Typer(help="独立MACD条件、五年研究回测和信号证据。")


def validate_research_window(calendar: list[str], configuration: ResearchConfiguration) -> None:
    """验证请求窗口和至少250个交易日预热；输入短缓存，输出ValueError。

    起止日期必须为已采集交易日，不能将一月实测标成五年研究；上市不足的证券另计。
    """
    if (configuration.start_date not in calendar or configuration.end_date not in calendar
            or calendar.index(configuration.start_date) < 250):
        raise ValueError("历史缓存未覆盖请求窗口或250个交易日预热")


@signal_research_app.command("run")
def run_command(
    history_root: Annotated[Path, typer.Option(help="已认证原始历史缓存目录")],
    output_directory: Annotated[Path, typer.Option(help="私有研究产物目录")],
    start_date: str = "20211018",
    end_date: str = "20261016",
    histogram_threshold: float = 0.002,
    confirmation_days: int = 3,
    chart_samples: int = 2,
) -> None:
    """校验缓存并运行离线研究，生成逐信号记录、消融表和复核截图。

    输入 --history-root cache --output-directory result，输出result/report.md路径。
    不读取凭据或发网络请求；数据不完整时不发布五年结论。
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    database_path = output_directory / "history.duckdb"
    data_manifest = load_history_database(history_root, database_path)
    assert_complete_history(data_manifest)
    calendar = data_manifest["trading_dates"]
    configuration = ResearchConfiguration(start_date, end_date, histogram_threshold,
                                          confirmation_days, chart_samples)
    validate_research_window(calendar, configuration)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        run_manifest = run_research(iter_security_history(connection), calendar,
                                    output_directory, configuration)
    with (output_directory / "signal_records.jsonl").open(encoding="utf-8") as stream:
        trades = (json.loads(line) for line in stream)
        summary = summarize_trades(trades, [item.name for item in research_experiments()])
    enrich_signal_percentiles(output_directory)
    manifest: dict[str, Any] = {"data": data_manifest, **run_manifest}
    markdown_path, _ = write_research_report(summary, manifest, output_directory)
    typer.echo(str(markdown_path))
