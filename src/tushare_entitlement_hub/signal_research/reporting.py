"""逐信号研究统计与中文报告，明确区分研究收益和组合绩效。"""

from __future__ import annotations

import html
import json
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any

import duckdb

RESEARCH_LIMITATIONS = (
    "这些数字是逐信号的费后研究收益估计，不等于组合收益。未模拟仓位和资本约束，"
    "没有组合净值曲线，也不能据此计算组合回撤。",
    "收益只汇总已退出且收益可核验的交易；未退出、截尾和跳过的信号单独列出。"
    "胜率分母是收益可核验的已退出交易，费后收益严格大于零才计为获胜。",
    "平均持有期只统计持有交易日数可核验的已退出交易。profit_factor 是正收益总和"
    "除以负收益总和的绝对值；没有负收益时为空，不声称无限盈利能力。",
    "费后收益按供应商复权因子和研究费率估算，没有重建分红现金、送转股权益和"
    "真实账户净损益；最低佣金及过户费等未核验费用必须保留假设说明。",
    "基本面和板块过滤依赖当时可得证据，未知状态不能当作通过。当前行业分类不能"
    "冒充历史行业，缺少历史生效时点时，历史行业条件无法核验。",
    "量价筛选未启用，不将成交量图示解释为量价规则已经参与实验。",
    "H > 0 在 DIF > DEA 且 H 与二者差值符号一致时是冗余条件，不能当成独立信号证据。",
    "观察尾端会截尾：未来没有可成交入场报价时为 censored，已经入场但未能退出时为 open。"
    "这些状态可能改变后续收益分布，不能只展示已退出交易而隐去尾端信号。",
    "公告发布时间、业务有效时间和价格观察时间分别保留。时间相邻不能证明事件导致涨跌，"
    "当时已知事件与事后新信息应在复核产物中分开。",
    "基线截图按确认年和未来20日收益方向分层，每层按代码、日期顺序选取前N条，默认两条。"
    "这属于确定性有序选取，不代表均匀样本，也不能替代全部实验的全量信号统计。",
)

CHARACTERISTIC_COLUMNS = (
    "efficiency_60",
    "wick_ratio_median_60",
    "market_cap",
    "turnover_median_20",
    "amount_median_20",
)


def enrich_signal_percentiles(output_directory: Path, *, memory_limit: str = "256MB") -> Path:
    """关联信号确认日的全市场有效特征分位，仅供展示，不参与硬筛或评分。

    输入含 signal_records.jsonl 和 daily_condition_matrix/*.csv.gz 的显式目录，
    输出 signal_characteristics.csv；原值 20 在同日有效值 [10,20,30] 中分位为 0.5。
    每日每证券一票，重复证券日期拒绝；UNKNOWN、空值及非有限数值不参与各特征排名。
    market_cap 保留来源单位，不将分位误称为实际收益；空信号生成只有表头的 CSV。
    只物化市场键/五项数值特征和信号必要字段，不复制完整宽矩阵或嵌套信号。
    DuckDB 使用临时磁盘数据库与可配置内存上限，完成后清理数据库及溢写目录。
    """
    signal_path = output_directory / "signal_records.jsonl"
    result_path = output_directory / "signal_characteristics.csv"
    header = [
        "ts_code",
        "confirmation_date",
        "signal_id",
        "experiment_id",
        *[item for column in CHARACTERISTIC_COLUMNS for item in (column, f"{column}_percentile")],
        "status",
        "entry_date",
        "exit_date",
        "fee_adjusted_research_return",
        "return_20d",
    ]
    with signal_path.open(encoding="utf-8") as signal_stream:
        has_signals = any(line.strip() for line in signal_stream)
    if not has_signals:
        import csv

        with result_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(header)
        return result_path
    matrix_paths = sorted(
        str(path) for path in (output_directory / "daily_condition_matrix").glob("*.csv.gz")
    )
    if not matrix_paths:
        raise ValueError("全市场日矩阵缺失，不能只用入选信号计算分位")
    with tempfile.TemporaryDirectory(
        prefix="signal-percentiles-", dir=output_directory
    ) as temporary_directory:
        database = duckdb.connect(str(Path(temporary_directory) / "percentiles.duckdb"))
        try:
            database.execute("SET memory_limit = ?", [memory_limit])
            database.execute("SET temp_directory = ?", [str(Path(temporary_directory) / "spill")])
            database.execute("SET preserve_insertion_order = false")
            database.execute("SET threads = 1")
            matrix_reader = (
                "read_csv(?, union_by_name=true, all_varchar=true, filename=true, "
                "buffer_size=1048576)"
            )
            available_columns = {
                str(row[0]) for row in database.execute(
                    "DESCRIBE SELECT * FROM " + matrix_reader, [matrix_paths],
                ).fetchall()
            }
            if not {"trade_date", *CHARACTERISTIC_COLUMNS} <= available_columns:
                raise ValueError("全市场日矩阵缺少特征列，无法核验分位")
            security_expression = (
                "ts_code"
                if "ts_code" in available_columns
                else "regexp_extract(filename, '([^/]+)\\.csv\\.gz$', 1)"
            )
            numeric_expressions = ", ".join(
                f"CASE WHEN isfinite(try_cast({column} AS DOUBLE)) "
                f"THEN try_cast({column} AS DOUBLE) END AS {column}"
                for column in CHARACTERISTIC_COLUMNS
            )
            database.execute(
                "CREATE TABLE market AS SELECT " + security_expression + " AS ts_code, "
                "coalesce(try_cast(trade_date AS DATE), try_strptime(trade_date, '%Y%m%d')::DATE) "
                "AS trade_date, " + numeric_expressions + " FROM " + matrix_reader,
                [matrix_paths],
            )
            invalid_keys = database.execute(
                "SELECT count(*) FROM market WHERE ts_code IS NULL OR ts_code = '' "
                "OR trade_date IS NULL"
            ).fetchone()
            if invalid_keys and invalid_keys[0]:
                raise ValueError("全市场日矩阵存在无法核验的证券或日期")
            duplicates = database.execute(
                "SELECT 1 FROM market GROUP BY ts_code, trade_date HAVING count(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicates:
                raise ValueError("全市场日矩阵证券日期重复，同日每证券必须只有一票")
            signal_reader = "read_ndjson(?, union_by_name=true)"
            signal_columns = {
                str(row[0]) for row in database.execute(
                    "DESCRIBE SELECT * FROM " + signal_reader, [str(signal_path)],
                ).fetchall()
            }
            result_expressions = [
                column if column in signal_columns else f"NULL AS {column}"
                for column in ("status", "entry_date", "exit_date", "fee_adjusted_research_return")
            ]
            result_expressions.append(
                "try_cast(json_extract(to_json(post_signal_returns), '$.return_20d') AS DOUBLE) "
                "AS return_20d"
                if "post_signal_returns" in signal_columns
                else "NULL AS return_20d"
            )
            database.execute(
                "CREATE TABLE signal_keys AS SELECT ts_code, "
                "coalesce(try_cast(confirmation_date AS DATE), "
                "try_strptime(cast(confirmation_date AS VARCHAR), '%Y%m%d')::DATE) "
                "AS confirmation_date, "
                "signal_id, experiment_id, "
                + ", ".join(result_expressions)
                + " FROM " + signal_reader,
                [str(signal_path)],
            )
            joins = []
            selections = [
                "signals.ts_code",
                "signals.confirmation_date",
                "signals.signal_id",
                "signals.experiment_id",
            ]
            for column in CHARACTERISTIC_COLUMNS:
                database.execute(
                    f"CREATE TABLE ranked_{column} AS SELECT ts_code, trade_date, {column}, "
                    f"percent_rank() OVER (PARTITION BY trade_date ORDER BY {column}) "
                    f"AS {column}_percentile FROM market WHERE {column} IS NOT NULL"
                )
                # 独立按特征有效样本排名，再按完整证券日期关联；未知值不会取得零分位。
                joins.append(
                    f"LEFT JOIN ranked_{column} AS rank_{column} ON "
                    f"signals.ts_code = rank_{column}.ts_code AND "
                    f"signals.confirmation_date = rank_{column}.trade_date"
                )
                selections.extend((f"rank_{column}.{column}", f"rank_{column}.{column}_percentile"))
            selections.extend(
                f"signals.{column}"
                for column in (
                    "status",
                    "entry_date",
                    "exit_date",
                    "fee_adjusted_research_return",
                    "return_20d",
                )
            )
            query = (
                "SELECT "
                + ", ".join(selections)
                + " FROM signal_keys AS signals "
                + " ".join(joins)
            )
            query += (
                " ORDER BY signals.confirmation_date, signals.ts_code, "
                "signals.signal_id, signals.experiment_id"
            )
            database.execute(
                "COPY (" + query + ") TO ? (HEADER, DELIMITER ',')", [str(result_path)]
            )
        finally:
            database.close()
    return result_path


def _finite_number(value: object) -> float | None:
    """读取有限研究指标；不把缺失、布尔值、非数字或无穷大改成零。

    输入 0.1，输出 0.1；输入 None、True 或 NaN，输出 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _summarize_group(
    trades: Sequence[Mapping[str, object]], experiment_id: str, experiment_name: str
) -> dict[str, Any]:
    """统计一组信号，所有状态完整计数，收益和持有期只采纳可核验的 closed。

    输入一笔 closed 收益 0.1 和一笔 open，输出 signals=2、closed=1、open=1、win_rate=1。
    缺失 closed 收益单列 unavailable；持有期分母独立列为 closed_holding_observations。
    """
    counts = {status: 0 for status in ("closed", "open", "censored", "skipped")}
    returns: list[float] = []
    holding_sessions: list[float] = []
    for trade in trades:
        status = str(trade.get("status", ""))
        if status not in counts:
            raise ValueError("交易状态必须为 closed/open/censored/skipped")
        counts[status] += 1
        if status == "closed":
            research_return = _finite_number(trade.get("fee_adjusted_research_return"))
            if research_return is not None:
                returns.append(research_return)
            holding_period = _finite_number(trade.get("holding_sessions"))
            if holding_period is not None and holding_period >= 0:
                holding_sessions.append(holding_period)
    positive_returns = sum(value for value in returns if value > 0)
    absolute_negative_returns = -sum(value for value in returns if value < 0)
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "signals": len(trades),
        **counts,
        "closed_return_observations": len(returns),
        "closed_return_unavailable": counts["closed"] - len(returns),
        "mean_fee_adjusted_research_return": mean(returns) if returns else None,
        "median_fee_adjusted_research_return": median(returns) if returns else None,
        "win_rate": sum(value > 0 for value in returns) / len(returns) if returns else None,
        "closed_holding_observations": len(holding_sessions),
        "average_holding_sessions": mean(holding_sessions) if holding_sessions else None,
        "profit_factor": positive_returns / absolute_negative_returns
        if absolute_negative_returns
        else None,
    }


def summarize_trades(
    trades: Iterable[Mapping[str, object]], experiment_names: Sequence[str] | Mapping[str, str]
) -> dict[str, Any]:
    """返回逐实验及逐确认年的研究汇总，不累乘信号收益形成伪组合绩效。

    输入 simulation 输出及 {'baseline':'基线'}，输出 experiments 和 by_confirmation_year。
    声明但无信号的实验显示零计数与空指标；未知实验保留原 ID，不丢弃任何信号。
    年份属于确认日而非退出日；所有闭合收益均以小数比例保存，不写成虚构百分数。
    """
    names = (
        dict(experiment_names)
        if isinstance(experiment_names, Mapping)
        else {name: name for name in experiment_names}
    )
    experiment_groups: dict[str, list[Mapping[str, object]]] = {name: [] for name in names}
    yearly_groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for trade in trades:
        experiment_id = str(trade.get("experiment_id", "default"))
        confirmation_year = date.fromisoformat(str(trade["confirmation_date"])).year
        experiment_groups.setdefault(experiment_id, []).append(trade)
        yearly_groups.setdefault((experiment_id, confirmation_year), []).append(trade)
    experiment_rows = [
        _summarize_group(group, experiment_id, names.get(experiment_id, experiment_id))
        for experiment_id, group in experiment_groups.items()
    ]
    yearly_rows = [
        {
            **_summarize_group(group, experiment_id, names.get(experiment_id, experiment_id)),
            "confirmation_year": confirmation_year,
        }
        for (experiment_id, confirmation_year), group in sorted(yearly_groups.items())
    ]
    return {
        "schema_version": 1,
        "return_semantics": "fee_adjusted_per_signal_research_estimate_not_portfolio_return",
        "portfolio_return_available": False,
        "portfolio_drawdown_available": False,
        "return_denominator": "closed_with_finite_fee_adjusted_research_return",
        "holding_denominator": "closed_with_finite_nonnegative_holding_sessions",
        "experiments": experiment_rows,
        "by_confirmation_year": yearly_rows,
        "limitations": list(RESEARCH_LIMITATIONS),
    }


def _markdown_cell(value: object) -> str:
    """将表格值安全转为 Markdown 单元格，空指标明确显示无法核验。

    输入 None，输出 '无法核验'；输入 '<script>|x\n'，输出转义后的单行文字。
    """
    if value is None:
        return "无法核验"
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    escaped_text = (
        html.escape(text, quote=True).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")
    )
    for character in ("\\", "[", "]", "(", ")", "`", "*", "_"):
        escaped_text = escaped_text.replace(character, "\\" + character)
    return escaped_text


def _report_manifest_excerpt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """压缩仅供人读的运行记录，完整日历和缺失日期始终保留在报告 JSON。

    输入 data.trading_dates=['20240102','20240103']，输出缓存日期起止和数量。
    接口覆盖只展示通过/缺失数量及状态；非标准字段按安全JSON文本截取，避免巨大元数据刷屏。
    """
    excerpt: dict[str, Any] = {}
    for key, value in manifest.items():
        if key == "data" and isinstance(value, Mapping):
            calendar = value.get("trading_dates", [])
            excerpt[key] = {
                "cache_start_date": calendar[0] if calendar else None,
                "cache_end_date": calendar[-1] if calendar else None,
                "cached_trading_sessions": len(calendar),
                "historical_fundamentals": value.get("historical_fundamentals", "unknown"),
                "historical_sector": value.get("historical_sector", "unknown"),
                "api_counts": value.get("api_counts", {}),
                "partition_coverage": {
                    api_name: {
                        "status": coverage.get("status", "unknown"),
                        "passed_date_count": len(coverage.get("passed_dates", [])),
                        "missing_date_count": len(coverage.get("missing_dates", [])),
                    }
                    for api_name, coverage in value.get("partition_coverage", {}).items()
                },
            }
        else:
            serialized_value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            excerpt[key] = (
                value if len(serialized_value) <= 2000 else serialized_value[:2000] + "（节选）"
            )
    return excerpt


def write_research_report(
    summary: Mapping[str, Any], manifest: Mapping[str, Any], output_directory: Path
) -> tuple[Path, Path]:
    """将真实汇总和运行假设保存为中文 Markdown 与 JSON；缓存位置由调用者明确指定。

    输入 summarize_trades 输出和不含凭证的运行 manifest，输出 (report.md, report.json)。
    空实验仍保留表头与限制说明，不推断收益、行业历史或未知基本面；非 JSON 值会报错。
    manifest 原值仅保存在 JSON，Markdown 中以 HTML 转义的预格式文本展示，避免执行注入。
    """
    document = {
        "summary": dict(summary),
        "manifest": dict(manifest),
        "research_limitations": list(RESEARCH_LIMITATIONS),
    }
    serialized_document = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    lines = [
        "# MACD 信号研究报告",
        "",
        *RESEARCH_LIMITATIONS[:2],
        "",
        "收益以小数比例表示。空指标表示没有可核验样本，不代表收益为零。",
        "",
    ]
    excerpt = _report_manifest_excerpt(manifest)
    protocol = manifest.get("protocol", {})
    data = excerpt.get("data", {})
    counts = manifest.get("counts", {})
    missing_quotes = counts.get("securities_without_valid_factor_quotes")
    missing_points = counts.get("securities_without_calendar_points")
    lines.extend(
        [
            "## 日期窗口与数据覆盖",
            "",
            f"请求确认窗口：{_markdown_cell(protocol.get('start_date'))} 至 "
            f"{_markdown_cell(protocol.get('end_date'))}。",
            f"实际缓存日历：{_markdown_cell(data.get('cache_start_date'))} 至 "
            f"{_markdown_cell(data.get('cache_end_date'))}，"
            f"交易日数 {_markdown_cell(data.get('cached_trading_sessions'))}。",
            f"完成证券数：{_markdown_cell(counts.get('securities_processed'))}；"
            f"缺有效因子报价证券数：{_markdown_cell(missing_quotes)}；"
            f"缺日历指标证券数：{_markdown_cell(missing_points)}。",
            f"历史基本面：{_markdown_cell(data.get('historical_fundamentals', 'unknown'))}；"
            f"历史行业：{_markdown_cell(data.get('historical_sector', 'unknown'))}。",
            "分区日期齐全仅说明接口日期覆盖，不能据此证明每个证券日都有可用因子、限价或基本面。"
            "本报告为技术条件研究，完整策略及历史事件覆盖仍未核验。",
            "",
            "| 接口 | 分区状态 | 通过日期数 | 缺失日期数 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for api_name, coverage in data.get("partition_coverage", {}).items():
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    api_name,
                    coverage["status"],
                    coverage["passed_date_count"],
                    coverage["missing_date_count"],
                )
            )
            + " |"
        )
    lines.append("")
    columns = (
        ("experiment_name", "实验"),
        ("signals", "信号"),
        ("closed", "已退出"),
        ("open", "未退出"),
        ("censored", "截尾"),
        ("skipped", "跳过"),
        ("closed_return_observations", "可核验收益"),
        ("closed_return_unavailable", "收益缺失"),
        ("mean_fee_adjusted_research_return", "平均费后研究收益"),
        ("median_fee_adjusted_research_return", "中位费后研究收益"),
        ("win_rate", "胜率"),
        ("average_holding_sessions", "平均持有交易日"),
        ("profit_factor", "profit_factor"),
    )
    for section_key, section_title in (
        ("experiments", "逐实验统计"),
        ("by_confirmation_year", "逐确认年统计"),
    ):
        section_columns = (
            (("confirmation_year", "确认年"), *columns)
            if section_key == "by_confirmation_year"
            else columns
        )
        lines.extend(
            [
                f"## {section_title}",
                "",
                "| " + " | ".join(label for _, label in section_columns) + " |",
                "| " + " | ".join("---" for _ in section_columns) + " |",
            ]
        )
        for row in summary.get(section_key, []):
            lines.append(
                "| " + " | ".join(_markdown_cell(row.get(key)) for key, _ in section_columns) + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## 假设与限制",
            "",
            *[f"- {item}" for item in RESEARCH_LIMITATIONS],
            "",
            "## 本次运行记录",
            "",
            "以下原始参数用于复核，未提供的条件不作推断。",
            "",
            "<pre>"
            + html.escape(json.dumps(excerpt, ensure_ascii=False, allow_nan=False, indent=2))
            + "</pre>",
            "",
        ]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    markdown_path, json_path = output_directory / "report.md", output_directory / "report.json"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(serialized_document, encoding="utf-8")
    return markdown_path, json_path
