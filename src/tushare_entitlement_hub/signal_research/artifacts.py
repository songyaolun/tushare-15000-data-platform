"""信号证据及离线截图；确认日快照与事后观察分别保存，事件时间不代表因果。"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

VALUE_COLUMNS = ("open", "high", "low", "close", "ma20", "ma60", "dif", "dea", "hist", "volume")


@dataclass(frozen=True)
class EventEvidence:
    """事件发布时间、历史系统可用时间及业务有效时间分别保留。

    输入示例：EventEvidence('公告', '2024-01-02T12:00:00+08:00', '2024-01-03')。
    输出示例：写入事件 JSON，注明发布时间、有效日期和无法据此确定因果。
    """

    title: str
    published_at: str | None = None
    effective_at: str | None = None
    verification_status: str = "verified"
    source_url: str = ""
    pit_available_at: str | None = None


@dataclass(frozen=True)
class SignalArtifact:
    """一条信号的完整复核输入；各列必须按日期升序、等长且包含确认日。

    输入示例：SignalArtifact('s1', '000001.SZ', '20240103',
    {'date': ['20240103'], 'open': [10], ... , 'volume': [100]})。
    输出示例：write_signal_artifacts 生成该信号的 JSON、CSV 和可选 PNG。
    价格单位、成交量单位和复权口径应写入 pre_signal_metrics 的来源说明。
    """

    signal_id: str
    ts_code: str
    confirmation_date: str
    dated_values: Mapping[str, Sequence[Any]]
    pre_signal_metrics: Mapping[str, Any] = field(default_factory=dict)
    rule_outcomes: Mapping[str, Any] = field(default_factory=dict)
    post_signal_returns: Mapping[str, Any] = field(default_factory=dict)
    events: Sequence[EventEvidence] = ()


@dataclass(frozen=True)
class ArtifactIndexEntry:
    """索引行的可点击相对路径；无截图时路径为空。

    输入示例：ArtifactIndexEntry('s1','000001.SZ','2024-01-03','a/evidence.json', ...)。
    输出示例：write_artifact_index 生成 CSV 和 HTML 中的单行。
    """

    signal_id: str
    ts_code: str
    confirmation_date: str
    evidence_json: str
    snapshot_csv: str
    post_review_csv: str
    snapshot_png: str = ""
    post_review_png: str = ""


def _normalized_date(value: str) -> str:
    """将 ISO 或 Tushare 紧凑日期转为 ISO；无效日期报错。

    输入 '20240103'，输出 '2024-01-03'；输入 'bad'，抛出 ValueError。
    """
    return date.fromisoformat(value).isoformat()


def _validated_rows(artifact: SignalArtifact) -> list[dict[str, Any]]:
    """验证时间轴和数值，生成可审计的逐日数据；不排序掩盖输入问题。

    输入日期升序、列等长的 SignalArtifact，输出 [{'date':'2024-01-03', ...}]。
    指标暖启动可为 None；OHLC 和成交量必须为有限数字，负成交量被拒绝。
    """
    required_columns = ("date", *VALUE_COLUMNS)
    if not artifact.signal_id or not artifact.ts_code:
        raise ValueError("signal_id 和 ts_code 不得为空")
    if any(column not in artifact.dated_values for column in required_columns):
        raise ValueError("缺少日期、OHLC、均线、MACD 或成交量列")
    dates = [_normalized_date(str(value)) for value in artifact.dated_values["date"]]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("日期必须非空、严格递增且没有重复")
    if _normalized_date(artifact.confirmation_date) not in dates:
        raise ValueError("时间轴必须包含信号确认日")
    if any(len(artifact.dated_values[column]) != len(dates) for column in VALUE_COLUMNS):
        raise ValueError("逐日列必须等长")
    rows: list[dict[str, Any]] = []
    for position, trade_date in enumerate(dates):
        row: dict[str, Any] = {"date": trade_date}
        for column in VALUE_COLUMNS:
            value = artifact.dated_values[column][position]
            if value is None and column in {"ma20", "ma60", "dif", "dea", "hist"}:
                row[column] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{column} 必须是有限数字或允许的指标空值")
            if not math.isfinite(value) or (column == "volume" and value < 0):
                raise ValueError(f"{column} 必须是有限数字且成交量不得为负")
            row[column] = value
        rows.append(row)
    return rows


def _event_record(event: EventEvidence, confirmation_date: str) -> dict[str, Any]:
    """同时核验公开发布时间与历史系统可用时间，不用业务有效时间替代任一时点。

    输入发布于确认日次日的事件，输出 information_state='post_signal_information'。
    历史发布事件缺失PIT时点、无时区、PIT早于发布或晚于确认时均为 unverifiable。
    未来公开事件可记 post_signal_information，但没有合格PIT证据时不认证历史可用性。
    """
    if event.verification_status not in {"verified", "not_found", "unverifiable"}:
        raise ValueError("未知事件核验状态")
    information_state = event.verification_status
    pit_availability_verified = False
    if information_state == "verified":
        try:
            published_at = datetime.fromisoformat(event.published_at or "")
            if published_at.tzinfo is None:
                raise ValueError("发布时间必须带时区")
            # 日线信号按上海时区确认日收盘判定；不把当日晚间公告视为已知。
            confirmation_cutoff = datetime.fromisoformat(f"{confirmation_date}T15:00:00+08:00")
            try:
                available_at = datetime.fromisoformat(event.pit_available_at or "")
                pit_availability_verified = (
                    available_at.tzinfo is not None and available_at >= published_at
                )
            except ValueError:
                available_at = None
            if published_at > confirmation_cutoff:
                information_state = "post_signal_information"
            elif (
                pit_availability_verified
                and available_at is not None
                and available_at <= confirmation_cutoff
            ):
                information_state = "known_at_confirmation"
            else:
                information_state = "unverifiable"
        except ValueError:
            information_state = "unverifiable"
    return {
        "title": event.title,
        "published_at": event.published_at,
        "pit_available_at": event.pit_available_at,
        "pit_availability_verified": pit_availability_verified,
        "pit_known_at_confirmation": information_state == "known_at_confirmation",
        "effective_at": event.effective_at,
        "verification_status": event.verification_status,
        "information_state": information_state,
        "source_url": event.source_url,
        "causal_claim": "not_established",
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """写入稳定列序的 UTF-8 CSV，不把 None 填成零。

    输入路径与 [{'date':'2024-01-03', ...}]，输出包含表头的 CSV 文件。
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("date", *VALUE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _render_chart(
    path: Path, rows: Sequence[Mapping[str, Any]], confirmation_date: str, *, post_review: bool
) -> None:
    """离线保存价格/均线、MACD、成交量三面板；事后区间用背景显式标识。

    输入确认日及逐日数据，输出静态 PNG；post_review=False 的调用只接收当时快照。
    使用 Figure/Agg 避免 GUI 和全局 pyplot 状态，指标空值绘为断线。
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8), dpi=120)
    FigureCanvasAgg(figure)
    price_axis, macd_axis, volume_axis = figure.subplots(3, 1, sharex=True)
    positions = list(range(len(rows)))
    for column, label in (("close", "Close"), ("ma20", "MA20"), ("ma60", "MA60")):
        price_axis.plot(positions, [row[column] for row in rows], label=label)
    for column, label in (("dif", "DIF"), ("dea", "DEA")):
        macd_axis.plot(positions, [row[column] for row in rows], label=label)
    macd_axis.bar(
        positions,
        [float("nan") if row["hist"] is None else row["hist"] for row in rows],
        label="Histogram",
    )
    volume_axis.bar(positions, [row["volume"] for row in rows], label="Volume (source unit)")
    confirmation_position = next(
        position for position, row in enumerate(rows) if row["date"] == confirmation_date
    )
    for axis in (price_axis, macd_axis, volume_axis):
        axis.axvline(confirmation_position, color="black", linestyle="--", linewidth=1)
        axis.legend(loc="upper left")
        axis.grid(alpha=0.2)
        if post_review and confirmation_position < len(rows) - 1:
            axis.axvspan(confirmation_position + 0.5, len(rows) - 0.5, color="orange", alpha=0.15)
    price_axis.set_title(
        "Post-event review: orange = future observations; no causal inference"
        if post_review
        else "Signal snapshot: observations through confirmation only"
    )
    tick_positions = sorted(set((0, confirmation_position, len(rows) - 1)))
    volume_axis.set_xticks(
        tick_positions, [str(rows[position]["date"]) for position in tick_positions]
    )
    figure.tight_layout()
    figure.savefig(path, metadata={"Software": "tushare signal research"})
    figure.clear()


def write_signal_artifacts(
    artifact: SignalArtifact, output_directory: Path, *, render_charts: bool = True
) -> ArtifactIndexEntry:
    """保存每条信号的完整证据；抽样仅控制截图，不减少 JSON/CSV 覆盖。

    输入 SignalArtifact 与显式指定的缓存根目录，输出相对路径 ArtifactIndexEntry。
    snapshot.json/csv/png 严格截至确认日；evidence.json 和 post_review.* 含独立事后区。
    同一 signal_id 重跑写入同一目录，关闭截图时移除该信号旧图，防止过期索引。
    """
    rows = _validated_rows(artifact)
    confirmation_date = _normalized_date(artifact.confirmation_date)
    snapshot_rows = [row for row in rows if row["date"] <= confirmation_date]
    post_rows = [row for row in rows if row["date"] > confirmation_date]
    event_records = [_event_record(event, confirmation_date) for event in artifact.events]
    snapshot = {
        "schema_version": 1,
        "signal_id": artifact.signal_id,
        "ts_code": artifact.ts_code,
        "confirmation_date": confirmation_date,
        "confirmation_cutoff": f"{confirmation_date}T15:00:00+08:00",
        "observations": snapshot_rows,
        "pre_signal_metrics": dict(artifact.pre_signal_metrics),
        "rule_outcomes": dict(artifact.rule_outcomes),
        "events": [
            event
            for event in event_records
            if event["information_state"] == "known_at_confirmation"
        ],
    }
    evidence = {
        "snapshot": snapshot,
        "post_signal_review": {
            "observations": post_rows,
            "returns": dict(artifact.post_signal_returns),
            "events": event_records,
            "event_search_state": "provided_evidence" if event_records else "unverifiable",
            "causal_claim": "not_established",
        },
    }
    # 在创建目录前拒绝不可序列化值或 NaN，避免留下伪完整证据。
    serialized_evidence = json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2)
    serialized_snapshot = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2)
    directory_name = hashlib.sha256(artifact.signal_id.encode("utf-8")).hexdigest()[:24]
    artifact_directory = output_directory / directory_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "evidence.json").write_text(serialized_evidence, encoding="utf-8")
    (artifact_directory / "snapshot.json").write_text(serialized_snapshot, encoding="utf-8")
    _write_rows(artifact_directory / "snapshot.csv", snapshot_rows)
    _write_rows(artifact_directory / "post_review.csv", post_rows)
    for file_name in ("snapshot.png", "post_review.png"):
        (artifact_directory / file_name).unlink(missing_ok=True)
    if render_charts:
        _render_chart(
            artifact_directory / "snapshot.png", snapshot_rows, confirmation_date, post_review=False
        )
        if post_rows:
            _render_chart(
                artifact_directory / "post_review.png", rows, confirmation_date, post_review=True
            )
    return ArtifactIndexEntry(
        artifact.signal_id,
        artifact.ts_code,
        confirmation_date,
        f"{directory_name}/evidence.json",
        f"{directory_name}/snapshot.csv",
        f"{directory_name}/post_review.csv",
        f"{directory_name}/snapshot.png" if render_charts else "",
        f"{directory_name}/post_review.png" if render_charts and post_rows else "",
    )


def select_chart_signal_ids(
    artifacts: Iterable[SignalArtifact], *, per_stratum: int = 2, return_key: str = "return_20d"
) -> frozenset[str]:
    """按确认年和事后收益符号确定性抽样；未知收益另成一层，内存随层数而非信号数增长。

    输入乱序信号序列、per_stratum=2，输出每年正/负/零/未知各至多两条 signal_id。
    事后收益仅用于审核截图抽样，绝不能用此子集代替全信号统计或构建信号。
    """
    if per_stratum < 0:
        raise ValueError("每层截图数量不得为负")
    samples: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for artifact in artifacts:
        return_value = artifact.post_signal_returns.get(return_key)
        if (
            isinstance(return_value, bool)
            or not isinstance(return_value, (float, int))
            or not math.isfinite(return_value)
        ):
            outcome = "unknown"
        else:
            outcome = "positive" if return_value > 0 else "negative" if return_value < 0 else "zero"
        stratum = (_normalized_date(artifact.confirmation_date)[:4], outcome)
        ranking = (
            hashlib.sha256(artifact.signal_id.encode("utf-8")).hexdigest(),
            artifact.signal_id,
        )
        selected = samples.setdefault(stratum, [])
        if ranking not in selected:
            selected.append(ranking)
            selected.sort()
            del selected[per_stratum:]
    return frozenset(signal_id for selected in samples.values() for _, signal_id in selected)


def write_artifact_index(
    entries: Iterable[ArtifactIndexEntry], output_directory: Path
) -> tuple[Path, Path]:
    """流式写入 CSV/静态 HTML 索引，所有文本转义、链接仅允许本地产物相对路径。

    输入 write_signal_artifacts 返回行，输出 (index.csv, index.html)；空输入仍生成表头。
    输入 '<script>' 会显示为普通文字；绝对 URL、路径穿越或危险链接被拒绝。
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path, html_path = output_directory / "index.csv", output_directory / "index.html"
    columns = tuple(ArtifactIndexEntry.__dataclass_fields__)
    with (
        csv_path.open("w", encoding="utf-8", newline="") as csv_stream,
        html_path.open("w", encoding="utf-8") as html_stream,
    ):
        writer = csv.DictWriter(csv_stream, fieldnames=columns)
        writer.writeheader()
        html_stream.write(
            '<!doctype html><meta charset="utf-8"><title>Signal evidence</title>'
            "<h1>Signal evidence</h1><p>Post-event observations are separate; "
            "event timing does not establish causation.</p><table><thead><tr>"
        )
        html_stream.write("".join(f"<th>{html.escape(column)}</th>" for column in columns))
        html_stream.write("</tr></thead><tbody>")
        for entry in entries:
            row = {column: getattr(entry, column) for column in columns}
            cells = []
            for column, value in row.items():
                safe_value = html.escape(value, quote=True)
                if column.endswith(("json", "csv", "png")) and value:
                    relative_path = Path(value)
                    if (
                        relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or ":" in value
                        or "\\" in value
                    ):
                        raise ValueError("索引只允许安全的本地相对路径")
                    cells.append(f'<td><a href="{safe_value}">{safe_value}</a></td>')
                else:
                    cells.append(f"<td>{safe_value}</td>")
            writer.writerow(row)
            html_stream.write("<tr>" + "".join(cells) + "</tr>")
        html_stream.write("</tbody></table>")
    return csv_path, html_path
