"""离线验证信号证据时间边界、抽样、静态图片及安全索引。"""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tushare_entitlement_hub.signal_research.artifacts import (
    VALUE_COLUMNS,
    EventEvidence,
    SignalArtifact,
    select_chart_signal_ids,
    write_artifact_index,
    write_signal_artifacts,
)


def example_artifact(signal_id="signal-1", return_value=0.1):
    """生成确定性三日证据；输入 id/收益，输出确认日为中间日的 SignalArtifact。

    示例：example_artifact('a', -0.1) → 2024-01-03 确认、下一日为事后区的信号。
    """
    values = {column: [10.0, 11.0, 12.0] for column in VALUE_COLUMNS}
    values["date"] = ["20240102", "20240103", "20240104"]
    values["ma60"][0] = None
    values["hist"][0] = None
    return SignalArtifact(
        signal_id,
        "000001.SZ",
        "20240103",
        values,
        {"source_units": {"volume": "lots"}},
        {"cross": True},
        {"return_20d": return_value},
        (
            EventEvidence(
                "known",
                "2024-01-03T07:00:00Z",
                "2024-02-01",
                pit_available_at="2024-01-03T15:00:00+08:00",
            ),
            EventEvidence("after close", "2024-01-03T15:01:00+08:00", "2024-01-02"),
            EventEvidence("missing publication"),
            EventEvidence("naive", "2024-01-02T12:00:00"),
            EventEvidence("not found", verification_status="not_found"),
            EventEvidence("cannot verify", verification_status="unverifiable"),
        ),
    )


def test_snapshot_and_post_review_have_distinct_evidence_and_png(tmp_path):
    """场景：前中后三日与多种事件状态；预期：快照无未来、事后独立、PNG真实可读。"""
    from PIL import Image

    entry = write_signal_artifacts(example_artifact(), tmp_path)
    evidence = json.loads((tmp_path / entry.evidence_json).read_text())
    snapshot = evidence["snapshot"]
    assert [row["date"] for row in snapshot["observations"]] == ["2024-01-02", "2024-01-03"]
    assert snapshot["events"][0]["title"] == "known"
    assert len(snapshot["events"]) == 1
    assert "post_signal_returns" not in snapshot
    assert (
        json.loads((tmp_path / entry.evidence_json).with_name("snapshot.json").read_text())
        == snapshot
    )
    post_review = evidence["post_signal_review"]
    assert post_review["observations"][0]["date"] == "2024-01-04"
    assert post_review["returns"]["return_20d"] == 0.1
    assert [event["information_state"] for event in post_review["events"]] == [
        "known_at_confirmation",
        "post_signal_information",
        "unverifiable",
        "unverifiable",
        "not_found",
        "unverifiable",
    ]
    assert all(event["causal_claim"] == "not_established" for event in post_review["events"])
    with (tmp_path / entry.snapshot_csv).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]["ma60"] == ""
    with (tmp_path / entry.post_review_csv).open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    for relative_path in (entry.snapshot_png, entry.post_review_png):
        with Image.open(tmp_path / relative_path) as image:
            assert image.size == (1440, 960)
            image.verify()


def test_no_future_chart_and_rerun_cleans_stale_images(tmp_path):
    """场景：只有确认日前数据以及禁用截图重跑；预期：无事后图、旧图清除且证据全保留。"""
    artifact = example_artifact()
    values = {column: sequence[:2] for column, sequence in artifact.dated_values.items()}
    artifact = replace(artifact, dated_values=values, events=())
    entry = write_signal_artifacts(artifact, tmp_path)
    assert entry.post_review_png == ""
    evidence = json.loads((tmp_path / entry.evidence_json).read_text())
    assert evidence["post_signal_review"]["event_search_state"] == "unverifiable"
    entry = write_signal_artifacts(artifact, tmp_path, render_charts=False)
    assert entry.snapshot_png == "" and entry.post_review_png == ""
    assert not (tmp_path / entry.evidence_json).with_name("snapshot.png").exists()


@pytest.mark.parametrize(
    "change",
    [
        "empty_id",
        "empty_code",
        "missing",
        "empty_dates",
        "duplicate",
        "unordered",
        "no_confirmation",
        "length",
        "invalid_date",
        "string",
        "boolean",
        "none_price",
        "infinite",
        "negative_volume",
        "invalid_event",
        "nan_metric",
    ],
)
def test_invalid_evidence_fails_before_writing(tmp_path, change):
    """场景：不完整或不可核验的结构输入；预期：明确拒绝且不留下看似完整产物。"""
    artifact = example_artifact()
    values = {column: list(sequence) for column, sequence in artifact.dated_values.items()}
    if change == "empty_id":
        artifact = replace(artifact, signal_id="")
    elif change == "empty_code":
        artifact = replace(artifact, ts_code="")
    elif change == "missing":
        del values["dif"]
    elif change == "empty_dates":
        values["date"] = []
    elif change == "duplicate":
        values["date"][1] = values["date"][0]
    elif change == "unordered":
        values["date"].reverse()
    elif change == "no_confirmation":
        artifact = replace(artifact, confirmation_date="20240105")
    elif change == "length":
        values["open"].pop()
    elif change == "invalid_date":
        values["date"][0] = "bad"
    elif change == "string":
        values["close"][0] = "10"
    elif change == "boolean":
        values["close"][0] = True
    elif change == "none_price":
        values["close"][0] = None
    elif change == "infinite":
        values["close"][0] = float("inf")
    elif change == "negative_volume":
        values["volume"][0] = -1
    elif change == "invalid_event":
        artifact = replace(
            artifact, events=(EventEvidence("event", verification_status="unknown"),)
        )
    else:
        artifact = replace(artifact, pre_signal_metrics={"x": float("nan")})
    artifact = replace(artifact, dated_values=values)
    with pytest.raises(ValueError):
        write_signal_artifacts(artifact, tmp_path)
    assert not list(tmp_path.iterdir())


def test_stratified_sampling_is_order_independent_and_bounded():
    """场景：各年份正负零及未知收益；预期：每层最多两条且输入顺序不影响选择。"""
    artifacts = [
        example_artifact(f"{year}-{outcome}-{number}", outcome)
        for year in (2023, 2024)
        for outcome in (0.1, -0.1, 0, None, True, float("nan"))
        for number in range(4)
    ]
    artifacts = [
        replace(artifact, confirmation_date=artifact.signal_id[:4] + "0103")
        for artifact in artifacts
    ]
    selected = select_chart_signal_ids(artifacts)
    assert selected == select_chart_signal_ids(reversed(artifacts))
    assert len(selected) == 16
    assert select_chart_signal_ids(artifacts, per_stratum=0) == frozenset()
    assert select_chart_signal_ids([artifacts[0], artifacts[0]]) == frozenset(
        {artifacts[0].signal_id}
    )
    with pytest.raises(ValueError, match="不得为负"):
        select_chart_signal_ids(artifacts, per_stratum=-1)


def test_index_escapes_text_and_links_to_all_evidence(tmp_path):
    """场景：提供含 HTML 的标识和路径；预期：脚本不执行且 JSON/CSV/PNG 均可点击。"""
    entry = write_signal_artifacts(
        replace(example_artifact(), signal_id='<script>"x"</script>'), tmp_path
    )
    csv_path, html_path = write_artifact_index(iter([entry]), tmp_path)
    report = html_path.read_text()
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert f'href="{entry.evidence_json}"' in report
    assert f'href="{entry.snapshot_png}"' in report
    with csv_path.open() as stream:
        assert list(csv.DictReader(stream))[0]["signal_id"] == entry.signal_id
    no_charts = replace(entry, snapshot_png="", post_review_png="")
    write_artifact_index([no_charts], tmp_path)
    write_artifact_index([], tmp_path)
    assert "</tbody></table>" in html_path.read_text()


@pytest.mark.parametrize("unsafe_path", ["https://example.com", "/tmp/a", "../a", r"a\b"])
def test_index_rejects_external_or_traversing_links(tmp_path, unsafe_path):
    """场景：索引路径被替换成外部或穿越链接；预期：拒绝危险链接。"""
    entry = write_signal_artifacts(example_artifact(), tmp_path, render_charts=False)
    with pytest.raises(ValueError, match="安全的本地相对路径"):
        write_artifact_index([replace(entry, evidence_json=unsafe_path)], tmp_path)


def test_signal_id_cannot_escape_cache_directory(tmp_path):
    """场景：signal_id 含目录穿越；预期：哈希目录限定在显式缓存根内。"""
    entry = write_signal_artifacts(
        replace(example_artifact(), signal_id="../../escape"), tmp_path, render_charts=False
    )
    assert len(Path(entry.evidence_json).parts) == 2
    assert (tmp_path / entry.evidence_json).resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    "published_at,available_at,expected_state,verified",
    [
        ("2024-01-02T12:00:00+08:00", None, "unverifiable", False),
        ("2024-01-02T12:00:00+08:00", "bad", "unverifiable", False),
        ("2024-01-02T12:00:00+08:00", "2024-01-02T13:00:00", "unverifiable", False),
        ("2024-01-02T12:00:00+08:00", "2024-01-01T13:00:00+08:00", "unverifiable", False),
        ("2024-01-02T12:00:00+08:00", "2024-01-03T15:01:00+08:00", "unverifiable", True),
        ("2024-01-03T16:00:00+08:00", None, "post_signal_information", False),
        ("2024-01-03T16:00:00+08:00", "2024-01-03T17:00:00+08:00", "post_signal_information", True),
    ],
)
def test_event_pit_contract_fails_closed_without_historical_availability(
    tmp_path,
    published_at,
    available_at,
    expected_state,
    verified,
):
    """场景：缺失、无时区、非法、早于发布或未来PIT；预期：不进入历史已知快照。"""
    event = EventEvidence("event", published_at, pit_available_at=available_at)
    entry = write_signal_artifacts(
        replace(example_artifact(), events=(event,)), tmp_path, render_charts=False
    )
    evidence = json.loads((tmp_path / entry.evidence_json).read_text())
    assert evidence["snapshot"]["events"] == []
    record = evidence["post_signal_review"]["events"][0]
    assert record["published_at"] == published_at and record["pit_available_at"] == available_at
    assert record["information_state"] == expected_state
    assert record["pit_availability_verified"] is verified
    assert record["pit_known_at_confirmation"] is False


def test_chart_candles_histogram_colors_and_identity_preserve_source_values(tmp_path, monkeypatch):
    """场景：涨跌平K线及缺失MACD；预期：OHLC位置准确、正红负绿、空柱不传NaN、标题可定位。"""
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    captured = []
    original_savefig = Figure.savefig

    def inspect_figure(figure, *arguments, **keywords):
        """截取保存时的完整三面板；输入figure，输出轴与patch数值供独立断言。"""
        captured.append(
            {
                "title": figure.axes[0].get_title(),
                "candles": [
                    (patch.get_x(), patch.get_y(), patch.get_height())
                    for patch in figure.axes[0].patches
                    if hasattr(patch, "get_height")
                ],
                "histogram": [
                    (patch.get_height(), patch.get_facecolor())
                        for patch in figure.axes[1].containers[0].patches
                ],
            }
        )
        return original_savefig(figure, *arguments, **keywords)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    artifact = example_artifact()
    values = dict(artifact.dated_values)
    values.update(
        open=[9, 12, 12],
        close=[10, 11, 12],
        high=[11, 13, 13],
        low=[8, 10, 11],
        hist=[None, 0.2, -0.3],
    )
    write_signal_artifacts(replace(artifact, dated_values=values), tmp_path)
    assert "000001.SZ | confirmation 2024-01-03" in captured[0]["title"]
    assert "Post-event review" in captured[1]["title"]
    assert captured[0]["candles"] == [(-0.3, 9, 1), (0.7, 11, 1)]
    assert captured[1]["candles"][2] == (1.7, 12, 0)
    assert captured[1]["histogram"] == [(0.2, to_rgba("#d73027")), (-0.3, to_rgba("#1a9850"))]
