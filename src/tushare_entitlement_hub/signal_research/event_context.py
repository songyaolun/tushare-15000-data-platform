"""按证券与真实披露时间复核交易事件，保留候选解释而不自动归因。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

EVENT_CATEGORIES = frozenset({"业绩", "减持", "监管", "企业行动", "市场", "行业"})
PUBLICATION_WINDOWS = (
    "signal_known",
    "pre_entry_new",
    "holding_new",
    "post_exit",
    "unknown_publication",
)


def _timestamp(value: Any) -> datetime | None:
    """将带时区时间或上海日粒度日期转 UTC；未知及无时区时间返回 None。

    日期在上海日末才可用，不用报告期或除权日替代披露时刻。
    例如输入 '2024-01-02' 输出 2024-01-02T15:59:59.999999+00:00；
    输入 '2024-01-02T09:00:00' 输出 None，避免默默猜测时区。
    """
    if not isinstance(value, (str, datetime)):
        return None
    try:
        if isinstance(value, str) and len(value) in {8, 10}:
            date_format = "%Y%m%d" if len(value) == 8 else "%Y-%m-%d"
            day = datetime.strptime(value, date_format).date()
            return datetime.combine(day, time.max, ZoneInfo("Asia/Shanghai")).astimezone(UTC)
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        return moment.astimezone(UTC) if moment.tzinfo is not None else None
    except ValueError:
        return None


def _window(
    published_at: datetime | None,
    signal_at: datetime,
    entry_at: datetime,
    exit_at: datetime,
) -> str:
    """按公开披露时间区分信号前、入场前、持有期、退出后和未知。

    边界使用 <=，入场当刻归入 pre_entry_new、退出当刻归入 holding_new。
    例如 published_at 等于 signal_at，输出 'signal_known'；缺失则输出
    'unknown_publication'。分组只描述披露先后，不证明历史系统已获得事件。
    """
    if published_at is None:
        return "unknown_publication"
    if published_at <= signal_at:
        return "signal_known"
    if published_at <= entry_at:
        return "pre_entry_new"
    if published_at <= exit_at:
        return "holding_new"
    return "post_exit"


def _structured_tags(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """仅接受六类结构化标签且要求对应原始证据与来源链接，不扫描标题猜标签。

    输入 structured_tags=[{'category':'业绩','evidence':'预亏','source_url':'https://x'}]
    输出该标签的副本；仅有 title='减持公告' 或裸字符串标签输出 []。
    标签由上游结构化来源或明确人工标注提供，本函数不核实证据真伪。
    """
    labels = record.get("structured_tags", [])
    if not isinstance(labels, (list, tuple)):
        return []
    return [
        deepcopy(dict(label))
        for label in labels
        if isinstance(label, Mapping)
        and isinstance(label.get("category"), str)
        and label["category"] in EVENT_CATEGORIES
        and isinstance(label.get("evidence"), str)
        and bool(label["evidence"].strip())
        and isinstance(label.get("source_url"), str)
        and bool(label["source_url"].strip())
    ]


def _verified_absence(
    verification: Mapping[str, Any] | None,
    security_code: str,
    signal_at: datetime,
    exit_at: datetime,
) -> bool:
    """要求显式同证券完整窗口与有来源的人工复核，才能确认无重大事件。

    输入 verification 包含 ts_code、coverage_start、coverage_end、
    coverage_complete=True、reviewed_by、source_urls 及
    conclusion='verified_no_material_event'，且范围覆盖 signal_at 至 exit_at，
    输出 True；没有检索结果或只有空记录列表输出 False。
    """
    if verification is None:
        return False
    start = _timestamp(verification.get("coverage_start"))
    end = _timestamp(verification.get("coverage_end"))
    sources = verification.get("source_urls")
    reviewer = verification.get("reviewed_by")
    return (
        verification.get("ts_code") == security_code
        and verification.get("conclusion") == "verified_no_material_event"
        and verification.get("coverage_complete") is True
        and isinstance(reviewer, str)
        and bool(reviewer.strip())
        and isinstance(sources, (tuple, list))
        and bool(sources)
        and all(isinstance(source, str) and bool(source.strip()) for source in sources)
        and start is not None
        and end is not None
        and start <= signal_at
        and end >= exit_at
    )


def review_event_context(
    security_code: str,
    signal_at: str | datetime,
    entry_at: str | datetime,
    exit_at: str | datetime,
    records: Iterable[Mapping[str, Any]],
    *,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """复核一个证券的一次交易的公告候选解释，纯函数且永不自动判断下跌原因。

    输入证券完整 ts_code、信号/入场/退出时点及公告 records。仅精确匹配
    record['ts_code']，市场/行业事件也需上游先提供有证据的证券映射。
    published_at 控制披露分组；pit_available_at 独立表示实际系统可用时刻，
    缺失或早于公开披露时刻时 fail closed。observed_at、effective_at、
    report_period 只保留作审计，不替代任何必要时点。所有原始记录保留深拷贝。

    candidate_explanations 仅含有结构化标签证据且在退出前披露的事件；
    缺历史 PIT 证据仍可用于事后复核，但不能作为历史信号已知输入。
    verified_no_material_event 需完整来源与人工复核，任何窗口内事件或
    未知披露事件都会保守阻止无事件确认。无匹配证据输出 no_event_evidence。
    返回 causality_unproven=True，无字段声称事件造成了跌幅。

    例如 review_event_context('000001.SZ','2024-01-01','2024-01-02',
    '2024-01-03',[]) 的 evidence_status 为 'no_event_evidence'，
    candidate_explanations=[]。无效或乱序交易时点抛出 ValueError。
    """
    signal_time, entry_time, exit_time = map(_timestamp, (signal_at, entry_at, exit_at))
    if (
        not isinstance(security_code, str)
        or not security_code.strip()
        or signal_time is None
        or entry_time is None
        or exit_time is None
        or not signal_time <= entry_time <= exit_time
    ):
        raise ValueError("证券及信号、入场、退出时间必须有效且按先后排列")
    windows: dict[str, list[dict[str, Any]]] = {name: [] for name in PUBLICATION_WINDOWS}
    candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("ts_code") != security_code:
            continue
        published_time = _timestamp(record.get("published_at"))
        available_time = _timestamp(record.get("pit_available_at"))
        valid_availability = (
            published_time is not None
            and available_time is not None
            and available_time >= published_time
        )
        window = _window(published_time, signal_time, entry_time, exit_time)
        tags = _structured_tags(record)
        event = {
            "event_id": record.get("event_id"),
            "ts_code": security_code,
            "published_at": record.get("published_at"),
            "pit_available_at": record.get("pit_available_at"),
            "source_url": record.get("source_url"),
            "publication_window": window,
            "structured_tags": tags,
            "pit_availability_verified": valid_availability,
            "pit_known_at_signal": bool(
                valid_availability and available_time is not None and available_time <= signal_time
            ),
            "pit_known_at_entry": bool(
                valid_availability and available_time is not None and available_time <= entry_time
            ),
            "pit_known_at_exit": bool(
                valid_availability and available_time is not None and available_time <= exit_time
            ),
            "causality_unproven": True,
            "raw_record": deepcopy(dict(record)),
        }
        windows[window].append(event)
        if tags and window not in {"post_exit", "unknown_publication"}:
            candidates.append(deepcopy(event))
    relevant_events = any(windows[name] for name in PUBLICATION_WINDOWS if name != "post_exit")
    verified = not relevant_events and _verified_absence(
        verification, security_code, signal_time, exit_time
    )
    status = (
        "verified_no_material_event"
        if verified
        else "event_evidence_present"
        if relevant_events
        else "no_event_evidence"
    )
    return {
        "ts_code": security_code,
        "signal_at": signal_time.isoformat(),
        "entry_at": entry_time.isoformat(),
        "exit_at": exit_time.isoformat(),
        "publication_windows": windows,
        "candidate_explanations": candidates,
        "evidence_status": status,
        "verification": deepcopy(dict(verification)) if verification is not None else None,
        "causality_unproven": True,
        "automatic_cause": None,
    }
