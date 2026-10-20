"""离线验证公告时间、结构化证据及无事件结论的保守边界。"""

from datetime import UTC, datetime

import pytest

from tushare_entitlement_hub.signal_research.event_context import (
    _timestamp,
    review_event_context,
)

SECURITY = "000001.SZ"
SIGNAL = "2024-01-01T08:00:00Z"
ENTRY = "2024-01-02T08:00:00Z"
EXIT = "2024-01-04T08:00:00Z"


def event(published_at=SIGNAL, **changes):
    """构造同证券有结构化来源的离线事件；例 event() 输出业绩事件 e1。"""
    return {
        "event_id": "e1",
        "ts_code": SECURITY,
        "published_at": published_at,
        "pit_available_at": published_at,
        "source_url": "https://www.cninfo.com.cn/announcement/e1",
        "structured_tags": [
            {"category": "业绩", "evidence": "预计年度亏损", "source_url": "https://source/e1"}
        ],
        **changes,
    }


def verification(**changes):
    """构造完整人工复核来源；例 verification() 输出同证券覆盖交易区间的声明。"""
    return {
        "ts_code": SECURITY,
        "coverage_start": SIGNAL,
        "coverage_end": EXIT,
        "coverage_complete": True,
        "reviewed_by": "reviewer",
        "source_urls": ["https://www.cninfo.com.cn/"],
        "conclusion": "verified_no_material_event",
        **changes,
    }


@pytest.mark.parametrize(
    ("published_at", "expected"),
    [
        (SIGNAL, "signal_known"),
        (ENTRY, "pre_entry_new"),
        ("2024-01-03T08:00:00Z", "holding_new"),
        (EXIT, "holding_new"),
        ("2024-01-05T08:00:00Z", "post_exit"),
        (None, "unknown_publication"),
    ],
)
def test_publication_windows_and_causality(published_at, expected):
    """场景：公告落在各披露边界；预期：精确分组，退出后/未知不列为候选且不判因果。"""
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [event(published_at)])
    assert len(result["publication_windows"][expected]) == 1
    assert bool(result["candidate_explanations"]) == (
        expected not in {"post_exit", "unknown_publication"}
    )
    assert result["causality_unproven"] is True
    assert result["automatic_cause"] is None
    assert result["publication_windows"][expected][0]["causality_unproven"] is True


def test_only_exact_security_is_matched_including_market_events():
    """场景：同代码、相似代码及市场事件混合；预期：只接受明确同证券映射的事件。"""
    records = [event(), event(ts_code="000002.SZ"), event(ts_code="000001"), event(ts_code="市场")]
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, records)
    assert len(result["candidate_explanations"]) == 1


@pytest.mark.parametrize("available_at", [None, "bad", "2023-12-31T08:00:00Z"])
def test_missing_invalid_or_premature_availability_fails_closed(available_at):
    """场景：系统可用时间缺失、非法或早于公开披露；预期：保留事后证据，历史可用性均为假。"""
    record = event(pit_available_at=available_at, observed_at=SIGNAL, effective_at=SIGNAL)
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [record])
    candidate = result["candidate_explanations"][0]
    assert candidate["publication_window"] == "signal_known"
    assert candidate["pit_availability_verified"] is False
    assert not any(
        candidate[key] for key in ("pit_known_at_signal", "pit_known_at_entry", "pit_known_at_exit")
    )
    assert candidate["raw_record"]["observed_at"] == SIGNAL


def test_later_system_availability_does_not_change_publication_window():
    """场景：公告信号前公开但系统入场后获得；预期：披露组保留，信号及入场 PIT 不可用。"""
    record = event(pit_available_at="2024-01-03T08:00:00Z")
    candidate = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [record])[
        "candidate_explanations"
    ][0]
    assert candidate["pit_availability_verified"] is True
    assert candidate["pit_known_at_signal"] is False
    assert candidate["pit_known_at_entry"] is False
    assert candidate["pit_known_at_exit"] is True


@pytest.mark.parametrize("category", ["业绩", "减持", "监管", "企业行动", "市场", "行业"])
def test_all_supported_structured_categories_preserve_evidence(category):
    """场景：六类显式结构化标签；预期：逐字保存证据和 URL，历史已知时点为真。"""
    record = event(
        structured_tags=[
            {"category": category, "evidence": "结构化原文", "source_url": "https://source/e1"}
        ]
    )
    candidate = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [record])[
        "candidate_explanations"
    ][0]
    assert candidate["structured_tags"] == record["structured_tags"]
    assert candidate["source_url"] == record["source_url"]
    assert all(
        candidate[key] for key in ("pit_known_at_signal", "pit_known_at_entry", "pit_known_at_exit")
    )


@pytest.mark.parametrize(
    "labels",
    [
        None,
        "减持",
        ["减持"],
        [{"category": "未知", "evidence": "原文", "source_url": "x"}],
        [{"category": "减持", "evidence": "", "source_url": "x"}],
        [{"category": "减持", "evidence": "原文"}],
        [{"category": []}],
        [{"category": "减持", "evidence": 1, "source_url": "x"}],
        [{"category": "减持", "evidence": "原文", "source_url": 1}],
        [{"category": "减持", "evidence": "原文", "source_url": " "}],
    ],
)
def test_title_and_unsubstantiated_labels_never_infer_event_category(labels):
    """场景：标题含减持但标签缺失、错误或无来源；预期：保存原事件而不猜类别或候选原因。"""
    result = review_event_context(
        SECURITY, SIGNAL, ENTRY, EXIT, [event(title="减持监管导致下跌", structured_tags=labels)]
    )
    assert result["candidate_explanations"] == []
    assert result["evidence_status"] == "event_evidence_present"


def test_report_period_and_effective_time_cannot_replace_publication():
    """场景：只有报告期、观察日和除权日；预期：披露未知且不作为交易期候选解释。"""
    record = event(None, report_period="2023-12-31", effective_at=ENTRY, observed_at=SIGNAL)
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [record])
    assert result["candidate_explanations"] == []
    assert result["publication_windows"]["unknown_publication"][0]["raw_record"] == record


def test_no_event_evidence_is_distinct_from_verified_absence():
    """场景：空事件列表与完整人工复核；预期：分别输出无证据与已复核无重大事件。"""
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [])
    assert result["evidence_status"] == "no_event_evidence"
    assert result["verification"] is None
    verified = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [], verification=verification())
    assert verified["evidence_status"] == "verified_no_material_event"
    assert verified["verification"] == verification()
    assert verified["causality_unproven"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"ts_code": "000002.SZ"},
        {"coverage_start": ENTRY},
        {"coverage_end": ENTRY},
        {"coverage_start": None},
        {"coverage_end": None},
        {"coverage_complete": False},
        {"reviewed_by": " "},
        {"reviewed_by": None},
        {"source_urls": []},
        {"source_urls": None},
        {"source_urls": [1]},
        {"source_urls": [" "]},
        {"conclusion": "no_results"},
    ],
)
def test_incomplete_verification_never_confirms_absence(changes):
    """场景：复核声明缺来源、人工、范围或结论；预期：保持无事件证据，不确认无重大事件。"""
    result = review_event_context(
        SECURITY, SIGNAL, ENTRY, EXIT, [], verification=verification(**changes)
    )
    assert result["evidence_status"] == "no_event_evidence"


@pytest.mark.parametrize("published_at", [SIGNAL, None])
def test_contradictory_or_unknown_events_prevent_verified_absence(published_at):
    """场景：无重大事件声明与实际/未知时点证据冲突；预期：保守返回存在事件证据。"""
    result = review_event_context(
        SECURITY, SIGNAL, ENTRY, EXIT, [event(published_at)], verification=verification()
    )
    assert result["evidence_status"] == "event_evidence_present"


def test_post_exit_event_does_not_contradict_review_of_trade_window():
    """场景：只有退出后公告且交易期已完整复核；预期：保留退出后记录，允许交易期无重大事件。"""
    result = review_event_context(
        SECURITY, SIGNAL, ENTRY, EXIT, [event("2024-02-01")], verification=verification()
    )
    assert result["evidence_status"] == "verified_no_material_event"
    assert result["candidate_explanations"] == []


def test_outputs_do_not_share_mutable_evidence_with_input_or_each_other():
    """场景：消费方修改候选、原始记录和复核声明；预期：输入及其他输出副本均不变。"""
    record = event()
    review = verification()
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [record], verification=review)
    result["candidate_explanations"][0]["raw_record"]["structured_tags"].clear()
    result["candidate_explanations"][0]["structured_tags"][0]["evidence"] = "changed"
    result["verification"]["source_urls"].clear()
    assert record == event()
    assert review == verification()
    assert result["publication_windows"]["signal_known"][0]["raw_record"] == record


@pytest.mark.parametrize("value", [None, 1, "bad", "2024-13-40", "2024-01-01T08:00:00"])
def test_invalid_timestamp_is_unknown(value):
    """场景：日期非法或时区缺失；预期：返回未知，不猜时间边界。"""
    assert _timestamp(value) is None


def test_date_precision_uses_shanghai_end_of_day_and_aware_datetime_is_supported():
    """场景：上海日粒度公告与显式时间；预期：日末可用，收盘时刻不得前视当天未知公告。"""
    expected = datetime(2024, 1, 1, 15, 59, 59, 999999, tzinfo=UTC)
    assert _timestamp("20240101") == expected
    assert _timestamp("2024-01-01") == expected
    assert _timestamp(expected) == expected
    assert _timestamp(datetime(2024, 1, 1)) is None
    result = review_event_context(SECURITY, SIGNAL, ENTRY, EXIT, [event("2024-01-01")])
    assert len(result["publication_windows"]["pre_entry_new"]) == 1


@pytest.mark.parametrize(
    "security,signal,entry,exit_time",
    [
        ("", SIGNAL, ENTRY, EXIT),
        (None, SIGNAL, ENTRY, EXIT),
        (SECURITY, "bad", ENTRY, EXIT),
        (SECURITY, SIGNAL, "bad", EXIT),
        (SECURITY, SIGNAL, ENTRY, "bad"),
        (SECURITY, ENTRY, SIGNAL, EXIT),
        (SECURITY, SIGNAL, EXIT, ENTRY),
    ],
)
def test_invalid_trade_identity_or_chronology_is_rejected(security, signal, entry, exit_time):
    """场景：证券空白、时点非法或入退出乱序；预期：拒绝复核以免生成错误交易证据。"""
    with pytest.raises(ValueError, match="按先后排列"):
        review_event_context(security, signal, entry, exit_time, [])
