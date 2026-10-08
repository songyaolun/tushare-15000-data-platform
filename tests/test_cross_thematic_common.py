"""三类研究产品的有限证据、单位和时间辅助离线测试。"""

from dataclasses import replace

import pytest
from test_research_governance import NOW, A, inputs, row

from tushare_entitlement_hub.cross_asset_research.thematic_common import (
    accepted_groups,
    comparable,
    date_text,
    finite,
    latest_snapshot,
    number,
    observation_summary,
    row_key,
    sampled_refs,
    scalar,
    text_values,
    unique_latest_by,
)


@pytest.mark.parametrize(
    "value,expected", [(None, None), ("bad", None), ("2026Q3", None), ("20260930", "2026-09-30")]
)
def test_date_conversion_never_invents_period_end(value, expected):
    """场景：合法日、缺值和季度文本；预期：只有明确日级日期可转换。"""
    assert date_text(value) == expected


def test_groups_recheck_visibility_quality_and_source_reasons():
    """场景：未来、隔离、来源拒绝及无实体行混入；预期：全部排除且保留受阻实体入口。"""
    good = row("dc_daily", raw={"ts_code": "I"})
    future = replace(
        good, availability=replace(good.availability, observed_at="2027-01-01T00:00:00Z")
    )
    bad = row("dc_daily", "bad", raw={"quality_tier": "quarantine"})
    flag = row("dc_daily", "flag", raw={"is_quarantined": True})
    unidentified = replace(good, entity_id=None)
    sample = inputs(good, future, bad, flag, unidentified)
    assert accepted_groups(sample, ("dc_daily",))[A]["dc_daily"] == (good,)
    assert (
        accepted_groups(replace(sample, source_reasons={"dc_daily": ("blocked",)}), ("dc_daily",))[
            A
        ]["dc_daily"]
        == ()
    )


def test_observation_annotations_preserve_rows_without_weakening_capacity_rejection():
    """场景：当前目录、来源内身份和字段缺口仅是观察注记；预期：保留已有行，容量拒绝仍整组关闭。"""
    observed = row("ci_index_member", raw={"ts_code": "S"})
    notes = (
        "directory_identity_observed_only",
        "source_local_entity_only",
        "consumption_fields_missing",
    )
    sample = inputs(observed, reasons={"ci_index_member": notes})
    assert accepted_groups(sample, ("ci_index_member",))[A]["ci_index_member"] == (observed,)
    assert sample.source_reasons["ci_index_member"] == notes
    blocked = replace(
        sample, source_reasons={"ci_index_member": notes + ("source_capacity_exceeded",)}
    )
    assert accepted_groups(blocked, ("ci_index_member",))[A]["ci_index_member"] == ()


def test_evidence_samples_preserve_full_count_and_deterministic_digest():
    """场景：100条成员按相反顺序输入；预期：计数100、摘要一致且证据最多8条。"""
    rows = tuple(row("dc_member", str(index), raw={"con_code": str(index)}) for index in range(100))
    summary = observation_summary(rows, ("con_code", "missing"))
    assert summary["row_count"] == 100
    assert summary["field_observed_counts"] == {"con_code": 100, "missing": 0}
    assert summary == observation_summary(rows[::-1], ("con_code", "missing"))
    assert sampled_refs(rows, ("con_code",)) == sampled_refs(rows[::-1], ("con_code",))
    assert len(sampled_refs(rows, ("con_code",))) == 8
    labels = text_values(rows, "con_code")
    assert labels["distinct_count"] == 100
    assert len(labels["samples"]) == 32
    assert row_key(rows[0])[2] == "dc_member/0"


def test_excluded_row_notes_do_not_clear_healthy_retained_rows():
    """场景：健康行同源存在已排除未来、隔离及未知质量行；预期：仅健康行保留，容量与歧义仍拒绝。"""
    good = row(
        "ths_daily",
        "good",
        raw={"quality_tier": "verified"},
        values={"turnover_rate": (0.03, "fraction")},
    )
    future = row("ths_daily", "future", public_available_at="2027-01-01T00:00:00Z")
    quarantined = row("ths_daily", "quarantine", raw={"quality_tier": "quarantine"})
    unknown = row("ths_daily", "unknown", raw={"quality_tier": "unknown"})
    notes = (
        "rows_excluded_by_adapter",
        "row_unavailable",
        "future_business_date",
        "quarantined_or_unknown_quality",
    )
    sample = inputs(good, future, quarantined, unknown, reasons={"ths_daily": notes})
    assert accepted_groups(sample, ("ths_daily",))[A]["ths_daily"] == (good,)
    assert sample.source_reasons["ths_daily"] == notes
    for hard in ("source_capacity_exceeded", "entity_unresolved"):
        rejected = replace(sample, source_reasons={"ths_daily": notes + (hard,)})
        assert accepted_groups(rejected, ("ths_daily",))[A]["ths_daily"] == ()


def test_numeric_helpers_refuse_currency_units_missing_and_overflow():
    """场景：同值不同币种、错误单位及溢出；预期：仅同口径事实能参与算术。"""
    left = row("fund_daily", values={"close": (10, "元"), "bad": (None, "元")})
    right = row("fund_nav", values={"unit_nav": (9, "元"), "usd": (9, "元", "USD")})
    assert comparable(left, "close", right, "unit_nav")
    assert not comparable(left, "close", right, "usd")
    assert not comparable(left, "bad", right, "unit_nav")
    assert number(None, "close") is None
    assert number(left, "close", "份") is None
    assert scalar("price", left, "close", "元").value == 10
    assert scalar("price", None, "close", "元").value is None
    assert (
        scalar(
            "ratio", row("daily", values={"pct_chg": (-0.1, "fraction")}), "pct_chg", "fraction"
        ).value
        == -0.1
    )
    assert finite(None) is None and finite(float("inf")) is None and finite(-2) == -2


def test_latest_snapshot_and_business_key_conflicts_fail_closed():
    """场景：两日快照、同日不同版本及缺键；预期：最新日完整保留，冲突与缺键不任意选行。"""
    old = row("dc_member", "old", day="2026-10-02", raw={"con_code": "A"})
    new = row("dc_member", "new", raw={"con_code": "A"})
    duplicate = row("dc_member", "conflict", raw={"con_code": "A"})
    missing = row("dc_member", "missing")
    assert latest_snapshot((old, new)) == (new,)
    assert latest_snapshot(()) == ()
    unknown = replace(
        new, availability=replace(new.availability, data_as_of=None, public_available_at=NOW)
    )
    assert latest_snapshot((unknown,)) == ()
    assert unique_latest_by((old, new), ("con_code",)) == ((new,), 0)
    assert unique_latest_by((new, duplicate, missing), ("con_code",)) == ((), 2)
