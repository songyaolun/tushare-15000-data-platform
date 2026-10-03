"""审查回归：严重异动必须同时满足质量、可见性与生效区间。"""

from dataclasses import replace

import pytest
from test_research_flow import COMPLETE, daily
from test_research_governance import NOW, A, inputs, row

from tushare_entitlement_hub.equity_research.flow import _shock, build_flow
from tushare_entitlement_hub.product_foundation.contracts import Truth


@pytest.mark.parametrize(
    ("raw", "times"),
    [
        ({"quality_tier": "verified"}, {"effective_from": "2026-10-04T00:00:00Z"}),
        ({"quality_tier": "verified"}, {"effective_from": None}),
        ({"quality_tier": "verified", "is_quarantined": True}, {}),
        ({"quality_tier": "unknown"}, {}),
        ({"quality_tier": "quarantine"}, {}),
        ({}, {}),
        ({"quality_tier": "verified"}, {"public_available_at": "2026-10-04T00:00:00Z"}),
        ({"quality_tier": "verified"}, {"observed_at": "2026-10-04T00:00:00Z"}),
    ],
)
@pytest.mark.parametrize("include_active", [False, True])
def test_rejected_shock_preserves_unknown_even_with_certified_scope(raw, times, include_active):
    """场景：未来、隔离或质量未知异动混入完整来源；预期：独立与混合情况下硬门禁均未知。"""
    source = row("stk_high_shock", "rejected", raw=raw, effective_from=NOW)
    source = replace(source, availability=replace(source.availability, **times))
    rows = (source,)
    if include_active:
        rows += (
            row("stk_high_shock", "active", raw={"quality_tier": "verified"}, effective_from=NOW),
        )
    context = inputs(daily(), *rows, complete=COMPLETE)
    assert _shock(context, A) is Truth.UNKNOWN
    gate = next(
        check
        for check in build_flow(context)[0].envelope.quality_gates
        if check.gate_id == "flow_severe_shock"
    )
    assert gate.result is Truth.UNKNOWN


@pytest.mark.parametrize("quality", ["verified", "usable_with_annotation"])
def test_certified_shock_effective_boundaries_are_half_open(quality):
    """场景：合格异动在截止时刻生效或结束；预期：生效即阻断，结束且来源完整才放行。"""
    active = row("stk_high_shock", raw={"quality_tier": quality}, effective_from=NOW)
    ended = replace(
        active,
        availability=replace(
            active.availability, effective_from="2026-10-01T00:00:00Z", effective_to=NOW
        ),
    )
    assert _shock(inputs(active, complete=COMPLETE), A) is Truth.FALSE
    assert _shock(inputs(ended, complete=COMPLETE), A) is Truth.TRUE
    assert _shock(inputs(ended), A) is Truth.UNKNOWN
    assert (
        _shock(inputs(active, reasons={"stk_high_shock": ("source_rejected",)}), A) is Truth.UNKNOWN
    )
