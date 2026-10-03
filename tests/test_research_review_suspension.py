"""审查回归：停牌来源不完整时，不把证券放入市场分母。"""

from dataclasses import replace

import pytest
from test_research_market import market_inputs, row

from tushare_entitlement_hub.equity_research.market import _suspension, build_market
from tushare_entitlement_hub.product_foundation.contracts import Truth

IDENTITY = "cn:security:0:listed"


@pytest.mark.parametrize("complete", [(), ("suspend_d",)])
def test_absent_suspension_requires_certified_complete_scope(complete):
    """场景：有有效行情但无停牌行；预期：只有完整认证的空来源才允许证券进入分母。"""
    inputs = market_inputs(1, (row(),), complete=complete)
    result = build_market(inputs)[0]
    assert result.details["denominator_count"] == int(bool(complete))
    assert result.details["excluded_suspension_unknown_count"] == int(not complete)
    assert _suspension(inputs, IDENTITY, None)[0] is Truth.UNKNOWN


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize(
    "reason", ["source_not_registered", "row_capacity_exceeded", "quarantined"]
)
def test_source_rejection_cannot_be_cleared_by_complete_scope(local, reason):
    """场景：停牌缺表、容量拒绝或隔离缺口同时声明完整；预期：全局和实体缺口均保留未知。"""
    inputs = market_inputs(1, (row(),))
    inputs = replace(
        inputs,
        **(
            {"entity_source_reasons": {"suspend_d/" + IDENTITY: (reason,)}}
            if local
            else {"source_reasons": {"suspend_d": (reason,)}}
        ),
    )
    result = build_market(inputs)[0]
    assert result.details["denominator_count"] == 0
    assert result.details["excluded_suspension_unknown_count"] == 1
    assert result.metric("stress_score") is None


@pytest.mark.parametrize(
    "patch",
    [{"is_quarantined": True}, {"quality_tier": "unknown"}, {"quality_tier": "quarantine"}],
)
def test_rejected_suspension_row_neither_blocks_nor_clears_current_state(patch):
    """场景：同日停牌行被隔离或质量未知；预期：不宣称已停牌或未停牌，不把坏行作为证据。"""
    suspended = row("suspend_d", fields={"trade_date": "20260930", "suspend_type": "S"})
    suspended = replace(suspended, raw=suspended.raw | patch)
    result = build_market(market_inputs(1, (row(), suspended)))[0]
    assert result.details["denominator_count"] == 0
    assert result.details["excluded_suspension_unknown_count"] == 1
    assert result.details["excluded_suspended_count"] == 0
    assert not any(ref.source_api == "suspend_d" for ref in result.envelope.evidence_refs)
