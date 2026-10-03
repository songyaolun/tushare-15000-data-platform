"""审查回归：历史审计证据不能替代当前报告期的审计门禁。"""

import pytest
from test_research_fundamental import reports, research, statement

from tushare_entitlement_hub.equity_research.fundamental import build_fundamental
from tushare_entitlement_hub.product_foundation.contracts import Truth


@pytest.mark.parametrize("opinion", ["标准无保留意见", "否定意见"])
def test_previous_period_audit_is_history_and_current_audit_stays_unknown(opinion):
    """场景：当前三表仅有上一年审计意见；预期：旧意见独立留证，当前门禁未知且不发质量分。"""
    current = tuple(row for row in reports() if row.source_api != "fina_audit")
    historical = statement(
        "fina_audit",
        record="historical",
        report="2025-12-31",
        fields={"end_date": "20251231", "audit_result": opinion},
    )
    result = build_fundamental(research(current + (historical,), count=1))[0]
    check = next(gate for gate in result.envelope.quality_gates if gate.gate_id == "audit_opinion")
    assert check.result is Truth.UNKNOWN
    assert check.evidence_refs == ()
    assert result.details["audit_period"] is None
    assert result.metric("quality_score") is None
    history = result.details["historical_audits"]
    assert len(history) == 1
    assert history[0]["report_period"] == "2025-12-31"
    assert history[0]["audit_opinion"] == opinion
    assert history[0]["applies_to_current_report"] is False
    assert history[0]["evidence_refs"][0]["source_record_id"] == "historical"
    assert any(ref.source_record_id == "historical" for ref in result.envelope.evidence_refs)


def test_current_audit_is_selected_without_losing_historical_evidence():
    """场景：当前否定意见与旧标准意见同时可见；预期：当前门禁失败，历史意见只作背景。"""
    historical = statement("fina_audit", record="old", report="2025-12-31")
    result = build_fundamental(
        research(reports(fields={"audit_result": "否定意见"}) + (historical,), count=1)
    )[0]
    check = next(gate for gate in result.envelope.quality_gates if gate.gate_id == "audit_opinion")
    assert check.result is Truth.FALSE
    assert result.details["audit_period"] == "2026-06-30"
    assert result.details["historical_audits"][0]["source_record_id"] == "old"
    assert {ref.source_record_id for ref in check.evidence_refs} == {"fina_audit0"}
