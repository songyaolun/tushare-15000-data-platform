"""实际披露日和预期披露日的公开时间边界，严格门禁与已知日程分别验证。"""

from dataclasses import replace

import pytest
from test_research_adapter import configured, reference_database, source
from test_research_governance import NOW

from tushare_entitlement_hub.equity_research.adapter import load_inputs
from tushare_entitlement_hub.product_foundation.bindings import TimeMapping


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("actual_date_unmapped", "public_field_mapping_missing"),
        ("actual_date_future", "row_unavailable"),
        ("expected_date_future", None),
    ],
)
def test_disclosure_publication_boundary_preserves_announced_future_schedule(
    tmp_path, scenario, reason
):
    """场景：遗漏实际披露日、未来实际披露日、已公告未来日程；预期：前两者拒绝，后者保留。"""
    api = "disclosure_date"
    record = source(
        api,
        record=scenario,
        end_date="2026-09-30",
        ann_date="2026-10-02",
        pre_date="2026-10-04",
        actual_date=(
            None
            if scenario == "expected_date_future"
            else "2026-10-04"
            if scenario == "actual_date_future"
            else "2026-10-02"
        ),
        modify_date=None,
    )
    mapping = TimeMapping(
        source_api=api,
        data_field="end_date",
        public_fields=("ann_date",),
        optional_public_fields=() if scenario == "actual_date_unmapped" else ("actual_date",),
        market_timezone="Asia/Shanghai",
    )
    base = configured(api)
    config = replace(base, foundation=replace(base.foundation, time_mappings=(mapping,)))
    reference = reference_database(tmp_path, {api: [record]})
    result, counts = load_inputs(reference, NOW, "disclosure-public-fields", config)
    assert counts["scanned_rows"] == 1
    assert counts["retained_rows"] == int(reason is None)
    assert counts["excluded_rows"] == int(reason is not None)
    assert counts["discarded_rows"] == 0
    if reason is not None:
        assert result.rows(api) == ()
        assert reason in result.source_reasons[api]
    else:
        assert len(result.rows(api)) == 1
        kept = result.rows(api)[0]
        assert kept.text("pre_date") == "2026-10-04"
        assert kept.text("ann_date") == "2026-10-02"
        assert kept.raw["actual_date"] is None
        evidence = kept.evidence("pre_date", "ann_date")
        assert {ref.source_record_id for ref in evidence} == {scenario}
        assert {ref.source_api for ref in evidence} == {api}
        assert {ref.fields for ref in evidence} == {("pre_date",), ("ann_date",)}
