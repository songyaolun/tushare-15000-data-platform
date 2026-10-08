"""审查回归：上下文分组的未知状态不得被其他已观测分组覆盖。"""

import pytest
from test_research_context_consumers import STOCK, consume, context_inputs, context_row

from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    count_consumption,
    summarize_consumption,
)


def test_partial_calendar_unknown_survives_product_and_run_summary():
    """场景：SSE日历完整而SZSE缺前交易日；预期：逐组证据保留且接口不计入已观测数。"""
    rows = tuple(
        context_row(
            "trade_cal",
            {"exchange": exchange, "cal_date": "20261003", "is_open": "0"}
            | ({"pretrade_date": "20260930"} if exchange == "SSE" else {}),
            record=exchange,
            day="2026-10-03",
            entity=None,
        )
        for exchange in ("SSE", "SZSE")
    )
    inputs = context_inputs(*rows)
    result = consume(inputs, "trade_cal")
    assert result["status"] == "unknown"
    assert {item["status"] for item in result["facts"]["observations"]} == {"observed", "unknown"}
    assert {ref.source_record_id for ref in result["evidence_refs"]} == {"SSE", "SZSE"}
    product = "market_context/v1"
    attached = attach_consumption(inputs, product, (make_result(inputs, product, STOCK),))[0]
    counts = {}
    count_consumption(counts, attached)
    summary = summarize_consumption(counts, [product])
    assert summary["observed_apis"] == 0
    assert summary["by_api"]["trade_cal"] == {"unknown": 1}


@pytest.mark.parametrize(
    ("assets", "incomplete", "expected"),
    [
        (("STK", "IDX"), False, "observed"),
        (("IDX", "FD"), False, "not_applicable"),
        (("STK", "IDX"), True, "unknown"),
        (("IDX", "FD"), True, "unknown"),
    ],
)
def test_factor_dictionary_mixed_applicability_requires_all_groups_known(
    assets, incomplete, expected
):
    """场景：股票与非股票因子字典混合；预期：完整组允许不适用，缺定义组始终使整体未知。"""
    rows = tuple(
        context_row(
            "factor_list",
            {"factor_name": asset, "asset_type": asset, "factor_type": "technical"}
            | ({} if incomplete and index else {"factor_desc": "合成定义"}),
            record=asset,
        )
        for index, asset in enumerate(assets)
    )
    result = consume(context_inputs(*rows), "factor_list")
    assert result["status"] == expected
    assert len(result["facts"]["observations"]) == 2


def test_history_limited_factor_does_not_hide_unknown_factor_group():
    """场景：一个因子有字典但历史不足，另一因子缺字典；预期：整体未知并保留历史受限组。"""
    dictionary = context_row(
        "factor_list",
        {"factor_name": "known", "asset_type": "STK", "factor_type": "技术", "factor_desc": "合成"},
    )
    rows = tuple(
        context_row(
            "factor_value",
            {"factor_name": name, "trade_date": "20260930", "factor_value": 0.5},
            record=name,
        )
        for name in ("known", "missing")
    )
    result = consume(context_inputs(dictionary, *rows), "factor_value")
    assert result["status"] == "unknown"
    assert {item["status"] for item in result["facts"]["observations"]} == {
        "unknown",
        "not_enabled_due_to_history",
    }
