"""显式组合与非组合执行不得解析无关基金持仓或扩大父结果预算。"""

from dataclasses import replace

import pytest
from test_all_products_execution import prepared_inputs
from test_research_shared import IDENTITY, NOW, source

from tushare_entitlement_hub.cross_asset_research import execution, portfolio
from tushare_entitlement_hub.cross_asset_research.registry import PRODUCTS
from tushare_entitlement_hub.equity_research.common import make_result


@pytest.mark.parametrize("mode", ["explicit_holdings", "without_portfolio"])
def test_unrelated_fund_holdings_never_expand_selected_parent_scope(tmp_path, monkeypatch, mode):
    """场景：显式A与基金B并存，或只选P02；预期：不解析基金，父预算1仍允许显式A执行。"""
    unrelated = "cn:security:B:listed"
    original = source(api="fund_portfolio")
    disclosure = replace(
        original,
        raw=original.raw | {"symbol": "000002.SZ"},
        availability=replace(original.availability, report_period="2026-06-30"),
    )
    config, context, _, _ = prepared_inputs(monkeypatch, (disclosure,))
    resolutions = []
    captured = []

    def resolve_fund(current, symbol, period):
        """记录基金证券解析；例：000002.SZ会解析到无关的B，正确路径应完全不调用。"""
        resolutions.append((symbol, period))
        return unrelated

    def theme(current):
        """生成A和B两条独立研究；例：显式仅持有A时B仍输出但不能占用组合父预算。"""
        return tuple(
            make_result(current, PRODUCTS["P02"], entity) for entity in (IDENTITY, unrelated)
        )

    def diagnostic(current, *, parents, holdings):
        """捕获组合父集合；例：容量1时只接收A，并原样保留显式权重1。"""
        captured.append((parents, holdings))
        return (make_result(current, PRODUCTS["P12"], "cn:portfolio:synthetic:explicit"),)

    monkeypatch.setattr(portfolio, "holding_entity", resolve_fund)
    monkeypatch.setattr(
        execution,
        "resolve_handler",
        lambda name: diagnostic if name.endswith("build_portfolio") else theme,
    )
    selected = ("P02", "P12") if mode == "explicit_holdings" else ("P02",)
    holdings = ({"entity_id": IDENTITY, "weight": 1.0},) if mode == "explicit_holdings" else ()
    state = execution.ExecutionState()
    results = tuple(
        execution.execute(
            tmp_path / "reference", NOW, context.run_id, config, selected, holdings,
            execution.RunLimits(maximum_parent_results=1), state,
        )
    )
    assert resolutions == []
    assert len(results) == state.result_count == (3 if mode == "explicit_holdings" else 2)
    if mode == "explicit_holdings":
        assert len(captured) == 1
        assert [item.envelope.entity_id for item in captured[0][0]] == [IDENTITY]
        assert captured[0][1] == holdings
    else:
        assert captured == []
