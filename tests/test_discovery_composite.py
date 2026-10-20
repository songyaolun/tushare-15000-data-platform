"""共享现金组合与原单策略执行的等价、归属及退出隔离测试。"""

from dataclasses import replace

import numpy as np
import pytest
import test_discovery_execution
from test_discovery_execution import single_signal

from tushare_entitlement_hub.strategy_discovery.composite import simulate_composite
from tushare_entitlement_hub.strategy_discovery.execution import simulate_strategy


@pytest.fixture
def discovery_panel():
    """场景：复用固定8日12证券面板；预期每个测试获得独立可修改副本。"""
    return test_discovery_execution.discovery_panel.__wrapped__()


@pytest.mark.parametrize("intraday", [False, True])
def test_one_strategy_reproduces_cash_ledger(discovery_panel, intraday):
    """场景：只有一个支路；预期所有日净值及交易现金流与原模拟器完全相同。"""
    panel = discovery_panel
    panel.values["raw_open"][3:] = 10.4
    panel.values["raw_close"][2:] = 10.4
    panel.values["raw_high"][2:] = 10.5
    definition = single_signal(panel, intraday_exits=intraday, maximum_pending_sessions=1)
    definition = replace(definition, policy=replace(definition.policy, take_profit=0.02))
    states = np.zeros(len(panel.calendar), dtype=np.int8)
    expected = simulate_strategy(panel, states, [], definition, start=panel.calendar[0])
    actual = simulate_composite(
        panel, states, [], (definition,), name="single", start=panel.calendar[0]
    )
    np.testing.assert_array_equal(
        [x["nav"] for x in actual["daily"]], [x["nav"] for x in expected["daily"]]
    )
    assert actual["trades"][0]["proceeds"] == expected["trades"][0]["proceeds"]
    assert actual["trades"][0]["exit"] == expected["trades"][0]["exit"]


def test_priority_freezes_exit_rule_and_deduplicates(discovery_panel):
    """场景：同股同时有2%与8%止盈信号；预期优先支路2%退出且只成交一次。"""
    panel = discovery_panel
    panel.values["raw_close"][2:] = 10.3
    panel.values["raw_open"][3:] = 10.3
    first = single_signal(panel, name="fast")
    first = replace(first, policy=replace(first.policy, take_profit=0.02))
    second = replace(first, name="slow", policy=replace(first.policy, take_profit=0.08))
    result = simulate_composite(
        panel, np.zeros(8, dtype=np.int8), [], (first, second), name="both", start=panel.calendar[0]
    )
    assert len(result["trades"]) == 1
    assert result["trades"][0]["family"] == "fast"
    assert result["trades"][0]["exit"] == 3
    assert len(result["entry_days"]) == 1


def test_pending_expiry_and_structure_exit_keep_original_semantics(discovery_panel):
    """场景：首日涨停买不到，或结构卖出；预期短单取消，长单可延后且结构有效。"""
    panel = discovery_panel
    panel.values["raw_open"][1, 0] = 11
    definition = single_signal(panel, maximum_pending_sessions=1)
    states = np.zeros(8, dtype=np.int8)
    result = simulate_composite(
        panel, states, [], (definition,), name="expire", start=panel.calendar[0]
    )
    assert result["entry_days"] == []
    exit_mask = np.zeros(definition.entry_mask.shape, dtype=bool)
    exit_mask[3, 0] = True
    definition = replace(definition, maximum_pending_sessions=3, exit_mask=exit_mask)
    result = simulate_composite(
        panel, states, [], (definition,), name="structure", start=panel.calendar[0]
    )
    assert result["entry_days"] == [2]
    assert result["trades"][0]["exit"] == 4
    assert result["trades"][0]["reason"] == "signal"


def test_entry_gap_rejected_and_invalid_definitions_fail(discovery_panel):
    """场景：高开超过追价、重复支路或无支路；预期分别取消买入与拒绝契约。"""
    panel = discovery_panel
    panel.values["raw_open"][1, 0] = 10.5
    definition = single_signal(panel, maximum_entry_gap=0.02)
    states = np.zeros(8, dtype=np.int8)
    result = simulate_composite(
        panel, states, [], (definition,), name="gap", start=panel.calendar[0]
    )
    assert not result["entry_days"]
    for definitions in [(), (definition, definition)]:
        with pytest.raises(ValueError):
            simulate_composite(panel, states, [], definitions, name="invalid")
