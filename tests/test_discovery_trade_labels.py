"""独立标签与实际单持仓账本对齐，覆盖跳空、停牌及分红。"""

from dataclasses import replace

import numpy as np
import pytest
import test_discovery_execution

from tushare_entitlement_hub.regime_research.accounting import Distribution
from tushare_entitlement_hub.strategy_discovery.execution import simulate_strategy
from tushare_entitlement_hub.strategy_discovery.trade_labels import independent_trade_label


@pytest.mark.parametrize("scenario", ["profit", "stop", "time", "distribution", "delayed"])
def test_matches_actual_single_trade(scenario):
    """场景：固定止盈、跳空止损、到期、送转分红或跌停延期；预期与真实账本逐项相等。"""
    panel = test_discovery_execution.discovery_panel.__wrapped__()
    definition = test_discovery_execution.single_signal(panel)
    definition = replace(
        definition,
        policy=replace(definition.policy, take_profit=0.02, stop_loss=0.08, maximum_holding=3),
    )
    events = []
    if scenario == "profit":
        panel.values["raw_close"][1, 0] = 10.5
    if scenario == "stop":
        panel.values["raw_close"][1, 0] = 8.5
        panel.values["raw_open"][2, 0] = 8.0
        panel.values["raw_down_limit"][2, 0] = 7.0
    if scenario == "distribution":
        events = [Distribution(0, 1, 2, 3, 4, 0.1, 0.1)]
    if scenario == "delayed":
        panel.values["raw_close"][1, 0] = 8.5
        panel.values["raw_open"][2, 0] = panel.values["raw_down_limit"][2, 0]
    result = simulate_strategy(
        panel, np.zeros(8, dtype=np.int8), events, definition, start=panel.calendar[0]
    )
    label = independent_trade_label(panel, 0, 0, events, definition.policy)
    trade = result["trades"][0]
    assert label["status"] == "closed"
    for key in ["entry", "exit", "reason", "cost", "proceeds", "dividend", "return"]:
        assert label[key] == trade[key]


def test_unfilled_censored_and_invalid():
    """场景：尾部、缺报价、涨停、预算不足与错误证券；预期缺失标签不冒充零收益。"""
    panel = test_discovery_execution.discovery_panel.__wrapped__()
    policy = test_discovery_execution.single_signal(panel).policy
    assert independent_trade_label(panel, 7, 0, [], policy)["status"] == "censored"
    assert independent_trade_label(panel, 6, 0, [], policy)["return"] is None
    assert independent_trade_label(panel, 0, 0, [], policy, budget=10)["status"] == "unfilled"
    panel.values["raw_open"][1, 0] = np.nan
    assert independent_trade_label(panel, 0, 0, [], policy)["status"] == "unfilled"
    panel.values["raw_open"][1, 0] = panel.values["raw_up_limit"][1, 0]
    assert independent_trade_label(panel, 0, 0, [], policy)["status"] == "unfilled"
    with pytest.raises(ValueError):
        independent_trade_label(panel, -1, 0, [], policy)
