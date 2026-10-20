"""初期快速下跌的边界与共享账本接入测试。"""

from dataclasses import replace

import numpy as np
import pytest
import test_discovery_execution

from tushare_entitlement_hub.strategy_discovery.composite import simulate_composite
from tushare_entitlement_hub.strategy_discovery.early_failure import EarlyFailure


@pytest.mark.parametrize("sessions,loss", [(0, 0.03), (True, 0.03), (3, 0), (3, np.nan), (3, True)])
def test_invalid_policy(sessions, loss):
    """场景：非法观察期/阈值；预期明确拒绝。"""
    with pytest.raises(ValueError):
        EarlyFailure(sessions, loss)


def test_boundary_missing_and_expired():
    """场景：精确触线、过期、缺值；预期只有窗口内有效价格触发。"""
    policy = EarlyFailure(3, 0.03)
    assert policy.triggered(1, 10, 9.7)
    assert policy.triggered(3, 10, 9.6)
    assert not policy.triggered(4, 10, 9.6)
    assert not policy.triggered(2, 10, 9.8)
    assert not policy.triggered(2, 10, np.nan)


def test_early_failure_sells_next_open_and_preserves_fixed_stop_priority():
    """场景：入场日跌4%但未触8%止损；预期早期退出次开盘成交，未知策略拒绝。"""
    panel = test_discovery_execution.discovery_panel.__wrapped__()
    definition = test_discovery_execution.single_signal(panel)
    definition = replace(definition, policy=replace(definition.policy, stop_loss=0.08))
    panel.values["raw_close"][1, 0] = 9.6
    panel.values["raw_open"][2, 0] = 9.5
    states = np.zeros(8, dtype=np.int8)
    result = simulate_composite(
        panel,
        states,
        [],
        (definition,),
        name="failure",
        start=panel.calendar[0],
        failure_policies={definition.name: EarlyFailure()},
    )
    assert result["trades"][0]["reason"] == "early_failure"
    assert result["trades"][0]["exit"] == 2
    with pytest.raises(ValueError):
        simulate_composite(
            panel,
            states,
            [],
            (definition,),
            name="bad",
            failure_policies={"unknown": EarlyFailure()},
        )
