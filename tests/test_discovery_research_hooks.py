"""联合研究接口必须保留收盘时点、硬退出优先级和真实资金约束。"""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY


def hook_account(panel, definition, **hooks):
    """输入合成面板和可选接口，返回8日预备账户；例预算接口.03限制信号预算3%。"""
    return simulate_candidate_allocation(
        panel,
        np.zeros(8, dtype=np.int8),
        [],
        definition,
        DEFAULT_POLICY,
        mode="exit_reserve",
        count=3,
        start=panel.calendar[0],
        **hooks,
    )


@pytest.mark.parametrize("invalid", [0, -0.01, 0.051, np.nan, np.inf, True, "0.03"])
def test_weight_provider_rejects_invalid_or_excessive_budget(invalid):
    """场景：预算为非正、超过5%、非有限、布尔或字符串；预期挂单时失败。"""
    panel, definition = allocation_fixture()

    def invalid_weight(*arguments):
        """输入任意候选上下文，输出参数化非法预算；例如0触发拒绝。"""
        return invalid

    with pytest.raises(ValueError, match="invalid_candidate_position_weight"):
        hook_account(panel, definition, position_weight_provider=invalid_weight)


def test_weight_provider_freezes_signal_budget_and_original_rank():
    """场景：第二天首名已持仓；预期新候选仍rank1/2，3%与2%预算不填满剩余现金。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, 0] = True
    mask[1, :3] = True
    definition = replace(definition, entry_mask=mask)
    observed = []

    def ranked_weight(current, day, security, score, rank, candidate_count, policy):
        """记录实际挂单上下文并给首名3%、其余2%；例持仓首名不再调用。"""
        observed.append((day, security, rank, candidate_count))
        return 0.03 if rank == 0 else 0.02

    result = hook_account(panel, definition, position_weight_provider=ranked_weight)
    assert observed == [(0, 0, 0, 1), (1, 1, 1, 3), (1, 2, 2, 3)]
    weights = result["daily"][2]["weights"]
    assert 0.028 < weights[0] < 0.03
    assert all(0.018 < weights[column] < 0.02 for column in (1, 2))
    assert result["daily"][2]["cash"] > 930000


def test_maximum_weight_hook_reproduces_default_account():
    """场景：接口总返回原5%上限；预期逐日、逐笔与无接口账户完全一致。"""
    panel, definition = allocation_fixture()

    def original_weight(current, day, security, score, rank, candidate_count, policy):
        """输入候选和政策，输出原单票预算；例默认政策返回.05。"""
        return policy.position_weight

    expected = hook_account(panel, definition)
    actual = hook_account(panel, definition, position_weight_provider=original_weight)
    assert actual == expected


@pytest.mark.parametrize("native", ["stop", "time", "signal", "custom"])
def test_position_exit_respects_native_priority_and_next_open(native):
    """场景：硬止损、期限、既有结构或新回调退出；预期前三者优先且不能收盘当日卖。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, 0] = True
    definition = replace(definition, entry_mask=mask)
    if native == "stop":
        panel.values["raw_close"][1, 0] = 9
    elif native == "time":
        definition = replace(definition, policy=replace(definition.policy, maximum_holding=1))
    elif native == "signal":
        exits = np.zeros((8, 12), dtype=bool)
        exits[1, 0] = True
        definition = replace(definition, exit_mask=exits)
    observed = []

    def close_state(current, holding, day, policy):
        """输入已有持仓收盘上下文，输出退出原因；例第1日触发、第2日才能卖。"""
        observed.append(day)
        return "custom"

    result = hook_account(panel, definition, position_exit_provider=close_state)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit"] == 2
    assert result["trades"][0]["reason"] == native
    assert observed == ([1] if native == "custom" else [])
