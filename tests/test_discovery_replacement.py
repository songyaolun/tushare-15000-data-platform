"""主动替换保持收盘决策、逐对卖出依赖和原账户的实际成交限制。"""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.regime_research.accounting import Holding
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.replacement import (
    ReplacementRule,
    fading_position,
    make_fading_exit,
    make_replacement_provider,
)


def rotation_fixture():
    """返回8日面板，首日两只入场、第二日两只新候选；例第0股和第2股构成换仓。"""
    panel, definition = allocation_fixture()
    panel.values["return5"] = np.full((8, 12), -0.02)
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, :2] = True
    mask[1, 2:4] = True
    return panel, replace(definition, entry_mask=mask)


def run_rotation(panel, definition, *, rule=None, policy=None, provider=None):
    """在合成双持仓账户运行替换；例默认最多替换两只，保留真实下一开盘成交。"""
    return simulate_candidate_allocation(
        panel,
        np.ones(8, dtype=np.int8),
        [],
        definition,
        policy or replace(DEFAULT_POLICY, non_stage_allocation=0.1),
        mode="exit_reserve",
        count=2,
        start=panel.calendar[0],
        replacement_provider=provider
        or make_replacement_provider(rule or ReplacementRule(1, 0.01, "momentum", 0, 2)),
    )


@pytest.mark.parametrize("blocked", ["none", "first", "second", "both", "gap", "missing"])
def test_replacement_requires_its_own_sale(blocked):
    """场景：两组配对的旧股独立阻塞；预期未卖掉的那组绝不能借另一组现金买入。"""
    panel, definition = rotation_fixture()
    if blocked in {"first", "both"}:
        panel.values["raw_open"][2:, 0] = 9
    if blocked in {"second", "both"}:
        panel.values["raw_open"][2:, 1] = 9
    if blocked == "missing":
        panel.values["raw_open"][2:, 0] = np.nan
    if blocked == "gap":
        definition = replace(definition, maximum_entry_gap=0.02)
        panel.values["raw_open"][2, 2] = 10.5
    result = run_rotation(panel, definition)
    assert result["replacement_decisions"] == [
        {"day": 1, "sell": 0, "buy": 2},
        {"day": 1, "sell": 1, "buy": 3},
    ]
    expected = 0 if blocked == "both" else 2 if blocked == "none" else 1
    assert result["entry_days"].count(2) == expected
    assert len(result["entry_days"]) == 2 + expected
    weights = result["daily"][2]["weights"]
    assert (2 in weights) == (blocked not in {"first", "both", "missing", "gap"})
    assert (3 in weights) == (blocked not in {"second", "both"})
    assert all(row["cash"] >= 0 for row in result["daily"])
    assert abs(result["reconciliation_error"]) < 1e-7


@pytest.mark.parametrize(
    "scenario", ["spare_slots", "over_capacity", "normal_exit", "no_candidate"]
)
def test_replacement_only_when_needed(scenario):
    """场景：有正常买入名额、弱市超额或自然卖单；预期不额外主动换仓。"""
    panel, definition = rotation_fixture()
    policy = replace(DEFAULT_POLICY, non_stage_allocation=0.1)
    if scenario == "spare_slots":
        policy = replace(policy, non_stage_allocation=0.2)
    if scenario == "over_capacity":
        # 用阶段切换将两只已买持仓收紧到一名额，不能借换仓继续超额补入。
        policy = replace(
            policy,
            stage_allocation=True,
            force_rebalance=False,
            allocations=(0.05, 0.1, 0.1, 0.1, 0.1),
        )
    if scenario == "normal_exit":
        panel.values["raw_close"][1, 0] = 10.5
    if scenario == "no_candidate":
        definition.entry_mask[1] = False
    states = np.ones(8, dtype=np.int8)
    if scenario == "over_capacity":
        states[1:] = 0
    result = simulate_candidate_allocation(
        panel,
        states,
        [],
        definition,
        policy,
        mode="exit_reserve",
        count=2,
        start=panel.calendar[0],
        replacement_provider=make_replacement_provider(ReplacementRule(1)),
    )
    assert result["replacement_decisions"] == []


@pytest.mark.parametrize("pairs", [[(9, 2)], [(0, 0)], [(0, 2), (0, 3)], [(0, 2), (1, 2)]])
def test_invalid_pair_rejected(pairs):
    """场景：卖不存在持仓、买已有持仓或重复配对；预期拒绝非法回调输出。"""
    panel, definition = rotation_fixture()

    def invalid_provider(*arguments):
        """返回本测试指定的非法配对；例[(9,2)]不能通过账户验证。"""
        return pairs

    with pytest.raises(ValueError, match="invalid_replacement_pairs"):
        run_rotation(panel, definition, provider=invalid_provider)


def test_fading_uses_only_known_prices_and_confirmation():
    """场景：同一持仓变更当前特征、未来价格和锁定状态；预期只依据已知且完整输入。"""
    panel, _ = rotation_fixture()
    holding = Holding(0, 0, 1, 100, 1000, 10, 10, 10, "example", 0, 1)
    rule = ReplacementRule(1)
    assert fading_position(panel, holding, 1, rule)
    assert not fading_position(panel, holding, 1, replace(rule, minimum_sessions=2))
    assert not fading_position(panel, replace(holding, locked=1), 1, rule)
    assert not fading_position(panel, replace(holding, sell_reason="stop"), 1, rule)
    assert not fading_position(panel, replace(holding, entry_adjusted=0), 1, rule)
    panel.values["raw_close"][2:, 0] = 1000
    assert fading_position(panel, holding, 1, rule)
    panel.values["raw_close"][1, 0] = 10.2
    assert not fading_position(panel, holding, 1, rule)
    panel.values["raw_close"][1, 0] = np.nan
    assert not fading_position(panel, holding, 1, rule)
    panel.values["raw_close"][1, 0] = 10
    panel.values["return5"][1, 0] = np.nan
    assert not fading_position(panel, holding, 1, rule)
    assert fading_position(panel, holding, 1, replace(rule, confirmation="age"))
    panel.values["return5"][1, 0] = -0.01
    panel.values["close"][1, 0] = 10
    panel.values["ma20"][1, 0] = 11
    assert fading_position(panel, holding, 1, replace(rule, confirmation="trend"))
    panel.values["sector_breadth"][1, 0] = 0.3
    assert fading_position(panel, holding, 1, replace(rule, confirmation="sector"))
    panel.values["sector_breadth"][1, 0] = 0.4
    assert not fading_position(panel, holding, 1, replace(rule, confirmation="sector"))
    assert make_fading_exit(rule)(panel, holding, 1, DEFAULT_POLICY) == "fading"
    assert (
        make_fading_exit(replace(rule, minimum_sessions=2))(panel, holding, 1, DEFAULT_POLICY) == ""
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_sessions": 0},
        {"minimum_sessions": 31},
        {"maximum_replacements": 0},
        {"maximum_return": float("nan")},
        {"maximum_return": 0.045},
        {"minimum_candidate_score": float("inf")},
        {"confirmation": "unknown"},
    ],
)
def test_rule_rejects_invalid_values(changes):
    """场景：非法研究阈值；预期在进入回测前失败，不能产生虚假正常结果。"""
    with pytest.raises(ValueError, match="invalid_replacement_rule"):
        ReplacementRule(**changes)


def test_replacement_candidate_threshold_and_pending():
    """场景：候选未达阈值或已经挂单；预期不产生配对，持仓对象也不被回调修改。"""
    panel, _ = rotation_fixture()
    holding = Holding(0, 0, 1, 100, 1000, 10, 10, 10, "example", 0, 99)
    provider = make_replacement_provider(ReplacementRule(1, minimum_candidate_score=0.05))
    assert (
        provider(panel, 1, DEFAULT_POLICY, (holding,), ((2, "example", 0.04),), frozenset()) == []
    )
    assert (
        provider(panel, 1, DEFAULT_POLICY, (holding,), ((2, "example", 0.06),), frozenset({2}))
        == []
    )
    assert provider(panel, 1, DEFAULT_POLICY, (holding,), ((2, "example", 0.06),), frozenset()) == [
        (0, 2)
    ]
    assert holding.sell_reason == ""
