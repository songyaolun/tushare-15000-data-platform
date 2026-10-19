"""三个候选机制独立对照，保持实际现金与先卖后买时点。"""

from dataclasses import replace

import numpy as np
import pytest
import test_discovery_execution

from tushare_entitlement_hub.regime_research.accounting import Distribution
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.composite import simulate_composite
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.signal_redesign import top_candidates


def allocation_fixture():
    """输入无，输出8日12股全候选定义，分数按证券顺序递减；例第0股最优。"""
    panel = test_discovery_execution.discovery_panel.__wrapped__()
    definition = test_discovery_execution.single_signal(panel)
    definition = replace(
        definition,
        entry_mask=np.ones((8, 12), dtype=bool),
        scores=np.tile(np.arange(12, 0, -1, dtype=float), (8, 1)),
        maximum_pending_sessions=1,
        policy=replace(DEFAULT_POLICY, take_profit=0.02, stop_loss=0.08, maximum_holding=20),
    )
    return panel, definition


def run_allocation(panel, definition, mode, *, policy=DEFAULT_POLICY, count=2, events=None):
    """输入合成面板及模式，输出从首日开始的账户；默认每日取两只候选。"""
    return simulate_candidate_allocation(
        panel,
        np.zeros(8, dtype=np.int8),
        events or [],
        definition,
        policy,
        mode=mode,
        count=count,
        start=panel.calendar[0],
    )


def test_baseline_matches_composite_exactly():
    """场景：机制关闭；预期逐日账户和所有交易与旧composite精确相同。"""
    panel, definition = allocation_fixture()
    panel.values["raw_close"][2, 0] = 11
    actual = run_allocation(panel, definition, "baseline")
    selected = replace(
        definition, entry_mask=top_candidates(definition.entry_mask, definition.scores, 2)
    )
    expected = simulate_composite(
        panel,
        np.zeros(8, dtype=np.int8),
        [],
        (selected,),
        name=definition.name,
        start=panel.calendar[0],
    )
    assert actual["daily"] == expected["daily"]
    assert actual["trades"] == expected["trades"]
    assert actual["entry_days"] == expected["entry_days"]


def test_common_first_refills_only_signal_day_qualification():
    """场景：最高分不满足信号日成交额资格；预期仅common_first从第三名补位。"""
    panel, definition = allocation_fixture()
    panel.values["raw_amount"][:, 0] = 19999
    base = run_allocation(panel, definition, "baseline")
    common = run_allocation(panel, definition, "common_first")
    assert base["candidate_selections"][0]["qualified"] == [1]
    assert common["candidate_selections"][0]["qualified"] == [1, 2]
    assert common["candidate_selections"][0]["common_rejected"] == 0
    # 次日开盘涨停不会倒改信号日候选，仍选1、2，不能从3补位。
    panel.values["raw_open"][1, 1] = 11
    changed = run_allocation(panel, definition, "common_first")
    assert changed["candidate_selections"][0] == common["candidate_selections"][0]
    assert changed["entry_days"].count(1) == 1


def test_holdings_first_excludes_positions_but_keeps_common_order():
    """场景：头两名已经买入，第三名共同资格失败；预期持仓前置后选2/3再剔除2。"""
    panel, definition = allocation_fixture()
    panel.values["raw_amount"][:, 2] = 19999
    actual = run_allocation(panel, definition, "holdings_first")
    assert actual["candidate_selections"][1]["selected"] == [2, 3]
    assert actual["candidate_selections"][1]["qualified"] == [3]
    assert actual["candidate_selections"][1]["common_rejected"] == 1
    assert run_allocation(panel, definition, "baseline")["candidate_selections"][1]["selected"] == [
        0,
        1,
    ]


@pytest.mark.parametrize("blocked", ["none", "limit", "missing", "locked", "all"])
def test_reserve_activates_only_after_actual_sales(blocked):
    """场景：两只满仓待卖，次开盘不同阻塞；预期实际卖几只最多启用几笔预备单。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, :2] = True
    mask[1, 2:4] = True
    definition = replace(definition, entry_mask=mask)
    policy = replace(DEFAULT_POLICY, non_stage_allocation=0.1)
    panel.values["raw_close"][1, :2] = 10.5
    events = []
    expected = 2
    if blocked in ("limit", "all"):
        panel.values["raw_open"][2, 0] = 9
        expected = 1
    if blocked == "all":
        panel.values["raw_open"][2, 1] = 9
        expected = 0
    if blocked == "missing":
        panel.values["raw_open"][2, 0] = np.nan
        expected = 1
    if blocked == "locked":
        events = [Distribution(0, 1, 2, 3, 5, 0, 0.1)]
        expected = 1
    actual = run_allocation(panel, definition, "exit_reserve", policy=policy, events=events)
    base = run_allocation(panel, definition, "baseline", policy=policy, events=events)
    assert actual["entry_days"].count(2) == expected
    assert base["entry_days"].count(2) == 0
    assert actual["reserve_orders"]["created"] == 2
    assert actual["reserve_orders"]["filled"] == expected
    assert len(actual["entry_days"]) == 2 + expected  # 阻塞单次日收盘失效，不在以后追补。
    assert all(row["cash"] >= 0 for row in actual["daily"])


def test_reserve_keeps_sector_cash_and_gap_constraints():
    """场景：卖出完成但替代股行业容量不足或追价超限；预期不能借预备权限绕过限制。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, :2] = True
    mask[1, 2:4] = True
    definition = replace(definition, entry_mask=mask, maximum_entry_gap=0.02)
    policy = replace(DEFAULT_POLICY, non_stage_allocation=0.1, sector_weight=0.05)
    panel.values["raw_close"][1, :2] = 10.5
    panel.values["sector_index"][:, 2:4] = 2
    actual = run_allocation(panel, definition, "exit_reserve", policy=policy)
    assert actual["entry_days"].count(2) == 1
    panel.values["raw_open"][2, 2:4] = 10.5
    rejected = run_allocation(panel, definition, "exit_reserve", policy=policy)
    assert rejected["entry_days"].count(2) == 0


def test_invalid_definition_and_structural_invalidation():
    """场景：非法模式或结构失效；预期非法输入拒绝，结构失效继续在TopK后剔除。"""
    panel, definition = allocation_fixture()
    with pytest.raises(ValueError):
        run_allocation(panel, definition, "bad")
    with pytest.raises(ValueError):
        run_allocation(panel, definition, "baseline", count=0)
    with pytest.raises(ValueError):
        run_allocation(panel, replace(definition, maximum_pending_sessions=3), "baseline")
    invalidation = np.zeros((8, 12), dtype=bool)
    invalidation[:, 0] = True
    actual = run_allocation(
        panel, replace(definition, entry_invalidation_mask=invalidation), "common_first"
    )
    assert actual["candidate_selections"][0]["qualified"] == [1]


@pytest.mark.parametrize("blocked", [True, False])
def test_reserve_requires_full_slot_deficit_when_regime_shrinks(blocked):
    """场景：两持仓转弱只容一名额；预期两只均卖才激活预备单，仅卖一只仍阻塞。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, :2] = True
    mask[1, 2] = True
    definition = replace(definition, entry_mask=mask)
    policy = replace(
        DEFAULT_POLICY,
        stage_allocation=True,
        allocations=(0.05, 0.1, 0.1, 0.1, 0.1),
        force_rebalance=False,
    )
    states = np.zeros(8, dtype=np.int8)
    states[0] = 1
    panel.values["raw_close"][1, :2] = 10.5
    if blocked:
        panel.values["raw_open"][2, 0] = 9
    # 大幅跌价让金额仓位允许少量买入；名额规则仍应要求两笔旧持仓均卖掉。
    panel.values["raw_down_limit"][2, 0] = 9
    result = simulate_candidate_allocation(
        panel, states, [], definition, policy, mode="exit_reserve", count=2, start=panel.calendar[0]
    )
    assert result["entry_days"].count(2) == int(not blocked)
    assert result["reserve_orders"]["blocked_open_attempts"] == int(blocked)
