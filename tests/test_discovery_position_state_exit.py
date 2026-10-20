"""持仓状态退出的纯函数、因果窗口和复权适配测试。"""

from dataclasses import replace

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.accounting import Holding
from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy, exit_reason
from tushare_entitlement_hub.strategy_discovery.position_state_exit import (
    PositionCloseState,
    PositionStateExitRule,
    known_sector_state,
    make_position_state_exit,
    position_state_exit_reason,
)


def state_fixture(**changes):
    """生成已赚5%且共同强势的状态；例stock_return=-.01可覆盖五日个股方向。"""
    return replace(PositionCloseState(8, 1.05, 1.05, True, 0.04, 0.6, 0.6, 0.03), **changes)


def callback_fixture():
    """生成12日单股面板及持仓；例第7日复权收盘21、成本20，盈利5%。"""
    values = {
        name: np.full((12, 1), value)
        for name, value in {
            "raw_close": 10.5,
            "raw_adj_factor": 2.0,
            "close": 21.0,
            "ma20": 20.0,
            "return5": 0.04,
            "sector_breadth": 0.6,
            "sector_mean_return": 0.01,
            "sector_index": 1.0,
        }.items()
    }
    panel = Panel([str(day) for day in range(12)], ["000001.SZ"], values, {}, {})
    holding = Holding(0, 4, 5, 100.0, 1000.0, 10.0, 20.0, 21.0, "test", 0, 1.0)
    return panel, holding


@pytest.mark.parametrize(
    "changes",
    [
        {"extended_profit": float("nan")},
        {"extended_profit": True},
        {"extended_profit": "bad"},
        {"extended_profit": 0.04},
        {"extended_profit": 1},
        {"minimum_breadth": -1},
        {"minimum_breadth": 2},
        {"failure_after": -1},
        {"failure_after": True},
        {"failure_return": -0.08},
        {"failure_return": 0.01},
        {"breadth_drop": 0},
        {"breadth_drop": 2},
        {"extend_profit": 1},
        {"require_sector_failure": 1},
    ],
)
def test_invalid_rules(changes):
    """场景：非有限数、布尔混数或越界；预期在回测前拒绝非法规则。"""
    with pytest.raises(ValueError, match="invalid_position_state_exit_rule"):
        PositionStateExitRule(**changes)


@pytest.mark.parametrize("multiple", [float("nan"), 0, -1])
def test_unknown_price_does_not_exit(multiple):
    """场景：当前复权价格缺失或无效；预期不虚构额外价格退出。"""
    assert (
        position_state_exit_reason(state_fixture(return_multiple=multiple), PositionStateExitRule())
        == ""
    )


def test_fixed_profit_and_extension_cap():
    """场景：基础止盈、强势延长及上限；预期只在启用延长且未到上限时继续持有。"""
    state = state_fixture()
    assert position_state_exit_reason(state, PositionStateExitRule()) == "state_profit"
    rule = PositionStateExitRule(extend_profit=True)
    assert position_state_exit_reason(state, rule) == ""
    assert position_state_exit_reason(state_fixture(return_multiple=1.08), rule) == "state_profit"
    assert (
        position_state_exit_reason(state_fixture(return_multiple=1.02, peak_multiple=1.02), rule)
        == ""
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"above_average": False},
        {"above_average": None},
        {"stock_return": 0},
        {"sector_breadth": 0.49},
        {"sector_return": 0},
        {"signal_sector_breadth": 0.8},
        {"sector_breadth": float("nan")},
    ],
)
def test_extension_needs_joint_known_strength(changes):
    """场景：延长中的共同强势任一条件消失；预期退出，即使收益已回落到4%以下。"""
    state = state_fixture(return_multiple=1.03, peak_multiple=1.05, **changes)
    assert (
        position_state_exit_reason(state, PositionStateExitRule(extend_profit=True))
        == "state_profit"
    )


def test_failure_requires_time_loss_and_joint_deterioration():
    """场景：持仓失败早退；预期必须满足持有时钟、亏损、个股和行业同时走弱。"""
    rule = PositionStateExitRule(failure_after=5)
    failed = state_fixture(
        return_multiple=0.98,
        peak_multiple=1.0,
        above_average=False,
        stock_return=-0.03,
        sector_breadth=0.4,
        sector_return=-0.02,
    )
    assert position_state_exit_reason(failed, rule) == "state_failed"
    for changes in (
        {"held_sessions": 4},
        {"return_multiple": 0.99},
        {"above_average": True},
        {"stock_return": 0.01},
        {"sector_breadth": 0.6},
        {"sector_return": 0.01},
        {"sector_breadth": float("nan")},
        {"above_average": None},
    ):
        assert position_state_exit_reason(replace(failed, **changes), rule) == ""
    assert (
        position_state_exit_reason(
            replace(failed, sector_breadth=0.6), replace(rule, require_sector_failure=False)
        )
        == "state_failed"
    )
    assert position_state_exit_reason(failed, PositionStateExitRule()) == ""


def test_sector_window_is_causal_and_rejects_changes_or_missing():
    """场景：五日窗口、未来篡改、缺失和换行业；预期只用完整同业的过去五日。"""
    panel, _ = callback_fixture()
    assert np.isnan(known_sector_state(panel, 3, 0)[0])
    assert known_sector_state(panel, 7, 0) == pytest.approx((0.6, 0.05))
    panel.values["sector_breadth"][8:] = 0.01
    assert known_sector_state(panel, 7, 0) == pytest.approx((0.6, 0.05))
    for key, value in [
        ("sector_index", 2),
        ("sector_index", -1),
        ("sector_index", np.nan),
        ("sector_breadth", np.nan),
        ("sector_mean_return", np.nan),
    ]:
        original = panel.values[key][6, 0]
        panel.values[key][6, 0] = value
        assert np.isnan(known_sector_state(panel, 7, 0)[0])
        panel.values[key][6, 0] = original


def test_callback_adjustment_future_and_industry_identity():
    """场景：原始价×因子、未来改写及信号至今换行业；预期收益同尺度且跨行业不延长。"""
    panel, holding = callback_fixture()
    callback = make_position_state_exit(PositionStateExitRule(extend_profit=True))
    policy = PortfolioPolicy(take_profit_enabled=False)
    assert callback(panel, holding, 7, policy) == ""
    for values in panel.values.values():
        values[8:] = np.nan
    assert callback(panel, holding, 7, policy) == ""
    panel.values["sector_index"][:5] = 2
    assert callback(panel, holding, 7, policy) == "state_profit"
    panel.values["sector_index"][:] = 1
    panel.values["ma20"][7] = np.nan
    assert callback(panel, holding, 7, policy) == "state_profit"
    panel.values["raw_close"][7] = np.nan
    assert callback(panel, holding, 7, policy) == ""


def test_hard_stop_and_expiry_remain_authoritative():
    """场景：状态允许延长但原账户触发8%止损或30日到期；预期原退出仍优先生效。"""
    _, holding = callback_fixture()
    policy = PortfolioPolicy(take_profit_enabled=False, stop_loss=0.08, maximum_holding=30)
    assert exit_reason(holding, 7, 18.4, policy) == "stop"
    assert exit_reason(holding, 34, 21, policy) == "time"


@pytest.mark.parametrize(
    "closing,factor,cost",
    [
        (-10.0, -2.0, 20.0),
        (0.0, 2.0, 20.0),
        (10.0, 0.0, 20.0),
        (10.0, 2.0, 0.0),
        (10.0, np.nan, 20.0),
    ],
)
def test_callback_rejects_nonpositive_price_components(closing, factor, cost):
    """场景：价格与因子同负、零值或入场成本无效；预期不把乘积伪装成有效收益。"""
    panel, holding = callback_fixture()
    panel.values["raw_close"][7] = closing
    panel.values["raw_adj_factor"][7] = factor
    holding.entry_adjusted = cost
    assert (
        make_position_state_exit(PositionStateExitRule())(panel, holding, 7, PortfolioPolicy())
        == ""
    )
