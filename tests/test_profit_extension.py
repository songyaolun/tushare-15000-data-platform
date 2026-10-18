"""固定止盈上调与单独关闭止盈的确定性离线检查。"""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_execution import discovery_panel as discovery_panel
from test_discovery_execution import run_strategy, single_signal
from test_discovery_onset_structure import _complete_fixture, _definitions
from test_regime_research import holding

from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy, exit_reason
from tushare_entitlement_hub.strategy_discovery.execution import intraday_fixed_exit
from tushare_entitlement_hub.strategy_discovery.profit_extension import build_variants


def test_variants_preserve_entry_and_risk_rules():
    """四组仅改变固定止盈，原信号、评分、行业退出、止损与追价保持一致。"""
    panel, features = _complete_fixture()
    base = _definitions(panel, features)["gap_2pct"]
    variants = build_variants(base)
    assert [item.policy.take_profit for item in variants] == [0.08, 0.12, 0.16, 0.08]
    assert [item.policy.take_profit_enabled for item in variants] == [True, True, True, False]
    for item in variants:
        assert item.entry_mask is base.entry_mask
        assert item.scores is base.scores
        assert item.exit_mask is base.exit_mask
        assert item.entry_invalidation_mask is base.entry_invalidation_mask
        assert item.maximum_entry_gap == 0.02
        assert item.policy.stop_loss == 0.08
        assert item.policy.maximum_holding == 30
    with pytest.raises(ValueError, match="requires_frozen_anchor"):
        build_variants(replace(base, name="other"))


def test_switch_rejects_non_boolean_and_retains_stop_and_time():
    """关闭止盈后上涨不卖，跌8%仍止损，30日到期仍卖，非法开关被拒绝。"""
    with pytest.raises(ValueError, match="invalid_portfolio_policy"):
        PortfolioPolicy(take_profit_enabled=1)
    policy = PortfolioPolicy(
        take_profit=0.08, stop_loss=0.08, maximum_holding=30, take_profit_enabled=False
    )
    assert exit_reason(holding(), 2, 12, policy) == ""
    assert exit_reason(holding(), 2, 9, policy) == "stop"
    assert exit_reason(holding(), 30, 12, policy) == "time"


def test_off_retains_industry_exit_in_real_account(discovery_panel):
    """上涨超过8%不固定止盈，随后行业退出仍在下一开盘卖出。"""
    panel = discovery_panel
    for name in ("raw_open", "raw_close", "close", "raw_high", "raw_low"):
        panel.values[name][2:, 0] = 12
    panel.values["raw_up_limit"][:, 0] = 15
    exits = np.zeros((8, 12), dtype=bool)
    exits[4, 0] = True
    definition = single_signal(panel, exit_mask=exits)
    definition = replace(
        definition, policy=replace(definition.policy, take_profit=0.08, take_profit_enabled=False)
    )
    result = run_strategy(panel, definition)
    assert result["trades"][0]["exit"] == 5
    assert result["trades"][0]["reason"] == "signal"


def test_intraday_off_keeps_stop_at_open_and_inside_bar(discovery_panel):
    """公共盘中模型也尊重关闭止盈，保留开盘和盘中止损。"""
    panel = discovery_panel
    policy = PortfolioPolicy(take_profit=0.08, stop_loss=0.08, take_profit_enabled=False)
    panel.values["raw_up_limit"][:, 0] = 15
    panel.values["raw_down_limit"][:, 0] = 8
    for name in ("raw_open", "raw_close", "raw_high", "raw_low"):
        panel.values[name][2, 0] = 12
    assert intraday_fixed_exit(panel, holding(), 2, policy) is None
    panel.values["raw_open"][2, 0] = 10
    panel.values["raw_low"][2, 0] = 9.1
    assert intraday_fixed_exit(panel, holding(), 2, policy) == pytest.approx((9.2, "stop"))
    panel.values["raw_open"][2, 0] = 9
    assert intraday_fixed_exit(panel, holding(), 2, policy) == (9, "stop")
