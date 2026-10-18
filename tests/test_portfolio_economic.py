"""资金执行配置的离线回归：默认兼容、成本预算、仓位和退出开关。"""

from dataclasses import replace

import numpy as np
import pytest
import test_regime_research

from tushare_entitlement_hub.regime_research.accounting import (
    affordable_shares,
    transaction_fee,
)
from tushare_entitlement_hub.regime_research.portfolio import (
    PortfolioPolicy,
    exit_reason,
    simulate_portfolio,
)


@pytest.fixture
def economic_panel():
    """输入无，输出固定10元、仅首日一只股票发信号的60日合成面板。"""
    panel = test_regime_research.panel.__wrapped__()
    for name in ("raw_open", "raw_close", "close", "raw_low", "raw_high"):
        panel.values[name][:] = 10
    panel.values["raw_up_limit"][:] = 11
    panel.values["raw_down_limit"][:] = 9
    for signals in panel.signals.values():
        signals[:] = False
        signals[0, 0] = True
    return panel


def replay(panel, **changes):
    """输入面板与配置覆盖项，输出1万元账户；例maximum_holding=2使第三日开盘卖出。"""
    policy = PortfolioPolicy(
        switching=False,
        stage_allocation=False,
        position_weight=1,
        sector_weight=1,
        maximum_holding=2,
    )
    return simulate_portfolio(
        panel,
        np.zeros(len(panel.calendar), dtype=np.int8),
        [],
        replace(policy, **changes),
        capital=10000,
        start=panel.calendar[0],
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"fixed_price_exits": 0},
        {"fixed_price_exits": "false"},
        {"slippage_rate": -0.001},
        {"slippage_rate": 1},
        {"slippage_rate": np.nan},
        {"slippage_rate": np.inf},
        {"non_stage_allocation": -0.1},
        {"non_stage_allocation": 1.1},
        {"non_stage_allocation": np.nan},
        {"non_stage_allocation": np.inf},
        {"fee_multiplier": -1},
        {"fee_multiplier": np.nan},
        {"fee_multiplier": np.inf},
    ],
)
def test_invalid_economic_policy_rejected(changes):
    """场景：越界或非有限成本/仓位及非布尔开关；预期构造时明确拒绝。"""
    with pytest.raises(ValueError, match="invalid_portfolio_policy"):
        PortfolioPolicy(**changes)


def test_explicit_defaults_preserve_complete_accounting(economic_panel):
    """场景：省略新参数与显式原默认值；预期逐日、交易和全部汇总完全一致。"""
    baseline = replay(economic_panel, position_weight=0.5)
    explicit = replay(
        economic_panel,
        position_weight=0.5,
        fixed_price_exits=True,
        slippage_rate=0.0005,
        non_stage_allocation=0.9,
        fee_multiplier=1,
    )
    assert explicit == baseline
    trade = baseline["trades"][0]
    assert trade["shares"] == 400
    assert trade["cost"] == pytest.approx(400 * 10.005 + 5.04002)
    assert trade["proceeds"] == pytest.approx(400 * 9.995 - 7.03898)


@pytest.mark.parametrize("closing,expected", [(8, "stop"), (12, "profit")])
def test_fixed_price_exits_switch_preserves_time_and_t_plus_one(
    economic_panel, closing, expected
):
    """场景：买入当日越过固定阈值；预期开关关闭后只按期限于后续开盘退出。"""
    economic_panel.values["raw_close"][1, 0] = closing
    baseline = replay(economic_panel, position_weight=0.5)
    disabled = replay(economic_panel, position_weight=0.5, fixed_price_exits=False)
    assert baseline["trades"][0]["reason"] == expected
    assert baseline["trades"][0]["exit"] == 2
    assert disabled["trades"][0]["reason"] == "time"
    assert disabled["trades"][0]["exit"] == 3
    assert disabled["trades"][0]["entry"] == 1


def test_disabled_fixed_exits_preserve_other_close_rules():
    """场景：直接调用关闭固定退出；预期外部开关不能重启，跟踪和反弹失败仍生效。"""
    policy = PortfolioPolicy(fixed_price_exits=False)
    assert exit_reason(test_regime_research.holding(), 2, 8, policy, fixed_exits=True) == ""
    assert exit_reason(test_regime_research.holding(), 2, 12, policy) == ""
    assert exit_reason(test_regime_research.holding(), 30, np.nan, policy) == "time"
    assert (
        exit_reason(
            test_regime_research.holding(stage=1, peak_adjusted=12),
            2,
            10,
            replace(policy, trailing=0.1),
        )
        == "trailing"
    )
    assert (
        exit_reason(test_regime_research.holding(), 3, 8, replace(policy, early_exit=2))
        == "failed_rebound"
    )


@pytest.mark.parametrize("fee_multiplier", [0, 1, 2, 200])
@pytest.mark.parametrize("slippage_rate", [0, 0.01, 0.2])
def test_cost_stress_uses_actual_fills_and_never_overdraws_cash(
    economic_panel, fee_multiplier, slippage_rate
):
    """场景：零成本及高滑点/手续费压力；预期整手预算、限价、双边费用和净值逐项对账。"""
    result = replay(
        economic_panel,
        slippage_rate=slippage_rate,
        fee_multiplier=fee_multiplier,
        non_stage_allocation=1,
        fixed_price_exits=False,
    )
    buying_price = min(10 * (1 + slippage_rate), 11)
    selling_price = max(10 * (1 - slippage_rate), 9)
    expected_shares = affordable_shares(
        10000, buying_price, "600000.SH", "20240102", fee_multiplier=fee_multiplier
    )
    trade = result["trades"][0]
    buying_notional = expected_shares * buying_price
    selling_notional = expected_shares * selling_price
    buying_fee = transaction_fee(buying_notional, "20240102", False) * fee_multiplier
    selling_fee = transaction_fee(selling_notional, "20240104", True) * fee_multiplier
    assert trade["shares"] == expected_shares
    assert trade["cost"] == pytest.approx(buying_notional + buying_fee)
    assert trade["proceeds"] == pytest.approx(selling_notional - selling_fee)
    assert result["daily"][1]["cash"] == pytest.approx(10000 - trade["cost"])
    assert result["daily"][1]["cost"] == pytest.approx(
        buying_fee + expected_shares * (buying_price - 10)
    )
    assert result["daily"][3]["cost"] == pytest.approx(
        selling_fee + expected_shares * (10 - selling_price)
    )
    assert result["fees"] == pytest.approx(buying_fee + selling_fee)
    assert min(row["cash"] for row in result["daily"]) >= 0
    assert result["daily"][-1]["nav"] == pytest.approx(
        10000 - trade["cost"] + trade["proceeds"]
    )
    assert abs(result["reconciliation_error"]) < 1e-8


@pytest.mark.parametrize("code,minimum", [("600000.SH", 100), ("688001.SH", 200)])
def test_affordable_shares_reserves_full_scaled_fee(code, minimum):
    """场景：预算只够原手续费或恰好双倍；预期倍增费用减少可买股数且零费用可买低价股。"""
    notional = minimum * 10
    fee = transaction_fee(notional, "20240101", False)
    assert affordable_shares(notional + fee, 10, code, "20240101") == minimum
    assert (
        affordable_shares(notional + fee, 10, code, "20240101", fee_multiplier=2) == 0
    )
    assert (
        affordable_shares(notional + 2 * fee, 10, code, "20240101", fee_multiplier=2)
        == minimum
    )
    assert affordable_shares(notional, 10, code, "20240101", fee_multiplier=0) == minimum
    assert affordable_shares(5, 0.01, code, "20240101", fee_multiplier=0) == 500


@pytest.mark.parametrize("multiplier", [-1, np.nan, np.inf])
def test_affordable_shares_rejects_invalid_fee_multiplier(multiplier):
    """场景：独立预算函数接收非法倍数；预期即使预算不足也先拒绝错误配置。"""
    with pytest.raises(ValueError, match="invalid_fee_multiplier"):
        affordable_shares(0, 10, "600000.SH", "20240101", fee_multiplier=multiplier)


@pytest.mark.parametrize("allocation,expected_positions", [(0, 0), (0.2, 2), (0.5, 5), (1, 10)])
def test_non_stage_allocation_controls_both_slots_and_cash_budget(
    economic_panel, allocation, expected_positions
):
    """场景：十只同期候选、10%单股仓位；预期总仓位同时限制订单数和成交预算。"""
    for signals in economic_panel.signals.values():
        signals[0, :10] = True
    result = replay(
        economic_panel,
        non_stage_allocation=allocation,
        position_weight=0.1,
        slippage_rate=0,
        fee_multiplier=0,
    )
    assert result["daily"][1]["positions"] == expected_positions
    assert result["daily"][1]["exposure"] == pytest.approx(allocation)
    assert result["daily"][1]["cash"] == pytest.approx(10000 * (1 - allocation))


def test_stage_allocation_ignores_non_stage_override(economic_panel):
    """场景：启用阶段仓位且非阶段上限为零；预期仍按阶段30%而非零上限买入。"""
    result = replay(
        economic_panel,
        stage_allocation=True,
        non_stage_allocation=0,
        position_weight=0.1,
        slippage_rate=0,
        fee_multiplier=0,
    )
    assert result["entry_days"] == [1]
    assert result["daily"][1]["exposure"] == pytest.approx(0.1)
