"""冻结短反资格、原信号时钟、单次开盘与五日真实资金账本验证。"""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from test_portfolio_economic import economic_panel as economic_panel

from tushare_entitlement_hub.economic_research import execution
from tushare_entitlement_hub.economic_research.reversal import ReversalStrategy
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy


def frozen_strategy(panel, name="reversal", security=0):
    """给固定10元面板构造单次冻结资格；例如delayed只在第1日派发原第0日信号。"""
    mask = np.zeros_like(panel.values["close"], dtype=bool)
    delay = int(name == "delayed")
    mask[delay, security] = True
    policy = PortfolioPolicy(
        name=name,
        switching=False,
        stage_allocation=False,
        force_rebalance=False,
        maximum_holding=5,
        fixed_price_exits=False,
        non_stage_allocation=0.5,
        position_weight=0.1,
        sector_weight=0.5,
        slippage_rate=0.001 if name == "cost_stress" else 0.0005,
        fee_multiplier=2 if name == "cost_stress" else 1,
    )
    return ReversalStrategy(
        name, mask, np.ones(mask.shape), np.arange(mask.shape[0]) - delay, policy
    )


def replay(panel, strategy):
    """重放10万元离线账户；例如基础单信号产生第1日买入、第6日开盘卖出。"""
    return execution.simulate_reversal(
        panel,
        np.zeros(len(panel.calendar), dtype=np.int8),
        [],
        strategy,
        capital=100000,
        start=panel.calendar[0],
    )


@pytest.mark.parametrize(("name", "entry", "exit_day"), [("reversal", 1, 6), ("delayed", 2, 7)])
def test_frozen_candidates_preserve_source_and_real_entry_holding_clock(
    economic_panel, name, entry, exit_day
):
    """场景：资格冻结后行情过滤字段失效且持仓浮亏；预期仍按唯一开盘成交，只在实际持有五日后卖出。"""
    strategy = frozen_strategy(economic_panel, name)
    dispatch = entry - 1
    for field in ("beta", "capitalization", "raw_amount"):
        economic_panel.values[field][dispatch] = np.nan
    economic_panel.values["valid"][dispatch] = 0
    economic_panel.values["raw_close"][entry:exit_day, 0] = 7
    result = replay(economic_panel, strategy)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert (trade["source_signal"], trade["confirmation"], trade["entry"], trade["exit"]) == (
        0,
        dispatch,
        entry,
        exit_day,
    )
    assert trade["reason"] == "time"
    assert trade["shares"] > 0 and trade["shares"] % 100 == 0
    assert result["execution_model"]["source_delay_sessions"] == int(name == "delayed")
    assert min(row["cash"] for row in result["daily"]) >= 0
    assert abs(result["reconciliation_error"]) < 1e-7
    for row in result["daily"]:
        assert row["nav"] == pytest.approx(
            row["cash"] + row["receivable"] + row["nav"] * row["exposure"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("raw_open", np.nan), ("raw_open", 11), ("raw_adj_factor", np.nan), ("raw_adj_factor", 0)],
)
@pytest.mark.parametrize("name", ["reversal", "delayed"])
def test_unfilled_next_open_is_cancelled_without_later_entry(economic_panel, field, value, name):
    """场景：唯一执行开盘缺失/涨停/因子无效，下日恢复；预期撤单且此后没有补买或费用。"""
    strategy = frozen_strategy(economic_panel, name)
    economic_panel.values[field][1 + int(name == "delayed"), 0] = value
    result = replay(economic_panel, strategy)
    assert result["trades"] == result["entry_days"] == []
    assert result["fees"] == 0 and result["open_positions"] == 0


def test_cost_stress_changes_real_fill_prices_and_scaled_fees(economic_panel):
    """场景：同一冻结资格用双倍滑点/费用；预期真实成本提高、净值降低，资格和时钟不变。"""
    base = replay(economic_panel, frozen_strategy(economic_panel))
    stress = replay(economic_panel, frozen_strategy(economic_panel, "cost_stress"))
    assert stress["entry_days"] == base["entry_days"]
    assert stress["fees"] > base["fees"]
    assert stress["daily"][-1]["nav"] < base["daily"][-1]["nav"]
    assert stress["trades"][0]["source_signal"] == base["trades"][0]["source_signal"]


def test_order_hooks_reject_late_open_and_cancel_only_after_dispatch(economic_panel):
    """场景：独立调用hook检查边界；预期第1日允许、第2日拒绝，派发日不撤单而次日撤单。"""
    assert execution.next_open_allowed(economic_panel, 0, 1, 0)
    assert not execution.next_open_allowed(economic_panel, 0, 2, 0)
    assert not execution.cancel_expired_order(economic_panel, 0, 0, 0)
    assert execution.cancel_expired_order(economic_panel, 0, 1, 0)


@pytest.mark.parametrize("field", ["entry_mask", "scores", "source_days"])
def test_misaligned_frozen_arrays_are_rejected(economic_panel, field):
    """场景：任一冻结矩阵轴与面板不符；预期模拟前拒绝，不能错位引用证券或信号日。"""
    strategy = frozen_strategy(economic_panel)
    object.__setattr__(strategy, field, getattr(strategy, field)[:-1])
    with pytest.raises(ValueError, match="reversal_strategy_shape_mismatch"):
        replay(economic_panel, strategy)


@pytest.mark.parametrize("source", [-1, 1])
def test_invalid_source_clock_is_rejected(economic_panel, source):
    """场景：基础派发记录负值或未来源日；预期明确拒绝而非悄悄改写延迟。"""
    strategy = frozen_strategy(economic_panel)
    changed = strategy.source_days.copy()
    changed[0] = source
    object.__setattr__(strategy, "source_days", changed)
    with pytest.raises(ValueError, match="reversal_source_clock_mismatch"):
        replay(economic_panel, strategy)


@pytest.mark.parametrize(
    "changes",
    [
        {"fixed_price_exits": True},
        {"maximum_holding": 6},
        {"trailing": 0.1},
        {"early_exit": 1},
        {"stage_allocation": True},
        {"force_rebalance": True},
    ],
)
def test_non_time_exit_policies_are_rejected(economic_panel, changes):
    """场景：传入固定阈值、跟踪、阶段卖出或非五日期限；预期拒绝违反预声明实验的政策。"""
    strategy = frozen_strategy(economic_panel)
    object.__setattr__(strategy, "policy", replace(strategy.policy, **changes))
    with pytest.raises(ValueError, match="reversal_requires_time_only_five_session_policy"):
        replay(economic_panel, strategy)


def test_unexpected_engine_late_fill_is_rejected(economic_panel, monkeypatch):
    """场景：下游返回违反dispatch+1的成交；预期适配层审计失败，不导出看似有效的交易。"""
    monkeypatch.setattr(
        execution,
        "simulate_portfolio",
        Mock(return_value={"trades": [{"confirmation": 0, "entry": 2}]}),
    )
    with pytest.raises(ValueError, match="reversal_fill_clock_violation"):
        replay(economic_panel, frozen_strategy(economic_panel))


def test_multiple_pending_sessions_policy_is_rejected(economic_panel):
    """场景：冻结策略声明多日挂单；预期拒绝，不能绕过唯一下一开盘契约。"""
    strategy = replace(frozen_strategy(economic_panel), maximum_pending_sessions=2)
    with pytest.raises(ValueError, match="reversal_requires_time_only_five_session_policy"):
        replay(economic_panel, strategy)
