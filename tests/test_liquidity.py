"""Past-only capacity and partial sale tests, independent of live credentials."""

from dataclasses import replace

import numpy as np
import pytest
import test_portfolio_economic
from test_discovery_replacement import rotation_fixture

from tushare_entitlement_hub.regime_research.accounting import Distribution
from tushare_entitlement_hub.regime_research.execution_audit import audit_execution
from tushare_entitlement_hub.regime_research.liquidity import (
    LiquidityRule,
    historical_notional_limit,
    liquidity_sale_shares,
)
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.replacement import (
    ReplacementRule,
    make_replacement_provider,
)
from tushare_entitlement_hub.strategy_discovery.runner import save_result


@pytest.mark.parametrize("changes", [
    {"maximum_participation": 0}, {"maximum_participation": -1},
    {"maximum_participation": 1.1}, {"maximum_participation": np.nan},
    {"maximum_participation": np.inf}, {"maximum_participation": True},
    {"lookback_sessions": 0}, {"lookback_sessions": 253},
    {"lookback_sessions": 1.2}, {"lookback_sessions": True},
])
def test_liquidity_rule_rejects_invalid_inputs(changes):
    """Invalid fractions/windows are rejected before simulation; no silent unlimited fallback."""
    with pytest.raises(ValueError, match="invalid_liquidity_rule"):
        LiquidityRule(**changes)


@pytest.mark.parametrize("last,expected", [(10000, 100000), (5000, 50000), (0, 0)])
def test_causal_notional_limit_keeps_today_and_future_unseen(last, expected):
    """Past weakness lowers the reference; current/future changes cannot alter the cap."""
    amounts = np.full((25, 1), 10000.)
    amounts[19, 0] = last
    assert historical_notional_limit(amounts, 19, 0, LiquidityRule()) == 0
    assert historical_notional_limit(amounts, 20, 0, LiquidityRule()) == expected
    amounts[20:] = np.nan
    assert historical_notional_limit(amounts, 20, 0, LiquidityRule()) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_history_freezes_capacity(value):
    """One unknown or negative past observation fails closed instead of shrinking the window."""
    amounts = np.full((21, 1), 10000.)
    amounts[5, 0] = value
    assert historical_notional_limit(amounts, 20, 0, LiquidityRule()) == 0


@pytest.mark.parametrize("shares,limit,code,expected", [
    (250, 2500, "600000.SH", 250), (250, 1500, "600000.SH", 100),
    (50, 400, "600000.SH", 0), (50, 500, "600000.SH", 50),
    (250, 1510, "688001.SH", 151), (250, 0, "600000.SH", 0),
])
def test_partial_sale_preserves_board_lots_and_odd_tail(shares, limit, code, expected):
    """Partial mainboard sales use 100-share steps; complete tails and STAR steps remain valid."""
    assert liquidity_sale_shares(shares, 10, limit, code) == expected


def constrained_fixture():
    """Return a 60-day case buying day21 before turnover drops; later sells must use many days."""
    panel = test_portfolio_economic.economic_panel.__wrapped__()
    for signals in panel.signals.values():
        signals[:] = False
        signals[20, 0] = True
    panel.values["raw_amount"][:] = 20000
    panel.values["raw_amount"][21:, 0] = 100
    policy = PortfolioPolicy(switching=False, stage_allocation=False, position_weight=1,
                             sector_weight=1, non_stage_allocation=1, maximum_holding=2)
    return panel, policy


@pytest.mark.parametrize("block", ["none", "suspension", "limit", "unknown_history", "bonus"])
def test_partial_exit_queue_cash_and_dividends(block, tmp_path):
    """Residual shares keep their exit order, cash/fees reconcile, and blocked exits never fill."""
    panel, policy = constrained_fixture()
    events = []
    if block == "suspension":
        panel.values["raw_open"][23:26, 0] = np.nan
    elif block == "limit":
        panel.values["raw_open"][23:26, 0] = 9
    elif block == "unknown_history":
        panel.values["raw_amount"][22, 0] = np.nan
    elif block == "bonus":
        events = [Distribution(0, 21, 22, 27, 25, 1, .2)]
    result = simulate_portfolio(
        panel, np.zeros(60, dtype=np.int8), events, policy, capital=10000,
        start=panel.calendar[0], liquidity_rule=LiquidityRule(), capture_execution_evidence=True,
    )
    fills = result["execution_evidence"]["fills"]
    sales = [row for row in fills if row["side"] == "sell"]
    expected_start = {"none": 23, "suspension": 26, "limit": 26,
                      "unknown_history": 43, "bonus": 25}[block]
    assert sales[0]["day"] == expected_start
    assert len(sales) > 1 and sales[0]["shares"] == 100
    assert all(row["day"] > row["entry"] for row in sales)
    expected_shares = fills[0]["shares"] * (1.2 if block == "bonus" else 1)
    assert sum(row["shares"] for row in sales) == pytest.approx(expected_shares)
    assert result["open_positions"] == 0
    assert all(row["cash"] >= 0 for row in result["daily"])
    assert abs(result["reconciliation_error"]) < 1e-8
    result["daily_returns"] = [0] * 60
    result["timed_market_returns"] = [0] * 60
    save_result(tmp_path, result, {}, {})
    audited = audit_execution(tmp_path, 10000, 1, .0005)
    assert audited["passed"], audited["errors"]
    assert (tmp_path / "liquidity_orders.csv").exists()


def test_partial_sale_does_not_release_paired_replacement_slot():
    """One paired seller only partly exits; its buyer cannot borrow the other seller's cash/slot."""
    panel, definition = rotation_fixture()
    panel.values["raw_amount"][1:, 0] = 1000
    result = simulate_candidate_allocation(
        panel, np.ones(8, dtype=np.int8), [], definition,
        replace(DEFAULT_POLICY, non_stage_allocation=.1), mode="exit_reserve", count=2,
        start=panel.calendar[0], liquidity_rule=LiquidityRule(.01, 1),
        capture_execution_evidence=True,
        replacement_provider=make_replacement_provider(ReplacementRule(1, .01, "momentum", 0, 2)),
    )
    fills = result["execution_evidence"]["fills"]
    assert any(row["side"] == "sell" and row["security"] == 0 and row["day"] == 2
               for row in fills)
    assert result["daily"][2]["weights"][0] > 0
    assert 2 not in result["daily"][2]["weights"]
    assert 3 in result["daily"][2]["weights"]


def test_disabled_rule_preserves_default_account_exactly():
    """Omitting the optional rule and passing None produce identical full legacy results."""
    panel, policy = constrained_fixture()
    arguments = dict(capital=10000, start=panel.calendar[0])
    states = np.zeros(60, dtype=np.int8)
    assert simulate_portfolio(panel, states, [], policy, **arguments) == simulate_portfolio(
        panel, states, [], policy, liquidity_rule=None, **arguments)


@pytest.mark.parametrize("zero_reference", [False, True])
def test_buy_reduction_and_terminal_residual_are_explicit(zero_reference, tmp_path):
    """A large buy is reduced, not topped up; zero historical median cancels it entirely."""
    panel, policy = constrained_fixture()
    if zero_reference:
        panel.values["raw_amount"][:20, 0] = 0
    result = simulate_portfolio(
        panel, np.zeros(60, dtype=np.int8), [], policy, capital=1_000_000,
        start=panel.calendar[0], liquidity_rule=LiquidityRule(), capture_execution_evidence=True,
    )
    order = result["liquidity_orders"][0]
    assert order["executed_shares"] < order["requested_shares"]
    if zero_reference:
        assert order["executed_shares"] == 0
        assert result["entry_days"] == []
    else:
        assert len(result["entry_days"]) == 1
        assert result["open_positions"] == 1
        assert result["execution_evidence"]["positions"][-1]["sell_reason"] == "time"
    result["daily_returns"] = [0] * 60
    result["timed_market_returns"] = [0] * 60
    save_result(tmp_path, result, {}, {})
    assert audit_execution(tmp_path, 1_000_000, 1, .0005)["passed"]


def test_open_and_intraday_sales_share_one_daily_limit():
    """A second sell attempt cannot consume the same security's historical budget again."""
    panel, policy = constrained_fixture()

    def intraday(current, holding, day, current_policy):
        """Request a synthetic sell at 10; engine T+1 and remaining daily budget still apply."""
        return 10., "fixture_intraday"

    result = simulate_portfolio(
        panel, np.zeros(60, dtype=np.int8), [], policy, capital=10000,
        start=panel.calendar[0], liquidity_rule=LiquidityRule(), capture_execution_evidence=True,
        intraday_exit_provider=intraday,
    )
    sales = [row for row in result["execution_evidence"]["fills"]
             if row["side"] == "sell" and row["day"] == 23]
    assert len(sales) == 1
    assert sum(row["notional"] for row in sales) <= 1000
    attempts = [row for row in result["liquidity_orders"]
                if row["side"] == "sell" and row["day"] == 23]
    assert len(attempts) == 2 and attempts[-1]["executed_shares"] == 0


def test_intraday_partial_exit_remains_queued_without_repeated_signal():
    """One intraday exit signal leaves a residual that must retry at the next open."""
    panel, policy = constrained_fixture()

    def intraday(current, holding, day, current_policy):
        """Signal only on day 22; later days return None to test persistent exit intent."""
        return (10., "single_intraday") if day == 22 else None

    result = simulate_portfolio(
        panel, np.zeros(60, dtype=np.int8), [], replace(policy, maximum_holding=59),
        capital=10000, start=panel.calendar[0], liquidity_rule=LiquidityRule(),
        capture_execution_evidence=True, intraday_exit_provider=intraday,
    )
    sales = [row for row in result["execution_evidence"]["fills"] if row["side"] == "sell"]
    assert sales[0]["day"] == 22
    assert sales[1]["day"] == 23
    assert all(row["reason"] == "single_intraday" for row in sales)
    assert result["open_positions"] == 0
