"""Opening purchases share a frozen cash balance without reusing same-day sale proceeds."""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.regime_research.accounting import Distribution
from tushare_entitlement_hub.regime_research.liquidity import LiquidityRule
from tushare_entitlement_hub.regime_research.portfolio import simulate_portfolio
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY


def cash_fixture(*, position_weight=0.45, fee_multiplier=1):
    """Build one sale and two later purchases; weight=.45 leaves cash for two reduced buys.

    Returns panel, definition and policy for a 10,000 CNY account; day1 buys security0,
    day2 sells it after a closing profit signal and attempts securities1/2.
    """
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, 0] = True
    mask[1, 1:3] = True
    panel.values["raw_close"][1, 0] = 10.5
    definition = replace(definition, entry_mask=mask)
    policy = replace(
        DEFAULT_POLICY, position_weight=position_weight, sector_weight=1,
        non_stage_allocation=1, slippage_rate=0, fee_multiplier=fee_multiplier,
    )
    return panel, definition, policy


def replay_cash(panel, definition, policy, **options):
    """Replay the cash fixture with optional engine controls; True enables frozen cash.

    Input opening_cash_only=True returns daily cash, fills and reached sizing constraints.
    Events default to none and the account starts with 10,000 CNY on panel day0.
    """
    return simulate_candidate_allocation(
        panel, np.zeros(8, dtype=np.int8), options.pop("events", []), definition, policy,
        mode="exit_reserve", count=2, start=panel.calendar[0], capital=10000,
        capture_execution_evidence=True, **options,
    )


@pytest.mark.parametrize("fee_multiplier", [1, 2])
def test_multiple_buys_share_opening_cash_including_fees(fee_multiplier):
    """Two buys follow a full sale; both share pre-sale cash and deduct actual multiplied fees."""
    panel, definition, policy = cash_fixture(fee_multiplier=fee_multiplier)
    legacy = replay_cash(panel, definition, policy)
    result = replay_cash(panel, definition, policy, opening_cash_only=True)
    fills = [row for row in result["execution_evidence"]["fills"] if row["day"] == 2]
    assert [row["side"] for row in fills] == ["sell", "buy", "buy"]
    assert [row["shares"] for row in fills[1:]] == [400, 100]
    opening_cash = result["daily"][1]["cash"]
    spending = sum(row["notional"] + row["fee"] for row in fills[1:])
    assert spending <= opening_cash
    legacy_spending = sum(
        row["notional"] + row["fee"] for row in legacy["execution_evidence"]["fills"]
        if row["day"] == 2 and row["side"] == "buy"
    )
    assert legacy_spending > opening_cash
    attempts = [row for row in result["execution_evidence"]["order_attempts"] if row["day"] == 2]
    assert attempts[0]["cash_available"] == opening_cash
    assert attempts[1]["cash_available"] == opening_cash - (
        fills[1]["notional"] + fills[1]["fee"]
    )
    assert attempts[1]["first_binding_constraint"] == "cash_available"
    assert result["daily"][2]["cash"] == pytest.approx(
        opening_cash + fills[0]["notional"] - fills[0]["fee"] - spending
    )


@pytest.mark.parametrize("payment_day", [None, 2, 3])
def test_zero_cash_uses_paid_dividends_and_resets_next_day(payment_day):
    """A fully invested account can buy only paid dividends; sale cash becomes usable tomorrow."""
    panel, definition, policy = cash_fixture(position_weight=1, fee_multiplier=0)
    definition.entry_mask[2, 2] = True
    events = [] if payment_day is None else [Distribution(0, 1, 2, payment_day, 0, 2.5, 0)]
    result = replay_cash(panel, definition, policy, events=events, opening_cash_only=True)
    assert result["daily"][1]["cash"] == 0
    attempts = result["execution_evidence"]["order_attempts"]
    second_day = next(row for row in attempts if row["day"] == 2)
    expected = 2000 if payment_day == 2 else 0
    assert second_day["cash_available"] == expected
    assert second_day["executed_shares"] == expected / 10
    assert result["daily"][2]["cash"] == 10000
    assert result["daily"][2]["receivable"] == (2000 if payment_day == 3 else 0)
    if payment_day != 2:
        third_day = next(row for row in attempts if row["day"] == 3)
        assert third_day["cash_available"] == 10000 + (2000 if payment_day == 3 else 0)
        assert third_day["executed_shares"] >= 1000


@pytest.mark.parametrize("sale_state", ["partial", "limit", "missing"])
def test_partial_or_blocked_sale_keeps_reserve_dependency(sale_state):
    """A sale that fails to close its lot cannot activate the dependent reserve purchase."""
    panel, definition, policy = cash_fixture()
    options = {}
    if sale_state == "partial":
        panel.values["raw_amount"][1, 0] = 150
        options["liquidity_rule"] = LiquidityRule(.01, 1)
    elif sale_state == "limit":
        panel.values["raw_open"][2, 0] = 9
    else:
        panel.values["raw_open"][2, 0] = np.nan
    result = replay_cash(panel, definition, policy, opening_cash_only=True, **options)
    attempts = [row for row in result["execution_evidence"]["order_attempts"] if row["day"] == 2]
    assert attempts[0]["cash_available"] == result["daily"][1]["cash"]
    assert attempts[1]["reason"] == "reserve_exit_wait"
    sales = [row for row in result["execution_evidence"]["fills"]
             if row["day"] == 2 and row["side"] == "sell"]
    assert len(sales) == int(sale_state == "partial")
    if sales:
        assert sales[0]["shares"] == 100
        assert any(row["day"] == 2 and row["security"] == 0 and row["shares"] == 300
                   for row in result["execution_evidence"]["positions"])


def test_explicit_false_preserves_complete_results_and_observer_is_optional():
    """Omitted and explicit False are identical; disabling evidence cannot change True accounts."""
    panel, definition, policy = cash_fixture()
    assert replay_cash(panel, definition, policy) == replay_cash(
        panel, definition, policy, opening_cash_only=False,
    )
    observed = replay_cash(panel, definition, policy, opening_cash_only=True)
    unobserved = simulate_candidate_allocation(
        panel, np.zeros(8, dtype=np.int8), [], definition, policy, mode="exit_reserve",
        count=2, start=panel.calendar[0], capital=10000, opening_cash_only=True,
    )
    observed.pop("execution_evidence")
    assert unobserved == observed


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize("entry_point", ["portfolio", "allocation"])
def test_non_boolean_cash_mode_fails_closed(invalid, entry_point):
    """Both public entry points reject truthy/coercible values instead of silently changing cash."""
    panel, definition, policy = cash_fixture()
    with pytest.raises(ValueError, match="invalid_simulation_inputs"):
        if entry_point == "allocation":
            replay_cash(panel, definition, policy, opening_cash_only=invalid)
        else:
            simulate_portfolio(panel, np.zeros(8, dtype=np.int8), [], policy,
                               opening_cash_only=invalid)
