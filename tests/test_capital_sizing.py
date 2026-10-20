"""Validate NAV sizing boundaries and the causal next-open account integration."""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.regime_research.capital_sizing import (
    CapitalSizingRule,
    SizingDecision,
    finite_number,
    parse_capital_tiers,
)
from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY


@pytest.mark.parametrize(
    "value,expected", [(1, True), (0.1, True), (True, False), (float("nan"), False), ("1", False)]
)
def test_finite_number(value, expected):
    """Finite numerical observations pass; Boolean, text and missing values fail."""
    assert finite_number(value) is expected


@pytest.mark.parametrize(
    "assets,expected",
    [(9_999_999, 0.1), (10_000_000, 0.05), (99_999_999, 0.05), (100_000_000, 0.02)],
)
def test_tier_boundaries(assets, expected):
    """Exact closing NAV thresholds enter the new tier with no future account information."""
    rule = CapitalSizingRule(((0.0, 0.1), (1e7, 0.05), (1e8, 0.02)))
    assert rule(assets, 1).position_weight == expected
    assert rule(assets, 1).maximum_total_weight == 1
    assert rule(120000, 0).maximum_total_weight == 0.15


def test_weak_cap_and_downward_crossing():
    """Weak 5% ceilings never enlarge a 2% tier; NAV drawdowns may reverse tiers."""
    rule = CapitalSizingRule(((0.0, 0.1), (1e7, 0.05), (1e8, 0.02)), 0.15, 0.05)
    assert rule(120000, 0).position_weight == 0.05
    assert rule(1e8, 0).position_weight == 0.02
    assert rule(1e7 - 1, 1).position_weight == 0.1
    assert CapitalSizingRule(weak_allocation=0)(120000, 0).maximum_total_weight == 0


@pytest.mark.parametrize(
    "arguments",
    [
        (0.0, 0.15, 0),
        (0.5, 0.15, 0),
        (0.1, -1, 0),
        (0.1, 1.1, 0),
        (0.1, 0.15, -1),
        (True, 0.15, 0),
        (0.1, float("nan"), 0),
    ],
)
def test_invalid_decisions(arguments):
    """Invalid fractions, leverage and missing values cannot enter the account."""
    with pytest.raises(ValueError, match="invalid_account_sizing"):
        SizingDecision(*arguments)


@pytest.mark.parametrize("tiers", [(), ((1, 0.1),), ((0, 0.1), (0, 0.05)), ((0, 0.05), (1, 0.1))])
def test_invalid_tier_orders(tiers):
    """Floors must start at zero and rise; weights cannot rise with account size."""
    with pytest.raises(ValueError):
        CapitalSizingRule(tiers)


@pytest.mark.parametrize(
    "assets,stage", [(0, 0), (-1, 0), (float("nan"), 0), (True, 0), (1, True), (1, 2)]
)
def test_unknown_account_inputs_fail(assets, stage):
    """Missing NAV and nonbinary market states fail instead of inventing a sizing tier."""
    with pytest.raises(ValueError, match="invalid_sizing_observation"):
        CapitalSizingRule()(assets, stage)


@pytest.mark.parametrize("text", ["", "0", "0:.1:2", "0:nan", "1:.1", "0:.1,1:.2"])
def test_invalid_cli_tiers(text):
    """Malformed or unsafe CLI tiers fail before loading any research data."""
    with pytest.raises(ValueError, match="invalid_capital_tiers"):
        parse_capital_tiers(text)


def test_cli_tier_parse():
    """Explicit CNY floors retain full numeric precision and weight fractions."""
    assert parse_capital_tiers("0:.1,10000000:.05") == ((0.0, 0.1), (1e7, 0.05))


def replay_fixture(rule=None, *, strong=False, changing_prices=False, future_price=None):
    """Replay an eight-day fixture; e.g. weak fixed10% opens one lot on day1."""
    panel, definition = allocation_fixture()
    definition = replace(
        definition, policy=replace(definition.policy, take_profit=0.9, stop_loss=0.9)
    )
    if changing_prices:
        panel.values["raw_close"][1, :] = 10.9
    if future_price is not None:
        panel.values["raw_close"][-1, :] = future_price
    policy = replace(
        DEFAULT_POLICY,
        position_weight=0.1,
        sector_weight=0.4,
        stage_allocation=True,
        force_rebalance=False,
        allocations=(0.15, 1.0, 1.0, 1.0, 1.0),
    )
    return simulate_candidate_allocation(
        panel,
        np.full(8, int(strong), dtype=np.int8),
        [],
        definition,
        policy,
        mode="exit_reserve",
        count=12,
        capital=1e6,
        start=panel.calendar[0],
        opening_cash_only=True,
        capture_execution_evidence=True,
        account_sizing_provider=rule,
    )


def test_constant_provider_matches_original_account():
    """Adding an identical sizing policy leaves every trade, fill, candidate and NAV unchanged."""
    expected, actual = replay_fixture(), replay_fixture(CapitalSizingRule())
    assert actual["trades"] == expected["trades"]
    assert actual["execution_evidence"] == expected["execution_evidence"]
    assert actual["candidate_selections"] == expected["candidate_selections"]
    assert [
        {key: value for key, value in row.items() if not key.startswith("sizing_")}
        for row in actual["daily"]
    ] == expected["daily"]


@pytest.mark.parametrize(
    "cap,weight,count", [(0.15, 0.1, 1), (0.15, 0.05, 3), (0.1, 0.05, 2), (0, 0.05, 0)]
)
def test_weak_slots_change_with_both_limits(cap, weight, count):
    """Total/stock ratios govern slots while the first signal remains entirely in cash."""
    result = replay_fixture(CapitalSizingRule(weak_allocation=cap, weak_position_weight=weight))
    assert result["daily"][0]["positions"] == 0
    assert result["daily"][1]["positions"] == count
    assert result["execution_evidence"]["allocation_states"][0]["scheduled"] == count


def test_tier_crossing_does_not_reprice_same_day_or_force_sell():
    """A day1 close crossing changes later slots, never day1 opening budgets or existing lots."""
    result = replay_fixture(
        CapitalSizingRule(((0.0, 0.1), (1_005_000.0, 0.05))), strong=True, changing_prices=True
    )
    assert result["daily"][0]["sizing_weight"] == 0.1
    assert result["daily"][1]["sizing_weight"] == 0.05
    assert result["daily"][2]["sizing_weight"] == 0.1
    assert result["daily"][1]["positions"] == 10
    # Two new slots exist, but residual opening cash can fund only one additional lot.
    assert result["execution_evidence"]["allocation_states"][1]["scheduled"] == 2
    assert result["daily"][2]["positions"] == 11
    assert any(
        row["day"] == 2 and row["reason"] == "sizing_zero"
        for row in result["execution_evidence"]["order_attempts"]
    )
    assert result["trades"] == []
    assert all(
        row["signal_budget"] == 100000
        for row in result["execution_evidence"]["order_attempts"]
        if row["day"] == 1 and row["price"] is not None
    )


def test_future_changes_cannot_change_earlier_sizing():
    """Changing only the final close cannot alter earlier NAV, fills or sizing decisions."""
    rule = CapitalSizingRule(((0.0, 0.1), (1_005_000.0, 0.05)))
    original, changed = replay_fixture(rule), replay_fixture(rule, future_price=20)
    assert original["daily"][:-1] == changed["daily"][:-1]
    assert original["execution_evidence"]["fills"] == changed["execution_evidence"]["fills"]


def test_invalid_provider_fails_before_scheduling():
    """A callback returning an arbitrary tuple cannot bypass the validated sizing contract."""
    with pytest.raises(ValueError, match="invalid_account_sizing_provider"):
        replay_fixture(lambda assets, stage: (0.1, 0.15))
