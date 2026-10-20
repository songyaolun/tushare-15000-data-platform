"""Executed-fill and dividend observations must preserve the original cash engine."""

import csv

import numpy as np
import pytest
import test_portfolio_economic

from tushare_entitlement_hub.regime_research.accounting import Distribution
from tushare_entitlement_hub.regime_research.execution_audit import audit_execution
from tushare_entitlement_hub.regime_research.execution_evidence import ExecutionEvidence
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from tushare_entitlement_hub.strategy_discovery.runner import save_result


@pytest.mark.parametrize("events", [
    [], [Distribution(0, 1, 2, 5, 4, 1, .2)],
    [Distribution(0, 1, 2, 70, 70, 1, .2)],
    [Distribution(0, 1, 2, 2, 0, 1, 0)],
])
def test_observations_preserve_account_and_rebuild_daily_cash(events, tmp_path):
    """Cash-only, delayed, unresolved and same-day dividends preserve every legacy result."""
    panel = test_portfolio_economic.economic_panel.__wrapped__()
    policy = PortfolioPolicy(switching=False, stage_allocation=False,
                             position_weight=1, sector_weight=1, maximum_holding=2,
                             non_stage_allocation=1)
    arguments = dict(capital=10000, start=panel.calendar[0])
    states = np.zeros(len(panel.calendar), dtype=np.int8)
    baseline = simulate_portfolio(panel, states, events, policy, **arguments)
    observed = simulate_portfolio(panel, states, events, policy,
                                  capture_execution_evidence=True, **arguments)
    evidence = observed.pop("execution_evidence")
    assert observed == baseline
    cash = 10000
    receivable = 0
    for row in observed["daily"]:
        day = row["day"]
        for event in evidence["distributions"]:
            if event["day"] == day and event["kind"] == "entitlement":
                receivable += event["cash"]
            if event["day"] == day and event["kind"] == "payment":
                cash += event["cash"]
                receivable -= event["cash"]
        for fill in evidence["fills"]:
            if fill["day"] != day:
                continue
            cash += (1 if fill["side"] == "sell" else -1) * fill["notional"] - fill["fee"]
            assert cash == pytest.approx(fill["cash_after"], abs=1e-9)
        marked = sum(position["market_value"] for position in evidence["positions"]
                     if position["day"] == day)
        assert cash == pytest.approx(row["cash"], abs=1e-9)
        assert receivable == pytest.approx(row["receivable"], abs=1e-9)
        assert cash + receivable + marked == pytest.approx(row["nav"], abs=1e-9)
    assert evidence["fills"][0]["raw_amount"] == 50000
    if events and events[0].listing == 4:
        assert evidence["fills"][1]["day"] == 4
        assert evidence["distributions"][-1]["kind"] == "payment"
    observed["execution_evidence"] = evidence
    observed["daily_returns"] = [0] * len(observed["daily"])
    observed["timed_market_returns"] = [0] * len(observed["daily"])
    save_result(tmp_path, observed, {}, {})
    with (tmp_path / "fills.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(evidence["fills"])
    assert (tmp_path / "distributions.csv").exists()
    assert (tmp_path / "positions.csv").exists()
    audit = audit_execution(tmp_path, 10000, 1, .0005)
    assert audit["passed"], audit["errors"]
    assert audit["total_fees"] == pytest.approx(observed["fees"])
    assert not audit["capacity"]["capacity_certified"]
    if not events:
        rows = list(csv.DictReader((tmp_path / "fills.csv").open()))
        rows[0]["fee"] = str(float(rows[0]["fee"]) + 1)
        with (tmp_path / "fills.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        broken = audit_execution(tmp_path, 10000, 1, .0005)
        assert not broken["passed"]
        assert any(error["kind"] == "fee" for error in broken["errors"])
        oversized = [rows[0], rows[1] | {"shares": str(float(rows[1]["shares"]) * 2)}]
        with (tmp_path / "fills.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(oversized)
        with pytest.raises(ValueError, match="sale_exceeds_remaining_shares"):
            audit_execution(tmp_path, 10000, 1, .0005)
        with (tmp_path / "fills.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        trades = list(csv.DictReader((tmp_path / "trades.csv").open()))
        trades[0]["ts_code"] = "600999.SH"
        with (tmp_path / "trades.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trades[0]))
            writer.writeheader()
            writer.writerows(trades)
        with pytest.raises(ValueError, match="trade_lot_mismatch"):
            audit_execution(tmp_path, 10000, 1, .0005)
        for field, value, error in (
            ("day", "999", "evidence_outside_daily_axis"),
            ("side", "unknown", "invalid_fill_identity_or_quantity"),
            ("shares", "nan", "invalid_fill_identity_or_quantity"),
        ):
            corrupted = [rows[0] | {field: value}, *rows[1:]]
            with (tmp_path / "fills.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(corrupted)
            with pytest.raises(ValueError, match=error):
                audit_execution(tmp_path, 10000, 1, .0005)
        (tmp_path / "daily.csv").write_text("")
        with pytest.raises(ValueError, match="invalid_daily_axis"):
            audit_execution(tmp_path, 10000, 1, .0005)
    else:
        rows = list(csv.DictReader((tmp_path / "distributions.csv").open()))
        rows[1]["kind"] = "unknown"
        with (tmp_path / "distributions.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError, match="unknown_distribution_kind"):
            audit_execution(tmp_path, 10000, 1, .0005)


def test_empty_evidence_and_unknown_market_observation():
    """Empty portfolios have empty tables; missing daily amounts remain unknown, never zero."""
    panel = test_portfolio_economic.economic_panel.__wrapped__()
    collector = ExecutionEvidence(panel)
    collector.snapshot(0, {})
    assert collector.result() == {"fills": [], "distributions": [], "positions": [],
                                  "order_attempts": [], "allocation_states": []}
    panel.values["raw_amount"][1, 0] = np.nan
    result = simulate_portfolio(
        panel, np.zeros(len(panel.calendar), dtype=np.int8), [],
        PortfolioPolicy(switching=False), start=panel.calendar[0],
        capture_execution_evidence=True,
    )
    assert np.isnan(result["execution_evidence"]["fills"][0]["raw_amount"])


def test_order_diagnostics_distinguish_waits_and_competing_budgets():
    """A blocked pair has unknown budgets; a sized order retains every competing amount."""
    panel = test_portfolio_economic.economic_panel.__wrapped__()
    collector = ExecutionEvidence(panel)
    collector.order_attempt(2, 0, 1, "paired_exit_wait")
    collector.order_attempt(2, 1, 1, "sizing_zero", {
        "signal_budget": 1000, "position_limit": 1000, "cash_available": 900,
        "target_available": 700, "sector_available": 50,
    }, price=10, requested_shares=0, executed_shares=0, liquidity_available=10000)
    collector.allocation_state(2, scheduled=0, unused_slots=1)
    rows = collector.result()["order_attempts"]
    assert rows[0]["cash_available"] is None
    assert rows[1]["first_binding_constraint"] == "sector_available"
    assert rows[1]["liquidity_available"] == 10000
    assert collector.result()["allocation_states"][0]["unused_slots"] == 1


@pytest.mark.parametrize("reason", [
    "expired", "opening_untradable", "entry_filter", "unknown_sector",
])
def test_blocked_entry_records_reason_without_inventing_fills(reason):
    """Blocked orders retain their reason; only a temporary opening block may fill later.

    Example: missing opens on days 1–3 expire on day 4 with no purchase; an
    unavailable industry after a one-day wait cancels the order without a fill.
    """
    panel = test_portfolio_economic.economic_panel.__wrapped__()
    if reason == "expired":
        panel.values["raw_open"][1:4, 0] = np.nan
    elif reason in {"opening_untradable", "unknown_sector"}:
        panel.values["raw_open"][1, 0] = np.nan
        if reason == "unknown_sector":
            panel.values["sector_index"][1, 0] = np.nan
    result = simulate_portfolio(
        panel, np.zeros(len(panel.calendar), dtype=np.int8), [],
        PortfolioPolicy(switching=False, stage_allocation=False,
                        position_weight=1, sector_weight=1, maximum_holding=2,
                        non_stage_allocation=1),
        capital=10000, start=panel.calendar[0], capture_execution_evidence=True,
        entry_filter=(lambda *_arguments: False) if reason == "entry_filter" else None,
    )
    evidence = result["execution_evidence"]
    rejected = [row for row in evidence["order_attempts"] if row["reason"] == reason]
    assert len(rejected) == 1
    assert rejected[0]["confirmation"] == 0
    purchases = [row for row in evidence["fills"] if row["side"] == "buy"]
    if reason == "opening_untradable":
        assert len(purchases) == 1 and purchases[0]["day"] == 2
    else:
        assert not purchases
        assert all(row["cash"] == 10000 for row in result["daily"])
