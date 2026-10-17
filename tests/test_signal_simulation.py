"""Offline execution tests use deterministic synthetic quotes only."""

import pytest

from tushare_entitlement_hub.signal_research.simulation import (
    SimulationConfiguration,
    _can_fill,
    _day,
    _fill_price,
    _number,
    _returns,
    simulate_signal_trades,
)


def bar(day: str, **changes: object) -> dict[str, object]:
    """Build a synthetic valid daily bar; input day, output quote mapping.

    Example: ``bar('2023-08-28')`` produces an active 10-yuan quote.
    """
    return {
        "ts_code": "000001.SZ",
        "trade_date": day,
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "vol": 100.0,
        "amount": 1000.0,
        "price_limits_exempt": not ("up_limit" in changes or "down_limit" in changes),
        "adj_factor": 1.0,
        "normalized_histogram": 0.02,
        **changes,
    }


def signal(day: str = "2023-08-24", **changes: object) -> dict[str, object]:
    """Build a confirmed signal; input day, output experiment/security record.

    Example: ``signal()`` confirms the default experiment on August 24.
    """
    return {"ts_code": "000001.SZ", "confirmation_date": day, "experiment_id": "test", **changes}


@pytest.mark.parametrize(
    "changes",
    [
        {"commission_rate": -1},
        {"commission_rate": float("nan")},
        {"adverse_slippage_rate": 1},
        {"maximum_holding_sessions": 0},
        {"price_stop_loss_fraction": 0},
        {"price_stop_loss_fraction": float("inf")},
    ],
)
def test_invalid_configuration(changes: dict[str, object]) -> None:
    """Scenario: invalid execution parameters; expected: fail before simulation."""
    with pytest.raises(ValueError):
        SimulationConfiguration(**changes)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (True, None),
        ("bad", None),
        ("inf", None),
        ("12.5", 12.5),
    ],
)
def test_number(value: object, expected: float | None) -> None:
    """Scenario: malformed/nonfinite inputs; expected: unavailable or finite value."""
    assert _number(value) == expected


def test_day() -> None:
    """Scenario: provider/ISO/invalid dates; expected: normalize or reject."""
    assert _day("20230828") == _day("2023-08-28") == "2023-08-28"
    with pytest.raises(ValueError):
        _day("bad")


@pytest.mark.parametrize(
    "changes, side",
    [
        ({"suspended": True}, "buy"),
        ({"tradable": False}, "buy"),
        ({"open": None}, "buy"),
        ({"close": 0}, "buy"),
        ({"high": 9}, "buy"),
        ({"low": 11}, "buy"),
        ({"vol": 0}, "buy"),
        ({"amount": None}, "sell"),
        ({"up_limit": 10}, "buy"),
        ({"down_limit": 10}, "sell"),
        ({"price_limits_exempt": False}, "buy"),
        ({"price_limits_exempt": False, "down_limit": 0}, "sell"),
        ({"price_limits_exempt": False, "up_limit": float("nan")}, "buy"),
    ],
)
def test_fill_fails_closed(changes: dict[str, object], side: str) -> None:
    """Scenario: suspension, malformed bar, missing activity or limit; expected: no fill."""
    assert not _can_fill(bar("2023-08-25", **changes), side)


def test_fill_slippage_bounds() -> None:
    """Scenario: adverse slip reaches limits; expected: bounded valid execution price."""
    quote = bar("2023-08-25", up_limit=10.01, down_limit=9.99)
    assumptions = SimulationConfiguration(adverse_slippage_rate=0.1)
    assert _can_fill(quote, "buy")
    assert _fill_price(quote, "buy", assumptions) == 10.01
    assert _fill_price(quote, "sell", assumptions) == 9.99
    unrestricted = bar("2023-08-25", up_limit=10.01, price_limits_exempt=True)
    assert _fill_price(unrestricted, "buy", assumptions) == 11


def test_missing_limit_blocks_both_sides_with_reasons() -> None:
    """场景：买卖方向限价缺失；预期：等待证据，阻塞次数可审计，显式豁免才成交。"""
    quotes = [
        bar("2023-08-25", price_limits_exempt=False),
        bar("2023-08-28", up_limit=11, normalized_histogram=-0.003),
        bar("2023-08-29", up_limit=11),
        bar("2023-08-30", down_limit=9),
    ]
    result = simulate_signal_trades(quotes, [signal()])[0]
    assert result["entry_date"] == "2023-08-28"
    assert result["exit_date"] == "2023-08-30"
    assert result["entry_blocked_reason_counts"] == {
        "missing_or_invalid_directional_price_limit": 1,
    }
    assert result["exit_blocked_reason_counts"] == {
        "missing_or_invalid_directional_price_limit": 1,
    }


def test_next_open_t_plus_one_and_adjustment() -> None:
    """Scenario: green on entry day plus split; expected: next-day exit and zero price return."""
    quotes = [
        bar("2023-08-24", normalized_histogram=-1),
        bar("2023-08-25", normalized_histogram=-0.02),
        bar("2023-08-28", open=5, high=5, low=5, close=5, adj_factor=2),
    ]
    result = simulate_signal_trades(
        quotes, [signal()], SimulationConfiguration(adverse_slippage_rate=0)
    )[0]
    assert result["entry_date"] == "2023-08-25"
    assert result["exit_date"] == "2023-08-28"
    assert result["holding_sessions"] == 1
    assert result["provider_adjusted_price_return"] == 0
    assert result["stamp_tax_rate"] == 0.0005
    assert result["fee_adjusted_research_return"] == pytest.approx(-0.0011)
    assert "not_dividend_cash" in str(result["return_semantics"])
    assert quotes[0]["trade_date"] == "2023-08-24"


def test_pending_entry_exit_and_overlapping_signals() -> None:
    """Scenario: blocked opens and overlap; expected: delayed fills and recorded skips."""
    quotes = [
        bar("2023-08-25", suspended=True),
        bar("2023-08-28", up_limit=10),
        bar("2023-08-29", normalized_histogram=-0.02),
        bar("2023-08-30", down_limit=10),
        bar("2023-08-31", suspended=True),
        bar("2023-09-01"),
        bar("2023-09-04"),
    ]
    results = simulate_signal_trades(
        quotes,
        [
            signal(),
            signal("2023-08-28"),
            signal("2023-09-01"),
            signal("2023-08-28", experiment_id="independent"),
        ],
    )
    assert results[0]["entry_date"] == "2023-08-29"
    assert results[0]["exit_date"] == "2023-09-01"
    assert results[1]["status"] == "skipped"
    assert results[2]["status"] == "closed"
    assert results[3]["entry_date"] == "2023-09-04"


def test_future_truncation_and_pending_censoring() -> None:
    """Scenario: remove future exit quote; expected: unchanged entry and censored open return."""
    quotes = [bar("2023-08-25", normalized_histogram=-0.02), bar("2023-08-28")]
    full = simulate_signal_trades(quotes, [signal()])[0]
    truncated = simulate_signal_trades(quotes[:1], [signal(), signal("2023-08-25")])
    assert truncated[0]["entry_price"] == full["entry_price"]
    assert truncated[0]["status"] == "open"
    assert truncated[0]["exit_date"] is None
    assert truncated[0]["provider_adjusted_price_return"] is None
    assert truncated[0]["censor_reason"] == "no_eligible_future_exit_quote"
    assert truncated[1]["status"] == "skipped"
    empty = simulate_signal_trades([], [signal(), signal("2023-08-25")])
    assert empty[0]["status"] == "censored"
    assert empty[1]["status"] == "skipped"


def test_time_cap_and_default_no_stop() -> None:
    """Scenario: falling raw prices without configured stop; expected: only time-cap exit."""
    quotes = [
        bar("2023-08-25"),
        bar("2023-08-28", open=5, high=5, low=5, close=5),
        bar("2023-08-29"),
    ]
    result = simulate_signal_trades(
        quotes, [signal()], SimulationConfiguration(maximum_holding_sessions=2)
    )[0]
    assert result["exit_reason"] == "research_time_cap"
    assert result["exit_trigger_date"] == "2023-08-28"
    assert result["exit_date"] == "2023-08-29"
    assert (
        simulate_signal_trades(quotes, [signal()])[0]["censor_reason"] == "no_exit_trigger_observed"
    )


def test_optional_stop_and_missing_factor() -> None:
    """Scenario: configured close stop, missing factor; expected: factor-aware fail-closed stop."""
    quotes = [
        bar("2023-08-25"),
        bar("2023-08-28", open=8, high=8, low=8, close=8),
        bar("2023-08-29"),
    ]
    configuration = SimulationConfiguration(price_stop_loss_fraction=0.1)
    result = simulate_signal_trades(quotes, [signal()], configuration)[0]
    assert result["exit_reason"] == "configured_close_observed_price_stop"
    quotes[1]["adj_factor"] = None
    assert simulate_signal_trades(quotes, [signal()], configuration)[0]["status"] == "open"
    quotes[1].update(suspended=True, close=None)
    assert simulate_signal_trades(quotes, [signal()], configuration)[0]["status"] == "open"


@pytest.mark.parametrize("factor", [None, 0, -1])
def test_missing_return_factor_and_old_tax(factor: object) -> None:
    """Scenario: invalid factor and pre-change tax date; expected: no fabricated price return."""
    result = _returns(
        bar("2023-08-24", adj_factor=factor), bar("2023-08-25"), 10, 10, SimulationConfiguration()
    )
    assert result["stamp_tax_rate"] == 0.001
    assert result["provider_adjusted_price_return"] is None
    assert result["return_unavailable_reason"] == "missing_or_invalid_adjustment_factor"


def test_alias_default_experiment_duplicate_and_missing_histogram() -> None:
    """Scenario: aliases and absent histogram; expected: stable API, duplicate rejection."""
    results = simulate_signal_trades(
        [bar("20230825", normalized_histogram=None)],
        [{"ts_code": "000001.SZ", "signal_date": "20230824"}],
    )
    assert results[0]["experiment_id"] == "default"
    assert results[0]["status"] == "open"
    with pytest.raises(ValueError, match="Duplicate"):
        simulate_signal_trades([bar("2023-08-25"), bar("20230825")], [signal()])
