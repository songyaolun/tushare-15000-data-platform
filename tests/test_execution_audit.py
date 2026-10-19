"""Independent auditor failure cases retain invalid observations and full denominators."""

import math

import pytest

from tushare_entitlement_hub.regime_research.execution_audit import audit_capacity, read_rows


@pytest.mark.parametrize("changes,expected", [
    ({}, []),
    ({"raw_amount": "nan"}, ["unknown_or_nonpositive_daily_amount"]),
    ({"raw_amount": "0"}, ["unknown_or_nonpositive_daily_amount"]),
    ({"notional": "0"}, ["invalid_fill_notional"]),
    ({"price": "nan"}, ["unknown_or_nonpositive_price"]),
    ({"price": "12"}, ["fill_outside_observed_daily_range"]),
    ({"raw_low": "12"}, ["fill_outside_observed_daily_range"]),
])
def test_capacity_retains_invalid_fills(changes, expected):
    """Unknown amounts and impossible prices remain in observations and the original count."""
    row = dict(sequence="0", date="20240101", ts_code="600000.SH", side="buy",
               notional="1000", raw_amount="10", raw_low="9", raw_high="11", price="10")
    result = audit_capacity([row | changes])
    assert result["fill_count"] == 1
    assert result["observations"][0]["anomalies"] == expected
    if not expected:
        assert result["maximum_fill_participation"] == .1
    assert result["known_amount_fill_count"] + result["unknown_amount_fill_count"] == 1


def test_conflicting_amounts_and_empty_csv(tmp_path):
    """Same-stock same-day conflicts cannot establish aggregate capacity; empty file stays empty."""
    row = dict(sequence="0", date="20240101", ts_code="600000.SH", side="buy",
               notional="1000", raw_amount="10", raw_low="9", raw_high="11", price="10")
    result = audit_capacity([row, row | {"raw_amount": "11", "sequence": "1"}])
    assert math.isnan(result["security_days"][0]["daily_amount_thousand_cny"])
    assert result["maximum_security_day_participation"] is None
    assert audit_capacity([])["maximum_fill_participation"] is None
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert read_rows(path) == []
    with pytest.raises(FileNotFoundError):
        read_rows(tmp_path / "missing.csv")
