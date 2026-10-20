"""Offline contracts for complete denominators and dated candidate mapping."""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.industry_crowding import (
    candidate_crowding_features,
    complete_industry_amount_shares,
)


def test_complete_denominator_preserves_unknown_and_zero() -> None:
    """A missing industry closes the whole date; a genuine zero amount stays zero."""
    amounts = np.array([[1, 3], [1, np.nan], [0, 0], [0, 2], [-1, 2], [1, np.inf]])
    expected = np.array(
        [
            [0.25, 0.75],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0, 1],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )
    np.testing.assert_equal(complete_industry_amount_shares(amounts), expected)
    np.testing.assert_equal(amounts[0], [1, 3])


@pytest.mark.parametrize("shape", [(3,), (2, 0)])
def test_invalid_amount_axes(shape: tuple[int, ...]) -> None:
    """A scalar industry list or empty catalog cannot define an auditable denominator."""
    with pytest.raises(ValueError, match="nonempty catalog"):
        complete_industry_amount_shares(np.zeros(shape))


def test_mapping_unknown_membership_changes_and_prefix() -> None:
    """Only a fully known stable industry interval yields change; future rows have no effect."""
    percentiles = np.column_stack((np.arange(9) / 10, np.ones(9)))
    sectors = np.zeros((9, 5))
    sectors[0, 1] = 1
    sectors[5, 2:] = [np.nan, -1, 0.5]
    days = np.array([5, 5, 5, 5, 5, 0, 8])
    securities = np.array([0, 1, 2, 3, 4, 0, 0])
    result = candidate_crowding_features(percentiles, sectors, days, securities)
    np.testing.assert_allclose(result[:2], [[0.5, 0.5], [0.5, np.nan]], equal_nan=True)
    assert np.isnan(result[2:5]).all()
    np.testing.assert_allclose(result[5:], [[0, np.nan], [0.8, 0.5]], equal_nan=True)
    prefix = candidate_crowding_features(percentiles[:6], sectors[:6], days[:6], securities[:6])
    np.testing.assert_equal(prefix, result[:6])
    percentiles[2, 0] = np.nan
    assert np.isnan(candidate_crowding_features(percentiles, sectors, days, securities)[0, 1])


@pytest.mark.parametrize("case", ["date", "security", "dtype", "infinity", "range", "lag"])
def test_invalid_mapping_contract(case: str) -> None:
    """Invalid axes and impossible percentile values fail before any candidate is scored."""
    percentiles = np.ones((2, 1))
    sectors = np.zeros((2, 1))
    days, securities, lag = np.array([0]), np.array([0]), 5
    if case == "date":
        days = np.array([-1])
    elif case == "security":
        securities = np.array([1])
    elif case == "dtype":
        days = np.array([0.5])
    elif case == "infinity":
        percentiles[0, 0] = np.inf
    elif case == "range":
        percentiles[0, 0] = 1.1
    else:
        lag = 0
    with pytest.raises(ValueError, match="invalid crowding"):
        candidate_crowding_features(percentiles, sectors, days, securities, change_sessions=lag)
