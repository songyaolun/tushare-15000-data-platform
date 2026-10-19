"""Historical candidate filters preserve original rows and never read future liquidity."""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.liquidity import LiquidityRule
from tushare_entitlement_hub.strategy_discovery.candidate_liquidity import filter_signal_liquidity


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, True])
def test_invalid_threshold_fails_closed(threshold):
    """Nonpositive, nonfinite and boolean monetary thresholds must not select candidates."""
    with pytest.raises(ValueError, match="invalid_signal_liquidity_filter"):
        filter_signal_liquidity(np.ones((3, 2)), np.ones((3, 2), dtype=bool),
                                threshold, LiquidityRule())


@pytest.mark.parametrize("amounts,eligible", [
    (np.ones(3), np.ones(3, dtype=bool)),
    (np.ones((3, 2)), np.ones((3, 1), dtype=bool)),
    (np.ones((3, 2)), np.ones((3, 2))),
])
def test_invalid_axes_and_nonboolean_mask_are_rejected(amounts, eligible):
    """Shape mismatch and numeric eligibility masks cannot silently alter the universe."""
    with pytest.raises(ValueError, match="invalid_signal_liquidity_filter"):
        filter_signal_liquidity(amounts, eligible, 100, LiquidityRule())


def test_every_original_candidate_has_reason_and_current_close_is_visible():
    """Insufficient, missing, negative, zero, small and accepted rows retain distinct reasons."""
    amounts = np.full((4, 6), 10000.)
    amounts[1, 0] = np.nan
    amounts[1, 1] = -1
    amounts[2, 2] = 0
    amounts[2, 3] = 5000
    eligible = np.zeros((4, 6), dtype=bool)
    eligible[0, 0] = True
    eligible[2] = True
    selected, records = filter_signal_liquidity(amounts, eligible, 100000, LiquidityRule(.01, 3))
    assert [row["reason"] for row in records] == [
        "insufficient_history", "missing_history", "negative_history", "zero_reference",
        "below_threshold", "passed", "passed",
    ]
    assert selected[2].tolist() == [False, False, False, False, True, True]
    assert len(records) == int(eligible.sum())
    amounts[3] = 1e12
    future_selected, future_records = filter_signal_liquidity(
        amounts, eligible, 100000, LiquidityRule(.01, 3),
    )
    assert np.array_equal(selected, future_selected)
    assert records == future_records
    assert records[1]["missing_observations"] == 1
    assert records[2]["negative_observations"] == 1
    assert records[-1]["history_end"] == 2
    assert records[-1]["reference_notional"] == 100000


def test_empty_candidate_mask_stays_empty():
    """A day without original candidates cannot acquire candidates from the liquidity filter."""
    selected, records = filter_signal_liquidity(
        np.ones((3, 2)), np.zeros((3, 2), dtype=bool), 100, LiquidityRule(.01, 1),
    )
    assert not selected.any() and records == []
