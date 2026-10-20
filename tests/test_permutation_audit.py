"""Daily missingness strata accounting without any model or network dependency."""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.permutation_audit import permutation_strata_audit


def test_daily_permutation_audit_separates_movement_from_value_change():
    """Scenario: finite swaps, singleton NaN and repeated NaNs; expect disjoint evidence."""
    original = np.array([[1., 2.], [2., 1.], [np.nan, np.nan],
                         [np.nan, np.nan], [np.nan, np.nan], [1., np.nan]])
    changed = original.copy()
    changed[:2] = changed[[1, 0]]
    rows = permutation_strata_audit(original, changed, np.array([0, 0, 0, 1, 1, 1]))
    assert rows[0] == {"day": 0, "rows": 3, "eligible_rows": 2, "singleton_rows": 1,
                       "all_missing_rows": 1, "frozen_missing_rows": 1,
                       "value_changed_rows": 2}
    assert rows[1] == {"day": 1, "rows": 3, "eligible_rows": 2, "singleton_rows": 1,
                       "all_missing_rows": 2, "frozen_missing_rows": 3,
                       "value_changed_rows": 0}


@pytest.mark.parametrize("original,changed,days", [
    (np.ones(2), np.ones(2), np.ones(2)),
    (np.ones((2, 1)), np.ones((3, 1)), np.ones(2)),
    (np.ones((2, 0)), np.ones((2, 0)), np.ones(2)),
    (np.ones((2, 1)), np.ones((2, 1)), np.ones(3)),
])
def test_permutation_audit_rejects_invalid_axes(original, changed, days):
    """Scenario: inconsistent dimensions; expect closed failure before aggregation."""
    with pytest.raises(ValueError, match="axes"):
        permutation_strata_audit(original, changed, days)


def test_permutation_audit_rejects_changed_missingness():
    """Scenario: a NaN becomes finite; expect failure, never a valid permutation audit."""
    with pytest.raises(ValueError, match="missingness"):
        permutation_strata_audit(np.array([[np.nan]]), np.array([[1.]]), np.array([0]))
