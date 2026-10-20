"""Bound export work deterministically without weakening native score identity."""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.boosted_tree_scoring import BoostedTreeModel
from tushare_entitlement_hub.strategy_discovery.portable_validation import (
    representative_rows,
    validate_portable_predictions,
)


def test_representative_rows_keep_extremes_missing_boundaries_and_budget():
    """Sampling retains first/last, finite extremes and first/last NaNs without input mutation."""
    features = np.arange(2000.0).reshape(1000, 2)
    features[123, 0], features[876, 0] = np.nan, np.nan
    features[321, 1], features[654, 1] = -999, 99999
    before = features.copy()
    rows = representative_rows(features, maximum_rows=32)
    assert len(rows) == 32 and {0, 999, 123, 876, 321, 654} <= set(rows)
    np.testing.assert_array_equal(rows, representative_rows(features, maximum_rows=32))
    np.testing.assert_array_equal(features, before)


def test_all_missing_small_matrix_retains_each_available_row():
    """All-NaN columns need no finite extrema and a small axis is fully sampled."""
    np.testing.assert_array_equal(representative_rows(np.full((3, 2), np.nan)), [0, 1, 2])


@pytest.mark.parametrize("features,budget", [
    (np.array([1.0]), 4), (np.empty((0, 1)), 4), (np.empty((2, 0)), 4),
    (np.ones((2, 1), dtype=int), 4), (np.array([[np.inf]]), 4),
    (np.ones((2, 1)), 0), (np.ones((2, 1)), True),
])
def test_invalid_sampling_contract_rejects(features, budget):
    """Malformed matrices and invalid row budgets reject before selecting observations."""
    with pytest.raises(ValueError, match="matrix_or_budget"):
        representative_rows(features, maximum_rows=budget)


def test_too_small_budget_cannot_drop_nan_boundaries():
    """A row budget below the required boundary count fails rather than hiding unknowns."""
    with pytest.raises(ValueError, match="omits_boundaries"):
        representative_rows(np.array([[1.0], [np.nan], [2.0]]), maximum_rows=2)


@pytest.mark.parametrize("mode,expected", [("bounded", 32), ("full", 1000)])
def test_validation_mode_reports_scope_without_replacing_native_scores(mode, expected):
    """Bounded checks certify only their subset; explicit full checks visit the full axis."""
    features = np.arange(1000.0)[:, None]
    native = np.full(1000, 0.25)
    model = BoostedTreeModel(("value",), 0.25, ())
    audit = validate_portable_predictions(model, features, native, feature_names=("value",),
                                          mode=mode, maximum_rows=32)
    assert audit["checked_rows"] == expected and audit["total_rows"] == 1000
    assert audit["full_portable_verified"] is (mode == "full")
    assert audit["maximum_absolute_error"] == 0
    np.testing.assert_array_equal(native, np.full(1000, 0.25))


@pytest.mark.parametrize("mode,native", [
    ("unknown", np.array([0.25])), ("bounded", np.array([0.25, 0.25])),
    ("full", np.array([np.nan])),
])
def test_invalid_native_axis_or_mode_rejects(mode, native):
    """Unknown modes, wrong row counts and nonfinite native scores cannot pass export checks."""
    with pytest.raises(ValueError, match="mode_or_native_axis"):
        validate_portable_predictions(
            BoostedTreeModel(("value",), 0.25, ()), np.array([[1.0]]), native,
            feature_names=("value",), mode=mode,
        )


def test_mismatched_portable_values_fail_without_modifying_native():
    """An incorrect portable model raises and leaves the authoritative score vector intact."""
    native = np.array([0.25])
    with pytest.raises(AssertionError):
        validate_portable_predictions(BoostedTreeModel(("value",), 0.50, ()),
                                      np.array([[1.0]]), native, feature_names=("value",))
    assert native[0] == 0.25
