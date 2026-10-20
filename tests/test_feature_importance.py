"""Deterministic tree usage, rank ties and strategy-selection diagnostic fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.feature_importance import (
    daily_diagnostics,
    rank_correlation,
    ranked_selection,
    tree_usage,
)


def test_tree_usage_excludes_leaf_sentinels_and_retains_unused_features():
    """Scenario: two trees with leaf gain -1; expect only internal gains/counts."""
    nodes = np.array([(0, 0, 2.), (0, 1, 6.), (1, 0, -1.)],
                     dtype=[("is_leaf", "i4"), ("feature_idx", "i4"), ("gain", "f8")])
    model = SimpleNamespace(_predictors=[[SimpleNamespace(nodes=nodes)],
                                         [SimpleNamespace(nodes=nodes)]])
    result = tree_usage(model, ["first", "second", "unused"])
    assert result[0] == {"feature": "first", "gain": 4., "splits": 2,
                         "gain_share": .25, "split_share": .5}
    assert result[1]["gain"] == 12.
    assert result[2]["gain_share"] == result[2]["splits"] == 0


@pytest.mark.parametrize("index,gain", [(3, 1.), (0, np.nan)])
def test_tree_usage_rejects_malformed_nodes(index, gain):
    """Scenario: out-of-axis index or NaN gain; expect a closed failure."""
    nodes = np.array([(0, index, gain)],
                     dtype=[("is_leaf", "i4"), ("feature_idx", "i4"), ("gain", "f8")])
    with pytest.raises(ValueError, match="invalid internal"):
        tree_usage(SimpleNamespace(_predictors=[[SimpleNamespace(nodes=nodes)]]), ["x"])


def test_tree_usage_leaf_only_model_is_zero():
    """Scenario: a constant estimator; expect defined zero shares, without NaN."""
    nodes = np.array([(1, 0, -1.)],
                     dtype=[("is_leaf", "i4"), ("feature_idx", "i4"), ("gain", "f8")])
    result = tree_usage(SimpleNamespace(_predictors=[[SimpleNamespace(nodes=nodes)]]), ["x"])
    assert result[0]["gain_share"] == result[0]["split_share"] == 0


def test_selection_applies_common_after_sector_cap_without_refill():
    """Scenario: the highest score fails common; expect no replacement by same sector."""
    selected = ranked_selection(np.array([9., 8., 7.]), np.array([1, 1, 2]),
                                np.array([False, True, True]), count=2)
    assert selected.tolist() == [2]


def test_selection_stable_ties_unknown_sector_and_nan_scores():
    """Scenario: tied scores and two unknown sectors; expect security-order tie breaks."""
    selected = ranked_selection(np.array([9., 9., 8., 7., np.nan]),
                                np.array([np.nan, -2, 2, 3, 4]), np.ones(5, dtype=bool))
    assert selected.tolist() == [0, 2, 3]
    assert ranked_selection(np.array([9., 9.]), np.array([1, 1]),
                            np.ones(2, dtype=bool), sector_cap=False).tolist() == [0, 1]


@pytest.mark.parametrize("scores,sectors,common,count", [
    (np.ones((1, 2)), np.ones(2), np.ones(2), 1),
    (np.ones(2), np.ones(3), np.ones(2), 1),
    (np.ones(2), np.ones(2), np.ones(3), 1),
    (np.ones(2), np.ones(2), np.ones(2), 0),
])
def test_selection_rejects_bad_axes_or_count(scores, sectors, common, count):
    """Scenario: incompatible axes or nonpositive count; expect ValueError."""
    with pytest.raises(ValueError):
        ranked_selection(scores, sectors, common, count=count)


@pytest.mark.parametrize("left,right,expected", [
    ([1, 2, 3], [3, 2, 1], -1.), ([1, 1, 3], [2, 2, 3], 1.),
    ([1], [2], None), ([1, 1], [2, 3], None), ([1, 2], [3, 3], None),
])
def test_rank_correlation_ties_and_degenerate_days(left, right, expected):
    """Scenario: ties/constant days; expect average-rank Spearman or explicit None."""
    result = rank_correlation(np.array(left), np.array(right))
    assert result is None if expected is None else result == pytest.approx(expected)


def test_daily_diagnostics_maturity_and_empty_qualified_selection():
    """Scenario: one immature day; expect no label error, but prediction/rank diagnostics."""
    rows = daily_diagnostics(
        np.array([3., 2., 1., 2.]), np.array([1., 2., 3., 2.]), np.array([0, 0, 0, 1]),
        np.array([1, 1, 2, 1]), np.array([True, True, True, False]),
        np.array([1., np.nan, 1., np.nan]), np.array([True, False, True, False]),
    )
    assert rows[0]["baseline_mse"] == rows[0]["perturbed_mse"] == 2.
    assert rows[0]["mse_delta"] == 0
    assert rows[0]["prediction_mae"] == pytest.approx(4 / 3)
    assert rows[0]["strategy"]["removed"] == 1
    assert rows[0]["strategy"]["intersection"] == 1
    assert rows[1]["baseline_mse"] is rows[1]["mse_delta"] is None
    assert rows[1]["strategy"]["jaccard"] == 1.
