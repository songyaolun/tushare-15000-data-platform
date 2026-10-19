"""Offline checks for exhaustive cases, matched placebos and portable exports."""

from types import SimpleNamespace

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.boosted_tree_scoring import BoostedTreeModel
from tushare_entitlement_hub.strategy_discovery.feature_experiments import (
    feature_cases,
    portable_numeric_model,
    stratified_feature_placebo,
)


def test_cases_include_every_subset_without_order_duplicates():
    """Three independent groups yield seven fixed cases and preserve column ordering."""
    cases = feature_cases(("event", "quality", "freefloat"))
    assert len(cases) == 7
    assert cases["event+quality"] == ("event", "quality")
    assert list(cases)[-1] == "event+quality+freefloat"


@pytest.mark.parametrize("groups", [(), ("a", "a"), ("a+b",), ("",), ("../a",)])
def test_invalid_group_names_fail_before_experiment(groups):
    """Empty, duplicate and path-like group identifiers cannot create ambiguous cases."""
    with pytest.raises(ValueError, match="groups"):
        feature_cases(groups)


def test_placebo_preserves_date_sector_missingness_and_column_pairs():
    """Shuffling changes matching rows only, retaining paired values and immutable inputs."""
    features = np.column_stack((np.arange(12.0), np.arange(12.0) + 100))
    features[10] = np.nan
    days = np.array([1] * 8 + [2] * 4)
    sectors = np.array([1] * 8 + [2, 2, 2, 3])
    before = features.copy()
    actual, audit = stratified_feature_placebo(features, days, sectors)
    repeated, _ = stratified_feature_placebo(features, days, sectors)
    np.testing.assert_array_equal(features, before)
    np.testing.assert_array_equal(actual, repeated)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(before))
    np.testing.assert_array_equal(actual[11], before[11])
    assert sorted(actual[:8, 0]) == list(range(8))
    np.testing.assert_array_equal(actual[:10, 1] - actual[:10, 0], np.full(10, 100))
    assert audit["eligible"] == 10 and audit["moved"] > 0
    assert audit["unchanged"] == 12 - audit["moved"]


@pytest.mark.parametrize("case", ["infinity", "shape", "days", "sector", "integer", "zero"])
def test_placebo_invalid_axes_fail_closed(case):
    """Wrong axes, empty columns, nonfloat features and infinity cannot enter a placebo."""
    values = np.ones((2, 1))
    days = sectors = np.ones(2, dtype=int)
    if case == "infinity":
        values[0, 0] = np.inf
    elif case == "shape":
        values = values[:, 0]
    elif case == "days":
        days = np.ones(3, dtype=int)
    elif case == "sector":
        sectors = sectors.astype(float)
    elif case == "integer":
        values = values.astype(int)
    else:
        values = np.empty((2, 0))
    with pytest.raises(ValueError, match="axes"):
        stratified_feature_placebo(values, days, sectors)


def test_portable_export_keeps_missing_split_and_leaf_values():
    """Missing-only positive-infinity split remains equivalent for finite and NaN inputs."""
    node = dict(
        feature_idx=0,
        num_threshold=np.inf,
        missing_go_to_left=False,
        left=1,
        right=2,
        is_leaf=False,
        value=0.0,
        is_categorical=False,
    )
    leaf = dict(node, num_threshold=0.0, left=0, right=0, is_leaf=True)
    nodes = [node, dict(leaf, value=0.2), dict(leaf, value=-0.3)]
    model = SimpleNamespace(
        _predictors=[[SimpleNamespace(nodes=nodes)]], _baseline_prediction=np.array([[0.1]])
    )
    payload = portable_numeric_model(model, ("one",))
    prediction = BoostedTreeModel.from_dict(payload).predict(
        np.array([[2.0], [np.nan]]), feature_names=("one",)
    )
    np.testing.assert_allclose(prediction, [0.3, -0.2])
    nodes[0]["is_categorical"] = True
    with pytest.raises(ValueError, match="categorical"):
        portable_numeric_model(model, ("one",))
