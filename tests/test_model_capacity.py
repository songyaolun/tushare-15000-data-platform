"""Research capacity configuration must preserve labels and non-capacity fit settings."""

from types import MappingProxyType

import pytest

from tushare_entitlement_hub.strategy_discovery.model_capacity import resolve_model_capacity


@pytest.mark.parametrize("overrides", [None, {}])
def test_omitted_capacity_returns_unchanged_copy(overrides):
    """No override returns an equal independent mapping; baseline configuration stays intact."""
    baseline = {"max_iter": 1600, "loss": "squared_error"}
    actual = resolve_model_capacity(baseline, overrides)
    assert actual == baseline and actual is not baseline


def test_capacity_overrides_leave_original_and_other_settings_unchanged():
    """Three allowed integer overrides change capacity only and support read-only mappings."""
    baseline = {"max_iter": 1600, "max_depth": 5, "max_leaf_nodes": 31, "random_state": 7}
    overrides = MappingProxyType({"max_iter": 3200, "max_depth": 6, "max_leaf_nodes": 63})
    assert resolve_model_capacity(baseline, overrides) == dict(overrides) | {"random_state": 7}
    assert baseline == {"max_iter": 1600, "max_depth": 5, "max_leaf_nodes": 31, "random_state": 7}


@pytest.mark.parametrize("overrides", [[], 1, "max_iter=3200", False])
def test_capacity_rejects_non_mapping_configuration(overrides):
    """Malformed parameter blocks fail before fitting instead of silently retaining defaults."""
    with pytest.raises(ValueError, match="invalid_model_capacity_overrides"):
        resolve_model_capacity({}, overrides)


@pytest.mark.parametrize(
    "overrides",
    [{"loss": "absolute_error"}, {"max_iters": 3200}, {"max_iter": True}, {"max_depth": 6.0}],
)
def test_capacity_rejects_unregistered_names_and_non_integer_values(overrides):
    """Unknown keys, target changes and integer-like bool/float values fail closed."""
    with pytest.raises(ValueError, match="invalid_model_capacity_parameter"):
        resolve_model_capacity({}, overrides)


@pytest.mark.parametrize("overrides", [{"max_iter": 0}, {"max_depth": -1}, {"max_leaf_nodes": 1}])
def test_capacity_enforces_each_parameter_lower_bound(overrides):
    """Zero rounds, negative depth and a one-leaf configuration are rejected."""
    with pytest.raises(ValueError, match="invalid_model_capacity_value"):
        resolve_model_capacity({}, overrides)


def test_capacity_accepts_exact_lower_bounds():
    """The documented integer boundaries are accepted without coercion or clipping."""
    overrides = {"max_iter": 1, "max_depth": 1, "max_leaf_nodes": 2}
    assert resolve_model_capacity({}, overrides) == overrides
