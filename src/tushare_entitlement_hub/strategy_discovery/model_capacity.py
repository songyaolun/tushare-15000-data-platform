"""Resolve explicit research tree-capacity overrides without changing other fit settings."""

from collections.abc import Mapping
from typing import Any


def resolve_model_capacity(
    baseline_parameters: Mapping[str, Any], overrides: object = None
) -> dict[str, Any]:
    """Copy baseline parameters and apply only predeclared positive tree-size limits.

    Example: baseline {'max_iter': 1600, 'loss': 'squared_error'} and overrides
    {'max_iter': 3200} return {'max_iter': 3200, 'loss': 'squared_error'} while
    leaving the input untouched. None returns an unchanged copy. Iterations and
    depth require integers >=1; leaf count requires >=2. Other settings, bools,
    floating-point values and non-mappings fail closed, including misspelled keys.
    This helper does not select parameters from outcomes or retrain a model.
    """
    result = dict(baseline_parameters)
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise ValueError("invalid_model_capacity_overrides")
    minimum_values = {"max_iter": 1, "max_depth": 1, "max_leaf_nodes": 2}
    for name, value in overrides.items():
        if name not in minimum_values or type(value) is not int:
            raise ValueError("invalid_model_capacity_parameter")
        if value < minimum_values[name]:
            raise ValueError("invalid_model_capacity_value")
        result[name] = value
    return result
