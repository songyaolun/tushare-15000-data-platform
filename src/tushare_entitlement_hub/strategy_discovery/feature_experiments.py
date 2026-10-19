"""Deterministic feature experiments on an immutable candidate axis."""

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray


def feature_cases(groups: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Enumerate every nonempty group subset once, without inspecting outcomes.

    Input ('event', 'quality') yields event, quality and event+quality. Names must
    be nonempty unique identifiers without '+', which is reserved for case names.
    """
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(not name or not name.replace("_", "").isalnum() for name in groups)
    ):
        raise ValueError("invalid feature groups")
    return {
        "+".join(subset): subset
        for length in range(1, len(groups) + 1)
        for subset in combinations(groups, length)
    }


def stratified_feature_placebo(
    features: NDArray[np.floating[Any]],
    days: NDArray[np.integer[Any]],
    sectors: NDArray[np.integer[Any]],
    *,
    seed: int = 20261019,
) -> tuple[NDArray[np.floating[Any]], dict[str, int]]:
    """Shuffle whole added-feature rows within date, sector and missingness strata.

    Original features and NaN masks stay untouched; a fixed seed makes the result
    reproducible. Example: two known rows from the same date/sector may exchange
    their paired values, but never exchange with an unknown row. Return a copy
    and counts of eligible, moved and unchanged rows. This is a diagnostic, not
    a significance test; event trajectories across dates are not preserved.
    """
    if (
        features.ndim != 2
        or not features.shape[1]
        or features.dtype.kind != "f"
        or days.shape != (len(features),)
        or sectors.shape != days.shape
        or days.dtype.kind not in "iu"
        or sectors.dtype.kind not in "iu"
        or np.isinf(features).any()
    ):
        raise ValueError("invalid placebo axes")
    strata = np.column_stack((days, sectors, np.isnan(features)))
    _, inverse = np.unique(strata, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    cuts = np.flatnonzero(np.diff(inverse[order])) + 1
    result = features.copy()
    random = np.random.default_rng(seed)
    eligible = moved = 0
    for indices in np.split(order, cuts):
        if len(indices) < 2:
            continue
        source = random.permutation(indices)
        result[indices] = features[source]
        eligible += len(indices)
        moved += int(np.count_nonzero(source != indices))
    return result, {
        "rows": len(features),
        "eligible": eligible,
        "moved": moved,
        "unchanged": len(features) - moved,
    }


def portable_numeric_model(model: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Export a fitted numeric HGB model using the existing portable v1 contract.

    Example: a numeric one-leaf predictor becomes one portable leaf and retains
    its already learning-rate-scaled value. Categorical trees fail closed. HGB
    infinity sentinels for missing-only splits become the largest finite float;
    callers must verify all candidate predictions against the fitted estimator.
    """
    trees = []
    for iteration in model._predictors:
        for tree in iteration:
            records = []
            for node in tree.nodes:
                if node["is_categorical"]:
                    raise ValueError("categorical model cannot use numeric export")
                threshold = float(node["num_threshold"])
                if np.isposinf(threshold):
                    threshold = float(np.finfo(float).max)
                records.append(
                    {
                        "feature_idx": int(node["feature_idx"]),
                        "numeric_threshold": threshold,
                        "missing_go_to_left": bool(node["missing_go_to_left"]),
                        "left": int(node["left"]),
                        "right": int(node["right"]),
                        "is_leaf": bool(node["is_leaf"]),
                        "value": float(node["value"]),
                    }
                )
            trees.append(records)
    return {
        "schema_version": 1,
        "feature_names": list(names),
        "baseline_prediction": float(model._baseline_prediction[0, 0]),
        "trees": trees,
    }
