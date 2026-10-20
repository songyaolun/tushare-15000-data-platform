"""Bound research export checks while preserving native scores as replay authority."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .boosted_tree_scoring import BoostedTreeModel


def representative_rows(
    features: NDArray[np.floating[Any]], *, maximum_rows: int = 512,
) -> NDArray[np.int64]:
    """Select deterministic representative rows and observed finite/NaN boundaries.

    Input is a nonempty float matrix. Keep first/last rows, each column's finite
    min/max rows and first/last NaN rows, then evenly spaced rows up to the budget.
    Example: [[1],[NaN],[3]] with budget=3 selects [0,1,2]. A budget unable to keep
    every observed boundary fails instead of silently dropping boundary coverage.
    """
    if (features.ndim != 2 or not len(features) or features.shape[1] == 0
            or features.dtype.kind != "f" or np.isinf(features).any()
            or type(maximum_rows) is not int or maximum_rows <= 0):
        raise ValueError("invalid_portable_validation_matrix_or_budget")
    selected = {0, len(features) - 1}
    for column in features.T:
        finite = np.flatnonzero(np.isfinite(column))
        missing = np.flatnonzero(np.isnan(column))
        if len(finite):
            selected.update((int(finite[np.argmin(column[finite])]),
                             int(finite[np.argmax(column[finite])])))
        if len(missing):
            selected.update((int(missing[0]), int(missing[-1])))
    if len(selected) > maximum_rows:
        raise ValueError("portable_validation_budget_omits_boundaries")
    for row in np.linspace(0, len(features) - 1, min(maximum_rows, len(features)), dtype=int):
        if len(selected) == maximum_rows:
            break
        selected.add(int(row))
    return np.asarray(sorted(selected), dtype=np.int64)


def validate_portable_predictions(
    model: BoostedTreeModel, features: NDArray[np.float64],
    native_predictions: NDArray[np.floating[Any]], *, feature_names: tuple[str, ...],
    mode: str = "bounded", maximum_rows: int = 512,
) -> dict[str, Any]:
    """Compare exact native/portable values on a fixed subset or explicit full axis.

    Example: mode='bounded', maximum_rows=512 checks at most 512 existing rows and
    returns full_portable_verified=False. mode='full' checks all existing rows.
    Native predictions are read-only and remain authoritative for account replay.
    A mismatch raises before emitting a successful audit, without changing any scores.
    """
    if (features.ndim != 2 or not len(features) or features.shape[1] == 0
            or features.dtype.kind != "f" or np.isinf(features).any()
            or mode not in {"bounded", "full"} or native_predictions.shape != (len(features),)
            or not np.isfinite(native_predictions).all()):
        raise ValueError("invalid_portable_validation_mode_or_native_axis")
    selected = (np.arange(len(features), dtype=np.int64) if mode == "full"
                else representative_rows(features, maximum_rows=maximum_rows))
    actual = model.predict(features[selected], feature_names=feature_names)
    np.testing.assert_array_equal(actual, native_predictions[selected])
    return {"mode": mode, "checked_rows": len(selected), "total_rows": len(features),
            "row_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "maximum_absolute_error": 0.0, "full_portable_verified": mode == "full",
            "replay_score_source": "native_predictions"}
