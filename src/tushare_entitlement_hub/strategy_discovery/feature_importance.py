"""Frozen-tree usage and bounded prediction diagnostics; never fit an estimator."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def tree_usage(model: Any, names: list[str]) -> list[dict[str, Any]]:
    """Sum internal-node gain and split counts, retaining within-model shares.

    Example: one split on feature 0 with gain 2 yields gain=2 and gain_share=1.
    Leaf sentinels are excluded. Gain is not multiplied by sample count and is
    not a linear coefficient; differently weighted fits have different scales.
    """
    gains = np.zeros(len(names))
    counts = np.zeros(len(names), dtype=np.int64)
    for iteration in model._predictors:
        for tree in iteration:
            nodes = tree.nodes[tree.nodes["is_leaf"] == 0]
            indices = nodes["feature_idx"].astype(int)
            values = nodes["gain"]
            if np.any(indices >= len(names)) or not np.isfinite(values).all():
                raise ValueError("invalid internal tree nodes")
            np.add.at(gains, indices, values)
            np.add.at(counts, indices, 1)
    return [
        {"feature": name, "gain": float(gains[index]), "splits": int(counts[index]),
         "gain_share": float(gains[index] / gains.sum()) if gains.sum() else 0.0,
         "split_share": float(counts[index] / counts.sum()) if counts.sum() else 0.0}
        for index, name in enumerate(names)
    ]


def ranked_selection(
    scores: NDArray[Any], sectors: NDArray[Any], common: NDArray[Any],
    *, count: int = 15, sector_cap: bool = True,
) -> NDArray[np.int64]:
    """Select stable score ranks, then apply common eligibility without refill.

    Inputs must already follow security-column order. Example: scores [9,8,7],
    sectors [1,1,2], common [False,True,True], count=2 yields [2]. Unknown sectors
    form one -1 bucket, matching the frozen strategy selector.
    """
    if scores.ndim != 1 or sectors.shape != scores.shape or common.shape != scores.shape:
        raise ValueError("invalid selection axes")
    if count < 1:
        raise ValueError("positive selection count required")
    ranked = np.flatnonzero(np.isfinite(scores))
    ranked = ranked[np.argsort(-scores[ranked], kind="stable")]
    selected: list[int] = []
    seen: set[int] = set()
    for index in ranked:
        sector = int(sectors[index]) if np.isfinite(sectors[index]) and sectors[index] >= 0 else -1
        if sector_cap and sector in seen:
            continue
        selected.append(int(index))
        seen.add(sector)
        if len(selected) == count:
            break
    return np.asarray([index for index in selected if common[index]], dtype=np.int64)


def rank_correlation(left: NDArray[Any], right: NDArray[Any]) -> float | None:
    """Return Spearman correlation with average ranks for ties, or None if constant.

    Example: [1,2,3] versus [3,2,1] returns -1; a one-row day returns None.
    """
    ranks = []
    for values in (left, right):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks.append((np.cumsum(counts) - (counts + 1) / 2)[inverse])
    if len(left) < 2 or any(np.std(values) == 0 for values in ranks):
        return None
    return float(np.corrcoef(ranks)[0, 1])


def daily_diagnostics(
    baseline: NDArray[Any], perturbed: NDArray[Any], days: NDArray[Any],
    sectors: NDArray[Any], common: NDArray[Any], targets: NDArray[Any],
    mature: NDArray[Any],
) -> list[dict[str, Any]]:
    """Compare daily predictions, mature-label MSE and exact strategy selection.

    Example: unchanged predictions return zero MAE/MSE delta and zero removed
    names. MSE uses only caller-pinned mature rows; selection uses all candidates.
    Raw ranks and sector-capped/common-qualified selections are kept separate.
    """
    rows = []
    for day in np.unique(days):
        mask = days == day
        before, after = baseline[mask], perturbed[mask]
        valid = mature[mask]
        target = targets[mask][valid]
        before_mse = float(np.mean((before[valid] - target) ** 2)) if valid.any() else None
        after_mse = float(np.mean((after[valid] - target) ** 2)) if valid.any() else None
        row: dict[str, Any] = {
            "day": int(day), "rows": int(mask.sum()), "mature_rows": int(valid.sum()),
            "prediction_mae": float(np.mean(np.abs(after - before))),
            "rank_correlation": rank_correlation(before, after),
            "baseline_mse": before_mse, "perturbed_mse": after_mse,
            "mse_delta": cast(float, after_mse) - before_mse if before_mse is not None else None,
        }
        for label, capped in (("raw", False), ("strategy", True)):
            eligibility = common[mask] if capped else np.ones(mask.sum(), dtype=bool)
            original = ranked_selection(before, sectors[mask], eligibility, sector_cap=capped)
            changed = ranked_selection(after, sectors[mask], eligibility, sector_cap=capped)
            intersection = len(set(original) & set(changed))
            union = len(set(original) | set(changed))
            row[label] = {"before": original.tolist(), "after": changed.tolist(),
                          "intersection": intersection, "removed": len(original) - intersection,
                          "jaccard": intersection / union if union else 1.0}
        rows.append(row)
    return rows
