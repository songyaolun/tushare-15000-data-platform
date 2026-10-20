"""Daily observability for same-date, missingness-preserving group permutations."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def permutation_strata_audit(
    original: NDArray[Any], changed: NDArray[Any], days: NDArray[Any],
) -> list[dict[str, int]]:
    """Count daily singleton/all-missing strata and actually changed feature rows.

    Example: two finite rows exchange and one NaN row stays put: rows=3,
    eligible_rows=2, singleton_rows=1, all_missing_rows=1, value_changed_rows=2.
    Eligible denotes movable row indices, not necessarily different values.
    Frozen-missing rows are the union of singleton strata and all-missing rows.
    """
    if (original.ndim != 2 or original.shape != changed.shape or not original.shape[1]
            or days.shape != (len(original),)):
        raise ValueError("invalid permutation audit axes")
    missing = np.isnan(original)
    if not np.array_equal(missing, np.isnan(changed)):
        raise ValueError("permutation changed missingness")
    _, inverse, counts = np.unique(np.column_stack((days, missing)), axis=0,
                                  return_inverse=True, return_counts=True)
    singleton = counts[inverse] == 1
    all_missing = missing.all(axis=1)
    value_changed = np.any(~((original == changed) | missing), axis=1)
    return [
        {"day": int(day), "rows": int((days == day).sum()),
         "eligible_rows": int(((days == day) & ~singleton).sum()),
         "singleton_rows": int(((days == day) & singleton).sum()),
         "all_missing_rows": int(((days == day) & all_missing).sum()),
         "frozen_missing_rows": int(((days == day) & (singleton | all_missing)).sum()),
         "value_changed_rows": int(((days == day) & value_changed).sum())}
        for day in np.unique(days)
    ]
