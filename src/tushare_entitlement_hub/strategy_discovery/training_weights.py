"""Auditable trading-session decay on top of frozen daily-equal training weights."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def daily_equal_training_weights(
    signal_days: NDArray[np.int64],
    exit_days: NDArray[np.float64],
    calendar: list[str],
    *,
    cutoff: str,
    half_life_sessions: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Validate mature fit labels and return mean-one daily-equal, optionally decayed weights.

    Each row first receives 1 / its signal day's fit-row count. Optional decay is
    2 ** (-(cutoff_index - signal_day) / half_life_sessions), then all weights are
    normalized to mean one, preserving total weight and regularization scale.
    All supplied labels must have exited by cutoff; invalid rows fail closed and
    are never silently dropped. Calendar distance includes sessions with no fit rows.
    The audit includes row/day effective sample sizes and annual total weight shares.

    Example: days=[0,0,1], exits=[1,1,1], two-day calendar and no half-life gives
    weights=[.75,.75,1.5]; a one-session half-life gives [.5,.5,2.]. These remain
    full-sample fitting weights, not an out-of-sample or historical-PIT certificate.
    """
    if (
        not calendar
        or calendar != sorted(set(calendar))
        or any(len(day) != 8 or not day.isdigit() for day in calendar)
        or cutoff not in calendar
    ):
        raise ValueError("invalid_training_calendar_or_cutoff")
    if half_life_sessions is not None and (
        type(half_life_sessions) is not int or half_life_sessions <= 0
    ):
        raise ValueError("invalid_training_half_life")
    signal_days = np.asarray(signal_days)
    exit_days = np.asarray(exit_days)
    cutoff_index = calendar.index(cutoff)
    if (
        signal_days.ndim != 1
        or signal_days.size == 0
        or not np.issubdtype(signal_days.dtype, np.integer)
        or exit_days.shape != signal_days.shape
        or not np.isfinite(exit_days).all()
        or np.any(exit_days != np.floor(exit_days))
        or np.any(signal_days < 0)
        or np.any(signal_days > cutoff_index)
        or np.any(exit_days < signal_days)
        or np.any(exit_days > cutoff_index)
    ):
        raise ValueError("invalid_or_immature_training_labels")
    days, inverse, counts = np.unique(signal_days, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].astype(float)
    if half_life_sessions is not None:
        weights *= np.exp2(-(cutoff_index - signal_days) / half_life_sessions)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("nonpositive_training_weights")
    weights /= weights.mean()
    day_weights = np.bincount(inverse, weights=weights)
    years = np.asarray([calendar[int(day)][:4] for day in signal_days])
    annual = {
        year: {
            "rows": int(np.sum(years == year)),
            "weight_share": float(weights[years == year].sum() / weights.sum()),
        }
        for year in np.unique(years)
    }
    return weights, {
        "method": "daily_equal" if half_life_sessions is None else "daily_equal_exponential",
        "half_life_sessions": half_life_sessions,
        "cutoff": cutoff,
        "cutoff_index": cutoff_index,
        "fit_rows": len(weights),
        "fit_signal_days": len(days),
        "first_signal_date": calendar[int(days[0])],
        "last_signal_date": calendar[int(days[-1])],
        "last_label_exit_date": calendar[int(exit_days.max())],
        "immature_fit_rows": 0,
        "weight_mean": float(weights.mean()),
        "weight_sum": float(weights.sum()),
        "weight_minimum": float(weights.min()),
        "weight_maximum": float(weights.max()),
        "row_effective_sample_size": float(weights.sum() ** 2 / np.dot(weights, weights)),
        "day_effective_sample_size": float(
            day_weights.sum() ** 2 / np.dot(day_weights, day_weights)
        ),
        "annual_weights": annual,
    }
