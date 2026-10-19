"""Signal-close liquidity eligibility before sector Top10; no account or future inputs."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.liquidity import LiquidityRule, historical_notional_limit


def filter_signal_liquidity(
    amounts: NDArray[np.float64], eligible: NDArray[np.bool_],
    minimum_notional: float, rule: LiquidityRule,
) -> tuple[NDArray[np.bool_], list[dict[str, Any]]]:
    """Filter original candidates using a fully known signal-close history window.

    For signal day 19, twenty values through day 19 of 10000 thousand CNY yield
    a 100000 CNY next-session reference. A threshold of 200000 rejects that row.
    The current execution session is never read; no NAV or account state enters.
    Return a fresh subset mask and one record for every original eligible row,
    including missing/negative/zero evidence instead of removing its denominator.
    """
    if (amounts.ndim != 2 or amounts.shape != eligible.shape or eligible.dtype != bool
            or isinstance(minimum_notional, bool) or not np.isfinite(minimum_notional)
            or minimum_notional <= 0):
        raise ValueError("invalid_signal_liquidity_filter")
    filtered = np.zeros(eligible.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    for day, security in zip(*np.nonzero(eligible), strict=True):
        start = int(day) + 1 - rule.lookback_sessions
        history = amounts[max(0, start):day + 1, security]
        missing = int((~np.isfinite(history)).sum())
        negative = int((history < 0).sum())
        reference = historical_notional_limit(amounts, int(day) + 1, int(security), rule)
        if start < 0:
            reason = "insufficient_history"
        elif missing:
            reason = "missing_history"
        elif negative:
            reason = "negative_history"
        elif reference == 0:
            reason = "zero_reference"
        elif reference < minimum_notional:
            reason = "below_threshold"
        else:
            reason = "passed"
        filtered[day, security] = reason == "passed"
        records.append({"day": int(day), "security": int(security),
                        "history_start": max(0, start), "history_end": int(day),
                        "history_sessions": len(history), "missing_observations": missing,
                        "negative_observations": negative, "reference_notional": reference,
                        "minimum_notional": minimum_notional, "reason": reason})
    return filtered, records
