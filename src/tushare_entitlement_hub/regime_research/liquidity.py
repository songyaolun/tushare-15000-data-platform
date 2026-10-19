"""Causal research order limits from past turnover, without current-session volume."""

from dataclasses import dataclass
from math import floor, isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LiquidityRule:
    """Limit gross daily notional per security; .01 means 1% of a past-only reference."""

    maximum_participation: float = .01
    lookback_sessions: int = 20

    def __post_init__(self) -> None:
        """Reject nonfinite proportions or invalid windows; rate=0 or lookback=0 raises."""
        if (isinstance(self.maximum_participation, bool)
                or not isfinite(self.maximum_participation)
                or not 0 < self.maximum_participation <= 1
                or type(self.lookback_sessions) is not int
                or not 1 <= self.lookback_sessions <= 252):
            raise ValueError("invalid_liquidity_rule")


def historical_notional_limit(
    amounts: NDArray[np.float64], day: int, security: int, rule: LiquidityRule,
) -> float:
    """Use min(previous amount, median of previous window), converting thousand CNY once.

    Example: twenty past values of 10000 and rate=.01 allow 100000 CNY. Today and
    later observations are never read. Known zero turnover remains in the window;
    negative, nonfinite or insufficient history yields zero capacity, not a fill.
    """
    if day < rule.lookback_sessions:
        return 0.
    history = amounts[day - rule.lookback_sessions:day, security]
    if not np.all(np.isfinite(history) & (history >= 0)):
        return 0.
    return float(min(history[-1], np.median(history)) * 1000 * rule.maximum_participation)


def liquidity_sale_shares(shares: float, price: float, limit: float, code: str) -> float:
    """Sell a full remaining lot when it fits, otherwise round a partial sale down.

    Example: 250 mainboard shares at 10 CNY with 1500 cap sells 100; with 2500 cap
    sells all 250 including the odd tail. STAR partial sales use one-share steps.
    """
    if shares * price <= limit:
        return shares
    step = 1 if code.startswith(("688", "689")) else 100
    return float(max(0, floor(limit / price / step)) * step)
