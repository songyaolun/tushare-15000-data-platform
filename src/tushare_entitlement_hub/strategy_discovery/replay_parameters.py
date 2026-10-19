"""Validate finite, unlevered execution experiments separately from model training."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ReplayParameters:
    """Frozen replay controls; defaults reproduce the established 47-column account.

    Example: ReplayParameters(take_profit=0.06) changes only the closing profit
    trigger. The existing next-open execution, cash and liquidity limits still apply.
    """

    take_profit: float = 0.045
    stop_loss: float = 0.08
    maximum_holding: int = 30
    position_weight: float = 0.05
    replacement_sessions: int = 10

    def __post_init__(self) -> None:
        """Reject non-finite thresholds, leverage and invalid session counts.

        Example: ReplayParameters(position_weight=1.1) raises ValueError;
        ReplayParameters(maximum_holding=15) preserves a valid finite experiment.
        """
        for value in (self.take_profit, self.stop_loss, self.position_weight):
            if isinstance(value, bool) or not isfinite(value) or not 0 < value < 1:
                raise ValueError("invalid_replay_fraction")
        if self.position_weight > .4:
            raise ValueError("invalid_replay_sector_weight")
        for value in (self.maximum_holding, self.replacement_sessions):
            if type(value) is not int or not 1 <= value <= 30:
                raise ValueError("invalid_replay_sessions")
