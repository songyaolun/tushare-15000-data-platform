"""Research-only sizing from known closing NAV; never changes scores or forces sales."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import cast


def finite_number(value: object) -> bool:
    """Accept finite real numbers, excluding Boolean flags; e.g. 1 -> True, NaN -> False."""
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class SizingDecision:
    """A closing instruction for the next open; e.g. (.05, .15, 1e7) permits three slots."""

    position_weight: float
    maximum_total_weight: float
    tier_floor: float

    def __post_init__(self) -> None:
        """Reject nonfinite, levered or negative sizing; e.g. weight=0 raises ValueError."""
        if not (
            all(
                finite_number(value)
                for value in (
                    self.position_weight,
                    self.maximum_total_weight,
                    self.tier_floor,
                )
            )
            and 0 < self.position_weight <= 0.4
            and 0 <= self.maximum_total_weight <= 1
            and self.tier_floor >= 0
        ):
            raise ValueError("invalid_account_sizing")


@dataclass(frozen=True)
class CapitalSizingRule:
    """NAV tiers and weak-market entry caps, evaluated only after that day's close.

    Example: tiers=((0,.1),(1e7,.05),(1e8,.02)), NAV=1e7 and stage=0
    yields weight=.05 and a .15 total cap. Crossings are reversible; existing
    holdings are neither topped up nor sold merely to match the new tier.
    """

    tiers: tuple[tuple[float, float], ...] = ((0.0, 0.1),)
    weak_allocation: float = 0.15
    weak_position_weight: float | None = None

    def __post_init__(self) -> None:
        """Require ascending NAV floors and nonincreasing weights; e.g. duplicate floors fail."""
        if not self.tiers or self.tiers[0][0] != 0:
            raise ValueError("capital_tiers_must_start_at_zero")
        previous_floor, previous_weight = -1.0, 1.0
        for floor, weight in self.tiers:
            SizingDecision(weight, self.weak_allocation, floor)
            if floor <= previous_floor or weight > previous_weight:
                raise ValueError("invalid_capital_tier_order")
            previous_floor, previous_weight = floor, weight
        if self.weak_position_weight is not None:
            SizingDecision(self.weak_position_weight, self.weak_allocation, 0.0)

    def __call__(self, assets: float, stage: int) -> SizingDecision:
        """Resolve closing NAV and binary breadth state; e.g. stage=1 retains 100% total cap.

        Weak position weight is a ceiling, so a 5% weak cap cannot enlarge a 2%
        large-account tier. Unknown/nonpositive NAV and unknown states fail closed.
        """
        if (
            not finite_number(assets)
            or assets <= 0
            or type(stage) is not int
            or stage not in (0, 1)
        ):
            raise ValueError("invalid_sizing_observation")
        floor, weight = next(pair for pair in reversed(self.tiers) if assets >= pair[0])
        if stage == 0 and self.weak_position_weight is not None:
            weight = min(weight, self.weak_position_weight)
        return SizingDecision(weight, self.weak_allocation if stage == 0 else 1.0, floor)


def parse_capital_tiers(value: str) -> tuple[tuple[float, float], ...]:
    """Parse explicit CNY floors and fractional weights; '0:0.1,10000000:0.05' -> two tiers."""
    try:
        tiers = tuple(
            tuple(float(number) for number in part.split(":")) for part in value.split(",")
        )
        if any(len(pair) != 2 for pair in tiers):
            raise ValueError("invalid_capital_tiers")
        validated_tiers = cast(tuple[tuple[float, float], ...], tiers)
        CapitalSizingRule(validated_tiers)
    except (ValueError, TypeError, IndexError) as error:
        raise ValueError("invalid_capital_tiers") from error
    return validated_tiers
