"""Optional observations of executed fills and corporate actions; never size orders."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .accounting import Distribution, Holding
    from .panel import Panel


class ExecutionEvidence:
    """Collect raw-share evidence with stable (security, entry) lot identities."""

    def __init__(self, panel: "Panel") -> None:
        """Bind calendar and raw market observations; an empty run yields three empty lists."""
        self.panel = panel
        self.fills: list[dict[str, Any]] = []
        self.distributions: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []

    def fill(
        self, holding: "Holding", day: int, side: str, price: float,
        reference_price: float, fee: float, cash_after: float, reason: str,
    ) -> None:
        """Observe an executed fill in CNY and shares, including unaltered daily amount.

        Example: 100 shares at 10 CNY records notional=1000; raw_amount=20 means
        20 thousand CNY. Unknown observations remain NaN for downstream rejection.
        """
        security = holding.security
        self.fills.append({
            "sequence": len(self.fills), "day": day, "date": self.panel.calendar[day],
            "security": security, "ts_code": self.panel.codes[security],
            "entry": holding.entry, "confirmation": holding.confirmation,
            "side": side, "shares": holding.shares, "price": price,
            "reference_price": reference_price, "notional": price * holding.shares,
            "fee": fee, "cash_after": cash_after, "reason": reason,
            **{name: float(self.panel.values[name][day, security]) for name in (
                "raw_open", "raw_high", "raw_low", "raw_close", "raw_amount",
                "raw_up_limit", "raw_down_limit",
            )},
        })

    def distribution(
        self, kind: str, day: int, event: "Distribution", holding: "Holding",
        shares: float, cash: float,
    ) -> None:
        """Observe the actual book transition with original event dates and lot identity.

        Example: 100 registered shares at 1 CNY produce an entitlement cash=80
        under the engine's 20% tax assumption; payment is a separate observation.
        """
        self.distributions.append({
            "kind": kind, "day": day, "date": self.panel.calendar[day],
            "ts_code": self.panel.codes[holding.security], "entry": holding.entry,
            "shares": shares, "cash": cash,
            **{f"event_{key}": value for key, value in asdict(event).items()},
        })

    def snapshot(self, day: int, holdings: dict[int, "Holding"]) -> None:
        """Copy each closing lot without mutable aliases; 100 shares at 10 records value=1000."""
        self.positions.extend({
            "day": day, "date": self.panel.calendar[day],
            "ts_code": self.panel.codes[position.security], **asdict(position),
            "market_value": position.shares * position.price,
        } for position in holdings.values())

    def result(self) -> dict[str, list[dict[str, Any]]]:
        """Return named CSV tables; an empty collector returns fills/distributions/positions=[] ."""
        return {"fills": self.fills, "distributions": self.distributions,
                "positions": self.positions}
