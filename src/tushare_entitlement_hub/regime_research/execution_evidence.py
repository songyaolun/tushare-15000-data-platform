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
        self.order_attempts: list[dict[str, Any]] = []
        self.allocation_states: list[dict[str, Any]] = []

    def order_attempt(
        self, day: int, security: int, confirmation: int, reason: str,
        constraints: dict[str, float] | None = None, *, price: float | None = None,
        requested_shares: float | None = None, executed_shares: float | None = None,
        liquidity_available: float | None = None,
    ) -> None:
        """Record one reached buy attempt, retaining competing limits without attribution claims.

        Example: cash=100 and sector=50 label sector as the first minimum. A paired
        exit wait has no sizing inputs, which stay None rather than invented zero.
        """
        limits = constraints or {}
        self.order_attempts.append({
            "day": day, "date": self.panel.calendar[day], "security": security,
            "ts_code": self.panel.codes[security], "confirmation": confirmation,
            "reason": reason, "price": price, "requested_shares": requested_shares,
            "executed_shares": executed_shares, "liquidity_available": liquidity_available,
            "first_binding_constraint": (
                min(limits, key=lambda name: limits[name]) if limits else ""
            ),
            **{name: limits.get(name) for name in (
                "signal_budget", "position_limit", "cash_available",
                "target_available", "sector_available",
            )},
        })

    def allocation_state(self, day: int, **values: Any) -> None:
        """Copy closing candidate/slot diagnostics; day0 scheduled=0 remains an explicit row."""
        self.allocation_states.append({"day": day, "date": self.panel.calendar[day], **values})

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
                "positions": self.positions, "order_attempts": self.order_attempts,
                "allocation_states": self.allocation_states}
