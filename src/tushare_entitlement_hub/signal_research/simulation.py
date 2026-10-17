"""Conservative daily-bar signal execution with explicitly limited return semantics.

Raw OHLC drives fills; provider factors drive a separate price-return estimate.
This module does not reconstruct dividend cash, share entitlements or account fees.
"""

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite


@dataclass(frozen=True)
class SimulationConfiguration:
    """Research execution assumptions, rather than certified account conditions.

    Example: ``SimulationConfiguration()`` uses 3 bp commission on each side,
    5 bp adverse slippage, a 40-session close-observed cap, and no price stop.
    Transfer fee zero means omitted/unverified, not a claim that no fee is owed.
    """

    green_histogram_threshold: float = 0.002
    maximum_holding_sessions: int = 40
    adverse_slippage_rate: float = 0.0005
    commission_rate: float = 0.0003
    transfer_fee_rate: float = 0.0
    price_stop_loss_fraction: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid assumptions before any fills are evaluated.

        Input example: ``maximum_holding_sessions=0``; output: ``ValueError``.
        All rates must be finite and nonnegative; price stops must lie in (0, 1).
        """
        rates = (
            self.green_histogram_threshold,
            self.adverse_slippage_rate,
            self.commission_rate,
            self.transfer_fee_rate,
        )
        if any(not isfinite(value) or value < 0 for value in rates):
            raise ValueError("Research rates must be finite and nonnegative")
        if self.adverse_slippage_rate >= 1:
            raise ValueError("Adverse slippage must be less than one")
        if self.maximum_holding_sessions < 1:
            raise ValueError("Holding-session cap must be positive")
        if self.price_stop_loss_fraction is not None and (
            not isfinite(self.price_stop_loss_fraction) or not 0 < self.price_stop_loss_fraction < 1
        ):
            raise ValueError("Price-stop fraction must lie strictly between zero and one")


def _number(value: object) -> float | None:
    """Parse a finite numeric source value without manufacturing missing data.

    Input examples: ``'12.5'``, ``None``; output: ``12.5``, ``None`` respectively.
    Boolean, malformed and nonfinite values are unavailable.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric_value = float(str(value))
    except ValueError:
        return None
    return numeric_value if isfinite(numeric_value) else None


def _day(value: object) -> str:
    """Normalize ISO or provider dates and reject invalid dates.

    Input: ``'20230828'`` or ``'2023-08-28'``; output: ``'2023-08-28'``.
    """
    text_value = str(value)
    if len(text_value) == 8 and text_value.isdigit():
        text_value = f"{text_value[:4]}-{text_value[4:6]}-{text_value[6:]}"
    return date.fromisoformat(text_value).isoformat()


def _fill_blocker(bar: Mapping[str, object], side: str) -> str | None:
    """Explain why an open cannot execute; None denotes an eligible quote.

    Input: an upper-limit locked bar and ``'buy'``; output: ``'price_limit_locked'``.
    Missing/nonpositive activity or OHLC fails closed. An open at the relevant
    limit cannot fill even when later daily-bar prices indicate an unlock.
    """
    if bar.get("suspended") or bar.get("tradable", True) is not True:
        return "suspended_or_not_tradable"
    required_values = [_number(bar.get(field)) for field in ("open", "high", "low", "close")]
    if any(value is None or value <= 0 for value in required_values):
        return "missing_or_invalid_ohlc"
    opening_price = _number(bar.get("open"))
    high_price = _number(bar.get("high"))
    low_price = _number(bar.get("low"))
    closing_price = _number(bar.get("close"))
    assert opening_price is not None and high_price is not None and low_price is not None
    assert closing_price is not None
    if (
        not low_price
        <= min(opening_price, closing_price)
        <= max(opening_price, closing_price)
        <= high_price
    ):
        return "inconsistent_ohlc_range"
    for activity_field in ("vol", "amount"):
        activity_value = _number(bar.get(activity_field))
        if activity_value is None or activity_value <= 0:
            return "missing_or_nonpositive_activity"
    if bar.get("price_limits_exempt") is True:
        return None
    limit_price = _number(bar.get("up_limit" if side == "buy" else "down_limit"))
    if limit_price is None or limit_price <= 0:
        return "missing_or_invalid_directional_price_limit"
    if (side == "buy" and opening_price >= limit_price) or (
        side == "sell" and opening_price <= limit_price
    ):
        return "price_limit_locked"
    return None


def _can_fill(bar: Mapping[str, object], side: str) -> bool:
    """Return open execution eligibility using fail-closed price-limit evidence.

    Input: valid activity but missing limits and no explicit exemption; output False.
    Explicit ``price_limits_exempt=True`` permits legitimate unrestricted sessions.
    """
    return _fill_blocker(bar, side) is None


def _fill_price(
    bar: Mapping[str, object], side: str, configuration: SimulationConfiguration
) -> float:
    """Apply adverse slippage bounded by known daily price limits.

    Input: open 10, buy, slippage 0.001; output: 10.01 (unless capped by up_limit).
    The caller must establish quote eligibility with ``_can_fill`` first.
    """
    opening_price = _number(bar["open"])
    assert opening_price is not None
    fill_price = opening_price * (
        1 + configuration.adverse_slippage_rate * (1 if side == "buy" else -1)
    )
    limit_price = _number(bar.get("up_limit" if side == "buy" else "down_limit"))
    if limit_price is not None and bar.get("price_limits_exempt") is not True:
        fill_price = min(fill_price, limit_price) if side == "buy" else max(fill_price, limit_price)
    return fill_price


def _returns(
    entry_bar: Mapping[str, object],
    exit_bar: Mapping[str, object],
    entry_price: float,
    exit_price: float,
    configuration: SimulationConfiguration,
) -> dict[str, object]:
    """Calculate factor-adjusted research returns and normalized fee estimates.

    Input: entry 10/factor 1, exit 5/factor 2; output: price return 0, rather
    than -0.5. Without positive factors, both return estimates remain ``None``.
    Fees use synthetic adjusted notionals; dividend cash and share changes are
    not reconstructed, and minimum commissions are not assumed.
    """
    stamp_tax_rate = 0.0005 if str(exit_bar["trade_date"]) >= "2023-08-28" else 0.001
    result: dict[str, object] = {
        "provider_adjusted_price_return": None,
        "fee_adjusted_research_return": None,
        "research_fee_fraction": None,
        "stamp_tax_rate": stamp_tax_rate,
    }
    entry_factor = _number(entry_bar.get("adj_factor"))
    exit_factor = _number(exit_bar.get("adj_factor"))
    if entry_factor is None or exit_factor is None or entry_factor <= 0 or exit_factor <= 0:
        result["return_unavailable_reason"] = "missing_or_invalid_adjustment_factor"
        return result
    gross_multiple = exit_price * exit_factor / (entry_price * entry_factor)
    fee_fraction = configuration.commission_rate + configuration.transfer_fee_rate
    fee_fraction += gross_multiple * (
        configuration.commission_rate + configuration.transfer_fee_rate + stamp_tax_rate
    )
    result.update(
        {
            "provider_adjusted_price_return": gross_multiple - 1,
            "fee_adjusted_research_return": gross_multiple - 1 - fee_fraction,
            "research_fee_fraction": fee_fraction,
        }
    )
    return result


def simulate_signal_trades(
    bars: Sequence[Mapping[str, object]],
    confirmations: Sequence[Mapping[str, object]],
    configuration: SimulationConfiguration | None = None,
) -> list[dict[str, object]]:
    """Simulate one concurrent position per security and experiment causally.

    Input: daily raw OHLC/activity/factor/histogram bars and confirmation records
    with ``ts_code``, ``confirmation_date``, ``experiment_id``. Output: one dict
    per signal with ``closed``, ``open``, ``censored`` or ``skipped`` status.
    ``signal_date``/``trade_date`` are accepted aliases for confirmation dates.
    Entries and close-observed exits fill at a later session's eligible open;
    pending orders wait through suspensions/limits. The union of supplied bar
    dates defines observed sessions, so the caller must supply the full market
    calendar as rows, including suspended sessions. All input is copied.
    Example: confirm Monday, enter Tuesday, green Tuesday, exit Wednesday.
    """
    assumptions = configuration or SimulationConfiguration()
    security_bars: dict[str, list[dict[str, object]]] = {}
    unique_keys: set[tuple[str, str]] = set()
    for source_bar in bars:
        copied_bar = dict(source_bar)
        security_code = str(copied_bar["ts_code"])
        copied_bar["trade_date"] = _day(copied_bar["trade_date"])
        bar_key = security_code, str(copied_bar["trade_date"])
        if bar_key in unique_keys:
            raise ValueError("Duplicate security/date bar")
        unique_keys.add(bar_key)
        security_bars.setdefault(security_code, []).append(copied_bar)
    session_dates = sorted({key[1] for key in unique_keys})
    session_indices = {session_date: index for index, session_date in enumerate(session_dates)}
    for daily_bars in security_bars.values():
        daily_bars.sort(key=lambda daily_bar: str(daily_bar["trade_date"]))
    security_dates = {
        security_code: [str(daily_bar["trade_date"]) for daily_bar in daily_bars]
        for security_code, daily_bars in security_bars.items()
    }
    signals: list[dict[str, object]] = []
    for source_signal in confirmations:
        copied_signal = dict(source_signal)
        confirmation_value = copied_signal.get(
            "confirmation_date", copied_signal.get("signal_date", copied_signal.get("trade_date"))
        )
        copied_signal["confirmation_date"] = _day(confirmation_value)
        signals.append(copied_signal)
    signals.sort(key=lambda signal: str(signal["confirmation_date"]))
    occupied_until: dict[tuple[str, str], str | None] = {}
    results: list[dict[str, object]] = []
    for signal in signals:
        security_code = str(signal["ts_code"])
        experiment_id = str(signal.get("experiment_id", "default"))
        confirmation_date = str(signal["confirmation_date"])
        position_key = security_code, experiment_id
        result: dict[str, object] = {
            **signal,
            "experiment_id": experiment_id,
            "status": "censored",
            "entry_date": None,
            "exit_date": None,
            "exit_trigger_date": None,
            "exit_reason": None,
            "entry_price": None,
            "exit_price": None,
            "holding_sessions": None,
            "provider_adjusted_price_return": None,
            "fee_adjusted_research_return": None,
            "entry_blocked_reason_counts": {},
            "exit_blocked_reason_counts": {},
            "return_semantics": "provider_factor_adjusted_price_estimate_not_dividend_cash_net_pnl",
            "fee_semantics": (
                "synthetic_adjusted_notional_research_assumption_no_minimum_commission"
            ),
            "transfer_fee_semantics": "unverified_research_assumption",
            "commission_rate": assumptions.commission_rate,
            "transfer_fee_rate": assumptions.transfer_fee_rate,
            "adverse_slippage_rate": assumptions.adverse_slippage_rate,
            "maximum_holding_sessions": assumptions.maximum_holding_sessions,
            "green_histogram_threshold": assumptions.green_histogram_threshold,
            "threshold_semantics": "unoptimized_research_ratio_0.02_histogram_at_10_CNY",
            "price_stop_loss_fraction": assumptions.price_stop_loss_fraction,
        }
        if position_key in occupied_until and (
            occupied_until[position_key] is None
            or confirmation_date < str(occupied_until[position_key])
        ):
            result.update(status="skipped", skip_reason="overlapping_position_or_pending_order")
            results.append(result)
            continue
        occupied_until[position_key] = None
        daily_bars = security_bars.get(security_code, [])
        entry_bar: dict[str, object] | None = None
        exit_bar: dict[str, object] | None = None
        entry_price = 0.0
        trigger_date: str | None = None
        start_index = bisect_right(security_dates.get(security_code, []), confirmation_date)
        for bar_index in range(start_index, len(daily_bars)):
            daily_bar = daily_bars[bar_index]
            current_date = str(daily_bar["trade_date"])
            if entry_bar is None:
                entry_blocker = _fill_blocker(daily_bar, "buy")
                if entry_blocker is not None:
                    entry_blocks = result["entry_blocked_reason_counts"]
                    assert isinstance(entry_blocks, dict)
                    entry_blocks[entry_blocker] = entry_blocks.get(entry_blocker, 0) + 1
                    continue
                entry_bar = daily_bar
                entry_price = _fill_price(daily_bar, "buy", assumptions)
                result.update(status="open", entry_date=current_date, entry_price=entry_price)
            elif trigger_date is not None and current_date > trigger_date:
                exit_blocker = _fill_blocker(daily_bar, "sell")
                if exit_blocker is None:
                    exit_bar = daily_bar
                    break
                exit_blocks = result["exit_blocked_reason_counts"]
                assert isinstance(exit_blocks, dict)
                exit_blocks[exit_blocker] = exit_blocks.get(exit_blocker, 0) + 1
            held_sessions = (
                session_indices[current_date] - session_indices[str(entry_bar["trade_date"])] + 1
            )
            result["holding_sessions"] = held_sessions
            if trigger_date is None:
                histogram_value = _number(daily_bar.get("normalized_histogram"))
                exit_reason = None
                closing_price = _number(daily_bar.get("close"))
                if (
                    histogram_value is not None
                    and histogram_value < -assumptions.green_histogram_threshold
                ):
                    exit_reason = "first_clear_green_histogram"
                elif (
                    assumptions.price_stop_loss_fraction is not None
                    and closing_price is not None
                    and closing_price > 0
                ):
                    stop_return = _returns(
                        entry_bar,
                        daily_bar,
                        entry_price,
                        closing_price,
                        assumptions,
                    )["provider_adjusted_price_return"]
                    if (
                        stop_return is not None
                        and float(str(stop_return)) <= -assumptions.price_stop_loss_fraction
                    ):
                        exit_reason = "configured_close_observed_price_stop"
                if exit_reason is None and held_sessions >= assumptions.maximum_holding_sessions:
                    exit_reason = "research_time_cap"
                if exit_reason is not None:
                    trigger_date = current_date
                    result.update(exit_trigger_date=current_date, exit_reason=exit_reason)
        if entry_bar is None:
            result["censor_reason"] = "no_eligible_future_entry_quote"
        elif exit_bar is None:
            result["censor_reason"] = (
                "no_eligible_future_exit_quote" if trigger_date else "no_exit_trigger_observed"
            )
        else:
            exit_price = _fill_price(exit_bar, "sell", assumptions)
            exit_date = str(exit_bar["trade_date"])
            result.update(status="closed", exit_date=exit_date, exit_price=exit_price)
            result["holding_sessions"] = (
                session_indices[exit_date] - session_indices[str(entry_bar["trade_date"])]
            )
            result.update(_returns(entry_bar, exit_bar, entry_price, exit_price, assumptions))
            occupied_until[position_key] = exit_date
        results.append(result)
    return results
