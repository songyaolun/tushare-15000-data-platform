"""Disclosure-dated quality and earnings yield, with no report or missing-value fallback."""

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]
QUALITY_FEATURE_NAMES = (
    "deducted_profit_growth", "deducted_growth_gap", "cash_profit_backing",
)
VALUATION_FEATURE_NAMES = ("earnings_to_price",)
REASONS = (
    "known", "no_visible_report", "missing_value", "conflicting_values",
    "missing_ttm_component", "incompatible_company_type", "nonindustrial_cashflow",
    "nonfinite_arithmetic", "nonpositive_net_profit", "invalid_market_cap",
)


@dataclass(frozen=True)
class ReportEvent:
    """One disclosure version; e.g. period=20251231/session=5 activates on session 5."""

    effective_session: int
    disclosure_date: str
    period: str
    company_type: str
    values: tuple[float, ...]
    reasons: tuple[int, ...]


def finite_number(value: Any) -> float:
    """Return a finite numeric value or NaN, without coercing text/bools into facts.

    Examples: 12 becomes 12.0; None, '12', True and infinity become NaN.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def valid_date(value: Any) -> bool:
    """Accept an exact valid YYYYMMDD string; '20240229' passes, '20240230' fails."""
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def report_events(
    records: Iterable[Mapping[str, Any]],
    calendar: Sequence[str],
    codes: Sequence[str],
    fields: tuple[str, ...],
    *,
    cumulative: bool,
) -> tuple[dict[int, list[ReportEvent]], dict[str, int]]:
    """Group source records into next-session versions; conflicts never pick update_flag.

    Cumulative sources accept report_type=1 only. Monetary fields remain yuan.
    Indicator sources have no report/company type and retain their own disclosure.
    Empty optional f_ann_date falls back to ann_date; a nonempty invalid date fails.
    Same security/period/effective disclosure date is one identity: each conflicting
    field becomes unknown independently; differing company types invalidate all fields.
    Missing and finite values at the same identity are ambiguous and also fail closed.
    Example: Friday ann_date activates Monday; two Friday values 10/12 yield NaN.
    observed_at is not backdated: this is supplier-reconstructed research, not PIT proof.
    """
    security_indices = {code: index for index, code in enumerate(codes)}
    grouped: dict[tuple[int, str, str], list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    audit: Counter[str] = Counter()
    for record in records:
        audit["source_rows"] += 1
        security = security_indices.get(str(record.get("ts_code", "")))
        if security is None:
            audit["outside_candidate_security_axis"] += 1
            continue
        announcement = record.get("ann_date")
        actual = record.get("f_ann_date") or announcement
        period = record.get("end_date")
        if not all(valid_date(value) for value in (announcement, actual, period)):
            audit["invalid_date"] += 1
            continue
        disclosure = max(str(announcement), str(actual))
        period = str(period)
        if period[4:] not in ("0331", "0630", "0930", "1231") or disclosure < period:
            audit["invalid_period_or_premature_disclosure"] += 1
            continue
        if cumulative and str(record.get("report_type")) != "1":
            audit["noncumulative_or_nonconsolidated_report"] += 1
            continue
        company_type = str(record.get("comp_type", "")) if cumulative else ""
        grouped[(security, period, disclosure)].append(
            (company_type, tuple(finite_number(record.get(field)) for field in fields))
        )
    events: dict[int, list[ReportEvent]] = defaultdict(list)
    for (security, period, disclosure), versions in grouped.items():
        companies = {version[0] for version in versions}
        values, reasons = [], []
        for field_index in range(len(fields)):
            observed = {
                version[1][field_index] if np.isfinite(version[1][field_index]) else None
                for version in versions
            }
            reason = 3 if len(observed) != 1 or len(companies) != 1 else 0
            value = next(iter(observed)) if reason == 0 else None
            if reason == 0 and value is None:
                reason = 2
            values.append(float(value) if value is not None else float("nan"))
            reasons.append(reason)
            audit[f"field_{fields[field_index]}_{REASONS[reason]}"] += 1
        company_type = next(iter(companies)) if len(companies) == 1 else ""
        events[security].append(ReportEvent(
            bisect_right(calendar, disclosure), disclosure, period, company_type,
            tuple(values), tuple(reasons),
        ))
    for security_events in events.values():
        security_events.sort(key=lambda event: (event.effective_session, event.disclosure_date))
    audit["disclosure_identities"] = sum(len(items) for items in events.values())
    return dict(events), dict(audit)


def report_value(
    latest_period: str,
    visible: Mapping[str, ReportEvent],
    field_index: int,
    *,
    cumulative: bool,
    industrial_only: bool,
) -> tuple[float, int]:
    """Read latest report or its TTM without falling back to an older report.

    Example: current H1=80, prior annual=100, prior H1=50 yields TTM=130.
    Every component uses its latest visible version; a missing/conflicting component
    invalidates the value. Annual values are already TTM. Company type must match.
    """
    current = visible[latest_period]
    if industrial_only and current.company_type != "1":
        return float("nan"), 6
    periods = [latest_period]
    if cumulative and latest_period[4:] != "1231":
        previous_year = str(int(latest_period[:4]) - 1)
        periods += [previous_year + "1231", previous_year + latest_period[4:]]
    if any(period not in visible for period in periods):
        return float("nan"), 4
    components = [visible[period] for period in periods]
    if cumulative and (
        not current.company_type
        or any(item.company_type != current.company_type for item in components)
    ):
        return float("nan"), 5
    for component in components:
        if component.reasons[field_index] != 0:
            return float("nan"), component.reasons[field_index]
    value = components[0].values[field_index]
    if len(components) == 3:
        value += components[1].values[field_index] - components[2].values[field_index]
    return (value, 0) if np.isfinite(value) else (float("nan"), 7)


def candidate_report_values(
    records: Iterable[Mapping[str, Any]],
    calendar: Sequence[str],
    codes: Sequence[str],
    candidate_days: IntegerArray,
    candidate_securities: IntegerArray,
    fields: tuple[str, ...],
    *,
    cumulative: bool,
    industrial_only: bool = False,
) -> tuple[FloatArray, NDArray[np.int8], dict[str, int]]:
    """Map direct financial values or TTM onto the unchanged candidate row order.

    Candidate axes must be integral and inside unique sorted calendar/security axes.
    Records may be unordered. Values become visible strictly after disclosure date;
    later corrections cannot alter earlier candidates. No labels or scores are inputs.
    Example: fields=('net_profit',), cumulative=True returns N×1 TTM and reason arrays.
    """
    if (
        not calendar or not codes or not fields or len(set(fields)) != len(fields)
        or list(calendar) != sorted(set(calendar)) or not all(valid_date(day) for day in calendar)
        or len(set(codes)) != len(codes) or candidate_days.ndim != 1
        or candidate_securities.shape != candidate_days.shape
        or not np.issubdtype(candidate_days.dtype, np.integer)
        or not np.issubdtype(candidate_securities.dtype, np.integer)
        or np.any(candidate_days < 0) or np.any(candidate_days >= len(calendar))
        or np.any(candidate_securities < 0) or np.any(candidate_securities >= len(codes))
        or (industrial_only and not cumulative)
    ):
        raise ValueError("invalid financial feature axes or statement contract")
    events, audit = report_events(records, calendar, codes, fields, cumulative=cumulative)
    output = np.full((len(candidate_days), len(fields)), np.nan)
    reasons = np.ones(output.shape, dtype=np.int8)
    rows_by_security: dict[int, list[int]] = defaultdict(list)
    for row, security in enumerate(candidate_securities):
        rows_by_security[int(security)].append(row)
    for security, rows in rows_by_security.items():
        ordered = sorted(rows, key=lambda row: candidate_days[row])
        security_events = events.get(security, [])
        position = 0
        visible: dict[str, ReportEvent] = {}
        latest_period = ""
        for row in ordered:
            while (
                position < len(security_events)
                and security_events[position].effective_session <= candidate_days[row]
            ):
                event = security_events[position]
                visible[event.period] = event
                latest_period = max(latest_period, event.period)
                position += 1
            if not latest_period:
                continue
            for field_index in range(len(fields)):
                output[row, field_index], reasons[row, field_index] = report_value(
                    latest_period, visible, field_index,
                    cumulative=cumulative, industrial_only=industrial_only,
                )
    return output, reasons, audit


def quality_features(indicators: FloatArray, cashflow: FloatArray) -> FloatArray:
    """Return fixed growth, growth-gap and cash-backing columns, retaining unknowns.

    Inputs are N×2: indicator [deducted_yoy_percent,parent_yoy_percent] and
    same-statement TTM [operating_cash_yuan,consolidated_profit_yuan]. Caller must
    already reject financial/nonindustrial cashflow and incompatible report types.
    Example: [[30,10]], [[150,100]] yields [[.3,.2,.2]]. NP<=0 makes cash NaN.
    """
    if indicators.ndim != 2 or indicators.shape[1] != 2 or cashflow.shape != indicators.shape:
        raise ValueError("quality requires matching N by 2 inputs")
    result = np.full((len(indicators), 3), np.nan)
    result[:, 0] = indicators[:, 0] / 100
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result[:, 1] = (indicators[:, 0] - indicators[:, 1]) / 100
        valid = np.isfinite(cashflow).all(axis=1) & (cashflow[:, 1] > 0)
        result[valid, 2] = (
            (cashflow[valid, 0] - cashflow[valid, 1])
            / (np.abs(cashflow[valid, 0]) + np.abs(cashflow[valid, 1]))
        )
    result[~np.isfinite(result)] = np.nan
    return result


def earnings_yield(parent_profit_ttm: FloatArray, market_cap_yuan: FloatArray) -> FloatArray:
    """Divide visible parent-owner TTM profit by same-day unadjusted market cap.

    Inputs are equally shaped vectors in yuan; negative/zero profits remain valid.
    Example: profit [-10,0,20], cap [100,100,0] yields [[-.1],[0],[NaN]].
    No PE inversion, adjusted price, future shares or cross-date carry is used.
    """
    if parent_profit_ttm.ndim != 1 or market_cap_yuan.shape != parent_profit_ttm.shape:
        raise ValueError("earnings yield requires matching one-dimensional inputs")
    result = np.full((len(parent_profit_ttm), 1), np.nan)
    valid = np.isfinite(parent_profit_ttm) & np.isfinite(market_cap_yuan) & (market_cap_yuan > 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result[valid, 0] = parent_profit_ttm[valid] / market_cap_yuan[valid]
    result[~np.isfinite(result)] = np.nan
    return result
