"""Independent CSV reconstruction; no calls into simulation or fee calculation code."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a required CSV, retaining every row; an empty existing file yields [] ."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_capacity(fills: list[dict[str, str]]) -> dict[str, Any]:
    """Audit execution-date notional against Tushare daily amount in thousand CNY.

    Example: notional=1000 and raw_amount=10 implies 1/10 daily participation.
    Invalid amounts retain their row and denominator; daily evidence never proves
    open-auction/minute capacity and is never fed back into order decisions.
    """
    observations = []
    anomalies = []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in fills:
        amount = float(row["raw_amount"])
        notional = float(row["notional"])
        price = float(row["price"])
        reasons = []
        ratio = None
        if not math.isfinite(amount) or amount <= 0:
            reasons.append("unknown_or_nonpositive_daily_amount")
        elif not math.isfinite(notional) or notional <= 0:
            reasons.append("invalid_fill_notional")
        else:
            ratio = notional / (amount * 1000)
        lower, upper = float(row["raw_low"]), float(row["raw_high"])
        if not all(math.isfinite(value) and value > 0 for value in (lower, upper, price)):
            reasons.append("unknown_or_nonpositive_price")
        elif lower > upper or not lower - 1e-8 <= price <= upper + 1e-8:
            reasons.append("fill_outside_observed_daily_range")
        record = {"sequence": row["sequence"], "date": row["date"],
                  "ts_code": row["ts_code"], "side": row["side"],
                  "notional_cny": notional, "daily_amount_thousand_cny": amount,
                  "participation": ratio, "anomalies": reasons}
        observations.append(record)
        if reasons:
            anomalies.append(record)
        key = row["date"], row["ts_code"]
        group = grouped.setdefault(key, {"date": key[0], "ts_code": key[1],
                                        "notional_cny": 0., "daily_amount_thousand_cny": amount})
        group["notional_cny"] += notional
        if amount != group["daily_amount_thousand_cny"]:
            group["daily_amount_thousand_cny"] = math.nan
    aggregate = []
    for row in grouped.values():
        amount = row["daily_amount_thousand_cny"]
        row["participation"] = (row["notional_cny"] / (amount * 1000)
                                if math.isfinite(amount) and amount > 0 else None)
        aggregate.append(row)
    ratios = [row["participation"] for row in observations if row["participation"] is not None]
    return {"fill_count": len(fills), "known_amount_fill_count": len(ratios),
            "unknown_amount_fill_count": len(fills) - len(ratios),
            "maximum_fill_participation": max(ratios, default=None),
            "maximum_security_day_participation": max(
                (row["participation"] for row in aggregate if row["participation"] is not None),
                default=None),
            "above_threshold_counts": {str(limit): sum(ratio > limit for ratio in ratios)
                                       for limit in (.01, .05, .1, 1.)},
            "observations": observations, "security_days": aggregate, "anomalies": anomalies,
            "capacity_certified": False, "participation_cap_enforced": False,
            "limitation": "Full-day turnover cannot establish opening execution capacity."}


def audit_execution(path: Path, capital: float, fee_multiplier: float,
                    slippage_rate: float) -> dict[str, Any]:
    """Reconstruct daily cash, fees, shares and dividends exclusively from exported CSVs.

    Example: a 1000 CNY purchase plus 5.01 fee reduces 10000 cash to 8994.99.
    Price/amount capacity anomalies are separate from arithmetic failures. The
    absolute monetary tolerance is 1e-5 CNY, and no missing row is silently removed.
    """
    tables = {name: read_rows(path / f"{name}.csv") for name in
              ("fills", "distributions", "positions", "daily", "trades")}
    dates = {int(row["day"]): row["date"] for row in tables["daily"]}
    if not dates or list(dates) != sorted(dates) or len(dates) != len(tables["daily"]):
        raise ValueError("invalid_daily_axis")
    by_day = {name: defaultdict(list) for name in ("fills", "distributions", "positions")}
    for name in by_day:
        for row in tables[name]:
            if dates.get(int(row["day"])) != row["date"]:
                raise ValueError("evidence_outside_daily_axis")
            by_day[name][int(row["day"])].append(row)
    for sequence, row in enumerate(tables["fills"]):
        if (int(row["sequence"]) != sequence or row["side"] not in {"buy", "sell"}
                or not all(math.isfinite(float(row[name])) and float(row[name]) > 0
                           for name in ("shares", "price", "notional"))):
            raise ValueError("invalid_fill_identity_or_quantity")
    cash, receivable, total_fees = capital, 0., 0.
    lots: dict[tuple[str, int], dict[str, float]] = {}
    registered: dict[tuple[str, ...], float] = {}
    sales: list[dict[str, Any]] = []
    errors = []
    maximum_errors: dict[str, float] = defaultdict(float)

    def compare(kind: str, actual: float, expected: float, location: Any) -> None:
        """Retain all mismatches, including nonfinite values; 1 versus 2 records error=1."""
        difference = abs(actual - expected)
        maximum_errors[kind] = max(maximum_errors[kind], difference)
        if not math.isfinite(difference) or difference > 1e-5:
            errors.append({"kind": kind, "location": location,
                           "actual": actual, "expected": expected})

    for daily in tables["daily"]:
        day = int(daily["day"])
        daily_fees = daily_slippage = 0.
        for row in by_day["distributions"][day]:
            if row["kind"] == "record":
                continue
            key = row["ts_code"], int(row["entry"])
            identity = tuple(row[name] for name in (
                "ts_code", "entry", "event_record", "event_ex", "event_pay",
                "event_listing", "event_cash", "event_stock"))
            amount, shares = float(row["cash"]), float(row["shares"])
            if row["kind"] == "entitlement":
                count = registered[identity]
                compare("dividend_entitlement", amount, count * float(row["event_cash"]) * .8, day)
                compare("stock_entitlement", shares, count * float(row["event_stock"]), day)
                compare("ex_date", day, int(row["event_ex"]), day)
                receivable += amount
                lots[key]["shares"] += shares
                lots[key]["locked"] += shares
                lots[key]["dividend"] += amount
            elif row["kind"] == "payment":
                compare("payment_date", day, int(row["event_pay"]), day)
                compare("payment_amount", amount,
                        registered[identity] * float(row["event_cash"]) * .8, day)
                cash += amount
                receivable -= amount
            elif row["kind"] == "unlock":
                compare("listing_date", day, int(row["event_listing"]), day)
                lots[key]["locked"] -= shares
            else:
                raise ValueError("unknown_distribution_kind")
        for row in by_day["fills"][day]:
            key = row["ts_code"], int(row["entry"])
            selling = row["side"] == "sell"
            shares, price, notional, fee = (float(row[name]) for name in
                                           ("shares", "price", "notional", "fee"))
            compare("notional", notional, shares * price, row["sequence"])
            transfer = .00001 if row["date"] >= "20220429" else .00002
            stamp = (.0005 if row["date"] >= "20230828" else .001) if selling else 0.
            expected_fee = (max(5., notional * .0003) + notional * (transfer + stamp))
            compare("fee", fee, expected_fee * fee_multiplier, row["sequence"])
            reference = float(row["reference_price"])
            expected_price = (max(reference * (1 - slippage_rate), float(row["raw_down_limit"]))
                              if selling else min(reference * (1 + slippage_rate),
                                                  float(row["raw_up_limit"])))
            compare("execution_price", price, expected_price, row["sequence"])
            cash += (notional if selling else -notional) - fee
            compare("fill_cash", float(row["cash_after"]), cash, row["sequence"])
            daily_fees += fee
            daily_slippage += (reference - price if selling else price - reference) * shares
            if selling:
                fraction = shares / lots[key]["shares"]
                if fraction > 1 + 1e-10:
                    raise ValueError("sale_exceeds_remaining_shares")
                compare("sell_unlocked", lots[key]["locked"], 0., row["sequence"])
                cost = lots[key]["cost"] * fraction
                dividend = lots[key]["dividend"] * fraction
                sales.append({"key": key, "exit": day, "shares": shares, "cost": cost,
                              "dividend": dividend, "proceeds": notional - fee})
                lots[key]["shares"] -= shares
                lots[key]["cost"] -= cost
                lots[key]["dividend"] -= dividend
            else:
                lots[key] = {"shares": shares, "locked": 0., "dividend": 0.,
                             "cost": notional + fee, "proceeds": 0.}
        total_fees += daily_fees
        marked = 0.
        for row in by_day["positions"][day]:
            key = row["ts_code"], int(row["entry"])
            for name in ("shares", "locked", "dividend", "cost"):
                compare(f"position_{name}", float(row[name]), lots[key][name], [day, key])
            value = float(row["shares"]) * float(row["price"])
            compare("market_value", float(row["market_value"]), value, [day, key])
            marked += value
        compare("position_count", len(by_day["positions"][day]),
                sum(lot["shares"] > 1e-8 for lot in lots.values()), day)
        compare("daily_cash", float(daily["cash"]), cash, day)
        compare("daily_receivable", float(daily["receivable"]), receivable, day)
        compare("daily_nav", float(daily["nav"]), cash + marked + receivable, day)
        compare("daily_cost", float(daily["cost"]), daily_fees + daily_slippage, day)
        for row in by_day["distributions"][day]:
            if row["kind"] != "record":
                continue
            key = row["ts_code"], int(row["entry"])
            identity = tuple(row[name] for name in (
                "ts_code", "entry", "event_record", "event_ex", "event_pay",
                "event_listing", "event_cash", "event_stock"))
            compare("registered_shares", float(row["shares"]), lots[key]["shares"], day)
            registered[identity] = float(row["shares"])
    compare("sale_count", len(tables["trades"]), len(sales), "trades")
    for row, sale in zip(tables["trades"], sales, strict=True):
        key = row["ts_code"], int(row["entry"])
        if key != sale["key"]:
            raise ValueError("trade_lot_mismatch")
        for name in ("cost", "proceeds", "dividend", "exit", "shares"):
            compare(f"trade_{name}", float(row[name]), sale[name], key)
    return {"passed": not errors, "errors": errors, "maximum_errors": dict(maximum_errors),
            "total_fees": total_fees, "final_cash": cash, "final_receivable": receivable,
            "lot_count": len(lots), "days": len(tables["daily"]),
            "capacity": audit_capacity(tables["fills"])}
