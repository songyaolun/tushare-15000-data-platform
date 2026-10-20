"""只读正式参考清单，验证ETF目标历史、分红与价格来源后才放行回放。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..reference_storage import verified_files
from .engine import Bar, Dividend


def date_key(value: Any) -> str:
    """统一接口日期为八位字符串；例如2021-10-18转换为20211018，空值保留空。"""
    return str(value).replace("-", "")[:8] if value else ""


def source_rows(root: Path, api: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """验证正式清单及文件完整性后读JSONL；例如缺清单拒绝，不把空值伪造为零行情。"""
    manifest = json.loads((root / "datasets" / api / "manifest.json").read_text())
    if not manifest.get("collection_status", "").startswith("complete"):
        raise ValueError("etf_source_incomplete:" + api)
    if not verified_files(manifest):
        raise ValueError("etf_source_integrity_failed:" + api)
    file = manifest.get("files", {}).get("accepted", {})
    if not file.get("jsonl"):
        raise ValueError("etf_accepted_source_missing:" + api)
    rows = [json.loads(line) for line in Path(file["jsonl"]).read_text().splitlines()]
    evidence = {"api": api, "parameters": manifest["parameters"],
                "observed_at": manifest["observed_at"], "rows": len(rows),
                "sha256": file["sha256"], "path": file["jsonl"],
                "collection_status": manifest["collection_status"]}
    quarantined = manifest.get("files", {}).get("quarantine", {})
    if quarantined.get("rows", 0):
        if api != "fund_div":
            raise ValueError("etf_quarantined_price_source")
        additional = [json.loads(line)
                      for line in Path(quarantined["jsonl"]).read_text().splitlines()]
        if any(row.get("issues") != ["conflicting_business_key"] for row in additional):
            raise ValueError("etf_dividend_quarantine_not_projectable")
        rows.extend(additional)
        evidence.update(cash_projection_quarantine_rows=len(additional),
                        quarantine_path=quarantined["jsonl"],
                        quarantine_sha256=quarantined["sha256"],
                        formal_quality_status=manifest.get("status", "partial"),
                        projection_policy="all cash entitlement fields must agree; "
                        "base_unit/net_ex_date not used for cash entitlement")
    return rows, evidence


def unique_dates(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """按日期建立唯一索引，重复立即拒绝；例：两条同日行情抛duplicate_etf_date。"""
    result = {date_key(row.get(field)): row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise ValueError("duplicate_or_missing_etf_date")
    return result


def dividend_events(rows: list[dict[str, Any]], start: str, end: str) -> list[Dividend]:
    """取目标期内已实施现金分红、去除完全重复事件并拒绝冲突或缺支付日期。

    例：实施记录record=20210101、ex=20210104、pay=20210105、div_cash=.1转现金事件。
    source中的每份派息已经为元，base_unit不是派息除数。
    """
    events: dict[str, Dividend] = {}
    signatures: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        ex_date = date_key(row.get("ex_date"))
        if not start <= ex_date <= end or row.get("div_proc") not in {"实施", "已实施"}:
            continue
        record, payment = date_key(row.get("record_date")), date_key(row.get("pay_date"))
        announced = date_key(row.get("imp_anndate") or row.get("ann_date"))
        amount = float(row.get("div_cash") or 0)
        if not (announced and record and payment and announced <= record < ex_date <= payment):
            raise ValueError("etf_dividend_dates_invalid")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("etf_dividend_amount_invalid")
        event = Dividend(ex_date, record, ex_date, payment, amount)
        signature = (row.get("ts_code"), announced, date_key(row.get("ann_date")),
                     record, ex_date, payment, amount, row.get("div_proc"))
        if ex_date in events and signatures[ex_date] != signature:
            raise ValueError("conflicting_etf_dividend")
        events[ex_date] = event
        signatures[ex_date] = signature
    return sorted(events.values(), key=lambda item: item.ex_date)


def prepare_data(
    rows: dict[str, list[dict[str, Any]]], start: str = "20210301", end: str = "20261016"
) -> tuple[list[str], dict[str, Bar], list[Dividend], dict[str, Any]]:
    """核验510300实际历史、因子和限价逐日覆盖；缺一天即阻止生产报告。

    输入五个接口记录集合，输出日历、原始价格Bar、现金事件和审计摘要。
    例：calendar有两个交易日而fund_daily只有一个时抛etf_history_calendar_mismatch。
    """
    for api in ("fund_daily", "fund_adj", "fund_div", "etf_limit"):
        if any(row.get("ts_code") != "510300.SH" for row in rows[api]):
            raise ValueError("unexpected_etf_security")
    calendar = sorted(date_key(row["cal_date"]) for row in rows["trade_cal"]
                      if int(row["is_open"]) == 1 and start <= date_key(row["cal_date"]) <= end)
    if len(calendar) != len(set(calendar)) or not calendar:
        raise ValueError("invalid_etf_source_calendar")
    quotes = unique_dates(rows["fund_daily"], "trade_date")
    factors = unique_dates(rows["fund_adj"], "trade_date")
    limits = unique_dates(rows["etf_limit"], "trade_date")
    for source in (quotes, factors, limits):
        if set(source) != set(calendar):
            raise ValueError("etf_history_calendar_mismatch")
    events = dividend_events(rows["fund_div"], start, end)
    bars: dict[str, Bar] = {}
    for day in calendar:
        quote, factor, limit = quotes[day], factors[day], limits[day]
        values = [float(quote[field]) for field in ("open", "high", "low", "close", "vol")]
        factor_value = float(factor["adj_factor"])
        upper, lower = float(limit["up_limit"]), float(limit["down_limit"])
        if any(not math.isfinite(value) or value <= 0
               for value in (*values[:4], factor_value, upper, lower)) or (
                   not math.isfinite(values[4]) or values[4] < 0):
            raise ValueError("invalid_etf_price")
        if not (values[2] <= values[0] <= values[1] and values[2] <= values[3] <= values[1]
                and lower < upper and lower - 1e-8 <= values[2]
                and values[1] <= upper + 1e-8):
            raise ValueError("invalid_etf_price_range")
        bars[day] = Bar(values[0], values[1], values[2], values[3], values[4],
                        factor_value, upper, lower)
    event_dates = {event.ex_date for event in events}
    changes = []
    for previous, current in zip(calendar, calendar[1:], strict=False):
        if not math.isclose(bars[previous].factor, bars[current].factor, rel_tol=1e-9):
            changes.append(current)
            if current not in event_dates:
                raise ValueError("unexplained_etf_adjustment")
    if any(event.record_date not in calendar or event.ex_date not in calendar for event in events):
        raise ValueError("etf_dividend_calendar_missing")
    factor_residuals = []
    for event in events:
        position = calendar.index(event.ex_date)
        prior_bar, current_bar = bars[calendar[position - 1]], bars[event.ex_date]
        if prior_bar.close <= event.cash_per_share:
            raise ValueError("etf_cash_adjustment_invalid")
        expected = prior_bar.close / (prior_bar.close - event.cash_per_share)
        observed = current_bar.factor / prior_bar.factor
        residual = observed / expected - 1
        if abs(residual) > .0002:
            raise ValueError("etf_cash_adjustment_mismatch")
        factor_residuals.append({"ex_date": event.ex_date, "relative_residual": residual})
    return calendar, bars, events, {
        "security": "510300.SH", "start": calendar[0], "end": calendar[-1],
        "sessions": len(calendar), "dividend_events": len(events), "factor_change_dates": changes,
        "validation": "complete_actual_etf_history", "missing_price_policy": "fail_closed",
        "listing_date": "20120528", "listing_basis": "pre-existing matching fund_basic/etf_basic",
        "cash_factor_reconciliation": factor_residuals,
        "factor_rounding_relative_tolerance": .0002,
    }


def load_data(root: Path) -> tuple[list[str], dict[str, Bar], list[Dividend], dict[str, Any]]:
    """从专用参考目录加载五接口并返回可复查证据；例如load_data(reference)不联网。"""
    rows, sources = {}, []
    for api in ("fund_daily", "fund_adj", "fund_div", "etf_limit", "trade_cal"):
        records, evidence = source_rows(root, api)
        rows[api] = records
        sources.append(evidence)
    calendar, bars, events, audit = prepare_data(rows)
    audit["sources"] = sources
    return calendar, bars, events, audit
