"""右侧交易结构的确定性合成行情与父证据，不读取真实供应商数据。"""

from dataclasses import replace
from datetime import date, timedelta

from test_research_candidates import research_inputs, research_parents, research_row

from tushare_entitlement_hub.equity_research.common import metric


def right_side_inputs(count=2, *, pattern="breakout", minimum=2):
    """生成61日回调、抬高低点与突破；例：默认现价10.12、整理高点10.02、历史峰12.02。"""
    inputs = research_inputs(count, minimum=minimum, flat=True)
    days = []
    cursor = date(2026, 9, 30)
    while len(days) < 61:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    days.reverse()
    rows = []
    for identity in inputs.entity_ids:
        closes = [12.0] * 21 + [12 - 0.3 * (index + 1) for index in range(10)]
        closes += [9.0] * 9 + [9.5] * 10 + [10.0] * 10 + [10.12]
        amounts = [100.0] * 60 + [130.0]
        if pattern == "retest":
            closes[-3:] = [10.12, 10.04, 10.08]
            amounts[-3:] = [130, 60, 80]
        elif pattern == "left_side":
            closes[-1] = 9.4
        elif pattern == "extended":
            closes[-1] = 10.8
        elif pattern == "late":
            closes[-11:] = [11.7] * 10 + [11.85]
        for index, day in enumerate(days):
            close = closes[index]
            fields = {
                "close": close,
                "high": close + 0.02,
                "low": close - 0.05,
                "vol": 100,
                "amount": amounts[index],
                "pct_chg": close / closes[index - 1] - 1 if index else 0,
                "trade_date": day,
            }
            if pattern == "retest" and index >= 58:
                fields["low"] = 10.02
            rows.append(
                research_row(identity, "daily", fields, day=day, record=f"daily-{identity}-{day}")
            )
    return replace(inputs, rows_by_api=inputs.rows_by_api | {"daily": tuple(rows)})


def right_side_parents(inputs, *, quality=80, flow_ratio=0.05):
    """提供可独立门禁的三父结果；例：flow_ratio=-0.1形成资金流出反例。"""
    fundamental, governance, funding = research_parents(inputs, quality=quality)
    updated = []
    for parent in funding:
        refs = parent.envelope.evidence_refs
        updated.append(
            replace(
                parent,
                metrics=(metric("large_flow_ratio", flow_ratio, "ratio", refs),),
                envelope=replace(parent.envelope, source_as_of={"moneyflow": "2026-09-30"}),
            )
        )
    return fundamental, governance, tuple(updated)
