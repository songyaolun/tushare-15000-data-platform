"""业绩预告的公告版本特征；供应商重建历史，不认证当时不可变归档。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = (
    "forecast_growth_midpoint", "forecast_growth_interval_width", "forecast_growth_revision"
)


def forecast_event_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把预告按证券、报告期和公告日保留各版本，生成三个无量纲特征。

    输入forecast/forecast_vip原始字段。增幅中点、区间宽度均从百分数除以100，
    修正为同证券同报告期相邻不同公告日的中点差。first_ann_date不能倒替ann_date，
    update_flag没有可靠版本语义，不参与判断。公告只有日期，统一延迟一个自然日。
    同日经济字段相同的重复行合并；冲突保留一个未知事件，并中断后续修正比较。
    例：同期间1月5日增幅10–20、1月8日20–30，第二事件1月9日起为[.25,.1,.1]。
    无法解析的日期丢弃；缺增幅或倒置区间保留未知事件，绝不补零。
    """
    groups: dict[tuple[str, str], dict[date, set[tuple[float, float] | None]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        try:
            code = str(row["ts_code"])
            period = date.fromisoformat(str(row["end_date"])).isoformat()
            announcement = date.fromisoformat(str(row["ann_date"]))
            if row.get("first_ann_date"):
                announcement = max(announcement, date.fromisoformat(str(row["first_ann_date"])))
        except (KeyError, ValueError, TypeError):
            continue
        interval = None
        try:
            lower, upper = float(row["p_change_min"]), float(row["p_change_max"])
            if np.isfinite(lower) and np.isfinite(upper) and lower <= upper:
                interval = (lower, upper)
        except (KeyError, ValueError, TypeError):
            pass
        groups[(code, period)][announcement].add(interval)
    events: list[dict[str, Any]] = []
    for (code, period), versions in sorted(groups.items()):
        prior_midpoint = None
        for announcement, intervals in sorted(versions.items()):
            interval = next(iter(intervals)) if len(intervals) == 1 else None
            values = [float("nan")] * 3
            midpoint = None
            if interval is not None:
                lower, upper = interval
                midpoint = (lower + upper) / 200
                values[:2] = [midpoint, (upper - lower) / 100]
                if prior_midpoint is not None:
                    values[2] = midpoint - prior_midpoint
            events.append({
                "ts_code": code, "period": period, "announcement_date": announcement,
                "available_date": announcement + timedelta(days=1), "features": values,
                "conflict": len(intervals) > 1,
            })
            prior_midpoint = midpoint
    return events


def candidate_forecast_features(
    events: list[dict[str, Any]], calendar: list[str], codes: list[str],
    candidate_days: NDArray[np.int64], candidate_securities: NDArray[np.int64],
    *, maximum_age_days: int = 180,
) -> NDArray[np.float32]:
    """将已延迟的预告映射到候选轴，最新报告期优先，旧期修正不覆盖新期。

    输入证券与日历轴及candidate索引，返回N×3矩阵；无事件、过期或冲突保持NaN。
    180自然日覆盖两个季度的事件观察窗，仅限制新增特征有效期，不筛候选或改标签。
    例：1月5日公告，1月5日候选未知，1月6日候选取得值；6月后的旧事件过期。
    索引、形状或日期轴非法时抛ValueError，保证排序和前缀不变性。
    """
    dates = [date.fromisoformat(day) for day in calendar]
    if (maximum_age_days < 1 or dates != sorted(set(dates))
            or candidate_days.shape != candidate_securities.shape or candidate_days.ndim != 1
            or candidate_days.dtype.kind not in "iu" or candidate_securities.dtype.kind not in "iu"
            or np.any(candidate_days < 0) or np.any(candidate_days >= len(calendar))
            or np.any(candidate_securities < 0) or np.any(candidate_securities >= len(codes))):
        raise ValueError("invalid_forecast_candidate_axis")
    result = np.full((len(candidate_days), 3), np.nan, dtype=np.float32)
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_code[event["ts_code"]].append(event)
    for security in np.unique(candidate_securities):
        positions = np.flatnonzero(candidate_securities == security)
        positions = positions[np.argsort(candidate_days[positions], kind="stable")]
        versions = sorted(by_code[codes[int(security)]], key=lambda event: (
            event["available_date"], event["period"]
        ))
        active: dict[str, dict[str, Any]] = {}
        pointer = 0
        for position in positions:
            current = dates[int(candidate_days[position])]
            while pointer < len(versions) and versions[pointer]["available_date"] <= current:
                event = versions[pointer]
                active[event["period"]] = event
                pointer += 1
            if active:
                latest = active[max(active)]
                if (current - latest["announcement_date"]).days <= maximum_age_days:
                    result[position] = latest["features"]
    return result
