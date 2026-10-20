"""公告年龄与公告后价格反应；只读历史前缀，不训练或改变候选资格。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .earnings_features import forecast_event_features

FEATURE_NAMES = ("forecast_age_sessions", "forecast_post_event_excess")


@dataclass(frozen=True)
class ForecastEventReaction:
    """完整候选轴及事件审计键；例无预告行features=[NaN,NaN]、event_ids=-1。

    event_keys为去重后的(code, period, announcement ISO日期)，event_ids逐行引用它。
    仅有效、未过期且前两项预告增长有限的版本可以绑定；价格缺失不抹去有效年龄。
    """

    features: NDArray[np.float32]
    event_ids: NDArray[np.int64]
    event_keys: tuple[tuple[str, str, str], ...]


def industry_relative_log_returns(
    adjusted_close: NDArray[np.float64],
    sector_index: NDArray[np.float64],
    valid: NDArray[np.float64],
    *,
    minimum_peers: int = 5,
) -> NDArray[np.float64]:
    """按前一日行业、资格和当前可核验价格计算剔除自身的行业超额log收益。

    输入同形状日×证券矩阵；行业非负整数、未知为NaN或负数，valid==1才可参与。
    每日同行简单收益取有限价格的合格同行均值，至少minimum_peers只且不含自身。
    输出首日与不可计算位置NaN；普通缺报价不补零，非正价格不参与计算。
    例自身涨10%、五只同行各涨2%，输出log(1.1)-log(1.02)。
    同一证券价格整段乘正常数不改变结果；不读取d之后的价格或行业。
    """
    arrays = (adjusted_close, sector_index, valid)
    if (
        type(minimum_peers) is not int
        or minimum_peers < 1
        or any(array.ndim != 2 or array.dtype.kind not in "fiu" for array in arrays)
        or any(array.shape != adjusted_close.shape for array in arrays)
        or any(np.isinf(array).any() for array in arrays)
        or np.any(
            np.isfinite(sector_index)
            & (sector_index >= 0)
            & (sector_index != np.floor(sector_index))
        )
        or np.any(np.isfinite(valid) & ~np.isin(valid, [0, 1]))
    ):
        raise ValueError("invalid_event_price_axes")
    output = np.full(adjusted_close.shape, np.nan, dtype=np.float64)
    for day in range(1, len(adjusted_close)):
        previous, current = adjusted_close[day - 1], adjusted_close[day]
        sectors = sector_index[day - 1]
        known = (
            (valid[day - 1] == 1)
            & np.isfinite(sectors)
            & (sectors >= 0)
            & np.isfinite(previous)
            & np.isfinite(current)
            & (previous > 0)
            & (current > 0)
        )
        columns = np.flatnonzero(known)
        if not len(columns):
            continue
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            returns = current[columns] / previous[columns] - 1
        representable = np.isfinite(returns) & (returns > -1)
        columns, returns = columns[representable], returns[representable]
        if not len(columns):
            continue
        _, inverse, counts = np.unique(sectors[columns], return_inverse=True, return_counts=True)
        totals = np.bincount(inverse, weights=returns)
        enough = counts[inverse] - 1 >= minimum_peers
        peer_returns = (totals[inverse[enough]] - returns[enough]) / (counts[inverse[enough]] - 1)
        output[day, columns[enough]] = np.log1p(returns[enough]) - np.log1p(peer_returns)
    return output


def candidate_forecast_event_features(
    rows: list[dict[str, Any]],
    calendar: list[str],
    codes: list[str],
    candidate_days: NDArray[np.int64],
    candidate_securities: NDArray[np.int64],
    relative_log_returns: NDArray[np.float64],
    *,
    maximum_age_days: int = 180,
) -> ForecastEventReaction:
    """绑定现有预告版本，返回年龄与从该版本公告后开始的行业超额价格反应。

    rows为forecast原始行，复用原三列的去重、冲突、max(ann,first_ann)、报告期优先。
    公告日A严格之后首交易日E才可见；年龄为log1p(t-E)，180自然日后整组未知。
    反应累加E到t的逐日相对log收益，等价公告日前最后收盘到信号收盘的相对涨幅。
    例周五公告、周一信号、周一相对log收益.02，输出[0,.02]；周二年龄log(2)。
    任何中间价格或同行路径未知，反应NaN但保留年龄；公告早于日历起点没有完整
    交易日年龄和价格锚点，因此两列均未知。候选顺序和重复候选保留，绝不删除行。
    """
    dates = [date.fromisoformat(day) for day in calendar]
    if (
        type(maximum_age_days) is not int
        or maximum_age_days < 1
        or dates != sorted(set(dates))
        or len(set(codes)) != len(codes)
        or candidate_days.ndim != 1
        or candidate_days.shape != candidate_securities.shape
        or candidate_days.dtype.kind not in "iu"
        or candidate_securities.dtype.kind not in "iu"
        or np.any(candidate_days < 0)
        or np.any(candidate_days >= len(dates))
        or np.any(candidate_securities < 0)
        or np.any(candidate_securities >= len(codes))
        or relative_log_returns.shape != (len(dates), len(codes))
        or relative_log_returns.dtype.kind not in "fiu"
        or np.isinf(relative_log_returns).any()
    ):
        raise ValueError("invalid_event_candidate_axis")
    events = forecast_event_features(rows)
    keys = tuple(
        (event["ts_code"], event["period"], event["announcement_date"].isoformat())
        for event in events
    )
    result = np.full((len(candidate_days), 2), np.nan, dtype=np.float32)
    event_ids = np.full(len(candidate_days), -1, dtype=np.int64)
    by_code: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for identifier, event in enumerate(events):
        by_code[event["ts_code"]].append((identifier, event))
    ordinals = np.array([day.toordinal() for day in dates])
    for security in np.unique(candidate_securities):
        positions = np.flatnonzero(candidate_securities == security)
        positions = positions[np.argsort(candidate_days[positions], kind="stable")]
        versions = sorted(
            by_code[codes[int(security)]],
            key=lambda item: (item[1]["available_date"], item[1]["period"]),
        )
        active: dict[str, tuple[int, dict[str, Any]]] = {}
        pointer = 0
        daily = relative_log_returns[:, security]
        finite = np.isfinite(daily)
        # 未知计数和数值前缀分别保留；补零只供求和，任何未知仍关闭整个事件路径。
        cumulative = np.r_[0.0, np.cumsum(np.where(finite, daily, 0.0))]
        unknown = np.r_[0, np.cumsum(~finite)]
        for position in positions:
            day = int(candidate_days[position])
            while pointer < len(versions) and versions[pointer][1]["available_date"] <= dates[day]:
                identifier, event = versions[pointer]
                active[event["period"]] = (identifier, event)
                pointer += 1
            if not active:
                continue
            identifier, latest = active[max(active)]
            if (
                dates[day] - latest["announcement_date"]
            ).days > maximum_age_days or not np.isfinite(latest["features"][:2]).all():
                continue
            visible = int(
                np.searchsorted(ordinals, latest["announcement_date"].toordinal(), side="right")
            )
            # 只有完整交易日历才可计算交易日年龄；左截断事件也不能假装刚刚公布。
            if visible == 0:
                continue
            event_ids[position] = identifier
            result[position, 0] = np.log1p(day - visible)
            if unknown[day + 1] == unknown[visible]:
                result[position, 1] = cumulative[day + 1] - cumulative[visible]
    return ForecastEventReaction(result, event_ids, keys)
