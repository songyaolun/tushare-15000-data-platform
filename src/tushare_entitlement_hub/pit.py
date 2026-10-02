"""只使用截止时点已被系统观察到的版本，并明确复权锚点。"""

import json
from copy import deepcopy
from datetime import UTC, datetime, time
from math import isfinite
from typing import Any

from .standard import (
    FINANCIAL_APIS,
    PERIOD_FIELDS,
    SNAPSHOT_APIS,
    normalize_date,
    normalize_null,
    normalize_timestamp,
)

REVISION_KEY_FIELDS = frozenset(
    {
        "update_flag",
        "revision",
        "revision_id",
        "revision_number",
        "version",
        "version_id",
        "ann_date",
        "f_ann_date",
        "observed_at",
        "response_hash",
        "source_hash",
    }
)


def time_boundary(value: Any, *, end_of_day: bool = False) -> datetime:
    """将日期或带时区时间转为 UTC 边界，不把无时区的时间当成已知时刻。

    输入日粒度日期时按 end_of_day 选择 UTC 日初或日末；时间输入必须带时区。
    例如 time_boundary('2024-01-02').isoformat() 返回 '2024-01-02T00:00:00+00:00'。
    日粒度披露采用日末，避免在当天披露时刻未知时过早使用数据。
    """
    if isinstance(value, datetime):
        return datetime.fromisoformat(normalize_timestamp(value))
    try:
        day = datetime.strptime(normalize_date(value), "%Y-%m-%d").date()
    except ValueError:
        return datetime.fromisoformat(normalize_timestamp(value))
    boundary = time.max if end_of_day else time.min
    return datetime.combine(day, boundary, tzinfo=UTC)


def is_financial(row: dict[str, Any]) -> bool:
    """识别需要披露时间的财务行，避免把市场日期强制当作公告日期。

    输入记录，输出是否属于财务数据。接口和分类优先，原始财务字段作为补充。
    例如 is_financial({'source_api': 'income'}) 返回 True。
    """
    return (
        row.get("source_api") in FINANCIAL_APIS
        or str(row.get("dataset_category", "")).lower()
        in {"financial", "finance", "financials", "fundamentals"}
        or ("end_date" in row and "report_type" in row)
    )


def _availability(row: dict[str, Any], cutoff: datetime) -> tuple[datetime, datetime] | None:
    """检查一行的必要披露、生效、失效与观察时间，未知必要时间直接排除。

    输入记录和 UTC 截止时刻；输出观察时刻及最近可用边界，不合格则返回 None。
    例如仅含 {'end_date': '2024-03-31'} 的记录返回 None，因为没有真实观察时间。
    """
    try:
        observation = datetime.fromisoformat(normalize_timestamp(row.get("observed_at")))
        if observation > cutoff:
            return None
        required = row.get("pit_required_fields", [])
        if not isinstance(required, list) or any(
            not isinstance(field, str) or normalize_null(row.get(field)) is None
            for field in required
        ):
            return None
        disclosures = [
            row[field]
            for field in ("ann_date", "f_ann_date")
            if normalize_null(row.get(field)) is not None
        ]
        if is_financial(row) and not disclosures:
            return None
        boundaries = [time_boundary(value, end_of_day=True) for value in disclosures]
        effective_fields = (
            "effective_at",
            "effective_time",
            "effective_date",
            "in_date",
            "start_date",
            "trade_date",
        )
        effective_values = [
            row[field] for field in effective_fields if normalize_null(row.get(field)) is not None
        ]
        if not disclosures and not effective_values:
            snapshot = (
                row.get("collection_mode") == "snapshot"
                or row.get("source_api") in SNAPSHOT_APIS
                or row.get("availability_basis") == "observed_snapshot"
            )
            if not snapshot:
                return None
            boundaries.append(observation)
        boundaries.extend(time_boundary(value) for value in effective_values)
        for field in required:
            if field in {"end_date", "report_date", "out_date", "observed_at"} | PERIOD_FIELDS:
                continue
            boundaries.append(time_boundary(row[field], end_of_day=field.endswith("ann_date")))
        if any(boundary > cutoff for boundary in boundaries):
            return None
        for field in ("out_date", "valid_to", "effective_to"):
            if normalize_null(row.get(field)) is not None and time_boundary(row[field]) <= cutoff:
                return None
        return observation, max(boundaries, default=observation)
    except (ValueError, TypeError, OverflowError):
        return None


def as_of(
    rows: list[dict[str, Any]],
    cutoff: str,
    primary_key: list[str],
) -> list[dict[str, Any]]:
    """选出截止时点已可见、已生效且已被本系统观察到的最新业务版本。

    输入包含历史版本的记录列表、截止日期或带时区时间以及业务主键；输出深拷贝列表。
    不删除输入版本。日期截止表示 UTC 日末；日粒度公告按 UTC 日末判定以避免日内前视。
    主键剔除 update_flag、revision/version 标记、观察时间和来源哈希；财务再剔除公告时间，
    事件记录保留公告日期这一业务身份。保留 report_type、end_date 等区分口径的字段。
    失效区间采用 [in_date, out_date)。
    财务必须有 ann_date 或 f_ann_date；所有行必须有 observed_at，未知必要时间则排除。
    声明 snapshot 或明确属于静态快照的接口可以从观察时刻起可见，绝不回溯快照内容。
    例如同一证券同一报告期的两版记录分别在 5 月和 7 月观察到，cutoff='2024-06-30'
    只返回 5 月版，即使 7 月版声称其公告日在 4 月。排序时间完全相同但内容冲突时不猜选。
    """
    cutoff_time = time_boundary(cutoff, end_of_day=True)
    disclosure_keys = {"ann_date", "f_ann_date"}
    generic_revision_keys = REVISION_KEY_FIELDS - disclosure_keys
    business_key = [field for field in primary_key if field not in generic_revision_keys]
    if not business_key:
        raise ValueError("business_primary_key_required")
    selected: dict[str, tuple[tuple[datetime, datetime], dict[str, Any], str]] = {}
    ambiguous: set[str] = set()
    for row in rows:
        row_key = [
            field for field in business_key if field not in disclosure_keys or not is_financial(row)
        ]
        if not row_key:
            continue
        if any(normalize_null(row.get(field)) is None for field in row_key):
            continue
        if any(isinstance(row[field], (dict, list, set)) for field in row_key):
            continue
        rank = _availability(row, cutoff_time)
        if rank is None:
            continue
        key = json.dumps([row[field] for field in row_key], sort_keys=True, default=str)
        fingerprint = json.dumps(row, sort_keys=True, default=str)
        previous = selected.get(key)
        if previous is None or rank > previous[0]:
            selected[key] = (rank, row, fingerprint)
            ambiguous.discard(key)
        elif rank == previous[0] and fingerprint != previous[2]:
            ambiguous.add(key)
    return [deepcopy(value[1]) for key, value in selected.items() if key not in ambiguous]


def adjust_prices(
    prices: list[float],
    factors: list[float],
    anchor_factor: float,
    mode: str,
) -> list[float]:
    """按调用方显式选择的锚点计算前复权或后复权，不读取未来因子。

    输入等长价格、逐日复权因子、有限正锚点及 qfq/hfq 模式；输出等长复权价格。
    两种模式均为 price × factor ÷ anchor_factor：qfq 锚点为分析截止日的因子，
    hfq 锚点为调用方固定的历史基准因子，不能默认以 1 替代真实基准。
    例如 adjust_prices([10.0], [2.0], 4.0, 'qfq') 返回 [5.0]；
    adjust_prices([10.0], [2.0], 0.5, 'hfq') 返回 [40.0]。
    长度不等、未知模式、非有限数或非正因子抛出不含原始值的 ValueError。
    """
    if mode not in {"qfq", "hfq"}:
        raise ValueError("unsupported_adjustment_mode")
    if len(prices) != len(factors):
        raise ValueError("price_factor_length_mismatch")
    if isinstance(anchor_factor, bool) or not isfinite(anchor_factor) or anchor_factor <= 0:
        raise ValueError("invalid_anchor_factor")
    adjusted: list[float] = []
    for price, factor in zip(prices, factors, strict=True):
        if isinstance(price, bool) or not isfinite(price):
            raise ValueError("invalid_price")
        if isinstance(factor, bool) or not isfinite(factor) or factor <= 0:
            raise ValueError("invalid_adjustment_factor")
        result = price * (factor / anchor_factor)
        if not isfinite(result):
            raise ValueError("nonfinite_adjusted_price")
        adjusted.append(result)
    return adjusted
