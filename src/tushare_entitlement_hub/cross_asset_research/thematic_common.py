"""主题、预期和基金研究的有限证据与数值辅助，不创建隐含关系。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from heapq import nsmallest
from math import isfinite
from typing import Any

from ..equity_research.common import latest, metric
from ..equity_research.models import Metric, ResearchInputs, ResearchRow
from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef
from ..standard import normalize_date
from ..store import encode

EVIDENCE_SAMPLE = 8
DETAIL_SAMPLE = 32
OBSERVATION_REASON_CODES = frozenset(
    {
        "directory_identity_observed_only",
        "source_local_entity_only",
        "consumption_fields_missing",
        "rows_excluded_by_adapter",
        "row_unavailable",
        "future_business_date",
        "quarantined_or_unknown_quality",
    }
)


def date_text(value: str | None) -> str | None:
    """规范化明确的日级字段，缺失、季度和非法值保持未知。

    输入'20260930'返回'2026-09-30'；输入'2026Q3'返回None，不补季度末。
    """
    try:
        return normalize_date(value) if value is not None else None
    except ValueError:
        return None


def row_key(row: ResearchRow) -> tuple[str, str, str]:
    """给已适配来源行生成稳定顺序，不序列化整个载荷。

    输入一条daily/r1行，输出(数据日,公开时点,'daily/r1')，用于确定性抽样。
    """
    return (
        row.availability.data_as_of or "",
        row.availability.public_available_at or "",
        row.source_api + "/" + str(row.raw["source_record_id"]),
    )


def accepted_groups(
    inputs: ResearchInputs, apis: Iterable[str]
) -> dict[str, dict[str, tuple[ResearchRow, ...]]]:
    """一次遍历已适配输入，按显式实体与来源分组，并再次封闭时点和隔离边界。

    输入实体A的两条可见行与一条未来行，返回A的两条行；无实体行不编造身份。
    来源或实体有硬拒绝原因时保留实体入口，但不让该组数据参与计算。
    目录／来源内身份的观察注记和部分消费字段缺失允许已有字段参与观察，缺口仍保留。
    已排除行的未来或质量原因不连带清空健康保留行；每条保留行仍重验自身可见性与质量。
    """
    groups: dict[str, dict[str, list[ResearchRow]]] = {}
    cutoff = time_boundary(inputs.analysis_as_of)
    for api in apis:
        for row in inputs.rows(api):
            if row.entity_id is None:
                continue
            bucket = groups.setdefault(row.entity_id, {}).setdefault(api, [])
            if (
                set(inputs.reasons(api, row.entity_id)) - OBSERVATION_REASON_CODES
                or time_boundary(row.availability.available_at) > cutoff
                or row.raw.get("quality_tier") in {"quarantine", "unknown"}
                or row.raw.get("is_quarantined") is True
            ):
                continue
            bucket.append(row)
    return {
        entity: {api: tuple(rows) for api, rows in sources.items()}
        for entity, sources in groups.items()
    }


def sampled_refs(
    rows: Iterable[ResearchRow], fields: Sequence[str], limit: int = EVIDENCE_SAMPLE
) -> tuple[EvidenceRef, ...]:
    """从全量输入确定性抽取字段引用；不将样本数量用于业务分母。

    输入100条成员及('con_code',)，返回最多8条来源证据，成员计数仍由调用方算100。
    """
    materialized = tuple(rows)
    return tuple(
        ref
        for field in fields
        for row in nsmallest(limit, (row for row in materialized if field in row.raw), key=row_key)
        for ref in row.evidence(field)
    )


def observation_summary(rows: Sequence[ResearchRow], fields: Sequence[str]) -> dict[str, Any]:
    """形成完整计数、字段覆盖及确定性来源摘要，避免复制原始载荷。

    输入十条记录，输出row_count=10、逐字段非空计数和来源身份SHA256；无样本截断计数。
    摘要只含来源身份及已选择业务字段，不处理凭证、运行环境或无关原始字段。
    """
    fingerprint = sha256()
    for row in sorted(rows, key=row_key):
        item = [row.source_api, row.raw["source_record_id"]]
        item.extend(row.text(field) for field in fields)
        fingerprint.update(encode(item).encode("utf-8"))
        fingerprint.update(b"\n")
    return {
        "row_count": len(rows),
        "field_observed_counts": {
            field: sum(row.text(field) is not None for row in rows) for field in fields
        },
        "source_digest": fingerprint.hexdigest(),
        "evidence_sample_limit": EVIDENCE_SAMPLE,
    }


def number(row: ResearchRow | None, field: str, unit: str | None = None) -> float | None:
    """读取认证有限值，拒绝单位、币种及隔离质量不满足的事实。

    输入认证amount=100元得到100；未认证原始amount='100'得到None。
    """
    return None if row is None else row.number(field, unit)


def finite(value: float | None) -> float | None:
    """将算术溢出变为未知，不把无穷写入JSON或结果模型。

    输入float('inf')返回None，输入-2返回-2，输入None仍返回None。
    """
    return value if value is not None and isfinite(value) else None


def comparable(left: ResearchRow, left_field: str, right: ResearchRow, right_field: str) -> bool:
    """只在两个合格数值的单位、量纲和币种完全相同时允许跨字段算术。

    输入CNY元净值与CNY元收盘价返回True；元与万份、CNY与HKD均返回False。
    """
    a, b = left.numeric.get(left_field), right.numeric.get(right_field)
    return (
        number(left, left_field) is not None
        and number(right, right_field) is not None
        and a is not None
        and b is not None
        and (a.unit, a.dimension, a.currency) == (b.unit, b.dimension, b.currency)
    )


def scalar(name: str, row: ResearchRow | None, field: str, unit: str) -> Metric:
    """把单个来源认证值变成指标，保留具体字段证据和缺口原因。

    输入row的pct_change=0.02 fraction返回值0.02；缺证书返回unit_or_value_unverified。
    有符号百分比以ratio输出，避免将-2%错误限制为非负比例。
    """
    return metric(
        name,
        number(row, field, unit),
        "ratio" if unit == "fraction" else unit,
        () if row is None else row.evidence(field),
        () if number(row, field, unit) is not None else ("unit_or_value_unverified",),
    )


def latest_snapshot(rows: Sequence[ResearchRow]) -> tuple[ResearchRow, ...]:
    """取最大已知数据日的完整横截面，不取任意单条成员或以观察日回填数据日。

    输入两个日期的五条成员，返回较新日的全部成员；所有数据日未知返回空元组。
    """
    day = max((row.availability.data_as_of or "" for row in rows), default="")
    return tuple(row for row in rows if day and row.availability.data_as_of == day)


def unique_latest_by(
    rows: Sequence[ResearchRow], fields: Sequence[str]
) -> tuple[tuple[ResearchRow, ...], int]:
    """按显式业务键选最新唯一版本，同序来源冲突单独计数。

    输入同一con_code两天记录输出较新版本；同一日两份不同记录输出冲突数1。
    缺键行也计入冲突，不丢掉分母后宣称完整覆盖。
    """
    indexed: dict[tuple[str, ...], list[ResearchRow]] = {}
    conflicts = 0
    for row in rows:
        key = tuple(row.text(field) or "" for field in fields)
        if not all(key):
            conflicts += 1
            continue
        indexed.setdefault(key, []).append(row)
    result = []
    for key in sorted(indexed):
        selected = latest(tuple(indexed[key]))
        if selected is None:
            conflicts += 1
        else:
            result.append(selected)
    return tuple(result), conflicts


def text_values(rows: Sequence[ResearchRow], field: str) -> dict[str, Any]:
    """汇总业务标签的去重范围，长文本只提供长度和摘要，不灌入结果。

    输入重复管理人名称A、A、B得到distinct_count=2和有界样本；样本不决定计数。
    """
    values = sorted({value for row in rows if (value := row.text(field)) is not None})
    return {
        "distinct_count": len(values),
        "samples": [value[:256] for value in values[:DETAIL_SAMPLE]],
        "values_digest": sha256(encode(values).encode("utf-8")).hexdigest(),
    }
