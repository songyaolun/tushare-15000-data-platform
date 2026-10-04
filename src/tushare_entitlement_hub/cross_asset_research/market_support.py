"""跨资产产品的认证数值、分层快照及有界证据摘要。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any

from ..equity_research.common import gate, make_result, metric
from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.numeric import TypedFact
from ..product_foundation.temporal import availability_gate
from ..store import digest, encode
from .thematic_common import OBSERVATION_REASON_CODES

SAMPLE_ROWS = 8
MAXIMUM_RESULT_BYTES = 8 * 1024 * 1024
LOCAL_OBSERVATION_CODES = OBSERVATION_REASON_CODES | {"market_timezone_uncertified"}
OBSERVED_SNAPSHOT_APIS = frozenset(
    {
        "fut_basic",
        "sge_basic",
        "fx_obasic",
        "hk_basic",
        "us_basic",
        "stock_company",
        "opt_basic",
        "rt_hk_k",
    }
)


def fact(row: ResearchRow, name: str) -> TypedFact | None:
    """读取与来源、实体和时间完全一致的认证数值，保留原币种及量纲。

    例如已认证 HKD close 可读取，未绑定实体、未知单位或其他行的事实返回 None。
    本函数不改变 ResearchRow.number 的人民币约束，也不从 raw 猜测数值。
    """
    value = row.numeric.get(name)
    if (
        value is None
        or value.value is None
        or value.reason_codes
        or value.quality_tier != "verified"
        or value.certification is None
        or not row.entity_id
        or value.entity_id != row.entity_id
        or value.field != name
        or value.availability != row.availability
        or value.certification.source_api != row.source_api
        or name not in row.raw
        or (value.dimension in {"money", "price"} and not value.currency)
    ):
        return None
    return value


def number(row: ResearchRow, name: str, dimensions: tuple[str, ...] = ()) -> float | None:
    """按认证量纲读取原币数值；例如 dimensions=('ratio',) 拒绝金额型 close。

    无量纲限制时仍要求完整 TypedFact 认证，缺失值永不解释成零。
    """
    value = fact(row, name)
    return (
        None if value is None or dimensions and value.dimension not in dimensions else value.value
    )


def comparable(left: ResearchRow, left_name: str, right: ResearchRow, right_name: str) -> bool:
    """校验同日、同单位、同币种和同量纲，实体关系由调用处另外证明。

    例如同日两个 USD 元价格返回 True，USD 和 HKD 即使都叫元也返回 False。
    """
    first, second = fact(left, left_name), fact(right, right_name)
    return bool(
        first is not None
        and second is not None
        and (
            left.source_api == right.source_api
            or left.availability.market_timezone is not None
            and right.availability.market_timezone is not None
        )
        and (first.unit, first.dimension, first.currency, first.availability.data_as_of)
        == (second.unit, second.dimension, second.currency, second.availability.data_as_of)
    )


def ratio(
    row: ResearchRow, numerator: str, denominator: str, *, change: bool = False
) -> float | None:
    """计算同口径比例或相对变化，分母须为正；例如 120/100 的 change 为 0.2。

    单位、币种或维度不一致返回 None，不将报价和成交金额相除。
    """
    first, second = number(row, numerator), number(row, denominator)
    if not comparable(row, numerator, row, denominator) or second is None or second <= 0:
        return None
    assert first is not None
    result = first / second - int(change)
    return result if isfinite(result) else None


def difference(
    row: ResearchRow, first: str, second: str, dimensions: tuple[str, ...] = ()
) -> float | None:
    """计算同口径差值；例如同为比例的长端 0.03 和短端 0.02 得到 0.01。

    返回值沿用来源认证单位，跨币种或缺认证返回 None。
    """
    left, right = number(row, first, dimensions), number(row, second, dimensions)
    value = (
        left - right
        if comparable(row, first, row, second) and left is not None and right is not None
        else None
    )
    return value if value is not None and isfinite(value) else None


def unit(row: ResearchRow, name: str, fallback: str = "source_unit") -> str:
    """给数值附原币种单位；例如 USD 元输出 USD:元，未知证书输出 source_unit。"""
    value = fact(row, name)
    return (
        fallback
        if value is None
        else f"{value.currency}:{value.unit}"
        if value.currency
        else str(value.unit)
    )


def binary_flag(row: ResearchRow, name: str) -> float | None:
    """读取严格的零一枚举，拒绝布尔值和任意非零数字的隐式真值转换。

    例如数值 1.0、整数 1 和文本 '1' 均得到 1.0；1.5、True 和文本 '1.0' 得 None。
    这是离散枚举解释，不把缺少单位认证的连续量当作可计算事实。
    """
    value = row.raw.get(name)
    if type(value) in {int, float} and value in {0, 1}:
        return float(value)
    if type(value) is str and value in {"0", "1"}:
        return float(value)
    return None


@dataclass
class Frame:
    """单来源、实体、细分维度的最新可见快照，保留全量计数与摘要。"""

    api: str
    entity_id: str
    scope: dict[str, str | None]
    fields: tuple[str, ...]
    rows: list[ResearchRow] = field(default_factory=list)
    observed_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0
    latest_period: str = ""
    reasons: set[str] = field(default_factory=set)
    scope_annotations: set[str] = field(default_factory=set)
    signatures: dict[str, str] = field(default_factory=dict)
    evidence_digest: Any = field(default_factory=sha256)
    supporting_references: list[EvidenceRef] = field(default_factory=list)
    supporting_reference_count: int = 0

    def add(self, inputs: ResearchInputs, row: ResearchRow) -> None:
        """流式处理一行；例如较新日期替换旧快照，超过容量后整组禁算。

        摘要覆盖每个输入来源身份及所有选定字段，采样不会改变完整计数。
        """
        self.observed_count += 1
        payload = [
            row.source_api,
            row.raw.get("source_record_id"),
            row.raw.get("contract_hash"),
            {name: row.raw.get(name) for name in self.fields},
        ]
        self.evidence_digest.update(encode(payload).encode())
        codes = set(inputs.reasons(row.source_api, row.entity_id or self.entity_id))
        self.reasons.update(codes - LOCAL_OBSERVATION_CODES)
        self.scope_annotations.update(codes & LOCAL_OBSERVATION_CODES)
        metadata_snapshot = row.source_api in OBSERVED_SNAPSHOT_APIS
        if (
            row.raw.get("source_api") != row.source_api
            or row.raw.get("source_record_id") != row.availability.source_record_id
        ):
            self.rejected_count += 1
            self.reasons.add("source_identity_unverified")
            return
        if (
            not availability_gate(row.availability, inputs.analysis_as_of).passed
            or row.availability.data_as_of is None
            and not metadata_snapshot
            or row.availability.data_as_of is not None
            and row.availability.data_as_of > inputs.analysis_as_of[:10]
            or row.raw.get("is_quarantined") is True
            or row.raw.get("quality_tier") in {"quarantine", "unknown"}
        ):
            self.rejected_count += 1
            self.scope_annotations.add("rows_rejected_time_or_quality")
            return
        period = row.availability.data_as_of or row.availability.observed_at[:10]
        if row.availability.data_as_of is None:
            self.scope_annotations.add("metadata_snapshot_observed_only")
        if period < self.latest_period:
            return
        if period > self.latest_period:
            self.latest_period, self.rows, self.signatures = period, [], {}
        signature = digest(payload[3])
        identity = (
            row.source_api
            + "/"
            + str(row.raw.get("business_group_key", row.raw.get("source_record_id")))
        )
        if identity in self.signatures:
            if self.signatures[identity] == signature:
                self.duplicate_count += 1
            else:
                self.reasons.add("source_version_conflict")
            return
        if len(self.rows) >= inputs.config.maximum_rows_per_entity_source:
            self.reasons.add("capacity_exceeded")
            return
        self.signatures[identity] = signature
        self.rows.append(row)

    def references(self, *fields: str) -> tuple[EvidenceRef, ...]:
        """返回最多八行的字段证据；例如千行快照仍有完整摘要而引用仅为样本。"""
        return tuple(
            ref for row in self.rows[:SAMPLE_ROWS] for ref in row.evidence(*(fields or self.fields))
        ) + tuple(self.supporting_references)

    def cite(self, references: Iterable[EvidenceRef]) -> None:
        """记录衍生计算的元数据和关系血缘，最多保留六十四条但摘要覆盖全部。

        例如近远月曲线同时引用合约 d_month 与 quote_unit，不只引用结算价格。
        """
        for reference in references:
            self.supporting_reference_count += 1
            self.evidence_digest.update(reference.to_json().encode())
            if len(self.supporting_references) < SAMPLE_ROWS * 8:
                self.supporting_references.append(reference)

    def value(
        self, name: str, value: float | None, output_unit: str = "ratio", *fields: str
    ) -> Metric:
        """生成门禁后的指标；例如容量溢出即使已算出 0.2 也输出 unknown。"""
        value = value if value is not None and isfinite(value) else None
        references = self.references(*fields)
        return metric(
            name,
            None if self.reasons or not references else value,
            output_unit,
            references,
            self.reasons,
        )

    def summary(self) -> dict[str, Any]:
        """返回计数、完整来源摘要和范围；例如 observed_count=100 不谎称八条样本为全量。"""
        return {
            "source_api": self.api,
            "scope": self.scope,
            "data_as_of": self.latest_period,
            "observed_count": self.observed_count,
            "snapshot_count": len(self.rows),
            "exact_duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "evidence_sample_rows": min(SAMPLE_ROWS, len(self.rows)),
            "evidence_complete": len(self.rows) <= SAMPLE_ROWS
            and self.supporting_reference_count <= SAMPLE_ROWS * 8
            and not self.reasons,
            "supporting_reference_count": self.supporting_reference_count,
            "evidence_digest": self.evidence_digest.hexdigest(),
            "reason_codes": sorted(self.reasons),
            "scope_annotations": sorted(self.scope_annotations),
        }


def frames(
    inputs: ResearchInputs, api: str, fields: tuple[str, ...], dimensions: tuple[str, ...] = ()
) -> tuple[Frame, ...]:
    """一次扫描按显式实体及细分维度分组；例如不同仓库/品级从不相加。

    未绑定行使用来源内上下文身份，附 entity_unverified，不能升级成证券结论。
    """
    result: dict[tuple[Any, ...], Frame] = {}
    for row in inputs.rows(api):
        scope = {name: row.text(name) for name in dimensions}
        key = (row.entity_id, *scope.values())
        if key not in result:
            identity = row.entity_id or "context:series:" + digest([api, scope])[:24] + ":observed"
            result[key] = Frame(api, identity, scope, fields)
            if row.entity_id is None:
                result[key].reasons.add("entity_unverified")
            if dimensions and any(value is None for value in scope.values()):
                result[key].reasons.add("scope_dimension_missing")
        result[key].add(inputs, row)
    return tuple(result.values())


def finish(
    inputs: ResearchInputs,
    product: str,
    frame: Frame,
    values: Iterable[Metric],
    details: dict[str, Any],
) -> ResearchResult:
    """组装标准信封并限制八 MiB；例如缺认证输出 incomplete，容量溢出输出 blocked。

    feature_profile 包含来源与细分范围，防止同实体多种观察覆盖彼此。
    """
    checks = [
        gate(reason, Truth.FALSE if "capacity" in reason else Truth.UNKNOWN, reason)
        for reason in sorted(frame.reasons)
    ]
    checks.extend(
        gate(reason, Truth.UNKNOWN, reason, hard=False)
        for reason in sorted(frame.scope_annotations)
    )
    result = make_result(
        inputs,
        product,
        frame.entity_id,
        metrics=values,
        gates=checks,
        references=frame.references(),
        profile=frame.api + "/" + digest(frame.scope)[:20],
        details={**frame.summary(), **details},
    )
    if len(result.to_json().encode()) > MAXIMUM_RESULT_BYTES:
        return make_result(
            inputs,
            product,
            frame.entity_id,
            gates=(gate("payload_capacity", Truth.FALSE, "result_payload_capacity_exceeded"),),
            profile=result.envelope.feature_profile,
            details={
                "source_api": frame.api,
                "observed_count": frame.observed_count,
                "evidence_digest": frame.evidence_digest.hexdigest(),
            },
        )
    return result


def run_sources(
    inputs: ResearchInputs,
    product: str,
    fields: dict[str, tuple[str, ...]],
    dimensions: dict[str, tuple[str, ...]],
    processor: Callable[[Frame], tuple[list[Metric], dict[str, Any]]],
) -> list[ResearchResult]:
    """执行逐接口独立加工；例如同代码港股与美股仍分来源、实体和 profile 输出。"""
    results = []
    for api, names in fields.items():
        for frame in frames(inputs, api, names, dimensions.get(api, ())):
            values, details = processor(frame)
            results.append(finish(inputs, product, frame, values, details))
    return results


def singleton(frame: Frame) -> ResearchRow | None:
    """只允许最新快照中的唯一业务行；例如两个不同版本不任意选择第一条。"""
    if len(frame.rows) != 1:
        frame.reasons.add("unique_snapshot_required")
        return None
    return frame.rows[0]


def spread(frame: Frame, field_name: str) -> tuple[float | None, str]:
    """计算同口径样本极差；例如银行报价 0.02、0.03 得到 0.01。

    全部样本须有认证且单位/币种一致；只有一份报价不能证明市场离散度为零。
    """
    if len(frame.rows) < 2:
        return None, "source_unit"
    first = frame.rows[0]
    values = [number(row, field_name) for row in frame.rows]
    if any(value is None for value in values) or not all(
        comparable(first, field_name, row, field_name) for row in frame.rows
    ):
        return None, unit(first, field_name)
    valid = [value for value in values if value is not None]
    return max(valid) - min(valid), unit(first, field_name)
