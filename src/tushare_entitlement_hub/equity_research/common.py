"""研究引擎共用的证据去重、特征、同行分位与信封构造。"""

from __future__ import annotations

from collections.abc import Iterable
from math import fsum
from typing import Any

from ..product_foundation.contracts import (
    ActionState,
    Coverage,
    DecisionEnvelope,
    EvidenceRef,
    FeatureContribution,
    Predicate,
    ProductState,
    QualityGateResult,
    Truth,
)
from ..product_foundation.features import PeerObservation, PercentileResult, percentile_ranks
from ..store import encode
from .models import RULE_VERSION, Metric, ResearchInputs, ResearchResult, ResearchRow


def unique_evidence(references: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    """按完整来源字段引用去重并稳定排序；例：同一引用两次只保留一次。

    VIP与普通接口的经济去重仍由具体产品处理，本函数不合并不同来源。
    """
    indexed = {ref.to_json(): ref for ref in references}
    return tuple(indexed[key] for key in sorted(indexed))


def metric(
    name: str,
    value: float | None,
    unit: str,
    references: Iterable[EvidenceRef] = (),
    reasons: Iterable[str] = (),
    *,
    not_applicable: bool = False,
) -> Metric:
    """构造带明确空值语义的特征；例：value=None得到unknown和metric_unavailable。

    已知不适用只能显式标注，不能输出0假装无风险。
    """
    return Metric(
        name=name,
        value=value,
        unit=unit,
        status="not_applicable" if not_applicable else "unknown" if value is None else "available",
        reason_codes=tuple(sorted(set(reasons)))
        or (("metric_unavailable",) if value is None else ()),
        evidence_refs=unique_evidence(references),
    )


def gate(
    name: str,
    result: Truth,
    reason: str,
    *,
    hard: bool = True,
    references: Iterable[EvidenceRef] = (),
) -> QualityGateResult:
    """生成三值门禁；例：UNKNOWN与unit_unverified保留未验证状态。

    真值不附失败原因，只有显式TRUE才表示该检查通过。
    """
    return QualityGateResult(
        gate_id=name,
        result=result,
        hard=hard,
        reason_codes=() if result is Truth.TRUE else (reason,),
        evidence_refs=unique_evidence(references),
    )


def latest(rows: tuple[ResearchRow, ...]) -> ResearchRow | None:
    """按业务日期、公开时点选择唯一最新来源版本，同序冲突返回None。

    例：同日两份不同source_record_id且同时公开的记录不会任意选第一条。
    """
    if not rows:
        return None
    key = max(
        (row.availability.data_as_of or "", row.availability.public_available_at or "")
        for row in rows
    )
    selected = [
        row
        for row in rows
        if (row.availability.data_as_of or "", row.availability.public_available_at or "") == key
    ]
    return selected[0] if len({row.raw.get("source_record_id") for row in selected}) == 1 else None


def percentiles(
    inputs: ResearchInputs,
    values: dict[str, float | None],
    profile: str,
    *,
    lower_is_better: bool = False,
) -> dict[str, PercentileResult]:
    """在显式同行路径内复用基础层平均秩；例：30个相同值均为0.5。

    未认证同行路径的证券不参与评分，避免默认为全市场可比。
    """
    rows = tuple(
        PeerObservation(
            entity_id=identity,
            value=value,
            analysis_as_of=inputs.analysis_as_of,
            feature_profile=profile,
            scope_id="observed_research",
            peer_path=inputs.config.peer_paths.get(identity, ("unclassified",)),
            eligible=identity in inputs.config.peer_paths,
        )
        for identity, value in values.items()
    )
    return percentile_ranks(rows, inputs.config.thresholds, lower_is_better=lower_is_better)


def make_result(
    inputs: ResearchInputs,
    product_id: str,
    entity_id: str,
    *,
    metrics: Iterable[Metric] = (),
    gates: Iterable[QualityGateResult] = (),
    details: dict[str, Any] | None = None,
    profile: str = "observed_research",
    contributions: Iterable[FeatureContribution] = (),
    references: Iterable[EvidenceRef] = (),
    contrary: Iterable[EvidenceRef] = (),
    parents: Iterable[str] = (),
    predicate_results: dict[str, Truth] | None = None,
    entry_conditions: tuple[Predicate, ...] = (),
    invalidation_conditions: tuple[Predicate, ...] = (),
) -> ResearchResult:
    """构造统一研究信封与业务扩展，硬失败或未知均禁用研究资格。

    例：无价格门禁为FALSE时state=blocked；缺财务特征时state=incomplete。
    所有结果默认只供观察；P03可在完成分组排名与门禁后显式开启人工研究。
    """
    values, checks, components = tuple(metrics), tuple(gates), tuple(contributions)
    evidence = unique_evidence(
        tuple(references)
        + tuple(ref for item in values for ref in item.evidence_refs)
        + tuple(ref for check in checks for ref in check.evidence_refs)
        + tuple(ref for component in components for ref in component.evidence_refs)
    )
    codes = set(reason for check in checks for reason in check.reason_codes)
    codes.update(reason for item in values for reason in item.reason_codes)
    blocked = any(check.hard and check.result is Truth.FALSE for check in checks)
    incomplete = (
        any(check.hard and check.result is Truth.UNKNOWN for check in checks)
        or any(item.status == "unknown" for item in values)
        or not evidence
    )
    state = (
        ProductState.BLOCKED
        if blocked
        else (ProductState.INCOMPLETE if incomplete else ProductState.ANNOTATED)
    )
    codes.add("historical_pit_uncertified")
    used_records = {(ref.source_api, ref.source_record_id) for ref in evidence}
    periods: dict[str, list[str]] = {}
    for api, record_id in used_records:
        period = inputs.record_periods.get((api, record_id))
        if period is not None:
            periods.setdefault(api, []).append(period)
    source_as_of = {api: max(values) for api, values in periods.items()}
    score = fsum(item.contribution for item in components if item.contribution is not None)
    envelope = DecisionEnvelope(
        product_id=product_id,
        entity_id=entity_id,
        scope_id="observed_research",
        run_id=inputs.run_id,
        model_version=RULE_VERSION,
        analysis_as_of=inputs.analysis_as_of,
        generated_at=inputs.analysis_as_of,
        data_as_of=max(source_as_of.values(), default=inputs.analysis_as_of[:10]),
        source_as_of=source_as_of,
        state=state,
        action_state=ActionState.REVIEW_REQUIRED
        if blocked
        else (ActionState.NO_CONCLUSION if incomplete else ActionState.WATCH),
        feature_profile=profile,
        score=score if components and all(c.contribution is not None for c in components) else None,
        gate_codes=tuple(sorted(codes)),
        missing_features=tuple(item.name for item in values if item.status == "unknown"),
        unavailable_apis=tuple(
            sorted(api for api, reasons in inputs.source_reasons.items() if reasons)
        ),
        coverage=Coverage(
            observed_count=len(inputs.entity_ids),
            scope_basis="explicit_bindings",
            reason_codes=("universe_coverage_not_certified",),
        ),
        contributions=components,
        evidence_refs=evidence,
        contrary_evidence=unique_evidence(contrary),
        quality_gates=checks,
        entry_conditions=entry_conditions,
        invalidation_conditions=invalidation_conditions,
    )
    return ResearchResult(
        envelope=envelope,
        metrics=values,
        details_json=encode(details or {}),
        parent_result_ids=tuple(sorted(set(parents))),
        predicate_results=predicate_results or {},
    )
