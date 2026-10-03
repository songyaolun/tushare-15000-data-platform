"""P13 证据事件与三值论点；实体候选、原文原因和同窗相关分别保存。"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef, Predicate, QualityGateResult, Truth
from ..product_foundation.numeric import price_gate
from ..product_foundation.signals import evaluate
from ..standard import normalize_date
from ..store import digest, encode
from .common import gate, make_result, metric, unique_evidence
from .event_parents import ParentBundle, index_parents
from .models import MARKET_ENTITY, RULE_VERSION, ResearchInputs, ResearchResult, ResearchRow

EVENT_APIS = ("major_news", "disclosure_date", "cn_schedule", "idx_anns", "namechange")
THESIS = Predicate(
    predicate_id="cash_support_and_supply",
    version=RULE_VERSION,
    operator="all",
    children=(
        Predicate(
            predicate_id="cash_support",
            version=RULE_VERSION,
            operator="ge",
            feature_id="cash_cover",
            threshold=1.0,
        ),
        Predicate(
            predicate_id="limited_unlock",
            version=RULE_VERSION,
            operator="lt",
            feature_id="unlock_30d",
            threshold=0.05,
        ),
    ),
)


@dataclass(frozen=True)
class _Event:
    """待生成的来源事件；例如一条披露记录分别形成计划与实际事件。"""

    event_id: str
    event_type: str
    source_api: str
    deadline: str | None
    row: ResearchRow | None
    entity_id: str | None = None


def _day(value: Any) -> str | None:
    """规范明确的日粒度字段，非法值只保留为未知；例：20260930→2026-09-30。

    不从文本中猜日期，不将报告期替换为事件的可用或生效日期。
    """
    try:
        return normalize_date(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _source_events(inputs: ResearchInputs) -> tuple[_Event, ...]:
    """按来源身份去重，拆分披露计划、实际和历次修正而不吞掉版本。

    例如 pre_date=20261020、actual_date=20261019 输出两个独立事件；缺少所有
    日程字段仍保留一条 disclosure_observation，并明确日期未知。
    """
    events: dict[str, _Event] = {}
    for api in EVENT_APIS:
        for row in inputs.rows(api):
            if api == "disclosure_date":
                dates = [
                    ("disclosure_planned", _day(row.raw.get("pre_date"))),
                    ("disclosure_actual", _day(row.raw.get("actual_date"))),
                ]
                changes = row.raw.get("modify_date")
                pieces = changes if isinstance(changes, list) else str(changes or "").split(",")
                dates.extend(
                    ("disclosure_revision", _day(piece.strip()))
                    for piece in pieces
                    if isinstance(piece, str)
                )
                dated: list[tuple[str, str | None]] = [
                    (kind, day) for kind, day in dates if day is not None
                ]
                dates = dated or [("disclosure_observation", None)]
            else:
                field = {
                    "major_news": "pub_time",
                    "cn_schedule": "publish_date",
                    "idx_anns": "ann_date",
                    "namechange": "start_date",
                }[api]
                raw_date = row.text(field)
                dates = [(api, _day(raw_date[:10]) if raw_date else None)]
            for kind, day in dates:
                event_id = (
                    "research:event:"
                    + digest([api, row.availability.source_record_id, kind, day])
                    + ":v1"
                )
                events[event_id] = _Event(event_id, kind, api, day, row)
    return tuple(events[key] for key in sorted(events))


def _entities(
    inputs: ResearchInputs,
    event: _Event,
) -> tuple[str | None, tuple[str, ...], tuple[EvidenceRef, ...], str | None]:
    """显式键或有效人工关系才确认实体，文本别名只返回候选及歧义原因。

    例如新闻标题匹配两家公司的有效别名，返回 confirmed=None 和两项候选；
    单一候选也需人工确认。人工关系以 event_id→entity_id 的 event_entity 表示。
    """
    if event.entity_id is not None:
        return (
            event.entity_id,
            (),
            (),
            "upstream_market_scope_not_security" if event.entity_id == MARKET_ENTITY else None,
        )
    row = event.row
    assert row is not None
    if event.source_api == "cn_schedule":
        return MARKET_ENTITY, (), (), None
    if row.entity_id is not None and event.source_api in {"disclosure_date", "namechange"}:
        return row.entity_id, (row.entity_id,), (), None
    relations = tuple(
        relation
        for relation in inputs.config.confirmed_relations
        if relation.from_entity_id == event.event_id
        and relation.confirmed
        and relation.relation_type == "event_entity"
        and relation.usable_at(inputs.analysis_as_of)
        and relation.to_entity_id
        in {entity.identity.entity_id for entity in inputs.config.foundation.entities}
    )
    confirmed = {relation.to_entity_id for relation in relations}
    references = tuple(ref for relation in relations for ref in relation.evidence_refs)
    if len(confirmed) == 1:
        identity = next(iter(confirmed))
        return identity, (identity,), references, None
    text = "\n".join(filter(None, (row.text("title"), row.text("content"), row.text("name"))))
    event_time = row.availability.public_available_at or row.availability.available_at
    aliases = tuple(
        alias
        for alias in inputs.config.foundation.aliases
        if alias.alias in text
        and alias.visible_at(inputs.analysis_as_of)
        and time_boundary(alias.effective_from) <= time_boundary(event_time)
        and (
            alias.effective_to is None
            or time_boundary(event_time) < time_boundary(alias.effective_to)
        )
    )
    candidates = tuple(sorted(confirmed | {alias.entity_id for alias in aliases}))
    references += tuple(ref for alias in aliases for ref in alias.evidence_refs)
    reason = (
        "entity_ambiguous"
        if len(candidates) > 1
        else "entity_confirmation_required"
        if candidates
        else "entity_unresolved"
    )
    return None, candidates, unique_evidence(references), reason


def _attribution(
    inputs: ResearchInputs,
    event: _Event,
    identity: str | None,
) -> tuple[str, list[dict[str, str]]]:
    """只有原文明确的原因句才能形成 documented_reason，并保存逐字片段。

    例如 content='利润减少，原因是成本上升。' 返回该原句；没有原因句但有
    同实体同日价格证据时仅返回 temporal_association，不作因果判断。
    """
    if event.row is None:
        return "evidence_only", []
    spans: list[dict[str, str]] = []
    for field in ("title", "content"):
        content = event.row.text(field)
        if content:
            spans.extend(
                {"field": field, "text": sentence.strip()}
                for sentence in re.split(r"(?<=[。！？\n])", content)
                if re.search(
                    r"原因(?:是|为|在于)|由于\S|因为\S|because\s+\S|due to\s+\S",
                    sentence,
                    re.IGNORECASE,
                )
                and not re.search(r"并非|不是|未知|不明|未披露|尚未|可能|或许|疑似", sentence)
            )
    if spans:
        return "documented_reason", spans
    associated = (
        identity is not None
        and event.deadline is not None
        and any(
            row.availability.data_as_of == event.deadline
            and row.number("pct_chg", "fraction") is not None
            and price_gate(row.raw, "return").passed
            for row in inputs.rows("daily", identity)
        )
    )
    return "temporal_association" if associated else "evidence_only", []


def _parent_metrics(
    parents: tuple[ResearchResult, ...],
) -> tuple[dict[str, float | None], tuple[EvidenceRef, ...]]:
    """从对应产品唯一结果提取论点输入，同一产品多版本时拒绝任意选取。

    例如同实体两份 P04 结果，即使现金覆盖数值相同，也输出 cash_cover=None；
    允许 P06 与 P07 同时存在，因为它们不替代现金覆盖的来源。
    """
    values: dict[str, float | None] = {}
    references: list[EvidenceRef] = []
    for name, product in (
        ("cash_cover", "fundamental_dossier/v1"),
        ("unlock_30d", "capital_governance_risks/v1"),
    ):
        selected = tuple(parent for parent in parents if parent.envelope.product_id == product)
        values[name] = selected[0].metric(name) if len(selected) == 1 else None
        references.extend(
            ref
            for parent in selected
            for item in parent.metrics
            if item.name == name
            for ref in item.evidence_refs
        )
    return values, unique_evidence(references)


def _upstream_hard_checks(parents: tuple[ResearchResult, ...]) -> tuple[QualityGateResult, ...]:
    """按父产品前缀及门禁身份归并硬检查，冲突时保留最严格的结论与全部证据。

    例如同产品同门禁分别为 true、unknown，输出一条 unknown；false 优先于 unknown。
    不同产品的同名门禁彼此独立，软检查不升级为硬阻断。
    """
    groups: dict[str, list[QualityGateResult]] = {}
    for parent in parents:
        for check in parent.envelope.quality_gates:
            if check.hard:
                name = parent.envelope.product_id + "/" + check.gate_id
                groups.setdefault(name, []).append(check)
    return tuple(
        QualityGateResult(
            gate_id=name,
            result=Truth.FALSE
            if any(check.result is Truth.FALSE for check in checks)
            else Truth.UNKNOWN
            if any(check.result is Truth.UNKNOWN for check in checks)
            else Truth.TRUE,
            reason_codes=tuple(
                sorted({reason for check in checks for reason in check.reason_codes})
            ),
            evidence_refs=unique_evidence(ref for check in checks for ref in check.evidence_refs),
        )
        for name, checks in sorted(groups.items())
    )


def _review(
    inputs: ResearchInputs, event: _Event, upstream: dict[str, ParentBundle]
) -> ResearchResult:
    """生成单事件的默认论点、反证、父血缘与排序键，保持持仓权重未知。

    例如现金覆盖=1.2、解禁=.01 得到 true；解禁缺失得到 unknown。硬风险越高
    越先复核，同风险先按已知期限排序，再按来源等级，最后按事件身份打破平局。
    父结果已按实体及截止索引；一次查询只取得当前实体的全部版本，不扫描其他实体。
    """
    identity, candidates, mapping_refs, conflict = _entities(inputs, event)
    bundle = upstream.get(identity) if identity is not None else None
    parents = bundle.parents if bundle else ()
    values, feature_refs = _parent_metrics(parents)
    results = {
        predicate.predicate_id: evaluate(predicate, values)
        for predicate in THESIS.children + (THESIS,)
    }
    results["entity_confirmation"] = (
        Truth.TRUE if identity is not None and conflict is None else Truth.UNKNOWN
    )
    row = event.row
    references = unique_evidence(
        mapping_refs
        + feature_refs
        + tuple(ref for parent in parents for ref in parent.envelope.evidence_refs)
        + (row.evidence(*tuple(row.raw)) if row else ())
    )
    hard_checks = _upstream_hard_checks(parents)
    if bundle and bundle.ambiguous_products:
        hard_checks += (
            gate("upstream_product_versions", Truth.UNKNOWN, "upstream_product_versions_ambiguous"),
        )
    hard_risk = (
        2
        if any(check.result is Truth.FALSE for check in hard_checks)
        else 1
        if any(check.result is Truth.UNKNOWN for check in hard_checks)
        else 0
    )
    attribution, spans = _attribution(inputs, event, identity)
    contrary = tuple(
        ref for check in hard_checks if not check.passed for ref in check.evidence_refs
    ) + tuple(ref for parent in parents for ref in parent.envelope.contrary_evidence)
    if results[THESIS.predicate_id] is Truth.FALSE:
        contrary += feature_refs
    source_level = (
        1 if event.source_api == "major_news" else 2 if event.source_api == "cn_schedule" else 3
    )
    next_check = (
        event.deadline + "T23:59:59Z"
        if event.deadline is not None
        and time_boundary(event.deadline + "T23:59:59Z") >= time_boundary(inputs.analysis_as_of)
        else None
    )
    details = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "source_api": event.source_api,
        "deadline": event.deadline,
        "entity_candidates": list(candidates),
        "confirmed_entity": identity,
        "conflict_reason": conflict,
        "evidence_spans": spans,
        "attribution_level": attribution,
        "attribution_statement": "原文原因陈述仅代表来源表述"
        if spans
        else "同窗出现仅表示时间相关"
        if attribution == "temporal_association"
        else "仅整理来源证据",
        "causality_established": False,
        "car": None,
        "car_reason": "historical_event_window_unavailable",
        "thesis_id": THESIS.predicate_id,
        "thesis_version": RULE_VERSION,
        "claim": "现金覆盖不低于1且30日解禁比例低于5%",
        "predicate": THESIS.to_dict(),
        "predicate_result": results[THESIS.predicate_id].value,
        "threshold_basis": "versioned_uncalibrated_default",
        "holding_weight": None,
        "hard_risk_level": hard_risk,
        "source_evidence_level": source_level,
        "review_priority_key": [-hard_risk, 1, None, event.deadline, -source_level, event.event_id],
        "observation_at": row.availability.observed_at if row else inputs.analysis_as_of,
        "public_available_at": row.availability.public_available_at if row else None,
        "effective_from": row.availability.effective_from if row else None,
        "report_period": row.availability.report_period if row else None,
        "supporting_features": [
            predicate.feature_id
            for predicate in THESIS.children
            if results[predicate.predicate_id] is Truth.TRUE
        ],
        "contradictory_features": [
            predicate.feature_id
            for predicate in THESIS.children
            if results[predicate.predicate_id] is Truth.FALSE
        ],
        "review_owner": None,
        "upstream_parent_count": len(parents),
        "upstream_fingerprint": bundle.fingerprint if bundle else None,
        "upstream_ambiguous_products": list(bundle.ambiguous_products) if bundle else [],
    }
    result = make_result(
        inputs,
        "event_thesis_review/v1",
        identity or MARKET_ENTITY,
        metrics=tuple(metric(name, value, "ratio", feature_refs) for name, value in values.items()),
        gates=(
            gate(
                "event_entity_confirmed",
                results["entity_confirmation"],
                conflict or "entity_unresolved",
                references=mapping_refs,
            ),
        )
        + hard_checks,
        references=references,
        contrary=contrary,
        details=details,
        parents=(parent.envelope.result_id for parent in parents),
        predicate_results=results,
        entry_conditions=(
            THESIS,
            Predicate(predicate_id="entity_confirmation", version=RULE_VERSION, operator="manual"),
        ),
        invalidation_conditions=(
            Predicate(
                predicate_id="thesis_invalidated",
                version=RULE_VERSION,
                operator="not",
                children=(THESIS,),
            ),
        ),
    )
    return replace(
        result, envelope=replace(result.envelope, scope_id=event.event_id, next_check_at=next_check)
    )


def build_events(
    inputs: ResearchInputs, upstream: tuple[ResearchResult, ...]
) -> tuple[ResearchResult, ...]:
    """整理直接来源及 P04/P06/P07 父证据事件，输出确定性的人工复核队列。

    例如同一新闻重复输入两次只产生一条来源事件；两条不同事件各有唯一 scope_id。
    同实体的P04/P06/P07形成一个upstream_bundle，全部父ID绑定事件身份；
    完整父内容由流式摘要覆盖，版本歧义、硬门禁和反证仍保留，不抽样父证据。
    空输入返回明确缺来源的观察结果，绝不用一个空列表表示所有检查通过。
    """
    indexed = index_parents(inputs, upstream)
    events = list(_source_events(inputs))
    for identity, bundle in indexed.items():
        events.append(
            _Event(
                "research:event:"
                + digest(["upstream_bundle", identity, bundle.result_ids])
                + ":v1",
                "upstream_bundle",
                "upstream_products",
                None,
                None,
                identity,
            )
        )
    if not events:
        return (
            make_result(
                inputs,
                "event_thesis_review/v1",
                MARKET_ENTITY,
                gates=(gate("event_sources", Truth.UNKNOWN, "event_sources_unavailable"),),
                details={
                    "event_count": 0,
                    "car": None,
                    "car_reason": "historical_event_window_unavailable",
                },
            ),
        )
    reviews = [_review(inputs, event, indexed) for event in events]
    reviews.sort(
        key=lambda result: (
            -result.details["hard_risk_level"],
            result.details["deadline"] or "9999-12-31",
            -result.details["source_evidence_level"],
            result.details["event_id"],
        )
    )
    return tuple(
        replace(result, details_json=encode({**result.details, "review_priority": index}))
        for index, result in enumerate(reviews, 1)
    )
