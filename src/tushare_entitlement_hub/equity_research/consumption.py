"""执行消费注册、把真实引用附加到产品，并重建可验证的运行账本。"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isfinite
from typing import Any

from ..product_foundation.contracts import EvidenceRef, QualityGateResult, Truth
from ..product_foundation.temporal import availability_gate
from ..store import encode
from .common import gate, make_result, unique_evidence
from .consumption_registry import (
    CONSUMPTION_VERSION,
    PRODUCT_ALIASES,
    ConsumptionRegistration,
    load_consumption_registry,
    registry_fingerprint,
    resolve_consumer,
)
from .context_aggregation import row_fingerprint
from .market_evidence import MAXIMUM_SAMPLE_ROWS
from .models import MARKET_ENTITY, ResearchInputs, ResearchResult, ResearchRow

CONSUMPTION_STATUSES = frozenset(
    {
        "observed",
        "unknown",
        "conflict",
        "not_enabled_due_to_history",
        "not_applicable",
    }
)


def consumption_reasons(value: Any, status: str) -> tuple[str, ...]:
    """校验状态原因必须为非空字符串序列，未知、冲突和历史受限不得省略原因。

    输入(['history_missing'], 'not_enabled_due_to_history')返回同内容元组；
    输入字符串、布尔元素或unknown配空列表均抛ValueError。
    """
    if (
        not isinstance(value, (list, tuple))
        or any(type(reason) is not str or not reason.strip() for reason in value)
        or (status in {"unknown", "conflict", "not_enabled_due_to_history"} and not value)
    ):
        raise ValueError("research_consumption_reason_codes_invalid")
    return tuple(value)


def consume_core(
    inputs: ResearchInputs,
    product: str,
    api: str,
    rows: tuple[ResearchRow, ...],
    entity_id: str,
) -> dict[str, Any]:
    """补记原48接口的当前业务字段质量，原产品引擎仍承担其公式与门禁。

    输入已适配daily.close=10且有证书，输出认证值10、字段引用和当前期间；
    无证书只记录字段存在，不把原始数字变成可计算值，无行则unknown。
    P01当期超过8行时全量审计，只展示最多8行认证值，披露全量认证覆盖和抽样范围。
    """
    registration = next(item for item in load_consumption_registry() if item.spec.api == api)
    fields = registration.spec.selected_fields
    current = max((row.availability.data_as_of or "" for row in rows), default="")
    selected = tuple(row for row in rows if (row.availability.data_as_of or "") == current)
    current_record_count = len(selected)
    aggregation = None
    certified_counts = {}
    if product == "P01" and current_record_count > MAXIMUM_SAMPLE_ROWS:
        from .market_evidence import MarketEvidence

        aggregation = MarketEvidence("core_" + api, fields, non_null_fields=True)
        for row in selected:
            aggregation.add(row, "current")
        certified_counts = {
            field: sum(row.number(field) is not None for row in selected) for field in fields
        }
        selected = tuple(row for _, (row, _) in sorted(aggregation.samples.items()))
    elif product == "P01":
        selected = tuple(sorted(selected, key=lambda row: row_fingerprint(row, fields)))
    references = unique_evidence(
        ref
        for row in selected
        for ref in row.evidence(*(field for field in fields if row.raw.get(field) is not None))
    )
    certified = {
        field: [
            {
                "source_record_id": str(row.raw["source_record_id"]),
                "value": row.number(field),
                "unit": row.numeric[field].unit,
            }
            for row in selected
            if row.number(field) is not None
        ]
        for field in fields
    }
    return {
        "status": "observed" if references else "unknown",
        "reason_codes": () if references else ("current_fields_unavailable",),
        "facts": {
            "engine": PRODUCT_ALIASES[product],
            "current_period": current or None,
            "current_record_count": current_record_count,
            "certified_values": certified,
            **(
                {
                    "market_evidence_audit": aggregation.summary(),
                    "certified_values_scope": "evidence_sample_only",
                    "certified_field_counts": certified_counts,
                }
                if aggregation is not None
                else {}
            ),
            "calculation_scope": "current_observation",
            "independent_evidence": False,
            "historical_pit_certified": False,
        },
        "evidence_refs": references,
        "gates": (
            gate(
                "current_fields",
                Truth.TRUE if references else Truth.UNKNOWN,
                "current_fields_unavailable",
                hard=False,
                references=references,
            ),
        ),
    }


def consumption_rows(
    inputs: ResearchInputs,
    registration: ConsumptionRegistration,
    entity_id: str,
) -> tuple[ResearchRow, ...]:
    """按领域范围选行，市场或来源结果保留未关联实体，股票只接显式关联。

    输入P06股票A和已认证转债关系，输出关联债券行；无关系输出空元组。
    factor_list/hm_list按实际因子或席位名称关联，未使用字典留在来源结果。
    """
    api, scope = registration.spec.api, registration.entity_scope
    if entity_id == MARKET_ENTITY or scope == "market":
        return inputs.rows(api)
    if scope == "bond":
        related = {
            relation.from_entity_id
            for relation in inputs.config.confirmed_relations
            if relation.relation_type == "cb_underlying"
            and relation.confirmed
            and relation.to_entity_id == entity_id
            and relation.usable_at(inputs.analysis_as_of)
        }
        return tuple(row for row in inputs.rows(api) if row.entity_id in related)
    if scope == "dictionary":
        if api == "factor_list":
            names = {row.text("factor_name") for row in inputs.rows("factor_value", entity_id)}
            return tuple(row for row in inputs.rows(api) if row.text("factor_name") in names)
        if api == "hm_list":
            names = {row.text("hm_name") for row in inputs.rows("hm_detail", entity_id)}
            return tuple(row for row in inputs.rows(api) if row.text("name") in names)
        aliases = {
            alias.alias
            for alias in inputs.config.foundation.aliases
            if alias.entity_id == entity_id and alias.visible_at(inputs.analysis_as_of)
        }
        return tuple(
            row
            for row in inputs.rows(api)
            if row.text("o_code") in aliases or row.text("n_code") in aliases
        )
    return inputs.rows(api, entity_id)


def field_quality(rows: tuple[ResearchRow, ...], fields: tuple[str, ...]) -> dict[str, Any]:
    """区分字段缺失、有限原值和已认证值，不由原始值猜单位。

    输入一条未认证close=10，输出present_rows=1、certified_rows=0；NaN另计为1。
    """
    return {
        field: {
            "present_rows": sum(row.raw.get(field) is not None for row in rows),
            "missing_rows": sum(row.raw.get(field) is None for row in rows),
            "certified_rows": sum(row.number(field) is not None for row in rows),
            "nonfinite_rows": sum(
                type(row.raw.get(field)) in {int, float} and not isfinite(row.raw[field])
                for row in rows
            ),
        }
        for field in fields
    }


def execute_consumption(
    inputs: ResearchInputs,
    registration: ConsumptionRegistration,
    entity_id: str,
) -> tuple[dict[str, Any], tuple[EvidenceRef, ...], tuple[QualityGateResult, ...]]:
    """执行真实处理器并叠加扫描缺口，引用和门禁均进入统一信封。

    输入缺实体绑定的API，输出executed=true但status=unknown及绑定原因；
    处理器返回非有限数值或无引用却声称已观测时拒绝伪造状态。
    """
    spec = registration.spec
    candidates = consumption_rows(inputs, registration, entity_id)
    rows = tuple(
        row
        for row in candidates
        if row.source_api == spec.api
        and row.raw.get("source_api") == spec.api
        and row.raw.get("quality_tier") in {"verified", "usable_with_annotation"}
        and row.raw.get("is_quarantined") is not True
        and availability_gate(row.availability, inputs.analysis_as_of).passed
    )
    outcome = resolve_consumer(registration.handler)(
        inputs, spec.product, spec.api, rows, entity_id
    )
    status = outcome["status"]
    if type(status) is not str or status not in CONSUMPTION_STATUSES:
        raise ValueError("research_consumption_status_invalid")
    references = unique_evidence(
        tuple(outcome["evidence_refs"])
        + tuple(reference for check in outcome["gates"] for reference in check.evidence_refs)
    )
    if status in {"observed", "not_enabled_due_to_history"}:
        local_fields = {
            (str(row.raw["source_record_id"]), field)
            for row in rows
            for field in spec.selected_fields
            if row.raw.get(field) is not None
        }
        if not rows or not any(
            reference.source_api == spec.api
            and any(
                (reference.source_record_id, field) in local_fields for field in reference.fields
            )
            for reference in references
        ):
            raise ValueError("research_consumption_observation_without_evidence")
    checks = tuple(
        replace(check, gate_id=f"consumption/{spec.api}/{check.gate_id}", hard=False)
        for check in outcome["gates"]
    )
    reasons = set(consumption_reasons(outcome["reason_codes"], "observed")) | set(
        inputs.reasons(spec.api, entity_id)
    )
    if entity_id == MARKET_ENTITY:
        reasons.update(
            reason
            for key, values in inputs.entity_source_reasons.items()
            if key.startswith(spec.api + "/")
            for reason in values
        )
    if not rows:
        reasons.add("consumption_rows_unavailable")
        status = "unknown"
    if len(rows) != len(candidates):
        reasons.add("consumption_rows_rejected")
    verified_reasons = consumption_reasons(sorted(reasons), status)
    checks += (
        gate(
            f"consumption/{spec.api}/execution_status",
            Truth.TRUE
            if status == "observed"
            else Truth.FALSE
            if status == "conflict"
            else Truth.UNKNOWN,
            "consumption_" + status,
            hard=False,
            references=references,
        ),
    )
    record = {
        "api": spec.api,
        "product": spec.product,
        "role": spec.role,
        "purpose": spec.purpose,
        "handler": registration.handler,
        "selected_fields": list(spec.selected_fields),
        "entity_scope": registration.entity_scope,
        "retention": registration.retention,
        "executed": True,
        "status": status,
        "reason_codes": list(verified_reasons),
        "retained_row_count": len(rows),
        "field_quality": field_quality(rows, spec.selected_fields),
        "facts": outcome["facts"],
        "evidence_refs": [ref.to_dict() for ref in references],
        "gates": [check.to_dict() for check in checks],
        "historical_pit_certified": False,
    }
    encode(record)
    return record, references, checks


def attach_consumption(
    inputs: ResearchInputs,
    product: str,
    results: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """为产品每条结果执行其矩阵分配，必要时增加独立来源状态承接非股票行。

    输入只有未关联转债的P06，输出保留债券引用的市场范围结果；
    原股票分数、排名与贡献保持原值，新增旁证门禁不提升研究资格。
    """
    registrations = tuple(
        item
        for item in load_consumption_registry()
        if PRODUCT_ALIASES[item.spec.product] == product
    )
    if not any(result.envelope.entity_id == MARKET_ENTITY for result in results) and any(
        item.entity_scope in {"bond", "dictionary", "market", "event"}
        and inputs.rows(item.spec.api)
        for item in registrations
    ):
        results += (
            make_result(
                inputs,
                product,
                MARKET_ENTITY,
                profile="source_observation",
                gates=(gate("source_scope", Truth.UNKNOWN, "source_scope_not_stock_universe"),),
                details={"mode": "source_observation", "stock_candidate": False},
            ),
        )
    attached = []
    for result in results:
        records: list[dict[str, Any]] = []
        references: list[EvidenceRef] = []
        checks: list[QualityGateResult] = []
        for item in registrations:
            record, evidence, gates = execute_consumption(inputs, item, result.envelope.entity_id)
            records.append(record)
            references.extend(evidence)
            checks.extend(gates)
        evidence = unique_evidence(result.envelope.evidence_refs + tuple(references))
        periods = dict(result.envelope.source_as_of)
        for reference in evidence:
            period = inputs.record_periods.get((reference.source_api, reference.source_record_id))
            if period is not None:
                periods[reference.source_api] = max(period, periods.get(reference.source_api, ""))
        envelope = replace(
            result.envelope,
            evidence_refs=evidence,
            source_as_of=periods,
            quality_gates=result.envelope.quality_gates + tuple(checks),
            gate_codes=tuple(
                sorted(
                    set(result.envelope.gate_codes)
                    | {reason for check in checks for reason in check.reason_codes}
                )
            ),
        )
        details = result.details | {
            "consumption_version": CONSUMPTION_VERSION,
            "consumption": records,
        }
        attached.append(replace(result, envelope=envelope, details_json=encode(details)))
    return tuple(attached)


def validate_result_consumption(result: ResearchResult) -> None:
    """重验每条结果的精确分配、处理器、字段质量与真实信封引用。

    输入删除一个API或伪造引用的持久化结果，抛ValueError；完整结果返回None。
    """
    details = result.details
    records = details.get("consumption")
    expected = {
        item.spec.api: item
        for item in load_consumption_registry()
        if PRODUCT_ALIASES[item.spec.product] == result.envelope.product_id
    }
    if details.get("consumption_version") != CONSUMPTION_VERSION or not isinstance(records, list):
        raise ValueError("research_consumption_missing")
    if len(records) != len(expected) or {item["api"] for item in records} != set(expected):
        raise ValueError("research_consumption_assignment_mismatch")
    envelope_refs = {ref.to_json() for ref in result.envelope.evidence_refs}
    envelope_gates = {check.to_json() for check in result.envelope.quality_gates}
    for record in records:
        item = expected[record["api"]]
        immutable = {
            "product": item.spec.product,
            "role": item.spec.role,
            "purpose": item.spec.purpose,
            "handler": item.handler,
            "selected_fields": list(item.spec.selected_fields),
            "entity_scope": item.entity_scope,
            "retention": item.retention,
        }
        if (
            any(record.get(key) != value for key, value in immutable.items())
            or type(record.get("status")) is not str
            or record.get("status") not in CONSUMPTION_STATUSES
            or record.get("executed") is not True
            or record.get("historical_pit_certified") is not False
        ):
            raise ValueError("research_consumption_contract_mismatch")
        consumption_reasons(record.get("reason_codes"), record["status"])
        count = record["retained_row_count"]
        quality = record["field_quality"]
        if type(count) is not int or count < 0 or set(quality) != set(item.spec.selected_fields):
            raise ValueError("research_consumption_field_quality_invalid")
        for values in quality.values():
            if (
                set(values) != {"present_rows", "missing_rows", "certified_rows", "nonfinite_rows"}
                or any(type(value) is not int or value < 0 for value in values.values())
                or values["present_rows"] + values["missing_rows"] != count
                or max(values["certified_rows"], values["nonfinite_rows"]) > values["present_rows"]
            ):
                raise ValueError("research_consumption_field_quality_invalid")
        refs = tuple(EvidenceRef.from_dict(ref) for ref in record["evidence_refs"])
        if any(ref.to_json() not in envelope_refs for ref in refs) or any(
            QualityGateResult.from_dict(check).to_json() not in envelope_gates
            for check in record["gates"]
        ):
            raise ValueError("research_consumption_reference_missing")
        if record["status"] in {"observed", "not_enabled_due_to_history"} and (
            not count
            or not any(
                reference.source_api == item.spec.api
                and any(
                    field in quality and quality[field]["present_rows"] > 0
                    for field in reference.fields
                )
                for reference in refs
            )
        ):
            raise ValueError("research_consumption_observation_without_evidence")


def count_consumption(counts: dict[str, Counter[str]], result: ResearchResult) -> None:
    """累计每接口执行状态，空间与99接口数量相关而不随证券数增长。

    输入两只股票的同一API观测，输出该API observed=2，分配接口数仍为1。
    """
    for record in result.details["consumption"]:
        counts.setdefault(record["api"], Counter())[record["status"]] += 1


def summarize_consumption(counts: dict[str, Counter[str]], products: list[str]) -> dict[str, Any]:
    """从实际结果重建分配、注册、执行和状态分解，空数据不冒充已观测。

    输入六产品全缺数据，输出assigned=registered=executed=99、observed=0。
    """
    expected = {
        item.spec.api
        for item in load_consumption_registry()
        if PRODUCT_ALIASES[item.spec.product] in products
    }
    return {
        "version": CONSUMPTION_VERSION,
        "registry_fingerprint": registry_fingerprint(),
        "total_assigned_apis": 99,
        "assigned_apis": len(expected),
        "registered_apis": len(expected),
        "executed_apis": len(set(counts) & expected),
        "observed_apis": sum(values["observed"] > 0 for values in counts.values()),
        "status_counts": {
            status: sum(values[status] for values in counts.values())
            for status in sorted(CONSUMPTION_STATUSES)
        },
        "by_api": {api: dict(sorted(values.items())) for api, values in sorted(counts.items())},
    }
