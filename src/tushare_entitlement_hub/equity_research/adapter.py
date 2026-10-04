"""只读参考库适配：逐批扫描、有限当前观察集、明确报价语义与时点门禁。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, cast

import duckdb

from ..catalog import load_policies
from ..product_foundation.batch_io import batches
from ..product_foundation.entities import resolve_alias
from ..product_foundation.numeric import (
    TypedFact,
    UnitCertification,
    adapt_numeric,
    source_evidence,
)
from ..product_foundation.progress import ProgressCallback
from ..product_foundation.reference import _inspect_source
from ..product_foundation.reference_scan import map_availability, reference_rows
from ..product_foundation.registry import ConsumerSpec, load_registry
from ..product_foundation.temporal import availability_gate
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..store import digest
from .binding_rules import PUBLIC_FIELDS, period_visible
from .consumption_registry import (
    EVENT_APIS,
    TWO_PERIOD_APIS,
    load_consumption_registry,
)
from .models import ResearchConfig, ResearchInputs, ResearchRow

SOURCE_APIS = frozenset(item.spec.api for item in load_consumption_registry())
UNRESOLVED_APIS = frozenset(
    item.spec.api
    for item in load_consumption_registry()
    if item.entity_scope in {"market", "bond", "dictionary", "event"}
)

QUOTE_FIELDS = {
    "cb_basic": {"conv_price", "par"},
    "cb_share": {"convert_price"},
    "cb_daily": {"cb_value"},
    "new_share": {"price"},
    "cyq_perf": {"cost_5pct", "cost_50pct", "cost_95pct", "weight_avg"},
    "stk_factor": {"close_qfq"},
    "stk_factor_pro": {"close_qfq"},
    "stk_week_month_adj": {"close_qfq", "close_hfq"},
    "cyq_chips": {"price"},
    "stk_limit": {"up_limit", "down_limit"},
    "block_trade": {"price"},
}


def quoted_numeric(row: ResearchRow, name: str, certificate: UnitCertification) -> TypedFact:
    """认证转股价、筹码价、涨跌停边界等非日线成交报价，保持其独立价格语义。

    例：cb_basic.conv_price=10且元/CNY证书得到10；不要求转债有当日成交。
    本函数仅供固定报价字段白名单调用，不用于daily.close的成交价格门禁。
    """
    raw = row.raw
    reasons: set[str] = set()
    value = raw.get(name)
    current_unit = raw.get("standard_units", {}).get(name) or raw.get("source_units", {}).get(name)
    if current_unit == certificate.target_unit:
        multiplier = 1.0
    elif current_unit == certificate.source_unit:
        multiplier = certificate.multiplier
    else:
        multiplier = 1.0
        reasons.add("unit_mismatch")
    if raw.get("quality_tier") != "verified" or raw.get("is_quarantined") is True:
        reasons.add("quote_quality_unverified")
    if raw.get("business_variant_key"):
        reasons.add("business_variant_unresolved")
    if certificate.dimension != "price" or certificate.currency != "CNY":
        reasons.add("quote_basis_unverified")
    if (
        type(value) not in {int, float}
        or not isfinite(cast(float, value))
        or cast(float, value) <= 0
    ):
        reasons.add("invalid_quote")
    elif not isfinite(cast(float, value) * multiplier):
        reasons.add("nonfinite_conversion")
    return TypedFact(
        fact_id=digest([row.source_api, raw["source_record_id"], name, certificate.to_dict()]),
        entity_id=row.entity_id or "cn:market:A:all",
        field=name,
        value=None if reasons else cast(float, value) * multiplier,
        unit=certificate.target_unit,
        dimension=certificate.dimension,
        currency=certificate.currency,
        availability=row.availability,
        quality_tier="verified" if not reasons else "unknown",
        certification=certificate,
        evidence_refs=(source_evidence(raw, name, row.availability),),
        reason_codes=tuple(sorted(reasons)),
    )


def prepare_numeric(
    row: ResearchRow,
    config: ResearchConfig,
    *,
    certificates: tuple[UnitCertification, ...] | None = None,
) -> ResearchRow:
    """仅适配调用方认证过且实际存在的字段；例：缺revenue证书不生成财务值。

    认证证据晚于研究时点时由扫描层先剔除对应证书，原始字段不改写。
    certificates可显式传入已过滤的来源证书，例如批处理daily时不遍历其他205个来源。
    """
    numeric = {}
    for certificate in config.foundation.certifications if certificates is None else certificates:
        if certificate.source_api != row.source_api or certificate.field not in row.raw:
            continue
        name = certificate.field
        numeric[name] = (
            quoted_numeric(row, name, certificate)
            if name in QUOTE_FIELDS.get(row.source_api, set())
            else adapt_numeric(
                row.raw, name, row.entity_id or "cn:market:A:all", row.availability, certificate
            )
        )
    return replace(row, numeric=numeric)


def adapt_row(
    raw: dict[str, Any],
    contract: ReferenceContract,
    config: ResearchConfig,
    cutoff: str,
    *,
    unresolved_apis: frozenset[str] = UNRESOLVED_APIS,
    event_apis: frozenset[str] = EVENT_APIS,
) -> ResearchRow:
    """在数值计算前核验来源、时点、实体及质量；例：隔离行或未来公告直接拒绝。

    文本事件允许entity_id=None，后续只能生成候选；其他接口必须有唯一有效别名。
    显式unresolved_apis允许新来源保留未解析证据，例如汇率不被解释为股票。
    """
    api = contract.policy.api_name
    if raw.get("source_api") != api or raw.get("contract_hash") != contract.fingerprint:
        raise ValueError("source_contract_mismatch")
    if raw.get("quality_tier") not in {"verified", "usable_with_annotation"} or (
        raw.get("is_quarantined") is True
    ):
        raise ValueError("quarantined_or_unknown_quality")
    mapping = next(
        (item for item in config.foundation.time_mappings if item.source_api == api), None
    )
    if mapping is None:
        raise ValueError("time_mapping_missing")
    if any(
        raw.get(field) is not None
        and field not in mapping.public_fields + mapping.optional_public_fields
        for field in PUBLIC_FIELDS
    ):
        raise ValueError("public_field_mapping_missing")
    availability = map_availability(raw, contract, mapping)
    if not availability_gate(availability, cutoff).passed:
        raise ValueError("row_unavailable")
    if (
        api not in event_apis
        and availability.data_as_of is not None
        and not period_visible(availability.data_as_of, cutoff, mapping.market_timezone or "UTC")
    ):
        raise ValueError("future_business_date")
    entity_mapping = next(
        (item for item in config.foundation.entity_mappings if item.source_api == api), None
    )
    entity_id = None
    if entity_mapping is not None:
        business_time = availability.data_as_of or availability.effective_from
        if business_time is None and api not in unresolved_apis:
            raise ValueError("entity_business_time_unknown")
        if business_time is not None:
            entity_id = resolve_alias(
                config.aliases_by_key.get(
                    (entity_mapping.namespace, str(raw.get(entity_mapping.key_field, ""))), ()
                ),
                entity_mapping.namespace,
                str(raw.get(entity_mapping.key_field, "")),
                cutoff,
                effective_at=business_time,
                market_timezone=mapping.market_timezone,
            )
    if entity_id is None and api not in unresolved_apis:
        raise ValueError("entity_unresolved")
    return ResearchRow(entity_id=entity_id, source_api=api, raw=raw, availability=availability)


def adapter_reason(error: Exception) -> str:
    """将行失败映射为有限原因码，禁止第三方异常或来源正文进入审计。

    例如ValueError('entity_unresolved')保留原码，未知异常返回row_adaptation_failed。
    """
    allowed = {
        "source_contract_mismatch",
        "quarantined_or_unknown_quality",
        "time_mapping_missing",
        "public_field_mapping_missing",
        "row_unavailable",
        "future_business_date",
        "entity_business_time_unknown",
        "entity_unresolved",
        "ambiguous_entity_alias",
        "source_pit_state_unverified",
        "public_time_missing",
        "market_timezone_required",
        "invalid_timestamp_or_timezone",
        "invalid_period",
        "invalid_effective_interval",
        "source_local_key_missing",
        "source_local_key_invalid",
        "business_variant_unresolved",
    }
    return str(error) if str(error) in allowed else "row_adaptation_failed"


def retain_current(
    groups: dict[tuple[str, str, str], list[ResearchRow]],
    row: ResearchRow,
    config: ResearchConfig,
    *,
    entity_counts: dict[str, int] | None = None,
) -> bool:
    """流式保留当前业务期间及同序变体，宏观保留最近两个期间；返回是否容量溢出。

    例：daily旧日后遇新日，旧行释放；share_float事件全部保留，超过容量整组拒绝。
    财务按报表口径分别保留最新期，未在内存物化完整历史。
    """
    dimensions = (row.text("report_type") or "", row.text("comp_type") or "")
    if row.source_api == "cb_rating":
        dimensions = (row.text("rating_com_name") or "", "")
    elif row.source_api in {"stk_weekly_monthly", "stk_week_month_adj", "stk_nineturn"}:
        dimensions = (row.text("freq") or "", "")
    elif row.source_api in {"factor_list", "factor_value"}:
        dimensions = (row.text("factor_name") or "", row.text("asset_type") or "")
    # 无实体债券与市场分项仍按来源代码分组，不把多个证券的当前期间相互淘汰。
    key = (retention_identity(row), *dimensions)
    candidates = groups.setdefault(key, [])
    previous_count = len(candidates)
    candidates.append(row)
    if row.source_api not in EVENT_APIS:
        periods = sorted({retention_period(item) for item in candidates}, reverse=True)
        width = 2 if row.source_api in TWO_PERIOD_APIS else 1
        chosen = set(periods[:width])
        candidates[:] = [item for item in candidates if retention_period(item) in chosen]
    if entity_counts is None:
        retained_count = sum(len(values) for group, values in groups.items() if group[0] == key[0])
    else:
        retained_count = entity_counts.get(key[0], 0) + len(candidates) - previous_count
        entity_counts[key[0]] = retained_count
    return retained_count > config.maximum_rows_per_entity_source


def retention_period(row: ResearchRow) -> str:
    """使用统计终点而非供应商周月标签日期，评级按公告期保留相邻状态。

    输入标签2026-10-05、end_date=2026-10-02的周线，输出2026-10-02。
    其他来源返回已经通过时间映射认证的业务期间，不推断缺失日期。
    """
    from ..standard import normalize_period

    field = (
        "end_date"
        if row.source_api in {"stk_weekly_monthly", "stk_week_month_adj"}
        else ("ann_date" if row.source_api == "cb_rating" else None)
    )
    value = row.text(field) if field is not None else None
    if value is not None:
        try:
            return normalize_period(value)
        except ValueError:
            return ""
    return row.availability.data_as_of or ""


def retention_identity(row: ResearchRow) -> str:
    """为未解析来源保存局部容量分组，使用来源内代码但不构造股票身份。

    输入无实体cb_daily代码B，输出unresolved/B；无代码新闻仍输出unresolved。
    """
    if row.entity_id is not None:
        return row.entity_id
    code = next(
        (
            row.text(field)
            for field in ("ts_code", "exchange", "name", "o_code")
            if row.text(field) is not None
        ),
        None,
    )
    return "unresolved" if code is None else "unresolved/" + code


def inspect_research_source(
    connection: duckdb.DuckDBPyConnection,
    entry: ConsumerSpec,
    contract: ReferenceContract,
    manifest: dict[str, Any] | None,
    tables: set[tuple[str, str]],
    cutoff: str,
    config: ResearchConfig,
    *,
    count_rows: bool = True,
) -> tuple[set[str], bool]:
    """允许局部业务字段缺失的来源做质量观察，主键、元数据和账本仍严格检查。

    输入有amount但缺close的daily，输出字段缺口并读取已有字段；缺source_record_id
    仍拒绝整表。未出现的字段不会被补齐成零或伪造引用。
    count_rows=False用于无绑定的快速审计，例如空配置只读目录和字段描述。
    """
    available = None
    if ("accepted", entry.api) in tables:
        available = {
            row[0] for row in connection.execute(f'DESCRIBE accepted."{entry.api}"').fetchall()
        }
    selected = (
        entry
        if available is None
        else replace(
            entry,
            selected_fields=tuple(field for field in entry.selected_fields if field in available),
        )
    )
    reasons, readable = _inspect_source(
        connection,
        selected,
        contract,
        manifest,
        tables,
        cutoff,
        config.thresholds,
        count_rows=count_rows,
    )
    if available is not None and not set(entry.selected_fields) <= available:
        reasons.add("consumption_fields_missing")
    return reasons, readable


def source_manifests(
    connection: duckdb.DuckDBPyConnection,
    apis: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """只投影当前来源的有界清单字段，不读取参数、目录额外项或任意载荷。

    例如目标daily外有重复派生日历，仍只返回daily；目标本身重复则拒绝。
    """
    query = """SELECT api_name, json_object(
        'contract_hash', manifest::JSON->>'contract_hash',
        'observed_at', manifest::JSON->>'observed_at',
        'collection_status', manifest::JSON->>'collection_status',
        'quality', json_object(
            'source_rows', manifest::JSON->'quality'->'source_rows',
            'accepted_rows', manifest::JSON->'quality'->'accepted_rows',
            'quarantined_rows', manifest::JSON->'quality'->'quarantined_rows',
            'exact_duplicate_rows', manifest::JSON->'quality'->'exact_duplicate_rows'))
        FROM catalog WHERE api_name IN (SELECT unnest(?))"""
    counts = connection.execute(
        "SELECT api_name, count(*) FROM catalog "
        "WHERE api_name IN (SELECT unnest(?)) GROUP BY api_name",
        [sorted(apis)],
    ).fetchall()
    if any(count != 1 for _, count in counts):
        raise ValueError("reference_catalog_duplicate")
    return {
        api: json.loads(payload)
        for api, payload in connection.execute(query, [sorted(apis)]).fetchall()
    }


def visible_config(config: ResearchConfig, cutoff: str) -> ResearchConfig:
    """只保留截止前已观察且已公开的单位证书，不改写调用方配置。

    例如当前行的单位依据明天才公开，返回配置中剔除该证书，原始行仍可审计。
    """
    from ..pit import time_boundary

    certificates = tuple(
        certificate
        for certificate in config.foundation.certifications
        if all(
            time_boundary(ref.observed_at) <= time_boundary(cutoff)
            and (
                ref.public_available_at is None
                or time_boundary(ref.public_available_at) <= time_boundary(cutoff)
            )
            for ref in certificate.evidence_refs
        )
    )
    return replace(config, foundation=replace(config.foundation, certifications=certificates))


def load_inputs(
    reference_path: Path,
    cutoff: str,
    run_id: str,
    config: ResearchConfig,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[ResearchInputs, dict[str, int]]:
    """只读按512行扫描已绑定来源，返回有限研究观察集与扫描账本。

    例：无实体/时间绑定时scanned_rows=0；缺表输出原因而非虚构空市场。
    任一来源超过全局容量则整个来源降级，决不使用被截断的精确总量。
    """
    effective_config = visible_config(config, cutoff)
    contracts = load_reference_contracts(load_policies())
    registry = {entry.api: entry for entry in load_registry()}
    rows_by_api: dict[str, tuple[ResearchRow, ...]] = {}
    source_reasons: dict[str, tuple[str, ...]] = {}
    local_reasons: dict[str, tuple[str, ...]] = {}
    counts = {"scanned_rows": 0, "retained_rows": 0, "excluded_rows": 0, "batches": 0}
    with duckdb.connect(
        str(reference_path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        tables = set(
            connection.execute(
                "SELECT table_schema,table_name FROM information_schema.tables"
            ).fetchall()
        )
        if ("main", "catalog") not in tables:
            raise ValueError("reference_catalog_missing")
        manifests = source_manifests(connection, SOURCE_APIS)
        for api in sorted(SOURCE_APIS):
            entry = registry.get(api)
            if entry is None:
                source_reasons[api] = ("source_not_registered",)
                continue
            entity_bound = any(item.source_api == api for item in config.foundation.entity_mappings)
            time_bound = any(item.source_api == api for item in config.foundation.time_mappings)
            configured = time_bound and (entity_bound or api in UNRESOLVED_APIS)
            reasons, readable = inspect_research_source(
                connection,
                entry,
                contracts[api],
                manifests.get(api),
                tables,
                cutoff,
                config,
                count_rows=configured,
            )
            if not configured:
                reasons.add("research_bindings_missing")
                readable = False
            if not readable:
                source_reasons[api] = tuple(sorted(reasons))
                continue
            groups: dict[tuple[str, str, str], list[ResearchRow]] = defaultdict(list)
            entity_counts: dict[str, int] = {}
            overflow_entities: set[str] = set()
            overflow_source = False
            for batch in batches(
                reference_rows(connection, api, batch_size=config.thresholds.fact_batch_size),
                config.thresholds.fact_batch_size,
            ):
                counts["batches"] += 1
                counts["scanned_rows"] += len(batch)
                for raw in batch:
                    try:
                        row = adapt_row(raw, contracts[api], effective_config, cutoff)
                    except (ValueError, TypeError, KeyError, OverflowError) as error:
                        counts["excluded_rows"] += 1
                        reasons.add("rows_excluded_by_adapter")
                        reasons.add(adapter_reason(error))
                        if (
                            api not in UNRESOLVED_APIS
                            and isinstance(error, ValueError)
                            and str(error) in {"entity_unresolved", "entity_business_time_unknown"}
                        ):
                            reasons.add("stock_entity_unresolved")
                        continue
                    identity = retention_identity(row)
                    if overflow_source or identity in overflow_entities:
                        continue
                    if retain_current(groups, row, config, entity_counts=entity_counts):
                        overflow_entities.add(identity)
                        local_reasons[api + "/" + identity] = ("entity_source_capacity_exceeded",)
                        groups = defaultdict(
                            list,
                            {key: values for key, values in groups.items() if key[0] != identity},
                        )
                retained_count = sum(map(len, groups.values()))
                if retained_count > min(
                    config.maximum_source_rows, config.maximum_total_rows - counts["retained_rows"]
                ):
                    overflow_source = True
                    groups.clear()
                    reasons.add("source_capacity_exceeded")
                if progress is not None:
                    progress({"stage": "scan_reference", "source_api": api, **counts})
            retained = tuple(
                prepare_numeric(row, effective_config)
                for key in sorted(groups)
                for row in sorted(groups[key], key=lambda item: str(item.raw["source_record_id"]))
            )
            counts["retained_rows"] += len(retained)
            rows_by_api[api] = retained
            source_reasons[api] = tuple(sorted(reasons))
    counts["discarded_rows"] = (
        counts["scanned_rows"] - counts["retained_rows"] - counts["excluded_rows"]
    )
    return ResearchInputs(
        analysis_as_of=cutoff,
        run_id=run_id,
        config=effective_config,
        rows_by_api=rows_by_api,
        source_reasons=source_reasons,
        entity_source_reasons=local_reasons,
    ), counts
