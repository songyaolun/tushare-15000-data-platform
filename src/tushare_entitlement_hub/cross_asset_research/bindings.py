"""全206接口绑定配置构建：旧99门禁复用，新增来源采用明确契约与命名空间。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb

from ..catalog import load_policies
from ..equity_research.adapter import inspect_research_source, source_manifests
from ..equity_research.binding_builder import (
    MASTER_FIELDS,
    METADATA,
    binding_availability,
    build_research_config,
    configuration_hash,
    safe_row_reason,
)
from ..equity_research.binding_entities import EntityBuild
from ..equity_research.binding_io import BuildLimits, check_paths, projected_rows
from ..equity_research.binding_rules import PUBLIC_FIELDS, period_visible
from ..equity_research.binding_units import UnitBuild
from ..equity_research.consumption_registry import load_consumption_registry
from ..equity_research.models import ResearchConfig
from ..product_foundation.bindings import TimeMapping
from ..product_foundation.registry import ConsumerSpec
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..reference_storage import file_checksum
from ..standard import is_standard_numeric_field, normalize_timestamp
from ..store import digest, encode
from .binding_rules import (
    ALLOWED_CURRENCIES,
    BINDING_VERSION,
    DIRECTORY_IDENTITY_FIELDS,
    EVENT_APIS,
    MASTER_KINDS,
    MASTER_NAMESPACES,
    OBSERVED_MASTERS,
    ROW_UNIT_VERSION,
    STOCK_APIS,
    cross_registry,
    entity_mapping,
    rules_fingerprint,
    time_registry,
)

DEFAULT_LIMITS = BuildLimits()


@dataclass(frozen=True)
class BuildResult:
    """配置与独立审计对象；例如result.report和result.audit引用同一份构建报告。"""

    config: ResearchConfig
    audit: dict[str, Any]

    @property
    def report(self) -> dict[str, Any]:
        """提供报告命名的兼容入口；例如result.report['version']返回统一绑定版本。"""
        return self.audit


def projection_fields(
    columns: set[str],
    entry: ConsumerSpec,
    mapping: TimeMapping,
) -> set[str]:
    """只投影消费、身份、单位和时间字段，正文保留受单行字节预算限制。

    例如fund_daily保留close、vol、amount及来源证据，不物化source_fields原始大对象。
    """
    temporal = {
        mapping.data_field,
        mapping.report_period_field,
        mapping.effective_from_field,
        mapping.effective_to_field,
    } - {None}
    return columns & (
        METADATA
        | MASTER_FIELDS
        | PUBLIC_FIELDS
        | set(entry.selected_fields + entry.business_key)
        | set(mapping.public_fields + mapping.optional_public_fields)
        | set(DIRECTORY_IDENTITY_FIELDS.get(entry.api, ()))
        | {entity_mapping(entry.api).key_field}
        | temporal
        | {
            "list_date",
            "delist_date",
            "vol",
            "amount",
            "price_type",
            "market_activity_status",
            "price_observation_status",
            "currency",
            "curr_type",
            "org_code",
        }
    )


def scan_source(
    source: duckdb.DuckDBPyConnection,
    contract: ReferenceContract,
    mapping: TimeMapping,
    fields: set[str],
    cutoff: str,
    limits: BuildLimits,
    entities: EntityBuild,
) -> tuple[dict[str, Any], UnitBuild, str | None]:
    """完整分批审计一源，复用公开时间与单位门禁，不按扫描数量宣称消费完成。

    例如两行etf_share_size.total_share均为万份时累积一份证书；单位冲突使整列未知。
    本行明确币种的金额字段另登记逐行认证许可，混币不被解释成来源没有数据。
    """
    counts: Counter[str] = Counter()
    units = UnitBuild(
        allowed_currencies=ALLOWED_CURRENCIES,
        inherit_stock_currency=mapping.source_api in STOCK_APIS,
        preserve_contract_lots=True,
        report_currency_variation=True,
        prefer_row_currency=True,
    )
    names = {
        item["name"]
        for item in contract.documentation["output_fields"]
        if is_standard_numeric_field(item["name"], item["type"], contract.policy)
    } & fields
    scanned = eligible = 0
    latest = None
    missing_date = False
    row_currency_fields: set[str] = set()
    for row in projected_rows(source, mapping.source_api, fields, limits):
        scanned += 1
        try:
            fact = binding_availability(row, contract, mapping, cutoff, event_apis=EVENT_APIS)
            eligible += 1
            units.add(row, fact, contract, names, entities)
            if row.get("currency") is not None or row.get("curr_type") is not None:
                row_units = UnitBuild(
                    allowed_currencies=ALLOWED_CURRENCIES,
                    inherit_stock_currency=False,
                    preserve_contract_lots=True,
                )
                row_units.add(row, fact, contract, names, entities)
                row_currency_fields.update(
                    certificate.field
                    for certificate in row_units.accepted()
                    if certificate.dimension in {"money", "price"}
                )
            if fact.data_as_of is None:
                missing_date = True
            elif period_visible(fact.data_as_of, cutoff, mapping.market_timezone or "UTC"):
                latest = max(latest or "", fact.data_as_of)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            counts[safe_row_reason(error)] += 1
    expected = latest if not missing_date and mapping.source_api not in EVENT_APIS else None
    if expected is None:
        counts["business_freshness_uncertified"] += 1
    if not eligible:
        counts["no_eligible_rows"] += 1
    if mapping.market_timezone is None:
        counts["market_timezone_uncertified"] += 1
    return (
        {
            "readable": True,
            "configured": True,
            "scanned_rows": scanned,
            "eligible_rows": eligible,
            "reason_counts": dict(sorted(counts.items())),
            "unit_gaps": {
                key: dict(sorted(values.items())) for key, values in sorted(units.gaps.items())
            },
            "row_currency_fields": sorted(row_currency_fields),
        },
        units,
        expected,
    )


def build_config(
    reference: Path,
    cutoff: str,
    *,
    limits: BuildLimits = DEFAULT_LIMITS,
) -> BuildResult:
    """从只读reference生成206接口配置与逐源审计，旧99仍经过原构建器全部门禁。

    输入含206个唯一清单的合成库，输出配置和all-products-bindings/v1报告；
    当前来源清单缺失、重复、契约不一致或容量溢出均拒绝，目录额外项不参与判断。
    """
    check_paths(reference)
    cutoff = normalize_timestamp(cutoff)
    legacy = build_research_config(reference, cutoff, limits=limits)
    before = legacy.audit["reference_sha256"]
    contracts = load_reference_contracts(load_policies())
    entries = {entry.api: entry for entry in cross_registry()}
    mappings = {mapping.source_api: mapping for mapping in time_registry(contracts)}
    sources = {
        api: dict(report, configured=True) for api, report in legacy.audit["sources"].items()
    }
    entities = EntityBuild(
        maximum_entities=limits.maximum_entities,
        extra_namespaces=MASTER_NAMESPACES,
        extra_kinds=MASTER_KINDS,
        extra_key_fields={api: entity_mapping(api).key_field for api in MASTER_NAMESPACES},
        observation_only=OBSERVED_MASTERS,
        directory_identity_fields=DIRECTORY_IDENTITY_FIELDS,
    )
    certificates = list(legacy.config.foundation.certifications)
    dates = dict(legacy.config.foundation.expected_data_as_of)
    total = legacy.audit["summary"]["scanned_rows"]
    with duckdb.connect(
        str(reference), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        tables = set(
            connection.execute(
                "SELECT table_schema,table_name FROM information_schema.tables"
            ).fetchall()
        )
        manifests = source_manifests(connection, frozenset(entries))
        if set(manifests) != set(entries):
            raise ValueError("reference_manifest_missing")
        projections = {}
        for api, entry in entries.items():
            if manifests[api]["contract_hash"] != contracts[api].fingerprint:
                raise ValueError("source_contract_mismatch")
            reasons, readable = inspect_research_source(
                connection,
                entry,
                contracts[api],
                manifests[api],
                tables,
                cutoff,
                legacy.config,
            )
            sources[api] = {
                "readable": readable,
                "configured": True,
                "scanned_rows": 0,
                "eligible_rows": 0,
                "reason_counts": {reason: 1 for reason in sorted(reasons)},
                "unit_gaps": {},
            }
            if not readable:
                continue
            count = manifests[api]["quality"]["accepted_rows"]
            total += count
            if count > limits.maximum_source_rows or total > limits.maximum_total_rows:
                raise ValueError("binding_scan_capacity_exceeded")
            columns = {
                row[0] for row in connection.execute(f'DESCRIBE accepted."{api}"').fetchall()
            }
            projections[api] = projection_fields(columns, entry, mappings[api])
        # 股票主数据复用相同身份构建器，供新增股票来源追溯币种证据。
        stock_mapping = next(
            item
            for item in legacy.config.foundation.time_mappings
            if item.source_api == "stock_basic"
        )
        stock_columns = {
            row[0] for row in connection.execute('DESCRIBE accepted."stock_basic"').fetchall()
        }
        master_sources = {"stock_basic": stock_columns & (METADATA | MASTER_FIELDS)} | {
            api: fields for api, fields in projections.items() if api in MASTER_NAMESPACES
        }
        for api, fields in master_sources.items():
            mapping = stock_mapping if api == "stock_basic" else mappings[api]
            for row in projected_rows(connection, api, fields, limits):
                try:
                    entities.add(row, binding_availability(row, contracts[api], mapping, cutoff))
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    if str(error) == "binding_entity_capacity_exceeded":
                        raise
                    entities.gap(api, safe_row_reason(error))
        entities.materialize()
        for api, fields in sorted(projections.items()):
            report, units, expected = scan_source(
                connection,
                contracts[api],
                mappings[api],
                fields,
                cutoff,
                limits,
                entities,
            )
            report["reason_counts"] = dict(
                sorted(
                    (
                        Counter(sources[api]["reason_counts"])
                        + Counter(report["reason_counts"])
                        + entities.reasons.get(api, Counter())
                    ).items()
                )
            )
            sources[api] = report
            if "public_field_mapping_missing" in report["reason_counts"]:
                raise ValueError("public_field_mapping_missing")
            certificates.extend(units.accepted())
            if expected is not None:
                dates[api] = expected
    check_paths(reference)
    if file_checksum(reference) != before:
        raise ValueError("reference_changed_during_read")
    all_entities = {item.identity.entity_id: item for item in legacy.config.foundation.entities}
    all_entities.update({item.identity.entity_id: item for item in entities.entities})
    if len(all_entities) > limits.maximum_entities:
        raise ValueError("binding_entity_capacity_exceeded")
    all_aliases = {
        item.to_json(): item for item in (*legacy.config.foundation.aliases, *entities.aliases)
    }
    config = replace(
        legacy.config,
        foundation=replace(
            legacy.config.foundation,
            entities=tuple(all_entities[key] for key in sorted(all_entities)),
            aliases=tuple(all_aliases[key] for key in sorted(all_aliases)),
            certifications=tuple(certificates),
            time_mappings=(*legacy.config.foundation.time_mappings, *mappings.values()),
            entity_mappings=(
                *legacy.config.foundation.entity_mappings,
                *(entity_mapping(api) for api in entries),
            ),
            expected_data_as_of=dates,
        ),
        binding_audit_json="{}",
    )
    configuration_fingerprint = configuration_hash(config)
    audit = {
        "version": BINDING_VERSION,
        "reference_sha256": before,
        "generated_as_of": cutoff,
        "rules_fingerprint": rules_fingerprint(contracts),
        "configuration_sha256": configuration_fingerprint,
        "config_sha256": configuration_fingerprint,
        "historical_pit_certified": False,
        "row_unit_version": ROW_UNIT_VERSION,
        "row_currency_fields": {
            api: report["row_currency_fields"]
            for api, report in sources.items()
            if report.get("row_currency_fields")
        },
        "sources": sources,
        "reason_codes": ["historical_pit_uncertified", "universe_scope_uncertified"],
        "summary": {
            "registered_apis": 206,
            "configured_apis": len(sources),
            "readable_apis": sum(report["readable"] for report in sources.values()),
            "scanned_rows": sum(report["scanned_rows"] for report in sources.values()),
            "observed_apis": 0,
            "entities": len(all_entities),
            "aliases": len(all_aliases),
            "certifications": len(certificates),
        },
    }
    audit["audit_sha256"] = digest(audit)
    return BuildResult(replace(config, binding_audit_json=encode(audit)), audit)


def equity_config(config: ResearchConfig) -> ResearchConfig:
    """提取旧99接口配置并清除统一审计嵌套，避免海外security进入A股候选范围。

    例如同时存在tushare_stock与tushare_us证券时，旧入口仅收到股票实体及旧来源绑定。
    """
    apis = {item.spec.api for item in load_consumption_registry()}
    foundation = config.foundation
    entities = tuple(
        item
        for item in foundation.entities
        if item.identity.namespace not in set(MASTER_NAMESPACES.values())
    )
    identities = {item.identity.entity_id for item in entities}
    return replace(
        config,
        foundation=replace(
            foundation,
            entities=entities,
            aliases=tuple(item for item in foundation.aliases if item.entity_id in identities),
            certifications=tuple(
                item for item in foundation.certifications if item.source_api in apis
            ),
            time_mappings=tuple(
                item for item in foundation.time_mappings if item.source_api in apis
            ),
            entity_mappings=tuple(
                item for item in foundation.entity_mappings if item.source_api in apis
            ),
            expected_data_as_of={
                api: day for api, day in foundation.expected_data_as_of.items() if api in apis
            },
        ),
        peer_paths={key: value for key, value in config.peer_paths.items() if key in identities},
        confirmed_relations=tuple(
            item
            for item in config.confirmed_relations
            if item.from_entity_id in identities and item.to_entity_id in identities
        ),
        binding_audit_json="{}",
    )


def verify_binding_audit(config: ResearchConfig, reference_hash: str, cutoff: str) -> None:
    """运行前重验统一配置、规则与来源指纹；手工空审计保留显式配置的既有语义。

    例如换了reference而仍使用原证书，抛binding_reference_mismatch。
    """
    audit = config.binding_audit
    if not audit:
        return
    expected = audit.pop("audit_sha256", None)
    if expected != digest(audit):
        raise ValueError("binding_audit_mismatch")
    if audit.get("reference_sha256") != reference_hash:
        raise ValueError("binding_reference_mismatch")
    if audit.get("generated_as_of") != cutoff:
        raise ValueError("binding_cutoff_mismatch")
    configuration_fingerprint = configuration_hash(config)
    if (
        audit.get("version") != BINDING_VERSION
        or audit.get("row_unit_version") != ROW_UNIT_VERSION
        or audit.get("rules_fingerprint")
        != rules_fingerprint(load_reference_contracts(load_policies()))
        or audit.get("configuration_sha256") != configuration_fingerprint
        or audit.get("config_sha256") != configuration_fingerprint
    ):
        raise ValueError("binding_configuration_mismatch")
