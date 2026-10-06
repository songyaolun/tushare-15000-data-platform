"""独立的正式reference研究绑定构建器：只读扫描、显式证据和确定性审计。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb

from ..catalog import load_policies
from ..pit import time_boundary
from ..product_foundation.bindings import EntityMapping, FoundationBindings, TimeMapping
from ..product_foundation.numeric import UnitCertification
from ..product_foundation.reference_scan import map_availability
from ..product_foundation.temporal import AvailabilityFact
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..reference_storage import file_checksum
from ..standard import normalize_timestamp
from ..store import digest, encode
from .adapter import inspect_research_source
from .binding_entities import MASTER_APIS, NAMESPACES, EntityBuild
from .binding_io import BuildLimits, check_paths, projected_rows, publish_files
from .binding_rules import (
    BINDING_VERSION,
    PUBLIC_FIELDS,
    period_visible,
    rules_fingerprint,
    time_registry,
)
from .binding_units import UnitBuild
from .consumption_registry import BOND_APIS, EVENT_APIS, INDEX_APIS, load_consumption_registry
from .models import ResearchConfig

DEFAULT_LIMITS = BuildLimits()

METADATA = {
    "source_api",
    "source_record_id",
    "business_group_key",
    "business_variant_key",
    "contract_hash",
    "observed_at",
    "quality_tier",
    "is_quarantined",
    "historical_pit_certified",
    "source_units",
    "standard_units",
    "source_occurrences",
    "identity_resolution",
    "currency",
    "curr_type",
}
MASTER_FIELDS = {
    "ts_code",
    "o_code",
    "n_code",
    "name",
    "list_date",
    "delist_date",
    "exp_date",
    "list_status",
    "industry",
    "market",
    "stk_code",
    "cb_type",
}


@dataclass(frozen=True)
class BuildResult:
    """生成配置和同内容审计；调用方可以先检查再显式保存。"""

    config: ResearchConfig
    audit: dict[str, Any]


def binding_availability(
    row: dict[str, Any],
    contract: ReferenceContract,
    mapping: TimeMapping,
    cutoff: str,
    *,
    event_apis: frozenset[str] = EVENT_APIS,
) -> AvailabilityFact:
    """对来源身份、变体和全部公开时间做失败关闭校验，再映射各类时间。

    输入旧报告期与未来公告抛binding_future_public_time；缺公开字段仍只从观察时点可用。
    已知未来事件允许保留生效日，但不得作为expected_data_as_of的新鲜度基准。
    例如显式event_apis含forecast时，已公告的未来报告期仍可作为预期证据。
    """
    if any(
        row.get(name) is not None
        and name not in (mapping.public_fields + mapping.optional_public_fields)
        for name in PUBLIC_FIELDS
    ):
        raise ValueError("public_field_mapping_missing")
    if (
        row.get("source_api") != contract.policy.api_name
        or row.get("contract_hash") != contract.fingerprint
    ):
        raise ValueError("source_contract_mismatch")
    if (
        row.get("business_variant_key")
        or row.get("identity_resolution") == "business_group_source_variant"
    ):
        raise ValueError("business_variant_unresolved")
    if row.get("quality_tier") != "verified" or row.get("is_quarantined") is True:
        raise ValueError("binding_quality_unverified")
    fact = map_availability(row, contract, mapping)
    if fact.public_available_at is not None and time_boundary(
        fact.public_available_at
    ) > time_boundary(cutoff):
        raise ValueError("binding_future_public_time")
    if time_boundary(fact.observed_at) > time_boundary(cutoff):
        raise ValueError("binding_future_observation")
    if (
        fact.data_as_of is not None
        and not period_visible(fact.data_as_of, cutoff, mapping.market_timezone or "Asia/Shanghai")
        and contract.policy.api_name not in event_apis
    ):
        raise ValueError("future_business_date")
    return fact


def projection_fields(
    columns: set[str],
    contract: ReferenceContract,
    mapping: TimeMapping,
) -> set[str]:
    """取有限数值、身份与时间列，明确排除正文、条款和原始字段大对象。

    输入major_news全部列，输出含pub_time和来源元数据而不含content/title/source_fields。
    公开字段即使不在注册里也投影，用于发现遗漏并阻断该接口。
    """
    numeric = {
        item["name"]
        for item in contract.documentation["output_fields"]
        if item["type"].lower() in {"int", "float", "double"}
    }
    selected = next(
        item.spec.selected_fields
        for item in load_consumption_registry()
        if item.spec.api == contract.policy.api_name
    )
    numeric &= set(selected)
    times = {
        mapping.data_field,
        mapping.report_period_field,
        mapping.effective_from_field,
        mapping.effective_to_field,
    } - {None}
    return columns & (METADATA | MASTER_FIELDS | PUBLIC_FIELDS | numeric | times)


def safe_row_reason(error: Exception) -> str:
    """只允许固定错误码进入审计，不回显异常载荷；例：非法JSON返回binding_row_invalid。

    已知future_public等原因保持精确，第三方异常或来源自带文本不会写入报告。
    """
    allowed = {
        "public_field_mapping_missing",
        "source_contract_mismatch",
        "business_variant_unresolved",
        "binding_quality_unverified",
        "binding_future_public_time",
        "binding_future_observation",
        "future_business_date",
        "invalid_effective_interval",
        "not_yet_effective",
        "no_longer_effective",
        "public_time_unverified",
        "source_pit_state_unverified",
        "invalid_timestamp_or_timezone",
        "invalid_period",
    }
    return str(error) if str(error) in allowed else "binding_row_invalid"


def entity_mappings(mappings: tuple[TimeMapping, ...]) -> tuple[EntityMapping, ...]:
    """给股票、指数和转债使用独立别名命名空间，市场及无实体事件不强行造身份。

    输入99个时间映射，输出需要身份的API绑定；例：cb_daily映射tushare_cb。
    """
    result = []
    for mapping in mappings:
        api = mapping.source_api
        if api in BOND_APIS:
            namespace = NAMESPACES["cb_basic"]
        elif api in INDEX_APIS | {"idx_anns"}:
            namespace = NAMESPACES["index_basic"]
        elif api not in {
            "trade_cal",
            "eco_cal",
            "daily_info",
            "sz_daily_info",
            "shibor_lpr",
            "stk_account",
            "moneyflow_mkt_dc",
            "cn_pmi",
            "cn_ppi",
            "cn_m",
            "cn_cpi",
            "cn_gdp",
            "sf_month",
            "factor_list",
            "factor_value",
            "hm_list",
            "bse_mapping",
            "major_news",
            "cn_schedule",
        }:
            namespace = NAMESPACES["stock_basic"]
        else:
            continue
        result.append(EntityMapping(source_api=api, namespace=namespace, key_field="ts_code"))
    return tuple(result)


def scan_source(
    source: duckdb.DuckDBPyConnection,
    contract: ReferenceContract,
    mapping: TimeMapping,
    fields: set[str],
    cutoff: str,
    limits: BuildLimits,
    entities: EntityBuild,
) -> tuple[dict[str, Any], UnitBuild, str | None]:
    """完整分批审计一个可读来源，保留字段级缺口及可证明的当前业务期间。

    输入两行daily，返回scanned_rows=2及一致单位证书；公开字段注册遗漏会阻断整源。
    expected日期只来自非事件、非快照的真实业务字段，含缺日期行则不声明完整日期。
    """
    counts: Counter[str] = Counter()
    units = UnitBuild()
    scanned = eligible = 0
    latest = None
    missing_date = False
    names = {
        field["name"]
        for field in contract.documentation["output_fields"]
        if field["type"].lower() in {"int", "float", "double"}
    } & fields
    for row in projected_rows(source, mapping.source_api, fields, limits):
        scanned += 1
        try:
            fact = binding_availability(row, contract, mapping, cutoff)
            eligible += 1
            units.add(row, fact, contract, names, entities)
            if fact.data_as_of is None:
                missing_date = True
            elif period_visible(
                fact.data_as_of, cutoff, mapping.market_timezone or "Asia/Shanghai"
            ):
                latest = max(latest or "", fact.data_as_of)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            counts[safe_row_reason(error)] += 1
    expected = latest if not missing_date and mapping.source_api not in EVENT_APIS else None
    if expected is None:
        counts["business_freshness_uncertified"] += 1
    if eligible == 0:
        counts["no_eligible_rows"] += 1
    return (
        {
            "readable": True,
            "scanned_rows": scanned,
            "eligible_rows": eligible,
            "reason_counts": dict(sorted(counts.items())),
            "unit_gaps": {
                name: dict(sorted(codes.items())) for name, codes in sorted(units.gaps.items())
            },
        },
        units,
        expected,
    )


def build_research_config(
    reference_path: Path,
    analysis_as_of: str,
    *,
    limits: BuildLimits = DEFAULT_LIMITS,
) -> BuildResult:
    """从只读正式reference结构构建确定性研究配置，不访问缓存、网络或凭据。

    输入合成stock_basic及daily，输出实体、时间、可认证单位和完整99源缺口报告。
    文件SHA256在扫描前后比对；99项必需manifest缺失、重复或契约不符均拒绝。
    目录仅查询这99项，额外派生日历等目录项不参与绑定或数量限制。
    """
    check_paths(reference_path)
    cutoff = normalize_timestamp(analysis_as_of)
    before = file_checksum(reference_path)
    contracts = load_reference_contracts(load_policies())
    mappings = time_registry(contracts)
    by_api = {mapping.source_api: mapping for mapping in mappings}
    entries = {item.spec.api: item.spec for item in load_consumption_registry()}
    entities = EntityBuild(limits.maximum_entities)
    sources: dict[str, dict[str, Any]] = {}
    projections = {}
    total = 0
    certificates: list[UnitCertification] = []
    expected_dates = {}
    with duckdb.connect(
        str(reference_path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as source:
        tables = set(
            source.execute(
                "SELECT table_schema,table_name FROM information_schema.tables"
            ).fetchall()
        )
        if ("main", "catalog") not in tables:
            raise ValueError("reference_catalog_missing")
        if ("accepted", "stock_basic") not in tables:
            raise ValueError("binding_stock_basic_missing")
        required_apis = sorted(entries)
        catalog_counts = source.execute(
            "SELECT api_name,count(*) FROM catalog "
            "WHERE api_name IN (SELECT unnest(?)) GROUP BY api_name",
            [required_apis],
        ).fetchall()
        if any(count != 1 for _, count in catalog_counts):
            raise ValueError("reference_catalog_duplicate")
        if {api for api, _ in catalog_counts} != set(required_apis):
            raise ValueError("reference_manifest_missing")
        manifests = {}
        # 只读消费所需的清单字段，不把请求参数或其他大对象载入Python。
        manifest_query = """SELECT api_name, json_object(
            'contract_hash', manifest::JSON->>'contract_hash',
            'observed_at', manifest::JSON->>'observed_at',
            'collection_status', manifest::JSON->>'collection_status',
            'quality', json_object(
                'source_rows', manifest::JSON->'quality'->'source_rows',
                'accepted_rows', manifest::JSON->'quality'->'accepted_rows',
                'quarantined_rows', manifest::JSON->'quality'->'quarantined_rows',
                'exact_duplicate_rows', manifest::JSON->'quality'->'exact_duplicate_rows'))
            FROM catalog WHERE api_name IN (SELECT unnest(?))"""
        for api, payload in source.execute(manifest_query, [required_apis]).fetchall():
            manifest = json.loads(payload)
            if manifest["contract_hash"] != contracts[api].fingerprint:
                raise ValueError("source_contract_mismatch")
            manifests[api] = manifest
        for mapping in mappings:
            api = mapping.source_api
            reasons, readable = inspect_research_source(
                source,
                entries[api],
                contracts[api],
                manifests.get(api),
                tables,
                cutoff,
                ResearchConfig(),
            )
            sources[api] = {
                "readable": readable,
                "scanned_rows": 0,
                "eligible_rows": 0,
                "reason_counts": {code: 1 for code in sorted(reasons)},
                "unit_gaps": {},
            }
            if not readable:
                continue
            count = source.execute(f'SELECT count(*) FROM accepted."{api}"').fetchall()[0][0]
            total += count
            if count > limits.maximum_source_rows or total > limits.maximum_total_rows:
                raise ValueError("binding_scan_capacity_exceeded")
            columns = {row[0] for row in source.execute(f'DESCRIBE accepted."{api}"').fetchall()}
            projections[api] = projection_fields(columns, contracts[api], mapping)
        # 主数据先建立有限身份集，其余大表仅流式累计单位和审计状态。
        for api in MASTER_APIS:
            if api not in projections:
                continue
            for row in projected_rows(source, api, projections[api], limits):
                try:
                    entities.add(
                        row, binding_availability(row, contracts[api], by_api[api], cutoff)
                    )
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    if str(error) == "binding_entity_capacity_exceeded":
                        raise
                    entities.gap(api, safe_row_reason(error))
        entities.materialize()
        for api in sorted(projections):
            report, units, expected = scan_source(
                source, contracts[api], by_api[api], projections[api], cutoff, limits, entities
            )
            report["reason_counts"] = dict(
                sorted(
                    (
                        Counter(sources[api]["reason_counts"]) + Counter(report["reason_counts"])
                    ).items()
                )
            )
            sources[api] = report
            if "public_field_mapping_missing" in report["reason_counts"]:
                raise ValueError("public_field_mapping_missing")
            certificates.extend(units.accepted())
            if expected is not None:
                expected_dates[api] = expected
    check_paths(reference_path)
    if file_checksum(reference_path) != before:
        raise ValueError("reference_changed_during_read")
    for api, entity_reasons in entities.reasons.items():
        sources[api]["reason_counts"] = dict(
            sorted((Counter(sources[api]["reason_counts"]) + entity_reasons).items())
        )
    selected_mappings = mappings
    config = ResearchConfig(
        foundation=FoundationBindings(
            entities=tuple(entities.entities),
            aliases=tuple(sorted(entities.aliases, key=lambda item: item.to_json())),
            certifications=tuple(certificates),
            time_mappings=selected_mappings,
            entity_mappings=entity_mappings(selected_mappings),
            expected_data_as_of=expected_dates,
        ),
        peer_paths=entities.peers,
        confirmed_relations=tuple(entities.relations),
    )
    audit = {
        "version": BINDING_VERSION,
        "reference_sha256": before,
        "generated_as_of": cutoff,
        "rules_fingerprint": rules_fingerprint(contracts),
        "configuration_sha256": configuration_hash(config),
        "mode": "observed_research",
        "historical_pit_certified": False,
        "peer_basis": "stock_basic.industry/current_observation_only",
        "reason_codes": [
            "report_types_uncertified",
            "historical_pit_uncertified",
            "universe_scope_uncertified",
        ],
        "sources": sources,
        "summary": {
            "configured_apis": len(selected_mappings),
            "readable_apis": sum(item["readable"] for item in sources.values()),
            "observed_apis": 0,
            "unresolved_apis": sum(
                bool(item["reason_counts"] or item["unit_gaps"]) for item in sources.values()
            ),
            "scanned_rows": sum(item["scanned_rows"] for item in sources.values()),
            "entities": len(entities.entities),
            "aliases": len(entities.aliases),
            "certifications": len(certificates),
            "relations": len(entities.relations),
        },
    }
    audit["audit_sha256"] = digest(audit)
    return BuildResult(replace(config, binding_audit_json=encode(audit)), audit)


def configuration_hash(config: ResearchConfig) -> str:
    """对实际绑定内容生成摘要，排除自引用的审计字段；例：改别名会改变64位摘要。"""
    payload = config.to_dict()
    payload.pop("binding_audit_json")
    return digest(payload)


def verify_binding_audit(config: ResearchConfig, reference_hash: str, cutoff: str) -> None:
    """运行前核对自动生成配置的来源、规则、截止和内容，不允许静默迁移认证。

    输入旧reference指纹抛binding_reference_mismatch；手工空审计配置保留既有显式行为。
    """
    audit = config.binding_audit
    if not audit:
        return
    expected_audit_hash = audit.pop("audit_sha256", None)
    if expected_audit_hash != digest(audit):
        raise ValueError("binding_audit_mismatch")
    if audit.get("reference_sha256") != reference_hash:
        raise ValueError("binding_reference_mismatch")
    if audit.get("generated_as_of") != cutoff:
        raise ValueError("binding_cutoff_mismatch")
    if (
        audit.get("version") != BINDING_VERSION
        or audit.get("rules_fingerprint")
        != rules_fingerprint(load_reference_contracts(load_policies()))
        or audit.get("configuration_sha256") != configuration_hash(config)
    ):
        raise ValueError("binding_configuration_mismatch")


def write_research_config(
    reference_path: Path,
    output_path: Path,
    analysis_as_of: str,
    *,
    audit_path: Path | None = None,
    limits: BuildLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """先完成全量审计再保存独立配置和报告，stdout摘要不含任何本机路径或来源大字段。

    输入config.json默认同时保存config.audit.json；重复运行两个文件均字节一致。
    输入输出或审计路径冲突时，在读取数据及创建输出目录之前失败。
    """
    report_path = audit_path if audit_path is not None else output_path.with_suffix(".audit.json")
    check_paths(reference_path, output_path, report_path)
    result = build_research_config(reference_path, analysis_as_of, limits=limits)
    check_paths(reference_path, output_path, report_path)
    if file_checksum(reference_path) != result.audit["reference_sha256"]:
        raise ValueError("reference_changed_during_read")
    publish_files(
        ((report_path, encode(result.audit) + "\n"), (output_path, result.config.to_json() + "\n"))
    )
    return {"status": "built", **result.audit}


def binding_summary(config: ResearchConfig, consumption: dict[str, Any]) -> dict[str, Any]:
    """区分配置、可读与实际消费接口，单位缺口即使已观察也继续披露。

    输入构建审计与真实消费状态，输出configured/readable/observed/unresolved计数及原因。
    observed只取本次消费者结果，构建器的扫描数从不冒充消费数。
    """
    audit = config.binding_audit
    sources = audit["sources"]
    observed = {api for api, values in consumption["by_api"].items() if values.get("observed", 0)}
    unresolved = {}
    for api, source in sources.items():
        reasons = set(source["reason_counts"])
        if source["unit_gaps"]:
            reasons.add("unit_certification_gaps")
        if api not in observed:
            reasons.add("no_observed_consumption")
        if reasons:
            unresolved[api] = sorted(reasons)
    return {
        "reference_sha256": audit["reference_sha256"],
        "generated_as_of": audit["generated_as_of"],
        "rules_fingerprint": audit["rules_fingerprint"],
        "audit_sha256": audit["audit_sha256"],
        "configured_apis": len(config.foundation.time_mappings),
        "readable_apis": sum(source["readable"] for source in sources.values()),
        "observed_apis": len(observed),
        "unresolved_apis": len(unresolved),
        "unresolved_reasons": unresolved,
    }


def build_failure_code(error: Exception) -> str:
    """把构建失败分为固定机器码，不将DuckDB或文件异常中的路径和载荷写到stdout。

    输入缺stock_basic异常返回binding_stock_basic_missing，未知错误返回binding_io_failed。
    """
    allowed = {
        "reference_library_missing",
        "input_output_path_collision",
        "invalid_batch_size",
        "invalid_timestamp_or_timezone",
        "binding_limits_invalid",
        "binding_reference_wal_present",
        "binding_row_capacity_exceeded",
        "binding_entity_capacity_exceeded",
        "binding_scan_capacity_exceeded",
        "reference_catalog_missing",
        "reference_catalog_duplicate",
        "reference_manifest_missing",
        "source_contract_mismatch",
        "binding_stock_basic_missing",
        "reference_changed_during_read",
        "public_field_mapping_missing",
        "binding_time_contract_mismatch",
        "binding_time_registry_mismatch",
    }
    return str(error) if str(error) in allowed else "binding_io_failed"
