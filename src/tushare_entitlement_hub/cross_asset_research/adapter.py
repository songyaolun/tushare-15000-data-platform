"""新增来源流式适配，公开时间和数值门禁复用旧研究层，来源内部身份不冒充关联。"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb

from ..catalog import load_policies
from ..equity_research.adapter import (
    adapt_row as shared_adapt_row,
)
from ..equity_research.adapter import (
    adapter_reason,
    inspect_research_source,
    prepare_numeric,
    source_manifests,
    visible_config,
)
from ..equity_research.binding_entities import EntityBuild
from ..equity_research.binding_io import BuildLimits, check_paths, projected_rows
from ..equity_research.binding_units import UnitBuild
from ..equity_research.models import ResearchConfig, ResearchInputs, ResearchRow
from ..product_foundation.batch_io import batches
from ..product_foundation.entities import resolve_alias
from ..product_foundation.numeric import UnitCertification
from ..product_foundation.progress import ProgressCallback
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..reference_storage import file_checksum
from ..standard import restore_standard_numeric_text
from ..store import digest, encode
from .binding_rules import (
    ALLOWED_CURRENCIES,
    EVENT_APIS,
    OBSERVED_MASTERS,
    SECURITY_APIS,
    SOURCE_SNAPSHOT_APIS,
    cross_registry,
    current_lifecycle_reason,
    entity_mapping,
)
from .bindings import projection_fields, verify_binding_audit
from .retention import RetentionGroup

SOURCE_APIS = frozenset(entry.api for entry in cross_registry())
UNRESOLVED_APIS = (SOURCE_APIS - SECURITY_APIS) | SOURCE_SNAPSHOT_APIS
MAX_RETAINED_BYTES = 256 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
# 预期和财务修正需保留公告版本；其余来源按最近两期观察，而不是累计截断总量。
VERSION_APIS = EVENT_APIS | frozenset({"express", "express_vip", "report_rc", "stk_surv"})
MARKET_APIS = frozenset(
    {
        "gz_index",
        "hibor",
        "shibor",
        "slb_len",
        "us_tbr",
        "us_tltr",
        "us_trltr",
        "us_trycr",
        "us_tycr",
        "vix_index",
        "wz_index",
        "hk_tradecal",
        "us_tradecal",
    }
)


def adapt_row(
    raw: dict[str, Any],
    contract: ReferenceContract,
    config: ResearchConfig,
    cutoff: str,
    *,
    documented_kinds: dict[str, str] | None = None,
) -> ResearchRow:
    """先通过共享来源、公开时间和实体门禁，再为非证券证据声明来源内序列身份。

    例如fx_daily中的ABC和港股ABC绝不合并；source_local只支持本来源研究，
    不认证证券生命周期、历史PIT或跨来源映射。证券来源缺主数据仍拒绝。
    """
    if (
        raw.get("business_variant_key")
        or raw.get("identity_resolution") == "business_group_source_variant"
    ):
        raise ValueError("business_variant_unresolved")
    row = shared_adapt_row(
        raw,
        contract,
        config,
        cutoff,
        unresolved_apis=UNRESOLVED_APIS,
        event_apis=EVENT_APIS,
    )
    kinds = documented_kinds if documented_kinds is not None else {
        item["name"]: item["type"] for item in contract.documentation["output_fields"]
    }
    raw = restore_standard_numeric_text(raw, kinds, contract.policy)
    row = replace(row, raw=raw)
    if lifecycle_reason := current_lifecycle_reason(row.source_api, row.availability, cutoff):
        raise ValueError(lifecycle_reason)
    if row.entity_id is not None:
        return replace(row, raw={**raw, "entity_basis": "certified_alias"})
    mapping = entity_mapping(row.source_api)
    value = row.text(mapping.key_field)
    if row.source_api in MARKET_APIS:
        value = "market"
    if value is None:
        raise ValueError("source_local_key_missing")
    if len(value) > 256 or not value.strip():
        raise ValueError("source_local_key_invalid")
    directory_aliases = tuple(
        alias
        for alias in config.aliases_by_key.get((mapping.namespace, value), ())
        if alias.source in OBSERVED_MASTERS
        or (alias.source == "ths_index" and row.availability.data_as_of is None)
    )
    identity = resolve_alias(
        directory_aliases,
        mapping.namespace,
        value,
        cutoff,
        effective_at=cutoff,
        market_timezone=row.availability.market_timezone,
    )
    if identity is not None:
        return replace(
            row,
            entity_id=identity,
            raw={
                **raw,
                "entity_basis": "observed_directory",
                "entity_namespace": mapping.namespace,
                "historical_pit_certified": False,
            },
        )
    # 代码可读时保留，复杂复合键仅用确定性摘要；API进入命名空间，禁止隐式跨源连接。
    key = value if re.fullmatch(r"[A-Za-z0-9_.-]+", value) else digest(value)
    identity = "source:series:" + row.source_api + ":" + key
    return replace(
        row,
        entity_id=identity,
        raw={
            **raw,
            "entity_basis": "source_local",
            "entity_namespace": mapping.namespace,
            "historical_pit_certified": False,
        },
    )


def retain_rows(
    rows: list[ResearchRow],
    candidate: ResearchRow,
) -> list[ResearchRow]:
    """保留两个完整业务期间及期间内全部成员；无业务期的快照与事件保留全部有界证据。

    例如新交易日到来后释放第三旧日期，两个当前日期的成员均保留；不挑任意前N行。
    """
    updated = [*rows, candidate]
    if candidate.source_api in VERSION_APIS:
        return updated
    periods = sorted({row.availability.data_as_of or "" for row in updated}, reverse=True)[:2]
    return [row for row in updated if (row.availability.data_as_of or "") in periods]


def prepare_row_numeric(
    row: ResearchRow,
    config: ResearchConfig,
    contract: ReferenceContract,
    certificates: tuple[UnitCertification, ...],
    currency_fields: frozenset[str],
) -> ResearchRow:
    """按已签名审计的字段授权补充本行币种证书，不覆盖已有整列证书。

    例如同一接口的USD和HKD金额各自保留本币证书；本行无币种、单位未知或声明冲突
    均不补证。证书只附在ResearchRow.numeric，本币金额不能借此跨币相加。
    """
    names = currency_fields - {certificate.field for certificate in certificates}
    if names and (row.raw.get("currency") is not None or row.raw.get("curr_type") is not None):
        units = UnitBuild(
            allowed_currencies=ALLOWED_CURRENCIES,
            inherit_stock_currency=False,
            preserve_contract_lots=True,
        )
        units.add(row.raw, row.availability, contract, set(names), EntityBuild(1))
        local = tuple(
            certificate
            for certificate in units.accepted()
            if certificate.dimension in {"money", "price"}
        )
        certificates = (*certificates, *local)
    return prepare_numeric(row, config, certificates=certificates)


def load_inputs(
    reference_path: Path,
    cutoff: str,
    run_id: str,
    config: ResearchConfig,
    *,
    progress: ProgressCallback | None = None,
    remaining_rows: int | None = None,
    source_apis: frozenset[str] | None = None,
) -> tuple[ResearchInputs, dict[str, int]]:
    """只读取新增107来源，以有限批扫描并记录排除、保留和历史释放数量。

    例如source_apis={'fx_daily'}时只扫描该接口；remaining_rows=0时仍扫描并拒绝整源。
    内存字节与行数分别设门禁，任一实体或来源溢出均清空整组，不输出截断后的总量。
    """
    check_paths(reference_path)
    if remaining_rows is not None and (type(remaining_rows) is not int or remaining_rows < 0):
        raise ValueError("input_capacity_invalid")
    total_budget = (
        config.maximum_total_rows
        if remaining_rows is None
        else min(
            remaining_rows,
            config.maximum_total_rows,
        )
    )
    effective = visible_config(config, cutoff)
    contracts = load_reference_contracts(load_policies())
    currency_fields = config.binding_audit.get("row_currency_fields", {})
    if currency_fields:
        verify_binding_audit(config, file_checksum(reference_path), cutoff)
    selected_apis = SOURCE_APIS if source_apis is None else source_apis
    if not isinstance(selected_apis, frozenset) or not selected_apis <= SOURCE_APIS:
        raise ValueError("input_source_selection_invalid")
    entries = {
        entry.api: entry for entry in cross_registry() if entry.api in selected_apis
    }
    mappings = {item.source_api: item for item in effective.foundation.time_mappings}
    entity_apis = {item.source_api for item in effective.foundation.entity_mappings}
    rows_by_api: dict[str, tuple[ResearchRow, ...]] = {}
    source_reasons: dict[str, tuple[str, ...]] = {}
    local_reasons: dict[str, tuple[str, ...]] = {}
    counts = {"scanned_rows": 0, "retained_rows": 0, "excluded_rows": 0, "batches": 0}
    total_bytes = 0
    limits = BuildLimits(batch_size=config.thresholds.fact_batch_size)
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
        for api, entry in sorted(entries.items()):
            configured = api in mappings and api in entity_apis
            reasons, readable = inspect_research_source(
                connection,
                entry,
                contracts[api],
                manifests.get(api),
                tables,
                cutoff,
                effective,
                count_rows=configured,
            )
            if not configured:
                reasons.add("research_bindings_missing")
            if not readable or not configured:
                source_reasons[api] = tuple(sorted(reasons))
                continue
            if mappings[api].market_timezone is None:
                reasons.add("market_timezone_uncertified")
            columns = {
                row[0] for row in connection.execute(f'DESCRIBE accepted."{api}"').fetchall()
            }
            fields = projection_fields(columns, entry, mappings[api])
            documented_kinds = {
                item["name"]: item["type"] for item in contracts[api].documentation["output_fields"]
            }
            groups: dict[str, RetentionGroup] = {}
            blocked: set[str] = set()
            overflow = False
            retained_count = retained_bytes = 0
            # 将单位配置缩到本源，避免每条行遍历206来源全部证书。
            certificates = tuple(
                item for item in effective.foundation.certifications if item.source_api == api
            )
            for batch in batches(
                projected_rows(connection, api, fields, limits), limits.batch_size
            ):
                counts["batches"] += 1
                counts["scanned_rows"] += len(batch)
                for raw in batch:
                    try:
                        row = adapt_row(raw, contracts[api], effective, cutoff,
                                        documented_kinds=documented_kinds)
                    except (ValueError, TypeError, KeyError, OverflowError) as error:
                        counts["excluded_rows"] += 1
                        reasons.update({"rows_excluded_by_adapter", adapter_reason(error)})
                        continue
                    if row.raw["entity_basis"] == "source_local":
                        reasons.add("source_local_entity_only")
                    elif row.raw["entity_basis"] == "observed_directory":
                        reasons.add("directory_identity_observed_only")
                    identity = str(row.entity_id)
                    if overflow or identity in blocked:
                        continue
                    group = groups.get(identity)
                    if group is None:
                        group = RetentionGroup(keep_versions=api in VERSION_APIS)
                        groups[identity] = group
                    count_delta, byte_delta = group.append(
                        row, len(encode(row.raw).encode("utf-8"))
                    )
                    retained_count += count_delta
                    retained_bytes += byte_delta
                    if group.row_count > config.maximum_rows_per_entity_source:
                        blocked.add(identity)
                        local_reasons[api + "/" + identity] = ("entity_source_capacity_exceeded",)
                        retained_count -= group.row_count
                        retained_bytes -= group.byte_count
                        del groups[identity]
                    del group
                    row_overflow = retained_count > min(
                        config.maximum_source_rows, total_budget - counts["retained_rows"]
                    )
                    key_overflow = (
                        len(groups) + len(blocked) > config.maximum_source_rows
                        or len(local_reasons) > config.maximum_total_rows
                    )
                    byte_overflow = retained_bytes > min(
                        MAX_SOURCE_BYTES, MAX_RETAINED_BYTES - total_bytes
                    )
                    if row_overflow or key_overflow or byte_overflow:
                        overflow = True
                        groups.clear()
                        blocked.clear()
                        retained_count = retained_bytes = 0
                        reasons.add(
                            "source_byte_capacity_exceeded"
                            if byte_overflow
                            else "source_capacity_exceeded"
                        )
                if progress is not None:
                    progress({"stage": "scan_cross_asset_reference", "source_api": api, **counts})
            retained_rows = tuple(
                prepare_row_numeric(
                    row,
                    effective,
                    contracts[api],
                    certificates,
                    frozenset(currency_fields.get(api, ())),
                )
                for identity in sorted(groups)
                for row in sorted(
                    groups[identity].iter_rows(),
                    key=lambda item: str(item.raw["source_record_id"]),
                )
            )
            counts["retained_rows"] += len(retained_rows)
            total_bytes += retained_bytes
            rows_by_api[api] = retained_rows
            source_reasons[api] = tuple(sorted(reasons))
    counts["discarded_rows"] = (
        counts["scanned_rows"] - counts["retained_rows"] - counts["excluded_rows"]
    )
    return ResearchInputs(
        analysis_as_of=cutoff,
        run_id=run_id,
        config=effective,
        rows_by_api=rows_by_api,
        source_reasons=source_reasons,
        entity_source_reasons=local_reasons,
    ), counts
