"""产品基础库离线初始化，参考库只读，质量状态与实现范围逐项披露。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from ..catalog import load_policies
from ..pit import time_boundary
from ..reference_contract import ReferenceContract, load_reference_contracts
from ..reference_storage import file_checksum
from ..standard import normalize_timestamp
from ..store import digest, encode
from .bindings import FoundationBindings
from .config import FoundationPolicy
from .contracts import QualityGateResult, Truth
from .reference_scan import consume_rows
from .registry import REGISTRY_VERSION, ConsumerSpec, load_registry, validate_registry
from .storage import ProductStore
from .storage_records import CoverageSlice, GateRecord


def _inspect_source(
    connection: duckdb.DuckDBPyConnection,
    entry: ConsumerSpec,
    contract: ReferenceContract,
    manifest: dict[str, Any] | None,
    tables: set[tuple[str, str]],
    analysis_as_of: str,
    policy: FoundationPolicy,
) -> tuple[set[str], bool]:
    """核查实际表、契约、行数账本及采集新鲜度，返回原因和是否允许逐行适配。

    输入无 accepted.daily 的参考库 → missing_table 与 False；不伪造零条覆盖。
    """
    reasons = set()
    if manifest is None or ("accepted", entry.api) not in tables:
        return {"missing_table" if manifest else "missing_interface"}, False
    columns = {row[0] for row in connection.execute(f'DESCRIBE accepted."{entry.api}"').fetchall()}
    metadata = {
        "source_api",
        "source_record_id",
        "business_group_key",
        "contract_hash",
        "observed_at",
        "quality_tier",
        "standard_units",
        "historical_pit_certified",
    }
    if not (set(entry.selected_fields + entry.business_key) | metadata) <= columns:
        return {"reference_fields_missing"}, False
    if manifest.get("contract_hash") != contract.fingerprint:
        return {"source_contract_mismatch"}, False
    quality = manifest.get("quality", {})
    counts = [
        quality.get(key)
        for key in ("source_rows", "accepted_rows", "quarantined_rows", "exact_duplicate_rows")
    ]
    if any(type(value) is not int or value < 0 for value in counts) or counts[0] != sum(counts[1:]):
        return {"source_accounting_invalid"}, False
    accepted = connection.execute(f'SELECT count(*) FROM accepted."{entry.api}"').fetchall()[0][0]
    quarantined = (
        connection.execute(f'SELECT count(*) FROM quarantine."{entry.api}"').fetchall()[0][0]
        if ("quarantine", entry.api) in tables
        else 0
    )
    if accepted != counts[1] or quarantined != counts[2]:
        return {"source_accounting_mismatch"}, False
    if quarantined:
        reasons.add("quarantined_rows_excluded")
    if not str(manifest.get("collection_status", "")).startswith("complete"):
        reasons.add("collection_incomplete")
    try:
        age = (
            time_boundary(analysis_as_of)
            - time_boundary(normalize_timestamp(manifest.get("observed_at")))
        ).total_seconds() / 86400
    except (ValueError, TypeError):
        return reasons | {"observation_time_unknown"}, False
    if age < 0:
        return reasons | {"not_yet_observed"}, False
    if age > policy.maximum_observation_age_days:
        return reasons | {"stale_observation"}, False
    return reasons, True


def initialize_library(
    reference_path: Path,
    product_path: Path,
    analysis_as_of: str,
    policy: FoundationPolicy,
    bindings: FoundationBindings | None = None,
) -> dict[str, Any]:
    """只读消费参考库，原子保存产品共享事实和 206 项覆盖状态，完全离线。

    输入仅含 daily 的合成参考库，输出 status=incomplete、registered_apis=206。
    相同参考指纹、截止、配置产生同批次同摘要；不生成任何 P01—P13 业务评分。
    新批次在同一事务内替换当前产品事实，保留历史运行摘要；失败时恢复旧快照。
    """
    if not reference_path.is_file():
        raise ValueError("reference_library_missing")
    if reference_path.resolve() == product_path.resolve() or (
        product_path.exists() and reference_path.samefile(product_path)
    ):
        raise ValueError("input_output_path_collision")
    cutoff = normalize_timestamp(analysis_as_of)
    entries = load_registry()
    contracts = load_reference_contracts(load_policies())
    validate_registry(entries, contracts)
    inputs = FoundationBindings() if bindings is None else bindings
    reference_hash = file_checksum(reference_path)
    run_id = digest([reference_hash, cutoff, policy.to_dict(), inputs.to_dict(), REGISTRY_VERSION])
    with duckdb.connect(
        str(reference_path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as source:
        tables = {
            (row[0], row[1])
            for row in source.execute(
                "SELECT table_schema,table_name FROM information_schema.tables"
            ).fetchall()
        }
        if ("main", "catalog") not in tables:
            raise ValueError("reference_catalog_missing")
        manifests = {}
        for name, payload in source.execute("SELECT api_name,manifest FROM catalog").fetchall():
            if name in manifests:
                raise ValueError("reference_catalog_duplicate")
            manifests[name] = json.loads(payload)
        with ProductStore(product_path) as store:
            previous = store.metadata("run_id")
            if previous is not None:
                store.verify_contents()
            if previous == run_id:
                return store.verify_snapshot({entry.api for entry in entries})
            for table in store.tables():
                store.connection.execute(f'DELETE FROM product."{table}"')
            for entity in inputs.entities:
                store.put("entity", entity)
            for alias in inputs.aliases:
                store.put("entity_alias", alias)
            reason_counts: Counter[str] = Counter()
            available_apis = 0
            for entry in entries:
                manifest = manifests.get(entry.api)
                reasons, readable = _inspect_source(
                    source, entry, contracts[entry.api], manifest, tables, cutoff, policy
                )
                data_as_of = None
                if readable:
                    row_codes, data_as_of = consume_rows(
                        source, store, entry, contracts[entry.api], inputs, cutoff, run_id
                    )
                    reasons.update(row_codes)
                    available_apis += 1
                expected = inputs.expected_data_as_of.get(entry.api)
                if expected is None:
                    reasons.add("business_freshness_not_checked")
                elif data_as_of is None or data_as_of < expected:
                    reasons.add("stale_business_data")
                reasons.add("historical_pit_uncertified")
                reasons.add("universe_coverage_not_checked")
                quality = manifest.get("quality", {}) if manifest else {}
                accounting_valid = not reasons & {
                    "source_accounting_invalid",
                    "source_accounting_mismatch",
                }
                known_counts = quality if accounting_valid else {}
                observed = None
                if manifest and "observation_time_unknown" not in reasons:
                    observed = manifest.get("observed_at")
                coverage = CoverageSlice(
                    source_api=entry.api,
                    scope_id="declared_request",
                    partition="snapshot",
                    run_id=run_id,
                    request_scope=encode(manifest.get("parameters", {}))
                    if manifest
                    else "not_observed",
                    source_rows=known_counts.get("source_rows"),
                    accepted_rows=known_counts.get("accepted_rows"),
                    quarantined_rows=known_counts.get("quarantined_rows"),
                    duplicate_rows=known_counts.get("exact_duplicate_rows"),
                    observed_at=observed,
                    data_as_of=data_as_of,
                    manifest_hash=digest(manifest) if manifest else None,
                    reason_codes=tuple(sorted(reasons)),
                )
                coverage_id = store.put("coverage_slice", coverage)
                store.put(
                    "quality_gate_result",
                    GateRecord(
                        result_id=coverage_id,
                        rule_version=policy.version,
                        gate=QualityGateResult(
                            gate_id="source_readiness",
                            result=Truth.UNKNOWN,
                            reason_codes=tuple(sorted(reasons)),
                        ),
                    ),
                )
                reason_counts.update(reasons)
            if file_checksum(reference_path) != reference_hash:
                raise ValueError("reference_changed_during_read")
            summary = {
                "schema_version": "product-foundation/v1",
                "run_id": run_id,
                "status": "incomplete",
                "registered_apis": len(entries),
                "readable_apis": available_apis,
                "analysis_as_of": cutoff,
                "historical_pit_certified": False,
                "business_engines_implemented": [],
                "reason_counts": dict(sorted(reason_counts.items())),
                "counts": store.counts(),
            }
            store.save_metadata("reference_sha256", reference_hash)
            store.save_metadata("run_id", run_id)
            store.save_metadata("policy", policy.to_dict())
            store.save_metadata("bindings", inputs.to_dict())
            store.save_metadata("summary", summary)
            store.save_metadata(
                "run:" + run_id,
                {
                    "reference_sha256": reference_hash,
                    "policy": policy.to_dict(),
                    "bindings": inputs.to_dict(),
                    "summary": summary,
                },
            )
            return summary
