"""组织独立于信息产品的本地数据参考发布，覆盖全部登记接口及可解释的不可用项。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from .access import PRIVATE_APIS, WRITE_APIS, bind_account
from .catalog import load_policies
from .client import Client, ProviderError
from .raw import read_raw_envelope
from .reference_audit import audit_reference
from .reference_collect import collect_reference_partitions
from .reference_contract import ReferenceContract, ReferenceRecord, load_reference_contracts
from .reference_quality import evaluate_records
from .reference_scope import build_scope_certificate, merge_scope_certificates
from .reference_storage import (
    atomic_json,
    file_checksum,
    merge_dataset_files,
    publish_library,
    verified_files,
    write_dataset_files,
)
from .store import Store, digest, encode

REFERENCE_VERSION = "reference-v4"


def _records_from_raw_pages(
    api_name: str,
    observed_at: str,
    pages: list[dict[str, Any]],
) -> list[ReferenceRecord]:
    """校验并恢复一个不可变批次的原始页，返回规范化前记录身份。

    输入 manifest 中的 api、观察时间和页清单，输出 ``ReferenceRecord`` 列表；任何文件
    校验和、请求身份、响应哈希或页行宽不一致都会抛异常，禁止部分重建。
    """
    records: list[ReferenceRecord] = []
    for page in pages:
        raw_path = Path(page["raw_path"])
        if file_checksum(raw_path) != page["file_sha256"]:
            raise ValueError("raw_checksum_mismatch")
        envelope = read_raw_envelope(raw_path)
        if (
            envelope["api_name"] != api_name
            or digest(envelope["response"]) != page["response_hash"]
            or envelope["params"] != page["parameters"]
            or envelope["observed_at"] != observed_at
            or envelope["request_id"]
            != digest({"api_name": api_name, "params": envelope["params"]})
        ):
            raise ValueError("raw_identity_mismatch")
        data = envelope["response"]["data"]
        records.extend(
            (
                dict(zip(data["fields"], values, strict=True)),
                page["response_hash"],
                index,
                envelope["request_id"],
            )
            for index, values in enumerate(data["items"])
            if index < page.get("processed_rows", len(data["items"]))
        )
    return records


def reference_collection_policy(
    contract: ReferenceContract,
    observed_at: str,
    failure_kind: str | None = None,
    default_rate_per_minute: float = 60,
) -> dict[str, Any]:
    """返回接口限速、分区和下次可重试时间，不启动后台调度或隐藏重放请求。

    输入参考契约、UTC观察时间和可选失败类型，输出可持久化政策。例如每小时一次的
    hk_daily发生transient_failure时给出一小时后的retry_eligible_at；普通daily只记录速率。
    """
    rules = contract.rules
    policy: dict[str, Any] = {
        "rate_per_minute": rules.get("rate_per_minute", default_rate_per_minute),
        "partition_strategy": rules.get("partition_strategy", "declared_request"),
    }
    cooldown = rules.get("cooldown_seconds")
    if failure_kind == "transient_failure" and isinstance(cooldown, int) and cooldown > 0:
        instant = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        policy["retry_eligible_at"] = (
            (instant + timedelta(seconds=cooldown)).isoformat().replace("+00:00", "Z")
        )
    return policy


def build_reference(
    store: Store,
    client: Client,
    destination: Path,
    clock: Callable[[], str],
    parameters: dict[str, dict[str, Any] | list[dict[str, Any]]] | None = None,
    api_names: list[str] | None = None,
    refresh: bool = False,
    source: str = "live",
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """采集正式批次、逐行验收、保存清单并发布通用数据参考，单接口失败不终止其他接口。

    输入目录、客户端、目录、时钟和完整参数映射；输出全目录状态及数据库路径。
    例如daily有合格行、news拒绝权限，前者可查询、后者仍在catalog中解释缺口。
    默认使用显式政策范围；参数映射完整替换，避免隐藏地合并窄范围样例。
    复用要求同账户、同请求、同契约且文件校验通过，不将原探测结果冒充正式采集。
    """
    if source not in {"fixture", "live"}:
        raise ValueError("invalid_source")
    if source == "live":
        bind_account(store, client.account)
    destination.mkdir(parents=True, exist_ok=True)
    binding = destination / "account.json"
    identity = {"account": client.account, "source": source}
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("reference_account_mismatch")
    atomic_json(binding, identity)
    contracts = load_reference_contracts(load_policies())
    manifests = []
    for entry in store.catalog():
        path = destination / "datasets" / entry.api_name / "manifest.json"
        if api_names is not None and entry.api_name not in api_names:
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("files") and not verified_files(previous):
                    previous.update(status="artifact_verification_failed", files={})
                manifests.append(previous)
            continue
        observed_at = clock()
        manifest: dict[str, Any] = {
            "api_name": entry.api_name,
            "title": entry.title,
            "account": client.account,
            "source": source,
            "observed_at": observed_at,
            "doc_url": entry.doc_url,
            "documented_permission_mode": entry.permission_mode,
            "lifecycle": entry.lifecycle,
        }
        if entry.api_name in WRITE_APIS | PRIVATE_APIS or entry.api_name not in contracts:
            manifest.update(status="excluded", reason="private_write_or_non_api")
        else:
            contract = contracts[entry.api_name]
            dictionary_path = path.parent / "source_dictionary.json"
            atomic_json(
                dictionary_path,
                {
                    "api_name": entry.api_name,
                    "documentation": contract.documentation,
                    "primary_key": contract.primary_key,
                    "rules": contract.rules,
                    "source_units": contract.policy.units,
                    "purpose": contract.policy.purpose,
                    "lifecycle": contract.policy.lifecycle,
                },
            )
            manifest["dictionary"] = str(dictionary_path)
            params = (parameters or {}).get(entry.api_name, dict(contract.policy.minimal_params))
            signature = digest([identity, params, contract.fingerprint, REFERENCE_VERSION])
            previous = json.loads(path.read_text()) if path.exists() else {}
            if (
                not refresh
                and previous.get("request_signature") == signature
                and previous.get("collection_status", "").startswith("complete")
                and verified_files(previous)
            ):
                manifests.append(previous)
                progress(encode({"api_name": entry.api_name, "status": "reused"}))
                continue
            manifest.update(
                parameters=params,
                request_signature=signature,
                contract_hash=contract.fingerprint,
                reference_version=REFERENCE_VERSION,
                scope="complete_declared_request_not_all_history_or_market_universe",
                collection_policy=reference_collection_policy(
                    contract,
                    observed_at,
                    default_rate_per_minute=(entry.frequency_per_minute or 60),
                ),
            )
            try:
                client.rate_per_minute = contract.rules.get(
                    "rate_per_minute", entry.frequency_per_minute or 60
                )
                pages, records, completeness, partitions = collect_reference_partitions(
                    entry, contract, client, params, observed_at, progress
                )
                manifest.update(pages=pages, collection_status=completeness, partitions=partitions)
                for page in pages:
                    page["file_sha256"] = file_checksum(Path(page["raw_path"]))
                accepted, quarantined, quality = evaluate_records(records, contract, observed_at)
                run_directory = path.parent / "runs" / digest([signature, observed_at])[:20]
                manifest.update(write_dataset_files(run_directory, accepted, quarantined, contract))
                manifest["quality"] = quality
                manifest["status"] = (
                    "accepted"
                    if quality["status"] == "passed" and completeness.startswith("complete")
                    else "partial"
                    if accepted
                    else "empty"
                    if not records
                    else "quarantined"
                )
                certificate = build_scope_certificate(
                    entry.api_name,
                    params,
                    partitions,
                    completeness,
                    observed_at,
                    quality,
                    accepted,
                )
                manifest["scope_certificates"] = [certificate]
                atomic_json(run_directory / "manifest.json", manifest)
            except (ProviderError, ValueError, OSError, duckdb.Error) as error:
                manifest.update(
                    status=error.kind if isinstance(error, ProviderError) else "processing_failed"
                )
                if isinstance(error, ProviderError):
                    manifest["collection_policy"] = reference_collection_policy(
                        contract, observed_at, error.kind, entry.frequency_per_minute or 60
                    )
            previous_valid = bool(previous.get("files")) and verified_files(previous)
            publishable = manifest.get("collection_status", "").startswith(
                "complete"
            ) and manifest.get("status") in {"accepted", "empty"}
            if publishable and previous_valid:
                manifest["files"] = merge_dataset_files(
                    previous.get("files", {}), manifest.get("files", {})
                )
                manifest["released_runs"] = [
                    *previous.get("released_runs", []),
                    {
                        "observed_at": observed_at,
                        "parameters": params,
                        "collection_status": manifest["collection_status"],
                        "partitions": manifest.get("partitions", []),
                        "pages": manifest.get("pages", []),
                        "quality": manifest.get("quality", {}),
                    },
                ]
                manifest["scope_certificates"] = merge_scope_certificates(
                    previous.get("scope_certificates", []),
                    manifest.get("scope_certificates", []),
                )
            elif publishable:
                manifest["released_runs"] = [
                    {
                        "observed_at": observed_at,
                        "parameters": params,
                        "collection_status": manifest["collection_status"],
                        "partitions": manifest.get("partitions", []),
                        "pages": manifest.get("pages", []),
                        "quality": manifest.get("quality", {}),
                    }
                ]
            elif previous_valid:
                attempt = {
                    "observed_at": observed_at,
                    "status": manifest.get("status", "processing_failed"),
                    "collection_status": manifest.get("collection_status"),
                    "parameters": params,
                }
                retained = dict(previous)
                retained.update(
                    latest_attempt=attempt,
                    release_status="retained_after_failed_attempt",
                )
                manifest = retained
        atomic_json(path, manifest)
        manifests.append(manifest)
        progress(
            encode(
                {
                    "api_name": entry.api_name,
                    "status": manifest["status"],
                    "rows": manifest.get("quality", {}).get("source_rows", 0),
                }
            )
        )
        atomic_json(destination / "catalog.json", {"datasets": manifests, **identity})
    database = publish_library(destination, manifests)
    for manifest in manifests:
        atomic_json(destination / "datasets" / manifest["api_name"] / "manifest.json", manifest)
    result = {
        **identity,
        "database": str(database),
        "datasets": manifests,
        "counts": dict(Counter(item["status"] for item in manifests)),
        "cross_dataset_quality": audit_reference(destination),
    }
    atomic_json(destination / "catalog.json", result)
    return result


def rebuild_reference(
    destination: Path,
    progress: Callable[[str], None] = print,
    api_names: list[str] | None = None,
) -> dict[str, Any]:
    """从校验通过的正式原始页离线重建标准层，不重新请求、不改变历史观察时间。

    输入发布根目录和进度回调，返回重建目录与计数。原始页损坏时保留错误状态，
    不复用旧合格产物；例如修复字段别名后运行即可重建Parquet而不再次使用账户额度。
    """
    identity = json.loads((destination / "account.json").read_text())
    contracts = load_reference_contracts(load_policies())
    manifests = []
    for path in sorted((destination / "datasets").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        name = manifest["api_name"]
        contract = contracts.get(name)
        if contract is not None:
            failure_kind = (
                "transient_failure" if manifest.get("status") == "transient_failure" else None
            )
            manifest["collection_policy"] = reference_collection_policy(
                contract, manifest["observed_at"], failure_kind
            )
        if api_names is not None and name not in api_names:
            if manifest.get("files") and not verified_files(manifest):
                manifest.update(status="artifact_verification_failed", files={})
            atomic_json(path, manifest)
            manifests.append(manifest)
            continue
        if manifest.get("pages") and contract is not None:
            try:
                released = manifest.get("released_runs") or [
                    {
                        "observed_at": manifest["observed_at"],
                        "parameters": manifest["parameters"],
                        "collection_status": manifest["collection_status"],
                        "partitions": manifest.get("partitions", []),
                        "pages": manifest["pages"],
                    }
                ]
                merged_files: dict[str, Any] = {}
                rebuilt_runs = []
                certificates: list[dict[str, Any]] = []
                latest_quality: dict[str, Any] = {}
                rebuilt_dictionary: dict[str, Any] = {}
                for released_run in released:
                    run_observed_at = released_run["observed_at"]
                    records = _records_from_raw_pages(name, run_observed_at, released_run["pages"])
                    accepted, quarantined, quality = evaluate_records(
                        records, contract, run_observed_at
                    )
                    signature = digest(
                        [
                            identity,
                            released_run["parameters"],
                            contract.fingerprint,
                            REFERENCE_VERSION,
                        ]
                    )
                    directory = (
                        path.parent / "runs" / digest([signature, run_observed_at, "rebuild"])[:20]
                    )
                    rebuilt = write_dataset_files(directory, accepted, quarantined, contract)
                    rebuilt_dictionary = rebuilt
                    merged_files = merge_dataset_files(merged_files, rebuilt["files"])
                    certificate = build_scope_certificate(
                        name,
                        released_run["parameters"],
                        released_run.get("partitions", []),
                        released_run["collection_status"],
                        run_observed_at,
                        quality,
                        accepted,
                    )
                    certificates.append(certificate)
                    rebuilt_runs.append({**released_run, "quality": quality})
                    latest_quality = quality
                manifest.update(
                    files=merged_files,
                    dictionary=rebuilt_dictionary["dictionary"],
                    dictionary_sha256=rebuilt_dictionary["dictionary_sha256"],
                    quality=latest_quality,
                    released_runs=rebuilt_runs,
                    scope_certificates=merge_scope_certificates([], certificates),
                    request_signature=digest(
                        [identity, manifest["parameters"], contract.fingerprint, REFERENCE_VERSION]
                    ),
                    contract_hash=contract.fingerprint,
                    reference_version=REFERENCE_VERSION,
                )
                complete = all(
                    run["collection_status"].startswith("complete") for run in rebuilt_runs
                )
                manifest["status"] = (
                    "accepted"
                    if latest_quality["status"] == "passed" and complete
                    else "partial"
                    if merged_files.get("accepted", {}).get("rows", 0)
                    else "quarantined"
                    if merged_files.get("quarantine", {}).get("rows", 0)
                    else "empty"
                )
            except (OSError, ValueError, KeyError, duckdb.Error):
                manifest.update(status="rebuild_failed", files={})
        atomic_json(path, manifest)
        manifests.append(manifest)
        progress(encode({"api_name": name, "status": manifest["status"]}))
    database = publish_library(destination, manifests)
    for manifest in manifests:
        atomic_json(destination / "datasets" / manifest["api_name"] / "manifest.json", manifest)
    result = {
        **identity,
        "database": str(database),
        "datasets": manifests,
        "counts": dict(Counter(item["status"] for item in manifests)),
        "cross_dataset_quality": audit_reference(destination),
    }
    atomic_json(destination / "catalog.json", result)
    return result


def _manifest_dates(
    database: Path,
    api_name: str,
) -> list[dict[str, Any]]:
    """从已发布 accepted 视图读取实际业务日期的去重小集合。

    输入 source reference.duckdb 和接口名，输出 ``[{trade_date: ...}, ...]`` 形式的小行；
    缺表返回空列表。只投影名称为 date 或以 ``_date`` 结尾的列，不读取其他业务载荷。
    """
    with duckdb.connect(
        str(database), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        columns = [
            row[0]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='accepted' AND table_name=? ORDER BY column_name",
                [api_name],
            ).fetchall()
            if row[0] == "date" or row[0].endswith("_date")
        ]
        rows: list[dict[str, Any]] = []
        for column in columns:
            rows.extend(
                {column: value}
                for (value,) in connection.execute(
                    f'SELECT DISTINCT "{column}"::VARCHAR FROM accepted."{api_name}" '
                    f'WHERE "{column}" IS NOT NULL ORDER BY 1'
                ).fetchall()
            )
        return rows


def compose_reference(
    destination: Path,
    sources: list[Path],
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """零复制合并多个正式参考库，按业务键暴露最新版本并生成统一范围证书。

    输入若干 reference 根目录，读取其 manifest 和 Parquet，不复制原始页或标准明细；
    输出新的轻量 DuckDB、逐接口活动 manifest 和跨表审计。账户、来源模式或契约冲突时
    失败关闭。旧 v3 完整批次会用实际 accepted 日期补发范围证书。
    """
    if not sources:
        raise ValueError("reference_sources_required")
    resolved_destination = destination.resolve()
    if any(source.resolve() == resolved_destination for source in sources):
        raise ValueError("reference_compose_output_collision")
    identities = []
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for source in sources:
        identity = json.loads((source / "account.json").read_text(encoding="utf-8"))
        identities.append(identity)
        database = source / "reference.duckdb"
        with duckdb.connect(
            str(database), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            for api_name, payload in connection.execute(
                "SELECT api_name,manifest FROM catalog ORDER BY api_name"
            ).fetchall():
                manifest = json.loads(payload)
                if manifest.get("files") and not verified_files(manifest):
                    raise ValueError("reference_source_artifact_verification_failed")
                grouped.setdefault(api_name, []).append((manifest, source))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("reference_account_mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "account.json", identities[0])
    manifests = []
    for api_name, candidates in sorted(grouped.items()):
        hashes = {
            manifest.get("contract_hash")
            for manifest, _ in candidates
            if manifest.get("contract_hash")
        }
        if len(hashes) > 1:
            raise ValueError("reference_contract_mismatch")
        candidates.sort(key=lambda item: str(item[0].get("observed_at", "")))
        manifest = dict(candidates[-1][0])
        merged_files: dict[str, Any] = {}
        certificates: list[dict[str, Any]] = []
        released_runs: dict[str, dict[str, Any]] = {}
        for source_manifest, source in candidates:
            merged_files = merge_dataset_files(merged_files, source_manifest.get("files", {}))
            source_certificates = list(source_manifest.get("scope_certificates", []))
            if (
                not source_certificates
                and source_manifest.get("collection_status", "").startswith("complete")
                and source_manifest.get("parameters") is not None
            ):
                source_certificates = [
                    build_scope_certificate(
                        api_name,
                        source_manifest["parameters"],
                        source_manifest.get("partitions", []),
                        source_manifest["collection_status"],
                        source_manifest["observed_at"],
                        source_manifest.get("quality", {}),
                        _manifest_dates(source / "reference.duckdb", api_name),
                    )
                ]
            certificates = merge_scope_certificates(certificates, source_certificates)
            runs = source_manifest.get("released_runs") or [
                {
                    "observed_at": source_manifest.get("observed_at"),
                    "parameters": source_manifest.get("parameters"),
                    "collection_status": source_manifest.get("collection_status"),
                    "partitions": source_manifest.get("partitions", []),
                    "pages": source_manifest.get("pages", []),
                    "quality": source_manifest.get("quality", {}),
                }
            ]
            for run in runs:
                released_runs[digest(run)] = run
        manifest.update(
            files=merged_files,
            scope_certificates=certificates,
            released_runs=[released_runs[key] for key in sorted(released_runs)],
            release_status="composed",
            composed_sources=[str(source) for _, source in candidates],
        )
        manifests.append(manifest)
        progress(encode({"api_name": api_name, "status": manifest.get("status", "unknown")}))
    database = publish_library(destination, manifests)
    for manifest in manifests:
        atomic_json(destination / "datasets" / manifest["api_name"] / "manifest.json", manifest)
    result = {
        **identities[0],
        "database": str(database),
        "datasets": manifests,
        "counts": dict(Counter(item.get("status", "unknown") for item in manifests)),
        "sources": [str(source) for source in sources],
        "cross_dataset_quality": audit_reference(destination),
    }
    atomic_json(destination / "catalog.json", result)
    return result
