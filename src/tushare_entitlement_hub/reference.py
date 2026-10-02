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
from .reference_audit import audit_reference
from .reference_collect import collect_reference_partitions
from .reference_contract import ReferenceContract, ReferenceRecord, load_reference_contracts
from .reference_quality import evaluate_records
from .reference_storage import (
    atomic_json,
    file_checksum,
    publish_library,
    verified_files,
    write_dataset_files,
)
from .store import Store, digest, encode

REFERENCE_VERSION = "reference-v3"


def reference_collection_policy(
    contract: ReferenceContract, observed_at: str, failure_kind: str | None = None,
    default_rate_per_minute: float = 60,
) -> dict[str, Any]:
    """返回接口限速、分区和下次可重试时间，不启动后台调度或隐藏重放请求。

    输入参考契约、UTC观察时间和可选失败类型，输出可持久化政策。例如每小时一次的
    hk_daily发生transient_failure时给出一小时后的retry_eligible_at；普通daily只记录速率。
    """
    rules = contract.rules
    policy: dict[str, Any] = {
        "rate_per_minute": rules.get(
            "rate_per_minute", default_rate_per_minute
        ),
        "partition_strategy": rules.get("partition_strategy", "declared_request"),
    }
    cooldown = rules.get("cooldown_seconds")
    if failure_kind == "transient_failure" and isinstance(cooldown, int) and cooldown > 0:
        instant = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        policy["retry_eligible_at"] = (instant + timedelta(seconds=cooldown)).isoformat().replace(
            "+00:00", "Z"
        )
    return policy


def build_reference(
    store: Store, client: Client, destination: Path, clock: Callable[[], str],
    parameters: dict[str, dict[str, Any] | list[dict[str, Any]]] | None = None,
    api_names: list[str] | None = None,
    refresh: bool = False, source: str = "live", progress: Callable[[str], None] = print,
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
            "api_name": entry.api_name, "title": entry.title, "account": client.account,
            "source": source, "observed_at": observed_at, "doc_url": entry.doc_url,
            "documented_permission_mode": entry.permission_mode, "lifecycle": entry.lifecycle,
        }
        if entry.api_name in WRITE_APIS | PRIVATE_APIS or entry.api_name not in contracts:
            manifest.update(status="excluded", reason="private_write_or_non_api")
        else:
            contract = contracts[entry.api_name]
            dictionary_path = path.parent / "source_dictionary.json"
            atomic_json(dictionary_path, {
                "api_name": entry.api_name, "documentation": contract.documentation,
                "primary_key": contract.primary_key, "rules": contract.rules,
                "source_units": contract.policy.units,
                "purpose": contract.policy.purpose, "lifecycle": contract.policy.lifecycle,
            })
            manifest["dictionary"] = str(dictionary_path)
            params = (parameters or {}).get(entry.api_name, dict(contract.policy.minimal_params))
            signature = digest([identity, params, contract.fingerprint, REFERENCE_VERSION])
            previous = json.loads(path.read_text()) if path.exists() else {}
            if (not refresh and previous.get("request_signature") == signature
                    and previous.get("collection_status", "").startswith("complete")
                    and verified_files(previous)):
                manifests.append(previous)
                progress(encode({"api_name": entry.api_name, "status": "reused"}))
                continue
            manifest.update(
                parameters=params, request_signature=signature,
                contract_hash=contract.fingerprint,
                reference_version=REFERENCE_VERSION,
                scope="complete_declared_request_not_all_history_or_market_universe",
                collection_policy=reference_collection_policy(
                    contract, observed_at, default_rate_per_minute=(
                        entry.frequency_per_minute or 60
                    )
                ),
            )
            try:
                client.rate_per_minute = (
                    contract.rules.get("rate_per_minute", entry.frequency_per_minute or 60)
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
                    "accepted" if quality["status"] == "passed"
                    and completeness.startswith("complete")
                    else "partial" if accepted else "empty" if not records else "quarantined"
                )
            except (ProviderError, ValueError, OSError, duckdb.Error) as error:
                manifest.update(status=error.kind if isinstance(error, ProviderError)
                                else "processing_failed")
                if isinstance(error, ProviderError):
                    manifest["collection_policy"] = reference_collection_policy(
                        contract, observed_at, error.kind, entry.frequency_per_minute or 60
                    )
        atomic_json(path, manifest)
        manifests.append(manifest)
        progress(encode({"api_name": entry.api_name, "status": manifest["status"],
                         "rows": manifest.get("quality", {}).get("source_rows", 0)}))
        atomic_json(destination / "catalog.json", {"datasets": manifests, **identity})
    database = publish_library(destination, manifests)
    result = {**identity, "database": str(database), "datasets": manifests,
              "counts": dict(Counter(item["status"] for item in manifests)),
              "cross_dataset_quality": audit_reference(destination)}
    atomic_json(destination / "catalog.json", result)
    return result


def rebuild_reference(
    destination: Path, progress: Callable[[str], None] = print,
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
                records: list[ReferenceRecord] = []
                for page in manifest["pages"]:
                    raw_path = Path(page["raw_path"])
                    if file_checksum(raw_path) != page["file_sha256"]:
                        raise ValueError("raw_checksum_mismatch")
                    envelope = json.loads(raw_path.read_text())
                    if (envelope["api_name"] != name
                            or digest(envelope["response"]) != page["response_hash"]
                            or envelope["params"] != page["parameters"]
                            or envelope["observed_at"] != manifest["observed_at"]
                            or envelope["request_id"] != digest(
                                {"api_name": name, "params": envelope["params"]}
                            )):
                        raise ValueError("raw_identity_mismatch")
                    data = envelope["response"]["data"]
                    records.extend((dict(zip(data["fields"], values, strict=True)),
                                    page["response_hash"], index, envelope["request_id"])
                                   for index, values in enumerate(data["items"])
                                   if index < page.get("processed_rows", len(data["items"])))
                accepted, quarantined, quality = evaluate_records(
                    records, contract, manifest["observed_at"]
                )
                signature = digest([identity, manifest["parameters"], contract.fingerprint,
                                    REFERENCE_VERSION])
                directory = path.parent / "runs" / digest(
                    [signature, manifest["observed_at"], "rebuild"]
                )[:20]
                manifest.update(write_dataset_files(directory, accepted, quarantined, contract))
                manifest.update(quality=quality, request_signature=signature,
                                contract_hash=contract.fingerprint,
                                reference_version=REFERENCE_VERSION)
                complete = manifest["collection_status"].startswith("complete")
                manifest["status"] = (
                    "accepted" if quality["status"] == "passed" and complete else "partial"
                    if accepted else "quarantined" if records else "empty"
                )
            except (OSError, ValueError, KeyError, duckdb.Error):
                manifest.update(status="rebuild_failed", files={})
        atomic_json(path, manifest)
        manifests.append(manifest)
        progress(encode({"api_name": name, "status": manifest["status"]}))
    database = publish_library(destination, manifests)
    result = {**identity, "database": str(database), "datasets": manifests,
              "counts": dict(Counter(item["status"] for item in manifests)),
              "cross_dataset_quality": audit_reference(destination)}
    atomic_json(destination / "catalog.json", result)
    return result
