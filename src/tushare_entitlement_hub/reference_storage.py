"""将逐接口标准数据发布为可校验的 Parquet、字段字典和轻量 DuckDB 目录。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb

from .reference_contract import ReferenceContract
from .reference_derived import publish_derived_datasets
from .standard import is_standard_numeric_field
from .store import encode


def file_checksum(path: Path) -> str:
    """分块计算数据文件校验和，不处理或访问凭据；空文件返回SHA256空内容摘要。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def reference_fingerprint(path: Path) -> str:
    """计算参考库及其外部 Parquet 的统一指纹，并拒绝清单外篡改。

    普通或旧式内嵌 DuckDB 只返回数据库 SHA256；若 ``catalog.manifest`` 声明外部
    Parquet，则逐片核对清单哈希，再对数据库哈希和有序分片身份生成统一摘要。
    示例：修改视图引用的一片 Parquet 会抛 ``reference_artifact_verification_failed``，
    防止下游只校验轻量 DuckDB 而漏掉真实明细变化。
    """
    database_hash = file_checksum(path)
    index_path = path.parent / "reference-artifacts.json"
    external: list[dict[str, Any]] = []
    index_hash = None
    if index_path.is_file():
        index_hash = file_checksum(index_path)
        decoded = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict) or not isinstance(decoded.get("artifacts"), list):
            raise ValueError("reference_artifact_index_invalid")
        external = [dict(item) for item in decoded["artifacts"] if isinstance(item, dict)]
    else:
        return database_hash
    identities = []
    for fragment in external:
        fragment_path = Path(fragment["parquet"])
        stat = fragment_path.stat()
        unchanged = (
            fragment.get("bytes") == stat.st_size
            and fragment.get("modified_ns") == stat.st_mtime_ns
        )
        actual = (
            str(fragment.get("parquet_sha256", "")) if unchanged else file_checksum(fragment_path)
        )
        if actual != fragment.get("parquet_sha256"):
            raise ValueError("reference_artifact_verification_failed")
        identities.append(
            {
                "api_name": str(fragment.get("api_name", "")),
                "dataset": str(fragment.get("dataset", "")),
                "path": str(fragment_path.resolve()),
                "sha256": actual,
            }
        )
    return (
        hashlib.sha256(
            encode(
                {
                    "database_sha256": database_hash,
                    "artifact_index_sha256": index_hash,
                    "external": sorted(identities, key=encode),
                }
            ).encode()
        ).hexdigest()
        if identities
        else database_hash
    )


def atomic_json(path: Path, value: Any) -> None:
    """临时文件完整写入后原子替换清单；例如写manifest失败不会截断旧manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encode(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_columns(
    contract: ReferenceContract,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """生成稳定列类型，避免首批全空数值列被自动推断成整数或不同批次类型漂移。

    输入契约和记录，返回字段到DuckDB类型映射；例如声明float的close始终为DOUBLE。
    原始字段、单位和来源列表保存JSON，分钟时间保存TIMESTAMPTZ，区间文本保持VARCHAR。
    """
    documented = {
        field["name"]: field["type"].lower()
        for field in contract.documentation.get("output_fields", [])
    }
    keys = set(documented) | {key for row in rows for key in row}
    columns = {}
    for field in sorted(keys):
        kind = documented.get(field, "")
        if field in {
            "source_fields",
            "source_units",
            "standard_units",
            "source_occurrences",
            "issues",
            "field_quality",
            "business_dimensions",
            "business_variant_dimensions",
        }:
            target = "JSON"
        elif field == "source_index":
            target = "BIGINT"
        elif field == "historical_pit_certified":
            target = "BOOLEAN"
        elif field in contract.rules.get("timestamp_fields", []) or field == "observed_at":
            target = "TIMESTAMPTZ"
        elif field in contract.rules.get("date_list_fields", []):
            target = "DATE[]"
        elif field in contract.rules.get("text_fields", []):
            target = "VARCHAR"
        elif field.endswith("_date") or field == "date":
            target = "DATE"
        elif is_standard_numeric_field(field, kind, contract.policy):
            target = "BIGINT" if kind == "int" and field not in contract.policy.units else "DOUBLE"
        else:
            target = "VARCHAR"
        columns[field] = target
    return columns


def write_dataset_files(
    destination: Path,
    accepted: list[dict[str, Any]],
    quarantined: list[dict[str, Any]],
    contract: ReferenceContract,
) -> dict[str, Any]:
    """写合格及隔离 Parquet 和字段字典，返回路径、摘要与实际列类型。

    JSONL 只作为同目录临时转换文件并在返回前删除，长期标准明细只保留 ZSTD
    Parquet；空数据在字典中保留行数和契约，不制造无类型空文件。
    示例：一个 daily 合格行生成 ``accepted.parquet``，原始字段仍可追溯。
    """
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for label, rows in (("accepted", accepted), ("quarantine", quarantined)):
        columns = reference_columns(contract, rows)
        files[label] = {
            "rows": len(rows),
            "storage": "parquet_zstd",
            "columns": [{"name": name, "type": kind} for name, kind in columns.items()],
        }
        if not rows:
            continue
        staging = destination / f".{label}.staging.jsonl"
        parquet = destination / f"{label}.parquet"
        try:
            with staging.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(encode(row) + "\n")
            with duckdb.connect() as connection:
                connection.execute("SET threads=1")
                connection.execute("SET memory_limit='512MB'")
                connection.execute(
                    "CREATE TABLE exported AS SELECT * FROM read_json(?, "
                    "format='newline_delimited', columns=?, "
                    "sample_size=-1, maximum_object_size=100000000)",
                    [str(staging), columns],
                )
                connection.execute(
                    "COPY exported TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)",
                    [str(parquet)],
                )
                schema_rows = connection.execute("DESCRIBE exported").fetchall()
        finally:
            staging.unlink(missing_ok=True)
        files[label].update(
            parquet=str(parquet),
            parquet_sha256=file_checksum(parquet),
            bytes=parquet.stat().st_size,
            modified_ns=parquet.stat().st_mtime_ns,
            columns=[{"name": row[0], "type": row[1]} for row in schema_rows],
        )
    dictionary = {
        "api_name": contract.policy.api_name,
        "contract_hash": contract.fingerprint,
        "primary_key": contract.primary_key,
        "nullable_key_fields": contract.rules.get("nullable_keys", []),
        "source_schema": contract.documentation,
        "source_units": contract.policy.units,
        "normalization_rules": contract.rules,
        "nullability": "除非空业务键和明确数值规则外，不假定官方业务字段永不为空",
        "identity": (
            "business_group_key定位业务事件组；business_variant_key区分同组来源变体；"
            "source_record_id定位原始页内行"
        ),
        "quality_tiers": {
            "verified": "声明业务键唯一且直接质量规则通过",
            "usable_with_annotation": "数据可用但需遵守无成交、空值或业务变体说明",
            "quarantine": "无法解释的字段或数值问题，默认消费不得使用",
        },
        "time_semantics": "观察时间不可回填；报告期不代表公告时间；历史PIT资格单独标记",
        "contract": asdict(contract),
        "files": files,
    }
    dictionary_path = destination / "dictionary.json"
    atomic_json(dictionary_path, dictionary)
    return {
        "files": files,
        "dictionary": str(dictionary_path),
        "dictionary_sha256": file_checksum(dictionary_path),
    }


def parquet_fragments(item: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """把新分片清单或旧单文件清单统一成不可变 Parquet 分片序列。

    输入 ``{"parquet": "a.parquet", "parquet_sha256": "h", "rows": 2}`` 返回一个
    分片；输入含 ``fragments`` 的新清单返回其副本。示例：空清单返回空元组。
    """
    fragments = item.get("fragments")
    if isinstance(fragments, list):
        return tuple(dict(fragment) for fragment in fragments if isinstance(fragment, dict))
    if "parquet" not in item:
        return ()
    return (
        {
            "parquet": item["parquet"],
            "parquet_sha256": item.get("parquet_sha256", ""),
            "rows": item.get("rows", 0),
            "bytes": item.get("bytes"),
            "modified_ns": item.get("modified_ns"),
        },
    )


def merge_dataset_files(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """合并新旧标准分片索引，不复制 Parquet，也不丢弃旧批次。

    输入两份 ``files`` 字典，按路径和校验和去重后返回 ``parquet_fragments`` 清单；
    当前批次的列定义优先。示例：旧一片、新一片得到行数为两片之和的活动数据集。
    """
    merged: dict[str, Any] = {}
    for label in sorted(set(previous) | set(current)):
        older, newer = previous.get(label, {}), current.get(label, {})
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for fragment in (*parquet_fragments(older), *parquet_fragments(newer)):
            fragment_path = Path(fragment["parquet"])
            stat = fragment_path.stat()
            fragment["bytes"] = stat.st_size
            fragment["modified_ns"] = stat.st_mtime_ns
            key = (str(fragment.get("parquet", "")), str(fragment.get("parquet_sha256", "")))
            if key[0]:
                indexed[key] = fragment
        fragments = [indexed[key] for key in sorted(indexed)]
        merged[label] = {
            "storage": "parquet_fragments",
            "rows": sum(int(fragment.get("rows", 0)) for fragment in fragments),
            "columns": newer.get("columns", older.get("columns", [])),
            "fragments": fragments,
        }
    return merged


def verified_files(manifest: dict[str, Any]) -> bool:
    """验证清单引用的发布文件和原始页未被修改；任一缺失或校验失败返回False。

    输入发布清单，输出布尔值；例如accepted.parquet被截断时不允许直接复用旧发布。
    """
    try:
        if (
            manifest.get("dictionary_sha256")
            and file_checksum(Path(manifest["dictionary"])) != manifest["dictionary_sha256"]
        ):
            return False
        for item in manifest.get("files", {}).values():
            if "jsonl" in item and file_checksum(Path(item["jsonl"])) != item["sha256"]:
                return False
            if "parquet" in item and file_checksum(Path(item["parquet"])) != item["parquet_sha256"]:
                return False
            for fragment in parquet_fragments(item):
                if file_checksum(Path(fragment["parquet"])) != fragment["parquet_sha256"]:
                    return False
        pages = list(manifest.get("pages", []))
        for released in manifest.get("released_runs", []):
            pages.extend(released.get("pages", []))
        seen_raw_paths: set[str] = set()
        for page in pages:
            if page["raw_path"] in seen_raw_paths:
                continue
            seen_raw_paths.add(page["raw_path"])
            if file_checksum(Path(page["raw_path"])) != page["file_sha256"]:
                return False
    except (OSError, KeyError):
        return False
    return bool(manifest.get("files"))


def publish_library(
    destination: Path, manifests: list[dict[str, Any]], *, materialize: bool = False
) -> Path:
    """原子发布轻量 DuckDB 目录，以视图读取 accepted 和 quarantine Parquet。

    DuckDB 默认不复制标准明细；多批次按业务键和观察时间暴露最新 accepted 版本，隔离层按
    来源记录去重。输入发布根目录及逐接口清单，返回数据库路径。
    示例：``accepted.daily`` 可直接查询外部 Parquet，数据库本体只保存目录和派生小表。
    ``materialize=True`` 仅供需要修改合成表的离线测试夹具，不用于生产发布。
    """
    temporary = destination / "reference.building.duckdb"
    temporary.unlink(missing_ok=True)
    with duckdb.connect(
        str(temporary), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        connection.execute("SET threads=1")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        connection.execute("CREATE TABLE catalog(api_name VARCHAR, status VARCHAR, manifest JSON)")
        connection.execute(
            "CREATE TABLE scope_certificate("
            "certificate_id VARCHAR PRIMARY KEY, api_name VARCHAR, status VARCHAR, "
            "observed_at TIMESTAMPTZ, payload JSON)"
        )
        for manifest in manifests:
            name = manifest["api_name"]
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError("invalid_api_identifier")
            connection.execute(
                "INSERT INTO catalog VALUES (?, ?, ?)", [name, manifest["status"], encode(manifest)]
            )
            for certificate in manifest.get("scope_certificates", []):
                connection.execute(
                    "INSERT OR REPLACE INTO scope_certificate VALUES (?,?,?,?,?)",
                    [
                        certificate["certificate_id"],
                        name,
                        certificate["status"],
                        certificate["observed_at"],
                        encode(certificate),
                    ],
                )
            for schema, item in manifest.get("files", {}).items():
                fragments = parquet_fragments(item)
                if not fragments:
                    continue
                if schema not in {"accepted", "quarantine"}:
                    raise ValueError("invalid_dataset_schema")
                paths = (
                    "["
                    + ",".join(
                        "'" + str(Path(fragment["parquet"]).resolve()).replace("'", "''") + "'"
                        for fragment in fragments
                    )
                    + "]"
                )
                identity = "business_key" if schema == "accepted" else "source_record_id"
                relation_kind = "TABLE" if materialize else "VIEW"
                connection.execute(
                    f'CREATE {relation_kind} {schema}."{name}" AS '
                    "SELECT * EXCLUDE (_release_row_number) FROM ("
                    f"SELECT *, row_number() OVER (PARTITION BY {identity} "
                    "ORDER BY observed_at DESC NULLS LAST, source_record_id DESC) "
                    f"AS _release_row_number FROM read_parquet({paths}, union_by_name=true)"
                    ") WHERE _release_row_number=1"
                )
            release_quality = {}
            for schema in ("accepted", "quarantine"):
                exists_row = connection.execute(
                    "SELECT count(*)=1 FROM information_schema.tables "
                    "WHERE table_schema=? AND table_name=?",
                    [schema, name],
                ).fetchone()
                exists = bool(exists_row and exists_row[0])
                count_row = (
                    connection.execute(f'SELECT count(*) FROM {schema}."{name}"').fetchone()
                    if exists
                    else None
                )
                release_quality[schema + "_rows"] = int(count_row[0]) if count_row else 0
            manifest["release_quality"] = release_quality
            manifest.setdefault("quality", {})["accepted_rows"] = release_quality["accepted_rows"]
            manifest.setdefault("quality", {})["quarantined_rows"] = release_quality[
                "quarantine_rows"
            ]
            connection.execute(
                "UPDATE catalog SET status=?,manifest=? WHERE api_name=?",
                [manifest["status"], encode(manifest), name],
            )
        publish_derived_datasets(connection)
        connection.execute("CHECKPOINT")
    with (destination / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "api_name",
                "title",
                "status",
                "collection_status",
                "parameters",
                "source_rows",
                "accepted_rows",
                "usable_with_annotation_rows",
                "business_variant_rows",
                "quarantined_rows",
                "observed_at",
                "dictionary",
            ]
        )
        for manifest in manifests:
            quality = manifest.get("quality", {})
            writer.writerow(
                [
                    manifest["api_name"],
                    manifest.get("title", ""),
                    manifest["status"],
                    manifest.get("collection_status", ""),
                    encode(manifest.get("parameters", {})),
                    quality.get("source_rows", 0),
                    quality.get("accepted_rows", 0),
                    quality.get("usable_with_annotation_rows", 0),
                    quality.get("business_variant_rows", 0),
                    quality.get("quarantined_rows", 0),
                    manifest.get("observed_at", ""),
                    manifest.get("dictionary", ""),
                ]
            )
    path = destination / "reference.duckdb"
    temporary.replace(path)
    artifacts: list[dict[str, Any]] = []
    for manifest in manifests:
        for label, item in manifest.get("files", {}).items():
            artifacts.extend(
                {
                    "api_name": manifest["api_name"],
                    "dataset": label,
                    **fragment,
                }
                for fragment in parquet_fragments(item)
            )
    atomic_json(destination / "reference-artifacts.json", {"artifacts": artifacts})
    return path
