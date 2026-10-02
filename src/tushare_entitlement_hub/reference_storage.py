"""将逐接口标准数据发布为可校验的 JSONL、Parquet、字段字典和 DuckDB。"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb

from .reference_contract import ReferenceContract
from .store import encode


def file_checksum(path: Path) -> str:
    """分块计算数据文件校验和，不处理或访问凭据；空文件返回SHA256空内容摘要。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """临时文件完整写入后原子替换清单；例如写manifest失败不会截断旧manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encode(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_columns(
    contract: ReferenceContract, rows: list[dict[str, Any]],
) -> dict[str, str]:
    """生成稳定列类型，避免首批全空数值列被自动推断成整数或不同批次类型漂移。

    输入契约和记录，返回字段到DuckDB类型映射；例如声明float的close始终为DOUBLE。
    原始字段、单位和来源列表保存JSON，分钟时间保存TIMESTAMPTZ，区间文本保持VARCHAR。
    """
    documented = {field["name"]: field["type"].lower()
                  for field in contract.documentation.get("output_fields", [])}
    keys = set(documented) | {key for row in rows for key in row}
    columns = {}
    for field in sorted(keys):
        kind = documented.get(field, "")
        if field in {"source_fields", "source_units", "standard_units", "source_occurrences",
                     "issues"}:
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
        elif kind in {"float", "double", "int"}:
            target = "BIGINT" if kind == "int" and field not in contract.policy.units else "DOUBLE"
        else:
            target = "VARCHAR"
        columns[field] = target
    return columns


def write_dataset_files(
    destination: Path, accepted: list[dict[str, Any]], quarantined: list[dict[str, Any]],
    contract: ReferenceContract,
) -> dict[str, Any]:
    """写合格及隔离数据和字段字典，返回文件路径、摘要与实际列类型。

    输入输出目录、两类记录及契约；输出文件索引。空数据保留空JSONL和空schema说明。
    示例：一个daily合格行生成accepted.jsonl和accepted.parquet，原始字段仍可追溯。
    """
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for label, rows in (("accepted", accepted), ("quarantine", quarantined)):
        path = destination / f"{label}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(encode(row) + "\n")
        files[label] = {"jsonl": str(path), "sha256": file_checksum(path), "rows": len(rows)}
        if rows:
            parquet = destination / f"{label}.parquet"
            with duckdb.connect() as connection:
                connection.execute("SET threads=1")
                columns = reference_columns(contract, rows) if label == "accepted" else None
                connection.execute(
                    "CREATE TABLE exported AS SELECT * FROM read_json(?, "
                    "format='newline_delimited', columns=?, "
                    "sample_size=-1, maximum_object_size=100000000)",
                    [str(path), columns],
                )
                connection.execute("COPY exported TO ? (FORMAT PARQUET)", [str(parquet)])
                schema_rows = connection.execute("DESCRIBE exported").fetchall()
            files[label].update(
                parquet=str(parquet), parquet_sha256=file_checksum(parquet),
                columns=[{"name": row[0], "type": row[1]} for row in schema_rows],
            )
    dictionary = {
        "api_name": contract.policy.api_name, "contract_hash": contract.fingerprint,
        "primary_key": contract.primary_key,
        "nullable_key_fields": contract.rules.get("nullable_keys", []),
        "source_schema": contract.documentation,
        "source_units": contract.policy.units,
        "normalization_rules": contract.rules,
        "nullability": "除非空业务键和明确数值规则外，不假定官方业务字段永不为空",
        "identity": "source_record_id定位原始页内行；business_key才是声明业务维度",
        "time_semantics": "观察时间不可回填；报告期不代表公告时间；历史PIT资格单独标记",
        "contract": asdict(contract),
        "files": files,
    }
    dictionary_path = destination / "dictionary.json"
    atomic_json(dictionary_path, dictionary)
    return {"files": files, "dictionary": str(dictionary_path),
            "dictionary_sha256": file_checksum(dictionary_path)}


def verified_files(manifest: dict[str, Any]) -> bool:
    """验证清单引用的发布文件和原始页未被修改；任一缺失或校验失败返回False。

    输入发布清单，输出布尔值；例如accepted.parquet被截断时不允许直接复用旧发布。
    """
    try:
        if (manifest.get("dictionary_sha256") and file_checksum(Path(manifest["dictionary"]))
                != manifest["dictionary_sha256"]):
            return False
        for item in manifest.get("files", {}).values():
            if file_checksum(Path(item["jsonl"])) != item["sha256"]:
                return False
            if "parquet" in item and file_checksum(Path(item["parquet"])) != item["parquet_sha256"]:
                return False
        for page in manifest.get("pages", []):
            if file_checksum(Path(page["raw_path"])) != page["file_sha256"]:
                return False
    except (OSError, KeyError):
        return False
    return bool(manifest.get("files"))


def publish_library(destination: Path, manifests: list[dict[str, Any]]) -> Path:
    """原子发布独立DuckDB数据库，按accepted和quarantine分开存储每个接口。

    输入发布根目录及逐接口清单，返回数据库路径。catalog保存全部接口及失败原因。
    示例：accepted.daily可查询合格行，quarantine.daily仅用于问题研究，不参与默认消费。
    """
    temporary = destination / "reference.building.duckdb"
    temporary.unlink(missing_ok=True)
    with duckdb.connect(str(temporary)) as connection:
        connection.execute("SET threads=1")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        connection.execute("CREATE TABLE catalog(api_name VARCHAR, status VARCHAR, manifest JSON)")
        for manifest in manifests:
            name = manifest["api_name"]
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError("invalid_api_identifier")
            connection.execute("INSERT INTO catalog VALUES (?, ?, ?)",
                               [name, manifest["status"], encode(manifest)])
            for schema, item in manifest.get("files", {}).items():
                if "parquet" in item:
                    if schema not in {"accepted", "quarantine"}:
                        raise ValueError("invalid_dataset_schema")
                    connection.execute(
                        f'CREATE TABLE {schema}."{name}" AS SELECT * FROM read_parquet(?)',
                        [item["parquet"]],
                    )
        connection.execute("CHECKPOINT")
    with (destination / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["api_name", "title", "status", "collection_status", "parameters",
                         "source_rows", "accepted_rows", "quarantined_rows", "observed_at",
                         "dictionary"])
        for manifest in manifests:
            quality = manifest.get("quality", {})
            writer.writerow([
                manifest["api_name"], manifest.get("title", ""), manifest["status"],
                manifest.get("collection_status", ""), encode(manifest.get("parameters", {})),
                quality.get("source_rows", 0), quality.get("accepted_rows", 0),
                quality.get("quarantined_rows", 0), manifest.get("observed_at", ""),
                manifest.get("dictionary", ""),
            ])
    path = destination / "reference.duckdb"
    temporary.replace(path)
    return path
