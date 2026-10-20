"""审计盈利预期缓存的历史可用性，不把当前查询结果认证为历史版本。"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tushare_entitlement_hub.raw import canonical_json

SHANGHAI = ZoneInfo("Asia/Shanghai")
APIS = ("forecast", "forecast_vip", "report_rc")


def available_at(
    api_name: str, row: dict[str, Any], observed_at: str
) -> datetime | None:
    """返回有实际观测支持的最早可用时间，缺日期或无时区观测则返回None。

    输入原始字段和带时区本地观测时间。预告公告日20–21点批次保守延至次日零点，
    研报延至次日09:15；再取首次公告、TS更新时间及实际观测的最晚值。
    TS create_time只有更新时间含义，绝不能据此把当前记录倒填到历史。
    例：20260105公告、20260106T02:00:00Z观测的预告返回北京时间1月6日10点；
    同日首次公告不能覆盖较晚修订公告。未知API抛ValueError。
    """
    if api_name not in APIS:
        raise ValueError("unsupported_earnings_api")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            return None
        public_date = date.fromisoformat(
            str(row["report_date" if api_name == "report_rc" else "ann_date"])
        )
        if api_name != "report_rc" and row.get("first_ann_date"):
            public_date = max(public_date, date.fromisoformat(str(row["first_ann_date"])))
        batch_time = time(9, 15) if api_name == "report_rc" else time(0)
        earliest = datetime.combine(public_date + timedelta(days=1), batch_time, SHANGHAI)
        if row.get("create_time"):
            updated = datetime.fromisoformat(str(row["create_time"]))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=SHANGHAI)
            earliest = max(earliest, updated)
        return max(earliest, observed.astimezone(SHANGHAI))
    except (ValueError, KeyError, TypeError):
        return None


def read_snapshot(path: Path) -> dict[str, Any]:
    """读取已有JSON/gzip原始观测并核对响应哈希，不修改来源文件。

    例：read_snapshot(Path('raw.json.gz'))返回含api_name、response和observed_at的字典。
    凭据字段、错误响应、错误哈希或字段长度不匹配时拒绝，异常不回显载荷。
    """
    encoded = path.read_bytes()
    decoded = gzip.decompress(encoded) if path.suffix == ".gz" else encoded
    envelope: dict[str, Any] = json.loads(decoded)
    canonical_json(envelope)
    response = envelope["response"]
    expected_hash = hashlib.sha256(canonical_json(response).encode()).hexdigest()
    if response.get("code") != 0 or expected_hash != envelope.get("response_hash"):
        raise ValueError("invalid_earnings_response")
    data = response["data"]
    if len(set(data["fields"])) != len(data["fields"]) or any(
        len(row) != len(data["fields"]) for row in data["items"]
    ):
        raise ValueError("invalid_earnings_fields")
    return envelope


def audit_snapshots(paths: list[Path], mature_cutoff: datetime) -> dict[str, Any]:
    """汇总全部给定raw的年份、版本和成熟日可见性，重复快照不增加分母。

    mature_cutoff必须带时区。返回清单与各接口计数，始终不自动认证历史完整性。
    例：仅有2026年观测、截止2025年底时，mature_available_records为0，
    historical_version_certified为False；同一raw复制两次只算一次。
    同机构同预测期间才构成研报群组；群组多日期仅表明存在可比记录候选，
    不代表各旧日期的当时版本已被认证，也不直接计算利润修正或收益。
    """
    if mature_cutoff.tzinfo is None:
        raise ValueError("mature_cutoff_requires_timezone")
    seen_snapshots: set[str] = set()
    records: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    for path in sorted(paths):
        envelope = read_snapshot(path)
        api_name = envelope["api_name"]
        if api_name not in APIS:
            raise ValueError("unsupported_earnings_api")
        digest = hashlib.sha256(canonical_json(envelope).encode()).hexdigest()
        duplicate = digest in seen_snapshots
        data = envelope["response"]["data"]
        manifest.append({
            "path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "envelope_sha256": digest, "api_name": api_name,
            "observed_at": envelope["observed_at"], "duplicate_snapshot": duplicate,
            "row_count": len(data["items"]), "fields": data["fields"],
            "params": envelope["params"],
        })
        if duplicate:
            continue
        seen_snapshots.add(digest)
        records[api_name].extend(
            (dict(zip(data["fields"], values, strict=True)), envelope["observed_at"])
            for values in data["items"]
        )
    summaries: dict[str, Any] = {}
    for api_name in APIS:
        rows = records[api_name]
        date_field = "report_date" if api_name == "report_rc" else "ann_date"
        dates = sorted({str(row[date_field]) for row, _ in rows if row.get(date_field)})
        observed_dates = sorted({observed[:10] for _, observed in rows})
        availability = [available_at(api_name, row, observed) for row, observed in rows]
        cohorts: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for row, _ in rows:
            keys = ("ts_code", "org_name", "quarter") if api_name == "report_rc" else (
                "ts_code", "end_date"
            )
            if all(row.get(key) for key in (*keys, date_field)):
                cohorts[tuple(str(row[key]) for key in keys)].add(str(row[date_field]))
        summaries[api_name] = {
            "records_after_snapshot_deduplication": len(rows),
            "unique_securities": len({row["ts_code"] for row, _ in rows}),
            "public_date_range": [dates[0], dates[-1]] if dates else [],
            "public_year_counts": dict(sorted(Counter(day[:4] for row, _ in rows
                if (day := str(row.get(date_field, "")))).items())),
            "observation_dates": observed_dates,
            "first_available_at": min(value for value in availability if value is not None
                ).isoformat() if any(value is not None for value in availability) else None,
            "mature_available_records": sum(value is not None and value <= mature_cutoff
                for value in availability),
            "unknown_availability_records": availability.count(None),
            "create_time_nonnull": sum(bool(row.get("create_time")) for row, _ in rows),
            "first_ann_date_nonnull": sum(bool(row.get("first_ann_date")) for row, _ in rows),
            "first_ann_date_differs_from_ann_date": sum(bool(row.get("first_ann_date")) and
                row["first_ann_date"] != row.get("ann_date") for row, _ in rows),
            "update_flag_counts": dict(Counter(str(row.get("update_flag")) for row, _ in rows)),
            "cohorts": len(cohorts),
            "cohorts_with_multiple_public_dates": sum(len(days) > 1 for days in cohorts.values()),
            "historical_version_certified": False,
        }
    return {
        "schema_version": 1, "mature_cutoff": mature_cutoff.isoformat(),
        "files": manifest, "file_count": len(paths),
        "unique_snapshots": len(seen_snapshots), "apis": summaries,
        "decision": "archive_audit_only_reconstruction_requires_separate_review",
        "reason": "late_observation_does_not_disprove_provider_reconstructed_versions",
        "training_performed": False, "financial_returns": None,
    }


def write_audit(paths_file: Path, output_file: Path, mature_cutoff: datetime) -> dict[str, Any]:
    """从明确路径清单重放离线审计，返回并写出摘要，不读取任何凭据。

    例：write_audit(Path('paths.txt'),Path('audit.json'),带时区截止时间)生成可回查JSON；
    清单每行一个raw文件，空行忽略，输出父目录由调用方预先建立。
    """
    paths = [Path(line) for line in paths_file.read_text().splitlines() if line.strip()]
    result = audit_snapshots(paths, mature_cutoff)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
