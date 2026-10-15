"""批量执行只读探测，提供可恢复的运行清单和逐接口原始研究样本。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .access import PRIVATE_APIS, WRITE_APIS, bind_account, latest_probe, readable_policy
from .client import Client
from .ingest import probe
from .quality import audit_rows
from .raw import read_raw_envelope
from .standard import normalize_rows
from .store import Store, encode


def sample_product(record: dict[str, Any], store: Store, destination: Path) -> dict[str, Any]:
    """为非空成功探测生成原始研究数据集和质量诊断，不冒充完整采集或推进水位。

    输入含原始路径的探测记录、数据库和输出目录，返回样本文件及质量状态。
    示例：返回了数据但缺少必需字段时，原始样本可读、quality 状态仍为待处理。
    """
    entry = next(entry for entry in store.catalog() if entry.api_name == record["api_name"])
    envelope = read_raw_envelope(Path(record["evidence"]["raw_path"]))
    data = envelope["response"]["data"]
    rows = [dict(zip(data["fields"], values, strict=True)) for values in data["items"]]
    policy = readable_policy(entry)
    standardized: list[dict[str, Any]] = []
    try:
        standardized = normalize_rows(
            rows, policy, record["observed_at"], record["evidence"]["response_hash"]
        )
        findings = audit_rows(standardized, policy)
    except ValueError:
        findings = [{"code": "normalization_failed", "severity": "error"}]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{entry.api_name}.json"
    payload = {
        "api_name": entry.api_name,
        "source": record["source"],
        "scope": "probe_sample_only",
        "observed_at": record["observed_at"],
        "raw_path": record["evidence"]["raw_path"],
        "rows": rows,
        "standardized_rows": standardized,
        "quality": findings,
        "consumers": policy.consumers,
        "purpose": policy.purpose,
    }
    path.write_text(encode(payload) + "\n", encoding="utf-8")
    return {
        "sample_path": str(path), "quality": findings[:20], "sample_rows": len(rows),
        "quality_counts": dict(Counter(item["code"] for item in findings)),
    }


def survey(
    store: Store,
    client: Client,
    destination: Path,
    clock: Callable[[], str],
    source: str = "live",
    parameters: dict[str, dict[str, Any]] | None = None,
    refresh: bool = False,
    api_names: list[str] | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """串行探测所有公共只读接口，成功证据可恢复，失败不妨碍后续接口。

    输入数据库、客户端、输出目录、时钟、来源和按接口覆盖的参数，返回完整汇总。
    默认不批量访问私有组合，不调用账户写接口；可显式 api_names 选择私有只读接口。
    示例：daily成功、news无权限，仍生成两条结果；重跑复用存在原始文件的成功结果。
    """
    if source not in {"fixture", "live"}:
        raise ValueError("invalid_source")
    if source == "live":
        bind_account(store, client.account)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in store.catalog():
        if api_names is not None and entry.api_name not in api_names:
            continue
        reason = ""
        if entry.api_name in WRITE_APIS:
            reason = "account_write_not_probeable"
        elif entry.api_name in PRIVATE_APIS and api_names is None:
            reason = "private_account_read_requires_selection"
        elif entry.policy is None:
            reason = "non_api_or_missing_contract"
        if reason:
            result: dict[str, Any] = {
                "api_name": entry.api_name,
                "status": "excluded",
                "reason": reason,
            }
        else:
            previous = latest_probe(store, entry, client.account) if source == "live" else None
            previous_evidence = json.loads(previous["evidence"]) if previous else {}
            override = (parameters or {}).get(entry.api_name)
            reusable = (
                previous is not None
                and previous["status"] in {"available", "empty", "unauthorized", "unsupported"}
                and (previous["status"] in {"unauthorized", "unsupported"}
                     or Path(previous_evidence.get("raw_path", "")).is_file())
                and not refresh
                and override is None
            )
            if reusable and previous is not None:
                result = {**previous, "evidence": previous_evidence}
            else:
                result = probe(entry, client, store, clock(), source, override)
            if result["status"] == "available":
                result.update(sample_product(result, store, destination / "samples"))
        results.append(result)
        progress(
            encode(
                {"api_name": entry.api_name, "status": result["status"], "completed": len(results)}
            )
        )
        report = {
            "account": client.account,
            "source": source,
            "results": results,
            "counts": dict(Counter(item["status"] for item in results)),
            "scope": "bounded_probe_samples_not_full_history",
        }
        (destination / "report.json").write_text(encode(report) + "\n", encoding="utf-8")
    return {
        "account": client.account,
        "source": source,
        "results": results,
        "counts": dict(Counter(item["status"] for item in results)),
    }
