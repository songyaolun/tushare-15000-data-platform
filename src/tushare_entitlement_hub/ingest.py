"""政策驱动任务计划、最小权限探活和可恢复的 raw-first 采集。"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

import duckdb

from .access import bind_account, readable_policy, verified_policy
from .client import Client, ProviderError
from .models import ENABLED, CatalogEntry, Policy
from .quality import audit_rows
from .raw import assert_public
from .standard import normalize_rows, normalize_timestamp
from .store import Store, digest, task_identity


class IngestionError(ValueError):
    """安全的采集门禁错误；异常文本只包含有限错误码，不含请求参数。"""


def require_policy(
    entry: CatalogEntry, *, for_probe: bool = False, store: Store | None = None,
    source: str = "fixture", account: str = "local",
) -> Policy:
    """只读探测不受文档权限标签阻挡；真实采集要求同账户同政策的成功证据。

    输入目录、探测标记及可选数据库/来源/账户，返回可执行政策副本。
    fixture 保留严格目录校验；live 不改文档事实，仍检查数据政策完整性。
    示例：fund_nav 文档待复核但 live 探测成功时允许规划，尚无证据则明确拒绝。
    """
    from .coverage import policy_debt

    if for_probe:
        try:
            return readable_policy(entry)
        except ValueError as error:
            raise IngestionError(str(error)) from None
    if source == "live":
        if store is None:
            raise IngestionError("live_evidence_store_required")
        try:
            return verified_policy(store, entry, account)
        except ValueError as error:
            raise IngestionError(str(error)) from None
    if entry.lifecycle not in ENABLED or entry.permission_mode != "points":
        raise IngestionError("interface_not_enabled")
    if entry.minimum_points is None or entry.minimum_points > 15000 or entry.needs_review:
        raise IngestionError("entitlement_needs_review")
    if entry.policy is None or policy_debt(entry.policy):
        raise IngestionError("incomplete_policy")
    return entry.policy


def plan_tasks(
    entry: CatalogEntry,
    start: str,
    end: str,
    mode: str,
    observed_at: str,
    store: Store | None = None,
    source: str = "fixture",
    parameters: dict[str, Any] | None = None,
    account: str = "local",
) -> list[dict[str, Any]]:
    """为指定日期区间生成稳定任务，增量从水位减回看窗口开始。

    示例：daily、2024-01-02 至 2024-01-03 → 两个交易日期分片；同一执行语义
    即使在不同 planned_at 重新规划也保持 ID 不变，规划时间仅用于审计。
    """
    policy = require_policy(entry, store=store, source=source, account=account)
    observed_at = normalize_timestamp(observed_at)
    if mode not in {"initialize", "incremental", "lookback", "event", "backfill"}:
        raise IngestionError("invalid_mode")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise IngestionError("invalid_range")
    if store is not None and mode in {"incremental", "lookback"}:
        watermark = store.watermark(entry.api_name, policy.date_field, source)
        if watermark:
            first = max(first, date.fromisoformat(watermark) - timedelta(days=policy.lookback_days))
    if first > last:
        return []
    if policy.shard_by == "period" and first != last and not (parameters or {}).get("period"):
        result = []
        for year in range(first.year, last.year + 1):
            for month in (3, 6, 9, 12):
                period = date(year, month, monthrange(year, month)[1])
                if first <= period <= last:
                    result.extend(
                        plan_tasks(
                            entry,
                            period.isoformat(),
                            period.isoformat(),
                            mode,
                            observed_at,
                            store,
                            source,
                            parameters,
                            account,
                        )
                    )
        return result
    probe_only = {
        "ts_code",
        "limit",
        "offset",
        "trade_date",
        "ann_date",
        "f_ann_date",
        "end_date",
        "period",
        "start_date",
        "cal_date",
        "m",
        "q",
        "start_m",
        "end_m",
        "start_q",
        "end_q",
    }
    params = {key: value for key, value in policy.minimal_params.items() if key not in probe_only}
    params.update(parameters or {})
    if "ts_codes" in params:
        securities = params.pop("ts_codes")
        if not isinstance(securities, list) or not securities:
            raise IngestionError("security_universe_required")
        return [
            task
            for security in securities
            for task in plan_tasks(
                entry, start, end, mode, observed_at, store, source,
                {**params, "ts_code": security}, account,
            )
        ]
    assert_public(params)
    if policy.shard_by == "ts_code" and not str(params.get("ts_code", "")).strip():
        raise IngestionError("security_shard_required")
    trading_dates = params.pop("trade_dates", None)
    if trading_dates is not None:
        trading_dates = {date.fromisoformat(value).isoformat() for value in trading_dates}
    tasks: list[dict[str, Any]] = []
    cursor = first
    by_day = policy.shard_by in {"trade_date", "ann_date", "f_ann_date", "cal_date", "date"}
    while cursor <= last:
        upper = cursor if by_day else last
        if by_day and trading_dates is not None and cursor.isoformat() not in trading_dates:
            cursor += timedelta(days=1)
            continue
        shard = dict(params)
        if by_day:
            shard[policy.date_field] = cursor.strftime("%Y%m%d")
        elif policy.date_field == "month":
            shard.update(start_m=cursor.strftime("%Y%m"), end_m=upper.strftime("%Y%m"))
        elif policy.date_field == "quarter":
            shard.update(
                start_q=f"{cursor.year}Q{(cursor.month - 1) // 3 + 1}",
                end_q=f"{upper.year}Q{(upper.month - 1) // 3 + 1}",
            )
        elif policy.shard_by == "period":
            if "period" not in shard:
                shard["period"] = cursor.strftime("%Y%m%d")
        elif policy.shard_by not in {"snapshot", "event"}:
            shard.update(start_date=cursor.strftime("%Y%m%d"), end_date=upper.strftime("%Y%m%d"))
        task: dict[str, Any] = {
            "api_name": entry.api_name,
            "date_field": policy.date_field,
            "start": cursor.isoformat(),
            "end": upper.isoformat(),
            "mode": mode,
            "params": shard,
            "source": source,
            "planned_at": observed_at,
            "policy_hash": digest(asdict(policy)),
        }
        if source == "live":
            task["account"] = account
        task["task_id"] = task_identity(task)
        tasks.append(task)
        cursor = upper + timedelta(days=1)
    return tasks


def probe(
    entry: CatalogEntry, client: Client, store: Store, observed_at: str, source: str = "fixture",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行政策最小请求并保存分类证据；成功零行是 empty，网络失败为 transient_failure。

    参数为目录、Client、Store、观察时间与来源；返回分类、时间及无凭据证据字典。
    示例：成功空行fixture返回 status='empty'；网络重试耗尽返回 transient_failure，
    两者不会混记，且失败证据不保存服务端原始错误正文。
    """
    policy = require_policy(entry, for_probe=True)
    if source == "live":
        bind_account(store, client.account)
    observed_at = normalize_timestamp(observed_at)
    parameters = {**policy.minimal_params, **(parameters or {})}
    if policy.supports_pagination:
        parameters.update(limit=1, offset=0)
    evidence: dict[str, Any] = {
        "request_policy": digest(asdict(entry.policy)) if entry.policy else "",
        "account": client.account,
        "parameters": parameters,
    }
    try:
        page = client.fetch(entry.api_name, parameters, observed_at)
        status = "available" if page.rows else "empty"
        evidence.update(
            response_hash=page.raw.response_hash, rows=len(page.rows), fields=page.fields,
            raw_path=str(page.raw.path),
        )
    except ProviderError as error:
        status = error.kind
        evidence["error_code"] = status
    store.record_probe(entry.api_name, observed_at, source, status, evidence)
    return {
        "api_name": entry.api_name,
        "status": status,
        "observed_at": observed_at,
        "source": source,
        "evidence": evidence,
    }


def run_task(
    task: dict[str, Any],
    entry: CatalogEntry,
    client: Client,
    store: Store,
    observed_at: str,
    *,
    expected_dates: Any = None,
    suspensions: Any = None,
    adj_factors: Any = None,
    previous_fields: list[str] | None = None,
) -> dict[str, Any]:
    """执行全分页、标准化和质量门禁后提交检查点，中断可用同一 task_id 恢复。

    示例：第二页失败 → raw 第一页仍存在，任务 failed 且水位不变；重跑后幂等提交。
    首批须显式提供完整 previous_fields 基线，后续优先使用历史成功字段；任何必需
    not_checked 都阻断提交。成功行的 quality_reference 保留所用字段基线及来源。
    """
    account = str(task.get("account", "local"))
    if task.get("source") == "live" and client.account != account:
        raise IngestionError("live_account_mismatch")
    policy = require_policy(entry, store=store, source=str(task["source"]), account=account)
    observed_at = normalize_timestamp(observed_at)
    assert_public(task)
    if task["api_name"] != entry.api_name or task["policy_hash"] != digest(asdict(policy)):
        raise IngestionError("plan_policy_changed")
    source = str(task["source"])
    if source not in {"fixture", "live"}:
        raise IngestionError("invalid_source")
    if not store.begin_task(task, observed_at, source):
        return {"task_id": task["task_id"], "state": "already_committed"}
    findings: list[dict[str, str]] = []
    try:
        historical_fields = store.previous_fields(entry.api_name, source)
        schema_reference = historical_fields if historical_fields is not None else previous_fields
        if schema_reference is not None and (
            not isinstance(schema_reference, list) or not schema_reference
            or not all(isinstance(field, str) and field for field in schema_reference)
            or len(set(schema_reference)) != len(schema_reference)
        ):
            raise IngestionError("invalid_schema_reference")
        pages = client.pages(
            entry.api_name,
            task["params"],
            observed_at,
            policy.primary_key,
            policy.page_size,
            policy.max_pages,
            policy.supports_pagination,
        )
        rows = [
            row
            for page in pages
            for row in normalize_rows(page.rows, policy, observed_at, page.raw.response_hash)
        ]
        if policy.date_field and policy.shard_by not in {"snapshot", "event"}:
            for row in rows:
                value = str(row.get(policy.date_field, ""))
                if len(value) == 7 and policy.date_field == "month":
                    if not task["start"][:7] <= value <= task["end"][:7]:
                        raise IngestionError("response_outside_planned_period")
                if len(value) == 10 and not task["start"] <= value <= task["end"]:
                    raise IngestionError("response_outside_planned_range")
        findings = audit_rows(
            rows,
            policy,
            expected_dates=expected_dates,
            suspensions=suspensions,
            adj_factors=adj_factors,
            previous_fields=schema_reference,
        )
        if findings:
            raise IngestionError("quality_failed")
        for row in rows:
            row["quality_reference"] = {
                "schema_fields": schema_reference,
                "origin": "previous_committed" if historical_fields is not None else
                "explicit_reference",
            }
        fields = pages[0].fields
        store.commit_task(task, rows, policy.primary_key, observed_at, source, fields, findings)
        return {
            "task_id": task["task_id"],
            "state": "committed",
            "rows": len(rows),
            "pages": len(pages),
            "quality": findings,
        }
    except (ProviderError, ValueError, OSError, duckdb.Error) as error:
        code = error.kind if isinstance(error, ProviderError) else "quality_or_storage_failed"
        store.fail_task(task["task_id"], code, findings)
        raise IngestionError(code) from None
