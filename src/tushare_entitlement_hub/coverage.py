"""基于登记、探活、任务和消费证据计算覆盖率，不以默认值抹平覆盖债务。"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from .models import ENABLED, LIFECYCLES, PRODUCTS, Policy
from .store import Store


def meaningful(value: str) -> bool:
    """拒绝空文本及未填写占位符；meaningful('TODO') 为 False，实际用途描述为 True。

    参数 value 是待审核文本，返回是否含有效内容，不修改输入。
    示例：meaningful('研究日度资金变化') → True；meaningful('  default  ') → False。
    """
    return value.strip().lower() not in {
        "",
        "unknown",
        "unclassified",
        "default",
        "todo",
        "tbd",
        "n/a",
        "none",
        "null",
        "未知",
        "待补充",
        "默认",
    }


def policy_debt(policy: Policy) -> list[str]:
    """检查采集和消费政策是否完整；无主键的 Policy('daily') 返回明确债务。

    参数是单个 Policy，返回稳定债务代码列表，空列表表示结构契约完整。
    示例：policy_debt(Policy('daily')) 包含 missing_primary_key 与 missing_consumer。
    """
    debt = []
    if not meaningful(policy.api_name) or not meaningful(policy.category):
        debt.append("missing_category")
    if policy.lifecycle not in ENABLED:
        debt.append("invalid_policy_lifecycle")
    if not policy.primary_key or any(not meaningful(field) for field in policy.primary_key):
        debt.append("missing_primary_key")
    if not policy.required_fields or not set(policy.primary_key).issubset(policy.required_fields):
        debt.append("missing_field_contract")
    if (
        not meaningful(policy.shard_by)
        or not meaningful(policy.collection_mode)
        or not meaningful(policy.purpose)
    ):
        debt.append("missing_collection_policy")
    if policy.shard_by != "snapshot" and not meaningful(policy.date_field):
        debt.append("missing_watermark_dimension")
    if policy.lookback_days < 0 or policy.page_size < 1 or policy.max_pages < 1:
        debt.append("invalid_collection_bounds")
    if policy.freshness_days < 1 or any(not unit.strip() for unit in policy.units.values()):
        debt.append("invalid_units_or_freshness")
    if not policy.consumers or any(
        consumer not in PRODUCTS
        and not (
            consumer.startswith(("research:", "archive:")) and meaningful(consumer.split(":", 1)[1])
        )
        for consumer in policy.consumers
    ):
        debt.append("missing_consumer")
    return debt


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    """返回含分子分母的可审计比例；ratio(0, 0) 的 rate=None，避免无任务冒充 100%。

    参数为非负分子及分母，返回 numerator、denominator、rate。
    示例：ratio(1, 2) → {'numerator': 1, 'denominator': 2, 'rate': 0.5}。
    """
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def coverage_report(store: Store, as_of_date: str, source: str = "fixture") -> dict[str, Any]:
    """计算六项覆盖指标并列出债务；fixture 探活率和真实账户探活率分别输出。

    参数为 Store、报告日期和 fixture/live 来源；返回指标、债务、门禁和分类计数。
    示例：空库报告 discovered=0、gate_passed=False、registration.rate=None。
    临时故障不计作已明确验证权限，离线证据始终不计入 live_probe。
    """
    entries = store.catalog()
    enabled = {
        entry.api_name: entry
        for entry in entries
        if entry.permission_mode == "points" and entry.lifecycle in ENABLED
    }
    debt: list[dict[str, str]] = []
    registered = 0
    for entry in entries:
        errors = []
        if entry.lifecycle not in LIFECYCLES or entry.lifecycle == "unclassified":
            errors.append("unclassified")
        if not entry.title.strip() or not entry.doc_url.startswith("https://tushare.pro/document/"):
            errors.append("missing_documentation")
        if entry.needs_review:
            errors.append("documentation_conflict")
        if not entry.last_verified.strip():
            errors.append("missing_verification_time")
        if not meaningful(entry.updated_at):
            errors.append("missing_update_schedule")
        if entry.lifecycle == "blocked":
            errors.append("blocked_interface")
        if entry.permission_mode not in {"points", "independent", "deprecated"}:
            errors.append("unknown_permission")
        if entry.permission_mode == "points" and entry.lifecycle in ENABLED:
            errors.extend(policy_debt(entry.policy) if entry.policy else ["missing_policy"])
            if entry.minimum_points is None or entry.minimum_points > 15000:
                errors.append("points_not_verified")
        if not errors:
            registered += 1
        debt.extend({"api_name": entry.api_name, "code": code} for code in errors)
    probes = store.records("probes")
    conclusive = {"available", "empty", "unauthorized", "unsupported"}
    tasks = [task for task in store.records("tasks") if task["source"] == source]
    committed = {
        task["api_name"]
        for task in tasks
        if task["state"] == "committed" and store.rows(task["api_name"], source)
    }
    consumed = {
        row["api_name"]
        for row in store.records("consumption")
        if row["source"] == source and row["row_count"] > 0
    }
    freshness = 0
    cutoff = date.fromisoformat(as_of_date[:10])
    for name, entry in enabled.items():
        policy = entry.policy
        if policy is None:
            continue
        records = store.rows(name, source)
        dates = [
            str(row.get(policy.date_field, ""))[:10]
            for row in records
            if row.get(policy.date_field)
        ]
        if policy.shard_by == "snapshot" or policy.collection_mode == "snapshot":
            dates = [str(row["observed_at"])[:10] for row in records]
        if dates:
            try:
                latest = date.fromisoformat(max(dates))
                freshness += int(0 <= (cutoff - latest).days <= policy.freshness_days)
            except ValueError:
                debt.append({"api_name": name, "code": "freshness_unknown"})
    backfills = [task for task in tasks if json.loads(task["payload"])["mode"] == "backfill"]
    for record in store.records("quality"):
        if record["task_id"] not in {task["task_id"] for task in tasks}:
            continue
        for finding in json.loads(record["findings"]):
            if finding["code"] != "schema_drift_not_checked":
                debt.append({"api_name": record["api_name"], "code": finding["code"]})
    for name in enabled:
        if name not in committed:
            debt.append({"api_name": name, "code": "not_collected"})
        if name not in consumed:
            debt.append({"api_name": name, "code": "not_consumed"})
    return {
        "as_of": as_of_date,
        "source": source,
        "discovered": len(entries),
        "enabled": len(enabled),
        "metrics": {
            "registration": ratio(registered, len(entries)),
            "capability_probe": ratio(
                len(
                    {
                        row["api_name"]
                        for row in probes
                        if row["source"] == source
                        and row["api_name"] in enabled
                        and row["status"] in conclusive
                    }
                ),
                len(enabled),
            ),
            "collection": ratio(len(committed & enabled.keys()), len(enabled)),
            "consumption": ratio(len(consumed & enabled.keys()), len(enabled)),
            "freshness": ratio(freshness, len(enabled)),
            "historical_backfill": ratio(
                sum(t["state"] == "committed" for t in backfills), len(backfills)
            ),
        },
        "live_probe": ratio(
            len(
                {
                    row["api_name"]
                    for row in probes
                    if row["source"] == "live"
                    and row["api_name"] in enabled
                    and row["status"] in conclusive
                }
            ),
            len(enabled),
        ),
        "probe_status_counts": {
            status: sum(row["source"] == source and row["status"] == status for row in probes)
            for status in (
                "available",
                "empty",
                "unauthorized",
                "unsupported",
                "transient_failure",
                "contract_failure",
            )
        },
        "debt": debt,
        "gate_passed": bool(entries) and not debt,
        "note": "离线演示证据，不代表真实账户权限或真实市场数据"
        if source == "fixture"
        else "账户实测证据；权限失败不计作空数据",
    }
