"""把账户实测与官方文档分开保存，实测成功可覆盖目录的保守限制。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any

from .models import CatalogEntry, Policy
from .store import Store, digest

WRITE_APIS = frozenset({"p_save", "p_delete"})
PRIVATE_APIS = frozenset({"p_get", "p_list"})


def readable_policy(entry: CatalogEntry) -> Policy:
    """取得只读接口政策，目录权限未知、停更或待复核均不阻止小范围探测。

    输入目录记录，返回用于执行的政策副本；账户写操作和非 API 交付拒绝。
    示例：independent_permission 的 daily 政策返回 research 副本，原记录保持不变。
    """
    if entry.api_name in WRITE_APIS:
        raise ValueError("account_write_not_probeable")
    if entry.policy is None or entry.api_name.startswith("unresolved_doc_"):
        raise ValueError("missing_api_contract")
    lifecycle = entry.policy.lifecycle
    if lifecycle not in {"production", "research", "archive"}:
        lifecycle = "archive" if entry.api_name in PRIVATE_APIS else "research"
    return replace(entry.policy, lifecycle=lifecycle)


def bind_account(store: Store, account: str) -> None:
    """将一个数据库的 live 数据绑定到公开账户别名，禁止跨账户复用运行证据。

    输入 Store 和不含凭据的别名，返回 None；不同账户须使用不同数据库。
    示例：先绑定 local 再绑定 local 成功，再绑定 another 抛 live_account_mismatch。
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", account):
        raise ValueError("invalid_account_alias")
    store.connection.execute(
        "CREATE TABLE IF NOT EXISTS live_account (singleton INTEGER PRIMARY KEY, name VARCHAR)"
    )
    row = store.connection.execute("SELECT name FROM live_account WHERE singleton=1").fetchone()
    if row is not None and row[0] != account:
        raise ValueError("live_account_mismatch")
    store.connection.execute("INSERT OR IGNORE INTO live_account VALUES (1, ?)", [account])


def latest_probe(store: Store, entry: CatalogEntry, account: str) -> dict[str, Any] | None:
    """取同账户、同政策的最新真实探测；fixture、旧政策和其他账户不能用于放行。

    输入数据库、目录和公开别名，返回探测记录或 None。
    示例：旧 available 后有新 unauthorized，返回 unauthorized，不继续借用旧成功。
    """
    fingerprint = digest(asdict(entry.policy)) if entry.policy is not None else ""
    matches = []
    for record in store.records("probes"):
        if record["api_name"] != entry.api_name or record["source"] != "live":
            continue
        evidence = json.loads(record["evidence"])
        if evidence.get("account") == account and evidence.get("request_policy") == fingerprint:
            matches.append(record)
    return max(matches, key=lambda record: record["observed_at"]) if matches else None


def verified_policy(store: Store, entry: CatalogEntry, account: str) -> Policy:
    """按实测结果放行采集，同时继续检查主键、消费用途等数据政策完整性。

    输入数据库、接口与账户，返回有效政策；非空成功或成功空响应均证明请求被接受。
    空响应不代表数据覆盖，采集质量检查仍会拒绝空批次。
    示例：fund_nav 实测 available 后即使目录 needs_review=True 也可返回采集政策。
    """
    from .coverage import policy_debt

    bind_account(store, account)
    policy = readable_policy(entry)
    record = latest_probe(store, entry, account)
    if record is None or record["status"] not in {"available", "empty"}:
        raise ValueError("successful_live_probe_required")
    if policy_debt(policy):
        raise ValueError("incomplete_policy")
    return policy
