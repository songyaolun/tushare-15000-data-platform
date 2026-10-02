"""目录、政策和证据的共享数据结构，不包含账户凭据。"""

from dataclasses import dataclass, field
from typing import Any

LIFECYCLES = frozenset(
    {
        "production",
        "research",
        "archive",
        "independent_permission",
        "deprecated",
        "blocked",
        "unclassified",
    }
)
ENABLED = frozenset({"production", "research", "archive"})
PRODUCTS = {
    "market": "市场总览",
    "rotation": "行业与题材轮动",
    "security": "个股全景档案",
    "capital": "资金行为",
    "events": "事件与风险日历",
    "fund": "ETF与基金中心",
    "derivatives": "衍生品与固收",
    "sentiment": "市场情绪",
    "macro": "宏观环境",
    "coverage": "接口利用率与覆盖债务",
}


@dataclass
class Policy:
    """一项显式采集政策；Policy('daily') 默认不代表已经完成覆盖。"""

    api_name: str
    category: str = "unclassified"
    lifecycle: str = "unclassified"
    consumers: list[str] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    date_field: str = ""
    shard_by: str = ""
    lookback_days: int = 0
    required_fields: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    minimal_params: dict[str, Any] = field(default_factory=dict)
    supports_pagination: bool = False
    page_size: int = 1000
    max_pages: int = 100
    freshness_days: int = 7
    pit_fields: list[str] = field(default_factory=list)
    collection_mode: str = ""
    purpose: str = ""


@dataclass
class CatalogEntry:
    """发现记录和来源证据；CatalogEntry('daily') 进入未分类债务。"""

    api_name: str
    title: str = ""
    doc_url: str = ""
    category: str = "unclassified"
    permission_mode: str = "unknown"
    minimum_points: int | None = None
    row_limit: int | None = None
    frequency_per_minute: int | None = None
    updated_at: str = ""
    last_verified: str = ""
    lifecycle: str = "unclassified"
    policy: Policy | None = None
    needs_review: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)


def policy_from_dict(value: dict[str, Any]) -> Policy:
    """从 JSON 对象恢复政策；例如 {'api_name': 'daily'} → Policy('daily')。

    参数为与 Policy 字段一致的公开字典，返回可供调度使用的数据类。
    示例：policy_from_dict({'api_name': 'daily'}).consumers → []，不会补造消费用途。
    """
    return Policy(**value)


def entry_from_dict(value: dict[str, Any]) -> CatalogEntry:
    """恢复目录及嵌套政策；例如 {'api_name': 'daily'} → 未分类目录记录。

    参数是包含可选嵌套 policy 的字典；返回 CatalogEntry 且不修改输入。
    示例：entry_from_dict({'api_name': 'daily', 'policy': {'api_name': 'daily'}})
    返回 entry.policy.api_name == 'daily'；缺政策时保留 None。
    """
    copied = dict(value)
    if copied.get("policy") is not None:
        copied["policy"] = policy_from_dict(copied["policy"])
    return CatalogEntry(**copied)
