"""逐产品核对范围证书和必需特征，避免把结构校验冒充可发布状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .reference_scope import certified_dates


@dataclass(frozen=True)
class ScopeRequirement:
    """一个产品对单接口的范围要求。"""

    api_name: str
    minimum_certified_days: int = 0
    date_field: str = "trade_date"
    minimum_completed_partitions: int = 0

    def __post_init__(self) -> None:
        """拒绝空接口名和负日期要求；例如 minimum_certified_days=20 合法。"""
        if (
            not self.api_name
            or self.minimum_certified_days < 0
            or self.minimum_completed_partitions < 0
        ):
            raise ValueError("invalid_scope_requirement")


@dataclass(frozen=True)
class ProductReadiness:
    """产品发布状态、缺口和各接口已认证日期数。"""

    product_id: str
    status: Literal["ready", "incomplete", "blocked"]
    reason_codes: tuple[str, ...]
    certified_day_counts: dict[str, int]

    @property
    def ready(self) -> bool:
        """只有状态为 ready 时返回 True；incomplete 和 blocked 均不能正式发布。"""
        return self.status == "ready"


P02_SCOPE_REQUIREMENTS = (
    ScopeRequirement("trade_cal", 20, "cal_date"),
    ScopeRequirement("stock_basic"),
    ScopeRequirement("ths_index"),
    ScopeRequirement("ths_member", minimum_completed_partitions=1666),
    ScopeRequirement("daily", 20),
    ScopeRequirement("stk_limit", 20),
    ScopeRequirement("ths_hot", 20),
    ScopeRequirement("moneyflow_cnt_ths", 20),
    ScopeRequirement("kpl_list", 20),
)


def assess_product_readiness(
    product_id: str,
    certificates: tuple[dict[str, Any], ...],
    requirements: tuple[ScopeRequirement, ...],
    *,
    target_date: str | None = None,
    blocked_reasons: tuple[str, ...] = (),
) -> ProductReadiness:
    """按接口和日期证书评估产品，不对缺少的范围做推断或补零。

    输入 P02 的九项要求、20 个完整交易日及目标日，返回 ready；少一天返回 incomplete
    和 ``scope_days_insufficient:<api>``。权限、参数等不可发布原因传入 blocked_reasons 后
    返回 blocked，并保留所有范围缺口供修复。
    """
    reasons = set(blocked_reasons)
    counts: dict[str, int] = {}
    for requirement in requirements:
        dates = certified_dates(certificates, requirement.api_name, requirement.date_field)
        counts[requirement.api_name] = len(dates)
        matching = [
            item
            for item in certificates
            if item.get("api_name") == requirement.api_name
            and item.get("status") == "certified"
        ]
        if not matching:
            reasons.add("scope_certificate_missing:" + requirement.api_name)
            continue
        if max(int(item.get("completed_partition_count", 0)) for item in matching) < (
            requirement.minimum_completed_partitions
        ):
            reasons.add("scope_partitions_insufficient:" + requirement.api_name)
        if len(dates) < requirement.minimum_certified_days:
            reasons.add("scope_days_insufficient:" + requirement.api_name)
        if (
            target_date is not None
            and requirement.minimum_certified_days > 0
            and target_date not in dates
        ):
            reasons.add("target_date_not_certified:" + requirement.api_name)
    status: Literal["ready", "incomplete", "blocked"] = (
        "blocked"
        if blocked_reasons
        else "incomplete"
        if reasons
        else "ready"
    )
    return ProductReadiness(product_id, status, tuple(sorted(reasons)), counts)
