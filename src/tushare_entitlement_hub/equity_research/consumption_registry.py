"""把六产品的离线消费矩阵绑定到可执行处理器及有限读取策略。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from ..product_foundation.registry import ConsumerSpec, load_registry
from ..store import digest
from .models import PRODUCT_IDS, ResearchInputs, ResearchRow

CONSUMPTION_VERSION = "equity-consumption/v8"
PRODUCT_ALIASES = dict(zip(("P01", "P04", "P06", "P07", "P03", "P13"), PRODUCT_IDS, strict=True))
CONTEXT_APIS = frozenset(
    {
        "trade_cal",
        "eco_cal",
        "daily_info",
        "sz_daily_info",
        "index_basic",
        "index_daily",
        "index_dailybasic",
        "index_weekly",
        "index_monthly",
        "idx_factor_pro",
        "shibor_lpr",
        "stk_account",
        "moneyflow_mkt_dc",
        "adj_factor",
        "bak_daily",
        "weekly",
        "monthly",
        "pro_bar",
        "stk_weekly_monthly",
        "stk_week_month_adj",
        "bse_mapping",
        "factor_list",
        "factor_value",
        "stk_factor",
        "stk_factor_pro",
        "stk_nineturn",
    }
)
DISCLOSURE_APIS = frozenset(
    {
        "bak_basic",
        "fina_indicator",
        "fina_indicator_vip",
        "new_share",
        "stk_managers",
        "stk_rewards",
        "cb_daily",
        "cb_factor_pro",
        "cb_issue",
        "cb_rate",
        "cb_rating",
        "top10_cb_holders",
        "stk_holdernumber",
        "top10_holders",
        "top10_floatholders",
    }
)
FLOW_APIS = frozenset(
    {
        "stock_hsgt",
        "hm_detail",
        "hm_list",
        "limit_step",
        "top_list",
        "margin_secs",
        "slb_sec",
        "slb_sec_detail",
        "hsgt_top10",
        "cyq_perf",
    }
)
CORE_APIS = frozenset(
    {
        "stock_basic",
        "daily",
        "daily_basic",
        "stk_limit",
        "suspend_d",
        "stk_shock",
        "stk_high_shock",
        "limit_list_d",
        "limit_list_ths",
        "moneyflow",
        "moneyflow_dc",
        "moneyflow_ths",
        "margin_detail",
        "cyq_chips",
        "block_trade",
        "top_inst",
        "rt_min",
        "income",
        "income_vip",
        "balancesheet",
        "balancesheet_vip",
        "cashflow",
        "cashflow_vip",
        "fina_audit",
        "share_float",
        "stk_seasoned",
        "pledge_stat",
        "pledge_detail",
        "st",
        "stock_st",
        "stk_alert",
        "cb_basic",
        "cb_share",
        "cb_call",
        "repurchase",
        "dividend",
        "stk_holdertrade",
        "cn_pmi",
        "cn_ppi",
        "cn_m",
        "cn_cpi",
        "cn_gdp",
        "sf_month",
        "major_news",
        "disclosure_date",
        "cn_schedule",
        "idx_anns",
        "namechange",
    }
)
BOND_APIS = frozenset(
    {
        "cb_basic",
        "cb_share",
        "cb_call",
        "cb_daily",
        "cb_factor_pro",
        "cb_issue",
        "cb_rate",
        "cb_rating",
        "top10_cb_holders",
    }
)
INDEX_APIS = frozenset(
    {
        "index_basic",
        "index_daily",
        "index_dailybasic",
        "index_weekly",
        "index_monthly",
        "idx_factor_pro",
    }
)
DICTIONARY_APIS = frozenset({"factor_list", "hm_list", "bse_mapping"})
UNRESOLVED_EVENT_APIS = frozenset({"major_news", "idx_anns", "cn_schedule"})
EVENT_APIS = frozenset(
    {
        "share_float",
        "stk_seasoned",
        "pledge_detail",
        "cb_basic",
        "cb_call",
        "repurchase",
        "dividend",
        "stk_holdertrade",
        "major_news",
        "disclosure_date",
        "cn_schedule",
        "idx_anns",
        "namechange",
        "stk_alert",
        "st",
        "stock_st",
        "cyq_chips",
        "new_share",
        "cb_issue",
        "cb_rate",
        "stk_managers",
        "eco_cal",
    }
)
TWO_PERIOD_APIS = frozenset(
    {
        "cn_pmi",
        "cn_ppi",
        "cn_m",
        "cn_cpi",
        "cn_gdp",
        "sf_month",
        "index_daily",
        "weekly",
        "monthly",
        "index_weekly",
        "index_monthly",
        "adj_factor",
        "stk_factor",
        "stk_factor_pro",
        "stk_weekly_monthly",
        "stk_week_month_adj",
        "shibor_lpr",
        "stk_account",
        "cb_rating",
        "stk_holdernumber",
        "fina_indicator",
        "fina_indicator_vip",
        "slb_sec",
        "stock_hsgt",
    }
)
Consumer = Callable[[ResearchInputs, str, str, tuple[ResearchRow, ...], str], dict[str, Any]]


@dataclass(frozen=True)
class ConsumptionRegistration:
    """接口事实与处理器、实体范围和保留策略的不可变绑定。"""

    spec: ConsumerSpec
    handler: str
    entity_scope: str
    retention: str


@lru_cache(maxsize=1)
def load_consumption_registry() -> tuple[ConsumptionRegistration, ...]:
    """从完整矩阵挑选六产品并逐一绑定执行器，遗漏或重复注册立即拒绝。

    输入无参数，输出精确99项；新增CSV接口但未绑定处理器会抛ValueError。
    """
    handlers = {
        "core": CORE_APIS,
        "context": CONTEXT_APIS,
        "disclosure": DISCLOSURE_APIS,
        "flow": FLOW_APIS,
    }
    entries = tuple(item for item in load_registry() if item.product in PRODUCT_ALIASES)
    declared = [api for apis in handlers.values() for api in apis]
    if (
        len(entries) != 99
        or len(declared) != len(set(declared))
        or set(declared) != {item.api for item in entries}
    ):
        raise ValueError("research_consumption_registry_mismatch")
    result = []
    for item in entries:
        scope = (
            "bond"
            if item.api in BOND_APIS
            else "dictionary"
            if item.api in DICTIONARY_APIS
            else "market"
            if item.product == "P01"
            else "event"
            if item.api in UNRESOLVED_EVENT_APIS
            else "stock"
        )
        retention = (
            "bounded_events"
            if item.api in EVENT_APIS
            else "two_periods"
            if item.api in TWO_PERIOD_APIS
            else "current_period"
        )
        result.append(
            ConsumptionRegistration(
                item,
                next(name for name, apis in handlers.items() if item.api in apis),
                scope,
                retention,
            )
        )
    return tuple(result)


def resolve_consumer(name: str) -> Consumer:
    """解析已注册的真实函数，拒绝只有名称而没有可调用实现的绑定。

    输入'context'，输出consume_context函数；输入未知标识则抛ValueError。
    """
    from .consumption import consume_core
    from .context_consumers import consume_context
    from .disclosure_consumers import consume_disclosure
    from .flow_consumers import consume_flow

    handler = {
        "core": consume_core,
        "context": consume_context,
        "disclosure": consume_disclosure,
        "flow": consume_flow,
    }.get(name)
    if not callable(handler):
        raise ValueError("research_consumption_handler_missing")
    return handler


def registry_fingerprint() -> str:
    """将消费规则和完整注册内容纳入指纹，防止旧结果被新处理器复用。

    输入当前注册，输出64位摘要；角色、字段或保留策略变化都会改变摘要。
    """
    return digest([CONSUMPTION_VERSION, [asdict(item) for item in load_consumption_registry()]])


def validate_consumption_registry() -> None:
    """校验99项分配和可调用绑定；例：缺失consume_flow时运行开始即失败。"""
    load_consumption_registry.cache_clear()
    for name in {item.handler for item in load_consumption_registry()}:
        resolve_consumer(name)
