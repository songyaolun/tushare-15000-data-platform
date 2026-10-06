"""206 个来源的唯一执行登记；基础层覆盖登记本身不代表业务执行。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, cast

from ..equity_research.consumption_registry import (
    CONSUMPTION_VERSION as EQUITY_VERSION,
)
from ..equity_research.consumption_registry import (
    PRODUCT_ALIASES as EQUITY_PRODUCTS,
)
from ..equity_research.consumption_registry import (
    load_consumption_registry,
    resolve_consumer,
)
from ..equity_research.models import CANDIDATE_RANKING_PRODUCT_IDS
from ..product_foundation.registry import ConsumerSpec, load_registry, validate_registry
from ..store import digest

VERSION = "all-products-consumption/v2"
NEW_PRODUCTS = {
    "P02": "theme_opportunities/v1",
    "P05": "expectation_revision/v1",
    "P08": "etf_exposure/v1",
    "P09": "commodity_transmission/v1",
    "P10": "international_transmission/v1",
    "P11": "financing_tail_risk/v1",
    "P12": "portfolio_diagnostics/v1",
}
RANKING_PRODUCTS = dict(
    zip(("P03S", "P03M", "P03L"), CANDIDATE_RANKING_PRODUCT_IDS, strict=True)
)
PRODUCTS = dict(sorted((EQUITY_PRODUCTS | NEW_PRODUCTS | RANKING_PRODUCTS).items()))
DEFAULT_PRODUCTS = frozenset(PRODUCTS) - frozenset(RANKING_PRODUCTS)
RANKING_SOURCE_PRODUCTS = {
    "P03S": frozenset({"P03"}),
    "P03M": frozenset({"P03"}),
    "P03L": frozenset({"P03", "P04"}),
}
HANDLERS = {
    "P02": "themes.build_theme_opportunities",
    "P05": "expectations.build_expectation_revision",
    "P08": "funds.build_etf_exposure",
    "P09": "commodities.build_commodities",
    "P10": "international.build_international",
    "P11": "financing.build_financing",
    "P12": "portfolio.build_portfolio",
}
EXPECTED_COUNTS = {"P02": 22, "P05": 7, "P08": 18, "P09": 19, "P10": 26, "P11": 14, "P12": 1}


def source_products(product: str) -> frozenset[str]:
    """返回一个可执行产品直接读取的登记来源产品集合。

    例如 P03L 返回 ``{'P03', 'P04'}`` 以读取估值和三年财务；普通 P05 返回自身。
    """
    return RANKING_SOURCE_PRODUCTS.get(product, frozenset({product}))


@dataclass(frozen=True)
class Registration:
    """不可变来源登记，使用字段与用途保留矩阵原文。"""

    spec: ConsumerSpec
    handler: str


def registrations() -> tuple[Registration, ...]:
    """核验权威矩阵并绑定全部接口，返回按 API 排序的 206 项登记。

    例如 forecast 绑定 P05 的预期处理器；若新增矩阵项未归属则立即拒绝。
    旧六产品继续使用原消费处理器，因而不把基础 coverage 当作执行证明。
    """
    entries = load_registry()
    validate_registry(entries)
    legacy = {item.spec.api: item for item in load_consumption_registry()}
    counts = Counter(item.product for item in entries if item.product in NEW_PRODUCTS)
    if dict(counts) != EXPECTED_COUNTS or len(legacy) != 99:
        raise ValueError("all_products_registry_mismatch")
    return tuple(
        Registration(
            item,
            HANDLERS[item.product]
            if item.product in NEW_PRODUCTS
            else "equity." + legacy[item.api].handler,
        )
        for item in entries
    )


def resolve_handler(name: str) -> Callable[..., Any]:
    """只解析本模块白名单内真实可调用处理器，不允许用户注入模块路径。

    输入 themes.build_theme_opportunities 返回主题构造函数；未知名称抛固定错误。
    """
    if name.startswith("equity."):
        return resolve_consumer(name.removeprefix("equity."))
    if name not in HANDLERS.values():
        raise ValueError("all_products_handler_missing")
    module, attribute = name.split(".")
    handler: Any = getattr(import_module(f"{__package__}.{module}"), attribute, None)
    if not callable(handler):
        raise ValueError("all_products_handler_missing")
    return cast(Callable[..., Any], handler)


def fingerprint() -> str:
    """把规则版本和逐项登记绑定到运行身份，防止跨版本错误复用。

    无参数输入得到 64 位摘要；字段、处理器、旧消费版本变化均改变结果。
    """
    return digest(
        [VERSION, EQUITY_VERSION, PRODUCTS, [asdict(item) for item in registrations()]]
    )


def validate_handlers() -> None:
    """在扫描前解析所有 206 项的处理器，缺实现不能记为 registered 完成。

    例如 financing 模块缺失时运行立即失败，而不会产出伪造的执行摘要。
    """
    for name in sorted({item.handler for item in registrations()}):
        resolve_handler(name)
    for item in registrations():
        if item.spec.product in NEW_PRODUCTS:
            module = import_module(f"{__package__}.{item.handler.split('.')[0]}")
            fields = getattr(module, "API_FIELDS", {})
            if tuple(fields.get(item.spec.api, ())) != item.spec.selected_fields:
                raise ValueError("all_products_handler_fields_mismatch")


def select_products(selection: tuple[str, ...]) -> tuple[str, ...]:
    """解析别名、展开真实依赖并按稳定顺序返回产品别名。

    空输入返回原有十三产品；P03S/M/L 只在显式选择时运行。P03/P03S
    展开 P04/P06/P07，P03M/P03L 展开 P04/P06，P12 扩展原有十三产品。
    """
    reverse = {value: key for key, value in PRODUCTS.items()}
    selected = (
        {reverse.get(item, item) for item in selection}
        if selection
        else set(DEFAULT_PRODUCTS)
    )
    if not selected <= PRODUCTS.keys():
        raise ValueError("unknown_all_products_selection")
    if "P12" in selected:
        selected.update(DEFAULT_PRODUCTS)
    if selected & {"P03", "P03S", "P13"}:
        selected.update({"P04", "P06", "P07"})
    if selected & {"P03M", "P03L"}:
        selected.update({"P04", "P06"})
    order = (
        "P01",
        "P04",
        "P06",
        "P07",
        "P03",
        "P03S",
        "P03M",
        "P03L",
        "P13",
        "P02",
        "P05",
        "P08",
        "P09",
        "P10",
        "P11",
        "P12",
    )
    return tuple(item for item in order if item in selected)
