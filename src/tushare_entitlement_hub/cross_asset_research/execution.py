"""按依赖顺序运行真实业务构造器，及时产出结果而不集中复制完整结果库。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..equity_research.adapter import SOURCE_APIS as EQUITY_SOURCE_APIS
from ..equity_research.adapter import load_inputs as load_equity_inputs
from ..equity_research.candidate_horizons import (
    build_long_term_candidates,
    build_medium_term_candidates,
    build_short_term_candidates,
)
from ..equity_research.candidates import build_candidates
from ..equity_research.common import gate, make_result
from ..equity_research.consumption import attach_consumption
from ..equity_research.events import build_events
from ..equity_research.flow import build_flow
from ..equity_research.fundamental import build_fundamental
from ..equity_research.governance import build_governance
from ..equity_research.market import build_market
from ..equity_research.models import MARKET_ENTITY, ResearchConfig, ResearchInputs, ResearchResult
from ..product_foundation.contracts import Truth
from ..product_foundation.progress import ProgressCallback
from .audit import ConsumptionAudit
from .registry import (
    HANDLERS,
    NEW_PRODUCTS,
    PRODUCTS,
    registrations,
    resolve_handler,
    source_products,
)

EQUITY_SUPPORT_APIS = {
    "P01": frozenset({"daily", "stk_limit", "suspend_d"}),
    "P02": frozenset({"daily"}),
    "P03": frozenset({"daily", "daily_basic"}),
    "P04": frozenset({"daily_basic"}),
    "P06": frozenset({"daily_basic"}),
    "P07": frozenset({"daily", "daily_basic", "stk_limit", "suspend_d"}),
    "P08": frozenset({"index_daily"}),
    "P13": frozenset({"daily"}),
}


@dataclass(frozen=True)
class RunLimits:
    """输出总量及组合父结果预算，单结果仍由公共存储强制限制为 8MiB。"""

    maximum_results: int = 100_000
    maximum_parent_results: int = 20_000
    maximum_output_bytes: int = 32 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        """拒绝布尔、零和负容量；例如 maximum_results=0 抛容量配置错误。"""
        if any(
            type(value) is not int or value < 1
            for value in (
                self.maximum_results,
                self.maximum_parent_results,
                self.maximum_output_bytes,
            )
        ):
            raise ValueError("all_products_capacity_invalid")


@dataclass
class ExecutionState:
    """只保存有界计数和最终消费摘要，不保存业务结果副本。"""

    processing: dict[str, int] = field(default_factory=dict)
    states: dict[str, Counter[str]] = field(default_factory=dict)
    consumption: dict[str, Any] = field(default_factory=dict)
    result_count: int = 0


def execute(
    reference: Path,
    cutoff: str,
    run_id: str,
    config: ResearchConfig,
    products: tuple[str, ...],
    holdings: tuple[dict[str, Any], ...],
    limits: RunLimits,
    state: ExecutionState,
    progress: ProgressCallback | None = None,
) -> Iterator[ResearchResult]:
    """逐产品构造并流式交给分区写入器，旧六产品复用原构造及消费逻辑。

    输入 P03 闭包时先运行 P04/P06/P07；P12 只保留本次所用持仓实体相关的父结果。
    空输入也必须实际调用处理器，并输出有原因的降级结果，不能伪造执行成功。
    组合父结果在附加消费审计后收集，与随后落库的模型完全一致，
    因此P12的父结果摘要可直接由其引用的已保存父结果复算。
    只有执行P12且未提供显式持仓时才解析基金披露，避免无关实体消耗父结果预算。
    """
    from .adapter import load_inputs
    from .bindings import equity_config
    from .portfolio import holding_entity

    legacy_config = equity_config(config)
    all_registrations = registrations()
    product_source_apis = {
        product: frozenset(
            item.spec.api
            for item in all_registrations
            if item.spec.product in source_products(product)
        )
        for product in PRODUCTS
    }
    shared_source_apis = frozenset().union(
        *(EQUITY_SUPPORT_APIS.get(product, frozenset()) for product in products)
    )
    shared_inputs = None
    state.processing = {}

    def record_processing(counts: dict[str, int]) -> None:
        """把互不重叠的来源批次累计进运行账本；例：两批各保留2行得到4行。"""
        for name, value in counts.items():
            state.processing[name] = state.processing.get(name, 0) + value
        if state.processing.get("retained_rows", 0) > config.maximum_total_rows:
            raise ValueError("source_capacity_exceeded")

    if shared_source_apis:
        shared_inputs, shared_counts = load_equity_inputs(
            reference,
            cutoff,
            run_id,
            legacy_config,
            progress=progress,
            source_apis=shared_source_apis,
        )
        record_processing(shared_counts)
    product_inputs: dict[str, tuple[ResearchInputs, tuple[ResearchInputs, ...]]] = {}

    def load_product_inputs(product: str) -> ResearchInputs:
        """按产品一次加载其独占来源并累计全局扫描预算，再合入共享旁证。

        例如先加载 P02 后加载 P05，第二次剩余行数扣除 P02 已保留行；同一产品
        重用缓存，不重复扫描参考库。返回上下文仅包含该产品来源和显式共享来源。
        """
        existing = product_inputs.get(product)
        if existing is not None:
            return existing[0]
        contexts = [shared_inputs] if shared_inputs is not None else []
        audit_contexts = []
        equity_apis = (product_source_apis[product] & EQUITY_SOURCE_APIS) - shared_source_apis
        if equity_apis:
            selected_equity, equity_counts = load_equity_inputs(
                reference,
                cutoff,
                run_id,
                legacy_config,
                progress=progress,
                source_apis=frozenset(equity_apis),
            )
            record_processing(equity_counts)
            contexts.append(selected_equity)
            audit_contexts.append(selected_equity)
        cross_apis = product_source_apis[product] - EQUITY_SOURCE_APIS
        if cross_apis:
            selected_cross, cross_counts = load_inputs(
                reference,
                cutoff,
                run_id,
                config,
                progress=progress,
                remaining_rows=max(
                    0, config.maximum_total_rows - state.processing.get("retained_rows", 0)
                ),
                source_apis=frozenset(cross_apis),
            )
            record_processing(cross_counts)
            contexts.append(selected_cross)
            audit_contexts.append(selected_cross)
        base = contexts[-1]
        rows_by_api = {}
        source_reasons = {}
        entity_source_reasons = {}
        for context in contexts:
            rows_by_api.update(context.rows_by_api)
            source_reasons.update(context.source_reasons)
            entity_source_reasons.update(context.entity_source_reasons)
        combined = replace(
            base,
            config=config if product in NEW_PRODUCTS else legacy_config,
            rows_by_api=rows_by_api,
            source_reasons=source_reasons,
            entity_source_reasons=entity_source_reasons,
        )
        product_inputs[product] = combined, tuple(audit_contexts)
        return combined

    parents: dict[str, tuple[ResearchResult, ...]] = {}
    parent_consumers = {"P03", "P03S", "P03M", "P03L", "P13"}.intersection(products)
    portfolio_parents: list[ResearchResult] = []
    portfolio_entities = {str(item.get("entity_id", "")) for item in holdings}
    if "P12" in products and not holdings:
        for row in load_product_inputs("P12").rows("fund_portfolio"):
            symbol, period = row.text("symbol"), row.availability.report_period
            if symbol is not None and period is not None:
                resolved = holding_entity(product_inputs["P12"][0], symbol, period)
                if resolved is not None:
                    portfolio_entities.add(resolved)
    builders = {
        "P01": build_market,
        "P04": build_fundamental,
        "P06": build_governance,
        "P07": build_flow,
    }
    with ConsumptionAudit(products, config.binding_audit) as audit:
        if shared_inputs is not None:
            audit.inputs(shared_inputs)
        for _, audit_contexts in product_inputs.values():
            for current in audit_contexts:
                audit.inputs(current)
        for product in products:
            already_loaded = product in product_inputs
            inputs = load_product_inputs(product)
            if not already_loaded:
                for current in product_inputs[product][1]:
                    audit.inputs(current)
            if product in builders:
                results = builders[product](inputs)
            elif product == "P03":
                results = build_candidates(inputs, parents["P04"], parents["P06"], parents["P07"])
            elif product == "P03S":
                results = build_short_term_candidates(
                    inputs, parents["P04"], parents["P06"], parents["P07"]
                )
            elif product == "P03M":
                results = build_medium_term_candidates(inputs, parents["P04"], parents["P06"])
            elif product == "P03L":
                results = build_long_term_candidates(inputs, parents["P04"], parents["P06"])
            elif product == "P13":
                results = build_events(
                    inputs,
                    tuple(result for name in ("P04", "P06", "P07") for result in parents[name]),
                )
            elif product == "P12":
                results = resolve_handler(HANDLERS[product])(
                    inputs, parents=tuple(portfolio_parents), holdings=holdings
                )
            else:
                results = resolve_handler(HANDLERS[product])(inputs)
            if not results:
                results = (
                    make_result(
                        inputs,
                        PRODUCTS[product],
                        MARKET_ENTITY,
                        gates=(gate("input_scope", Truth.UNKNOWN, "empty_input"),),
                        details={"reason": "empty_input"},
                    ),
                )
            if product in {"P04", "P06", "P07"}:
                parents[product] = results
            if product not in NEW_PRODUCTS:
                results = attach_consumption(inputs, PRODUCTS[product], results)
            for parent_candidate in results:
                if "P12" in products and parent_candidate.envelope.entity_id in portfolio_entities:
                    if len(portfolio_parents) >= limits.maximum_parent_results:
                        raise ValueError("portfolio_parent_capacity_exceeded")
                    portfolio_parents.append(parent_candidate)
            for result in results:
                state.result_count += 1
                if state.result_count > limits.maximum_results:
                    raise ValueError("all_products_result_capacity_exceeded")
                audit.result(product, result)
                state.states.setdefault(product, Counter())[result.envelope.state.value] += 1
                yield result
            parent_consumers.discard(product)
            if not parent_consumers:
                parents.clear()
            if progress is not None:
                progress(
                    {
                        "stage": "product_complete",
                        "product": product,
                        "results": len(results),
                        "total_results": state.result_count,
                    }
                )
            product_inputs.pop(product, None)
        state.consumption = audit.summary()
