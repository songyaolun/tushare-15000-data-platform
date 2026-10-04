"""按依赖顺序运行真实业务构造器，及时产出结果而不集中复制完整结果库。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..equity_research.adapter import load_inputs as load_equity_inputs
from ..equity_research.candidates import build_candidates
from ..equity_research.common import gate, make_result
from ..equity_research.consumption import attach_consumption
from ..equity_research.events import build_events
from ..equity_research.flow import build_flow
from ..equity_research.fundamental import build_fundamental
from ..equity_research.governance import build_governance
from ..equity_research.market import build_market
from ..equity_research.models import MARKET_ENTITY, ResearchConfig, ResearchResult
from ..product_foundation.contracts import Truth
from ..product_foundation.progress import ProgressCallback
from .audit import ConsumptionAudit
from .registry import HANDLERS, NEW_PRODUCTS, PRODUCTS, resolve_handler


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

    equity, equity_counts = load_equity_inputs(reference, cutoff, run_id, equity_config(config))
    cross, cross_counts = load_inputs(
        reference,
        cutoff,
        run_id,
        config,
        progress=progress,
        remaining_rows=max(0, config.maximum_total_rows - equity_counts["retained_rows"]),
    )
    state.processing = {
        name: equity_counts.get(name, 0) + cross_counts.get(name, 0)
        for name in sorted(equity_counts.keys() | cross_counts.keys())
    }
    cross = replace(
        cross,
        rows_by_api=equity.rows_by_api | cross.rows_by_api,
        source_reasons=equity.source_reasons | cross.source_reasons,
        entity_source_reasons=equity.entity_source_reasons | cross.entity_source_reasons,
    )
    parents: dict[str, tuple[ResearchResult, ...]] = {}
    portfolio_parents: list[ResearchResult] = []
    portfolio_entities = {str(item.get("entity_id", "")) for item in holdings}
    if "P12" in products and not holdings:
        for row in cross.rows("fund_portfolio"):
            symbol, period = row.text("symbol"), row.availability.report_period
            if symbol is not None and period is not None:
                resolved = holding_entity(cross, symbol, period)
                if resolved is not None:
                    portfolio_entities.add(resolved)
    builders = {
        "P01": build_market,
        "P04": build_fundamental,
        "P06": build_governance,
        "P07": build_flow,
    }
    with ConsumptionAudit(products, config.binding_audit) as audit:
        audit.inputs(cross)
        for product in products:
            inputs = cross if product in NEW_PRODUCTS else equity
            if product in builders:
                results = builders[product](equity)
            elif product == "P03":
                results = build_candidates(equity, parents["P04"], parents["P06"], parents["P07"])
            elif product == "P13":
                results = build_events(
                    equity,
                    tuple(result for name in ("P04", "P06", "P07") for result in parents[name]),
                )
            elif product == "P12":
                results = resolve_handler(HANDLERS[product])(
                    cross, parents=tuple(portfolio_parents), holdings=holdings
                )
            else:
                results = resolve_handler(HANDLERS[product])(cross)
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
            if product == "P13" or (product == "P03" and "P13" not in products):
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
        state.consumption = audit.summary()
