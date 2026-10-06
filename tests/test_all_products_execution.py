"""执行编排的依赖传递、有界父结果及真实执行计数契约。"""

from dataclasses import replace

import pytest
from test_research_shared import IDENTITY, NOW, inputs, source

from tushare_entitlement_hub.cross_asset_research import adapter, execution
from tushare_entitlement_hub.cross_asset_research.registry import (
    PRODUCTS,
    registrations,
    select_products,
)
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.models import ResearchConfig


def prepared_inputs(monkeypatch, row_values=()):
    """替换只读输入边界以单测编排；例：equity保留2行，cross保留1行，计数合为3。

    业务构造和消费审计保留真实实现，避免通过模拟审计掩盖输出登记问题。
    """
    config = ResearchConfig()
    context = inputs(row_values, config)
    cross_calls, equity_calls = [], []

    def equity_loader(*args, **kwargs):
        """返回确定性股票输入计数；例：retained_rows=2参与全局剩余预算扣减。"""
        equity_calls.append(kwargs)
        return context, {"retained_rows": 2, "equity_only": 1}

    def cross_loader(*args, **kwargs):
        """返回确定性跨资产输入并记录剩余预算；例：row_values不被复制或截断。"""
        cross_calls.append(kwargs)
        return context, {"retained_rows": 1, "cross_only": 1}

    monkeypatch.setattr(execution, "load_equity_inputs", equity_loader)
    monkeypatch.setattr(adapter, "load_inputs", cross_loader)
    return config, context, cross_calls, equity_calls


@pytest.mark.parametrize("selection", [("P03",), ("P13",), ("P03", "P13")])
def test_dependency_builders_receive_real_parent_results_in_stable_order(
    tmp_path, monkeypatch, selection
):
    """场景：只选旧产品及其依赖；预期：父结果顺序稳定且不扫描无关跨资产来源。"""
    config, context, cross_calls, equity_calls = prepared_inputs(monkeypatch)
    seen = []

    def parent_builder(product):
        """给不同父产品创建独立结果；例：P04返回fundamental标识的来源空降级结果。"""

        def build(current):
            """记录父构造并返回合法信封；例：输入context输出一条P04结果。"""
            seen.append(product)
            return (make_result(current, PRODUCTS[product], IDENTITY),)

        return build

    for product, name in (
        ("P04", "build_fundamental"),
        ("P06", "build_governance"),
        ("P07", "build_flow"),
    ):
        monkeypatch.setattr(execution, name, parent_builder(product))

    def candidates(current, fundamental, governance, flow):
        """核对三个依赖组再构造候选；例：父产品P04、P06、P07依次到达。"""
        assert [group[0].envelope.product_id for group in (fundamental, governance, flow)] == [
            PRODUCTS[key] for key in ("P04", "P06", "P07")
        ]
        return (make_result(current, PRODUCTS["P03"], IDENTITY),)

    def events(current, parents):
        """核对事件收到同一批三个父模型；例：每类父结果各一条且无候选结果混入。"""
        assert [result.envelope.product_id for result in parents] == [
            PRODUCTS[key] for key in ("P04", "P06", "P07")
        ]
        return (make_result(current, PRODUCTS["P13"], IDENTITY),)

    monkeypatch.setattr(execution, "build_candidates", candidates)
    monkeypatch.setattr(execution, "build_events", events)
    state, progress = execution.ExecutionState(), []
    products = select_products(selection)
    results = tuple(
        execution.execute(
            tmp_path / "reference",
            NOW,
            context.run_id,
            config,
            products,
            (),
            execution.RunLimits(),
            state,
            progress.append,
        )
    )
    assert seen == ["P04", "P06", "P07"]
    assert len(results) == state.result_count == len(products)
    equity_load_count = len(products) + 1
    assert state.processing == {
        "equity_only": equity_load_count,
        "retained_rows": 2 * equity_load_count,
    }
    shared = frozenset().union(
        *(execution.EQUITY_SUPPORT_APIS.get(product, frozenset()) for product in products)
    )
    expected_equity_calls = [shared] + [
        (
            frozenset(
                item.spec.api for item in registrations() if item.spec.product == product
            )
            - shared
        )
        for product in products
    ]
    assert [call["source_apis"] for call in equity_calls] == expected_equity_calls
    assert cross_calls == []
    assert [event["product"] for event in progress] == list(products)
    assert all(event["stage"] == "product_complete" for event in progress)


def test_empty_builder_still_returns_explicit_degraded_result(tmp_path, monkeypatch):
    """场景：新产品构造器确实执行但返回空；预期：一条empty_input结果且执行账本有记录。"""
    config, context, _, _ = prepared_inputs(monkeypatch)
    monkeypatch.setattr(execution, "resolve_handler", lambda name: lambda current: ())
    state = execution.ExecutionState()
    results = tuple(
        execution.execute(
            tmp_path / "reference",
            NOW,
            context.run_id,
            config,
            ("P05",),
            (),
            execution.RunLimits(),
            state,
        )
    )
    assert len(results) == 1
    assert results[0].details["reason"] == "empty_input"
    assert "empty_input" in results[0].envelope.gate_codes
    assert state.consumption["executed"] == 7 and state.consumption["observed"] == 0


def test_cross_sources_are_loaded_once_per_selected_product(tmp_path, monkeypatch):
    """场景：同时运行两个新产品；预期：各自只扫描所属接口且沿用同一全局行数预算。"""
    config, context, received, _ = prepared_inputs(monkeypatch)

    def resolve(name):
        """按注册处理器反查产品并构造结果；例：themes处理器返回P02产品标识。"""
        product = next(
            product for product, handler in execution.HANDLERS.items() if handler == name
        )

        def builder(current):
            """返回当前处理器的一条确定性结果；例：P05处理器得到预期产品标识。"""
            return (make_result(current, PRODUCTS[product], IDENTITY),)

        return builder

    monkeypatch.setattr(execution, "resolve_handler", resolve)
    state = execution.ExecutionState()
    tuple(
        execution.execute(
            tmp_path / "reference",
            NOW,
            context.run_id,
            config,
            ("P02", "P05"),
            (),
            execution.RunLimits(),
            state,
        )
    )
    expected = {
        product: frozenset(
            item.spec.api for item in registrations() if item.spec.product == product
        )
        for product in ("P02", "P05")
    }
    assert [call["source_apis"] for call in received] == [expected["P02"], expected["P05"]]
    assert [call["remaining_rows"] for call in received] == [
        config.maximum_total_rows - 2,
        config.maximum_total_rows - 3,
    ]
    assert state.processing == {"cross_only": 2, "equity_only": 1, "retained_rows": 4}


def test_execution_rejects_accumulated_rows_over_persisted_limit(tmp_path, monkeypatch):
    """场景：共享来源保留行超过配置总预算；预期：构造产品前立即拒绝，不能继续消费。"""
    config, context, _, _ = prepared_inputs(monkeypatch)
    constrained = replace(config, maximum_total_rows=1)
    with pytest.raises(ValueError, match="source_capacity_exceeded"):
        tuple(
            execution.execute(
                tmp_path / "reference",
                NOW,
                context.run_id,
                constrained,
                ("P01",),
                (),
                execution.RunLimits(),
                execution.ExecutionState(),
            )
        )


@pytest.mark.parametrize(
    "mode", ["holdings", "source_symbol", "missing_symbol", "missing_period", "unresolved"]
)
def test_portfolio_receives_only_relevant_parent_results(tmp_path, monkeypatch, mode):
    """场景：显式持仓或基金报告证券命中父结果；预期：只保留相关实体，不复制无关结果。"""
    from tushare_entitlement_hub.cross_asset_research import portfolio as portfolio_module

    original_row = source(api="fund_portfolio")
    row = replace(
        original_row,
        raw=original_row.raw | {"symbol": "000001.SZ"},
        availability=replace(original_row.availability, report_period="2026-06-30"),
    )
    if mode == "missing_symbol":
        row = replace(row, raw=row.raw | {"symbol": None})
    elif mode == "missing_period":
        row = replace(row, availability=replace(row.availability, report_period=None))
    resolutions = []

    def resolve_report_holding(current, symbol, period):
        """记录报告代码及报告期后返回显式实体；例：000001.SZ在报告日解析到IDENTITY。"""
        resolutions.append((symbol, period))
        return None if mode == "unresolved" else IDENTITY

    monkeypatch.setattr(portfolio_module, "holding_entity", resolve_report_holding)
    config, context, _, _ = prepared_inputs(
        monkeypatch, (row,) if mode != "holdings" else ()
    )
    captured = []

    def theme(current):
        """返回相关和无关两个实体；例：只有IDENTITY应进入后续组合父集合。"""
        return tuple(
            make_result(current, PRODUCTS["P02"], entity)
            for entity in (IDENTITY, "cn:security:B:listed")
        )

    def portfolio(current, *, parents, holdings):
        """捕获组合依赖后返回组合实体结果；例：收到一个父模型且显式权重不变。"""
        captured.append((parents, holdings))
        return (make_result(current, PRODUCTS["P12"], "cn:portfolio:synthetic:explicit"),)

    monkeypatch.setattr(
        execution,
        "resolve_handler",
        lambda name: portfolio if name.endswith("build_portfolio") else theme,
    )
    holdings = ({"entity_id": IDENTITY, "weight": 0.5},) if mode == "holdings" else ()
    state = execution.ExecutionState()
    result = tuple(
        execution.execute(
            tmp_path / "reference",
            NOW,
            context.run_id,
            config,
            ("P02", "P12"),
            holdings,
            execution.RunLimits(),
            state,
        )
    )
    expected_parent_count = int(mode in {"holdings", "source_symbol"})
    assert len(result) == 3 and len(captured[0][0]) == expected_parent_count
    if expected_parent_count:
        assert captured[0][0][0].envelope.entity_id == IDENTITY
    assert captured[0][1] == holdings
    assert resolutions == (
        [("000001.SZ", "2026-06-30")] if mode in {"source_symbol", "unresolved"} else []
    )


def test_parent_budget_rejects_overflow_before_portfolio_invocation(tmp_path, monkeypatch):
    """场景：同持仓实体有两条父结果而容量1；预期：直接拒绝，不把部分父集合交给组合。"""
    config, context, _, _ = prepared_inputs(monkeypatch)

    def builder(current):
        """返回同实体两条不同profile研究；例：父集合超1时必须整体拒绝。"""
        return tuple(
            make_result(current, PRODUCTS["P02"], IDENTITY, profile=profile)
            for profile in ("first", "second")
        )

    monkeypatch.setattr(execution, "resolve_handler", lambda name: builder)
    with pytest.raises(ValueError, match="portfolio_parent_capacity_exceeded"):
        tuple(
            execution.execute(
                tmp_path / "reference",
                NOW,
                context.run_id,
                config,
                ("P02", "P12"),
                ({"entity_id": IDENTITY, "weight": 1.0},),
                execution.RunLimits(maximum_parent_results=1),
                execution.ExecutionState(),
            )
        )
