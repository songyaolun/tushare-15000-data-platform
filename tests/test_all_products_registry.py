"""全206接口执行注册及依赖选择的确定性离线验收。"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tushare_entitlement_hub.cross_asset_research import registry
from tushare_entitlement_hub.cross_asset_research.execution import RunLimits
from tushare_entitlement_hub.product_foundation.registry import load_registry


def test_registry_has_exact_unique_matrix_entries_and_callable_handlers():
    """场景：权威消费矩阵；预期：206项唯一、13产品、用途字段逐项一致且处理器可调用。"""
    entries = registry.registrations()
    matrix = {entry.api: entry for entry in load_registry()}
    assert len(entries) == len({entry.spec.api for entry in entries}) == 206
    assert {entry.spec.product for entry in entries} == set(registry.PRODUCTS)
    assert len(registry.PRODUCTS) == 13
    assert (
        Counter(
            entry.spec.product for entry in entries if entry.spec.product in registry.NEW_PRODUCTS
        )
        == registry.EXPECTED_COUNTS
    )
    assert sum(entry.handler.startswith("equity.") for entry in entries) == 99
    assert all(entry.spec == matrix[entry.spec.api] for entry in entries)
    assert all(entry.spec.purpose and entry.spec.selected_fields for entry in entries)
    for entry in entries:
        assert callable(registry.resolve_handler(entry.handler))
    registry.validate_handlers()
    assert registry.fingerprint() == registry.fingerprint()
    assert len(registry.fingerprint()) == 64


@pytest.mark.parametrize("variant", ["counts", "legacy"])
def test_registry_rejects_drift_from_authoritative_scope(monkeypatch, variant):
    """场景：新产品计数或旧99项缺失；预期：固定登记错误，不把范围缩水当成功。"""
    if variant == "counts":
        monkeypatch.setattr(registry, "EXPECTED_COUNTS", {})
    else:
        monkeypatch.setattr(registry, "load_consumption_registry", lambda: ())
    with pytest.raises(ValueError, match="all_products_registry_mismatch"):
        registry.registrations()


@pytest.mark.parametrize("value", ["unknown.fn", "os.system", "equity.unknown.fn"])
def test_handler_resolver_rejects_unregistered_paths(value):
    """场景：未登记或任意模块路径；预期：拒绝解析，不能从输入执行任意模块。"""
    with pytest.raises(ValueError):
        registry.resolve_handler(value)


def test_handler_resolver_rejects_missing_callable(monkeypatch):
    """场景：白名单函数模块存在但函数缺失；预期：明确返回handler_missing。"""
    monkeypatch.setattr(registry, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ValueError, match="all_products_handler_missing"):
        registry.resolve_handler(registry.HANDLERS["P02"])


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (("P01",), ("P01",)),
        (("expectation_revision/v1", "P05"), ("P05",)),
        (("P03",), ("P04", "P06", "P07", "P03")),
        (("P13",), ("P04", "P06", "P07", "P13")),
        (("P03", "P13"), ("P04", "P06", "P07", "P03", "P13")),
    ],
)
def test_product_selection_expands_dependencies_and_deduplicates(selection, expected):
    """场景：别名、重复及依赖产品；预期：稳定有序闭包，依赖只执行一次。"""
    assert registry.select_products(selection) == expected


def test_all_and_portfolio_selection_cover_thirteen_products():
    """场景：空选择或组合诊断；预期：均展开13个真实产品，组合最后执行。"""
    assert registry.select_products(("P12",)) == registry.select_products(())
    assert len(registry.select_products(())) == 13
    assert registry.select_products(())[-1] == "P12"
    with pytest.raises(ValueError, match="unknown_all_products_selection"):
        registry.select_products(("P99",))


def test_fingerprint_changes_when_registered_purpose_changes(monkeypatch):
    """场景：登记用途改变；预期：指纹变化，旧执行身份不可复用。"""
    before = registry.fingerprint()
    entries = registry.registrations()
    changed = (
        replace(entries[0], spec=replace(entries[0].spec, purpose="新研究用途")),
        *entries[1:],
    )
    monkeypatch.setattr(registry, "registrations", lambda: changed)
    assert registry.fingerprint() != before


@pytest.mark.parametrize(
    "field", ["maximum_results", "maximum_parent_results", "maximum_output_bytes"]
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", None])
def test_run_limits_reject_nonpositive_or_noninteger_capacity(field, value):
    """场景：结果或父结果容量不是严格正整数；预期：拒绝，布尔值不冒充容量1。"""
    with pytest.raises(ValueError, match="all_products_capacity_invalid"):
        RunLimits(**{field: value})


def test_run_limits_accept_minimum_and_defaults():
    """场景：最小有效预算及默认预算；预期：精确保留1、100000和20000。"""
    assert RunLimits(1, 1).maximum_results == 1
    assert RunLimits().maximum_results == 100_000
    assert RunLimits().maximum_parent_results == 20_000


def test_handler_field_drift_is_rejected_before_execution(monkeypatch):
    """场景：P02实现漏掉矩阵要求字段；预期：执行前固定错误拒绝，不声称接口已完整消费。"""
    from tushare_entitlement_hub.cross_asset_research import themes

    monkeypatch.setattr(themes, "API_FIELDS", {})
    with pytest.raises(ValueError, match="all_products_handler_fields_mismatch"):
        registry.validate_handlers()
