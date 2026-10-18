"""有限结构退出变体的入场冻结、阈值、保参和因果性测试。"""

from dataclasses import fields, replace

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.refinement import (
    defensive_exit_variants,
    structural_variants,
)


def refinement_fixture(family="boll_reentry"):
    """构造三日两证券锚；例首日首证券entry和exit同时为真，全部变体应禁止该入场。"""
    definition = StrategyDefinition(
        name=f"reversion_{family}_deep_sector",
        family=family,
        entry_mask=np.ones((3, 2), dtype=bool),
        scores=np.arange(6, dtype=float).reshape(3, 2),
        policy=PortfolioPolicy(
            switching=False,
            stage_allocation=False,
            force_rebalance=False,
            take_profit=0.09,
            stop_loss=0.05,
            maximum_holding=12,
        ),
        exit_mask=np.array([[True, False], [False, False], [False, True]]),
        maximum_entry_gap=0.02,
        trailing_stop=0.03,
    )
    features = {
        "valid": np.ones((3, 2)),
        "close": np.full((3, 2), 10.0),
        "ma20": np.full((3, 2), 1.0),
        "boll_position": np.array([[0.64, 0.65], [0.79, 0.80], [0.81, 0.1]]),
        "rsi6": np.array([[64.0, 65.0], [69.0, 70.0], [71.0, 10.0]]),
    }
    return definition, features


@pytest.mark.parametrize("family", ["boll_reentry", "rsi_recovery"])
def test_five_variants_have_fixed_entries_and_expected_exit_semantics(family):
    """场景：BOLL/RSI锚含原入场退出冲突；预期：五变体统一冻结有效入场并精确应用阈值。"""
    definition, features = refinement_fixture(family)
    variants = structural_variants(definition, features)
    assert [variant.name for variant in variants] == [
        definition.name + "__struct_" + label
        for label in ("signal_off", "signal_medium", "signal_high", "early3", "early5")
    ]
    expected_entry = definition.entry_mask & ~definition.exit_mask
    for variant in variants:
        np.testing.assert_array_equal(variant.entry_mask, expected_entry)
        assert variant.entry_mask is not definition.entry_mask
        assert variant.policy.name == variant.name
    assert variants[0].exit_mask is None
    np.testing.assert_array_equal(
        variants[1].exit_mask, [[False, True], [True, True], [True, False]]
    )
    np.testing.assert_array_equal(
        variants[2].exit_mask, [[False, False], [False, True], [True, False]]
    )
    assert variants[3].exit_mask is definition.exit_mask
    assert variants[4].exit_mask is definition.exit_mask
    assert [variant.policy.early_exit for variant in variants] == [0, 0, 0, 3, 5]
    # 即便所有close都高于MA20，新RSI退出也不能通过原均线OR条件。
    assert not variants[1].exit_mask[0, 0]


def test_other_execution_and_policy_fields_are_preserved_by_replace():
    """场景：任意现有/新增契约字段；预期：除命名、入场冻结、退出和early_exit外全部保留。"""
    definition, features = refinement_fixture()
    definition = replace(definition, policy=replace(definition.policy, early_exit=2))
    for variant in structural_variants(definition, features):
        for field in fields(definition):
            if field.name not in {"name", "entry_mask", "exit_mask", "policy"}:
                original = getattr(definition, field.name)
                current = getattr(variant, field.name)
                if isinstance(original, np.ndarray):
                    assert current is original
                else:
                    assert current == original
        for field in fields(definition.policy):
            if field.name not in {"name", "early_exit"}:
                assert getattr(variant.policy, field.name) == getattr(definition.policy, field.name)
    assert [v.policy.early_exit for v in structural_variants(definition, features)][:3] == [2] * 3


def test_absent_original_exit_preserves_all_original_entries():
    """场景：原锚没有退出矩阵；预期：保留原入场，early变体仍没有指标退出。"""
    definition, features = refinement_fixture()
    definition = replace(definition, exit_mask=None)
    variants = structural_variants(definition, features)
    for variant in variants:
        np.testing.assert_array_equal(variant.entry_mask, definition.entry_mask)
    assert variants[3].exit_mask is None and variants[4].exit_mask is None


@pytest.mark.parametrize("family", ["boll_reentry", "rsi_recovery"])
@pytest.mark.parametrize("field", ["valid", "close", "indicator"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_unknown_features_never_generate_new_indicator_exit(family, field, value):
    """场景：所需有效标志、报价或指标未知/无穷；预期：中高阈值退出全部关闭。"""
    definition, features = refinement_fixture(family)
    actual_field = (
        ("boll_position" if family == "boll_reentry" else "rsi6") if field == "indicator" else field
    )
    features[actual_field][:] = value
    variants = structural_variants(definition, features)
    assert not variants[1].exit_mask.any()
    assert not variants[2].exit_mask.any()


@pytest.mark.parametrize("field", ["valid", "close"])
def test_suspended_or_nonpositive_quote_cannot_trigger_new_exit(field):
    """场景：停牌标记或零收盘价；预期：即使指标超过阈值也不生成新退出。"""
    definition, features = refinement_fixture()
    features[field][:] = 0
    variants = structural_variants(definition, features)
    assert not variants[1].exit_mask.any()
    assert not variants[2].exit_mask.any()


def test_future_perturbation_leaves_past_exits_and_frozen_entries_unchanged():
    """场景：改写末日指标与报价；预期：此前所有变体退出、入场、排名保持一致。"""
    definition, features = refinement_fixture()
    original = structural_variants(definition, features)
    altered = {name: values.copy() for name, values in features.items()}
    for values in altered.values():
        values[-1] = np.nan
    changed = structural_variants(definition, altered)
    for old, new in zip(original, changed, strict=True):
        np.testing.assert_array_equal(old.entry_mask, new.entry_mask)
        np.testing.assert_array_equal(old.scores, new.scores)
        if old.exit_mask is not None:
            np.testing.assert_array_equal(old.exit_mask[:-1], new.exit_mask[:-1])


def test_inputs_are_not_mutated_and_variant_entries_do_not_alias():
    """场景：构造后局部修改一个变体入场；预期：原定义、特征及其他变体都保持原值。"""
    definition, features = refinement_fixture()
    before = {name: values.copy() for name, values in features.items()}
    original_entry, original_exit = definition.entry_mask.copy(), definition.exit_mask.copy()
    variants = structural_variants(definition, features)
    variants[0].entry_mask[:] = False
    np.testing.assert_array_equal(definition.entry_mask, original_entry)
    np.testing.assert_array_equal(definition.exit_mask, original_exit)
    assert variants[1].entry_mask.any()
    for name, values in features.items():
        np.testing.assert_array_equal(values, before[name])


@pytest.mark.parametrize("mutation", ["family", "missing", "shape"])
def test_unsupported_or_misaligned_inputs_fail_explicitly(mutation):
    """场景：其他策略族、缺字段或矩阵错位；预期：明确ValueError，不猜测含义或广播。"""
    definition, features = refinement_fixture()
    if mutation == "family":
        definition = replace(definition, family="trend")
    elif mutation == "missing":
        del features["boll_position"]
    else:
        features["boll_position"] = np.ones((2, 3))
    with pytest.raises(ValueError, match="structural_variants_"):
        structural_variants(definition, features)


@pytest.mark.parametrize("take_profit", [0.02, 0.04, 0.06])
@pytest.mark.parametrize("stop_loss", [0.08, 0.12])
@pytest.mark.parametrize("maximum_holding", [20, 40])
def test_defensive_grid_changes_only_three_exit_parameters_and_names(
    take_profit, stop_loss, maximum_holding
):
    """场景：十二个防御退出组合；预期：只改三个退出参数和名称，其余字段及矩阵对象保留。"""
    original, _ = refinement_fixture("defensive_rotation")
    original = replace(
        original,
        intraday_exits=True,
        policy=replace(
            original.policy,
            take_profit=0.95,
            stop_loss=0.08,
            maximum_holding=60,
            early_exit=7,
            trailing=0.04,
        ),
    )
    variants = defensive_exit_variants(original)
    assert len(variants) == len({variant.name for variant in variants}) == 12
    name = f"{original.name}__defexit_tp{take_profit:g}_sl{stop_loss:g}_h{maximum_holding}"
    selected = next(variant for variant in variants if variant.name == name)
    assert selected.policy.name == name
    assert selected.policy.take_profit == take_profit
    assert selected.policy.stop_loss == stop_loss
    assert selected.policy.maximum_holding == maximum_holding
    for field in fields(original):
        if field.name not in {"name", "policy"}:
            expected = getattr(original, field.name)
            actual = getattr(selected, field.name)
            if isinstance(expected, np.ndarray):
                assert actual is expected
                assert actual.shape == expected.shape
            else:
                assert actual == expected
    for field in fields(original.policy):
        if field.name not in {"name", "take_profit", "stop_loss", "maximum_holding"}:
            assert getattr(selected.policy, field.name) == getattr(original.policy, field.name)
    assert original.policy.take_profit == 0.95
    assert original.policy.stop_loss == 0.08
    assert original.policy.maximum_holding == 60
    assert original.name == "reversion_defensive_rotation_deep_sector"


def test_defensive_grid_preserves_absent_exit_and_default_execution():
    """场景：原防御策略没有指标退出且使用默认收盘执行；预期：十二变体保持None和False。"""
    definition, _ = refinement_fixture("defensive_rotation")
    definition = replace(definition, exit_mask=None, trailing_stop=0.0)
    for variant in defensive_exit_variants(definition):
        assert variant.exit_mask is None
        assert variant.intraday_exits is False
        assert variant.trailing_stop == 0.0
        assert variant.entry_mask is definition.entry_mask
        assert variant.scores is definition.scores


@pytest.mark.parametrize("family", ["boll_reentry", "rsi_recovery", "trend"])
def test_defensive_grid_rejects_other_entry_families(family):
    """场景：给防御网格传入回归或趋势策略；预期：明确拒绝而不将退出探索扩大到其他族。"""
    definition, _ = refinement_fixture(family)
    with pytest.raises(ValueError, match="defensive_exit_variants_unsupported_family"):
        defensive_exit_variants(definition)
