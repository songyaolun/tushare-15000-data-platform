"""利润保护声明的八组映射、固定锚选择与完整字段保留测试。"""

from dataclasses import fields, replace

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy
from tushare_entitlement_hub.strategy_discovery import onset_protection as protection
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.profit_protection import ProfitProtection


def protection_fixture():
    """构造已有独立入场否决的行业启动定义；例TP8%/SL8%/H30且两个矩阵互不相同。"""
    return StrategyDefinition(
        name=protection.BASELINE_NAME,
        family="sector_breadth_onset",
        entry_mask=np.array([[True, False], [True, True]]),
        scores=np.array([[0.5, 0.2], [0.6, 0.4]]),
        exit_mask=np.array([[False, True], [False, False]]),
        entry_invalidation_mask=np.array([[True, False], [False, False]]),
        policy=PortfolioPolicy(
            name=protection.BASELINE_NAME,
            switching=False,
            stage_allocation=False,
            force_rebalance=False,
            take_profit=0.08,
            stop_loss=0.08,
            maximum_holding=30,
        ),
        maximum_entry_gap=0.01,
        trailing_stop=0.04,
        intraday_exits=True,
    )


@pytest.mark.parametrize(
    ("label", "activation", "floor", "drawdown"),
    [
        ("floor_arm3_lock0", 0.03, 0.0, None),
        ("floor_arm3_lock1", 0.03, 0.01, None),
        ("floor_arm5_lock0", 0.05, 0.0, None),
        ("floor_arm5_lock2", 0.05, 0.02, None),
        ("trail_arm3_dd2", 0.03, None, 0.02),
        ("trail_arm3_dd3", 0.03, None, 0.03),
        ("trail_arm5_dd2", 0.05, None, 0.02),
        ("trail_arm5_dd3", 0.05, None, 0.03),
    ],
)
def test_eight_explicit_mappings_preserve_every_other_field(label, activation, floor, drawdown):
    """场景：八个激活/地板/回撤映射；预期：精确参数，所有数组原对象与其他政策字段保留。"""
    original = protection_fixture()
    variants = protection.build_variants(original)
    assert len(variants) == len({item.name for item in variants}) == 8
    name = original.name + "__protect_" + label
    selected = next(item for item in variants if item.name == name)
    assert selected.profit_protection == ProfitProtection(
        activation,
        minimum_return=floor,
        trailing_drawdown=drawdown,
    )
    assert selected.policy.name == name
    assert replace(selected.policy, name=original.policy.name) == original.policy
    for field in fields(original):
        if field.name not in {"name", "policy", "profit_protection"}:
            expected, actual = getattr(original, field.name), getattr(selected, field.name)
            if isinstance(expected, np.ndarray):
                assert actual is expected
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
    assert original.profit_protection is None
    assert original.name == protection.BASELINE_NAME


@pytest.mark.parametrize(
    "family", ["baseline_volume", "defensive_rotation", "strong_sector_pullback"]
)
def test_protection_rejects_other_strategy_families(family):
    """场景：试图给非行业初期族增加保护；预期：明确拒绝，不扩大预声明研究范围。"""
    definition = replace(protection_fixture(), family=family)
    with pytest.raises(ValueError, match="onset_protection_requires_unprotected_onset"):
        protection.build_variants(definition)


def test_existing_protection_cannot_be_silently_replaced():
    """场景：输入已配置利润保护；预期：拒绝覆盖已有保护设置。"""
    definition = replace(
        protection_fixture(), profit_protection=ProfitProtection(0.03, minimum_return=0)
    )
    with pytest.raises(ValueError, match="onset_protection_requires_unprotected_onset"):
        protection.build_variants(definition)


def test_generic_onset_preserves_absent_masks_and_nondefault_policy():
    """场景：后续交叉输入不带结构退出/否决且政策不同；预期：仅声明保护，不强制重置为原锚。"""
    original = protection_fixture()
    original = replace(
        original,
        exit_mask=None,
        entry_invalidation_mask=None,
        intraday_exits=False,
        maximum_entry_gap=None,
        trailing_stop=0,
        policy=replace(
            original.policy, take_profit=0.06, stop_loss=0.05, maximum_holding=20, early_exit=3
        ),
    )
    for variant in protection.build_variants(original):
        assert variant.exit_mask is None and variant.entry_invalidation_mask is None
        assert not variant.intraday_exits and variant.maximum_entry_gap is None
        assert variant.trailing_stop == 0
        assert replace(variant.policy, name=original.policy.name) == original.policy


def real_onset_fixture():
    """构造两个行业各六股，行业1在第30日由弱广度跃升；例市场广度.7通过50%过滤。"""
    shape = (45, 12)
    amplitude = np.tile(np.arange(1, 7), 2) * 0.0001
    daily = np.repeat([0.001, 0.002], 6)[None, :] + np.where(
        np.arange(45)[:, None] % 2,
        amplitude,
        -amplitude,
    )
    close = 10 * np.cumprod(1 + daily, axis=0)
    features = {
        "valid": np.ones(shape),
        "close": close,
        "ma20": close * 0.99,
        "return1": daily,
        "return5": np.full(shape, -0.02),
        "return20": np.full(shape, 0.05),
        "turnover": np.broadcast_to(np.tile(np.arange(1, 7), 2), shape).astype(float),
        "sector_index": np.broadcast_to(np.repeat([0, 1], 6), shape).astype(float),
    }
    features["ma20"][:30, 6:10] = close[:30, 6:10] * 1.01
    panel = Panel(
        [f"day{day}" for day in range(45)],
        [f"stock{number}" for number in range(12)],
        features,
        {},
        {"breadth": np.full(45, 0.7)},
    )
    return panel, features


def test_builder_selects_exact_quality_and_management_winner_without_control():
    """场景：真实既有builder链条；预期：八项取指定质量+动量退出锚，候选非空且无重复控制。"""
    panel, features = real_onset_fixture()
    quality = next(
        item
        for item in protection.build_quality_strategies(panel, features)
        if item.name == protection.QUALITY_BASELINE_NAME
    )
    baseline = next(
        item
        for item in protection.build_management_variants(quality, features)
        if item.name == protection.BASELINE_NAME
    )
    variants = protection.build_strategies(panel, features)
    assert len(variants) == 8
    assert baseline.entry_mask.any()
    for variant in variants:
        assert variant.name.startswith(protection.BASELINE_NAME + "__protect_")
        assert variant.profit_protection is not None
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            np.testing.assert_array_equal(getattr(variant, field), getattr(baseline, field))
        assert replace(variant.policy, name=baseline.policy.name) == baseline.policy
        assert (
            variant.policy.take_profit,
            variant.policy.stop_loss,
            variant.policy.maximum_holding,
        ) == (0.08, 0.08, 30)


def test_missing_market_breadth_keeps_original_quality_filter_closed():
    """场景：市场广度缺失；预期：仍生成八项规格，但继承原质量过滤的空候选。"""
    panel, features = real_onset_fixture()
    panel.market = {}
    assert all(
        not variant.entry_mask.any() for variant in protection.build_strategies(panel, features)
    )
