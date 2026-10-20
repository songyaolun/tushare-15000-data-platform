"""趋势与收缩突破的离线契约、因果性及停牌缺值回归。"""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.trend import (
    REQUIRED_FEATURES,
    _finite_mask,
    _lag_observations,
    build_strategies,
)


@pytest.fixture
def trend_inputs():
    """场景：上升均线中回调、首列第8日突破；预期：两类均能独立触发。

    输入fixture请求，输出12日×3证券的面板与技术特征；旧策略信号为空。
    """
    shape = (12, 3)
    features = {name: np.full(shape, 0.0) for name in REQUIRED_FEATURES}
    average20 = np.broadcast_to(10 + np.arange(12)[:, None] * 0.02, shape).copy()
    features.update(
        {
            "valid": np.ones(shape),
            "close": average20 + 0.2,
            "low": average20.copy(),
            "ma20": average20,
            "ma60": np.full(shape, 9.0),
            "return1": np.full(shape, 0.02),
            "return5": np.full(shape, -0.01),
            "return20": np.full(shape, 0.12),
            "extension_atr": np.full(shape, 0.5),
            "close_location": np.full(shape, 0.85),
            "boll_width": np.full(shape, 0.06),
            "boll_width_reference": np.full(shape, 0.1),
            "previous_high20": average20 + 1,
            "previous_low10": np.full(shape, 8.0),
            "sector_peers": np.full(shape, 6.0),
            "sector_coverage": np.full(shape, 0.9),
            "sector_breadth": np.full(shape, 0.6),
        }
    )
    features["previous_high20"][8, 0] = features["close"][8, 0] - 0.05
    panel = Panel([f"202301{day + 1:02d}" for day in range(12)], ["A", "B", "C"], {}, {}, {})
    return panel, features


def test_configuration_count_policy_and_independent_signals(trend_inputs):
    """场景：旧信号全无；预期：16个唯一配置、每类8个，资金与宽止盈契约一致。"""
    panel, features = trend_inputs
    definitions = build_strategies(panel, features)
    assert len(definitions) == len({item.name for item in definitions}) == 16
    assert sum(item.family == "compression_breakout" for item in definitions) == 8
    assert sum(item.family == "trend_pullback" for item in definitions) == 8
    for item in definitions:
        assert item.entry_mask.dtype == np.bool_
        assert item.entry_mask.shape == item.scores.shape == (12, 3)
        assert item.entry_mask[8, 0]
        assert np.all(np.isfinite(item.scores[item.entry_mask]))
        assert not item.policy.switching
        assert not item.policy.stage_allocation
        assert not item.policy.force_rebalance
        assert item.policy.position_weight == 0.05
        assert item.policy.sector_weight == 0.25
        assert item.policy.take_profit in (0.2, 0.4)
        assert item.policy.stop_loss in (0.07, 0.1)
        assert item.policy.maximum_holding in (30, 60)
        assert item.trailing_stop in (0.08, 0.12)
        assert item.maximum_entry_gap == (0.03 if item.name.startswith("breakout") else 0.02)
    assert not definitions[0].entry_mask[7, 0]


@pytest.mark.parametrize("name", REQUIRED_FEATURES)
def test_missing_or_misaligned_required_features_fail_closed(trend_inputs, name):
    """场景：必需字段缺失或错误形状；预期：输入错误显式拒绝，不广播填充。"""
    panel, features = trend_inputs
    original = features.pop(name)
    with pytest.raises(ValueError, match="invalid_trend_feature_matrix"):
        build_strategies(panel, features)
    features[name] = original[:, 0]
    with pytest.raises(ValueError, match="invalid_trend_feature_matrix"):
        build_strategies(panel, features)


@pytest.mark.parametrize("name", ("sector_peers", "sector_coverage", "sector_breadth"))
@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0))
def test_sector_filter_is_optional_and_unknown_fails_closed(trend_inputs, name, value):
    """场景：行业覆盖/同业/广度未知或不足；预期：只阻止行业过滤版。"""
    panel, features = trend_inputs
    features[name][8, 0] = value
    for item in build_strategies(panel, features):
        assert item.entry_mask[8, 0] == item.name.endswith("_all")


def test_thresholds_have_distinct_economic_effect(trend_inputs):
    """场景：80%带宽及距均线1.3ATR；预期：宽松版入选、严格版拒绝。"""
    panel, features = trend_inputs
    features["boll_width"][7, 0] = 0.08
    features["extension_atr"][8, 1] = 1.3
    features["low"][8, 1] = features["ma20"][8, 1] * 1.02
    for item in build_strategies(panel, features):
        column = 0 if item.family == "compression_breakout" else 1
        assert item.entry_mask[8, column] == ("_broad_" in item.name)


def test_breakout_requires_prior_compression_and_first_cross(trend_inputs):
    """场景：只有当天带宽收缩或此前已突破；预期：不能冒充首次收缩突破。"""
    panel, features = trend_inputs
    features["boll_width"][7, 0] = 0.11
    assert not any(item.entry_mask[8, 0] for item in build_strategies(panel, features)[:8])
    features["boll_width"][7, 0] = 0.06
    features["previous_high20"][7, 0] = features["close"][7, 0] - 0.01
    assert not any(item.entry_mask[8, 0] for item in build_strategies(panel, features)[:8])


def test_trend_requires_pullback_and_rising_average(trend_inputs):
    """场景：此前未回调或MA20没有上行；预期：所有趋势回调配置拒绝。"""
    panel, features = trend_inputs
    features["return5"][7, 0] = 0.01
    features["ma20"][3, 1] = features["ma20"][8, 1]
    assert all(not item.entry_mask[8, :2].any() for item in build_strategies(panel, features)[8:])


def test_future_perturbation_and_prefix_leave_history_identical(trend_inputs):
    """场景：任意扰动未来所有特征及截短面板；预期：此前信号、排序、退出逐值一致。"""
    panel, features = trend_inputs
    original = build_strategies(panel, features)
    prefix = Panel(panel.calendar[:9], panel.codes, {}, {}, {})
    shortened = build_strategies(prefix, {name: value[:9] for name, value in features.items()})
    changed = {name: value.copy() for name, value in features.items()}
    for value in changed.values():
        value[9:] = 999.0
    perturbed = build_strategies(panel, changed)
    for reference, limited, altered in zip(original, shortened, perturbed, strict=True):
        for attribute in ("entry_mask", "scores", "exit_mask"):
            np.testing.assert_array_equal(
                getattr(reference, attribute)[:9], getattr(limited, attribute)
            )
            np.testing.assert_array_equal(
                getattr(reference, attribute)[:9], getattr(altered, attribute)[:9]
            )


def test_lag_skips_suspension_but_retains_valid_technical_unknowns():
    """场景：多证券不同停牌与技术缺值；预期：按列滞后，技术缺值保留，不互相串列。"""
    values = np.array([[1, 10], [np.nan, 20], [3, np.nan], [4, 40]], dtype=float)
    valid = np.array([[1, 1], [0, 1], [1, 1], [1, 1]], dtype=bool)
    np.testing.assert_array_equal(
        _lag_observations(values, valid, 1),
        [[np.nan, np.nan], [np.nan, 10], [1, 20], [3, np.nan]],
    )
    np.testing.assert_array_equal(
        _lag_observations(values, valid, 2),
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, 10], [1, 20]],
    )
    assert _lag_observations(np.empty((0, 0)), np.empty((0, 0), dtype=bool), 1).shape == (0, 0)


def test_suspension_does_not_emit_signals_or_break_resumption(trend_inputs):
    """场景：信号前一日停牌；预期：停牌无信号，复牌从前一有效日恢复指标历史。"""
    panel, features = trend_inputs
    for name, value in features.items():
        value[7, 0] = 0 if name == "valid" else np.nan
    for item in build_strategies(panel, features):
        assert not item.entry_mask[7, 0]
        assert not item.exit_mask[7, 0]
        assert item.entry_mask[8, 0]


@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_unknown_or_invalid_compression_reference_rejects_breakout(trend_inputs, value):
    """场景：前日带宽基准缺失或非正；预期：无除零警告且所有突破配置拒绝。"""
    panel, features = trend_inputs
    features["boll_width_reference"][7, 0] = value
    with np.errstate(all="raise"):
        definitions = build_strategies(panel, features)
    assert all(not item.entry_mask[8, 0] for item in definitions[:8])


def test_finite_mask_and_structural_exits_handle_unknowns(trend_inputs):
    """场景：MA20破坏、十日低点破坏、全部未知；预期：对应退出，缺值不伪造退出。"""
    panel, features = trend_inputs
    features["close"][8] = [9.5, 9.5, np.nan]
    features["previous_low10"][8] = [8, 9.6, np.nan]
    features["ma60"][8] = [9, np.nan, np.nan]
    np.testing.assert_array_equal(_finite_mask(features, ("close",))[8], [True, True, False])
    for item in build_strategies(panel, features):
        expected = [True, True, False] if "_balanced_" in item.name else [False, True, False]
        np.testing.assert_array_equal(item.exit_mask[8], expected)
    features["close"][8, 0] = 8.5
    assert all(item.exit_mask[8, 0] for item in build_strategies(panel, features))


def test_empty_panel_and_all_unknown_history_are_safe(trend_inputs):
    """场景：空面板或无合格历史；预期：仍有16个配置且全部无入场信号。"""
    panel, features = trend_inputs
    for value in features.values():
        value[:] = np.nan
    assert not any(item.entry_mask.any() for item in build_strategies(panel, features))
    empty = Panel([], [], {}, {}, {})
    assert len(build_strategies(empty, {name: np.empty((0, 0)) for name in features})) == 16


def test_inputs_unchanged_and_old_oscillator_does_not_affect_ranking(trend_inputs):
    """场景：额外KDJ/反转信号任意变动；预期：输入不修改，策略和价格排序无变化。"""
    panel, features = trend_inputs
    snapshot = {name: value.copy() for name, value in features.items()}
    original = build_strategies(panel, features)
    features["kdj_j"] = np.full((12, 3), -999.0)
    panel.signals["volume_reversal__sector_group"] = np.zeros((12, 3), dtype=bool)
    altered = build_strategies(panel, features)
    for name, value in snapshot.items():
        np.testing.assert_array_equal(features[name], value)
    for before, after in zip(original, altered, strict=True):
        np.testing.assert_array_equal(before.entry_mask, after.entry_mask)
        np.testing.assert_array_equal(before.scores, after.scores)
