"""超跌回归候选的独立入场、风险配置、缺失保护和时序因果测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.reversion import (
    REQUIRED_FEATURES,
    _cross_section_percentile,
    _previous,
    build_strategies,
)


def reversion_fixture(warmup_days=0):
    """构造三日五证券，可前置有效历史：首日超跌，次日收回，末日回归均线。

    输入warmup_days=5，输出8日面板；倒数第二日证券0五日收益-12%，截面最弱。
    """
    shape = (3, 5)
    features = {name: np.ones(shape) for name in REQUIRED_FEATURES}
    features.update(
        close=np.array([[8.0] * 5, [8.1] * 5, [10.2] * 5]),
        ma20=np.full(shape, 10.0),
        rsi6=np.array([[15.0] * 5, [32.0] * 5, [65.0] * 5]),
        rsi14=np.full(shape, 30.0),
        boll_position=np.array([[-0.2] * 5, [0.1] * 5, [0.6] * 5]),
        return1=np.array([[-0.03] * 5, [0.01] * 5, [0.04] * 5]),
        return5=np.array([[-0.12] * 5, [-0.12, -0.10, -0.06, -0.02, 0.01], [0.05] * 5]),
        sector_breadth=np.full(shape, 0.6),
        sector_peers=np.full(shape, 10.0),
    )
    features = {
        name: np.concatenate((np.repeat(value[:1], warmup_days, axis=0), value))
        for name, value in features.items()
    }
    calendar = [f"202412{day + 1:02}" for day in range(warmup_days)]
    panel = Panel(calendar + ["20250102", "20250103", "20250106"], list("ABCDE"), {}, {}, {})
    return panel, features


def test_grid_contains_twenty_independent_reasonable_policies():
    """场景：完整特征且无原策略信号/成交量；预期：二十组均有独立入场及盈亏约束。"""
    panel, features = reversion_fixture(5)
    strategies = build_strategies(panel, features)
    assert len(strategies) == len({strategy.name for strategy in strategies}) == 20
    assert len({strategy.family for strategy in strategies}) == 5
    for strategy in strategies:
        assert strategy.name.startswith("reversion_")
        assert strategy.entry_mask.shape == (8, 5)
        assert strategy.entry_mask.dtype == bool
        assert strategy.entry_mask.any()
        assert np.isfinite(strategy.scores[strategy.entry_mask]).all()
        assert strategy.exit_mask[-1].all()
        assert not strategy.exit_mask[:-1].any()
        policy = strategy.policy
        assert not policy.switching and not policy.stage_allocation and not policy.force_rebalance
        assert policy.allocations == (0.9,) * 5
        assert policy.position_weight == 0.05 and policy.sector_weight == 0.25
        assert policy.take_profit > policy.stop_loss
        assert policy.stop_loss <= 0.05
        assert policy.maximum_holding in (5, 8, 12)
        assert strategy.maximum_entry_gap == 0.02
        assert strategy.trailing_stop == 0
        expected_day = 5 if strategy.family in ("rsi_low", "boll_deviation") else 6
        assert strategy.entry_mask[expected_day, 0]
        assert not strategy.entry_mask[-1].any()


def test_sector_filter_is_paired_and_does_not_gate_unfiltered_signals():
    """场景：行业数据未知；预期：十个无行业候选保持原信号，行业候选全部关闭。"""
    panel, features = reversion_fixture()
    original = build_strategies(panel, features)
    for field in ("sector_breadth", "sector_coverage", "sector_peers"):
        features[field][:] = np.nan
    changed = build_strategies(panel, features)
    for baseline, strategy in zip(original, changed, strict=True):
        if strategy.name.endswith("_sector"):
            assert not strategy.entry_mask.any()
        else:
            np.testing.assert_array_equal(strategy.entry_mask, baseline.entry_mask)
        np.testing.assert_array_equal(strategy.scores, baseline.scores)


def test_profiles_change_oversold_thresholds_and_recovery_events():
    """场景：温和超跌及RSI从23修复到28；预期：宽档接温和偏离，深档捕捉25收回。"""
    panel, features = reversion_fixture(5)
    features["rsi6"][5] = 23
    features["rsi6"][6] = 28
    features["rsi14"][:] = 38
    features["boll_position"][5] = -0.05
    features["return5"][5] = -0.05
    features["return5"][6] = [-0.04, -0.03, -0.02, -0.01, 0]
    strategies = {strategy.name: strategy for strategy in build_strategies(panel, features)}
    for family, day in (
        ("rsi_low", 5),
        ("boll_deviation", 5),
        ("boll_reentry", 6),
        ("cross_section", 6),
    ):
        assert strategies[f"reversion_{family}_balanced_none"].entry_mask[day, 0]
        assert not strategies[f"reversion_{family}_deep_none"].entry_mask[day, 0]
    assert strategies["reversion_rsi_recovery_deep_none"].entry_mask[6, 0]
    assert not strategies["reversion_rsi_recovery_balanced_none"].entry_mask[6, 0]


@pytest.mark.parametrize("field", ["sector_breadth", "sector_coverage", "sector_peers"])
def test_sector_filter_requires_each_evidence_threshold(field):
    """场景：行业广度/覆盖/同业数量任一不足；预期：行业版本没有信号。"""
    panel, features = reversion_fixture()
    features[field][:] = 0
    assert all(
        not strategy.entry_mask.any()
        for strategy in build_strategies(panel, features)
        if strategy.name.endswith("_sector")
    )


def test_cross_section_uses_current_peers_with_equal_rank_for_ties():
    """场景：五个不同值、全同值、有效数不足；预期：中点分位、同值.5、缺失排名。"""
    values = np.array([[1, 2, 3, 4, 5, np.nan], [2] * 6, [1, 2, 3, 4, np.inf, 6]])
    valid = np.ones(values.shape, dtype=bool)
    valid[2, 5] = False
    result = _cross_section_percentile(values, valid)
    np.testing.assert_allclose(result[0, :5], [0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.isnan(result[0, 5])
    np.testing.assert_allclose(result[1], 0.5)
    assert np.isnan(result[2]).all()
    tied = _cross_section_percentile(np.array([[1.0, 1.0, 2.0, 3.0, 4.0]]), valid[:1, :5])
    np.testing.assert_allclose(tied, [[0.2, 0.2, 0.5, 0.7, 0.9]])


def test_previous_does_not_wrap_or_cross_security_boundaries():
    """场景：二维矩阵、单日与空矩阵；预期：只滞后日轴，首日未知，形状保持。"""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_previous(values), [[np.nan, np.nan], [1.0, 2.0]])
    assert np.isnan(_previous(values[:1])).all()
    assert _previous(values[:0]).shape == (0, 2)


@pytest.mark.parametrize("field", ["valid", "close"])
@pytest.mark.parametrize("missing", [np.nan, np.inf, 0.0])
def test_invalid_observations_cannot_enter_or_exit(field, missing):
    """场景：报价/有效标志未知、无穷或为零；预期：全部入场和指标退出关闭。"""
    panel, features = reversion_fixture()
    features[field][:] = missing
    for strategy in build_strategies(panel, features):
        assert not strategy.entry_mask.any()
        assert not strategy.exit_mask.any()
        assert np.isnan(strategy.scores).all()


def test_all_unknown_features_produce_no_signals():
    """场景：特征全部NaN；预期：未知不得成为超跌、回归或退出信号。"""
    panel, features = reversion_fixture()
    for values in features.values():
        values[:] = np.nan
    for strategy in build_strategies(panel, features):
        assert not strategy.entry_mask.any()
        assert not strategy.exit_mask.any()


@pytest.mark.parametrize(
    ("family", "field", "day"),
    [
        ("rsi_low", "rsi6", 0),
        ("rsi_low", "rsi14", 0),
        ("rsi_low", "return5", 0),
        ("rsi_recovery", "rsi6", 1),
        ("rsi_recovery", "rsi14", 1),
        ("rsi_recovery", "return1", 1),
        ("boll_deviation", "boll_position", 0),
        ("boll_deviation", "return5", 0),
        ("boll_reentry", "boll_position", 1),
        ("boll_reentry", "return1", 1),
        ("cross_section", "return5", 1),
        ("cross_section", "return1", 1),
    ],
)
def test_missing_required_input_never_fires_family_signal(family, field, day):
    """场景：某策略所需字段缺失；预期：该策略当天全部信号关闭。"""
    panel, features = reversion_fixture(5)
    day += 5
    features[field][day] = np.nan
    for strategy in build_strategies(panel, features):
        if strategy.family == family:
            assert not strategy.entry_mask[day].any()


def test_recovery_requires_known_previous_tradable_observation():
    """场景：前日停牌但遗留技术值；预期：RSI和布林收回都不把遗留值当有效前日。"""
    panel, features = reversion_fixture()
    features["valid"][0] = 0
    for strategy in build_strategies(panel, features):
        if strategy.family in ("rsi_recovery", "boll_reentry"):
            assert not strategy.entry_mask[:2].any()


def test_future_perturbation_and_history_prefix_leave_past_unchanged():
    """场景：改写末日全部字段并截断历史；预期：此前入场、分数、退出完全一致。"""
    panel, features = reversion_fixture()
    original = build_strategies(panel, features)
    changed_features = {name: values.copy() for name, values in features.items()}
    for values in changed_features.values():
        values[2] = np.arange(5) + 1234
    changed = build_strategies(panel, changed_features)
    prefix_panel = Panel(panel.calendar[:2], panel.codes, {}, {}, {})
    prefix = build_strategies(prefix_panel, {name: values[:2] for name, values in features.items()})
    for baseline, mutated, shortened in zip(original, changed, prefix, strict=True):
        for field in ("entry_mask", "scores", "exit_mask"):
            np.testing.assert_array_equal(getattr(baseline, field)[:2], getattr(mutated, field)[:2])
            np.testing.assert_array_equal(getattr(baseline, field)[:2], getattr(shortened, field))


@pytest.mark.parametrize("mutation", ["missing", "shape"])
def test_missing_or_misaligned_feature_contract_is_rejected(mutation):
    """场景：字段缺失或证券/日轴错位；预期：明确拒绝，不使用广播猜测。"""
    panel, features = reversion_fixture()
    if mutation == "missing":
        del features["rsi6"]
    else:
        features["rsi6"] = np.ones((5, 3))
    with pytest.raises(ValueError, match="reversion_features_missing_or_misaligned"):
        build_strategies(panel, features)


def test_empty_panel_keeps_twenty_empty_definitions_and_inputs_unchanged():
    """场景：零日零证券；预期：构造二十个空策略且不修改调用方数组。"""
    panel = Panel([], [], {}, {}, {})
    features = {name: np.empty((0, 0)) for name in REQUIRED_FEATURES}
    assert all(
        strategy.entry_mask.shape == (0, 0) for strategy in build_strategies(panel, features)
    )
    normal_panel, normal_features = reversion_fixture()
    before = {name: values.copy() for name, values in normal_features.items()}
    build_strategies(normal_panel, normal_features)
    for name in normal_features:
        np.testing.assert_array_equal(normal_features[name], before[name])


def test_cross_section_waits_six_market_observations_but_rsi_and_boll_keep_semantics():
    """场景：六股截面最弱股停牌后复牌；预期六日齐备才排名，RSI/BOLL仍按观察窗口。"""
    _, original = reversion_fixture()
    features = {
        name: np.repeat(value[:1, [0, 1, 2, 3, 4, 4]], 24, axis=0)
        for name, value in original.items()
    }
    features["return5"][:] = [-0.12, -0.10, -0.08, -0.06, -0.04, -0.02]
    features["return1"][:] = 0
    features["valid"][7:12, 0] = 0
    panel = Panel([f"day{day:02}" for day in range(24)], list("ABCDEF"), {}, {}, {})
    definitions = build_strategies(panel, features)
    for strategy in definitions:
        if strategy.family == "cross_section":
            assert not strategy.entry_mask[12:17, 0].any()
            assert np.isnan(strategy.scores[12:17, 0]).all()
            assert strategy.entry_mask[17, 0]
            assert strategy.entry_mask[12, 1]
        elif strategy.family in ("rsi_low", "boll_deviation"):
            assert strategy.entry_mask[12, 0]
    prefix = Panel(panel.calendar[:16], panel.codes, {}, {}, {})
    shortened = build_strategies(prefix, {name: value[:16] for name, value in features.items()})
    changed = {name: value.copy() for name, value in features.items()}
    for value in changed.values():
        value[16:] = np.nan
    perturbed = build_strategies(panel, changed)
    for full, short, altered in zip(definitions, shortened, perturbed, strict=True):
        for name in ("scores", "entry_mask", "exit_mask"):
            np.testing.assert_array_equal(getattr(full, name)[:16], getattr(short, name))
            np.testing.assert_array_equal(getattr(full, name)[:16], getattr(altered, name)[:16])
