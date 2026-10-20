"""离线验证行业历史归属、停牌、轮动时钟和信号前缀不变性。"""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.sector import (
    REQUIRED_FEATURES,
    _percentile,
    _policy,
    _sector_features,
    build_strategies,
)


def _fixture(days: int = 60) -> tuple[Panel, dict[str, np.ndarray]]:
    """生成四行业各三证券的确定性行情；输入60输出(60,12)完整特征矩阵。"""
    daily_returns = np.linspace(0.001, 0.012, 12)
    close = 10 * (1 + daily_returns[None, :]) ** np.arange(days)[:, None]
    features = {
        "valid": np.ones((days, 12)),
        "close": close,
        "ma20": close * 0.95,
        "return1": np.broadcast_to(daily_returns, close.shape).copy(),
        "return5": np.broadcast_to((1 + daily_returns) ** 5 - 1, close.shape).copy(),
        "return20": np.broadcast_to((1 + daily_returns) ** 20 - 1, close.shape).copy(),
        "sector_index": np.broadcast_to(np.repeat(np.arange(4), 3), close.shape).astype(float),
    }
    panel = Panel(
        [f"day{day:03}" for day in range(days)],
        [f"security{column}" for column in range(12)],
        features,
        {},
        {},
    )
    return panel, features


def test_percentiles_preserve_ties_unknowns_and_singletons() -> None:
    """场景：并列、全缺失、单证券；预期平均秩、不填缺失、不伪造唯一领先者。"""
    values = np.array([[1, 2, 2, np.nan], [np.nan] * 4, [3, np.nan, np.nan, np.nan]])
    expected = np.array([[0, 0.75, 0.75, np.nan], [np.nan] * 4, [0.5, np.nan, np.nan, np.nan]])
    np.testing.assert_allclose(_percentile(values), expected, equal_nan=True)


def test_industry_returns_follow_daily_membership_without_backfill() -> None:
    """场景：第15日两只股票交换行业；预期此前行业收益保留原成员，之后用新成员。"""
    _, features = _fixture()
    features["sector_index"][15:, 0] = 1
    features["sector_index"][15:, 3] = 0
    result = _sector_features(features)
    old_mean = np.mean(features["return1"][0, :3])
    new_mean = np.mean(features["return1"][0, [1, 2, 3]])
    expected = (1 + old_mean) ** 9 * (1 + new_mean) - 1
    assert result["momentum10"][15, 3] == pytest.approx(expected)
    assert result["momentum10"][14, 0] == pytest.approx((1 + old_mean) ** 10 - 1)
    assert np.all(np.isnan(result["momentum10"][:10]))


def test_suspension_resume_and_insufficient_coverage_fail_closed() -> None:
    """场景：三成员行业一只停牌后巨幅复牌；预期停牌和复牌日均不拼造行业动量。"""
    panel, features = _fixture()
    features["valid"][25, 0] = 0
    features["return1"][26, 0] = 0.8
    features["ma20"][30, 1] = np.nan
    result = _sector_features(features)
    assert np.isnan(result["momentum10"][25:36, :3]).all()
    assert np.isfinite(result["momentum10"][36, :3]).all()
    assert np.isnan(result["breadth"][30, :3]).all()
    assert all(not strategy.entry_mask[25, 0] for strategy in build_strategies(panel, features))


def test_coverage_threshold_and_invalid_returns() -> None:
    """场景：单行业12成员仅三只有效或出现-100%收益；预期低于60%覆盖不生成指标。"""
    _, features = _fixture()
    features["sector_index"][:] = 0
    features["valid"][25, 3:] = 0
    result = _sector_features(features)
    assert np.isnan(result["breadth"][25]).all()
    assert np.isnan(result["momentum10"][25]).all()
    features["valid"][:] = 1
    features["return1"][25, :] = -1
    assert np.isnan(_sector_features(features)["momentum10"][25]).all()


def test_configuration_count_policies_and_distinct_economic_entries() -> None:
    """场景：上涨行业广度由三分之一升至全体；预期20配置、两族及三种行业逻辑触发。"""
    panel, features = _fixture()
    features["ma20"][20:25, [9, 10]] = features["close"][20:25, [9, 10]] * 1.05
    strategies = build_strategies(panel, features)
    assert len(strategies) == len({strategy.name for strategy in strategies}) == 20
    assert sum(strategy.family == "sector_momentum" for strategy in strategies) == 12
    for strategy in strategies:
        assert strategy.entry_mask.shape == (60, 12)
        assert strategy.entry_mask.dtype == bool
        assert strategy.policy.allocations == (0.9,) * 5
        assert strategy.policy.position_weight == 0.05
        assert strategy.policy.sector_weight == 0.25
        assert not strategy.policy.switching
        assert not strategy.policy.stage_allocation
        assert not strategy.policy.force_rebalance
        assert np.isfinite(strategy.scores[strategy.entry_mask]).all()
        if strategy.name.startswith("sector_"):
            assert strategy.entry_mask[25:30].any()
    assert _policy("short", 20).stop_loss == 0.07
    assert _policy("long", 40).take_profit == 0.25


def test_rotation_uses_fixed_clock_and_buffered_rank_exits() -> None:
    """场景：连续上涨横截面；预期仅5/10日时钟调仓，中间排名缓冲区既不买也不卖。"""
    panel, features = _fixture()
    rotations = [
        strategy
        for strategy in build_strategies(panel, features)
        if strategy.family == "cross_sectional_rotation"
    ]
    for strategy in rotations:
        interval = 10 if "_d10_" in strategy.name else 5
        outside_clock = np.arange(60) % interval != 0
        assert not strategy.entry_mask[outside_clock].any()
        assert not strategy.exit_mask[outside_clock].any()
        assert strategy.entry_mask[20].any()
        assert strategy.exit_mask[20].any()
        assert (~strategy.entry_mask[20] & ~strategy.exit_mask[20]).any()
        assert not (strategy.entry_mask & strategy.exit_mask).any()


def test_future_changes_and_prefix_truncation_preserve_all_previous_signals() -> None:
    """场景：未来价格、行业和有效性全部改变；预期历史评分、入场和退出逐元素不变。"""
    panel, features = _fixture()
    original = build_strategies(panel, features)
    changed = {name: values.copy() for name, values in features.items()}
    for name in REQUIRED_FEATURES:
        changed[name][35:] = 100 if name == "sector_index" else np.nan
    altered = build_strategies(panel, changed)
    prefix_features = {name: values[:35] for name, values in features.items()}
    prefix_panel = Panel(panel.calendar[:35], panel.codes, prefix_features, {}, {})
    prefix = build_strategies(prefix_panel, prefix_features)
    for initial, modified, truncated in zip(original, altered, prefix, strict=True):
        for attribute in ("entry_mask", "scores", "exit_mask"):
            np.testing.assert_allclose(
                getattr(initial, attribute)[:35], getattr(modified, attribute)[:35], equal_nan=True
            )
            np.testing.assert_allclose(
                getattr(initial, attribute)[:35], getattr(truncated, attribute), equal_nan=True
            )


def test_unknown_industry_empty_history_and_shape_rejection() -> None:
    """场景：未知或非法行业、空历史及错位输入；预期无入场，错位明确拒绝。"""
    panel, features = _fixture()
    features["sector_index"][:] = np.nan
    features["sector_index"][:, 0] = -1
    features["sector_index"][:, 1] = 0.5
    strategies = build_strategies(panel, features)
    assert all(not strategy.entry_mask.any() for strategy in strategies)
    empty_panel, empty_features = _fixture(0)
    assert all(
        strategy.entry_mask.shape == (0, 12)
        for strategy in build_strategies(empty_panel, empty_features)
    )
    features["return5"] = features["return5"][:-1]
    with pytest.raises(ValueError, match="sector_feature_shape_mismatch"):
        build_strategies(panel, features)


def test_sector_exit_when_participation_deteriorates() -> None:
    """场景：行业仅三分之一成员站上均线；预期行业持仓发退出而非借大盘状态决定。"""
    panel, features = _fixture()
    features["ma20"][30:, [9, 10]] = features["close"][30:, [9, 10]] * 1.05
    sectors = [
        strategy
        for strategy in build_strategies(panel, features)
        if strategy.family == "sector_momentum"
    ]
    assert all(strategy.exit_mask[30, 9:12].all() for strategy in sectors)


def test_resumed_stock_waits_for_complete_market_window_despite_healthy_industry() -> None:
    """场景：六成员强行业仍完整，但最强股票长期停牌；预期复牌后21日才恢复排名。"""
    panel, features = _fixture(90)
    features["sector_index"][:] = np.repeat([0, 1], 6)
    features["valid"][30:50, 11] = 0
    features["return20"][50:, 11] = 0.5
    industry = _sector_features(features)
    assert np.isfinite(industry["momentum20"][50, 11])
    assert industry["breadth"][50, 11] == 1
    definitions = build_strategies(panel, features)
    for strategy in definitions:
        assert not strategy.entry_mask[50:70, 11].any()
        assert np.isnan(strategy.scores[50:70, 11]).all()
        assert np.isfinite(strategy.scores[70, 11])
        if strategy.family == "cross_sectional_rotation":
            assert strategy.entry_mask[70, 11]
            assert strategy.exit_mask[50, 11]
    prefix = Panel(panel.calendar[:65], panel.codes, {}, {}, {})
    shortened = build_strategies(prefix, {name: values[:65] for name, values in features.items()})
    for full, short in zip(definitions, shortened, strict=True):
        for name in ("scores", "entry_mask", "exit_mask"):
            np.testing.assert_array_equal(getattr(full, name)[:65], getattr(short, name))
