"""独立行业初期/回调策略的历史归属、窗口、同排名和缺失因果验证。"""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.sector import _sector_features
from tushare_entitlement_hub.strategy_discovery.sector_onset import (
    REQUIRED_FEATURES,
    _previous_industry,
    _stock_statistics,
    build_strategies,
)


def _fixture(days: int = 80) -> tuple[Panel, dict[str, np.ndarray]]:
    """生成两行业各六股，行业1在第30日扩散；输入80输出(80,12)特征与Panel。"""
    shape = days, 12
    amplitudes = np.tile(np.arange(1, 7), 2) * 0.0001
    daily = np.repeat([0.001, 0.002], 6)[None, :] + np.where(
        np.arange(days)[:, None] % 2,
        amplitudes,
        -amplitudes,
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
    return Panel(
        [f"day{day:03}" for day in range(days)],
        [f"stock{column}" for column in range(12)],
        features,
        {},
        {},
    ), features


def test_sixteen_declared_configurations_and_independent_signals() -> None:
    """场景：第30日行业扩散且个股温和回调；预期16项均可触发，不需旧反转或市场状态。"""
    panel, features = _fixture()
    definitions = build_strategies(panel, features)
    assert len(definitions) == len({item.name for item in definitions}) == 16
    assert sum(item.family == "sector_breadth_onset" for item in definitions) == 8
    assert sum(item.family == "strong_sector_pullback" for item in definitions) == 8
    for item in definitions:
        assert item.entry_mask.shape == item.scores.shape == (80, 12)
        assert item.entry_mask.dtype == item.exit_mask.dtype == np.bool_
        assert item.entry_mask[30, 6]
        assert np.isfinite(item.scores[item.entry_mask]).all()
        assert not item.entry_mask[:20].any()
        assert not item.policy.switching and not item.policy.stage_allocation
        assert not item.policy.force_rebalance and not item.intraday_exits
        assert item.policy.allocations == (0.9,) * 5
        assert (item.policy.position_weight, item.policy.sector_weight) == (0.05, 0.25)
        assert (item.policy.take_profit, item.policy.stop_loss, item.policy.maximum_holding) in (
            (0.04, 0.10, 20),
            (0.08, 0.08, 30),
        )
        assert item.maximum_entry_gap is None and item.trailing_stop == item.policy.trailing == 0


def test_onset_is_a_crossing_not_continuously_strong_breadth() -> None:
    """场景：广度在第30日上穿并持续强；预期onset只触发首次日期，不将持续强势称初期。"""
    panel, features = _fixture()
    for item in build_strategies(panel, features):
        if item.family == "sector_breadth_onset":
            assert item.entry_mask[30, 6]
            assert not item.entry_mask[31:, 6:].any()
    features["ma20"][:] = features["close"] * 0.99
    assert not any(
        item.entry_mask.any()
        for item in build_strategies(panel, features)
        if item.family == "sector_breadth_onset"
    )


def test_industry_switch_does_not_invent_breadth_crossing() -> None:
    """场景：股票由弱行业转入昨日已强且仍在扩散的行业；预期按新行业昨日值拒绝伪上穿。"""
    panel, features = _fixture()
    features["ma20"][:] = features["close"] * 0.99
    features["ma20"][:, :4] = features["close"][:, :4] * 1.01
    features["ma20"][:25, 6:9] = features["close"][:25, 6:9] * 1.01
    features["ma20"][25:30, 6:8] = features["close"][25:30, 6:8] * 1.01
    features["sector_index"][30:, 0] = 1
    features["ma20"][30:, 0] = features["close"][30:, 0] * 0.99
    industry = _sector_features(features)
    previous = _previous_industry(industry["breadth"], features["sector_index"])
    assert industry["breadth"][29, 0] == pytest.approx(1 / 3)
    assert previous[30, 0] == pytest.approx(2 / 3)
    assert industry["diffusion"][30, 0] >= 0.15
    assert not any(
        item.entry_mask[30, 0]
        for item in build_strategies(panel, features)
        if item.family == "sector_breadth_onset"
    )


def test_previous_industry_unknown_and_new_identifier_fail_closed() -> None:
    """场景：新行业首现、昨日值缺失或非法行业；预期前值未知，不回填旧行业或未来值。"""
    membership = np.array([[0, 1, 1], [2, 1, np.nan], [2, 1, -1]], dtype=float)
    values = np.array([[0.2, 0.8, 0.8], [0.9, np.nan, 0.1], [0.9, 0.7, 0.1]])
    expected = np.array([[np.nan] * 3, [np.nan, 0.8, np.nan], [0.9, np.nan, np.nan]])
    np.testing.assert_allclose(_previous_industry(values, membership), expected, equal_nan=True)


def test_risk_statistics_and_suspension_use_continuous_market_days() -> None:
    """场景：20收益窗口和一次停牌后巨幅复牌；预期完整21日后恢复，不压缩或吸收复牌累计涨幅。"""
    _, features = _fixture()
    volatility, turnover = _stock_statistics(features)
    assert np.isnan(volatility[:20]).all() and np.isnan(turnover[:20]).all()
    assert volatility[20, 0] == pytest.approx(0.0001)
    assert turnover[20, 0] == 1
    features["valid"][25, 0] = 0
    features["return1"][26, 0] = 0.8
    volatility, turnover = _stock_statistics(features)
    assert np.isnan(volatility[25:46, 0]).all()
    assert volatility[46, 0] == pytest.approx(0.0001)
    assert turnover[46, 0] == 1


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -1.0, -2.0))
def test_invalid_daily_returns_break_full_window(invalid: float) -> None:
    """场景：收益窗口含缺值、无穷或<=-100%；预期波动未知且对应证券所有入场被拒绝。"""
    panel, features = _fixture()
    features["return1"][25, 6] = invalid
    volatility, _ = _stock_statistics(features)
    assert np.isnan(volatility[25:45, 6]).all()
    assert np.isfinite(volatility[45, 6])
    assert all(not item.entry_mask[30, 6] for item in build_strategies(panel, features))


def test_moderate_strength_rank_prefers_five_percent_over_extreme_gain() -> None:
    """场景：5%与9%涨幅同波动、另有40%过热股；预期中等强度优先，过热股所有配置拒绝。"""
    panel, features = _fixture()
    features["return1"][:, 7] = features["return1"][:, 6]
    features["return20"][30, 6:9] = [0.05, 0.09, 0.4]
    for item in build_strategies(panel, features):
        assert not item.entry_mask[30, 8]
        if "reclaim_mid" in item.name:
            assert item.scores[30, 6] > item.scores[30, 7]


def test_profiles_and_confirmations_have_distinct_effects() -> None:
    """场景：12%前期涨幅或-0.5%小跌；预期严格涨幅门拒绝，安静确认可接纳未转涨个股。"""
    panel, features = _fixture()
    features["return20"][30, 6] = 0.12
    for item in build_strategies(panel, features):
        assert item.entry_mask[30, 6] == ("_broad_" in item.name)
    features["return20"][30, 6] = 0.05
    features["return1"][30, 6] = -0.005
    other_columns = np.arange(12) != 6
    features["return1"][:30, other_columns] += np.where(
        np.arange(30)[:, None] % 2,
        0.01,
        -0.01,
    )
    for item in build_strategies(panel, features):
        assert item.entry_mask[30, 6] == ("quiet_risk" in item.name)


@pytest.mark.parametrize("prior_average", (9.0, 10.0, np.nan, np.inf, 0.0, -1.0))
def test_reclaim_requires_previous_close_strictly_below_known_average(prior_average: float) -> None:
    """场景：昨收已在均线上/恰等均线或前均线非法；预期不冒充今日首次收复。"""
    panel, features = _fixture()
    features["close"][29, 6] = 10
    features["ma20"][29, 6] = prior_average
    for item in build_strategies(panel, features):
        if "reclaim_mid" in item.name:
            assert not item.entry_mask[30, 6]


def test_reclaim_accepts_current_close_at_average_and_rejects_persistent_above() -> None:
    """场景：昨收在均线下、今收恰等均线，之后持续在上方；预期只认真实回收日。"""
    panel, features = _fixture()
    features["ma20"][30, 6] = features["close"][30, 6]
    for item in build_strategies(panel, features):
        if "reclaim_mid" in item.name:
            assert item.entry_mask[30, 6]
            assert not item.entry_mask[31:, 6].any()


def test_quiet_confirmation_rejects_unknown_and_high_turnover() -> None:
    """场景：换手20日窗口缺失或高换手；预期quiet拒绝，reclaim不依赖换手继续触发。"""
    panel, features = _fixture()
    for value in (np.nan, np.inf, -1.0, 100.0):
        features["turnover"][:, 6] = value
        for item in build_strategies(panel, features):
            assert item.entry_mask[30, 6] == ("reclaim_mid" in item.name)


def test_unknown_sector_inadequate_coverage_and_missing_inputs_reject_entries() -> None:
    """场景：未知行业、六成员仅两有效、技术缺值；预期不能构造有效群动或回调。"""
    panel, features = _fixture()
    features["valid"][30, 8:] = 0
    assert all(not item.entry_mask[30, 6:].any() for item in build_strategies(panel, features))
    panel, features = _fixture()
    features["sector_index"][:, 6:] = np.nan
    assert all(not item.entry_mask[:, 6:].any() for item in build_strategies(panel, features))
    for name in ("ma20", "return5", "return20"):
        panel, features = _fixture()
        features[name][30, 6] = np.nan
        assert all(not item.entry_mask[30, 6] for item in build_strategies(panel, features))


def test_equal_ranks_preserve_equal_scores_without_artificial_security_preference() -> None:
    """场景：全部股票波动、换手、涨幅相等；预期平均秩得相同分数，不靠证券列伪造强弱。"""
    panel, features = _fixture()
    features["return1"][:] = 0.002
    features["turnover"][:] = 1
    for item in build_strategies(panel, features):
        np.testing.assert_allclose(item.scores[30], np.full(12, 0.5))
        expected = not (item.family == "strong_sector_pullback" and "_selective_" in item.name)
        assert item.entry_mask[30, 6] == expected


def test_sector_deterioration_triggers_exit_without_market_stage() -> None:
    """场景：行业广度降至三分之一；预期所有配置追加退出，不读取任何大盘阶段。"""
    panel, features = _fixture()
    features["ma20"][40:, 6:10] = features["close"][40:, 6:10] * 1.01
    assert all(item.exit_mask[40, 6:].all() for item in build_strategies(panel, features))


def test_future_perturbation_prefix_and_input_immutability() -> None:
    """场景：未来价格和行业ID任意变化、截短历史；预期此前入场、退出、评分逐元素一致。"""
    panel, features = _fixture()
    copied = {name: values.copy() for name, values in features.items()}
    original = build_strategies(panel, features)
    short = Panel(panel.calendar[:41], panel.codes, {}, {}, {})
    prefix = build_strategies(short, {name: values[:41] for name, values in features.items()})
    changed = {name: values.copy() for name, values in features.items()}
    for name, values in changed.items():
        values[41:] = 99 if name == "sector_index" else np.nan
    altered = build_strategies(panel, changed)
    for before, limited, after in zip(original, prefix, altered, strict=True):
        for attribute in ("entry_mask", "exit_mask", "scores"):
            np.testing.assert_allclose(
                getattr(before, attribute)[:41], getattr(limited, attribute), equal_nan=True
            )
            np.testing.assert_allclose(
                getattr(before, attribute)[:41], getattr(after, attribute)[:41], equal_nan=True
            )
    for name, values in copied.items():
        np.testing.assert_array_equal(features[name], values)


@pytest.mark.parametrize("name", REQUIRED_FEATURES)
def test_missing_key_or_shape_is_rejected(name: str) -> None:
    """场景：必需特征缺失或shape错位；预期明确拒绝，不静默广播或填充。"""
    panel, features = _fixture()
    values = features.pop(name)
    with pytest.raises(ValueError, match="sector_onset_features_missing_or_misaligned"):
        build_strategies(panel, features)
    features[name] = values[:, 0]
    with pytest.raises(ValueError, match="sector_onset_features_missing_or_misaligned"):
        build_strategies(panel, features)


def test_empty_or_all_unknown_history_returns_sixteen_safe_definitions() -> None:
    """场景：空日期/空证券/全未知；预期16配置shape保持一致，全部无入场。"""
    for shape in ((0, 0), (0, 3), (30, 0), (30, 3)):
        panel = Panel(
            [str(day) for day in range(shape[0])],
            [str(column) for column in range(shape[1])],
            {},
            {},
            {},
        )
        features = {name: np.full(shape, np.nan) for name in REQUIRED_FEATURES}
        definitions = build_strategies(panel, features)
        assert len(definitions) == 16
        assert all(
            item.entry_mask.shape == shape and not item.entry_mask.any() for item in definitions
        )
