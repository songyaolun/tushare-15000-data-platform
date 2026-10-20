"""原信号选股/择时对照的参数隔离、连续窗口和因果边界测试。"""

from dataclasses import replace

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.registry import control_candidates
from tushare_entitlement_hub.strategy_discovery.selection_timing import (
    REQUIRED_FEATURES,
    SOURCE_SIGNAL,
    _first_confirmation,
    _historical_risk,
    _previous,
    build_strategies,
)


def selection_fixture():
    """构造30日三证券，20/24日原信号，次日上涨且首个信号第二日突破源高。

    输入无；输出(panel,features)，证券0波动最低、证券2换手最低、证券1行业热度适中。
    """
    shape = (30, 3)
    features = {name: np.ones(shape) for name in REQUIRED_FEATURES}
    features["close"][:] = 10
    features["close"][21:26] = np.array([10.5, 11.2, 10.8, 11, 11.2])[:, None]
    features["high"] = features["close"] + 1
    features["high"][20] = 11
    features["high"][24] = 11.5
    features["return1"] = np.where(np.arange(30)[:, None] % 2, 1, -1) * [0.01, 0.02, 0.04]
    features["return20"][:] = [-0.25, 0.05, 0.30]
    features["turnover"][:] = [8, 3, 1]
    features["sector_heat"][:] = [0.4, 0.6, 0.95]
    features["sector_breadth"][:] = 0.7
    features["kdj_j"] = np.broadcast_to(np.arange(30)[:, None], shape).astype(float)
    features["boll_position"][:] = 0.1
    features["boll_position"][20] = [0.1, 0.3, 0.7]
    features["boll_position"][24] = 0.3
    mask = np.zeros(shape, dtype=bool)
    mask[[20, 24]] = True
    panel = Panel(
        [f"day{day:02}" for day in range(30)], list("ABC"), features, {SOURCE_SIGNAL: mask}, {}
    )
    return panel, features


def definitions_by_suffix(panel, features):
    """按省略volume_前缀的名称索引；例返回字典的selection_control为原信号对照。"""
    return {item.name.removeprefix("volume_"): item for item in build_strategies(panel, features)}


def test_seventeen_predeclared_variants_keep_original_risk_policy_and_control():
    """场景：完整特征；预期：17个唯一配置，原控制与control_candidates完全一致。"""
    panel, features = selection_fixture()
    original = control_candidates(panel, features)[0]
    variants = build_strategies(panel, features)
    assert len(variants) == len({variant.name for variant in variants}) == 17
    assert sum(item.family == "volume_timing" for item in variants) == 8
    control = variants[0]
    np.testing.assert_array_equal(control.entry_mask, original.entry_mask)
    np.testing.assert_array_equal(control.scores, original.scores)
    for variant in variants:
        assert replace(variant.policy, name=original.policy.name) == original.policy
        assert variant.policy.take_profit == 0.04
        assert variant.policy.stop_loss == 0.1
        assert variant.policy.maximum_holding == 20
        assert variant.policy.name == variant.name
        assert variant.entry_mask.dtype == bool and variant.entry_mask.shape == (30, 3)
        assert variant.exit_mask is None
        assert not variant.intraday_exits
        assert variant.trailing_stop == 0


def test_original_control_preserves_nan_score_fallbacks():
    """场景：旧评分字段缺失；预期：内部control保持旧NaN回退语义，与原控制逐元素相同。"""
    panel, features = selection_fixture()
    for field in ("sector_heat", "sector_breadth", "kdj_j"):
        features[field][20, 0] = np.nan
    original = control_candidates(panel, features)[0]
    actual = build_strategies(panel, features)[0]
    np.testing.assert_array_equal(actual.scores, original.scores)
    assert actual.scores[20, 0] == -0.5


def test_selection_rankings_and_single_filters_have_declared_meaning():
    """场景：三证券的波动、换手、热度及涨跌幅不同；预期：各单项对照改变对应排序/筛选。"""
    panel, features = selection_fixture()
    definitions = definitions_by_suffix(panel, features)
    for suffix, winner in (
        ("low_vol20", 0),
        ("low_turnover20", 2),
        ("sector_moderate", 1),
        ("risk_adjusted", 0),
    ):
        strategy = definitions["selection_" + suffix]
        assert strategy.entry_mask[20].all()
        assert np.argmax(strategy.scores[20]) == winner
    for suffix, expected in (
        ("heat_cap", [True, True, False]),
        ("no_runup", [True, True, False]),
        ("no_collapse", [False, True, True]),
        ("boll_lower", [True, True, False]),
    ):
        strategy = definitions["selection_" + suffix]
        np.testing.assert_array_equal(strategy.entry_mask[20], expected)
        np.testing.assert_array_equal(strategy.scores, definitions["selection_control"].scores)


def test_timing_uses_confirmation_close_and_unchanged_current_day_ranking():
    """场景：源后次日上涨、第二日突破；预期：在对应收盘才发信号，未提前到源日。"""
    panel, features = selection_fixture()
    definitions = definitions_by_suffix(panel, features)
    for suffix in ("delay1", "up_within3"):
        assert definitions["timing_" + suffix].entry_mask[21].all()
        assert not definitions["timing_" + suffix].entry_mask[20].any()
    assert definitions["timing_high_within3"].entry_mask[22].all()
    assert not definitions["timing_high_within3"].entry_mask[21].any()
    for number, gap in ((0, 0.0), (1, 0.01), (2, 0.02)):
        strategy = definitions[f"timing_gap_{number}pct"]
        assert strategy.maximum_entry_gap == gap
        np.testing.assert_array_equal(strategy.entry_mask, panel.signals[SOURCE_SIGNAL])
    for name, strategy in definitions.items():
        if name.startswith("timing_"):
            np.testing.assert_array_equal(strategy.scores, definitions["selection_control"].scores)
    assert definitions["timing_kdj_rising"].entry_mask[20].all()
    np.testing.assert_array_equal(
        definitions["timing_boll_reclaim"].entry_mask[20], [False, True, True]
    )


def test_historical_risk_has_full_market_windows_and_zero_volatility_is_finite():
    """场景：20日收益/换手窗口及零波动；预期：波动等21日有效报价，风险分使用.005下限。"""
    panel, features = selection_fixture()
    volatility, turnover = _historical_risk(features)
    assert np.isnan(volatility[:20]).all()
    np.testing.assert_allclose(volatility[20], [0.01, 0.02, 0.04])
    assert np.isnan(turnover[:19]).all()
    np.testing.assert_allclose(turnover[19], [8, 3, 1])
    features["return1"][:] = 0
    strategy = definitions_by_suffix(panel, features)["selection_risk_adjusted"]
    assert strategy.entry_mask[20].all()
    np.testing.assert_allclose(strategy.scores[20], (features["sector_heat"][20] + 0.6) / 0.005)


@pytest.mark.parametrize(("field", "bad"), [("valid", 0), ("close", np.nan)])
def test_suspension_breaks_twenty_day_return_comparability(field, bad):
    """场景：20日收益观察跨度中有停牌/缺价；预期：压缩观察收益不能用于波动或20日涨跌过滤。"""
    panel, features = selection_fixture()
    features[field][5, 0] = bad
    panel.signals[SOURCE_SIGNAL][:] = True
    definitions = definitions_by_suffix(panel, features)
    for suffix in ("low_vol20", "risk_adjusted", "no_runup", "no_collapse"):
        assert not definitions["selection_" + suffix].entry_mask[20:26, 0].any()
    assert definitions["selection_no_runup"].entry_mask[26, 0]
    assert definitions["selection_low_vol20"].entry_mask[26, 0]


@pytest.mark.parametrize(
    ("field", "bad"), [("return1", np.nan), ("return1", -1), ("turnover", np.inf), ("turnover", -1)]
)
def test_invalid_history_is_not_zero_filled(field, bad):
    """场景：收益/换手窗口含未知或非法值；预期：依赖该历史的排序无法入场。"""
    panel, features = selection_fixture()
    features[field][10, 0] = bad
    definitions = definitions_by_suffix(panel, features)
    suffix = "low_vol20" if field == "return1" else "low_turnover20"
    assert not definitions["selection_" + suffix].entry_mask[20, 0]


@pytest.mark.parametrize(
    ("suffix", "field"),
    [
        ("sector_moderate", "sector_heat"),
        ("heat_cap", "sector_heat"),
        ("no_runup", "return20"),
        ("no_collapse", "return20"),
        ("boll_lower", "boll_position"),
    ],
)
def test_unknown_selection_field_fails_its_filter(suffix, field):
    """场景：所需选股字段NaN；预期：不因比较或评分回退变成有效选择信号。"""
    panel, features = selection_fixture()
    features[field][20] = np.nan
    assert not definitions_by_suffix(panel, features)["selection_" + suffix].entry_mask[20].any()


@pytest.mark.parametrize("breakout", [False, True])
@pytest.mark.parametrize("hit_day", [1, 3, 4])
def test_confirmation_window_includes_days_one_to_three_only(breakout, hit_day):
    """场景：源后第1、3或4市场日才满足；预期：只接受前三日首次确认且源日不确认。"""
    close = np.full((6, 1), 10.0)
    high = np.full((6, 1), 10.5)
    close[hit_day:] = 11
    source = np.zeros(close.shape, dtype=bool)
    source[0] = True
    actual = _first_confirmation(source, close, high, np.ones_like(source), breakout=breakout)
    assert not actual[0].any()
    assert int(actual.sum()) == (1 if hit_day <= 3 else 0)
    if hit_day <= 3:
        assert actual[hit_day, 0]


def test_overlapping_sources_do_not_extend_original_confirmation_deadline():
    """场景：等待期间连续源信号，原窗口过期后才突破；预期：不能靠重叠事件延长期限。"""
    close = np.array([[10.0], [9.0], [9.0], [9.0], [11.0], [12.0]])
    source = np.array([[True], [True], [True], [False], [False], [False]])
    valid = np.ones_like(source)
    assert not _first_confirmation(source, close, close + 0.5, valid, breakout=True).any()
    source[4] = True
    result = _first_confirmation(source, close, close + 0.5, valid, breakout=True)
    assert result[5, 0]


@pytest.mark.parametrize("breakout", [False, True])
def test_invalid_intermediate_market_day_cancels_event(breakout):
    """场景：源后首日停牌，随后价格满足；预期：已取消事件不在复牌日恢复。"""
    close = np.array([[10.0], [10.0], [12.0], [13.0]])
    source = np.array([[True], [False], [False], [False]])
    valid = np.array([[True], [False], [True], [True]])
    assert not _first_confirmation(source, close, close + 0.5, valid, breakout=breakout).any()


def test_breakout_requires_valid_source_high_and_up_requires_rising_today():
    """场景：源高缺失或价格仍低于源收盘；预期：不建立突破事件、不误认低位上涨为回收。"""
    close = np.array([[10.0], [9.0], [9.5], [9.9]])
    high = close + 0.5
    high[0] = np.nan
    source = np.array([[True], [False], [False], [False]])
    valid = np.ones_like(source)
    assert not _first_confirmation(source, close, high, valid, breakout=True).any()
    assert not _first_confirmation(source, close, high, valid, breakout=False).any()


def test_delayed_and_same_day_confirmations_refuse_invalid_or_unknown_days():
    """场景：次日无有效报价、前日J/BOLL缺失；预期：延后和技术确认全部拒绝相应信号。"""
    panel, features = selection_fixture()
    features["valid"][21] = 0
    features["kdj_j"][19] = np.nan
    features["boll_position"][19] = np.nan
    definitions = definitions_by_suffix(panel, features)
    for name in ("timing_delay1", "timing_up_within3", "timing_high_within3"):
        assert not definitions[name].entry_mask[21:24].any()
    assert not definitions["timing_kdj_rising"].entry_mask[20].any()
    assert not definitions["timing_boll_reclaim"].entry_mask[20].any()


def test_future_changes_and_prefix_truncation_leave_prior_signals_and_scores_equal():
    """场景：改写22日以后全部特征或截断到22日；预期：此前所有策略信号/分数逐元素相同。"""
    panel, features = selection_fixture()
    original = build_strategies(panel, features)
    altered = {name: values.copy() for name, values in features.items()}
    for values in altered.values():
        values[22:] = 999
    changed_panel = Panel(panel.calendar, panel.codes, altered, panel.signals, {})
    changed = build_strategies(changed_panel, altered)
    prefix_features = {name: values[:22] for name, values in features.items()}
    prefix_panel = Panel(
        panel.calendar[:22],
        panel.codes,
        prefix_features,
        {SOURCE_SIGNAL: panel.signals[SOURCE_SIGNAL][:22]},
        {},
    )
    prefix = build_strategies(prefix_panel, prefix_features)
    for before, after, shortened in zip(original, changed, prefix, strict=True):
        for field in ("entry_mask", "scores"):
            np.testing.assert_array_equal(getattr(before, field)[:22], getattr(after, field)[:22])
            np.testing.assert_array_equal(getattr(before, field)[:22], getattr(shortened, field))


def test_first_and_empty_days_never_wrap_and_inputs_are_not_mutated():
    """场景：首日、空面板及正常构造；预期：滞后不绕到末日、空输出形状保持、输入不变。"""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_previous(values), [[np.nan, np.nan], [1.0, 2.0]])
    assert _previous(values[:0]).shape == (0, 2)
    panel, features = selection_fixture()
    before = {name: value.copy() for name, value in features.items()}
    variants = build_strategies(panel, features)
    assert all(not item.entry_mask[0].any() for item in variants)
    for name, value in features.items():
        np.testing.assert_array_equal(value, before[name])
    empty_features = {name: value[:0, :0] for name, value in features.items()}
    empty_panel = Panel([], [], empty_features, {SOURCE_SIGNAL: np.zeros((0, 0), dtype=bool)}, {})
    assert all(
        item.entry_mask.shape == (0, 0) for item in build_strategies(empty_panel, empty_features)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "feature_missing",
        "feature_shape",
        "signal_missing",
        "signal_shape",
        "control_missing",
        "control_shape",
    ],
)
def test_missing_or_misaligned_contract_is_rejected(mutation):
    """场景：特征、原信号或原评分面板缺失/错轴；预期：明确拒绝，不广播或猜测。"""
    panel, features = selection_fixture()
    panel.values = dict(panel.values)
    if mutation == "feature_missing":
        del features["turnover"]
    elif mutation == "feature_shape":
        features["close"] = np.ones((3, 30))
    elif mutation == "signal_missing":
        del panel.signals[SOURCE_SIGNAL]
    elif mutation == "signal_shape":
        panel.signals[SOURCE_SIGNAL] = np.zeros((3, 30), dtype=bool)
    elif mutation == "control_missing":
        del panel.values["sector_heat"]
    else:
        panel.values["sector_heat"] = np.ones((3, 30))
    with pytest.raises(ValueError, match="selection_timing_"):
        build_strategies(panel, features)
