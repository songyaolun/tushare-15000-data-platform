"""连续增量特征的时点、缺失分母、停牌和公司行动契约。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.extended_nonlinear_features import (
    EXTENDED_FEATURE_NAMES,
    extended_nonlinear_features,
    support_spread,
)


def feature_fixture(days=90, securities=6):
    """生成恒定正行情；例90日6股同行、价格10、ATR.2、成交量100、换手2。"""
    return {
        name: np.full((days, securities), value, dtype=float)
        for name, value in {
            "valid": 1,
            "close": 10,
            "raw_adj_factor": 1,
            "turnover": 2,
            "atr": 0.2,
            "raw_vol": 100,
            "previous_high20": 11,
            "sector_index": 0,
        }.items()
    }


def test_constant_values_and_exact_window_boundaries():
    """场景：完整恒定行情；预期两比值为1，动量/价托/量托为0，窗口前未知。"""
    values = extended_nonlinear_features(feature_fixture())
    assert tuple(values) == EXTENDED_FEATURE_NAMES
    starts = {
        "relative_turnover20": 20,
        "atr_expansion60": 60,
        "momentum60": 60,
        "price_support_spread": 19,
        "volume_support_spread": 19,
        "sector_breakout_diffusion5": 25,
    }
    for name, first in starts.items():
        assert np.isnan(values[name][:first]).all()
        expected = 1 if name in ("relative_turnover20", "atr_expansion60") else 0
        np.testing.assert_allclose(values[name][first:], expected, atol=1e-12)


def test_relative_measures_exclude_current_day_from_reference():
    """场景：只把当日换手与ATR翻倍；预期比值2，分母不被当前数值稀释。"""
    features = feature_fixture()
    features["turnover"][70] = 4
    features["atr"][70] = 0.4
    values = extended_nonlinear_features(features)
    np.testing.assert_allclose(values["relative_turnover20"][70], 2)
    np.testing.assert_allclose(values["atr_expansion60"][70], 2)
    features["close"][70] = 11
    values = extended_nonlinear_features(features)
    np.testing.assert_allclose(values["momentum60"][70], 0.1)
    np.testing.assert_allclose(values["atr_expansion60"][70], 2 / 1.1)


def test_support_spread_is_weaker_of_two_relative_gaps():
    """场景：阶梯成交量最近5日20、前5日10、此前5；预期选择较弱均线间距。"""
    values = np.full((25, 1), 5.0)
    values[-10:-5] = 10
    values[-5:] = 20
    assert support_spread(values)[-1, 0] == pytest.approx(min(20 / 15 - 1, 15 / 10 - 1))
    values[10] = np.nan
    assert np.isnan(support_spread(values)[-1, 0])
    assert np.isnan(support_spread(np.zeros((25, 1)))).all()


@pytest.mark.parametrize(
    "field,value", [("valid", 0), ("close", np.nan), ("close", -1), ("raw_adj_factor", 0)]
)
def test_quote_gap_invalidates_complete_market_windows(field, value):
    """场景：第60日无效；预期不能把有效交易根数压缩成完整市场日历史。"""
    features = feature_fixture(130)
    features[field][60] = value
    values = extended_nonlinear_features(features)
    for name in EXTENDED_FEATURE_NAMES:
        assert np.isnan(values[name][60]).all()
    assert np.isnan(values["price_support_spread"][79]).all()
    assert np.isfinite(values["price_support_spread"][80]).all()
    assert np.isnan(values["momentum60"][120]).all()
    assert np.isfinite(values["momentum60"][121]).all()


@pytest.mark.parametrize(
    "field,value,target",
    [
        ("turnover", np.nan, "relative_turnover20"),
        ("turnover", -1, "relative_turnover20"),
        ("atr", np.inf, "atr_expansion60"),
        ("atr", 0, "atr_expansion60"),
        ("raw_vol", 0, "volume_support_spread"),
        ("raw_vol", np.nan, "volume_support_spread"),
    ],
)
def test_field_missingness_is_not_filled_or_shared_across_features(field, value, target):
    """场景：单一源字段无效；预期对应窗口未知，独立的价格动量不受影响。"""
    features = feature_fixture()
    features[field][65] = value
    values = extended_nonlinear_features(features)
    assert np.isnan(values[target][70]).all()
    np.testing.assert_allclose(values["momentum60"][70], 0)
    features["turnover"][:] = 0
    assert np.isnan(extended_nonlinear_features(features)["relative_turnover20"]).all()


def test_adjustment_change_resets_volume_support_but_preserves_adjusted_price():
    """场景：第60日拆股、复权收盘连续且量翻倍；预期19日内禁量托，价格动量仍0。"""
    features = feature_fixture()
    features["raw_adj_factor"][60:] = 2
    features["raw_vol"][60:] = 200
    values = extended_nonlinear_features(features)
    assert np.isnan(values["volume_support_spread"][60:79]).all()
    np.testing.assert_allclose(values["volume_support_spread"][79:], 0, atol=1e-12)
    np.testing.assert_allclose(values["momentum60"][60:], 0)


def test_sector_excludes_self_and_unknown_high_from_denominator():
    """场景：六股中一股突破、另一股高点未知；预期其余股同业广度为1/4，突破股为0。"""
    features = feature_fixture()
    features["close"][70, 0] = 12
    features["previous_high20"][:, 1] = np.nan
    values = extended_nonlinear_features(features)["sector_breakout_diffusion5"]
    assert values[70, 5] == pytest.approx(0.25)
    assert values[70, 0] == 0
    assert values[75, 5] == pytest.approx(-0.25)
    # 进一步减少分母至3个已知同业：恰好60%仍有效，不能把缺高点算未突破。
    features["previous_high20"][:, 2] = np.nan
    assert extended_nonlinear_features(features)["sector_breakout_diffusion5"][
        70, 5
    ] == pytest.approx(1 / 3)
    features["previous_high20"][:, 3] = np.nan
    assert np.isnan(extended_nonlinear_features(features)["sector_breakout_diffusion5"][70, 5])


def test_sector_coverage_threshold_and_continuous_high_history():
    """场景：七股仅3个合格同业或其中一股中断历史；预期60%覆盖与连续21日均强制。"""
    features = feature_fixture(securities=7)
    features["previous_high20"][:, :3] = np.nan
    # 第6股仅3/6合格同业，虽然达到3个仍不足60%。
    assert np.isnan(extended_nonlinear_features(features)["sector_breakout_diffusion5"][70, 6])
    features = feature_fixture(securities=4)
    features["valid"][60, 0] = 0
    assert np.isnan(extended_nonlinear_features(features)["sector_breakout_diffusion5"][80, 3])
    assert np.isfinite(extended_nonlinear_features(features)["sector_breakout_diffusion5"][86, 3])


def test_industry_change_and_unknown_membership_fail_closed():
    """场景：全部股票换行业或行业ID无效；预期换行业后五日不可比较，无效ID全未知。"""
    features = feature_fixture()
    features["sector_index"][70:] = 1
    values = extended_nonlinear_features(features)["sector_breakout_diffusion5"]
    assert np.isnan(values[70:75]).all()
    np.testing.assert_allclose(values[75:], 0)
    for invalid in [-1, np.nan, 0.5]:
        features["sector_index"][:] = invalid
        assert np.isnan(extended_nonlinear_features(features)["sector_breakout_diffusion5"]).all()


def test_prefix_causality_future_mutation_and_input_immutability():
    """场景：历史前缀、未来极值和新行业ID；预期历史六特征完全一致，输入不变。"""
    features = feature_fixture()
    features["close"][60:70, 0] = 12
    snapshots = {name: values.copy() for name, values in features.items()}
    original = extended_nonlinear_features(features)
    prefix = extended_nonlinear_features({name: values[:73] for name, values in features.items()})
    for name in EXTENDED_FEATURE_NAMES:
        np.testing.assert_array_equal(original[name][:73], prefix[name])
    for name, values in features.items():
        np.testing.assert_array_equal(values, snapshots[name])
        values[73:] = 99999
    changed = extended_nonlinear_features(features)
    for name in EXTENDED_FEATURE_NAMES:
        np.testing.assert_array_equal(original[name][:73], changed[name][:73])


@pytest.mark.parametrize(
    "scenario", ["missing", "onedimensional", "empty", "shape", "text", "complex"]
)
def test_invalid_input_contract(scenario):
    """场景：缺字段、非二维/同形、空轴、文本或复数；预期在计算前拒绝。"""
    features = feature_fixture()
    if scenario == "missing":
        del features["atr"]
    elif scenario == "onedimensional":
        features["close"] = np.ones(90)
    elif scenario == "empty":
        features = {name: values[:0] for name, values in features.items()}
    elif scenario == "shape":
        features["atr"] = np.ones((5, 6))
    elif scenario == "text":
        features["atr"] = np.full((90, 6), "a")
    else:
        features["atr"] = np.ones((90, 6), dtype=complex)
    with pytest.raises(ValueError, match="extended_feature_inputs"):
        extended_nonlinear_features(features)
