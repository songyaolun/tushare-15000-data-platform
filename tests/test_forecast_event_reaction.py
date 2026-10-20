"""公告事件两列的时间、版本、行业路径和候选轴离线验收。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.earnings_features import (
    candidate_forecast_features,
    forecast_event_features,
)
from tushare_entitlement_hub.strategy_discovery.forecast_event_reaction import (
    FEATURE_NAMES,
    candidate_forecast_event_features,
    industry_relative_log_returns,
)


def disclosure(day="20260109", period="20251231", **changes):
    """生成单证券预告；例默认周五公告净利增长10–20%，周一才可见。"""
    return {
        "ts_code": "000001.SZ",
        "ann_date": day,
        "end_date": period,
        "p_change_min": 10,
        "p_change_max": 20,
        **changes,
    }


def event_inputs():
    """返回完整四交易日单证券轴；例周一相对log收益.02，周二.03。"""
    return (
        ["20260109", "20260112", "20260113", "20260114"],
        ["000001.SZ"],
        np.arange(4),
        np.zeros(4, dtype=int),
        np.array([[np.nan], [0.02], [0.03], [0.04]]),
    )


def test_relative_return_excludes_self_and_uses_previous_industry():
    """自身10%与五同行2%得到精确超额；当日改行业不影响当日，次日才生效。"""
    prices = np.array([[100.0] * 6, [110.0, 102.0, 102.0, 102.0, 102.0, 102.0], [121.0] * 6])
    sectors = np.zeros_like(prices)
    sectors[1:, 0] = 1
    result = industry_relative_log_returns(prices, sectors, np.ones_like(prices))
    assert np.isnan(result[0]).all() and np.isnan(result[2]).all()
    assert result[1, 0] == pytest.approx(np.log(1.1) - np.log(1.02))
    assert result[1, 1] == pytest.approx(np.log(1.02) - np.log(1.036))


def test_missing_price_is_excluded_never_zero_and_scale_invariance():
    """七证券一只缺价后还足五同行；缺价不计0，常数复权尺度不改变收益。"""
    prices = np.array([[100.0] * 7, [110.0, 102.0, 102.0, 102.0, 102.0, 102.0, np.nan]])
    sectors = np.zeros_like(prices)
    result = industry_relative_log_returns(prices, sectors, np.ones_like(prices))
    assert result[1, 0] == pytest.approx(np.log(1.1) - np.log(1.02))
    assert np.isnan(result[1, -1])
    scaled = industry_relative_log_returns(prices * np.arange(1, 8), sectors, np.ones_like(prices))
    np.testing.assert_allclose(scaled, result, atol=1e-15)


def test_missing_industry_invalid_membership_and_nonpositive_price():
    """未知行业、前日无资格或非正价格不参与；全无有效证券返回全NaN。"""
    prices = np.array(
        [[100.0, 100.0, 100.0, 100.0], [110.0, 110.0, 0.0, -1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    sectors = np.array([[np.nan, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = np.ones_like(prices)
    valid[1, :2] = 0
    assert np.isnan(industry_relative_log_returns(prices, sectors, valid)).all()


def test_nonrepresentable_price_ratios_fail_closed_without_infinite_features():
    """极端有限价格的比值上溢／下溢不能输出无穷log收益，也不能假装同行涨幅为0。"""
    prices = np.array([[1e-300, 1e300], [1e300, 1e-300]])
    result = industry_relative_log_returns(
        prices, np.zeros_like(prices), np.ones_like(prices), minimum_peers=1
    )
    assert np.isnan(result).all()


@pytest.mark.parametrize(
    "kind",
    ["shape", "dimension", "dtype", "infinity", "sector", "valid", "peers_zero", "peers_bool"],
)
def test_invalid_price_contract(kind):
    """形状、类型、无穷、非法行业或资格及非法同行门槛均失败关闭。"""
    prices = np.ones((2, 6))
    sectors = np.zeros_like(prices)
    valid = prices.copy()
    peers = 5
    if kind == "shape":
        sectors = sectors[:, :2]
    elif kind == "dimension":
        prices = prices[0]
    elif kind == "dtype":
        prices = prices.astype(str)
    elif kind == "infinity":
        prices[0, 0] = np.inf
    elif kind == "sector":
        sectors[0, 0] = 0.5
    elif kind == "valid":
        valid[0, 0] = 2
    elif kind == "peers_zero":
        peers = 0
    else:
        peers = True
    with pytest.raises(ValueError, match="invalid_event_price_axes"):
        industry_relative_log_returns(prices, sectors, valid, minimum_peers=peers)


def test_friday_weekend_announcement_age_and_duplicate_events():
    """周五与周末公告均周一可见，首日年龄0；同经济版本重复不增加事件。"""
    arguments = event_inputs()
    for announcement in ["20260109", "20260110", "20260111"]:
        row = disclosure(announcement)
        result = candidate_forecast_event_features([row, row], *arguments)
        assert len(result.event_keys) == 1 and result.event_ids.tolist() == [-1, 0, 0, 0]
        assert np.isnan(result.features[0]).all()
        np.testing.assert_allclose(result.features[1:, 0], np.log1p([0, 1, 2]), atol=1e-7)
        np.testing.assert_allclose(result.features[1:, 1], [0.02, 0.05, 0.09], atol=1e-7)
    assert FEATURE_NAMES == ("forecast_age_sessions", "forecast_post_event_excess")


def test_revision_resets_anchor_old_period_cannot_replace_new_period():
    """新公告重置年龄与路径，旧报告期迟修订不能覆盖已选新期，与原3列绑定一致。"""
    rows = [
        disclosure(),
        disclosure("20260112", "20260331", p_change_min=50, p_change_max=60),
        disclosure("20260113", "20251231"),
    ]
    arguments = event_inputs()
    result = candidate_forecast_event_features(rows, *arguments)
    prior = candidate_forecast_features(forecast_event_features(rows), *arguments[:4])
    assert np.array_equal(np.isfinite(result.features[:, 0]), np.isfinite(prior[:, 0]))
    np.testing.assert_allclose(result.features[1:], [[0, 0.02], [0, 0.03], [np.log(2), 0.07]])
    assert result.event_keys[result.event_ids[2]][1] == "2026-03-31"
    assert result.event_ids[2] == result.event_ids[3]


def test_later_first_announcement_conflict_and_unknown_growth_close_both_columns():
    """first_ann只能延后；同日冲突或未知增长不能暴露事件年龄，坏行不变成事件。"""
    arguments = event_inputs()
    result = candidate_forecast_event_features([disclosure(first_ann_date="20260112")], *arguments)
    assert np.isnan(result.features[:2]).all() and result.features[2, 0] == 0
    for rows in [
        [{}, disclosure("bad")],
        [disclosure(), disclosure(p_change_min=11)],
        [disclosure(p_change_min=None)],
    ]:
        result = candidate_forecast_event_features(rows, *arguments)
        assert np.isnan(result.features).all() and (result.event_ids == -1).all()


def test_missing_intermediate_path_keeps_age_until_a_new_announcement():
    """中间缺行业收益让后续累计反应未知，年龄仍可用；新公告可重新形成完整路径。"""
    calendar, codes, days, securities, returns = event_inputs()
    returns[2] = np.nan
    result = candidate_forecast_event_features(
        [disclosure()], calendar, codes, days, securities, returns
    )
    assert np.isnan(result.features[2:, 1]).all() and np.isfinite(result.features[2:, 0]).all()
    reset = candidate_forecast_event_features(
        [disclosure(), disclosure("20260113")], calendar, codes, days, securities, returns
    )
    np.testing.assert_allclose(reset.features[3], [0, 0.04])


def test_expiry_left_truncation_absent_code_and_empty_axis():
    """超过180自然日、无完整公告前日历、无记录证券保持未知，空轴不崩溃。"""
    result = candidate_forecast_event_features(
        [disclosure()],
        ["20260109", "20260709"],
        ["000001.SZ", "000002.SZ"],
        np.array([1, 1]),
        np.array([0, 1]),
        np.zeros((2, 2)),
    )
    assert np.isnan(result.features).all()
    truncated = candidate_forecast_event_features([disclosure("20260108")], *event_inputs())
    assert np.isnan(truncated.features).all()
    empty = candidate_forecast_event_features(
        [], [], [], np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 0))
    )
    assert empty.features.shape == (0, 2) and empty.event_ids.shape == (0,)


def test_prefix_future_prices_and_reordered_duplicate_candidates():
    """截断未来公告价格仍逐值一致；乱序重复候选输出保持原轴，原始输入不修改。"""
    calendar, codes, days, securities, returns = event_inputs()
    rows = [disclosure(), disclosure("20260113", p_change_min=40, p_change_max=60)]
    result = candidate_forecast_event_features(rows, calendar, codes, days, securities, returns)
    prefix = candidate_forecast_event_features(
        rows[:1], calendar[:3], codes, days[:3], securities[:3], returns[:3]
    )
    np.testing.assert_array_equal(prefix.features, result.features[:3])
    order = np.array([3, 0, 2, 2, 1])
    reordered = candidate_forecast_event_features(
        rows[::-1], calendar, codes, days[order], securities[order], returns
    )
    np.testing.assert_array_equal(reordered.features, result.features[order])
    assert returns[2, 0] == 0.03 and len(rows) == 2


@pytest.mark.parametrize(
    "kind",
    [
        "calendar",
        "duplicate_code",
        "negative_day",
        "high_day",
        "negative_security",
        "high_security",
        "shape",
        "dimension",
        "float_day",
        "float_security",
        "return_shape",
        "return_type",
        "infinite_return",
        "age_zero",
        "age_bool",
    ],
)
def test_invalid_candidate_contract(kind):
    """日期／代码重复、索引类型越界、收益轴不符及非法有效期均拒绝计算。"""
    calendar, codes, days, securities, returns = event_inputs()
    age = 180
    if kind == "calendar":
        calendar[1] = calendar[0]
    elif kind == "duplicate_code":
        codes = codes * 2
    elif kind == "negative_day":
        days[0] = -1
    elif kind == "high_day":
        days[0] = 4
    elif kind == "negative_security":
        securities[0] = -1
    elif kind == "high_security":
        securities[0] = 1
    elif kind == "shape":
        securities = securities[:2]
    elif kind == "dimension":
        days = days[:, None]
        securities = securities[:, None]
    elif kind == "float_day":
        days = days.astype(float)
    elif kind == "float_security":
        securities = securities.astype(float)
    elif kind == "return_shape":
        returns = returns[:, :0]
    elif kind == "return_type":
        returns = returns.astype(str)
    elif kind == "infinite_return":
        returns[1, 0] = np.inf
    elif kind == "age_zero":
        age = 0
    else:
        age = True
    with pytest.raises(ValueError, match="invalid_event_candidate_axis"):
        candidate_forecast_event_features(
            [], calendar, codes, days, securities, returns, maximum_age_days=age
        )
