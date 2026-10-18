"""候选诊断的时间方向、成交缺口、同组对照与分数并列测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.signal_diagnostics import (
    event_summary,
    forward_open_returns,
    liquidity_buckets,
    matched_daily_effect,
    score_groups,
)


def test_forward_returns_use_next_open_and_explicit_missing():
    """场景：开盘逐日上涨且末端不足；预期从次开盘计收益，不以信号收盘买入。"""
    opening = np.arange(10.0, 18.0)[:, None]
    factors = np.ones_like(opening)
    result = forward_open_returns(opening, factors, opening + 2, opening - 2, opening, 2, 1.0)
    np.testing.assert_allclose(
        result[:5, 0], [13 / 11 - 1, 14 / 12 - 1, 15 / 13 - 1, 16 / 14 - 1, 17 / 15 - 1]
    )
    assert np.isnan(result[5:]).all()
    assert np.isnan(forward_open_returns(opening, factors, opening, opening - 2, opening, 2)).all()
    assert np.isnan(
        forward_open_returns(opening, factors, opening + 2, opening, opening, 2, 1.0)
    ).all()
    assert np.isnan(
        forward_open_returns(opening, factors, opening + 2, opening - 2, opening, 20)
    ).all()


def test_adjustments_and_quote_missing_do_not_become_gains():
    """场景：除权后价格减半、因子翻倍；预期收益零，缺失或零因子拒绝标签。"""
    opening = np.array([[10.0], [10.0], [5.0], [5.0]])
    adjustment = np.array([[1.0], [1.0], [2.0], [2.0]])
    result = forward_open_returns(opening, adjustment, opening * 1.1, opening * 0.9, opening, 1)
    assert result[0, 0] == 0
    adjustment[2] = np.nan
    assert np.isnan(
        forward_open_returns(opening, adjustment, opening * 1.1, opening * 0.9, opening, 1)[0]
    )


@pytest.mark.parametrize("holding,gap", [(0, 0.02), (True, 0.02), (1, -0.1), (1, np.nan)])
def test_forward_invalid_contract(holding, gap):
    """场景：期限或追价非法；预期明确报错。"""
    values = np.ones((3, 2))
    with pytest.raises(ValueError):
        forward_open_returns(values, values, values, values, values, holding, gap)
    with pytest.raises(ValueError):
        forward_open_returns(values, values[:, :1], values, values, values, 1)


def test_buckets_are_causal_and_keep_ties_together():
    """场景：同额/同分和无候选日；预期并列不拆分、空日-1，未来不影响历史。"""
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [8.0, 8.0, 8.0, 8.0, 8.0, 8.0]])
    universe = np.ones(values.shape, dtype=bool)
    np.testing.assert_array_equal(liquidity_buckets(values, universe)[0], [0, 0, 1, 1, 2, 2])
    assert (score_groups(values, universe)[1] == 0).all()
    np.testing.assert_array_equal(
        score_groups(values, universe)[:1], score_groups(values[:1], universe[:1])
    )
    universe[:] = False
    assert (liquidity_buckets(values, universe) == -1).all()
    assert (score_groups(values, universe) == -1).all()
    with pytest.raises(ValueError):
        liquidity_buckets(values, universe.astype(float))
    with pytest.raises(ValueError):
        score_groups(values, universe, 1)


def test_matched_effect_uses_unhit_same_cell_only():
    """场景：行业间收益差异很大；预期只与同行业同流动性未命中比较。"""
    outcome = np.array([[0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9, 0.8], [0.2] * 8])
    universe = np.ones(outcome.shape, dtype=bool)
    hit = np.zeros_like(universe)
    hit[:, [0, 6]] = True
    industry = np.array([[0.0, 0, 0, 0, 0, 0, 1, 1]] * 2)
    buckets = np.zeros(outcome.shape, dtype=np.int8)
    effect, counts = matched_daily_effect(outcome, universe, hit, industry, buckets)
    np.testing.assert_allclose(effect, [0.02, 0])
    np.testing.assert_array_equal(counts, [1, 1])
    hit[:] = True
    assert np.isnan(matched_daily_effect(outcome, universe, hit, industry, buckets)[0]).all()
    hit[:] = False
    assert np.isnan(matched_daily_effect(outcome, universe, hit, industry, buckets)[0]).all()
    hit[:, 6] = True
    industry[:, 6] = 8
    assert np.isnan(matched_daily_effect(outcome, universe, hit, industry, buckets)[0]).all()
    with pytest.raises(ValueError):
        matched_daily_effect(outcome, universe, hit, industry, buckets, 0)


def test_event_summary_preserves_missing_denominator_and_day_weight():
    """场景：候选收益缺失、日期事件数不同；预期覆盖率不隐去缺失并区分两种均值。"""
    outcome = np.array([[0.1, 0.3], [-0.1, np.nan]])
    selected = np.ones(outcome.shape, dtype=bool)
    report = event_summary(outcome, selected)
    assert report["events"] == 4 and report["observed"] == 3
    assert report["coverage"] == 0.75
    assert report["daily_equal_mean"] == pytest.approx(0.05)
    assert report["mean_gross_return"] == pytest.approx(0.1)
    assert report["gross_win_rate"] == pytest.approx(2 / 3)
    assert event_summary(outcome, ~selected)["mean_gross_return"] is None
    with pytest.raises(ValueError):
        event_summary(outcome, selected.astype(float))
