"""均线与三次交叉的确定性、缺失和因果窗口测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.event_windows import (
    moving_average,
    recent_event,
    shift_matrix,
    support_formation,
)


def test_shift_does_not_wrap_and_mean_requires_complete_window():
    """场景：缺值、无穷及样本不足；预期只在完整有限窗口产出均线。"""
    values = np.array([[1.0], [3], [np.nan], [7], [9], [np.inf], [2], [4]])
    np.testing.assert_allclose(
        moving_average(values, 2)[:, 0], [np.nan, 2, np.nan, np.nan, 8, np.nan, np.nan, 3]
    )
    np.testing.assert_allclose(shift_matrix(values)[:, 0], [np.nan, 1, 3, np.nan, 7, 9, np.inf, 2])
    assert np.isnan(shift_matrix(values, 20)).all()
    assert np.isnan(moving_average(values, 20)).all()
    assert moving_average(values[:0], 2).shape == (0, 1)


@pytest.mark.parametrize("function", [shift_matrix, moving_average, recent_event])
@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_windows_reject_invalid_window_contract(function, window):
    """场景：窗口不为正整数；预期明确拒绝，不隐式取整或接受布尔值。"""
    with pytest.raises(ValueError):
        function(np.ones((5, 1), dtype=bool), window)


@pytest.mark.parametrize("function", [shift_matrix, moving_average, recent_event])
def test_windows_reject_one_dimensional_input(function):
    """场景：传入一维数组；预期拒绝，以免按证券轴滚动。"""
    with pytest.raises(ValueError):
        function(np.ones(5), 2)


def test_recent_event_clock_expires_without_future_fill():
    """场景：仅第1日有事件，窗口2；预期第1、2日有效，第3日过期。"""
    events = np.array([[False], [True], [False], [False]])
    np.testing.assert_array_equal(recent_event(events, 2)[:, 0], [False, True, True, False])
    assert recent_event(events[:0], 2).shape == (0, 1)
    with pytest.raises(ValueError):
        recent_event(events.astype(float), 2)


def test_support_requires_ordered_crosses_and_only_final_day():
    """场景：短线依次越过中长线，中线最后越过长线；预期仅最后一叉发单。"""
    short = np.array([8, 9.5, 10.5, 11, 12, 12], dtype=float)[:, None]
    middle = np.array([9, 9, 9.5, 9.8, 10.5, 11], dtype=float)[:, None]
    long = np.full((6, 1), 10.0)
    np.testing.assert_array_equal(
        support_formation(short, middle, long)[:, 0], [False, False, False, False, True, False]
    )
    # 短线最后才上穿，不能把顺序倒置解释为同一价托。
    short[:5] = 8
    assert not support_formation(short, middle, long).any()


def test_support_crosses_expire_reset_and_flat_roundoff_is_not_cross():
    """场景：金叉隔太久、均线断档及微小计算误差；预期均不产生有效托。"""
    short = np.array([8, 9.5, 11, *([12] * 12)], dtype=float)[:, None]
    middle = np.array([9, 9, *([9.8] * 11), 10.5, 11], dtype=float)[:, None]
    long = np.full((15, 1), 10.0)
    assert not support_formation(short, middle, long).any()
    short[5] = np.nan
    assert not support_formation(short, middle, long).any()
    short[:] = 10 + 1e-12
    middle[:] = 10 + 5e-13
    assert not support_formation(short, middle, long).any()
    assert support_formation(short[:0], middle[:0], long[:0]).shape == (0, 1)
    with pytest.raises(ValueError):
        support_formation(short, middle[:-1], long)


def test_prefix_of_every_window_matches_full_history():
    """场景：未来行情改变；预期均线、滞后及事件钟在历史前缀逐元素一致。"""
    values = np.arange(40, dtype=float).reshape(20, 2)
    for function in (moving_average, shift_matrix):
        np.testing.assert_array_equal(function(values, 5)[:12], function(values[:12], 5))
    events = values % 7 == 0
    np.testing.assert_array_equal(recent_event(events, 5)[:12], recent_event(events[:12], 5))
