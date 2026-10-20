"""退出目标与双期限评分的离线契约测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.joint_exit_scoring import (
    EXIT_TARGET_NAMES,
    dual_horizon_priority,
    exit_training_targets,
    within_date_center,
)


def test_targets_keep_all_realized_rows_and_exclude_censored():
    """场景：盈利、亏损、零收益及三类未实现标签；预期仅已平仓进入四目标。"""
    labels = np.array(
        [[-.08, 5, 9, 3], [.1, 2, 2, 3], [0, 2, 20, 3]]
        + [[np.nan, np.nan, np.nan, status] for status in [0, 1, 2]]
    )
    mask, targets = exit_training_targets(labels)
    assert EXIT_TARGET_NAMES == ("net_return", "tail_loss", "holding_sessions", "win_indicator")
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])
    np.testing.assert_allclose(targets, [[-.08, .03, 5, 0], [.1, 0, 2, 1], [0, 0, 2, 0]])
    assert exit_training_targets(np.empty((0, 4)))[1].shape == (0, 4)


@pytest.mark.parametrize(
    "labels",
    [np.zeros(4), np.zeros((2, 3)), np.array([[0, 2, 3, 4]]),
     np.array([[np.nan, 2, 3, 3]]), np.array([[0, 1, 3, 3]]),
     np.array([[0, 2.5, 3, 3]]), np.array([[0, 2, 1, 3]]), np.zeros((1, 4))],
)
def test_invalid_labels_fail_closed(labels):
    """场景：错轴、未知状态、无效持有期或未平仓补零；预期拒绝训练。"""
    with pytest.raises(ValueError):
        exit_training_targets(labels)


def test_dual_horizon_mixture_preserves_fixed_scale():
    """场景：短长目标相反且等权；预期逐项可解释、追加候选不改变已有分数。"""
    short = np.array([.01, -.03])
    long = np.array([.03, -.01])
    utility, scores = dual_horizon_priority(short, long)
    np.testing.assert_allclose(utility, [.02, -.02])
    np.testing.assert_allclose(scores, 50 + 50 * np.tanh(utility / .02))
    assert dual_horizon_priority(short[:1], long[:1])[1][0] == scores[0]
    np.testing.assert_allclose(dual_horizon_priority(short, long, short_weight=1)[0], short)
    np.testing.assert_allclose(dual_horizon_priority(short, long, short_weight=0)[0], long)


@pytest.mark.parametrize(
    ("short", "long", "weight"),
    [(np.zeros((1, 1)), np.zeros((1, 1)), .5), (np.zeros(1), np.zeros(2), .5),
     (np.zeros(1), np.zeros(1), -1), (np.zeros(1), np.zeros(1), np.nan),
     (np.array([np.nan]), np.zeros(1), .5)],
)
def test_invalid_dual_objectives_are_rejected(short, long, weight):
    """场景：错轴、非法权重或缺失预测；预期报错而非产生可排名分数。"""
    with pytest.raises(ValueError):
        dual_horizon_priority(short, long, short_weight=weight)


def test_within_date_center_preserves_axis_and_removes_daily_constants():
    """场景：日期乱序、单候选日和日期共同偏移；预期保持原轴且同日均值归零。"""
    values = np.array([[1., 4.], [8., 3.], [3., 6.]])
    days = np.array([2, 1, 2])
    original = values.copy()
    expected = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    np.testing.assert_array_equal(within_date_center(values, days), expected)
    np.testing.assert_array_equal(values, original)
    offsets = np.array([[20., -10.], [-50., 80.], [20., -10.]])
    np.testing.assert_array_equal(within_date_center(values + offsets, days), expected)
    assert within_date_center(np.empty((3, 0)), days).shape == (3, 0)


@pytest.mark.parametrize(
    ("values", "days"),
    [(np.zeros(2), np.array([0, 1])), (np.zeros((2, 1)), np.array([[0, 1]])),
     (np.zeros((0, 1)), np.array([], dtype=int)), (np.zeros((2, 1)), np.array([0])),
     (np.zeros((2, 1)), np.array([0., 1.])), (np.array([[np.nan]]), np.array([0]))],
)
def test_invalid_within_date_inputs_fail_closed(values, days):
    """场景：错轴、空训练、非整数日期或未知数值；预期拒绝生成相对训练矩阵。"""
    with pytest.raises(ValueError):
        within_date_center(values, days)
