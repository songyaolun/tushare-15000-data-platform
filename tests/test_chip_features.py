"""筹码特征的离线口径、缺失和因果窗口测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.chip_features import candidate_chip_features


def test_chip_features_values_and_scaling():
    """场景：成本整体缩放；预期三个无量纲特征及明确计算值不变。"""
    costs = np.array([[[8., 10., 12., 10.]], [[8., 10., 14., 8.]]])
    actual = candidate_chip_features(costs, np.array([0, 1]), np.array([0, 0]),
                                     change_sessions=1)
    np.testing.assert_allclose(actual, [[0, .4, np.nan], [.25, .6, .2]])
    np.testing.assert_allclose(actual, candidate_chip_features(
        costs * np.array([2., 3.])[:, None, None], np.array([0, 1]), np.array([0, 0]),
        change_sessions=1))


def test_chip_features_missing_prefix_and_invalid_costs():
    """场景：窗口缺行、负成本和颠倒分位；预期未知且前缀不受未来影响。"""
    costs = np.tile([[[8., 10., 12., 10.]]], (8, 2, 1))
    costs[2, 0] = np.nan
    costs[3, 1, 0] = -1
    costs[4, 1, 2] = 9
    days = np.repeat(np.arange(8), 2)
    securities = np.tile([0, 1], 8)
    actual = candidate_chip_features(costs, days, securities)
    assert np.isnan(actual[:, 2]).all()
    assert np.isnan(actual[[4, 7, 9]]).all()
    np.testing.assert_allclose(actual[:12], candidate_chip_features(
        costs[:6], days[:12], securities[:12]))
    assert np.isfinite(candidate_chip_features(costs[:1], np.array([0]),
                                               np.array([0]))[0, :2]).all()


@pytest.mark.parametrize("costs,days,securities,window", [
    (np.ones((2, 4)), np.array([0]), np.array([0]), 5),
    (np.ones((2, 1, 3)), np.array([0]), np.array([0]), 5),
    (np.ones((2, 1, 4)), np.array([[0]]), np.array([0]), 5),
    (np.ones((2, 1, 4)), np.array([0]), np.array([0, 0]), 5),
    (np.ones((2, 1, 4)), np.array([0.]), np.array([0]), 5),
    (np.ones((2, 1, 4)), np.array([0]), np.array([0.]), 5),
    (np.ones((2, 1, 4)), np.array([0]), np.array([0]), True),
    (np.ones((2, 1, 4)), np.array([0]), np.array([0]), 0),
    (np.ones((2, 1, 4)), np.array([-1]), np.array([0]), 5),
    (np.ones((2, 1, 4)), np.array([2]), np.array([0]), 5),
    (np.ones((2, 1, 4)), np.array([0]), np.array([-1]), 5),
    (np.ones((2, 1, 4)), np.array([0]), np.array([1]), 5),
])
def test_chip_features_reject_invalid_axis(costs, days, securities, window):
    """场景：维度、索引或窗口不符合契约；预期关闭而非隐式截断或补零。"""
    with pytest.raises(ValueError, match="invalid_chip_feature_axis"):
        candidate_chip_features(costs, days, securities, change_sessions=window)
