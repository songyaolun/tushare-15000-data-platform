"""市场日窗口完整性的离线边界、停牌与因果测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.windows import complete_history


def test_window_requires_current_and_all_prior_observations():
    """场景：不同证券在不同日期停牌；预期独立计数，恰好lookback+1日才恢复。"""
    valid = np.array([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1]])
    expected = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 1], [1, 1]], dtype=bool)
    np.testing.assert_array_equal(complete_history(valid.astype(float), 2), expected)
    np.testing.assert_array_equal(complete_history(valid.astype(bool), 2), expected)


@pytest.mark.parametrize("unknown", (np.nan, np.inf, -np.inf, 0, -1, 2))
def test_nonunit_validity_breaks_window_without_becoming_truthy(unknown):
    """场景：数值型有效标记有缺失、无穷或非法值；预期均中断历史，不按真值强转。"""
    valid = np.ones((8, 2))
    valid[3, 0] = unknown
    result = complete_history(valid, 2)
    assert result[2, 0]
    assert not result[3:6, 0].any()
    assert result[6:, 0].all()
    assert result[2:, 1].all()


@pytest.mark.parametrize("lookback", (0, -1, 1.5, True))
def test_invalid_lookback_is_rejected(lookback):
    """场景：回看期为非正数、非整数或布尔值；预期显式拒绝契约错误。"""
    with pytest.raises(ValueError, match="invalid_complete_history_contract"):
        complete_history(np.ones((3, 2), dtype=bool), lookback)


def test_shape_empty_history_and_window_longer_than_panel():
    """场景：一维输入、空日或空证券、历史过短；预期错维拒绝，其余返回同形状全假。"""
    with pytest.raises(ValueError, match="invalid_complete_history_contract"):
        complete_history(np.ones(3, dtype=bool), 1)
    for shape in ((0, 0), (0, 3), (3, 0), (3, 2)):
        result = complete_history(np.ones(shape, dtype=bool), 5)
        assert result.shape == shape
        assert result.dtype == np.bool_
        assert not result.any()


def test_future_changes_prefix_and_inputs_remain_independent():
    """场景：未来标记扰动、历史截短；预期历史结果不变，输入矩阵不被修改。"""
    valid = np.ones((12, 2), dtype=bool)
    valid[3, 0] = False
    before = valid.copy()
    reference = complete_history(valid, 3)
    np.testing.assert_array_equal(valid, before)
    np.testing.assert_array_equal(reference[:8], complete_history(valid[:8], 3))
    valid[8:] = False
    np.testing.assert_array_equal(reference[:8], complete_history(valid, 3)[:8])
