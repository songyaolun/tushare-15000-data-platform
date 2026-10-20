"""高分排序的并列、缺失、归属与因果前缀测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.evidence_scoring import (
    priority_scores,
    signal_evidence_scores,
)
from tushare_entitlement_hub.strategy_discovery.signal_redesign import top_candidates


def test_priority_preserves_ties_eligibility_and_top_order():
    """场景：并列、缺分、仅单只及空候选日；预期分数不伪造区分，Top排序精确一致。"""
    values = np.array([[1, 2, 2, 4], [3, 3, 3, 3], [1, np.nan, 3, 4], [np.nan] * 4])
    eligible = np.ones(values.shape, dtype=bool)
    eligible[2, 2:] = False
    scores = priority_scores(values, eligible)
    np.testing.assert_array_equal(scores[0], [0, 50, 50, 100])
    np.testing.assert_array_equal(scores[1], [50] * 4)
    assert scores[2, 0] == 50 and np.isnan(scores[2, 1:]).all()
    assert np.isnan(scores[3]).all()
    np.testing.assert_array_equal(
        top_candidates(eligible, scores, 2), top_candidates(eligible, values, 2)
    )
    np.testing.assert_array_equal(priority_scores(values[:2], eligible[:2]), scores[:2])


def test_signal_contributions_global_daily_and_missing():
    """场景：两信号单独及共同命中、状态切换；预期贡献可相加复算且只影响当日。"""
    patterns = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint32)
    weights = np.array([0.02, -0.01])
    average = signal_evidence_scores(patterns, weights)
    np.testing.assert_allclose(average[0], [np.nan, 0.02, -0.01, 0.005])
    np.testing.assert_allclose(
        signal_evidence_scores(patterns, weights, average_hits=False)[0],
        [np.nan, 0.02, -0.01, 0.01],
    )
    daily = np.array([weights, -weights])
    scored = signal_evidence_scores(patterns, daily)
    np.testing.assert_allclose(scored[1], -scored[0])
    np.testing.assert_allclose(signal_evidence_scores(patterns[:1], daily[:1]), scored[:1])


@pytest.mark.parametrize(
    "values,eligible",
    [
        (np.ones(2), np.ones(2, dtype=bool)),
        (np.ones((2, 2)), np.ones((2, 3), dtype=bool)),
        (np.ones((2, 2)), np.ones((2, 2))),
    ],
)
def test_bad_priority_inputs(values, eligible):
    """场景：错维度、错轴或非布尔资格；预期拒绝。"""
    with pytest.raises(ValueError):
        priority_scores(values, eligible)


@pytest.mark.parametrize(
    "patterns,weights",
    [
        (np.ones(2, dtype=np.uint32), np.ones(2)),
        (np.ones((2, 2)), np.ones(2)),
        (np.ones((2, 2), dtype=np.uint32), np.ones((1, 2))),
        (np.ones((2, 2), dtype=np.uint32), np.array([np.nan])),
        (np.array([[4]], dtype=np.uint32), np.ones(2)),
        (np.ones((2, 2), dtype=np.uint32), np.ones(0)),
    ],
)
def test_bad_evidence_inputs(patterns, weights):
    """场景：错轴、未知位及非有限权重；预期拒绝生成评分。"""
    with pytest.raises(ValueError):
        signal_evidence_scores(patterns, weights)
