"""广度候选的因果状态、参数和既有规则兼容性。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.breadth_candidate import (
    breadth_low_score_candidate,
)


def test_defaults_rank_risk_missing_and_prefix():
    """场景：广度跨阈值、缺失、未来追加；预期状态不前移且历史信号不变。"""
    eligible = np.ones((3, 12), dtype=bool)
    technical = eligible.copy()
    technical[:, 0] = False
    scores = np.tile(np.arange(12, dtype=float), (3, 1))
    breadth = np.array([0.39, 0.4, np.nan])
    definition, states, policy = breadth_low_score_candidate(eligible, technical, scores, breadth)
    np.testing.assert_array_equal(states, [0, 1, 0])
    np.testing.assert_array_equal(np.flatnonzero(definition.entry_mask[0]), np.arange(1, 11))
    assert policy.allocations == (0.15, 0.9, 0.9, 0.9, 0.9)
    assert policy.stage_allocation and not policy.force_rebalance
    assert definition.policy.take_profit == 0.02
    assert definition.policy.stop_loss == 0.08
    assert definition.policy.maximum_holding == 20
    assert definition.maximum_pending_sessions == 1
    assert definition.maximum_entry_gap == 0.02
    prefix, prefix_states, _ = breadth_low_score_candidate(
        eligible[:2], technical[:2], scores[:2], breadth[:2]
    )
    np.testing.assert_array_equal(prefix.entry_mask, definition.entry_mask[:2])
    np.testing.assert_array_equal(prefix_states, states[:2])


@pytest.mark.parametrize(
    "changes",
    [
        {"breadth": np.ones((2, 1))},
        {"breadth": np.ones(1)},
        {"breadth": np.array([-0.1, 0.5])},
        {"breadth_threshold": 1},
        {"breadth_threshold": True},
        {"weak_allocation": np.nan},
        {"weak_allocation": 0.95},
        {"candidate_count": 0},
        {"eligible": np.ones(2, dtype=bool)},
    ],
)
def test_invalid_inputs_rejected(changes):
    """场景：错轴、非法比例及候选数；预期在生成可交易定义前明确拒绝。"""
    inputs = {
        "eligible": np.ones((2, 3), dtype=bool),
        "technical_pass": np.ones((2, 3), dtype=bool),
        "production_scores": np.ones((2, 3)),
        "breadth": np.array([0.3, 0.5]),
    }
    inputs.update(changes)
    with pytest.raises(ValueError):
        breadth_low_score_candidate(**inputs)
