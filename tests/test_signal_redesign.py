"""信号改造保持因果、首次确认和独立退出的离线测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.signal_redesign import (
    confirmation_entries,
    exit_variants,
    low_score_candidate,
    top_candidates,
)


def test_top_candidates_are_stable_and_reject_unknown():
    """场景：同分、缺分和不合格证券；预期稳定选股且不凑满人数。"""
    scores = np.array([[3.0, 3.0, 2.0, np.nan], [1.0, 4.0, 2.0, 3.0]])
    eligible = np.array([[1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool)
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(top_candidates(eligible, scores, 1), expected)
    with pytest.raises(ValueError):
        top_candidates(eligible, scores, 0)


def test_first_retest_and_followthrough_have_distinct_confirmation():
    """场景：先回踩再突破且源日连续重复；预期回踩日或突破日各首次确认一次。"""
    closing = np.array([[10.0], [10.05], [10.3], [10.4]])
    high = np.array([[10.2], [10.1], [10.35], [10.5]])
    low = np.array([[9.8], [9.95], [10.1], [10.2]])
    volume = np.array([[100.0], [70.0], [110.0], [100.0]])
    source = np.array([[1], [1], [0], [0]], dtype=bool)
    for mode, expected in [("retest", [0, 1, 0, 0]), ("followthrough", [0, 0, 1, 0])]:
        result = confirmation_entries(source, closing, high, low, volume, mode)
        np.testing.assert_array_equal(result[:, 0], expected)
        np.testing.assert_array_equal(
            result[:2],
            confirmation_entries(source[:2], closing[:2], high[:2], low[:2], volume[:2], mode),
        )


def test_failed_structure_missing_quote_and_timeout_cancel_event():
    """场景：事件等候中破位、缺报价或过期；预期以后反弹不复活原事件。"""
    closing = np.full((5, 3), 10.0)
    closing[3:] = 11
    high = closing + 0.1
    low = closing - 0.1
    low[1, 0] = 9
    closing[1, 1] = np.nan
    volume = np.ones(closing.shape)
    source = np.zeros(closing.shape, dtype=bool)
    source[0] = True
    assert not confirmation_entries(source, closing, high, low, volume, "followthrough", 2).any()
    with pytest.raises(ValueError):
        confirmation_entries(source, closing, high, low, volume, "unknown")


def test_exit_variants_preserve_entry_and_separate_structural_sell():
    """场景：新增结构卖出条件；预期六种退出共用入场且不把卖出条件变成买入禁令。"""
    mask = np.ones((8, 2), dtype=bool)
    scores = np.ones(mask.shape)
    original = StrategyDefinition("base", "test", mask, scores, maximum_entry_gap=0.02)
    variants = exit_variants(original, mask)
    assert len(variants) == len({item.name for item in variants}) == 6
    assert all(item.entry_mask is mask and item.scores is scores for item in variants)
    assert all(item.maximum_entry_gap == 0.02 for item in variants)
    assert variants[4].exit_mask is mask
    assert not variants[4].entry_invalidation_mask.any()
    assert not variants[3].policy.fixed_price_exits
    assert variants[5].profit_protection is not None
    assert original.exit_mask is None
    with pytest.raises(ValueError):
        exit_variants(original, scores)


def test_low_score_candidate_filters_and_never_reads_future():
    """场景：24只候选中含风控失败和低分；预期只选合格低分20只，配置独立。"""
    scores = np.tile(np.arange(24, dtype=float), (3, 1))
    eligible = np.ones(scores.shape, dtype=bool)
    passed = eligible.copy()
    passed[:, 0] = False
    result = low_score_candidate(eligible, passed, scores)
    assert result.entry_mask[:, 1:21].all()
    assert not result.entry_mask[:, 0].any()
    assert not result.entry_mask[:, 21:].any()
    assert result.policy.take_profit == 0.02 and result.policy.stop_loss == 0.08
    assert result.maximum_pending_sessions == 1 and result.maximum_entry_gap == 0.02
    scores[2] *= -1
    updated = low_score_candidate(eligible, passed, scores)
    np.testing.assert_array_equal(result.entry_mask[:2], updated.entry_mask[:2])
    with pytest.raises(ValueError):
        low_score_candidate(eligible, passed, scores, target=True)
    with pytest.raises(ValueError):
        low_score_candidate(eligible, passed.astype(float), scores)
