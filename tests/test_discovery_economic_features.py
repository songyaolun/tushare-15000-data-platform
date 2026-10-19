"""固定经济特征的含义、轴一致性和前缀因果性。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.economic_features import (
    FEATURE_BUCKETS,
    economic_features,
)


def test_named_features_and_interactions_are_causal():
    """场景：信号相同但市场状态和价格位置不同；预期交互按当前事实产生且名称唯一。"""
    raw = {name: np.zeros(3) for name in FEATURE_BUCKETS}
    raw["market_breadth"] = np.array([0.3, 0.5, np.nan])
    raw["extension_atr"] = np.array([1.0, 3.0, 0.0])
    hits = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    features, names, groups = economic_features(raw, hits, ("breakout", "rebound"))
    assert len(set(names)) == len(names) == len(groups) == features.shape[1]
    np.testing.assert_array_equal(
        features[:, names.index("interaction:breakout:strong_market")], [0, 1, 0]
    )
    np.testing.assert_array_equal(
        features[:, names.index("interaction:rebound:extended_price")], [0, 1, 0]
    )
    prefix, _, _ = economic_features(
        {k: v[:2] for k, v in raw.items()}, hits[:2], ("breakout", "rebound")
    )
    np.testing.assert_array_equal(prefix, features[:2])
    with pytest.raises(ValueError):
        economic_features({}, hits, ("breakout", "rebound"))


def test_scoring_interface_aligns_schema_and_absolute_scores():
    """场景：模型列顺序与原始特征不同及未知列；预期按名对齐，未知拒绝，分数绝对可比。"""
    from tushare_entitlement_hub.strategy_discovery.economic_features import (
        score_economic_candidates,
    )
    from tushare_entitlement_hub.strategy_discovery.utility_scoring import AdditiveModel

    raw = {name: np.zeros(2) for name in FEATURE_BUCKETS}
    hits = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = AdditiveModel(
        ("signal:rebound", "signal:breakout"),
        ("net_return", "tail_loss", "holding_sessions", "win_indicator"),
        np.array([0.0, 0.0, 5.0, 0.5]),
        np.array([[0.01, 0.0, 0.0, 0.0], [-0.01, 0.0, 0.0, 0.0]]),
    )
    output = score_economic_candidates(
        raw, hits, ("breakout", "rebound"), model, tail_penalty=0, daily_cost=0
    )
    np.testing.assert_allclose(output["expected_net_return"], [-0.01, 0.01])
    assert output["score"][0] < 50 < output["score"][1]
    bad = AdditiveModel(("unknown",), model.target_names, model.intercept, np.zeros((1, 4)))
    with pytest.raises(ValueError):
        score_economic_candidates(raw, hits, ("breakout", "rebound"), bad)
