"""可解释效用模型的软件正确性，不是金融样本外检验。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.utility_scoring import (
    AdditiveModel,
    bucket_features,
    fit_additive_model,
    utility_priority,
)


def test_buckets_missing_boundaries_and_prefix():
    """场景：边界、缺值、追加未来数据；预期分箱确定且历史不改变。"""
    values = np.array([-1.0, 0.0, 2.0, np.nan])
    result = bucket_features(values, (0, 1))
    np.testing.assert_array_equal(result, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(bucket_features(values[:2], (0, 1)), result[:2])
    with pytest.raises(ValueError):
        bucket_features(values, (1, 0))


def test_fit_explanations_and_daily_weighting():
    """场景：线性真值、多目标和同日整批复制；预期拟合准确、贡献闭合、复制不增权。"""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    targets = np.column_stack((features[:, 0] * 0.02 + 0.01, features[:, 0] + 2))
    days = np.array([1, 1, 2, 2], dtype=np.int64)
    model = fit_additive_model(
        features, targets, days, ("x",), ("net", "holding"), regularization=1e-8
    )
    np.testing.assert_allclose(model.predict(features), targets, atol=1e-7)
    np.testing.assert_allclose(
        model.contributions(features, 0).sum(axis=1), model.predict(features)[:, 0]
    )
    selected = np.array([0, 1, 0, 1, 2, 3])
    copied = fit_additive_model(
        features[selected],
        targets[selected],
        days[selected],
        ("x",),
        ("net", "holding"),
        regularization=1e-8,
    )
    np.testing.assert_allclose(copied.coefficients, model.coefficients)
    with pytest.raises(ValueError):
        model.predict(np.ones((1, 2)))
    with pytest.raises(ValueError):
        model.contributions(features, 2)
    with pytest.raises(ValueError):
        fit_additive_model(features, targets, days, ("x",), ("net", "holding"), regularization=0)
    with pytest.raises(ValueError):
        AdditiveModel(("x", "x"), ("net",), np.zeros(1), np.zeros((2, 1)))


def test_absolute_utility_tail_and_capital_penalties():
    """场景：全池预期亏损和等收益不同尾险/持有期；预期无强造高分且风险增加会减分。"""
    returns = np.array([-0.01, -0.02, -0.03])
    tail = np.array([0.0, 0.01, 0.01])
    holding = np.array([5.0, 5.0, 10.0])
    utility, score = utility_priority(returns, tail, holding)
    assert np.all(score < 50) and np.all(np.diff(score) < 0)
    np.testing.assert_allclose(utility, [-0.011, -0.031, -0.042])
    _, prefix_score = utility_priority(returns[:1], tail[:1], holding[:1])
    assert prefix_score[0] == score[0]
    with pytest.raises(ValueError):
        utility_priority(returns, tail, holding, scale=0)
