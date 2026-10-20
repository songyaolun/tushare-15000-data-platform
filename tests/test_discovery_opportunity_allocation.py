"""排名预算梯度、等预算对照及接口异常边界。"""

import pytest
from test_discovery_candidate_allocation import allocation_fixture
from test_discovery_research_hooks import hook_account

from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.opportunity_allocation import (
    RankWeightRule,
    make_rank_weight_provider,
)


@pytest.mark.parametrize(
    "first,last",
    [(0, 0.01), (0.03, 0.05), (1.1, 0.05), (True, 0.03), (0.05, float("nan")), (".05", 0.03)],
)
def test_invalid_rank_rule(first, last):
    """场景：无效预算区间、布尔、非有限或非数字；预期创建时拒绝。"""
    with pytest.raises(ValueError, match="invalid_rank_weight_rule"):
        RankWeightRule(first, last)


@pytest.mark.parametrize("rank,count", [(-1, 3), (3, 3), (0, 0), (True, 3), (0, 3.0)])
def test_invalid_rank_context(rank, count):
    """场景：排名越界或非整数上下文；预期拒绝而非默默增加预算。"""
    panel, _ = allocation_fixture()
    with pytest.raises(ValueError, match="invalid_rank_weight_context"):
        make_rank_weight_provider(RankWeightRule())(panel, 0, 0, 50, rank, count, DEFAULT_POLICY)


def test_linear_weights_singleton_and_equal_control():
    """场景：三个排名、单候选和4%等权对照；预期5/4/3%、单只5%、等权均4%。"""
    panel, _ = allocation_fixture()
    provider = make_rank_weight_provider(RankWeightRule())
    assert [
        provider(panel, 0, 0, 50, rank, 3, DEFAULT_POLICY) for rank in range(3)
    ] == pytest.approx([0.05, 0.04, 0.03])
    assert provider(panel, 0, 0, 50, 0, 1, DEFAULT_POLICY) == 0.05
    equal = make_rank_weight_provider(RankWeightRule(0.04, 0.04))
    assert [equal(panel, 0, 0, 50, rank, 3, DEFAULT_POLICY) for rank in range(3)] == [0.04] * 3
    with pytest.raises(ValueError, match="invalid_rank_weight_context"):
        make_rank_weight_provider(RankWeightRule(0.06, 0.03))(panel, 0, 0, 50, 0, 3, DEFAULT_POLICY)


def test_account_gradient_keeps_cash_and_maximum_budget():
    """场景：实际账户选三只且按5/4/3分预算；预期仓位递减、现金余额非负。"""
    panel, definition = allocation_fixture()
    result = hook_account(
        panel, definition, position_weight_provider=make_rank_weight_provider(RankWeightRule())
    )
    weights = result["daily"][1]["weights"]
    assert 0.048 < weights[0] < 0.05
    assert 0.038 < weights[1] < 0.04
    assert 0.028 < weights[2] < 0.03
    assert all(row["cash"] >= 0 for row in result["daily"])
