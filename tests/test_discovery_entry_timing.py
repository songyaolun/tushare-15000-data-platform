"""开盘入场区间的除权、时点与账户集成测试，无网络和真实行情。"""

from dataclasses import replace

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.strategy_discovery.candidate_allocation import (
    simulate_candidate_allocation,
)
from tushare_entitlement_hub.strategy_discovery.contracts import DEFAULT_POLICY
from tushare_entitlement_hub.strategy_discovery.entry_timing import (
    OpeningGapRule,
    opening_gap_allowed,
)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(True, None), (None, float("nan")), (-1, None), (None, 1), (0.02, 0.01), ("bad", None)],
)
def test_invalid_gap_rule(minimum, maximum):
    """场景：布尔、缺失数、越界、倒置或非数字；预期构造时拒绝。"""
    with pytest.raises(ValueError, match="invalid_opening_gap_rule"):
        OpeningGapRule(minimum, maximum)


@pytest.mark.parametrize(
    ("opening", "expected"), [(4.84, False), (4.85, True), (5.05, True), (5.06, False)]
)
def test_adjusted_gap_boundaries_and_future_independence(opening, expected):
    """场景：次日除权、上下界及随后行情任意改变；预期仅按已知复权开盘判断。"""
    panel, _ = allocation_fixture()
    panel.values["raw_close"][0, 0] = 10
    panel.values["raw_adj_factor"][0, 0] = 2
    panel.values["raw_adj_factor"][1, 0] = 4
    panel.values["raw_open"][1, 0] = opening
    rule = OpeningGapRule(-0.03, 0.01)
    assert opening_gap_allowed(panel, 0, 1, 0, rule) is expected
    panel.values["raw_close"][1:, :] = np.nan
    panel.values["raw_open"][2:, :] = np.nan
    assert opening_gap_allowed(panel, 0, 1, 0, rule) is expected


@pytest.mark.parametrize(
    "field,day", [("raw_close", 0), ("raw_adj_factor", 0), ("raw_open", 1), ("raw_adj_factor", 1)]
)
@pytest.mark.parametrize("value", [np.nan, 0, -1])
def test_missing_or_nonpositive_prices_fail_closed(field, day, value):
    """场景：任一价格或复权因子缺失、零、负；预期即使不设价格界也拒绝入场。"""
    panel, _ = allocation_fixture()
    panel.values[field][day, 0] = value
    assert not opening_gap_allowed(panel, 0, 1, 0, OpeningGapRule())


def test_one_session_only_and_optional_bounds():
    """场景：无界、单端区间、同日及过期请求；预期仅下一日且满足已配置边界通过。"""
    panel, _ = allocation_fixture()
    assert opening_gap_allowed(panel, 0, 1, 0, OpeningGapRule())
    assert opening_gap_allowed(panel, 0, 1, 0, OpeningGapRule(minimum=-0.03))
    assert opening_gap_allowed(panel, 0, 1, 0, OpeningGapRule(maximum=0.01))
    assert not opening_gap_allowed(panel, 0, 0, 0, OpeningGapRule())
    assert not opening_gap_allowed(panel, 0, 2, 0, OpeningGapRule())


def test_gap_filter_keeps_top_candidates_and_expires_without_refill():
    """场景：Top2分别大低开与正常开盘，第三名更合适；预期拒绝首名且不事后补第三名。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, :3] = True
    definition = replace(definition, entry_mask=mask, maximum_entry_gap=0.02)
    panel.values["raw_open"][1, 0] = 9.5
    result = simulate_candidate_allocation(
        panel,
        np.zeros(8, dtype=np.int8),
        [],
        definition,
        DEFAULT_POLICY,
        mode="exit_reserve",
        count=2,
        start=panel.calendar[0],
        opening_gap_rule=OpeningGapRule(-0.03, 0.01),
    )
    assert result["candidate_selections"][0]["selected"] == [0, 1]
    assert result["entry_days"] == [1]
    assert result["daily"][1]["positions"] == 1


def test_additional_gap_rule_intersects_existing_limit():
    """场景：附加区间允许3%但原追价限制2%；预期仍拒绝2.5%高开。"""
    panel, definition = allocation_fixture()
    mask = np.zeros((8, 12), dtype=bool)
    mask[0, 0] = True
    definition = replace(definition, entry_mask=mask, maximum_entry_gap=0.02)
    panel.values["raw_open"][1, 0] = 10.25
    result = simulate_candidate_allocation(
        panel,
        np.zeros(8, dtype=np.int8),
        [],
        definition,
        DEFAULT_POLICY,
        mode="exit_reserve",
        start=panel.calendar[0],
        opening_gap_rule=OpeningGapRule(maximum=0.03),
    )
    assert result["entry_days"] == []
