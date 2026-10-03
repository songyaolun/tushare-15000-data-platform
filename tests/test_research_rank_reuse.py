"""同行池回退与删因子重复输入的操作计数回归，不使用墙钟阈值。"""

from dataclasses import replace

import pytest
from test_research_candidates import research_inputs, research_parents
from test_research_candidates_index import candidate_fingerprint

from tushare_entitlement_hub.equity_research import candidates
from tushare_entitlement_hub.product_foundation import features
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.features import PeerObservation, percentile_ranks


@pytest.mark.parametrize(
    "scenario,expected_calls,expected_fingerprint",
    [
        ("missing_risk", 4, "e4eca4ec886ed0c944735f0be374e4c1b96c9f156f6f2c01c6479f01487976af"),
        ("zero_quality", 6, "3e5b9b90bb5a8fc304c94411dfa9d40b8d1b81ab57a49eb7a78dc3b45189effe"),
    ],
)
def test_leave_one_factor_reuses_only_identical_populations(
    monkeypatch, scenario, expected_calls, expected_fingerprint
):
    """真实P03中全缺失和零贡献场景少算相同输入，完整JSON与cb4e879保持一致，场景名称不丢失。"""
    context = research_inputs(4, minimum=2)
    groups = list(research_parents(context, quality=0 if scenario == "zero_quality" else 80))
    if scenario == "missing_risk":
        groups[1] = ()
    original = candidates.percentiles
    calls = []

    def counted(inputs, values, profile, **kwargs):
        """记录真实分位调用输入并透传；例：基线与删除零贡献因子的值相同时只计算一次。"""
        calls.append((profile, tuple(values.items())))
        return original(inputs, values, profile, **kwargs)

    monkeypatch.setattr(candidates, "percentiles", counted)
    results = candidates.build_candidates(context, *groups)
    assert len(calls) == expected_calls
    # 前三次分别生成PE、PB、amount分位；之后是基线及删因子场景。
    assert len(calls[3:]) == len(set(calls[3:]))
    assert candidate_fingerprint(results) == expected_fingerprint
    for result in results:
        assert set(result.details["leave_one_factor_ranks"]) == {
            "quality_score",
            "value_score",
            "liquidity_score",
        }
    # 同一函数重入仍重新计算本次输入，不跨构建保留排名结果。
    calls.clear()
    repeated = candidates.build_candidates(context, *groups)
    assert len(calls) == expected_calls
    assert [result.to_json() for result in repeated] == [result.to_json() for result in results]


@pytest.mark.parametrize("lower_is_better", [False, True])
def test_fallback_pools_sort_once_and_exclude_missing_and_ineligible(monkeypatch, lower_is_better):
    """两个小叶子共享一次上级排序，另一叶子独立排序；缺值、无资格和不足样本不计入有效分母。"""
    base = PeerObservation(
        entity_id="cn:security:template:life",
        value=1,
        analysis_as_of="2026-10-03T00:00:00Z",
        feature_profile="complete",
        scope_id="observed_research",
        peer_path=("CN", "sector", "a"),
    )
    data = (
        ("a1", 1, ("CN", "sector", "a"), True),
        ("a2", 1, ("CN", "sector", "a"), True),
        ("b", 2, ("CN", "sector", "b"), True),
        ("missing", None, ("CN", "sector", "a"), True),
        ("ineligible", 999, ("CN", "sector", "a"), False),
        ("x", 0, ("CN", "other", "c"), True),
        ("y", 10, ("CN", "other", "c"), True),
        ("z", 10, ("CN", "other", "c"), True),
        ("single", 1, ("isolated",), True),
    )
    rows = tuple(
        replace(
            base,
            entity_id=f"cn:security:{name}:life",
            value=value,
            peer_path=path,
            eligible=eligible,
        )
        for name, value, path, eligible in data
    )
    calls = []
    original = features._average_ranks

    def counted(pool):
        """按参与实体记录实际池；例：a1/a2/b只允许作为一个完整池排序一次。"""
        calls.append(frozenset(pool))
        return original(pool)

    monkeypatch.setattr(features, "_average_ranks", counted)
    policy = FoundationPolicy(version="test", minimum_peer_count=3)
    result = percentile_ranks(rows, policy, lower_is_better=lower_is_better)
    assert len(calls) == len(set(calls)) == 2
    assert set(calls) == {
        frozenset(row.entity_id for row in rows[:3]),
        frozenset(row.entity_id for row in rows[5:8]),
    }
    for row, expected in zip(rows[:3] + rows[5:8], (0.25, 0.25, 1.0, 0.0, 0.75, 0.75), strict=True):
        item = result[row.entity_id]
        assert item.percentile == (1 - expected if lower_is_better else expected)
        assert item.peer_count == 3
    for row in rows[:3]:
        assert result[row.entity_id].peer_level == "sector"
        assert result[row.entity_id].reason_codes == ("peer_level_fallback",)
    for row in rows[3:5] + rows[8:]:
        item = result[row.entity_id]
        assert item.percentile is None and item.peer_count == 0 and item.peer_level is None
        assert item.reason_codes == (
            ("insufficient_peer_sample",) if row is rows[-1] else ("peer_value_unavailable",)
        )
    calls.clear()
    reversed_result = percentile_ranks(
        tuple(reversed(rows)), policy, lower_is_better=lower_is_better
    )
    assert len(calls) == len(set(calls)) == 2
    assert {key: value.to_json() for key, value in result.items()} == {
        key: value.to_json() for key, value in reversed_result.items()
    }
