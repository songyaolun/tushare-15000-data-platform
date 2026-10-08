"""基础评分排名按完整口径分组，只为每个合格组计算一次降序平均秩。"""

import hashlib
from collections import Counter
from dataclasses import replace

import pytest
from test_product_foundation_edges import sample_feature, sample_score_policy

from tushare_entitlement_hub.product_foundation import features
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.features import (
    ScoreResult,
    rank_scores,
    weight_stability,
)
from tushare_entitlement_hub.store import encode


def score_result(identity="A", score=50, **changes):
    """生成有限、已声明口径的合成评分，不读取行情或外部配置。

    例：score=None构造缺值，reason_codes=('excluded',)构造有值但禁止参与排名的记录。
    """
    values = {
        "entity_id": f"cn:security:{identity}:life",
        "analysis_as_of": "2026-10-03T00:00:00Z",
        "scope_id": "observed_research",
        "model_version": "model/v1",
        "feature_profile": "complete",
        "score": score,
        "weight_coverage": 1,
        "components": (),
        "missing_features": (),
        "reason_codes": (),
    }
    return ScoreResult(**(values | changes))


def mixed_score_groups():
    """生成五个仅口径不同的合格组及两个不可排名组，每组含并列或排除值。

    例：同一实体在不同模型版本下属于不同组；缺值仍报告其合格同组样本数4。
    """
    changes = (
        {},
        {"analysis_as_of": "2026-10-04T00:00:00Z"},
        {"scope_id": "other_scope"},
        {"model_version": "model/v2"},
        {"feature_profile": "partial"},
    )
    results = tuple(
        score_result(identity, score, **context, reason_codes=reasons)
        for index, context in enumerate(changes)
        for identity, score, reasons in (
            ("missing", None, ()),
            ("excluded", 100, ("excluded",)),
            ("A", 10 + index, ()),
            ("B", 10 + index, ()),
            ("C", 30 + index, ()),
            ("D", 0, ()),
        )
    )
    results += (
        score_result("A", 20, feature_profile="small"),
        score_result("B", 50, feature_profile="small"),
        score_result("A", None, feature_profile="unavailable"),
        score_result("B", 50, feature_profile="unavailable", reason_codes=("excluded",)),
    )
    return results, FoundationPolicy(version="test", minimum_peer_count=3)


def rank_json(ranked):
    """规范序列化完整排名映射，包含rank、peer_count及全部原因，不受字典插入顺序影响。

    例：缺样本输出rank=null及insufficient_rank_sample，不能只比较有值的排名。
    """
    return encode({key: item.to_dict() for key, item in ranked.items()})


def test_shared_score_group_is_sorted_once_for_5911_entities(monkeypatch):
    """5911实体共享组只排序一次，11档含零分并列值保持降序平均秩，不以墙钟作断言。"""
    results = tuple(score_result(f"S{index:04}", (index % 11) * 10) for index in range(5911))
    counts = Counter(item.score for item in results)
    calls = []
    original = features._average_ranks

    def counted(group, *, descending=False):
        """记录实际排序输入；例：第二次排序立即失败，避免旧算法跑完5911次排序。"""
        calls.append((len(group), descending))
        assert len(calls) == 1, "评分组被逐实体重复排序"
        return original(group, descending=descending)

    monkeypatch.setattr(features, "_average_ranks", counted)
    ranked = rank_scores(results, FoundationPolicy(version="test"))
    assert calls == [(5911, True)]
    assert len(ranked) == 5911
    for item in results:
        above = sum(count for value, count in counts.items() if value > item.score)
        position = ranked[item.result_id]
        assert position.rank == above + (counts[item.score] + 1) / 2
        assert position.peer_count == 5911 and position.reason_codes == ()


def test_score_groups_are_isolated_and_exclusions_keep_original_counts(monkeypatch):
    """五个不同口径组各排序一次，缺值/带原因记录排除，小组与空组不排序，逆序输入JSON不变。"""
    results, policy = mixed_score_groups()
    calls = []
    original = features._average_ranks

    def counted(group, *, descending=False):
        """按完整成员集合记录池；例：同实体不同scope的result_id集合不同，不能复用彼此排名。"""
        calls.append((frozenset(group), descending))
        return original(group, descending=descending)

    monkeypatch.setattr(features, "_average_ranks", counted)
    ranked = rank_scores(results, policy)
    assert len(calls) == len(set(calls)) == 5
    assert all(len(group) == 4 and descending for group, descending in calls)
    for index, item in enumerate(results[:30]):
        value = ranked[item.result_id]
        assert value.peer_count == 4
        assert value.rank == (None, None, 2.5, 2.5, 1, 4)[index % 6]
        assert value.reason_codes == (("insufficient_rank_sample",) if index % 6 < 2 else ())
    for item in results[30:]:
        value = ranked[item.result_id]
        assert value.rank is None
        assert value.peer_count == (2 if item.feature_profile == "small" else 0)
        assert value.reason_codes == ("insufficient_rank_sample",)
    before = rank_json(ranked)
    # 在391212f上记录的完整JSON摘要，包含所有组、名次、分母和原因。
    assert (
        hashlib.sha256(before.encode()).hexdigest()
        == "79338ebb7f205d41bc9b7ed74a3ef6c9155e82592609c96768ea423222aa8c04"
    )
    calls.clear()
    assert rank_json(rank_scores(tuple(reversed(results)), policy)) == before
    assert len(calls) == len(set(calls)) == 5


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_score_ids_fail_before_any_sort(monkeypatch, conflicting):
    """完全重复和同ID不同分数均在任何排序前以原错误拒绝，不能靠组字典覆盖掉重复。"""
    item = score_result()
    duplicate = replace(item, score=90) if conflicting else item

    def forbidden(*args, **kwargs):
        """重复ID必须先拒绝；任意排序调用都表示错误顺序发生变化。"""
        pytest.fail("重复ID触发了排序")

    monkeypatch.setattr(features, "_average_ranks", forbidden)
    with pytest.raises(ValueError, match="^duplicate_score_result$"):
        rank_scores((item, duplicate), FoundationPolicy(version="test"))
    assert rank_scores((), FoundationPolicy(version="test")) == {}


def test_weight_stability_json_matches_before_group_index():
    """真实权重敏感性调用链的完整JSON与391212f一致，分组索引不改变扰动场景和排名区间。"""
    first = (sample_feature(value=1), sample_feature("liquidity", 0))
    second = tuple(
        replace(item, entity_id="cn:security:B:life", value=value)
        for item, value in zip(first, (0, 0.98), strict=True)
    )
    result = weight_stability((first, second), sample_score_policy())
    assert (
        hashlib.sha256(result.to_json().encode()).hexdigest()
        == "ac27c8552d3cbaad806e3eb506cf2479a01f77516ef8a2e08b5312e9f9cdf4d1"
    )
