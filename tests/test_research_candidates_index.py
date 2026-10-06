"""P03 父结果索引与同行池排序的规模回归，使用合成实体和操作计数。"""

import hashlib
from dataclasses import replace
from time import perf_counter

import pytest
from test_research_candidates import research_inputs, research_parents

from tushare_entitlement_hub.equity_research.candidates import build_candidates
from tushare_entitlement_hub.product_foundation import features
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.features import PeerObservation, percentile_ranks

SCENARIOS = (
    "normal",
    "market",
    "ties",
    "0_missing",
    "0_duplicate",
    "0_future",
    "0_mixed",
    "1_missing",
    "1_duplicate",
    "1_future",
    "1_mixed",
    "2_missing",
    "2_duplicate",
    "2_future",
    "2_mixed",
)
EXPECTED_COMPLETE_SCORES = {
    "normal": 57.25,
    "market": 38.33333333333333,
    "ties": 63.5,
}


class SinglePassParents(tuple):
    """对父集合的第二次遍历立即报错，旧二次算法无需完成一亿次比较即可被检出。"""

    def __new__(cls, values):
        """包装已有父对象，不改变顺序或重复性；例：两条父结果初始访问数为0。"""
        instance = super().__new__(cls, values)
        instance.passes = 0
        instance.visits = 0
        return instance

    def __iter__(self):
        """允许一次完整索引扫描，记录每个输入；例：第二次iter立即触发断言。"""
        self.passes += 1
        assert self.passes == 1, "父结果集合被逐实体重复扫描"
        for result in super().__iter__():
            self.visits += 1
            yield result


def candidate_scenario(scenario="normal", count=4):
    """生成稳定的候选输入，区分缺失、重复、错截止和当前/未来混合重复。

    例：'1_duplicate'使首个实体的P06出现两次；'0_missing'只缺P04，仍可走市场档。
    """
    context = research_inputs(count, minimum=2, flat=scenario == "ties")
    groups = list(research_parents(context, quality=None if scenario == "market" else 80))
    if "_" in scenario:
        index, change = scenario.split("_", 1)
        index = int(index)
        parent, *others = groups[index]
        if change == "missing":
            groups[index] = tuple(others)
        elif change == "duplicate":
            groups[index] = (parent, parent, *others)
        else:
            future = replace(
                parent,
                envelope=replace(
                    parent.envelope,
                    analysis_as_of="2026-10-03T01:00:00Z",
                    generated_at="2026-10-03T01:00:00Z",
                ),
            )
            groups[index] = (future, parent, *others) if change == "mixed" else (future, *others)
    return context, tuple(groups)


def candidate_fingerprint(results):
    """逐个累计完整JSON字节，不保留第二份大结果列表；例：顺序或任何字段变化都会改变摘要。"""
    digest = hashlib.sha256()
    for result in results:
        payload = result.to_json().encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def test_parent_collections_are_indexed_once_for_5911_entities(monkeypatch):
    """完整5911构建只扫描17733条父记录、排序7次，父ID和全部JSON字节与cb4e879一致。"""
    context, original = candidate_scenario(count=5911)
    groups = tuple(SinglePassParents(group) for group in original)
    sorted_pools = []
    average = features._average_ranks

    def counted(pool):
        """三项因子、基线和三个不同删因子输入各排序一次；例：第八次说明仍有重复排序。"""
        sorted_pools.append(len(pool))
        assert len(sorted_pools) <= 7
        return average(pool)

    monkeypatch.setattr(features, "_average_ranks", counted)
    started = perf_counter()
    results = build_candidates(context, *groups)
    seconds = perf_counter() - started
    assert len(results) == 5911
    assert [group.passes for group in groups] == [1, 1, 1]
    assert [group.visits for group in groups] == [5911, 5911, 5911]
    assert sorted_pools == [5911] * 7
    for result, *parents in zip(results, *original, strict=True):
        assert result.parent_result_ids == tuple(
            sorted(item.envelope.result_id for item in parents)
        )
        assert result.envelope.peer_count == 5911
        assert all(check.passed for check in result.envelope.quality_gates)
    fingerprint = candidate_fingerprint(results)
    assert len(fingerprint) == 64
    print(
        f"5911实体：父访问17733次，排序7次，build_candidates={seconds:.3f}s，完整结果SHA256={fingerprint}"
    )


@pytest.mark.parametrize("lower_is_better", [False, True])
def test_each_shared_peer_pool_is_ranked_once_for_5911_entities(monkeypatch, lower_is_better):
    """5911个同池实体只计算一次平均秩，正反向与同值平均秩均保持精确原值。"""
    rows = tuple(
        PeerObservation(
            entity_id=f"cn:security:S{index:04}:life",
            value=float(index % 11),
            analysis_as_of="2026-10-03T00:00:00Z",
            feature_profile="complete",
            scope_id="observed_research",
            peer_path=("CN", "industry"),
        )
        for index in range(5911)
    )
    calls = []
    average = features._average_ranks
    expected = average({row.entity_id: row.value for row in rows})

    def counted(values, **kwargs):
        """同池第二次排序直接失败；例：5911行仍只允许一次完整排序。"""
        calls.append(len(values))
        assert len(calls) == 1, "同一同行池被逐实体重复排序"
        return average(values, **kwargs)

    monkeypatch.setattr(features, "_average_ranks", counted)
    results = percentile_ranks(
        rows, FoundationPolicy(version="test"), lower_is_better=lower_is_better
    )
    assert calls == [5911]
    for row in rows:
        percentile = (expected[row.entity_id] - 1) / 5910
        assert results[row.entity_id].percentile == (
            1 - percentile if lower_is_better else percentile
        )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_parent_semantics_match_9b00b05_bytes_including_duplicates(scenario):
    """正常、并列、缺失、重复、错截止及混合重复保持语义，重排父集合不改变结果。"""
    context, groups = candidate_scenario(scenario)
    results = build_candidates(context, *groups)
    reversed_results = build_candidates(context, *(tuple(reversed(group)) for group in groups))
    assert [result.to_json() for result in reversed_results] == [
        result.to_json() for result in results
    ]
    if "_" in scenario:
        index = int(scenario.split("_", 1)[0])
        affected = results[0]
        expected = {
            parent.envelope.result_id
            for number, group in enumerate(groups)
            if number != index
            for parent in group
            if parent.envelope.entity_id == affected.envelope.entity_id
        }
        assert set(affected.parent_result_ids) == expected
        codes = {check.gate_id: check.result.value for check in affected.envelope.quality_gates}
        if index == 0:
            assert affected.details["score_profile"] == "market_only"
            assert "fundamental_parent" not in codes
        else:
            assert codes[("governance" if index == 1 else "flow") + "_parent"] == "unknown"
            assert affected.envelope.score is None
    else:
        assert results[1].envelope.score == EXPECTED_COMPLETE_SCORES[scenario]


def test_peer_rank_cache_keeps_cutoff_scope_profile_and_path_separate(monkeypatch):
    """五个只差一个口径字段的同行池独立排序一次，同值平均秩和分母不得跨池复用。"""
    base = PeerObservation(
        entity_id="cn:security:A:life",
        value=1,
        analysis_as_of="2026-10-03T00:00:00Z",
        feature_profile="complete",
        scope_id="observed_research",
        peer_path=("CN", "industry"),
    )
    variations = (
        {},
        {"analysis_as_of": "2026-10-04T00:00:00Z"},
        {"feature_profile": "market"},
        {"scope_id": "other_scope"},
        {"peer_path": ("CN", "other_industry")},
    )
    rows = tuple(
        replace(base, entity_id=f"cn:security:S{group}_{index}:life", value=value, **changes)
        for group, changes in enumerate(variations)
        for index, value in enumerate((1, 1, 2))
    )
    calls = []
    average = features._average_ranks

    def counted(pool):
        """记录每个被排序的池的实体键；例：三行同池的键集合只出现一次。"""
        calls.append(frozenset(pool))
        return average(pool)

    monkeypatch.setattr(features, "_average_ranks", counted)
    ranked = percentile_ranks(rows, FoundationPolicy(version="test", minimum_peer_count=3))
    assert len(calls) == len(set(calls)) == 5
    assert [ranked[row.entity_id].percentile for row in rows] == [0.25, 0.25, 1.0] * 5
    assert {value.peer_count for value in ranked.values()} == {3}


def test_empty_and_triplicate_parent_groups_remain_fail_closed():
    """空股票集合仍返回空结果，三份相同父结果仍无效，不能因第二份已标无效而重新接纳第三份。"""
    empty, groups = candidate_scenario(count=0)
    assert build_candidates(empty, *groups) == ()
    context, groups = candidate_scenario()
    fundamental, governance, flow = groups
    results = build_candidates(context, fundamental, (governance[0],) * 3 + governance[1:], flow)
    first, *others = results
    assert first.envelope.score is None and first.envelope.rank is None
    assert all(item.envelope.score is not None for item in others)
