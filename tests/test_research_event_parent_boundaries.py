"""评审回归：错截止父结果先排除，当前实体第65个唯一版本立即停止读取输入流。"""

from dataclasses import replace
from itertools import chain

import pytest
from test_research_events import research_inputs, thesis_parents

from tushare_entitlement_hub.equity_research import event_parents
from tushare_entitlement_hub.equity_research.events import build_events
from tushare_entitlement_hub.equity_research.models import ResearchResult


def future_parent(parent):
    """生成合法但属于其他截止时点的父结果；例：当前10月3日输入得到10月4日父版本。"""
    return replace(
        parent,
        envelope=replace(
            parent.envelope,
            analysis_as_of="2026-10-04T00:00:00Z",
            generated_at="2026-10-04T00:00:00Z",
        ),
    )


@pytest.mark.parametrize("damage", ["conflict", "missing_entity"])
def test_wrong_cutoff_precedes_entity_validation_and_duplicate_comparison(monkeypatch, damage):
    """其他截止父的同ID冲突或缺实体均先被排除；混入前后不改变本轮索引、摘要和P13字节。"""
    inputs = research_inputs(1)
    current = thesis_parents(inputs)
    first = future_parent(current[0])
    second = replace(first, details_json='{"different":true}')
    if damage == "missing_entity":
        object.__setattr__(second.envelope, "entity_id", None)
    excluded = (first, second)
    expected_index = event_parents.index_parents(inputs, current)
    expected_json = tuple(result.to_json() for result in build_events(inputs, current))
    original_json = ResearchResult.to_json

    def checked_json(parent):
        """错截止对象不得进入完整JSON比较；当前结果仍使用原有严格序列化。"""
        assert parent is not first and parent is not second
        return original_json(parent)

    monkeypatch.setattr(ResearchResult, "to_json", checked_json)
    assert event_parents.index_parents(inputs, excluded) == {}
    for ordered in (excluded + current, current + excluded[::-1]):
        assert event_parents.index_parents(inputs, ordered) == expected_index
        assert tuple(result.to_json() for result in build_events(inputs, ordered)) == expected_json


@pytest.mark.parametrize("repeat", [False, True])
def test_sixty_fifth_unique_current_parent_stops_before_consuming_stream_tail(monkeypatch, repeat):
    """第65个当前唯一父立刻报容量错误；即使每条都重复，仍不读取流尾或提前计算指纹。"""
    inputs = research_inputs(1)
    parent = thesis_parents(inputs)[0]
    seen = []
    repeated = []

    def upstream():
        """惰性产生65个唯一版本，流尾为失败哨兵；有重复时前64个各额外产出一份相同内容。"""
        for index in range(65):
            candidate = replace(
                parent, envelope=replace(parent.envelope, scope_id=f"version-{index}")
            )
            seen.append(index)
            yield candidate
            if repeat:
                repeated.append(index)
                yield replace(candidate)
        pytest.fail("容量检查耗尽了父结果流")

    def unexpected_fingerprint(value):
        """输入尚未通过容量边界时不能构造任何bundle指纹；任意调用均说明失败过晚。"""
        pytest.fail("容量拒绝前计算了父结果指纹")

    monkeypatch.setattr(event_parents, "parent_fingerprint", unexpected_fingerprint)
    with pytest.raises(ValueError, match="^research_event_parent_capacity_exceeded$"):
        event_parents.index_parents(inputs, upstream())
    assert seen == list(range(65))
    assert repeated == (list(range(64)) if repeat else [])


def test_capacity_counts_only_current_unique_versions_within_each_entity():
    """两个实体各64个当前版本合法；重复及80个错截止版本不占容量，逆序流保持相同bundle。"""
    inputs = research_inputs(2)
    parent = thesis_parents(inputs)[0]
    current = tuple(
        replace(
            parent,
            envelope=replace(parent.envelope, entity_id=identity, scope_id=f"version-{index}"),
        )
        for identity in inputs.entity_ids
        for index in range(64)
    )
    excluded = tuple(
        future_parent(
            replace(parent, envelope=replace(parent.envelope, scope_id=f"ignored-{index}"))
        )
        for index in range(80)
    )
    expected = event_parents.index_parents(inputs, current)
    actual = event_parents.index_parents(inputs, chain(excluded, reversed(current), current))
    assert actual == expected
    assert len(actual) == 2
    assert all(len(bundle.parents) == 64 for bundle in actual.values())
