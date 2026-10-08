"""P13 原文、实体确认、三值论点与确定性复核顺序的离线场景。"""

from dataclasses import replace

import pytest
from test_research_candidates import AS_OF, research_inputs, research_parents, research_row

from tushare_entitlement_hub.equity_research.common import gate, metric
from tushare_entitlement_hub.equity_research.events import THESIS, build_events
from tushare_entitlement_hub.product_foundation.contracts import ActionState, ProductState, Truth
from tushare_entitlement_hub.product_foundation.entities import EntityAlias, EntityRelation


def thesis_parents(inputs, cash=1.2, unlock=0.01, *, risk=Truth.TRUE):
    """场景：两个版本化默认论点输入；预期：现金覆盖与解禁来自其对应父产品。

    例如 cash=1.2、unlock=.01 返回支持论点的 P04/P06，risk 控制 P07 硬检查。
    """
    groups = research_parents(inputs)
    references = inputs.rows("daily")[0].evidence("amount")
    fundamental = replace(groups[0][0], metrics=(metric("cash_cover", cash, "ratio", references),))
    governance = replace(groups[1][0], metrics=(metric("unlock_30d", unlock, "ratio", references),))
    flow = replace(
        groups[2][0],
        envelope=replace(
            groups[2][0].envelope,
            quality_gates=(gate("flow_price", risk, "price_unavailable", references=references),),
        ),
    )
    return fundamental, governance, flow


def event_inputs(inputs, api, fields, *, identity=None, record="event-source"):
    """场景：添加一条带来源身份的事件；预期：与现有合成行情共存，无隐式实体解析。"""
    row = research_row(identity, api, fields, record=record)
    return replace(inputs, rows_by_api={**inputs.rows_by_api, api: (row,)})


def entity_alias(inputs, identity, alias, **changes):
    """构造有证据的时点别名；例：同名两实体形成待人工确认候选，而非自动连接。"""
    values = dict(
        namespace="names",
        alias=alias,
        entity_id=identity,
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        observed_at=AS_OF,
        source="manual",
        version="aliases/v1",
        confidence=100,
        evidence_refs=inputs.rows("daily")[0].evidence("amount"),
    )
    return EntityAlias(**{**values, **changes})


def event_relation(inputs, event_id, identity, **changes):
    """构造事件到实体的人审关系；例：confirmed=False 不赋予新闻实体确认资格。"""
    values = dict(
        from_entity_id=event_id,
        to_entity_id=identity,
        relation_type="event_entity",
        confirmed=True,
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        observed_at=AS_OF,
        source="manual",
        version="relations/v1",
        confidence=100,
        evidence_refs=inputs.rows("daily")[0].evidence("amount"),
    )
    return EntityRelation(**{**values, **changes})


def test_all_direct_sources_keep_independent_disclosure_stages():
    """场景：五个事件入口及一条披露多日期；预期：计划、实际和修正分别生成唯一结果。"""
    inputs = research_inputs(1)
    identity = inputs.entity_ids[0]
    for api, fields in {
        "major_news": {
            "title": "公开新闻",
            "content": "事实记录",
            "pub_time": "2026-09-30 10:00:00",
        },
        "disclosure_date": {
            "ann_date": "20260930",
            "end_date": "20260630",
            "pre_date": "20261020",
            "actual_date": "20261019",
            "modify_date": "20261018,20261017,bad",
        },
        "cn_schedule": {"title": "公开统计日程", "publish_date": "20261008", "data_api": "cn_cpi"},
        "idx_anns": {"title": "指数调整", "ann_date": "20260930", "source": "指数提供方"},
        "namechange": {"name": "企业新名", "start_date": "20261001", "ann_date": "20260930"},
    }.items():
        inputs = event_inputs(
            inputs,
            api,
            fields,
            identity=identity if api in {"disclosure_date", "namechange"} else None,
        )
    results = build_events(inputs, ())
    assert len(results) == 8
    assert len({result.envelope.result_id for result in results}) == 8
    assert len({result.envelope.scope_id for result in results}) == 8
    assert {result.details["event_type"] for result in results} >= {
        "disclosure_planned",
        "disclosure_actual",
        "disclosure_revision",
    }
    assert all(result.details["car"] is None for result in results)
    assert all(not result.details["causality_established"] for result in results)
    assert any(result.envelope.next_check_at == "2026-10-20T23:59:59Z" for result in results)
    reordered = replace(
        inputs, rows_by_api={key: rows + rows for key, rows in reversed(inputs.rows_by_api.items())}
    )
    assert results == build_events(reordered, ())


@pytest.mark.parametrize(
    "cash,unlock,expected",
    [
        (1.2, 0.01, Truth.TRUE),
        (0.5, 0.01, Truth.FALSE),
        (1.2, 0.05, Truth.FALSE),
        (1.2, 1.2, Truth.FALSE),
        (None, 0.01, Truth.UNKNOWN),
        (1.2, None, Truth.UNKNOWN),
    ],
)
def test_default_thesis_uses_three_valued_inputs(cash, unlock, expected):
    """场景：现金覆盖或解禁过阈值、缺失；预期：支持、反证和未知严格按三值逻辑计算。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs, cash, unlock)
    results = build_events(inputs, parents)
    assert len(results) == 1
    assert results[0].details["event_type"] == "upstream_bundle"
    for result in results:
        assert result.predicate_results[THESIS.predicate_id] is expected
        assert result.details["predicate"]["children"][1]["threshold"] == 0.05
        assert result.details["holding_weight"] is None
        assert len(result.parent_result_ids) == 3
        assert result.envelope.invalidation_conditions[0].operator == "not"
        if expected is Truth.FALSE:
            assert result.envelope.contrary_evidence
            assert result.details["contradictory_features"]


def test_news_original_reason_does_not_confirm_entity_or_causality():
    """场景：新闻原文含原因但无实体键；预期：逐字原因片段、实体未知，不能推出市场因果。"""
    inputs = event_inputs(
        research_inputs(1),
        "major_news",
        {
            "title": "盈利变动",
            "content": "公司披露：利润减少，原因是成本上升。其他事实。",
            "pub_time": "20260930",
        },
    )
    result = build_events(inputs, ())[0]
    assert result.details["attribution_level"] == "documented_reason"
    assert result.details["evidence_spans"] == [
        {"field": "content", "text": "公司披露：利润减少，原因是成本上升。"}
    ]
    assert result.details["confirmed_entity"] is None
    assert not result.details["causality_established"]
    assert result.envelope.state is ProductState.INCOMPLETE


@pytest.mark.parametrize(
    "alias_count,expected",
    [(0, "entity_unresolved"), (1, "entity_confirmation_required"), (2, "entity_ambiguous")],
)
def test_alias_candidates_remain_unconfirmed(alias_count, expected):
    """场景：文本别名匹配零、一、两个实体；预期：均不强归因，歧义明确要求人工确认。"""
    inputs = event_inputs(research_inputs(2), "idx_anns", {"title": "同名公司指数变化"})
    aliases = tuple(
        entity_alias(inputs, identity, "同名公司") for identity in inputs.entity_ids[:alias_count]
    )
    aliases += (
        entity_alias(inputs, inputs.entity_ids[0], "同名公司", effective_to="2026-09-01T00:00:00Z"),
        entity_alias(
            inputs, inputs.entity_ids[0], "同名公司", effective_from="2026-11-01T00:00:00Z"
        ),
        entity_alias(inputs, inputs.entity_ids[0], "别的名字"),
    )
    inputs = replace(
        inputs,
        config=replace(
            inputs.config, foundation=replace(inputs.config.foundation, aliases=aliases)
        ),
    )
    result = build_events(inputs, ())[0]
    assert result.details["confirmed_entity"] is None
    assert len(result.details["entity_candidates"]) == alias_count
    assert result.details["conflict_reason"] == expected


def test_manual_event_relations_confirm_only_one_valid_target():
    """场景：明确人审关系与冲突关系；预期：唯一有效目标确认，多目标保留歧义。"""
    inputs = event_inputs(research_inputs(2), "major_news", {"title": "无键新闻"})
    event_id = build_events(inputs, ())[0].details["event_id"]
    identities = inputs.entity_ids
    valid = event_relation(inputs, event_id, identities[0])
    ignored = (
        event_relation(inputs, event_id, identities[1], confirmed=False),
        event_relation(inputs, event_id, identities[1], relation_type="industry"),
        event_relation(inputs, "research:event:other:v1", identities[1]),
        event_relation(inputs, event_id, "cn:security:missing:life"),
        event_relation(inputs, event_id, identities[1], effective_to="2026-09-01T00:00:00Z"),
    )
    confirmed = replace(
        inputs, config=replace(inputs.config, confirmed_relations=(valid,) + ignored)
    )
    result = build_events(confirmed, ())[0]
    assert result.details["confirmed_entity"] == identities[0]
    conflicting = replace(
        inputs,
        config=replace(
            inputs.config,
            confirmed_relations=(valid, event_relation(inputs, event_id, identities[1])),
        ),
    )
    assert build_events(conflicting, ())[0].details["conflict_reason"] == "entity_ambiguous"


@pytest.mark.parametrize(
    "day,pct,amount,expected",
    [
        ("2026-09-30", 0.01, 100, "temporal_association"),
        ("2026-09-29", 0.01, 100, "evidence_only"),
        ("2026-09-30", None, 100, "evidence_only"),
        ("2026-09-30", 0.01, 0, "evidence_only"),
    ],
)
def test_temporal_association_needs_same_day_certified_traded_return(day, pct, amount, expected):
    """场景：同日涨跌、跨日、缺涨跌或无成交；预期：只有可用同日涨跌能标为时间相关。"""
    inputs = research_inputs(1)
    identity = inputs.entity_ids[0]
    daily = research_row(
        identity, "daily", {"close": 10, "vol": 10, "amount": amount, "pct_chg": pct}, day=day
    )
    inputs = replace(inputs, rows_by_api={**inputs.rows_by_api, "daily": (daily,)})
    inputs = event_inputs(inputs, "namechange", {"start_date": "20260930"}, identity=identity)
    result = build_events(inputs, ())[0]
    assert result.details["attribution_level"] == expected
    assert result.envelope.next_check_at is None


def test_missing_and_list_disclosure_dates_preserve_observation():
    """场景：披露日期缺失、非法值及列表修正；预期：缺口保留观察，合法修正逐项输出。"""
    inputs = event_inputs(research_inputs(1), "disclosure_date", {"modify_date": [None, 2, "bad"]})
    assert build_events(inputs, ())[0].details["event_type"] == "disclosure_observation"
    inputs = event_inputs(inputs, "disclosure_date", {"modify_date": ["20261004", "20261005"]})
    assert len(build_events(inputs, ())) == 2


@pytest.mark.parametrize("risk,level", [(Truth.FALSE, 2), (Truth.UNKNOWN, 1), (Truth.TRUE, 0)])
def test_review_priority_is_hard_risk_then_deadline_then_source(risk, level):
    """场景：同实体硬风险等级不同且事件期限不同；预期：记录确定性优先级，不生成概率。"""
    inputs = research_inputs(1)
    identity = inputs.entity_ids[0]
    inputs = event_inputs(inputs, "namechange", {"start_date": "20261008"}, identity=identity)
    inputs = event_inputs(inputs, "disclosure_date", {"pre_date": "20261005"}, identity=identity)
    parents = thesis_parents(inputs, risk=risk)
    results = build_events(inputs, parents)
    assert results[0].details["deadline"] == "2026-10-05"
    assert all(result.details["hard_risk_level"] == level for result in results)
    expected_state = {
        Truth.FALSE: ProductState.BLOCKED,
        Truth.UNKNOWN: ProductState.INCOMPLETE,
        Truth.TRUE: ProductState.ANNOTATED,
    }[risk]
    expected_action = {
        Truth.FALSE: ActionState.REVIEW_REQUIRED,
        Truth.UNKNOWN: ActionState.NO_CONCLUSION,
        Truth.TRUE: ActionState.WATCH,
    }[risk]
    assert all(result.envelope.state is expected_state for result in results)
    assert all(result.envelope.action_state is expected_action for result in results)
    assert all(
        any(
            check.gate_id == "flow_feasibility/v1/flow_price"
            for check in result.envelope.quality_gates
        )
        for result in results
    )
    assert [result.details["review_priority"] for result in results] == list(range(1, 4))
    assert results == build_events(inputs, tuple(reversed(parents)))


def test_parent_duplicate_and_conflict_and_cutoff_are_not_silently_selected():
    """场景：重复父结果、多版本和错截止；预期：重复幂等、冲突特征未知、未来父不消费。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    assert build_events(inputs, parents) == build_events(inputs, parents + parents)
    other = replace(parents[0], envelope=replace(parents[0].envelope, scope_id="other-scope"))
    results = build_events(inputs, parents + (other,))
    assert all(result.metric("cash_cover") is None for result in results)
    future = replace(
        parents[0],
        envelope=replace(
            parents[0].envelope,
            analysis_as_of="2026-10-04T00:00:00Z",
            generated_at="2026-10-04T00:00:00Z",
        ),
    )
    assert len(build_events(inputs, (future,))[0].parent_result_ids) == 0


def test_empty_events_remain_incomplete_and_without_car():
    """场景：没有事件和父证据；预期：明确缺来源状态，CAR为空而不是0。"""
    result = build_events(research_inputs(0), ())[0]
    assert result.envelope.state is ProductState.INCOMPLETE
    assert result.details["event_count"] == 0
    assert result.details["car"] is None


@pytest.mark.parametrize(
    "content",
    [
        "利润变动原因尚未查明。",
        "可能由于成本上升。",
        "利润减少并非因为订单变化。",
        "公司未披露原因。",
    ],
)
def test_unconfirmed_reason_text_cannot_become_documented_reason(content):
    """场景：原文没有给出、否认或猜测原因；预期：仅保留证据，不输出明确原因归因。"""
    inputs = event_inputs(research_inputs(1), "major_news", {"content": content})
    result = build_events(inputs, ())[0]
    assert result.details["attribution_level"] == "evidence_only"
    assert result.details["evidence_spans"] == []


def test_upstream_hard_checks_deduplicate_by_source_and_preserve_strictest_result():
    """场景：同来源同门禁重复且结果冲突；预期：只保留一条失败硬门禁，软门禁不升级。"""
    inputs = research_inputs(1)
    fundamental, governance, flow = thesis_parents(inputs)
    references = inputs.rows("daily")[0].evidence("amount")
    checks = (
        gate("flow_price", Truth.TRUE, ""),
        gate("flow_price", Truth.UNKNOWN, "not_checked"),
        gate("flow_price", Truth.FALSE, "price_unavailable", references=references),
        gate("optional_hint", Truth.FALSE, "hint_unavailable", hard=False),
    )
    flow = replace(flow, envelope=replace(flow.envelope, quality_gates=checks))
    for result in build_events(inputs, (fundamental, governance, flow)):
        output = [
            check
            for check in result.envelope.quality_gates
            if check.gate_id.startswith("flow_feasibility/v1/")
        ]
        assert len(output) == 1
        assert output[0].result is Truth.FALSE
        assert output[0].reason_codes == ("not_checked", "price_unavailable")
        assert output[0].evidence_refs == references
        assert result.envelope.action_state is ActionState.REVIEW_REQUIRED


def test_ambiguous_event_does_not_inherit_any_candidate_entity_risk():
    """场景：新闻匹配两个实体且其中一家有硬风险；预期：新闻保持待确认，不强归因风险。"""
    inputs = event_inputs(research_inputs(2), "major_news", {"title": "同名企业发布公告"})
    aliases = tuple(entity_alias(inputs, identity, "同名企业") for identity in inputs.entity_ids)
    inputs = replace(
        inputs,
        config=replace(
            inputs.config, foundation=replace(inputs.config.foundation, aliases=aliases)
        ),
    )
    parents = thesis_parents(inputs, risk=Truth.FALSE)
    result = next(
        result
        for result in build_events(inputs, parents)
        if result.details["source_api"] == "major_news"
    )
    assert result.details["confirmed_entity"] is None
    assert result.details["conflict_reason"] == "entity_ambiguous"
    assert result.envelope.state is ProductState.INCOMPLETE
    assert result.parent_result_ids == ()
    assert len(result.envelope.quality_gates) == 1
