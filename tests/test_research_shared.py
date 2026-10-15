"""研究共享输入、数值门禁、证据与信封构造的离线测试。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.equity_research.common import (
    gate,
    latest,
    make_result,
    metric,
    percentiles,
    unique_evidence,
)
from tushare_entitlement_hub.equity_research.models import (
    RULE_VERSION,
    Metric,
    ResearchConfig,
    ResearchInputs,
    ResearchResult,
    ResearchRow,
    default_thresholds,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import (
    ActionState,
    FeatureContribution,
    Predicate,
    ProductState,
    Truth,
)
from tushare_entitlement_hub.product_foundation.entities import (
    Entity,
    EntityIdentity,
    EntityKind,
    EntityRelation,
)
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-02T12:00:00Z"
IDENTITY = "cn:security:A:listed"


def source(
    record="row",
    value=10.0,
    *,
    api="daily",
    entity=IDENTITY,
    data="2026-09-30",
    public="2026-09-30T16:00:00Z",
    currency="CNY",
):
    """构造认证金额证据；例：默认输入得到daily.amount=10元的可见来源行。"""
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="fixture-time",
        observed_at=NOW,
        data_as_of=data,
        public_available_at=public,
    )
    row = ResearchRow(
        entity_id=entity,
        source_api=api,
        raw={
            "amount": value,
            "source_api": api,
            "source_record_id": record,
            "business_group_key": record,
            "contract_hash": "fixture-contract",
        },
        availability=availability,
    )
    references = row.evidence("amount")
    certificate = UnitCertification(
        version="fixture-unit",
        source_api=api,
        field="amount",
        source_unit="元",
        target_unit="元",
        dimension="money",
        currency=currency,
        multiplier=1,
        evidence_refs=references,
    )
    fact = TypedFact(
        fact_id=record,
        entity_id=entity,
        field="amount",
        value=value,
        unit="元",
        dimension="money",
        currency=currency,
        availability=availability,
        quality_tier="verified",
        certification=certificate,
        evidence_refs=references,
    )
    return replace(row, numeric={"amount": fact})


def inputs(rows=(), config=None):
    """构造最小共享研究输入；例：空行得到无已知来源的合法输入。"""
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="shared-fixture",
        config=config or ResearchConfig(),
        rows_by_api={
            api: tuple(row for row in rows if row.source_api == api)
            for api in {row.source_api for row in rows}
        },
    )


def test_research_config_defaults_roundtrip_and_independent_thresholds():
    """场景：默认配置及序列化回放；预期：认证为空、阈值30且每次独立构造。"""
    config = ResearchConfig()
    assert config.version == RULE_VERSION
    assert config.thresholds.minimum_peer_count == 30
    assert config.maximum_rows_per_entity_source == 512
    assert config.maximum_source_rows == 32768
    assert config.maximum_total_rows == 131072
    assert config == ResearchConfig.from_json(config.to_json())
    assert default_thresholds() is not default_thresholds()


@pytest.mark.parametrize(
    "change",
    [
        {"version": "unsupported"},
        {"maximum_rows_per_entity_source": 0},
        {"maximum_source_rows": 0},
        {"maximum_total_rows": 0},
        {"peer_paths": {IDENTITY: ()}},
    ],
)
def test_research_config_rejects_unusable_boundaries(change):
    """场景：版本不支持、零容量或空同行路径；预期：配置创建立即失败。"""
    with pytest.raises(ValueError, match="invalid_research_configuration"):
        ResearchConfig(**change)


def test_research_row_numeric_never_falls_back_to_raw():
    """场景：已认证、字段缺失、单位错配、有原因及外币数值；预期：仅合格人民币值可读。"""
    row = source()
    assert row.number("amount") == row.number("amount", "元") == 10
    assert row.number("absent") is None
    assert row.number("amount", "股") is None
    assert replace(row, numeric={}).number("amount") is None
    invalid = replace(row.numeric["amount"], reason_codes=("unit_unverified",))
    assert replace(row, numeric={"amount": invalid}).number("amount") is None
    assert source(currency="HKD").number("amount") is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "1"),
        (1.5, "1.5"),
        ("类型", "类型"),
        (" ", None),
        (None, None),
        (True, None),
        ({}, None),
        ([], None),
    ],
)
def test_research_row_text_keeps_only_real_scalar_identifiers(value, expected):
    """场景：文本、数值编码和非标识对象；预期：标量变文本，空值与结构化对象保持未知。"""
    row = source()
    assert replace(row, raw=row.raw | {"type": value}).text("type") == expected
    assert row.text("absent") is None


def test_research_row_evidence_requires_real_fields_and_identity():
    """场景：请求不存在字段或错配来源身份；预期：缺字段不造证据，错配直接拒绝。"""
    row = source()
    assert row.evidence("amount", "absent") == row.evidence("amount")
    assert row.evidence("absent") == ()
    with pytest.raises(ValueError, match="source_identity_mismatch"):
        replace(row, raw=row.raw | {"source_record_id": "other"}).evidence("amount")


def test_inputs_filter_entities_rows_and_merge_reason_codes():
    """场景：股票和宏观系列共存，接口及实体分别有缺口；预期：只登记股票并去重合并原因。"""
    stock = Entity(
        identity=EntityIdentity(
            namespace="cn", kind=EntityKind.SECURITY, local_id="A", lifecycle_id="listed"
        ),
        name="股票",
    )
    series = Entity(
        identity=EntityIdentity(
            namespace="cn", kind=EntityKind.SERIES, local_id="pmi", lifecycle_id="v1"
        ),
        name="序列",
    )
    row = source()
    other = source("other", entity="cn:security:B:listed")
    context = replace(
        inputs(
            (row, other), ResearchConfig(foundation=FoundationBindings(entities=(series, stock)))
        ),
        source_reasons={"daily": ("source_gap",)},
        entity_source_reasons={"daily/" + IDENTITY: ("source_gap", "unit_gap")},
    )
    assert context.entity_ids == (IDENTITY,)
    assert context.rows("daily") == (row, other)
    assert context.rows("daily", IDENTITY) == (row,)
    assert context.rows("absent") == ()
    assert context.reasons("daily", IDENTITY) == ("source_gap", "unit_gap")
    assert context.reasons("absent", IDENTITY) == ()


@pytest.mark.parametrize(
    "value,unit", [(1.1, "fraction"), (-0.1, "fraction"), (101, "score"), (-1, "score")]
)
def test_metrics_reject_out_of_range_scores_and_fractions(value, unit):
    """场景：广度或评分越界；预期：拒绝输出，原始ratio另有宽范围。"""
    with pytest.raises(ValueError, match="metric_range_invalid"):
        Metric(name="test", value=value, unit=unit, evidence_refs=source().evidence("amount"))


@pytest.mark.parametrize(
    "change",
    [
        {"value": None},
        {"evidence_refs": ()},
        {"status": "unknown", "value": 0.0, "reason_codes": ("missing",)},
        {"status": "unknown", "value": None, "reason_codes": ()},
    ],
)
def test_metric_evidence_and_unknown_value_invariants(change):
    """场景：已知值无证据或未知状态携带伪零；预期：统一模型拒绝。"""
    values = {
        "name": "test",
        "value": 1.0,
        "unit": "ratio",
        "evidence_refs": source().evidence("amount"),
    }
    with pytest.raises(ValueError):
        Metric(**(values | change))


def test_metric_builder_keeps_negative_ratio_and_explicit_missing_states():
    """场景：负比率、未知和行业不适用；预期：负值保留，另外两种空状态带独立原因。"""
    assert metric("cash", -0.5, "ratio", source().evidence("amount")).value == -0.5
    assert metric("cash", None, "ratio").reason_codes == ("metric_unavailable",)
    item = metric("cash", None, "ratio", reasons=("industry", "industry"), not_applicable=True)
    assert item.status == "not_applicable"
    assert item.reason_codes == ("industry",)


def test_unique_evidence_consumes_generators_once_and_preserves_sources():
    """场景：一次性迭代器含重复和不同API证据；预期：去重后保留不同来源，顺序确定。"""
    first, second = source().evidence("amount")[0], source(api="income").evidence("amount")[0]
    assert unique_evidence(ref for ref in (first, second, first)) == unique_evidence(
        (second, first)
    )
    assert len(unique_evidence((first, second))) == 2


@pytest.mark.parametrize(
    "truth,hard", [(Truth.TRUE, True), (Truth.FALSE, True), (Truth.UNKNOWN, False)]
)
def test_gate_builder_preserves_three_valued_logic(truth, hard):
    """场景：真、假和未知门禁；预期：只有真不带失败原因，硬软属性保持原样。"""
    check = gate("test", truth, "reason", hard=hard, references=source().evidence("amount"))
    assert check.result is truth
    assert check.hard is hard
    assert check.reason_codes == (() if truth is Truth.TRUE else ("reason",))


def test_latest_selects_unique_date_and_public_version():
    """场景：多个日期、同日修订、重复来源及同刻冲突；预期：只返回唯一最新版本。"""
    original = source("old", data="2026-09-29")
    current = source("new")
    assert latest(()) is None
    assert latest((current, original, current)) == current
    assert latest((current, source("conflict"))) is None
    unknown = source("unknown", data=None, public=None)
    assert latest((unknown, current)) == current
    earlier = source("earlier", public="2026-09-29T16:00:00Z")
    assert latest((earlier, current)) == current


def test_percentiles_require_explicit_peers_and_reverse_low_values():
    """场景：两条认证同行加一个无路径实体；预期：后者不计分母，低值方向可显式反转。"""
    first, second, unknown = (f"cn:security:{code}:listed" for code in ("A", "B", "unknown"))
    context = inputs(
        config=ResearchConfig(
            peer_paths={first: ("industry",), second: ("industry",)},
            thresholds=FoundationPolicy(version="test", minimum_peer_count=2),
        )
    )
    direct = percentiles(context, {first: 1.0, second: 2.0, unknown: 100.0}, "test")
    reverse = percentiles(context, {first: 1.0, second: 2.0}, "test", lower_is_better=True)
    assert direct[first].percentile == 0
    assert direct[second].peer_count == 2
    assert direct[unknown].percentile is None
    assert reverse[first].percentile == 1


@pytest.mark.parametrize(
    "checks,values,state,action",
    [
        ((), (), ProductState.ANNOTATED, ActionState.WATCH),
        (
            (gate("hard", Truth.FALSE, "blocked"),),
            (),
            ProductState.BLOCKED,
            ActionState.REVIEW_REQUIRED,
        ),
        (
            (gate("hard", Truth.UNKNOWN, "unknown"),),
            (),
            ProductState.INCOMPLETE,
            ActionState.NO_CONCLUSION,
        ),
        (
            (gate("soft", Truth.FALSE, "soft", hard=False),),
            (),
            ProductState.ANNOTATED,
            ActionState.WATCH,
        ),
        (
            (),
            (metric("missing", None, "ratio"),),
            ProductState.INCOMPLETE,
            ActionState.NO_CONCLUSION,
        ),
    ],
)
def test_result_state_uses_gates_and_missing_features(checks, values, state, action):
    """场景：硬失败、未知门禁、软警告和缺特征；预期：状态分别传播且默认不给研究权限。"""
    result = make_result(
        inputs(),
        "test/v1",
        IDENTITY,
        gates=checks,
        metrics=values,
        references=source().evidence("amount"),
    )
    assert result.envelope.state is state
    assert result.envelope.action_state is action
    assert not result.envelope.decision_allowed
    assert "historical_pit_uncertified" in result.envelope.gate_codes


def test_result_without_evidence_is_incomplete():
    """场景：没有任何已知证据；预期：信封保持无结论且分数为空。"""
    result = make_result(inputs(), "test/v1", IDENTITY)
    assert result.envelope.state is ProductState.INCOMPLETE
    assert result.envelope.score is None


def test_result_becomes_ready_only_with_explicit_readiness_certificate():
    """场景：完整证据分别未声明和已声明产品就绪；预期：前者 annotated、后者 ready。"""
    references = source().evidence("amount")
    annotated = make_result(inputs(), "test/v1", IDENTITY, references=references)
    ready = make_result(
        inputs(), "test/v1", IDENTITY, references=references, readiness_certified=True
    )
    assert annotated.envelope.state is ProductState.ANNOTATED
    assert ready.envelope.state is ProductState.READY


def test_result_contributions_lineage_and_json_roundtrip():
    """场景：带贡献、三值谓词、父结果、来源缺口的完整结果；预期：加总正确且可无损回放。"""
    row = source()
    refs = row.evidence("amount")
    contribution = FeatureContribution(
        feature_id="x",
        raw_value=10.0,
        unit="元",
        transformed_value=0.5,
        weight=1.0,
        contribution=50.0,
        evidence_refs=refs,
    )
    condition = Predicate(predicate_id="ready", version="v1", operator="manual")
    context = replace(inputs((row,)), source_reasons={"income": ("not_bound",), "daily": ()})
    result = make_result(
        context,
        "test/v1",
        IDENTITY,
        metrics=(metric("amount", 10.0, "元", refs),),
        gates=(gate("good", Truth.TRUE, "unused", references=refs),),
        details={"description": "合成结果"},
        contributions=(contribution,),
        references=refs,
        contrary=refs,
        parents=("b", "a", "a"),
        predicate_results={"ready": Truth.UNKNOWN},
        entry_conditions=(condition,),
        invalidation_conditions=(condition,),
    )
    assert result.envelope.score == 50
    assert result.parent_result_ids == ("a", "b")
    assert result.envelope.unavailable_apis == ("income",)
    assert result.envelope.source_as_of == {"daily": "2026-09-30"}
    assert result.metric("amount") == 10
    assert result.metric("absent") is None
    assert ResearchResult.from_json(result.to_json()) == result
    details = result.details
    details["description"] = "修改副本"
    assert result.details["description"] == "合成结果"


def test_incomplete_contributions_do_not_publish_partial_score():
    """场景：贡献中一个已知一个未知；预期：贡献原值保留，总分为空。"""
    row = source()
    good = FeatureContribution(
        feature_id="good",
        raw_value=1.0,
        unit="ratio",
        transformed_value=1.0,
        weight=0.5,
        contribution=50.0,
        evidence_refs=row.evidence("amount"),
    )
    missing = FeatureContribution(
        feature_id="missing",
        raw_value=None,
        unit="ratio",
        transformed_value=None,
        weight=0.5,
        contribution=None,
        missing_reason="unavailable",
    )
    result = make_result(inputs((row,)), "test/v1", IDENTITY, contributions=(good, missing))
    assert result.envelope.score is None


@pytest.mark.parametrize(
    "details", ["[]", "null", '{"value":NaN}', '{"value":Infinity}', "invalid"]
)
def test_result_rejects_nonobject_invalid_or_nonfinite_details(details):
    """场景：扩展详情不是对象、非法JSON或含非有限数；预期：拒绝持久化。"""
    result = make_result(inputs(), "test/v1", IDENTITY)
    with pytest.raises(ValueError):
        replace(result, details_json=details)


def test_result_rejects_duplicate_feature_names():
    """场景：同一结果重复特征名称；预期：拒绝，避免metric读取时任意取第一条。"""
    item = metric("duplicate", None, "ratio")
    result = make_result(inputs(), "test/v1", IDENTITY)
    with pytest.raises(ValueError, match="research_result_details_invalid"):
        replace(result, metrics=(item, item))


def test_source_periods_only_include_used_evidence():
    """场景：输入有其他实体及未来日期行；预期：信封只报告实际已用证据的日期。"""
    older = source("older", data="2026-09-29")
    selected = source("selected")
    future = source("unused-future", data="2026-10-10")
    other = source("other", entity="cn:security:B:listed", data="2026-10-01")
    income = source("income", api="income", data="2026-06-30")
    undated = source("undated", api="events", data=None)
    context = inputs((older, selected, future, other, income, undated))
    references = tuple(
        ref for row in (older, selected, income, undated) for ref in row.evidence("amount")
    )
    result = make_result(context, "test/v1", IDENTITY, references=references)
    assert result.envelope.data_as_of == "2026-09-30"
    assert result.envelope.source_as_of == {"daily": "2026-09-30", "income": "2026-06-30"}
    assert ("events", "undated") not in context.record_periods
    empty = make_result(
        inputs((undated,)), "test/v1", IDENTITY, references=undated.evidence("amount")
    )
    assert empty.envelope.source_as_of == {}
    assert empty.envelope.data_as_of == NOW[:10]


def test_input_indexes_are_cached_references_and_rebuilt_after_replace():
    """场景：同API多实体和未知实体，随后替换输入批次；预期：索引复用原对象且新批次独立。"""
    first = source("first")
    second = source("second")
    unknown = replace(source("unknown"), entity_id=None)
    context = inputs((first, second, unknown))
    assert context.entity_rows[("daily", IDENTITY)] == (first, second)
    assert context.rows("daily", IDENTITY)[0] is first
    assert context.rows("daily", "cn:security:missing:listed") == ()
    assert context.rows("daily") == (first, second, unknown)
    assert context.record_periods is context.record_periods
    assert context.entity_rows is context.entity_rows
    assert all(key[1] is not None for key in context.entity_rows)
    new = replace(context, rows_by_api={"daily": (second,)})
    assert new.record_periods == {("daily", "second"): "2026-09-30"}
    assert new.entity_rows[("daily", IDENTITY)] == (second,)


def test_stock_universe_excludes_bonds_by_metadata_or_underlying_relation():
    """场景：债券也登记为security，但有转债元数据或正股关系；预期：股票范围剔除债券。"""
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind=EntityKind.SECURITY, local_id=code, lifecycle_id="listed"
            ),
            name=code,
        )
        for code in ("A", "metadata_bond", "related_bond", "other_stock")
    )
    metadata = source("metadata", api="cb_basic", entity=entities[1].identity.entity_id)
    relation = EntityRelation(
        from_entity_id=entities[2].identity.entity_id,
        to_entity_id=IDENTITY,
        relation_type="cb_underlying",
        confirmed=True,
        effective_from="2020-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="fixture",
        version="v1",
        confidence=100,
        evidence_refs=metadata.evidence("amount"),
    )
    unrelated = replace(
        relation, from_entity_id=entities[3].identity.entity_id, relation_type="same_issuer"
    )
    context = inputs(
        (metadata,),
        ResearchConfig(
            foundation=FoundationBindings(entities=entities),
            confirmed_relations=(relation, unrelated),
        ),
    )
    assert context.entity_ids == (IDENTITY, entities[3].identity.entity_id)
