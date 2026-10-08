"""产品公共契约的先行离线测试，合成证据不表示真实数据已获认证。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import (
    ActionState,
    ConfidenceComponents,
    DecisionEnvelope,
    EvidenceRef,
    ProductState,
    QualityGateResult,
    Truth,
)
from tushare_entitlement_hub.product_foundation.entities import (
    Entity,
    EntityAlias,
    EntityIdentity,
    EntityKind,
    EntityRelation,
    resolve_alias,
)
from tushare_entitlement_hub.product_foundation.features import (
    FeatureValue,
    FeatureWeight,
    PeerObservation,
    ScorePolicy,
    percentile_ranks,
    rank_scores,
    score_features,
    weight_stability,
)
from tushare_entitlement_hub.product_foundation.numeric import (
    TypedFact,
    UnitCertification,
    adapt_numeric,
    arithmetic,
    price_gate,
)
from tushare_entitlement_hub.product_foundation.signals import Predicate, SignalFact, evaluate
from tushare_entitlement_hub.product_foundation.temporal import (
    AvailabilityFact,
    availability_gate,
    public_time_boundary,
)

NOW = "2026-10-02T00:00:00Z"
ENTITY = "cn:security:A.SZ:listing1"


def evidence(**changes):
    """构造可追溯的合成证据；输入字段覆盖，输出 daily 来源引用。"""
    values = dict(
        source_api="daily",
        source_record_id="row1",
        business_group_key="group",
        business_variant_key=None,
        contract_hash="contract",
        fields=("close",),
        observed_at=NOW,
        locator="fixture:row1",
    )
    return EvidenceRef(**(values | changes))


def availability(**changes):
    """构造观察后可用的合成事实；默认输入，输出未认证历史 PIT 的时间事实。"""
    values = dict(
        source_api="daily",
        source_record_id="row1",
        certification_version="v1",
        observed_at=NOW,
        data_as_of="2026-09-30",
    )
    return AvailabilityFact(**(values | changes))


def feature(name="quality", value=0.5, **changes):
    """构造合成归一特征；输入 value=0.5，输出同一研究时点的特征记录。"""
    values = dict(
        entity_id=ENTITY,
        feature_id=name,
        analysis_as_of=NOW,
        scope_id="sample",
        window="spot",
        feature_profile="base",
        version="v1",
        value=value,
        unit="ratio",
        input_count=1,
        expected_count=1,
        evidence_refs=(evidence(),),
    )
    return FeatureValue(**(values | changes))


def test_envelope_roundtrip_and_research_only():
    """场景：未知证据的研究输出；预期：可序列化、身份稳定且不授予历史交易资格。

    输入未评分信封，序列化再读取后与原对象相等；decision_allowed 保持 False。
    """
    envelope = DecisionEnvelope(
        product_id="foundation",
        entity_id=ENTITY,
        scope_id="sample",
        run_id="run1",
        model_version="v1",
        analysis_as_of=NOW,
        generated_at=NOW,
        data_as_of="2026-09-30",
        state=ProductState.INCOMPLETE,
        action_state=ActionState.NO_CONCLUSION,
        gate_codes=("unit_unverified",),
        evidence_refs=(evidence(),),
    )
    assert envelope.schema_version == "DecisionEnvelope/v1"
    assert DecisionEnvelope.from_dict(envelope.to_dict()) == envelope
    assert DecisionEnvelope.from_json(envelope.to_json()) == envelope
    assert not envelope.historical_pit_certified and not envelope.decision_allowed
    assert envelope.result_id == replace(envelope, generated_at="2026-10-03T00:00:00Z").result_id
    with pytest.raises(ValueError):
        replace(envelope, decision_allowed=True)


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf"), True])
def test_score_and_confidence_reject_invalid_numbers(value):
    """场景：越界、非有限或布尔分数；预期：全部拒绝，不能发布为合法评分。

    输入 101、NaN、Infinity 或 True，输出 ValueError。
    """
    with pytest.raises(ValueError):
        ConfidenceComponents(quality=value, coverage=80, mapping=80, temporal=80)


@pytest.mark.parametrize(
    "change",
    [{"source_api": ""}, {"observed_at": "2026-10-02"}, {"source_record_id": " "}, {"fields": ()}],
)
def test_evidence_requires_identity_time_and_fields(change):
    """场景：证据缺身份、时区或字段；预期：在模型边界失败。

    输入空 source_api 或没有时区的观察日期，输出 ValueError。
    """
    with pytest.raises(ValueError):
        evidence(**change)


def test_gate_unknown_and_reason_validation():
    """场景：未知门禁和非法原因码；预期：未知不通过，错误原因码被拒绝。

    输入 UNKNOWN 门禁，passed 返回 False；输入空原因码则拒绝。
    """
    gate = QualityGateResult(
        gate_id="unit", result=Truth.UNKNOWN, reason_codes=("unit_unverified",)
    )
    assert not gate.passed
    with pytest.raises(ValueError):
        replace(gate, reason_codes=("",))
    with pytest.raises(ValueError):
        replace(gate, reason_codes=())


def test_entities_alias_lifecycle_and_ambiguity():
    """场景：证券代码在不同生命周期复用；预期：按区间唯一匹配，歧义不猜选。

    输入 A.SZ 的唯一有效别名，输出统一实体；两个有效目标则拒绝。
    """
    identity = EntityIdentity(
        namespace="cn", kind=EntityKind.SECURITY, local_id="A.SZ", lifecycle_id="listing1"
    )
    entity = Entity(identity=identity, name="合成证券")
    assert identity.entity_id == ENTITY
    assert Entity.from_json(entity.to_json()) == entity
    alias = EntityAlias(
        namespace="tushare",
        alias="A.SZ",
        entity_id=ENTITY,
        effective_from="2020-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="fixture",
        version="v1",
        evidence_refs=(evidence(),),
        confidence=100,
    )
    assert resolve_alias((alias,), "tushare", "A.SZ", NOW) == ENTITY
    assert resolve_alias((alias,), "other", "A.SZ", NOW) is None
    assert resolve_alias((alias,), "tushare", "A.SZ", "2019-01-01T00:00:00Z") is None
    other = replace(alias, entity_id="us:security:A.SZ:listing2")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_alias((alias, other), "tushare", "A.SZ", NOW)
    with pytest.raises(ValueError):
        replace(alias, effective_to="2010-01-01T00:00:00Z")
    relation = EntityRelation(
        from_entity_id=ENTITY,
        to_entity_id="cn:company:org:life",
        relation_type="issued_by",
        effective_from=NOW,
        effective_to=None,
        observed_at=NOW,
        source="fixture",
        version="v1",
        evidence_refs=(evidence(),),
        confidence=90,
        confirmed=True,
    )
    assert relation.confirmed


def test_public_time_and_observed_availability():
    """场景：日期级公告、晚采旧数据及未来生效事件；预期：保守时间且可见与生效分开。

    输入上海市场 2026-09-30 公告日，输出 2026-09-30T16:00:00Z。
    """
    assert public_time_boundary("2026-09-30", "Asia/Shanghai") == "2026-09-30T16:00:00Z"
    fact = availability(
        public_available_at="2026-09-30T16:00:00Z",
        effective_from="2026-11-01T00:00:00Z",
        report_period="2026-Q3",
    )
    assert availability_gate(fact, NOW).passed
    assert not availability_gate(fact, NOW, require_effective=True).passed
    assert not availability_gate(fact, "2026-09-30T23:00:00Z").passed
    assert not availability_gate(fact, NOW, mode="historical").passed
    assert (
        "historical_pit_uncertified" in availability_gate(fact, NOW, mode="historical").reason_codes
    )
    assert availability_gate(availability(), NOW).passed


def test_certified_units_avoid_double_conversion_and_unknowns():
    """场景：标准层已转元、原值万元及未知单位；预期：只转换一次，未知值保持空。

    输入已标准化的 2000 元，输出仍为 2000 元；缺证书输出空值。
    """
    rule = UnitCertification(
        version="v1",
        source_api="daily",
        field="amount",
        source_unit="千元",
        target_unit="元",
        dimension="money",
        currency="CNY",
        multiplier=1000,
        evidence_refs=(evidence(),),
    )
    row = {
        "amount": 2000,
        "source_api": "daily",
        "source_record_id": "row1",
        "business_group_key": "group",
        "contract_hash": "contract",
        "standard_units": {"amount": "元"},
        "quality_tier": "verified",
    }
    fact = adapt_numeric(row, "amount", ENTITY, availability(), rule)
    assert fact.value == 2000 and fact.unit == "元"
    assert (
        adapt_numeric(
            row | {"amount": 2, "standard_units": {"amount": "千元"}},
            "amount",
            ENTITY,
            availability(),
            rule,
        ).value
        == 2000
    )
    assert adapt_numeric(row, "amount", ENTITY, availability(), None).value is None
    assert (
        adapt_numeric(
            row | {"quality_tier": "quarantine"}, "amount", ENTITY, availability(), rule
        ).value
        is None
    )
    assert arithmetic(fact, fact, "divide").value == 1
    assert arithmetic(fact, fact, "add").value == 4000
    with pytest.raises(ValueError):
        arithmetic(fact, replace(fact, currency="USD"), "add")
    with pytest.raises(ValueError):
        arithmetic(fact, replace(fact, value=0), "divide")
    assert TypedFact.from_json(fact.to_json()) == fact


@pytest.mark.parametrize("purpose", ["return", "volatility", "price_rank", "feasibility"])
def test_reference_close_never_used_as_traded_price(purpose):
    """场景：有参考收盘但无成交；预期：收益、波动、价格排名及可行性全部拦截。

    输入 close=10、vol=0、amount=0，四类用途均返回不通过。
    """
    row = {
        "close": 10,
        "vol": 0,
        "amount": 0,
        "price_observation_status": "reference_close_without_trade",
    }
    assert not price_gate(row, purpose).passed
    assert price_gate({"close": 10, "vol": 1, "amount": 10}, purpose).passed


def test_percentiles_ties_fallback_and_profile_isolation():
    """场景：同值、小行业样本及不同特征档；预期：平均秩、上级回退且不跨档计数。

    输入 [10,10,20]，输出 [0.25,0.25,1]；换档样本不计入分母。
    """
    policy = FoundationPolicy(version="test", minimum_peer_count=3)
    rows = tuple(
        PeerObservation(
            entity_id=f"cn:security:{i}:life",
            value=value,
            analysis_as_of=NOW,
            feature_profile="base",
            scope_id="sample",
            peer_path=("market", group),
        )
        for i, value, group in [(0, 10, "a"), (1, 10, "a"), (2, 20, "b")]
    )
    ranked = percentile_ranks(rows, policy)
    assert ranked[rows[0].entity_id].percentile == 0.25
    assert ranked[rows[2].entity_id].percentile == 1
    assert ranked[rows[0].entity_id].peer_level == "market"
    assert (
        percentile_ranks(tuple(replace(row, value=10) for row in rows), policy)[
            rows[0].entity_id
        ].percentile
        == 0.5
    )
    mixed = rows[:2] + (replace(rows[2], feature_profile="other"),)
    assert percentile_ranks(mixed, policy)[rows[0].entity_id].percentile is None


def test_weighted_missing_threshold_contributions_and_stability():
    """场景：加权评分和缺核心项；预期：贡献可复算、缺失门槛生效并报告扰动稳定性。

    输入权重 0.8/0.2、值 0.5/1，输出 60 分及四个扰动场景。
    """
    config = FoundationPolicy(version="test", minimum_peer_count=2, top_k=1)
    policy = ScorePolicy(
        version="v1",
        weights=(
            FeatureWeight(feature_id="quality", weight=0.8, required=True),
            FeatureWeight(feature_id="liquidity", weight=0.2),
        ),
        thresholds=config,
    )
    features = (feature(value=0.5), feature("liquidity", 1))
    result = score_features(features, policy)
    assert result.score == pytest.approx(60)
    assert sum(item.contribution for item in result.components) == pytest.approx(result.score)
    missing = score_features((features[0],), policy)
    assert missing.score == 50 and missing.feature_profile != result.feature_profile
    assert score_features((features[1],), policy).score is None
    other = tuple(replace(item, entity_id="cn:security:B:life", value=0.6) for item in features)
    ranked = rank_scores((result, score_features(other, policy), missing), config)
    assert ranked[missing.result_id].rank is None
    stability = weight_stability((features, other), policy)
    assert len(stability.scenarios) == 4
    assert stability.rank_intervals


def test_three_valued_predicates_and_signal_evidence():
    """场景：条件缺数或需人工确认；预期：未知传播，信号保留反证及后续复核条件。

    输入阈值 0.5 与值 0.8，返回 TRUE；缺输入返回 UNKNOWN。
    """
    predicate = Predicate(
        predicate_id="threshold", version="v1", feature_id="quality", operator="ge", threshold=0.5
    )
    assert evaluate(predicate, {"quality": 0.8}) == Truth.TRUE
    assert evaluate(predicate, {"quality": 0.2}) == Truth.FALSE
    assert evaluate(predicate, {}) == Truth.UNKNOWN
    conjunction = Predicate(
        predicate_id="all",
        version="v1",
        operator="all",
        children=(predicate, replace(predicate, feature_id="missing")),
    )
    assert evaluate(conjunction, {"quality": 0.8}) == Truth.UNKNOWN
    assert evaluate(conjunction, {"quality": 0.2}) == Truth.FALSE
    signal = SignalFact(
        entity_id=ENTITY,
        rule_id="signal",
        rule_version="v1",
        triggered_at=NOW,
        predicate=predicate,
        result=Truth.UNKNOWN,
        evidence_refs=(evidence(),),
        contrary_evidence=(evidence(),),
        invalidation_conditions=(predicate,),
        next_check_at="2026-10-03T00:00:00Z",
    )
    assert SignalFact.from_json(signal.to_json()) == signal
