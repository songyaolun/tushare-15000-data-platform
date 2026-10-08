"""公共产品基础层边界：模型输入、时间语义、统计门槛和证据闭环。"""

import json
from dataclasses import replace
from datetime import UTC, datetime

import duckdb
import pytest
from test_product_storage import bindings, reference_fixture

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.product_foundation.bindings import (
    EntityMapping,
    TimeMapping,
)
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy, load_policy
from tushare_entitlement_hub.product_foundation.contracts import (
    ActionState,
    Coverage,
    DecisionEnvelope,
    EvidenceRef,
    FeatureContribution,
    Predicate,
    ProductState,
    QualityGateResult,
    Truth,
)
from tushare_entitlement_hub.product_foundation.entities import (
    Entity,
    EntityAlias,
    EntityIdentity,
    EntityKind,
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
    UnitCertification,
    adapt_numeric,
    arithmetic,
    price_gate,
    source_evidence,
)
from tushare_entitlement_hub.product_foundation.reference import initialize_library
from tushare_entitlement_hub.product_foundation.reference_scan import map_availability
from tushare_entitlement_hub.product_foundation.signals import SignalFact, evaluate
from tushare_entitlement_hub.product_foundation.storage import ProductStore, validate_library
from tushare_entitlement_hub.product_foundation.storage_records import CoverageSlice
from tushare_entitlement_hub.product_foundation.temporal import (
    AvailabilityFact,
    availability_gate,
    public_time_boundary,
)
from tushare_entitlement_hub.reference_contract import load_reference_contracts

NOW = "2026-10-02T00:00:00Z"
ENTITY = "cn:security:A.SZ:listing1"


def sample_evidence(**changes):
    """构造确定性证据；输入 fields=('close',)，输出对应 daily 来源引用。

    默认时间为固定 UTC 时点，可用 changes 覆盖单个边界字段。
    """
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


def sample_envelope(**changes):
    """构造待补证据的研究信封；默认输入，输出 INCOMPLETE 且禁止行动的模型。

    例如 changes={'generated_at': NOW} 保持稳定的研究时点与结果身份。
    """
    values = dict(
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
        evidence_refs=(sample_evidence(),),
    )
    return DecisionEnvelope(**(values | changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_peer_count": 1},
        {"top_k": 0},
        {"weight_perturbation": 1},
        {"minimum_weight_coverage": -0.1},
        {"contribution_tolerance": 0},
        {"minimum_peer_count": -1},
        {"top_k": True},
        {"version": " "},
    ],
)
def test_policy_rejects_invalid_statistical_thresholds(changes):
    """场景：版本或统计阈值无效；预期：读取配置前阻止不合理计算参数。

    输入示例：minimum_peer_count=1；输出：ValueError。
    """
    with pytest.raises(ValueError):
        FoundationPolicy(**({"version": "v1"} | changes))


def test_policy_file_and_strict_model_deserialization(tmp_path):
    """场景：有效文件及非法 JSON 模型结构；预期：有效配置往返、错误结构明确拒绝。

    输入：含 version 的 JSON；输出：minimum_peer_count=30 的配置。
    """
    path = tmp_path / "policy.json"
    path.write_text('{"version":"v1"}', encoding="utf-8")
    policy = load_policy(path)
    assert policy.minimum_peer_count == 30
    assert FoundationPolicy.from_dict(policy.to_dict()) == policy
    for payload, error in [
        ("not-json", "invalid_model_json"),
        ("[]", "model_mapping_required"),
        ('{"version":"v1","extra":1}', "model_fields_invalid"),
        ("{}", "model_fields_invalid"),
    ]:
        with pytest.raises(ValueError, match=error):
            FoundationPolicy.from_json(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "automatic_trade"},
        {"schema_version": "DecisionEnvelope/v2"},
        {"source_as_of": []},
        {"evidence_refs": "row1"},
        {"data_as_of": True},
        {"generated_at": "2026-10-01T00:00:00Z"},
        {"next_check_at": "2026-10-01T00:00:00Z"},
        {"rank": 1},
        {"contribution_tolerance": 0},
        {"contribution_tolerance": 0.01},
        {"score": 10},
        {"historical_tradable": True},
        {"gate_codes": ()},
        {"entity_id": "cross_api_hash"},
    ],
)
def test_envelope_rejects_invalid_contract_state(changes):
    """场景：非法类型、身份、时间或研究状态；预期：JSON 边界拒绝发布。

    输入示例：rank=1 且 score 未知；输出：ValueError。
    """
    with pytest.raises(ValueError):
        sample_envelope(**changes)


@pytest.mark.parametrize("field", ["observed_at", "public_available_at"])
def test_envelope_rejects_future_supporting_or_contrary_evidence(field):
    """场景：支持或反证晚于研究时点；预期：不能把未来证据写入当前研究结论。

    输入：任一证据时间为2026-10-03；输出：future_evidence。
    """
    evidence = sample_evidence(**{field: "2026-10-03T00:00:00Z"})
    for changes in ({"evidence_refs": (evidence,)}, {"contrary_evidence": (evidence,)}):
        with pytest.raises(ValueError, match="future_evidence"):
            sample_envelope(**changes)


def test_envelope_contributions_and_research_permission():
    """场景：贡献一致且证据门禁齐备；预期：允许人工研究，贡献误差仍失败。

    输入：50 分唯一贡献与真门禁；输出：decision_allowed=True，非自动交易。
    """
    contribution = FeatureContribution(
        feature_id="quality",
        raw_value=0.5,
        unit="ratio",
        transformed_value=0.5,
        weight=1,
        contribution=50,
    )
    predicate = Predicate(predicate_id="review", version="v1", operator="manual")
    envelope = sample_envelope(
        state=ProductState.READY,
        action_state=ActionState.RESEARCH_FIRST,
        score=50,
        rank=1,
        peer_count=1,
        contributions=(contribution,),
        quality_gates=(QualityGateResult(gate_id="unit", result=Truth.TRUE),),
        entry_conditions=(predicate,),
        decision_allowed=True,
        source_as_of={"daily": "2026-09-30"},
    )
    assert envelope.decision_allowed and not envelope.historical_tradable
    assert DecisionEnvelope.from_json(envelope.to_json()) == envelope
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        replace(envelope, contributions=(replace(contribution, contribution=49),))
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        replace(envelope, contributions=(replace(contribution, contribution=None),))
    with pytest.raises(ValueError, match="invalid_rank"):
        replace(envelope, rank=2)
    with pytest.raises(ValueError, match="decision_not_allowed"):
        replace(
            envelope,
            quality_gates=(
                QualityGateResult(
                    gate_id="unit", result=Truth.UNKNOWN, reason_codes=("unit_unverified",)
                ),
            ),
        )


def test_evidence_coverage_and_truth_boundaries():
    """场景：重复字段、空位置、越界观测数和三值真值；预期：非法结构拒绝，未知为假。

    输入示例：observed_count=2、expected_count=1；输出：coverage_count_mismatch。
    """
    assert bool(Truth.TRUE)
    assert not bool(Truth.FALSE) and not bool(Truth.UNKNOWN)
    assert Coverage(observed_count=0, expected_count=0).observed_count == 0
    with pytest.raises(ValueError, match="coverage_count_mismatch"):
        Coverage(observed_count=2, expected_count=1)
    for changes in ({"fields": ("close", "close")}, {"locator": " "}):
        with pytest.raises(ValueError, match="evidence_fields_and_locator_required"):
            sample_evidence(**changes)
    assert sample_evidence(observed_at="2026-10-02T08:00:00+08:00").observed_at == NOW


@pytest.mark.parametrize(
    "changes",
    [
        {"operator": "all", "children": ()},
        {"operator": "ge", "feature_id": "quality", "threshold": None},
        {
            "operator": "all",
            "feature_id": "quality",
            "children": (Predicate(predicate_id="manual", version="v1", operator="manual"),),
        },
    ],
)
def test_predicate_rejects_incomplete_or_mixed_forms(changes):
    """场景：组合条件为空或混入比较参数；预期：规则模型拒绝歧义结构。

    输入示例：all 没有 children；输出：ValueError。
    """
    with pytest.raises(ValueError):
        Predicate(predicate_id="check", version="v1", **changes)


def sample_availability(**changes):
    """构造固定观察时点；默认输入，输出仅允许观察后研究的 daily 时间事实。

    示例：effective_from=NOW 可用于验证生效区间的左边界。
    """
    values = dict(
        source_api="daily",
        source_record_id="row1",
        certification_version="v1",
        observed_at=NOW,
        data_as_of="2026-09-30",
    )
    return AvailabilityFact(**(values | changes))


def test_temporal_unknown_expiry_and_invalid_inputs():
    """场景：未知生效日、区间到期及非法模式；预期：未知保留三值，到期关闭。

    输入：effective_from=None、require_effective=True；输出：UNKNOWN。
    """
    fact = sample_availability()
    gate = availability_gate(fact, NOW, require_effective=True)
    assert gate.result is Truth.UNKNOWN and gate.reason_codes == ("effective_time_unknown",)
    expired = replace(fact, effective_from="2026-09-01T00:00:00Z", effective_to=NOW)
    assert availability_gate(expired, NOW, require_effective=True).reason_codes == (
        "no_longer_effective",
    )
    assert availability_gate(replace(fact, effective_from=NOW), NOW, require_effective=True).passed
    with pytest.raises(ValueError, match="unknown_availability_mode"):
        availability_gate(fact, NOW, mode="trading")
    for changes in (
        {"historical_pit_certified": True},
        {"effective_to": NOW},
        {"effective_from": NOW, "effective_to": NOW},
    ):
        with pytest.raises(ValueError):
            replace(fact, **changes)
    assert public_time_boundary("2026-10-02T08:00:00+08:00", "Asia/Shanghai") == NOW
    for value, zone in [("bad", "UTC"), ("2026-10-02", "Invalid/Zone")]:
        with pytest.raises(ValueError, match="public_time_unverified"):
            public_time_boundary(value, zone)


def test_entity_alias_interval_observation_and_source_evidence():
    """场景：别名左闭右开及证据时间；预期：未公开或已过期不匹配，身份分隔冲突拒绝。

    输入：终止时点等于查询时点；输出：usable_at=False。
    """
    with pytest.raises(ValueError, match="ambiguous_entity_identity"):
        EntityIdentity(
            namespace="cn", kind=EntityKind.SECURITY, local_id="A:B", lifecycle_id="life"
        )
    alias = EntityAlias(
        namespace="tushare",
        alias="A.SZ",
        entity_id=ENTITY,
        effective_from=NOW,
        effective_to="2026-10-03T00:00:00Z",
        observed_at=NOW,
        source="fixture",
        version="v1",
        evidence_refs=(sample_evidence(),),
        confidence=100,
    )
    assert alias.usable_at(NOW)
    assert not alias.usable_at(alias.effective_to)
    assert not replace(
        alias, evidence_refs=(sample_evidence(public_available_at="2026-10-03T00:00:00Z"),)
    ).usable_at(NOW)
    for changes, error in [
        ({"alias": " "}, "alias_required"),
        ({"evidence_refs": ()}, "assertion_evidence_required"),
        ({"observed_at": "2026-10-01T00:00:00Z"}, "assertion_before_evidence"),
    ]:
        with pytest.raises(ValueError, match=error):
            replace(alias, **changes)


@pytest.mark.parametrize(
    "operator,values,result",
    [
        ("gt", [1, None], Truth.TRUE),
        ("ge", [0.5, None], Truth.TRUE),
        ("lt", [0.2, None], Truth.TRUE),
        ("le", [0.5, None], Truth.TRUE),
        ("eq", [0.5, None], Truth.TRUE),
        ("any", [1, None], Truth.TRUE),
        ("any", [0.2, None], Truth.UNKNOWN),
        ("any", [0.2, 0.2], Truth.FALSE),
        ("all", [1, 1], Truth.TRUE),
        ("not", [1], Truth.FALSE),
        ("not", [0.2], Truth.TRUE),
        ("not", [None], Truth.UNKNOWN),
    ],
)
def test_predicates_comparison_and_kleene_truth_table(operator, values, result):
    """场景：比较、析取和否定含未知值；预期：遵循三值逻辑，未知不冒充证据。

    输入示例：any(false, unknown)；输出：UNKNOWN。
    """
    children = tuple(
        Predicate(
            predicate_id=f"p{i}", version="v1", feature_id=f"f{i}", operator="ge", threshold=0.5
        )
        for i in range(len(values))
    )
    if operator in {"any", "all", "not"}:
        predicate = Predicate(
            predicate_id="root", version="v1", operator=operator, children=children
        )
    else:
        predicate = replace(children[0], operator=operator)
    assert evaluate(predicate, {f"f{i}": value for i, value in enumerate(values)}) is result


def test_manual_results_and_signal_time_guards():
    """场景：人工结论、坏数值及信号证据时间；预期：未确认未知，未来与缺证据拒绝。

    输入示例：人工条件未提供结论；输出：UNKNOWN。
    """
    manual = Predicate(predicate_id="manual", version="v1", operator="manual")
    assert evaluate(manual, {}) is Truth.UNKNOWN
    assert evaluate(manual, {}, {"manual": Truth.FALSE}) is Truth.FALSE
    with pytest.raises(ValueError, match="invalid_manual_result"):
        evaluate(manual, {}, {"manual": True})
    comparison = Predicate(
        predicate_id="p", version="v1", operator="ge", feature_id="x", threshold=0
    )
    for value in (True, float("nan"), float("inf")):
        assert evaluate(comparison, {"x": value}) is Truth.UNKNOWN
    signal = SignalFact(
        entity_id=ENTITY,
        rule_id="r",
        rule_version="v1",
        triggered_at=NOW,
        predicate=manual,
        result=Truth.UNKNOWN,
        evidence_refs=(sample_evidence(),),
        contrary_evidence=(),
        invalidation_conditions=(),
        next_check_at=None,
    )
    for changes, error in [
        ({"next_check_at": "2026-10-01T00:00:00Z"}, "next_check_before_signal"),
        ({"evidence_refs": ()}, "signal_evidence_required"),
        (
            {"evidence_refs": (sample_evidence(observed_at="2026-10-03T00:00:00Z"),)},
            "signal_before_evidence",
        ),
    ]:
        with pytest.raises(ValueError, match=error):
            replace(signal, **changes)


def numeric_inputs():
    """构造有来源身份的金额事实输入；无参数输入，输出合成行和千元转元证书。

    示例：返回行 amount=2 千元，经 adapt_numeric 输出 2000 元。
    """
    row = {
        "amount": 2,
        "source_api": "daily",
        "source_record_id": "row1",
        "business_group_key": "group",
        "contract_hash": "contract",
        "quality_tier": "verified",
        "standard_units": {"amount": "千元"},
    }
    certificate = UnitCertification(
        version="v1",
        source_api="daily",
        field="amount",
        source_unit="千元",
        target_unit="元",
        dimension="money",
        currency="CNY",
        multiplier=1000,
        evidence_refs=(sample_evidence(),),
    )
    return row, certificate


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"amount": None}, "missing_numeric_value"),
        ({"amount": True}, "invalid_numeric_value"),
        ({"amount": float("inf")}, "invalid_numeric_value"),
        ({"quality_tier": "new_tier"}, "quality_unknown"),
        ({"quality_tier": "usable_with_annotation"}, "annotation_not_adapted"),
        ({"is_quarantined": True}, "quality_unknown"),
        ({"business_variant_key": "variant1"}, "business_variant_unresolved"),
        ({"standard_units": {"amount": "美元"}}, "unit_mismatch"),
        ({"amount": 1e308}, "nonfinite_conversion"),
    ],
)
def test_numeric_adaptation_preserves_unusable_reasons(changes, reason):
    """场景：参考行缺值、注释未适配、单位错配或换算溢出；预期：空值附具体原因。

    输入示例：amount=1e308 千元；输出：value=None、nonfinite_conversion。
    """
    row, certificate = numeric_inputs()
    result = adapt_numeric(row | changes, "amount", ENTITY, sample_availability(), certificate)
    assert result.value is None and reason in result.reason_codes
    assert result.evidence_refs[0].source_record_id == "row1"


def test_numeric_certification_and_fact_boundaries():
    """场景：证书无币种、倍率错误或事实与证书冲突；预期：不产生伪认证数值。

    输入示例：money 证书 currency=None；输出：currency_unverified。
    """
    row, certificate = numeric_inputs()
    for changes, code in [
        ({"currency": None}, "currency_unverified"),
        ({"source_unit": ""}, "unit_evidence_required"),
        ({"evidence_refs": ()}, "unit_evidence_required"),
        ({"source_unit": "元"}, "same_unit_multiplier_invalid"),
    ]:
        with pytest.raises(ValueError, match=code):
            replace(certificate, **changes)
    fact = adapt_numeric(row, "amount", ENTITY, sample_availability(), certificate)
    for changes, code in [
        ({"value": None}, "missing_value_reason_required"),
        ({"quality_tier": "quarantine"}, "uncertified_value"),
        ({"unit": "USD"}, "fact_certification_mismatch"),
        ({"evidence_refs": ()}, "fact_evidence_required"),
    ]:
        with pytest.raises(ValueError, match=code):
            replace(fact, **changes)
    with pytest.raises(ValueError, match="source_identity_mismatch"):
        source_evidence(row | {"source_record_id": "different"}, "amount", sample_availability())
    with pytest.raises(ValueError, match="unit_certificate_field_mismatch"):
        adapt_numeric(row, "amount", ENTITY, sample_availability(), replace(certificate, field="x"))
    result = adapt_numeric(
        row | {"standard_units": {}, "source_units": {"amount": "千元"}},
        "amount",
        ENTITY,
        sample_availability(),
        certificate,
    )
    assert result.value == 2000


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"source_api": "repo_daily"}, "not_traded_price_type"),
        ({"price_type": "settlement"}, "not_traded_price_type"),
        ({"is_quarantined": True}, "quarantined_record"),
        ({"amount": None}, "price_activity_unknown"),
        ({"amount": 0}, "no_traded_price"),
    ],
)
def test_price_gate_distinguishes_traded_prices(changes, code):
    """场景：利率、结算、隔离或量额不足；预期：默认价格用途门禁失败。

    输入示例：price_type='settlement'；输出：not_traded_price_type。
    """
    row = {"close": 10, "vol": 1, "amount": 10}
    gate = price_gate(row | changes, "premium")
    assert gate.result is Truth.FALSE and code in gate.reason_codes
    with pytest.raises(ValueError, match="unknown_price_purpose"):
        price_gate(row, "trade_now")


def test_arithmetic_operations_contexts_and_nonfinite_results():
    """场景：四则运算与实体、币种、量纲不一致；预期：可比事实复算，不可比输入拒绝。

    输入：两份2000元事实；输出：subtract=0、multiply=4000000元*元。
    """
    row, certificate = numeric_inputs()
    fact = adapt_numeric(row, "amount", ENTITY, sample_availability(), certificate)
    assert arithmetic(fact, fact, "subtract").value == 0
    product = arithmetic(fact, fact, "multiply")
    assert product.value == 4_000_000 and product.unit == "元*元"
    ratio_certificate = replace(
        certificate,
        source_unit="ratio",
        target_unit="ratio",
        multiplier=1,
        dimension="ratio",
        currency=None,
    )
    ratio = adapt_numeric(
        row | {"standard_units": {"amount": "ratio"}},
        "amount",
        ENTITY,
        sample_availability(),
        ratio_certificate,
    )
    quotient = arithmetic(fact, ratio, "divide")
    assert quotient.value == 1000 and quotient.unit == "元/ratio" and quotient.currency == "CNY"
    dollars = replace(fact, currency="USD", certification=replace(certificate, currency="USD"))
    for right, operation, code in [
        (fact, "pow", "unknown_arithmetic_operation"),
        (replace(fact, reason_codes=("stale",)), "add", "numeric_fact_not_eligible"),
        (replace(fact, entity_id="cn:security:B:life"), "add", "numeric_context_mismatch"),
        (dollars, "add", "currency_mismatch"),
        (ratio, "add", "dimension_mismatch"),
        (replace(fact, value=0), "divide", "zero_denominator"),
    ]:
        with pytest.raises(ValueError, match=code):
            arithmetic(fact, right, operation)
    with pytest.raises(ValueError, match="invalid_finite_number"):
        arithmetic(replace(fact, value=1e308), replace(fact, value=1e308), "multiply")


def sample_feature(feature_id="quality", value=0.5, **changes):
    """构造固定截止和口径的特征；输入 value=0.5，输出有证据的归一特征。

    示例：feature_id='liquidity' 可与 quality 按配置权重组合。
    """
    values = dict(
        entity_id=ENTITY,
        feature_id=feature_id,
        analysis_as_of=NOW,
        scope_id="sample",
        window="spot",
        feature_profile="base",
        version="v1",
        value=value,
        unit="ratio",
        input_count=1,
        expected_count=1,
        evidence_refs=(sample_evidence(),),
    )
    return FeatureValue(**(values | changes))


def sample_score_policy():
    """构造等权双特征配置；无参数输入，输出最少2个同行、Top-1的评分政策。

    示例：quality=1、liquidity=0 的合法结果为50分。
    """
    return ScorePolicy(
        version="v1",
        weights=(
            FeatureWeight(feature_id="quality", weight=0.5),
            FeatureWeight(feature_id="liquidity", weight=0.5),
        ),
        thresholds=FoundationPolicy(version="v1", minimum_peer_count=2, top_k=1),
    )


def test_feature_validation_and_score_context_failures():
    """场景：缺失解释、未来证据、混合实体及重复特征；预期：模型或评分入口明确拒绝。

    输入示例：同一个特征重复传入两次；输出：score_feature_context_mismatch。
    """
    first = sample_feature()
    for changes, code in [
        ({"input_count": 2}, "feature_count_mismatch"),
        ({"value": None}, "feature_missing_reason_required"),
        ({"input_count": 0}, "feature_evidence_required"),
        (
            {"evidence_refs": (sample_evidence(observed_at="2026-10-03T00:00:00Z"),)},
            "feature_before_evidence",
        ),
    ]:
        with pytest.raises(ValueError, match=code):
            replace(first, **changes)
    policy = sample_score_policy()
    for features, code in [
        ((), "score_context_required"),
        ((first, first), "score_feature_context_mismatch"),
        ((replace(first, feature_id="unknown"),), "score_feature_context_mismatch"),
        (
            (first, sample_feature("liquidity", entity_id="cn:security:B:life")),
            "score_feature_context_mismatch",
        ),
        ((replace(first, value=1.01),), "score_feature_not_normalized"),
    ]:
        with pytest.raises(ValueError, match=code):
            score_features(features, policy)
    for weights in (
        (),
        (policy.weights[0], policy.weights[0]),
        (replace(policy.weights[0], weight=1e308), replace(policy.weights[1], weight=1e308)),
    ):
        with pytest.raises(ValueError, match="invalid_score_weights"):
            replace(policy, weights=weights)
    missing = score_features((replace(first, value=None, reason_codes=("missing",)),), policy)
    assert missing.score is None and missing.weight_coverage == 0
    assert "insufficient_weight_coverage" in missing.reason_codes


def test_percentile_exclusions_reverse_order_and_duplicate_guards():
    """场景：无资格、缺值、反向排名及重复实体；预期：排除坏样本，重复拒绝。

    输入：两个合格值1、2及一个空值；输出：反向分位1、0、None。
    """
    rows = tuple(
        PeerObservation(
            entity_id=f"cn:security:{i}:life",
            value=value,
            analysis_as_of=NOW,
            feature_profile="base",
            scope_id="sample",
            peer_path=("market",),
        )
        for i, value in enumerate([1, 2, None])
    )
    policy = sample_score_policy().thresholds
    results = percentile_ranks(rows, policy, lower_is_better=True)
    assert [results[row.entity_id].percentile for row in rows] == [1, 0, None]
    assert results[rows[2].entity_id].reason_codes == ("peer_value_unavailable",)
    assert percentile_ranks((), policy) == {}
    assert percentile_ranks((replace(rows[0], eligible=False),), policy)[
        rows[0].entity_id
    ].reason_codes == ("peer_value_unavailable",)
    with pytest.raises(ValueError, match="peer_path_required"):
        replace(rows[0], peer_path=())
    with pytest.raises(ValueError, match="duplicate_peer_entity"):
        percentile_ranks((rows[0], rows[0]), policy)


def test_stability_rank_flip_unavailable_samples_and_profile_guards():
    """场景：接近同分的权重扰动、空样本及混合档位；预期：排名翻转被标记敏感。

    输入：A的特征1/0、B的特征0/0.98；输出：部分扰动Top-1改变，sensitive=True。
    """
    policy = sample_score_policy()
    first = (sample_feature(value=1), sample_feature("liquidity", 0))
    second = (
        sample_feature(value=0, entity_id="cn:security:B:life"),
        sample_feature("liquidity", 0.98, entity_id="cn:security:B:life"),
    )
    result = weight_stability((first, second), policy)
    assert result.sensitive and result.reason_codes == ("weight_sensitive",)
    assert result.rank_intervals[ENTITY] == (1, 2)
    assert {scenario.top_k_jaccard for scenario in result.scenarios} == {0, 1}
    empty = weight_stability((), policy)
    assert empty.reason_codes == ("stability_sample_unavailable",) and not empty.rank_intervals
    assert all(scenario.top_k_jaccard is None for scenario in empty.scenarios)
    score = score_features(first, policy)
    with pytest.raises(ValueError, match="duplicate_score_result"):
        rank_scores((score, score), policy.thresholds)
    with pytest.raises(ValueError, match="stability_profile_mismatch"):
        weight_stability((first, second[:1]), policy)


def test_product_store_rejects_invalid_references_and_rolls_back(tmp_path):
    """场景：写入不匹配模型、悬空实体及批次异常；预期：明确拒绝，事务回滚。

    输入示例：未登记实体的 feature_value；输出：referenced_entity_missing，库仍为空。
    """
    path = tmp_path / "products.duckdb"
    store = ProductStore(path)
    assert store.metadata("missing") is None
    with pytest.raises(ValueError, match="product_table_model_mismatch"):
        store.put("entity", sample_feature())
    with pytest.raises(ValueError, match="referenced_entity_missing"):
        store.put("feature_value", sample_feature())
    with pytest.raises(ValueError, match="unknown_product_table"):
        store.records("unknown")
    store.close()
    entity = Entity(
        identity=EntityIdentity(
            namespace="cn", kind=EntityKind.SECURITY, local_id="A.SZ", lifecycle_id="listing1"
        ),
        name="合成证券",
    )
    with pytest.raises(RuntimeError, match="rollback_fixture"):
        with ProductStore(path) as transaction:
            transaction.put("entity", entity)
            raise RuntimeError("rollback_fixture")
    with ProductStore(path) as reopened:
        assert reopened.counts()["entity"] == 0
        identity = reopened.put("entity", entity)
        assert reopened.put("entity", entity) == identity
        with pytest.raises(ValueError, match="immutable_record_conflict"):
            reopened.put("entity", replace(entity, name="不同名称"))
        assert reopened.counts()["entity"] == 1
    with ProductStore(path, read_only=True) as readonly:
        assert readonly.records("entity") == [entity]


def test_product_store_schema_and_coverage_accounting_guards(tmp_path):
    """场景：产品库缺失、版本错误、计数不守恒；预期：初始化和记录边界拒绝。

    输入示例：source_rows=3但合格/隔离/重复合计2；输出：coverage_accounting_mismatch。
    """
    with pytest.raises(ValueError, match="product_library_missing"):
        ProductStore(tmp_path / "missing.duckdb", read_only=True)
    path = tmp_path / "products.duckdb"
    with ProductStore(path) as store:
        store.save_metadata("schema_version", "unknown/v2")
    with pytest.raises(ValueError, match="product_schema_version_mismatch"):
        ProductStore(path, read_only=True)
    complete = CoverageSlice(
        source_api="daily",
        scope_id="sample",
        partition="all",
        run_id="r1",
        request_scope="合成样本",
        source_rows=2,
        accepted_rows=1,
        quarantined_rows=1,
        duplicate_rows=0,
        observed_at=NOW,
        data_as_of="2026-09-30",
        manifest_hash="manifest",
        reason_codes=(),
    )
    assert complete.source_rows == 2
    assert replace(complete, source_rows=None).source_rows is None
    with pytest.raises(ValueError, match="coverage_accounting_mismatch"):
        replace(complete, source_rows=3)


def test_bindings_reject_duplicate_sources_and_time_mapping():
    """场景：重复来源绑定和显式时间映射；预期：拒绝歧义，公开与生效时间分别保留。

    输入：公开日2026-09-30、生效日2026-10-02、上海时区；输出不同UTC边界。
    """
    configured = bindings()
    for changes in (
        {"entities": configured.entities * 2},
        {"certifications": configured.certifications * 2},
        {"entity_mappings": configured.entity_mappings * 2},
        {"time_mappings": configured.time_mappings * 2},
    ):
        with pytest.raises(ValueError, match="duplicate_foundation_binding"):
            replace(configured, **changes)
    contract = load_reference_contracts(load_policies())["daily"]
    mapping = TimeMapping(
        source_api="daily",
        data_field="trade_date",
        public_fields=("announcement",),
        market_timezone="Asia/Shanghai",
        report_period_field="report_period",
        effective_from_field="effective_from",
        effective_to_field="effective_to",
    )
    row = dict(
        source_api="daily",
        source_record_id="row1",
        observed_at=NOW,
        historical_pit_certified=False,
        trade_date=datetime(2026, 9, 30, tzinfo=UTC),
        announcement="2026-09-30",
        report_period="2026-Q3",
        effective_from="2026-10-02",
        effective_to="2026-10-03",
    )
    fact = map_availability(row, contract, mapping)
    assert fact.public_available_at == "2026-09-30T16:00:00Z"
    assert fact.effective_from == "2026-10-01T16:00:00Z"
    assert fact.effective_to == "2026-10-02T16:00:00Z"
    assert fact.report_period == "2026-Q3" and fact.data_as_of == "2026-09-30"
    for value in (NOW, datetime(2026, 10, 2, tzinfo=UTC)):
        assert (
            map_availability(row | {"announcement": value}, contract, mapping).public_available_at
            == NOW
        )
    for changed_row, changed_mapping, code in [
        (row | {"announcement": None}, mapping, "public_time_missing"),
        (row, replace(mapping, market_timezone=None), "market_timezone_required"),
        (row | {"historical_pit_certified": None}, mapping, "source_pit_state_unverified"),
    ]:
        with pytest.raises(ValueError, match=code):
            map_availability(changed_row, contract, changed_mapping)


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("missing_table", "missing_table"),
        ("missing_column", "reference_fields_missing"),
        ("contract", "source_contract_mismatch"),
        ("accounting_invalid", "source_accounting_invalid"),
        ("accounting_mismatch", "source_accounting_mismatch"),
        ("incomplete", "collection_incomplete"),
        ("unknown_time", "observation_time_unknown"),
        ("future_time", "not_yet_observed"),
        ("stale_time", "stale_observation"),
    ],
)
def test_initialization_reports_source_degradation(tmp_path, scenario, reason):
    """场景：参考层表、字段、账本或新鲜度不合格；预期：摘要给出准确降级原因。

    输入示例：清单观测时间晚于研究截止；输出：not_yet_observed=1，不适配该接口。
    """
    reference = reference_fixture(tmp_path)
    with duckdb.connect(
        str(reference), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        manifest = json.loads(
            connection.execute("SELECT manifest FROM catalog WHERE api_name='daily'").fetchone()[0]
        )
        if scenario == "missing_table":
            connection.execute("DROP TABLE accepted.daily")
        elif scenario == "missing_column":
            connection.execute("ALTER TABLE accepted.daily DROP COLUMN close")
        elif scenario == "contract":
            manifest["contract_hash"] = "different"
        elif scenario == "accounting_invalid":
            manifest["quality"]["source_rows"] = -1
        elif scenario == "accounting_mismatch":
            manifest["quality"].update(
                source_rows=0, accepted_rows=0, quarantined_rows=0, exact_duplicate_rows=0
            )
        elif scenario == "incomplete":
            manifest["collection_status"] = "partial"
        elif scenario == "unknown_time":
            manifest["observed_at"] = "unknown"
        elif scenario == "future_time":
            manifest["observed_at"] = "2026-10-03T00:00:00Z"
        else:
            manifest["observed_at"] = "2026-09-01T00:00:00Z"
        connection.execute(
            "UPDATE catalog SET manifest=? WHERE api_name='daily'", [json.dumps(manifest)]
        )
    summary = initialize_library(
        reference, tmp_path / "products.duckdb", NOW, FoundationPolicy(version="v1")
    )
    assert summary["status"] == "incomplete"
    assert summary["reason_counts"][reason] == 1
    if scenario != "incomplete":
        assert summary["readable_apis"] == 0


@pytest.mark.parametrize("scenario", ["missing_catalog", "duplicate_catalog"])
def test_initialization_rejects_invalid_catalog(tmp_path, scenario):
    """场景：参考库缺目录或重复API目录项；预期：初始化失败，不生成部分产品库。

    输入示例：catalog 存在重复 daily；输出：reference_catalog_duplicate。
    """
    reference = reference_fixture(tmp_path)
    with duckdb.connect(
        str(reference), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        if scenario == "missing_catalog":
            connection.execute("DROP TABLE catalog")
        else:
            connection.execute("CREATE TABLE duplicate_catalog AS SELECT * FROM catalog")
            connection.execute("INSERT INTO duplicate_catalog SELECT * FROM catalog")
            connection.execute("DROP TABLE catalog")
            connection.execute("ALTER TABLE duplicate_catalog RENAME TO catalog")
    output = tmp_path / "products.duckdb"
    expected = (
        "reference_catalog_missing"
        if scenario == "missing_catalog"
        else "reference_catalog_duplicate"
    )
    with pytest.raises(ValueError, match=expected):
        initialize_library(reference, output, NOW, FoundationPolicy(version="v1"))
    assert not output.exists()


@pytest.mark.parametrize("scenario", ["source_mismatch", "future_row", "no_alias"])
def test_initialization_row_failures_and_business_freshness(tmp_path, scenario):
    """场景：行来源错配、未来行或未解析实体；预期：逐行原因与业务过期可追踪。

    输入：期望业务日期2026-10-01但参考行为2026-09-30；输出：stale_business_data。
    """
    reference = reference_fixture(tmp_path)
    configured = replace(bindings(), expected_data_as_of={"daily": "2026-10-01"})
    with duckdb.connect(
        str(reference), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        if scenario == "source_mismatch":
            connection.execute("UPDATE accepted.daily SET contract_hash='different'")
        elif scenario == "future_row":
            connection.execute("UPDATE accepted.daily SET observed_at='2026-10-03T00:00:00Z'")
        else:
            configured = replace(
                configured,
                entity_mappings=(
                    EntityMapping(source_api="daily", namespace="different", key_field="ts_code"),
                ),
            )
    summary = initialize_library(
        reference, tmp_path / "products.duckdb", NOW, FoundationPolicy(version="v1"), configured
    )
    reason = {
        "source_mismatch": "row_adaptation_failed",
        "future_row": "row_unavailable",
        "no_alias": "entity_unresolved",
    }[scenario]
    assert summary["reason_counts"][reason] == 1
    assert summary["reason_counts"]["stale_business_data"] == 1


@pytest.mark.parametrize(
    "scenario", ["missing_table", "wrong_identity", "missing_summary", "same_path"]
)
def test_library_validation_rejects_corrupted_product_state(tmp_path, scenario):
    """场景：产品表、记录身份或运行摘要被破坏；预期：只读验证失败且不给历史资格。

    输入示例：删除 summary 元数据；输出：valid=False、product_validation_failed。
    """
    reference = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(reference, output, NOW, FoundationPolicy(version="v1"), bindings())
    with duckdb.connect(str(output), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        if scenario == "missing_table":
            connection.execute("DROP TABLE product.entity")
        elif scenario == "wrong_identity":
            connection.execute("UPDATE product.entity SET record_id='tampered'")
        elif scenario == "missing_summary":
            connection.execute("DELETE FROM product_metadata WHERE key='summary'")
    checked = validate_library(reference, reference if scenario == "same_path" else output)
    assert checked == {
        "valid": False,
        "status": "failed",
        "reason_codes": ["product_validation_failed"],
        "historical_pit_certified": False,
    }


@pytest.mark.parametrize("location", ["direct", "contribution", "gate"])
def test_envelope_rejects_fractional_future_and_nested_evidence(location):
    """场景：证据仅晚半秒或藏于贡献、门禁；预期：所有证据统一遵守研究截止。

    输入：截止00:00:00Z、证据00:00:00.500000Z；输出：ValueError。
    """
    future = sample_evidence(observed_at="2026-10-02T00:00:00.500000Z")
    changes = {"evidence_refs": (future,)}
    if location == "contribution":
        changes = {
            "contributions": (
                FeatureContribution(
                    feature_id="quality",
                    raw_value=0.5,
                    unit="ratio",
                    transformed_value=0.5,
                    weight=1,
                    contribution=50,
                    evidence_refs=(future,),
                ),
            )
        }
    elif location == "gate":
        changes = {
            "quality_gates": (
                QualityGateResult(gate_id="unit", result=Truth.TRUE, evidence_refs=(future,)),
            )
        }
    with pytest.raises(ValueError):
        sample_envelope(**changes)


@pytest.mark.parametrize("mutation", ["projection", "lineage"])
def test_library_detects_projection_and_lineage_tampering_before_reuse(tmp_path, mutation):
    """场景：仅改SQL投影或删血缘而payload不变；预期：校验失败，幂等重跑拒绝复用。

    输入示例：将typed_fact.value改为999；输出：valid=False，重跑ValueError。
    """
    reference = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    configured = bindings()
    initialize_library(reference, output, NOW, policy, configured)
    with duckdb.connect(str(output), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        if mutation == "projection":
            connection.execute("UPDATE product.typed_fact SET value=999")
        else:
            connection.execute("DELETE FROM product.lineage_edge")
    assert not validate_library(reference, output)["valid"]
    with pytest.raises(ValueError):
        initialize_library(reference, output, NOW, policy, configured)


@pytest.mark.parametrize("scenario", ["eligible", "quarantine", "unknown", "null", "future"])
def test_feature_adapter_requires_eligible_and_available_typed_fact(scenario):
    """场景：从认证事实构建默认特征；预期：仅有效且当前可见值进入特征，其他附原因。

    输入示例：2000元认证事实；输出：FeatureValue.value=2000且证据保留。
    """
    from tushare_entitlement_hub.product_foundation.features import feature_from_fact

    row, certificate = numeric_inputs()
    availability = sample_availability()
    if scenario in {"quarantine", "unknown"}:
        row["quality_tier"] = scenario
    elif scenario == "null":
        row["amount"] = None
    elif scenario == "future":
        availability = replace(availability, observed_at="2026-10-03T00:00:00Z")
    fact = adapt_numeric(row, "amount", ENTITY, availability, certificate)
    feature = feature_from_fact(
        fact,
        feature_id="amount",
        analysis_as_of=NOW,
        scope_id="sample",
        window="spot",
        feature_profile="base",
        version="v1",
    )
    assert feature.entity_id == ENTITY and feature.unit == "元"
    if scenario == "eligible":
        assert feature.value == 2000 and feature.input_count == 1 and not feature.reason_codes
        assert feature.evidence_refs == fact.evidence_refs
    else:
        assert feature.value is None and feature.input_count == 0 and feature.reason_codes


def test_event_fact_preserves_future_plan_and_late_observation():
    """场景：未来生效计划与晚到旧事件；预期：保留各时间和来源阶段，不自动变成新信号。

    输入：9月生效但10月才观察的事件；输出：原生效日和观察日分别保存且没有触发时间。
    """
    from tushare_entitlement_hub.product_foundation.signals import EventFact

    plan = EventFact(
        event_id="plan1",
        event_type="listing",
        version="v1",
        entity_id=ENTITY,
        availability=sample_availability(effective_from="2026-11-01T00:00:00Z"),
        evidence_refs=(sample_evidence(),),
        stage="planned",
    )
    assert plan.stage == "planned"
    assert plan.availability.effective_from == "2026-11-01T00:00:00Z"
    late = replace(
        plan,
        event_id="old1",
        entity_id=None,
        stage="effective",
        availability=sample_availability(effective_from="2026-09-01T00:00:00Z"),
    )
    assert late.availability.observed_at == NOW
    assert late.availability.effective_from == "2026-09-01T00:00:00Z"
    assert "triggered_at" not in late.to_dict()
    assert EventFact.from_json(late.to_json()) == late
    with pytest.raises(ValueError):
        replace(plan, stage="trade_now")


def test_score_many_equal_weights_keeps_exact_upper_boundary():
    """场景：十二个满分归一特征等权合成；预期：得到合法100分，舍入不导致越界。

    输入：12项value=1、weight=1；输出：score=100且贡献和与分数一致。
    """
    policy = ScorePolicy(
        version="v1",
        thresholds=FoundationPolicy(version="v1"),
        weights=tuple(FeatureWeight(feature_id=f"feature{i}", weight=1) for i in range(12)),
    )
    features = tuple(sample_feature(f"feature{i}", 1) for i in range(12))
    result = score_features(features, policy)
    assert result.score == 100
    assert sum(item.contribution for item in result.components) == pytest.approx(100)


def test_score_profiles_include_actual_weight_configuration():
    """场景：策略版本文本相同但实际权重不同；预期：不能把不同评分口径混在同一档位。

    输入：同version下权重0.5/0.5与0.9/0.1；输出：feature_profile不同。
    """
    policy = sample_score_policy()
    features = (sample_feature(value=1), sample_feature("liquidity", 0))
    changed = replace(
        policy,
        weights=(replace(policy.weights[0], weight=0.9), replace(policy.weights[1], weight=0.1)),
    )
    baseline = score_features(features, policy)
    alternative = score_features(features, changed)
    assert baseline.score == 50 and alternative.score == 90
    assert baseline.feature_profile != alternative.feature_profile


def test_score_contribution_guard_honors_strict_tolerance():
    """场景：容差严于真实浮点运算误差；预期：贡献与直接加权公式不一致时拒绝。

    输入：等权1/1、数值0.1/0.2、容差1e-20；输出：contribution_sum_mismatch。
    """
    policy = ScorePolicy(
        version="v1",
        thresholds=FoundationPolicy(version="v1", contribution_tolerance=1e-20),
        weights=(
            FeatureWeight(feature_id="quality", weight=1),
            FeatureWeight(feature_id="liquidity", weight=1),
        ),
    )
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        score_features((sample_feature(value=0.1), sample_feature("liquidity", 0.2)), policy)


def test_numeric_price_certificate_still_requires_trade_evidence():
    """场景：单位已认证但价格无成交；预期：认证不能绕过价格用途门禁。

    输入：price证书和缺量价证明的参考行；输出：value=None、no_traded_price。
    """
    row, certificate = numeric_inputs()
    fact = adapt_numeric(
        row, "amount", ENTITY, sample_availability(), replace(certificate, dimension="price")
    )
    assert fact.value is None and "no_traded_price" in fact.reason_codes


def test_event_requires_evidence_and_effective_stage_time():
    """场景：事件无来源或已生效但缺生效日；预期：拒绝无依据阶段声明。

    输入示例：stage='effective'、effective_from=None；输出：ValueError。
    """
    from tushare_entitlement_hub.product_foundation.signals import EventFact

    event = EventFact(
        event_id="event1",
        event_type="listing",
        version="v1",
        entity_id=None,
        availability=sample_availability(),
        evidence_refs=(sample_evidence(),),
        stage="unknown",
    )
    for changes in ({"evidence_refs": ()}, {"stage": "effective"}):
        with pytest.raises(ValueError, match="event_evidence_or_effective_time_required"):
            replace(event, **changes)


def test_reference_reader_decodes_string_json_metadata():
    """场景：参考表以VARCHAR存储JSON元数据；预期：分批读取后恢复字典结构。

    输入：source_units='{"amount":"元"}'；输出：source_units={'amount':'元'}。
    """
    from tushare_entitlement_hub.product_foundation.reference_scan import reference_rows

    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        connection.execute("CREATE SCHEMA accepted")
        connection.execute(
            "CREATE TABLE accepted.daily(source_record_id VARCHAR,source_units VARCHAR)"
        )
        connection.execute("INSERT INTO accepted.daily VALUES (?,?)", ["row1", '{"amount":"元"}'])
        assert list(reference_rows(connection, "daily")) == [
            {"source_record_id": "row1", "source_units": {"amount": "元"}}
        ]


def test_product_store_score_projection_and_missing_table_guard(tmp_path):
    """场景：保存评分贡献及直接验证缺表库；预期：SQL投影准确，缺表完整性失败。

    输入：唯一50分贡献；输出：投影quality/50，删除entity表后校验ValueError。
    """
    from tushare_entitlement_hub.product_foundation.storage_records import ScoreComponent

    component = FeatureContribution(
        feature_id="quality",
        raw_value=0.5,
        unit="ratio",
        transformed_value=0.5,
        weight=1,
        contribution=50,
    )
    record = ScoreComponent(result_id="result1", component=component)
    with ProductStore(tmp_path / "products.duckdb") as store:
        store.put("score_component", record)
        assert store.records("score_component") == [record]
        assert store.connection.execute(
            "SELECT feature_id,contribution FROM product.score_component"
        ).fetchone() == ("quality", 50)
        store.verify_contents()
        store.connection.execute("DROP TABLE product.entity")
        with pytest.raises(ValueError, match="product_tables_missing"):
            store.verify_contents()


def test_library_validation_detects_deleted_entity_reference(tmp_path):
    """场景：持久化后实体行被删除；预期：其余别名、事实仍在也不能通过校验。

    输入：删除product.entity全部行；输出：valid=False。
    """
    reference = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(reference, output, NOW, FoundationPolicy(version="v1"), bindings())
    with duckdb.connect(str(output), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        connection.execute("DELETE FROM product.entity")
    assert not validate_library(reference, output)["valid"]


def test_initialization_detects_source_file_change_during_read(tmp_path, monkeypatch):
    """场景：读取期间来源文件实际变化；预期：最终摘要比对失败并回滚产品批次。

    输入：最终校验前给临时参考文件追加字节；输出：reference_changed_during_read。
    """
    from tushare_entitlement_hub.product_foundation import reference as reference_module

    reference = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    original_checksum = reference_module.file_checksum
    calls = 0

    def checksum_after_external_change(path):
        """模拟外部写入后真实计算摘要；第二次输入同路径，输出改变后的真实摘要。"""
        nonlocal calls
        calls += 1
        if calls == 2:
            with path.open("ab") as handle:
                handle.write(b"external-change")
        return original_checksum(path)

    monkeypatch.setattr(reference_module, "file_checksum", checksum_after_external_change)
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        initialize_library(reference, output, NOW, FoundationPolicy(version="v1"))
    assert calls == 2
    with ProductStore(output) as store:
        assert sum(store.counts().values()) == 0


def test_cli_validate_missing_library_returns_failure_json(tmp_path):
    """场景：CLI校验不存在的产品库；预期：退出1并输出机器可读失败，不创建库。

    输入：missing.duckdb路径；输出：valid=False、exit_code=1。
    """
    from typer.testing import CliRunner

    from tushare_entitlement_hub.cli import app

    result = CliRunner().invoke(
        app,
        [
            "product-foundation",
            "validate",
            "--reference",
            str(tmp_path / "missing-reference.duckdb"),
            "--output",
            str(tmp_path / "missing-products.duckdb"),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["valid"] is False
    assert not (tmp_path / "missing-products.duckdb").exists()
