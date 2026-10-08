"""公共信封回归：保留可选缺失贡献，并拒绝未经实现的历史认证声明。"""

import json
from dataclasses import replace

import pytest

from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import (
    ActionState,
    DecisionEnvelope,
    EvidenceRef,
    ProductState,
)
from tushare_entitlement_hub.product_foundation.features import (
    FeatureValue,
    FeatureWeight,
    ScorePolicy,
    ScoreResult,
    score_features,
)

ANALYSIS_AT = "2026-10-02T00:00:00Z"
ENTITY_ID = "cn:security:A.SZ:listing1"


def envelope_values() -> dict[str, object]:
    """提供固定研究时点的公共信封字段，不开启行动或历史用途。

    输入无参数，输出含 READY 状态、WATCH 行动和稳定身份的字典；调用方补充评分。
    """
    return {
        "product_id": "foundation",
        "entity_id": ENTITY_ID,
        "scope_id": "sample",
        "run_id": "run1",
        "model_version": "v1",
        "analysis_as_of": ANALYSIS_AT,
        "generated_at": ANALYSIS_AT,
        "data_as_of": "2026-09-30",
        "state": ProductState.READY,
        "action_state": ActionState.WATCH,
    }


def optional_missing_score(value: float = 0.5) -> ScoreResult:
    """计算核心权重达到门槛、可选项缺失的真实评分结果。

    输入核心特征 value=0.5、权重 0.8，输出 50 分及一个带缺失原因的空贡献；
    输入 value=0，输出合法 0 分，缺失项仍保持空值。
    """
    evidence = EvidenceRef(
        source_api="daily",
        source_record_id="row1",
        business_group_key="group",
        business_variant_key=None,
        contract_hash="contract",
        fields=("close",),
        observed_at=ANALYSIS_AT,
        locator="fixture:row1",
    )
    feature = FeatureValue(
        entity_id=ENTITY_ID,
        feature_id="core",
        analysis_as_of=ANALYSIS_AT,
        scope_id="sample",
        window="daily",
        feature_profile="fixture",
        version="v1",
        value=value,
        unit="ratio",
        input_count=1,
        expected_count=1,
        evidence_refs=(evidence,),
    )
    policy = ScorePolicy(
        version="v1",
        thresholds=FoundationPolicy(version="v1", minimum_weight_coverage=0.8),
        weights=(
            FeatureWeight(feature_id="core", weight=0.8, required=True),
            FeatureWeight(feature_id="optional", weight=0.2),
        ),
    )
    return score_features((feature,), policy)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_envelope_accepts_scored_optional_missing_contributions(value):
    """场景：核心权重恰好达标，可选项缺失；预期：信封可封装且完整往返。

    输入核心值 0、0.5 或 1，输出对应 0、50 或 100 分；缺失贡献不补零。
    """
    result = optional_missing_score(value)
    assert result.score == pytest.approx(value * 100)
    assert result.weight_coverage == pytest.approx(0.8)
    envelope = DecisionEnvelope(
        **envelope_values(),
        score=result.score,
        feature_profile=result.feature_profile,
        missing_features=result.missing_features,
        features=result.components,
        contributions=result.components,
    )
    assert envelope.score == result.score
    assert envelope.missing_features == ("optional",)
    assert envelope.contributions[1].contribution is None
    assert envelope.contributions[1].missing_reason == "feature_unavailable"
    assert DecisionEnvelope.from_dict(envelope.to_dict()) == envelope
    assert DecisionEnvelope.from_json(envelope.to_json()) == envelope


def test_envelope_rejects_missing_contribution_without_reason():
    """场景：已评分结果中的空贡献没有缺失原因；预期：继续拒绝不完整解释。

    输入 50 分有效贡献与无原因空贡献，输出 contribution_sum_mismatch。
    """
    result = optional_missing_score()
    unexplained = replace(result.components[1], missing_reason=None)
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        DecisionEnvelope(
            **envelope_values(),
            score=result.score,
            contributions=(result.components[0], unexplained),
        )


def test_envelope_rejects_wrong_available_contribution_sum():
    """场景：空贡献有原因，但实际贡献之和与评分不符；预期：仍拒绝错误总分。

    输入实际贡献 50、带原因的空贡献和 score=49，输出 contribution_sum_mismatch。
    """
    result = optional_missing_score()
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        DecisionEnvelope(**envelope_values(), score=49, contributions=result.components)


@pytest.mark.parametrize("empty_components", [False, True])
def test_envelope_requires_available_contribution_for_zero_score(empty_components):
    """场景：0 分没有任何实际贡献支持；预期：空列表与全部缺失均拒绝。

    输入 score=0、空贡献列表或仅有带原因的缺失项，输出 contribution_sum_mismatch。
    """
    result = optional_missing_score()
    components = () if empty_components else (result.components[1],)
    with pytest.raises(ValueError, match="contribution_sum_mismatch"):
        DecisionEnvelope(**envelope_values(), score=0, contributions=components)


def test_envelope_preserves_unscored_missing_components():
    """场景：结果尚未评分但需保留缺失说明；预期：允许空评分及缺失贡献。

    输入 score=None、仅有 optional 缺失项，输出可往返且仍无评分的公共信封。
    """
    result = optional_missing_score()
    envelope = DecisionEnvelope(
        **envelope_values(), score=None, contributions=(result.components[1],)
    )
    assert envelope.score is None
    assert DecisionEnvelope.from_json(envelope.to_json()) == envelope


@pytest.mark.parametrize("state", list(ProductState))
def test_envelope_rejects_historical_pit_certification(state):
    """场景：任意准备状态直接声明历史 PIT 已认证；预期：当前阶段一律拒绝。

    输入 READY 等各状态及 historical_pit_certified=True，输出
    historical_certification_not_implemented，状态本身不构成认证依据。
    """
    with pytest.raises(ValueError, match="historical_certification_not_implemented"):
        DecisionEnvelope(
            **(envelope_values() | {"state": state}),
            gate_codes=("observed_only",),
            historical_pit_certified=True,
        )


@pytest.mark.parametrize("encoding", ["dict", "json"])
def test_envelope_deserialization_rejects_historical_pit_certification(encoding):
    """场景：持久化输入携带未经认证的历史声明；预期：反序列化也执行拒绝规则。

    输入字典或 JSON 中的 historical_pit_certified=true，输出
    historical_certification_not_implemented，不让外部载荷绕过构造校验。
    """
    payload = DecisionEnvelope(**envelope_values()).to_dict()
    payload["historical_pit_certified"] = True
    with pytest.raises(ValueError, match="historical_certification_not_implemented"):
        if encoding == "dict":
            DecisionEnvelope.from_dict(payload)
        else:
            DecisionEnvelope.from_json(json.dumps(payload))


def test_envelope_preserves_unverified_historical_defaults():
    """场景：调用方省略历史认证或显式设为 false；预期：普通研究信封保持兼容。

    输入默认字段或 historical_pit_certified=False，输出相同信封，序列化仍为 false。
    """
    envelope = DecisionEnvelope(**envelope_values())
    explicit = replace(envelope, historical_pit_certified=False)
    assert explicit == envelope
    assert envelope.to_dict()["historical_pit_certified"] is False
    assert DecisionEnvelope.from_json(envelope.to_json()) == envelope
