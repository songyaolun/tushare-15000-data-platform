"""不可用评分特征的审计信息与排除血缘回归测试，全程使用合成数据。"""

from pathlib import Path

import pytest
from test_product_envelope_regressions import envelope_values
from test_product_foundation import evidence, feature

from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import DecisionEnvelope
from tushare_entitlement_hub.product_foundation.features import (
    FeatureWeight,
    ScorePolicy,
    score_features,
)
from tushare_entitlement_hub.product_foundation.storage import ProductStore
from tushare_entitlement_hub.product_foundation.storage_records import ScoreComponent


def audit_score_policy() -> ScorePolicy:
    """构造审计回归使用的评分策略，合格主项占八成且必须提供。

    输入无参数，输出 quality/liquidity 权重为 0.8/0.2 的策略；
    只有 quality=0.5 可用时，允许输出 50 分及 0.8 权重覆盖率。
    """
    return ScorePolicy(
        version="audit-v1",
        weights=(
            FeatureWeight(feature_id="quality", weight=0.8, required=True),
            FeatureWeight(feature_id="liquidity", weight=0.2),
        ),
        thresholds=FoundationPolicy(version="audit-v1"),
    )


@pytest.mark.parametrize(
    ("value", "raw_value", "reason_codes"),
    [
        (None, 2500.0, ("observation_after_cutoff",)),
        (None, 0.0, ("unit_unverified", "fact_not_feature_eligible")),
        (0.95, 95.0, ("quality_quarantine",)),
        (17.0, None, ("unit_unverified",)),
    ],
)
def test_unavailable_feature_preserves_audit_fields(
    value: float | None, raw_value: float | None, reason_codes: tuple[str, ...]
) -> None:
    """场景：特征被时点、单位或隔离门禁排除；预期：审计字段完整且不参与评分。

    输入原值 2500、不可用原因 observation_after_cutoff 的流动性特征，
    输出贡献的原值仍为 2500、权重为 0、贡献为空，并保留所有原因和来源。
    """
    # 场景：不可用值也可能非空甚至超出归一范围；预期：只保留审计，主项仍得 50 分。
    excluded_reference = evidence(
        source_record_id="excluded-liquidity",
        observed_at="2026-10-03T00:00:00Z",
        locator="fixture:excluded-liquidity",
    )
    source_references = (
        (evidence(source_record_id="rejected-liquidity", locator="fixture:rejected-liquidity"),)
        if value is not None
        else ()
    )
    rejected = feature(
        "liquidity",
        value,
        raw_value=raw_value,
        unit="万元",
        transform="certified_scale",
        input_count=int(value is not None),
        evidence_refs=source_references,
        excluded_evidence=(excluded_reference,),
        reason_codes=reason_codes,
    )

    result = score_features((feature(), rejected), audit_score_policy())
    component = next(item for item in result.components if item.feature_id == "liquidity")

    assert component.raw_value == (raw_value if raw_value is not None else value)
    assert component.unit == "万元"
    assert component.transform == "certified_scale"
    assert component.missing_reason == reason_codes[0]
    assert component.reason_codes == reason_codes
    assert component.evidence_refs == ()
    assert set(component.excluded_evidence) == {excluded_reference, *source_references}
    assert component.transformed_value is None
    assert component.contribution is None
    assert component.weight == 0
    assert result.score == 50
    assert result.weight_coverage == 0.8
    assert result.missing_features == ("liquidity",)


def test_absent_feature_keeps_generic_missing_reason() -> None:
    """场景：调用方完全未提供非核心特征；预期：继续使用明确的通用缺失解释。

    输入只有 quality=0.5 的特征集合，输出 liquidity 的原值和贡献为空、
    单位为 unknown，缺失原因是 feature_unavailable，且不虚构来源证据。
    """
    # 场景：没有原始特征可供审计；预期：默认缺失语义和可用主项评分保持一致。
    result = score_features((feature(),), audit_score_policy())
    component = next(item for item in result.components if item.feature_id == "liquidity")

    assert component.raw_value is None
    assert component.unit == "unknown"
    assert component.transform == "unavailable"
    assert component.missing_reason == "feature_unavailable"
    assert component.reason_codes == ("feature_unavailable",)
    assert component.evidence_refs == ()
    assert component.excluded_evidence == ()
    assert component.transformed_value is None
    assert component.contribution is None
    assert component.weight == 0
    assert result.score == 50


def test_scored_feature_retains_excluded_partial_inputs() -> None:
    """场景：可用特征同时保留未采用的部分输入；预期：贡献沿用全部审计记录。

    输入 quality=0、原值为 0 且含一条被排除的来源，输出 0 分贡献、
    原值仍为 0，可用证据与排除证据分别保留，缺失原因为空。
    """
    # 场景：零是合法可用值，部分输入被排除；预期：评分和两类证据不相互混淆。
    excluded_reference = evidence(source_record_id="partial-input", locator="fixture:partial")
    usable = feature(value=0.0, raw_value=0.0, excluded_evidence=(excluded_reference,))

    result = score_features((usable,), audit_score_policy())
    component = next(item for item in result.components if item.feature_id == "quality")

    assert component.raw_value == 0
    assert component.transformed_value == 0
    assert component.contribution == 0
    assert component.weight == 1
    assert component.evidence_refs == usable.evidence_refs
    assert component.excluded_evidence == (excluded_reference,)
    assert component.reason_codes == ()
    assert component.missing_reason is None
    assert result.score == 0


def test_score_component_persists_excluded_lineage(tmp_path: Path) -> None:
    """场景：评分贡献落入产品库并重新打开；预期：排除来源形成持久化排除血缘。

    输入不可用 liquidity 及其两条来源，存储后输出两条 relationship=excluded
    的血缘；可用 quality 的来源仍为 used，原始值、单位和原因完整往返。
    """
    # 场景：有值但被隔离的来源和时点不可用来源共同排除；预期：均不登记成 used。
    rejected_reference = evidence(source_record_id="rejected-value", locator="fixture:rejected")
    future_reference = evidence(
        source_record_id="future-value",
        observed_at="2026-10-03T00:00:00Z",
        locator="fixture:future",
    )
    rejected = feature(
        "liquidity",
        0.9,
        raw_value=9000.0,
        unit="元",
        reason_codes=("quality_quarantine", "unit_unverified"),
        evidence_refs=(rejected_reference,),
        excluded_evidence=(future_reference,),
    )
    result = score_features((feature(), rejected), audit_score_policy())
    product_path = tmp_path / "audit-products.duckdb"
    with ProductStore(product_path) as store:
        identities = {
            component.feature_id: store.put(
                "score_component", ScoreComponent(result_id=result.result_id, component=component)
            )
            for component in result.components
        }

    with ProductStore(product_path, read_only=True) as store:
        lineage = store.connection.execute(
            "SELECT output_id, source_record_id, relationship FROM product.lineage_edge"
        ).fetchall()
        assert set(lineage) == {
            (identities["quality"], "row1", "used"),
            (identities["liquidity"], "rejected-value", "excluded"),
            (identities["liquidity"], "future-value", "excluded"),
        }
        components = {
            record.component.feature_id: record.component
            for record in store.records("score_component")
        }
        assert components == {component.feature_id: component for component in result.components}
        assert components["liquidity"].raw_value == 9000
        assert components["liquidity"].unit == "元"
        assert components["liquidity"].reason_codes == ("quality_quarantine", "unit_unverified")
        store.verify_contents()


def test_optional_excluded_evidence_survives_scored_envelope_roundtrip() -> None:
    """场景：可选特征因来源晚于截止而缺失；预期：带分数信封保留排除证据。

    输入可用主项与未来可选来源，输出 50 分信封；未来来源只在排除列表，
    JSON 往返仍保留原值 2500、单位、具体缺失原因，不成为已使用证据。
    """
    # 场景：评分与公共信封连用；预期：合法缺失不阻断序列化，未来来源不参与贡献。
    future = evidence(source_record_id="future-optional", observed_at="2026-10-03T00:00:00Z")
    unavailable = feature(
        "liquidity",
        None,
        raw_value=2500.0,
        unit="元",
        input_count=0,
        evidence_refs=(),
        excluded_evidence=(future,),
        reason_codes=("not_yet_available",),
    )
    score = score_features((feature(), unavailable), audit_score_policy())
    result = DecisionEnvelope(
        **envelope_values(),
        score=score.score,
        feature_profile=score.feature_profile,
        missing_features=score.missing_features,
        features=score.components,
        contributions=score.components,
    )
    restored = DecisionEnvelope.from_json(result.to_json())
    optional = next(item for item in restored.contributions if item.feature_id == "liquidity")
    assert restored == result
    assert restored.score == 50
    assert optional.raw_value == 2500
    assert optional.unit == "元"
    assert optional.reason_codes == ("not_yet_available",)
    assert optional.excluded_evidence == (future,)
    assert optional.evidence_refs == ()
    assert optional.contribution is None
