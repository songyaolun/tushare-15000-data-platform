"""审查回归：配置内参考库证据必须像普通来源证据一样复验。"""

from dataclasses import replace

import pytest
import test_research_source_validation as validation_fixtures
from test_research_adapter import certification, configured, reference_database, source
from test_research_governance import NOW, relation
from test_research_source_validation import proof, put

from tushare_entitlement_hub.equity_research.source_validation import verify_source_evidence

products = validation_fixtures.products


def assertion_config(kind, reference):
    """把指定引用写入真实配置断言；例：kind=alias将证据绑定两证券的别名，仍保留原时间映射。"""
    config = configured("daily")
    if kind == "alias":
        return replace(
            config,
            foundation=replace(
                config.foundation,
                aliases=tuple(
                    replace(
                        alias,
                        evidence_refs=(reference,),
                        observed_at=max(NOW, reference.observed_at),
                    )
                    for alias in config.foundation.aliases
                ),
            ),
        )
    if kind == "certification":
        return replace(
            config,
            foundation=replace(
                config.foundation,
                certifications=(replace(certification(), evidence_refs=(reference,)),),
            ),
        )
    return replace(
        config,
        confirmed_relations=(
            replace(
                relation(), evidence_refs=(reference,), observed_at=max(NOW, reference.observed_at)
            ),
        ),
    )


@pytest.mark.parametrize("kind", ["alias", "certification", "relation"])
@pytest.mark.parametrize(
    ("patch", "reference_patch", "error"),
    [
        ({}, {}, None),
        ({}, {"source_record_id": "missing"}, "research_evidence_source_missing"),
        ({"quality_tier": "unknown"}, {}, "research_evidence_source_missing"),
        ({"is_quarantined": True}, {}, "research_evidence_source_missing"),
        ({"contract_hash": "tampered"}, {}, "research_evidence_contract_mismatch"),
        ({}, {"business_group_key": "tampered"}, "research_evidence_identity_mismatch"),
        ({"observed_at": "2026-10-04T00:00:00Z"}, {}, "row_unavailable"),
        ({}, {"fields": ("invented",)}, "research_evidence_field_missing"),
    ],
)
def test_reference_assertions_are_rechecked_against_accepted_rows(
    tmp_path, products, kind, patch, reference_patch, error
):
    """场景：三类配置引用正常或被篡改、未来、隔离；预期：只允许当前契约下真实可见accepted行。"""
    record = source() | patch
    path = reference_database(tmp_path, {"daily": [record]})
    reference = replace(proof(record), **reference_patch)
    config = assertion_config(kind, reference)
    put(products, reference)
    if error is None:
        verify_source_evidence(path, products, "current", config, analysis_as_of=NOW)
    else:
        with pytest.raises(ValueError, match=error):
            verify_source_evidence(path, products, "current", config, analysis_as_of=NOW)


@pytest.mark.parametrize("locator", ["assertion:manual", "synthetic:unit-certification"])
@pytest.mark.parametrize("future_field", [None, "observed_at", "public_available_at"])
def test_external_assertion_type_still_requires_exact_visible_configuration(
    tmp_path, products, locator, future_field
):
    """场景：独立人工或合成断言可见或晚于截止时点；预期：精确配置可复验，未来证据被拒。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    reference = replace(
        proof(record),
        locator=locator,
        **({future_field: "2026-10-04T00:00:00Z"} if future_field else {}),
    )
    config = assertion_config("certification", reference)
    put(products, reference)
    if future_field:
        with pytest.raises(ValueError, match="research_assertion_not_yet_available"):
            verify_source_evidence(path, products, "current", config, analysis_as_of=NOW)
    else:
        verify_source_evidence(path, products, "current", config, analysis_as_of=NOW)


def test_unknown_assertion_locator_cannot_bypass_reference_identity(tmp_path, products):
    """场景：伪造配置断言使用未登记定位协议；预期：不能靠配置成员关系跳过来源身份校验。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    reference = replace(proof(record), locator="unverified:manual")
    put(products, reference)
    with pytest.raises(ValueError, match="research_evidence_identity_mismatch"):
        verify_source_evidence(
            path, products, "current", assertion_config("relation", reference), analysis_as_of=NOW
        )
