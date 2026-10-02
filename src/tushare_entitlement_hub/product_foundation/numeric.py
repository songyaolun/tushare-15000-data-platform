"""单位认证与数值门禁；原始值的可查询性不自动等于特征可用性。"""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, cast

from ..store import digest
from .contracts import EvidenceRef, QualityGateResult, Truth
from .model import EntityId, Identifier, Model, Positive, ReasonCode
from .temporal import AvailabilityFact


@dataclass(frozen=True, kw_only=True)
class UnitCertification(Model):
    """显式来源字段换算证书；元单位仍须明确 CNY/HKD 等币种。"""

    version: Identifier
    source_api: Identifier
    field: Identifier
    source_unit: str
    target_unit: str
    dimension: Identifier
    currency: Identifier | None
    multiplier: Positive
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        """证书必须有单位、依据及金额币种；例如 money 且 currency=None → 拒绝。

        输入千元→元、1000 倍及 CNY，输出可审计证书；同单位只能使用 1 倍。
        """
        super().__post_init__()
        if not self.source_unit.strip() or not self.target_unit.strip() or not self.evidence_refs:
            raise ValueError("unit_evidence_required")
        if self.dimension in {"money", "price"} and self.currency is None:
            raise ValueError("currency_unverified")
        if self.source_unit == self.target_unit and self.multiplier != 1:
            raise ValueError("same_unit_multiplier_invalid")


@dataclass(frozen=True, kw_only=True)
class TypedFact(Model):
    """原始来源数值适配结果；空值必须附原因，隔离记录不保留默认特征数值。"""

    fact_id: Identifier
    entity_id: EntityId
    field: Identifier
    value: float | None
    unit: str | None
    dimension: Identifier | None
    currency: Identifier | None
    availability: AvailabilityFact
    quality_tier: Literal["verified", "usable_with_annotation", "quarantine", "unknown"]
    certification: UnitCertification | None
    evidence_refs: tuple[EvidenceRef, ...]
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        """阻断伪认证、空值无原因及隔离值；例如 quarantine 且 value=1 → 拒绝。

        输入已认证的合格数值，输出可供后续算术使用的事实。
        """
        super().__post_init__()
        if self.value is None and not self.reason_codes:
            raise ValueError("missing_value_reason_required")
        if self.value is not None and (
            self.quality_tier in {"quarantine", "unknown"} or self.certification is None
        ):
            raise ValueError("uncertified_value")
        if self.certification is not None and (
            self.unit != self.certification.target_unit
            or self.dimension != self.certification.dimension
            or self.currency != self.certification.currency
            or self.field != self.certification.field
            or self.availability.source_api != self.certification.source_api
        ):
            raise ValueError("fact_certification_mismatch")
        if not self.evidence_refs:
            raise ValueError("fact_evidence_required")


@dataclass(frozen=True, kw_only=True)
class NumericResult(Model):
    """算术结果保留输出量纲、币种及输入事实，不能自动成为新的认证输入。"""

    value: float
    unit: str
    dimension: str
    currency: str | None
    input_fact_ids: tuple[Identifier, ...]


def price_gate(row: dict[str, Any], purpose: str) -> QualityGateResult:
    """验证成交参考是否允许进入价格用途；量额缺失或矛盾均不证明已成交。

    输入 close=10、vol=0，return 用途 → FALSE；有效正价格和正量额 → TRUE。
    repo_daily.close 是利率，settle 是结算价，均不冒充成交价格。
    """
    if purpose not in {"return", "volatility", "price_rank", "feasibility", "premium"}:
        raise ValueError("unknown_price_purpose")
    values = [row.get(name) for name in ("close", "vol", "amount")]
    codes = []
    if row.get("quality_tier") == "quarantine" or row.get("is_quarantined") is True:
        codes.append("quarantined_record")
    if row.get("source_api") == "repo_daily" or row.get("price_type") == "settlement":
        codes.append("not_traded_price_type")
    if row.get("market_activity_status") == "no_trade" or row.get("price_observation_status") in {
        "reference_close_without_trade",
        "no_price_observed",
    }:
        codes.append("no_traded_price")
    if any(type(value) not in {float, int} or not isfinite(cast(float, value)) for value in values):
        codes.append("price_activity_unknown")
    elif any(cast(float, value) <= 0 for value in values):
        codes.append("no_traded_price")
    return QualityGateResult(
        gate_id="price_" + purpose,
        result=Truth.FALSE if codes else Truth.TRUE,
        reason_codes=tuple(sorted(set(codes))),
    )


def source_evidence(row: dict[str, Any], field: str, availability: AvailabilityFact) -> EvidenceRef:
    """从参考层身份构造字段级引用，不接受缺失或错配的记录身份。

    输入带 contract_hash/业务组/来源行身份的 accepted 行，输出一个字段的 EvidenceRef。
    """
    if (
        row.get("source_api") != availability.source_api
        or row.get("source_record_id") != availability.source_record_id
    ):
        raise ValueError("source_identity_mismatch")
    return EvidenceRef(
        source_api=availability.source_api,
        source_record_id=availability.source_record_id,
        business_group_key=cast(str, row.get("business_group_key")),
        business_variant_key=row.get("business_variant_key"),
        contract_hash=cast(str, row.get("contract_hash")),
        fields=(field,),
        observed_at=availability.observed_at,
        public_available_at=availability.public_available_at,
        locator="reference:accepted/"
        + availability.source_api
        + "/"
        + availability.source_record_id,
        source_occurrences=tuple(row.get("source_occurrences", ())),
    )


def adapt_numeric(
    row: dict[str, Any],
    field: str,
    entity_id: str,
    availability: AvailabilityFact,
    certification: UnitCertification | None,
) -> TypedFact:
    """消费参考层现有标准单位，仅在显式证书匹配时换算，未知返回空值和原因。

    例如标准层 amount=2000 元及千元→元证书，输出 2000 元；不会再乘 1000。
    本适配器不重写 source_fields，也不会把隔离、未知注释或来源变体用于默认特征。
    """
    evidence = source_evidence(row, field, availability)
    reasons = []
    value = row.get(field)
    quality = row.get("quality_tier", "unknown")
    if quality not in {"verified", "usable_with_annotation", "quarantine", "unknown"}:
        quality = "unknown"
    if quality in {"quarantine", "unknown"} or row.get("is_quarantined") is True:
        reasons.append("quarantined_record" if quality == "quarantine" else "quality_unknown")
    if (
        row.get("business_variant_key")
        or row.get("identity_resolution") == "business_group_source_variant"
    ):
        reasons.append("business_variant_unresolved")
    if quality == "usable_with_annotation":
        reasons.append("annotation_not_adapted")
    if value is None:
        reasons.append("missing_numeric_value")
    elif type(value) not in {int, float} or not isfinite(value):
        reasons.append("invalid_numeric_value")
    if certification is None:
        reasons.append("unit_unverified")
    elif certification.source_api != availability.source_api or certification.field != field:
        raise ValueError("unit_certificate_field_mismatch")
    else:
        standard_unit = row.get("standard_units", {}).get(field)
        multiplier = 1.0
        if standard_unit == certification.target_unit:
            pass
        elif standard_unit == certification.source_unit:
            multiplier = certification.multiplier
        elif (
            standard_unit is None
            and row.get("source_units", {}).get(field) == certification.source_unit
        ):
            multiplier = certification.multiplier
        else:
            reasons.append("unit_mismatch")
        if certification.dimension == "price" and not price_gate(row, "return").passed:
            reasons.append("no_traded_price")
        if not reasons:
            value = cast(float, value) * multiplier
            if not isfinite(value):
                reasons.append("nonfinite_conversion")
    return TypedFact(
        fact_id=digest(
            [
                entity_id,
                availability.to_dict(),
                field,
                certification.to_dict() if certification else None,
            ]
        ),
        entity_id=entity_id,
        field=field,
        value=None if reasons else value,
        unit=certification.target_unit if certification else None,
        dimension=certification.dimension if certification else None,
        currency=certification.currency if certification else None,
        availability=availability,
        quality_tier=quality,
        certification=certification,
        evidence_refs=(evidence,),
        reason_codes=tuple(sorted(set(reasons))),
    )


def arithmetic(left: TypedFact, right: TypedFact, operation: str) -> NumericResult:
    """只有可比的已认证数值允许四则运算，币种、实体、期间和加减量纲须一致。

    输入两个同日 CNY 2000 元事实，divide → 1 比例，add → 4000 元。
    非有限结果、零分母和未知认证抛固定原因码；不跨币种猜测汇率。
    """
    if operation not in {"add", "subtract", "multiply", "divide"}:
        raise ValueError("unknown_arithmetic_operation")
    for fact in (left, right):
        if (
            fact.value is None
            or fact.certification is None
            or fact.reason_codes
            or fact.quality_tier != "verified"
        ):
            raise ValueError("numeric_fact_not_eligible")
    if (
        left.entity_id != right.entity_id
        or left.availability.data_as_of != right.availability.data_as_of
    ):
        raise ValueError("numeric_context_mismatch")
    if left.currency != right.currency and left.currency is not None and right.currency is not None:
        raise ValueError("currency_mismatch")
    assert left.value is not None and right.value is not None
    if operation in {"add", "subtract"}:
        if (left.unit, left.dimension, left.currency) != (
            right.unit,
            right.dimension,
            right.currency,
        ):
            raise ValueError("dimension_mismatch")
        value = left.value + right.value if operation == "add" else left.value - right.value
        unit, dimension, currency = str(left.unit), str(left.dimension), left.currency
    elif operation == "divide":
        if right.value == 0:
            raise ValueError("zero_denominator")
        value = left.value / right.value
        same = (left.unit, left.dimension, left.currency) == (
            right.unit,
            right.dimension,
            right.currency,
        )
        unit = "ratio" if same else f"{left.unit}/{right.unit}"
        dimension = "ratio" if same else f"{left.dimension}/{right.dimension}"
        currency = None if same else left.currency or right.currency
    else:
        value = left.value * right.value
        unit, dimension = f"{left.unit}*{right.unit}", f"{left.dimension}*{right.dimension}"
        currency = left.currency or right.currency
    return NumericResult(
        value=value,
        unit=unit,
        dimension=dimension,
        currency=currency,
        input_fact_ids=(left.fact_id, right.fact_id),
    )
