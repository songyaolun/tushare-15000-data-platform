"""公共输出信封、质量、贡献和证据契约；研究动作不包含自动交易。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from ..pit import time_boundary
from ..store import digest
from .config import MAXIMUM_CONTRIBUTION_ERROR
from .model import (
    Count,
    EntityId,
    Fraction,
    Identifier,
    Model,
    Period,
    ReasonCode,
    Score,
    Timestamp,
)


class Truth(StrEnum):
    """缺失值保留 unknown，只有 TRUE 才能通过门禁。"""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:
        """只将真映射为通过；例如 bool(Truth.UNKNOWN) → False。"""
        return self is Truth.TRUE


class ProductState(StrEnum):
    """共享的产品准备状态，不表达市场方向。"""

    READY = "ready"
    ANNOTATED = "annotated"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


class ActionState(StrEnum):
    """仅支持人工研究、观察和复核。"""

    RESEARCH_FIRST = "research_first"
    WATCH = "watch"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    NO_CONCLUSION = "no_conclusion"


@dataclass(frozen=True, kw_only=True)
class EvidenceRef(Model):
    """保留来源的三层身份；组键和变体键只能在 source_api 内解释。"""

    source_api: Identifier
    source_record_id: Identifier
    business_group_key: Identifier
    business_variant_key: Identifier | None
    contract_hash: Identifier
    fields: tuple[Identifier, ...]
    observed_at: Timestamp
    locator: str
    public_available_at: Timestamp | None = None
    source_occurrences: tuple[Identifier, ...] = ()

    def __post_init__(self) -> None:
        """引用必须定位至少一个字段和原始位置；例如 fields=() → 拒绝。

        输入完整证据，输出可逆模型；多个重复来源在 source_occurrences 中保留。
        """
        super().__post_init__()
        if not self.fields or len(set(self.fields)) != len(self.fields) or not self.locator.strip():
            raise ValueError("evidence_fields_and_locator_required")


@dataclass(frozen=True, kw_only=True)
class QualityGateResult(Model):
    """门禁失败和未检查必须带原因；硬门禁无法由分数抵消。"""

    gate_id: Identifier
    result: Truth
    reason_codes: tuple[ReasonCode, ...] = ()
    hard: bool = True
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        """校验结果与原因一致性；例如 UNKNOWN 没有原因 → 拒绝。

        输入真值及原因，输出有效门禁；TRUE 可以附 observed_only 等限制原因。
        """
        super().__post_init__()
        if self.result is not Truth.TRUE and not self.reason_codes:
            raise ValueError("gate_reason_required")

    @property
    def passed(self) -> bool:
        """只有 true 返回通过；例如 unknown → False，不隐式转为通过。"""
        return self.result is Truth.TRUE


@dataclass(frozen=True, kw_only=True)
class ConfidenceComponents(Model):
    """各分量为 0—100 证据质量分，未知为空，不合成上涨概率。"""

    quality: Score | None = None
    coverage: Score | None = None
    mapping: Score | None = None
    temporal: Score | None = None


@dataclass(frozen=True, kw_only=True)
class Coverage(Model):
    """目标分母与实际可用数分开；分母未知时不输出覆盖比例。"""

    observed_count: Count = 0
    expected_count: Count | None = None
    scope_basis: str = "not_checked"
    reason_codes: tuple[ReasonCode, ...] = ("coverage_not_checked",)

    def __post_init__(self) -> None:
        """拒绝大于目标分母的观测数；例如 observed=2、expected=1 → 错误。"""
        super().__post_init__()
        if self.expected_count is not None and self.observed_count > self.expected_count:
            raise ValueError("coverage_count_mismatch")


@dataclass(frozen=True, kw_only=True)
class Freshness(Model):
    """业务截止日、观察年龄和预期发布时刻分别保留。"""

    observed_at: Timestamp | None = None
    expected_at: Timestamp | None = None
    data_as_of: Period | None = None
    reason_codes: tuple[ReasonCode, ...] = ("freshness_not_checked",)


@dataclass(frozen=True, kw_only=True)
class FeatureContribution(Model):
    """评分分量保留归一变换、原值单位、权重及贡献。"""

    feature_id: Identifier
    raw_value: float | None
    unit: str
    transformed_value: Fraction | None
    weight: Fraction
    contribution: Score | None
    transform: str = "identity"
    peer_level: str | None = None
    missing_reason: ReasonCode | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()
    excluded_evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Predicate(Model):
    """可序列化的三值比较或组合条件，人工条件需显式提供结论。"""

    predicate_id: Identifier
    version: Identifier
    operator: Literal["gt", "ge", "lt", "le", "eq", "all", "any", "not", "manual"]
    feature_id: Identifier | None = None
    threshold: float | None = None
    children: tuple[Predicate, ...] = ()

    def __post_init__(self) -> None:
        """比较要求字段和阈值，组合要求子项；例如 ge 且无 threshold → 拒绝。"""
        super().__post_init__()
        if self.operator in {"all", "any", "not"}:
            if not self.children or (self.operator == "not" and len(self.children) != 1):
                raise ValueError("predicate_children_invalid")
            if self.feature_id is not None or self.threshold is not None:
                raise ValueError("predicate_mixed_forms")
        elif self.operator != "manual":
            if self.feature_id is None or self.threshold is None or self.children:
                raise ValueError("predicate_comparison_invalid")


@dataclass(frozen=True, kw_only=True)
class DecisionEnvelope(Model):
    """DecisionEnvelope/v1：各具体产品共同继承的研究结果边界。"""

    product_id: Identifier
    entity_id: EntityId
    scope_id: Identifier
    run_id: Identifier
    model_version: Identifier
    analysis_as_of: Timestamp
    generated_at: Timestamp
    data_as_of: Period
    state: ProductState
    action_state: ActionState
    schema_version: Literal["DecisionEnvelope/v1"] = "DecisionEnvelope/v1"
    source_as_of: dict[str, Period] = field(default_factory=dict)
    feature_profile: Identifier = "unscored"
    score: Score | None = None
    rank: float | None = None
    peer_count: Count = 0
    gate_codes: tuple[ReasonCode, ...] = ()
    missing_features: tuple[Identifier, ...] = ()
    unavailable_apis: tuple[Identifier, ...] = ()
    coverage: Coverage = field(default_factory=Coverage)
    freshness: Freshness = field(default_factory=Freshness)
    confidence_components: ConfidenceComponents = field(default_factory=ConfidenceComponents)
    features: tuple[FeatureContribution, ...] = ()
    contributions: tuple[FeatureContribution, ...] = ()
    contrary_evidence: tuple[EvidenceRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    quality_gates: tuple[QualityGateResult, ...] = ()
    entry_conditions: tuple[Predicate, ...] = ()
    invalidation_conditions: tuple[Predicate, ...] = ()
    next_check_at: Timestamp | None = None
    historical_pit_certified: bool = False
    historical_tradable: bool = False
    decision_allowed: bool = False
    contribution_tolerance: float = MAXIMUM_CONTRIBUTION_ERROR

    def __post_init__(self) -> None:
        """核验时点、贡献和、行动资格；不完整结果不能开启研究行动。

        输入 INCOMPLETE 且 decision_allowed=True → 拒绝；未认证历史交易同样拒绝。
        输入 50 分实际贡献与带原因的可选缺失项，输出保留缺失项的合法 50 分信封。
        输入 historical_pit_certified=True → 拒绝，本阶段尚未实现历史认证。
        """
        super().__post_init__()
        if self.historical_pit_certified:
            raise ValueError("historical_certification_not_implemented")
        cutoff = time_boundary(self.analysis_as_of)
        if time_boundary(self.generated_at) < cutoff:
            raise ValueError("generated_before_analysis")
        if self.next_check_at is not None and time_boundary(self.next_check_at) < cutoff:
            raise ValueError("next_check_before_analysis")
        nested = tuple(
            evidence
            for component in self.features + self.contributions
            for evidence in component.evidence_refs
        )
        nested += tuple(evidence for gate in self.quality_gates for evidence in gate.evidence_refs)
        for evidence in self.evidence_refs + self.contrary_evidence + nested:
            if time_boundary(evidence.observed_at) > cutoff or (
                evidence.public_available_at is not None
                and time_boundary(evidence.public_available_at) > cutoff
            ):
                raise ValueError("future_evidence")
        if self.rank is not None and (self.score is None or not 1 <= self.rank <= self.peer_count):
            raise ValueError("invalid_rank")
        if not 0 < self.contribution_tolerance <= MAXIMUM_CONTRIBUTION_ERROR:
            raise ValueError("invalid_contribution_tolerance")
        if self.score is not None:
            available_contributions = [
                item.contribution for item in self.contributions if item.contribution is not None
            ]
            if (
                not available_contributions
                or any(
                    item.contribution is None and item.missing_reason is None
                    for item in self.contributions
                )
                or abs(sum(available_contributions) - self.score) > self.contribution_tolerance
            ):
                raise ValueError("contribution_sum_mismatch")
        if self.historical_tradable:
            raise ValueError("historical_trading_not_supported")
        if self.decision_allowed and (
            self.state not in {ProductState.READY, ProductState.ANNOTATED}
            or self.action_state in {ActionState.BLOCKED, ActionState.NO_CONCLUSION}
            or not self.evidence_refs
            or not self.entry_conditions
            or not self.quality_gates
            or any(gate.hard and not gate.passed for gate in self.quality_gates)
        ):
            raise ValueError("decision_not_allowed")
        if self.state not in {ProductState.READY, ProductState.ANNOTATED} and not self.gate_codes:
            raise ValueError("product_reason_required")

    @property
    def result_id(self) -> str:
        """按身份、版本和批次生成稳定结果键；只改 generated_at 不改变键。

        输入当前信封，输出 SHA256 文本，可作为贡献和血缘的结果外键。
        """
        return digest(
            [
                self.product_id,
                self.entity_id,
                self.scope_id,
                self.run_id,
                self.model_version,
                self.feature_profile,
                self.analysis_as_of,
            ]
        )
