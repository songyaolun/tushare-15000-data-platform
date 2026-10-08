"""可复算三值谓词及研究信号；无下单、持仓变更或自动买卖状态。"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from ..pit import time_boundary
from .contracts import EvidenceRef, Predicate, Truth
from .model import EntityId, Identifier, Model, Timestamp
from .temporal import AvailabilityFact


def evaluate(
    predicate: Predicate,
    values: dict[str, float | None],
    manual_results: dict[str, Truth] | None = None,
) -> Truth:
    """实现 Kleene 三值逻辑；缺失、非有限数或未确认人工条件返回 unknown。

    例如 ge(quality,0.5) 对 {'quality':0.8} → true；空输入 → unknown。
    all(false,unknown) 为 false，any(true,unknown) 为 true，不把未知当支持。
    """
    if predicate.operator == "manual":
        answer = (manual_results or {}).get(predicate.predicate_id, Truth.UNKNOWN)
        if not isinstance(answer, Truth):
            raise ValueError("invalid_manual_result")
        return answer
    if predicate.operator in {"all", "any", "not"}:
        children = tuple(evaluate(child, values, manual_results) for child in predicate.children)
        if predicate.operator == "not":
            return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE, Truth.UNKNOWN: Truth.UNKNOWN}[
                children[0]
            ]
        deciding = Truth.FALSE if predicate.operator == "all" else Truth.TRUE
        if deciding in children:
            return deciding
        if Truth.UNKNOWN in children:
            return Truth.UNKNOWN
        return Truth.TRUE if predicate.operator == "all" else Truth.FALSE
    value = values.get(predicate.feature_id or "")
    if value is None or type(value) not in {int, float} or not isfinite(value):
        return Truth.UNKNOWN
    threshold = predicate.threshold
    assert threshold is not None
    results = {
        "gt": value > threshold,
        "ge": value >= threshold,
        "lt": value < threshold,
        "le": value <= threshold,
        "eq": value == threshold,
    }
    return Truth.TRUE if results[predicate.operator] else Truth.FALSE


@dataclass(frozen=True, kw_only=True)
class SignalFact(Model):
    """规则触发事实保留支持、反证和失效条件，不假定空反证意味着无风险。"""

    entity_id: EntityId
    rule_id: Identifier
    rule_version: Identifier
    triggered_at: Timestamp
    predicate: Predicate
    result: Truth
    evidence_refs: tuple[EvidenceRef, ...]
    contrary_evidence: tuple[EvidenceRef, ...]
    invalidation_conditions: tuple[Predicate, ...]
    next_check_at: Timestamp | None

    def __post_init__(self) -> None:
        """信号不能早于证据，复核不能早于触发；未来证据输入 → ValueError。

        输入完整三值信号，输出可逆模型，未知结果仍保留证据链。
        """
        super().__post_init__()
        cutoff = time_boundary(self.triggered_at)
        if self.next_check_at is not None and time_boundary(self.next_check_at) < cutoff:
            raise ValueError("next_check_before_signal")
        if not self.evidence_refs:
            raise ValueError("signal_evidence_required")
        if any(
            time_boundary(item.observed_at) > cutoff
            or (
                item.public_available_at is not None
                and time_boundary(item.public_available_at) > cutoff
            )
            for item in self.evidence_refs + self.contrary_evidence
        ):
            raise ValueError("signal_before_evidence")


@dataclass(frozen=True, kw_only=True)
class EventFact(Model):
    """版本化来源事件，观察时间与计划或实际生效阶段分开保存。"""

    event_id: Identifier
    event_type: Identifier
    version: Identifier
    entity_id: EntityId | None
    availability: AvailabilityFact
    evidence_refs: tuple[EvidenceRef, ...]
    stage: Literal["planned", "effective", "cancelled", "unknown"]

    def __post_init__(self) -> None:
        """事件必须有来源，已生效阶段须有生效时间；未知实体可留空等待消歧。

        输入未来生效的 planned 事件，输出原计划；晚采旧事件不会增加 triggered_at。
        """
        super().__post_init__()
        if not self.evidence_refs or (
            self.stage == "effective" and self.availability.effective_from is None
        ):
            raise ValueError("event_evidence_or_effective_time_required")
