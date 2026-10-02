"""有命名空间及生命周期的实体、时点别名和有证据的关系。"""

from dataclasses import dataclass
from enum import StrEnum

from ..pit import time_boundary
from .contracts import EvidenceRef
from .model import EntityId, Identifier, Model, Score, Timestamp


class EntityKind(StrEnum):
    """证券与公司、基金份额及衍生品合约分别标识，不能只凭代码跨类型连接。"""

    SECURITY = "security"
    COMPANY = "company"
    ETF_SHARE = "etf_share"
    INDEX = "index"
    FUTURES_PRODUCT = "futures_product"
    FUTURES_CONTRACT = "futures_contract"
    OPTION_CONTRACT = "option_contract"
    MARKET = "market"
    SERIES = "series"


@dataclass(frozen=True, kw_only=True)
class EntityIdentity(Model):
    """local_id 可为业务代码，但 lifecycle_id 必须显式给出，不从代码猜生命周期。"""

    namespace: Identifier
    kind: EntityKind
    local_id: Identifier
    lifecycle_id: Identifier

    def __post_init__(self) -> None:
        """拒绝会造成分隔歧义的身份；例如 local_id='a:b' → ValueError。"""
        super().__post_init__()
        if any(":" in part for part in (self.namespace, self.local_id, self.lifecycle_id)):
            raise ValueError("ambiguous_entity_identity")

    @property
    def entity_id(self) -> str:
        """返回命名空间、类型、代码和生命周期组合；cn/security/A/life → cn:security:A:life。"""
        return f"{self.namespace}:{self.kind.value}:{self.local_id}:{self.lifecycle_id}"


@dataclass(frozen=True, kw_only=True)
class Entity(Model):
    """统一实体记录；名称只是展示，不作为跨接口关系依据。"""

    identity: EntityIdentity
    name: str


@dataclass(frozen=True, kw_only=True)
class TemporalAssertion(Model):
    """别名与关系共用的有效区间及观察版本，采用左闭右开区间。"""

    effective_from: Timestamp
    effective_to: Timestamp | None
    observed_at: Timestamp
    source: Identifier
    version: Identifier
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: Score

    def __post_init__(self) -> None:
        """验证区间和证据齐全；例如终点早于起点或证据为空 → 拒绝。"""
        super().__post_init__()
        if self.effective_to is not None and time_boundary(self.effective_to) <= time_boundary(
            self.effective_from
        ):
            raise ValueError("invalid_effective_interval")
        if not self.evidence_refs:
            raise ValueError("assertion_evidence_required")
        if any(
            time_boundary(item.observed_at) > time_boundary(self.observed_at)
            for item in self.evidence_refs
        ):
            raise ValueError("assertion_before_evidence")

    def usable_at(self, cutoff: str) -> bool:
        """只在已观察、已公开且有效的时段内返回真；未来观察的旧别名 → False。"""
        instant = time_boundary(cutoff)
        return (
            time_boundary(self.observed_at) <= instant
            and time_boundary(self.effective_from) <= instant
            and (self.effective_to is None or instant < time_boundary(self.effective_to))
            and all(
                item.public_available_at is None
                or time_boundary(item.public_available_at) <= instant
                for item in self.evidence_refs
            )
        )


@dataclass(frozen=True, kw_only=True)
class EntityAlias(TemporalAssertion):
    """来源别名到统一实体的时点断言；BSE 新旧代码可指向同一实体。"""

    namespace: Identifier
    alias: str
    entity_id: EntityId

    def __post_init__(self) -> None:
        """拒绝空别名；例如 alias=' ' → ValueError，其余由共用断言校验。"""
        super().__post_init__()
        if not self.alias.strip():
            raise ValueError("alias_required")


@dataclass(frozen=True, kw_only=True)
class EntityRelation(TemporalAssertion):
    """显式语义关系，不提供 business_group_key 相等连接捷径。"""

    from_entity_id: EntityId
    to_entity_id: EntityId
    relation_type: Identifier
    confirmed: bool = False


def resolve_alias(
    aliases: tuple[EntityAlias, ...],
    namespace: str,
    alias: str,
    cutoff: str,
) -> str | None:
    """仅解析同命名空间、研究截止前可见的精确别名，多实体冲突直接拒绝。

    输入一个有效 A.SZ 别名，输出 cn:security:A.SZ:listing1；未匹配返回 None。
    """
    matches = {
        item.entity_id
        for item in aliases
        if item.namespace == namespace and item.alias == alias and item.usable_at(cutoff)
    }
    if len(matches) > 1:
        raise ValueError("ambiguous_entity_alias")
    return next(iter(matches), None)
