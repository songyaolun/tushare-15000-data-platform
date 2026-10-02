"""有命名空间及生命周期的实体、时点别名和有证据的关系。"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from ..pit import time_boundary
from ..standard import normalize_period
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

    def visible_at(self, cutoff: str) -> bool:
        """仅检查断言及其证据是否在截止前可见，不据此判断业务有效期。

        输入2026年截止和2020年失效但已观察的别名，输出True；未来公开证据输出False。
        """
        instant = time_boundary(cutoff)
        return time_boundary(self.observed_at) <= instant and all(
            item.public_available_at is None or time_boundary(item.public_available_at) <= instant
            for item in self.evidence_refs
        )

    def usable_at(self, cutoff: str) -> bool:
        """只在已观察、已公开且当时有效的时段内返回真，保留原来的单时点语义。

        输入2026年截止和2025年失效的别名输出False；同日仍有效且可见的别名输出True。
        """
        instant = time_boundary(cutoff)
        return (
            self.visible_at(cutoff)
            and time_boundary(self.effective_from) <= instant
            and (self.effective_to is None or instant < time_boundary(self.effective_to))
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


def _business_time_bounds(value: str, market_timezone: str | None) -> tuple[datetime, datetime]:
    """把业务时间展开为需由同一生命周期完整覆盖的闭区间，不补造公开时间。

    带时区时间戳返回同一时点；日、月、季、年覆盖完整市场期间。市场时区未知时，
    在UTC期间两端各扩一天以包住所有合法UTC偏移，不假设市场。输入2020-06配UTC，
    输出2020-06-01T00:00:00Z至2020-06-30T23:59:59.999999Z。
    """
    try:
        period = normalize_period(value)
    except ValueError:
        instant = time_boundary(value)
        return instant, instant
    year = int(period[:4])
    if len(period) == 10:
        start = datetime.fromisoformat(period)
        stop = start + timedelta(days=1)
    else:
        if len(period) == 4:
            month, width = 1, 12
        elif "Q" in period:
            month, width = (int(period[-1]) - 1) * 3 + 1, 3
        else:
            month, width = int(period[5:]), 1
        start = datetime(year, month, 1)
        next_month = month - 1 + width
        stop = datetime(year + next_month // 12, next_month % 12 + 1, 1)
    if market_timezone is None:
        lower = start.replace(tzinfo=UTC) - timedelta(days=1)
        upper = stop.replace(tzinfo=UTC) + timedelta(days=1)
    else:
        timezone = ZoneInfo(market_timezone)
        lower = start.replace(tzinfo=timezone).astimezone(UTC)
        upper = stop.replace(tzinfo=timezone).astimezone(UTC)
    return lower, upper - timedelta(microseconds=1)


def resolve_alias(
    aliases: tuple[EntityAlias, ...],
    namespace: str,
    alias: str,
    cutoff: str,
    *,
    effective_at: str | None = None,
    market_timezone: str | None = None,
) -> str | None:
    """独立校验证据可见截止与业务有效时间，只返回完整覆盖业务期间的唯一实体。

    输入cutoff=2026年、effective_at=2020年及两次上市别名，输出旧实体；期间任何部分
    有多实体冲突均拒绝。effective_at省略或为None保留旧四参调用的截止时点语义，
    因此业务时间未知的扫描调用方必须先拒绝解析。未匹配返回None。
    """
    lower, upper = (
        (time_boundary(cutoff), time_boundary(cutoff))
        if effective_at is None
        else _business_time_bounds(effective_at, market_timezone)
    )
    candidates = tuple(
        item
        for item in aliases
        if item.namespace == namespace
        and item.alias == alias
        and item.visible_at(cutoff)
        and time_boundary(item.effective_from) <= upper
        and (item.effective_to is None or lower < time_boundary(item.effective_to))
    )
    if len({item.entity_id for item in candidates}) > 1:
        raise ValueError("ambiguous_entity_alias")
    matches = {
        item.entity_id
        for item in candidates
        if time_boundary(item.effective_from) <= lower
        and (item.effective_to is None or upper < time_boundary(item.effective_to))
    }
    return next(iter(matches), None)
