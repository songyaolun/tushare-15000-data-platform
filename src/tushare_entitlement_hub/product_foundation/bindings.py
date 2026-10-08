"""外部调用方显式提供的实体、时间和单位适配配置。"""

from collections.abc import Hashable
from dataclasses import dataclass, field

from .entities import Entity, EntityAlias
from .model import Identifier, Model, Period
from .numeric import UnitCertification


@dataclass(frozen=True, kw_only=True)
class EntityMapping(Model):
    """每个 API 的业务字段绑定命名空间；不按接口间业务哈希推断关系。"""

    source_api: Identifier
    namespace: Identifier
    key_field: Identifier


@dataclass(frozen=True, kw_only=True)
class TimeMapping(Model):
    """字段语义与市场时区显式登记；报告期不充当可用时间。"""

    source_api: Identifier
    version: Identifier = "time-mapping/v1"
    data_field: Identifier | None = None
    public_fields: tuple[Identifier, ...] = ()
    optional_public_fields: tuple[Identifier, ...] = ()
    market_timezone: str | None = None
    report_period_field: Identifier | None = None
    effective_from_field: Identifier | None = None
    effective_to_field: Identifier | None = None


@dataclass(frozen=True, kw_only=True)
class FoundationBindings(Model):
    """只对证据明确的字段启用适配；默认空配置不会猜测身份或认证单位。"""

    entities: tuple[Entity, ...] = ()
    aliases: tuple[EntityAlias, ...] = ()
    certifications: tuple[UnitCertification, ...] = ()
    entity_mappings: tuple[EntityMapping, ...] = ()
    time_mappings: tuple[TimeMapping, ...] = ()
    expected_data_as_of: dict[Identifier, Period] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """拒绝同一来源或字段重复配置；例如同 daily.amount 两份证书 → 拒绝。

        输入版本化显式绑定，输出无歧义配置，别名歧义留给时点解析拒绝。
        """
        super().__post_init__()
        groups: list[list[Hashable]] = [
            [item.identity.entity_id for item in self.entities],
            [(item.source_api, item.field) for item in self.certifications],
            [item.source_api for item in self.entity_mappings],
            [item.source_api for item in self.time_mappings],
        ]
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("duplicate_foundation_binding")
