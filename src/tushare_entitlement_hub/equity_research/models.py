"""六产品共用的显式适配输入与可复算结果，不改变基础层数据契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

from ..product_foundation.bindings import FoundationBindings
from ..product_foundation.config import FoundationPolicy
from ..product_foundation.contracts import DecisionEnvelope, EvidenceRef, Truth
from ..product_foundation.entities import EntityAlias, EntityRelation
from ..product_foundation.model import Count, EntityId, Identifier, Model, ReasonCode
from ..product_foundation.numeric import TypedFact, source_evidence
from ..product_foundation.temporal import AvailabilityFact
from ..store import encode

RULE_VERSION = "equity-research/v1"
MARKET_ENTITY = "cn:market:A:all"
PRODUCT_IDS = (
    "market_context/v1",
    "fundamental_dossier/v1",
    "capital_governance_risks/v1",
    "flow_feasibility/v1",
    "equity_research_candidates/v1",
    "event_thesis_review/v1",
)
CANDIDATE_RANKING_PRODUCT_IDS = (
    "equity_short_term_candidates/v1",
    "equity_medium_term_candidates/v1",
    "equity_long_term_candidates/v1",
)
ALL_EQUITY_PRODUCT_IDS = PRODUCT_IDS[:5] + CANDIDATE_RANKING_PRODUCT_IDS + PRODUCT_IDS[5:]


def default_thresholds() -> FoundationPolicy:
    """返回公开初始阈值；例：默认同行最小样本为30，批大小为512。

    每次构造独立配置，不在导入时打开数据库或读取外部文件。
    """
    return FoundationPolicy(version="research-thresholds/v1")


@dataclass(frozen=True, kw_only=True)
class ResearchConfig(Model):
    """范围及口径认证由调用方显式声明，空配置只产生降级结果。"""

    version: Identifier = RULE_VERSION
    foundation: FoundationBindings = field(default_factory=FoundationBindings)
    thresholds: FoundationPolicy = field(default_factory=default_thresholds)
    peer_paths: dict[EntityId, tuple[Identifier, ...]] = field(default_factory=dict)
    certified_report_types: tuple[Identifier, ...] = ()
    complete_scopes: tuple[Identifier, ...] = ()
    confirmed_relations: tuple[EntityRelation, ...] = ()
    maximum_rows_per_entity_source: Count = 512
    maximum_source_rows: Count = 32768
    maximum_total_rows: Count = 131072
    binding_audit_json: str = "{}"

    def __post_init__(self) -> None:
        """检查配置版本、容量与同行路径；例：容量为0或空同行路径均拒绝。

        输入合法配置得到不可变认证边界，complete_scopes须使用API名称。
        """
        super().__post_init__()
        if not isinstance(json.loads(self.binding_audit_json), dict):
            raise ValueError("invalid_binding_audit")
        if (
            self.version != RULE_VERSION
            or self.maximum_rows_per_entity_source < 1
            or self.maximum_source_rows < 1
            or self.maximum_total_rows < 1
            or any(not path for path in self.peer_paths.values())
        ):
            raise ValueError("invalid_research_configuration")

    @cached_property
    def aliases_by_key(self) -> dict[tuple[str, str], tuple[EntityAlias, ...]]:
        """按命名空间和代码一次建立别名索引，保留同代码的全部生命周期及冲突。

        输入5000个股票别名，逐行解析只检查当前代码的候选；同代码两实体仍失败关闭。
        """
        grouped: dict[tuple[str, str], list[EntityAlias]] = {}
        for alias in self.foundation.aliases:
            grouped.setdefault((alias.namespace, alias.alias), []).append(alias)
        return {key: tuple(values) for key, values in grouped.items()}

    @cached_property
    def relations_by_source(self) -> dict[tuple[str, str], tuple[EntityRelation, ...]]:
        """按来源实体和关系类型一次索引声明关系，保留原顺序、重复及未确认条目。

        例如(A, 'cb_underlying')只返回A的转股关系候选；确认与有效期仍由调用方逐条校验。
        索引只持有原对象引用，不进入配置序列化或改变运行指纹。
        """
        grouped: dict[tuple[str, str], list[EntityRelation]] = {}
        for relation in self.confirmed_relations:
            grouped.setdefault((relation.from_entity_id, relation.relation_type), []).append(
                relation
            )
        return {key: tuple(values) for key, values in grouped.items()}

    @property
    def binding_audit(self) -> dict[str, Any]:
        """返回生成配置的独立审计对象；例：默认配置返回{}，不自动信任来源。

        输入配置内的规范JSON，输出可供运行边界核验的指纹、截止及来源缺口。
        """
        result: dict[str, Any] = json.loads(self.binding_audit_json)
        return result


@dataclass(frozen=True, kw_only=True)
class ResearchRow:
    """一条已通过时点和来源身份门禁的accepted证据，数值另经单位认证。"""

    entity_id: str | None
    source_api: str
    raw: dict[str, Any]
    availability: AvailabilityFact
    numeric: dict[str, TypedFact] = field(default_factory=dict)

    def number(self, name: str, unit: str | None = None) -> float | None:
        """读取合格数值，单位不匹配保留未知；例：amount的元证书对股请求返回None。

        输入字段名与可选目标单位，输出有限数值或None，绝不从原始字符串猜数。
        """
        fact = self.numeric.get(name)
        if (
            fact is None
            or fact.reason_codes
            or (unit is not None and fact.unit != unit)
            or (fact.dimension in {"money", "price"} and fact.currency != "CNY")
        ):
            return None
        return fact.value

    def text(self, name: str) -> str | None:
        """读取已有文本标识，数值编码转字符串；例：comp_type=1得到'1'，缺失得到None。

        输入字段名，输出非空标识；不把结构化对象或布尔值解释成业务标签。
        """
        value = self.raw.get(name)
        return str(value) if type(value) in {str, int, float} and str(value).strip() else None

    def evidence(self, *names: str) -> tuple[EvidenceRef, ...]:
        """对实际存在字段生成引用；例：evidence('close','absent')只返回close引用。

        来源三层身份和公开时间全部保留，未声明字段不会产生伪造血缘。
        """
        return tuple(
            source_evidence(self.raw, name, self.availability) for name in names if name in self.raw
        )


@dataclass(frozen=True, kw_only=True)
class ResearchInputs:
    """扫描后按来源与实体分组的有限观察集，容量溢出整组拒绝而非截断计分。"""

    analysis_as_of: str
    run_id: str
    config: ResearchConfig
    rows_by_api: dict[str, tuple[ResearchRow, ...]] = field(default_factory=dict)
    source_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    entity_source_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @cached_property
    def entity_kinds(self) -> dict[str, str]:
        """一次索引显式实体类型，避免高基数上下文逐行重复遍历全部实体。

        例如登记cn:index:I:published后，该键映射index；未登记身份不产生猜测值。
        """
        return {
            entity.identity.entity_id: entity.identity.kind.value
            for entity in self.config.foundation.entities
        }

    @cached_property
    def record_periods(self) -> dict[tuple[str, str], str]:
        """一次建立来源行的时间索引；例：(daily,id)映射交易日，不复制来源载荷。

        信封按证据直接取日期，避免逐证券重复扫描整批来源。
        """
        return {
            (api, str(row.raw["source_record_id"])): row.availability.data_as_of
            for api, rows in self.rows_by_api.items()
            for row in rows
            if row.availability.data_as_of is not None
        }

    @cached_property
    def entity_rows(self) -> dict[tuple[str, str], tuple[ResearchRow, ...]]:
        """一次按接口及实体建立对象引用索引；例：daily/A只含A的来源对象。

        不复制raw数据，实体无歧义时后续按键直接取行。
        """
        indexed: dict[tuple[str, str], list[ResearchRow]] = {}
        for api, rows in self.rows_by_api.items():
            for row in rows:
                if row.entity_id is not None:
                    indexed.setdefault((api, row.entity_id), []).append(row)
        return {key: tuple(rows) for key, rows in indexed.items()}

    @cached_property
    def code_rows(self) -> dict[tuple[str, str], tuple[ResearchRow, ...]]:
        """按接口和text('ts_code')的原有口径一次索引来源行，供无实体旁证精确查询。

        例如('cb_basic', 'CB1')只返回该代码的行，缺代码不入索引；不合并不同接口。
        保留行顺序、重复、已绑定及未绑定行，不复制载荷，也不预先过滤时间、质量或隔离。
        """
        indexed: dict[tuple[str, str], list[ResearchRow]] = {}
        for api, rows in self.rows_by_api.items():
            for row in rows:
                code = row.text("ts_code")
                if code is not None:
                    indexed.setdefault((api, code), []).append(row)
        return {key: tuple(rows) for key, rows in indexed.items()}

    @cached_property
    def entity_ids(self) -> tuple[str, ...]:
        """一次缓存输入快照中显式登记的股票实体；例：一个security和一个series得到一个股票身份。

        不从新闻标题或跨接口业务哈希推断证券身份。逐结果读取不重复遍历和排序证券集；
        数据变化须以replace创建新ResearchInputs，与entity_rows等既有索引一起重新计算。
        """
        from .consumption_registry import BOND_APIS, INDEX_APIS

        bond_ids = {
            row.entity_id for api in BOND_APIS | INDEX_APIS for row in self.rows_by_api.get(api, ())
        }
        bond_ids.update(
            relation.from_entity_id
            for relation in self.config.confirmed_relations
            if relation.relation_type == "cb_underlying"
        )
        return tuple(
            sorted(
                entity.identity.entity_id
                for entity in self.config.foundation.entities
                if entity.identity.kind.value == "security"
                and entity.identity.entity_id not in bond_ids
            )
        )

    def rows(self, api: str, entity_id: str | None = None) -> tuple[ResearchRow, ...]:
        """按API和可选实体取证据；例：rows('daily',股票A)不含股票B记录。

        entity_id为None表示全来源，包括尚未确认实体的事件。
        """
        return (
            self.rows_by_api.get(api, ())
            if entity_id is None
            else self.entity_rows.get((api, entity_id), ())
        )

    def reasons(self, api: str, entity_id: str) -> tuple[str, ...]:
        """合并接口和实体局部缺口；例：缺单位与容量溢出都保留而非覆盖。

        输入API、实体身份，输出排序且去重的原因码。
        """
        return tuple(
            sorted(
                set(self.source_reasons.get(api, ()))
                | set(self.entity_source_reasons.get(api + "/" + entity_id, ()))
            )
        )


@dataclass(frozen=True, kw_only=True)
class Metric(Model):
    """特征原值与缺失状态分开，unknown和not_applicable均不可转换成数值0。"""

    name: Identifier
    value: float | None
    unit: str
    status: Literal["available", "unknown", "not_applicable"] = "available"
    reason_codes: tuple[ReasonCode, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        """验证空值、原因、范围和证据；例：fraction=1.1或未知值带0均拒绝。

        比值ratio不强行裁剪到0—1，保留负现金流或高杠杆等异常原值。
        """
        super().__post_init__()
        if self.status == "available":
            if self.value is None or not self.evidence_refs:
                raise ValueError("metric_evidence_required")
            if (self.unit == "fraction" and not 0 <= self.value <= 1) or (
                self.unit == "score" and not 0 <= self.value <= 100
            ):
                raise ValueError("metric_range_invalid")
        elif self.value is not None or not self.reason_codes:
            raise ValueError("metric_unknown_invalid")


@dataclass(frozen=True, kw_only=True)
class ResearchResult(Model):
    """业务结果包含公共信封、结构化特征、上游引用及三值论点。"""

    envelope: DecisionEnvelope
    metrics: tuple[Metric, ...] = ()
    details_json: str = "{}"
    parent_result_ids: tuple[Identifier, ...] = ()
    predicate_results: dict[Identifier, Truth] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """重验特征唯一性与扩展JSON；例：重复cash_cover或JSON含NaN均拒绝。

        输出只包含有限数值和无凭据字段的可重放结果。
        """
        super().__post_init__()
        details = json.loads(self.details_json)
        if not isinstance(details, dict) or len({m.name for m in self.metrics}) != len(
            self.metrics
        ):
            raise ValueError("research_result_details_invalid")
        encode(details)

    def metric(self, name: str) -> float | None:
        """读取一个可用特征；例：不存在的momentum_12_1返回None，不能当作0。

        输入特征标识，输出其有限数值或未知。
        """
        return next((item.value for item in self.metrics if item.name == name), None)

    @property
    def details(self) -> dict[str, Any]:
        """返回扩展解释的独立字典；例：'{}'得到{}，修改它不影响持久化内容。"""
        result: dict[str, Any] = json.loads(self.details_json)
        return result
