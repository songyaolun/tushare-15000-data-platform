"""从accepted主数据构建有生命周期的证券身份及当前观察关系。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from ..pit import time_boundary
from ..product_foundation.entities import (
    Entity,
    EntityAlias,
    EntityIdentity,
    EntityKind,
    EntityRelation,
)
from ..product_foundation.numeric import source_evidence
from ..product_foundation.temporal import AvailabilityFact
from ..store import digest
from .binding_rules import BINDING_VERSION

MASTER_APIS = ("stock_basic", "index_basic", "cb_basic", "bse_mapping")
NAMESPACES = {
    "stock_basic": "tushare_stock",
    "index_basic": "tushare_index",
    "cb_basic": "tushare_cb",
}
KINDS = {
    "stock_basic": EntityKind.SECURITY,
    "index_basic": EntityKind.INDEX,
    "cb_basic": EntityKind.CONVERTIBLE_BOND,
}
STOCK_CODE = re.compile(r"(?:6\d{5}\.SH|(?:00|30)\d{4}\.SZ|(?:[48]\d{5}|92\d{4})\.BJ)")
SECURITY_CODE = re.compile(r"\d{6}\.[A-Z]{2,6}")


@dataclass
class EntityBuild:
    """有限主数据集合及其输出；重复代码整组拒绝，不挑一个来源版本。"""

    maximum_entities: int
    rows: dict[tuple[str, str], tuple[dict[str, Any], AvailabilityFact]] = field(
        default_factory=dict
    )
    conflicts: set[tuple[str, str]] = field(default_factory=set)
    entities: list[Entity] = field(default_factory=list)
    aliases: list[EntityAlias] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
    peers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reasons: dict[str, Counter[str]] = field(default_factory=dict)
    extra_namespaces: dict[str, str] = field(default_factory=dict)
    extra_kinds: dict[str, EntityKind] = field(default_factory=dict)
    extra_key_fields: dict[str, str] = field(default_factory=dict)
    observation_only: frozenset[str] = frozenset()
    directory_identity_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def gap(self, api: str, reason: str) -> None:
        """按来源累计实体缺口而不保存来源载荷；例：缺行业增加industry_unclassified计数。"""
        self.reasons.setdefault(api, Counter())[reason] += 1

    def add(self, row: dict[str, Any], availability: AvailabilityFact) -> None:
        """接收已通过来源及时间校验的主数据，重复或变体代码不参与后续身份推断。

        输入一条stock_basic得到一个待构建对象；同代码第二条记录将整组标记冲突。
        显式observation_only目录仅保存最新完整业务期，例如DC旧日期不会淘汰当前目录。
        """
        api = availability.source_api
        code = row.get(
            self.extra_key_fields.get(api, "o_code" if api == "bse_mapping" else "ts_code")
        )
        pattern = (
            re.compile(r"[A-Za-z0-9_.-]+")
            if api in self.extra_namespaces
            else STOCK_CODE
            if api in {"stock_basic", "bse_mapping"}
            else SECURITY_CODE
        )
        if not isinstance(code, str) or not pattern.fullmatch(code):
            self.gap(api, "entity_code_unrecognized")
            return
        key = (api, code)
        if api in self.directory_identity_fields:
            self.add_directory(row, availability, key)
            return
        if api in self.observation_only and key in self.rows:
            previous = self.rows[key][1].data_as_of or ""
            current = availability.data_as_of or ""
            if current < previous:
                return
            if current > previous:
                self.rows.pop(key)
                self.conflicts.discard(key)
        if key in self.rows:
            self.conflicts.add(key)
            self.gap(api, "entity_duplicate_conflict")
        self.rows[key] = (row, availability)
        if len(self.rows) > self.maximum_entities:
            raise ValueError("binding_entity_capacity_exceeded")

    def add_directory(
        self,
        row: dict[str, Any],
        availability: AvailabilityFact,
        key: tuple[str, str],
    ) -> None:
        """聚合同一来源显式提供的目录代码和名称，检查所有成员行的一致性。

        例如CI两成员均声明l3_code=CI1且l3_name=行业甲时保留一份当前目录依据；
        任一名称缺失或冲突即拒绝整个代码。代表证据按观察时间和来源ID确定，顺序无关。
        """
        fields = self.directory_identity_fields[availability.source_api]
        previous = self.rows.get(key)
        if any(not isinstance(row.get(name), str) or not row[name].strip() for name in fields):
            self.conflicts.add(key)
            self.gap(availability.source_api, "directory_identity_incomplete")
        elif previous is not None and any(row[name] != previous[0].get(name) for name in fields):
            self.conflicts.add(key)
            self.gap(availability.source_api, "directory_identity_conflict")
        if previous is None or (availability.observed_at, availability.source_record_id) > (
            previous[1].observed_at,
            previous[1].source_record_id,
        ):
            self.rows[key] = (row, availability)
        if len(self.rows) > self.maximum_entities:
            raise ValueError("binding_entity_capacity_exceeded")

    def materialize(self) -> None:
        """按代码排序创建主身份，再构建BSE别名和转债关系，输出与行顺序无关。

        输入两条无冲突股票得到两个实体及别名；缺行业股票保留身份并报告未分类。
        显式当前目录使用series身份和observed_at作为断言起点，不伪造历史上市日期。
        """
        for (api, code), (row, availability) in sorted(self.rows.items()):
            if (api, code) in self.conflicts or api == "bse_mapping":
                continue
            observed_directory = api in self.observation_only
            effective_from = (
                availability.observed_at if observed_directory else availability.effective_from
            )
            if effective_from is None:
                self.gap(api, "entity_lifecycle_missing")
                continue
            if api == "stock_basic" and (
                row.get("list_status") not in {"L", "D", "P"}
                or (row.get("list_status") == "D") != (availability.effective_to is not None)
                or row.get("market") == "CDR"
            ):
                self.gap(api, "entity_lifecycle_conflict")
                continue
            identity = EntityIdentity(
                namespace=(self.extra_namespaces | NAMESPACES)[api],
                kind=(self.extra_kinds | KINDS)[api],
                local_id=code,
                lifecycle_id=("observed_" + digest(availability.observed_at))
                if observed_directory
                else effective_from[:10],
            )
            self.entities.append(Entity(identity=identity, name=code))
            key_field = self.extra_key_fields.get(api, "ts_code")
            evidence = source_evidence(row, key_field, availability)
            evidence = replace(
                evidence,
                fields=tuple(
                    sorted(
                        ({key_field} if observed_directory else {key_field, "list_date"})
                        | set(self.directory_identity_fields.get(api, ()))
                        | ({"delist_date", "curr_type", "industry"} & row.keys())
                    )
                ),
            )
            self.aliases.append(
                EntityAlias(
                    namespace=(self.extra_namespaces | NAMESPACES)[api],
                    alias=code,
                    entity_id=identity.entity_id,
                    effective_from=effective_from,
                    effective_to=None if observed_directory else availability.effective_to,
                    observed_at=availability.observed_at,
                    source=api,
                    version="research-directory/v1" if observed_directory else BINDING_VERSION,
                    confidence=100,
                    evidence_refs=(evidence,),
                )
            )
            if api == "stock_basic":
                industry = row.get("industry")
                if isinstance(industry, str) and industry.strip():
                    self.peers[identity.entity_id] = ("observed-industry/" + digest(industry),)
                else:
                    self.gap(api, "industry_unclassified")
        self.bse_aliases()
        self.underlying_relations()

    def bse_aliases(self) -> None:
        """只接受旧新代码对中恰有一个主身份且名称、上市日一致的BSE证据。

        输入旧代码已知的新旧映射，新增另一代码的观察期别名；两端已有不同实体或
        一对多映射均报告冲突。无换码生效日时别名仅从当前观察起有效，不回填历史。
        """
        mappings = [
            (row, fact)
            for (api, _), (row, fact) in sorted(self.rows.items())
            if api == "bse_mapping"
        ]
        counts = Counter(
            code for row, _ in mappings for code in (row.get("o_code"), row.get("n_code"))
        )
        for row, fact in mappings:
            old, new = row.get("o_code"), row.get("n_code")
            if (
                not isinstance(new, str)
                or not STOCK_CODE.fullmatch(new)
                or not (str(old).endswith(".BJ") and new.endswith(".BJ"))
            ):
                self.gap("bse_mapping", "bse_code_unrecognized")
                continue
            known = [
                alias
                for alias in self.aliases
                if alias.namespace == "tushare_stock" and alias.alias in {old, new}
            ]
            if (
                len(known) != 1
                or (("stock_basic", str(old)) in self.rows and ("stock_basic", new) in self.rows)
                or old == new
                or counts[old] != 1
                or counts[new] != 1
                or ("bse_mapping", str(old)) in self.conflicts
            ):
                self.gap("bse_mapping", "bse_mapping_conflict")
                continue
            alias = known[0]
            master, _ = self.rows[("stock_basic", alias.alias)]
            if (
                not row.get("name")
                or row.get("name") != master.get("name")
                or row.get("list_date") != master.get("list_date")
            ):
                self.gap("bse_mapping", "bse_evidence_insufficient")
                continue
            start = max(alias.effective_from, fact.observed_at, key=time_boundary)
            if alias.effective_to is not None and time_boundary(start) >= time_boundary(
                alias.effective_to
            ):
                self.gap("bse_mapping", "bse_lifecycle_expired")
                continue
            evidence = replace(
                source_evidence(row, "o_code", fact),
                fields=("o_code", "n_code", "name", "list_date"),
            )
            self.aliases.append(
                replace(
                    alias,
                    alias=new if alias.alias == old else str(old),
                    effective_from=start,
                    observed_at=max(alias.observed_at, fact.observed_at, key=time_boundary),
                    source="bse_mapping",
                    evidence_refs=(*alias.evidence_refs, evidence),
                )
            )
            self.gap("bse_mapping", "bse_observed_alias_only")

    def underlying_relations(self) -> None:
        """仅按cb_basic.stk_code关联唯一股票，关系有效期取两个生命周期交集。

        输入明确正股代码的CB得到cb_underlying；缺字段、EB或正股未知保持关系缺口。
        证据可见性独立保留，关系不能认证历史行业或历史交易资格。
        """
        stocks = {item.alias: item for item in self.aliases if item.namespace == "tushare_stock"}
        for bond in [item for item in self.aliases if item.namespace == "tushare_cb"]:
            row, fact = self.rows[("cb_basic", bond.alias)]
            stock = stocks.get(str(row.get("stk_code", "")))
            if stock is None or row.get("cb_type") == "EB":
                self.gap("cb_basic", "cb_underlying_unresolved")
                continue
            start = max(bond.effective_from, stock.effective_from, key=time_boundary)
            ends = [value for value in (bond.effective_to, stock.effective_to) if value is not None]
            end = min(ends, key=time_boundary) if ends else None
            if end is not None and time_boundary(start) >= time_boundary(end):
                self.gap("cb_basic", "cb_underlying_lifecycle_conflict")
                continue
            self.relations.append(
                EntityRelation(
                    from_entity_id=bond.entity_id,
                    to_entity_id=stock.entity_id,
                    relation_type="cb_underlying",
                    confirmed=True,
                    effective_from=start,
                    effective_to=end,
                    observed_at=max(bond.observed_at, stock.observed_at, key=time_boundary),
                    source="cb_basic",
                    version=BINDING_VERSION,
                    confidence=100,
                    evidence_refs=(source_evidence(row, "stk_code", fact), *stock.evidence_refs),
                )
            )

    @cached_property
    def currency_sources(self) -> dict[str, tuple[dict[str, Any], AvailabilityFact]]:
        """为已接受的股票别名一次建立币种证据索引，避免每行遍历全部实体。

        输入5000只股票的主数据，输出按ts_code查询的字典；冲突代码不进入索引。
        """
        records = {row["source_record_id"]: (row, fact) for row, fact in self.rows.values()}
        return {
            alias.alias: records[alias.evidence_refs[0].source_record_id]
            for alias in self.aliases
            if alias.namespace == "tushare_stock"
        }
