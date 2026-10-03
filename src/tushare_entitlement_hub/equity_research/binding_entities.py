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

    def gap(self, api: str, reason: str) -> None:
        """按来源累计实体缺口而不保存来源载荷；例：缺行业增加industry_unclassified计数。"""
        self.reasons.setdefault(api, Counter())[reason] += 1

    def add(self, row: dict[str, Any], availability: AvailabilityFact) -> None:
        """接收已通过来源及时间校验的主数据，重复或变体代码不参与后续身份推断。

        输入一条stock_basic得到一个待构建对象；同代码第二条记录将整组标记冲突。
        """
        api = availability.source_api
        code = row.get("o_code" if api == "bse_mapping" else "ts_code")
        pattern = STOCK_CODE if api in {"stock_basic", "bse_mapping"} else SECURITY_CODE
        if not isinstance(code, str) or not pattern.fullmatch(code):
            self.gap(api, "entity_code_unrecognized")
            return
        key = (api, code)
        if key in self.rows:
            self.conflicts.add(key)
            self.gap(api, "entity_duplicate_conflict")
        self.rows[key] = (row, availability)
        if len(self.rows) > self.maximum_entities:
            raise ValueError("binding_entity_capacity_exceeded")

    def materialize(self) -> None:
        """按代码排序创建主身份，再构建BSE别名和转债关系，输出与行顺序无关。

        输入两条无冲突股票得到两个实体及别名；缺行业股票保留身份并报告未分类。
        """
        for (api, code), (row, availability) in sorted(self.rows.items()):
            if (api, code) in self.conflicts or api == "bse_mapping":
                continue
            if availability.effective_from is None:
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
                namespace=NAMESPACES[api],
                kind=KINDS[api],
                local_id=code,
                lifecycle_id=availability.effective_from[:10],
            )
            self.entities.append(Entity(identity=identity, name=code))
            evidence = source_evidence(row, "ts_code", availability)
            evidence = replace(
                evidence,
                fields=tuple(
                    sorted(
                        {"ts_code", "list_date"}
                        | ({"delist_date", "curr_type", "industry"} & row.keys())
                    )
                ),
            )
            self.aliases.append(
                EntityAlias(
                    namespace=NAMESPACES[api],
                    alias=code,
                    entity_id=identity.entity_id,
                    effective_from=availability.effective_from,
                    effective_to=availability.effective_to,
                    observed_at=availability.observed_at,
                    source=api,
                    version=BINDING_VERSION,
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
