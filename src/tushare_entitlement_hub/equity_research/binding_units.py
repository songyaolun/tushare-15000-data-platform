"""只认证标准层可证明且全行一致的单位，保留可回查来源证据。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..product_foundation.contracts import EvidenceRef
from ..product_foundation.numeric import UnitCertification, source_evidence
from ..product_foundation.temporal import AvailabilityFact
from ..reference_contract import ReferenceContract
from ..standard import EQUITY_LOT_APIS, UNIT_CONVERSIONS, _convert_unit
from .binding_entities import EntityBuild
from .binding_rules import BINDING_VERSION
from .consumption_registry import BOND_APIS, INDEX_APIS

PRICE_FIELDS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "price",
        "conv_price",
        "first_conv_price",
        "up_limit",
        "down_limit",
        "close_qfq",
        "close_hfq",
        "weight_avg",
        "issue_price",
        "par",
        "convert_price",
        "cb_value",
        "cost_5pct",
        "cost_50pct",
        "cost_95pct",
    }
)
FRACTION_FIELDS = frozenset(
    {
        "turnover_rate",
        "turnover_rate_f",
        "pledge_ratio",
        "float_ratio",
        "percent",
        "hold_ratio",
        "hold_float_ratio",
        "change_ratio",
        "after_ratio",
        "winner_rate",
    }
)


def currency_evidence(
    row: dict[str, Any],
    fact: AvailabilityFact,
    entities: EntityBuild,
    *,
    allowed_currencies: frozenset[str] = frozenset({"CNY"}),
    inherit_stock_currency: bool = True,
) -> tuple[str | None, tuple[EvidenceRef, ...]]:
    """币种只能来自本行声明或已确认股票主数据；不同声明相互冲突时拒绝。

    输入daily代码对应stock_basic.curr_type=CNY，输出CNY和主数据引用；无证据返回None。
    元、万元等金额单位本身不能替代币种声明，转债也不从正股推断发行币种。
    例如显式allowed_currencies={'USD'}及inherit_stock_currency=False，只认证行内USD。
    """
    references: list[EvidenceRef] = []
    values = []
    for name in ("currency", "curr_type"):
        if row.get(name) is not None:
            values.append(row[name])
            references.append(source_evidence(row, name, fact))
    master = (
        entities.currency_sources.get(str(row.get("ts_code", "")))
        if inherit_stock_currency and fact.source_api not in BOND_APIS | INDEX_APIS
        else None
    )
    if master is not None and master[0].get("curr_type") is not None:
        values.append(master[0]["curr_type"])
        references.append(source_evidence(master[0], "curr_type", master[1]))
    values = ["CNY" if value == "人民币" else value for value in values]
    if not values or any(value not in allowed_currencies for value in values):
        return None, ()
    if len(set(values)) != 1:
        return None, ()
    return values[0], tuple(references)


def unit_definition(
    row: dict[str, Any],
    name: str,
    contract: ReferenceContract,
    *,
    preserve_contract_lots: bool = False,
) -> tuple[str, str, str, float]:
    """复用标准转换表，返回当前单位、目标语义、量纲及尚需应用的倍数。

    输入已转为元的daily.amount，输出(元,元,money,1)；输入比例转ratio也只乘1。
    未知文档、声明冲突、标准单位偏离转换规则及非支持单位抛固定缺口码。
    preserve_contract_lots=True仅允许期货/期权已明确保留的手数，例如手→手、quantity、1倍。
    """
    policy_unit = contract.policy.units.get(name)
    source_units = row.get("source_units", {})
    standard_units = row.get("standard_units", {})
    if not isinstance(source_units, dict) or not isinstance(standard_units, dict):
        raise ValueError("unit_metadata_invalid")
    source = source_units.get(name) or policy_unit or standard_units.get(name)
    if policy_unit and "未说明" in policy_unit:
        raise ValueError("unit_documentation_unknown")
    if policy_unit and source != policy_unit:
        raise ValueError("unit_policy_conflict")
    if (
        preserve_contract_lots
        and contract.policy.category in {"futures", "options"}
        and policy_unit == "手"
    ):
        if source_units.get(name) != "手" or standard_units.get(name) != "手":
            raise ValueError("contract_lot_unit_unverified")
        return "手", "手", "quantity", 1.0
    if not isinstance(source, str) or not (
        source in UNIT_CONVERSIONS
        or (source == "手" and contract.policy.api_name in EQUITY_LOT_APIS)
    ):
        raise ValueError("unit_unknown")
    multiplier, target = _convert_unit(1, source, contract.policy)
    current = standard_units.get(name)
    if current is None and policy_unit is not None and source != target:
        raise ValueError("unit_standard_missing")
    if current is not None and current != target:
        raise ValueError("unit_standard_conflict")
    if current is not None:
        source, multiplier = current, 1.0
    dimension = {
        "元": "price" if name in PRICE_FIELDS else "money",
        "元/股": "price",
        "股": "quantity",
        "张": "quantity",
        "份": "quantity",
        "比例": "ratio",
    }[target]
    if target == "比例":
        target = "fraction" if name in FRACTION_FIELDS else "ratio"
    return source, target, dimension, float(multiplier)


@dataclass
class UnitBuild:
    """单接口按字段累积一致性状态；只保留一份代表证据，内存不随行数增长。"""

    definitions: dict[str, tuple[str, str, str, str | None, float]] = field(default_factory=dict)
    certificates: dict[str, UnitCertification] = field(default_factory=dict)
    gaps: dict[str, Counter[str]] = field(default_factory=dict)
    allowed_currencies: frozenset[str] = frozenset({"CNY"})
    inherit_stock_currency: bool = True
    preserve_contract_lots: bool = False
    report_currency_variation: bool = False
    prefer_row_currency: bool = False

    def add(
        self,
        row: dict[str, Any],
        fact: AvailabilityFact,
        contract: ReferenceContract,
        names: set[str],
        entities: EntityBuild,
    ) -> None:
        """逐字段核验所有有值行，任一冲突使该字段最终不发证书。

        输入两行amount标准单位元且CNY，保留记录ID较小的证据；不同币种报告缺口。
        行值不会写入配置，来源文件指纹约束证书适用的完整扫描范围。
        新调用方可优先采用本行币种，并将混币报告为currency_varies_by_row；整列仍不发证。
        """
        for name in sorted(names):
            if row.get(name) is None:
                self.gaps.setdefault(name, Counter())["unit_value_missing"] += 1
                continue
            try:
                source, target, dimension, multiplier = unit_definition(
                    row, name, contract, preserve_contract_lots=self.preserve_contract_lots
                )
                currency, references = (
                    currency_evidence(
                        row,
                        fact,
                        entities,
                        allowed_currencies=self.allowed_currencies,
                        inherit_stock_currency=self.inherit_stock_currency
                        and not (
                            self.prefer_row_currency
                            and (
                                row.get("currency") is not None or row.get("curr_type") is not None
                            )
                        ),
                    )
                    if dimension in {"money", "price"}
                    else (None, ())
                )
                if dimension in {"money", "price"} and currency is None:
                    raise ValueError("currency_unverified")
                definition = (source, target, dimension, currency, multiplier)
                if name in self.definitions and self.definitions[name] != definition:
                    previous_definition = self.definitions[name]
                    if (
                        self.report_currency_variation
                        and previous_definition[:3] == definition[:3]
                        and previous_definition[4] == definition[4]
                    ):
                        raise ValueError("currency_varies_by_row")
                    raise ValueError("unit_cross_row_conflict")
                self.definitions[name] = definition
                previous = self.certificates.get(name)
                if (
                    previous is not None
                    and fact.source_record_id >= previous.evidence_refs[0].source_record_id
                ):
                    continue
                certificate = UnitCertification(
                    version=BINDING_VERSION,
                    source_api=fact.source_api,
                    field=name,
                    source_unit=source,
                    target_unit=target,
                    dimension=dimension,
                    currency=currency,
                    multiplier=multiplier,
                    evidence_refs=(source_evidence(row, name, fact), *references),
                )
                self.certificates[name] = certificate
            except (ValueError, TypeError, KeyError) as error:
                allowed = {
                    "unit_metadata_invalid",
                    "unit_documentation_unknown",
                    "unit_policy_conflict",
                    "unit_unknown",
                    "unit_standard_conflict",
                    "unit_standard_missing",
                    "currency_unverified",
                    "unit_cross_row_conflict",
                    "contract_lot_unit_unverified",
                    "currency_varies_by_row",
                }
                reason = str(error) if str(error) in allowed else "unit_evidence_invalid"
                self.gaps.setdefault(name, Counter())[reason] += 1

    def accepted(self) -> tuple[UnitCertification, ...]:
        """只返回没有任何有值行认证失败的证书；例：一个单位冲突剔除整列证书。

        空值仍可诚实unknown，不使其他已证明有值行失去单位说明。
        """
        return tuple(
            self.certificates[name]
            for name in sorted(self.certificates)
            if not (set(self.gaps.get(name, {})) - {"unit_value_missing"})
        )
