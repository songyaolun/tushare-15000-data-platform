"""P06市场债券的全量分组消费、单位隔离统计和有界证据；不生成投资信号。"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from typing import Any

from ..product_foundation.contracts import Truth
from ..store import digest, encode
from .common import gate
from .consumption_registry import Consumer, ConsumptionRegistration
from .context_aggregation import ContextAggregate, fingerprint_leaf, row_fingerprint
from .context_digest import DiskKeySet, MultisetDigest
from .disclosure_consumers import _date, _underlying_relations
from .models import MARKET_ENTITY, ResearchInputs, ResearchRow

BOND_AGGREGATION_VERSION = "bond-market-aggregate/v2"
MAXIMUM_DETAILED_BOND_ROWS = 8
MAXIMUM_GROUP_ROWS = 512
MAXIMUM_DIMENSION_VALUES = 8
MAXIMUM_DIMENSION_BYTES = 128
RAW_DIMENSIONS = {
    "cb_basic": ("cb_type",),
    "cb_call": ("call_type", "is_call", "call_date"),
    "cb_daily": ("trade_date",),
    "cb_factor_pro": ("trade_date",),
    "cb_issue": ("ann_date", "res_ann_date"),
    "cb_rate": ("rate_start_date", "rate_end_date", "rate_freq"),
    "cb_rating": ("rating_date", "rating_com_name", "rating_type", "rating", "rating_outlook"),
    "cb_share": ("publish_date", "end_date"),
    "top10_cb_holders": ("end_date", "holder_rank"),
}
NUMERIC_FIELDS = {
    "cb_basic": ("par", "conv_price", "remain_size"),
    "cb_call": ("call_amount",),
    "cb_daily": ("close", "cb_value", "cb_over_rate", "vol", "amount"),
    "cb_factor_pro": ("close", "vol", "amount"),
    "cb_issue": ("plan_issue_size", "issue_size"),
    "cb_rate": ("coupon_rate", "rate_freq"),
    "cb_rating": (),
    "cb_share": ("convert_price", "convert_vol", "acc_convert_vol", "remain_size"),
    "top10_cb_holders": ("hold_ratio",),
}
AUDIT_FIELDS = (
    "ts_code",
    "stk_code",
    "cb_type",
    "conv_start_date",
    "conv_end_date",
    "ann_date",
    "f_ann_date",
    "pub_date",
    "publish_date",
    "rating_type",
)


def bond_full_fingerprint(api: str, analysis_as_of: str, summary: dict[str, Any]) -> str:
    """绑定全量行、分组、证据及规范旁证来源期间，供生成与重建使用同一指纹算法。

    例：cb_daily原值与引用不变，仅未抽样cb_basic使source_as_of变化时，返回不同的64位摘要。
    """
    return digest(
        [
            BOND_AGGREGATION_VERSION,
            api,
            analysis_as_of,
            summary["rows_fingerprint"],
            summary["groups_fingerprint"],
            summary["evidence_fingerprint"],
            summary["source_as_of"],
        ]
    )


def bond_audit_fingerprint(summary: dict[str, Any]) -> str:
    """校验整个持久化审计对象，包含计数、统计、抽样声明及全量指纹，不包含自身。

    例：只改usable_rows或sample_strata，输出摘要也改变；JSON键顺序不影响结果。
    该摘要用于完整性检查，不是签名，也不能替代对参考库的来源验证。
    """
    return digest({key: value for key, value in summary.items() if key != "audit_fingerprint"})


def _group_key(row: ResearchRow) -> tuple[str, str]:
    """以实体、明确代码或独立来源身份分组，不把缺代码债券合并为同一债券。

    例：有实体A返回('entity', A)；缺实体和代码时按source_record_id独立承接。
    """
    if row.entity_id is not None:
        return "entity", row.entity_id
    code = row.text("ts_code")
    return ("code", code) if code is not None else ("record", row.availability.source_record_id)


class _Distribution:
    """保存最小8个分类的精确频次，其余计入other，空间不随分类数量增长。"""

    def __init__(self) -> None:
        """初始化分类计数；例：空输入的total、missing、oversized均为0。"""
        self.total = self.missing = self.oversized = 0
        self.values: Counter[str] = Counter()

    def add(self, value: Any) -> None:
        """记录一个观察标签；例：第9个分类按字典序淘汰，超128字节只计oversized而不输出原文。

        保留阈值单调减小，被淘汰分类不会重新进入，保留分类的频次因此仍是全量精确值。
        """
        self.total += 1
        if value is None or value == "":
            self.missing += 1
            return
        text = str(value)
        if len(text.encode("utf-8")) > MAXIMUM_DIMENSION_BYTES:
            self.oversized += 1
        elif text in self.values:
            self.values[text] += 1
        elif len(self.values) < MAXIMUM_DIMENSION_VALUES or text < max(self.values):
            if len(self.values) == MAXIMUM_DIMENSION_VALUES:
                del self.values[max(self.values)]
            self.values[text] = 1

    def summary(self) -> dict[str, Any]:
        """输出完整分母与有界分类，不将缺值归入其他；例：10种各1行得到8项及other=2。"""
        return {
            "total": self.total,
            "missing": self.missing,
            "oversized": self.oversized,
            "values": dict(sorted(self.values.items())),
            "other": self.total - self.missing - self.oversized - sum(self.values.values()),
            "value_limit": MAXIMUM_DIMENSION_VALUES,
            "value_byte_limit": MAXIMUM_DIMENSION_BYTES,
        }


class _Statistics:
    """全量业务标签、认证值范围及行摘要；不跨单位汇总，也不累加重复快照的存量。"""

    def __init__(self, api: str, fields: tuple[str, ...]) -> None:
        """绑定固定接口字段；例：cb_daily统计日期与五个认证字段，未认证字段不生成数值范围。"""
        self.api, self.fields = api, fields
        self.dimensions: dict[str, _Distribution] = {}
        self.numbers: dict[str, dict[str, Any]] = {}
        self.rows = MultisetDigest()
        self.distinct_holders = 0

    def count(self, name: str, value: Any) -> None:
        """为固定业务维度增加一次观察；例：relation/unknown使对应全量频次增加1。"""
        self.dimensions.setdefault(name, _Distribution()).add(value)

    def add(self, row: ResearchRow, holders: DiskKeySet, *, eligible: bool = True) -> None:
        """每条保留行进入摘要、标签和认证统计；例：未抽样的remain_size变化也改变完整摘要。

        数值只统计count/min/max，混合单位或币种时清空范围；正负零统一展示为0，原指纹保留区别。
        持有人去重键包含债券、规范报告期和名称；缺期或缺姓名单列，不猜测报告期。
        私有临时集合可溢写，不保留Python全量哈希集合。
        eligible=False的拒绝行仍进入行摘要和原始分类，但不进入认证数值统计。
        """
        self.rows.add(row_fingerprint(row, self.fields))
        for field in RAW_DIMENSIONS[self.api]:
            self.count(field, row.text(field))
        for field in NUMERIC_FIELDS[self.api]:
            value = row.number(field) if eligible else None
            self.count(
                field + "_availability",
                "certified" if value is not None else "unverified_or_missing",
            )
            record = self.numbers.setdefault(
                field,
                {
                    "certified_rows": 0,
                    "unit_basis": None,
                    "mixed_units": False,
                    "minimum": None,
                    "maximum": None,
                },
            )
            if value is None:
                continue
            fact = row.numeric[field]
            unit = (fact.unit, fact.dimension, fact.currency)
            if len(encode(unit).encode("utf-8")) > MAXIMUM_DIMENSION_BYTES:
                raise ValueError("research_bond_unit_capacity_exceeded")
            if record["certified_rows"] == 0:
                record["unit_basis"] = unit
            elif record["unit_basis"] != unit:
                record.update(unit_basis=None, mixed_units=True, minimum=None, maximum=None)
            record["certified_rows"] += 1
            if not record["mixed_units"]:
                value = 0.0 if value == 0 else value
                record["minimum"] = (
                    value if record["minimum"] is None else min(value, record["minimum"])
                )
                record["maximum"] = (
                    value if record["maximum"] is None else max(value, record["maximum"])
                )
        if self.api == "top10_cb_holders":
            name = row.text("holder_name")
            period = _date(row, "end_date")
            self.count("holder_name_availability", "present" if name is not None else "missing")
            self.count(
                "holder_period_availability",
                "known" if period is not None else "missing_or_invalid",
            )
            if name is not None and period is not None:
                self.distinct_holders += holders.add(
                    "holders", encode([_group_key(row), period, name]).encode()
                )

    def group(self, outcome: dict[str, Any]) -> None:
        """记录原处理器已经核验的业务阶段和核对状态，不从供应商原值推断结论。

        例：报价核对agreement=False计conflict；发行stage来自原处理器，票息期间逐项计状态。
        """
        observations = outcome["facts"].get("observations", ())
        facts = observations[0] if observations else {}
        self.count("underlying_relation_status", outcome["facts"]["binding_status"])
        if self.api in {"cb_daily", "cb_factor_pro"}:
            for name in ("close", "vol", "amount"):
                agreement = facts.get("quote_checks", {}).get(name, {}).get("agreement")
                self.count(
                    "quote_" + name,
                    "unknown" if agreement is None else "match" if agreement else "conflict",
                )
        if self.api == "cb_issue":
            self.count("issuance_stage", facts.get("stage"))
        if self.api == "cb_rate":
            for period in facts.get("periods", ()):
                self.count("coupon_period_state", period["period_state"])

    def summary(self) -> dict[str, Any]:
        """输出有界统计及明确口径；例：持有人覆盖仅代表已报告的债券/期间/持有人组合。"""
        return {
            "dimensions": {
                name: values.summary() for name, values in sorted(self.dimensions.items())
            },
            "numeric_ranges": self.numbers,
            "numeric_basis": "certified_usable_rows_no_snapshot_sum",
            "dimension_basis": {
                "raw_fields_and_availability": "all_retained_rows",
                "relation_issuance_quote": "bond_groups",
                "coupon_period_state": "consumer_reported_periods",
            },
            "distinct_bond_period_holder_pairs": self.distinct_holders,
            "holder_coverage": "reported_rows_only_not_full_holder_population",
        }


def aggregate_bonds(
    inputs: ResearchInputs,
    registration: ConsumptionRegistration,
    rows: tuple[ResearchRow, ...],
    consumer: Consumer,
    *,
    retained_rows: tuple[ResearchRow, ...] | None = None,
) -> dict[str, Any]:
    """逐债券执行原处理器，汇总全部保留行及结果，仅输出每状态/来源角色一个代表引用。

    输入P06市场cb_basic的1200行，输出total_rows=1200及最多10个可回查引用；
    不改变逐股或低基数路径。单债券超过512行或配置限额时明确失败，绝不截断输入。
    retained_rows包含执行层拒绝行，rows仅含通过执行门禁的行；例：9行中1行未来，
    total_rows=9、usable_rows=8、rejected_rows=1，拒绝行仍进入全量摘要。
    """
    api = registration.spec.api
    if (
        api not in RAW_DIMENSIONS
        or registration.spec.product != "P06"
        or registration.entity_scope != "bond"
    ):
        raise ValueError("research_bond_aggregation_binding_invalid")
    all_rows = rows if retained_rows is None else retained_rows
    if len(all_rows) > inputs.config.maximum_source_rows:
        raise ValueError("research_bond_source_capacity_exceeded")
    fields = tuple(
        sorted(
            set(
                registration.spec.selected_fields
                + RAW_DIMENSIONS[api]
                + NUMERIC_FIELDS[api]
                + AUDIT_FIELDS
            )
        )
    )
    grouped: dict[tuple[str, str], list[ResearchRow]] = defaultdict(list)
    for row in all_rows:
        group = grouped[_group_key(row)]
        group.append(row)
        if len(group) > min(MAXIMUM_GROUP_ROWS, inputs.config.maximum_rows_per_entity_source):
            raise ValueError("research_bond_group_capacity_exceeded")
    usable_groups: dict[tuple[str, str], list[ResearchRow]] = defaultdict(list)
    for row in rows:
        usable_groups[_group_key(row)].append(row)
    statistics = _Statistics(api, fields)
    source_periods: dict[str, str] = {}
    with (
        ContextAggregate(
            api, all_rows, rows, registration.spec.selected_fields, inputs.analysis_as_of
        ) as aggregate,
        closing(DiskKeySet()) as holders,
    ):
        for key, group in sorted(grouped.items()):
            ordered = tuple(sorted(group, key=lambda row: row_fingerprint(row, fields)))
            usable_ids = {id(row) for row in usable_groups.get(key, ())}
            for row in ordered:
                statistics.add(row, holders, eligible=id(row) in usable_ids)
            usable = tuple(row for row in ordered if id(row) in usable_ids)
            outcome = consumer(inputs, "P06", api, usable, MARKET_ENTITY)
            relations = _underlying_relations(inputs, usable[0]) if usable else ()
            targets = {relation.to_entity_id for relation in relations}
            binding = "confirmed" if len(targets) == 1 else "unknown" if not targets else "conflict"
            relation_digest = MultisetDigest()
            for relation in relations:
                relation_digest.add(fingerprint_leaf(relation.to_dict()))
            outcome["facts"] = outcome["facts"] | {
                "binding_status": binding,
                "binding_fingerprint": relation_digest.hexdigest(),
            }
            outcome["evidence_refs"] = tuple(outcome["evidence_refs"]) + tuple(
                ref for relation in relations for ref in relation.evidence_refs
            )
            for reference in outcome["evidence_refs"]:
                period = inputs.record_periods.get(
                    (reference.source_api, reference.source_record_id)
                )
                if period is not None:
                    source_periods[reference.source_api] = max(
                        period, source_periods.get(reference.source_api, "")
                    )
            statistics.group(outcome)
            aggregate.add(key, len(ordered), outcome)
        # ContextAggregate保留每层最小4个；取其中最小者等价于全量每层直接选1个。
        for stratum, samples in aggregate.samples.items():
            selected = min(samples)
            aggregate.samples[stratum] = {selected: samples[selected]}
        result = aggregate.finish()
        summary = result["facts"]["aggregation"]
        summary.update(
            version=BOND_AGGREGATION_VERSION,
            business_statistics=statistics.summary(),
            rows_fingerprint=statistics.rows.hexdigest(),
            group_reason_codes=sorted(aggregate.reasons),
            audited_fields=fields,
            source_as_of=dict(sorted(source_periods.items())),
        )
        summary["sampling_policy"].update(per_stratum_limit=1, maximum_samples=10)
        summary["sample_strata"] = [
            {"status": status, "source_role": role, "reference_fingerprint": next(iter(sample))[0]}
            for (status, role), sample in sorted(aggregate.samples.items())
        ]
        summary["full_fingerprint"] = bond_full_fingerprint(api, inputs.analysis_as_of, summary)
        summary["audit_fingerprint"] = bond_audit_fingerprint(summary)
        if registration.handler == "core":
            # 保持原core公开字段可观察即observed的口径，不因单位或其他债券缺字段伪装unknown。
            current = max((row.availability.data_as_of or "" for row in rows), default="")
            observed = any(
                (row.availability.data_as_of or "") == current
                and any(
                    row.raw.get(field) is not None for field in registration.spec.selected_fields
                )
                for row in rows
            )
            result["status"] = "observed" if observed else "unknown"
            result["reason_codes"] = () if observed else ("current_fields_unavailable",)
        result["facts"].update(
            independent_evidence=False,
            historical_pit_certified=False,
            calculation_scope="all_retained_bonds_grouped_current_observations",
            investment_signal="not_supported",
        )
        status = result["status"]
        result["gates"] = (
            gate(
                "bond_market_aggregate",
                Truth.TRUE
                if status == "observed"
                else Truth.FALSE
                if status == "conflict"
                else Truth.UNKNOWN,
                result["reason_codes"][0] if result["reason_codes"] else "bond_market_unavailable",
                hard=False,
                references=result["evidence_refs"],
            ),
        )
        return result
