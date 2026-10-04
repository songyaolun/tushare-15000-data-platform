"""由实际产品引用核算消费，磁盘去重防止证据数量随实体规模占满内存。"""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any

from ..catalog import load_policies
from ..equity_research.consumption import validate_result_consumption
from ..equity_research.consumption_registry import load_consumption_registry
from ..equity_research.key_index import DiskKeyIndex
from ..equity_research.models import ResearchInputs, ResearchResult, ResearchRow
from ..product_foundation.contracts import Truth
from ..product_foundation.temporal import availability_gate
from ..reference_contract import load_reference_contracts
from ..standard import is_standard_numeric_field
from .registry import NEW_PRODUCTS, PRODUCTS, Registration, registrations

SOURCE_SERIES = frozenset(
    item.spec.api
    for item in load_consumption_registry()
    if item.entity_scope in {"market", "dictionary"}
)
NUMERIC_FIELDS = {
    api: frozenset(
        field["name"]
        for field in contract.documentation.get("output_fields", ())
        if is_standard_numeric_field(field["name"], field.get("type", ""), contract.policy)
    )
    for api, contract in load_reference_contracts(load_policies()).items()
}


def certified_fields(row: ResearchRow, item: Registration, cutoff: str) -> tuple[str, ...]:
    """列出可认证的实际使用字段，数值必须有 TypedFact 单位与来源认证。

    输入带合格 amount 数值及 ts_code 的行返回这两个字段；缺单位时仅可认证身份文本。
    无实体、未来或隔离行返回空元组，不能由引用存在倒推出数据合格。
    契约明确为数值的字符串也必须有事实证书，例如未认证的price_limit="0.1"被拒绝；
    数值语义表由共享标准层一次构造，逐行审计不重复加载契约或猜测文本含义。
    """
    if (
        (
            row.entity_id is None
            and not (
                item.spec.api in SOURCE_SERIES
                and all(row.raw.get(name) is not None for name in item.spec.business_key)
            )
        )
        or row.raw.get("source_api") != item.spec.api
        or row.raw.get("is_quarantined") is True
        or row.raw.get("quality_tier") not in {"verified", "usable_with_annotation"}
        or not availability_gate(row.availability, cutoff).passed
    ):
        return ()
    valid = []
    for name in item.spec.selected_fields:
        value = row.raw.get(name)
        if value is None:
            continue
        if type(value) in {int, float} or name in NUMERIC_FIELDS[item.spec.api]:
            fact = row.numeric.get(name)
            if fact is None or fact.value is None or fact.reason_codes:
                continue
        valid.append(name)
    return tuple(valid)


class ConsumptionAudit(DiskKeyIndex):
    """逐产品执行账本，原始数据与使用引用以 SQLite B-tree 精确关联。"""

    def __init__(self, products: tuple[str, ...], binding_audit: dict[str, Any]) -> None:
        """创建有界磁盘账本；例如选择 P05 只分配其七个接口，注册总量仍为 206。

        父类缓存固定 8MiB，临时文件只在当前进程内使用，退出时清理。
        """
        super().__init__()
        self.entries = tuple(item for item in registrations() if item.spec.product in products)
        self.binding_audit = binding_audit
        self.executed: set[str] = set()
        self.reasons: dict[str, Counter[str]] = {}
        self.consumption_reasons: dict[str, Counter[str]] = {}
        self.retained: Counter[str] = Counter()
        self.legacy_observed: Counter[str] = Counter()
        self.connection.execute(
            "CREATE TABLE eligible(api TEXT,id TEXT,field TEXT,"
            "PRIMARY KEY(api,id,field)) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TABLE used(api TEXT,id TEXT,field TEXT,PRIMARY KEY(api,id,field)) WITHOUT ROWID"
        )

    def inputs(self, inputs: ResearchInputs) -> None:
        """登记有界保留集的合格字段和所有原因，不能将 unreadable 来源当空数据成功。

        例如十行输入只有两行实际被产品引用，最终 observed_rows 只能是两行。
        """
        for item in self.entries:
            api = item.spec.api
            if api not in inputs.rows_by_api and api not in inputs.source_reasons:
                continue
            rows = inputs.rows(api)
            self.retained[api] = len(rows)
            reasons = self.reasons.setdefault(api, Counter())
            reasons.update(inputs.source_reasons.get(api, ()))
            for key, values in inputs.entity_source_reasons.items():
                if key.startswith(api + "/"):
                    reasons.update(values)
            for row in rows:
                fields = certified_fields(row, item, inputs.analysis_as_of)
                if not fields:
                    reasons["record_entity_time_or_unit_unresolved"] += 1
                self.connection.executemany(
                    "INSERT OR IGNORE INTO eligible VALUES (?,?,?)",
                    ((api, str(row.raw["source_record_id"]), name) for name in fields),
                )

    def result(self, product: str, result: ResearchResult) -> None:
        """仅在真实处理器返回后登记执行，并核验旧消费记录或登记新产品实际证据。

        例如主题结果引用 ths_daily.close，只有该字段在适配后合格才计入 observed。
        重复结果引用同一字段只在磁盘登记一次，不重复统计来源行。
        同时存在带本来源引用的未知比较时，保留 observed 并另计业务缺口；
        同一指标或门禁的重复接口引用、重复原因码各算一次，独立诊断分别计数。
        """
        self.executed.add(product)
        if result.envelope.product_id != PRODUCTS[product]:
            raise ValueError("all_products_result_product_mismatch")
        if product not in NEW_PRODUCTS:
            validate_result_consumption(result)
            for record in result.details["consumption"]:
                api = record["api"]
                self.legacy_observed[api] += record["status"] == "observed"
                self.consumption_reasons.setdefault(api, Counter()).update(
                    record["reason_codes"]
                )
        assigned = {item.spec.api for item in self.entries if item.spec.product == product}
        if product in NEW_PRODUCTS:
            gaps = chain(
                (
                    (item.evidence_refs, item.reason_codes)
                    for item in result.metrics
                    if item.status == "unknown"
                ),
                (
                    (item.evidence_refs, item.reason_codes)
                    for item in result.envelope.quality_gates
                    if item.result is Truth.UNKNOWN
                ),
            )
            for gap_references, reason_codes in gaps:
                referenced_apis = assigned.intersection(
                    ref.source_api for ref in gap_references
                )
                diagnostic_reasons = sorted(set(reason_codes))
                for api in sorted(referenced_apis):
                    self.consumption_reasons.setdefault(api, Counter()).update(diagnostic_reasons)
        references = (
            tuple(
                ref
                for metric in result.metrics
                if metric.status == "available"
                for ref in metric.evidence_refs
            )
            + tuple(
                ref
                for gate in result.envelope.quality_gates
                if gate.result is Truth.TRUE
                for ref in gate.evidence_refs
            )
            if product in NEW_PRODUCTS
            else result.envelope.evidence_refs
        )
        for ref in references:
            if ref.source_api in assigned:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO used VALUES (?,?,?)",
                    ((ref.source_api, ref.source_record_id, name) for name in ref.fields),
                )

    def summary(self) -> dict[str, Any]:
        """输出逐接口七种独立状态及原因；registered 或 executed 不保证 observed。

        输入全产品无数据执行，返回 registered=assigned=executed=206、observed=0。
        observed 与 unresolved 可同时为真，表示部分证据可用而仍存在口径缺口。
        例如一行支撑两个未知比较，业务诊断计两次，独立来源行仍只有一条。
        """
        sources = self.binding_audit.get("sources", {})
        observed = dict(
            self.connection.execute(
                "SELECT api,count(*) FROM (SELECT DISTINCT e.api,e.id FROM eligible e "
                "JOIN used u USING(api,id,field)) GROUP BY api"
            ).fetchall()
        )
        by_api = {}
        for item in self.entries:
            api = item.spec.api
            binding = sources.get(api, {})
            reasons = self.reasons.get(api, Counter()).copy()
            consumption_reasons = self.consumption_reasons.get(api, Counter())
            reasons.update(consumption_reasons)
            reasons.update(binding.get("reason_counts", {}))
            for gaps in binding.get("unit_gaps", {}).values():
                reasons.update(gaps)
            used_rows = int(observed.get(api, 0))
            is_observed = used_rows > 0
            if item.spec.product not in NEW_PRODUCTS:
                is_observed = is_observed and self.legacy_observed[api] > 0
            if not is_observed:
                reasons["no_certified_business_observation"] += 1
            if self.retained[api] > used_rows:
                reasons["retained_rows_without_certified_product_reference"] += (
                    self.retained[api] - used_rows
                )
            by_api[api] = {
                "product": item.spec.product,
                "handler": item.handler,
                "purpose": item.spec.purpose,
                "selected_fields": list(item.spec.selected_fields),
                "registered": True,
                "assigned": True,
                "executed": item.spec.product in self.executed,
                "readable": binding.get("readable", False) is True,
                "configured": binding.get("configured", False) is True,
                "observed": is_observed,
                "unresolved": bool(reasons),
                "retained_rows": self.retained[api],
                "observed_rows": used_rows if is_observed else 0,
                "reason_counts": dict(sorted(reasons.items())),
                "consumption_reason_counts": dict(sorted(consumption_reasons.items())),
            }
        states = (
            "registered",
            "assigned",
            "executed",
            "readable",
            "configured",
            "observed",
            "unresolved",
        )
        return {
            "total_registered": 206,
            **{state: sum(record[state] for record in by_api.values()) for state in states},
            "by_api": by_api,
        }
