"""只读参考库的逐接口、逐行适配；不访问清单外部路径及原始文件。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Any, cast

import duckdb

from ..pit import time_boundary
from ..reference_contract import ReferenceContract
from ..standard import normalize_date, normalize_period, normalize_timestamp
from ..store import digest
from .batch_io import batches
from .bindings import FoundationBindings, TimeMapping
from .config import DEFAULT_FACT_BATCH_SIZE, READ_BATCH_SIZE, validate_batch_size
from .contracts import QualityGateResult, Truth
from .entities import resolve_alias
from .model import Model
from .numeric import TypedFact, adapt_numeric
from .registry import ConsumerSpec
from .storage import ProductStore
from .storage_records import GateRecord
from .temporal import AvailabilityFact, availability_gate, public_time_boundary


def reference_rows(
    connection: duckdb.DuckDBPyConnection, api: str, *, batch_size: int = READ_BATCH_SIZE
) -> Iterator[dict[str, Any]]:
    """分批读取已验证接口名的 accepted 表，只解码已知 JSON 元数据。

    输入 daily，逐条产出来源记录；永不读取 quarantine 行内容。
    batch_size允许1到512，但每次实际读取最多128行；输入512也只fetch至多128行。
    非法大小在执行SQL之前拒绝，读取默认值与事实写入配置独立。
    """
    validate_batch_size(batch_size)
    read_size = min(batch_size, READ_BATCH_SIZE)
    cursor = connection.execute(f'SELECT to_json(source_row) FROM accepted."{api}" source_row')
    while batch := cursor.fetchmany(read_size):
        for values in batch:
            row = json.loads(values[0])
            for key in (
                "source_units",
                "standard_units",
                "source_occurrences",
                "issues",
                "field_quality",
            ):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
            yield row


def _time_value(value: Any, timezone: str | None, *, public: bool) -> str | None:
    """按显式语义解析可空时间；日粒度公开使用次日，生效使用市场日初。

    例如公开日期 2026-09-30 配 Asia/Shanghai → 2026-09-30T16:00:00Z。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return normalize_timestamp(value)
    text = str(value)
    if "T" in text or " " in text:
        return normalize_timestamp(text)
    if timezone is None:
        raise ValueError("market_timezone_required")
    if public:
        return public_time_boundary(text, timezone)
    from zoneinfo import ZoneInfo

    return normalize_timestamp(
        datetime.fromisoformat(normalize_date(value)).replace(tzinfo=ZoneInfo(timezone))
    )


def map_availability(
    row: dict[str, Any],
    contract: ReferenceContract,
    mapping: TimeMapping | None,
) -> AvailabilityFact:
    """通过明确映射区分公告、生效和报告期；没有映射时保持 observed_only。

    输入旧报告期及当前观察时间，输出当前可见事实，不回写历史观察时间。
    """
    data_field = mapping.data_field if mapping else contract.policy.date_field
    data_value = row.get(data_field) if data_field else None
    if isinstance(data_value, datetime):
        data_value = data_value.date()
    data_as_of = normalize_period(data_value) if isinstance(data_value, (str, int, date)) else None
    public_times = []
    effective_from = effective_to = report_period = None
    timezone = mapping.market_timezone if mapping else None
    if mapping:
        for field in mapping.public_fields:
            parsed = _time_value(row.get(field), timezone, public=True)
            if parsed is None:
                raise ValueError("public_time_missing")
            public_times.append(parsed)
        effective_from = _time_value(
            row.get(mapping.effective_from_field or ""), timezone, public=False
        )
        effective_to = _time_value(
            row.get(mapping.effective_to_field or ""), timezone, public=False
        )
        period_value = row.get(mapping.report_period_field or "")
        report_period = normalize_period(period_value) if period_value is not None else None
    if row.get("historical_pit_certified") is not False:
        raise ValueError("source_pit_state_unverified")
    return AvailabilityFact(
        source_api=cast(str, row.get("source_api")),
        source_record_id=cast(str, row.get("source_record_id")),
        certification_version=digest(
            [contract.fingerprint, mapping.to_dict() if mapping else None]
        ),
        observed_at=normalize_timestamp(row.get("observed_at")),
        data_as_of=data_as_of,
        public_available_at=max(public_times, key=time_boundary) if public_times else None,
        effective_from=effective_from,
        effective_to=effective_to,
        report_period=report_period,
        market_timezone=timezone,
        time_precision="timestamp" if public_times else "observed_only",
    )


def consume_rows(
    connection: duckdb.DuckDBPyConnection,
    store: ProductStore,
    entry: ConsumerSpec,
    contract: ReferenceContract,
    bindings: FoundationBindings,
    analysis_as_of: str,
    run_id: str,
    *,
    batch_size: int = DEFAULT_FACT_BATCH_SIZE,
    progress: Callable[[int], None] | None = None,
) -> tuple[set[str], str | None]:
    """将 accepted 行适配为时间事实及可认证数值，保留逐行失败门禁。

    输入 daily 合格行和 amount 证书，输出 typed_fact 与血缘；无实体映射只登记原因。
    返回接口原因集合和最近业务期间，隔离行从不进入此函数。业务时间未知时不解析
    实体；例如2020年旧行在2026年观察后按2020年生命周期归属，不按当前代码复用归属。
    每批集合式写入时间、数值和门禁，并通过 progress 报告本批行数；无实体映射不扫描。
    读取保持独立小批；例如512条来源各产生两个数值事实，数值缓冲分两次512条写入。
    写入异常在单行适配的异常处理之外传播，由调用者事务回滚，不转成行失败原因。
    """
    validate_batch_size(batch_size)
    time_mapping = next(
        (item for item in bindings.time_mappings if item.source_api == entry.api), None
    )
    entity_mapping = next(
        (item for item in bindings.entity_mappings if item.source_api == entry.api), None
    )
    certificates = {
        item.field: item for item in bindings.certifications if item.source_api == entry.api
    }
    numeric_fields = {
        item["name"]
        for item in contract.documentation["output_fields"]
        if item["type"].lower() in {"float", "double", "int"}
    }
    if entity_mapping is None:
        return {"entity_mapping_missing"}, None
    reasons: set[str] = set()
    dates = set()
    for source_batch in batches(reference_rows(connection, entry.api), batch_size):
        availability_facts: list[Model] = []
        typed_facts: list[Model] = []
        gates: list[Model] = []
        for row in source_batch:
            row_typed_facts: list[TypedFact] = []
            row_codes: set[str] = set()
            row_id = digest([run_id, entry.api, str(row.get("source_record_id"))])
            try:
                if (
                    row.get("contract_hash") != contract.fingerprint
                    or row.get("source_api") != entry.api
                ):
                    raise ValueError("source_contract_mismatch")
                availability = map_availability(row, contract, time_mapping)
                availability_facts.append(availability)
                if availability.data_as_of is not None:
                    dates.add(availability.data_as_of)
                gate = availability_gate(availability, analysis_as_of)
                row_codes.update(gate.reason_codes)
                if not gate.passed:
                    row_codes.add("row_unavailable")
                else:
                    business_time = availability.data_as_of or availability.effective_from
                    entity_id = (
                        resolve_alias(
                            bindings.aliases,
                            entity_mapping.namespace,
                            str(row.get(entity_mapping.key_field, "")),
                            analysis_as_of,
                            effective_at=business_time,
                            market_timezone=availability.market_timezone,
                        )
                        if entity_mapping and business_time is not None
                        else None
                    )
                    if entity_id is None:
                        row_codes.add("entity_unresolved")
                    for field in sorted(set(entry.selected_fields) & numeric_fields):
                        if field not in certificates:
                            row_codes.add("unit_unverified")
                        if entity_id is not None:
                            fact = adapt_numeric(
                                row, field, entity_id, availability, certificates.get(field)
                            )
                            row_typed_facts.append(fact)
                            row_codes.update(fact.reason_codes)
            except (ValueError, TypeError, OverflowError, KeyError):
                row_codes.add("row_adaptation_failed")
            for fact in row_typed_facts:
                typed_facts.append(fact)
                if len(typed_facts) == batch_size:
                    store.put_many("typed_fact", typed_facts, batch_size=batch_size)
                    typed_facts.clear()
            if row_codes:
                gates.append(
                    GateRecord(
                        result_id=row_id,
                        rule_version="v1",
                        gate=QualityGateResult(
                            gate_id="row_eligibility",
                            result=Truth.UNKNOWN,
                            reason_codes=tuple(sorted(row_codes)),
                        ),
                    ),
                )
            reasons.update(row_codes)
        store.put_many("availability_fact", availability_facts, batch_size=batch_size)
        store.put_many("typed_fact", typed_facts, batch_size=batch_size)
        store.put_many("quality_gate_result", gates, batch_size=batch_size)
        if progress is not None:
            progress(len(source_batch))
    return reasons, max(dates) if dates else None
