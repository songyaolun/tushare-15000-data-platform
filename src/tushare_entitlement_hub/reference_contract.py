"""与产品无关的来源字段契约、可空业务维度及标准类型转换。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
from math import isfinite
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .models import Policy
from .standard import (
    _convert_unit,
    normalize_date,
    normalize_null,
    normalize_period,
    normalize_timestamp,
)
from .store import digest

ReferenceRecord = tuple[dict[str, Any], str, int, str]


@dataclass
class ReferenceContract:
    """接口字段契约；例如 daily 契约同时保留官方字段说明和独立的业务主键。"""

    policy: Policy
    documentation: dict[str, Any]
    rules: dict[str, Any]

    @property
    def primary_key(self) -> list[str]:
        """返回声明的业务维度；例如 ths_member 使用代码对，不虚构暂无的纳入日期。"""
        return list(self.rules.get("primary_key", self.policy.primary_key))

    @cached_property
    def fingerprint(self) -> str:
        """返回不含凭据的契约摘要；相同政策、说明和规则产生相同摘要。"""
        return digest(asdict(self))

    @property
    def requested_fields(self) -> str:
        """请求全部官方输出字段；例如 cb_rate 显式请求默认隐藏的付息日期。"""
        return ",".join(field["name"] for field in self.documentation.get("output_fields", []))


def load_reference_contracts(policies: dict[str, Policy]) -> dict[str, ReferenceContract]:
    """结合显式政策、打包官方字段和例外规则，返回按接口名索引的契约。

    输入政策映射，输出契约映射；例如 daily 保留原单位及文档 URL。
    仅 SDK 组合方法允许无独立 HTTP 字段说明，其字段由底层响应提供。
    """
    directory = Path(__file__).parent / "data"
    documents = {
        item["api_name"]: item
        for item in json.loads((directory / "reference_schema.json").read_text())
    }
    documents["pro_bar"] = {**documents["daily"], "api_name": "pro_bar",
                            "schema_basis": "daily_fields_used_by_https_composition"}
    rules = json.loads((directory / "reference_rules.json").read_text())
    return {
        name: ReferenceContract(policy, documents.get(name, {}), rules.get(name, {}))
        for name, policy in policies.items()
    }


def canonical_fields(source: dict[str, Any], contract: ReferenceContract) -> dict[str, Any]:
    """修正可验证的字段大小写和显式别名，冲突时拒绝覆盖来源值。

    输入原始行与契约，输出新行；例如 cn_pmi 的 MONTH 变为 month。
    同时存在 MONTH 和 month 时抛字段冲突，原始响应仍独立保存。
    """
    result: dict[str, Any] = {}
    for name, value in source.items():
        canonical = contract.rules.get("aliases", {}).get(name.lower(), name.lower())
        if canonical in result:
            raise ValueError("source_field_collision")
        result[canonical] = value
    return result


def reference_value(value: Any, field: str, kind: str, contract: ReferenceContract) -> Any:
    """按已声明语义转换一个值，未知内容不补零、不推断日期。

    输入来源值、字段、官方类型和契约，返回标准值；例如月份202609为2026-09。
    文档没有非空承诺的业务数值允许空；单独的 '-' 作为提供方缺失占位保存为None。
    分钟时间仅在接口明确声明时区时补上时区，交易时间说明保持文本。
    """
    value = normalize_null(value)
    if value is None or value == "-":
        return None
    if field in contract.rules.get("date_list_fields", []):
        return [normalize_date(item.strip()) for item in str(value).split(",")]
    if field in contract.rules.get("text_fields", []):
        return str(value)
    if field in contract.rules.get("timestamp_fields", []):
        instant = datetime.fromisoformat(str(value))
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=ZoneInfo(contract.rules["timezone"]))
        return normalize_timestamp(instant)
    if field in {"month", "quarter", "period"}:
        return normalize_period(value, field)
    if field.endswith("_date") or field in {"date", "cal_date"}:
        candidate = str(value)
        if len(candidate) > 10 and candidate[10] in {"T", " "}:
            candidate = datetime.fromisoformat(candidate).date().isoformat()
        return normalize_date(candidate)
    if kind in {"float", "double", "int"}:
        if isinstance(value, bool):
            raise ValueError("invalid_numeric_value")
        number = float(value)
        if not isfinite(number) or (kind == "int" and not number.is_integer()):
            raise ValueError("invalid_numeric_value")
        return int(number) if kind == "int" else number
    if field == "ts_code" or field.endswith("_code"):
        return str(value).strip().upper()
    return value


def normalize_reference_row(
    source: dict[str, Any], contract: ReferenceContract, observed_at: str,
    response_hash: str, source_index: int, request_id: str = "",
) -> dict[str, Any]:
    """保留原始证据并生成有类型、单位、记录身份和可见性说明的标准行。

    输入来源行、契约、真实观察时间、原始摘要和页内位置，返回独立标准行。
    例如缺公告日的主营业务构成只能从 observed_at 之后使用，不能回填历史可见性。
    source_record_id 仅定位来源记录，不冒充业务主键；可空键仍保留实际 None。
    """
    fields = canonical_fields(source, contract)
    types = {item["name"]: item["type"].lower() for item in
             contract.documentation.get("output_fields", [])}
    row: dict[str, Any] = {}
    units = {}
    for field, original in fields.items():
        try:
            value = reference_value(original, field, types.get(field, ""), contract)
            if field in contract.policy.units:
                value, units[field] = _convert_unit(
                    value, contract.policy.units[field], contract.policy
                )
        except (ValueError, TypeError, OverflowError):
            raise ValueError("invalid_field:" + field) from None
        row[field] = value
    for field in contract.primary_key:
        if field not in row:
            raise ValueError("business_key_field_missing:" + field)
        if row[field] is None and field not in contract.rules.get("nullable_keys", []):
            raise ValueError("business_key_null:" + field)
        if isinstance(row[field], (dict, list)):
            raise ValueError("business_key_not_scalar")
    observation = normalize_timestamp(observed_at)
    disclosure = next((row.get(name) for name in ("f_ann_date", "ann_date", "notice_date")
                       if row.get(name)), None)
    row.update(
        source_fields=source,
        source_units=contract.policy.units,
        standard_units=units,
        source_api=contract.policy.api_name,
        observed_at=observation,
        response_hash=response_hash, source_request_id=request_id, source_index=source_index,
        source_record_id=digest(
            [contract.policy.api_name, observation, response_hash, request_id, source_index]
        ),
        business_key=digest([row.get(name) for name in contract.primary_key]),
        contract_hash=contract.fingerprint,
        availability_basis="observed_only" if not disclosure or contract.rules.get("snapshot")
        else "disclosed_and_observed",
        historical_pit_certified=False,
    )
    return row
