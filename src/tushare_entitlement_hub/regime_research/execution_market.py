"""离线核验逐证券日市场证据；数据完整不代表订单实际可成交。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

from ..raw import canonical_json
from ..reference_contract import ReferenceContract
from ..standard import normalize_date

REQUIRED_APIS = ("stk_mins", "stk_auction_o")
PRICE_FIELDS = ("open", "high", "low", "close")
NUMERIC_FIELDS = (*PRICE_FIELDS, "vol", "amount")
ABSENCE_STATES = frozenset({"suspended", "not_listed", "non_trading_day"})


@dataclass(frozen=True)
class ExecutionDemand:
    """一个冻结证券日及应有分钟标签；空标签表示时间覆盖尚未明确。"""

    ts_code: str
    date: str
    expected_minutes: tuple[str, ...] = ()
    simulated_fill: bool = True
    session_evidence: bytes | None = None
    session_sha256: str = ""


@dataclass(frozen=True)
class ExecutionArtifact:
    """已保存原始响应及声明的哈希；构造本身不读取文件或取得权限。"""

    api_name: str
    ts_code: str
    date: str
    payload: bytes
    content_sha256: str
    contract_hash: str


def _decode(payload: bytes, expected_hash: str) -> dict[str, Any]:
    """验证完整字节并解析公开JSON对象；例如合法{}字节及其哈希返回空字典。

    哈希不符返回固定错误，不回显输入。
    """
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("content_hash_mismatch")
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError
        canonical_json(value)
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("invalid_public_json") from None
    return value


def _session_state(demand: ExecutionDemand) -> str:
    """验证调用方提供的状态证据；例如同证券日trading声明返回trading。

    证据须有完整字节哈希及非空source_reference；本函数不替提供方认证该引用。
    缺证返回unknown，日期、证券或状态冲突返回固定错误。
    """
    if demand.session_evidence is None:
        return "unknown"
    evidence = _decode(demand.session_evidence, demand.session_sha256)
    if (evidence.get("ts_code") != demand.ts_code or evidence.get("date") != demand.date
            or not isinstance(evidence.get("source_reference"), str)
            or not evidence["source_reference"].strip()):
        raise ValueError("session_evidence_mismatch")
    state = evidence.get("state")
    if not isinstance(state, str) or state not in {"trading", *ABSENCE_STATES}:
        raise ValueError("unknown_session_state")
    return str(state)


def _contract_errors(contract: ReferenceContract) -> list[str]:
    """核对执行所需显式口径；现有未知单位的竞价契约会返回unit_unknown。"""
    errors = []
    units = contract.policy.units
    if (any(units.get(field) not in {"元", "元/股"} for field in PRICE_FIELDS)
            or units.get("vol") != "股" or units.get("amount") != "元"):
        errors.append("unit_unknown")
    rules = contract.rules
    if rules.get("timezone") != "Asia/Shanghai":
        errors.append("timezone_unknown")
    expected = "bar_end_1min" if contract.policy.api_name == "stk_mins" else "opening_auction"
    if (rules.get("execution_time_semantics") != expected
            or not isinstance(rules.get("execution_semantics_evidence"), str)
            or not rules["execution_semantics_evidence"].strip()):
        errors.append("time_semantics_unknown")
    if contract.policy.page_size <= 0:
        errors.append("row_limit_unknown")
    try:
        start = datetime.strptime(rules["execution_window_start"], "%H:%M:%S").time()
        end = datetime.strptime(rules["execution_window_end"], "%H:%M:%S").time()
        if (start >= end or start.strftime("%H:%M:%S") != rules["execution_window_start"]
                or end.strftime("%H:%M:%S") != rules["execution_window_end"]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("execution_window_unknown")
    return errors


def _response_rows(
    artifact: ExecutionArtifact, contract: ReferenceContract,
) -> tuple[list[dict[str, Any]], dict[str, Any], datetime]:
    """核对RawStore信封及列式响应；输入两行items返回两份原始字段字典。

    请求、响应或合同指纹不一致、截断、缺字段和错误响应均拒绝；不调用网络。
    """
    if artifact.contract_hash != contract.fingerprint:
        raise ValueError("contract_hash_mismatch")
    envelope = _decode(artifact.payload, artifact.content_sha256)
    parameters, response = envelope.get("params"), envelope.get("response")
    if (envelope.get("api_name") != artifact.api_name or not isinstance(parameters, dict)
            or not isinstance(response, dict)):
        raise ValueError("invalid_envelope")
    request_hash = hashlib.sha256(canonical_json(
        {"api_name": artifact.api_name, "params": parameters}
    ).encode()).hexdigest()
    response_hash = hashlib.sha256(canonical_json(response).encode()).hexdigest()
    if (envelope.get("request_id") != request_hash
            or envelope.get("response_hash") != response_hash):
        raise ValueError("envelope_hash_mismatch")
    try:
        observed = datetime.fromisoformat(envelope["observed_at"].replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError
    except (KeyError, AttributeError, TypeError, ValueError):
        raise ValueError("observation_time_invalid") from None
    if type(response.get("code")) is not int or response["code"] != 0:
        raise ValueError("provider_response_failed")
    if parameters.get("ts_code") != artifact.ts_code:
        raise ValueError("request_scope_mismatch")
    if artifact.api_name == "stk_mins":
        if parameters.get("freq") != "1min":
            raise ValueError("frequency_mismatch")
        try:
            start = datetime.fromisoformat(parameters["start_date"])
            end = datetime.fromisoformat(parameters["end_date"])
            if (start.date().isoformat() != normalize_date(artifact.date)
                    or end.date() != start.date() or start > end):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("request_window_invalid") from None
    elif parameters.get("trade_date") != artifact.date:
        raise ValueError("request_scope_mismatch")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError("response_shape_invalid")
    fields, items = data.get("fields"), data.get("items")
    if (not isinstance(fields, list) or not all(isinstance(field, str) for field in fields)
            or len(set(fields)) != len(fields) or not isinstance(items, list)):
        raise ValueError("response_shape_invalid")
    if data.get("has_more") not in (None, False) or len(items) >= contract.policy.page_size:
        raise ValueError("response_truncated")
    time_field = "trade_time" if artifact.api_name == "stk_mins" else "trade_date"
    required = {"ts_code", time_field, *NUMERIC_FIELDS, *contract.policy.required_fields}
    if not required.issubset(fields):
        raise ValueError("required_fields_missing")
    if any(not isinstance(item, list) or len(item) != len(fields) for item in items):
        raise ValueError("row_shape_invalid")
    return [dict(zip(fields, item, strict=True)) for item in items], parameters, observed


def _row_key(
    row: dict[str, Any], artifact: ExecutionArtifact, parameters: dict[str, Any],
    observed: datetime,
) -> tuple[str, tuple[float, ...]]:
    """校验同证券日量价并返回时间键和值；09:31分钟生成09:31:00键。

    分钟按已声明的上海时间解析，拒绝分钟之外的秒数、范围外行和无效量价。
    """
    if row["ts_code"] != artifact.ts_code:
        raise ValueError("row_scope_mismatch")
    if artifact.api_name == "stk_mins":
        try:
            instant = datetime.fromisoformat(row["trade_time"])
            instant = (instant.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                       if instant.tzinfo is None else instant.astimezone(ZoneInfo("Asia/Shanghai")))
            local = instant.replace(tzinfo=None)
            start = datetime.fromisoformat(parameters["start_date"])
            end = datetime.fromisoformat(parameters["end_date"])
            if (instant.date().isoformat() != normalize_date(artifact.date)
                    or instant.second or instant.microsecond or not start <= local <= end
                    or instant > observed):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("row_time_invalid") from None
        key = instant.strftime("%H:%M:%S")
    else:
        if row["trade_date"] != artifact.date:
            raise ValueError("row_scope_mismatch")
        observed_date = observed.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        if normalize_date(artifact.date) > observed_date:
            raise ValueError("observation_before_session")
        key = artifact.date
    try:
        if any(isinstance(row[field], bool) for field in NUMERIC_FIELDS):
            raise ValueError
        values = tuple(float(row[field]) for field in NUMERIC_FIELDS)
        opening, high, low, close, volume, amount = values
        if (not all(isfinite(value) for value in values) or min(values[:4]) <= 0
                or min(volume, amount) < 0 or not low <= opening <= high
                or not low <= close <= high or (volume == 0) != (amount == 0)):
            raise ValueError
        if volume > 0 and not low * volume - .01 <= amount <= high * volume + .01:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("row_values_invalid") from None
    return key, values


def audit_execution_market(
    demands: list[ExecutionDemand], artifacts: list[ExecutionArtifact],
    contracts: dict[str, ReferenceContract],
) -> dict[str, Any]:
    """对完整需求输出逐证券日缺口，始终保留capacity_certified=False。

    输入冻结需求、RawStore格式字节及现有接口契约，输出分母、覆盖和错误。
    例：一条需求配空artifacts得到required_security_days=1、market_covered=0。
    状态证据与时间语义由调用方显式提供；离线通过不认证权限、PIT或真实成交。
    """
    keys = {(demand.ts_code, demand.date) for demand in demands}
    if len(keys) != len(demands):
        raise ValueError("duplicate_demand")
    for demand in demands:
        try:
            if normalize_date(demand.date).replace("-", "") != demand.date:
                raise ValueError
        except ValueError:
            raise ValueError("demand_date_invalid") from None
        try:
            for label in demand.expected_minutes:
                if datetime.strptime(label, "%H:%M:%S").strftime("%H:%M:%S") != label:
                    raise ValueError
        except ValueError:
            raise ValueError("minute_schedule_invalid") from None
        if len(set(demand.expected_minutes)) != len(demand.expected_minutes):
            raise ValueError("minute_schedule_duplicate")
    grouped: dict[tuple[str, str, str], list[ExecutionArtifact]] = defaultdict(list)
    unexpected = 0
    for artifact in artifacts:
        if ((artifact.ts_code, artifact.date) not in keys
                or artifact.api_name not in REQUIRED_APIS):
            unexpected += 1
        else:
            grouped[artifact.ts_code, artifact.date, artifact.api_name].append(artifact)
    results = []
    for demand in demands:
        errors: list[str] = []
        try:
            state = _session_state(demand)
        except ValueError as error:
            state = "unknown"
            errors.append(str(error))
        if state == "unknown":
            errors.append("session_state_unknown")
        if state in ABSENCE_STATES and demand.simulated_fill:
            errors.append("absence_conflicts_with_fill")
        rows_by_api: dict[str, int] = {}
        for api in REQUIRED_APIS:
            selected = grouped[demand.ts_code, demand.date, api]
            rows: dict[str, tuple[float, ...]] = {}
            contract = contracts.get(api)
            if not selected:
                if state not in ABSENCE_STATES:
                    errors.append(api + ":missing_data")
            elif contract is None or contract.policy.api_name != api:
                errors.append(api + ":contract_missing")
            else:
                contract_errors = _contract_errors(contract)
                errors.extend(api + ":" + error for error in contract_errors)
                for artifact in selected:
                    try:
                        source_rows, parameters, observed = _response_rows(artifact, contract)
                        for row in source_rows:
                            key, values = _row_key(row, artifact, parameters, observed)
                            if "execution_window_unknown" not in contract_errors:
                                start = contract.rules["execution_window_start"]
                                end = contract.rules["execution_window_end"]
                                if api == "stk_mins" and not start <= key <= end:
                                    raise ValueError("row_outside_execution_window")
                                closing = datetime.fromisoformat(
                                    normalize_date(artifact.date) + "T" + end
                                ).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                                if api == "stk_auction_o" and observed < closing:
                                    raise ValueError("observation_before_execution_window")
                            if key in rows and rows[key] != values:
                                raise ValueError("duplicate_conflict")
                            rows[key] = values
                    except ValueError as error:
                        errors.append(api + ":" + str(error))
                if state in ABSENCE_STATES and rows:
                    errors.append(api + ":absence_conflicts_with_rows")
                if state == "trading":
                    if api == "stk_mins":
                        if not demand.expected_minutes:
                            errors.append(api + ":minute_schedule_unknown")
                        elif set(rows) != set(demand.expected_minutes):
                            errors.append(api + ":minute_coverage_incomplete")
                    elif len(rows) != 1:
                        errors.append(api + ":auction_coverage_incomplete")
            rows_by_api[api] = len(rows)
        status = ("rejected" if errors else "absence_evidenced" if state in ABSENCE_STATES
                  else "market_data_verified")
        results.append({"ts_code": demand.ts_code, "date": demand.date, "status": status,
                        "session_state": state, "rows_by_api": rows_by_api,
                        "errors": sorted(set(errors))})
    covered = sum(item["status"] == "market_data_verified" for item in results)
    absences = sum(item["status"] == "absence_evidenced" for item in results)
    return {"required_security_days": len(demands), "market_covered": covered,
            "absence_evidenced": absences, "unresolved": len(demands) - covered - absences,
            "unexpected_artifacts": unexpected, "results": results,
            "passed": bool(demands) and covered + absences == len(demands) and unexpected == 0,
            "capacity_certified": False, "historical_pit_certified": False,
            "current_permission_verified": False}
