"""按明确字段契约和参考数据审计质量，保留未检查项而不伪造通过。"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any, cast

from .models import Policy
from .pit import is_financial, time_boundary
from .standard import (
    EQUITY_LOT_APIS,
    METADATA_FIELDS,
    PERIOD_FIELDS,
    normalize_date,
    normalize_null,
    normalize_period,
    normalize_timestamp,
)

Issue = dict[str, str]
ReferenceRows = Iterable[dict[str, Any] | tuple[str, str]]


def _issue(code: str, message: str, row: int | None = None, **context: str) -> Issue:
    """创建统一问题对象，仅包含规则和定位信息，不嵌入原始字段值。

    输入错误码、中文说明、可选行号及字符串上下文；输出字符串字典。
    例如 _issue('primary_key_null', '主键为空', 0) 包含 code、message、severity 与 row='0'。
    """
    result = {"code": code, "message": message, "severity": "error", **context}
    if row is not None:
        result["row"] = str(row)
    return result


def _number(value: Any) -> float | None:
    """读取有限数值，拒绝布尔值、空值和不能解析的文本。

    输入任意字段值，输出 float 或 None。例如 _number('2.5') 返回 2.5。
    """
    if normalize_null(value) is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if isfinite(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _pair(row: Mapping[str, Any], date_field: str) -> tuple[str, str] | None:
    """从参考记录提取证券日期组合，缺失字段不会退化为全市场豁免。

    输入记录和日期字段，输出统一代码及日期，无法解析时返回 None。
    例如 _pair({'ts_code': 'a.sz', 'trade_date': '20240102'}, 'trade_date') 返回
    ('A.SZ', '2024-01-02')。
    """
    code = normalize_null(row.get("ts_code"))
    date_value = normalize_null(row.get(date_field))
    if code is None or date_value is None:
        return None
    try:
        return str(code).upper(), normalize_date(date_value)
    except ValueError:
        return None


def _reference_pairs(
    references: ReferenceRows, *, suspension: bool = False
) -> set[tuple[str, str]]:
    """提取精确的参考证券日期，复权因子必须为正，复牌记录不能作为停牌解释。

    输入停牌记录、因子记录或显式的 (证券, 日期) 对；输出有效组合集合。
    例如 _reference_pairs([('A.SZ', '20240102')], suspension=True) 返回一个日期组合。
    """
    pairs: set[tuple[str, str]] = set()
    for reference in references:
        if isinstance(reference, tuple):
            try:
                pairs.add((reference[0].upper(), normalize_date(reference[1])))
            except (ValueError, IndexError, AttributeError):
                continue
            continue
        if suspension:
            status = str(reference.get("suspend_type", "S")).upper()
            if status not in {"S", "SUSPEND", "停牌"}:
                continue
        else:
            factor = _number(reference.get("adj_factor"))
            if factor is None or factor <= 0:
                continue
        pair = _pair(reference, "trade_date") or _pair(reference, "suspend_date")
        if pair is not None:
            pairs.add(pair)
    return pairs


def _audit_contract(
    rows: list[dict[str, Any]],
    policy: Policy,
    previous_fields: Iterable[str] | None,
) -> list[Issue]:
    """检查主键、逐行必需字段和来源 schema，分别报告缺失与新增字段。

    输入记录、政策及可选上次来源字段集合；输出问题列表，不修改数据。
    例如空主键值会返回 primary_key_null，同批相同主键返回 duplicate_primary_key。
    """
    issues: list[Issue] = []
    if not policy.primary_key:
        issues.append(_issue("primary_key_required", "政策尚未声明业务主键"))
    seen: set[str] = set()
    observed_fields: set[str] = set()
    schemas: set[frozenset[str]] = set()
    for index, row in enumerate(rows):
        source = row.get("source_fields", row)
        fields = set(source) if isinstance(source, dict) else set(row) - METADATA_FIELDS
        schemas.add(frozenset(fields))
        observed_fields.update(fields)
        missing_key = [
            field for field in policy.primary_key if normalize_null(row.get(field)) is None
        ]
        for field in missing_key:
            issues.append(_issue("primary_key_null", "业务主键字段为空或缺失", index, field=field))
        if policy.primary_key and not missing_key:
            key_values = [row[field] for field in policy.primary_key]
            if any(isinstance(value, (dict, list, set)) for value in key_values):
                issues.append(_issue("primary_key_not_scalar", "业务主键必须是标量", index))
            else:
                key = json.dumps(key_values, sort_keys=True, default=str)
                if key in seen:
                    issues.append(_issue("duplicate_primary_key", "同批记录的业务主键重复", index))
                seen.add(key)
        for field in policy.required_fields:
            if field not in row:
                issues.append(
                    _issue("required_field_missing", "缺少契约要求的字段", index, field=field)
                )
            elif normalize_null(row[field]) is None:
                issues.append(
                    _issue("required_field_null", "契约要求的字段为空", index, field=field)
                )
    if len(schemas) > 1:
        issues.append(_issue("inconsistent_schema", "同批来源记录的字段集合不一致"))
    if previous_fields is not None and rows:
        prior = set(previous_fields) - METADATA_FIELDS
        for field in sorted(prior - observed_fields):
            issues.append(_issue("schema_field_removed", "来源字段相较前一版本被移除", field=field))
        for field in sorted(observed_fields - prior):
            issues.append(
                _issue(
                    "schema_field_added",
                    "来源新增字段，需要核对契约",
                    field=field,
                    severity="warning",
                )
            )
    elif previous_fields is None:
        issues.append(
            _issue(
                "schema_drift_not_checked",
                "未提供前一版本字段，尚未检查 schema 漂移",
                severity="not_checked",
            )
        )
    return issues


def _audit_values(rows: list[dict[str, Any]], policy: Policy) -> list[Issue]:
    """验证日期、OHLC 价格关系与非负量额，数值异常明确报告而不替换。

    输入记录与政策；输出错误列表。例如 high 小于 low 返回 invalid_ohlc_range。
    """
    issues: list[Issue] = []
    for index, row in enumerate(rows):
        for field, value in row.items():
            if normalize_null(value) is None:
                continue
            if field.endswith("_date") or field == policy.date_field or field in PERIOD_FIELDS:
                try:
                    if field in PERIOD_FIELDS:
                        normalize_period(value, field)
                    elif field.endswith(("_time", "_at")):
                        normalize_timestamp(value)
                    else:
                        normalize_date(value)
                except ValueError:
                    issues.append(_issue("invalid_date", "日期或时间格式无效", index, field=field))
        ohlc_fields = {"open", "high", "low", "close"}
        contract_ohlc = ohlc_fields & set(policy.required_fields)
        if len(ohlc_fields & row.keys()) >= 2 or len(contract_ohlc) >= 2:
            values = {field: _number(row.get(field)) for field in ohlc_fields}
            if any(value is None for value in values.values()):
                issues.append(_issue("incomplete_ohlc", "OHLC 字段缺失或不是有限数值", index))
            else:
                opening, high, low, close = (
                    cast(float, values[field]) for field in ("open", "high", "low", "close")
                )
                if low > high or not low <= opening <= high or not low <= close <= high:
                    issues.append(_issue("invalid_ohlc_range", "OHLC 高低价关系不成立", index))
        for field in ("vol", "volume", "amount", "turnover", "trade_volume"):
            if field not in row or normalize_null(row[field]) is None:
                continue
            number = _number(row[field])
            if number is None:
                issues.append(
                    _issue("invalid_numeric_value", "量额字段不是有限数值", index, field=field)
                )
            elif number < 0:
                issues.append(
                    _issue("negative_volume_or_amount", "量额字段不能为负", index, field=field)
                )
    return issues


def _audit_times(rows: list[dict[str, Any]], policy: Policy) -> list[Issue]:
    """检查财务披露、观察时间和历史成分区间，报告期只参与时间顺序检查。

    输入记录与政策；输出问题列表。例如财务记录只有 end_date 会返回 missing_disclosure_time。
    同一指数同一成员的 [in_date, out_date) 区间重叠会返回 overlapping_membership_period。
    """
    issues: list[Issue] = []
    intervals: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for index, row in enumerate(rows):
        contextual = {**row, "source_api": policy.api_name, "dataset_category": policy.category}
        financial = is_financial(contextual)
        for field in policy.pit_fields:
            if normalize_null(row.get(field)) is None:
                issues.append(
                    _issue("missing_pit_field", "缺少政策要求的时点字段", index, field=field)
                )
        if financial:
            disclosures = [
                row[field]
                for field in ("ann_date", "f_ann_date")
                if normalize_null(row.get(field)) is not None
            ]
            if not disclosures:
                issues.append(
                    _issue("missing_disclosure_time", "财务记录缺少公告或实际披露日期", index)
                )
            observation = row.get("observed_at")
            if normalize_null(observation) is None:
                issues.append(_issue("missing_observation_time", "财务记录缺少系统观察时间", index))
            try:
                observed = (
                    None if observation is None else time_boundary(normalize_timestamp(observation))
                )
                for disclosure in disclosures:
                    published = time_boundary(disclosure)
                    if observed is not None and published > observed:
                        issues.append(
                            _issue("disclosure_after_observation", "披露日期晚于观察时间", index)
                        )
                    if normalize_null(row.get("end_date")) is not None:
                        if published < time_boundary(row["end_date"]):
                            if policy.api_name not in {"forecast"}:
                                issues.append(
                                    _issue(
                                        "disclosure_before_period_end",
                                        "财报披露日期早于报告期结束",
                                        index,
                                    )
                                )
            except ValueError:
                issues.append(_issue("invalid_pit_time", "财务披露或观察时间无效", index))
        if "in_date" in row or "out_date" in row:
            try:
                start = normalize_date(row.get("in_date"))
                end = (
                    None
                    if normalize_null(row.get("out_date")) is None
                    else normalize_date(row["out_date"])
                )
                if end is not None and end <= start:
                    issues.append(
                        _issue("invalid_membership_period", "成分失效日期必须晚于生效日期", index)
                    )
                    continue
                code = row.get("con_code", row.get("ts_code"))
                if code is None:
                    issues.append(
                        _issue("missing_membership_identity", "历史成分缺少成员代码", index)
                    )
                    continue
                identity_fields = [
                    field
                    for field in policy.primary_key
                    if not field.endswith(("_date", "_at", "_time"))
                    and field not in {"update_flag", "version", "revision"}
                ]
                membership_scope = [row.get(field) for field in identity_fields]
                membership_scope.extend(
                    row.get(field)
                    for field in ("index_code", "index", "l1_code", "l2_code", "l3_code")
                )
                identity = json.dumps([*membership_scope, code], default=str)
                for prior_start, prior_end in intervals[identity]:
                    if (end is None or prior_start < end) and (
                        prior_end is None or start < prior_end
                    ):
                        issues.append(
                            _issue("overlapping_membership_period", "同一成员的有效期重叠", index)
                        )
                        break
                intervals[identity].append((start, end))
            except ValueError:
                issues.append(
                    _issue("invalid_membership_period", "历史成分生效日期缺失或无效", index)
                )
    return issues


def _audit_coverage(
    rows: list[dict[str, Any]],
    policy: Policy,
    expected_dates: Iterable[str] | Mapping[str, Iterable[str]] | None,
    suspensions: ReferenceRows | None,
    adj_factors: ReferenceRows | None,
) -> list[Issue]:
    """按证券和日期核对覆盖，只允许对应证券当天的停牌记录解释缺口。

    输入行情、政策、交易日列表或证券到日期的映射，以及可选停牌和因子记录；输出问题。
    例如 A 缺少 2024-01-02 时，B 同日停牌仍会返回 missing_trading_day。
    未提供参考数据返回 severity='not_checked'，不把缺证据算作通过。
    复权因子仅适用股票和基金价格；期货、指数、宏观等不会被强制套用股票因子。
    """
    issues: list[Issue] = []
    date_field = policy.date_field or "trade_date"
    actual = {pair for row in rows if (pair := _pair(row, date_field)) is not None}
    is_market = policy.date_field == "trade_date" or any(
        {"open", "high", "low", "close"} & row.keys() for row in rows
    )
    if expected_dates is None:
        if is_market:
            issues.append(
                _issue(
                    "calendar_coverage_not_checked",
                    "未提供交易日历，尚未检查交易日覆盖",
                    severity="not_checked",
                )
            )
    else:
        expected: set[tuple[str, str]] = set()
        try:
            if isinstance(expected_dates, Mapping):
                expected = {
                    (code.upper(), normalize_date(day))
                    for code, days in expected_dates.items()
                    for day in days
                }
            else:
                days = {normalize_date(day) for day in expected_dates}
                codes = {code for code, _ in actual}
                if not codes and days:
                    issues.append(
                        _issue("coverage_universe_missing", "没有证券范围，无法核对预期交易日")
                    )
                expected = {(code, day) for code in codes for day in days}
        except (ValueError, TypeError, AttributeError):
            issues.append(_issue("invalid_expected_dates", "预期交易日或证券范围格式无效"))
        gaps = expected - actual
        suspended = set() if suspensions is None else _reference_pairs(suspensions, suspension=True)
        if gaps and suspensions is None:
            issues.append(
                _issue(
                    "suspension_explanation_not_checked",
                    "未提供停牌记录，缺口尚未解释",
                    severity="not_checked",
                )
            )
        for code, day in sorted(gaps - suspended):
            issues.append(
                _issue(
                    "missing_trading_day", "缺少应有行情且没有对应停牌证据", ts_code=code, date=day
                )
            )
    factor_assets = {"equity", "equities", "stock", "stocks", "fund", "funds", "etf"}
    factor_apis = EQUITY_LOT_APIS | {"fund_daily", "fund_weekly", "fund_monthly"}
    factor_applicable = (
        policy.api_name in factor_apis or policy.category.lower() in factor_assets
    ) and any("close" in row for row in rows)
    if adj_factors is None:
        if is_market and factor_applicable:
            issues.append(
                _issue(
                    "adj_factor_coverage_not_checked",
                    "未提供复权因子，尚未检查因子覆盖",
                    severity="not_checked",
                )
            )
    elif factor_applicable:
        covered = _reference_pairs(adj_factors)
        for code, day in sorted(actual - covered):
            issues.append(
                _issue("missing_adj_factor", "行情缺少对应的有限正复权因子", ts_code=code, date=day)
            )
    return issues


def audit_rows(
    rows: list[dict[str, Any]],
    policy: Policy,
    expected_dates: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    suspensions: ReferenceRows | None = None,
    adj_factors: ReferenceRows | None = None,
    previous_fields: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """执行主键、字段、价格量额、覆盖、财务 PIT、成分有效期和 schema 门禁。

    输入记录与采集政策；expected_dates 可为全批交易日或 {证券: 日期列表}，后者可检测
    完全没有返回的证券。停牌和因子参考使用记录列表或精确证券日期对；previous_fields
    使用上次来源字段集合。输出每个问题的 code、message、severity 及可选 row/field/date。
    severity='error' 表示缺陷，warning 表示新增字段，not_checked 表示参考数据缺失。
    例如两条相同业务主键会返回 duplicate_primary_key；正常 OHLC 在参考数据齐全时
    不产生 OHLC 或覆盖错误。空批次返回 empty_response，错误描述不带原始行或凭据。
    """
    issues = [] if rows else [_issue("empty_response", "来源返回空数据，不能据此认定完成覆盖")]
    issues.extend(_audit_contract(rows, policy, previous_fields))
    issues.extend(_audit_values(rows, policy))
    issues.extend(_audit_times(rows, policy))
    issues.extend(_audit_coverage(rows, policy, expected_dates, suspensions, adj_factors))
    return issues
