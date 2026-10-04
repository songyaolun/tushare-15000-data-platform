"""保留来源证据，并把日期、代码、空值和已声明单位转换为统一表示。"""

from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

from .models import Policy

CONVERSION_VERSION = "1"
METADATA_FIELDS = frozenset(
    {
        "source_fields",
        "source_units",
        "standard_units",
        "conversion_version",
        "response_hash",
        "observed_at",
        "effective_at",
        "source_api",
        "dataset_category",
        "pit_required_fields",
        "collection_mode",
        "availability_basis",
        "period_granularity",
        "period_semantics",
    }
)
FINANCIAL_APIS = frozenset(
    {
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "fina_audit",
        "forecast",
        "express",
        "income_vip",
        "balancesheet_vip",
        "cashflow_vip",
        "fina_indicator_vip",
    }
)
NULL_TEXT = frozenset({"", "none", "null", "nan", "n/a", "--"})
UNIT_CONVERSIONS: dict[str, tuple[str, Decimal]] = {
    "元": ("元", Decimal(1)),
    "千元": ("元", Decimal(1000)),
    "万元": ("元", Decimal(10000)),
    "亿元": ("元", Decimal(100000000)),
    "股": ("股", Decimal(1)),
    "万股": ("股", Decimal(10000)),
    "亿股": ("股", Decimal(100000000)),
    "张": ("张", Decimal(1)),
    "份": ("份", Decimal(1)),
    "万份": ("份", Decimal(10000)),
    "%": ("比例", Decimal("0.01")),
    "百分比": ("比例", Decimal("0.01")),
    "比例": ("比例", Decimal(1)),
    "元/股": ("元/股", Decimal(1)),
}
PERIOD_FIELDS = frozenset({"period", "month", "quarter", "year"})
SNAPSHOT_APIS = frozenset({"stock_basic", "fund_basic", "index_basic", "fut_basic", "opt_basic"})
EQUITY_LOT_APIS = frozenset({"daily", "weekly", "monthly", "stk_mins", "bak_daily"})


def normalize_null(value: Any) -> Any:
    """统一来源空值，但保留零、False 和非空对象。

    输入可以是来源字段的任意值；输出为原值或 None。字符串会去掉首尾空白。
    例如 normalize_null(' NULL ') 返回 None，normalize_null(0) 返回 0。
    """
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped.lower() in NULL_TEXT else stripped
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    return value


def normalize_date(value: Any) -> str:
    """把日粒度日期转换为 YYYY-MM-DD，不接受无效日期或猜测缺失月份。

    输入支持日期对象、YYYYMMDD 整数或字符串、ISO 日期和斜杠日期；输出为日期字符串。
    例如 normalize_date('20240102') 返回 '2024-01-02'；非法输入抛出脱敏 ValueError。
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid_date")
    candidate = str(value).strip().replace("/", "-")
    try:
        if len(candidate) == 8 and candidate.isdigit():
            return datetime.strptime(candidate, "%Y%m%d").date().isoformat()
        if len(candidate) != 10:
            raise ValueError("invalid_date")
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        raise ValueError("invalid_date") from None


def normalize_period(value: Any, field: str = "period") -> str:
    """按来源粒度规范报告期间，不把月、季或年伪造为某一天的披露时间。

    输入期间值和可选字段名 month/quarter/year/period，输出 YYYY-MM、YYYY-QN、YYYY
    或完整日日期；显式字段名与值粒度不一致时拒绝。例如 normalize_period('202401',
    'month') 返回 '2024-01'，normalize_period('2024Q1', 'quarter') 返回 '2024-Q1'。
    非法月份、季度或年份抛出不含来源值的 ValueError。
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, date)):
        raise ValueError("invalid_period")
    candidate = str(value).strip().upper().replace("/", "-")
    try:
        if len(candidate) == 4 and candidate.isdigit():
            if field not in {"year", "period"}:
                raise ValueError("invalid_period_granularity")
            date(int(candidate), 1, 1)
            return candidate
        compact_quarter = candidate.replace("-Q", "Q")
        if len(compact_quarter) == 6 and compact_quarter[4] == "Q":
            if field not in {"quarter", "period"} or compact_quarter[5] not in "1234":
                raise ValueError("invalid_period_granularity")
            date(int(compact_quarter[:4]), 1, 1)
            return compact_quarter[:4] + "-Q" + compact_quarter[5]
        compact_month = candidate.replace("-", "")
        month_shape = (len(candidate) == 6 and candidate.isdigit()) or (
            len(candidate) == 7 and candidate[4] == "-"
        )
        if month_shape and len(compact_month) == 6 and compact_month.isdigit():
            if field not in {"month", "period"}:
                raise ValueError("invalid_period_granularity")
            parsed = date(int(compact_month[:4]), int(compact_month[4:]), 1)
            return f"{parsed.year:04d}-{parsed.month:02d}"
        if field != "period":
            raise ValueError("invalid_period_granularity")
        return normalize_date(value)
    except (ValueError, TypeError):
        raise ValueError("invalid_period") from None


def normalize_timestamp(value: Any) -> str:
    """把带时区的真实时间转换为 UTC，拒绝缺少时区的模糊观察时间。

    输入为 datetime 或 ISO 8601 字符串，输出统一以 Z 结尾。
    例如 normalize_timestamp('2024-01-02T08:00:00+08:00') 返回 '2024-01-02T00:00:00Z'。
    """
    try:
        timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timezone_required")
        return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid_timestamp_or_timezone") from None


def _convert_unit(value: Any, unit: str, policy: Policy) -> tuple[Any, str]:
    """只转换政策明确声明的单位，未知单位按原样保留，非数值不能静默变成零。

    输入为字段值、来源单位和接口政策，输出为标准值与标准单位。
    例如 _convert_unit('2', '万元', Policy('daily')) 返回 (20000.0, '元')。
    手仅对明确的股票行情接口按每手 100 股转换，期货等接口保留手数。
    """
    conversion = UNIT_CONVERSIONS.get(unit)
    if unit == "手" and policy.api_name in EQUITY_LOT_APIS:
        conversion = ("股", Decimal(100))
    if conversion is None:
        return value, unit
    target_unit, multiplier = conversion
    if value is None:
        return None, target_unit
    try:
        if isinstance(value, bool):
            raise ValueError("invalid_unit_value")
        number = Decimal(str(value)) * multiplier
        result = float(number)
        if not number.is_finite() or not isfinite(result):
            raise ValueError("invalid_unit_value")
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        raise ValueError("invalid_unit_value") from None
    return result, target_unit


def is_standard_numeric_field(field: str, documented_kind: str, policy: Policy) -> bool:
    """按字段类型与已声明单位判断标准层数值语义，不以当前Python值类型猜测。

    例如fut_trade_param.price_limit文档为str但单位为%，返回True；无单位业务文本返回False。
    已声明数值类型的未知单位字段仍是数值语义，后续必须继续通过独立单位认证。
    """
    unit = policy.units.get(field)
    return documented_kind.lower() in {"int", "float", "double"} or unit in UNIT_CONVERSIONS or (
        unit == "手" and policy.api_name in EQUITY_LOT_APIS
    )


def restore_standard_numeric_text(
    row: dict[str, Any], documented_kinds: dict[str, str], policy: Policy,
) -> dict[str, Any]:
    """恢复旧参考库中已标准化却按VARCHAR保存的数值，保留原文本且不重复缩放。

    输入price_limit='0.1'、来源单位%及标准单位比例，输出price_limit=0.1；
    单位缺失、冲突、未知或数值无效时保留原文本，后续数值门禁继续失败关闭。
    """
    source_units, standard_units = row.get("source_units", {}), row.get("standard_units", {})
    if not isinstance(source_units, dict) or not isinstance(standard_units, dict):
        return row
    restored = dict(row)
    original_text = {}
    for field, value in row.items():
        declared = policy.units.get(field)
        if not isinstance(value, str) or not is_standard_numeric_field(
            field, documented_kinds.get(field, ""), policy
        ) or not isinstance(declared, str) or declared not in UNIT_CONVERSIONS:
            continue
        if source_units.get(field) != declared:
            continue
        _, expected = _convert_unit(1, declared, policy)
        if standard_units.get(field) != expected:
            continue
        try:
            converted, target = _convert_unit(value, expected, policy)
        except (ValueError, TypeError, OverflowError):
            continue
        if target == expected and type(converted) in {int, float}:
            restored[field] = converted
            original_text[field] = value
    if original_text:
        restored["source_numeric_text"] = original_text
    return restored


def _effective_time(row: dict[str, Any], policy: Policy) -> str | None:
    """查找实际生效日期，绝不把财务报告期当成生效日期。

    输入为已完成字段转换的记录和政策，输出明确的日期或时间，未知则为 None。
    例如记录 {'trade_date': '2024-01-02'} 配合 date_field='trade_date' 返回 '2024-01-02'。
    """
    for field in ("effective_at", "effective_time", "effective_date", "in_date", "start_date"):
        if row.get(field) is not None:
            if field in {"effective_at", "effective_time"}:
                value = row[field]
                if isinstance(value, datetime):
                    return normalize_timestamp(value)
                try:
                    return normalize_date(value)
                except ValueError:
                    return normalize_timestamp(value)
            return normalize_date(row[field])
    excluded = {"ann_date", "f_ann_date", "end_date", "report_date", "out_date"} | PERIOD_FIELDS
    if policy.date_field and policy.date_field not in excluded:
        value = row.get(policy.date_field)
        if value is not None:
            if policy.date_field.endswith(("_time", "_at")):
                return normalize_timestamp(value)
            return normalize_date(value)
    return None


def normalize_rows(
    rows: list[dict[str, Any]],
    policy: Policy,
    observed_at: str,
    response_hash: str,
) -> list[dict[str, Any]]:
    """标准化一批记录，同时保存原字段、来源单位和响应哈希供审计。

    输入 rows 必须具有政策声明的非空业务主键，observed_at 必须带时区；不修改输入。
    输出逐行增加 source_fields、source_units、standard_units、conversion_version、
    response_hash、source_api、dataset_category、pit_required_fields、observed_at、effective_at。
    所有 *_date 和日粒度 date_field 使用 YYYY-MM-DD；月、季、年频字段归一为报告期间。
    period_granularity 明确粒度，period_semantics 标记 reporting_period，不推导披露时间。
    快照标记 observed_snapshot，仅从观察时刻起可用；未知真实生效日期仍为 None。
    ann_date、f_ann_date、report_type、update_flag 独立保留，不补造未知披露日期。
    例如 [{'ts_code': ' a.sz ', 'trade_date': '20240102', 'vol': 2}] 在 daily 政策指定
    主键 ['ts_code', 'trade_date']、单位 {'vol': '手'} 时，输出代码 A.SZ、日期 2024-01-02、
    成交量 200.0 股，source_fields 中成交量仍为 2。非法主键或日期抛出固定错误码。
    """
    if not policy.primary_key:
        raise ValueError("primary_key_required")
    if not response_hash or not isinstance(response_hash, str):
        raise ValueError("response_hash_required")
    observation = normalize_timestamp(observed_at)
    normalized_rows: list[dict[str, Any]] = []
    for source in rows:
        row: dict[str, Any] = {}
        units: dict[str, str] = {}
        for field, original in source.items():
            value = normalize_null(original)
            if value is not None:
                if field in PERIOD_FIELDS:
                    value = normalize_period(value, field)
                elif field.endswith("_date") or field == policy.date_field:
                    if field.endswith(("_time", "_at")):
                        value = normalize_timestamp(value)
                    else:
                        value = normalize_date(value)
                elif field == "ts_code" or field.endswith("_code"):
                    value = str(value).strip().upper()
            if field in policy.units:
                value, units[field] = _convert_unit(value, policy.units[field], policy)
            row[field] = value
        for field in policy.primary_key:
            if row.get(field) is None:
                raise ValueError("primary_key_null")
            if isinstance(row[field], (dict, list, set)):
                raise ValueError("primary_key_not_scalar")
        period_source = policy.date_field if policy.date_field in PERIOD_FIELDS else ""
        if period_source and row.get(period_source) is not None and row.get("period") is None:
            row["period"] = row[period_source]
        if row.get("period") is not None:
            period_text = str(row["period"])
            row["period_granularity"] = (
                "quarter"
                if "Q" in period_text
                else "year"
                if len(period_text) == 4
                else "month"
                if len(period_text) == 7
                else "day"
            )
            row["period_semantics"] = "reporting_period"
        row["effective_at"] = _effective_time(row, policy)
        row["collection_mode"] = policy.collection_mode
        row["availability_basis"] = (
            "explicit_effective_time"
            if row["effective_at"] is not None
            else "observed_snapshot"
            if policy.collection_mode == "snapshot" or policy.api_name in SNAPSHOT_APIS
            else "unknown"
        )
        row["source_fields"] = deepcopy(source)
        row["source_units"] = dict(policy.units)
        row["standard_units"] = units
        row["conversion_version"] = CONVERSION_VERSION
        row["response_hash"] = response_hash
        row["source_api"] = policy.api_name
        row["dataset_category"] = policy.category
        row["pit_required_fields"] = list(policy.pit_fields)
        row["observed_at"] = observation
        normalized_rows.append(row)
    return normalized_rows
