"""离线验证来源保真、质量契约和防前视边界，全部使用人工构造数据。"""

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from tushare_entitlement_hub.models import Policy
from tushare_entitlement_hub.pit import adjust_prices, as_of, is_financial, time_boundary
from tushare_entitlement_hub.quality import _number, _pair, _reference_pairs, audit_rows
from tushare_entitlement_hub.standard import (
    _convert_unit,
    _effective_time,
    normalize_date,
    normalize_null,
    normalize_period,
    normalize_rows,
    normalize_timestamp,
)

OBSERVED = "2024-07-01T08:00:00Z"
MARKET_ROW = {
    "ts_code": "A.SZ",
    "trade_date": "2024-01-02",
    "open": 10.0,
    "high": 12.0,
    "low": 9.0,
    "close": 11.0,
    "vol": 2.0,
    "amount": 100.0,
}
FINANCIAL_ROW = {
    "ts_code": "A.SZ",
    "source_api": "income",
    "end_date": "2024-03-31",
    "report_type": "1",
    "ann_date": "2024-04-30",
    "observed_at": "2024-05-01T00:00:00Z",
    "revenue": 100.0,
    "update_flag": "0",
}


def issue_codes(rows: list[dict[str, Any]], policy: Policy, **kwargs: Any) -> set[str]:
    """提取审计错误码便于验证业务规则，保留实际审计路径。

    输入为人工记录、政策及参考参数，输出错误码集合；例如空行列表包含 empty_response。
    """
    return {issue["code"] for issue in audit_rows(rows, policy, **kwargs)}


@pytest.fixture
def market_policy() -> Policy:
    """提供明确的股票日行情契约，避免测试依赖在线政策。

    无输入；输出含证券日期主键、完整 OHLC 和成交量额字段的 daily 政策。
    """
    return Policy(
        "daily",
        category="market",
        primary_key=["ts_code", "trade_date"],
        date_field="trade_date",
        required_fields=list(MARKET_ROW),
        units={"vol": "手"},
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " NULL ",
        "nan",
        "N/A",
        "--",
        float("nan"),
        float("inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_null_markers(value: Any) -> None:
    """来源空值必须归一而不流入数值计算；输入 NULL、NaN 等均预期输出 None。"""
    assert normalize_null(value) is None


@pytest.mark.parametrize("value,expected", [(0, 0), (False, False), (" x ", "x"), ([1], [1])])
def test_nonnull_values_survive(value: Any, expected: Any) -> None:
    """合法零值和对象不可当作空值；例如输入 0 预期仍输出 0。"""
    assert normalize_null(value) == expected


@pytest.mark.parametrize(
    "value",
    ["20240102", 20240102, "2024/01/02", "2024-01-02", date(2024, 1, 2), datetime(2024, 1, 2, 9)],
)
def test_daily_date_forms(value: Any) -> None:
    """不同日日期格式必须一致；例如输入 20240102 预期输出 2024-01-02。"""
    assert normalize_date(value) == "2024-01-02"


@pytest.mark.parametrize("value", [True, [], "20240230", "2024", "2024-13-01"])
def test_invalid_daily_dates_fail_closed(value: Any) -> None:
    """拒绝不存在的日期与模糊粒度；例如 20240230 预期抛出固定 invalid_date。"""
    with pytest.raises(ValueError, match="^invalid_date$"):
        normalize_date(value)


@pytest.mark.parametrize(
    "value,field,expected",
    [
        ("202401", "month", "2024-01"),
        ("2024-01", "period", "2024-01"),
        ("2024Q1", "quarter", "2024-Q1"),
        ("2024-Q2", "period", "2024-Q2"),
        (2024, "year", "2024"),
        ("2024", "period", "2024"),
        ("20240102", "period", "2024-01-02"),
        (date(2024, 1, 2), "period", "2024-01-02"),
    ],
)
def test_reporting_period_granularity(value: Any, field: str, expected: str) -> None:
    """报告期间保持原粒度；例如季度 2024Q1 预期成为 2024-Q1 而不是某天。"""
    assert normalize_period(value, field) == expected


@pytest.mark.parametrize(
    "value,field",
    [
        (True, "period"),
        ([], "period"),
        ("2024", "month"),
        ("0000", "year"),
        ("2024Q5", "quarter"),
        ("2024Q1", "year"),
        ("abcdQ1", "quarter"),
        ("202401", "quarter"),
        ("202413", "month"),
        ("x", "month"),
        ("20-2401", "period"),
        ("bad", "period"),
    ],
)
def test_invalid_reporting_periods(value: Any, field: str) -> None:
    """非法粒度和期间不得被补成日期；例如季度 Q5 预期抛出 invalid_period。"""
    with pytest.raises(ValueError, match="^invalid_period$"):
        normalize_period(value, field)


@pytest.mark.parametrize("value", ["2024-01-02T08:00:00+08:00", datetime(2024, 1, 2, tzinfo=UTC)])
def test_timestamps_are_utc(value: Any) -> None:
    """观察时间统一 UTC；输入东八区早八点预期输出 UTC 零点。"""
    assert normalize_timestamp(value) == "2024-01-02T00:00:00Z"


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-02",
        "bad",
        datetime(2024, 1, 2),
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_ambiguous_or_overflow_timestamps(value: Any) -> None:
    """缺时区或越界时间不可推测；例如无时区日期预期抛出固定时间错误码。"""
    with pytest.raises(ValueError, match="^invalid_timestamp_or_timezone$"):
        normalize_timestamp(value)


@pytest.mark.parametrize(
    "value,unit,api,expected",
    [
        ("2", "万元", "daily", (20000.0, "元")),
        (2, "手", "daily", (200.0, "股")),
        (2, "手", "fut_daily", (2, "手")),
        (None, "万元", "daily", (None, "元")),
        (3, "%", "daily", (0.03, "比例")),
        ("opaque", "未知单位", "daily", ("opaque", "未知单位")),
    ],
)
def test_unit_conversions_are_explicit(value: Any, unit: str, api: str, expected: Any) -> None:
    """只换算明确来源单位；例如期货 2 手预期仍为 2 手，不冒充 200 股。"""
    assert _convert_unit(value, unit, Policy(api)) == expected


@pytest.mark.parametrize("value", [True, "not-a-number", "1e10000", "NaN"])
def test_invalid_numeric_units_are_rejected(value: Any) -> None:
    """单位换算不可把异常数值替换成零；例如文本数值错误预期固定 invalid_unit_value。"""
    with pytest.raises(ValueError, match="^invalid_unit_value$"):
        _convert_unit(value, "元", Policy("daily"))


def test_normalization_keeps_provenance_and_financial_concepts() -> None:
    """标准化必须保留原始证据和财务口径；输入公告、实际披露与报告期预期独立保留。"""
    source = {
        "ts_code": " a.sz ",
        "end_date": "20240331",
        "ann_date": "20240430",
        "f_ann_date": "20240501",
        "report_type": "1",
        "update_flag": "0",
        "revenue": "2",
        "detail": ["original"],
    }
    policy = Policy(
        "income",
        category="financial",
        primary_key=["ts_code", "end_date"],
        date_field="ann_date",
        units={"revenue": "万元"},
    )
    result = normalize_rows([source], policy, OBSERVED, "response-digest")[0]
    assert result["source_fields"] == source
    assert result["ts_code"] == "A.SZ"
    assert result["end_date"] == "2024-03-31"
    assert result["ann_date"] == "2024-04-30"
    assert result["f_ann_date"] == "2024-05-01"
    assert result["report_type"] == "1" and result["update_flag"] == "0"
    assert result["effective_at"] is None
    assert result["revenue"] == 20000.0
    assert result["source_units"] == {"revenue": "万元"}
    assert result["standard_units"] == {"revenue": "元"}
    assert result["conversion_version"] and result["response_hash"] == "response-digest"
    source["detail"].append("later")
    assert result["source_fields"]["detail"] == ["original"]


@pytest.mark.parametrize(
    "field,value,expected,granularity",
    [
        ("month", "202401", "2024-01", "month"),
        ("quarter", "2024Q1", "2024-Q1", "quarter"),
        ("year", "2024", "2024", "year"),
        ("period", "20240102", "2024-01-02", "day"),
    ],
)
def test_macro_periods_are_observed_snapshots(
    field: str,
    value: str,
    expected: str,
    granularity: str,
) -> None:
    """宏观期间不能冒充发布日期；输入月度快照预期仅在观察后可见且公告为空。"""
    policy = Policy(
        "macro_demo",
        category="macro",
        primary_key=[field],
        date_field=field,
        collection_mode="snapshot",
        pit_fields=[field],
    )
    row = normalize_rows([{field: value}], policy, OBSERVED, "digest")[0]
    assert row[field] == row["period"] == expected
    assert row["period_granularity"] == granularity
    assert row["period_semantics"] == "reporting_period"
    assert "ann_date" not in row and row["effective_at"] is None
    assert as_of([row], "2024-06-30", [field]) == []
    assert as_of([row], OBSERVED, [field]) == [row]
    assert "invalid_date" not in issue_codes([row], policy)


@pytest.mark.parametrize(
    "row,policy,code",
    [
        ({"id": "x"}, Policy("x"), "primary_key_required"),
        ({}, Policy("x", primary_key=["id"]), "primary_key_null"),
        ({"id": []}, Policy("x", primary_key=["id"]), "primary_key_not_scalar"),
    ],
)
def test_normalization_requires_business_identity(
    row: dict[str, Any], policy: Policy, code: str
) -> None:
    """无法识别业务行时拒绝落库；例如缺 id 预期返回 primary_key_null。"""
    with pytest.raises(ValueError, match=f"^{code}$"):
        normalize_rows([row], policy, OBSERVED, "digest")


def test_response_hash_and_null_fields() -> None:
    """来源哈希必须存在且空字段不伪造日期；输入空公告预期保留 None。"""
    policy = Policy("demo", primary_key=["id"])
    with pytest.raises(ValueError, match="response_hash_required"):
        normalize_rows([], policy, OBSERVED, "")
    row = normalize_rows([{"id": "x", "ann_date": "", "amount": None}], policy, OBSERVED, "digest")[
        0
    ]
    assert row["ann_date"] is None and row["amount"] is None
    assert normalize_rows([], policy, OBSERVED, "digest") == []


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("effective_at", "20240102", "2024-01-02"),
        ("effective_time", "2024-01-02T08:00:00+08:00", "2024-01-02T00:00:00Z"),
        ("effective_date", "20240102", "2024-01-02"),
        ("trade_date", "20240102", "2024-01-02"),
        ("trade_time", "2024-01-02T08:00:00+08:00", "2024-01-02T00:00:00Z"),
    ],
)
def test_effective_time_sources(field: str, value: str, expected: str) -> None:
    """明确生效来源按精度保留；例如带时区交易时间预期仍保留时分秒。"""
    policy = Policy("demo", primary_key=["id"], date_field=field)
    assert _effective_time({field: value}, policy) == expected
    if field != "effective_at":
        assert (
            normalize_rows([{"id": "x", field: value}], policy, OBSERVED, "digest")[0][
                "effective_at"
            ]
            == expected
        )
    assert _effective_time({}, policy) is None


def test_datetime_effective_boundary_never_becomes_midnight() -> None:
    """回归生效 datetime 被截断缺陷；输入晚六点生效事件预期中午不可见。"""
    policy = Policy("event_demo", primary_key=["id"])
    future = datetime(2024, 7, 1, 18, tzinfo=UTC)
    row = normalize_rows(
        [{"id": "x", "effective_at": future}], policy, "2024-07-01T08:00:00Z", "digest"
    )[0]
    assert row["effective_at"] == "2024-07-01T18:00:00Z"
    assert as_of([row], "2024-07-01T12:00:00Z", ["id"]) == []
    assert as_of([row], "2024-07-01T18:00:00Z", ["id"]) == [row]


def test_date_and_timestamp_boundaries() -> None:
    """日期与完整时间分别处理；输入日期可得日末，完整时间预期精确保留。"""
    assert time_boundary("2024-01-02") == datetime(2024, 1, 2, tzinfo=UTC)
    assert time_boundary("2024-01-02", end_of_day=True).hour == 23
    timestamp = datetime(2024, 1, 2, 12, tzinfo=UTC)
    assert time_boundary(timestamp) == timestamp
    assert time_boundary("2024-01-02T12:00:00Z") == timestamp
    assert is_financial({"dataset_category": "financial"})
    assert is_financial({"end_date": "20240331", "report_type": "1"})
    assert not is_financial({"trade_date": "20240102"})


def test_financial_revisions_follow_observation_time() -> None:
    """迟到修订不可提前进入回测；输入五月版和七月版，六月预期只见五月版。"""
    original = deepcopy(FINANCIAL_ROW)
    revised = {**original, "revenue": 200.0, "update_flag": "1", "observed_at": OBSERVED}
    rows = [original, revised]
    primary = ["ts_code", "end_date", "report_type", "ann_date", "update_flag"]
    result = as_of(rows, "2024-06-30", primary)
    assert result == [original]
    assert as_of(rows, "2024-07-02", primary) == [revised]
    result[0]["revenue"] = 999
    assert original["revenue"] == 100.0 and len(rows) == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"observed_at": None},
        {"observed_at": "2024-08-01T00:00:00Z"},
        {"pit_required_fields": "ann_date"},
        {"pit_required_fields": [3]},
        {"pit_required_fields": ["f_ann_date"]},
        {"ann_date": None},
        {"ann_date": "2024-08-01"},
        {"f_ann_date": "2024-08-01"},
        {"effective_date": "2024-08-01"},
        {"ann_date": "not-date"},
        {"out_date": "2024-06-01"},
        {"ts_code": None},
        {"ts_code": []},
    ],
)
def test_pit_rejects_unknown_or_future_requirements(updates: dict[str, Any]) -> None:
    """缺必要时间或未来信息必须排除；例如实际披露在八月，七月查询预期为空。"""
    row = {**FINANCIAL_ROW, **updates}
    assert as_of([row], "2024-07-02", ["ts_code", "end_date"]) == []


def test_pit_required_fields_snapshot_and_ambiguous_revisions() -> None:
    """快照只在观察后可见，冲突修订不猜选；同时间不同收入预期排除该业务键。"""
    unknown = {"id": "x", "observed_at": OBSERVED}
    assert as_of([unknown], "2024-07-02", ["id"]) == []
    snapshot = {**unknown, "source_api": "stock_basic"}
    assert as_of([snapshot], "2024-07-02", ["id"]) == [snapshot]
    required = {**FINANCIAL_ROW, "pit_required_fields": ["ann_date", "end_date", "observed_at"]}
    assert as_of([required], "2024-07-02", ["ts_code", "end_date"]) == [required]
    conflict = {**required, "revenue": 200.0}
    assert as_of([required, conflict], "2024-07-02", ["ts_code", "end_date"]) == []
    later = {**required, "observed_at": OBSERVED, "revenue": 300.0}
    assert as_of([required, conflict, later, later], "2024-07-02", ["ts_code", "end_date"]) == [
        later
    ]
    with pytest.raises(ValueError, match="business_primary_key_required"):
        as_of([], "2024-07-02", ["update_flag"])


@pytest.mark.parametrize("mode,anchor,expected", [("qfq", 4.0, [5.0]), ("hfq", 0.5, [40.0])])
def test_adjustment_anchor_is_explicit(mode: str, anchor: float, expected: list[float]) -> None:
    """前后复权使用指定锚点；输入价格10、因子2、前复权锚点4预期输出5。"""
    assert adjust_prices([10.0], [2.0], anchor, mode) == expected
    assert adjust_prices([], [], anchor, mode) == []


@pytest.mark.parametrize(
    "prices,factors,anchor,mode,code",
    [
        ([1.0], [1.0], 1.0, "bad", "unsupported_adjustment_mode"),
        ([1.0], [], 1.0, "qfq", "price_factor_length_mismatch"),
        ([1.0], [1.0], 0.0, "qfq", "invalid_anchor_factor"),
        ([1.0], [1.0], float("nan"), "qfq", "invalid_anchor_factor"),
        ([float("inf")], [1.0], 1.0, "qfq", "invalid_price"),
        ([True], [1.0], 1.0, "qfq", "invalid_price"),
        ([1.0], [-1.0], 1.0, "qfq", "invalid_adjustment_factor"),
        ([1.0], [float("nan")], 1.0, "qfq", "invalid_adjustment_factor"),
        ([1e308], [2.0], 1.0, "qfq", "nonfinite_adjusted_price"),
    ],
)
def test_invalid_adjustments_fail_without_raw_values(
    prices: list[float],
    factors: list[float],
    anchor: float,
    mode: str,
    code: str,
) -> None:
    """复权异常不能产生无限价格；例如零锚点预期固定 invalid_anchor_factor。"""
    with pytest.raises(ValueError, match=f"^{code}$"):
        adjust_prices(prices, factors, anchor, mode)


def test_numeric_and_reference_parsing() -> None:
    """参考数据必须能定位证券日期且值有限；输入复牌记录预期不成为停牌豁免。"""
    assert _number("2.5") == 2.5
    for value in (None, True, "bad", [], "inf"):
        assert _number(value) is None
    assert _pair({"ts_code": "a.sz", "trade_date": "20240102"}, "trade_date") == (
        "A.SZ",
        "2024-01-02",
    )
    assert _pair({}, "trade_date") is None
    assert _pair({"ts_code": "A.SZ", "trade_date": "bad"}, "trade_date") is None
    references = [
        {"ts_code": "A.SZ", "trade_date": "20240102", "suspend_type": "R"},
        {"ts_code": "B.SZ", "suspend_date": "20240102", "suspend_type": "S"},
        ("C.SZ", "20240102"),
        ("D.SZ", "bad"),
        {},
    ]
    assert _reference_pairs(references, suspension=True) == {
        ("B.SZ", "2024-01-02"),
        ("C.SZ", "2024-01-02"),
    }
    assert _reference_pairs(
        [
            {"ts_code": "A.SZ", "trade_date": "20240102", "adj_factor": 0},
            {"ts_code": "B.SZ", "trade_date": "20240102", "adj_factor": 2},
        ]
    ) == {("B.SZ", "2024-01-02")}


def test_contract_empty_null_duplicate_schema() -> None:
    """主键和字段契约独立审计；输入重复 id 与缺失值预期报告对应错误码。"""
    policy = Policy("demo", primary_key=["id"], required_fields=["value"])
    rows = [{"id": "a", "value": 1}, {"id": "a", "value": None}, {"id": None}, {"id": []}]
    codes = issue_codes(rows, policy, previous_fields=["id", "value", "removed"])
    assert {
        "duplicate_primary_key",
        "primary_key_null",
        "primary_key_not_scalar",
        "required_field_null",
        "required_field_missing",
        "inconsistent_schema",
        "schema_field_removed",
    } <= codes
    assert {"empty_response", "primary_key_required"} <= issue_codes([], Policy("demo"))
    findings = audit_rows(
        [{"id": "a", "added": 1}], Policy("demo", primary_key=["id"]), previous_fields=["id"]
    )
    assert any(
        item["code"] == "schema_field_added" and item["severity"] == "warning" for item in findings
    )
    assert "schema_drift_not_checked" in issue_codes([{"id": "a"}], policy)
    assert "schema_field_removed" not in issue_codes(
        [{"id": "a", "source_fields": "invalid"}],
        Policy("demo", primary_key=["id"]),
        previous_fields=["id"],
    )


def test_ohlc_and_volume_contracts(market_policy: Policy) -> None:
    """价格范围和量额符号必须满足契约；输入高价低于收盘预期报告价格关系错误。"""
    invalid = {**MARKET_ROW, "high": 10.0, "vol": -1, "amount": "bad", "trade_date": "bad"}
    assert {
        "invalid_ohlc_range",
        "negative_volume_or_amount",
        "invalid_numeric_value",
        "invalid_date",
    } <= issue_codes([invalid], market_policy)
    assert "incomplete_ohlc" in issue_codes([{**MARKET_ROW, "open": None}], market_policy)
    assert "invalid_ohlc_range" not in issue_codes([MARKET_ROW], market_policy)
    time_policy = Policy("intraday", primary_key=["id"], date_field="trade_time")
    assert "invalid_date" not in issue_codes(
        [{"id": "x", "trade_time": OBSERVED, "amount": None}], time_policy
    )


@pytest.mark.parametrize(
    "category,api",
    [
        ("fund", "fund_demo"),
        ("index", "index_daily"),
        ("futures", "fut_daily"),
        ("macro", "macro_demo"),
    ],
)
def test_close_only_is_not_incomplete_ohlc(category: str, api: str) -> None:
    """收盘价专用接口无需补齐全天价格；输入单 close 预期没有 incomplete_ohlc。"""
    policy = Policy(api, category=category, primary_key=["ts_code"], required_fields=["close"])
    codes = issue_codes([{"ts_code": "X", "close": 2}], policy, adj_factors=[])
    assert "incomplete_ohlc" not in codes
    if category != "fund":
        assert "missing_adj_factor" not in codes and "adj_factor_coverage_not_checked" not in codes


def test_static_basic_has_no_trading_calendar_requirement() -> None:
    """回归静态表误判行情；输入 stock_basic 无日期政策预期不要求交易日历。"""
    policy = Policy(
        "stock_basic", category="basic", primary_key=["ts_code"], collection_mode="snapshot"
    )
    row = normalize_rows([{"ts_code": "A.SZ"}], policy, OBSERVED, "digest")[0]
    codes = issue_codes([row], policy)
    assert "calendar_coverage_not_checked" not in codes
    assert as_of([row], "2024-06-30", ["ts_code"]) == []
    assert as_of([row], OBSERVED, ["ts_code"]) == [row]


def test_financial_quality_time_contracts() -> None:
    """财务披露与报告期不能互换；输入只有报告期预期缺少披露和观察时间。"""
    policy = Policy(
        "income", category="financial", primary_key=["ts_code"], pit_fields=["ann_date"]
    )
    assert {
        "missing_pit_field",
        "missing_disclosure_time",
        "missing_observation_time",
    } <= issue_codes([{"ts_code": "A.SZ", "end_date": "20240331"}], policy)
    assert "disclosure_after_observation" in issue_codes(
        [{**FINANCIAL_ROW, "observed_at": "2024-04-01T00:00:00Z"}], policy
    )
    assert "disclosure_before_period_end" in issue_codes(
        [{**FINANCIAL_ROW, "ann_date": "2024-01-01"}], policy
    )
    assert "invalid_pit_time" in issue_codes([{**FINANCIAL_ROW, "observed_at": "bad"}], policy)
    forecast = Policy("forecast", category="financial", primary_key=["ts_code"])
    assert "disclosure_before_period_end" not in issue_codes(
        [{**FINANCIAL_ROW, "ann_date": "2024-01-01"}], forecast
    )


def test_membership_periods_and_industry_scope() -> None:
    """成分有效区间按行业业务键隔离；不同 l3_code 同证券区间预期不互相冲突。"""
    policy = Policy(
        "index_member_all", category="index", primary_key=["l3_code", "ts_code", "in_date"]
    )
    base = {"ts_code": "A.SZ", "l3_code": "ONE", "in_date": "20240101", "out_date": "20241231"}
    second = {**base, "l3_code": "TWO"}
    assert "overlapping_membership_period" not in issue_codes([base, second], policy)
    assert "overlapping_membership_period" in issue_codes(
        [base, {**base, "in_date": "20240601", "out_date": None}], policy
    )
    assert "invalid_membership_period" in issue_codes([{**base, "out_date": "20230101"}], policy)
    assert "invalid_membership_period" in issue_codes([{**base, "in_date": None}], policy)
    assert "missing_membership_identity" in issue_codes(
        [{"l3_code": "ONE", "in_date": "20240101"}], policy
    )
    assert "overlapping_membership_period" not in issue_codes(
        [base, {**base, "in_date": "20241231", "out_date": None}], policy
    )


def test_precise_coverage_suspensions_and_factors(market_policy: Policy) -> None:
    """停牌只能解释对应证券当天缺口；输入 B 停牌不能豁免 A 缺行情。"""
    expected = {"A.SZ": ["20240102", "20240103"], "B.SZ": ["20240103"]}
    findings = audit_rows(
        [MARKET_ROW],
        market_policy,
        expected_dates=expected,
        suspensions=[{"ts_code": "B.SZ", "trade_date": "20240103"}],
        adj_factors=[{"ts_code": "A.SZ", "trade_date": "20240102", "adj_factor": 1}],
    )
    gaps = [finding for finding in findings if finding["code"] == "missing_trading_day"]
    assert [(gap["ts_code"], gap["date"]) for gap in gaps] == [("A.SZ", "2024-01-03")]
    assert "missing_adj_factor" not in {finding["code"] for finding in findings}
    assert "missing_adj_factor" in issue_codes([MARKET_ROW], market_policy, adj_factors=[])
    codes = issue_codes([MARKET_ROW], market_policy, expected_dates=["20240103"])
    assert {"missing_trading_day", "suspension_explanation_not_checked"} <= codes
    assert {"calendar_coverage_not_checked", "adj_factor_coverage_not_checked"} <= issue_codes(
        [MARKET_ROW], market_policy
    )
    assert "coverage_universe_missing" in issue_codes(
        [], market_policy, expected_dates=["20240102"]
    )
    assert "invalid_expected_dates" in issue_codes(
        [MARKET_ROW], market_policy, expected_dates=["bad"]
    )


def test_probe_security_is_not_production_coverage_universe(market_policy: Policy) -> None:
    """回归探测样本污染范围；输入仅 B 行情与样本 A 政策，预期不报告 A 缺失。"""
    market_policy.minimal_params = {"ts_code": "A.SZ"}
    row = {**MARKET_ROW, "ts_code": "B.SZ"}
    assert "missing_trading_day" not in issue_codes(
        [row],
        market_policy,
        expected_dates=["20240102"],
        suspensions=[],
        adj_factors=[{"ts_code": "B.SZ", "trade_date": "20240102", "adj_factor": 1}],
        previous_fields=list(row),
    )


def test_event_announcement_dates_remain_business_identity() -> None:
    """独立事件不能按财报修订合并；输入同证券两次公告，预期保留两条事件。

    例如五月与六月各一条公告，在七月查询应输出两条；仅公告构成财务键时则拒绝猜选。
    """
    first = {"ts_code": "A.SZ", "ann_date": "2024-05-01", "observed_at": OBSERVED}
    second = {**first, "ann_date": "2024-06-01"}
    assert as_of([first, second], "2024-07-02", ["ts_code", "ann_date"]) == [first, second]
    assert as_of([FINANCIAL_ROW], "2024-07-02", ["ann_date"]) == []
