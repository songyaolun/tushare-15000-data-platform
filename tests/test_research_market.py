"""市场环境离线合成回归，覆盖有效分母、时点、单位及月度相邻性。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.equity_research.market import build_market
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity, EntityKind
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-02T12:00:00Z"


def market_inputs(count=2, rows=(), *, complete=("suspend_d",)):
    """创建显式证券和合成停牌完整范围；例：complete=()可测试未认证的来源缺口。"""
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind=EntityKind.SECURITY, local_id=str(index), lifecycle_id="listed"
            ),
            name="合成证券",
        )
        for index in range(count)
    )
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="market-fixture",
        config=ResearchConfig(
            foundation=FoundationBindings(entities=entities), complete_scopes=complete
        ),
        rows_by_api={
            api: tuple(row for row in rows if row.source_api == api)
            for api in {row.source_api for row in rows}
        },
    )


def row(
    api="daily",
    index=0,
    record="a",
    fields=None,
    numeric=None,
    *,
    observed=NOW,
    public="2026-09-30T16:00:00Z",
    data="2026-09-30",
    currency="CNY",
):
    """创建有来源和单位证据的合成行；例：daily默认包含正成交和上涨1%。"""
    fields = (
        {"trade_date": "20260930", "close": 10.0, "vol": 100.0, "amount": 100.0, "pct_chg": 0.01}
        if fields is None
        else fields
    )
    raw = {
        **fields,
        "source_api": api,
        "source_record_id": record,
        "business_group_key": record,
        "contract_hash": "fixture-contract",
        "quality_tier": "verified",
    }
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="fixture-time",
        observed_at=observed,
        public_available_at=public,
        data_as_of=data,
    )
    item = ResearchRow(
        entity_id=f"cn:security:{index}:listed", source_api=api, raw=raw, availability=availability
    )
    facts = {}
    for name, value in (fields if numeric is None else numeric).items():
        if type(value) not in {int, float}:
            continue
        unit, dimension = (
            ("元", "price")
            if name in {"close", "down_limit"}
            else (
                ("元", "money")
                if name in {"amount", "inc_month"}
                else ("index", "index")
                if name == "pmi010000"
                else ("ratio", "ratio")
            )
        )
        refs = item.evidence(name)
        certificate = UnitCertification(
            version="fixture-unit",
            source_api=api,
            field=name,
            source_unit=unit,
            target_unit=unit,
            dimension=dimension,
            currency=currency if unit == "元" else None,
            multiplier=1,
            evidence_refs=refs,
        )
        facts[name] = TypedFact(
            fact_id=record + name,
            entity_id=item.entity_id,
            field=name,
            value=value,
            unit=unit,
            dimension=dimension,
            currency=certificate.currency,
            availability=availability,
            quality_tier="verified",
            certification=certificate,
            evidence_refs=refs,
        )
    return replace(item, numeric=facts)


def test_market_hand_calculation_and_evidence():
    """场景：两股一涨一跌且一股跌停；预期：广度0.5、集中0.75、E0为58.33。"""
    rows = (
        row(),
        row(
            index=1,
            record="b",
            fields={
                "trade_date": "20260930",
                "close": 9.0,
                "vol": 100.0,
                "amount": 300.0,
                "pct_chg": -0.1,
            },
        ),
        row("stk_limit", record="la", fields={"trade_date": "20260930", "down_limit": 9.0}),
        row(
            "stk_limit", index=1, record="lb", fields={"trade_date": "20260930", "down_limit": 9.0}
        ),
    )
    (result,) = build_market(market_inputs(rows=rows))
    assert result.metric("breadth") == 0.5
    assert result.metric("liquidity_concentration") == 0.75
    assert result.metric("down_limit_share") == 0.5
    assert result.metric("stress_score") == pytest.approx(100 * 1.75 / 3)
    # 场景：普通求和与精确求和的舍入次序不同；预期：仅容忍机器精度级误差。
    assert result.envelope.score == pytest.approx(
        result.metric("stress_score"), rel=1e-14, abs=1e-12
    )
    assert sum(item.contribution for item in result.envelope.contributions) == pytest.approx(
        result.envelope.score, rel=1e-14, abs=1e-12
    )
    assert result.details["denominator_count"] == 2
    assert not result.envelope.decision_allowed


def test_empty_market_degrades():
    """场景：空范围空数据；预期：仍返回一个未知市场结果，所有分数为空。"""
    (result,) = build_market(market_inputs(0))
    assert result.metric("stress_score") is None
    assert result.details["denominator_count"] == 0
    assert result.envelope.score is None
    assert all(state["status"] == "unknown" for state in result.details["macro_states"])


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"vol": 0.0}, "excluded_no_trade_count"),
        ({"market_activity_status": "no_trade"}, "excluded_no_trade_count"),
        ({"market_activity_status": "suspended", "amount": 0.0}, "excluded_suspended_count"),
        ({"market_activity_status": "suspended"}, "excluded_suspended_count"),
        ({"close": None}, "excluded_price_unknown_count"),
        ({"close": -1.0}, "excluded_price_unknown_count"),
        ({"price_type": "reference"}, "excluded_price_unknown_count"),
        ({"business_variant_key": "unresolved"}, "excluded_price_unknown_count"),
    ],
)
def test_invalid_market_observations_are_not_denominators(change, expected):
    """场景：无成交、未知价、参考价或变体；预期：不计广度分母并保留反证。"""
    source = row()
    invalid = replace(source, raw=source.raw | change)
    (result,) = build_market(market_inputs(rows=(invalid, row(index=1, record="b"))))
    assert result.details["denominator_count"] == 1
    assert result.details[expected] == 1
    assert result.envelope.contrary_evidence


def test_market_numeric_and_currency_certification_required():
    """场景：原值存在但数值未认证或非人民币；预期：排除，不偷偷读取raw补数。"""
    (result,) = build_market(
        market_inputs(rows=(row(numeric={}), row(index=1, record="b", currency="HKD")))
    )
    assert result.details["excluded_price_unknown_count"] == 2
    assert result.metric("breadth") is None


def test_market_latest_day_does_not_mix_stale_and_future():
    """场景：一股只有旧日数据，另有未来观察和未来交易日；预期：只计最新已知有效日。"""
    old = row(
        index=1,
        record="old",
        fields={
            "trade_date": "20260929",
            "close": 10.0,
            "vol": 100.0,
            "amount": 100.0,
            "pct_chg": -1.0,
        },
        data="2026-09-29",
    )
    future = row(index=1, record="future", observed="2026-10-03T00:00:00Z")
    future_day = row(
        record="future-day", fields=old.raw | {"trade_date": "20261005"}, data="2026-10-05"
    )
    (result,) = build_market(market_inputs(rows=(row(), old, future, future_day)))
    assert result.details["trade_date"] == "2026-09-30"
    assert result.envelope.data_as_of == "2026-09-30"
    assert result.details["excluded_stale_count"] == 1
    assert result.metric("breadth") == 1
    assert all(ref.source_record_id != "future" for ref in result.envelope.evidence_refs)


@pytest.mark.parametrize(
    "fields,data",
    [
        ({}, "2026-09-30"),
        ({"trade_date": "garbage"}, "2026-09-30"),
        ({"trade_date": "20260929"}, "2026-09-30"),
    ],
)
def test_market_invalid_date_cannot_create_sample(fields, data):
    """场景：日期缺失、非法或适配冲突；预期：不建立行情分母。"""
    (result,) = build_market(market_inputs(1, (row(fields=fields, data=data),)))
    assert result.details["denominator_count"] == 0


def test_same_day_ambiguity_is_not_arbitrarily_selected():
    """场景：同日同时公开两份来源行；预期：拒绝任意择一，保留未知价计数。"""
    (result,) = build_market(market_inputs(1, (row(), row(record="other"))))
    assert result.details["denominator_count"] == 0
    assert result.details["excluded_price_unknown_count"] == 1


@pytest.mark.parametrize("limit", [None, 0.0, -1.0])
def test_limit_missing_or_invalid_blocks_stress_score(limit):
    """场景：跌停价缺失或非正；预期：保留广度但不发布跌停占比与E0。"""
    rows = (
        row(),
        row("stk_limit", record="limit", fields={"trade_date": "20260930", "down_limit": limit}),
    )
    (result,) = build_market(market_inputs(1, rows))
    assert result.metric("breadth") == 1
    assert result.metric("down_limit_share") is None
    assert result.metric("stress_score") is None


def test_top_ten_percent_rounds_up_and_stress_topics():
    """场景：11股均跌停且等金额；预期：前10%向上取2股，触发流动性复核。"""
    rows = []
    for index in range(11):
        source = row(index=index, record=str(index))
        rows.append(
            replace(
                source,
                numeric=source.numeric
                | {"pct_chg": replace(source.numeric["pct_chg"], value=-0.1)},
            )
        )
        rows.append(
            row(
                "stk_limit",
                index=index,
                record="limit" + str(index),
                fields={"trade_date": "20260930", "down_limit": 10.0},
            )
        )
    (result,) = build_market(market_inputs(11, rows))
    assert result.metric("liquidity_concentration") == pytest.approx(2 / 11)
    assert result.details["research_topics"] == ["优先复核流动性"]


@pytest.mark.parametrize(
    "value,direction", [(51.0, "expansion"), (49.0, "contraction"), (50.0, "neutral")]
)
def test_pmi_current_level(value, direction):
    """场景：制造业PMI在50上下或等于50；预期：输出相应状态，宏观不参与E0。"""
    macro = row("cn_pmi", fields={"month": "202609", "pmi010000": value})
    (result,) = build_market(market_inputs(0, (macro,)))
    state = result.details["macro_states"][0]
    assert state["direction"] == direction
    assert state["mode"] == "observed_current_only"
    assert result.metric("stress_score") is None


def test_macro_adjacent_month_change_and_social_increment():
    """场景：PPI跨年相邻、M1/M2相邻及社融累计并存；预期：变化按相邻月且社融只用当月。"""
    rows = (
        row("cn_ppi", record="p0", fields={"month": "202512", "ppi_yoy": -0.02}),
        row("cn_ppi", record="p1", fields={"month": "202601", "ppi_yoy": -0.01}),
        row("cn_m", record="m0", fields={"month": "2026-08", "m2_yoy": 0.08, "m1_yoy": 0.02}),
        row("cn_m", record="m1", fields={"month": "202609", "m2_yoy": 0.10, "m1_yoy": 0.03}),
        row(
            "sf_month",
            record="s",
            fields={"month": "202609", "inc_month": 100.0, "inc_cumval": 1000.0},
        ),
    )
    (result,) = build_market(market_inputs(0, rows))
    states = {state["source_api"]: state for state in result.details["macro_states"]}
    assert states["cn_ppi"]["change"] == pytest.approx(0.01)
    assert states["cn_m"]["value"] == pytest.approx(0.07)
    assert states["cn_m"]["change"] == pytest.approx(0.01)
    assert states["sf_month"]["value"] == 100
    assert all(ref.fields != ("inc_cumval",) for ref in result.envelope.evidence_refs)


@pytest.mark.parametrize("prior", [None, "202607", "invalid", "202613", "202611"])
def test_macro_missing_adjacent_period_is_current_only(prior):
    """场景：没有相邻有效月份，包括空值、跨月、非法月和未来月；预期：只显示当前值。"""
    rows = (
        row("cn_ppi", fields={"month": "202609", "ppi_yoy": 0.01}),
        row("cn_ppi", record="old", fields={"month": prior, "ppi_yoy": 0.02}),
    )
    (result,) = build_market(market_inputs(0, rows))
    state = result.details["macro_states"][1]
    assert state["change"] is None
    assert state["status"] == "current_only"


@pytest.mark.parametrize(
    "kind",
    ["duplicate", "variant", "unverified", "previous_unverified", "previous_variant", "foreign"],
)
def test_macro_ambiguity_and_units_fail_closed(kind):
    """场景：宏观多版本、未知单位或社融外币；预期：歧义与单位缺口保持未知，不填零。"""
    api = "sf_month" if kind == "foreign" else "cn_ppi"
    field = "inc_month" if kind == "foreign" else "ppi_yoy"
    current = row(
        api, fields={"month": "202609", field: 1.0}, currency="HKD" if kind == "foreign" else "CNY"
    )
    rows = [current]
    if kind == "duplicate":
        rows.append(row(api, record="duplicate", fields={"month": "202609", field: 1.0}))
    if kind == "variant":
        rows[0] = replace(current, raw=current.raw | {"business_variant_key": "variant"})
    if kind == "unverified":
        rows[0] = replace(current, numeric={})
    if kind.startswith("previous"):
        previous = row(api, record="previous", fields={"month": "202608", field: 0.0})
        rows.append(
            replace(previous, numeric={})
            if kind == "previous_unverified"
            else replace(previous, raw=previous.raw | {"business_variant_key": "variant"})
        )
    (result,) = build_market(market_inputs(0, rows))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == api)
    assert state["change"] is None
    assert state["reason_codes"]


@pytest.mark.parametrize("kind", ["current", "change"])
def test_macro_overflow_is_not_serialized(kind):
    """场景：有限输入相减后溢出；预期：当前值或变化值保持未知，输出可序列化。"""
    rows = (row("cn_m", fields={"month": "202609", "m2_yoy": 1e308, "m1_yoy": -1e308}),)
    if kind == "change":
        rows = (
            row("cn_ppi", fields={"month": "202609", "ppi_yoy": 1e308}),
            row("cn_ppi", record="previous", fields={"month": "202608", "ppi_yoy": -1e308}),
        )
    (result,) = build_market(market_inputs(0, rows))
    state = result.details["macro_states"][2 if kind == "current" else 1]
    assert "nonfinite_macro_calculation" in state["reason_codes"]
    assert state["change"] is None


@pytest.mark.parametrize(
    "timing,scope",
    [(None, "full_day"), ("09:30-10:30", "intraday"), ("missing", "unknown"), ({}, "unknown")],
)
def test_same_day_suspension_excludes_traded_price_without_claiming_no_trade(timing, scope):
    """场景：有效日线同时有同日S记录；预期：停牌股不计广度，日内与未知时段不冒称无成交。"""
    fields = {"trade_date": "20260930", "suspend_type": "S"}
    if timing != "missing":
        fields["suspend_timing"] = timing
    suspended = row("suspend_d", record="suspended", fields=fields)
    (result,) = build_market(market_inputs(rows=(row(), row(index=1, record="b"), suspended)))
    assert result.details["denominator_count"] == 1
    assert result.details["excluded_suspended_count"] == 1
    assert result.details["excluded_no_trade_count"] == 0
    assert result.details["suspension_observations"][0]["scope"] == scope
    assert any(ref.source_api == "suspend_d" for ref in result.envelope.contrary_evidence)
    assert not any(ref.source_api == "suspend_d" for ref in result.envelope.evidence_refs)


def test_suspended_stock_without_daily_price_is_still_counted():
    """场景：停牌股没有日线但其他股提供当日日期；预期：仍按停牌计数，不当成未知价格。"""
    suspended = row("suspend_d", fields={"trade_date": "20260930", "suspend_type": "S"})
    (result,) = build_market(market_inputs(rows=(row(index=1, record="b"), suspended)))
    assert result.details["denominator_count"] == 1
    assert result.details["excluded_suspended_count"] == 1
    assert result.details["excluded_price_unknown_count"] == 0


@pytest.mark.parametrize("kind", ["resume", "old", "other_entity", "future", "bad_date"])
def test_suspension_join_requires_same_visible_security_and_trade_date(kind):
    """场景：复牌、异日、其他实体、未来公开或日期异常；预期：均不当作当前股票明确停牌。"""
    fields = {
        "trade_date": "20260929" if kind == "old" else "bad" if kind == "bad_date" else "20260930",
        "suspend_type": "R" if kind == "resume" else "S",
        "suspend_timing": None,
    }
    suspended = row(
        "suspend_d",
        index=1 if kind == "other_entity" else 0,
        fields=fields,
        data="2026-09-29" if kind == "old" else "2026-09-30",
        public="2026-10-03T00:00:00Z" if kind == "future" else "2026-09-30T16:00:00Z",
    )
    (result,) = build_market(market_inputs(1, (row(), suspended)))
    assert result.details["denominator_count"] == (0 if kind == "future" else 1)
    assert result.details["excluded_suspension_unknown_count"] == (1 if kind == "future" else 0)
    assert result.details["excluded_suspended_count"] == 0


@pytest.mark.parametrize("kind", ["unknown_type", "variant"])
def test_uncertain_suspension_excludes_without_asserting_confirmed_state(kind):
    """场景：未知停复牌类型或未解决变体；预期：排除不确定样本，计入未知停牌状态而非明确停牌。"""
    fields = {"trade_date": "20260930", "suspend_type": "S" if kind == "variant" else "X"}
    if kind == "variant":
        fields["business_variant_key"] = "variant"
    suspended = row("suspend_d", fields=fields)
    (result,) = build_market(market_inputs(1, (row(), suspended)))
    assert result.details["denominator_count"] == 0
    assert result.details["excluded_suspension_unknown_count"] == 1
    assert result.details["excluded_suspended_count"] == 0


def test_gdp_quarter_and_cpi_month_keep_period_and_availability_independent():
    """场景：季度GDP和月度CPI与日行情并存；预期：各自保留报告期与可用时点，GDP只用当季同比。"""
    gdp = row(
        "cn_gdp",
        record="gdp",
        fields={"quarter": "2026Q2", "gdp_yoy": 0.05, "gdp": 500000.0},
        data="2026-Q2",
        public="2026-07-20T08:00:00Z",
    )
    cpi = row(
        "cn_cpi",
        record="cpi",
        fields={"month": "202608", "nt_yoy": 0.01, "nt_mom": -0.01, "nt_val": 101.0},
        data="2026-08",
    )
    (result,) = build_market(market_inputs(1, (row(), gdp, cpi)))
    states = {state["source_api"]: state for state in result.details["macro_states"]}
    assert result.details["trade_date"] == "2026-09-30"
    assert result.envelope.data_as_of == "2026-09-30"
    assert result.envelope.source_as_of["cn_gdp"] == "2026-Q2"
    assert states["cn_gdp"]["frequency"] == "quarter"
    assert states["cn_gdp"]["period"] == "2026-Q2"
    assert states["cn_gdp"]["value"] == 0.05
    assert states["cn_gdp"]["status"] == "current_only"
    assert states["cn_gdp"]["available_at"] == NOW
    assert states["cn_gdp"]["public_available_at"] == "2026-07-20T08:00:00Z"
    assert states["cn_cpi"]["period"] == "2026-08"
    assert states["cn_cpi"]["value"] == 0.01
    assert all(ref.fields != ("gdp",) for ref in result.envelope.evidence_refs)


def test_gdp_adjacent_quarters_work_across_year_boundary():
    """场景：2025Q4和2026Q1两季均已公开；预期：计算当季同比的差，保留上一季度时间。"""
    older = row(
        "cn_gdp", record="older", fields={"quarter": "2025-Q4", "gdp_yoy": 0.05}, data="2025-Q4"
    )
    newer = row(
        "cn_gdp", record="newer", fields={"quarter": "2026Q1", "gdp_yoy": 0.04}, data="2026-Q1"
    )
    (result,) = build_market(market_inputs(0, (older, newer)))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_gdp")
    assert state["change"] == pytest.approx(-0.01)
    assert state["previous_period"] == "2025-Q4"
    assert result.envelope.data_as_of == NOW[:10]
    assert state["previous_available_at"] == NOW
    assert state["status"] == "adjacent_change"


@pytest.mark.parametrize("quarter", [None, "invalid", "2026Q5", "202601", "2027Q1"])
def test_gdp_invalid_or_future_quarter_cannot_supply_current_state(quarter):
    """场景：GDP季度缺失、非法、月度伪装或未来季度；预期：不生成季度状态。"""
    gdp = row("cn_gdp", fields={"quarter": quarter, "gdp_yoy": 0.05})
    (result,) = build_market(market_inputs(0, (gdp,)))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_gdp")
    assert state["status"] == "unknown"
    assert state["period"] is None


def test_gdp_report_period_mismatch_and_uncertified_yoy_fail_closed():
    """场景：季度与时间适配冲突或只有累计GDP没有同比证书；预期：均不补造季度增速。"""
    original = row("cn_gdp", fields={"quarter": "2026Q2", "gdp_yoy": 0.05})
    mismatch = replace(
        original, availability=replace(original.availability, report_period="2026-Q1")
    )
    (result,) = build_market(market_inputs(0, (mismatch,)))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_gdp")
    assert state["value"] is None
    unverified = replace(original, numeric={})
    (result,) = build_market(market_inputs(0, (unverified,)))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_gdp")
    assert state["value"] is None
    assert "macro_unit_or_value_unverified" in state["reason_codes"]


def test_gdp_gap_and_future_publication_leave_current_only():
    """场景：旧季度不相邻且下一季度尚未公开；预期：当前可见季只展示当前值，无变化。"""
    older = row("cn_gdp", record="older", fields={"quarter": "2025Q4", "gdp_yoy": 0.05})
    current = row("cn_gdp", record="current", fields={"quarter": "2026Q2", "gdp_yoy": 0.04})
    future = row(
        "cn_gdp",
        record="future",
        fields={"quarter": "2026Q3", "gdp_yoy": 0.03},
        public="2026-10-03T00:00:00Z",
    )
    (result,) = build_market(market_inputs(0, (older, current, future)))
    state = next(item for item in result.details["macro_states"] if item["source_api"] == "cn_gdp")
    assert state["period"] == "2026-Q2"
    assert state["status"] == "current_only"
    assert state["change"] is None
    assert not any(ref.source_record_id == "future" for ref in result.envelope.evidence_refs)


def test_gdp_and_cpi_do_not_change_daily_stress_formula():
    """场景：同一日频样本追加季度及月度宏观；预期：E0及三个贡献值完全不变。"""
    daily = (
        row(),
        row("stk_limit", record="limit", fields={"trade_date": "20260930", "down_limit": 9.0}),
    )
    macros = (
        row("cn_gdp", fields={"quarter": "2026Q2", "gdp_yoy": -0.05}),
        row("cn_cpi", fields={"month": "202608", "nt_yoy": 0.2}),
    )
    (original,) = build_market(market_inputs(1, daily))
    (expanded,) = build_market(market_inputs(1, daily + macros))
    assert original.metric("stress_score") == expanded.metric("stress_score")
    assert original.envelope.contributions == expanded.envelope.contributions
