"""市场和技术旁证的离线领域回归，所有数值与来源均由确定性合成夹具构造。"""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from tushare_entitlement_hub.equity_research.consumption import execute_consumption
from tushare_entitlement_hub.equity_research.consumption_registry import load_consumption_registry
from tushare_entitlement_hub.equity_research.context_consumers import (
    CONTEXT_FIELDS,
    consume_context,
)
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import EntityMapping, FoundationBindings
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityAlias, EntityIdentity
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-03T12:00:00Z"
STOCK = "cn:security:A:listed"
INDEX = "cn:index:I:published"
MARKET = "cn:market:SZ:all"


def context_row(
    api,
    fields,
    *,
    entity=STOCK,
    record=None,
    day="2026-09-30",
    units=None,
    unverified=(),
    observed=NOW,
):
    """构造有完整来源引用的合成行；例：close=10默认认证人民币价格，index则为指数点。

    units可显式改变量纲，unverified保留raw数值却不生成认证值，用于检查未知不被补成0。
    """
    identifier = record or api + "/" + str(entity) + "/" + day
    raw = dict(fields) | {
        "source_api": api,
        "source_record_id": identifier,
        "business_group_key": identifier,
        "contract_hash": "synthetic",
        "quality_tier": "verified",
    }
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=identifier,
        certification_version="synthetic",
        observed_at=observed,
        data_as_of=day,
        public_available_at="2026-09-01T00:00:00Z",
        market_timezone="Asia/Shanghai",
    )
    row = ResearchRow(entity_id=entity, source_api=api, raw=raw, availability=availability)
    numeric = {}
    for field, value in fields.items():
        if type(value) not in {int, float} or field in unverified:
            continue
        unit, dimension, currency = (
            ("index", "index", None)
            if entity == INDEX and field in {"close", "pre_close"}
            else ("元", "price", "CNY")
            if field in {"close", "pre_close", "close_qfq", "close_hfq"}
            else ("元", "money", "CNY")
            if field in {"amount", "total_mv", "circ_mv", "net_amount"}
            else ("个", "quantity", None)
            if field in {"com_count", "count", "total", "weekly_new", "weekly_hold", "weekly_trade"}
            else ("股", "quantity", None)
            if field == "vol"
            else ("ratio", "ratio", None)
        )
        unit, dimension, currency = (units or {}).get(field, (unit, dimension, currency))
        refs = row.evidence(field)
        certification = UnitCertification(
            version="synthetic",
            source_api=api,
            field=field,
            source_unit=unit,
            target_unit=unit,
            dimension=dimension,
            currency=currency,
            multiplier=1,
            evidence_refs=refs,
        )
        numeric[field] = TypedFact(
            fact_id=identifier + "/" + field,
            entity_id=entity or MARKET,
            field=field,
            value=value,
            unit=unit,
            dimension=dimension,
            currency=currency,
            availability=availability,
            quality_tier="verified",
            certification=certification,
            evidence_refs=refs,
        )
    return replace(row, numeric=numeric)


def context_inputs(*rows, complete=(), aliases=(), mappings=()):
    """创建股票、指数和市场分离的显式范围；例：全部三类存在也只有A是股票候选。

    仅启用调用者声明的完整来源范围，不提供默认别名、时间推断或单位外部数据。
    """
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind=kind, local_id=code, lifecycle_id=lifecycle
            ),
            name=code,
        )
        for kind, code, lifecycle in (
            ("security", "A", "listed"),
            ("index", "I", "published"),
            ("market", "SZ", "all"),
        )
    )
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="context-fixture",
        config=ResearchConfig(
            foundation=FoundationBindings(
                entities=entities, aliases=aliases, entity_mappings=mappings
            ),
            complete_scopes=complete,
        ),
        rows_by_api={
            api: tuple(row for row in rows if row.source_api == api)
            for api in {row.source_api for row in rows}
        },
    )


def consume(inputs, api, entity=STOCK):
    """调用真实消费入口并保持P01/P03归属；例：index_daily交P01，factor_value交P03。

    返回完整处理器结果，测试对领域事实、证据和软门禁进行断言，不替换实现分支。
    """
    product = (
        "P01"
        if api
        in {
            "daily_info",
            "eco_cal",
            "idx_factor_pro",
            "index_basic",
            "index_daily",
            "index_dailybasic",
            "index_monthly",
            "index_weekly",
            "moneyflow_mkt_dc",
            "shibor_lpr",
            "stk_account",
            "sz_daily_info",
            "trade_cal",
        }
        else "P03"
    )
    return consume_context(inputs, product, api, inputs.rows(api), entity)


@pytest.mark.parametrize("api", sorted(CONTEXT_FIELDS))
def test_every_context_api_has_empty_state_with_no_fabricated_evidence(api):
    """场景：26个分配接口逐一无数据；预期：入口均可执行，unknown且不捏造观察或引用。"""
    result = consume(context_inputs(), api)
    assert result["status"] == "unknown"
    assert result["facts"]["observations"] == []
    assert result["evidence_refs"] == ()
    assert result["facts"]["independent_evidence_count"] == 0
    assert result["facts"]["core_score_contribution"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"quality_tier": "quarantine"},
        {"quality_tier": "unknown"},
        {"is_quarantined": True},
        {"business_variant_key": "ambiguous"},
    ],
)
def test_quarantined_and_unresolved_index_technical_rows_are_rejected(change):
    """场景：指数技术数据被隔离或存在业务变体；预期：不消费其数字、字段和伪通过门禁。"""
    source = context_row(
        "idx_factor_pro",
        {"ts_code": "I", "trade_date": "20260930", "close": 100.0, "vol": 10.0, "amount": 100.0},
        entity=INDEX,
    )
    source = replace(source, raw=source.raw | change)
    result = consume(context_inputs(source), "idx_factor_pro")
    assert result["status"] == "unknown"
    assert result["evidence_refs"] == ()
    assert "context_rows_rejected_by_quality_or_time" in result["reason_codes"]


def test_future_visible_time_cannot_become_current_observation():
    """场景：LPR行只能在分析时点以后观察；预期：字段存在也不进入当前利率背景。"""
    source = context_row(
        "shibor_lpr", {"date": "20260930", "1y": 0.03, "5y": 0.04}, observed="2026-10-04T00:00:00Z"
    )
    assert consume(context_inputs(source), "shibor_lpr")["evidence_refs"] == ()


def test_index_identity_return_and_metadata_are_separate_from_stock_candidates():
    """场景：认证指数有编制元数据、102/100点位和2%涨幅；预期：核对匹配且候选仍只有股票A。"""
    metadata = context_row(
        "index_basic",
        {
            "ts_code": "I",
            "market": "SSE",
            "publisher": "发布方",
            "index_type": "宽基",
            "weight_rule": "市值加权",
        },
        entity=INDEX,
    )
    quote = context_row(
        "index_daily",
        {
            "ts_code": "I",
            "trade_date": "20260930",
            "close": 102.0,
            "pre_close": 100.0,
            "pct_chg": 0.02,
            "amount": 100.0,
        },
        entity=INDEX,
    )
    inputs = context_inputs(metadata, quote)
    assert inputs.entity_ids == (STOCK,)
    assert consume(inputs, "index_basic")["status"] == "observed"
    result = consume(inputs, "index_daily")
    assert result["facts"]["observations"][0]["return_check"] == "matched"
    assert result["facts"]["observations"][0]["recomputed_return"] == pytest.approx(0.02)
    wrong = replace(quote, entity_id=STOCK)
    assert consume(context_inputs(wrong), "index_daily")["status"] == "unknown"


def test_index_return_conflict_and_unverified_estimate_do_not_affect_core_score():
    """场景：指数点位涨2%但供应商报负2%，另有未认证估值；预期：冲突软门禁、估值未知。"""
    quote = context_row(
        "index_daily",
        {
            "trade_date": "20260930",
            "close": 102.0,
            "pre_close": 100.0,
            "pct_chg": -0.02,
            "amount": 100.0,
        },
        entity=INDEX,
    )
    valuation = context_row(
        "index_dailybasic",
        {"trade_date": "20260930", "pe_ttm": 10.0},
        entity=INDEX,
        unverified=("pe_ttm",),
    )
    result = consume(context_inputs(quote), "index_daily")
    assert result["status"] == "conflict"
    assert all(not check.hard for check in result["gates"])
    assert consume(context_inputs(valuation), "index_dailybasic")["status"] == "unknown"


@pytest.mark.parametrize("mismatch", ["none", "amount", "scope", "unit"])
def test_market_summary_comparison_requires_same_certified_scope_and_units(mismatch):
    """场景：深市两个汇总接口在金额、实体范围或单位上不同；预期：只对同范围同单位对账。"""
    left = context_row(
        "daily_info",
        {
            "trade_date": "20260930",
            "ts_code": "SZ_A",
            "exchange": "SZ",
            "com_count": 20.0,
            "amount": 100.0,
            "total_mv": 200.0,
        },
        entity=MARKET,
    )
    right = context_row(
        "sz_daily_info",
        {
            "trade_date": "20260930",
            "ts_code": "股票",
            "count": 20.0,
            "amount": 110.0 if mismatch == "amount" else 100.0,
            "total_mv": 200.0,
        },
        entity=None if mismatch == "scope" else MARKET,
        units={"amount": ("万元", "money", "CNY")} if mismatch == "unit" else None,
    )
    result = consume(context_inputs(left, right), "daily_info")
    assert result["status"] == (
        "observed" if mismatch == "none" else "conflict" if mismatch == "amount" else "unknown"
    )
    assert not result["facts"]["observations"][0]["local_sample_is_exchange_total"]


def test_market_summary_keeps_local_sample_distinct_from_exchange_total():
    """场景：官方市场金额100而本地仅有股票A金额20；预期：只报告本地样本，不判汇总冲突。"""
    summary = context_row(
        "daily_info",
        {
            "trade_date": "20260930",
            "ts_code": "SZ_A",
            "exchange": "SZ",
            "com_count": 20.0,
            "amount": 100.0,
            "total_mv": 200.0,
        },
        entity=MARKET,
    )
    metadata = context_row("stock_basic", {"ts_code": "A", "exchange": "SZSE"})
    daily = context_row("daily", {"ts_code": "A", "trade_date": "20260930", "amount": 20.0})
    result = consume(context_inputs(summary, metadata, daily), "daily_info")
    state = result["facts"]["observations"][0]["local_exchange_sample"]
    assert result["status"] == "unknown"
    assert state["amount_sum_cny"] == 20.0
    assert state["observed_stock_count"] == 1
    assert not state["comparable_to_reported_scope"]
    assert any(ref.source_api == "daily" for ref in result["evidence_refs"])


def test_macro_context_preserves_rate_accounts_and_provider_flow_meanings():
    """场景：LPR期限差1%、账户20/100参与交易、主力净额为负；预期：各自计算且不混为资金。"""
    sources = (
        context_row("shibor_lpr", {"date": "20260930", "1y": 0.03, "5y": 0.04}),
        context_row(
            "stk_account",
            {
                "date": "20260930",
                "weekly_new": 3.0,
                "total": 100.0,
                "weekly_hold": 50.0,
                "weekly_trade": 20.0,
            },
        ),
        context_row(
            "moneyflow_mkt_dc",
            {"trade_date": "20260930", "net_amount": -5.0, "net_amount_rate": -0.02},
        ),
    )
    inputs = context_inputs(*sources)
    assert consume(inputs, "shibor_lpr")["facts"]["observations"][0][
        "term_spread"
    ] == pytest.approx(0.01)
    accounts = consume(inputs, "stk_account")["facts"]["observations"][0]
    assert accounts["weekly_trade_share"] == 0.2
    assert not accounts["monetary_flow_interpretation"]
    assert consume(inputs, "moneyflow_mkt_dc")["facts"]["observations"][0]["direction"] == "outflow"


def test_calendar_closed_does_not_assert_stock_suspension_and_bse_is_unknown():
    """场景：SSE休市记录和未经派生的BSE记录；预期：只认交易所休市，个股停牌仍为空。"""
    closed = context_row(
        "trade_cal",
        {"exchange": "SSE", "cal_date": "20261003", "is_open": "0", "pretrade_date": "20260930"},
        day="2026-10-03",
        entity=None,
    )
    state = consume(context_inputs(closed), "trade_cal")["facts"]["observations"][0]
    assert state["exchange_state"] == "exchange_closed"
    assert state["individual_suspension"] is None
    bse = replace(closed, raw=closed.raw | {"exchange": "BSE"})
    assert consume(context_inputs(bse), "trade_cal")["status"] == "unknown"


def test_economic_calendar_keeps_forecast_surprise_disabled_and_current_change_auditable():
    """场景：经济事件今值3前值2预测值2.5；预期：认证今前值差为1，未知预测版本不算预期差。"""
    source = context_row(
        "eco_cal",
        {
            "date": "20260930",
            "time": "09:30",
            "country": "中国",
            "event": "合成发布",
            "value": 3.0,
            "pre_value": 2.0,
            "fore_value": 2.5,
        },
        entity=None,
    )
    result = consume(context_inputs(source), "eco_cal")
    state = result["facts"]["observations"][0]
    assert state["previous_value_change"] == 1.0
    assert state["forecast_surprise"] is None
    assert state["schedule_state"] == "release_observed"
    assert any(ref.fields == ("fore_value",) for ref in result["evidence_refs"])


@pytest.mark.parametrize("certified", [False, True])
def test_bse_mapping_requires_two_visible_certified_aliases(certified):
    """场景：旧新代码来源相同但别名认证有无；预期：只有两侧均指股票A时确认同实体。"""
    source = context_row(
        "bse_mapping", {"o_code": "OLD", "n_code": "NEW", "list_date": "20200101"}, entity=None
    )
    aliases = (
        tuple(
            EntityAlias(
                namespace="tushare",
                alias=code,
                entity_id=STOCK,
                effective_from="2019-01-01T00:00:00Z",
                effective_to=None,
                observed_at=NOW,
                source="synthetic",
                version="v1",
                confidence=100,
                evidence_refs=source.evidence("o_code"),
            )
            for code in ("OLD", "NEW")
        )
        if certified
        else ()
    )
    inputs = context_inputs(
        source,
        aliases=aliases,
        mappings=(
            EntityMapping(source_api="bse_mapping", namespace="tushare", key_field="n_code"),
        ),
    )
    result = consume(inputs, "bse_mapping")
    assert result["status"] == ("observed" if certified else "unknown")
    assert result["facts"]["observations"][0]["same_entity_confirmed"] is certified
    assert not result["facts"]["observations"][0]["creates_additional_security"]


@pytest.mark.parametrize("asset", ["STK", "IDX"])
def test_factor_values_join_stock_dictionary_without_independent_score(asset):
    """场景：同名因子字典分别声明股票或指数；预期：股票定义才可旁证，公式历史都不启用。"""
    dictionary = context_row(
        "factor_list",
        {
            "factor_name": "momentum",
            "asset_type": asset,
            "factor_type": "technical",
            "factor_desc": "供应商定义",
        },
        entity=None,
    )
    source = context_row(
        "factor_value",
        {"ts_code": "A", "trade_date": "20260930", "factor_name": "momentum", "factor_value": 0.2},
    )
    result = consume(context_inputs(dictionary, source), "factor_value")
    assert result["status"] == ("not_enabled_due_to_history" if asset == "STK" else "unknown")
    if asset == "STK":
        assert any(ref.source_api == "factor_list" for ref in result["evidence_refs"])
        assert result["facts"]["observations"][0]["supplier_value"]["factor_value"]["value"] == 0.2
    assert not result["facts"]["independent_evidence"]
    assert consume(context_inputs(dictionary), "factor_list")["status"] == (
        "observed" if asset == "STK" else "not_applicable"
    )


def test_adjustment_anchor_is_real_row_with_adjacent_change_not_historical_certification():
    """场景：两天复权因子2和2.2；预期：当前锚点绑定后一天真实记录，变化1.1且历史关闭。"""
    previous = context_row(
        "adj_factor",
        {"ts_code": "A", "trade_date": "20260929", "adj_factor": 2.0},
        day="2026-09-29",
    )
    current = context_row(
        "adj_factor", {"ts_code": "A", "trade_date": "20260930", "adj_factor": 2.2}
    )
    result = consume(context_inputs(previous, current), "adj_factor")
    state = result["facts"]["observations"][0]
    assert result["status"] == "not_enabled_due_to_history"
    assert state["anchor_record_id"] == current.availability.source_record_id
    assert state["factor_change_ratio"] == 1.1
    assert not state["historical_adjustment_enabled"]
    assert len({ref.source_record_id for ref in result["evidence_refs"]}) == 2


@pytest.mark.parametrize("api", ["bak_daily", "pro_bar", "idx_factor_pro"])
@pytest.mark.parametrize("zero", [False, True])
def test_quote_reconciliation_requires_traded_prices_and_never_fills_zero(api, zero):
    """场景：辅助行情与主日线相同或辅助量为0；预期：同口径匹配，0量则未知且不补主行情。"""
    entity = INDEX if api == "idx_factor_pro" else STOCK
    fields = {
        "ts_code": "I" if entity == INDEX else "A",
        "trade_date": "20260930",
        "close": 10.0,
        "vol": 10.0,
        "amount": 100.0,
    }
    main = context_row("index_daily" if entity == INDEX else "daily", fields, entity=entity)
    source = context_row(api, fields | {"vol": 0.0 if zero else 10.0}, entity=entity)
    result = consume(context_inputs(main, source), api)
    assert result["status"] == ("unknown" if zero else "observed")
    assert result["facts"]["core_score_contribution"] == 0


@pytest.mark.parametrize("api", ["weekly", "monthly", "index_weekly", "index_monthly"])
def test_period_sources_preserve_current_close_when_full_history_is_missing(api):
    """场景：周月行只有统计终点日线；预期：当前收盘核对匹配，完整期间明确因历史不足关闭。"""
    entity = INDEX if api.startswith("index_") else STOCK
    fields = {
        "ts_code": "I" if entity == INDEX else "A",
        "trade_date": "20260930",
        "close": 10.0,
        "vol": 10.0,
        "amount": 100.0,
    }
    main = context_row("index_daily" if entity == INDEX else "daily", fields, entity=entity)
    source = context_row(api, fields, entity=entity)
    result = consume(context_inputs(main, source), api)
    state = result["facts"]["observations"][0]
    assert result["status"] == "not_enabled_due_to_history"
    assert state["current_close_check"]["status"] == "matched"
    assert state["full_period_check"] == "unknown"
    assert {ref.source_api for ref in result["evidence_refs"]} == {api, main.source_api}


def test_combined_period_uses_statistical_end_instead_of_friday_label():
    """场景：周五标签对应周三统计终点，周五日线故意不同；预期：仅以周三价格核对。"""
    source = context_row(
        "stk_weekly_monthly",
        {
            "ts_code": "A",
            "trade_date": "20261002",
            "end_date": "20260930",
            "freq": "week",
            "close": 10.0,
            "vol": 10.0,
        },
        day="2026-10-02",
    )
    wednesday = context_row(
        "daily",
        {"ts_code": "A", "trade_date": "20260930", "close": 10.0, "vol": 10.0, "amount": 100.0},
    )
    friday = context_row(
        "daily",
        {"ts_code": "A", "trade_date": "20261002", "close": 99.0, "vol": 10.0, "amount": 100.0},
        day="2026-10-02",
    )
    result = consume(context_inputs(source, wednesday, friday), "stk_weekly_monthly")
    state = result["facts"]["observations"][0]
    assert state["label_date"] == "2026-10-02"
    assert state["statistical_end_date"] == "2026-09-30"
    assert state["current_close_check"]["status"] == "matched"
    assert all(
        ref.source_record_id != friday.availability.source_record_id
        for ref in result["evidence_refs"]
    )


@pytest.mark.parametrize("reported_volume", [50.0, 60.0])
def test_complete_week_requires_all_calendar_days_and_reconciles_daily_totals(reported_volume):
    """场景：完整七日日历和五个开市日日线；预期：周成交量50匹配，报60触发软冲突。"""
    start = date(2026, 9, 21)
    calendar = tuple(
        context_row(
            "trade_cal",
            {
                "exchange": "SSE",
                "cal_date": (start + timedelta(days=offset)).isoformat(),
                "is_open": "1" if offset < 5 else "0",
                "pretrade_date": "20260918",
            },
            day=(start + timedelta(days=offset)).isoformat(),
            entity=None,
        )
        for offset in range(7)
    )
    daily = tuple(
        context_row(
            "daily",
            {
                "ts_code": "A",
                "trade_date": (start + timedelta(days=offset)).isoformat(),
                "close": 10.0,
                "vol": 10.0,
                "amount": 100.0,
            },
            day=(start + timedelta(days=offset)).isoformat(),
        )
        for offset in range(5)
    )
    metadata = context_row("stock_basic", {"ts_code": "A", "exchange": "SSE"})
    source = context_row(
        "weekly",
        {
            "ts_code": "A",
            "trade_date": "20260925",
            "close": 10.0,
            "vol": reported_volume,
            "amount": 500.0,
        },
        day="2026-09-25",
    )
    result = consume(
        context_inputs(*calendar, *daily, metadata, source, complete=("trade_cal",)), "weekly"
    )
    state = result["facts"]["observations"][0]
    assert result["status"] == ("observed" if reported_volume == 50 else "conflict")
    assert state["period_phase"] == "complete"
    assert state["aggregation_checks"]["vol"]["daily_sum"] == 50.0
    assert state["expected_trading_days"] == 5


def test_adjusted_period_does_not_guess_supplier_normalization_anchor():
    """场景：复权周线有前后复权价、主日线和当前因子；预期：仍要求供应商固定锚点证据。"""
    source = context_row(
        "stk_week_month_adj",
        {
            "ts_code": "A",
            "trade_date": "20261002",
            "end_date": "20260930",
            "freq": "week",
            "close_qfq": 5.0,
            "close_hfq": 20.0,
        },
        day="2026-10-02",
    )
    daily = context_row(
        "daily", {"trade_date": "20260930", "close": 10.0, "vol": 10.0, "amount": 100.0}
    )
    factor = context_row("adj_factor", {"trade_date": "20260930", "adj_factor": 2.0})
    result = consume(context_inputs(source, daily, factor), "stk_week_month_adj")
    assert result["status"] == "not_enabled_due_to_history"
    state = result["facts"]["observations"][0]
    assert state["current_adjusted_quotes"]["close_qfq"]["value"] == 5.0
    assert state["supplier_normalization_anchor"] is None
    assert any(ref.source_api == "adj_factor" for ref in result["evidence_refs"])


@pytest.mark.parametrize("api", ["stk_factor", "stk_factor_pro"])
def test_supplier_technical_fields_are_current_only_with_same_day_adjustment_checks(api):
    """场景：供应商因子与同日复权因子一致；预期：保存当前技术值和核对，公式历史不进入评分。"""
    source = context_row(
        api,
        {
            "ts_code": "A",
            "trade_date": "20260930",
            "adj_factor": 2.0,
            "close_qfq": 5.0,
            "macd": 0.1,
            "rsi_6": 0.6,
            "pe_ttm": 10.0,
            "circ_mv": 100.0,
        },
    )
    factor = context_row("adj_factor", {"trade_date": "20260930", "adj_factor": 2.0})
    valuation = context_row(
        "daily_basic", {"trade_date": "20260930", "pe_ttm": 10.0, "circ_mv": 100.0}
    )
    result = consume(context_inputs(source, factor, valuation), api)
    state = result["facts"]["observations"][0]
    assert result["status"] == "not_enabled_due_to_history"
    assert state["same_day_checks"]["adj_factor"]["status"] == "matched"
    assert not state["formula_validated"]
    if api == "stk_factor_pro":
        assert state["same_day_checks"]["pe_ttm"]["status"] == "matched"


def test_nineturn_timestamp_preserves_market_date_and_supplier_event_semantics():
    """场景：九转字段带已认证市场时区时间戳；预期：提取市场日期并保存上九转事件，历史关闭。"""
    source = context_row(
        "stk_nineturn",
        {
            "ts_code": "A",
            "trade_date": "2026-09-30T15:00:00+08:00",
            "freq": "daily",
            "nine_up_turn": 9.0,
            "nine_down_turn": 0.0,
        },
    )
    result = consume(context_inputs(source), "stk_nineturn")
    state = result["facts"]["observations"][0]
    assert result["status"] == "not_enabled_due_to_history"
    assert state["trade_date"] == "2026-09-30"
    assert state["up_event"] and not state["down_event"]


def test_ambiguous_current_versions_are_not_arbitrarily_chosen():
    """场景：同日同公开时点两份不同LPR记录；预期：当前版本未知且保留两份实际字段引用。"""
    first = context_row("shibor_lpr", {"date": "20260930", "1y": 0.03, "5y": 0.04}, record="first")
    second = context_row(
        "shibor_lpr", {"date": "20260930", "1y": 0.04, "5y": 0.05}, record="second"
    )
    result = consume(context_inputs(first, second), "shibor_lpr")
    assert result["status"] == "unknown"
    assert "context_current_version_ambiguous" in result["reason_codes"]
    assert {ref.source_record_id for ref in result["evidence_refs"]} == {"first", "second"}


@pytest.mark.parametrize("net,expected", [(5.0, "inflow"), (0.0, "balanced"), (None, "unknown")])
def test_market_flow_direction_distinguishes_zero_from_unavailable(net, expected):
    """场景：市场净额为正、零或缺失；预期：依次为流入、平衡、未知，不将缺失补零。"""
    source = context_row("moneyflow_mkt_dc", {"trade_date": "20260930", "net_amount": net})
    result = consume(context_inputs(source), "moneyflow_mkt_dc")
    assert result["facts"]["observations"][0]["direction"] == expected
    assert result["status"] == ("unknown" if net is None else "observed")


@pytest.mark.parametrize(
    "total,held,status",
    [(0.0, 0.0, "unknown"), (100.0, -1.0, "conflict"), (100.0, 101.0, "conflict")],
)
def test_account_participation_rejects_impossible_counts(total, held, status):
    """场景：账户总量零或持有数超出0至总量范围；预期：未知或冲突，仍不解释为资金。"""
    source = context_row("stk_account", {"date": "20260930", "total": total, "weekly_hold": held})
    result = consume(context_inputs(source), "stk_account")
    assert result["status"] == status
    state = result["facts"]["observations"][0]
    assert state["weekly_hold_share"] == (None if total == 0 else held / total)
    assert state["weekly_trade_share"] is None
    assert state["monetary_flow_interpretation"] is False


@pytest.mark.parametrize("date_text", ["not-a-date", "2026-02-30", "20260929"])
def test_current_dates_must_be_valid_and_match_certified_business_date(date_text):
    """场景：LPR日期非法或和认证日期不符；预期：观测日期未知，不计算利差。"""
    source = context_row("shibor_lpr", {"date": date_text, "1y": 0.03, "5y": 0.04})
    result = consume(context_inputs(source), "shibor_lpr")
    assert result["status"] == "unknown"
    assert "observation_date_unverified" in result["reason_codes"]
    assert result["facts"]["observations"][0]["observation_date"] is None


@pytest.mark.parametrize(
    "units", [{"5y": ("fraction", "ratio", None)}, {"1y": ("元", "money", "CNY")}]
)
def test_lpr_spread_requires_matching_certified_ratio_units(units):
    """场景：LPR两期限的单位不同或量纲误认证；预期：不计算期限差并保留单位未知原因。"""
    source = context_row("shibor_lpr", {"date": "20260930", "1y": 0.03, "5y": 0.04}, units=units)
    result = consume(context_inputs(source), "shibor_lpr")
    assert result["status"] == "unknown"
    assert result["facts"]["observations"][0]["term_spread"] is None
    assert "lpr_units_unverified" in result["reason_codes"]


@pytest.mark.parametrize(
    "timestamp,timezone",
    [
        ("2026-09-30T15:00:00", "Asia/Shanghai"),
        ("2026-09-30T15:00:00+08:00", None),
        ("2026-09-30T15:00:00+08:00", "Invalid/Zone"),
        ("bad-timestamp", "Asia/Shanghai"),
    ],
)
def test_nineturn_requires_explicit_valid_timestamp_timezone(timestamp, timezone):
    """场景：九转时间戳缺偏移、缺市场时区或时区非法；预期：日期未知，不进入历史状态。"""
    source = context_row(
        "stk_nineturn", {"trade_date": timestamp, "freq": "daily", "nine_up_turn": 9.0}
    )
    source = replace(source, availability=replace(source.availability, market_timezone=timezone))
    result = consume(context_inputs(source), "stk_nineturn")
    assert result["status"] == "unknown"
    assert result["facts"]["observations"][0]["trade_date"] is None


@pytest.mark.parametrize(
    "close,previous", [(0.0, 100.0), (102.0, 0.0), (None, 100.0), (1e308, 1e-308)]
)
def test_index_return_cannot_use_zero_missing_or_overflowed_levels(close, previous):
    """场景：指数价格零、缺失或有限输入除法溢出；预期：收益未知，不生成非有限数值。"""
    source = context_row(
        "index_daily",
        {"trade_date": "20260930", "close": close, "pre_close": previous, "pct_chg": 0.02},
        entity=INDEX,
    )
    result = consume(context_inputs(source), "index_daily")
    state = result["facts"]["observations"][0]
    assert result["status"] == "unknown"
    assert state["recomputed_return"] is None
    assert state["return_check"] == "unknown"


@pytest.mark.parametrize(
    "api,fields,reason",
    [
        ("weekly", {"trade_date": "invalid"}, "period_date_or_frequency_unverified"),
        (
            "stk_weekly_monthly",
            {"trade_date": "20260930", "end_date": "20260930", "freq": "year"},
            "period_date_or_frequency_unverified",
        ),
        (
            "stk_weekly_monthly",
            {"trade_date": "20260930", "end_date": "20260920", "freq": "week"},
            "period_label_mismatch",
        ),
        (
            "stk_weekly_monthly",
            {"trade_date": "20260930", "end_date": "20260831", "freq": "month"},
            "period_label_mismatch",
        ),
    ],
)
def test_period_labels_and_frequency_cannot_be_guessed(api, fields, reason):
    """场景：周期日期非法、频率不支持或跨周月标签；预期：未知并明确对应日期口径原因。"""
    source = context_row(api, fields | {"close": 10.0, "vol": 10.0})
    result = consume(context_inputs(source), api)
    assert result["status"] == "unknown"
    assert reason in result["reason_codes"]
    assert result["facts"]["observations"][0]["full_period_check"] == "unknown"


def complete_period_inputs(*, mode="normal", api="weekly"):
    """构造完整交易周及可控质量缺口；例：normal得到周量50，unit得到单位不一致。"""
    start = date(2026, 9, 21)
    calendar = tuple(
        context_row(
            "trade_cal",
            {
                "exchange": "SSE",
                "cal_date": (start + timedelta(days=i)).isoformat(),
                "is_open": "1" if i < 5 else "0",
                "pretrade_date": "20260918",
            },
            day=(start + timedelta(days=i)).isoformat(),
            entity=None,
        )
        for i in range(7)
    )
    if mode == "duplicate_calendar":
        calendar += (
            replace(
                calendar[0],
                raw=calendar[0].raw | {"source_record_id": "duplicate"},
                availability=replace(calendar[0].availability, source_record_id="duplicate"),
            ),
        )
    daily = tuple(
        context_row(
            "daily",
            {
                "ts_code": "A",
                "trade_date": (start + timedelta(days=i)).isoformat(),
                "close": 10.0,
                "vol": 1e308 if mode == "overflow" else 10.0,
                "amount": 100.0,
            },
            day=(start + timedelta(days=i)).isoformat(),
            units={"vol": ("手", "quantity", None)} if mode == "unit" and i == 0 else None,
            unverified=("vol",) if mode == "missing_value" and i == 0 else (),
        )
        for i in range(5)
    )
    if mode == "missing_day":
        daily = daily[1:]
    metadata = context_row("stock_basic", {"ts_code": "A", "exchange": "SSE"})
    source = context_row(
        api,
        {
            "ts_code": "A",
            "trade_date": "20260925",
            "end_date": "20260925",
            "freq": "week",
            "close": 11.0 if mode == "close_conflict" else 10.0,
            "vol": 50.0,
            "amount": 500.0,
        },
        day="2026-09-25",
    )
    return context_inputs(*calendar, *daily, metadata, source, complete=("trade_cal",))


@pytest.mark.parametrize("mode", ["unit", "missing_value", "overflow"])
def test_period_aggregation_preserves_unknown_for_uncertified_or_overflowing_volume(mode):
    """场景：完整周存在量单位不同、未认证日量或求和溢出；预期：聚合未知，不能报告通过。"""
    result = consume(complete_period_inputs(mode=mode), "weekly")
    state = result["facts"]["observations"][0]
    assert result["status"] == "unknown"
    assert state["period_phase"] == "complete"
    assert state["aggregation_checks"]["vol"]["status"] == "unknown"
    assert state["aggregation_checks"]["amount"]["status"] == "matched"
    if mode == "overflow":
        assert state["aggregation_checks"]["vol"]["daily_sum"] is None


@pytest.mark.parametrize("mode", ["duplicate_calendar", "missing_day"])
def test_complete_period_rejects_duplicate_calendar_and_missing_traded_day(mode):
    """场景：日历重复或实际日线少一天；预期：当前收盘仍可核对，但完整周历史不可认证。"""
    result = consume(complete_period_inputs(mode=mode), "weekly")
    state = result["facts"]["observations"][0]
    assert result["status"] == "not_enabled_due_to_history"
    assert state["current_close_check"]["status"] == "matched"
    assert state["full_period_check"] == "unknown"
    assert state["expected_trading_days"] is None


@pytest.mark.parametrize("api", ["weekly", "stk_weekly_monthly"])
def test_complete_period_close_conflict_stops_aggregate_acceptance(api):
    """场景：周收11而统计末日日线10；预期：关闭完整聚合并输出价格冲突和两侧真实引用。"""
    result = consume(complete_period_inputs(mode="close_conflict", api=api), api)
    assert result["status"] == "conflict"
    assert "period_close_conflict" in result["reason_codes"]
    assert result["facts"]["observations"][0]["current_close_check"]["status"] == "conflict"
    assert {api, "daily"} <= {ref.source_api for ref in result["evidence_refs"]}


def test_combined_complete_week_aggregates_only_its_declared_volume_field():
    """场景：组合周线已匹配完整交易周；预期：仅核对声明成交量50，不凭空添加金额聚合。"""
    result = consume(complete_period_inputs(api="stk_weekly_monthly"), "stk_weekly_monthly")
    assert result["status"] == "observed"
    assert result["facts"]["observations"][0]["aggregation_checks"] == {
        "vol": {"status": "matched", "daily_sum": 50.0, "reported": 50.0}
    }


def test_context_helpers_fail_closed_for_missing_entity_and_uncertified_comparison():
    """场景：未绑定实体或对照值无证书；预期：不猜交易所和同日对照，数值比较unknown。"""
    from tushare_entitlement_hub.equity_research.context_consumers import (
        _comparison,
        _exchange,
        _peer,
    )

    unbound = context_row("pro_bar", {"trade_date": "20260930", "close": 10.0}, entity=None)
    missing = context_row("daily", {"trade_date": "20260930", "close": 10.0}, unverified=("close",))
    assert _peer(context_inputs(unbound, missing), unbound, "daily") is None
    assert _exchange(context_inputs(unbound), unbound) is None
    assert _comparison(unbound, "close", missing, "close") == {
        "status": "unknown",
        "reason": "comparison_unit_or_source_unavailable",
    }
    high = context_row("daily", {"close": 1e308})
    low = context_row("daily", {"close": -1e308})
    assert _comparison(high, "close", low, "close") == {
        "status": "unknown",
        "reason": "nonfinite_comparison",
    }


@pytest.mark.parametrize(
    "api,fields,entity",
    [
        ("index_daily", {"trade_date": "invalid", "close": 10.0}, INDEX),
        ("weekly", {"trade_date": "20260930", "close": 10.0}, INDEX),
        ("pro_bar", {"trade_date": "20260930", "close": 10.0, "vol": 10.0, "amount": 10.0}, STOCK),
    ],
)
def test_context_missing_identity_date_or_peer_preserves_unknown(api, fields, entity):
    """场景：指数日期非法、股票周线错绑指数或组合行情缺主行情；预期：unknown且评分贡献0。"""
    source = context_row(api, fields, entity=entity)
    result = consume(context_inputs(source), api)
    assert result["status"] == "unknown"
    assert result["facts"]["core_score_contribution"] == 0


@pytest.mark.parametrize("product,api", [("P06", "daily_info"), ("P01", "not_registered")])
def test_context_dispatch_rejects_unregistered_product_or_api(product, api):
    """场景：上下文处理器被错误产品或未知接口调用；预期：明确拒绝错误绑定。"""
    with pytest.raises(ValueError, match="unsupported_context_consumer"):
        consume_context(context_inputs(), product, api, (), STOCK)


@pytest.mark.parametrize("mode", ["no_quote", "no_amount", "overflow"])
def test_market_local_sample_handles_missing_quote_and_nonfinite_sum(mode):
    """场景：样本股无当日报价、无金额证书或多股金额求和溢出；预期：样本总额未知。"""
    summary = context_row(
        "daily_info",
        {
            "trade_date": "20260930",
            "ts_code": "SZ",
            "exchange": "SZ",
            "com_count": 2.0,
            "amount": 100.0,
            "total_mv": 100.0,
        },
        entity=MARKET,
    )
    metadata = context_row("stock_basic", {"ts_code": "A", "exchange": "SZSE"})
    amount = 1e308 if mode == "overflow" else 100.0
    quote = context_row(
        "daily",
        {"ts_code": "A", "trade_date": "20260930", "amount": amount},
        unverified=("amount",) if mode == "no_amount" else (),
    )
    sources = (summary, metadata) if mode == "no_quote" else (summary, metadata, quote)
    context = context_inputs(*sources)
    if mode == "overflow":
        other_id = "cn:security:B:listed"
        other_metadata = context_row(
            "stock_basic", {"ts_code": "B", "exchange": "SZSE"}, entity=other_id
        )
        other_quote = context_row(
            "daily", {"ts_code": "B", "trade_date": "20260930", "amount": amount}, entity=other_id
        )
        context = context_inputs(*sources, other_metadata, other_quote)
        first = context.config.foundation.entities[0]
        other_entity = replace(
            first,
            identity=EntityIdentity(
                namespace="cn", kind="security", local_id="B", lifecycle_id="listed"
            ),
        )
        context = replace(
            context,
            config=replace(
                context.config,
                foundation=replace(
                    context.config.foundation,
                    entities=context.config.foundation.entities + (other_entity,),
                ),
            ),
        )
    result = consume(context, "daily_info")
    state = result["facts"]["observations"][0]["local_exchange_sample"]
    assert result["status"] == "unknown"
    assert state["amount_sum_cny"] is None
    assert state["observed_stock_count"] == (
        0 if mode == "no_quote" else 2 if mode == "overflow" else 1
    )
    assert state["comparable_to_reported_scope"] is False


@pytest.mark.parametrize("previous", [None, 0.0])
def test_adjustment_anchor_without_valid_previous_value_keeps_change_unknown(previous):
    """场景：当前复权锚点2但无前值或前值零；预期：当前锚点可用，相邻变化未知。"""
    current = context_row("adj_factor", {"trade_date": "20260930", "adj_factor": 2.0})
    prior = (
        ()
        if previous is None
        else (
            context_row(
                "adj_factor", {"trade_date": "20260929", "adj_factor": previous}, day="2026-09-29"
            ),
        )
    )
    result = consume(context_inputs(*prior, current), "adj_factor")
    assert result["status"] == "not_enabled_due_to_history"
    state = result["facts"]["observations"][0]
    assert state["anchor_factor"] == 2.0
    assert state["factor_change_ratio"] is None


def test_adjusted_week_without_peer_or_anchor_retains_unknown():
    """场景：复权周线仅有自身价格且无主日线或锚点；预期：unknown，不凭供应商值启用历史。"""
    source = context_row(
        "stk_week_month_adj",
        {"trade_date": "20260930", "end_date": "20260930", "freq": "week", "close_qfq": 5.0},
    )
    result = consume(context_inputs(source), "stk_week_month_adj")
    assert result["status"] == "unknown"
    state = result["facts"]["observations"][0]
    assert state["supplier_normalization_anchor"] is None
    assert "observed_factor" not in state


@pytest.mark.parametrize("api", ["stk_factor", "stk_factor_pro"])
def test_supplier_technical_values_without_local_peers_remain_nonindependent(api):
    """场景：供应商技术记录缺本地复权和估值对照；预期：当前值保留，对账unknown且历史关闭。"""
    source = context_row(api, {"trade_date": "20260930", "adj_factor": 2.0, "close_qfq": 5.0})
    result = consume(context_inputs(source), api)
    assert result["status"] == "not_enabled_due_to_history"
    assert all(
        check["status"] == "unknown"
        for check in result["facts"]["observations"][0]["same_day_checks"].values()
    )
    assert result["facts"]["independent_evidence"] is False


@pytest.mark.parametrize(
    "certified_field",
    ["missing_numeric", "missing_certification", "pe_ttm", "pb", "turnover_rate"],
)
def test_index_valuation_consumption_requires_at_least_one_certified_numeric(certified_field):
    """消费入口必须区分原始数字与认证值，来源无其他缺口时仍返回明确估值认证原因。

    输入pe_ttm=10、pb=2、turnover_rate=0.03，numeric缺失或所有真实TypedFact的
    certification=None时输出unknown及index_valuation_units_unverified；分别只认证
    三字段之一时输出observed、空原因集合，并且仅该字段的估值状态为observed。
    """
    values = {"pe_ttm": 10.0, "pb": 2.0, "turnover_rate": 0.03}
    source = context_row(
        "index_dailybasic", {"ts_code": "I", "trade_date": "20260930", **values}, entity=INDEX
    )
    if certified_field == "missing_numeric":
        source = replace(source, numeric={})
    else:
        source = replace(
            source,
            numeric={
                field: fact
                if field == certified_field
                else replace(
                    fact,
                    value=None,
                    quality_tier="unknown",
                    certification=None,
                    reason_codes=("unit_unverified",),
                )
                for field, fact in source.numeric.items()
            },
        )
    inputs = replace(context_inputs(source), source_reasons={"index_dailybasic": ()})
    registration = next(
        item for item in load_consumption_registry() if item.spec.api == "index_dailybasic"
    )
    assert (registration.spec.product, registration.handler) == ("P01", "context")
    assert inputs.reasons("index_dailybasic", INDEX) == ()
    assert {field: source.raw[field] for field in values} == values
    if certified_field == "missing_certification":
        assert set(source.numeric) == set(values)
        assert all(
            isinstance(fact, TypedFact) and fact.certification is None
            for fact in source.numeric.values()
        )
    record, references, _ = execute_consumption(inputs, registration, INDEX)
    observed = certified_field in values
    assert record["executed"] is True
    assert record["retained_row_count"] == 1
    assert record["status"] == ("observed" if observed else "unknown")
    assert record["reason_codes"] == ([] if observed else ["index_valuation_units_unverified"])
    assert references
    valuation = record["facts"]["observations"][0]["valuation_context"]
    for field, value in values.items():
        certified = field == certified_field
        assert valuation[field]["status"] == ("observed" if certified else "unknown")
        assert valuation[field]["value"] == (value if certified else None)
