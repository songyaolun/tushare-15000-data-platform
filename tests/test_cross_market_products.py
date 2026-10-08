"""P09/P10/P11 的单位、实体、时点、交叉加工和有界证据测试。"""

from dataclasses import replace
from math import inf

import pytest

from tushare_entitlement_hub.cross_asset_research import commodities as commodity
from tushare_entitlement_hub.cross_asset_research import financing, international
from tushare_entitlement_hub.cross_asset_research import market_support as support
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.entities import EntityAlias, EntityRelation
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-03T12:00:00Z"
DAY = "2026-10-02"
A, B, C = "test:security:A:life", "test:security:B:life", "test:security:C:life"


def source(
    api, record="one", entity=A, raw=None, values=None, day=DAY, observed=NOW, timezone="UTC"
):
    """构造明确量纲合成证据；例如 close=(10,'元','price','USD') 保留美元价格。"""
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="test/v1",
        observed_at=observed,
        data_as_of=day,
        market_timezone=timezone,
    )
    payload = {
        "source_api": api,
        "source_record_id": record,
        "business_group_key": record,
        "contract_hash": "synthetic",
        "trade_date": day,
        "date": day,
        "ts_code": entity.split(":")[2] if entity else "unknown",
        **(raw or {}),
    }
    payload.update({key: spec[0] for key, spec in (values or {}).items()})
    row = ResearchRow(entity_id=entity, source_api=api, raw=payload, availability=availability)
    numeric = {}
    for name, specification in (values or {}).items():
        value, unit, dimension, currency = specification
        refs = row.evidence(name)
        certificate = UnitCertification(
            version="test/v1",
            source_api=api,
            field=name,
            source_unit=unit,
            target_unit=unit,
            dimension=dimension,
            currency=currency,
            multiplier=1,
            evidence_refs=refs,
        )
        numeric[name] = TypedFact(
            fact_id=api + "/" + record + "/" + name,
            entity_id=entity or A,
            field=name,
            value=value,
            unit=unit,
            dimension=dimension,
            currency=currency,
            availability=availability,
            quality_tier="verified",
            certification=certificate,
            evidence_refs=refs,
            reason_codes=("test_missing",) if value is None else (),
        )
    return replace(row, numeric=numeric)


def value(number, dimension="ratio", unit="ratio", currency=None):
    """生成认证元组；例如 value(10,'quantity','手') 是十手而非十股。"""
    return number, unit, dimension, currency


def inputs(*rows, relations=(), aliases=(), reasons=None, limit=512):
    """组装完全离线输入；例如 limit=1 使同快照第二条有效业务记录触发容量拒绝。"""
    grouped = {}
    for row in rows:
        grouped.setdefault(row.source_api, []).append(row)
    config = ResearchConfig(
        foundation=FoundationBindings(aliases=aliases),
        confirmed_relations=relations,
        maximum_rows_per_entity_source=limit,
    )
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="cross-test",
        config=config,
        rows_by_api={api: tuple(items) for api, items in grouped.items()},
        source_reasons=reasons or {},
    )


def relation(kind, first=A, second=B, confirmed=True):
    """构造显式且有效的实体关系；例如 option_underlying 将期权 A 对应标的 B。"""
    return EntityRelation(
        from_entity_id=first,
        to_entity_id=second,
        relation_type=kind,
        confirmed=confirmed,
        effective_from="2000-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="synthetic",
        version="v1",
        confidence=100,
        evidence_refs=source("relation", raw={"kind": kind}).evidence("kind"),
    )


def find(results, name):
    """查找唯一的业务指标；例如 find(results,'net_income_margin') 返回包含净利率的结果。"""
    return next(result for result in results if any(item.name == name for item in result.metrics))


def commodity_sources():
    """覆盖十九个商品API的合成快照；预期每个来源都有具体加工和自身字段血缘。"""
    definitions = {
        "fut_basic": (
            {
                "fut_code": "CU",
                "exchange": "SHFE",
                "d_month": "202611",
                "quote_unit": "元/吨",
                "trade_unit": "吨",
                "per_unit": "5",
            },
            {},
        ),
        "fut_daily": (
            {},
            {
                "settle": value(110),
                "close": value(100),
                "vol": value(10, "quantity", "手"),
                "oi": value(20, "quantity", "手"),
            },
        ),
        "fut_daily_adj": ({}, {"settle": value(110), "close": value(100), "vol": value(10)}),
        "fut_holding": (
            {"symbol": "CU", "exchange": "SHFE", "broker": "member1"},
            {"long_hld": value(10, "quantity", "手"), "short_hld": value(20, "quantity", "手")},
        ),
        "fut_index_daily": ({}, {"close": value(100), "pct_chg": value(0.02)}),
        "fut_inv_weekly": (
            {"exchange": "SHFE", "warehouse": "W", "grade": "G", "fut_name": "CU", "unit": "吨"},
            {"cur_total": value(120, "quantity", "吨"), "pre_total": value(100, "quantity", "吨")},
        ),
        "fut_mapping": ({"mapping_ts_code": "A"}, {}),
        "fut_rcpt_mat": (
            {"exchange": "SHFE", "fut_name": "CU", "unit": "吨"},
            {"cur_month": value(10), "next_month": value(20)},
        ),
        "fut_settle": (
            {},
            {
                "settle": value(110),
                "long_margin_rate": value(0.1),
                "short_margin_rate": value(0.08),
            },
        ),
        "fut_trade_cal": ({"exchange": "SHFE", "cal_date": DAY, "is_open": 1}, {}),
        "fut_trade_param": (
            {"exchange": "SHFE", "fut_code": "CU"},
            {"tick_size": value(1), "price_limit": value(0.1), "position_limit": value(100)},
        ),
        "fut_weekly_detail": (
            {"exchange": "SHFE", "prd": "CU"},
            {"vol": value(10), "open_interest": value(20)},
        ),
        "fut_weekly_monthly": (
            {"freq": "W"},
            {"close": value(100), "settle": value(110), "vol": value(10)},
        ),
        "fut_wsr": (
            {"symbol": "CU", "warehouse": "W", "grade": "G", "is_ct": "0", "unit": "吨"},
            {"vol": value(100), "vol_chg": value(5)},
        ),
        "ft_limit": ({}, {"up_limit": value(110), "down_limit": value(90), "m_ratio": value(0.1)}),
        "sge_basic": (
            {"trade_type": "spot", "t_unit": "克", "p_unit": "元/克", "trade_time": "session"},
            {},
        ),
        "sge_daily": (
            {},
            {"close": value(100), "price_avg": value(101), "vol": value(10), "oi": value(20)},
        ),
        "fina_mainbz": (
            {"bz_item": "铜材", "curr_type": "CNY"},
            {
                "bz_sales": value(100, "money", "元", "CNY"),
                "bz_cost": value(60, "money", "元", "CNY"),
            },
        ),
        "fina_mainbz_vip": (
            {"bz_item": "铜材", "curr_type": "CNY"},
            {
                "bz_sales": value(100, "money", "元", "CNY"),
                "bz_cost": value(60, "money", "元", "CNY"),
            },
        ),
    }
    return tuple(source(api, raw=raw, values=values) for api, (raw, values) in definitions.items())


def test_all_commodity_sources_produce_deep_metrics_and_evidence():
    """场景：十九来源齐全；预期：库存、持仓、分部利润率和去重核对均可复算。"""
    results = commodity.build_commodities(inputs(*commodity_sources()))
    assert set(commodity.API_FIELDS) == {result.details["source_api"] for result in results}
    assert find(results, "inventory_change_ratio").metric(
        "inventory_change_ratio"
    ) == pytest.approx(0.2)
    assert find(results, "segment_gross_margin").metric("segment_gross_margin") == pytest.approx(
        0.4
    )
    assert find(results, "settle_source_difference").metric("settle_source_difference") == 0
    assert (
        find(results, "continuous_actual_close_deviation").metric(
            "continuous_actual_close_deviation"
        )
        == 0
    )
    assert all(len(result.to_json().encode()) < support.MAXIMUM_RESULT_BYTES for result in results)
    assert all(result.envelope.evidence_refs for result in results)


@pytest.mark.parametrize(
    "module,builder",
    [
        (commodity, commodity.build_commodities),
        (international, international.build_international),
        (financing, financing.build_financing),
    ],
)
def test_missing_and_duplicate_snapshots_fail_closed(module, builder):
    """场景：每个API分别缺字段及同日多行；预期：保留结果和未知状态，不伪造精确指标。"""
    rows = [source(api, raw={}) for api in module.API_FIELDS]
    assert builder(inputs(*rows))
    doubled = rows + [
        replace(
            row,
            raw={**row.raw, "business_group_key": "second", "source_record_id": "second"},
            availability=replace(row.availability, source_record_id="second"),
        )
        for row in rows
    ]
    assert builder(inputs(*doubled))
    assert builder(inputs()) == ()


@pytest.mark.parametrize(
    "far,expected", [(110, "contango"), (90, "backwardation"), (100, "flat"), (None, "unknown")]
)
def test_futures_curve_uses_explicit_month_and_specs(far, expected):
    """场景：同品种同规格双合约；预期：按显式交割月分类曲线且引用规格证据。"""
    basic = commodity_sources()[0]
    farther = replace(basic, entity_id=B, raw={**basic.raw, "d_month": "2026-12"})
    near_row = source("fut_daily", values={"settle": value(100)})
    far_row = source("fut_daily", "far", entity=B, values={"settle": value(far)})
    results = commodity.build_commodities(inputs(basic, farther, near_row, far_row))
    curve = find(results, "far_near_settlement_spread")
    assert curve.details["curve_state"] == expected
    assert any(ref.source_api == "fut_basic" for ref in curve.envelope.evidence_refs)


@pytest.mark.parametrize("month", ["bad", "202613", "202611", None])
def test_futures_curve_missing_or_duplicate_month(month):
    """场景：交割月非法、缺失或重复；预期：不编造期限差，冲突月份整体拒绝。"""
    basic = commodity_sources()[0]
    farther = replace(basic, entity_id=B, raw={**basic.raw, "d_month": month})
    rows = (
        basic,
        farther,
        source("fut_daily", values={"settle": value(100)}),
        source("fut_daily", "far", entity=B, values={"settle": value(110)}),
    )
    curves = commodity._curves(inputs(*rows))
    if month == "202611":
        assert curves[0].metric("far_near_settlement_spread") is None
        assert curves[0].details["curve_state"] == "unknown"
    else:
        assert curves == []


def test_commodity_relations_enable_only_explicit_attribution():
    """场景：明确产业及可比现期关系；预期：增加证据归因，无关系时不出现。"""
    rows = list(commodity_sources())
    spot = source("sge_daily", entity=B, values={"close": value(100)})
    relations = (relation("commodity_exposure"), relation("spot_futures_comparable", B, A))
    results = commodity.build_commodities(inputs(*rows, spot, relations=relations))
    assert find(results, "futures_spot_basis_ratio").metric("futures_spot_basis_ratio") == 0
    assert any(
        result.details.get("stock_attribution") == "confirmed_relation_only" for result in results
    )
    assert not any(
        result.details.get("stock_attribution") == "confirmed_relation_only"
        for result in commodity.build_commodities(inputs(*rows))
    )


def test_member_hhi_does_not_mix_totals_or_duplicate_brokers():
    """场景：两个等额会员及混入合计；预期：HHI为0.5，合计混入则未知。"""
    first = commodity_sources()[3]
    second = replace(first, raw={**first.raw, "business_group_key": "second", "broker": "member2"})
    result = find(commodity.build_commodities(inputs(first, second)), "long_hld_sample_hhi")
    assert result.metric("long_hld_sample_hhi") == 0.5
    total = replace(second, raw={**second.raw, "broker": "合计"})
    assert (
        find(commodity.build_commodities(inputs(first, total)), "long_hld_sample_hhi").metric(
            "long_hld_sample_hhi"
        )
        is None
    )


def international_sources():
    """覆盖所有海外源的合成样本；预期：外币保留、各收益率期限独立计算。"""
    rows = []
    for api in international.API_FIELDS:
        raw = {name: "metadata" for name in international.API_FIELDS[api]}
        raw.update(
            exchange="FX",
            report_type="annual",
            currency="USD",
            ind_type="consolidated",
            curr_type="USD",
            cal_date=DAY,
            pretrade_date="2026-10-01",
            is_open=1,
        )
        numerical = {
            name: value(0.03)
            for name in (
                "on",
                "1m",
                "3m",
                "12m",
                "w4_bd",
                "w4_ce",
                "w13_bd",
                "w13_ce",
                "ltc",
                "cmt",
                "ltr_avg",
                "y5",
                "y7",
                "y10",
                "y20",
                "y30",
                "m3",
                "y2",
                "gross_profit_ratio",
                "ocf_sales",
                "ah_premium",
                "pct_chg",
                "pct_change",
            )
            if name in international.API_FIELDS[api]
        }
        if api == "fx_daily":
            numerical.update(bid_close=value(7), ask_close=value(7.2))
        if api.endswith(("_income", "_balancesheet", "_cashflow")):
            labels = (
                [("net_income", 20), ("revenue", 100)]
                if api.endswith("income")
                else [("total_liabilities", 20), ("total_assets", 100)]
                if api.endswith("balancesheet")
                else [
                    ("operating_cashflow", 20),
                    ("investing_cashflow", -10),
                    ("financing_cashflow", 10),
                ]
            )
            rows.extend(
                source(
                    api,
                    label,
                    raw={**raw, "ind_name": label},
                    values={"ind_value": value(amount, "money", "元", "USD")},
                )
                for label, amount in labels
            )
        else:
            rows.append(source(api, raw=raw, values=numerical))
    return tuple(rows)


def test_all_international_sources_preserve_currency_and_market_scope():
    """场景：二十六来源齐备；预期：美元财务净利率可算、所有来源自带证据。"""
    results = international.build_international(inputs(*international_sources()))
    assert set(international.API_FIELDS) == {result.details["source_api"] for result in results}
    assert find(results, "net_income_margin").metric("net_income_margin") == 0.2
    assert find(results, "quoted_midpoint").metric("quoted_midpoint") == 7.1
    assert find(results, "ten_year_breakeven_proxy").metric("ten_year_breakeven_proxy") == 0
    assert find(results, "observed_market_return").envelope.product_id == international.PRODUCT_ID


def test_ah_pair_requires_matching_explicit_hong_kong_alias():
    """场景：A/H有关系但港股代码必须匹配有效别名；预期：只有完整配对输出溢价。"""
    row = source("stk_ah_comparison", raw={"hk_code": "0001"}, values={"ah_premium": value(0.1)})
    link = relation("ah_pair")
    alias = EntityAlias(
        namespace="tushare_hk",
        alias="0001",
        entity_id=B,
        effective_from="2000-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="test",
        version="v1",
        confidence=100,
        evidence_refs=row.evidence("hk_code"),
    )
    assert (
        international.build_international(inputs(row, relations=(link,), aliases=(alias,)))[
            0
        ].metric("ah_premium")
        == 0.1
    )
    conflict = replace(alias, entity_id=C)
    assert (
        international.build_international(
            inputs(row, relations=(link,), aliases=(alias, conflict))
        )[0].metric("ah_premium")
        is None
    )


def test_financial_longtable_conflict_currency_and_denominator():
    """场景：重复指标、外币不同或收入零；预期：不产生可比净利率。"""
    revenue = source(
        "us_income",
        "revenue",
        raw={"ind_type": "annual", "report_type": "annual", "ind_name": "revenue"},
        values={"ind_value": value(100, "money", "元", "USD")},
    )
    profit = source(
        "us_income",
        "profit",
        raw={
            **revenue.raw,
            "source_record_id": "profit",
            "business_group_key": "profit",
            "ind_name": "net_income",
        },
        values={"ind_value": value(20, "money", "元", "USD")},
    )
    duplicate = replace(profit, raw={**profit.raw, "business_group_key": "duplicate"})
    assert (
        international.build_international(inputs(revenue, profit, duplicate))[0].metric(
            "net_income_margin"
        )
        is None
    )
    foreign = source(
        "us_income",
        "foreign",
        raw={"ind_type": "annual", "report_type": "annual", "ind_name": "net_income"},
        values={"ind_value": value(20, "money", "元", "HKD")},
    )
    assert (
        international.build_international(inputs(revenue, foreign))[0].metric("net_income_margin")
        is None
    )


def financing_sources():
    """覆盖十四个融资源的确定性样本；预期：报价、成交、融资本金和利率保持区分。"""
    rows = []
    for api, fields in financing.API_FIELDS.items():
        raw = {name: "spec" for name in fields}
        raw.update(
            remain_maturity="5Y",
            repo_maturity="1D",
            exchange_id="SSE",
            bank="bank1",
            qt_time="09:00:00",
            call_put="C",
            maturity_date="2026-12-31",
            exercise_type="E",
            opt_type="ETF",
            opt_multiplier="100",
        )
        values = {
            name: value(0.03)
            for name in fields
            if name
            in {
                "best_buy_yield",
                "best_sell_yield",
                "buy_yield",
                "sell_yield",
                "close",
                "weight",
                "pct_change",
                "d10_rate",
                "m1_rate",
                "m3_rate",
                "long_rate",
                "comp_rate",
                "micro_rate",
                "on",
                "1w",
                "1m",
                "3m",
                "1y",
                "on_b",
                "on_a",
                "1m_b",
                "1m_a",
            }
        }
        values.update(
            {
                name: value(amount, "money", "元", "CNY")
                for name, amount in {
                    "rzye": 100,
                    "rzmre": 30,
                    "rzche": 20,
                    "rzrqye": 125,
                    "ob": 100,
                    "auc_amount": 30,
                    "repo_amount": 10,
                    "repay_amount": 20,
                    "cb": 120,
                    "amount": 100,
                }.items()
                if name in fields
            }
        )
        values.update(
            {
                name: value(amount, "quantity", "手")
                for name, amount in {"vol": 10, "oi": 20}.items()
                if name in fields
            }
        )
        if "price" in fields:
            values["price"] = value(100, "price", "元", "CNY")
        rows.append(source(api, raw=raw, values=values))
    return tuple(rows)


def test_all_financing_sources_distinguish_rates_quotes_and_executions():
    """场景：十四融资来源齐备；预期：净融资强度0.1、转融资恒等式残差0。"""
    results = financing.build_financing(inputs(*financing_sources()))
    assert set(financing.API_FIELDS) == {result.details["source_api"] for result in results}
    assert find(results, "net_financing_balance_ratio").metric("net_financing_balance_ratio") == 0.1
    assert (
        find(results, "refinancing_balance_identity_residual").metric(
            "refinancing_balance_identity_residual"
        )
        == 0
    )
    assert (
        find(results, "closing_weighted_funding_spread").details["close_semantics"]
        == "funding_rate_not_security_price"
    )


def test_option_pcr_requires_explicit_underlying_and_separate_expiry():
    """场景：同标的同到期期权成交看跌20/看涨10；预期PCR=2并引用元数据和关系。"""
    spec = next(row for row in financing_sources() if row.source_api == "opt_basic")
    put_spec = replace(spec, entity_id=B, raw={**spec.raw, "call_put": "P"})
    call = source(
        "opt_daily", values={"vol": value(10, "quantity", "手"), "oi": value(30, "quantity", "手")}
    )
    put = source(
        "opt_daily",
        "put",
        entity=B,
        values={"vol": value(20, "quantity", "手"), "oi": value(60, "quantity", "手")},
    )
    links = (relation("option_underlying", A, C), relation("option_underlying", B, C))
    result = find(
        financing.build_financing(inputs(spec, put_spec, call, put, relations=links)),
        "put_call_vol_ratio",
    )
    assert result.metric("put_call_vol_ratio") == 2
    assert result.metric("put_call_oi_ratio") == 2
    assert {ref.source_api for ref in result.envelope.evidence_refs} >= {
        "opt_daily",
        "opt_basic",
        "relation",
    }
    assert financing._option_groups(inputs(spec, call)) == []


def test_quote_latest_bank_snapshot_and_trade_amount_concentration():
    """场景：银行重复更新时间和两笔成交；预期仅最新银行报价参与极差、成交不混报价。"""
    first = source(
        "bc_otcqt",
        raw={"remain_maturity": "5Y", "bank": "B1", "qt_time": "09:00"},
        values={"buy_yield": value(0.03), "sell_yield": value(0.02)},
    )
    newer = source(
        "bc_otcqt",
        "newer",
        raw={"remain_maturity": "5Y", "bank": "B1", "qt_time": "10:00"},
        values={"buy_yield": value(0.04), "sell_yield": value(0.03)},
    )
    other = source(
        "bc_otcqt",
        "other",
        raw={"remain_maturity": "5Y", "bank": "B2", "qt_time": "10:00"},
        values={"buy_yield": value(0.05), "sell_yield": value(0.04)},
    )
    result = financing.build_financing(inputs(first, newer, other))[0]
    assert result.metric("buy_yield_bank_range") == pytest.approx(0.01)
    assert result.details["bank_count"] == 2
    executions = [
        source(
            "bond_blk",
            str(index),
            values={"price": value(100 + index), "amount": value(amount, "money", "元", "CNY")},
        )
        for index, amount in enumerate((60, 40))
    ]
    assert (
        financing.build_financing(inputs(*executions))[0].metric("largest_block_amount_share")
        == 0.6
    )


def test_support_fact_checks_dimensions_currency_and_arithmetic():
    """场景：外币认证、事实错实体、空认证与单位不同；预期严格读取并阻断不当算术。"""
    row = source(
        "us_daily",
        values={"close": value(10, "price", "元", "USD"), "other": value(20, "price", "元", "USD")},
    )
    assert row.number("close") is None
    assert support.number(row, "close") == 10
    assert support.number(row, "close", ("money",)) is None
    assert support.unit(row, "close") == "USD:元"
    assert support.unit(row, "absent") == "source_unit"
    assert support.ratio(row, "other", "close", change=True) == 1
    assert support.difference(row, "other", "close") == 10
    assert support.fact(replace(row, entity_id=B), "close") is None
    assert support.fact(replace(row, entity_id=None), "close") is None
    assert support.fact(replace(row, raw={}), "close") is None
    assert support.fact(row, "missing") is None
    wrong = replace(
        row, numeric={"close": replace(row.numeric["close"], reason_codes=("unverified",))}
    )
    assert support.fact(wrong, "close") is None
    assert support.ratio(row, "close", "absent") is None
    assert support.difference(row, "close", "absent") is None
    overflow = source(
        "rate",
        values={"a": value(1e308), "b": value(-1e308), "tiny": value(1e-308), "zero": value(0)},
    )
    assert support.ratio(overflow, "a", "tiny") is None
    assert support.ratio(overflow, "a", "zero") is None
    assert support.difference(overflow, "a", "b") is None


def test_frame_full_counts_dedup_conflict_capacity_and_bounded_lineage():
    """场景：重复、冲突、较旧行、容量溢出与大量关系证据；预期精确计数但不输出截断统计。"""
    row = source("x", values={"a": value(1)})
    config = inputs(row, limit=1)
    frame = support.frames(config, "x", ("a",))[0]
    frame.add(config, row)
    frame.add(config, replace(row, raw={**row.raw, "a": 2}))
    frame.add(config, replace(row, raw={**row.raw, "business_group_key": "second"}))
    frame.add(config, source("x", "old", day="2026-10-01"))
    frame.cite(row.evidence("a") * 70)
    assert frame.observed_count == 5
    assert frame.duplicate_count == 1
    assert frame.value("derived", inf).value is None
    assert len(frame.supporting_references) == 64
    assert frame.summary()["supporting_reference_count"] == 70
    assert "capacity_exceeded" in frame.reasons
    result = support.finish(config, "test/v1", frame, [frame.value("derived", 1)], {})
    assert result.envelope.state.value == "blocked"


@pytest.mark.parametrize(
    "changed",
    [
        {"day": None},
        {"day": "2027-01-01"},
        {"observed": "2027-01-01T00:00:00Z"},
        {"raw": {"is_quarantined": True}},
        {"raw": {"quality_tier": "quarantine"}},
        {"raw": {"source_api": "wrong"}},
        {"raw": {"source_record_id": "wrong"}},
    ],
)
def test_frame_rejects_time_quality_and_identity(changed):
    """场景：未来、隔离或身份错误来源；预期整组被拒绝且不访问错误血缘。"""
    row = source("x", **changed)
    frame = support.frames(inputs(row), "x", ("trade_date",))[0]
    assert frame.rejected_count == 1
    assert frame.rows == []


def test_frame_missing_scope_unknown_entity_and_payload_guard(monkeypatch):
    """场景：缺实体/维度及输出容量阈值过小；预期未知实体不升级证券，超限结果结构化拒绝。"""
    row = source("x", entity=None)
    config = inputs(row)
    frame = support.frames(config, "x", ("trade_date",), ("warehouse",))[0]
    assert frame.reasons >= {"entity_unverified", "scope_dimension_missing"}
    monkeypatch.setattr(support, "MAXIMUM_RESULT_BYTES", 1)
    result = support.finish(config, "test/v1", frame, (), {})
    assert "result_payload_capacity_exceeded" in result.envelope.gate_codes


def test_spread_rejects_unknown_mixed_units_and_singletons():
    """场景：少于两样本、缺值、不同单位和正常样本；预期只有完整可比样本输出极差。"""
    rows = [
        source("x", str(index), values={"a": value(number)}) for index, number in enumerate((1, 3))
    ]
    frame = support.frames(inputs(*rows), "x", ("a",))[0]
    assert support.spread(frame, "a") == (2, "ratio")
    assert support.spread(frame, "absent")[0] is None
    frame.rows = frame.rows[:1]
    assert support.spread(frame, "a")[0] is None


def test_cross_source_comparisons_reject_missing_peer_and_uncertified_price():
    """场景：只存在核对左源、现货目标缺失或映射价格未认证；预期无虚假价差。"""
    rows = [row for row in commodity_sources() if row.source_api in {"fut_settle", "sge_daily"}]
    assert (
        commodity._cross_checks(
            inputs(*rows, relations=(relation("spot_futures_comparable", A, C),))
        )
        == []
    )
    actual = source("fut_daily", values={"close": value(None)})
    adjusted = source("fut_daily_adj", values={"close": value(100)})
    mapping = source("fut_mapping", raw={"mapping_ts_code": "A"})
    assert (
        commodity._cross_checks(inputs(actual, adjusted, mapping))[0].metric(
            "continuous_actual_close_deviation"
        )
        is None
    )


def test_latest_bank_quote_conflict_and_old_quote_do_not_change_precision():
    """场景：新报价先到、旧报价后到及同刻冲突；预期忽略旧报价且同刻冲突保留未知。"""
    newer = source(
        "bc_otcqt",
        "newer",
        raw={"remain_maturity": "5Y", "bank": "B1", "qt_time": "10:00"},
        values={"buy_yield": value(0.04)},
    )
    older = source(
        "bc_otcqt",
        "older",
        raw={"remain_maturity": "5Y", "bank": "B1", "qt_time": "09:00"},
        values={"buy_yield": value(0.03)},
    )
    conflict = source(
        "bc_otcqt",
        "conflict",
        raw={"remain_maturity": "5Y", "bank": "B1", "qt_time": "10:00"},
        values={"buy_yield": value(0.05)},
    )
    result = financing.build_financing(inputs(newer, older, conflict))[0]
    assert "bank_quote_time_conflict" in result.envelope.gate_codes
    assert result.metric("buy_yield_bank_range") is None


def test_option_quantity_failure_and_zero_revenue_remain_unknown():
    """场景：期权持仓负数或缺认证、财务收入零；预期PCR和净利率均不产出。"""
    spec = next(row for row in financing_sources() if row.source_api == "opt_basic")
    invalid = source("opt_daily", values={"vol": value(-1, "quantity", "手")})
    result = financing._option_groups(
        inputs(spec, invalid, relations=(relation("option_underlying"),))
    )[0]
    assert result.metric("put_call_vol_ratio") is None
    assert result.metric("put_call_oi_ratio") is None
    financial_rows = [
        source(
            "hk_income",
            name,
            raw={"ind_name": name},
            values={"ind_value": value(amount, "money", "元", "HKD")},
        )
        for name, amount in (("revenue", 0), ("net_income", 10))
    ]
    assert (
        international.build_international(inputs(*financial_rows))[0].metric("net_income_margin")
        is None
    )


def test_quote_time_and_rate_dimensions_cannot_be_guessed():
    """场景：不可解析报价时分及错误金额型利率；预期报价与期限差均未知。"""
    quote = source(
        "bc_otcqt",
        raw={"remain_maturity": "5Y", "bank": "B", "qt_time": "9:00"},
        values={"buy_yield": value(0.03)},
    )
    assert (
        "bank_quote_time_unverified"
        in financing.build_financing(inputs(quote))[0].envelope.gate_codes
    )
    rates = source(
        "shibor",
        values={"on": value(1, "money", "元", "CNY"), "1y": value(2, "money", "元", "CNY")},
    )
    assert financing.build_financing(inputs(rates))[0].metric("one_year_overnight_spread") is None


@pytest.mark.parametrize("maturity", ["unknown", "2020-01-01"])
def test_option_invalid_or_expired_maturity_not_grouped(maturity):
    """场景：到期日非法或成交时已过期；预期不构建虚假的同到期期权组。"""
    spec = next(row for row in financing_sources() if row.source_api == "opt_basic")
    spec = replace(spec, raw={**spec.raw, "maturity_date": maturity})
    quote = source("opt_daily", values={"vol": value(10, "quantity", "手")})
    assert (
        financing._option_groups(inputs(spec, quote, relations=(relation("option_underlying"),)))
        == []
    )


def test_source_local_observation_annotations_preserve_certified_local_metrics():
    """场景：真实适配层来源内身份注记及部分缺列；预期可用利率正常加工、范围注记保留。"""
    row = source(
        "shibor",
        entity="source:series:shibor:market",
        values={"on": value(0.01), "1m": value(0.02), "1y": value(0.03)},
    )
    result = financing.build_financing(
        inputs(row, reasons={"shibor": ("source_local_entity_only", "consumption_fields_missing")})
    )[0]
    assert result.metric("one_year_overnight_spread") == pytest.approx(0.02)
    assert result.details["scope_annotations"] == [
        "consumption_fields_missing",
        "source_local_entity_only",
    ]
    assert not result.envelope.decision_allowed
    blocked = financing.build_financing(
        inputs(row, reasons={"shibor": ("source_local_entity_only", "capacity_exceeded")})
    )[0]
    assert blocked.metric("one_year_overnight_spread") is None


def test_observed_metadata_without_business_date_keeps_snapshot_boundary():
    """场景：合约目录只有观察时点、没有业务日期；预期规格完整度可用且明确观察后快照。"""
    row = source(
        "fut_basic",
        day=None,
        raw={
            "fut_code": "CU",
            "exchange": "SHFE",
            "d_month": "202611",
            "quote_unit": "元/吨",
            "trade_unit": "吨",
            "per_unit": "5",
        },
    )
    result = commodity.build_commodities(
        inputs(row, reasons={"fut_basic": ("directory_identity_observed_only",)})
    )[0]
    assert result.metric("specification_completeness") == 1
    assert "metadata_snapshot_observed_only" in result.details["scope_annotations"]


def test_unknown_timezone_allows_local_curve_but_blocks_cross_market_comparison():
    """场景：时区未知但当前本源观察有效；预期本源期限差可用，跨源名义实际比较未知。"""
    nominal = source("us_tycr", timezone=None, values={"y2": value(0.02), "y10": value(0.03)})
    real = source("us_trycr", values={"y10": value(0.01)})
    results = international.build_international(
        inputs(nominal, real, reasons={"us_tycr": ("market_timezone_uncertified",)})
    )
    assert find(results, "nominal_ten_two_year_spread").metric(
        "nominal_ten_two_year_spread"
    ) == pytest.approx(0.01)
    assert find(results, "ten_year_breakeven_proxy").metric("ten_year_breakeven_proxy") is None


@pytest.mark.parametrize("api", ["fut_trade_cal", "hk_tradecal", "us_tradecal"])
@pytest.mark.parametrize(
    "raw_value,expected",
    [
        (1, 1.0),
        (1.0, 1.0),
        ("1", 1.0),
        (0, 0.0),
        (0.0, 0.0),
        ("0", 0.0),
        (True, None),
        (False, None),
        (1.5, None),
        (-1, None),
        (2, None),
        ("1.0", None),
        (None, None),
    ],
)
def test_calendar_binary_enum_keeps_normalized_float_semantics(api, raw_value, expected):
    """场景：正式标准化零一数字及非法枚举；预期精确0/1有效，任意数字或布尔值不猜开市。"""
    row = source(
        api,
        raw={
            "is_open": raw_value,
            "exchange": "SHFE",
            "cal_date": DAY,
            "pretrade_date": "2026-10-01",
        },
    )
    builder = (
        commodity.build_commodities if api == "fut_trade_cal" else international.build_international
    )
    result = builder(inputs(row))[0]
    assert (
        result.details["session_state"] == {0.0: "closed", 1.0: "open", None: "unknown"}[expected]
    )
    if api == "fut_trade_cal":
        assert result.metric("session_open") == expected


@pytest.mark.parametrize("bad_kind", ["future", "quarantine", "unknown"])
def test_retained_good_rows_survive_rejected_rows_and_source_audit_annotations(bad_kind):
    """场景：同源健康行和未来/隔离行并存；预期跳过坏行、保留统计与注记并继续健康期限差。"""
    good = source("shibor", values={"on": value(0.01), "1m": value(0.02), "1y": value(0.03)})
    bad = source(
        "shibor",
        "bad",
        day="2027-01-01" if bad_kind == "future" else DAY,
        raw={} if bad_kind == "future" else {"quality_tier": bad_kind},
    )
    annotations = (
        "rows_excluded_by_adapter",
        "row_unavailable",
        "future_business_date",
        "quarantined_or_unknown_quality",
    )
    result = financing.build_financing(inputs(good, bad, reasons={"shibor": annotations}))[0]
    assert result.metric("one_year_overnight_spread") == pytest.approx(0.02)
    assert result.details["rejected_count"] == 1
    assert result.details["observed_count"] == 2
    assert set(result.details["scope_annotations"]) >= {
        *annotations,
        "rows_rejected_time_or_quality",
    }
    assert all(ref.source_record_id != "bad" for ref in result.envelope.evidence_refs)


def test_retained_good_row_still_rejects_identity_or_capacity_failures():
    """场景：健康行与身份错配记录或硬容量失败并存；预期硬失败仍阻断精确指标。"""
    good = source("shibor", values={"on": value(0.01), "1y": value(0.03)})
    wrong = source("shibor", "wrong", raw={"source_api": "other"})
    result = financing.build_financing(inputs(good, wrong))[0]
    assert result.metric("one_year_overnight_spread") is None
    assert "source_identity_unverified" in result.envelope.gate_codes
    rejected = financing.build_financing(
        inputs(good, reasons={"shibor": ("rows_excluded_by_adapter", "capacity_exceeded")})
    )[0]
    assert rejected.metric("one_year_overnight_spread") is None
