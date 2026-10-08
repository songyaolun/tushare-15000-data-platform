"""P08全部基金接口、明确指数映射和保守缺口的离线场景。"""

from dataclasses import replace

import pytest
from test_research_governance import NOW, A, B, inputs, row

from tushare_entitlement_hub.cross_asset_research.funds import (
    API_FIELDS,
    _adjusted_return,
    _basket,
    _events,
    _fund_observations,
    _price_limits,
    _reference_index,
    _relative,
    _same_day,
    _shares,
    _traded,
    _weights,
    build_etf_exposure,
)


def fund_quote(record="price", day="2026-10-03", close=11, **raw):
    """场景：有量有额且收益口径显式的ETF日线；预期：可计算有效成交和固定锚收益。"""
    return row(
        "fund_daily",
        record,
        day=day,
        raw={"ts_code": "ETF", "return_basis": "total_return", **raw},
        values={"close": (close, "元"), "amount": (100, "元"), "vol": (10, "手")},
    )


def fund_size(record="size", day="2026-10-03", shares=110, nav=10):
    """场景：认证份额、净值和规模恒等；预期：规模为shares×nav，份额变化可复算。"""
    return row(
        "etf_share_size",
        record,
        day=day,
        values={
            "total_share": (shares, "份"),
            "total_size": (shares * nav, "元"),
            "nav": (nav, "元"),
            "close": (nav * 1.1, "元"),
        },
    )


def full_fund_rows():
    """场景：18个基金接口和同端点基准行情；预期：每个API具备自身可用业务证据。"""
    sources = [
        fund_quote(),
        fund_quote("old", "2026-10-02", 10),
        fund_size(),
        fund_size("oldsize", "2026-10-02", 100),
        row(
            "etf_basic",
            raw={"ts_code": "ETF", "index_code": "I", "etf_type": "权益", "mgr_name": "管理人"},
            values={"mgt_fee": (0.005, "fraction")},
        ),
        row(
            "fund_basic",
            raw={
                "ts_code": "ETF",
                "management": "管理人",
                "fund_type": "股票型",
                "market": "E",
                "benchmark": "基准",
            },
            values={"m_fee": (0.005, "fraction"), "c_fee": (0.001, "fraction")},
        ),
        row(
            "fund_company",
            entity=B,
            raw={"name": "管理人", "shortname": "简称", "org_code": "ORG", "credit_code": "CREDIT"},
        ),
        row(
            "etf_index",
            entity=B,
            raw={"ts_code": "I", "pub_party_name": "发布方", "adj_circle": "季度"},
        ),
        row(
            "mkt_idx_bmk",
            entity=B,
            raw={
                "ts_code": "I",
                "bmk_level": "1",
                "bmk_type": "宽基",
                "bmk_src": "机构",
                "idx_type": "权益",
            },
        ),
        row("etf_limit", values={"up_limit": (12, "元"), "down_limit": (10, "元")}),
        row(
            "fund_nav",
            raw={"nav_date": "20261003", "ann_date": "20261003"},
            public_available_at=NOW,
            values={"unit_nav": (10, "元"), "adj_nav": (10, "元"), "net_asset": (1100, "元")},
        ),
        row("fund_share", values={"fd_share": (110, "份")}),
        row(
            "fund_factor_pro",
            values={"close": (11, "元"), "amount": (100, "元"), "vol": (10, "手")},
        ),
        row(
            "slb_len_mm",
            values={
                "ope_inv": (10, "股"),
                "cls_inv": (12, "股"),
                "lent_qnt": (2, "股"),
                "end_bal": (100, "元"),
            },
        ),
        row(
            "fund_div",
            raw={
                "ann_date": "20260901",
                "ex_date": "20261001",
                "net_ex_date": "20261001",
                "div_proc": "实施",
            },
            values={"div_cash": (0.1, "元")},
            public_available_at=NOW,
        ),
        row(
            "fund_manager",
            raw={
                "name": "经理",
                "ann_date": "20260901",
                "begin_date": "20260101",
                "end_date": None,
            },
        ),
    ]
    for api in ("etf_sh_cons", "etf_sz_cons"):
        values = {"qty": (100, "股"), "cpr": (0.1, "fraction"), "rdr": (0.2, "fraction")}
        values.update(
            {
                field: (5, "元")
                for field in (("sca",) if api == "etf_sh_cons" else ("sub_cc", "red_cc"))
            }
        )
        sources.append(row(api, raw={"con_code": "S", "sub_flag": "允许"}, values=values))
    sources.append(
        row(
            "index_weight",
            entity=B,
            raw={"index_code": "I", "con_code": "S"},
            values={"weight": (1, "fraction")},
        )
    )
    for day, value in (("2026-10-02", 100), ("2026-10-03", 105)):
        sources.append(row("fund_adj", day, day=day, values={"adj_factor": (1, "ratio")}))
        sources.append(
            row(
                "index_daily",
                day,
                entity=B,
                day=day,
                raw={"ts_code": "I", "return_basis": "total_return"},
                values={"close": (value, "ratio")},
            )
        )
    return sources


def test_all_fund_interfaces_process_business_roles_with_own_evidence():
    """场景：完整ETF及全部18来源；预期：本体、基准重合、申赎代理和追踪差均可复算。"""
    result = build_etf_exposure(inputs(*full_fund_rows(), entities=(A, B)))[0]
    assert len(API_FIELDS) == 18
    assert set(API_FIELDS) <= {ref.source_api for ref in result.envelope.evidence_refs}
    assert result.metric("share_change") == 10
    assert result.metric("share_change_nav_proxy") == 100
    assert result.metric("price_nav_premium") == pytest.approx(0.1)
    assert result.metric("adjusted_price_return") == pytest.approx(0.1)
    assert result.metric("benchmark_tracking_difference") == pytest.approx(0.05)
    assert result.metric("etf_sh_cons_benchmark_observed_overlap") == 1
    assert result.metric("benchmark_observed_hhi") == 1
    assert result.metric("lending_inventory_change") == 2
    assert result.metric("active_fund_managers") == 1
    assert result.metric("same_manager_observed_etfs") == 1
    assert result.metric("portfolio_overlap") is None
    assert result.details["product_scope"] == "etf_standalone"
    assert result.details["fund_observation"]["price_status"] == "traded_observation"
    assert not result.envelope.decision_allowed


def test_no_user_portfolio_preserves_standalone_etf_observation():
    """场景：只提供ETF本体行情，没有组合或指数映射；预期：保留成交额并显式缺组合。"""
    result = build_etf_exposure(inputs(fund_quote()))[0]
    assert result.metric("fund_turnover_amount") == 100
    assert result.metric("portfolio_overlap") is None
    assert "user_portfolio_not_provided" in result.envelope.gate_codes
    assert "etf_identity_unverified" in result.envelope.gate_codes
    assert build_etf_exposure(inputs()) == ()
    assert build_etf_exposure(inputs(row("fund_basic", raw={"market": "O"}))) == ()


def test_metadata_only_and_conflicting_manager_identity_are_conservative():
    """场景：管理人全名有两个主体代码，指数代码缺失；预期：不猜管理人身份或基准。"""
    basic = row("etf_basic", raw={"mgr_name": "管理人", "etf_type": "权益"})
    companies = (
        row("fund_company", "a", entity=B, raw={"name": "管理人", "org_code": "A"}),
        row("fund_company", "b", entity=B, raw={"name": "管理人", "org_code": "B"}),
    )
    result = build_etf_exposure(inputs(basic, *companies))[0]
    assert result.metric("same_manager_observed_etfs") is None
    assert result.details["benchmark_mapping_basis"] == "unknown"
    empty_id = row("fund_company", entity=B, raw={"name": "管理人"})
    assert (
        build_etf_exposure(inputs(basic, empty_id))[0].metric("same_manager_observed_etfs") is None
    )
    assert _reference_index(inputs(row("etf_index"))) == {}


def test_price_comparisons_require_same_day_currency_and_positive_denominator():
    """场景：错日、外币、零净值和缺行；预期：相对差均未知，不默认折溢价为0。"""
    quote = fund_quote()
    for other in (
        None,
        row("fund_nav", day="2026-10-02", values={"unit_nav": (10, "元")}),
        row("fund_nav", values={"unit_nav": (10, "元", "HKD")}),
        row("fund_nav", values={"unit_nav": (0, "元")}),
    ):
        assert _relative("test", quote, "close", other, "unit_nav").value is None
    unknown = replace(quote, availability=replace(quote.availability, data_as_of=None))
    assert not _same_day(unknown, unknown)
    assert not _same_day(None, quote)
    assert _relative("test", None, "close", quote, "close").value is None


def test_unknown_units_and_inactive_prices_cannot_become_trade_feasibility():
    """场景：原始数值存在但未认证，或成交量为0；预期：交易状态未验证且涨跌停距离未知。"""
    untyped = replace(fund_quote(), numeric={})
    values, details = _fund_observations({"fund_daily": (untyped,)})
    assert values[0].value is None
    assert details["price_status"] == "price_activity_unverified"
    assert _price_limits({}).value is None
    quote = fund_quote()
    limits = row("etf_limit", values={"up_limit": (10, "元"), "down_limit": (9, "元")})
    assert _price_limits({"fund_daily": (quote,), "etf_limit": (limits,)}).value is None


def test_share_proxy_requires_two_unique_comparable_snapshots():
    """场景：缺历史、零起始份额及同日冲突；预期：申赎代理未知，不截断冲突快照计算。"""
    old, new = fund_size("old", "2026-10-02", 0), fund_size()
    for rows in (
        (),
        (new,),
        (old, new),
        (fund_size("old", "2026-10-02", 100), new, fund_size("duplicate")),
    ):
        metrics, _ = _shares(rows)
        assert metrics[0].value is None
        assert metrics[-1].value is None
    untyped = replace(
        new, numeric={name: fact for name, fact in new.numeric.items() if name != "nav"}
    )
    metrics, _ = _shares((fund_size("old", "2026-10-02", 100), untyped))
    assert metrics[0].value == 10 and metrics[-1].value is None


def test_fixed_anchor_adjustment_does_not_skip_missing_factor_or_guess_index_basis():
    """场景：缺因子、无基准口径以及乘法溢出；预期：不跳过失败端点制造收益或追踪差。"""
    old, new = fund_quote("old", "2026-10-02", 10), fund_quote()
    factor = row("fund_adj", "new", values={"adj_factor": (1, "ratio")})
    assert (
        _adjusted_return({"fund_daily": (old, new), "fund_adj": (factor,)}, ())[0][0].value is None
    )
    oldfactor = row("fund_adj", "old", day="2026-10-02", values={"adj_factor": (1, "ratio")})
    metrics, details = _adjusted_return(
        {"fund_daily": (old, new), "fund_adj": (oldfactor, factor)}, ()
    )
    assert metrics[0].value == pytest.approx(0.1) and metrics[1].value is None
    assert details["anchor_start"] == "2026-10-02"
    overflow = fund_quote("huge", close=1e308)
    bigfactor = row("fund_adj", values={"adj_factor": (1e308, "ratio")})
    assert (
        _adjusted_return({"fund_daily": (old, overflow), "fund_adj": (oldfactor, bigfactor)}, ())[
            0
        ][0].value
        is None
    )


@pytest.mark.parametrize("api", ["etf_sh_cons", "etf_sz_cons"])
def test_basket_quantity_and_cash_substitution_units_do_not_become_weights(api):
    """场景：一名成分零股数、一名正股数但现金单位缺失；预期：成分广度0.5而现金金额未知。"""
    members = tuple(
        row(
            api,
            str(index),
            raw={"con_code": str(index), "sub_flag": "允许"},
            values={"qty": (qty, "股"), "cpr": (0.1, "fraction"), "rdr": (0.2, "fraction")},
        )
        for index, qty in enumerate((0, 100))
    )
    metrics, details, codes, _ = _basket(api, members)
    assert metrics[0].value == 2 and metrics[1].value == 0.5
    assert metrics[-1].value is None
    assert codes == {"0", "1"}
    assert details["quantity_scope"] == "shares_per_security_not_portfolio_weights"
    assert _basket(api, ())[0][0].value is None
    conflict = row(api, "conflict", raw={"con_code": "0"})
    assert _basket(api, (*members, conflict))[2] == set()


@pytest.mark.parametrize(
    "weights,expected",
    [
        ((0.6, 0.4), 0.52),
        ((0.6, 0.6), None),
        ((None, 0.4), None),
        ((0, 0), None),
        ((-0.1, 0.4), None),
    ],
)
def test_weight_coverage_and_concentration_do_not_hide_invalid_denominator(weights, expected):
    """场景：正常、超额、缺失、零和负权重；预期：只有合法完整认证权重可得集中度。"""
    rows = tuple(
        row(
            "index_weight",
            str(index),
            raw={"con_code": str(index)},
            values={"weight": (value, "fraction")},
        )
        for index, value in enumerate(weights)
    )
    metrics, details, _, _ = _weights(rows)
    assert metrics[1].value == expected
    assert details["constituent_count"] == 2
    assert _weights(())[0][0].value is None


def test_weight_conflict_and_stale_basket_have_no_observed_overlap():
    """场景：权重同成分冲突，或篮子与基准错日；预期：重合度未知且冲突计数保留。"""
    member = row("index_weight", "x", raw={"con_code": "S"}, values={"weight": (1, "fraction")})
    conflict = row("index_weight", "y", raw={"con_code": "S"}, values={"weight": (1, "fraction")})
    assert _weights((member, conflict))[2] == set()
    sources = full_fund_rows()
    sources = [
        replace(source, availability=replace(source.availability, data_as_of="2026-10-02"))
        if source.source_api == "index_weight"
        else source
        for source in sources
    ]
    result = build_etf_exposure(inputs(*sources))[0]
    assert result.metric("etf_sh_cons_benchmark_observed_overlap") is None


def test_dividend_stage_and_manager_dates_do_not_create_unsupported_cash_or_performance():
    """场景：计划分红、未来除息及未知经理日期；预期：已实施事件数0、经理数未知。"""
    dividends = (
        row("fund_div", "plan", raw={"div_proc": "预案", "ex_date": "20261001"}),
        row("fund_div", "future", raw={"div_proc": "实施", "ex_date": "20261010"}),
        row("fund_div", "bad", raw={"div_proc": "实施", "ex_date": "bad"}),
    )
    managers = (row("fund_manager", raw={"name": "经理", "begin_date": "bad", "end_date": "bad"}),)
    metrics, details = _events(inputs(), {"fund_div": dividends, "fund_manager": managers})
    assert metrics[0].value == 0 and metrics[1].value is None
    assert details["manager_performance_attribution"] == "not_inferred"
    ended = row(
        "fund_manager",
        "ended",
        raw={"name": "经理", "begin_date": "20260101", "end_date": "20261001"},
    )
    assert _events(inputs(), {"fund_manager": (ended,)})[0][1].value == 0
    assert _events(inputs(), {})[0][0].value is None


def test_reference_identity_and_evidence_limits_do_not_inflate_exposure_count():
    """场景：100个指数成分且两只ETF使用同一基准；预期：每只保留完整100计数和有限结果体积。"""
    basics = (
        row("etf_basic", "a", raw={"index_code": "I"}),
        row("etf_basic", "b", entity=B, raw={"index_code": "I"}),
    )
    weights = tuple(
        row(
            "index_weight",
            str(index),
            raw={"index_code": "I", "con_code": str(index)},
            values={"weight": (0.01, "fraction")},
        )
        for index in range(100)
    )
    results = build_etf_exposure(inputs(*basics, *weights, entities=(A, B)))
    assert len(results) == 2
    for result in results:
        assert result.details["benchmark_exposure"]["constituent_count"] == 100
        assert result.metric("benchmark_observed_weight") == 1
        assert len(result.to_json().encode()) < 8 * 1024 * 1024


def test_reference_close_and_unknown_nav_publication_are_not_traded_or_published():
    """场景：价格有量额却标记参考价，净值无公告时间；预期：收益及公告后净值核对均未知。"""
    reference = fund_quote(price_type="reference")
    assert not _traded(reference)
    assert not _traded(fund_quote(market_activity_status="no_trade"))
    assert not _traded(fund_quote(price_observation_status="no_price_observed"))
    assert not _traded(None)
    sources = {
        "fund_daily": (reference,),
        "etf_share_size": (fund_size(),),
        "fund_nav": (row("fund_nav", values={"unit_nav": (10, "元")}),),
    }
    metrics, _ = _fund_observations(sources)
    assert next(item for item in metrics if item.name == "published_nav_difference").value is None
