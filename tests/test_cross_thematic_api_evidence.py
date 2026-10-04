"""P02/P05/P08/P12逐接口业务证据验收，全部输入由正式契约离线生成。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from all_products_fixture import AS_OF, write_reference

from tushare_entitlement_hub.cross_asset_research.adapter import load_inputs
from tushare_entitlement_hub.cross_asset_research.bindings import build_config, equity_config
from tushare_entitlement_hub.cross_asset_research.expectations import build_expectation_revision
from tushare_entitlement_hub.cross_asset_research.funds import build_etf_exposure
from tushare_entitlement_hub.cross_asset_research.portfolio import build_portfolio
from tushare_entitlement_hub.cross_asset_research.themes import build_theme_opportunities
from tushare_entitlement_hub.equity_research.adapter import load_inputs as equity_inputs

FUND = "510001.SH"
INDEX = "000300.SH"
STOCKS = ("000001.SZ", "000002.SZ")
DAY = "20261002"
MANAGER = "合成基金管理人"
_SCENARIO = None


def datasets():
    """构造48来源及明确目录锚点；例：ETF.index_code明确指向INDEX，篮子股数来自股单位。

    复用现有夹具审定的ths_daily与fund_portfolio百分比来源声明；其余单位由正式政策决定。
    CNY仅来自合成来源显式currency/stock_basic.curr_type；不填profit_scope或return_basis。
    """
    data = {
        "dc_index": [
            {
                "ts_code": "BK0001.DC",
                "trade_date": DAY,
                "name": "合成东财主题",
                "idx_type": "概念",
                "level": "1",
            }
        ],
        "tdx_index": [
            {
                "ts_code": "TDX1",
                "trade_date": DAY,
                "name": "合成通达信主题",
                "idx_type": "概念",
                "idx_count": 2,
            }
        ],
        "ths_index": [
            {
                "ts_code": "885001.TI",
                "name": "合成同花顺主题",
                "list_date": "20000101",
                "type": "N",
                "exchange": "A",
            }
        ],
        "index_classify": [
            {
                "index_code": "SW1",
                "parent_code": "SW0",
                "level": "L3",
                "industry_name": "合成申万行业",
                "src": "SW2021",
            }
        ],
        "dc_daily": [
            {
                "ts_code": "BK0001.DC",
                "trade_date": DAY,
                "pct_change": 2,
                "amount": 1000,
                "category": "概念",
                "currency": "CNY",
            }
        ],
        "tdx_daily": [
            {
                "ts_code": "TDX1",
                "trade_date": DAY,
                "pct_change": 2,
                "up_num": 3,
                "down_num": 1,
                "limit_up_num": 1,
                "limit_down_num": 0,
            }
        ],
        "ths_daily": [
            {
                "ts_code": "885001.TI",
                "trade_date": DAY,
                "pct_change": -2,
                "vol": 100,
                "turnover_rate": 3,
            }
        ],
        "ci_daily": [
            {
                "ts_code": "CI1",
                "trade_date": DAY,
                "close": 100,
                "pct_change": 2,
                "amount": 10,
                "currency": "CNY",
            }
        ],
        "sw_daily": [
            {
                "ts_code": "SW1",
                "trade_date": DAY,
                "pct_change": 2,
                "amount": 10,
                "pe": 20,
                "pb": 2,
                "currency": "CNY",
            }
        ],
        "kpl_list": [
            {
                "ts_code": STOCKS[0],
                "trade_date": DAY,
                "tag": "涨停",
                "theme": "合成题材",
                "lu_desc": "供应商归因",
                "status": "涨停",
            }
        ],
        "limit_cpt_list": [
            {
                "ts_code": "LIMIT1",
                "trade_date": DAY,
                "cons_nums": 10,
                "up_nums": 3,
                "pct_chg": 2,
                "rank": 1,
            }
        ],
        "moneyflow_cnt_ths": [
            {
                "ts_code": "885001.TI",
                "trade_date": DAY,
                "net_amount": 1,
                "company_num": 2,
                "currency": "CNY",
            }
        ],
        "moneyflow_ind_ths": [
            {
                "ts_code": "885001.TI",
                "trade_date": DAY,
                "industry": "合成行业",
                "net_amount": 2,
                "company_num": 2,
                "currency": "CNY",
            }
        ],
        "moneyflow_ind_dc": [
            {
                "ts_code": "BK0001.DC",
                "trade_date": DAY,
                "content_type": "行业",
                "net_amount": 300,
                "net_amount_rate": 2,
                "currency": "CNY",
            }
        ],
        "broker_recommend": [{"month": "202609", "broker": "合成机构", "ts_code": STOCKS[0]}],
        "stk_surv": [
            {
                "ts_code": STOCKS[0],
                "surv_date": "20261001",
                "rece_org": "合成接待方",
                "rece_mode": "现场",
                "content": "合成经营交流",
            }
        ],
        "etf_basic": [
            {
                "ts_code": FUND,
                "index_code": INDEX,
                "etf_type": "权益",
                "mgr_name": MANAGER,
                "mgt_fee": 0.5,
            }
        ],
        "fund_basic": [
            {
                "ts_code": FUND,
                "list_date": "20000101",
                "market": "E",
                "management": MANAGER,
                "fund_type": "股票型",
                "benchmark": "合成基准",
                "m_fee": 0.5,
                "c_fee": 0.1,
            }
        ],
        "fund_company": [
            {
                "name": MANAGER,
                "shortname": "合成管理",
                "org_code": "SYNTHETIC_ORG",
                "credit_code": "SYNTHETIC_CREDIT",
                "setup_date": "20000101",
            }
        ],
        "etf_index": [
            {
                "ts_code": INDEX,
                "pub_date": "20000101",
                "pub_party_name": "合成指数发布方",
                "adj_circle": "季度",
            }
        ],
        "mkt_idx_bmk": [
            {
                "ts_code": INDEX,
                "bmk_level": "1",
                "bmk_type": "宽基",
                "bmk_src": "合成发布方",
                "idx_type": "权益",
            }
        ],
        "etf_limit": [
            {"ts_code": FUND, "trade_date": DAY, "up_limit": 12, "down_limit": 9, "currency": "CNY"}
        ],
        "etf_share_size": [
            {
                "ts_code": FUND,
                "trade_date": day,
                "total_share": share,
                "total_size": share * 10,
                "nav": 10,
                "close": 11,
                "currency": "CNY",
            }
            for day, share in (("20261001", 1), (DAY, 2))
        ],
        "fund_daily": [
            {
                "ts_code": FUND,
                "trade_date": day,
                "close": close,
                "amount": 100,
                "vol": 10,
                "currency": "CNY",
            }
            for day, close in (("20261001", 10), (DAY, 11))
        ],
        "fund_adj": [
            {"ts_code": FUND, "trade_date": day, "adj_factor": 1} for day in ("20261001", DAY)
        ],
        "fund_div": [
            {
                "ts_code": FUND,
                "ann_date": "20261001",
                "base_date": "20260930",
                "ex_date": DAY,
                "div_cash": 0.1,
                "div_proc": "实施",
                "net_ex_date": DAY,
                "currency": "CNY",
            }
        ],
        "fund_factor_pro": [
            {
                "ts_code": FUND,
                "trade_date": DAY,
                "close": 11,
                "amount": 100,
                "vol": 10,
                "currency": "CNY",
            }
        ],
        "fund_manager": [
            {"ts_code": FUND, "name": "合成经理", "ann_date": "20261001", "begin_date": "20260101"}
        ],
        "fund_nav": [
            {
                "ts_code": FUND,
                "ann_date": DAY,
                "nav_date": DAY,
                "unit_nav": 10,
                "adj_nav": 10,
                "net_asset": 200000,
                "currency": "CNY",
            }
        ],
        "fund_share": [{"ts_code": FUND, "trade_date": DAY, "fd_share": 2}],
        "slb_len_mm": [
            {
                "ts_code": FUND,
                "trade_date": DAY,
                "ope_inv": 1,
                "lent_qnt": 0.2,
                "cls_inv": 1.2,
                "end_bal": 12,
                "currency": "CNY",
            }
        ],
        "fund_portfolio": [
            {
                "ts_code": FUND,
                "ann_date": "20261001",
                "end_date": "20260930",
                "symbol": STOCKS[0],
                "mkv": 100,
                "amount": 10,
                "stk_mkv_ratio": 60,
            }
        ],
    }
    for api, code in (
        ("dc_member", "BK0001.DC"),
        ("tdx_member", "TDX1"),
        ("kpl_concept_cons", "KP1"),
    ):
        data[api] = [
            {
                "ts_code": code,
                "con_code": stock,
                "trade_date": DAY,
                "desc": "合成成员归因",
                "hot_num": 3,
            }
            for stock in STOCKS
        ]
    data["ths_member"] = [
        {"ts_code": "885001.TI", "con_code": stock, "weight": 50, "in_date": "20200101"}
        for stock in STOCKS
    ]
    for api, code in (("ci_index_member", "CI1"), ("index_member_all", "SW1")):
        data[api] = [
            {
                "l1_code": "L1",
                "l2_code": "L2",
                "l3_code": code,
                "l3_name": "合成行业",
                "ts_code": stock,
                "in_date": "20200101",
            }
            for stock in STOCKS
        ]
    for api in ("dc_hot", "ths_hot"):
        data[api] = [
            {
                "ts_code": STOCKS[0],
                "trade_date": DAY,
                "data_type": "热股",
                "rank_time": "15:00:00",
                "rank": 2,
                "rank_reason": "合成关注理由",
                "hot": 100,
            }
        ]
    for api in ("forecast", "forecast_vip"):
        data[api] = [
            {
                "ts_code": STOCKS[0],
                "ann_date": "20260920",
                "first_ann_date": "20260920",
                "end_date": "20260930",
                "type": "预增",
                "net_profit_min": 10,
                "net_profit_max": 20,
                "change_reason": "成本改善",
            }
        ]
    data["forecast_vip"].append(
        {
            "ts_code": STOCKS[0],
            "ann_date": "20260910",
            "first_ann_date": "20260910",
            "end_date": "20260930",
            "type": "预增",
            "net_profit_min": 8,
            "net_profit_max": 10,
            "change_reason": "先前预告",
        }
    )
    for api in ("express", "express_vip"):
        data[api] = [
            {
                "ts_code": STOCKS[0],
                "ann_date": "20261001",
                "end_date": "20260930",
                "n_income": 180000,
                "revenue": 2000000,
                "is_audit": 1,
            }
        ]
    data["report_rc"] = [
        {
            "ts_code": STOCKS[0],
            "report_date": date,
            "org_name": "合成机构",
            "author_name": "合成作者",
            "quarter": "2026Q4",
            "report_title": title,
            "np": value,
            "eps": value / 10,
            "create_time": date + " 10:00:00",
        }
        for date, title, value in (("20260910", "前次预测", 10), ("20260920", "更新预测", 20))
    ]
    for api in ("etf_sh_cons", "etf_sz_cons"):
        data[api] = [
            {
                "ts_code": FUND,
                "trade_date": DAY,
                "con_code": stock,
                "qty": quantity,
                "sub_flag": "允许",
                "cpr": 10,
                "rdr": 20,
                "sca": 5,
                "sub_cc": 5,
                "red_cc": 6,
            }
            for stock, quantity in zip(STOCKS, (0, 100), strict=True)
        ]
    data["index_weight"] = [
        {"index_code": INDEX, "con_code": stock, "trade_date": DAY, "weight": 50}
        for stock in STOCKS
    ]
    return data


# 每项均为具名业务指标或明确业务状态，不把来源计数、非空metrics作为验收替代。
CASES = [
    ("P02", "dc_daily", "metric", "dc_daily_amount", 1000, None, ("amount", "trade_date")),
    (
        "P02",
        "dc_hot",
        "metric",
        "dc_hot_inverse_rank_0",
        None,
        "source_rank_unverified",
        ("rank", "rank_time", "data_type"),
    ),
    (
        "P02",
        "dc_index",
        "metric",
        "dc_index_catalog_consistent",
        1,
        None,
        ("ts_code", "name", "idx_type", "level"),
    ),
    (
        "P02",
        "dc_member",
        "metric",
        "dc_member_observed_members",
        2,
        None,
        ("ts_code", "con_code", "trade_date"),
    ),
    (
        "P02",
        "kpl_concept_cons",
        "metric",
        "kpl_concept_cons_observed_members",
        2,
        None,
        ("con_code", "desc", "hot_num"),
    ),
    (
        "P02",
        "kpl_list",
        "metric",
        "kpl_list_observed_events",
        1,
        None,
        ("tag", "theme", "lu_desc", "status"),
    ),
    (
        "P02",
        "limit_cpt_list",
        "metric",
        "limit_cpt_list_limit_up_breadth",
        None,
        "breadth_denominator_unverified",
        ("cons_nums", "up_nums"),
    ),
    (
        "P02",
        "tdx_daily",
        "metric",
        "tdx_daily_directional_breadth",
        None,
        "breadth_denominator_unverified",
        ("up_num", "down_num"),
    ),
    (
        "P02",
        "tdx_index",
        "metric",
        "tdx_index_catalog_consistent",
        1,
        None,
        ("ts_code", "name", "idx_type", "idx_count"),
    ),
    (
        "P02",
        "tdx_member",
        "metric",
        "tdx_member_observed_members",
        2,
        None,
        ("ts_code", "con_code", "trade_date"),
    ),
    (
        "P02",
        "ths_daily",
        "metric",
        "ths_daily_pct_change",
        -0.02,
        None,
        ("pct_change", "trade_date"),
    ),
    (
        "P02",
        "ths_hot",
        "metric",
        "ths_hot_inverse_rank_0",
        None,
        "source_rank_unverified",
        ("rank", "rank_time", "rank_reason"),
    ),
    (
        "P02",
        "ths_index",
        "metric",
        "ths_index_catalog_consistent",
        1,
        None,
        ("ts_code", "name", "type", "exchange"),
    ),
    (
        "P02",
        "ths_member",
        "metric",
        "ths_member_observed_members",
        2,
        None,
        ("ts_code", "con_code", "in_date"),
    ),
    ("P02", "ci_daily", "metric", "ci_daily_amount", 100000, None, ("amount", "trade_date")),
    (
        "P02",
        "ci_index_member",
        "metric",
        "ci_index_member_observed_members",
        2,
        None,
        ("l1_code", "l2_code", "l3_code", "ts_code", "in_date"),
    ),
    (
        "P02",
        "index_classify",
        "metric",
        "index_classify_catalog_consistent",
        1,
        None,
        ("index_code", "parent_code", "industry_name", "src"),
    ),
    (
        "P02",
        "index_member_all",
        "metric",
        "index_member_all_observed_members",
        2,
        None,
        ("l1_code", "l2_code", "l3_code", "ts_code"),
    ),
    (
        "P02",
        "sw_daily",
        "metric",
        "sw_daily_amount",
        100000,
        None,
        ("amount", "trade_date", "pe", "pb"),
    ),
    (
        "P02",
        "moneyflow_cnt_ths",
        "metric",
        "moneyflow_cnt_ths_net_amount",
        100000000,
        None,
        ("net_amount", "company_num"),
    ),
    (
        "P02",
        "moneyflow_ind_dc",
        "metric",
        "moneyflow_ind_dc_net_amount",
        300,
        None,
        ("net_amount", "content_type"),
    ),
    (
        "P02",
        "moneyflow_ind_ths",
        "metric",
        "moneyflow_ind_ths_net_amount",
        200000000,
        None,
        ("net_amount", "industry"),
    ),
    (
        "P05",
        "forecast",
        "metric",
        "forecast_economic_events",
        2,
        None,
        ("ann_date", "end_date", "type", "net_profit_min", "net_profit_max"),
    ),
    (
        "P05",
        "forecast_vip",
        "metric",
        "forecast_profit_midpoint",
        150000,
        None,
        ("net_profit_min", "net_profit_max", "change_reason"),
    ),
    (
        "P05",
        "express",
        "metric",
        "express_economic_events",
        1,
        None,
        ("ann_date", "end_date", "n_income", "is_audit"),
    ),
    (
        "P05",
        "express_vip",
        "metric",
        "express_net_income",
        180000,
        None,
        ("n_income", "revenue", "is_audit"),
    ),
    (
        "P05",
        "report_rc",
        "suffix",
        "_np_revision",
        100000,
        None,
        ("report_date", "org_name", "quarter", "np", "create_time"),
    ),
    (
        "P05",
        "broker_recommend",
        "metric",
        "broker_recommend_observed_coverage",
        1,
        None,
        ("month", "broker", "ts_code"),
    ),
    (
        "P05",
        "stk_surv",
        "metric",
        "stk_surv_observed_coverage",
        1,
        None,
        ("surv_date", "rece_org", "rece_mode", "content"),
    ),
    (
        "P08",
        "etf_basic",
        "metric",
        "same_manager_observed_etfs",
        1,
        None,
        ("ts_code", "index_code", "mgr_name"),
    ),
    (
        "P08",
        "etf_index",
        "state",
        ("metadata", "etf_index", "pub_party_name", "samples"),
        ["合成指数发布方"],
        None,
        ("ts_code", "pub_party_name", "adj_circle"),
    ),
    (
        "P08",
        "etf_limit",
        "metric",
        "price_limit_distance",
        None,
        "fund_trade_or_limit_unverified",
        ("up_limit", "down_limit", "trade_date"),
    ),
    (
        "P08",
        "etf_sh_cons",
        "metric",
        "etf_sh_cons_positive_quantity_fraction",
        0.5,
        None,
        ("con_code", "qty", "sub_flag", "cpr", "rdr", "sca"),
    ),
    (
        "P08",
        "etf_sz_cons",
        "metric",
        "etf_sz_cons_positive_quantity_fraction",
        0.5,
        None,
        ("con_code", "qty", "sub_flag", "cpr", "rdr", "sub_cc", "red_cc"),
    ),
    (
        "P08",
        "etf_share_size",
        "metric",
        "share_change",
        10000,
        None,
        ("total_share", "trade_date", "nav", "total_size"),
    ),
    (
        "P08",
        "fund_adj",
        "metric",
        "adjusted_price_return",
        None,
        "fixed_anchor_adjustment_unavailable",
        ("adj_factor", "trade_date"),
    ),
    (
        "P08",
        "fund_basic",
        "state",
        ("metadata", "fund_basic", "market", "samples"),
        ["E"],
        None,
        ("management", "fund_type", "market", "benchmark"),
    ),
    (
        "P08",
        "fund_company",
        "metric",
        "same_manager_observed_etfs",
        1,
        None,
        ("name", "shortname", "org_code", "credit_code"),
    ),
    (
        "P08",
        "fund_daily",
        "metric",
        "fund_turnover_amount",
        100000,
        None,
        ("amount", "vol", "close", "trade_date"),
    ),
    (
        "P08",
        "fund_div",
        "metric",
        "implemented_dividend_events",
        1,
        None,
        ("ann_date", "ex_date", "div_proc", "net_ex_date"),
    ),
    (
        "P08",
        "fund_factor_pro",
        "metric",
        "provider_price_difference",
        None,
        "aligned_comparable_values_unavailable",
        ("close", "amount", "vol"),
    ),
    (
        "P08",
        "fund_manager",
        "metric",
        "active_fund_managers",
        1,
        None,
        ("name", "ann_date", "begin_date"),
    ),
    (
        "P08",
        "fund_nav",
        "metric",
        "published_nav_difference",
        None,
        "aligned_comparable_values_unavailable",
        ("ann_date", "nav_date", "unit_nav", "adj_nav"),
    ),
    (
        "P08",
        "fund_share",
        "metric",
        "published_share_difference",
        None,
        "aligned_comparable_values_unavailable",
        ("trade_date", "fd_share"),
    ),
    (
        "P08",
        "mkt_idx_bmk",
        "state",
        ("metadata", "mkt_idx_bmk", "bmk_type", "samples"),
        ["宽基"],
        None,
        ("ts_code", "bmk_level", "bmk_type", "bmk_src", "idx_type"),
    ),
    (
        "P08",
        "index_weight",
        "metric",
        "benchmark_observed_hhi",
        None,
        "benchmark_weights_unverified",
        ("index_code", "con_code", "weight", "trade_date"),
    ),
    (
        "P08",
        "slb_len_mm",
        "metric",
        "lending_inventory_change",
        2000,
        None,
        ("ope_inv", "cls_inv", "lent_qnt", "end_bal"),
    ),
    (
        "P12",
        "fund_portfolio",
        "metric",
        "holding_hhi",
        0.36,
        None,
        ("ann_date", "end_date", "symbol", "mkv", "stk_mkv_ratio"),
    ),
]


@pytest.fixture
def thematic_api_scenario(tmp_path_factory, deny_network):
    """在禁网fixture生效后构建一次真实契约合成库，后续48项只读复用不可变输入与结果。

    例如fund_portfolio百分数60经既有声明转为0.6；未知adj_factor单位保留未知。
    """
    global _SCENARIO
    if _SCENARIO is None:
        reference = write_reference(tmp_path_factory.mktemp("thematic-api-evidence"), datasets())
        config = build_config(reference, AS_OF).config
        old, _ = equity_inputs(reference, AS_OF, "thematic-api-evidence", equity_config(config))
        new, _ = load_inputs(reference, AS_OF, "thematic-api-evidence", config)
        inputs = replace(
            new,
            rows_by_api={**old.rows_by_api, **new.rows_by_api},
            source_reasons={**old.source_reasons, **new.source_reasons},
            entity_source_reasons={**old.entity_source_reasons, **new.entity_source_reasons},
        )
        results = {
            "P02": build_theme_opportunities(inputs),
            "P05": build_expectation_revision(inputs),
            "P08": build_etf_exposure(inputs),
            "P12": build_portfolio(inputs),
        }
        _SCENARIO = inputs, results
    return _SCENARIO


@pytest.mark.parametrize(
    "product,api,kind,target,expected,reason,fields", CASES, ids=[case[1] for case in CASES]
)
def test_each_thematic_api_has_business_result_and_exact_source_evidence(
    thematic_api_scenario, product, api, kind, target, expected, reason, fields
):
    """场景：逐API真实契约、显式关系锚点与合法单位；预期：具体业务结果/未知原因及逐行字段证据一致。

    每个参数项均指定具名业务指标或语义路径，不以API被执行、非空metrics或来源集合替代断言。
    """
    inputs, products = thematic_api_scenario
    rows = inputs.rows(api)
    assert rows, (api, inputs.source_reasons.get(api))
    candidates = [
        result
        for result in products[product]
        if any(ref.source_api == api for ref in result.envelope.evidence_refs)
    ]
    assert len(candidates) == 1, (api, [item.details for item in products[product]])
    result = candidates[0]
    if kind == "state":
        actual = result.details
        for part in target:
            actual = actual[part]
        assert actual == expected
    else:
        metrics = [
            item
            for item in result.metrics
            if item.name == target or kind == "suffix" and item.name.endswith(target)
        ]
        assert len(metrics) == 1, (api, target, [item.name for item in result.metrics])
        value = metrics[0]
        if expected is None:
            assert value.value is None and value.status == "unknown"
            assert reason in value.reason_codes
        else:
            assert value.status == "available"
            assert value.value == pytest.approx(expected)
    # 每个小型夹具的全部记录都须定位到本API、本source_record_id及实际非空字段。
    for observed in rows:
        record_id = observed.raw["source_record_id"]
        own = [
            ref
            for ref in result.envelope.evidence_refs
            if ref.source_api == api and ref.source_record_id == record_id
        ]
        assert own, (api, record_id)
        evidence_fields = {field for reference in own for field in reference.fields}
        assert set(fields) <= evidence_fields
        assert all(observed.text(field) is not None for field in fields)
        assert all(reference.contract_hash == observed.raw["contract_hash"] for reference in own)
    assert result.envelope.historical_pit_certified is False
    assert not result.envelope.decision_allowed
