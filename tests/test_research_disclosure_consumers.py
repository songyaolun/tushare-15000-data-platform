"""财务、治理与转债新增消费的离线合成用例，不读取令牌、不访问供应商。"""

from dataclasses import replace

import pytest
from test_research_governance import BOND, DAY, A, B, inputs, relation, row

from tushare_entitlement_hub.equity_research.disclosure_consumers import (
    BOND_APIS,
    DISCLOSURE_FIELDS,
    FINANCIAL_APIS,
    _comparison,
    _conversion_value,
    _current,
    _date,
    _ratio_value,
    _statement_for,
    _value,
    consume_disclosure,
)
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.product_foundation.contracts import Truth


def source(api, record="source", *, raw=None, values=None, entity=A, **times):
    """构造已观察且有认证数值的合成来源；例：pb=(2,'ratio')创建可核对市净率2。"""
    return row(
        api,
        record,
        entity,
        values,
        {
            "quality_tier": "verified",
            "ts_code": entity.split(":")[2] if entity else "UNMAPPED",
            **(raw or {}),
        },
        **times,
    )


def consume(api, *sources, context=None, entity=A):
    """通过产品分配接口调用处理器，便于检查真实返回契约；例：空bak_basic输出unknown。"""
    context = context or inputs(*sources)
    result = consume_disclosure(
        context,
        "P04" if api in FINANCIAL_APIS else "P06",
        api,
        tuple(item for item in sources if item.source_api == api),
        entity,
    )
    assert all(not check.hard for check in result["gates"])
    assert result["facts"]["independent_evidence"] is False
    return result


def observation(result):
    """取一个证券的业务事实，保留测试对分组结果的检查；例：单股消费只含一项observations。"""
    assert len(result["facts"]["observations"]) == 1
    return result["facts"]["observations"][0]


def financial_sources(*, scope=True, liability=40, comp_type="1"):
    """创建同报告期指标、负债表及现金利润口径；例：40/100杠杆与20/10现金利润均可核对。"""
    context = {"ann_date": "20260830", "end_date": "20260630", "update_flag": "0"}
    if scope:
        context.update(report_type="1", comp_type=comp_type)
    indicator = {
        "roe": (0.1, "ratio"),
        "debt_to_assets": (0.4, "ratio"),
        "ocf_to_profit": (2, "ratio"),
    }
    return (
        source("fina_indicator", raw=context, values=indicator),
        source("fina_indicator_vip", raw=context, values=indicator),
        source(
            "balancesheet_vip",
            raw=context,
            values={"total_liab": (liability, "元"), "total_assets": (100, "元")},
        ),
        source("income_vip", raw=context, values={"operate_profit": (10, "元")}),
        source("cashflow_vip", raw=context, values={"n_cashflow_act": (20, "元")}),
    )


def certified_context(*sources):
    """认证合成报表类型，保持无网络的显式范围；例：report_type=1被允许做本地非金融比率。"""
    context = inputs(*sources)
    return replace(context, config=replace(context.config, certified_report_types=("1",)))


def bond_context(*sources, kind="CB", terms=None, stock=None, confirmed=True):
    """建立可追溯债券关系、面值与同日正股价；例：100元面值/10元转股价×10元股价=100元。"""
    basic = terms or source(
        "cb_basic",
        "terms",
        entity=BOND,
        raw={
            "cb_type": kind,
            "stk_code": "A",
            "conv_start_date": "20260101",
            "conv_end_date": "20271231",
        },
        values={"remain_size": (1000, "元"), "par": (100, "元"), "conv_price": (10, "元")},
    )
    equity = stock or source(
        "daily", "stock", values={"close": (10, "元"), "vol": (100, "手"), "amount": (1000, "元")}
    )
    linked = replace(relation(confirmed=confirmed), evidence_refs=basic.evidence("cb_type"))
    return inputs(*sources, basic, equity, entities=(A, BOND), relations=(linked,))


@pytest.mark.parametrize("api", tuple(DISCLOSURE_FIELDS))
def test_every_assigned_source_has_an_explicit_empty_status(api):
    """场景：十五个新增来源均无数据；预期：逐来源unknown和明确缺口，不能伪称已消费。"""
    result = consume(api)
    assert result["status"] == "unknown"
    assert "source_rows_missing" in result["reason_codes"]
    assert result["evidence_refs"] == ()


@pytest.mark.parametrize(
    "raw,times",
    [
        ({"is_quarantined": True}, {}),
        ({"quality_tier": "quarantine"}, {}),
        ({"quality_tier": "unknown"}, {}),
        ({"ann_date": "20261004"}, {}),
        ({"ann_date": "bad"}, {}),
        ({}, {"public_available_at": "2026-10-04T00:00:00Z"}),
    ],
)
def test_rejected_disclosures_never_contribute_evidence(raw, times):
    """场景：隔离、未来公告或不可见观察绕过适配层直接调用；预期：拒绝引用且保留unknown。"""
    record = source("stk_managers", raw=raw, **times)
    result = consume("stk_managers", record)
    assert "disclosure_unavailable_rows_rejected" in result["reason_codes"]
    assert result["evidence_refs"] == ()


def test_product_source_and_entity_mismatches_are_explicit():
    """场景：错产品、错API或错股票身份调用；预期：无法绑定不伪造归因，仍可保留独立来源事实。"""
    assert consume_disclosure(inputs(), "P06", "bak_basic", (), A)["status"] == "unknown"
    assert consume_disclosure(inputs(), "P04", "unassigned", (), A)["status"] == "unknown"
    record = source(
        "stk_managers", raw={"name": "甲", "title": "经理", "begin_date": "20260101"}, entity=B
    )
    result = consume("stk_managers", record)
    assert "disclosure_entity_binding_unverified" in result["reason_codes"]
    assert observation(result)["source_entity_id"] == B
    rejected = consume_disclosure(inputs(record), "P04", "bak_basic", (record,), A)
    assert rejected["evidence_refs"] == ()


def test_numeric_helpers_never_guess_unknown_units_or_currency():
    """场景：原始数值、错币种、缺日期及同日版本冲突；预期：认证读取与版本选择均失败关闭。"""
    record = source(
        "bak_basic",
        raw={"end_date": "bad", "pb": 3},
        values={"total_assets": (100, "元", "USD"), "roe": (40, "%"), "float_share": (1, "亿股")},
    )
    assert _date(record, "end_date") is None
    assert _value(record, "pb", "ratio") is None
    assert _value(record, "total_assets", "元") is None
    assert _value(None, "field", "元") is None
    assert _ratio_value(record, "roe") == 0.4
    assert _ratio_value(record, "total_assets") is None
    assert _ratio_value(record, "float_share") is None
    assert _ratio_value(None, "field") is None
    assert _current((), "end_date") is None
    assert _current((record,), "end_date") is None
    assert _current((record, source("bak_basic", "other")), "trade_date") is None
    assert _comparison(None, 0)["agreement"] is None


def test_backup_uses_certified_same_day_capital_and_preserves_pe_basis():
    """场景：备份股本和市净率与当日底表一致；预期：可核对二者，动态PE和无报告期资产不强比。"""
    backup = source(
        "bak_basic",
        values={
            "pb": (2, "ratio"),
            "pe": (10, "ratio"),
            "float_share": (1000, "股"),
            "total_assets": (5000, "元"),
        },
    )
    daily = source("daily_basic", values={"pb": (2, "ratio"), "float_share": (1000, "股")})
    result = consume("bak_basic", backup, daily)
    facts = observation(result)
    assert result["status"] == "observed"
    assert facts["comparisons"]["float_share"]["agreement"] is True
    assert facts["pe_comparison_status"] == "unknown"
    assert facts["assets_comparison_status"] == "unknown"
    assert {ref.source_api for ref in result["evidence_refs"]} == {"bak_basic", "daily_basic"}


@pytest.mark.parametrize(
    "shares,peer_day,status",
    [("亿股", DAY, "unknown"), ("股", "20261002", "unknown"), ("股", DAY, "conflict")],
)
def test_backup_rejects_billion_unit_date_mismatch_and_contradiction(shares, peer_day, status):
    """场景：亿股未换算、异日旁证或认证后数值不同；预期：分别unknown、unknown、conflict。"""
    backup = source("bak_basic", values={"pb": (2, "ratio"), "float_share": (1, shares)})
    daily = source(
        "daily_basic",
        raw={"trade_date": peer_day},
        values={"pb": (2, "ratio"), "float_share": (1000, "股")},
    )
    assert consume("bak_basic", backup, daily)["status"] == status


def test_indicator_schema_gap_is_visible_without_losing_provider_facts():
    """场景：指标结构无report_type和comp_type；预期：保留当前值与口径缺口。

    例：供应商资产负债率0.4仍可展示，但缺少合并范围时本地比率核对保持未知。
    """
    records = financial_sources(scope=False)
    result = consume("fina_indicator_vip", *records, context=certified_context(*records))
    facts = observation(result)
    assert result["status"] == "unknown"
    assert "indicator_report_scope_schema_gap" in result["reason_codes"]
    assert facts["provider_ratios"]["debt_to_assets"] == 0.4
    assert facts["local_ratio_checks"]["debt_to_assets"]["right"] is None


def test_indicator_local_and_duplicate_checks_require_complete_context():
    """场景：显式完整合成口径及单位与版本一致；预期：杠杆0.4、现金利润比2可核对，VIP不加分。"""
    records = financial_sources()
    for api in ("fina_indicator", "fina_indicator_vip"):
        result = consume(api, *records, context=certified_context(*records))
        facts = observation(result)
        assert result["status"] == "observed"
        assert facts["ordinary_vip_checks"]["roe"]["agreement"] is True
        assert facts["local_ratio_checks"]["debt_to_assets"]["right"] == 0.4
        assert facts["scoring_policy"] == "corroboration_only_no_duplicate_score"
        assert facts["roe_local_comparison_reason"] == "average_equity_basis_unavailable"
    assert facts["local_ratio_checks"]["ocf_to_profit"]["right"] == 2


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"report_type": "2"}, "indicator_duplicate_scope_or_version_unverified"),
        ({"comp_type": "2"}, "indicator_duplicate_scope_or_version_unverified"),
        ({"update_flag": "1"}, "indicator_duplicate_scope_or_version_unverified"),
        ({"ann_date": "20260901"}, "indicator_duplicate_scope_or_version_unverified"),
    ],
)
def test_indicator_ordinary_vip_scope_and_revision_mismatch_are_not_deduplicated(change, reason):
    """场景：普通版和VIP合并范围、公司类型或修订不同；预期：不据同数字判一致，版本缺口明确。"""
    records = list(financial_sources())
    records[0] = replace(records[0], raw=records[0].raw | change)
    result = consume("fina_indicator_vip", *records, context=certified_context(*records))
    assert reason in result["reason_codes"]
    assert observation(result)["ordinary_vip_checks"]["roe"]["agreement"] is None


def test_indicator_conflicts_and_financial_industry_do_not_override_core_score():
    """场景：供应商杠杆与财报不同或金融公司不适用；预期：旁证冲突为软门禁，金融本地比率未知。"""
    records = financial_sources(liability=80)
    result = consume("fina_indicator_vip", *records, context=certified_context(*records))
    assert result["status"] == "conflict"
    assert result["gates"][0].result is Truth.FALSE
    financial = financial_sources(comp_type="2")
    result = consume("fina_indicator", *financial, context=certified_context(*financial))
    assert observation(result)["industry_scope"] == "financial"
    assert observation(result)["local_ratio_checks"]["debt_to_assets"]["right"] is None


def test_statement_comparison_preserves_missing_units_and_same_time_conflicts():
    """场景：相同口径报表普通VIP数值不同或同刻多版；预期：本地核对不任选一份，保留引用。"""
    records = financial_sources()
    ordinary = source(
        "balancesheet",
        values={"total_liab": (90, "元"), "total_assets": (100, "元")},
        raw=records[2].raw
        | {
            "source_api": "balancesheet",
            "source_record_id": "ordinary",
            "business_group_key": "ordinary",
        },
        record="ordinary",
    )
    chosen, refs = _statement_for(
        certified_context(*records, ordinary),
        records[1],
        "balancesheet",
        ("total_liab", "total_assets"),
    )
    assert chosen is None and refs
    conflict = replace(records[2], raw=records[2].raw | {"business_variant_key": "candidate"})
    chosen, _ = _statement_for(
        certified_context(records[1], conflict), records[1], "balancesheet", ("total_assets",)
    )
    assert chosen is None


@pytest.mark.parametrize(
    "result_day,stage,actual",
    [(None, "planned", None), ("20261010", "planned", None), ("20261002", "implemented", 900)],
)
def test_cb_issuance_separates_plan_and_public_result(result_day, stage, actual):
    """场景：发行结果未知、未来或已公告；预期：计划1000元始终独立，只有已公告结果900元算实施。"""
    record = source(
        "cb_issue",
        entity=BOND,
        raw={"ann_date": "20261001", "res_ann_date": result_day},
        values={"plan_issue_size": (1000, "元"), "issue_size": (900, "元")},
    )
    result = consume("cb_issue", record, context=bond_context(record))
    facts = observation(result)
    assert facts["stage"] == stage
    assert facts["planned_amount"] == 1000
    assert facts["implemented_amount"] == actual


@pytest.mark.parametrize(
    "ipo,listing,stage",
    [
        ("20261010", None, "planned"),
        ("20261001", "20261010", "subscription_started"),
        ("20260901", "20260910", "listed"),
    ],
)
def test_new_listing_supply_has_its_own_lifecycle(ipo, listing, stage):
    """场景：新股待申购、已申购或已上市；预期：分离阶段与股数，不混成存量股票稀释。"""
    record = source(
        "new_share",
        raw={"ipo_date": ipo, "issue_date": listing},
        values={"amount": (1000, "股"), "price": (10, "元")},
    )
    facts = observation(consume("new_share", record))
    assert facts["stage"] == stage
    assert facts["equity_dilution_ratio"] is None
    assert facts["listed_supply_shares"] == (1000 if stage == "listed" else None)


def test_issuance_unknown_dates_units_and_reversed_dates_remain_unknown():
    """场景：缺发行日期、金额单位或上市早于申购；预期：unknown且不给精确新增股数。"""
    assert consume("cb_issue", source("cb_issue", entity=BOND))["status"] == "unknown"
    record = source(
        "new_share",
        raw={"ipo_date": "20261001", "issue_date": "20260901"},
        values={"amount": (1, "万股"), "price": (-1, "元")},
    )
    result = consume("new_share", record)
    assert "ipo_date_order_invalid" in result["reason_codes"]
    assert observation(result)["listed_supply_shares"] is None


def test_coupon_period_scenario_retains_active_past_and_future_periods():
    """场景：过去、当前和未来三段票息；预期：各段独立，以1000元×2%÷2次形成10元情景。"""
    records = tuple(
        source(
            "cb_rate",
            str(index),
            entity=BOND,
            raw={"rate_start_date": start, "rate_end_date": end},
            values={"coupon_rate": (2, "%"), "rate_freq": (2, "次/年")},
        )
        for index, (start, end) in enumerate(
            (("20250101", "20251231"), ("20260101", "20261231"), ("20270101", "20271231"))
        )
    )
    result = consume("cb_rate", *records, context=bond_context(*records))
    facts = observation(result)
    assert [period["period_state"] for period in facts["periods"]] == [
        "completed",
        "current",
        "planned",
    ]
    assert all(period["payment_scenario_amount"] == 10 for period in facts["periods"])
    assert facts["accrued_interest"] is None


@pytest.mark.parametrize(
    "patch,frequency",
    [({"rate_start_date": None}, 2), ({"rate_start_date": "20270101"}, 2), ({}, 0), ({}, 1.5)],
)
def test_coupon_requires_valid_interval_and_certified_frequency(patch, frequency):
    """场景：票息区间缺失、颠倒或频率非法；预期：金额情景未知，禁止仅凭票面利率算支出。"""
    record = source(
        "cb_rate",
        entity=BOND,
        raw={"rate_start_date": "20260101", "rate_end_date": "20261231"} | patch,
        values={"coupon_rate": (0.02, "ratio"), "rate_freq": (frequency, "次/年")},
    )
    result = consume("cb_rate", record, context=bond_context(record))
    assert observation(result)["periods"][0]["payment_scenario_amount"] is None


def test_overlapping_and_duplicate_coupon_periods_do_not_sum():
    """场景：两个当前票息期间重叠或业务键重复；预期：所有情景金额置未知，不重复计票息。"""
    record = source(
        "cb_rate",
        entity=BOND,
        raw={"rate_start_date": "20260101", "rate_end_date": "20261231"},
        values={"coupon_rate": (0.02, "ratio"), "rate_freq": (2, "次/年")},
    )
    second = replace(
        record,
        raw=record.raw | {"source_record_id": "second", "business_group_key": "second"},
        availability=replace(record.availability, source_record_id="second"),
    )
    result = consume("cb_rate", record, second, context=bond_context(record, second))
    assert "overlapping_current_coupon_periods" in result["reason_codes"]
    assert all(item["payment_scenario_amount"] is None for item in observation(result)["periods"])


def test_ratings_compare_only_the_same_agency_and_method():
    """场景：同机构两期评级加另一机构评级；预期：前者保留变化，后者仅当前状态且不跨机构排位。"""
    records = tuple(
        source(
            "cb_rating",
            str(index),
            entity=BOND,
            raw={
                "ann_date": day,
                "rating_date": day,
                "rating_com_name": agency,
                "rating_type": "债项",
                "rating": rating,
                "rating_outlook": outlook,
            },
        )
        for index, (day, agency, rating, outlook) in enumerate(
            (
                ("20260601", "机构甲", "AAA", "稳定"),
                ("20260901", "机构甲", "AA+", "负面"),
                ("20260901", "机构乙", "AA", "稳定"),
            )
        )
    )
    result = consume("cb_rating", *records, context=bond_context(*records))
    facts = observation(result)["agency_ratings"]
    current = next(item for item in facts if item["agency"] == "机构甲")
    assert current["previous_rating"] == "AAA"
    assert current["rating_changed"] and current["outlook_changed"]
    assert (
        next(item for item in facts if item["agency"] == "机构乙")["change_status"]
        == "not_enabled_due_to_history"
    )


def test_unmapped_bonds_keep_independent_current_ratings_without_stock_attribution():
    """场景：无股票范围且两只债券关系未认证；预期：保存两个独立评级，状态unknown，来源不消失。"""
    records = tuple(
        source(
            "cb_rating",
            str(index),
            entity=None,
            raw={
                "ts_code": code,
                "ann_date": "20260901",
                "rating_date": "20260901",
                "rating_com_name": "机构甲",
                "rating_type": "债项",
                "rating": "AAA",
            },
        )
        for index, code in enumerate(("BOND1", "BOND2"))
    )
    result = consume(
        "cb_rating", *records, context=inputs(*records, entities=()), entity=MARKET_ENTITY
    )
    assert result["status"] == "unknown"
    assert len(result["facts"]["observations"]) == 2
    assert result["evidence_refs"]
    assert all(item["underlying_entity_id"] is None for item in result["facts"]["observations"])


@pytest.mark.parametrize(
    "api,field,scope",
    [
        ("top10_holders", "hold_ratio", "total_shares"),
        ("top10_floatholders", "hold_float_ratio", "floating_shares"),
        ("top10_cb_holders", "hold_ratio", "bond_outstanding"),
    ],
)
def test_holder_concentration_has_explicit_denominator_and_partial_coverage(api, field, scope):
    """场景：当前两名持有人占20%和10%，另有旧期；预期：当前集中度0.3，债券不归成股票持有人。"""
    entity = BOND if api in BOND_APIS else A
    records = tuple(
        source(
            api,
            str(index),
            entity=entity,
            raw={"end_date": period, "holder_name": name, "holder_rank": str(index + 1)},
            values={field: (value, "%"), "hold_amount": (100, "股"), "hold_change": (10, "股")},
        )
        for index, (period, name, value) in enumerate(
            (("20260630", "甲", 20), ("20260630", "乙", 10), ("20260331", "甲", 90))
        )
    )
    result = consume(api, *records, context=bond_context(*records) if api in BOND_APIS else None)
    facts = observation(result)
    assert facts["observed_concentration_ratio"] == pytest.approx(0.3)
    assert facts["observed_holder_count"] == 2
    assert facts["denominator_scope"] == scope
    assert facts["full_holder_distribution"] is None
    assert facts["stock_holder_attribution"] is (api != "top10_cb_holders")


def test_holder_variants_invalid_ratios_and_scope_overflow_are_not_aggregated():
    """场景：比例和超过100%、同名变体或超过前十范围；预期：不产生集中度或伪造全体分布。"""
    records = tuple(
        source(
            "top10_holders",
            str(index),
            raw={"end_date": "20260630", "holder_name": str(index)},
            values={"hold_ratio": (0.6, "ratio")},
        )
        for index in range(2)
    )
    result = consume("top10_holders", *records)
    assert result["status"] == "conflict"
    assert observation(result)["observed_concentration_ratio"] is None
    duplicate = replace(records[1], raw=records[1].raw | {"holder_name": "0"})
    assert consume("top10_holders", records[0], duplicate)["status"] == "unknown"
    many = tuple(
        source(
            "top10_holders",
            str(index),
            raw={"end_date": "20260630", "holder_name": str(index)},
            values={"hold_ratio": (0.01, "ratio")},
        )
        for index in range(11)
    )
    assert "top10_scope_exceeded" in consume("top10_holders", *many)["reason_codes"]


@pytest.mark.parametrize("prior", [None, 100, 0])
def test_holder_count_changes_need_two_distinct_certified_periods(prior):
    """场景：仅一期或前期100户/0户，当前90户；预期：仅100户作正分母得到-10%，其余变化未知。"""
    current = source(
        "stk_holdernumber", raw={"end_date": "20260630"}, values={"holder_num": (90, "户")}
    )
    records = (
        (current,)
        if prior is None
        else (
            current,
            source(
                "stk_holdernumber",
                "prior",
                raw={"end_date": "20260331"},
                values={"holder_num": (prior, "户")},
            ),
        )
    )
    facts = observation(consume("stk_holdernumber", *records))
    assert facts["holder_count"] == 90
    assert facts["change_ratio"] == (pytest.approx(-0.1) if prior == 100 else None)


@pytest.mark.parametrize(
    "begin,end,state",
    [
        ("20261010", None, "planned"),
        ("20260101", "20261001", "ended"),
        ("20260101", "20261010", "active"),
        ("20260101", None, "open_end_observed"),
        (None, None, "unknown"),
        ("20260101", "bad", "unknown"),
        ("20260101", "20250101", "unknown"),
    ],
)
def test_manager_tenure_preserves_dates_and_does_not_infer_ability(begin, end, state):
    """场景：任职待开始、结束、区间内、开放尾或非法日期；预期：各自事实状态独立，不评价能力。"""
    record = source(
        "stk_managers", raw={"name": "甲", "title": "经理", "begin_date": begin, "end_date": end}
    )
    facts = observation(consume("stk_managers", record))
    assert facts["management_facts"][0]["tenure_state"] == state
    assert facts["management_ability_inference"] == "not_supported"


def test_compensation_uses_latest_period_and_keeps_multiple_roles_without_sum():
    """场景：当前同人两岗位薪酬与旧期记录；预期：只保留当前事实，不将两职薪酬相加或推导能力。"""
    records = tuple(
        source(
            "stk_rewards",
            str(index),
            raw={"end_date": period, "name": "甲", "title": title},
            values={"reward": (100, "元"), "hold_vol": (10, "股")},
        )
        for index, (period, title) in enumerate(
            (("20260630", "经理"), ("20260630", "董事"), ("20260331", "经理"))
        )
    )
    facts = observation(consume("stk_rewards", *records))
    assert len(facts["management_facts"]) == 2
    assert all(item["reward"] == 100 for item in facts["management_facts"])
    assert facts["aggregation_policy"] == "distinct_person_title_facts_no_sum"


def quote_sources(*, close=110, volume=100, premium=0.1):
    """创建同日两份转债报价，数值全部有认证；例：110/100-1与报告10%一致。"""
    values = {"close": (close, "元"), "vol": (volume, "手"), "amount": (11000, "元")}
    return (
        source(
            "cb_daily",
            entity=BOND,
            values=values | {"cb_value": (100, "元"), "cb_over_rate": (premium, "ratio")},
        ),
        source("cb_factor_pro", entity=BOND, values=values),
    )


def test_bond_quotes_and_conversion_premium_are_cross_checked_without_factor_score():
    """场景：认证CB关系、同日正股、面值及两份债券报价一致；预期：转股价值100元与10%溢价一致。"""
    records = quote_sources()
    for api in ("cb_daily", "cb_factor_pro"):
        result = consume(api, *records, context=bond_context(*records))
        facts = observation(result)
        assert result["status"] == "observed"
        assert facts["conversion_premium_check"]["agreement"] is True
        assert facts["underlying_conversion_value_check"]["agreement"] is True
        assert facts["technical_factor_score_increment"] == 0
        assert {"cb_basic", "daily", "cb_daily", "cb_factor_pro"} <= {
            ref.source_api for ref in result["evidence_refs"]
        }


@pytest.mark.parametrize("volume,premium,status", [(0, 0.1, "unknown"), (100, 0.5, "conflict")])
def test_bond_reference_close_and_premium_conflict_are_explicit(volume, premium, status):
    """场景：零成交参考价或报告溢价与报价不符；预期：分别unknown/conflict，旁证不增加技术分数。"""
    records = quote_sources(volume=volume, premium=premium)
    result = consume("cb_daily", *records, context=bond_context(*records))
    assert result["status"] == status
    if volume == 0:
        assert observation(result)["conversion_premium_check"]["right"] is None


def test_bond_conversion_requires_confirmed_cb_relation_and_never_uses_eb_dilution():
    """场景：未认证正股关系、EB或同日正股不存在；预期：不计算本地转股价值，EB无发行人稀释。"""
    records = quote_sources()
    context = bond_context(*records, confirmed=False)
    assert _conversion_value(context, records[0])[0] is None
    result = consume("cb_daily", *records, context=context)
    assert "cb_relation_unverified" in result["reason_codes"]
    eb = consume("cb_daily", *records, context=bond_context(*records, kind="EB"))
    assert observation(eb)["equity_dilution_applicable"] is False
    wrong_day = source(
        "daily",
        raw={"trade_date": "20261002"},
        values={"close": (10, "元"), "vol": (100, "手"), "amount": (1000, "元")},
    )
    context = bond_context(*records, stock=wrong_day)
    assert (
        _conversion_value(context, records[0])[2]
        == "conversion_period_or_underlying_quote_unverified"
    )


@pytest.mark.parametrize(
    "api,date_field",
    [
        ("fina_indicator", "end_date"),
        ("bak_basic", "trade_date"),
        ("cb_daily", "trade_date"),
        ("top10_holders", "end_date"),
        ("stk_rewards", "end_date"),
        ("cb_rating", "rating_date"),
    ],
)
def test_future_business_snapshots_are_rejected_even_if_observed_early(api, date_field):
    """场景：观察时间合格但报告期/行情日/评级日来自未来；预期：无引用，不能提前消费未来快照。"""
    record = source(api, raw={date_field: "20261004"}, entity=BOND if api in BOND_APIS else A)
    result = consume(api, record)
    assert result["evidence_refs"] == ()
    assert "disclosure_unavailable_rows_rejected" in result["reason_codes"]


@pytest.mark.parametrize(
    "api,patch",
    [
        ("bak_basic", {"trade_date": None}),
        ("fina_indicator", {"end_date": None}),
        ("cb_daily", {"trade_date": None}),
        ("top10_holders", {"end_date": None}),
        ("stk_holdernumber", {"end_date": None}),
        ("stk_rewards", {"end_date": None}),
        ("stk_managers", {"name": None}),
        ("cb_rating", {"rating_com_name": None, "rating_type": None}),
    ],
)
def test_incomplete_business_identifiers_retain_unknown_source_observations(api, patch):
    """场景：已观察行缺少关键日期或身份字段；预期：来源仍有独立观测结果，业务计算保持未知。"""
    record = source(api, raw=patch, entity=BOND if api in BOND_APIS else A)
    result = consume(api, record, context=bond_context(record) if api in BOND_APIS else None)
    assert result["status"] == "unknown"
    assert len(result["facts"]["observations"]) == 1


def test_unknown_and_negative_holder_and_compensation_values_are_not_calculable():
    """场景：股东户数非整数、集中比例缺失、薪酬为负；预期：数值均未知，不能以零替代缺失。"""
    count = source(
        "stk_holdernumber", raw={"end_date": "20260630"}, values={"holder_num": (1.5, "户")}
    )
    assert observation(consume("stk_holdernumber", count))["holder_count"] is None
    holder = source("top10_holders", raw={"end_date": "20260630", "holder_name": "甲"})
    assert observation(consume("top10_holders", holder))["observed_concentration_ratio"] is None
    reward = source(
        "stk_rewards",
        raw={"end_date": "20260630", "name": "甲"},
        values={"reward": (-1, "元"), "hold_vol": (-1, "股")},
    )
    facts = observation(consume("stk_rewards", reward))
    assert facts["management_facts"][0]["reward"] is None
    assert facts["management_facts"][0]["holding_shares"] is None


def test_financial_statement_fallback_never_hides_unknown_units():
    """场景：只有普通报表可用或VIP同字段单位未认证；预期：普通表可替代，存在不可比双源时未知。"""
    records = financial_sources()
    ordinary = source(
        "balancesheet",
        "ordinary",
        raw={"end_date": "20260630", "report_type": "1", "comp_type": "1"},
        values={"total_liab": (40, "元"), "total_assets": (100, "元")},
    )
    selected, _ = _statement_for(
        certified_context(records[1], ordinary), records[1], "balancesheet", ("total_assets",)
    )
    assert selected is ordinary
    selected, _ = _statement_for(
        certified_context(*records, ordinary), records[1], "balancesheet", ("total_assets",)
    )
    assert selected is records[2]
    unknown = replace(records[2], numeric={})
    selected, _ = _statement_for(
        certified_context(records[1], unknown, ordinary),
        records[1],
        "balancesheet",
        ("total_assets",),
    )
    assert selected is None
    invalid_ratio = source("fina_indicator", values={"roe": (None, "ratio")})
    assert _ratio_value(invalid_ratio, "roe") is None


def test_bond_conversion_rejects_missing_units_and_overflow():
    """场景：面值无认证或认证极端值溢出；预期：转股价值未知，JSON中不会写入无穷大。"""
    records = quote_sources()
    raw = {"cb_type": "CB", "conv_start_date": "20260101", "conv_end_date": "20271231"}
    incomplete = source(
        "cb_basic", "terms", entity=BOND, raw=raw, values={"conv_price": (10, "元")}
    )
    context = bond_context(*records, terms=incomplete)
    assert _conversion_value(context, records[0])[2] == "conversion_value_units_unverified"
    huge = source(
        "cb_basic",
        "terms",
        entity=BOND,
        raw=raw,
        values={"par": (1e308, "元"), "conv_price": (1e-308, "元")},
    )
    context = bond_context(*records, terms=huge)
    assert _conversion_value(context, records[0])[2] == "nonfinite_conversion_value"


def test_unmapped_matching_bond_codes_are_only_independent_corroboration():
    """场景：未映射的债券代码可连接其条款，但无认证正股关系；预期：独立票息事实保留且归因未知。"""
    rate = source(
        "cb_rate",
        entity=None,
        raw={"ts_code": "CB1", "rate_start_date": "20260101", "rate_end_date": "20261231"},
        values=None,
    )
    basic = source("cb_basic", "terms", entity=None, raw={"ts_code": "CB1", "cb_type": "CB"})
    result = consume(
        "cb_rate", rate, basic, context=inputs(rate, basic, entities=()), entity=MARKET_ENTITY
    )
    assert observation(result)["bond_type"] == "CB"
    assert observation(result)["underlying_relation_status"] == "unknown"
    assert {"cb_basic", "cb_rate"} <= {ref.source_api for ref in result["evidence_refs"]}


@pytest.mark.parametrize(
    "result_day,planned", [("invalid", 100.0), ("20260801", 100.0), ("20260930", -1.0)]
)
def test_bond_issuance_requires_ordered_result_date_and_nonnegative_amount(result_day, planned):
    """场景：发行结果日期非法、早于公告或计划金额负；预期：输出unknown且保留对应缺口原因。"""
    item = source(
        "cb_issue",
        entity=BOND,
        raw={"ann_date": "20260901", "res_ann_date": result_day},
        values={"plan_issue_size": (planned, "元"), "issue_size": (100, "元")},
    )
    result = consume("cb_issue", item, context=bond_context(item), entity=A)
    assert result["status"] == "unknown"
    assert (
        "issuance_amount_unverified" if planned < 0 else "issuance_result_date_unverified"
    ) in result["reason_codes"]
    if planned >= 0:
        assert observation(result)["stage"] == "unknown"
        assert observation(result)["implemented_amount"] is None


def test_future_rating_business_date_does_not_become_current_rating():
    """场景：记录已观察但评级业务日期在未来；预期：当前评级被清空，状态unknown并保留原因。"""
    item = source(
        "cb_rating",
        entity=BOND,
        raw={
            "rating_date": "20261004",
            "rating_comp": "机构",
            "rating_type": "债项",
            "rating": "AA",
        },
    )
    result = consume("cb_rating", item, context=bond_context(item), entity=A)
    assert result["status"] == "unknown"
    assert "disclosure_unavailable_rows_rejected" in result["reason_codes"]
    from tushare_entitlement_hub.equity_research.disclosure_consumers import _ratings

    direct = _ratings(bond_context(item), "cb_rating", (item,))
    assert "rating_date_in_future" in direct["reason_codes"]


def test_financial_observations_without_peers_never_claim_reconciliation():
    """场景：备用估值和财务指标没有普通VIP对照；预期：保留当前值、对账未知并说明来源缺口。"""
    backup = source(
        "bak_basic",
        raw={"trade_date": "20261003"},
        values={"pb": (2, "ratio"), "float_share": (100, "股")},
    )
    result = consume("bak_basic", backup)
    assert result["status"] == "unknown"
    assert "valuation_snapshot_date_mismatch" in result["reason_codes"]
    assert observation(result)["comparisons"]["pb"]["agreement"] is None
    indicator = financial_sources()[0]
    result = consume("fina_indicator", indicator, context=certified_context(indicator))
    assert result["status"] == "unknown"
    assert "indicator_duplicate_scope_or_version_unverified" in result["reason_codes"]


def test_bond_quote_and_coupon_without_basic_or_daily_dependencies_keep_unknown():
    """场景：债券虽有认证关系但缺条款、正股行情和债券对照；预期：转股计算未知而非零。"""
    quote = source(
        "cb_daily",
        entity=BOND,
        raw={"trade_date": "20261003"},
        values={"close": (110, "元"), "vol": (10, "手"), "amount": (1100, "元")},
    )
    context = inputs(quote, entities=(A, BOND), relations=(relation(),))
    value, refs, reason = _conversion_value(context, quote)
    assert value is None
    assert reason == "conversion_period_or_underlying_quote_unverified"
    result = consume("cb_daily", quote, context=context)
    assert result["status"] == "unknown"
    coupon = source(
        "cb_rate",
        entity=BOND,
        raw={"rate_start_date": "20260101", "rate_end_date": "20261231"},
        values={"coupon_rate": (0.02, "ratio"), "pay_per_year": (1, "次")},
    )
    result = consume("cb_rate", coupon)
    assert result["status"] == "unknown"
