"""P07新增资金消费的离线合成场景，不访问供应商或使用真实凭证。"""

from dataclasses import replace

import pytest
from test_research_flow import daily
from test_research_governance import DAY, A, B, inputs, row

from tushare_entitlement_hub.equity_research.flow import (
    AUXILIARY_FIELDS,
    _auxiliary_rows,
    build_flow,
)
from tushare_entitlement_hub.equity_research.flow_consumers import (
    FLOW_CONSUMER_FIELDS,
    _aliases,
    _fields,
    _value,
    auxiliary_observation,
    consume_flow,
)
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.product_foundation.contracts import Truth


def consume(source, *others, entity=A, complete=(), expected=None, reasons=None):
    """在显式股票范围内调用一个来源；例：认证融资名单返回仅资格可见的观察。"""
    context = inputs(source, *others, complete=complete, expected=expected, reasons=reasons)
    return consume_flow(context, "P07", source.source_api, (source,), entity)


def interpretation(result, index=0):
    """读取指定来源的领域解释；例：默认取第一条借券期限分层。"""
    return result["facts"]["observations"][index]["interpretation"]


def chip_summary(**changes):
    """生成半数8元、半数12元对应的供应商摘要；例：均价10、50分位8、获利比例0.5。"""
    values = {
        "cost_5pct": (8, "元"),
        "cost_50pct": (8, "元"),
        "cost_95pct": (12, "元"),
        "weight_avg": (10, "元"),
        "winner_rate": (0.5, "fraction"),
    }
    return row("cyq_perf", values=values | changes)


def distribution(day=DAY):
    """生成完整离散筹码分布；例：两档权重各0.5，总和为1。"""
    return (
        row("cyq_chips", "low", day=day, values={"price": (8, "元"), "percent": (0.5, "fraction")}),
        row(
            "cyq_chips", "high", day=day, values={"price": (12, "元"), "percent": (0.5, "fraction")}
        ),
    )


def seat(label="甲", organization="合成营业部", **changes):
    """生成有明确标签的10元买入、3元卖出席位；例：净额7满足内部核对。"""
    return row(
        "hm_detail",
        values={"buy_amount": (10, "元"), "sell_amount": (3, "元"), "net_amount": (7, "元")},
        raw={"hm_name": label, "hm_orgs": organization} | changes,
    )


def label_row(name="甲", record="label", **changes):
    """生成无股票绑定的全局游资字典；例：甲关联两个文本席位，不代表最终投资人。"""
    return row(
        "hm_list",
        record,
        entity=None,
        day=None,
        raw={"name": name, "desc": "供应商标签说明", "orgs": "合成营业部、另一营业部"} | changes,
    )


@pytest.mark.parametrize("api", tuple(FLOW_CONSUMER_FIELDS))
def test_all_new_consumers_preserve_empty_unknown(api):
    """场景：十个新增接口均无行；预期：各自执行且unknown，不声称观测到数据。"""
    result = consume_flow(inputs(), "P07", api, (), A)
    assert result["status"] == "unknown"
    assert result["facts"]["observations"] == []
    assert result["evidence_refs"] == ()
    assert result["gates"][0].result is Truth.UNKNOWN
    assert not result["gates"][0].hard


def test_unsupported_dispatch_is_rejected():
    """场景：非P07产品或未注册API；预期：明确拒绝错误调度。"""
    for product, api in (("P01", "cyq_perf"), ("P07", "daily")):
        with pytest.raises(ValueError, match="unsupported_flow_consumer"):
            consume_flow(inputs(), product, api, (), A)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"entity_id": None}, "entity_binding_unknown"),
        ({"entity_id": B}, "entity_binding_unknown"),
        ({"raw_change": {"quality_tier": "quarantine"}}, "quarantined_record"),
        ({"raw_change": {"is_quarantined": True}}, "quarantined_record"),
        ({"raw_change": {"business_variant_key": "unresolved"}}, "business_variant_unresolved"),
        ({"observed_at": "2026-10-04T00:00:00Z"}, "not_yet_available"),
        ({"data_as_of": None}, "business_date_unverified"),
        ({"data_as_of": "2026-10-04"}, "business_date_unverified"),
    ],
)
def test_rejected_rows_never_become_local_evidence(mutation, reason):
    """场景：无绑定、隔离、变体、未来或无日期行；预期：无数字和引用，状态unknown。"""
    source = row("slb_sec", values={"lent_qnt": (10, "股"), "end_bal": (100, "元")})
    if "raw_change" in mutation:
        source = replace(source, raw=source.raw | mutation["raw_change"])
    elif "entity_id" in mutation:
        source = replace(source, entity_id=mutation["entity_id"])
    else:
        source = replace(source, availability=replace(source.availability, **mutation))
    result = consume(source)
    assert result["status"] == "unknown"
    assert reason in result["reason_codes"]
    assert result["evidence_refs"] == ()
    assert "interpretation" not in result["facts"]["observations"][0]


def test_source_failures_dates_and_api_mismatch_are_preserved():
    """场景：已绑定行但来源失败、过期或混入其他API；预期：三个问题均不能被数值覆盖。"""
    source = row("margin_secs", raw={"exchange": "SSE"})
    result = consume(source, reasons={"margin_secs": ("partial_scan",)})
    assert "partial_scan" in result["reason_codes"]
    assert consume(source, expected={"daily": "2026-10-02"})["status"] == "unknown"
    result = consume_flow(inputs(source), "P07", "stock_hsgt", (source,), A)
    assert "source_api_mismatch" in result["reason_codes"]


def test_numeric_fields_need_certificates_currency_and_units():
    """场景：原始金额、美元、错误单位和未认证数值；预期：均不作为可计算人民币。"""
    raw = row("slb_sec", raw={"end_bal": 100})
    foreign = row("slb_sec", values={"end_bal": (100, "元", "USD")})
    wrong = row("slb_sec", values={"end_bal": (100, "股")})
    for source in (raw, foreign, wrong):
        assert _value(source, "end_bal") is None
        fields, quality = _fields(source, ("end_bal", "absent"))
        assert fields == {"end_bal": None, "absent": None}
        assert quality["end_bal"] == "unit_or_value_unverified"
        assert quality["absent"] == "missing"
    assert _value(row("slb_sec", values={"end_bal": (100, "元")}), "end_bal") == 100


def test_chip_costs_crosscheck_only_complete_same_day_distribution():
    """场景：认证报价10元、两档完整筹码；预期：均价及三个分位一致，引用覆盖双方字段。"""
    result = consume(chip_summary(), daily(), *distribution(), complete=("cyq_chips",))
    details = interpretation(result)
    assert result["status"] == "observed"
    assert details["distribution_comparison"] == "matched"
    assert details["quote_vs_average_cost"] == 0
    assert details["recomputed_costs_cny"]["cost_50pct"] == 8
    assert details["winner_rate_semantics"] == "vendor_chip_profit_fraction_not_strategy_win_rate"
    assert {ref.source_api for ref in result["evidence_refs"]} == {"daily", "cyq_chips", "cyq_perf"}
    assert not result["facts"]["independent_evidence"]


@pytest.mark.parametrize(
    "variant",
    [
        "no_scope",
        "stale_distribution",
        "no_quote",
        "reference_quote",
        "no_units",
        "partial_weights",
    ],
)
def test_chip_incomplete_context_stays_unknown(variant):
    """场景：筹码缺范围、异日、参考价、缺单位或不完整权重；预期：成本核对保留unknown。"""
    source, quote, chips = chip_summary(), daily(), distribution()
    complete = ("cyq_chips",)
    if variant == "no_scope":
        complete = ()
    elif variant == "stale_distribution":
        chips = distribution("2026-10-02")
    elif variant == "no_quote":
        quote = row("daily")
    elif variant == "reference_quote":
        quote = daily(raw={"price_type": "reference"})
    elif variant == "no_units":
        source = replace(source, numeric={})
    elif variant == "partial_weights":
        chips = chips[:1]
    result = consume(source, quote, *chips, complete=complete)
    assert result["status"] == "unknown"
    assert interpretation(result)["distribution_comparison"] == "unknown"


@pytest.mark.parametrize(
    "changes",
    [{"cost_50pct": (13, "元")}, {"weight_avg": (11, "元")}, {"winner_rate": (1.1, "fraction")}],
)
def test_chip_contradictions_are_soft_conflicts(changes):
    """场景：分位颠倒、均价不符或获利比例越界；预期：局部冲突不会成为停牌门禁。"""
    result = consume(chip_summary(**changes), daily(), *distribution(), complete=("cyq_chips",))
    assert result["status"] == "conflict"
    assert result["gates"][0].result is Truth.FALSE
    assert not result["gates"][0].hard


def test_chip_unknown_winner_does_not_claim_strategy_win_rate():
    """场景：成本完全一致但winner_rate无单位；预期：核对可见且整体unknown，不生成胜率。"""
    source = chip_summary(winner_rate=(None, "fraction"))
    result = consume(source, daily(), *distribution(), complete=("cyq_chips",))
    assert result["status"] == "unknown"
    assert interpretation(result)["distribution_comparison"] == "matched"
    assert interpretation(result)["winner_rate"] is None


def test_global_dictionary_links_local_seats_without_investor_attribution():
    """场景：甲字典无股票绑定，A有甲席位；预期：精确连接说明和别名，不识别最终投资人。"""
    label, source = label_row(), seat()
    result = consume(source, label)
    assert result["status"] == "observed"
    assert interpretation(result)["alias_match"] == "matched"
    assert interpretation(result)["beneficial_owner"] == "unknown"
    assert not interpretation(result)["institutional_investor_evidence"]
    assert {ref.fields[0] for ref in result["evidence_refs"] if ref.source_api == "hm_list"} == {
        "name",
        "desc",
        "orgs",
    }
    global_result = consume_flow(
        inputs(label, entities=()), "P07", "hm_list", (label,), MARKET_ENTITY
    )
    assert global_result["status"] == "observed"
    assert interpretation(global_result)["equity_attribution"] == "unknown"
    local_result = consume(label, source)
    assert interpretation(local_result)["linked_seat_records"] == ["one"]
    assert consume(label)["status"] == "unknown"


def test_seat_aliases_and_ambiguous_dictionary_remain_unknown():
    """场景：未知席位、同名两个字典、无标签或匿名机构；预期：不通过模糊名称作实体推断。"""
    assert consume(seat(organization="未知席位"), label_row())["status"] == "unknown"
    assert consume(seat(), label_row(), label_row(record="duplicate"))["status"] == "unknown"
    assert consume(seat(), label_row(orgs=None))["status"] == "unknown"
    anonymous = consume(seat(organization="机构专用"), label_row(orgs="机构专用"))
    assert not interpretation(anonymous)["institutional_investor_evidence"]
    assert not interpretation(anonymous)["anonymous_seats_aggregated"]
    assert _aliases({"unexpected": "object"}) == ()
    assert _aliases(["甲", "乙"]) == ("甲", "乙")


def test_seat_net_amount_conflict_is_preserved():
    """场景：同日已认证买10卖3却报净8；预期：精确字典连接不能掩盖金额冲突。"""
    source = seat()
    wrong = row(
        "hm_detail",
        values={"buy_amount": (10, "元"), "sell_amount": (3, "元"), "net_amount": (8, "元")},
        raw=source.raw,
    )
    assert consume(wrong, label_row())["status"] == "conflict"


@pytest.mark.parametrize(
    "api,raw,expected",
    [
        ("margin_secs", {"exchange": "SSE"}, "observed"),
        ("margin_secs", {"exchange": "unknown"}, "unknown"),
        ("stock_hsgt", {"type": "HK_SH", "type_name": "沪股通(港>沪)"}, "observed"),
        ("stock_hsgt", {"type": "HK_SZ", "type_name": "深股通(港>深)"}, "observed"),
        ("stock_hsgt", {"type": "SH_HK", "type_name": "港股通(沪>港)"}, "unknown"),
        ("stock_hsgt", {"type": "HK_SH", "type_name": "不明标签"}, "unknown"),
    ],
)
def test_eligibility_is_separate_from_tradability_and_holdings(api, raw, expected):
    """场景：融资或通道名单不同类型；预期：只确认明确资格，南向类型不变成A股北向资格。"""
    result = consume(row(api, raw=raw))
    details = interpretation(result)
    assert result["status"] == expected
    assert details["tradability"] == "unknown"
    assert details["holdings"] == "unknown"
    assert not details["absence_implies_ineligible"]
    assert not result["gates"][0].hard


def test_lending_uses_real_units_and_retained_observation_changes():
    """场景：同股两次认证融出量及余额；预期：保留区间变化，不能推断可借量或全量空头。"""
    old = row(
        "slb_sec", "old", day="2026-10-02", values={"lent_qnt": (10, "股"), "end_bal": (100, "元")}
    )
    current = row("slb_sec", values={"lent_qnt": (12, "股"), "end_bal": (130, "元")})
    result = consume(current, old, expected={"daily": DAY})
    details = interpretation(result)
    assert result["status"] == "observed"
    assert details["balance_change_cny"] == 30
    assert details["lent_change_shares"] == 2
    assert details["borrowing_availability"] == "unknown"
    assert details["total_short_interest"] == "not_supported"
    assert details["change_period"] == ["2026-10-02", DAY]
    assert any(ref.source_record_id == "old" for ref in result["evidence_refs"])


def test_previous_lending_observation_still_requires_availability():
    """场景：较早业务日的借券数据直到分析日后才采到；预期：不回填余额变化或引用。"""
    current = row("slb_sec", values={"lent_qnt": (12, "股"), "end_bal": (130, "元")})
    old = row(
        "slb_sec", "old", day="2026-10-02", values={"lent_qnt": (10, "股"), "end_bal": (100, "元")}
    )
    old = replace(old, availability=replace(old.availability, observed_at="2026-10-04T00:00:00Z"))
    result = consume(current, old, expected={"daily": DAY})
    assert interpretation(result)["balance_change_cny"] is None
    assert all(ref.source_record_id != "old" for ref in result["evidence_refs"])


def test_northbound_channel_observations_do_not_prove_membership_change():
    """场景：同一证券两个观察日都在沪股通名单；预期：记录通道未变，成员增减仍unknown。"""
    raw = {"type": "HK_SH", "type_name": "沪股通(港>沪)"}
    old = row("stock_hsgt", "old", day="2026-10-02", raw=raw)
    current = row("stock_hsgt", raw=raw)
    result = consume(current, old, expected={"daily": DAY})
    change = interpretation(result)["channel_observation_change"]
    assert change["changed"] is False
    assert change["membership_change"] == "unknown"
    assert any(ref.source_record_id == "old" for ref in result["evidence_refs"])


def test_dictionary_ambiguous_name_cannot_become_confirmed_local_metadata():
    """场景：字典里同名甲有两份说明；预期：股票标签连接保留unknown。"""
    source = label_row()
    result = consume(source, label_row(record="duplicate"), seat())
    assert result["status"] == "unknown"


@pytest.mark.parametrize(
    "quantity,balance,status",
    [(None, 100, "unknown"), (10, None, "unknown"), (-1, 100, "conflict")],
)
def test_lending_missing_and_negative_values_do_not_become_pressure(quantity, balance, status):
    """场景：借券缺字段或数量为负；预期：unknown或conflict，不以零替代缺失。"""
    source = row("slb_sec", values={"lent_qnt": (quantity, "股"), "end_bal": (balance, "元")})
    assert consume(source)["status"] == status


@pytest.mark.parametrize(
    "tenor,rate,quantity,status",
    [
        (7, 0.02, 100, "observed"),
        (None, 0.02, 100, "unknown"),
        (0, 0.02, 100, "conflict"),
        (7, -0.1, 100, "conflict"),
    ],
)
def test_lending_detail_keeps_tenor_fee_rate_and_quantity(tenor, rate, quantity, status):
    """场景：认证期限费率分层及缺失异常；预期：使用tenor/fee_rate/lent_qnt，不臆造rate/term。"""
    source = row(
        "slb_sec_detail",
        values={
            "tenor": (tenor, "天"),
            "fee_rate": (rate, "fraction"),
            "lent_qnt": (quantity, "股"),
        },
    )
    result = consume(source)
    assert result["status"] == status
    assert interpretation(result)["strata_aggregated"] is False
    assert interpretation(result)["annualized_rate"] == "unknown"


@pytest.mark.parametrize(
    "count,status", [(3, "observed"), (0, "conflict"), (2.5, "conflict"), (None, "unknown")]
)
def test_limit_ladder_is_fragility_background_only(count, status):
    """场景：三连板、非法或未知连板数；预期：仅解释当前拥挤背景，不给未来收益。"""
    result = consume(row("limit_step", values={"nums": (count, "count")}))
    assert result["status"] == status
    assert interpretation(result)["return_forecast"] == "not_supported"
    assert not result["facts"]["affects_core_score"]


@pytest.mark.parametrize(
    "net,amount,market,status",
    [
        (10, 100, "1", "observed"),
        (-10, 100, "3", "observed"),
        (101, 100, "1", "conflict"),
        (None, 100, "1", "unknown"),
        (10, 100, "2", "unknown"),
    ],
)
def test_top10_is_always_a_selected_sample(net, amount, market, status):
    """场景：前十样本净额与成交额；预期：只算样本比例，不外推全市场流入和持仓。"""
    result = consume(
        row(
            "hsgt_top10",
            raw={"market_type": market},
            values={"net_amount": (net, "元"), "amount": (amount, "元")},
        )
    )
    details = interpretation(result)
    assert result["status"] == status
    assert details["full_market_net_flow"] == "not_supported"
    assert details["holdings_inference"] == "not_supported"


def test_top_list_checks_each_seat_without_summing_anonymous_rows():
    """场景：两条匿名席位各买7而榜单总买10；预期：逐席位上界可核对，不相加成14误报。"""
    summary = row(
        "top_list",
        raw={"reason": "合成上榜原因"},
        values={"l_buy": (10, "元"), "l_sell": (3, "元"), "net_amount": (7, "元")},
    )
    seats = [
        row(
            "top_inst",
            str(i),
            raw={"reason": "合成上榜原因", "exalter": "机构专用"},
            values={"buy": (7, "元"), "sell": (2, "元")},
        )
        for i in range(2)
    ]
    result = consume(summary, *seats)
    assert result["status"] == "observed"
    assert len(interpretation(result)["seat_checks"]) == 2
    assert not interpretation(result)["seat_totals_aggregated"]
    too_large = row(
        "top_inst",
        "large",
        raw={"reason": "合成上榜原因"},
        values={"buy": (11, "元"), "sell": (2, "元")},
    )
    assert consume(summary, too_large)["status"] == "conflict"
    other_reason = replace(too_large, raw=too_large.raw | {"reason": "另一个原因"})
    assert interpretation(consume(summary, other_reason))["seat_checks"] == []


def test_auxiliary_existing_sources_use_actual_fields_and_linked_metadata():
    """场景：既有资金结果含THS、转融券和全局名录；预期：真实字段进入局部证据且只绑定本股标签。"""
    source, label, other_label = seat(), label_row(), label_row("乙", "other")
    context = inputs(
        source,
        label,
        other_label,
        row(
            "limit_list_ths",
            raw={"limit_type": "涨停", "lu_desc": "合成原因"},
            values={"open_num": (2, "次")},
        ),
        row("slb_sec", values={"lent_qnt": (10, "股"), "end_bal": (100, "元")}),
    )
    assert _auxiliary_rows(context, "hm_list", A) == (label,)
    assert _auxiliary_rows(context, "hm_detail", A) == (source,)
    result = build_flow(context)[0]
    auxiliary = {item["source_api"]: item for item in result.details["auxiliary"]}
    assert auxiliary["slb_sec"]["source_fields"] == {"lent_qnt": 10, "end_bal": 100}
    assert "open_num" in auxiliary["limit_list_ths"]["source_fields"]
    assert "tag" not in auxiliary["limit_list_ths"]["source_fields"]
    assert auxiliary["hm_list"]["source_fields"]["name"] == "甲"
    assert len(result.envelope.quality_gates) == 6
    assert "rqye" not in AUXILIARY_FIELDS["slb_sec"]
    assert AUXILIARY_FIELDS["slb_sec_detail"] == ("tenor", "fee_rate", "lent_qnt")


def test_auxiliary_unbound_or_unverified_values_remain_unknown():
    """场景：局部席位无认证金额或被隔离；预期：保留字段质量，数值不作为可计算证据。"""
    source = row("top_inst", raw={"exalter": "机构专用", "buy": 100})
    result = auxiliary_observation(inputs(source), source, A, ("exalter", "buy"))
    assert result["status"] == "unknown"
    assert result["source_fields"]["buy"] is None
    assert result["field_quality"]["buy"] == "unit_or_value_unverified"
    rejected = replace(source, raw=source.raw | {"is_quarantined": True})
    result = auxiliary_observation(inputs(rejected), rejected, A, ("exalter", "buy"))
    assert result["source_fields"] == {"exalter": None, "buy": None}


def test_missing_buy_amount_and_stock_reason_remain_unknown():
    """场景：游资缺买入额、榜单缺上榜原因；预期：净额核对及局部状态unknown而非零流入。"""
    missing = replace(seat(), numeric={})
    assert consume(missing, label_row())["status"] == "unknown"
    listing = row(
        "top_list", values={"l_buy": (10, "元"), "l_sell": (3, "元"), "net_amount": (7, "元")}
    )
    result = consume(listing)
    assert result["status"] == "unknown"
    assert interpretation(result)["net_amount_consistency"] == "true"
    assert interpretation(result)["seat_checks"] == []


@pytest.mark.parametrize("rejected", [False, True])
def test_seat_with_missing_amount_or_quarantine_cannot_certify_stock_total(rejected):
    """场景：同原因席位缺金额或隔离；预期：席位上界unknown，隔离席位不贡献业务引用。"""
    listing = row(
        "top_list",
        raw={"reason": "涨幅"},
        values={"l_buy": (10, "元"), "l_sell": (3, "元"), "net_amount": (7, "元")},
    )
    item = row("top_inst", "seat", raw={"reason": "涨幅", "is_quarantined": rejected})
    result = consume(listing, item)
    assert interpretation(result)["seat_checks"] == [
        {"source_record_id": "seat", "within_stock_total": "unknown"}
    ]
    assert not interpretation(result)["seat_totals_aggregated"]
    if rejected:
        assert all(ref.source_api != "top_inst" for ref in result["evidence_refs"])


def test_chip_without_quote_and_invalid_previous_channel_preserve_unknown():
    """场景：筹码没有主行情、北向名单前次渠道不明；预期：价格未知且不推断渠道变化。"""
    result = consume(chip_summary(), *distribution(), complete=("cyq_chips",))
    assert result["status"] == "unknown"
    assert interpretation(result)["quote_cny"] is None
    current = row("stock_hsgt", raw={"type": "HK_SH", "type_name": "沪股通(港>沪)"})
    previous = row(
        "stock_hsgt", "previous", day="2026-10-02", raw={"type": "UNKNOWN", "type_name": "未知"}
    )
    state = interpretation(consume(current, previous))
    assert state["eligibility"] == "listed"
    assert state["membership_change"] == "unknown"
    assert "channel_observation_change" not in state
