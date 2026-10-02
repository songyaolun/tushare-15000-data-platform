"""资金与可行性的离线合成测试，确认方向、范围与门禁均诚实降级。"""

from dataclasses import replace

import pytest
from test_research_governance import DAY, NOW, A, B, inputs, row

from tushare_entitlement_hub.equity_research.flow import (
    _activity,
    _aligned,
    _block_trades,
    _freshness,
    _leverage,
    _limits,
    _overhang,
    _shock,
    _source_flow,
    _suspension,
    build_flow,
)
from tushare_entitlement_hub.product_foundation.contracts import Truth

COMPLETE = ("suspend_d", "stk_high_shock", "cyq_chips")


def daily(entity=A, values=None, raw=None, day=DAY):
    """构造收盘10、成交100元、10股的有效日线；例：覆盖vol=0可测试量额冲突。"""
    return row(
        "daily",
        entity=entity,
        day=day,
        values={"close": (10, "元"), "amount": (100, "元"), "vol": (10, "股")} | (values or {}),
        raw=raw,
    )


def full_rows(entity=A, scale=1):
    """生成同日资金净流出、杠杆及完整筹码分布；例：scale=1的大单比例为-0.1。"""
    return [
        daily(entity),
        row(
            "moneyflow",
            entity=entity,
            values={
                "buy_lg_amount": (0, "元"),
                "buy_elg_amount": (0, "元"),
                "sell_lg_amount": (5 * scale, "元"),
                "sell_elg_amount": (5 * scale, "元"),
            },
        ),
        row("moneyflow_dc", entity=entity, values={"net_amount": (-20 * scale, "元")}),
        row("moneyflow_ths", entity=entity, values={"net_amount": (-30 * scale, "元")}),
        row(
            "daily_basic",
            entity=entity,
            values={"circ_mv": (100, "元"), "turnover_rate": (0.1 * scale, "fraction")},
        ),
        row("margin_detail", entity=entity, values={"rzye": (10 * scale, "元")}),
        row(
            "cyq_chips",
            "low",
            entity=entity,
            values={"price": (8, "元"), "percent": (1 - 0.3 * scale, "fraction")},
        ),
        row(
            "cyq_chips",
            "high",
            entity=entity,
            values={"price": (12, "元"), "percent": (0.3 * scale, "fraction")},
        ),
        row("stk_limit", entity=entity, values={"up_limit": (11, "元"), "down_limit": (9, "元")}),
    ]


def test_empty_flow_is_unknown_and_has_independent_gates():
    """场景：无绑定或无观察；预期：空集合或全部未知，六个门禁独立保留。"""
    assert build_flow(inputs(entities=())) == ()
    result = build_flow(inputs())[0]
    assert len(result.envelope.quality_gates) == 6
    assert all(check.result is Truth.UNKNOWN for check in result.envelope.quality_gates)
    assert result.metric("flow_support") is None
    assert result.metric("crowding") is None
    assert result.metric("flow_direction") is None
    assert result.details["market_activity"] == "unknown"


def test_negative_sources_keep_direction_and_two_axes():
    """场景：全来源流出但A相对分位最高；预期：支持轴可高，方向仍负，拥挤轴独立。"""
    sources = full_rows(A) + full_rows(B, 2)
    sources.extend(
        [
            row("block_trade", values={"price": (9, "元")}),
            row("top_inst", raw={"business_variant_key": "anonymous", "exalter": "机构专用"}),
            row("top_inst", "second", raw={"business_variant_key": "other", "exalter": "机构专用"}),
            row("rt_min", raw={"trade_time": "10:00:00"}),
            row("hsgt_top10", raw={"rank": 1}),
        ]
    )
    results = build_flow(
        inputs(*sources, entities=(A, B), complete=COMPLETE, expected={"daily": DAY})
    )
    first, second = results
    assert first.metric("flow_direction") == -1
    assert first.metric("flow_source_median") == -0.2
    assert first.metric("sign_agreement") == 1
    assert first.metric("flow_support") == 100
    assert first.metric("crowding") == 0
    assert second.metric("crowding") == 100
    assert first.envelope.score is None
    assert all(check.result is Truth.TRUE for check in first.envelope.quality_gates)
    assert first.metric("limit_distance") == 0.1
    assert first.metric("overhang") == 0.3
    assert first.details["block_trades"][0]["block_discount"] == pytest.approx(0.1)
    assert len(first.details["auxiliary"]) == 4
    assert "minutes_do_not_extend_observed_scope" in first.details["scope_notes"]
    assert (
        sum(item["contribution"] for item in first.details["score_components"]["flow_support"])
        == 100
    )


def test_zero_directions_count_and_single_source_is_insufficient():
    """场景：来源包含零方向及相反方向；预期：零保留分母，单来源不产生一致性。"""
    source = row(
        "moneyflow",
        values={
            "buy_lg_amount": (10, "元"),
            "buy_elg_amount": (0, "元"),
            "sell_lg_amount": (0, "元"),
            "sell_elg_amount": (0, "元"),
        },
    )
    zero = row("moneyflow_dc", values={"net_amount": (0, "元")})
    negative = row("moneyflow_ths", values={"net_amount": (-10, "元")})
    result = build_flow(inputs(daily(), source, zero, negative))[0]
    assert result.metric("flow_direction") == 0
    assert result.metric("sign_agreement") == 0
    assert result.metric("other_flow_ratio_median") == -0.05
    result = build_flow(inputs(daily(), source))[0]
    assert result.metric("flow_direction") == 1
    assert result.metric("sign_agreement") is None


@pytest.mark.parametrize("api", ["moneyflow_dc", "moneyflow_ths"])
def test_vendor_net_amount_is_real_field_and_currency_is_not_guessed(api):
    """场景：只给其他字段、美元金额或异日金额；预期：不臆造net_amount或跨币种日期相除。"""
    wrong = row(api, values={"net_mf_amount": (10, "元")})
    foreign = row(api, values={"net_amount": (10, "元", "USD")})
    stale = row(api, day="2026-10-02", values={"net_amount": (10, "元")})
    for source in (wrong, foreign, stale):
        result = _source_flow(inputs(source), A, api, daily())
        assert result.value is None
    assert (
        _source_flow(
            inputs(row(api, values={"net_amount": (10, "元")})),
            A,
            api,
            daily(values={"amount": (0, "元")}),
        ).value
        is None
    )


def test_flow_source_reasons_and_missing_price_do_not_disappear():
    """场景：来源局部拒绝但有数据；预期：来源原因仍使相关资金比率未知。"""
    source = row("moneyflow_dc", values={"net_amount": (10, "元")})
    context = inputs(source, reasons={"moneyflow_dc": ("row_limit_exceeded",)})
    assert _source_flow(context, A, "moneyflow_dc", daily()).value is None
    assert not _aligned(source, None)
    assert not _aligned(
        replace(source, availability=replace(source.availability, data_as_of=None)), daily()
    )


@pytest.mark.parametrize(
    "changes,price,consistency,activity",
    [
        ({"amount": (None, "元")}, Truth.UNKNOWN, Truth.UNKNOWN, "unknown"),
        ({"vol": (-1, "股")}, Truth.FALSE, Truth.FALSE, "contradictory"),
        ({"amount": (-1, "元")}, Truth.FALSE, Truth.FALSE, "contradictory"),
        ({"vol": (0, "股")}, Truth.FALSE, Truth.FALSE, "contradictory"),
        ({"vol": (0, "股"), "amount": (0, "元")}, Truth.FALSE, Truth.TRUE, "no_trade"),
        ({"close": (0, "元")}, Truth.FALSE, Truth.TRUE, "traded"),
        ({"close": (None, "元")}, Truth.UNKNOWN, Truth.TRUE, "traded"),
        ({}, Truth.TRUE, Truth.TRUE, "traded"),
    ],
)
def test_price_and_volume_amount_gates_are_separate(changes, price, consistency, activity):
    """场景：量额缺失、负值、矛盾、无成交或零价；预期：价格与量额门禁各按事实判断。"""
    assert _activity(daily(values=changes)) == (price, consistency, activity)


def test_reference_and_no_trade_metadata_block_price():
    """场景：正价格却标明参考价或无成交；预期：不能用于交易可行性。"""
    for raw in ({"price_type": "reference"}, {"market_activity_status": "no_trade"}):
        assert _activity(daily(raw=raw))[0] is Truth.FALSE
    source = daily()
    fact = replace(source.numeric["close"], value=None, reason_codes=("no_traded_price",))
    assert _activity(replace(source, numeric=source.numeric | {"close": fact}))[0] is Truth.FALSE


@pytest.mark.parametrize(
    "raw,complete,expected",
    [
        ({"suspend_type": "S", "suspend_timing": None}, (), Truth.FALSE),
        ({"suspend_type": "S", "suspend_timing": "09:30-10:00"}, (), Truth.FALSE),
        ({"suspend_type": "S"}, COMPLETE, Truth.UNKNOWN),
        ({"suspend_type": "X"}, COMPLETE, Truth.UNKNOWN),
        ({"suspend_type": "R"}, (), Truth.TRUE),
    ],
)
def test_suspension_timing_is_explicit(raw, complete, expected):
    """场景：全天、日内、未知时段及复牌；预期：未知时段不当全天停牌或通过。"""
    assert _suspension(inputs(row("suspend_d", raw=raw), complete=complete), A, daily()) is expected


def test_suspension_scope_and_source_failure():
    """场景：无记录但采集范围未知或失败；预期：不能当作未停牌。"""
    assert _suspension(inputs(), A, daily()) is Truth.UNKNOWN
    assert _suspension(inputs(complete=COMPLETE), A, daily()) is Truth.TRUE
    context = inputs(
        row("suspend_d", raw={"suspend_type": "R"}), reasons={"suspend_d": ("missing_scope",)}
    )
    assert _suspension(context, A, daily()) is Truth.UNKNOWN


@pytest.mark.parametrize(
    "lower,upper,close,day,expected",
    [
        (9, 11, 10, DAY, Truth.TRUE),
        (9, 10, 10, DAY, Truth.FALSE),
        (10, 11, 10, DAY, Truth.FALSE),
        (11, 10, 10, DAY, Truth.UNKNOWN),
        (9, 11, 0, DAY, Truth.UNKNOWN),
        (9, 11, 10, "2026-10-02", Truth.UNKNOWN),
        (0, 11, 10, DAY, Truth.UNKNOWN),
        (9, None, 10, DAY, Truth.UNKNOWN),
    ],
)
def test_limit_prices_require_same_day_and_valid_bounds(lower, upper, close, day, expected):
    """场景：触及涨跌停、界限颠倒或异日限价；预期：只有合法同日区间内通过。"""
    source = row(
        "stk_limit", day=day, values={"up_limit": (upper, "元"), "down_limit": (lower, "元")}
    )
    state, distance = _limits(inputs(source), A, daily(values={"close": (close, "元")}))
    assert state is expected
    assert (distance.value is None) == (expected is Truth.UNKNOWN)


def test_freshness_requires_expected_business_day_and_recent_observation():
    """场景：缺预期日期、落后、未来日期及旧采集；预期：当前性不足不通过。"""
    context = inputs(expected={"daily": DAY})
    assert _freshness(context, None) is Truth.UNKNOWN
    assert _freshness(inputs(), daily()) is Truth.UNKNOWN
    source = daily()
    assert (
        _freshness(
            context, replace(source, availability=replace(source.availability, data_as_of=None))
        )
        is Truth.UNKNOWN
    )
    assert _freshness(context, daily(day="2026-10-02")) is Truth.FALSE
    assert _freshness(context, daily(day="2026-10-04")) is Truth.UNKNOWN
    assert _freshness(context, daily()) is Truth.TRUE
    old = replace(
        source, availability=replace(source.availability, observed_at="2026-09-01T00:00:00Z")
    )
    assert _freshness(context, old) is Truth.FALSE


def test_severe_shock_period_text_never_implies_end_date():
    """场景：严重异动有文本期间但没有认证解除时点；预期：持续阻断，认证结束后才可通过。"""
    active = row("stk_high_shock", raw={"period": "2026年9月期间"})
    assert _shock(inputs(active), A) is Truth.FALSE
    later = row(
        "stk_high_shock", effective_from="2026-10-01T00:00:00Z", effective_to="2026-10-04T00:00:00Z"
    )
    assert _shock(inputs(later), A) is Truth.FALSE
    ended = row("stk_high_shock", effective_from="2026-10-01T00:00:00Z", effective_to=NOW)
    assert _shock(inputs(ended, complete=COMPLETE), A) is Truth.TRUE
    assert _shock(inputs(ended), A) is Truth.UNKNOWN
    assert (
        _shock(inputs(complete=COMPLETE, reasons={"stk_high_shock": ("missing_scope",)}), A)
        is Truth.UNKNOWN
    )


@pytest.mark.parametrize(
    "weights,prices,day,complete",
    [
        ((0.2, 0.3), (8, 12), DAY, COMPLETE),
        ((-0.1, 1.1), (8, 12), DAY, COMPLETE),
        ((None, 1), (8, 12), DAY, COMPLETE),
        ((0.2, 0.8), (8, None), DAY, COMPLETE),
        ((0.2, 0.8), (8, 12), "2026-10-02", COMPLETE),
        ((0.2, 0.8), (8, 12), DAY, ()),
    ],
)
def test_chip_distribution_is_complete_and_comparable(weights, prices, day, complete):
    """场景：筹码权重不完整、非法、缺价或异日；预期：不发布供应商套牢比例。"""
    sources = [
        row(
            "cyq_chips",
            str(index),
            day=day,
            values={"percent": (weight, "fraction"), "price": (price, "元")},
        )
        for index, (weight, price) in enumerate(zip(weights, prices, strict=True))
    ]
    assert _overhang(inputs(*sources, complete=complete), A, daily()).value is None


def test_chip_variants_price_and_source_failures():
    """场景：完整权重仍存在来源变体或无当前价；预期：相关指标继续未知。"""
    chip = row(
        "cyq_chips",
        values={"price": (12, "元"), "percent": (1, "fraction")},
        raw={"business_variant_key": "unresolved"},
    )
    assert _overhang(inputs(chip, complete=COMPLETE), A, daily()).value is None
    clean = replace(chip, raw=chip.raw | {"business_variant_key": None})
    assert _overhang(inputs(clean, complete=COMPLETE), A, None).value is None
    assert (
        _overhang(
            inputs(clean, complete=COMPLETE, reasons={"cyq_chips": ("partial_distribution",)}),
            A,
            daily(),
        ).value
        is None
    )


def test_margin_and_turnover_do_not_join_other_dates():
    """场景：融资或市值不在同一天，或余额为负；预期：杠杆和换手独立降级。"""
    for day, balance in (("2026-10-02", 10), (DAY, -1)):
        context = inputs(
            row(
                "daily_basic",
                day=day,
                values={"circ_mv": (100, "元"), "turnover_rate": (-1, "fraction")},
            ),
            row("margin_detail", values={"rzye": (balance, "元")}),
        )
        metrics = _leverage(context, A, daily())
        assert metrics[0].value is None and metrics[1].value is None
    assert (
        _leverage(inputs(row("daily_basic", values={"circ_mv": (0, "元")})), A, daily())[0].value
        is None
    )


def test_block_trade_price_and_date_are_checked_per_trade():
    """场景：大宗成交价缺失或异日；预期：各笔保留未知折价，不外推交易条件。"""
    context = inputs(
        row("block_trade"), row("block_trade", "old", day="2026-10-02", values={"price": (9, "元")})
    )
    output = _block_trades(context, A, daily())
    assert len(output) == 2 and all(item["block_discount"] is None for item in output)
    assert _block_trades(context, A, None)[0]["reason"] == "block_price_or_date_unverified"
