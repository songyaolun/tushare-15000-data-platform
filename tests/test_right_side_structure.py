"""回调后右侧结构、价格口径与初始策略边界测试。"""

from dataclasses import replace

import pytest
from right_side_fixture import right_side_inputs

from tushare_entitlement_hub.equity_research.right_side_policy import RightSidePolicy
from tushare_entitlement_hub.equity_research.right_side_structure import (
    observe_structure,
    price_window,
    structure_at,
)


@pytest.mark.parametrize("pattern", ["breakout", "retest"])
def test_breakout_and_retest_report_price_risk_and_source_evidence(pattern):
    """场景：回调后当日突破或突破后缩量回踩转强；预期：结构确认并给出来源价格与风险。"""
    inputs = right_side_inputs(1, pattern=pattern)
    observation = observe_structure(inputs, inputs.entity_ids[0])
    assert observation.reason is None
    assert observation.details["pattern"] == pattern
    assert observation.values["entry_risk"] < 0.08
    assert observation.values["structural_reward_risk"] >= 2
    assert observation.values["breakout_extension"] < 0.03
    assert observation.values["historical_resistance"] > observation.values["entry_price"]
    assert len(observation.references) == 61
    assert all(
        {"close", "high", "low", "pct_chg"} <= set(ref.fields) for ref in observation.references
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing_date", "right_side_date_missing"),
        ("duplicate", "right_side_date_conflict"),
        ("short", "right_side_history_incomplete"),
        ("price", "right_side_price_unverified"),
        ("price_unit", "right_side_price_unverified"),
        ("price_currency", "right_side_price_unverified"),
        ("ohlc", "right_side_ohlc_invalid"),
        ("amount", "right_side_amount_unverified"),
        ("return", "right_side_return_unverified"),
        ("return_unit", "right_side_return_unverified"),
        ("linked_overflow", "right_side_linked_price_invalid"),
    ],
)
def test_price_window_rejects_ambiguous_dates_and_unverified_values(case, reason):
    """场景：窗口或单位、币种、价格、金额、收益存在缺口；预期：返回原因而不计算结构。"""
    inputs = right_side_inputs(1)
    rows = list(inputs.rows("daily"))
    first = rows[0]
    if case == "missing_date":
        rows[0] = replace(first, availability=replace(first.availability, data_as_of=None))
    elif case == "duplicate":
        rows.append(first)
    elif case == "short":
        rows.pop()
    else:
        field = (
            "close"
            if case in {"price", "price_unit", "price_currency"}
            else ("high" if case == "ohlc" else "amount" if case == "amount" else "pct_chg")
        )
        numeric = dict(first.numeric)
        changes = (
            {"value": None, "reason_codes": ("missing_price",)}
            if case == "price"
            else (
                {"unit": "USD"}
                if case == "price_unit"
                else {"currency": "USD"}
                if case == "price_currency"
                else {"value": 1}
                if case == "ohlc"
                else {"value": 0}
                if case == "amount"
                else {"dimension": "price", "currency": "CNY"}
                if case == "return_unit"
                else {"value": -1}
            )
        )
        if case == "linked_overflow":
            for index, row in enumerate(rows):
                rows[index] = replace(
                    row,
                    numeric=row.numeric
                    | {"pct_chg": replace(row.numeric["pct_chg"], value=-0.9999999999999999)},
                )
        else:
            if case in {"price_unit", "price_currency", "return_unit"}:
                certificate_changes = (
                    {"source_unit": "USD", "target_unit": "USD"}
                    if case == "price_unit"
                    else (
                        {"currency": "USD"}
                        if case == "price_currency"
                        else {"dimension": "price", "currency": "CNY"}
                    )
                )
                changes["certification"] = replace(
                    numeric[field].certification, **certificate_changes
                )
            numeric[field] = replace(numeric[field], **changes)
            rows[0] = replace(first, numeric=numeric)
    changed = replace(inputs, rows_by_api={"daily": tuple(rows)})
    bars, _, actual = price_window(changed, inputs.entity_ids[0])
    assert bars == () and actual == reason
    assert observe_structure(changed, inputs.entity_ids[0]).reason == reason


def test_share_split_does_not_create_artificial_pullback_or_breakout():
    """场景：前60日裸价加倍而供应商收益不变；预期：链接价、确认点和风险与原窗口相同。"""
    inputs = right_side_inputs(1)
    original = observe_structure(inputs, inputs.entity_ids[0])
    rows = tuple(
        replace(
            row,
            numeric=row.numeric
            | {
                name: replace(row.numeric[name], value=row.numeric[name].value * 2)
                for name in ("close", "high", "low")
            },
        )
        if index < 60
        else row
        for index, row in enumerate(inputs.rows("daily"))
    )
    changed = replace(inputs, rows_by_api=inputs.rows_by_api | {"daily": rows})
    observation = observe_structure(changed, inputs.entity_ids[0])
    assert observation.values == original.values
    assert observation.reason is None


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("no_pullback", "right_side_pullback_missing"),
        ("no_higher_low", "right_side_higher_low_missing"),
        ("wide_base", "right_side_base_too_wide"),
        ("left_side", "right_side_trend_unconfirmed"),
        ("no_breakout", "right_side_breakout_unconfirmed"),
        ("no_amount", "right_side_amount_confirmation_missing"),
        ("bad_retest", "right_side_retest_unconfirmed"),
        ("extended", "right_side_entry_extended"),
        ("risk", "right_side_entry_risk_excessive"),
        ("late", "right_side_reward_risk_insufficient"),
    ],
)
def test_structure_failure_conditions_are_independent_and_explained(case, reason):
    """场景：各结构条件独立失败；预期：返回对应原因且失败条件不被其他强项抵消。"""
    inputs = right_side_inputs(
        1,
        pattern="retest"
        if case == "bad_retest"
        else case
        if case in {"left_side", "extended", "late"}
        else "breakout",
    )
    bars, _, _ = price_window(inputs, inputs.entity_ids[0])
    policy = inputs.config.right_side
    index = len(bars) - 3 if case == "bad_retest" else len(bars) - 1
    if case == "no_pullback":
        policy = replace(policy, minimum_pullback=0.5)
    elif case == "no_higher_low":
        policy = replace(policy, minimum_higher_low=0.5)
    elif case == "wide_base":
        policy = replace(policy, maximum_base_range=0.001)
    elif case == "no_breakout":
        bars = (*bars[:-1], replace(bars[-1], close=10.01))
    elif case == "no_amount":
        bars = (*bars[:-1], replace(bars[-1], amount=50))
    elif case == "bad_retest":
        bars = (*bars[:-1], replace(bars[-1], low=9.5))
    elif case == "risk":
        policy = replace(policy, maximum_entry_risk=0.001)
    observation = structure_at(bars, index, policy)
    assert observation.reason == reason
    assert observation.details["structure_checks"][reason] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"history_observations": 20},
        {"consolidation_sessions": 2},
        {"confirmation_sessions": 0},
        {"consolidation_sessions": 40},
        {"consolidation_sessions": 28},
        {"consolidation_sessions": 30},
        {"minimum_pullback": 0},
        {"maximum_base_range": 0},
        {"maximum_breakout_extension": 0},
        {"maximum_entry_risk": 0},
        {"invalidation_buffer": 1},
        {"minimum_amount_ratio": 0.5},
    ],
)
def test_invalid_right_side_policy_is_rejected(changes):
    """场景：窗口无法计算或阈值退化；预期：配置构造立即拒绝，不放宽为默认规则。"""
    with pytest.raises(ValueError, match="invalid_right_side_policy"):
        RightSidePolicy(**changes)


def test_minimum_peak_zone_remains_before_pullback_and_consolidation():
    """场景：61条窗口、27日整理和5日确认；预期：最早突破仍有独立历史峰区，不抛异常。"""
    inputs = right_side_inputs(1)
    inputs = replace(
        inputs,
        config=replace(inputs.config, right_side=RightSidePolicy(consolidation_sessions=27)),
    )
    bars, _, reason = price_window(inputs, inputs.entity_ids[0])
    assert reason is None
    earliest_index = len(bars) - inputs.config.right_side.confirmation_sessions
    observation = structure_at(bars, earliest_index, inputs.config.right_side)
    assert observation.values["historical_resistance"] == pytest.approx(bars[0].high)
    assert observation.details["base_start_date"] == bars[29].day


def test_policy_roundtrip_and_configuration_identity_include_thresholds():
    """场景：调整最大入场风险；预期：JSON往返一致，运行身份变化且不复用旧结果。"""
    from tushare_entitlement_hub.equity_research.models import ResearchConfig
    from tushare_entitlement_hub.equity_research.storage import run_identity

    config = ResearchConfig()
    assert ResearchConfig.from_json(config.to_json()) == config
    changed = replace(config, right_side=replace(config.right_side, maximum_entry_risk=0.04))
    assert run_identity("reference", "2026-10-03T00:00:00Z", config, ()) != run_identity(
        "reference", "2026-10-03T00:00:00Z", changed, ()
    )


@pytest.mark.parametrize(
    "field,dimension,reason",
    [
        ("close", "money", "right_side_price_unverified"),
        ("amount", "price", "right_side_amount_unverified"),
    ],
)
def test_price_and_amount_dimensions_are_not_interchangeable(field, dimension, reason):
    """场景：元金额被认证为价格或反之；预期：同币种同单位也不能跨量纲计算。"""
    inputs = right_side_inputs(1)
    first, *others = inputs.rows("daily")
    fact = first.numeric[field]
    changed_fact = replace(
        fact, dimension=dimension, certification=replace(fact.certification, dimension=dimension)
    )
    changed = replace(first, numeric=first.numeric | {field: changed_fact})
    inputs = replace(inputs, rows_by_api={"daily": (changed, *others)})
    assert price_window(inputs, inputs.entity_ids[0])[2] == reason


def test_nonfinite_structure_ratios_return_unknown_instead_of_crashing():
    """场景：有限输入的极端阻力/入场比例溢出；预期：失败关闭，不保存无限值。"""
    inputs = right_side_inputs(1)
    bars, _, _ = price_window(inputs, inputs.entity_ids[0])
    scaled = tuple(
        replace(bar, close=bar.close * 1e-300, high=bar.high * 1e-300, low=bar.low * 1e-300)
        for bar in bars
    )
    scaled = (replace(scaled[0], high=1e308), *scaled[1:])
    observation = structure_at(scaled, 60, inputs.config.right_side)
    assert observation.reason == "right_side_calculation_invalid"
    assert observation.values == {}
