"""P03S、P03M、P03L 三套独立排名的确定性离线测试。"""

from dataclasses import replace
from datetime import date, timedelta

import pytest
from test_research_candidates import research_inputs, research_row
from test_research_fundamental import research, statement

from tushare_entitlement_hub.equity_research.candidate_horizons import (
    LONG_TERM_PRODUCT_ID,
    MEDIUM_TERM_PRODUCT_ID,
    SHORT_TERM_PRODUCT_ID,
    CandidateDraft,
    FactorDefinition,
    _daily_window,
    _financial_years,
    _rank,
    _ratio_value,
    _stable,
    build_long_term_candidates,
    build_medium_term_candidates,
    build_short_term_candidates,
)
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.fundamental import build_fundamental
from tushare_entitlement_hub.product_foundation.contracts import ActionState, Truth


def ranking_parents(inputs, *, include_flow_metrics=True):
    """构造三类带真实指标的父结果。

    输入两个实体时输出等长 P04/P06/P07 元组；资金指标可关闭，用于验证短期榜失败关闭。
    """
    output = [[], [], []]
    for index, identity in enumerate(inputs.entity_ids):
        source_rows = inputs.rows("daily", identity) or inputs.rows("daily_basic", identity)
        references = source_rows[0].evidence("amount", "pb")
        output[0].append(
            make_result(
                inputs,
                "fundamental_dossier/v1",
                identity,
                metrics=(
                    metric("cash_margin", 0.10 + index * 0.01, "ratio", references),
                    metric("leverage", 0.50 - index * 0.01, "ratio", references),
                    metric("cash_cover", 0.30 + index * 0.01, "ratio", references),
                    metric("quality_score", 80, "score", references),
                ),
                gates=(gate("verified", Truth.TRUE, ""),),
                references=references,
                details={"comp_type": "1", "report_period": "2025-12-31"},
            )
        )
        output[1].append(
            make_result(
                inputs,
                "capital_governance_risks/v1",
                identity,
                metrics=(metric("R_supply", 20 - index, "score", references),),
                gates=(gate("verified", Truth.TRUE, ""),),
                references=references,
            )
        )
        output[2].append(
            make_result(
                inputs,
                "flow_feasibility/v1",
                identity,
                metrics=(
                    (metric("large_flow_ratio", 0.01 + index * 0.01, "ratio", references),)
                    if include_flow_metrics
                    else ()
                ),
                gates=(gate("verified", Truth.TRUE, ""),),
                references=references,
            )
        )
    return tuple(tuple(group) for group in output)


def with_daily_history(inputs):
    """为每个测试实体补 61 个有序日观察。

    输入当前截面夹具，输出保留原配置并带完整短期窗口的新 ResearchInputs。
    """
    rows = []
    first_day = date(2026, 7, 1)
    for entity_index, identity in enumerate(inputs.entity_ids):
        for offset in range(61):
            day = (first_day + timedelta(days=offset)).isoformat()
            rows.append(
                research_row(
                    identity,
                    "daily",
                    {
                        "close": 10,
                        "vol": 100,
                        "amount": 100 * (entity_index + 1),
                        "pct_chg": 0.001 * (entity_index + 1),
                    },
                    day=day,
                    record=f"daily-{entity_index}-{offset}",
                )
            )
    return replace(inputs, rows_by_api={**inputs.rows_by_api, "daily": tuple(rows)})


def test_short_term_ranking_requires_complete_history_and_flow():
    """场景：两个实体有 61 日行情和资金父指标；预期：独立短期榜产生可比较名次。"""
    inputs = with_daily_history(research_inputs(2, minimum=2))
    _, governance, flow = ranking_parents(inputs)
    results = build_short_term_candidates(inputs, (), governance, flow)
    assert len(results) == 2
    assert {item.envelope.product_id for item in results} == {SHORT_TERM_PRODUCT_ID}
    assert all(item.details["ranking_horizon"] == "short_term" for item in results)
    assert [item.envelope.rank for item in results] == [2, 1]
    assert all(item.envelope.score is not None for item in results)
    assert all(item.envelope.action_state is ActionState.NO_CONCLUSION for item in results)
    assert all(not item.envelope.decision_allowed for item in results)


def test_short_term_missing_history_does_not_fall_back_to_snapshot_score():
    """场景：只有一个交易日；预期：短期榜仍有结果，但历史门禁、分数和名次均为空。"""
    inputs = research_inputs(2, minimum=2)
    fundamental, governance, flow = ranking_parents(inputs)
    results = build_short_term_candidates(inputs, fundamental, governance, flow)
    assert all(item.envelope.score is None and item.envelope.rank is None for item in results)
    assert all("daily_history_window_incomplete" in item.envelope.gate_codes for item in results)


def test_medium_term_ranking_uses_current_financial_and_valuation_factors():
    """场景：两个实体具备同口径财务和正估值；预期：中期榜独立评分且第二实体排名更高。"""
    inputs = research_inputs(2, minimum=2)
    fundamental, governance, _ = ranking_parents(inputs)
    results = build_medium_term_candidates(inputs, fundamental, governance)
    assert len(results) == 2
    assert {item.envelope.product_id for item in results} == {MEDIUM_TERM_PRODUCT_ID}
    assert all(item.details["ranking_horizon"] == "medium_term" for item in results)
    assert [item.envelope.rank for item in results] == [2, 1]
    assert all(item.envelope.score is not None for item in results)


def annual_inputs(count=2, years=(2023, 2024, 2025)):
    """构造多个实体的连续年度三表、同期审计和当前 PB。

    输入默认两实体三年度，输出可供 P04 与 P03L 同时复算的 ResearchInputs。
    """
    rows = []
    for index in range(count):
        for year in years:
            report_period = f"{year}-12-31"
            publication = f"{year + 1}-04-30T00:00:00Z"
            fields = {
                "end_date": f"{year}1231",
                "ann_date": f"{year + 1}0430",
            }
            rows.extend(
                statement(
                    api,
                    index,
                    record=f"{api}-{index}-{year}",
                    fields=fields,
                    report=report_period,
                    public=publication,
                )
                for api in ("income_vip", "cashflow_vip", "balancesheet_vip", "fina_audit")
            )
        rows.append(statement("daily_basic", index, record=f"valuation-{index}"))
    return research(tuple(rows), count=count)


def test_long_term_ranking_requires_three_consecutive_annual_reports():
    """场景：两个实体有连续三年三表、同期审计和供给风险；预期：长期榜产生独立名次。"""
    inputs = annual_inputs()
    fundamental = build_fundamental(inputs)
    governance = []
    for index, identity in enumerate(inputs.entity_ids):
        references = inputs.rows("daily_basic", identity)[0].evidence("pb")
        governance.append(
            make_result(
                inputs,
                "capital_governance_risks/v1",
                identity,
                metrics=(metric("R_supply", 20 - index, "score", references),),
                gates=(gate("verified", Truth.TRUE, ""),),
                references=references,
            )
        )
    results = build_long_term_candidates(inputs, fundamental, tuple(governance))
    assert len(results) == 2
    assert {item.envelope.product_id for item in results} == {LONG_TERM_PRODUCT_ID}
    assert all(
        item.details["annual_report_periods"]
        == ["2023-12-31", "2024-12-31", "2025-12-31"]
        for item in results
    )
    assert all(item.envelope.rank is not None for item in results)


def test_long_term_gap_does_not_reuse_medium_term_rank():
    """场景：只有一个年度；预期：长期榜明确缺三年历史，不复用中期分数或名次。"""
    inputs = annual_inputs(years=(2025,))
    fundamental = build_fundamental(inputs)
    results = build_long_term_candidates(inputs, fundamental, ())
    assert all(item.envelope.score is None and item.envelope.rank is None for item in results)
    assert all(
        "three_year_financial_history_incomplete" in item.envelope.gate_codes
        for item in results
    )


def test_long_term_positive_cashflow_fraction_uses_selected_cashflow_evidence():
    """场景：三年中一年经营现金流为负；预期：所选现金流证据给出三分之二正值占比。"""
    inputs = annual_inputs()
    cashflow_rows = tuple(
        replace(
            row,
            raw=row.raw | {"n_cashflow_act": -20.0},
            numeric=row.numeric
            | {
                "n_cashflow_act": replace(
                    row.numeric["n_cashflow_act"], value=-20.0
                )
            },
        )
        if row.availability.report_period == "2024-12-31"
        else row
        for row in inputs.rows("cashflow_vip")
    )
    changed = replace(
        inputs,
        rows_by_api={**inputs.rows_by_api, "cashflow_vip": cashflow_rows},
    )
    fundamental = build_fundamental(changed)
    governance = ranking_parents(changed)[1]
    results = build_long_term_candidates(changed, fundamental, governance)
    assert all(item.metric("cash_margin_three_year_median") == 0.2 for item in results)
    assert all(
        item.metric("positive_operating_cashflow_year_fraction") == 2 / 3 for item in results
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing_date", "daily_history_window_incomplete"),
        ("duplicate_date", "daily_history_date_conflict"),
        ("invalid_return", "daily_return_unverified"),
        ("invalid_amount", "daily_amount_window_unverified"),
    ],
)
def test_short_term_window_rejects_incomplete_or_ambiguous_observations(case, reason):
    """场景：短期窗口缺日期、同日重复或数值不合格；预期：返回精确原因且不计算。"""
    inputs = with_daily_history(research_inputs(2, minimum=2))
    rows = list(inputs.rows("daily"))
    if case == "missing_date":
        rows[0] = replace(
            rows[0], availability=replace(rows[0].availability, data_as_of=None)
        )
    elif case == "duplicate_date":
        rows.append(replace(rows[0], raw=rows[0].raw | {"source_record_id": "duplicate"}))
    elif case == "invalid_return":
        rows[0] = replace(
            rows[0],
            numeric=rows[0].numeric
            | {"pct_chg": replace(rows[0].numeric["pct_chg"], value=-1)},
        )
    else:
        rows[60] = replace(
            rows[60],
            numeric=rows[60].numeric
            | {"amount": replace(rows[60].numeric["amount"], value=0)},
        )
    changed = replace(inputs, rows_by_api={**inputs.rows_by_api, "daily": tuple(rows)})
    values, _, actual, _ = _daily_window(changed, inputs.entity_ids[0])
    assert values == {} and actual == reason


def test_numeric_helpers_and_invalid_factor_weights_fail_closed():
    """场景：非有限数值、非比例指标及权重和错误；预期：分别拒绝或返回未知。"""
    inputs = research_inputs(1, minimum=2)
    with pytest.raises(ValueError, match="candidate_nonfinite_value"):
        _stable(float("nan"))
    assert _ratio_value(None) is None
    references = inputs.rows("daily", inputs.entity_ids[0])[0].evidence("amount")
    assert _ratio_value(metric("amount", 1, "元", references)) is None
    draft = CandidateDraft("entity", "profile", (), (), (), (), {})
    with pytest.raises(ValueError, match="candidate_factor_weights_invalid"):
        _rank(
            inputs,
            SHORT_TERM_PRODUCT_ID,
            "short_term",
            "test",
            (FactorDefinition("factor", 0.5),),
            (draft,),
        )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("nonconsecutive", "three_year_financial_history_not_consecutive"),
        ("ambiguous", "three_year_financial_scope_ambiguous"),
        ("bad_audit", "three_year_financial_quality_unverified"),
        ("cashflow_record_collision", "three_year_financial_quality_unverified"),
    ],
)
def test_long_term_history_rejects_invalid_annual_windows(case, reason):
    """场景：年度跳跃、同年多口径或审计缺失；预期：不构造长期财务窗口。"""
    years = (2022, 2024, 2025) if case == "nonconsecutive" else (2023, 2024, 2025)
    inputs = annual_inputs(count=1, years=years)
    if case == "ambiguous":
        extra = statement(
            "income_vip",
            0,
            record="income-alternate",
            fields={"end_date": "20251231", "ann_date": "20260430", "report_type": "2"},
            report="2025-12-31",
            public="2026-04-30T00:00:00Z",
        )
        inputs = replace(
            inputs,
            config=replace(inputs.config, certified_report_types=("1", "2")),
            rows_by_api={
                **inputs.rows_by_api,
                "income_vip": inputs.rows("income_vip") + (extra,),
            },
        )
    elif case == "bad_audit":
        inputs = replace(inputs, rows_by_api={**inputs.rows_by_api, "fina_audit": ()})
    elif case == "cashflow_record_collision":
        duplicate = statement(
            "cashflow_vip",
            0,
            record="cashflow_vip-0-2023",
            fields={"end_date": "20221231", "ann_date": "20230430"},
            report="2022-12-31",
            public="2023-04-30T00:00:00Z",
        )
        inputs = replace(
            inputs,
            rows_by_api={
                **inputs.rows_by_api,
                "cashflow_vip": inputs.rows("cashflow_vip") + (duplicate,),
            },
        )
    dossiers, actual = _financial_years(inputs, inputs.entity_ids[0])
    assert dossiers == () and actual == reason
