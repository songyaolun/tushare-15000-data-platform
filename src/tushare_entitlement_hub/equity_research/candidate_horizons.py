"""P03 的短期、中期和长期独立候选排名。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, fsum, isfinite, prod, sqrt
from statistics import median
from typing import Any, cast

from ..product_foundation.contracts import (
    ActionState,
    EvidenceRef,
    FeatureContribution,
    ProductState,
    QualityGateResult,
    Truth,
)
from ..product_foundation.numeric import price_gate
from ..store import encode
from .common import gate, latest, make_result, metric, percentiles, unique_evidence
from .fundamental import _context, _dossier
from .models import CANDIDATE_RANKING_PRODUCT_IDS, Metric, ResearchInputs, ResearchResult

SHORT_TERM_PRODUCT_ID, MEDIUM_TERM_PRODUCT_ID, LONG_TERM_PRODUCT_ID = (
    CANDIDATE_RANKING_PRODUCT_IDS
)
SHORT_TERM_REQUIRED_OBSERVATIONS = 61
LONG_TERM_REQUIRED_YEARS = 3


@dataclass(frozen=True)
class FactorDefinition:
    """声明一个排名因子的方向和权重。

    例如 ``FactorDefinition('volatility_20', 0.2, True)`` 表示低波动取得更高分位。
    """

    name: str
    weight: float
    lower_is_better: bool = False


@dataclass(frozen=True)
class CandidateDraft:
    """保存排名前的单实体特征、门禁与血缘。

    例如特征缺失时仍保留该实体草稿，最终结果的分数和名次保持空值。
    """

    entity_id: str
    profile: str
    metrics: tuple[Metric, ...]
    gates: tuple[QualityGateResult, ...]
    references: tuple[EvidenceRef, ...]
    parents: tuple[ResearchResult, ...]
    details: dict[str, Any]


SHORT_TERM_FACTORS = (
    FactorDefinition("return_5_sessions", 0.20),
    FactorDefinition("return_20_sessions", 0.25),
    FactorDefinition("return_60_sessions", 0.10),
    FactorDefinition("volatility_20_sessions", 0.20, True),
    FactorDefinition("median_amount_20_sessions", 0.15),
    FactorDefinition("large_flow_ratio", 0.10),
)
MEDIUM_TERM_FACTORS = (
    FactorDefinition("cash_margin", 0.30),
    FactorDefinition("leverage", 0.20, True),
    FactorDefinition("cash_cover", 0.15),
    FactorDefinition("inverse_pe_ttm", 0.20),
    FactorDefinition("inverse_pb", 0.15),
)
LONG_TERM_FACTORS = (
    FactorDefinition("cash_margin_three_year_median", 0.25),
    FactorDefinition("cash_cover_three_year_minimum", 0.20),
    FactorDefinition("leverage_three_year_maximum", 0.20, True),
    FactorDefinition("positive_operating_cashflow_year_fraction", 0.15),
    FactorDefinition("cash_margin_three_year_range", 0.10, True),
    FactorDefinition("inverse_pb", 0.10),
)


def _parent_index(results: tuple[ResearchResult, ...]) -> dict[str, ResearchResult | None]:
    """按实体索引唯一父结果，重复实体返回空标记。

    输入实体 A 的两条父结果时输出 ``{'A': None}``；单条父结果保持原对象。
    """
    indexed: dict[str, ResearchResult | None] = {}
    for result in results:
        entity_id = result.envelope.entity_id
        indexed[entity_id] = None if entity_id in indexed else result
    return indexed


def _parent_metric(parent: ResearchResult | None, name: str) -> Metric | None:
    """读取父结果中的唯一可用指标。

    输入含一个 ``quality_score`` 的父结果时返回该指标；缺失、重复或未知均返回 ``None``。
    """
    if parent is None:
        return None
    matches = [item for item in parent.metrics if item.name == name and item.status == "available"]
    return matches[0] if len(matches) == 1 and matches[0].value is not None else None


def _parent_checks(
    inputs: ResearchInputs,
    identity: str,
    groups: tuple[tuple[str, dict[str, ResearchResult | None], bool], ...],
) -> tuple[tuple[ResearchResult, ...], tuple[QualityGateResult, ...]]:
    """核验父结果身份、截止和硬门禁，允许可选父结果仅传播明确失败。

    例如必需的 P06 缢失时增加 UNKNOWN 门禁；可选 P04 缢失时不阻断短期排名。
    """
    parents: list[ResearchResult] = []
    checks: list[QualityGateResult] = []
    for label, indexed, required in groups:
        parent = indexed.get(identity)
        if parent is None or parent.envelope.analysis_as_of != inputs.analysis_as_of:
            if required:
                checks.append(gate(label + "_parent", Truth.UNKNOWN, label + "_parent_unavailable"))
            continue
        parents.append(parent)
        checks.append(gate(label + "_parent", Truth.TRUE, ""))
        checks.extend(
            replace(item, gate_id=label + "/" + item.gate_id)
            for item in parent.envelope.quality_gates
            if required or item.result is Truth.FALSE
        )
    return tuple(parents), tuple(checks)


def _stable(value: float) -> float:
    """把派生浮点数固定到十二位小数，避免解释器求和实现改变结果字节。

    输入 ``0.1 + 0.2`` 返回 ``0.3``；非有限值抛出 ``ValueError``。
    """
    if not isfinite(value):
        raise ValueError("candidate_nonfinite_value")
    return round(value, 12)


def _ratio_value(metric_value: Metric | None) -> float | None:
    """读取比例或分数指标的有限值。

    输入 available 的 ratio/score 指标返回数值；未知、非有限值或其他单位返回 ``None``。
    """
    if (
        metric_value is None
        or metric_value.value is None
        or metric_value.unit not in {"ratio", "fraction", "score"}
        or not isfinite(metric_value.value)
    ):
        return None
    return metric_value.value


def _daily_window(
    inputs: ResearchInputs, identity: str
) -> tuple[dict[str, float | None], tuple[EvidenceRef, ...], str | None, tuple[str, ...]]:
    """从 61 个唯一观察日计算 5/20/60 日收益、20 日波动和成交额中位数。

    输入 61 条已认证日收益时返回五项数值；日期重复、单位不合格或不足 61 条时
    返回空指标及明确原因。收益使用供应商日涨跌幅复利，不声称已完成复权价重建。
    """
    rows_by_date: dict[str, list[Any]] = {}
    for row in inputs.rows("daily", identity):
        if row.availability.data_as_of is not None:
            rows_by_date.setdefault(row.availability.data_as_of, []).append(row)
    if any(len(rows) != 1 for rows in rows_by_date.values()):
        return {}, (), "daily_history_date_conflict", tuple(sorted(rows_by_date))
    ordered = [rows_by_date[day][0] for day in sorted(rows_by_date)]
    dates = tuple(row.availability.data_as_of or "" for row in ordered)
    references = unique_evidence(
        reference
        for row in ordered[-SHORT_TERM_REQUIRED_OBSERVATIONS:]
        for reference in row.evidence("pct_chg", "amount", "trade_date")
    )
    if len(ordered) < SHORT_TERM_REQUIRED_OBSERVATIONS:
        return {}, references, "daily_history_window_incomplete", dates
    selected = ordered[-SHORT_TERM_REQUIRED_OBSERVATIONS:]
    returns = []
    for row in selected:
        fact = row.numeric.get("pct_chg")
        value = row.number("pct_chg")
        if (
            fact is None
            or fact.dimension != "ratio"
            or fact.unit not in {"ratio", "fraction"}
            or value is None
            or value <= -1
        ):
            return {}, references, "daily_return_unverified", dates[-61:]
        returns.append(value)
    amounts = []
    for row in selected[-20:]:
        fact = row.numeric.get("amount")
        value = row.number("amount", "元")
        if fact is None or fact.currency != "CNY" or value is None or value <= 0:
            return {}, references, "daily_amount_window_unverified", dates[-61:]
        amounts.append(value)

    def compounded(width: int) -> float:
        """复合最近若干条已认证简单收益；例如两日 10% 与 -10% 得 -1%。"""
        return _stable(prod(1 + value for value in returns[-width:]) - 1)

    recent = returns[-20:]
    center = fsum(recent) / len(recent)
    volatility = _stable(sqrt(fsum((value - center) ** 2 for value in recent) / len(recent)))
    return (
        {
            "return_5_sessions": compounded(5),
            "return_20_sessions": compounded(20),
            "return_60_sessions": compounded(60),
            "volatility_20_sessions": volatility,
            "median_amount_20_sessions": _stable(float(median(amounts))),
        },
        references,
        None,
        dates[-61:],
    )


def _financial_years(
    inputs: ResearchInputs, identity: str
) -> tuple[tuple[ResearchResult, ...], str | None]:
    """构造最近三个连续完整年度的同口径财务档案。

    输入 2023、2024、2025 三个年末且均通过 P04 门禁的三表与审计，返回三个档案；
    缺年、混合报表类型或任一年度门禁未通过时返回空元组和原因。
    """
    statement_apis = {
        "income",
        "income_vip",
        "cashflow",
        "cashflow_vip",
        "balancesheet",
        "balancesheet_vip",
    }
    contexts = {
        context
        for api in statement_apis
        for row in inputs.rows(api, identity)
        if (context := _context(row)) is not None
        and context[0].endswith("-12-31")
        and context[1] in inputs.config.certified_report_types
        and context[2] == "1"
    }
    periods = sorted({context[0] for context in contexts})
    if len(periods) < LONG_TERM_REQUIRED_YEARS:
        return (), "three_year_financial_history_incomplete"
    selected_periods = periods[-LONG_TERM_REQUIRED_YEARS:]
    years = [int(period[:4]) for period in selected_periods]
    if years != list(range(years[0], years[0] + LONG_TERM_REQUIRED_YEARS)):
        return (), "three_year_financial_history_not_consecutive"
    selected_contexts = []
    for period in selected_periods:
        matches = [context for context in contexts if context[0] == period]
        if len(matches) != 1:
            return (), "three_year_financial_scope_ambiguous"
        selected_contexts.append(matches[0])
    dossiers = []
    for context in selected_contexts:
        rows_by_api = dict(inputs.rows_by_api)
        for api in statement_apis:
            rows_by_api[api] = tuple(
                row for row in inputs.rows(api, identity) if _context(row) == context
            )
        rows_by_api["fina_audit"] = tuple(
            row
            for row in inputs.rows("fina_audit", identity)
            if row.availability.report_period == context[0]
        )
        dossier = _dossier(replace(inputs, rows_by_api=rows_by_api), identity)
        required = tuple(
            _parent_metric(dossier, name) for name in ("cash_margin", "leverage", "cash_cover")
        )
        if any(item is None for item in required) or any(
            item.hard and item.result is not Truth.TRUE
            for item in dossier.envelope.quality_gates
        ):
            return (), "three_year_financial_quality_unverified"
        cashflow_references = tuple(
            reference
            for reference in dossier.envelope.evidence_refs
            if reference.source_api in {"cashflow", "cashflow_vip"}
            and "n_cashflow_act" in reference.fields
        )
        cashflow_rows = tuple(
            row
            for api in ("cashflow", "cashflow_vip")
            for row in inputs.rows(api, identity)
            if any(
                reference.source_api == api
                and reference.source_record_id == row.availability.source_record_id
                for reference in cashflow_references
            )
        )
        operating_cashflows = tuple(
            value
            for row in cashflow_rows
            if (value := row.number("n_cashflow_act", "元")) is not None
        )
        if len(cashflow_rows) != 1 or len(operating_cashflows) != 1:
            return (), "three_year_financial_quality_unverified"
        dossiers.append(
            replace(
                dossier,
                metrics=dossier.metrics
                + (
                    metric(
                        "operating_cashflow",
                        operating_cashflows[0],
                        "元",
                        cashflow_references,
                    ),
                ),
            )
        )
    return tuple(dossiers), None


def _rank(
    inputs: ResearchInputs,
    product_id: str,
    horizon: str,
    holding_period: str,
    definitions: tuple[FactorDefinition, ...],
    drafts: tuple[CandidateDraft, ...],
) -> tuple[ResearchResult, ...]:
    """按产品和档位独立计算因子分位、总分与名次。

    输入同一行业 30 个完整草稿时输出该产品的 30 个独立名次；任一必需因子未知，
    该实体的总分和名次为空。初始策略只允许人工研究，不开放交易决策。
    """
    if abs(fsum(item.weight for item in definitions) - 1) > 1e-12:
        raise ValueError("candidate_factor_weights_invalid")
    by_profile: dict[str, list[CandidateDraft]] = {}
    for draft in drafts:
        by_profile.setdefault(draft.profile, []).append(draft)
    output: list[ResearchResult] = []
    for profile, group in sorted(by_profile.items()):
        eligible = {
            draft.entity_id: all(
                check.result is Truth.TRUE for check in draft.gates if check.hard
            )
            and all(item.status == "available" for item in draft.metrics)
            for draft in group
        }
        factor_ranks = {
            definition.name: percentiles(
                inputs,
                {
                    draft.entity_id: (
                        next(item.value for item in draft.metrics if item.name == definition.name)
                        if eligible[draft.entity_id]
                        else None
                    )
                    for draft in group
                },
                product_id + "/" + profile + "/" + definition.name,
                lower_is_better=definition.lower_is_better,
            )
            for definition in definitions
        }
        preliminary: list[ResearchResult] = []
        for draft in group:
            components = []
            for definition in definitions:
                raw_metric = next(item for item in draft.metrics if item.name == definition.name)
                ranked = factor_ranks[definition.name][draft.entity_id]
                contribution = (
                    _stable(100 * definition.weight * ranked.percentile)
                    if ranked.percentile is not None
                    else None
                )
                components.append(
                    FeatureContribution(
                        feature_id=definition.name,
                        raw_value=raw_metric.value,
                        unit=raw_metric.unit,
                        transformed_value=ranked.percentile,
                        weight=definition.weight,
                        contribution=contribution,
                        transform=(
                            "lower_peer_percentile"
                            if definition.lower_is_better
                            else "peer_percentile"
                        ),
                        peer_level=ranked.peer_level,
                        missing_reason=(
                            "candidate_factor_unavailable"
                            if ranked.percentile is None
                            else None
                        ),
                        evidence_refs=raw_metric.evidence_refs,
                        reason_codes=ranked.reason_codes,
                    )
                )
            result = make_result(
                inputs,
                product_id,
                draft.entity_id,
                metrics=draft.metrics,
                gates=draft.gates,
                contributions=components,
                references=draft.references,
                contrary=(
                    reference
                    for check in draft.gates
                    if not check.passed
                    for reference in check.evidence_refs
                ),
                parents=(parent.envelope.result_id for parent in draft.parents),
                profile=profile,
                details={
                    **draft.details,
                    "ranking_horizon": horizon,
                    "holding_period": holding_period,
                    "factor_weights": {
                        item.name: item.weight for item in definitions
                    },
                    "initial_policy_unvalidated": True,
                    "historical_pit_certified": False,
                },
            )
            preliminary.append(result)
        ranks = percentiles(
            inputs,
            {result.envelope.entity_id: result.envelope.score for result in preliminary},
            product_id + "/" + profile + "/final",
        )
        for result in preliminary:
            ranked = ranks[result.envelope.entity_id]
            rank_value = (
                1 + (1 - ranked.percentile) * (ranked.peer_count - 1)
                if ranked.percentile is not None
                else None
            )
            details = result.details | {
                "rank_peer_level": ranked.peer_level,
                "rank_reason_codes": list(ranked.reason_codes),
                "research_capacity": min(
                    inputs.config.thresholds.top_k,
                    ceil(ranked.peer_count * 0.2),
                ),
            }
            checks = result.envelope.quality_gates + (
                gate(
                    "ranking_sample",
                    Truth.TRUE if rank_value is not None else Truth.UNKNOWN,
                    "insufficient_rank_sample",
                ),
            )
            envelope = replace(
                result.envelope,
                rank=rank_value,
                peer_count=ranked.peer_count,
                quality_gates=checks,
                decision_allowed=False,
                action_state=ActionState.NO_CONCLUSION,
                state=(
                    ProductState.INCOMPLETE
                    if rank_value is None and result.envelope.state is not ProductState.BLOCKED
                    else result.envelope.state
                ),
                gate_codes=tuple(
                    sorted(set(result.envelope.gate_codes) | set(ranked.reason_codes))
                ),
            )
            output.append(replace(result, envelope=envelope, details_json=encode(details)))
    return tuple(sorted(output, key=lambda item: item.envelope.entity_id))


def build_short_term_candidates(
    inputs: ResearchInputs,
    fundamental: tuple[ResearchResult, ...],
    governance: tuple[ResearchResult, ...],
    flow: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """构建 1 至 20 个交易日观察用途的短期排名。

    输入 61 个完整日收益观察及 P06/P07 父结果时输出动量、波动、流动性和资金排名；
    窗口不完整时仍输出实体结果，但分数、名次和决策资格均为空。
    """
    fundamental_index, governance_index, flow_index = (
        _parent_index(group) for group in (fundamental, governance, flow)
    )
    drafts = []
    selected_date = max(
        (row.availability.data_as_of or "" for row in inputs.rows("daily")), default=""
    )
    for identity in inputs.entity_ids:
        parents, checks = _parent_checks(
            inputs,
            identity,
            (
                ("fundamental", fundamental_index, False),
                ("governance", governance_index, True),
                ("flow", flow_index, True),
            ),
        )
        values, history_refs, reason, dates = _daily_window(inputs, identity)
        flow_parent = flow_index.get(identity)
        flow_metric = _parent_metric(flow_parent, "large_flow_ratio")
        values["large_flow_ratio"] = _ratio_value(flow_metric)
        metrics = tuple(
            metric(
                definition.name,
                values.get(definition.name),
                "元" if definition.name == "median_amount_20_sessions" else "ratio",
                history_refs
                if definition.name != "large_flow_ratio"
                else (() if flow_metric is None else flow_metric.evidence_refs),
                ()
                if values.get(definition.name) is not None
                else ((reason or "short_term_flow_unavailable"),),
            )
            for definition in SHORT_TERM_FACTORS
        )
        daily = latest(inputs.rows("daily", identity))
        checks += (
            gate(
                "short_term_history",
                Truth.TRUE if reason is None else Truth.UNKNOWN,
                reason or "daily_history_window_incomplete",
                references=history_refs,
            ),
            replace(
                price_gate(daily.raw, "price_rank"),
                evidence_refs=() if daily is None else daily.evidence("close", "vol", "amount"),
            )
            if daily is not None
            else gate("price_short_term_rank", Truth.UNKNOWN, "price_unavailable"),
            gate(
                "short_term_market_date",
                Truth.TRUE
                if daily is not None and daily.availability.data_as_of == selected_date
                else Truth.UNKNOWN,
                "market_date_mismatch",
            ),
        )
        references = unique_evidence(
            history_refs
            + tuple(
                reference
                for parent in parents
                for reference in parent.envelope.evidence_refs
            )
        )
        drafts.append(
            CandidateDraft(
                identity,
                "short_term/observed_61_sessions",
                metrics,
                checks,
                references,
                parents,
                {
                    "formula": (
                        "100*(0.20*P(r5)+0.25*P(r20)+0.10*P(r60)"
                        "+0.20*P_low(vol20)+0.15*P(amount20)+0.10*P(flow))"
                    ),
                    "return_basis": "provider_daily_pct_chg_compounded",
                    "required_observations": SHORT_TERM_REQUIRED_OBSERVATIONS,
                    "observed_dates": list(dates),
                },
            )
        )
    return _rank(
        inputs,
        SHORT_TERM_PRODUCT_ID,
        "short_term",
        "1_to_20_trading_sessions",
        SHORT_TERM_FACTORS,
        tuple(drafts),
    )


def build_medium_term_candidates(
    inputs: ResearchInputs,
    fundamental: tuple[ResearchResult, ...],
    governance: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """构建最近报告期与当前估值结合的中期排名。

    输入 P04 三项财务比率、P06 风险门禁及正 PE/PB，输出五因子独立排名；
    任一必需项缺失都保留 unknown，不退回旧 P03 市场档。
    """
    fundamental_index, governance_index = (
        _parent_index(group) for group in (fundamental, governance)
    )
    drafts = []
    selected_date = max(
        (row.availability.data_as_of or "" for row in inputs.rows("daily_basic")), default=""
    )
    for identity in inputs.entity_ids:
        parents, checks = _parent_checks(
            inputs,
            identity,
            (
                ("fundamental", fundamental_index, True),
                ("governance", governance_index, True),
            ),
        )
        fundamental_parent = fundamental_index.get(identity)
        valuation = latest(inputs.rows("daily_basic", identity))
        pe = None if valuation is None else valuation.number("pe_ttm", "ratio")
        pb = None if valuation is None else valuation.number("pb", "ratio")
        parent_metrics = {
            name: _parent_metric(fundamental_parent, name)
            for name in ("cash_margin", "leverage", "cash_cover")
        }
        values = {
            name: _ratio_value(value) for name, value in parent_metrics.items()
        }
        values.update(
            inverse_pe_ttm=(1 / pe if pe is not None and pe > 0 and isfinite(1 / pe) else None),
            inverse_pb=(1 / pb if pb is not None and pb > 0 and isfinite(1 / pb) else None),
        )
        valuation_refs = (
            () if valuation is None else valuation.evidence("pe_ttm", "pb", "trade_date")
        )
        metrics = tuple(
            metric(
                definition.name,
                values.get(definition.name),
                "ratio",
                valuation_refs
                if definition.name.startswith("inverse_")
                else (
                    ()
                    if parent_metrics.get(definition.name) is None
                    else cast(Metric, parent_metrics[definition.name]).evidence_refs
                ),
                ()
                if values.get(definition.name) is not None
                else ("medium_term_factor_unavailable",),
            )
            for definition in MEDIUM_TERM_FACTORS
        )
        checks += (
            gate(
                "medium_term_valuation",
                Truth.TRUE
                if pe is not None and pe > 0 and pb is not None and pb > 0
                else Truth.FALSE
                if pe is not None and pe <= 0 or pb is not None and pb <= 0
                else Truth.UNKNOWN,
                "positive_valuation_unavailable",
                references=valuation_refs,
            ),
            gate(
                "medium_term_market_date",
                Truth.TRUE
                if valuation is not None and valuation.availability.data_as_of == selected_date
                else Truth.UNKNOWN,
                "market_date_mismatch",
            ),
        )
        company_type = (
            fundamental_parent.details.get("comp_type")
            if fundamental_parent is not None
            else "unknown"
        )
        references = unique_evidence(
            valuation_refs
            + tuple(
                reference
                for parent in parents
                for reference in parent.envelope.evidence_refs
            )
        )
        drafts.append(
            CandidateDraft(
                identity,
                "medium_term/comp_type_" + str(company_type or "unknown"),
                metrics,
                checks,
                references,
                parents,
                {
                    "formula": (
                        "100*(0.30*P(cash_margin)+0.20*P_low(leverage)"
                        "+0.15*P(cash_cover)+0.20*P(1/pe_ttm)+0.15*P(1/pb))"
                    ),
                    "report_period": None
                    if fundamental_parent is None
                    else fundamental_parent.details.get("report_period"),
                    "valuation_date": None
                    if valuation is None
                    else valuation.availability.data_as_of,
                },
            )
        )
    return _rank(
        inputs,
        MEDIUM_TERM_PRODUCT_ID,
        "medium_term",
        "one_to_three_month_research_review",
        MEDIUM_TERM_FACTORS,
        tuple(drafts),
    )


def build_long_term_candidates(
    inputs: ResearchInputs,
    fundamental: tuple[ResearchResult, ...],
    governance: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """构建要求三个连续完整年度的长期配置排名。

    输入同口径三年三表、同期标准审计、当前正 PB 和 P06 供给压力时输出六因子排名；
    正式数据缺少任一年度时结果保持 unknown，不用季度记录拼接成年报。
    """
    fundamental_index, governance_index = (
        _parent_index(group) for group in (fundamental, governance)
    )
    drafts = []
    for identity in inputs.entity_ids:
        parents, checks = _parent_checks(
            inputs,
            identity,
            (
                ("fundamental", fundamental_index, True),
                ("governance", governance_index, True),
            ),
        )
        dossiers, history_reason = _financial_years(inputs, identity)
        annual_metrics = {
            name: [cast(float, result.metric(name)) for result in dossiers]
            for name in ("cash_margin", "leverage", "cash_cover", "operating_cashflow")
        } if dossiers else {}
        margins = annual_metrics.get("cash_margin", [])
        cash_covers = annual_metrics.get("cash_cover", [])
        leverages = annual_metrics.get("leverage", [])
        operating_cashflows = annual_metrics.get("operating_cashflow", [])
        valuation = latest(inputs.rows("daily_basic", identity))
        pb = None if valuation is None else valuation.number("pb", "ratio")
        governance_parent = governance_index.get(identity)
        supply = _parent_metric(governance_parent, "R_supply")
        supply_value = _ratio_value(supply)
        values = {
            "cash_margin_three_year_median": (
                _stable(float(median(margins))) if margins else None
            ),
            "cash_cover_three_year_minimum": min(cash_covers) if cash_covers else None,
            "leverage_three_year_maximum": max(leverages) if leverages else None,
            "positive_operating_cashflow_year_fraction": (
                fsum(value > 0 for value in operating_cashflows) / LONG_TERM_REQUIRED_YEARS
                if operating_cashflows
                else None
            ),
            "cash_margin_three_year_range": (
                _stable(max(margins) - min(margins)) if margins else None
            ),
            "inverse_pb": 1 / pb if pb is not None and pb > 0 and isfinite(1 / pb) else None,
        }
        history_refs = unique_evidence(
            reference
            for dossier in dossiers
            for reference in dossier.envelope.evidence_refs
        )
        valuation_refs = () if valuation is None else valuation.evidence("pb", "trade_date")
        metrics = tuple(
            metric(
                definition.name,
                values.get(definition.name),
                "fraction" if "year_fraction" in definition.name else "ratio",
                valuation_refs if definition.name == "inverse_pb" else history_refs,
                ()
                if values.get(definition.name) is not None
                else ((history_reason or "long_term_factor_unavailable"),),
            )
            for definition in LONG_TERM_FACTORS
        )
        checks += (
            gate(
                "long_term_financial_history",
                Truth.TRUE if history_reason is None else Truth.UNKNOWN,
                history_reason or "three_year_financial_history_incomplete",
                references=history_refs,
            ),
            gate(
                "long_term_positive_pb",
                Truth.TRUE
                if pb is not None and pb > 0
                else Truth.FALSE
                if pb is not None
                else Truth.UNKNOWN,
                "positive_pb_unavailable",
                references=valuation_refs,
            ),
            gate(
                "long_term_supply_risk",
                Truth.TRUE if supply_value is not None else Truth.UNKNOWN,
                "supply_risk_unavailable",
                references=() if supply is None else supply.evidence_refs,
            ),
        )
        references = unique_evidence(
            history_refs
            + valuation_refs
            + tuple(
                reference
                for parent in parents
                for reference in parent.envelope.evidence_refs
            )
        )
        years = [result.details.get("report_period") for result in dossiers]
        drafts.append(
            CandidateDraft(
                identity,
                "long_term/three_annual_reports",
                metrics,
                checks,
                references,
                parents,
                {
                    "formula": (
                        "100*(0.25*P(cash_margin_median)+0.20*P(cash_cover_min)"
                        "+0.20*P_low(leverage_max)+0.15*P(positive_cashflow_years)"
                        "+0.10*P_low(cash_margin_range)+0.10*P(1/pb))"
                    ),
                    "annual_report_periods": years,
                    "required_annual_reports": LONG_TERM_REQUIRED_YEARS,
                },
            )
        )
    return _rank(
        inputs,
        LONG_TERM_PRODUCT_ID,
        "long_term",
        "six_to_twenty_four_month_research_review",
        LONG_TERM_FACTORS,
        tuple(drafts),
    )
