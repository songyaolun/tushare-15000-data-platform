"""同口径财务三表的版本核验、现金质量与审计门禁。"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any, cast

from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef, FeatureContribution, Truth
from ..product_foundation.features import PercentileResult
from ..product_foundation.temporal import availability_gate
from ..standard import normalize_date
from .common import gate, latest, make_result, metric, percentiles, unique_evidence
from .models import Metric, ResearchInputs, ResearchResult, ResearchRow

STATEMENT_FIELDS = {
    "income": ("revenue",),
    "cashflow": ("n_cashflow_act",),
    "balancesheet": (
        "total_assets",
        "total_liab",
        "money_cap",
        "total_cur_liab",
        "total_hldr_eqy_inc_min_int",
    ),
}


def _context(row: ResearchRow) -> tuple[str, str, str] | None:
    """返回完整报表连接键并核验报告期；例：20260630、1、1得到同口径元组。

    缺失类型、非法报告期或时间适配报告期冲突返回None，公告日期无需跨表相等。
    """
    try:
        period = normalize_date(row.text("end_date"))
    except (ValueError, TypeError):
        return None
    report_type, comp_type = row.text("report_type"), row.text("comp_type")
    if (
        not report_type
        or not comp_type
        or (row.availability.report_period is not None and row.availability.report_period != period)
    ):
        return None
    return period, report_type, comp_type


def _version(rows: tuple[ResearchRow, ...]) -> tuple[ResearchRow | None, str | None]:
    """以公开时间选唯一可见修订，绝不以update_flag大小排序。

    例：旧版update_flag=1、新版update_flag=0，公开时间较晚的新版本被选中；同刻多版拒绝。
    多版若缺公开时点无法确定顺序，原始变体也不能通过选择消除歧义。
    """
    if not rows:
        return None, "statement_missing"
    if any(row.raw.get("business_variant_key") for row in rows):
        return None, "business_variant_unresolved"
    unique = {row.availability.source_record_id: row for row in rows}
    if len(unique) == 1:
        return next(iter(unique.values())), None
    if any(row.availability.public_available_at is None for row in rows):
        return None, "revision_time_unknown"
    newest = max(time_boundary(cast(str, row.availability.public_available_at)) for row in rows)
    choices = {
        row.availability.source_record_id: row
        for row in rows
        if time_boundary(cast(str, row.availability.public_available_at)) == newest
    }
    return (
        (next(iter(choices.values())), None) if len(choices) == 1 else (None, "revision_conflict")
    )


def _money(row: ResearchRow | None, field: str) -> float | None:
    """读取已认证人民币金额；例：同为元但币种HKD的资产返回None。

    空报表、空字段和非金额量纲均不允许参与跨表计算，也不对源值再次换算。
    """
    if row is None:
        return None
    value = row.number(field, "元")
    fact = row.numeric.get(field)
    return (
        value if fact is not None and fact.currency == "CNY" and fact.dimension == "money" else None
    )


def _ratio(
    name: str,
    left: ResearchRow | None,
    numerator: str,
    right: ResearchRow | None,
    denominator: str,
    *,
    usable: bool,
    not_applicable: bool = False,
) -> Metric:
    """以正分母计算可审计财务比率；例：经营现金流-10/收入100得到-0.1。

    不可比报表、未知单位、非正分母或溢出输出空值及原因，金融行业明确标为不适用。
    """
    top, bottom = _money(left, numerator), _money(right, denominator)
    references = (() if left is None else left.evidence(numerator)) + (
        () if right is None else right.evidence(denominator)
    )
    reason = "financial_scope_unverified"
    value = None
    if usable:
        if top is None or bottom is None:
            reason = "financial_unit_or_value_unverified"
        elif bottom <= 0:
            reason = "nonpositive_financial_denominator"
        elif isfinite(top / bottom):
            value = top / bottom
        else:
            reason = "nonfinite_calculation"
    return metric(
        name,
        value,
        "ratio",
        references,
        () if value is not None else (reason,),
        not_applicable=not_applicable,
    )


def _dossier(inputs: ResearchInputs, identity: str) -> ResearchResult:
    """生成一个实体的未排名三表档案；例：缺现金流表时仍保留资产表证据和缺口。

    以最新已可见报告期为准，同期多个报表口径不猜选；普通版仅用于核对或核心源缺失时
    替代，核对一致不会再增加一份计分证据。审计意见和资产平衡异常独立进入硬门禁。
    已认证总权益或合并净利润为负时保留问题研究标记和反证，无数值认证不推断正负。
    """
    all_rows = tuple(
        row
        for family in STATEMENT_FIELDS
        for api in (family + "_vip", family)
        for row in inputs.rows(api, identity)
    )
    contexts = {
        context
        for row in all_rows
        if (context := _context(row)) is not None and context[0] <= inputs.analysis_as_of[:10]
    }
    period = max((context[0] for context in contexts), default=None)
    candidates = {context for context in contexts if context[0] == period}
    context = next(iter(candidates)) if len(candidates) == 1 else None
    comparable = context is not None and not any(_context(row) is None for row in all_rows)
    selected: dict[str, ResearchRow | None] = {}
    checks = []
    references: list[EvidenceRef] = []
    contrary: list[EvidenceRef] = []
    conflicts: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    checks.append(
        gate(
            "financial_context",
            Truth.TRUE if comparable else Truth.UNKNOWN,
            "financial_context_ambiguous",
        )
    )
    for family, fields in STATEMENT_FIELDS.items():
        sources = {}
        for api in (family + "_vip", family):
            rows = tuple(row for row in inputs.rows(api, identity) if _context(row) == context)
            chosen, reason = _version(rows)
            sources[api] = chosen
            if reason not in {None, "statement_missing"}:
                conflicts.append({"source_api": api, "reason": reason})
                contrary.extend(ref for row in rows for ref in row.evidence(*fields))
        vip, ordinary = sources[family + "_vip"], sources[family]
        chosen = vip or ordinary
        selected[family] = chosen
        matched = None
        if vip is not None and ordinary is not None:
            if (
                family == "income"
                and _money(vip, "n_income") is not None
                and _money(ordinary, "n_income") is not None
            ):
                fields += ("n_income",)
            vip_values = tuple(_money(vip, field) for field in fields)
            ordinary_values = tuple(_money(ordinary, field) for field in fields)
            if all(value is not None for value in vip_values + ordinary_values):
                matched = vip_values == ordinary_values
                if not matched:
                    conflicts.append({"source_api": family, "reason": "vip_ordinary_conflict"})
                    contrary.extend(ordinary.evidence(*fields))
        comparisons.append(
            {
                "family": family,
                "agreement": matched,
                "selected_api": None if chosen is None else chosen.source_api,
                "ordinary_record_id": None
                if ordinary is None
                else ordinary.availability.source_record_id,
                "vip_record_id": None if vip is None else vip.availability.source_record_id,
            }
        )
        if chosen is not None:
            references.extend(chosen.evidence("end_date", "report_type", "comp_type", *fields))
        checks.append(
            gate(
                family + "_available",
                Truth.TRUE if chosen is not None else Truth.UNKNOWN,
                "statement_missing",
            )
        )
    checks.append(
        gate(
            "financial_versions",
            Truth.FALSE if conflicts else Truth.TRUE,
            "financial_revision_conflict",
            references=contrary,
        )
    )
    report_certified = context is not None and context[1] in inputs.config.certified_report_types
    nonfinancial = context is not None and context[2] == "1"
    financial = context is not None and context[2] in {"2", "3", "4", "7"}
    checks.extend(
        (
            gate(
                "report_type_certified",
                Truth.TRUE if report_certified else Truth.UNKNOWN,
                "report_type_uncertified",
            ),
            gate(
                "nonfinancial_scope",
                Truth.TRUE if nonfinancial else Truth.FALSE if financial else Truth.UNKNOWN,
                "nonfinancial_model_not_applicable",
            ),
        )
    )
    usable = (
        comparable
        and report_certified
        and nonfinancial
        and not conflicts
        and all(row is not None for row in selected.values())
    )
    income, cashflow, balance = selected["income"], selected["cashflow"], selected["balancesheet"]
    problem_research_flags = []
    for flag, statement, field in (
        ("negative_equity", balance, "total_hldr_eqy_inc_min_int"),
        ("negative_earnings", income, "n_income"),
    ):
        amount = _money(statement, field)
        if (
            comparable
            and not conflicts
            and amount is not None
            and amount < 0
            and statement is not None
        ):
            problem_research_flags.append(flag)
            problem_refs = statement.evidence(field)
            references.extend(problem_refs)
            contrary.extend(problem_refs)
    features = [
        _ratio(
            "cash_margin",
            cashflow,
            "n_cashflow_act",
            income,
            "revenue",
            usable=usable,
            not_applicable=financial,
        ),
        _ratio(
            "leverage",
            balance,
            "total_liab",
            balance,
            "total_assets",
            usable=usable,
            not_applicable=financial,
        ),
        _ratio(
            "cash_cover",
            balance,
            "money_cap",
            balance,
            "total_cur_liab",
            usable=usable,
            not_applicable=financial,
        ),
    ]
    assets, liabilities, equity = (
        _money(balance, field)
        for field in ("total_assets", "total_liab", "total_hldr_eqy_inc_min_int")
    )
    error = None
    if (
        usable
        and assets is not None
        and liabilities is not None
        and equity is not None
        and assets > 0
    ):
        candidate_error = abs(assets - liabilities - equity) / max(abs(assets), 1)
        if isfinite(candidate_error):
            error = candidate_error
    balance_refs = (
        ()
        if balance is None
        else balance.evidence("total_assets", "total_liab", "total_hldr_eqy_inc_min_int")
    )
    features.append(
        metric(
            "balance_error",
            error,
            "ratio",
            balance_refs,
            () if error is not None else ("balance_check_unavailable",),
            not_applicable=financial,
        )
    )
    checks.append(
        gate(
            "balance_identity",
            Truth.UNKNOWN if error is None else Truth.FALSE if error > 0.001 else Truth.TRUE,
            "balance_identity_review",
            references=balance_refs,
        )
    )
    audit_rows = tuple(
        row
        for row in inputs.rows("fina_audit", identity)
        if row.availability.report_period is not None
        and period is not None
        and row.availability.report_period <= period
    )
    audit, audit_reason = _version(
        tuple(
            row
            for row in audit_rows
            if row.availability.report_period
            == max(
                (cast(str, item.availability.report_period) for item in audit_rows), default=None
            )
        )
    )
    opinion = None if audit is None else audit.text("audit_result")
    audit_refs = () if audit is None else audit.evidence("audit_result", "end_date", "ann_date")
    standard = opinion == "标准无保留意见"
    checks.append(
        gate(
            "audit_opinion",
            Truth.UNKNOWN if opinion is None else Truth.TRUE if standard else Truth.FALSE,
            audit_reason or "audit_opinion_review",
            references=audit_refs,
        )
    )
    if opinion is not None and not standard:
        contrary.extend(audit_refs)
    valuation = latest(inputs.rows("daily_basic", identity))
    valuation_details: dict[str, Any] = {"basis": "provider_reported", "pe_ttm": None, "pb": None}
    if valuation is not None:
        valuation_details.update(
            pe_ttm=valuation.number("pe_ttm", "ratio"), pb=valuation.number("pb", "ratio")
        )
        references.extend(valuation.evidence("pe_ttm", "pb", "trade_date"))
    missing_reasons = sorted(
        {
            reason
            for family in STATEMENT_FIELDS
            for api in (family, family + "_vip")
            for reason in inputs.reasons(api, identity)
        }
    )
    checks.append(
        gate(
            "financial_source_completeness",
            Truth.UNKNOWN if missing_reasons else Truth.TRUE,
            "financial_source_incomplete",
        )
    )
    return make_result(
        inputs,
        "fundamental_dossier/v1",
        identity,
        metrics=features,
        gates=checks,
        references=references,
        contrary=contrary,
        profile="nonfinancial_ytd/" + "/".join(context or ("unknown",)),
        details={
            "report_period": period,
            "report_type": None if context is None else context[1],
            "comp_type": None if context is None else context[2],
            "revision_conflicts": conflicts,
            "source_comparisons": comparisons,
            "audit_flags": [] if standard else [opinion or "audit_unknown"],
            "audit_period": None if audit is None else audit.availability.report_period,
            "valuation_basis": valuation_details,
            "problem_research_flags": problem_research_flags,
            "source_reason_codes": missing_reasons,
            "quality_scope_eligible": usable,
            "interpretation": "single_report_period",
        },
    )


def build_fundamental(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """为所有显式股票实体生成三表档案，并在相同报告口径的认证同行内计算Q0。

    例：同组三个现金比率及低杠杆方向分位都为0.5，则quality_score与Q0均为50；
    缺同行路径或样本不足时保留原始比率且两项分数为空。审计异常不被高分抵消。
    未来记录不作为证据；空实体集合返回空元组，处理过程中不访问网络。
    """
    inputs = replace(
        inputs,
        rows_by_api={
            api: tuple(
                row
                for row in rows
                if availability_gate(row.availability, inputs.analysis_as_of).passed
            )
            for api, rows in inputs.rows_by_api.items()
        },
    )
    rows_by_entity: dict[str | None, dict[str, list[ResearchRow]]] = {}
    for api, rows in inputs.rows_by_api.items():
        for row in rows:
            rows_by_entity.setdefault(row.entity_id, {}).setdefault(api, []).append(row)
    dossiers = tuple(
        _dossier(
            replace(
                inputs,
                rows_by_api={
                    api: tuple(rows) for api, rows in rows_by_entity.get(identity, {}).items()
                },
            ),
            identity,
        )
        for identity in inputs.entity_ids
    )
    groups: dict[str, list[ResearchResult]] = {}
    for dossier in dossiers:
        groups.setdefault(dossier.envelope.feature_profile, []).append(dossier)
    rankings: dict[str, dict[str, dict[str, PercentileResult]]] = {}
    for profile, group in groups.items():
        rankings[profile] = {
            name: percentiles(
                inputs,
                {item.envelope.entity_id: item.metric(name) for item in group},
                profile,
                lower_is_better=name == "leverage",
            )
            for name in ("cash_margin", "leverage", "cash_cover")
        }
    results = []
    for dossier in dossiers:
        profile = dossier.envelope.feature_profile
        components = []
        peer_details = {}
        for name, weight in (("cash_margin", 0.4), ("leverage", 0.35), ("cash_cover", 0.25)):
            rank = rankings[profile][name][dossier.envelope.entity_id]
            feature = next(item for item in dossier.metrics if item.name == name)
            components.append(
                FeatureContribution(
                    feature_id=name,
                    raw_value=feature.value,
                    unit="ratio",
                    transformed_value=rank.percentile,
                    weight=weight,
                    contribution=None
                    if rank.percentile is None
                    else 100 * weight * rank.percentile,
                    transform="one_minus_peer_percentile"
                    if name == "leverage"
                    else "peer_percentile",
                    peer_level=rank.peer_level,
                    missing_reason=None if rank.percentile is not None else rank.reason_codes[0],
                    evidence_refs=feature.evidence_refs,
                    reason_codes=rank.reason_codes,
                )
            )
            peer_details[name] = rank.to_dict()
        score = (
            sum(cast(float, item.contribution) for item in components)
            if all(item.contribution is not None for item in components)
            else None
        )
        score_refs = unique_evidence(ref for item in components for ref in item.evidence_refs)
        reasons = tuple(sorted({reason for item in components for reason in item.reason_codes}))
        metrics = dossier.metrics + tuple(
            metric(name, score, "score", score_refs, reasons) for name in ("quality_score", "Q0")
        )
        results.append(
            make_result(
                inputs,
                "fundamental_dossier/v1",
                dossier.envelope.entity_id,
                metrics=metrics,
                gates=dossier.envelope.quality_gates,
                references=dossier.envelope.evidence_refs,
                contrary=dossier.envelope.contrary_evidence,
                contributions=components,
                profile=profile,
                details=dossier.details | {"peer_statistics": peer_details},
            )
        )
    return tuple(results)
