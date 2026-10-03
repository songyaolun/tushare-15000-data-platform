"""P03 当前截面候选排序；风险门禁、缺失档位与评分分别解释。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, isfinite, log1p
from typing import cast

from ..product_foundation.contracts import (
    ActionState,
    EvidenceRef,
    FeatureContribution,
    Predicate,
    ProductState,
    QualityGateResult,
    Truth,
)
from ..product_foundation.numeric import price_gate
from ..store import encode
from .common import gate, latest, make_result, metric, percentiles, unique_evidence
from .models import RULE_VERSION, ResearchInputs, ResearchResult


@dataclass(frozen=True)
class _Candidate:
    """计算中的候选证据集；例如 eligible=False 的证券不会进入因子分母。"""

    profile: str
    quality: float | None
    pe: float | None
    pb: float | None
    amount: float | None
    checks: tuple[QualityGateResult, ...]
    references: tuple[EvidenceRef, ...]
    parents: tuple[ResearchResult, ...]
    eligible: bool
    problem_flags: tuple[str, ...]
    exclusion_reason: str | None


def _index_parents(results: tuple[ResearchResult, ...]) -> dict[str, ResearchResult | None]:
    """一次索引全部父结果，只保留每实体的唯一对象，重复实体标记为无效。

    先判同实体条数，再由调用方校验截止；不去重相同对象，也不预先丢弃错截止父结果。
    例：输入[A当前、A未来、B当前]返回{A: None, B: B当前}，A不能任选当前版本。
    扫描时间与父条数成正比，后续查找为常量时间，不序列化或复制父模型。
    """
    indexed: dict[str, ResearchResult | None] = {}
    for result in results:
        identity = result.envelope.entity_id
        indexed[identity] = None if identity in indexed else result
    return indexed


def _parent_context(
    inputs: ResearchInputs,
    identity: str,
    groups: tuple[dict[str, ResearchResult | None], ...],
) -> tuple[tuple[ResearchResult, ...], tuple[QualityGateResult, ...], float | None, str]:
    """选择同实体、同截止的唯一父结果，风险和可行性缺失均保留未知。

    例如缺少 P06 时返回 governance_parent=unknown；P04 缺质量分允许市场档，
    P04 已知硬失败仍传播，避免负面审计被缺财务降级掩盖。
    """
    parents, checks = [], []
    quality = None
    company_type = "unknown"
    for index, (name, results) in enumerate(
        zip(("fundamental", "governance", "flow"), groups, strict=True)
    ):
        parent = results.get(identity)
        if parent is None or parent.envelope.analysis_as_of != inputs.analysis_as_of:
            if index:
                checks.append(gate(name + "_parent", Truth.UNKNOWN, name + "_parent_unavailable"))
            continue
        parents.append(parent)
        if index == 0:
            raw_type = parent.details.get("comp_type")
            company_type = (
                raw_type
                if isinstance(raw_type, str) and raw_type in {"1", "2", "3", "4", "7"}
                else "unknown"
            )
            quality = parent.metric("quality_score")
            if company_type != "1" or any(
                check.hard and not check.passed for check in parent.envelope.quality_gates
            ):
                quality = None
            checks.extend(
                replace(check, gate_id=name + "/" + check.gate_id)
                for check in parent.envelope.quality_gates
                if check.hard and check.result is Truth.FALSE
            )
        else:
            checks.append(gate(name + "_parent", Truth.TRUE, ""))
            checks.extend(
                replace(check, gate_id=name + "/" + check.gate_id)
                for check in parent.envelope.quality_gates
            )
            if not parent.envelope.quality_gates:
                checks.append(gate(name + "_checks", Truth.UNKNOWN, name + "_checks_missing"))
    return tuple(parents), tuple(checks), quality, company_type


def _rank_and_explain(
    inputs: ResearchInputs,
    drafts: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """只对同档且通过硬门禁的评分排名，并逐个删因子重排名。

    例如完整档 30 只、市场档 29 只，后者名次全部未知；删因子后保留其余权重
    相对比例，输出包含基线的最小和最大名次。完整档前 20% 才能进入优先研究。
    同一分组内实体及数值完全相同的场景复用分位，场景名称仍独立保留；
    例如全为None的基线与三个删因子输入只计算一次。缓存最多包含基线加三个场景，
    不跨同行分组、档位或构建调用复用，也不把仅排序相同的不同数值当成相同输入。
    """
    finished = []
    groups: dict[tuple[str, tuple[str, ...]], list[ResearchResult]] = {}
    for result in drafts:
        key = (result.envelope.feature_profile, tuple(result.details["factor_peer_path"]))
        groups.setdefault(key, []).append(result)
    for (profile, _), group in sorted(groups.items()):
        scores = {result.envelope.entity_id: result.envelope.score for result in group}
        ranked = percentiles(inputs, scores, profile)
        ranking_cache = {tuple(scores.items()): ranked}
        scenarios: dict[str, dict[str, float]] = {}
        for factor in sorted(
            {
                component.feature_id
                for result in group
                for component in result.envelope.contributions
            }
        ):
            remaining = {
                result.envelope.entity_id: sum(
                    component.contribution or 0
                    for component in result.envelope.contributions
                    if component.feature_id != factor
                )
                if result.envelope.score is not None
                else None
                for result in group
            }
            population = tuple(remaining.items())
            if population not in ranking_cache:
                ranking_cache[population] = percentiles(inputs, remaining, profile)
            scenarios[factor] = {
                identity: 1 + (1 - item.percentile) * (item.peer_count - 1)
                for identity, item in ranking_cache[population].items()
                if item.percentile is not None
            }
        for result in group:
            identity = result.envelope.entity_id
            position = ranked[identity]
            rank = (
                1 + (1 - position.percentile) * (position.peer_count - 1)
                if position.percentile is not None
                else None
            )
            intervals = (
                ([rank] + [values[identity] for values in scenarios.values() if identity in values])
                if rank is not None
                else []
            )
            capacity = min(inputs.config.thresholds.top_k, ceil(position.peer_count * 0.2))
            priority = (
                profile.startswith("fundamental_complete/")
                and rank is not None
                and rank <= capacity
            )
            eligible = rank is not None and all(
                check.passed for check in result.envelope.quality_gates if check.hard
            )
            details = result.details
            details.update(
                {
                    "rank_peer_level": position.peer_level,
                    "rank_reason_codes": list(position.reason_codes),
                    "rank_interval_without_one_factor": [min(intervals), max(intervals)]
                    if intervals
                    else None,
                    "leave_one_factor_ranks": {
                        name: values.get(identity) for name, values in scenarios.items()
                    },
                    "research_capacity": capacity,
                    "rank_distance_to_capacity": rank - capacity if rank is not None else None,
                }
            )
            checks = result.envelope.quality_gates + (
                gate(
                    "ranking_sample",
                    Truth.TRUE if rank is not None else Truth.UNKNOWN,
                    "insufficient_rank_sample",
                ),
            )
            envelope = replace(
                result.envelope,
                rank=rank,
                peer_count=position.peer_count,
                quality_gates=checks,
                action_state=ActionState.RESEARCH_FIRST
                if priority and eligible
                else ActionState.NO_CONCLUSION
                if rank is None and result.envelope.state is not ProductState.BLOCKED
                else result.envelope.action_state,
                decision_allowed=bool(priority and eligible),
                state=ProductState.INCOMPLETE
                if rank is None and result.envelope.state is not ProductState.BLOCKED
                else result.envelope.state,
                gate_codes=tuple(
                    sorted(set(result.envelope.gate_codes) | set(position.reason_codes))
                ),
            )
            finished.append(replace(result, envelope=envelope, details_json=encode(details)))
    return tuple(sorted(finished, key=lambda item: item.envelope.entity_id))


def build_candidates(
    inputs: ResearchInputs,
    fundamental: tuple[ResearchResult, ...],
    governance: tuple[ResearchResult, ...],
    flow: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """构造证据可复算的 P03 候选，完全不使用历史动量或供应商技术信号。

    例如 Q0=80、V0=L0=0.5 得到完整档 63.5 分；缺 Q0 时市场档为 50 分。
    PE/PB 非正值进入 problem_research，缺风险检查不会被任何高分抵消。
    三类父集合各索引一次；例如5911实体和三类各5911父结果只扫描17733条父记录。
    """
    parent_indexes = tuple(_index_parents(group) for group in (fundamental, governance, flow))
    facts: dict[str, _Candidate] = {}
    selected_date = max(
        (row.availability.data_as_of or "" for row in inputs.rows("daily")), default=""
    )
    for identity in inputs.entity_ids:
        parents, parent_gates, quality, company_type = _parent_context(
            inputs,
            identity,
            parent_indexes,
        )
        daily = latest(inputs.rows("daily", identity))
        valuation = latest(inputs.rows("daily_basic", identity))
        pe = valuation.number("pe_ttm", "ratio") if valuation else None
        pb = valuation.number("pb", "ratio") if valuation else None
        amount = daily.number("amount", "元") if daily else None
        amount = (
            amount
            if daily and daily.numeric.get("amount") and daily.numeric["amount"].currency == "CNY"
            else None
        )
        problem_flags = tuple(
            sorted(
                {
                    flag
                    for parent in parents
                    if parent.envelope.product_id == "fundamental_dossier/v1"
                    for flag in parent.details.get("problem_research_flags", ())
                    if flag in {"negative_equity", "negative_earnings"}
                }
            )
        )
        nonpositive = any(value is not None and value <= 0 for value in (pe, pb))
        problem = nonpositive or bool(problem_flags)
        exclusion_reason = (
            "nonpositive_valuation"
            if nonpositive
            else "financial_problem_research"
            if problem_flags
            else None
        )
        profile = (
            (
                "problem_research"
                if problem
                else "fundamental_complete"
                if quality is not None
                else "market_only"
            )
            + "/comp_type_"
            + company_type
        )
        references = unique_evidence(
            (daily.evidence("close", "vol", "amount", "trade_date") if daily else ())
            + (valuation.evidence("pe_ttm", "pb", "trade_date") if valuation else ())
            + tuple(ref for parent in parents for ref in parent.envelope.evidence_refs)
        )
        checks = parent_gates + (
            replace(
                price_gate(daily.raw, "price_rank"),
                evidence_refs=daily.evidence("close", "vol", "amount"),
            )
            if daily
            else gate("price_price_rank", Truth.UNKNOWN, "price_unavailable"),
            gate(
                "valuation_positive",
                Truth.FALSE
                if problem
                else Truth.TRUE
                if pe is not None and pb is not None
                else Truth.UNKNOWN,
                exclusion_reason or "valuation_unavailable",
            ),
            gate(
                "valuation_inverse_finite",
                Truth.TRUE
                if all(
                    value is not None and value > 0 and isfinite(1 / value) for value in (pe, pb)
                )
                else Truth.UNKNOWN,
                "valuation_inverse_unavailable",
            ),
            gate(
                "amount_cny",
                Truth.UNKNOWN if amount is None else Truth.TRUE if amount > 0 else Truth.FALSE,
                "amount_cny_unavailable",
            ),
            gate(
                "aligned_market_date",
                Truth.UNKNOWN
                if daily is None or valuation is None
                else Truth.TRUE
                if daily.availability.data_as_of
                == valuation.availability.data_as_of
                == selected_date
                else Truth.FALSE,
                "market_date_mismatch",
            ),
        )
        eligible = all(check.passed for check in checks if check.hard)
        facts[identity] = _Candidate(
            profile,
            quality,
            pe,
            pb,
            amount,
            checks,
            references,
            parents,
            eligible,
            problem_flags,
            exclusion_reason,
        )
    drafts = []
    for profile in sorted({values.profile for values in facts.values()}):
        score_profile = profile.split("/", 1)[0]
        group = {
            identity: values for identity, values in facts.items() if values.profile == profile
        }
        pe_ranks = percentiles(
            inputs,
            {
                identity: 1 / cast(float, values.pe) if values.eligible else None
                for identity, values in group.items()
            },
            profile,
        )
        pb_ranks = percentiles(
            inputs,
            {
                identity: 1 / cast(float, values.pb) if values.eligible else None
                for identity, values in group.items()
            },
            profile,
        )
        amount_ranks = percentiles(
            inputs,
            {
                identity: log1p(cast(float, values.amount)) if values.eligible else None
                for identity, values in group.items()
            },
            profile,
        )
        for identity, values in group.items():
            quality, pe, pb, amount = values.quality, values.pe, values.pb, values.amount
            checks, references, parents = values.checks, values.references, values.parents
            pe_rank, pb_rank = pe_ranks[identity], pb_ranks[identity]
            path = inputs.config.peer_paths.get(identity, ())
            peer_path = (
                path[: path.index(pe_rank.peer_level) + 1] if pe_rank.peer_level in path else ()
            )
            liquidity = amount_ranks[identity].percentile
            value = (
                (pe_rank.percentile + pb_rank.percentile) / 2
                if pe_rank.percentile is not None and pb_rank.percentile is not None
                else None
            )
            components = []
            weights = (
                (
                    ("quality_score", quality, 0.45),
                    ("value_score", value, 0.35),
                    ("liquidity_score", liquidity, 0.20),
                )
                if score_profile == "fundamental_complete"
                else (("value_score", value, 0.6), ("liquidity_score", liquidity, 0.4))
            )
            for name, observed, weight in weights:
                normalized = (
                    observed / 100 if observed is not None and name == "quality_score" else observed
                )
                components.append(
                    FeatureContribution(
                        feature_id=name,
                        raw_value=observed,
                        unit="score" if name == "quality_score" else "fraction",
                        transformed_value=normalized,
                        weight=weight,
                        contribution=100 * weight * normalized if normalized is not None else None,
                        transform="quality_divide_100"
                        if name == "quality_score"
                        else "positive_inverse_pe_pb_mean_percentile"
                        if name == "value_score"
                        else "log1p_amount_cny_percentile",
                        peer_level=pe_rank.peer_level,
                        missing_reason="candidate_factor_unavailable" if observed is None else None,
                        evidence_refs=references if observed is not None else (),
                    )
                )
            contrary = tuple(
                ref for check in checks if not check.passed for ref in check.evidence_refs
            ) + tuple(ref for parent in parents for ref in parent.envelope.contrary_evidence)
            drafts.append(
                make_result(
                    inputs,
                    "equity_research_candidates/v1",
                    identity,
                    profile=profile,
                    metrics=(
                        metric(
                            "quality_score",
                            quality,
                            "score",
                            references,
                            ("financial_quality_not_available",) if quality is None else (),
                            not_applicable=score_profile != "fundamental_complete"
                            and quality is None,
                        ),
                        metric(
                            "value_score",
                            100 * value if value is not None else None,
                            "score",
                            references,
                        ),
                        metric(
                            "liquidity_score",
                            100 * liquidity if liquidity is not None else None,
                            "score",
                            references,
                        ),
                    ),
                    gates=checks,
                    contributions=components,
                    references=references,
                    contrary=contrary,
                    parents=(parent.envelope.result_id for parent in parents),
                    entry_conditions=(
                        Predicate(
                            predicate_id="manual_candidate_review",
                            version=RULE_VERSION,
                            operator="manual",
                        ),
                    ),
                    details={
                        "formula": "0.45*Q0+35*V0+20*L0"
                        if score_profile == "fundamental_complete"
                        else "100*(0.6*V0+0.4*L0)"
                        if score_profile == "market_only"
                        else None,
                        "selected_market_date": selected_date or None,
                        "pe_ttm": pe,
                        "pb": pb,
                        "amount_cny": amount,
                        "factor_peer_count": pe_rank.peer_count,
                        "factor_peer_path": list(peer_path),
                        "score_profile": score_profile,
                        "comp_type": profile.split("comp_type_", 1)[1],
                        "risk_flags": sorted(
                            {
                                reason
                                for check in checks
                                if not check.passed
                                for reason in check.reason_codes
                            }
                        ),
                        "exclusion_reason": values.exclusion_reason,
                        "problem_research_flags": list(values.problem_flags),
                        "next_research_question": "核查亏损或负净资产的原因及持续性"
                        if score_profile == "problem_research"
                        else "补齐财务质量证据并复核风险门禁"
                        if score_profile == "market_only"
                        else "复核现金支持及未来股权供给能否改变当前判断",
                        "historical_features": "not_enabled",
                    },
                )
            )
    return _rank_and_explain(inputs, tuple(drafts))
