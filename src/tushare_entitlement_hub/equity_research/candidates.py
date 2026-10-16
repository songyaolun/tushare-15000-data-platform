"""统一 P03：只对已确认且入场不过度延伸的右侧候选排序。"""

from __future__ import annotations

from dataclasses import replace
from math import fsum, isfinite

from ..product_foundation.contracts import (
    ActionState,
    FeatureContribution,
    Predicate,
    ProductState,
    Truth,
)
from ..store import encode
from .common import gate, latest, make_result, metric, percentiles
from .models import PRODUCT_IDS, RULE_VERSION, ResearchInputs, ResearchResult
from .right_side_structure import observe_structure

FACTORS = (
    ("entry_risk", 0.30, True),
    ("structural_reward_risk", 0.30, False),
    ("amount_confirmation_ratio", 0.20, False),
    ("large_flow_ratio", 0.10, False),
    ("quality_score", 0.10, False),
)
PRICE_METRICS = frozenset(
    {"entry_price", "confirmation_price", "invalidation_price", "historical_resistance"}
)


def index_parents(results: tuple[ResearchResult, ...]) -> dict[str, ResearchResult | None]:
    """一次索引父结果，重复实体失败关闭；例：A出现两次返回{'A': None}。"""
    indexed: dict[str, ResearchResult | None] = {}
    for result in results:
        identity = result.envelope.entity_id
        indexed[identity] = None if identity in indexed else result
    return indexed


def parent_metric(parent: ResearchResult | None, name: str) -> float | None:
    """读取唯一且有限的比例/分数；例：缺quality_score、错误单位或未知值返回None。"""
    if parent is None:
        return None
    matches = [item for item in parent.metrics if item.name == name and item.status == "available"]
    if len(matches) != 1:
        return None
    item = matches[0]
    return (
        item.value
        if (
            item.value is not None
            and isfinite(item.value)
            and item.unit in {"ratio", "fraction", "score"}
        )
        else None
    )


def build_candidates(
    inputs: ResearchInputs,
    fundamental: tuple[ResearchResult, ...],
    governance: tuple[ResearchResult, ...],
    flow: tuple[ResearchResult, ...],
) -> tuple[ResearchResult, ...]:
    """先验证结构、财务、风险和交易门禁，再按入场条件评分，绝不奖励历史涨幅。

    输入回调后突破且风险<=8%、结构空间/风险>=2的可比实体输出独立名次；
    继续下跌、突破过远、缺证据或风险门禁失败的实体保留诊断且分数/名次为空。
    初始规则未经回测，仅用于人工复核，decision_allowed始终为False。
    """
    indexes = tuple(index_parents(group) for group in (fundamental, governance, flow))
    selected_date = max(
        (row.availability.data_as_of or "" for row in inputs.rows("daily")), default=""
    )
    drafts = []
    for identity in inputs.entity_ids:
        parents, checks = [], []
        for label, product_id, indexed in zip(
            ("fundamental", "governance", "flow"), PRODUCT_IDS[1:4], indexes, strict=True
        ):
            parent = indexed.get(identity)
            if (
                parent is None
                or parent.envelope.analysis_as_of != inputs.analysis_as_of
                or parent.envelope.product_id != product_id
            ):
                checks.append(gate(label + "_parent", Truth.UNKNOWN, label + "_parent_unavailable"))
                continue
            parents.append(parent)
            checks.append(gate(label + "_parent", Truth.TRUE, ""))
            checks.extend(
                replace(check, gate_id=label + "/" + check.gate_id)
                for check in parent.envelope.quality_gates
            )
            if not parent.envelope.quality_gates:
                checks.append(gate(label + "_checks", Truth.UNKNOWN, label + "_checks_missing"))
        by_product = {parent.envelope.product_id: parent for parent in parents}
        financial = by_product.get(PRODUCT_IDS[1])
        funding = by_product.get(PRODUCT_IDS[3])
        quality = parent_metric(financial, "quality_score")
        large_flow = parent_metric(funding, "large_flow_ratio")
        problem_flags = financial.details.get("problem_research_flags", ()) if financial else ()
        company_type = financial.details.get("comp_type") if financial else None
        observation = observe_structure(inputs, identity)
        valuation = latest(inputs.rows("daily_basic", identity))
        price_to_earnings_ratio = valuation.number("pe_ttm", "ratio") if valuation else None
        price_to_book_ratio = valuation.number("pb", "ratio") if valuation else None
        valuation_refs = valuation.evidence("pe_ttm", "pb", "trade_date") if valuation else ()
        history_reason = inputs.reasons("daily", identity)
        flow_date = funding.envelope.source_as_of.get("moneyflow") if funding else None
        checks.extend(
            (
                gate(
                    "right_side_structure",
                    Truth.TRUE
                    if observation.reason is None
                    else Truth.FALSE
                    if observation.values
                    else Truth.UNKNOWN,
                    observation.reason or "right_side_unconfirmed",
                    references=observation.references,
                ),
                gate(
                    "right_side_history_scope",
                    Truth.UNKNOWN if history_reason else Truth.TRUE,
                    "right_side_source_incomplete",
                    references=observation.references,
                ),
                gate(
                    "financial_quality",
                    Truth.TRUE if quality is not None and company_type == "1" else Truth.UNKNOWN,
                    "financial_quality_unavailable",
                ),
                gate(
                    "financial_problem",
                    Truth.FALSE if problem_flags else Truth.TRUE,
                    "financial_problem_research",
                ),
                gate(
                    "valuation_positive",
                    Truth.UNKNOWN
                    if price_to_earnings_ratio is None or price_to_book_ratio is None
                    else Truth.TRUE
                    if price_to_earnings_ratio > 0 and price_to_book_ratio > 0
                    else Truth.FALSE,
                    "valuation_nonpositive_or_missing",
                    references=valuation_refs,
                ),
                gate(
                    "aligned_market_date",
                    Truth.TRUE
                    if valuation
                    and selected_date
                    and observation.details.get("data_as_of")
                    == valuation.availability.data_as_of
                    == selected_date
                    else Truth.UNKNOWN,
                    "market_date_mismatch",
                ),
                gate(
                    "flow_date",
                    Truth.TRUE if flow_date == selected_date and selected_date else Truth.UNKNOWN,
                    "flow_date_unaligned",
                ),
                gate(
                    "flow_support",
                    Truth.UNKNOWN
                    if large_flow is None
                    else Truth.TRUE
                    if large_flow > 0
                    else Truth.FALSE,
                    "right_side_flow_not_positive",
                ),
            )
        )
        references = (
            observation.references
            + valuation_refs
            + tuple(ref for parent in parents for ref in parent.envelope.evidence_refs)
        )
        values = observation.values | {"quality_score": quality, "large_flow_ratio": large_flow}
        metrics = tuple(
            metric(
                name,
                value,
                "元" if name in PRICE_METRICS else "score" if name == "quality_score" else "ratio",
                references,
            )
            for name, value in sorted(values.items())
        )
        # 缺结构时仍声明全部评分因子，未知值不进入分位分母。
        metrics += tuple(
            metric(name, None, "ratio", (), ("right_side_history_incomplete",))
            for name, _, _ in FACTORS
            if name not in values
        )
        conditions: tuple[Predicate, ...] = ()
        invalidations: tuple[Predicate, ...] = ()
        if observation.values:
            conditions = (
                Predicate(
                    predicate_id="entry_near_confirmation",
                    version=RULE_VERSION,
                    operator="le",
                    feature_id="breakout_extension",
                    threshold=inputs.config.right_side.maximum_breakout_extension,
                ),
                Predicate(
                    predicate_id="manual_right_side_review", version=RULE_VERSION, operator="manual"
                ),
            )
            invalidations = (
                Predicate(
                    predicate_id="structure_price_failure",
                    version=RULE_VERSION,
                    operator="lt",
                    feature_id="entry_price",
                    threshold=observation.values["invalidation_price"],
                ),
            )
        drafts.append(
            make_result(
                inputs,
                PRODUCT_IDS[4],
                identity,
                metrics=metrics,
                gates=checks,
                references=references,
                contrary=(
                    ref for check in checks if not check.passed for ref in check.evidence_refs
                ),
                parents=(parent.envelope.result_id for parent in parents),
                profile="right_side/comp_type_" + str(company_type or "unknown"),
                entry_conditions=conditions,
                invalidation_conditions=invalidations,
                details=observation.details
                | {
                    "strategy": "confirmed_pullback_right_side",
                    "policy": inputs.config.right_side.to_dict(),
                    "historical_returns_scored": False,
                    "initial_policy_unvalidated": True,
                    "problem_research_flags": list(problem_flags),
                    "structure_reason": observation.reason,
                    "next_research_question": "复核催化与盈利预期、结构失效及入场价格",
                },
            )
        )
    return rank_candidates(inputs, tuple(drafts))


def rank_candidates(
    inputs: ResearchInputs, drafts: tuple[ResearchResult, ...]
) -> tuple[ResearchResult, ...]:
    """只在相同类型与同行回退层内排列通过全部门禁的候选。

    例如两只完整同值候选在最小样本2时均50分、1.5名；缺样本或任一门禁失败不评分。
    历史涨幅没有贡献项，名次也不被解释为收益预测。
    """
    eligible = {
        item.envelope.entity_id: all(
            check.passed for check in item.envelope.quality_gates if check.hard
        )
        for item in drafts
    }
    profiles = sorted({item.envelope.feature_profile for item in drafts})
    scored = []
    for profile in profiles:
        group = [item for item in drafts if item.envelope.feature_profile == profile]
        ranks = {
            name: percentiles(
                inputs,
                {
                    item.envelope.entity_id: item.metric(name)
                    if eligible[item.envelope.entity_id]
                    else None
                    for item in group
                },
                profile + "/" + name,
                lower_is_better=lower,
            )
            for name, _, lower in FACTORS
        }
        for item in group:
            components = []
            identity = item.envelope.entity_id
            for name, weight, lower in FACTORS:
                position = ranks[name][identity]
                components.append(
                    FeatureContribution(
                        feature_id=name,
                        raw_value=item.metric(name),
                        unit="score" if name == "quality_score" else "ratio",
                        transformed_value=position.percentile,
                        weight=weight,
                        contribution=round(100 * weight * position.percentile, 12)
                        if position.percentile is not None
                        else None,
                        transform="lower_peer_percentile" if lower else "peer_percentile",
                        peer_level=position.peer_level,
                        missing_reason="right_side_factor_unavailable"
                        if position.percentile is None
                        else None,
                        reason_codes=position.reason_codes,
                        evidence_refs=item.envelope.evidence_refs,
                    )
                )
            score: float | None = round(
                fsum(component.contribution or 0 for component in components), 12
            )
            score = (
                score
                if all(component.contribution is not None for component in components)
                else None
            )
            level = ranks[FACTORS[0][0]][identity].peer_level
            path = inputs.config.peer_paths.get(identity, ())
            peer_path = path[: path.index(level) + 1] if level in path else ()
            details = item.details | {
                "factor_peer_path": list(peer_path),
                "factor_weights": {name: weight for name, weight, _ in FACTORS},
            }
            scored.append(
                replace(
                    item,
                    envelope=replace(item.envelope, score=score, contributions=tuple(components)),
                    details_json=encode(details),
                )
            )
    groups: dict[tuple[str, tuple[str, ...]], list[ResearchResult]] = {}
    for item in scored:
        key = (item.envelope.feature_profile, tuple(item.details["factor_peer_path"]))
        groups.setdefault(key, []).append(item)
    finished = []
    for (profile, _), group in sorted(groups.items()):
        final_ranks = percentiles(
            inputs,
            {item.envelope.entity_id: item.envelope.score for item in group},
            profile + "/final",
        )
        for item in group:
            position = final_ranks[item.envelope.entity_id]
            rank = (
                1 + (1 - position.percentile) * (position.peer_count - 1)
                if (position.percentile is not None)
                else None
            )
            checks = item.envelope.quality_gates + (
                gate(
                    "ranking_sample",
                    Truth.TRUE if rank is not None else Truth.UNKNOWN,
                    "insufficient_rank_sample",
                ),
            )
            envelope = replace(
                item.envelope,
                rank=rank,
                peer_count=position.peer_count,
                score=item.envelope.score if rank is not None else None,
                quality_gates=checks,
                state=ProductState.INCOMPLETE
                if rank is None and item.envelope.state is not ProductState.BLOCKED
                else item.envelope.state,
                decision_allowed=False,
                action_state=ActionState.WATCH if rank is not None else ActionState.NO_CONCLUSION,
                gate_codes=tuple(
                    sorted(
                        set(item.envelope.gate_codes)
                        | set(position.reason_codes)
                        | {reason for check in checks for reason in check.reason_codes}
                    )
                ),
            )
            finished.append(replace(item, envelope=envelope))
    return tuple(sorted(finished, key=lambda item: item.envelope.entity_id))
