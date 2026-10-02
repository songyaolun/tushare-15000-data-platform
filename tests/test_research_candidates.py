"""P03 确定性截面场景；不使用真实行情、凭证或网络。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.equity_research.candidates import build_candidates
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import ActionState, ProductState, Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity, EntityKind
from tushare_entitlement_hub.product_foundation.numeric import UnitCertification, adapt_numeric
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact
from tushare_entitlement_hub.store import encode

AS_OF = "2026-10-03T00:00:00Z"


def research_row(identity, api, fields, *, day="2026-09-30", record="row", currency="CNY"):
    """构造已认证的合成证据；例：amount=100 输出 CNY 元事实，缺值保持 None。"""
    raw = {
        "source_api": api,
        "source_record_id": record,
        "business_group_key": record,
        "business_variant_key": None,
        "contract_hash": "contract-v1",
        "quality_tier": "verified",
        **fields,
    }
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="time/v1",
        observed_at=AS_OF,
        data_as_of=day,
        public_available_at="2026-10-01T00:00:00Z",
    )
    row = ResearchRow(entity_id=identity, source_api=api, raw=raw, availability=availability)
    numeric = {}
    for name in fields:
        if name not in {"pe_ttm", "pb", "amount", "pct_chg"}:
            continue
        unit = "元" if name == "amount" else "fraction" if name == "pct_chg" else "ratio"
        raw.setdefault("source_units", {})[name] = unit
        certificate = UnitCertification(
            version="unit/v1",
            source_api=api,
            field=name,
            source_unit=unit,
            target_unit=unit,
            dimension="money" if name == "amount" else "ratio",
            currency=currency if name == "amount" else None,
            multiplier=1,
            evidence_refs=row.evidence(name),
        )
        numeric[name] = adapt_numeric(raw, name, identity, availability, certificate)
    return replace(row, numeric=numeric)


def research_inputs(count=30, *, minimum=30, flat=False):
    """场景：显式同组股票与当前截面；预期：30 个样本能排名，29 个不能。

    例如 flat=True 时 PE、PB、成交额相同，各因子平均秩分位均为0.5。
    """
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn",
                kind=EntityKind.SECURITY,
                local_id=f"A{index:03d}",
                lifecycle_id="life",
            ),
            name=f"企业{index}",
        )
        for index in range(count)
    )
    identities = tuple(entity.identity.entity_id for entity in entities)
    return ResearchInputs(
        analysis_as_of=AS_OF,
        run_id="research-run",
        config=ResearchConfig(
            foundation=FoundationBindings(entities=entities),
            thresholds=FoundationPolicy(version="test/v1", minimum_peer_count=minimum),
            peer_paths={identity: ("CN", "nonfinancial", "industry") for identity in identities},
        ),
        rows_by_api={
            "daily": tuple(
                research_row(
                    identity,
                    "daily",
                    {"close": 10, "vol": 100, "amount": 100 if flat else 100 * (index + 1)},
                    record="daily-" + identity,
                )
                for index, identity in enumerate(identities)
            ),
            "daily_basic": tuple(
                research_row(
                    identity,
                    "daily_basic",
                    {"pe_ttm": 10 if flat else count - index, "pb": 2},
                    record="basic-" + identity,
                )
                for index, identity in enumerate(identities)
            ),
        },
    )


def research_parents(inputs, *, quality=None, checks=None):
    """构造三个明确父产品；例：quality=80 为每个实体提供 P04 的 quality_score。

    quality 字典支持同批次完整档和市场档，checks 只覆盖第一个证券的风险门禁。
    """
    groups = [[], [], []]
    products = ("fundamental_dossier/v1", "capital_governance_risks/v1", "flow_feasibility/v1")
    for identity in inputs.entity_ids:
        references = inputs.rows("daily", identity)[0].evidence("amount")
        value = quality.get(identity) if isinstance(quality, dict) else quality
        for index, product in enumerate(products):
            metrics = (metric("quality_score", value, "score", references),) if index == 0 else ()
            gates = (
                checks
                if checks is not None and index == 1 and identity == inputs.entity_ids[0]
                else (gate("verified", Truth.TRUE, ""),)
            )
            groups[index].append(
                make_result(
                    inputs,
                    product,
                    identity,
                    metrics=metrics,
                    references=references,
                    gates=gates,
                    details={"comp_type": "1"} if index == 0 else {},
                )
            )
    return tuple(tuple(group) for group in groups)


def test_complete_formula_ties_and_research_capacity():
    """场景：30 只完全同值且 Q0=80；预期：63.5 分、15.5 名且不跨越前20%容量。"""
    inputs = research_inputs(flat=True)
    results = build_candidates(inputs, *research_parents(inputs, quality=80))
    assert len(results) == 30
    for result in results:
        assert result.envelope.feature_profile == "fundamental_complete/comp_type_1"
        assert result.envelope.score == 63.5
        assert result.envelope.rank == 15.5
        assert result.details["rank_interval_without_one_factor"] == [15.5, 15.5]
        assert result.details["research_capacity"] == 6
        assert result.envelope.action_state is ActionState.WATCH
        assert not result.envelope.decision_allowed
        assert len(result.parent_result_ids) == 3
        assert result.metric("momentum_12_1") is None


def test_complete_high_rank_enables_only_manual_research():
    """场景：完整档单调因子且硬门禁通过；预期：首名允许人工研究并保留删因子区间。"""
    inputs = research_inputs()
    results = build_candidates(inputs, *research_parents(inputs, quality=80))
    best = results[-1]
    assert best.envelope.rank == 1
    assert best.envelope.decision_allowed
    assert best.envelope.action_state is ActionState.RESEARCH_FIRST
    assert best.envelope.entry_conditions[0].operator == "manual"
    assert best.details["rank_interval_without_one_factor"] == [1, 1]
    assert (
        sum(component.contribution for component in best.envelope.contributions)
        == best.envelope.score
    )


def test_market_formula_and_profile_isolation():
    """场景：同批30只完整档、29只市场档；预期：不混排，市场档没有足够分位或名次。"""
    inputs = research_inputs(59, flat=True)
    quality = {identity: 80 for identity in inputs.entity_ids[:30]}
    results = build_candidates(inputs, *research_parents(inputs, quality=quality))
    assert results[0].envelope.score == 63.5
    assert results[-1].envelope.feature_profile == "market_only/comp_type_1"
    assert results[-1].envelope.score is None
    assert results[-1].envelope.rank is None
    assert results[-1].details["rank_interval_without_one_factor"] is None
    market_inputs = research_inputs(flat=True)
    market = build_candidates(market_inputs, *research_parents(market_inputs))
    assert market[0].envelope.score == 50
    assert market[0].metric("quality_score") is None
    assert market[0].metrics[0].status == "not_applicable"
    assert not market[0].envelope.decision_allowed


@pytest.mark.parametrize("truth", [Truth.FALSE, Truth.UNKNOWN])
def test_hard_risk_removes_candidate_before_ranking(truth):
    """场景：31只中一只风险失败或未知；预期：该股票无分数，剩余30只可排名。"""
    inputs = research_inputs(31)
    references = inputs.rows("daily", inputs.entity_ids[0])[0].evidence("amount")
    parents = research_parents(
        inputs,
        quality=100,
        checks=(gate("governance_st", truth, "risk_unresolved", references=references),),
    )
    result, *others = build_candidates(inputs, *parents)
    assert result.envelope.score is None
    assert result.envelope.rank is None
    assert not result.envelope.decision_allowed
    assert result.envelope.contrary_evidence
    assert all(other.envelope.rank is not None for other in others)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"pe_ttm": -1}, "problem_research"),
        ({"pb": 0}, "problem_research"),
        ({"pe_ttm": None}, "market_only"),
        ({"pe_ttm": 1e-320}, "market_only"),
    ],
)
def test_nonpositive_and_missing_valuation_stays_unscored(changes, expected):
    """场景：负PE、零PB或缺估值；预期：问题档或缺口档，不把无效估值当便宜。"""
    inputs = research_inputs(1)
    identity = inputs.entity_ids[0]
    valuation = research_row(identity, "daily_basic", {"pe_ttm": 10, "pb": 2, **changes})
    inputs = replace(inputs, rows_by_api={**inputs.rows_by_api, "daily_basic": (valuation,)})
    result = build_candidates(inputs, *research_parents(inputs))[0]
    assert result.details["score_profile"] == expected
    assert result.envelope.score is None
    assert result.details["next_research_question"]


@pytest.mark.parametrize(
    "kind",
    [
        "missing_daily",
        "missing_valuation",
        "date_mismatch",
        "older_daily",
        "currency",
        "missing_amount",
        "zero_amount",
    ],
)
def test_market_input_failures_are_explicit(kind):
    """场景：行情缺失、日期错配或金额认证不合格；预期：核心门禁不通过且无分数。"""
    inputs = research_inputs(2)
    parents = research_parents(inputs)
    daily = inputs.rows("daily")[0]
    basic = inputs.rows("daily_basic")[0]
    rows = dict(inputs.rows_by_api)
    if kind == "missing_daily":
        rows["daily"] = ()
    elif kind == "missing_valuation":
        rows["daily_basic"] = ()
    elif kind in {"date_mismatch", "older_daily"}:
        target = basic if kind == "date_mismatch" else daily
        changed = replace(
            target, availability=replace(target.availability, data_as_of="2026-09-29")
        )
        rows[target.source_api] = (changed, rows[target.source_api][1])
    else:
        changed = research_row(
            inputs.entity_ids[0],
            "daily",
            {
                "close": 10,
                "vol": 10,
                "amount": None if kind == "missing_amount" else 0 if kind == "zero_amount" else 100,
            },
            currency="USD" if kind == "currency" else "CNY",
        )
        rows["daily"] = (changed, rows["daily"][1])
    result = build_candidates(replace(inputs, rows_by_api=rows), *parents)[0]
    assert result.envelope.score is None
    assert any(not check.passed for check in result.envelope.quality_gates)


def test_missing_and_conflicting_parent_results_fail_closed():
    """场景：父结果缺失、多版本、错截止或缺门禁；预期：不能以高质量分取得研究资格。"""
    inputs = research_inputs(2, minimum=2)
    fundamental, governance, flow = research_parents(inputs, quality=100)
    conflicts = (fundamental[0], fundamental[0], fundamental[1])
    wrong_time = replace(
        flow[0],
        envelope=replace(
            flow[0].envelope,
            analysis_as_of="2026-10-03T01:00:00Z",
            generated_at="2026-10-03T01:00:00Z",
        ),
    )
    empty_checks = replace(
        governance[1], envelope=replace(governance[1].envelope, quality_gates=())
    )
    results = build_candidates(inputs, conflicts, (empty_checks,), (wrong_time, flow[1]))
    assert results[0].envelope.feature_profile == "market_only/comp_type_unknown"
    assert results[0].envelope.score is None
    assert results[1].envelope.score is None


def test_fundamental_unknown_downgrades_but_hard_false_still_blocks():
    """场景：财务未知和审计硬失败；预期：前者可用市场档，后者仍进入复核队列。"""
    inputs = research_inputs(2, minimum=2)
    fundamental, governance, flow = research_parents(inputs, quality=80)
    missing = replace(
        fundamental[0],
        envelope=replace(
            fundamental[0].envelope,
            quality_gates=(gate("statement", Truth.UNKNOWN, "statement_missing"),),
        ),
    )
    bad = replace(
        fundamental[1],
        envelope=replace(
            fundamental[1].envelope, quality_gates=(gate("audit", Truth.FALSE, "audit_adverse"),)
        ),
    )
    results = build_candidates(inputs, (missing, bad), governance, flow)
    assert results[0].envelope.feature_profile == "market_only/comp_type_1"
    assert results[1].envelope.state is ProductState.BLOCKED
    assert "audit_adverse" in results[1].envelope.gate_codes


def test_empty_inputs_produce_no_fabricated_security():
    """场景：没有登记任何股票；预期：P03返回空股票集合，不虚构可排名实体。"""
    assert build_candidates(research_inputs(0), (), (), ()) == ()


def test_accounting_type_isolates_financial_and_unknown_market_profiles():
    """场景：工业样本足够、金融及未知口径不足；预期：后两者不借工业样本取得分位。"""
    inputs = research_inputs(33, flat=True)
    fundamental, governance, flow = research_parents(inputs)
    changes = {30: "2", 31: None, 32: "unexpected"}
    fundamental = tuple(
        replace(parent, details_json=encode({"comp_type": changes[index]}))
        if index in changes
        else parent
        for index, parent in enumerate(fundamental)
    )
    results = build_candidates(inputs, fundamental, governance, flow)
    assert all(result.envelope.rank is not None for result in results[:30])
    assert results[30].envelope.feature_profile == "market_only/comp_type_2"
    assert results[31].envelope.feature_profile == "market_only/comp_type_unknown"
    assert all(result.envelope.score is None for result in results[30:])


def test_distinct_percentile_fallback_levels_never_share_final_ranking():
    """场景：30只在叶行业取分位、1只回退上级；预期：回退样本不与叶行业分数混排。"""
    inputs = research_inputs(31)
    paths = {
        identity: ("CN", "industry", "large" if index < 30 else "small")
        for index, identity in enumerate(inputs.entity_ids)
    }
    inputs = replace(inputs, config=replace(inputs.config, peer_paths=paths))
    results = build_candidates(inputs, *research_parents(inputs, quality=80))
    assert results[0].details["factor_peer_path"] == ["CN", "industry", "large"]
    assert results[-1].details["factor_peer_path"] == ["CN", "industry"]
    assert results[-1].envelope.score is not None
    assert results[-1].envelope.rank is None
    assert results[-1].envelope.action_state is ActionState.NO_CONCLUSION
    assert all(result.envelope.peer_count == 30 for result in results[:30])


@pytest.mark.parametrize("flag", ["negative_equity", "negative_earnings"])
def test_certified_financial_problem_overrides_positive_market_valuation(flag):
    """场景：PE/PB仍为正但财务已认证负权益或亏损；预期：问题研究档保留父反证且禁排名。"""
    inputs = research_inputs(1)
    fundamental, governance, flow = research_parents(inputs, quality=100)
    parent = replace(
        fundamental[0],
        details_json=encode({"comp_type": "1", "problem_research_flags": [flag]}),
        envelope=replace(
            fundamental[0].envelope, contrary_evidence=fundamental[0].envelope.evidence_refs
        ),
    )
    result = build_candidates(inputs, (parent,), governance, flow)[0]
    assert result.details["score_profile"] == "problem_research"
    assert result.details["problem_research_flags"] == [flag]
    assert result.details["exclusion_reason"] == "financial_problem_research"
    assert result.envelope.contrary_evidence == parent.envelope.contrary_evidence
    assert result.envelope.score is None
    assert not result.envelope.decision_allowed
