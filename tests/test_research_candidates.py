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
from tushare_entitlement_hub.product_foundation.contracts import ProductState, Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity, EntityKind
from tushare_entitlement_hub.product_foundation.numeric import UnitCertification, adapt_numeric
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

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
        if name not in {"pe_ttm", "pb", "amount", "pct_chg", "close", "high", "low"}:
            continue
        unit = (
            "元"
            if name in {"amount", "close", "high", "low"}
            else "fraction"
            if name == "pct_chg"
            else "ratio"
        )
        raw.setdefault("source_units", {})[name] = unit
        certificate = UnitCertification(
            version="unit/v1",
            source_api=api,
            field=name,
            source_unit=unit,
            target_unit=unit,
            dimension="money"
            if name == "amount"
            else "price"
            if name in {"close", "high", "low"}
            else "ratio",
            currency=currency if unit == "元" else None,
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


def test_confirmed_candidates_rank_without_historical_return_rewards():
    """场景：两只同值回调突破且父门禁齐备；预期：50分、1.5名，历史收益不参与贡献。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    inputs = right_side_inputs()
    results = build_candidates(inputs, *right_side_parents(inputs))
    for result in results:
        assert result.envelope.product_id == "equity_right_side_candidates/v1"
        assert result.envelope.score == 50
        assert result.envelope.rank == 1.5
        assert not result.envelope.decision_allowed
        assert result.details["historical_returns_scored"] is False
        assert result.details["strategy"] == "confirmed_pullback_right_side"
        assert not any("return" in item.feature_id for item in result.envelope.contributions)
        assert result.envelope.entry_conditions
        assert result.envelope.invalidation_conditions[0].threshold == result.metric(
            "invalidation_price"
        )
        assert len(result.parent_result_ids) == 3


@pytest.mark.parametrize("pattern", ["left_side", "extended", "late"])
def test_left_side_extended_and_insufficient_headroom_never_rank(pattern):
    """场景：仍下跌、突破过远或接近历史阻力；预期：无评分名次且保留结构失败原因。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    inputs = right_side_inputs(pattern=pattern)
    for result in build_candidates(inputs, *right_side_parents(inputs)):
        assert result.envelope.score is None and result.envelope.rank is None
        assert result.details["structure_reason"] is not None
        assert not result.envelope.decision_allowed


@pytest.mark.parametrize("truth", [Truth.FALSE, Truth.UNKNOWN])
def test_hard_parent_risk_cannot_be_compensated_by_confirmed_structure(truth):
    """场景：右侧结构已确认但治理风险失败或未知；预期：高质量与资金不能抵消门禁。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    inputs = right_side_inputs(3)
    fundamental, governance, funding = right_side_parents(inputs)
    bad = replace(
        governance[0],
        envelope=replace(
            governance[0].envelope,
            quality_gates=(
                gate(
                    "risk",
                    truth,
                    "risk_unresolved",
                    references=governance[0].envelope.evidence_refs,
                ),
            ),
        ),
    )
    first, *others = build_candidates(inputs, fundamental, (bad, *governance[1:]), funding)
    assert first.envelope.score is None and first.envelope.rank is None
    assert "risk_unresolved" in first.envelope.gate_codes
    assert first.envelope.contrary_evidence
    assert all(item.envelope.rank is not None for item in others)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "future", "wrong_product", "empty_gates"]
)
def test_parent_identity_time_uniqueness_and_checks_are_required(index, failure):
    """场景：各父结果缺失、重复、未来、错误产品或无门禁；预期：受影响实体不评分。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    inputs = right_side_inputs(3)
    groups = list(right_side_parents(inputs))
    first, *others = groups[index]
    if failure == "missing":
        groups[index] = tuple(others)
    elif failure == "duplicate":
        groups[index] = (first, first, *others)
    else:
        envelope = first.envelope
        if failure == "future":
            envelope = replace(
                envelope, analysis_as_of="2026-10-04T00:00:00Z", generated_at="2026-10-04T00:00:00Z"
            )
        elif failure == "wrong_product":
            envelope = replace(envelope, product_id="wrong/v1")
        else:
            envelope = replace(envelope, quality_gates=())
        groups[index] = (replace(first, envelope=envelope), *others)
    result = build_candidates(inputs, *groups)[0]
    assert result.envelope.rank is None and result.envelope.score is None


@pytest.mark.parametrize(
    "case",
    [
        "missing_pe",
        "negative_pb",
        "stale_valuation",
        "stale_flow",
        "negative_flow",
        "unknown_quality",
        "problem",
        "source_gap",
        "financial_type",
    ],
)
def test_valuation_flow_quality_and_source_gaps_fail_closed(case):
    """场景：估值、资金、财务类型或来源存在反证/缺口；预期：结构确认也不取得名次。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    from tushare_entitlement_hub.store import encode

    inputs = right_side_inputs()
    groups = list(
        right_side_parents(
            inputs,
            quality=None if case == "unknown_quality" else 80,
            flow_ratio=-0.1 if case == "negative_flow" else 0.05,
        )
    )
    identity = inputs.entity_ids[0]
    if case in {"missing_pe", "negative_pb", "stale_valuation"}:
        row = research_row(
            identity,
            "daily_basic",
            {
                "pe_ttm": None if case == "missing_pe" else 10,
                "pb": -1 if case == "negative_pb" else 2,
            },
            day="2026-09-29" if case == "stale_valuation" else "2026-09-30",
        )
        inputs = replace(
            inputs,
            rows_by_api=inputs.rows_by_api | {"daily_basic": (row, inputs.rows("daily_basic")[1])},
        )
    elif case == "stale_flow":
        first = groups[2][0]
        groups[2] = (
            replace(
                first, envelope=replace(first.envelope, source_as_of={"moneyflow": "2026-09-29"})
            ),
            groups[2][1],
        )
    elif case in {"problem", "financial_type"}:
        first = groups[0][0]
        details = {
            "comp_type": "2" if case == "financial_type" else "1",
            "problem_research_flags": ["negative_equity"] if case == "problem" else [],
        }
        groups[0] = (replace(first, details_json=encode(details)), groups[0][1])
    elif case == "source_gap":
        inputs = replace(
            inputs, entity_source_reasons={"daily/" + identity: ("capacity_exceeded",)}
        )
    result = build_candidates(inputs, *groups)[0]
    assert result.envelope.rank is None and result.envelope.score is None


def test_missing_window_and_small_sample_do_not_fabricate_scores():
    """场景：窗口不足或完整候选少于最小样本；预期：保留未知原因，分数名次均为空。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    missing = research_inputs(2, minimum=2)
    assert all(
        item.envelope.score is None
        for item in build_candidates(missing, *research_parents(missing))
    )
    inputs = right_side_inputs(1)
    result = build_candidates(inputs, *right_side_parents(inputs))[0]
    assert result.envelope.score is None and result.envelope.rank is None
    assert "insufficient_rank_sample" in result.envelope.gate_codes
    assert result.envelope.state is ProductState.INCOMPLETE
    assert build_candidates(research_inputs(0), (), (), ()) == ()


def test_parent_metric_rejects_missing_unknown_and_wrong_units():
    """场景：父指标未知或不是比例/分数；预期：不从金额猜测资金比例。"""
    from tushare_entitlement_hub.equity_research.candidates import parent_metric

    inputs = research_inputs(1)
    parent = research_parents(inputs)[0][0]
    assert parent_metric(None, "quality_score") is None
    assert parent_metric(parent, "quality_score") is None
    refs = parent.envelope.evidence_refs
    changed = replace(parent, metrics=(metric("quality_score", 10, "元", refs),))
    assert parent_metric(changed, "quality_score") is None


def test_peer_fallback_levels_are_not_mixed_in_final_ranking():
    """场景：两只在叶行业取分位，一只回退上级；预期：回退者不借叶行业名次。"""
    from right_side_fixture import right_side_inputs, right_side_parents

    inputs = right_side_inputs(3)
    paths = {
        identity: ("CN", "industry", "large" if index < 2 else "small")
        for index, identity in enumerate(inputs.entity_ids)
    }
    inputs = replace(inputs, config=replace(inputs.config, peer_paths=paths))
    results = build_candidates(inputs, *right_side_parents(inputs))
    assert all(item.envelope.rank is not None for item in results[:2])
    assert results[-1].envelope.rank is None and results[-1].envelope.score is None
