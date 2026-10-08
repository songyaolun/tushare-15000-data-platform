"""P12显式持仓、基金披露范围、父结果及容量边界的离线业务场景。"""

from dataclasses import replace

import pytest
from test_research_governance import NOW, A, B, row

from tushare_entitlement_hub.cross_asset_research import portfolio
from tushare_entitlement_hub.equity_research.common import make_result, metric
from tushare_entitlement_hub.equity_research.models import ResearchConfig, ResearchInputs
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.contracts import Truth
from tushare_entitlement_hub.product_foundation.entities import (
    Entity,
    EntityAlias,
    EntityIdentity,
    EntityKind,
)

FUND = "cn:etf_share:FUND:life"
OTHER_FUND = "cn:etf_share:OTHER:life"
FUTURE = "cn:futures_contract:FUT:life"


def stock_alias(symbol="S1", entity=A, namespace="tushare_stock", **changes):
    """构造有来源证据的时点别名；例：S1→A可解析，修改entity可测试跨市场冲突。"""
    values = {
        "namespace": namespace,
        "alias": symbol,
        "entity_id": entity,
        "effective_from": "2000-01-01T00:00:00Z",
        "effective_to": None,
        "observed_at": NOW,
        "source": "stock_basic",
        "version": "portfolio-test/v1",
        "confidence": 100,
        "evidence_refs": row("stock_basic", raw={"ts_code": symbol}).evidence("ts_code"),
    }
    return EntityAlias(**(values | changes))


def portfolio_inputs(*rows, entities=(A, B, FUND, OTHER_FUND), aliases=None, reasons=None):
    """生成显式资产类型及离线来源输入；例：FUND登记为etf_share，FUTURE登记为futures_contract。"""
    registered = []
    for identity in entities:
        namespace, kind, local_id, lifecycle_id = identity.split(":")
        registered.append(
            Entity(
                identity=EntityIdentity(
                    namespace=namespace,
                    kind=EntityKind(kind),
                    local_id=local_id,
                    lifecycle_id=lifecycle_id,
                ),
                name=local_id,
            )
        )
    grouped = {}
    for observed in rows:
        grouped.setdefault(observed.source_api, []).append(observed)
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="portfolio-test-run",
        config=ResearchConfig(
            foundation=FoundationBindings(
                entities=tuple(registered),
                aliases=tuple(aliases)
                if aliases is not None
                else (stock_alias(), stock_alias("S2", B)),
            )
        ),
        rows_by_api={api: tuple(values) for api, values in grouped.items()},
        source_reasons=reasons or {},
    )


def explicit_holdings(entities=(A, B), weights=(0.6, 0.4), **changes):
    """构造完整用户持仓；例：默认返回A六成、B四成，不自动归一化或解析百分数字符串。"""
    return tuple(
        {
            "entity_type": identity.split(":")[1],
            "entity_id": identity,
            "as_of": NOW,
            "weight": weight,
            **changes,
        }
        for identity, weight in zip(entities, weights, strict=True)
    )


def fund_holding(
    record="first",
    symbol="S1",
    weight=0.6,
    *,
    fund=FUND,
    period="2026-09-30",
    public="2026-10-01T00:00:00Z",
    unit="fraction",
    raw=None,
    market_value=1000000,
    currency="CNY",
    value_unit="元",
):
    """构造已公告的基金股票部分；例：weight=.6表示披露股票口径六成，mkv不能用于补足未知权重。"""
    return row(
        "fund_portfolio",
        record,
        entity=fund,
        day=period,
        raw={
            "quality_tier": "verified",
            "symbol": symbol,
            "ann_date": public,
            "end_date": period,
            **(raw or {}),
        },
        values={"stk_mkv_ratio": (weight, unit), "mkv": (market_value, value_unit, currency)},
        report_period=period,
        public_available_at=public,
    )


def parent_result(
    inputs,
    entity=A,
    value=0.2,
    *,
    name="risk",
    unit="fraction",
    product="flow_feasibility/v1",
    metrics=None,
):
    """生成当前上下文的父研究结果；例：A的risk=.2与B的risk=.4可按真实持仓权重汇总。"""
    refs = row("daily", entity, entity=entity, raw={"close": 10}).evidence("close")
    values = tuple(metrics) if metrics is not None else (metric(name, value, unit, refs),)
    return make_result(inputs, product, entity, metrics=values, references=refs)


def test_explicit_portfolio_concentration_exposures_and_scenario_are_recomputable():
    """场景：六成股票、四成ETF的完整持仓；预期：HHI=.52、最大权重=.6、固定下跌情景为-.2。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings((A, FUND))
    result = portfolio.build_portfolio(inputs, holdings=holdings)[0]
    assert result.details["status"] == "diagnosed"
    assert result.metric("disclosed_weight") == 1
    assert result.metric("holding_hhi") == pytest.approx(0.52)
    assert result.metric("largest_holding_weight") == 0.6
    assert result.metric("top_five_weight") == 1
    assert result.metric("effective_holdings") == pytest.approx(1 / 0.52)
    assert result.metric("unknown_stock_sleeve_weight") == 0
    assert result.metric("asset_type_weight/security") == 0.6
    assert result.metric("asset_type_weight/etf_share") == 0.4
    assert result.metric("scenario_equity_minus_20pct") == -0.2
    assert result.details["scenario_assumptions"]["prediction"] is False
    assert {ref.source_api for ref in result.envelope.evidence_refs} == {"user_holdings"}
    assert not result.envelope.decision_allowed
    assert (
        portfolio.build_portfolio(inputs, holdings=holdings[::-1])[0].to_json() == result.to_json()
    )


def test_position_weight_and_declaration_time_are_preserved():
    """场景：乱序显式持仓及等价带时区时间；预期：按实体稳定排序，原权重不归一化。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings(
        (B, A), (0.4, 0.6), portfolio_id="自有组合", as_of="2026-10-03T20:00:00+08:00"
    )
    positions, name, when = portfolio.explicit_positions(inputs, holdings)
    assert [(item.entity_id, item.weight) for item in positions] == [(A, 0.6), (B, 0.4)]
    assert name == "自有组合" and when == NOW
    assert positions[0].evidence[0].locator.startswith("input:holdings/")
    assert positions[0].evidence[0].fields == ("entity_type", "entity_id", "as_of", "weight")


@pytest.mark.parametrize("field", ["entity_type", "entity_id", "as_of", "weight"])
def test_missing_user_holding_field_requires_input(field):
    """场景：用户持仓缺任一必填字段；预期：校验给固定原因，产品输出结构化输入要求。"""
    holdings = list(explicit_holdings())
    holdings[0] = {key: value for key, value in holdings[0].items() if key != field}
    inputs = portfolio_inputs()
    with pytest.raises(ValueError, match="portfolio_holding_fields_missing"):
        portfolio.explicit_positions(inputs, tuple(holdings))
    assert (
        portfolio.build_portfolio(inputs, holdings=tuple(holdings))[0].details["status"]
        == "input_required"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"entity_id": "cn:security:UNKNOWN:life"},
        {"entity_id": 1},
        {"entity_type": "etf_share"},
        {"entity_type": None},
    ],
)
def test_unregistered_or_mistyped_entity_is_not_inferred(changes):
    """场景：未知实体或声明类型与注册类型不符；预期：不按代码形状猜资产身份。"""
    holdings = explicit_holdings()
    bad = (holdings[0] | changes, holdings[1])
    with pytest.raises(ValueError, match="portfolio_entity_unresolved"):
        portfolio.explicit_positions(portfolio_inputs(), bad)


@pytest.mark.parametrize(
    "weight", [None, True, "0.6", "60%", 60, -0.1, 1.1, float("nan"), float("inf")]
)
def test_invalid_weight_type_unit_or_range_is_rejected(weight):
    """场景：比例被写成百分数字符串、百分数原值或非法数；预期：拒绝而不是猜单位或裁剪。"""
    holdings = explicit_holdings()
    with pytest.raises(ValueError, match="portfolio_weight_invalid"):
        portfolio.explicit_positions(
            portfolio_inputs(), (holdings[0] | {"weight": weight}, holdings[1])
        )


def test_duplicates_total_and_capacity_do_not_autocorrect_holdings(monkeypatch):
    """场景：重复证券、权重不满一及超过持仓预算；预期：全部拒绝，不去重后重算或自动归一化。"""
    inputs = portfolio_inputs()
    with pytest.raises(ValueError, match="portfolio_duplicate_entity"):
        portfolio.explicit_positions(inputs, explicit_holdings((A, A), (0.5, 0.5)))
    with pytest.raises(ValueError, match="portfolio_weight_total_invalid"):
        portfolio.explicit_positions(inputs, explicit_holdings(weights=(0.2, 0.3)))
    with pytest.raises(ValueError, match="portfolio_holdings_missing_or_capacity"):
        portfolio.explicit_positions(inputs, ())
    monkeypatch.setattr(portfolio, "MAXIMUM_HOLDINGS", 1)
    with pytest.raises(ValueError, match="portfolio_holdings_missing_or_capacity"):
        portfolio.explicit_positions(inputs, explicit_holdings())


@pytest.mark.parametrize("name", [None, 1, "", "   ", "名" * 129])
def test_invalid_portfolio_name_has_a_fixed_rejection_reason(name):
    """场景：组合标识不是非空短文本；预期：返回固定校验错误，不将其写进结果身份。"""
    with pytest.raises(ValueError, match="portfolio_id_invalid"):
        portfolio.explicit_positions(portfolio_inputs(), explicit_holdings(portfolio_id=name))


def test_future_mixed_times_mixed_names_and_malformed_records_require_input():
    """场景：未来持仓、混合时点、混合组合或格式错误；预期：拒绝拼成一个组合快照。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings()
    with pytest.raises(ValueError, match="portfolio_future_holding"):
        portfolio.explicit_positions(inputs, explicit_holdings(as_of="2027-01-01T00:00:00Z"))
    for change in ({"as_of": "2026-10-02T12:00:00Z"}, {"portfolio_id": "另一组合"}):
        with pytest.raises(ValueError, match="portfolio_snapshot_ambiguous"):
            portfolio.explicit_positions(inputs, (holdings[0] | change, holdings[1]))
    for invalid in (
        (None,),
        (holdings[0] | {"as_of": "bad-time"}, holdings[1]),
        (holdings[0] | {"as_of": None}, holdings[1]),
    ):
        assert (
            portfolio.build_portfolio(inputs, holdings=invalid)[0].details["status"]
            == "input_required"
        )


def test_non_equity_position_keeps_exposure_but_has_unknown_scenario_sensitivity():
    """场景：股票与期货各半；预期：资产类型暴露保留，不能把股票下跌情景套用到期货。"""
    inputs = portfolio_inputs(entities=(A, FUTURE))
    result = portfolio.build_portfolio(inputs, holdings=explicit_holdings((A, FUTURE), (0.5, 0.5)))[
        0
    ]
    assert result.metric("asset_type_weight/futures_contract") == 0.5
    assert result.metric("scenario_equity_minus_20pct") is None
    assert "scenario_asset_sensitivity_unknown" in result.envelope.gate_codes


def test_parent_features_use_entire_holding_denominator_and_exact_context():
    """场景：两项持仓均有同产品同单位风险特征；预期：权重六四汇总为.28，血缘指向两个父结果。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings()
    parents = (parent_result(inputs, A, 0.2), parent_result(inputs, B, 0.4))
    result = portfolio.build_portfolio(inputs, parents, holdings)[0]
    assert result.metric("weighted/flow_feasibility/v1/risk/fraction") == pytest.approx(0.28)
    assert result.metric("upstream_research_covered_weight") == 1
    assert set(result.parent_result_ids) == {parent.envelope.result_id for parent in parents}
    assert (
        portfolio.build_portfolio(inputs, parents[::-1], holdings)[0].to_json() == result.to_json()
    )


def test_missing_unknown_and_duplicate_parent_metrics_do_not_fill_zero():
    """场景：缺父特征、父值未知或同一持仓有重复候选；预期：完整权重汇总未知，不补零。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings()
    first = parent_result(inputs, A, 0.2)
    for parents in (
        (first,),
        (first, parent_result(inputs, B, None)),
        (first, first, parent_result(inputs, B, 0.4)),
    ):
        result = portfolio.build_portfolio(inputs, parents, holdings)[0]
        assert (
            next(
                item
                for item in result.metrics
                if item.name == "weighted/flow_feasibility/v1/risk/fraction"
            ).value
            is None
        )
        assert "portfolio_parent_metric_missing_or_ambiguous" in result.envelope.gate_codes
    no_parents = portfolio.build_portfolio(inputs, holdings=holdings)[0]
    assert no_parents.metric("upstream_research_covered_weight") == 0


def test_parent_product_and_unit_scopes_are_not_mixed():
    """场景：不同产品同名特征，以及仅有金额特征；预期：不同产品分别汇总，金额不做比例加权。"""
    inputs = portfolio_inputs()
    holdings = explicit_holdings()
    parents = (
        parent_result(inputs, A, 0.2),
        parent_result(inputs, B, 0.4, product="fundamental_dossier/v1"),
        parent_result(inputs, A, 100, name="amount", unit="元"),
    )
    result = portfolio.build_portfolio(inputs, parents, holdings)[0]
    assert (
        next(
            item
            for item in result.metrics
            if item.name == "weighted/flow_feasibility/v1/risk/fraction"
        ).value
        is None
    )
    assert (
        next(
            item
            for item in result.metrics
            if item.name == "weighted/fundamental_dossier/v1/risk/fraction"
        ).value
        is None
    )
    assert not any(
        item.name.startswith("weighted/") and "/amount/" in item.name for item in result.metrics
    )
    assert result.metric("upstream_research_covered_weight") == 1


def test_same_named_parent_metrics_with_different_units_preserve_valid_holdings():
    """场景：同产品同名特征在两项持仓上分别为ratio和fraction；预期：可比性未知，不使有效持仓失效。"""
    inputs = portfolio_inputs()
    parents = (
        parent_result(inputs, A, 0.2, unit="ratio"),
        parent_result(inputs, B, 0.4, unit="fraction"),
    )
    result = portfolio.build_portfolio(inputs, parents, explicit_holdings())[0]
    assert result.details["status"] == "diagnosed"
    weighted = [item for item in result.metrics if item.name.startswith("weighted/")]
    assert weighted and all(item.value is None for item in weighted)
    assert len({item.name for item in result.metrics}) == len(result.metrics)


@pytest.mark.parametrize(
    "field,value", [("run_id", "another-run"), ("analysis_as_of", "2026-10-04T12:00:00Z")]
)
def test_parent_context_mismatch_is_detected_but_irrelevant_parent_is_ignored(field, value):
    """场景：持仓对应父结果时点或运行不匹配；预期：明确拒绝，非持仓实体的父结果不影响诊断。"""
    inputs = portfolio_inputs()
    positions, _, _ = portfolio.explicit_positions(inputs, explicit_holdings())
    parent = parent_result(inputs)
    updates = {field: value}
    if field == "analysis_as_of":
        updates["generated_at"] = value
    wrong = replace(parent, envelope=replace(parent.envelope, **updates))
    with pytest.raises(ValueError, match="portfolio_parent_context_mismatch"):
        portfolio.parent_metrics(inputs, positions, (wrong,))
    unrelated = replace(wrong, envelope=replace(wrong.envelope, entity_id=FUND))
    metrics, ids, details = portfolio.parent_metrics(inputs, positions, (unrelated,))
    assert metrics[0].value == 0 and ids == () and details["parent_metric_count"] == 0


def test_parent_metric_limit_keeps_full_counts_and_multiset_digest():
    """场景：129个上游比例指标超过展示上限；预期：展示128个、完整计数129、摘要不依赖输入顺序。"""
    inputs = portfolio_inputs()
    refs = row("daily", raw={"close": 10}).evidence("close")
    parents = tuple(
        parent_result(
            inputs,
            entity,
            metrics=tuple(metric(f"risk_{index:03}", 0.2, "ratio", refs) for index in range(129)),
        )
        for entity in (A, B)
    )
    result = portfolio.build_portfolio(inputs, parents, explicit_holdings())[0]
    assert result.details["parent_metric_count"] == 129
    assert result.details["parent_metric_sample_count"] == 128
    assert result.details["parent_metric_truncated"] is True
    assert len([item for item in result.metrics if item.name.startswith("weighted/")]) == 128
    assert (
        result.details["parent_multiset_sha256"]
        == portfolio.build_portfolio(inputs, parents[::-1], explicit_holdings())[0].details[
            "parent_multiset_sha256"
        ]
    )


def test_fund_disclosed_stock_part_is_not_normalized_or_completed_from_market_value():
    """场景：基金披露六成股票口径且两条市值相同；预期：原权重.4/.2保留，未披露股票部分.4。"""
    first, second = fund_holding(weight=0.4), fund_holding("second", "S2", 0.2)
    inputs = portfolio_inputs(first, second)
    result = portfolio.build_portfolio(inputs)[0]
    assert result.details["weight_basis"] == "disclosed_stock_sleeve"
    assert result.metric("disclosed_weight") == pytest.approx(0.6)
    assert result.metric("holding_hhi") == pytest.approx(0.2)
    assert result.metric("largest_holding_weight") == 0.4
    assert result.metric("effective_holdings") is None
    assert result.metric("unknown_stock_sleeve_weight") == pytest.approx(0.4)
    assert result.metric("scenario_equity_minus_20pct") == pytest.approx(-0.12)
    assert result.details["full_fund_nonstock_weight"] is None
    assert "fund_nonstock_exposure_unknown" in result.envelope.gate_codes
    assert {ref.source_api for ref in result.envelope.evidence_refs} == {"fund_portfolio"}
    assert {field for ref in result.envelope.evidence_refs for field in ref.fields} >= {
        "ann_date",
        "end_date",
        "symbol",
        "mkv",
        "stk_mkv_ratio",
    }


def test_fund_and_publication_versions_remain_distinct_and_deterministic():
    """场景：两只基金及同基金两次公告；预期：三个版本独立诊断，不合并持仓或权重。"""
    rows = (
        fund_holding("a", weight=0.2),
        fund_holding("b", weight=0.3, fund=OTHER_FUND),
        fund_holding("revised", weight=0.4, public="2026-10-02T00:00:00Z"),
    )
    results = portfolio.build_portfolio(portfolio_inputs(*rows))
    assert len(results) == 3
    assert {result.metric("disclosed_weight") for result in results} == {0.2, 0.3, 0.4}
    assert len({result.envelope.result_id for result in results}) == 3
    assert [result.to_json() for result in results] == [
        result.to_json() for result in portfolio.build_portfolio(portfolio_inputs(*rows[::-1]))
    ]


def test_explicit_user_holdings_take_priority_over_fund_disclosures():
    """场景：用户持仓与基金披露同时存在；预期：只诊断用户声明组合，不拼入基金持仓。"""
    results = portfolio.build_portfolio(
        portfolio_inputs(fund_holding()), holdings=explicit_holdings()
    )
    assert len(results) == 1
    assert results[0].details["weight_basis"] == "explicit_portfolio"
    assert results[0].metric("disclosed_weight") == 1


@pytest.mark.parametrize(
    "change", ["unresolved_entity", "future", "quarantine", "flagged", "unknown_quality"]
)
def test_unusable_fund_sources_require_input_instead_of_empty_diagnosis(change):
    """场景：基金来源未绑定、未来可见或质量不可用；预期：排除来源并输出输入要求。"""
    observed = fund_holding()
    if change == "unresolved_entity":
        observed = replace(observed, entity_id=None)
    elif change == "future":
        observed = replace(
            observed,
            availability=replace(observed.availability, public_available_at="2027-01-01T00:00:00Z"),
        )
    else:
        payload = (
            {"quality_tier": "quarantine"}
            if change == "quarantine"
            else {"is_quarantined": True}
            if change == "flagged"
            else {"quality_tier": None}
        )
        observed = replace(observed, raw=observed.raw | payload)
    result = portfolio.build_portfolio(portfolio_inputs(observed))[0]
    assert result.details["status"] == "input_required"
    assert result.details["reason_codes"] == ["portfolio_holdings_missing"]


@pytest.mark.parametrize("period,public", [(None, "2026-10-01T00:00:00Z"), ("2026-09-30", None)])
def test_disclosure_period_or_publication_missing_never_uses_observation_time(period, public):
    """场景：基金披露缺报告期或公开时间；预期：整个版本降级，不用观察时间补公告日期。"""
    result = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(period=period, public=public))
    )[0]
    assert result.details["status"] == "input_required"
    assert result.metric("holding_hhi") is None


@pytest.mark.parametrize(
    "weight,unit,symbol",
    [
        (None, "fraction", "S1"),
        (-0.1, "fraction", "S1"),
        (1.1, "fraction", "S1"),
        (0.5, "%", "S1"),
        (0.5, "fraction", "UNKNOWN"),
        (0.5, "fraction", None),
    ],
)
def test_fund_weight_or_stock_identity_failure_degrades_the_entire_version(weight, unit, symbol):
    """场景：基金披露权重未认证、单位错误或股票身份不明；预期：版本整体降级而不是只保留好行。"""
    result = portfolio.build_portfolio(
        portfolio_inputs(
            fund_holding(weight=weight, unit=unit, symbol=symbol), fund_holding("valid", "S2", 0.2)
        )
    )[0]
    assert result.details["status"] == "input_required"
    assert "fund_holding_weight_or_entity_unverified" in result.details["reason_codes"]


def test_duplicate_total_and_capacity_failures_preserve_fund_version_boundary(monkeypatch):
    """场景：同股票重复、权重超一或披露行数超预算；预期：拒绝该版本，不按市值猜去重或比例。"""
    duplicate = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(), fund_holding("duplicate"))
    )[0]
    assert duplicate.details["reason_codes"] == ["fund_holding_weight_or_entity_unverified"]
    excessive = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(weight=0.8), fund_holding("second", "S2", 0.5))
    )[0]
    assert excessive.details["reason_codes"] == ["fund_holding_weight_total_invalid"]
    monkeypatch.setattr(portfolio, "MAXIMUM_HOLDINGS", 1)
    over_budget = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(), fund_holding("second", "S2", 0.4))
    )[0]
    assert over_budget.details["reason_codes"] == [
        "fund_portfolio_disclosure_time_or_capacity_unknown"
    ]


def test_holding_aliases_require_unique_market_identity_and_valid_lifecycle():
    """场景：同码跨市场冲突、已过期别名及同实体跨命名空间；预期：只有唯一有效实体可解析。"""
    assert portfolio.holding_entity(portfolio_inputs(), "S1", "2026-09-30") == A
    ambiguous = portfolio_inputs(
        aliases=(stock_alias(), stock_alias(entity=B, namespace="tushare_hk"))
    )
    assert portfolio.holding_entity(ambiguous, "S1", "2026-09-30") is None
    expired = portfolio_inputs(aliases=(stock_alias(effective_to="2025-01-01T00:00:00Z"),))
    assert portfolio.holding_entity(expired, "S1", "2026-09-30") is None
    identical = portfolio_inputs(aliases=(stock_alias(), stock_alias(namespace="tushare_us")))
    assert portfolio.holding_entity(identical, "S1", "2026-09-30") == A


def test_alias_to_unregistered_stock_does_not_crash_fund_diagnosis():
    """场景：别名指向未登记的股票实体；预期：保持实体未知并给输入要求，不能触发KeyError。"""
    missing = "cn:security:ORPHAN:life"
    inputs = portfolio_inputs(fund_holding(), aliases=(stock_alias(entity=missing),))
    result = portfolio.build_portfolio(inputs)[0]
    assert result.details["status"] == "input_required"
    assert "fund_holding_weight_or_entity_unverified" in result.details["reason_codes"]


def test_fund_parent_context_mismatch_returns_structured_input_requirement():
    """场景：基金持仓关联到另一运行的父结果；预期：该版本结构化降级，不能中断整次产品执行。"""
    inputs = portfolio_inputs(fund_holding())
    parent = parent_result(inputs)
    parent = replace(parent, envelope=replace(parent.envelope, run_id="another-run"))
    result = portfolio.build_portfolio(inputs, (parent,))[0]
    assert result.details["status"] == "input_required"
    assert "portfolio_parent_context_or_metric_invalid" in result.details["reason_codes"]


def test_near_one_excess_fund_weights_do_not_escape_as_metric_range_errors():
    """场景：权重总和略大于一但小于浮点容差；预期：结构化拒绝或合法界内诊断，不能抛范围异常。"""
    result = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(weight=0.6), fund_holding("second", "S2", 0.400000005))
    )[0]
    assert result.details["status"] == "input_required" or result.metric("disclosed_weight") <= 1


def test_input_required_contains_the_contract_and_no_fabricated_metrics():
    """场景：没有用户或基金持仓；预期：声明四个所需字段、不生成空排名或零风险指标。"""
    inputs = portfolio_inputs()
    result = portfolio.build_portfolio(inputs)[0]
    assert result.details["status"] == "input_required" and result.metrics == ()
    assert result.details["required_fields"] == ["entity_type", "entity_id", "as_of", "weight"]
    assert result.envelope.quality_gates[0].result is Truth.UNKNOWN
    custom = portfolio.input_required(inputs, "explicit_reason", profile="custom_profile")
    assert custom.envelope.feature_profile == "custom_profile"
    assert custom.details["reason_codes"] == ["explicit_reason"]


def test_holding_samples_and_evidence_stay_bounded_without_changing_full_denominators():
    """场景：300项合法持仓及128个父特征；预期：完整计数/分母保留、展示32项、结果不超过8MiB。"""
    identities = tuple(f"cn:security:P{index:04}:life" for index in range(300))
    inputs = portfolio_inputs(entities=identities, aliases=())
    holdings = explicit_holdings(identities, tuple(1 / 300 for _ in identities))
    refs = row("daily", raw={"close": 10}).evidence("close")
    parent = parent_result(
        inputs,
        identities[0],
        metrics=tuple(metric(f"risk_{index:03}", 0.2, "ratio", refs) for index in range(128)),
    )
    result = portfolio.build_portfolio(inputs, (parent,), holdings)[0]
    assert result.details["status"] == "diagnosed"
    assert result.details["position_count"] == 300
    assert result.details["position_sample_count"] == 32
    assert result.details["positions_truncated"] is True
    assert result.metric("disclosed_weight") == pytest.approx(1)
    assert result.metric("holding_hhi") == pytest.approx(1 / 300)
    assert result.metric("top_five_weight") == pytest.approx(5 / 300)
    assert result.metric("upstream_research_covered_weight") == pytest.approx(1 / 300)
    assert all(
        value.value is None for value in result.metrics if value.name.startswith("weighted/")
    )
    serialized_bytes = len(result.to_json().encode())
    assert serialized_bytes <= 8 * 1024 * 1024


def test_certified_fund_market_values_are_summed_in_one_explicit_currency():
    """场景：同币种认证基金市值两项且股票披露权重不足一；预期：汇总原币市值，不用市值补足权重。"""
    first = fund_holding(weight=0.2, market_value=120)
    second = fund_holding("second", "S2", 0.3, market_value=80)
    result = portfolio.build_portfolio(portfolio_inputs(first, second))[0]
    value = next(item for item in result.metrics if item.name == "disclosed_market_value")
    assert value.value == 200 and value.unit == "CNY:元"
    assert result.metric("disclosed_weight") == 0.5
    assert result.metric("liquidation_days") is None
    assert "portfolio_notional_and_participation_required" in result.envelope.gate_codes
    assert any("mkv" in ref.fields for ref in value.evidence_refs)


@pytest.mark.parametrize(
    "options",
    [{"market_value": None}, {"market_value": -1}, {"value_unit": "ratio"}, {"value_unit": "万元"}],
)
def test_invalid_valuation_does_not_destroy_certified_weights(options):
    """场景：市值缺失、为负或未经目标单位认证；预期：权重仍可诊断，估值保持未知而不猜换算。"""
    result = portfolio.build_portfolio(portfolio_inputs(fund_holding(**options)))[0]
    assert result.details["status"] == "diagnosed"
    assert result.metric("disclosed_weight") == 0.6
    assert result.metric("disclosed_market_value") is None
    assert "holding_valuation_or_currency_unverified" in result.envelope.gate_codes


def test_mixed_valuation_currencies_and_missing_value_fact_remain_unknown():
    """场景：两笔市值币种不同，或权重已认证但市值事实缺失；预期：不进行隐式换汇或补值。"""
    mixed = portfolio.build_portfolio(
        portfolio_inputs(fund_holding(weight=0.2), fund_holding("hk", "S2", 0.3, currency="HKD"))
    )[0]
    assert mixed.metric("disclosed_market_value") is None
    first = fund_holding()
    untyped = replace(first, numeric={"stk_mkv_ratio": first.numeric["stk_mkv_ratio"]})
    assert (
        portfolio.build_portfolio(portfolio_inputs(untyped))[0].metric("disclosed_market_value")
        is None
    )
    declared = portfolio.build_portfolio(
        portfolio_inputs(), holdings=explicit_holdings(currency="CNY")
    )[0]
    assert declared.metric("disclosed_market_value") is None
    assert declared.metric("liquidation_days") is None


@pytest.mark.parametrize("field", ["observed_at", "public_available_at"])
def test_future_parent_metric_evidence_is_rejected_even_if_envelope_is_current(field):
    """场景：父信封当前有效，但嵌套指标引用未来观察或公开证据；预期：拒绝未来证据并结构化降级。"""
    inputs = portfolio_inputs(fund_holding())
    parent = parent_result(inputs)
    value = parent.metrics[0]
    future = replace(value.evidence_refs[0], **{field: "2027-01-01T00:00:00Z"})
    contaminated = replace(parent, metrics=(replace(value, evidence_refs=(future,)),))
    positions, _, _ = portfolio.explicit_positions(inputs, explicit_holdings())
    with pytest.raises(ValueError, match="portfolio_parent_evidence_unavailable"):
        portfolio.parent_metrics(inputs, positions, (contaminated,))
    result = portfolio.build_portfolio(inputs, (contaminated,))[0]
    assert result.details["status"] == "input_required"


def test_explicit_near_one_excess_is_rejected_without_clipping():
    """场景：用户权重总和1.000000005；预期：拒绝大于一的合计，不裁剪或归一化。"""
    with pytest.raises(ValueError, match="portfolio_weight_total_invalid"):
        portfolio.explicit_positions(
            portfolio_inputs(), explicit_holdings(weights=(0.6, 0.400000005))
        )
