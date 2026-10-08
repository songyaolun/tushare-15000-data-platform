"""财务研究的离线合成测试，不访问真实令牌或外部服务。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.equity_research.fundamental import build_fundamental
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity, EntityKind
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-02T12:00:00Z"


def statement(
    api,
    index=0,
    record=None,
    fields=None,
    *,
    public="2026-08-30T16:00:00Z",
    observed=NOW,
    report="2026-06-30",
    currency="CNY",
    dimension="money",
):
    """构造声明币种和报告期的合成报表；例：资产100、负债40及权益60可核对平衡。"""
    defaults = {
        "income": {"revenue": 100.0},
        "cashflow": {"n_cashflow_act": 20.0},
        "balancesheet": {
            "total_assets": 100.0,
            "total_liab": 40.0,
            "money_cap": 30.0,
            "total_cur_liab": 50.0,
            "total_hldr_eqy_inc_min_int": 60.0,
        },
        "fina_audit": {"audit_result": "标准无保留意见"},
        "daily_basic": {"pe_ttm": 10.0, "pb": 2.0, "trade_date": "20260930"},
    }[api.removesuffix("_vip")]
    values = {
        "end_date": "20260630",
        "report_type": "1",
        "comp_type": "1",
        "ann_date": "20260830",
        "update_flag": "0",
        **defaults,
        **(fields or {}),
    }
    record = record or api + str(index)
    raw = {
        **values,
        "source_api": api,
        "source_record_id": record,
        "business_group_key": record,
        "contract_hash": "fixture-contract",
    }
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="fixture-time",
        observed_at=observed,
        public_available_at=public,
        data_as_of=report,
        report_period=report,
    )
    item = ResearchRow(
        entity_id=f"cn:security:{index}:listed", source_api=api, raw=raw, availability=availability
    )
    facts = {}
    for field, value in values.items():
        if type(value) not in {float, int}:
            continue
        unit = "ratio" if api == "daily_basic" else "元"
        selected_dimension = "ratio" if api == "daily_basic" else dimension
        refs = item.evidence(field)
        certificate = UnitCertification(
            version="fixture-unit",
            source_api=api,
            field=field,
            source_unit=unit,
            target_unit=unit,
            dimension=selected_dimension,
            currency=None if api == "daily_basic" else currency,
            multiplier=1,
            evidence_refs=refs,
        )
        facts[field] = TypedFact(
            fact_id=record + field,
            entity_id=item.entity_id,
            field=field,
            value=value,
            unit=unit,
            dimension=selected_dimension,
            currency=certificate.currency,
            availability=availability,
            quality_tier="verified",
            certification=certificate,
            evidence_refs=refs,
        )
    return replace(item, numeric=facts)


def reports(index=0, fields=None):
    """构造完整三表和标准审计；例：默认输入得到可计算现金比率的四条合成记录。"""
    return tuple(
        statement(api, index, fields=fields)
        for api in ("income_vip", "cashflow_vip", "balancesheet_vip", "fina_audit")
    )


def research(rows=(), count=2, *, peers=True, certified=("1",)):
    """构造最低同行数量为2的认证测试配置；例：peers=False时仍有实体但不能评分。"""
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind=EntityKind.SECURITY, local_id=str(index), lifecycle_id="listed"
            ),
            name="合成证券",
        )
        for index in range(count)
    )
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="financial-fixture",
        config=ResearchConfig(
            foundation=FoundationBindings(entities=entities),
            thresholds=FoundationPolicy(version="test", minimum_peer_count=2),
            peer_paths={item.identity.entity_id: ("industry",) for item in entities}
            if peers
            else {},
            certified_report_types=certified,
        ),
        rows_by_api={
            api: tuple(row for row in rows if row.source_api == api)
            for api in {row.source_api for row in rows}
        },
    )


def test_financial_hand_calculation_and_tied_quality_score():
    """场景：两个同业三表完全相同；预期：现金比率0.2、杠杆0.4、现金覆盖0.6、Q0=50。"""
    results = build_fundamental(research(reports() + reports(1)))
    for result in results:
        assert result.metric("cash_margin") == 0.2
        assert result.metric("leverage") == 0.4
        assert result.metric("cash_cover") == 0.6
        assert result.metric("balance_error") == 0
        assert result.metric("quality_score") == result.metric("Q0") == 50
        assert result.envelope.score == 50
        assert all(check.result is Truth.TRUE for check in result.envelope.quality_gates)
        assert result.details["audit_flags"] == []
        assert not result.envelope.decision_allowed


def test_financial_rank_directions_and_valuation_basis():
    """场景：第二家现金更强且杠杆更低；预期：Q0为100/0，估值保留供应商口径。"""
    rows = reports() + reports(
        1,
        {
            "n_cashflow_act": 30.0,
            "total_liab": 30.0,
            "total_hldr_eqy_inc_min_int": 70.0,
            "money_cap": 40.0,
        },
    )
    rows += (statement("daily_basic"),)
    first, second = build_fundamental(research(rows))
    assert first.metric("Q0") == 0
    assert second.metric("Q0") == 100
    assert first.details["valuation_basis"] == {"basis": "provider_reported", "pe_ttm": 10, "pb": 2}


def test_empty_financial_and_missing_statements():
    """场景：没有显式实体或没有报表；预期：空实体返回空元组，已绑定实体返回未知档案。"""
    assert build_fundamental(research(count=0)) == ()
    (result,) = build_fundamental(research(count=1))
    assert result.metric("Q0") is None
    assert result.metric("cash_margin") is None
    assert "statement_missing" in result.envelope.gate_codes


@pytest.mark.parametrize("peers,count", [(False, 2), (True, 1)])
def test_peer_path_and_sample_minimum_are_required(peers, count):
    """场景：同行路径未认证或人数不足；预期：保留财务原比率，不发布Q0。"""
    result = build_fundamental(research(reports() + reports(1), peers=peers, count=count))[0]
    assert result.metric("cash_margin") == 0.2
    assert result.metric("Q0") is None


def test_different_reporting_periods_never_share_percentiles():
    """场景：同行公司报告期不同；预期：各组只有一家，Q0均为空。"""
    older = tuple(
        statement(api, 1, fields={"end_date": "20260331"}, report="2026-03-31")
        for api in ("income_vip", "cashflow_vip", "balancesheet_vip", "fina_audit")
    )
    result = build_fundamental(research(reports() + older))
    assert all(item.metric("Q0") is None for item in result)


def test_latest_public_revision_ignores_update_flag_and_distinct_announcement_dates():
    """场景：旧版标记1、新版标记0且三表公告日期不同；预期：按公开时间选新版，正常连接。"""
    old = statement(
        "income_vip",
        record="old",
        fields={"revenue": 80.0, "update_flag": "1"},
        public="2026-08-01T16:00:00Z",
    )
    rows = reports() + (old,)
    (result,) = build_fundamental(research(rows, count=1))
    assert result.metric("cash_margin") == 0.2
    assert result.details["revision_conflicts"] == []
    assert all(ref.source_record_id != "old" for ref in result.envelope.evidence_refs)


def test_matching_ordinary_sources_do_not_duplicate_scoring_evidence():
    """场景：普通三表与VIP逐字段一致；预期：核对一致且核心证据只用VIP。"""
    rows = reports() + tuple(statement(api) for api in ("income", "cashflow", "balancesheet"))
    (result,) = build_fundamental(research(rows, count=1))
    assert all(item["agreement"] is True for item in result.details["source_comparisons"])
    assert all(
        ref.source_api.endswith("_vip")
        for feature in result.metrics[:4]
        for ref in feature.evidence_refs
    )


def test_ordinary_fallback_and_crosscheck_unknown():
    """场景：VIP缺失时普通版替代，另有普通字段未知；预期：替代可计算，未知核对不伪报一致。"""
    rows = tuple(statement(api) for api in ("income", "cashflow", "balancesheet", "fina_audit"))
    (result,) = build_fundamental(research(rows, count=1))
    assert result.metric("cash_margin") == 0.2
    unknown = replace(statement("income"), numeric={})
    (result,) = build_fundamental(research(reports() + (unknown,), count=1))
    assert result.details["source_comparisons"][0]["agreement"] is None


@pytest.mark.parametrize("kind", ["ordinary_conflict", "same_time", "unknown_time", "variant"])
def test_financial_revision_conflicts_block_metrics(kind):
    """场景：普通版冲突、同刻双版本、修订时间未知或原始变体；预期：阻断计分并保留反证。"""
    extra = statement(
        "income" if kind == "ordinary_conflict" else "income_vip",
        record="extra",
        fields={"revenue": 200.0},
        public=None if kind == "unknown_time" else "2026-08-30T16:00:00Z",
    )
    if kind == "variant":
        extra = replace(extra, raw=extra.raw | {"business_variant_key": "variant"})
    (result,) = build_fundamental(research(reports() + (extra,), count=1))
    assert result.metric("cash_margin") is None
    assert result.details["revision_conflicts"]
    assert result.envelope.contrary_evidence


@pytest.mark.parametrize(
    "fields,report",
    [
        ({"end_date": None}, "2026-06-30"),
        ({"end_date": "invalid"}, "2026-06-30"),
        ({"end_date": "20260331"}, "2026-06-30"),
        ({"report_type": None}, "2026-06-30"),
        ({"comp_type": None}, "2026-06-30"),
        ({"report_type": "2"}, "2026-06-30"),
    ],
)
def test_mismatched_or_unknown_statement_context_fails_closed(fields, report):
    """场景：三表报告期、类型或时间适配不一致；预期：不做跨表比率。"""
    changed = statement("income_vip", fields=fields, report=report)
    rows = tuple(item for item in reports() if item.source_api != "income_vip") + (changed,)
    (result,) = build_fundamental(research(rows, count=1))
    assert result.metric("cash_margin") is None


@pytest.mark.parametrize("comp_type", ["2", "3", "4", "7", "unknown"])
def test_financial_industries_never_use_nonfinancial_model(comp_type):
    """场景：银行、保险、证券、多元金融或未知类型；预期：不发布非金融Q0。"""
    (result,) = build_fundamental(research(reports(fields={"comp_type": comp_type}), count=1))
    assert result.metric("Q0") is None
    assert result.metrics[0].status == ("unknown" if comp_type == "unknown" else "not_applicable")


def test_report_type_requires_explicit_certification():
    """场景：三表类型一致但配置未认证；预期：原始证据可读，财务比率和Q0为空。"""
    (result,) = build_fundamental(research(reports(), count=1, certified=()))
    assert result.metric("cash_margin") is None
    assert "report_type_uncertified" in result.envelope.gate_codes


@pytest.mark.parametrize(
    "kind", ["missing", "foreign", "dimension", "zero", "negative", "overflow"]
)
def test_financial_money_and_denominator_gates(kind):
    """场景：收入未认证、非人民币、量纲不符、非正或极小；预期：比率未知，禁止单位猜测。"""
    changed = statement(
        "income_vip",
        fields={
            "revenue": 0.0
            if kind == "zero"
            else -1.0
            if kind == "negative"
            else 1e-320
            if kind == "overflow"
            else 100.0
        },
        currency="HKD" if kind == "foreign" else "CNY",
        dimension="ratio" if kind == "dimension" else "money",
    )
    if kind == "missing":
        changed = replace(changed, numeric={})
    rows = tuple(item for item in reports() if item.source_api != "income_vip") + (changed,)
    (result,) = build_fundamental(research(rows, count=1))
    assert result.metric("cash_margin") is None


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"total_hldr_eqy_inc_min_int": None, "total_hldr_eqy_exc_min_int": 60.0}, None),
        ({"total_hldr_eqy_inc_min_int": 59.0}, 0.01),
        ({"total_assets": 0.0}, None),
        ({"total_assets": 1e308, "total_liab": -1e308}, None),
    ],
)
def test_balance_identity_uses_total_equity_and_preserves_errors(fields, expected):
    """场景：少数权益缺失、平衡误差或无效资产；预期：只用总权益，超过0.001触发复核。"""
    (result,) = build_fundamental(research(reports(fields=fields), count=1))
    assert result.metric("balance_error") == expected
    assert "balance_identity_review" in result.envelope.gate_codes


def test_negative_cash_flow_is_not_clipped():
    """场景：经营现金流为负；预期：现金利润率保留负值，不裁成零。"""
    (result,) = build_fundamental(research(reports(fields={"n_cashflow_act": -20.0}), count=1))
    assert result.metric("cash_margin") == -0.2


@pytest.mark.parametrize("opinion", ["保留意见", None])
def test_audit_opinion_is_independent_of_quality_score(opinion):
    """场景：非标审计或意见未知；预期：Q0仍可描述，但审计门禁无法通过。"""
    rows = reports(fields={"audit_result": opinion}) + reports(1)
    result = build_fundamental(research(rows))[0]
    assert result.metric("Q0") == 50
    assert not next(
        check for check in result.envelope.quality_gates if check.gate_id == "audit_opinion"
    ).passed
    assert not result.envelope.decision_allowed


def test_future_versions_and_audit_period_are_not_backfilled():
    """场景：未来公开修订、未来报告期及未来审计；预期：当前档案不使用未来值。"""
    future = statement(
        "income_vip", record="future", fields={"revenue": 1000.0}, public="2026-10-03T00:00:00Z"
    )
    future_period = statement(
        "income_vip", record="future-period", fields={"end_date": "20261231"}, report="2026-12-31"
    )
    future_audit = statement(
        "fina_audit",
        record="future-audit",
        fields={"audit_result": "否定意见"},
        report="2026-12-31",
    )
    (result,) = build_fundamental(
        research(reports() + (future, future_period, future_audit), count=1)
    )
    assert result.metric("cash_margin") == 0.2
    assert result.details["audit_flags"] == []
    assert all(ref.source_record_id != "future" for ref in result.envelope.evidence_refs)


def test_source_reasons_remain_visible_and_duplicate_record_is_idempotent():
    """场景：同一来源重复输入且扫描报告容量缺口；预期：数值不翻倍，缺口仍阻止完整状态。"""
    inputs = replace(
        research(reports() + (reports()[0],), count=1),
        source_reasons={"income_vip": ("source_capacity_exceeded",)},
    )
    (result,) = build_fundamental(inputs)
    assert result.metric("cash_margin") == 0.2
    assert result.details["source_reason_codes"] == ["source_capacity_exceeded"]


def test_negative_equity_and_earnings_have_evidence_despite_positive_valuation():
    """场景：PB和PE均为正但认证总权益、净利润为负；预期：输出两个问题研究标记和反证。"""
    rows = reports(
        fields={
            "total_assets": 100.0,
            "total_liab": 110.0,
            "total_hldr_eqy_inc_min_int": -10.0,
            "n_income": -5.0,
        }
    )
    (result,) = build_fundamental(research(rows + (statement("daily_basic"),), count=1))
    assert result.details["valuation_basis"]["pb"] == 2
    assert result.details["problem_research_flags"] == ["negative_equity", "negative_earnings"]
    for field in ("total_hldr_eqy_inc_min_int", "n_income"):
        used = [ref for ref in result.envelope.evidence_refs if ref.fields == (field,)]
        assert used
        assert all(ref in result.envelope.contrary_evidence for ref in used)


@pytest.mark.parametrize(
    "kind", ["uncertified", "foreign", "positive", "zero", "conflicting", "ambiguous"]
)
def test_problem_research_flags_require_certified_unique_negative_values(kind):
    """场景：字段未认证、外币、非负或报表冲突；预期：不猜测负净资产与亏损标记。"""
    value = 5.0 if kind == "positive" else 0.0 if kind == "zero" else -5.0
    rows = reports(fields={"total_hldr_eqy_inc_min_int": value, "n_income": value})
    if kind == "uncertified":
        rows = tuple(
            replace(
                row,
                numeric={
                    field: fact
                    for field, fact in row.numeric.items()
                    if field not in {"n_income", "total_hldr_eqy_inc_min_int"}
                },
            )
            for row in rows
        )
    if kind == "foreign":
        rows = tuple(
            statement(
                row.source_api,
                fields={"n_income": value, "total_hldr_eqy_inc_min_int": value},
                currency="HKD",
            )
            for row in rows
        )
    if kind == "conflicting":
        rows += (statement("income", fields={"n_income": 10.0}),)
    if kind == "ambiguous":
        rows += (statement("income_vip", record="other-report", fields={"report_type": "2"}),)
    (result,) = build_fundamental(research(rows, count=1))
    assert result.details["problem_research_flags"] == []
