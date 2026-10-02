"""股权研究引擎的确定性合成测试，不访问网络或使用真实证券凭据。"""

from collections import defaultdict
from dataclasses import replace

import pytest

from tushare_entitlement_hub.equity_research.governance import (
    _ambiguous,
    _convertible,
    _day,
    _event,
    _number,
    _placement,
    _regulatory,
    _supply_metric,
    _unlock,
    build_governance,
)
from tushare_entitlement_hub.equity_research.models import (
    ResearchConfig,
    ResearchInputs,
    ResearchRow,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.contracts import Truth
from tushare_entitlement_hub.product_foundation.entities import (
    Entity,
    EntityIdentity,
    EntityKind,
    EntityRelation,
)
from tushare_entitlement_hub.product_foundation.numeric import TypedFact, UnitCertification
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

NOW = "2026-10-03T12:00:00Z"
DAY = "2026-10-03"
A = "cn:security:A:life"
B = "cn:security:B:life"
BOND = "cn:security:BOND:life"
COMPLETE = ("share_float", "stk_seasoned", "cb_basic", "st", "stock_st", "stk_alert")


def row(api, record="one", entity=A, values=None, raw=None, day=DAY, **time_changes):
    """构造显式认证合成行；例：values={'amount':(10,'元')}得到10元合格事实。"""
    availability = AvailabilityFact(
        source_api=api,
        source_record_id=record,
        certification_version="synthetic/v1",
        observed_at=NOW,
        data_as_of=day,
        **time_changes,
    )
    payload = {
        "source_api": api,
        "source_record_id": record,
        "business_group_key": record,
        "contract_hash": "synthetic-contract",
        "trade_date": day,
    } | (raw or {})
    payload.update({name: value[0] for name, value in (values or {}).items()})
    result = ResearchRow(entity_id=entity, source_api=api, raw=payload, availability=availability)
    numeric = {}
    for name, parts in (values or {}).items():
        value, unit = parts[:2]
        currency = parts[2] if len(parts) > 2 else "CNY" if unit == "元" else None
        certificate = UnitCertification(
            version="synthetic/v1",
            source_api=api,
            field=name,
            source_unit=unit,
            target_unit=unit,
            dimension="money" if unit == "元" else "ratio",
            currency=currency,
            multiplier=1,
            evidence_refs=result.evidence(name),
        )
        numeric[name] = TypedFact(
            fact_id=api + "/" + record + "/" + name,
            entity_id=entity,
            field=name,
            value=value,
            unit=unit,
            dimension=certificate.dimension,
            currency=currency,
            availability=availability,
            quality_tier="verified",
            certification=certificate,
            evidence_refs=result.evidence(name),
            reason_codes=("unit_unverified",) if value is None else (),
        )
    return replace(result, numeric=numeric)


def inputs(*rows, entities=(A,), complete=(), relations=(), reasons=None, expected=None):
    """组合显式股票与来源，默认最小同行数为2；例：零行但有股票得到有范围的空观察。"""
    grouped = defaultdict(list)
    for item in rows:
        grouped[item.source_api].append(item)
    stocks = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn",
                kind=EntityKind.SECURITY,
                local_id=identity.split(":")[2],
                lifecycle_id="life",
            ),
            name="合成股票",
        )
        for identity in entities
    )
    return ResearchInputs(
        analysis_as_of=NOW,
        run_id="synthetic-run",
        source_reasons=reasons or {},
        config=ResearchConfig(
            foundation=FoundationBindings(entities=stocks, expected_data_as_of=expected or {}),
            complete_scopes=complete,
            confirmed_relations=relations,
            peer_paths={identity: ("all",) for identity in entities},
            thresholds=FoundationPolicy(version="test/v1", minimum_peer_count=2),
        ),
        rows_by_api={api: tuple(items) for api, items in grouped.items()},
    )


def relation(**changes):
    """创建有可见证据的正股关系；例：默认BOND到A且confirmed=True。"""
    reference = row("cb_basic", raw={"cb_type": "CB"}).evidence("cb_type")
    values = dict(
        from_entity_id=BOND,
        to_entity_id=A,
        relation_type="cb_underlying",
        confirmed=True,
        effective_from="2020-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="synthetic",
        version="v1",
        evidence_refs=reference,
        confidence=100,
    )
    return EntityRelation(**(values | changes))


def basic(entity=A, **changes):
    """生成流通1000股、总股本2000股的认证底表；例：total_share=0构造非法分母场景。"""
    values = {"float_share": (1000, "股"), "total_share": (2000, "股")}
    values.update({name: (value, "股") for name, value in changes.items()})
    return row("daily_basic", entity=entity, values=values)


def cb(record="bond", kind="CB", **changes):
    """生成转股期内1000元余额与10元转股价；例：默认潜在新增100股。"""
    return row(
        "cb_basic",
        record,
        BOND,
        values={"remain_size": (1000, "元"), "conv_price": (10, "元")},
        raw={"cb_type": kind, "conv_start_date": "20260101", "conv_end_date": "20271231"} | changes,
    )


def test_empty_observations_preserve_unknown_and_explicit_absence():
    """场景：无股票、缺采或认证无事件；预期：空集合、未知和不适用三者分别表达。"""
    assert build_governance(inputs(entities=())) == ()
    result = build_governance(inputs())[0]
    assert result.metric("unlock_30d") is None
    assert result.envelope.state.value == "incomplete"
    complete = build_governance(inputs(complete=COMPLETE))[0]
    metrics = {item.name: item for item in complete.metrics}
    assert metrics["unlock_30d"].status == "not_applicable"
    assert metrics["cb_potential_dilution"].value is None
    assert metrics["financing_dilution"].status == "not_applicable"
    failed = build_governance(
        inputs(complete=COMPLETE, reasons={"share_float": ("missing_table",)})
    )[0]
    assert next(item for item in failed.metrics if item.name == "unlock_30d").status == "unknown"


def test_supply_scores_are_explained_and_hard_risk_is_independent():
    """场景：两只股票全部供给指标可比且A已ST；预期：分数可分解，A仍被硬门禁阻断。"""
    rows = []
    relations = []
    for entity, shares in ((A, 100), (B, 200)):
        rows.extend(
            [
                basic(entity),
                row(
                    "share_float",
                    entity=entity,
                    values={"float_share": (shares, "股")},
                    raw={"ann_date": "20261001", "float_date": "20261010"},
                ),
                row(
                    "stk_seasoned",
                    "plan",
                    entity,
                    {"fo_vol_high": (shares, "股")},
                    {"cur_stage": "董事会预案", "first_ann_date": "20260101", "fo_type": "非公开"},
                ),
                row(
                    "stk_seasoned",
                    "actual",
                    entity,
                    {"fo_vol_act": (shares / 2, "股")},
                    {"cur_stage": "实施", "first_ann_date": "20260102", "fo_type": "非公开"},
                ),
                row(
                    "pledge_stat",
                    entity=entity,
                    values={"pledge_ratio": (shares / 1000, "fraction")},
                ),
            ]
        )
        bond_id = "cn:security:BOND" + entity.split(":")[2] + ":life"
        rows.append(replace(cb(), entity_id=bond_id))
        relations.append(relation(from_entity_id=bond_id, to_entity_id=entity))
    rows.append(row("st", raw={"imp_date": "20261001", "st_type": "ST"}))
    results = build_governance(
        inputs(*rows, entities=(A, B), relations=tuple(relations), complete=COMPLETE)
    )
    assert results[0].metric("unlock_30d") == 0.1
    assert results[0].metric("placement_dilution") == 0.05
    assert results[0].metric("placement_executed_dilution") == 0.025
    assert results[0].metric("cb_potential_dilution") == 0.05
    assert results[0].metric("R_supply") == 0
    assert results[1].metric("R_supply") == 100
    assert results[0].envelope.state.value == "blocked"
    assert sum(item.contribution for item in results[1].envelope.contributions) == 100
    assert all(item.peer_level for item in results[1].envelope.contributions)


@pytest.mark.parametrize(
    "date_value,expected",
    [("20261102", 0.1), ("20261103", None), ("20261003", None), ("20260901", None)],
)
def test_unlock_uses_thirty_calendar_days(date_value, expected):
    """场景：解禁位于第30/31天、当日或过去；预期：仅未来30自然日计入。"""
    source = row(
        "share_float",
        values={"float_share": (100, "股")},
        raw={"ann_date": "20261001", "float_date": date_value},
    )
    result = _unlock(inputs(basic(), source, complete=COMPLETE), A, _day(DAY))
    assert result.value == expected


@pytest.mark.parametrize(
    "patch", [{"ann_date": None}, {"float_date": "invalid"}, {"ann_date": "20261004"}]
)
def test_unlock_unknown_dates_block_precision(patch):
    """场景：公告或解禁日期缺失/未来公告；预期：不发布精确窗口总量。"""
    source = row(
        "share_float",
        values={"float_share": (100, "股")},
        raw={"ann_date": "20261001", "float_date": "20261101"} | patch,
    )
    assert "event_date_unverified" in _unlock(inputs(source), A, _day(DAY)).reason_codes
    assert _day(None) is None


def test_ambiguous_batches_and_invalid_denominators_fail_closed():
    """场景：匿名变体、同组冲突和缺少股本；预期：保留未知，不加总候选值。"""
    source = row("share_float", values={"float_share": (100, "股")})
    variant = row(
        "share_float",
        "other",
        values={"float_share": (200, "股")},
        raw={"business_group_key": "one"},
    )
    assert _ambiguous((source, variant))
    assert not _ambiguous((source, row("repurchase")))
    for rows in (
        (source, variant),
        (replace(source, raw=source.raw | {"business_variant_key": "v"}),),
    ):
        result = _supply_metric(
            inputs(basic()), A, "share_float", "unlock_30d", rows, "float_share", "float_share"
        )
        assert "business_variant_unresolved" in result.reason_codes
    for base in (None, basic(float_share=0)):
        context = inputs() if base is None else inputs(base)
        assert (
            _supply_metric(
                context, A, "share_float", "unlock_30d", (source,), "float_share", "float_share"
            ).value
            is None
        )
    assert (
        _supply_metric(
            inputs(basic()),
            A,
            "share_float",
            "unlock_30d",
            (row("share_float", values={"float_share": (-1, "股")}),),
            "float_share",
            "float_share",
        ).value
        is None
    )
    assert (
        _supply_metric(
            inputs(basic()),
            A,
            "share_float",
            "unlock_30d",
            (row("share_float"),),
            "float_share",
            "float_share",
        ).value
        is None
    )


@pytest.mark.parametrize("stage", [None, "意向", "未知阶段"])
def test_placement_intent_is_not_confirmed_issuance(stage):
    """场景：意向或未识别发行阶段；预期：计划与实际稀释均未知。"""
    context = inputs(basic(), row("stk_seasoned", raw={"cur_stage": stage}))
    assert _placement(context, A).value is None
    assert _placement(context, A, executed=True).value is None


def test_placement_termination_and_approved_plan_are_separate():
    """场景：终止方案及监管审批方案；预期：终止不参与，审批股数进入计划列。"""
    context = inputs(
        basic(),
        row(
            "stk_seasoned",
            "ended",
            raw={"cur_stage": "终止", "first_ann_date": "20260101", "fo_type": "非公开"},
        ),
        row(
            "stk_seasoned",
            "approved",
            values={"fo_vol_high": (50, "股")},
            raw={"cur_stage": "监管审批", "first_ann_date": "20260102", "fo_type": "非公开"},
        ),
        complete=COMPLETE,
    )
    assert _placement(context, A).value == 0.025
    assert _placement(context, A, executed=True).status == "not_applicable"


@pytest.mark.parametrize(
    "patch",
    [
        {"cb_type": "UNKNOWN"},
        {"conv_start_date": None},
        {"conv_end_date": None},
        {"conv_start_date": "20270101"},
        {"conv_end_date": "20260101"},
    ],
)
def test_cb_identity_and_conversion_period_are_required(patch):
    """场景：CB身份或转股期间不确定；预期：不输出潜在稀释率。"""
    result = _convertible(inputs(basic(), cb(**patch), relations=(relation(),)), A, _day(DAY))
    assert "cb_terms_unverified" in result.reason_codes


@pytest.mark.parametrize(
    "values",
    [
        {"remain_size": (None, "元"), "conv_price": (10, "元")},
        {"remain_size": (-1, "元"), "conv_price": (10, "元")},
        {"remain_size": (10, "元"), "conv_price": (0, "元")},
        {"remain_size": (10, "元"), "conv_price": (None, "元")},
        {"remain_size": (10, "元", "USD"), "conv_price": (10, "元")},
    ],
)
def test_cb_requires_positive_price_and_currency(values):
    """场景：余额未知/负值、零转股价或币种不一致；预期：数值认证失败。"""
    source = row(
        "cb_basic",
        entity=BOND,
        values=values,
        raw={"cb_type": "CB", "conv_start_date": "20260101", "conv_end_date": "20271231"},
    )
    assert (
        _convertible(inputs(basic(), source, relations=(relation(),)), A, _day(DAY)).value is None
    )


def test_cb_relations_exchangeable_bonds_and_denominator():
    """场景：关系缺失、EB、混合债种或缺股本；预期：EB不稀释发行人，未知不补0。"""
    assert _convertible(inputs(cb(), complete=COMPLETE), A, _day(DAY)).status == "unknown"
    assert (
        _convertible(inputs(cb(), relations=(relation(confirmed=False),)), A, _day(DAY)).value
        is None
    )
    eb = cb(kind="EB")
    assert (
        _convertible(inputs(basic(), eb, relations=(relation(),)), A, _day(DAY)).status
        == "not_applicable"
    )
    context = inputs(basic(), cb(), cb("exchange", kind="EB"), relations=(relation(),))
    assert _convertible(context, A, _day(DAY)).value == 0.05
    for base in (None, basic(total_share=0)):
        context = inputs(cb(), relations=(relation(),))
        if base:
            context = inputs(cb(), base, relations=(relation(),))
        assert "share_denominator_unverified" in _convertible(context, A, _day(DAY)).reason_codes
    variant = replace(cb(), raw=cb().raw | {"business_variant_key": "candidate"})
    assert (
        "business_variant_unresolved"
        in _convertible(
            inputs(basic(), variant, relations=(relation(),)), A, _day(DAY)
        ).reason_codes
    )
    assert _number(None, "x", "元") is None


@pytest.mark.parametrize(
    "raw,stage,planned,executed",
    [
        ({"proc": "董事会预案"}, "planned", 100, None),
        ({"proc": "实施"}, "implemented", None, 100),
        ({"div_proc": "预案"}, "planned", 100, None),
        ({"float_date": "20261002"}, "effective", None, None),
        ({"float_date": "20261004"}, "planned", 100, None),
        ({}, "unknown", None, None),
    ],
)
def test_event_stages_and_repurchase_snapshots(raw, stage, planned, executed):
    """场景：回购计划、实施及日期事件；预期：阶段分离且快照金额不累计。"""
    source = row("repurchase", values={"amount": (100, "元")}, raw=raw)
    event = _event(source, _day(DAY))
    assert event["event_stage"] == stage
    assert event["planned_amount"] == planned
    assert event["executed_amount"] == executed
    assert event["trading_days_to_event"] is None
    result = build_governance(
        inputs(
            source, row("repurchase", "next", values={"amount": (200, "元")}, raw={"proc": "实施"})
        )
    )[0]
    assert len(result.details["events"]) == 2
    assert result.details["repurchase_snapshot_policy"] == "preserve_separate_no_sum"


@pytest.mark.parametrize(
    "records,expected",
    [
        ([{"imp_date": "20261001", "st_type": "ST"}], Truth.FALSE),
        ([{"imp_date": "20261004", "st_type": "ST"}], Truth.UNKNOWN),
        ([{"imp_date": None, "st_type": "ST"}], Truth.UNKNOWN),
        ([{"imp_date": "20261001", "st_type": "unmapped"}], Truth.UNKNOWN),
        (
            [
                {"imp_date": "20261001", "st_type": "ST"},
                {"imp_date": "20261002", "st_type": "撤销ST"},
            ],
            Truth.TRUE,
        ),
        (
            [
                {"imp_date": "20261001", "st_type": "ST"},
                {"imp_date": "20261001", "st_type": "撤销ST"},
            ],
            Truth.FALSE,
        ),
    ],
)
def test_st_implementation_reversal_and_conflict(records, expected):
    """场景：ST计划、生效、撤销或同日冲突；预期：只确认最新实施状态，冲突独立阻断。"""
    context = inputs(*(row("st", str(index), raw=raw) for index, raw in enumerate(records)))
    result = _regulatory(context, A, _day(DAY))
    assert result[0].result is expected
    if len(records) == 2 and records[0]["imp_date"] == records[1]["imp_date"]:
        assert result[2].result is Truth.FALSE


def test_stock_st_snapshot_and_major_conflict():
    """场景：当日ST列表与撤销公告冲突，或来源报告重大冲突；预期：冲突门禁失败。"""
    context = inputs(
        row("st", raw={"imp_date": "20261002", "st_type": "撤销ST"}),
        row("stock_st", raw={"type": "ST"}),
        expected={"stock_st": DAY},
    )
    assert _regulatory(context, A, _day(DAY))[2].result is Truth.FALSE
    explicit = inputs(row("dividend", raw={"state_conflict": True}))
    assert _regulatory(explicit, A, _day(DAY))[2].result is Truth.FALSE


@pytest.mark.parametrize(
    "start,end,complete,expected",
    [
        (None, None, (), Truth.UNKNOWN),
        ("20261001", None, (), Truth.FALSE),
        ("20261001", "2026-10-04T00:00:00Z", (), Truth.FALSE),
        ("20261001", "2026-10-02T00:00:00Z", COMPLETE, Truth.TRUE),
        ("20261004", None, COMPLETE, Truth.TRUE),
    ],
)
def test_monitoring_is_not_cleared_by_reference_end_date(start, end, complete, expected):
    """场景：未知/有效/已确认结束监控；预期：参考end_date不自动解除监管门禁。"""
    kwargs = {"effective_from": "2026-10-01T00:00:00Z", "effective_to": end} if end else {}
    source = row("stk_alert", raw={"start_date": start, "end_date": "20261001"}, **kwargs)
    assert _regulatory(inputs(source, complete=complete), A, _day(DAY))[1].result is expected


def test_placement_revision_requires_identity_and_strictly_later_availability():
    """场景：相同方案有新版、同公开时点冲突或缺修订证据；预期：只选择严格更晚的明确修订。"""
    identity = {"first_ann_date": "20260101", "fo_type": "非公开", "cur_stage": "董事会预案"}
    original = row(
        "stk_seasoned",
        "old",
        values={"fo_vol_high": (100, "股")},
        raw=identity | {"update_flag": "0"},
        day="2026-10-01",
    )
    original = replace(
        original, availability=replace(original.availability, observed_at="2026-10-01T12:00:00Z")
    )
    revised = row(
        "stk_seasoned",
        "new",
        values={"fo_vol_high": (200, "股")},
        raw=identity | {"update_flag": "1"},
    )
    assert _placement(inputs(basic(), original, revised), A).value == 0.1
    same_time = replace(
        original, availability=replace(original.availability, observed_at=NOW, data_as_of=DAY)
    )
    assert _placement(inputs(basic(), same_time, revised), A).value is None
    for changes in ({"first_ann_date": None}, {"update_flag": "0"}):
        older = replace(original, raw=original.raw | changes)
        newer = replace(revised, raw=revised.raw | changes)
        assert _placement(inputs(basic(), older, newer), A).value is None
    later_business_date = replace(
        same_time, availability=replace(same_time.availability, data_as_of="2026-10-02")
    )
    assert _placement(inputs(basic(), later_business_date, revised), A).value is None
    implemented = row(
        "stk_seasoned",
        "implemented",
        values={"fo_vol_act": (150, "股")},
        raw=identity | {"update_flag": "1", "cur_stage": "实施"},
    )
    context = inputs(basic(), original, implemented, complete=COMPLETE)
    assert _placement(context, A).status == "not_applicable"
    assert _placement(context, A, executed=True).value == 0.075


def test_complete_scope_never_overrides_entity_source_rejection():
    """场景：完整API配置但实体分组因容量或来源错误被清空；预期：全部对应风险保留未知。"""
    context = inputs(complete=COMPLETE)
    context = replace(
        context,
        entity_source_reasons={
            api + "/" + A: ("entity_source_row_limit_exceeded",)
            for api in ("share_float", "cb_basic", "stk_seasoned", "st", "stock_st", "stk_alert")
        },
    )
    result = build_governance(context)[0]
    metrics = {item.name: item for item in result.metrics}
    assert metrics["unlock_30d"].status == "unknown"
    assert metrics["cb_potential_dilution"].status == "unknown"
    assert metrics["placement_dilution"].status == "unknown"
    assert result.envelope.quality_gates[0].result is Truth.UNKNOWN
    assert result.envelope.quality_gates[1].result is Truth.UNKNOWN


def test_currency_certificate_misclassification_still_cannot_enable_conversion():
    """场景：金额单位的证书误登记非货币量纲；预期：美元仍不能当人民币进入计算。"""
    source = row("cb_basic", values={"remain_size": (10, "元", "USD")})
    fact = source.numeric["remain_size"]
    certificate = replace(fact.certification, dimension="ratio")
    fact = replace(fact, dimension="ratio", certification=certificate)
    source = replace(source, numeric={"remain_size": fact})
    assert _number(source, "remain_size", "元") is None


def test_unknown_st_date_is_not_erased_by_known_revocation():
    """场景：已撤销记录外仍有未知实施日的ST；预期：撤销不能替未知来源做结论。"""
    context = inputs(
        row("st", "clear", raw={"st_type": "撤销ST", "imp_date": "20261001"}),
        row("st", "unknown", raw={"st_type": "ST", "imp_date": None}),
    )
    assert _regulatory(context, A, _day(DAY))[0].result is Truth.UNKNOWN


def test_stock_st_missing_current_date_remains_unknown():
    """场景：已采ST列表但没有认证当前交易日；预期：不能把缺日期记录当已生效状态。"""
    context = inputs(row("stock_st", day=None, raw={"type": "ST"}), complete=COMPLETE)
    assert _regulatory(context, A, _day(DAY))[0].result is Truth.UNKNOWN


def test_related_cb_events_remain_linked_to_the_underlying_stock():
    """场景：正股有确认转债关系与强赎公告；预期：正股事件清单保留转债来源证据。"""
    source = row("cb_call", entity=BOND, raw={"call_type": "强赎提示", "ann_date": "20261002"})
    result = build_governance(inputs(basic(), cb(), source, relations=(relation(),)))[0]
    assert any(event["source_api"] == "cb_call" for event in result.details["events"])
    assert any(ref.source_api == "cb_call" for ref in result.envelope.evidence_refs)


def test_cb_referenced_bond_failures_propagate_to_stock_risk():
    """场景：已确认关联债券缺采或被容量拒绝；预期：正股稀释保留未知，不按完整范围标无事件。"""
    config = inputs(basic(), relations=(relation(),), complete=COMPLETE)
    missing = _convertible(config, A, _day(DAY))
    assert missing.status == "unknown" and "cb_reference_missing" in missing.reason_codes
    rejected = replace(
        inputs(basic(), cb(), relations=(relation(),), complete=COMPLETE),
        entity_source_reasons={"cb_basic/" + BOND: ("entity_source_capacity_exceeded",)},
    )
    assert "entity_source_capacity_exceeded" in _convertible(rejected, A, _day(DAY)).reason_codes
    eb = replace(rejected, rows_by_api={"cb_basic": (cb(kind="EB"),)})
    assert _convertible(eb, A, _day(DAY)).status == "unknown"
    second_relation = relation(from_entity_id="cn:security:MISSING:life")
    partial = inputs(basic(), cb(), relations=(relation(), second_relation), complete=COMPLETE)
    assert _convertible(partial, A, _day(DAY)).value is None


def test_same_plan_implemented_and_terminated_is_a_hard_conflict():
    """场景：同一发行计划在同可用时点同时标实施和终止；预期：重大状态冲突硬门禁失败。"""
    identity = {"first_ann_date": "20260101", "fo_type": "非公开"}
    context = inputs(
        row("stk_seasoned", "done", raw=identity | {"cur_stage": "实施"}),
        row("stk_seasoned", "stop", raw=identity | {"cur_stage": "终止"}),
    )
    checks = _regulatory(context, A, _day(DAY))
    assert checks[2].result is Truth.FALSE
    assert any("cur_stage" in ref.fields for ref in checks[2].evidence_refs)


def test_unidentified_plans_are_not_falsely_joined_as_hard_conflicts():
    """场景：两个发行记录缺少计划身份；预期：保留指标未知，不凭阶段文字强行关联成冲突。"""
    context = inputs(
        row("stk_seasoned", "done", raw={"cur_stage": "实施"}),
        row("stk_seasoned", "stop", raw={"cur_stage": "终止"}),
    )
    assert _regulatory(context, A, _day(DAY))[2].result is Truth.TRUE
    assert _placement(context, A).status == "unknown"
