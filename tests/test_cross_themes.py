"""P02全接口角色、成员关系、同日确认和未知边界的离线场景。"""

from dataclasses import replace

import pytest
from test_research_governance import NOW, A, B, inputs, relation, row

from tushare_entitlement_hub.cross_asset_research.themes import (
    API_FIELDS,
    CATALOG_FIELDS,
    DAILY_FIELDS,
    FLOW_APIS,
    MEMBER_FIELDS,
    _breadth,
    _daily,
    _flow,
    _hot,
    _member_quote,
    _member_snapshot,
    _members,
    _signed_return,
    _theme_state,
    build_theme_opportunities,
)
from tushare_entitlement_hub.product_foundation.entities import EntityAlias


def theme_rows():
    """场景：22个权威来源均有独立证据；预期：每条只给其定义字段，数值显式认证。"""
    result = []
    for api, fields in API_FIELDS.items():
        raw = dict.fromkeys(fields, "示例") | {
            "ts_code": "I",
            "con_code": "S",
            "trade_date": "2026-10-03",
        }
        values = {}
        if api in DAILY_FIELDS:
            values = {
                field: (0.02 if unit == "fraction" else 10, unit)
                for field, unit in DAILY_FIELDS[api].items()
            }
        if api in FLOW_APIS:
            values = {
                "net_amount": (100, "元"),
                "company_num": (10, "count"),
                "net_amount_rate": (0.1, "fraction"),
            }
        if api in {"dc_hot", "ths_hot"}:
            values = {"rank": (2, "count"), "hot": (100, "count")}
        if api == "limit_cpt_list":
            values = {
                "cons_nums": (10, "count"),
                "up_nums": (3, "count"),
                "pct_chg": (0.1, "fraction"),
                "rank": (1, "count"),
            }
        if api == "ths_member":
            values = {"weight": (1, "fraction")}
        if api == "kpl_concept_cons":
            values = {"hot_num": (5, "count")}
        result.append(row(api, api, raw=raw, values=values))
    return result


def test_every_theme_api_has_domain_processing_and_own_evidence():
    """场景：完整22接口快照；预期：真实资金确认、板块广度及每个接口自身证据均存在。"""
    result = build_theme_opportunities(inputs(*theme_rows()))[0]
    assert len(API_FIELDS) == 22
    assert {ref.source_api for ref in result.envelope.evidence_refs} == set(API_FIELDS)
    assert result.metric("price_flow_confirmation") == 1
    assert result.metric("limit_cpt_list_limit_up_breadth") == 0.3
    assert result.metric("moneyflow_ind_ths_per_company_flow") == 10
    assert result.details["theme_state"] == "price_and_flow_positive"
    assert not result.envelope.decision_allowed
    assert (
        result.details["sources"]["kpl_list"]["interpretation"]
        == "provider_limit_reason_not_disclosure_fact"
    )


def test_member_contribution_requires_explicit_relation_and_complete_prices():
    """场景：两名成员、两条有效关系和同日涨跌；预期：广度0.5、等权收益0.05且价格证据可追溯。"""
    members = tuple(
        row("dc_member", code, raw={"con_code": code}, effective_from="2026-01-01T00:00:00Z")
        for code in ("S1", "S2")
    )
    prices = {
        A: (row("daily", "p1", raw={"ts_code": "S1"}, values={"pct_chg": (0.2, "fraction")}),),
        B: (
            row(
                "daily",
                "p2",
                entity=B,
                raw={"ts_code": "S2"},
                values={"pct_chg": (-0.1, "fraction")},
            ),
        ),
    }
    relations = tuple(
        relation(from_entity_id=A, to_entity_id=entity, relation_type="theme_member")
        for entity in (A, B)
    )
    sample = inputs(*members, relations=relations)
    metrics, details = _members(sample, A, "dc_member", members, prices)
    values = {item.name: item.value for item in metrics}
    assert values["dc_member_effective_members"] == 2
    assert values["dc_member_member_price_coverage"] == 1
    assert values["dc_member_member_up_fraction"] == 0.5
    assert values["dc_member_equal_weight_return"] == pytest.approx(0.05)
    assert details["membership_basis"] == "certified_effective_interval"
    assert any(ref.source_api == "daily" for ref in metrics[-1].evidence_refs)
    missing, _ = _members(sample, A, "dc_member", members, {A: prices[A]})
    assert missing[-1].value is None and missing[-3].value == 0.5
    stale_prices = {
        A: (
            replace(
                prices[A][0],
                availability=replace(prices[A][0].availability, data_as_of="2026-10-02"),
            ),
        )
    }
    stale, _ = _members(sample, A, "dc_member", members, stale_prices)
    assert stale[-1].value is None and stale[-3].value == 0


@pytest.mark.parametrize("api", tuple(MEMBER_FIELDS))
def test_member_intervals_and_missing_snapshots_never_become_historical_pit(api):
    """场景：成员只含文本纳入日期或数据日未知；预期：可观察成员数但有效期及历史用途仍未知。"""
    member = row(api, raw={MEMBER_FIELDS[api]: "S", "in_date": "20200101", "out_date": ""})
    metrics, details = _members(inputs(member), A, api, (member,), {})
    assert metrics[0].value == 1 and metrics[1].value is None
    assert details["membership_basis"] == "observed_snapshot_only"
    unknown = replace(member, availability=replace(member.availability, data_as_of=None))
    observed_metrics, observed_details = _members(inputs(unknown), A, api, (unknown,), {})
    assert observed_metrics[0].value == 1
    assert all(item.value is None for item in observed_metrics[1:])
    assert observed_details["snapshot_date"] is None
    assert observed_details["snapshot_observed_at"] == NOW


def test_expired_members_and_unmatched_prices_do_not_become_current_contributions():
    """场景：成员有效期已结束且价格错日；预期：有效成员数0、收益贡献未知。"""
    member = row(
        "ths_member",
        raw={"con_code": "S"},
        values={"weight": (1, "fraction")},
        effective_from="2026-01-01T00:00:00Z",
        effective_to="2026-10-01T00:00:00Z",
    )
    price = row(
        "daily", day="2026-10-02", raw={"ts_code": "S"}, values={"pct_chg": (0.1, "fraction")}
    )
    rel = relation(from_entity_id=A, to_entity_id=A, relation_type="theme_member")
    metrics, _ = _members(
        inputs(member, relations=(rel,)), A, "ths_member", (member,), {A: (price,)}
    )
    assert metrics[1].value == 0
    assert metrics[-2].value is None
    assert metrics[-1].value == 1


@pytest.mark.parametrize(
    "up,total,expected", [(3, 10, 0.3), (3, 2, None), (0, 0, None), (-1, 10, None)]
)
def test_breadth_denominator_is_validated(up, total, expected):
    """场景：正常、倒置、零分母和负数计数；预期：只返回有效来源内广度。"""
    observed = row(
        "limit_cpt_list", values={"up_nums": (up, "count"), "cons_nums": (total, "count")}
    )
    assert _breadth("breadth", observed, "up_nums", "cons_nums").value == expected
    assert _breadth("missing", None, "up_nums", "cons_nums").value is None


def test_hot_rank_conflicts_and_source_grouping_are_explicit():
    """场景：来源排名两类、同时刻冲突及零排名；预期：独立分组，不用跨源排名互比。"""
    good = row(
        "dc_hot",
        "good",
        raw={"data_type": "concept", "rank_time": "15:00"},
        values={"rank": (2, "count")},
    )
    unknown = row("dc_hot", "unknown", values={"rank": (0, "count")})
    metrics, details = _hot("dc_hot", (good, unknown))
    assert {item.value for item in metrics} == {0.5, None}
    assert details["rank_group_count"] == 2
    conflict = replace(
        good,
        raw=good.raw | {"source_record_id": "conflict"},
        availability=replace(good.availability, source_record_id="conflict"),
    )
    assert _hot("dc_hot", (good, conflict))[0][0].value is None
    assert _hot("ths_hot", ())[0][-1].value is None


@pytest.mark.parametrize(
    "amount,direction", [(-10, "outflow"), (0, "flat"), (10, "inflow"), (None, "unknown")]
)
def test_flow_sign_and_unknown_company_denominator(amount, direction):
    """场景：正负零及缺资金数值；预期：方向保留，零公司分母不产生强度。"""
    observed = row(
        "moneyflow_cnt_ths", values={"net_amount": (amount, "元"), "company_num": (0, "count")}
    )
    metrics, details = _flow("moneyflow_cnt_ths", (observed,))
    assert details["direction"] == direction
    assert metrics[-1].value is None
    assert _flow("moneyflow_ind_dc", ())[1]["direction"] == "unknown"


def test_theme_confirmation_rejects_stale_flow_and_never_requires_hot_rank():
    """场景：资金错日、同日净流出及无输入；预期：错日未知、同日负向0、空输入无结果。"""
    price = row("ths_daily", values={"pct_change": (0.1, "fraction")})
    stale = row("moneyflow_cnt_ths", day="2026-10-02", values={"net_amount": (10, "元")})
    assert _theme_state({"ths_daily": (price,), "moneyflow_cnt_ths": (stale,)})[0].value is None
    negative = row("moneyflow_cnt_ths", values={"net_amount": (-10, "元")})
    assert _theme_state({"ths_daily": (price,), "moneyflow_cnt_ths": (negative,)})[0].value == 0
    assert build_theme_opportunities(inputs()) == ()
    assert _daily("tdx_daily", ())[0][-1].value is None


@pytest.mark.parametrize("api", tuple(CATALOG_FIELDS))
def test_ambiguous_catalog_has_unknown_status(api):
    """场景：同时间存在两份目录版本；预期：目录一致性未知，来源可追溯。"""
    result = build_theme_opportunities(
        inputs(row(api, "x", raw={"name": "A"}), row(api, "y", raw={"name": "B"}))
    )[0]
    assert result.metric(api + "_catalog_consistent") is None


def test_blocked_sources_and_large_member_counts_preserve_scope():
    """场景：拒绝的题材事件和100名成员；预期：拒绝行不算事件，证据抽样不改变100的分母。"""
    event = row("kpl_list", raw={"theme": "题材"})
    blocked = build_theme_opportunities(
        inputs(event, reasons={"kpl_list": ("capacity_exceeded",)})
    )[0]
    assert blocked.metric("kpl_list_observed_events") is None
    members = tuple(
        row("dc_member", str(index), raw={"con_code": str(index)}) for index in range(100)
    )
    result = build_theme_opportunities(inputs(*members))[0]
    assert result.metric("dc_member_observed_members") == 100
    assert result.details["sources"]["dc_member"]["row_count"] == 100
    assert len(result.to_json().encode()) < 8 * 1024 * 1024


def test_member_contribution_sampling_keeps_all_ten_member_denominators():
    """场景：10名成员均有显式关系和同日价格；预期：抽样只限证据，广度分母仍为10。"""
    members = tuple(
        row("dc_member", str(index), raw={"con_code": str(index)}) for index in range(10)
    )
    identities = tuple(f"cn:security:M{index}:life" for index in range(10))
    prices = {
        identity: (
            row(
                "daily",
                str(index),
                entity=identity,
                raw={"ts_code": str(index)},
                values={"pct_chg": (0.1 if index < 9 else -0.1, "fraction")},
            ),
        )
        for index, identity in enumerate(identities)
    }
    relations = tuple(
        relation(from_entity_id=A, to_entity_id=identity, relation_type="theme_member")
        for identity in identities
    )
    metrics, details = _members(
        inputs(*members, relations=relations), A, "dc_member", members, prices
    )
    assert metrics[-2].value == 0.9
    assert metrics[-1].value == pytest.approx(0.08)
    assert details["contribution_member_count"] == 10
    assert (
        sum(
            ref.source_api == "daily" and ref.fields == ("pct_chg",)
            for ref in metrics[-1].evidence_refs
        )
        == 8
    )


def test_member_code_reused_by_two_explicit_entities_is_ambiguous():
    """场景：两个显式证券实体复用同一成员代码；预期：不按遍历顺序选证券，贡献保持未知。"""
    member = row("dc_member", raw={"con_code": "S"})
    relations = tuple(
        relation(from_entity_id=A, to_entity_id=identity, relation_type="theme_member")
        for identity in (A, B)
    )
    prices = {
        identity: (
            row(
                "daily",
                identity,
                entity=identity,
                raw={"ts_code": "S"},
                values={"pct_chg": (0.1, "fraction")},
            ),
        )
        for identity in (A, B)
    }
    metrics, details = _members(
        inputs(member, relations=relations), A, "dc_member", (member,), prices
    )
    assert metrics[-1].value is None
    assert details["ambiguous_member_code_count"] == 1


def test_member_source_and_stock_alias_form_an_explicit_price_evidence_chain():
    """场景：成员来源、股票时点别名及同日行情完整；预期：无需人工重复填关系也能计算成员贡献。"""
    member = row("dc_member", raw={"con_code": "S"})
    price = row("daily", "price", raw={"ts_code": "S"}, values={"pct_chg": (0.1, "fraction")})
    alias = EntityAlias(
        namespace="tushare_stock",
        alias="S",
        entity_id=A,
        effective_from="2000-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="stock_basic",
        version="test/v1",
        confidence=100,
        evidence_refs=row("stock_basic", raw={"ts_code": "S"}).evidence("ts_code"),
    )
    sample = inputs(member, price)
    sample = replace(
        sample,
        config=replace(
            sample.config, foundation=replace(sample.config.foundation, aliases=(alias,))
        ),
    )
    result = build_theme_opportunities(sample)[0]
    assert result.metric("dc_member_equal_weight_return") == 0.1
    assert {"stock_basic", "dc_member", "daily"} <= {
        ref.source_api for ref in result.envelope.evidence_refs
    }
    assert _member_quote(sample, member, "missing", {}) is None
    assert (
        _member_quote(
            sample,
            replace(member, availability=replace(member.availability, data_as_of=None)),
            "con_code",
            {},
        )
        is None
    )
    assert _member_quote(sample, member, "con_code", {}) is None


@pytest.mark.parametrize("api", ["ci_index_member", "ths_member"])
def test_real_contract_undated_members_have_observation_counts_only(tmp_path, api):
    """场景：真实CI及同花顺契约只含成员关系、没有业务日期；预期：当前观察计数2，同日覆盖未知。"""
    from test_cross_asset_bindings import write_all_reference
    from test_research_binding_fixture import AS_OF

    from tushare_entitlement_hub.cross_asset_research.adapter import load_inputs
    from tushare_entitlement_hub.cross_asset_research.bindings import build_config

    if api == "ci_index_member":
        datasets = {
            api: [
                {
                    "l1_code": "L1",
                    "l2_code": "L2",
                    "l3_code": "CI1",
                    "l3_name": "合成行业",
                    "ts_code": code,
                    "in_date": "20200101",
                }
                for code in ("000001.SZ", "000002.SZ")
            ],
            "ci_daily": [{"ts_code": "CI1", "trade_date": "20261002", "close": 10}],
        }
    else:
        datasets = {
            "ths_index": [{"ts_code": "TH1", "list_date": "20000101", "name": "合成主题"}],
            api: [
                {"ts_code": "TH1", "con_code": code, "con_name": "合成股票"}
                for code in ("000001.SZ", "000002.SZ")
            ],
        }
    reference = write_all_reference(tmp_path, datasets)
    config = build_config(reference, AS_OF).config
    observed, _ = load_inputs(reference, AS_OF, "undated-member-observation", config)
    members = observed.rows(api)
    assert len(members) == 2
    assert all(member.availability.data_as_of is None for member in members)
    result = next(
        item
        for item in build_theme_opportunities(observed)
        if item.envelope.entity_id == members[0].entity_id
    )
    details = result.details["sources"][api]
    assert result.metric(api + "_observed_members") == 2
    assert result.metric(api + "_member_price_coverage") is None
    assert result.metric(api + "_member_up_fraction") is None
    assert result.metric(api + "_equal_weight_return") is None
    assert details["snapshot_date"] is None
    assert details["snapshot_observed_at"] == AS_OF
    assert details["observed_snapshot_record_count"] == 2
    assert details["observed_member_key_coverage"] == 1
    assert details["membership_basis"] == "observed_snapshot_only"
    assert "membership_business_date_unknown" in result.envelope.gate_codes
    assert result.details["historical_pit_certified"] is False
    if api == "ci_index_member":
        assert result.metric(api + "_effective_members") == 2
        assert details["effective_interval_certified"] is True
    else:
        assert result.metric(api + "_effective_members") is None


def test_undated_members_use_latest_observation_group_without_merging_old_records():
    """场景：旧观察组一条、新观察组两条均无业务日期；预期：当前计数2，保留来源观察时刻而非日期。"""
    old = row("ci_index_member", "old", raw={"ts_code": "OLD"})
    old = replace(
        old,
        availability=replace(old.availability, observed_at="2026-10-02T10:00:00Z", data_as_of=None),
    )
    recent = tuple(
        replace(item, availability=replace(item.availability, data_as_of=None))
        for item in (
            row("ci_index_member", "new1", raw={"ts_code": "S1"}),
            row("ci_index_member", "new2", raw={"ts_code": "S2"}),
        )
    )
    snapshot, observation_only, observed_at = _member_snapshot((old, *recent))
    assert snapshot == recent
    assert observation_only and observed_at == NOW
    metrics, details = _members(inputs(old, *recent), A, "ci_index_member", (old, *recent), {})
    assert metrics[0].value == 2
    assert details["snapshot_scope"] == "latest_observed_at_records"
    assert details["snapshot_date"] is None
    assert _member_snapshot(()) == ((), False, None)


def test_undated_member_keys_and_date_less_prices_do_not_create_false_coverage():
    """场景：无业务日成员缺键，或成员与行情日期同时为None；预期：记录键覆盖可见，价格覆盖保持未知。"""
    member = row("ci_index_member", "member", raw={"ts_code": "S"})
    member = replace(member, availability=replace(member.availability, data_as_of=None))
    missing = row("ci_index_member", "missing")
    missing = replace(missing, availability=replace(missing.availability, data_as_of=None))
    metrics, details = _members(
        inputs(member, missing), A, "ci_index_member", (member, missing), {}
    )
    assert metrics[0].value is None
    assert details["observed_snapshot_record_count"] == 2
    assert details["observed_member_key_coverage"] == 0.5
    price = row("daily", "price", raw={"ts_code": "S"}, values={"pct_chg": (0.1, "fraction")})
    price = replace(price, availability=replace(price.availability, data_as_of=None))
    rel = relation(from_entity_id=A, to_entity_id=A, relation_type="theme_member")
    metrics, _ = _members(
        inputs(member, relations=(rel,)), A, "ci_index_member", (member,), {A: (price,)}
    )
    assert metrics[0].value == 1
    assert all(item.value is None for item in metrics[1:])


@pytest.mark.parametrize("unit", ["ratio", "fraction"])
def test_certified_signed_returns_accept_normalized_ratio_storage(unit):
    """场景：规范化负收益用ratio或fraction存储；预期：都以有符号ratio输出，并参与主题与成员计算。"""
    observed = row("ths_daily", values={"pct_change": (-0.02, unit)})
    result = build_theme_opportunities(inputs(observed))[0]
    value = next(item for item in result.metrics if item.name == "ths_daily_pct_change")
    assert value.value == -0.02 and value.unit == "ratio"
    flow = row("moneyflow_cnt_ths", values={"net_amount": (100, "元")})
    assert _theme_state({"ths_daily": (observed,), "moneyflow_cnt_ths": (flow,)})[0].value == 0
    member = row("dc_member", raw={"con_code": "S"})
    price = row("daily", "price", raw={"ts_code": "S"}, values={"pct_chg": (-0.02, unit)})
    rel = relation(from_entity_id=A, to_entity_id=A, relation_type="theme_member")
    metrics, _ = _members(
        inputs(member, relations=(rel,)), A, "dc_member", (member,), {A: (price,)}
    )
    assert metrics[-1].value == -0.02
    limit = row("limit_cpt_list", values={"pct_chg": (-0.02, unit)})
    assert build_theme_opportunities(inputs(limit))[0].metric("limit_cpt_list_pct_chg") == -0.02


def test_signed_return_requires_ratio_dimension_and_complete_source_certification():
    """场景：同名数值为金额、未换算百分数或事实来源不匹配；预期：不以任意数值替代收益认证。"""
    money = row("ths_daily", values={"pct_change": (-0.02, "元")})
    percent = row("ths_daily", values={"pct_change": (-2, "%")})
    untyped = row("ths_daily", raw={"pct_change": -0.02})
    original = row("ths_daily", values={"pct_change": (-0.02, "ratio")})
    mismatched = replace(
        original, availability=replace(original.availability, data_as_of="2026-10-02")
    )
    for item in (None, money, percent, untyped, mismatched):
        assert _signed_return(item, "pct_change") is None
