"""债券聚合的边界、未抽样变化、单位冲突和低基数原语义回归。"""

from dataclasses import replace

import pytest
from test_research_bond_market import COUNTS, bond_market_inputs
from test_research_consumption import registration
from test_research_governance import DAY, A, B, relation

from tushare_entitlement_hub.equity_research import consumption
from tushare_entitlement_hub.equity_research.bond_aggregation import _Distribution, aggregate_bonds
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    execute_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import (
    PRODUCT_ALIASES,
    resolve_consumer,
)
from tushare_entitlement_hub.equity_research.context_aggregation import row_fingerprint
from tushare_entitlement_hub.equity_research.context_digest import MultisetDigest
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.store import encode


def record_for(context, api):
    """实际执行指定API的市场消费；例：9行cb_basic返回带完整摘要的消费记录。"""
    return execute_consumption(context, registration(api), MARKET_ENTITY)[0]


def replace_rows(context, api, rows):
    """创建全新输入快照，使实体、代码和日期索引随输入更新；例：替换cb_call后不沿用旧日期缓存。"""
    return replace(context, rows_by_api=context.rows_by_api | {api: tuple(rows)})


@pytest.mark.parametrize("api", tuple(COUNTS))
def test_unsampled_bond_row_changes_full_fingerprint(api):
    """九种来源的未抽样行关键字段变化均改变全量摘要，其他来源及总行数不受影响。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 12))
    first = record_for(context, api)
    sampled = {
        ref["source_record_id"] for ref in first["evidence_refs"] if ref["source_api"] == api
    }
    sources = list(context.rows(api))
    index = next(
        i for i, row in enumerate(sources) if row.availability.source_record_id not in sampled
    )
    sources[index] = replace(sources[index], raw=sources[index].raw | {"ts_code": "CHANGED_CODE"})
    second = record_for(replace_rows(context, api, sources), api)
    assert first["retained_row_count"] == second["retained_row_count"] == 12
    assert (
        first["facts"]["aggregation"]["full_fingerprint"]
        != second["facts"]["aggregation"]["full_fingerprint"]
    )
    assert second["facts"]["investment_signal"] == "not_supported"


@pytest.mark.parametrize("api", tuple(COUNTS))
def test_low_cardinality_and_linked_stock_keep_exact_consumer_facts(monkeypatch, api):
    """八行市场来源及九行已关联股票来源均走原处理器，事实、状态和引用不被聚合替换。"""
    for count, identity in ((8, MARKET_ENTITY), (9, A)):
        context = bond_market_inputs(dict.fromkeys(COUNTS, count))
        assigned = registration(api)
        rows = consumption.consumption_rows(context, assigned, identity)
        legacy = resolve_consumer(assigned.handler)(context, "P06", api, rows, identity)

        def forbidden(*args, **kwargs):
            """低基数或股票范围不允许进入聚合；任意调用均表示范围隔离回归。"""
            pytest.fail("不应聚合该消费范围")

        with monkeypatch.context() as patch:
            patch.setattr(consumption, "aggregate_bonds", forbidden)
            actual, references, _ = execute_consumption(context, assigned, identity)
        assert actual["facts"] == legacy["facts"]
        assert actual["status"] == legacy["status"]
        assert {ref.to_json() for ref in references} == {
            ref.to_json() for ref in legacy["evidence_refs"]
        }


def test_mixed_quote_states_preserve_legacy_status_and_each_sample_stratum():
    """同API同时有已核对、零成交未知和报价冲突时保留三种状态，整体conflict与旧处理器一致。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    rows = list(context.rows("cb_daily"))
    rows[0] = replace(rows[0], raw=rows[0].raw | {"vol": 0})
    rows[1] = replace(
        rows[1],
        numeric=rows[1].numeric | {"close": replace(rows[1].numeric["close"], value=120)},
        raw=rows[1].raw | {"close": 120},
    )
    context = replace_rows(context, "cb_daily", rows)
    actual = record_for(context, "cb_daily")
    legacy = resolve_consumer("disclosure")(context, "P06", "cb_daily", tuple(rows), MARKET_ENTITY)
    summary = actual["facts"]["aggregation"]
    assert actual["status"] == legacy["status"] == "conflict"
    assert summary["status_counts"] == {"conflict": 1, "observed": 7, "unknown": 1}
    assert {sample["status"] for sample in summary["sample_strata"]} == {
        "conflict",
        "observed",
        "unknown",
    }
    assert summary["business_statistics"]["dimensions"]["quote_close"]["values"] == {
        "conflict": 1,
        "match": 7,
        "unknown": 1,
    }
    assert "bond_quote_or_premium_conflict" in actual["reason_codes"]


def test_all_unverified_units_remain_unknown_for_disclosure_but_observable_for_core():
    """缺认证的债券报价保持unknown，core可观察字段仍observed；两者认证计数均为0。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    for api in ("cb_daily", "cb_basic"):
        context = replace_rows(
            context, api, (replace(row, numeric={}) for row in context.rows(api))
        )
    for api, expected in (("cb_daily", "unknown"), ("cb_basic", "observed")):
        record = record_for(context, api)
        assert record["status"] == expected
        if expected == "unknown":
            assert record["reason_codes"]
        assert all(item["certified_rows"] == 0 for item in record["field_quality"].values())
        assert all(
            item["minimum"] is None
            for item in record["facts"]["aggregation"]["business_statistics"][
                "numeric_ranges"
            ].values()
        )


def test_unknown_codes_missing_fields_and_duplicate_rows_are_audited():
    """无实体时明确代码仍分组，缺代码则各自独立；无公开字段保持unknown，重复行不从摘要消失。"""
    context = bond_market_inputs({"cb_call": 9})
    rows = [
        replace(
            row,
            entity_id=None,
            raw={
                key: value
                for key, value in row.raw.items()
                if key not in registration("cb_call").spec.selected_fields
            },
        )
        for row in context.rows("cb_call")
    ]
    rows[0] = replace(rows[0], raw=rows[0].raw | {"ts_code": None})
    context = replace_rows(context, "cb_call", rows + [rows[1]])
    record = record_for(context, "cb_call")
    summary = record["facts"]["aggregation"]
    assert record["status"] == "unknown" and "current_fields_unavailable" in record["reason_codes"]
    assert summary["total_rows"] == 10 and summary["group_count"] == 9
    assert summary["sample_count"] == 0 and record["evidence_refs"] == []
    complete = MultisetDigest()
    for row in context.rows("cb_call"):
        complete.add(row_fingerprint(row, tuple(summary["audited_fields"])))
    assert summary["rows_fingerprint"] == complete.hexdigest()


def test_rejected_retained_rows_still_enter_full_audit():
    """九条未来观察行均被执行门禁排除，仍保留total=9/rejected=9及完整指纹，不能静默消失。"""
    context = bond_market_inputs({"cb_call": 9})
    future = tuple(
        replace(row, availability=replace(row.availability, observed_at="2026-10-04T12:00:00Z"))
        for row in context.rows("cb_call")
    )
    context = replace_rows(context, "cb_call", future)
    record = record_for(context, "cb_call")
    summary = record["facts"]["aggregation"]
    assert record["retained_row_count"] == summary["usable_rows"] == 0
    assert summary["total_rows"] == summary["rejected_rows"] == 9
    assert record["status"] == "unknown" and "consumption_rows_rejected" in record["reason_codes"]
    changed = list(future)
    changed[0] = replace(changed[0], raw=changed[0].raw | {"is_call": "Y"})
    assert (
        record_for(replace_rows(context, "cb_call", changed), "cb_call")["facts"]["aggregation"][
            "full_fingerprint"
        ]
        != summary["full_fingerprint"]
    )


def test_unit_conflicts_and_signed_zero_have_order_independent_statistics():
    """混合单位清空范围并披露冲突，零值范围不受正负零输入顺序影响；全部认证行仍计数。"""
    context = bond_market_inputs({"cb_share": 9})
    rows = list(context.rows("cb_share"))
    for index, value in enumerate((0.0, -0.0)):
        rows[index] = replace(
            rows[index],
            numeric=rows[index].numeric
            | {"convert_vol": replace(rows[index].numeric["convert_vol"], value=value)},
            raw=rows[index].raw | {"convert_vol": value},
        )
    amount = rows[0].numeric["remain_size"]
    rows[0] = replace(
        rows[0],
        numeric=rows[0].numeric
        | {
            "remain_size": replace(
                amount,
                unit="万元",
                certification=replace(amount.certification, source_unit="万元", target_unit="万元"),
            )
        },
    )
    first = record_for(replace_rows(context, "cb_share", rows), "cb_share")
    second = record_for(replace_rows(context, "cb_share", reversed(rows)), "cb_share")
    assert encode(first) == encode(second)
    numbers = first["facts"]["aggregation"]["business_statistics"]["numeric_ranges"]
    assert numbers["remain_size"] == {
        "certified_rows": 9,
        "unit_basis": None,
        "mixed_units": True,
        "minimum": None,
        "maximum": None,
    }
    assert numbers["convert_vol"]["minimum"] == 0.0


def test_foreign_currency_keeps_existing_cny_only_certification_boundary():
    """外币行不进入现有CNY认证统计，原值仍可观察且明确记录1行单位或币种不可用。"""
    context = bond_market_inputs({"cb_share": 9})
    rows = list(context.rows("cb_share"))
    amount = rows[0].numeric["remain_size"]
    rows[0] = replace(
        rows[0],
        numeric=rows[0].numeric
        | {
            "remain_size": replace(
                amount, currency="USD", certification=replace(amount.certification, currency="USD")
            )
        },
    )
    record = record_for(replace_rows(context, "cb_share", rows), "cb_share")
    statistics = record["facts"]["aggregation"]["business_statistics"]
    assert record["status"] == "observed" and record["retained_row_count"] == 9
    assert statistics["numeric_ranges"]["remain_size"]["certified_rows"] == 8
    assert (
        statistics["dimensions"]["remain_size_availability"]["values"]["unverified_or_missing"] == 1
    )


def test_missing_holder_name_and_business_date_remain_explicit_gaps():
    """缺持有人姓名不进入唯一持有人数；只有观察时点的来源不虚构业务日期。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    rows = list(context.rows("top10_cb_holders"))
    rows[0] = replace(rows[0], raw=rows[0].raw | {"holder_name": None})
    record = record_for(replace_rows(context, "top10_cb_holders", rows), "top10_cb_holders")
    assert record["status"] == "unknown"
    assert (
        record["facts"]["aggregation"]["business_statistics"]["distinct_bond_period_holder_pairs"]
        == 8
    )
    no_terms = bond_market_inputs({"cb_call": 9})
    rows = tuple(
        replace(row, availability=replace(row.availability, data_as_of=None))
        for row in no_terms.rows("cb_call")
    )
    record = record_for(replace_rows(no_terms, "cb_call", rows), "cb_call")
    assert record["facts"]["aggregation"]["source_as_of"] == {}


def test_holder_period_normalization_deduplicates_without_guessing_missing_dates():
    """同债券同姓名的两种等价报告期只计一个组合；缺报告期的行只计缺口，不能当成新报告期。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    rows = list(context.rows("top10_cb_holders"))
    rows[1] = replace(
        rows[1],
        raw=rows[1].raw | {"holder_name": rows[0].raw["holder_name"], "end_date": "2026-06-30"},
    )
    rows[2] = replace(rows[2], raw=rows[2].raw | {"end_date": None})
    record = record_for(replace_rows(context, "top10_cb_holders", rows), "top10_cb_holders")
    stats = record["facts"]["aggregation"]["business_statistics"]
    assert record["status"] == "unknown"
    assert stats["distinct_bond_period_holder_pairs"] == 7
    assert stats["dimensions"]["holder_period_availability"]["values"] == {
        "known": 8,
        "missing_or_invalid": 1,
    }


def test_dimension_limits_are_exact_deterministic_and_do_not_expose_large_fields():
    """分类超过8项及长字段仍计入other/oversized，字典序样本频次精确，反序结果一致。"""
    values = ["z"] * 3 + [str(index) for index in range(10)] + ["0", None, "", "大" * 129]
    first, second = _Distribution(), _Distribution()
    for value in values:
        first.add(value)
    for value in reversed(values):
        second.add(value)
    assert first.summary() == second.summary()
    summary = first.summary()
    assert summary["total"] == 17 and summary["missing"] == 2 and summary["oversized"] == 1
    assert summary["other"] == 5 and summary["values"]["0"] == 2
    assert "大" * 129 not in encode(summary)


@pytest.mark.parametrize("damage", ["binding", "source", "group", "unit"])
def test_bond_aggregation_capacity_and_binding_errors_fail_closed(damage):
    """错误绑定、来源超限、同债券超限及超长单位均以固定原因失败，不截断或发布部分结果。"""
    context = bond_market_inputs({"cb_basic": 9})
    assigned = registration("cb_basic")
    rows = context.rows("cb_basic")
    if damage == "binding":
        assigned = replace(assigned, entity_scope="stock")
    elif damage == "source":
        context = replace(context, config=replace(context.config, maximum_source_rows=8))
    elif damage == "group":
        context = replace(context, config=replace(context.config, maximum_rows_per_entity_source=1))
        rows = (rows[0], rows[0])
    else:
        fact = rows[0].numeric["par"]
        fact = replace(
            fact, unit="u" * 129, certification=replace(fact.certification, target_unit="u" * 129)
        )
        rows = (replace(rows[0], numeric=rows[0].numeric | {"par": fact}),)
    reason = (
        "research_bond_aggregation_binding_invalid"
        if damage == "binding"
        else f"research_bond_{damage}_capacity_exceeded"
    )
    with pytest.raises(ValueError, match="^" + reason + "$"):
        aggregate_bonds(context, assigned, rows, resolve_consumer(assigned.handler))


def test_conflicting_relations_remain_unknown_and_bind_full_relation_fingerprint():
    """两条可用关系冲突时不任选正股；同样冲突数量下更改目标仍改变完整摘要。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    first = context.config.confirmed_relations[0]
    other = relation(
        from_entity_id=first.from_entity_id, to_entity_id=B, evidence_refs=first.evidence_refs
    )
    context = replace(
        context,
        config=replace(
            context.config, confirmed_relations=context.config.confirmed_relations + (other,)
        ),
    )
    original = record_for(context, "cb_daily")
    assert original["status"] == "unknown"
    assert (
        original["facts"]["aggregation"]["business_statistics"]["dimensions"][
            "underlying_relation_status"
        ]["values"]["conflict"]
        == 1
    )
    changed = replace(
        context,
        config=replace(
            context.config,
            confirmed_relations=context.config.confirmed_relations[:-1]
            + (replace(other, to_entity_id="cn:security:C:life"),),
        ),
    )
    assert (
        record_for(changed, "cb_daily")["facts"]["aggregation"]["full_fingerprint"]
        != original["facts"]["aggregation"]["full_fingerprint"]
    )


def test_unsampled_latest_source_date_is_preserved_in_market_envelope():
    """最新业务日只在未抽样行时，信封仍披露全量已消费来源的最新日期，不能从样本推断。"""
    context = bond_market_inputs({"cb_call": 9})
    rows = [
        replace(row, availability=replace(row.availability, data_as_of="2026-10-01"))
        for row in context.rows("cb_call")
    ]
    context = replace_rows(context, "cb_call", rows)
    baseline = record_for(context, "cb_call")
    selected = {ref["source_record_id"] for ref in baseline["evidence_refs"]}
    index = next(
        i for i, row in enumerate(rows) if row.availability.source_record_id not in selected
    )
    rows[index] = replace(
        rows[index], availability=replace(rows[index].availability, data_as_of=DAY)
    )
    context = replace_rows(context, "cb_call", rows)
    result = attach_consumption(context, PRODUCT_ALIASES["P06"], ())[0]
    assert result.envelope.source_as_of["cb_call"] == DAY
    assert rows[index].availability.source_record_id not in {
        ref.source_record_id for ref in result.envelope.evidence_refs
    }


def test_unsampled_supporting_period_changes_full_bond_fingerprint():
    """仅改变未抽样cb_basic旁证的业务期间，引用与报价判断不变，全量指纹仍须识别溯源变化。"""
    context = bond_market_inputs(dict.fromkeys(COUNTS, 9))
    terms = [
        replace(row, availability=replace(row.availability, data_as_of="2026-10-01"))
        for row in context.rows("cb_basic")
    ]
    context = replace_rows(context, "cb_basic", terms)
    first = record_for(context, "cb_daily")
    selected = {
        ref["source_record_id"]
        for ref in first["evidence_refs"]
        if ref["source_api"] == "cb_basic"
    }
    index = next(
        index
        for index, row in enumerate(terms)
        if row.availability.source_record_id not in selected
    )
    terms[index] = replace(
        terms[index], availability=replace(terms[index].availability, data_as_of=DAY)
    )
    second = record_for(replace_rows(context, "cb_basic", terms), "cb_daily")
    before, after = first["facts"]["aggregation"], second["facts"]["aggregation"]
    assert first["status"] == second["status"] == "observed"
    assert first["evidence_refs"] == second["evidence_refs"]
    assert before["source_as_of"]["cb_basic"] == "2026-10-01"
    assert after["source_as_of"]["cb_basic"] == DAY
    for name in ("rows_fingerprint", "groups_fingerprint", "evidence_fingerprint"):
        assert before[name] == after[name]
    assert before["full_fingerprint"] != after["full_fingerprint"]
