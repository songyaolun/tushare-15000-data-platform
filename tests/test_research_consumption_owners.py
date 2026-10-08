"""P13 多事件共享实体时，stock 来源只由一个确定性结果承接。"""

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import duckdb
import pytest
from test_research_candidates import research_inputs, research_parents, research_row
from test_research_consumption import registration

from tushare_entitlement_hub.equity_research import consumption
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    count_consumption,
    execute_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.events import build_events
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.storage import initialize_tables, save_results
from tushare_entitlement_hub.store import encode


def multi_event_inputs(revisions=1):
    """构造两实体的披露、改名及无实体新闻，日期均显式绑定，不访问真实参考库。

    例：revisions=1时，每实体一条披露生成计划、实际、修订三个事件，另有一个改名事件。
    """
    context = research_inputs(2)
    revisions = [
        (date(2026, 10, 20) + timedelta(days=index)).isoformat() for index in range(revisions)
    ]
    datasets = {
        api: tuple(
            research_row(identity, api, fields, record=identity) for identity in context.entity_ids
        )
        for api, fields in {
            "disclosure_date": {
                "ann_date": "20260930",
                "end_date": "20260630",
                "pre_date": "20261010",
                "actual_date": "20261011",
                "modify_date": revisions,
            },
            "namechange": {"name": "新名", "start_date": "20261001", "ann_date": "20260930"},
        }.items()
    }
    datasets["major_news"] = (
        research_row(None, "major_news", {"title": "公开事实", "pub_time": "20260930"}),
    )
    return replace(context, rows_by_api={**context.rows_by_api, **datasets})


@pytest.mark.parametrize("revisions,with_parents", [(1, False), (1, True), (32, False)])
def test_real_events_consume_each_stock_row_once_and_preserve_events(
    tmp_path, monkeypatch, revisions, with_parents
):
    """真实多阶段事件及bundle只消费一次stock行；事件事实不变、顺序无关且保存重建后仍守恒。"""
    context = multi_event_inputs(revisions)
    parents = (
        tuple(parent for group in research_parents(context) for parent in group)
        if with_parents
        else ()
    )
    base = build_events(context, parents)
    consumed = Counter()
    resolve = consumption.resolve_consumer

    def tracked_consumer(handler):
        """包装真实处理器并累计实际传入行数；例：多个事件的同实体披露仅累加一行。"""
        consumer = resolve(handler)

        def tracked(inputs, product, api, rows, entity_id):
            """保留实际处理器语义；例：空行执行保留调用记录，但不增加已消费行数。"""
            consumed[api, entity_id] += len(rows)
            return consumer(inputs, product, api, rows, entity_id)

        return tracked

    with monkeypatch.context() as patch:
        patch.setattr(consumption, "resolve_consumer", tracked_consumer)
        results = attach_consumption(context, PRODUCT_ALIASES["P13"], base)
    assert len(results) == len(base) == 2 * (3 + revisions + int(with_parents)) + 1
    counts = {}
    for original, result in zip(base, results, strict=True):
        assert result.details.items() >= original.details.items()
        assert result.metrics == original.metrics
        assert result.predicate_results == original.predicate_results
        assert result.parent_result_ids == original.parent_result_ids
        assert result.envelope.contrary_evidence == original.envelope.contrary_evidence
        assert set(original.envelope.quality_gates) <= set(result.envelope.quality_gates)
        validate_result_consumption(result)
        count_consumption(counts, result)
    for identity in context.entity_ids:
        own = [result for result in results if result.envelope.entity_id == identity]
        owner = min(result.envelope.result_id for result in own)
        for api in ("disclosure_date", "namechange"):
            assert consumed[api, identity] == 1
            records = [
                (result.envelope.result_id, record)
                for result in own
                for record in result.details["consumption"]
                if record["api"] == api
            ]
            assert sum(record["retained_row_count"] for _, record in records) == 1
            for result_id, record in records:
                if result_id == owner:
                    assert record["retained_row_count"] == 1
                    assert record["status"] == "observed"
                else:
                    assert record["stock_consumption_owner_result_id"] == owner
                    assert record["retained_row_count"] == 0 and record["evidence_refs"] == []
                    assert record["status"] == "unknown"
                    assert "stock_consumption_owned_by_other_result" in record["reason_codes"]
                    assert all(
                        not any(values.values()) for values in record["field_quality"].values()
                    )
    assert counts["disclosure_date"]["observed"] == counts["namechange"]["observed"] == 2
    market = next(result for result in results if result.envelope.entity_id == MARKET_ENTITY)
    assert (
        next(item for item in market.details["consumption"] if item["api"] == "major_news")[
            "retained_row_count"
        ]
        == 1
    )
    assert "stock_scope_gaps" not in market.details
    reversed_results = attach_consumption(context, PRODUCT_ALIASES["P13"], tuple(reversed(base)))
    assert {item.envelope.result_id: item.to_json() for item in results} == {
        item.envelope.result_id: item.to_json() for item in reversed_results
    }
    assert max(len(item.to_json().encode()) for item in results) < 8 * 1024 * 1024
    with duckdb.connect(
        str(tmp_path / "events.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        assert save_results(connection, parents + results, 32)["result"] == len(parents) + len(
            results
        )
        rebuilt = tuple(
            ResearchResult.from_json(payload)
            for (payload,) in connection.execute(
                "SELECT payload FROM research.result WHERE payload->'envelope'->>'product_id'=?",
                [PRODUCT_ALIASES["P13"]],
            ).fetchall()
        )
    assert {item.envelope.result_id: item.to_json() for item in rebuilt} == {
        item.envelope.result_id: item.to_json() for item in results
    }


def test_nonowner_runs_real_handler_without_reading_stock_source(monkeypatch):
    """非承载事件仍实际调用处理器，但不读取来源，未知记录保留可回查的承载者ID。"""

    def forbidden(*args, **kwargs):
        """非承载结果不应请求来源；任意调用均表示仍有重复读路径。"""
        pytest.fail("非承载事件读取stock来源")

    context = multi_event_inputs()
    monkeypatch.setattr(ResearchInputs, "rows", forbidden)
    record, refs, gates = execute_consumption(
        context,
        registration("disclosure_date"),
        context.entity_ids[0],
        stock_owner_result_id="owner",
    )
    assert record["executed"] is True and record["status"] == "unknown"
    assert record["retained_row_count"] == 0 and refs == ()
    assert record["stock_consumption_owner_result_id"] == "owner"
    assert all(not check.evidence_refs for check in gates)


@pytest.mark.parametrize(
    "api,owner", [("major_news", "owner"), ("disclosure_date", ""), ("disclosure_date", True)]
)
def test_invalid_owner_arguments_fail_closed(api, owner):
    """非stock范围或空/错类型承载者ID均拒绝，避免错误限制其他来源或伪造所有权。"""
    context = research_inputs(1)
    with pytest.raises(ValueError, match="^research_consumption_scope_mismatch$"):
        execute_consumption(
            context, registration(api), context.entity_ids[0], stock_owner_result_id=owner
        )


@pytest.mark.parametrize(
    "damage", ["nonstock", "type", "empty", "self", "count", "status", "evidence", "reason"]
)
def test_persisted_nonowner_cannot_claim_stock_consumption(damage):
    """篡改承载者、范围、状态、行数或证据时验证失败，不能把非承载记录伪装为真实消费。"""
    context = multi_event_inputs()
    product = PRODUCT_ALIASES["P13"]
    original = make_result(context, product, context.entity_ids[0])
    result = attach_consumption(context, product, (original,))[0]
    details = result.details
    record, _, checks = execute_consumption(
        context,
        registration("disclosure_date"),
        context.entity_ids[0],
        stock_owner_result_id="owner",
    )
    if damage == "nonstock":
        record = next(item for item in details["consumption"] if item["api"] == "major_news")
        record["stock_consumption_owner_result_id"] = "owner"
    elif damage in {"type", "empty", "self"}:
        record["stock_consumption_owner_result_id"] = {
            "type": True,
            "empty": "",
            "self": result.envelope.result_id,
        }[damage]
    elif damage == "count":
        record["retained_row_count"] = 1
    elif damage == "status":
        record["status"] = "not_applicable"
    elif damage == "evidence":
        record["evidence_refs"] = [
            context.rows("disclosure_date")[0].evidence("ann_date")[0].to_dict()
        ]
    else:
        record["reason_codes"].remove("stock_consumption_owned_by_other_result")
    details["consumption"] = [
        record if item["api"] == record["api"] else item for item in details["consumption"]
    ]
    result = replace(
        result,
        details_json=encode(details),
        envelope=replace(result.envelope, quality_gates=result.envelope.quality_gates + checks),
    )
    with pytest.raises(ValueError, match="^research_consumption_scope_mismatch$"):
        validate_result_consumption(result)
