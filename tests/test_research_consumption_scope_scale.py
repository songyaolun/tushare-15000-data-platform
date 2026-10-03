"""超过72000条股票来源行的真实消费分流及有界市场结果回归。"""

from dataclasses import replace
from datetime import date, timedelta

import duckdb
from test_research_consumption import registration
from test_research_governance import DAY, inputs, row

from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    consumption_rows,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.storage import initialize_tables, save_results
from tushare_entitlement_hub.store import encode


def high_cardinality_inputs():
    """生成150实体各487条可见解禁记录，总计73050条，单实体未超过512行保留合同。

    业务日覆盖487日，实际消费retained_row_count仍包含全部事件；当前字段证据由现有
    core处理器按最新日生成。另有一条无股票绑定的转债来源，必须由市场结果承接。
    """
    identities = tuple(f"cn:security:S{index:03}:life" for index in range(150))
    base = inputs(entities=identities)
    days = tuple(
        (date.fromisoformat(DAY) - timedelta(days=offset)).isoformat() for offset in range(487)
    )
    rows = tuple(
        row(
            "share_float",
            f"share-{stock}-{offset}",
            entity=identity,
            day=day,
            raw={
                "quality_tier": "verified",
                "ts_code": f"S{stock:03}",
                "ann_date": day,
                "float_date": day,
                "float_share": 100.0,
                "float_ratio": 0.001,
            },
        )
        for stock, identity in enumerate(identities)
        for offset, day in enumerate(days)
    )
    bond = row(
        "cb_call",
        "bond",
        entity=None,
        raw={
            "quality_tier": "verified",
            "ts_code": "SYNTHETIC_CB",
            "ann_date": DAY,
            "call_type": "不赎回",
        },
    )
    return replace(
        base,
        config=replace(base.config, maximum_source_rows=80000),
        rows_by_api={"share_float": rows, "cb_call": (bond,)},
    )


def test_large_stock_source_is_consumed_per_entity_and_never_duplicated_in_market(
    tmp_path, monkeypatch
):
    """73050行逐股真实消费计数守恒，市场stock计数为0而转债仍可见，缩小stock输入不改变市场JSON。"""
    context = high_cardinality_inputs()
    product = PRODUCT_ALIASES["P06"]
    assigned = registration("share_float")
    source_rows = context.rows("share_float")
    routed = tuple(
        item
        for identity in context.entity_ids
        for item in consumption_rows(context, assigned, identity)
    )
    assert len(source_rows) == len(routed) == 73050
    assert {id(item) for item in routed} == {id(item) for item in source_rows}

    def forbidden(*args, **kwargs):
        """市场stock路由必须在请求任何来源数据前返回空，不能靠读取后截断过关。"""
        raise AssertionError("市场结果读取了stock来源")

    with monkeypatch.context() as patch:
        patch.setattr(ResearchInputs, "rows", forbidden)
        assert consumption_rows(context, assigned, MARKET_ENTITY) == ()
    base = tuple(make_result(context, product, identity) for identity in context.entity_ids)
    results = attach_consumption(context, product, base)
    stocks = [result for result in results if result.envelope.entity_id != MARKET_ENTITY]
    market = next(result for result in results if result.envelope.entity_id == MARKET_ENTITY)
    stock_records = [
        next(record for record in result.details["consumption"] if record["api"] == "share_float")
        for result in stocks
    ]
    assert len(stocks) == 150
    assert all(
        record["retained_row_count"] == 487 and record["status"] == "observed"
        for record in stock_records
    )
    assert sum(record["retained_row_count"] for record in stock_records) == 73050
    market_records = {record["api"]: record for record in market.details["consumption"]}
    assert market_records["share_float"]["retained_row_count"] == 0
    assert market_records["share_float"]["evidence_refs"] == []
    assert market_records["cb_call"]["retained_row_count"] == 1
    assert market_records["cb_call"]["status"] == "observed"
    assert "stock_scope_gaps" not in market.details
    market_bytes = len(market.to_json().encode())
    assert market_bytes < 256 * 1024
    reduced = replace(context, rows_by_api={**context.rows_by_api, "share_float": source_rows[:1]})
    reduced_market = next(
        result
        for result in attach_consumption(reduced, product, base)
        if result.envelope.entity_id == MARKET_ENTITY
    )
    assert market.to_json() == reduced_market.to_json()
    with duckdb.connect(
        str(tmp_path / "scope-scale.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        assert save_results(connection, results, 128)["result"] == 151
        rebuilt_count = 0
        retained_rows = 0
        cursor = connection.execute("SELECT payload FROM research.result")
        while batch := cursor.fetchmany(128):
            for (payload,) in batch:
                result = ResearchResult.from_json(payload)
                validate_result_consumption(result)
                retained_rows += next(
                    record
                    for record in result.details["consumption"]
                    if record["api"] == "share_float"
                )["retained_row_count"]
                rebuilt_count += 1
        assert rebuilt_count == 151 and retained_rows == 73050
    print(
        "消费范围高基数验收 "
        + encode(
            {
                "stock_source_rows": 73050,
                "stock_results": 150,
                "stock_retained_row_count": retained_rows,
                "market_stock_retained_row_count": 0,
                "market_bond_retained_row_count": 1,
                "market_payload_bytes": market_bytes,
                "stored_results": rebuilt_count,
            }
        )
    )
