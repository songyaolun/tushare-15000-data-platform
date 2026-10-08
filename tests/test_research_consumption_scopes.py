"""消费范围隔离、未承接股票行的显式缺口及参考库只读回归。"""

import json
from dataclasses import replace

import duckdb
import pytest
from test_research_adapter import configured, reference_database, source
from test_research_consumption import context_rows, registration
from test_research_governance import DAY, NOW, A, B, inputs, row

from tushare_entitlement_hub.equity_research.common import gate, make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    consumption_rows,
    execute_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.pipeline import run_research
from tushare_entitlement_hub.equity_research.storage import validate_research
from tushare_entitlement_hub.product_foundation.contracts import ProductState, Truth
from tushare_entitlement_hub.reference_storage import file_checksum


@pytest.mark.parametrize(
    "api,scope",
    [
        ("share_float", "stock"),
        ("moneyflow", "stock"),
        ("daily", "stock"),
        ("disclosure_date", "stock"),
        ("cb_call", "bond"),
        ("factor_list", "dictionary"),
        ("major_news", "event"),
        ("cn_cpi", "market"),
    ],
)
def test_market_row_selection_obeys_registered_scope(api, scope):
    """P06/P07/P03/P13的stock行只按实体取；bond/dictionary/event及P01 market仍返回全来源。"""
    rows = (row(api, "mapped", entity=A), row(api, "unmapped", entity=None))
    context = context_rows({api: rows})
    assigned = registration(api)
    assert assigned.entity_scope == scope
    assert consumption_rows(context, assigned, MARKET_ENTITY) == (
        () if scope == "stock" else context.rows(api)
    )
    if scope in {"stock", "event"}:
        assert consumption_rows(context, assigned, A) == context.rows(api, A)
    if scope == "market":
        assert consumption_rows(context, assigned, A) == context.rows(api)


def test_market_stock_execution_does_not_even_read_the_source(monkeypatch):
    """市场share_float消费不访问全量行或实体行索引；仍实际执行处理器并诚实保留空范围原因。"""
    context = inputs()

    def forbidden(*args, **kwargs):
        """stock范围市场消费不得请求来源行；任意调用均表示仍有全量复制入口。"""
        pytest.fail("市场stock消费访问了来源行")

    monkeypatch.setattr(ResearchInputs, "rows", forbidden)
    record, refs, _ = execute_consumption(context, registration("share_float"), MARKET_ENTITY)
    assert record["retained_row_count"] == 0 and refs == ()
    assert record["executed"] is True and record["status"] == "unknown"
    assert "stock_scope_requires_entity" in record["reason_codes"]


@pytest.mark.parametrize("api", ["cb_call", "factor_list", "major_news", "cn_cpi"])
def test_nonstock_market_records_keep_bound_and_unbound_source_evidence(api):
    """P06转债、P03字典、P13事件及P01市场来源在真实消费附加后仍保留绑定和未绑定两行证据。"""
    assigned = registration(api)
    rows = tuple(
        row(
            api,
            label,
            entity=identity,
            raw={
                **{field: "synthetic" for field in assigned.spec.selected_fields},
                "quality_tier": "verified",
                "ts_code": label,
                "factor_name": label,
                "asset_type": "STK",
            },
        )
        for label, identity in (("bound", A), ("unbound", None))
    )
    context = context_rows({api: rows})
    product = PRODUCT_ALIASES[assigned.spec.product]
    result = attach_consumption(context, product, (make_result(context, product, MARKET_ENTITY),))[
        0
    ]
    record = next(record for record in result.details["consumption"] if record["api"] == api)
    assert record["retained_row_count"] == 2
    assert record["status"] == "observed"
    assert {
        ref["source_record_id"] for ref in record["evidence_refs"] if ref["source_api"] == api
    } == {"bound", "unbound"}
    validate_result_consumption(result)


@pytest.mark.parametrize("existing", ["absent", "annotated", "blocked", "unsupported"])
def test_unmapped_or_unrepresented_stock_rows_are_counted_without_market_consumption(existing):
    """无实体、误绑市场和无对应结果的股票行均记录有界缺口；已有市场状态不被错误升级。"""
    records = tuple(
        row("share_float", str(index), entity=entity, raw={"float_share": 100})
        for index, entity in enumerate((A, B, None, MARKET_ENTITY))
    )
    context = context_rows({"share_float": records})
    product = PRODUCT_ALIASES["P06"]
    base = (make_result(context, product, A),)
    if existing != "absent":
        market = make_result(
            context,
            product,
            MARKET_ENTITY,
            references=records[0].evidence("float_share"),
            gates=(
                gate("base", Truth.FALSE if existing == "blocked" else Truth.TRUE, "base_blocked"),
            ),
        )
        if existing == "unsupported":
            market = replace(
                market, envelope=replace(market.envelope, state=ProductState.UNSUPPORTED)
            )
        base += (market,)
    results = attach_consumption(context, product, base)
    assert len(results) == 2
    stock = next(item for item in results if item.envelope.entity_id == A)
    market = next(item for item in results if item.envelope.entity_id == MARKET_ENTITY)
    own = next(item for item in stock.details["consumption"] if item["api"] == "share_float")
    aggregate = next(item for item in market.details["consumption"] if item["api"] == "share_float")
    assert own["retained_row_count"] == 1
    assert aggregate["retained_row_count"] == 0 and aggregate["evidence_refs"] == []
    assert market.details["stock_scope_gaps"] == {
        "share_float": {
            "unmapped_retained_rows": 2,
            "unrepresented_retained_rows": 1,
            "adapter_entity_rejections": False,
        }
    }
    assert "stock_rows_without_entity_result" in market.envelope.gate_codes
    assert market.envelope.state.value == (
        existing if existing in {"blocked", "unsupported"} else "incomplete"
    )
    assert not market.envelope.decision_allowed
    assert "stock_scope_gaps" not in stock.details
    for result in results:
        validate_result_consumption(result)


def test_only_unmapped_stock_rows_still_create_an_explicit_diagnostic():
    """没有任何实体结果或非股票来源时，一条无实体stock行仍形成unknown诊断，不能静默丢弃。"""
    context = inputs(row("share_float", entity=None), entities=())
    result = attach_consumption(context, PRODUCT_ALIASES["P06"], ())[0]
    assert result.envelope.entity_id == MARKET_ENTITY
    assert result.details["stock_scope_gaps"]["share_float"]["unmapped_retained_rows"] == 1
    assert all(item["retained_row_count"] == 0 for item in result.details["consumption"])
    assert result.envelope.evidence_refs == ()
    assert result.envelope.state is ProductState.INCOMPLETE


def test_persisted_market_stock_count_cannot_claim_entity_rows():
    """将有一条stock消费的证券结果改成市场身份后，验证器拒绝伪造的范围消费。"""
    context = context_rows({"share_float": (row("share_float", raw={"float_share": 100}),)})
    product = PRODUCT_ALIASES["P06"]
    stock = attach_consumption(context, product, (make_result(context, product, A),))[0]
    forged = replace(stock, envelope=replace(stock.envelope, entity_id=MARKET_ENTITY))
    with pytest.raises(ValueError, match="^research_consumption_scope_mismatch$"):
        validate_result_consumption(forged)


def test_adapter_unmapped_stock_rejection_remains_visible_in_real_run(tmp_path):
    """合成参考库一条映射A、一条未知代码；逐股消费1行、适配排除1行且市场仅记录缺口。"""
    rows = [
        source("share_float", "mapped", float_date=DAY, float_share=100, ts_code="A"),
        source("share_float", "unmapped", float_date=DAY, float_share=100, ts_code="UNKNOWN"),
    ]
    reference = reference_database(tmp_path, {"share_float": rows})
    checksum = file_checksum(reference)
    output = tmp_path / "scope-products.duckdb"
    summary = run_research(reference, output, NOW, configured("share_float"), ("P06",))
    assert summary["processing"]["scanned_rows"] == 2
    assert summary["processing"]["retained_rows"] == 1
    assert summary["processing"]["excluded_rows"] == 1
    assert summary["status"] == "incomplete"
    with duckdb.connect(
        str(output), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        results = [
            ResearchResult.from_json(payload)
            for (payload,) in connection.execute("SELECT payload FROM research.result").fetchall()
        ]
    market = next(item for item in results if item.envelope.entity_id == MARKET_ENTITY)
    assert market.details["stock_scope_gaps"]["share_float"] == {
        "unmapped_retained_rows": 0,
        "unrepresented_retained_rows": 0,
        "adapter_entity_rejections": True,
    }
    records = [
        record
        for item in results
        for record in item.details["consumption"]
        if record["api"] == "share_float"
    ]
    assert sum(record["retained_row_count"] for record in records) == 1
    assert all("stock_entity_unresolved" in record["reason_codes"] for record in records)
    assert (
        next(record for record in market.details["consumption"] if record["api"] == "share_float")[
            "retained_row_count"
        ]
        == 0
    )
    assert validate_research(reference, output)["valid"]
    assert file_checksum(reference) == checksum
    assert "UNKNOWN" not in json.dumps(market.details["stock_scope_gaps"])
