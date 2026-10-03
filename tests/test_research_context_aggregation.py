"""高基数市场旁证的全量统计、有界输出和真实内存DuckDB持久化回归。"""

from dataclasses import replace

import duckdb
import pytest
from test_research_context_consumers import INDEX, context_inputs, context_row

from tushare_entitlement_hub.equity_research import context_aggregation as aggregation
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    execute_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import (
    PRODUCT_ALIASES,
    load_consumption_registry,
)
from tushare_entitlement_hub.equity_research.context_consumers import consume_context
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.equity_research.storage import (
    fingerprints,
    initialize_tables,
    record_rows,
    save_results,
    stored_rows,
)
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity
from tushare_entitlement_hub.store import digest, encode


def formal_identifiers(row):
    """统一合成行及证书的正式长度身份；例如原行与close证书都引用同一64位契约哈希。"""
    updated = replace(
        row,
        raw=row.raw
        | {
            "contract_hash": "a" * 64,
            "business_group_key": digest([row.source_api, row.entity_id]),
        },
    )
    return replace(
        updated,
        numeric={
            field: replace(
                fact,
                evidence_refs=updated.evidence(field),
                certification=replace(fact.certification, evidence_refs=updated.evidence(field)),
            )
            for field, fact in row.numeric.items()
        },
    )


@pytest.fixture(scope="module")
def high_cardinality_inputs():
    """创建12000个真实独立指数身份与3000对同日因子/日线，全部为确定性合成数据。

    index_basic均元数据完整；idx_factor_pro各1000组匹配、冲突、未认证，原始字段都存在。
    所有身份与来源哈希采用正式长度；不读取文件、真实行情或凭据。
    """
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind="index", local_id=f"I{i:05}", lifecycle_id="published"
            ),
            name=f"指数{i}",
        )
        for i in range(12000)
    )
    datasets = {"index_basic": [], "idx_factor_pro": [], "index_daily": []}
    for i, entity in enumerate(entities):
        code = f"I{i:05}"
        datasets["index_basic"].append(
            context_row(
                "index_basic",
                {
                    "ts_code": code,
                    "market": "SH" if i % 2 else "SZ",
                    "publisher": f"publisher-{i % 20:02}",
                    "index_type": "综合",
                    "weight_rule": "市值加权",
                },
                entity=entity.identity.entity_id,
                record=digest(["index_basic", i]),
            )
        )
        if i < 3000:
            values = {
                "ts_code": code,
                "trade_date": "20260930",
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 100.0,
                "amount": 1000.0,
            }
            for api in ("idx_factor_pro", "index_daily"):
                datasets[api].append(
                    context_row(
                        api,
                        values | ({"close": 11.0} if api == "index_daily" and i % 3 == 1 else {}),
                        entity=entity.identity.entity_id,
                        record=digest([api, i]),
                        units={
                            "close": ("index", "index", None),
                            "pre_close": ("index", "index", None),
                        },
                        unverified=("close", "vol", "amount")
                        if api == "idx_factor_pro" and i % 3 == 2
                        else (),
                    )
                )
    inputs = context_inputs()
    return replace(
        inputs,
        config=replace(
            inputs.config, foundation=replace(inputs.config.foundation, entities=entities)
        ),
        rows_by_api={
            api: tuple(formal_identifiers(row) for row in rows) for api, rows in datasets.items()
        },
    )


def consume_record(inputs, api):
    """调用真实消费封装；例如index_basic返回保留行数、字段质量和可复核证据样本。"""
    registration = next(item for item in load_consumption_registry() if item.spec.api == api)
    return execute_consumption(inputs, registration, MARKET_ENTITY)[0]


@pytest.fixture(scope="module")
def high_cardinality_result(high_cardinality_inputs):
    """复用不可变的完整P01基线，避免各个变更回归重复计算相同的18000行观察集。"""
    product = PRODUCT_ALIASES["P01"]
    inputs = high_cardinality_inputs
    return attach_consumption(inputs, product, (make_result(inputs, product, MARKET_ENTITY),))[0]


def test_high_cardinality_result_is_bounded_deterministic_and_persistable(
    high_cardinality_inputs,
    high_cardinality_result,
):
    """12000指数和3000因子全部计入汇总，反转输入字节一致，512MB内存库完整保存及重建。

    每来源至多40个样本、2个门禁；完整P01载荷小于1MiB，不依靠数据库增加内存。
    样本身份、字段、契约与时间逐一回查输入；统计和指纹都覆盖未抽样的行。
    """
    inputs = high_cardinality_inputs
    product = PRODUCT_ALIASES["P01"]
    result = high_cardinality_result
    validate_result_consumption(result)
    records = {record["api"]: record for record in result.details["consumption"]}
    source_rows = {
        (row.source_api, row.raw["source_record_id"]): row
        for rows in inputs.rows_by_api.values()
        for row in rows
    }
    for api, count in (("index_basic", 12000), ("idx_factor_pro", 3000)):
        record = records[api]
        summary = record["facts"]["aggregation"]
        assert record["executed"] and record["retained_row_count"] == count
        assert summary["total_rows"] == summary["usable_rows"] == summary["group_count"] == count
        assert sum(summary["status_counts"].values()) == count
        assert sum(summary["rows_by_group_status"].values()) == count
        assert summary["sample_count"] == len(record["evidence_refs"]) <= 40
        assert len(record["gates"]) == 2
        assert summary["truncated"] is True
        assert summary["sampling_policy"]["statistics_scope"] == "all_rows_and_all_latest_groups"
        assert len(summary["full_fingerprint"]) == len(summary["rows_fingerprint"]) == 64
        assert record["facts"]["independent_evidence"] is False
        assert record["facts"]["core_score_contribution"] == 0
        assert "observations" not in record["facts"]
        for reference in record["evidence_refs"]:
            row = source_rows[(reference["source_api"], reference["source_record_id"])]
            assert reference["contract_hash"] == row.raw["contract_hash"]
            assert reference["observed_at"] == row.availability.observed_at
            assert all(field in row.raw for field in reference["fields"])
            assert reference["locator"].endswith("/" + row.raw["source_record_id"])
    basic = records["index_basic"]["facts"]["aggregation"]
    assert records["index_basic"]["status"] == "observed"
    assert records["index_basic"]["reason_codes"] == []
    assert basic["metadata_availability"]["market"]["values"] == [
        {"value": "SH", "groups": 6000},
        {"value": "SZ", "groups": 6000},
    ]
    assert basic["metadata_availability"]["publisher"]["other_groups"] == 2400
    assert basic["metadata_availability"]["publisher"]["distinct_values"] == 20
    factor = records["idx_factor_pro"]["facts"]["aggregation"]
    assert factor["status_counts"] == {"conflict": 1000, "observed": 1000, "unknown": 1000}
    assert factor["field_coverage"]["close"] == {"present_rows": 3000, "certified_rows": 2000}
    assert factor["quote_check_counts"]["close"] == {
        "conflict": 1000,
        "matched": 1000,
        "unknown": 1000,
    }
    reversed_inputs = replace(
        inputs, rows_by_api={api: tuple(reversed(rows)) for api, rows in inputs.rows_by_api.items()}
    )
    repeated = attach_consumption(
        reversed_inputs, product, (make_result(reversed_inputs, product, MARKET_ENTITY),)
    )[0]
    assert encode(result.to_dict()) == encode(repeated.to_dict())
    projected = tuple(record_rows(result))
    sizes = {
        table: sum(len(encode(row["payload"]).encode()) for name, row in projected if name == table)
        for table in ("result", "gate", "lineage", "evidence")
    }
    assert sizes["result"] < 1024 * 1024
    assert len(result.envelope.evidence_refs) <= 120
    assert len(result.envelope.quality_gates) <= 80
    print(
        "聚合输出实测 "
        + encode(
            sizes
            | {
                "details_bytes": len(result.details_json.encode()),
                "envelope_refs": len(result.envelope.evidence_refs),
                "gates": len(result.envelope.quality_gates),
                "index_basic_record_bytes": len(encode(records["index_basic"]).encode()),
                "idx_factor_pro_record_bytes": len(encode(records["idx_factor_pro"]).encode()),
            }
        )
    )
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        initialize_tables(connection)
        counts = save_results(connection, (result,), 128)
        assert counts["result"] == 1
        assert fingerprints(projected) == fingerprints(stored_rows(connection, inputs.run_id, 128))


@pytest.mark.parametrize("api", ["index_basic", "idx_factor_pro"])
def test_unsampled_row_changes_full_fingerprint(
    high_cardinality_inputs, high_cardinality_result, api
):
    """修改未抽样行仍改变全量指纹；因子原先匹配的未抽样组变化必须增加冲突计数。"""
    inputs = high_cardinality_inputs
    previous = next(
        record for record in high_cardinality_result.details["consumption"] if record["api"] == api
    )
    sampled = {ref["source_record_id"] for ref in previous["evidence_refs"]}
    rows = list(inputs.rows(api))
    index = next(
        i for i, row in enumerate(rows) if row.raw["source_record_id"] not in sampled and i % 3 == 0
    )
    original = rows[index]
    if api == "index_basic":
        rows[index] = replace(original, raw=original.raw | {"market": "BJ"})
    else:
        rows[index] = replace(
            original,
            raw=original.raw | {"close": 12.0},
            numeric=original.numeric | {"close": replace(original.numeric["close"], value=12.0)},
        )
    changed = consume_record(
        replace(inputs, rows_by_api=inputs.rows_by_api | {api: tuple(rows)}), api
    )
    before, after = (record["facts"]["aggregation"] for record in (previous, changed))
    assert before["total_rows"] == after["total_rows"]
    assert before["full_fingerprint"] != after["full_fingerprint"]
    assert before["rows_fingerprint"] != after["rows_fingerprint"]
    assert before["groups_fingerprint"] != after["groups_fingerprint"]
    if api == "idx_factor_pro":
        assert after["status_counts"]["conflict"] == before["status_counts"]["conflict"] + 1


@pytest.mark.parametrize("api", ["index_basic", "index_dailybasic", "idx_factor_pro", "shibor_lpr"])
def test_unverified_high_cardinality_context_remains_unknown(api):
    """同类市场来源65组原始数字存在但未认证时，均保持unknown、有原因，证据输出有界。"""
    rows = tuple(
        context_row(
            api,
            {
                "ts_code": str(i),
                "trade_date": "20260930",
                "date": "20260930",
                "pe_ttm": 10.0,
                "pb": 2.0,
                "turnover_rate": 0.03,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
                "1y": 0.03,
                "5y": 0.04,
            },
            entity=INDEX,
            record=str(i),
            unverified=("pe_ttm", "pb", "turnover_rate", "close", "vol", "amount", "1y", "5y"),
        )
        for i in range(65)
    )
    record = consume_record(context_inputs(*rows), api)
    assert record["status"] == "unknown"
    assert record["reason_codes"]
    if api == "index_dailybasic":
        assert record["reason_codes"] == ["index_valuation_units_unverified"]
    summary = record["facts"]["aggregation"]
    assert summary["status_counts"] == {"unknown": 65}
    assert summary["total_rows"] == summary["group_count"] == 65
    assert len(record["evidence_refs"]) <= 40


@pytest.mark.parametrize("count", [64, 65])
def test_aggregation_threshold_and_duplicate_rows_are_explicit(count):
    """64条以内保持细节，65条相同来源记录仍全量计数；单证据未截断，最新组只有1个。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    result = consume_context(
        context_inputs(row), "P01", "index_basic", (row,) * count, MARKET_ENTITY
    )
    if count == 64:
        assert len(result["facts"]["observations"]) == 1
    else:
        summary = result["facts"]["aggregation"]
        assert summary["total_rows"] == 65
        assert summary["group_count"] == summary["sample_count"] == 1
        assert summary["truncated"] is False


def test_rejected_rows_and_ambiguous_groups_are_counted_and_hashed():
    """质量拒绝行进入输入指纹；同日多版本保持歧义unknown，不能任取一条或丢弃尾组。"""
    rows = tuple(
        context_row("index_basic", {"ts_code": "I"}, entity=INDEX, record=str(i)) for i in range(65)
    )
    context = context_inputs(*rows)
    result = consume_context(context, "P01", "index_basic", rows, MARKET_ENTITY)
    summary = result["facts"]["aggregation"]
    assert summary["group_count"] == 1
    assert summary["rows_by_group_status"] == {"unknown": 65}
    assert result["reason_codes"] == ("context_current_version_ambiguous",)
    rejected = tuple(replace(row, raw=row.raw | {"is_quarantined": True}) for row in rows)
    failed = consume_context(context, "P01", "index_basic", rejected, MARKET_ENTITY)
    rejected_summary = failed["facts"]["aggregation"]
    assert rejected_summary["rejected_rows"] == rejected_summary["total_rows"] == 65
    assert rejected_summary["group_count"] == rejected_summary["sample_count"] == 0
    assert rejected_summary["rows_fingerprint"] != summary["rows_fingerprint"]
    assert failed["status"] == "unknown" and failed["reason_codes"]


def test_oversized_metadata_is_counted_without_echoing_text():
    """超长权重规则不复制到聚合输出，仍计为可用并计入超长组和全量指纹。"""
    value = "市值" * 100
    rows = tuple(
        context_row(
            "index_basic",
            {
                "ts_code": str(i),
                "market": "SH",
                "publisher": "P",
                "index_type": "综合",
                "weight_rule": value,
            },
            entity=INDEX,
            record=str(i),
        )
        for i in range(65)
    )
    result = consume_record(context_inputs(*rows), "index_basic")
    stats = result["facts"]["aggregation"]["metadata_availability"]["weight_rule"]
    assert stats["oversized_value_groups"] == stats["available_groups"] == 65
    assert stats["values"] == []
    assert value not in encode(result)


def test_oversized_sample_reference_fails_before_unbounded_output(monkeypatch):
    """单证据超过明确字节边界时固定错误失败，不截断可回查的来源定位信息。"""
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    monkeypatch.setattr(aggregation, "MAXIMUM_REFERENCE_BYTES", 1)
    with pytest.raises(ValueError, match="^research_context_evidence_capacity_exceeded$"):
        consume_context(context_inputs(row), "P01", "index_basic", (row,) * 65, MARKET_ENTITY)


def test_fingerprint_distinguishes_nonfinite_source_values_and_types():
    """被数值门禁拒绝的NaN/Infinity仍能审计，且与同拼写字符串及缺字段区分。"""
    assert aggregation.fingerprint_leaf(float("nan")) != aggregation.fingerprint_leaf("NaN")
    row = context_row("index_basic", {"ts_code": "I"}, entity=INDEX)
    values = [None, float("inf"), float("nan"), "NaN"]
    hashes = {
        aggregation.row_fingerprint(replace(row, raw=row.raw | {"close": value}), ("close",))
        for value in values
    }
    assert len(hashes) == 4
    assert aggregation.row_fingerprint(row, ("close",)) not in hashes


@pytest.mark.parametrize(
    "states,expected",
    [
        (set(), "unknown"),
        ({"not_applicable"}, "not_applicable"),
        ({"observed", "not_applicable"}, "observed"),
        ({"observed", "not_enabled_due_to_history"}, "not_enabled_due_to_history"),
        ({"unknown", "observed"}, "unknown"),
        ({"unknown", "conflict"}, "conflict"),
    ],
)
def test_aggregate_status_preserves_fail_closed_precedence(states, expected):
    """聚合沿用原状态优先级，未知/冲突/历史不足均不能被已观察组冲淡。"""
    assert aggregation.context_status(states) == expected
