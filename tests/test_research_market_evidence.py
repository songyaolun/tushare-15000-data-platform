"""真实市场主体的高基数证据、确定性及512MB持久化回归，全部输入为合成行。"""

from dataclasses import replace

import duckdb
import pytest
from test_research_context_aggregation import formal_identifiers
from test_research_market import market_inputs, row

from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.market import build_market
from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS, ResearchResult
from tushare_entitlement_hub.equity_research.storage import (
    MAXIMUM_RESULT_JSON_BYTES,
    fingerprints,
    initialize_tables,
    record_rows,
    save_results,
    stored_rows,
)
from tushare_entitlement_hub.store import digest, encode


@pytest.fixture(scope="module")
def large_market_inputs():
    """创建1536只股票、1024条边界和1280条停复牌记录，使用正式长度来源标识。

    前1024只有效且同日复牌，随后256只无成交，最后256只停牌；全部同日同币种。
    旧build_market仅此规模即超过8MiB，不能用空make_result替代实际产品计算。
    """
    rows = []
    for index in range(1536):
        daily = row(index=index, record=digest(["daily", index]))
        if 1024 <= index < 1280:
            daily = replace(daily, raw=daily.raw | {"vol": 0.0})
        rows.append(daily)
        if index < 1024:
            rows.append(
                row(
                    "stk_limit",
                    index=index,
                    record=digest(["stk_limit", index]),
                    fields={"trade_date": "20260930", "down_limit": 10.0},
                )
            )
        if index < 1024 or index >= 1280:
            rows.append(
                row(
                    "suspend_d",
                    index=index,
                    record=digest(["suspend_d", index]),
                    fields={
                        "trade_date": "20260930",
                        "suspend_type": "R" if index < 1024 else "S",
                        "suspend_timing": None,
                    },
                )
            )
    return market_inputs(1536, tuple(formal_identifiers(item) for item in rows))


@pytest.fixture(scope="module")
def large_market_result(large_market_inputs):
    """调用真实build_market再附加P01消费；例如1536股仍只产生一个带完整指标的市场结果。"""
    return attach_consumption(
        large_market_inputs, PRODUCT_IDS[0], build_market(large_market_inputs)
    )[0]


def test_real_market_payload_is_bounded_and_rebuildable(large_market_inputs, large_market_result):
    """旧实现超过8MiB的真实市场结果须小于门禁，512MB保存、反序列化、派生重建完全一致。"""
    result = large_market_result
    size = len(result.to_json().encode("utf-8"))
    print(f"真实build_market结果字节数={size}")
    assert size < MAXIMUM_RESULT_JSON_BYTES
    validate_result_consumption(result)
    assert result.details["denominator_count"] == 1024
    assert result.details["excluded_no_trade_count"] == 256
    assert result.details["excluded_suspended_count"] == 256
    assert result.details["down_limit_known_count"] == 1024
    assert result.metric("breadth") == result.metric("down_limit_share") == 1
    assert result.metric("liquidity_concentration") == 103 / 1024
    assert result.metric("stress_score") == pytest.approx(100 / 3 * (1 + 103 / 1024))
    assert len(result.details["suspension_observations"]) <= 8
    assert len(result.envelope.evidence_refs) <= 80
    assert len(result.envelope.contrary_evidence) <= 48
    assert max(len(item.evidence_refs) for item in result.metrics) <= 56
    print(
        encode(
            {
                "result_bytes": size,
                "details_bytes": len(result.details_json.encode()),
                "envelope_refs": len(result.envelope.evidence_refs),
                "contrary_refs": len(result.envelope.contrary_evidence),
                "gates": len(result.envelope.quality_gates),
            }
        )
    )
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        initialize_tables(connection)
        assert save_results(connection, (result,), 128)["result"] == 1
        serialized = connection.execute("SELECT payload FROM research.result").fetchone()[0]
        rebuilt = ResearchResult.from_json(serialized)
        assert rebuilt.to_json() == result.to_json()
        validate_result_consumption(rebuilt)
        assert fingerprints(record_rows(rebuilt)) == fingerprints(
            stored_rows(connection, large_market_inputs.run_id, 128)
        )


def test_market_all_roles_count_every_row_and_samples_resolve(
    large_market_inputs, large_market_result
):
    """全量输入、合格、排除、边界和停复牌分别计数；样本身份与字段均可回查且披露截断。"""
    audit = large_market_result.details["market_evidence_audit"]
    expected = {
        "daily_input": (1536, 0),
        "limit_input": (1024, 0),
        "suspension_input": (1280, 0),
        "eligible_daily": (1024, 5120),
        "excluded_daily": (512, 1536),
        "limit_used": (1024, 2048),
        "suspension_used": (1280, 3840),
        "suspension_contrary": (256, 768),
    }
    sources = {
        (item.source_api, item.raw["source_record_id"]): item
        for rows in large_market_inputs.rows_by_api.values()
        for item in rows
    }
    for role, (count, candidates) in expected.items():
        summary = audit[role]
        assert summary["total_rows"] == count
        assert summary["candidate_reference_count"] == candidates
        assert sum(summary["status_counts"].values()) == count
        assert len(summary["full_fingerprint"]) == 64
        assert summary["sample_count"] <= 8
        assert summary["truncated"] is True
        assert summary["sampling_policy"]["statistics_scope"] == "all_role_rows"
        for sample in summary["samples"]:
            assert (sample["source_api"], sample["source_record_id"]) in sources
    for reference in (
        large_market_result.envelope.evidence_refs + large_market_result.envelope.contrary_evidence
    ):
        source = sources[reference.source_api, reference.source_record_id]
        assert reference in source.evidence(*reference.fields)
    assert audit["suspension_used"]["status_counts"] == {"false": 1024, "true": 256}


def test_market_reversing_all_inputs_is_byte_identical(large_market_inputs, large_market_result):
    """同时反转实体和全部来源行；预期：真实市场主体与附加消费的完整JSON字节一致。"""
    inputs = large_market_inputs
    reversed_inputs = replace(
        inputs,
        config=replace(
            inputs.config,
            foundation=replace(
                inputs.config.foundation,
                entities=tuple(reversed(inputs.config.foundation.entities)),
            ),
        ),
        rows_by_api={api: tuple(reversed(rows)) for api, rows in inputs.rows_by_api.items()},
    )
    result = attach_consumption(reversed_inputs, PRODUCT_IDS[0], build_market(reversed_inputs))[0]
    assert result.to_json() == large_market_result.to_json()


@pytest.mark.parametrize(
    "api,role,field,value",
    [
        ("daily", "eligible_daily", "amount", 200.0),
        ("daily", "excluded_daily", "close", 11.0),
        ("stk_limit", "limit_used", "down_limit", 11.0),
        ("suspend_d", "suspension_used", "suspend_timing", "09:30-10:30"),
    ],
)
def test_unsampled_market_row_changes_full_fingerprint(
    large_market_inputs, large_market_result, api, role, field, value
):
    """改变各角色未抽样行的关键值；预期：全量指纹变化且计数完整，不因未展示而漏掉变化。"""
    inputs = large_market_inputs
    audit = large_market_result.details["market_evidence_audit"]
    sampled = {item["source_record_id"] for item in audit[role]["samples"]}
    rows = list(inputs.rows(api))
    index = next(
        i
        for i, item in enumerate(rows)
        if item.raw["source_record_id"] not in sampled
        and (1024 <= i < 1280 if role == "excluded_daily" else i < 1024)
    )
    original = rows[index]
    numeric = dict(original.numeric)
    if field in numeric:
        numeric[field] = replace(numeric[field], value=value)
    rows[index] = replace(original, raw=original.raw | {field: value}, numeric=numeric)
    changed = replace(inputs, rows_by_api=inputs.rows_by_api | {api: tuple(rows)})
    result = build_market(changed)[0]
    after = result.details["market_evidence_audit"][role]
    assert after["total_rows"] == audit[role]["total_rows"]
    assert after["full_fingerprint"] != audit[role]["full_fingerprint"]
    if role == "eligible_daily":
        assert result.metric("liquidity_concentration") == 104 / 1025
