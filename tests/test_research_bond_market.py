"""P06市场债券的全量消费、有界载荷与确定性合成回归。"""

import hashlib
from dataclasses import replace

import duckdb
import pytest
from test_research_disclosure_consumers import source
from test_research_governance import DAY, A, inputs, relation

from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    count_consumption,
    summarize_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.context_aggregation import row_fingerprint
from tushare_entitlement_hub.equity_research.context_digest import MultisetDigest
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY, ResearchResult
from tushare_entitlement_hub.equity_research.storage import initialize_tables, save_results
from tushare_entitlement_hub.product_foundation.contracts import EvidenceRef
from tushare_entitlement_hub.store import encode

COUNTS = {
    "cb_basic": 1200,
    "cb_call": 24,
    "cb_daily": 320,
    "cb_factor_pro": 320,
    "cb_issue": 12,
    "cb_rate": 10,
    "cb_rating": 10,
    "cb_share": 240,
    "top10_cb_holders": 60,
}


def bond_market_inputs(counts=None):
    """生成覆盖正式行数的九类债券来源，逐债券有明确关系及同日核价依据。

    例：1200个cb_basic及320组双源报价；持有人每债券10行，不将不同债券合并计算集中度。
    counts可缩至9行边界测试；所有值、日期与证据身份均为确定性合成数据。
    """
    counts = COUNTS if counts is None else counts
    raw_fields = {
        "cb_basic": {
            "cb_type": "CB",
            "stk_code": "A",
            "conv_start_date": "20260101",
            "conv_end_date": "20271231",
        },
        "cb_call": {"call_type": "不赎回", "is_call": "N", "ann_date": DAY, "call_date": DAY},
        "cb_daily": {},
        "cb_factor_pro": {},
        "cb_issue": {"ann_date": DAY, "res_ann_date": DAY},
        "cb_rate": {"rate_start_date": "20260101", "rate_end_date": "20261231"},
        "cb_rating": {
            "ann_date": DAY,
            "rating_date": DAY,
            "rating_com_name": "合成机构",
            "rating_type": "债项",
            "rating": "AA",
            "rating_outlook": "稳定",
        },
        "cb_share": {"publish_date": DAY, "end_date": "20260630"},
        "top10_cb_holders": {"ann_date": DAY, "end_date": "20260630"},
    }
    numeric = {
        "cb_basic": {"par": (100, "元"), "conv_price": (10, "元"), "remain_size": (1000, "元")},
        "cb_call": {"call_amount": (100, "元")},
        "cb_daily": {
            "close": (110, "元"),
            "cb_value": (100, "元"),
            "cb_over_rate": (0.1, "ratio"),
            "vol": (10, "手"),
            "amount": (1100, "元"),
        },
        "cb_factor_pro": {"close": (110, "元"), "vol": (10, "手"), "amount": (1100, "元")},
        "cb_issue": {"plan_issue_size": (1000, "元"), "issue_size": (900, "元")},
        "cb_rate": {"coupon_rate": (0.02, "ratio"), "rate_freq": (2, "次/年")},
        "cb_rating": {},
        "cb_share": {
            "convert_price": (10, "元"),
            "convert_vol": (10, "股"),
            "acc_convert_vol": (100, "股"),
            "remain_size": (900, "元"),
        },
        "top10_cb_holders": {"hold_ratio": (0.05, "ratio")},
    }
    rows = []
    for api, count in counts.items():
        for index in range(count):
            bond_index = index // 10 if api == "top10_cb_holders" else index
            code = f"CB{bond_index:04}"
            raw = raw_fields[api] | {"ts_code": code}
            if api == "top10_cb_holders":
                raw.update(holder_rank=index % 10 + 1, holder_name=f"合成持有人{index % 10}")
            rows.append(
                source(
                    api,
                    hashlib.sha256(f"{api}/{index}".encode()).hexdigest(),
                    entity=f"cn:security:{code}:life",
                    raw=raw,
                    values=numeric[api],
                )
            )
    terms = [row for row in rows if row.source_api == "cb_basic"]
    relations = tuple(
        relation(from_entity_id=row.entity_id, evidence_refs=row.evidence("cb_type"))
        for row in terms
    )
    stock = source(
        "daily", "stock", values={"close": (10, "元"), "vol": (100, "手"), "amount": (1000, "元")}
    )
    return inputs(*rows, stock, entities=(A,), relations=relations)


@pytest.mark.parametrize(
    ("source_counts", "unverified_quotes"),
    [
        pytest.param(COUNTS, False, id="all_certified_above_formal_counts"),
        pytest.param(
            {
                "cb_basic": 1171,
                "cb_call": 22,
                "cb_daily": 316,
                "cb_factor_pro": 316,
                "cb_issue": 9,
                "cb_rate": 5,
                "cb_rating": 5,
                "cb_share": 224,
                "top10_cb_holders": 40,
            },
            True,
            id="formal_counts_with_unverified_quotes",
        ),
    ],
)
def test_formal_scale_bond_market_result_is_bounded_and_rebuildable(
    tmp_path, source_counts, unverified_quotes
):
    """九来源全认证或同正式行数且两源报价未认证时均小于1MiB，全量计数及真实状态可重建。

    未认证场景的cb_basic仍observed，两组316行报价保持unknown；五行来源保留精确观察。
    两场景均验证全部保留行指纹、样本回查、逆序字节一致和512MB下保存重建。
    """
    context = bond_market_inputs(source_counts)
    quote_apis = {"cb_daily", "cb_factor_pro"}
    if unverified_quotes:
        context = replace(
            context,
            rows_by_api={
                api: tuple(replace(row, numeric={}) for row in rows)
                if api in quote_apis
                else rows
                for api, rows in context.rows_by_api.items()
            },
        )
    result = attach_consumption(context, PRODUCT_ALIASES["P06"], ())[0]
    payload_bytes = len(result.to_json().encode())
    assert payload_bytes < 1024 * 1024
    assert result.envelope.entity_id == MARKET_ENTITY
    records = {item["api"]: item for item in result.details["consumption"]}
    for api, count in source_counts.items():
        record = records[api]
        expected_status = "unknown" if unverified_quotes and api in quote_apis else "observed"
        assert record["status"] == expected_status
        assert record["retained_row_count"] == count
        assert len(encode(record).encode()) < 128 * 1024
        if expected_status == "unknown":
            assert "bond_quote_peer_date_or_unit_unverified" in record["reason_codes"]
            assert all(
                quality
                == {
                    "present_rows": count,
                    "missing_rows": 0,
                    "certified_rows": 0,
                    "nonfinite_rows": 0,
                }
                for quality in record["field_quality"].values()
            )
        if count <= 8:
            assert "aggregation" not in record["facts"]
            assert len(record["facts"]["observations"]) == count
            continue
        summary = record["facts"]["aggregation"]
        assert summary["total_rows"] == count
        assert summary["group_count"] == (count // 10 if api == "top10_cb_holders" else count)
        assert sum(summary["rows_by_group_status"].values()) == count
        assert summary["status_counts"] == {expected_status: summary["group_count"]}
        assert len(record["evidence_refs"]) <= 10 and len(record["gates"]) == 2
        assert summary["sample_count"] == len(record["evidence_refs"])
        assert summary["truncated"] is True
        assert len(summary["full_fingerprint"]) == 64
        complete = MultisetDigest()
        for source_row in context.rows(api):
            complete.add(row_fingerprint(source_row, tuple(summary["audited_fields"])))
        assert summary["rows_fingerprint"] == complete.hexdigest()
        assert summary["usable_rows"] == count and summary["rejected_rows"] == 0
        assert all(value["present_rows"] == count for value in record["field_quality"].values())
        assert summary["sampling_policy"]["per_stratum_limit"] == 1
        for evidence in record["evidence_refs"]:
            reference = EvidenceRef.from_dict(evidence)
            found = next(
                row
                for row in context.rows(reference.source_api)
                if row.availability.source_record_id == reference.source_record_id
            )
            assert reference in found.evidence(*reference.fields)
    assert (
        records["top10_cb_holders"]["facts"]["aggregation"]["business_statistics"][
            "distinct_bond_period_holder_pairs"
        ]
        == source_counts["top10_cb_holders"]
    )
    validate_result_consumption(result)
    counts = {}
    count_consumption(counts, result)
    consumption = summarize_consumption(counts, [PRODUCT_ALIASES["P06"]])
    assert consumption["observed_apis"] == len(source_counts) - (2 if unverified_quotes else 0)
    assert consumption["executed_apis"] == consumption["assigned_apis"]
    assert sum(records[api]["retained_row_count"] for api in source_counts) == sum(
        source_counts.values()
    )
    reversed_context = replace(
        context,
        rows_by_api={
            api: tuple(reversed(rows)) for api, rows in reversed(context.rows_by_api.items())
        },
    )
    assert (
        attach_consumption(reversed_context, PRODUCT_ALIASES["P06"], ())[0].to_json()
        == result.to_json()
    )
    with duckdb.connect(
        str(tmp_path / "bonds.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        assert save_results(connection, (result,), 128)["result"] == 1
        restored = ResearchResult.from_json(
            connection.execute("SELECT payload FROM research.result").fetchone()[0]
        )
    assert restored.to_json() == result.to_json()
    validate_result_consumption(restored)
    print(
        f"P06市场债券：{sum(source_counts.values())}行，最大结果{payload_bytes}字节，信封引用{len(result.envelope.evidence_refs)}，门禁{len(result.envelope.quality_gates)}"
    )
