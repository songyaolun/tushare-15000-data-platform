"""分区分派的缺元数据、合约歧义、旧旁证和显式现货关系边界。"""

from dataclasses import replace

import pytest
from test_cross_capacity_equivalence import (
    assert_same_results,
    canonical_inputs,
    partitioned_sample,
)
from test_cross_market_products import A as MARKET_A
from test_cross_market_products import B as MARKET_B
from test_cross_market_products import C as MARKET_C
from test_cross_market_products import commodity_sources, relation, source, value
from test_cross_market_products import inputs as market_inputs
from test_research_governance import A, B, inputs, row
from test_research_governance import relation as theme_relation

from tushare_entitlement_hub.cross_asset_research.registry import HANDLERS, resolve_handler


@pytest.mark.parametrize(
    "variant", ["missing_spec", "duplicate_spec", "duplicate_quote", "no_relation"]
)
def test_option_partitions_preserve_ambiguous_and_missing_group_members(variant: str) -> None:
    """场景：规格缺失、规格或行情歧义、标的关系缺失；预期：不拼出虚假完整 PCR。

    例如同一看涨合约有两份同时报价时，分区计算与原完整集合的未知结果完全一致。
    """
    common = {
        "maturity_date": "2026-12-31",
        "exercise_type": "E",
        "opt_type": "ETF",
        "opt_multiplier": "100",
        "opt_code": "OPT",
    }
    records = [
        source("opt_basic", "call-spec", entity=MARKET_A, raw={**common, "call_put": "C"}),
        source("opt_basic", "put-spec", entity=MARKET_B, raw={**common, "call_put": "P"}),
        source(
            "opt_daily",
            "call-quote",
            entity=MARKET_A,
            values={"vol": value(10, "quantity", "手"), "oi": value(30, "quantity", "手")},
        ),
        source(
            "opt_daily",
            "put-quote",
            entity=MARKET_B,
            values={"vol": value(20, "quantity", "手"), "oi": value(60, "quantity", "手")},
        ),
    ]
    if variant == "missing_spec":
        records.pop(0)
    elif variant == "duplicate_spec":
        records.append(
            source("opt_basic", "call-alt", entity=MARKET_A, raw={**common, "call_put": "P"})
        )
    elif variant == "duplicate_quote":
        records.append(
            source(
                "opt_daily",
                "call-alt",
                entity=MARKET_A,
                values={"vol": value(11, "quantity", "手")},
            )
        )
    relations = (
        ()
        if variant == "no_relation"
        else (
            relation("option_underlying", MARKET_A, MARKET_C),
            relation("option_underlying", MARKET_B, MARKET_C),
        )
    )
    sample = canonical_inputs(market_inputs(*records, relations=relations))
    expected = tuple(resolve_handler(HANDLERS["P11"])(sample))
    actual = partitioned_sample("P11", sample)
    assert_same_results(actual, expected)
    assert all(result.metric("put_call_vol_ratio") is None for result in actual)


def test_commodity_partition_keeps_old_segment_support_and_confirmed_spot_relation() -> None:
    """场景：普通与 VIP 主营同快照，以及现货与期货的显式关系；预期：两类旁证不丢失。

    例如旧 fina_mainbz 仍参与 VIP 核对，确认的现货期货关系仍生成完整比较结果。
    """
    records = list(commodity_sources())
    extra = source(
        "fut_daily",
        "linked-future",
        entity=MARKET_B,
        values={"close": value(100), "settle": value(110), "vol": value(10, "quantity", "手")},
    )
    records.append(extra)
    proof = source("daily", "old-relation-proof", day="2026-10-01")
    confirmed = replace(
        relation("spot_futures_comparable", MARKET_A, MARKET_B),
        evidence_refs=proof.evidence("ts_code"),
    )
    sample = canonical_inputs(market_inputs(*records, proof, relations=(confirmed,)))
    expected = tuple(resolve_handler(HANDLERS["P09"])(sample))
    actual = partitioned_sample("P09", sample)
    assert_same_results(actual, expected)
    assert any(
        {"fina_mainbz", "fina_mainbz_vip"}
        <= {reference.source_api for reference in result.envelope.evidence_refs}
        for result in actual
    )
    assert any(
        {"sge_daily", "fut_daily", "daily"}
        <= {reference.source_api for reference in result.envelope.evidence_refs}
        for result in actual
    )
    assert any(result.envelope.source_as_of.get("daily") == "2026-10-01" for result in actual)


def test_theme_partition_reuses_unique_prices_and_preserves_price_conflicts() -> None:
    """场景：旧层有唯一最新行情和同序冲突行情；预期：分区复用价格索引仍保留冲突语义。

    例如 A 的唯一行情可用，B 的两条同日冲突行情不能被任意挑选为可用价格。
    """
    members = [
        row("dc_member", code, raw={"con_code": code}, effective_from="2026-01-01T00:00:00Z")
        for code in ("S1", "S2")
    ]
    sample = canonical_inputs(
        inputs(
            *members,
            row(
                "daily",
                "price-a",
                entity=A,
                raw={"ts_code": "S1"},
                values={"pct_chg": (0.1, "fraction")},
            ),
            row(
                "daily",
                "price-b1",
                entity=B,
                raw={"ts_code": "S2"},
                values={"pct_chg": (0.1, "fraction")},
            ),
            row(
                "daily",
                "price-b2",
                entity=B,
                raw={"ts_code": "S2"},
                values={"pct_chg": (0.2, "fraction")},
            ),
            entities=(A, B),
            relations=tuple(
                theme_relation(from_entity_id=A, to_entity_id=entity, relation_type="theme_member")
                for entity in (A, B)
            ),
        )
    )
    expected = tuple(resolve_handler(HANDLERS["P02"])(sample))
    actual = partitioned_sample("P02", sample)
    assert_same_results(actual, expected)
    assert actual[0].metric("dc_member_effective_members") == 2
    assert actual[0].metric("dc_member_member_price_coverage") == 0.5
    assert actual[0].metric("dc_member_equal_weight_return") is None


@pytest.mark.parametrize("variant", ["fund_profile_only", "no_index", "no_manager"])
def test_fund_partition_does_not_invent_missing_profile_links(variant: str) -> None:
    """场景：仅有基金档案，或 ETF 未声明指数/管理人；预期：保留未知关系，完整结果等价。

    例如 ETF 没有 index_code 时不能把其他指数成分放进该基金分区。
    """
    if variant == "fund_profile_only":
        records = [row("fund_basic", raw={"ts_code": "ETF", "market": "E"})]
    else:
        fields = {"ts_code": "ETF", "mgr_name": "管理人", "index_code": "IDX"}
        fields.pop("index_code" if variant == "no_index" else "mgr_name")
        records = [row("etf_basic", raw=fields)]
    records.extend(
        [
            row(
                "fund_company",
                "company",
                entity=B,
                raw={"name": "管理人", "org_code": "ORG", "credit_code": "CREDIT"},
            ),
            row(
                "index_weight",
                "index",
                entity=B,
                raw={"index_code": "IDX", "con_code": "A"},
                values={"weight": (1, "fraction")},
            ),
        ]
    )
    sample = canonical_inputs(inputs(*records, entities=(A, B)))
    expected = tuple(resolve_handler(HANDLERS["P08"])(sample))
    actual = partitioned_sample("P08", sample)
    assert_same_results(actual, expected)
    assert len(actual) == 1
