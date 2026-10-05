"""分区复用全局实体索引的精确计算次数、对象共享及完整结果等价回归。"""

from dataclasses import FrozenInstanceError, replace
from functools import cached_property

import pytest
from test_cross_capacity_equivalence import assert_same_results
from test_cross_market_products import source, value
from test_research_governance import BOND, A, B, inputs, relation, row

from tushare_entitlement_hub.cross_asset_research.partitioned_execution import partitioned_results
from tushare_entitlement_hub.cross_asset_research.partitioned_inputs import InputSpool
from tushare_entitlement_hub.cross_asset_research.registry import HANDLERS, resolve_handler
from tushare_entitlement_hub.equity_research.models import ResearchInputs

INDEX = "cn:security:INDEX:life"


def scope_sample(*, relation_exclusion=False, extra_entities=()):
    """构造两只股票和被旧债券、指数规则排除的登记实体，保留全部观察范围。

    例如 A、B、BOND、INDEX 均显式登记，BOND 和 INDEX 由完整模板规则排除。
    """
    specifications = [
        source(
            "opt_basic",
            code,
            entity=entity,
            raw={
                "ts_code": code,
                "opt_code": "UNDERLYING",
                "opt_type": "ETF",
                "call_put": side,
                "exercise_type": "E",
                "exercise_price": 2,
                "maturity_date": "2026-12-31",
                "opt_multiplier": "100",
            },
        )
        for code, entity, side in (("CALL", A, "C"), ("PUT", B, "P"))
    ]
    old_rows = [row("index_daily", "old-index", entity=INDEX)]
    if not relation_exclusion:
        old_rows.append(row("cb_basic", "old-bond", entity=BOND))
    return inputs(
        *specifications,
        *old_rows,
        source(
            "opt_daily",
            "quote",
            entity=A,
            values={"vol": value(10, "quantity", "手"), "oi": value(20, "quantity", "手")},
        ),
        entities=(A, B, BOND, INDEX, *extra_entities),
        relations=(relation(),) if relation_exclusion else (),
    )


def test_partition_indexes_are_computed_once_and_shared_across_all_partitions(monkeypatch):
    """场景：多个实体及空分区反复读取全局索引；预期：模板两索引各计算一次，分区共享对象。

    保留 32 个未参与当前分区的股票范围，不复制全局集合，不依赖耗时判断复杂度。
    """
    extra = tuple(f"cn:security:EXTRA{index}:life" for index in range(32))
    sample = scope_sample(extra_entities=extra)
    calls = {"entity_ids": [], "entity_kinds": []}
    for name in calls:
        original = getattr(ResearchInputs, name).func

        def counted(instance, original=original, name=name):
            """登记底层索引的真实计算对象，例如所有分区共享同一个模板计算。"""
            calls[name].append(instance)
            return original(instance)

        descriptor = cached_property(counted)
        descriptor.__set_name__(ResearchInputs, name)
        monkeypatch.setattr(ResearchInputs, name, descriptor)
    with InputSpool() as spool:
        for record in sample.rows("opt_basic"):
            spool.append(record)
        parts = [
            spool.materialize(sample, ("opt_basic",), entity=entity)
            for entity in (A, B, "absent", A, B, "absent", A, B)
        ]
        actual = [(part.entity_ids, part.entity_kinds) for part in parts]
        expected = (sample.entity_ids, sample.entity_kinds)
        assert len(calls["entity_ids"]) == len(calls["entity_kinds"]) == 1
        assert calls["entity_ids"][0] is calls["entity_kinds"][0] is sample
        assert expected[0] == tuple(sorted((A, B, *extra)))
        assert len(expected[1]) == 36
        assert all(ids is expected[0] and kinds is expected[1] for ids, kinds in actual)
        assert parts[2].rows("opt_basic") == ()
        assert parts[2].entity_ids == expected[0]
        with pytest.raises(FrozenInstanceError):
            parts[0].entity_ids = ()


@pytest.mark.parametrize(
    "relation_exclusion", [False, True], ids=["legacy_source_rows", "confirmed_bond_relation"]
)
def test_partition_results_preserve_global_bond_and_index_exclusion(relation_exclusion):
    """场景：完整输入含旧债券、指数排除依据；预期：分区与原计算完整JSON一致，观察股票数为2。

    例如 P11 分区自身没有 cb_basic/index_daily 行，仍不得把这两个登记实体计入覆盖范围。
    """
    sample = scope_sample(relation_exclusion=relation_exclusion)
    template = replace(
        sample,
        rows_by_api={
            api: records
            for api, records in sample.rows_by_api.items()
            if api not in {"opt_basic", "opt_daily"}
        },
    )
    handler = resolve_handler(HANDLERS["P11"])
    expected = tuple(handler(sample))
    with InputSpool() as spool:
        for api in ("opt_basic", "opt_daily"):
            for record in reversed(sample.rows(api)):
                spool.append(record)
        actual = tuple(partitioned_results("P11", spool, template, handler))
    assert len(actual) == len(expected) == 3
    assert all(result.envelope.coverage.observed_count == 2 for result in expected)
    assert_same_results(actual, expected)
    assert all(result.envelope.coverage.observed_count == 2 for result in actual)
