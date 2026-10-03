"""6000实体×3父结果的真实P13构建、消费附加和512MB落库回归，不以墙钟设门槛。"""

from collections import Counter
from dataclasses import replace

import duckdb
from test_research_candidates import AS_OF, research_row

from tushare_entitlement_hub.equity_research import event_parents, events
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.models import (
    PRODUCT_IDS,
    ResearchConfig,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.storage import (
    MAXIMUM_RESULT_JSON_BYTES,
    initialize_tables,
    save_results,
)
from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
from tushare_entitlement_hub.product_foundation.contracts import Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity
from tushare_entitlement_hub.store import encode


class CountedParents(tuple):
    """记录父输入实际遍历次数；例如18000个唯一父只应被输入索引遍历18000次。"""

    visits = 0

    def __iter__(self):
        """逐项透传真实父对象并计数；不替换构建函数或省略父载荷。"""
        for parent in super().__iter__():
            self.visits += 1
            yield parent


def scale_inputs_and_parents(count=6000):
    """生成显式6000实体、6000合成行情行和18000父结果，每实体三个独立来源字段引用。

    P04现金覆盖1.2、P06解禁.01、P07观察值.5；每97个实体留一项硬风险与反证。
    每父还保留未直接用于论点的明细，验证摘要没有抽样或只覆盖两项指标。
    """
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind="security", local_id=f"S{index:04}", lifecycle_id="synthetic"
            ),
            name=f"合成证券{index}",
        )
        for index in range(count)
    )
    rows = tuple(
        research_row(
            entity.identity.entity_id,
            "daily",
            {"close": 10, "vol": 100, "amount": 1000},
            record=f"synthetic-{index}",
        )
        for index, entity in enumerate(entities)
    )
    inputs = ResearchInputs(
        analysis_as_of=AS_OF,
        run_id="event-scale",
        config=ResearchConfig(foundation=FoundationBindings(entities=entities)),
        rows_by_api={"daily": rows},
    )
    parents = []
    for index, row in enumerate(rows):
        for product, field, name, value in zip(
            PRODUCT_IDS[1:4],
            ("close", "vol", "amount"),
            ("cash_cover", "unlock_30d", "flow_observation"),
            (1.2, 0.01, 0.5),
            strict=True,
        ):
            references = row.evidence(field)
            blocked = product == PRODUCT_IDS[3] and index % 97 == 0
            parents.append(
                make_result(
                    inputs,
                    product,
                    row.entity_id,
                    metrics=(metric(name, value, "ratio", references),),
                    references=references,
                    gates=(
                        gate(
                            "verified",
                            Truth.FALSE if blocked else Truth.TRUE,
                            "synthetic_risk",
                            references=references,
                        ),
                    ),
                    contrary=references if blocked else (),
                    details={"unselected_detail": index, "product": product},
                )
            )
    return inputs, tuple(parents)


def test_six_thousand_entities_build_attach_and_persist_without_quadratic_parent_work(
    monkeypatch, tmp_path
):
    """真实18000父输入产生6000个bundle；父遍历与实体候选为线性，唯一父to_json调用为零。

    逆序并重复输入逐字节幂等；最后一个父的未展示字段变化改变摘要。全部6000个结果
    实际附加5项P13消费、在512MB DuckDB保存并逐条重建，全部父血缘存在且输出未超8MiB。
    """
    inputs, parents = scale_inputs_and_parents()
    inputs = replace(inputs)
    parent_objects = {id(parent) for parent in parents}
    counts = Counter()
    original_json = ResearchResult.to_json
    original_index = events.index_parents
    original_metrics = events._parent_metrics
    original_fingerprint = event_parents.parent_fingerprint
    original_entity_id = EntityIdentity.entity_id.fget

    class CountedIndex(dict):
        """透传真实实体索引，只记录按键查询；review若迭代全索引立即失败。"""

        def get(self, key, default=None):
            """查询一次对应一个事件，不访问其他实体；6000个bundle预期6000次。"""
            counts["entity_lookups"] += 1
            return super().get(key, default)

        def __iter__(self):
            """review不得把索引重新当全量父序列过滤；任意全量迭代立即报错。"""
            raise AssertionError("审阅全量扫描了实体索引")

    def indexed(context, candidates):
        """沿用真实索引器，只为返回字典增加访问计数，不替换去重或时点语义。"""
        return CountedIndex(original_index(context, candidates))

    def checked_json(result):
        """唯一父结果禁止完整序列化；新建P13结果仍执行原序列化和校验。"""
        if id(result) in parent_objects:
            counts["parent_to_json"] += 1
            raise AssertionError("唯一父结果调用了to_json")
        return original_json(result)

    def checked_metrics(candidates):
        """核对每次审阅只有同实体三个父；累计18000个候选，而非18000×18000。"""
        counts["review_parent_candidates"] += len(candidates)
        assert len(candidates) == 3
        assert len({parent.envelope.entity_id for parent in candidates}) == 1
        return original_metrics(candidates)

    def fingerprint(parent):
        """计数全量流式父摘要，每个唯一有效父每轮只计算一次。"""
        counts["parent_fingerprints"] += 1
        return original_fingerprint(parent)

    def entity_id(identity):
        """对整个构建计数实体身份访问，证明6000条结果不会各自重建6000实体的集合。"""
        counts["entity_identity_reads"] += 1
        return original_entity_id(identity)

    source = CountedParents(parents)
    with monkeypatch.context() as patch:
        patch.setattr(ResearchResult, "to_json", checked_json)
        patch.setattr(events, "index_parents", indexed)
        patch.setattr(events, "_parent_metrics", checked_metrics)
        patch.setattr(event_parents, "parent_fingerprint", fingerprint)
        patch.setattr(EntityIdentity, "entity_id", property(entity_id))
        results = events.build_events(inputs, source)
        assert source.visits == 18000
        assert len(results) == 6000
        assert counts == {
            "entity_lookups": 6000,
            "review_parent_candidates": 18000,
            "parent_fingerprints": 18000,
            "entity_identity_reads": 12000,
        }
        repeated_source = CountedParents(tuple(reversed(parents)) * 2)
        repeated = events.build_events(inputs, repeated_source)
        assert repeated_source.visits == 36000
        assert len(repeated) == len(results)
        assert all(
            left.to_json() == right.to_json() for left, right in zip(results, repeated, strict=True)
        )
        assert counts == {
            "entity_lookups": 12000,
            "review_parent_candidates": 36000,
            "parent_fingerprints": 36000,
            "entity_identity_reads": 12000,
        }
    del repeated
    last = next(
        result for result in results if result.envelope.entity_id == parents[-1].envelope.entity_id
    )
    changed = replace(parents[-1], details_json='{"unselected_detail":"changed"}')
    revised = events.build_events(inputs, parents[-3:-1] + (changed,))[0]
    assert revised.parent_result_ids == last.parent_result_ids
    assert revised.details["upstream_fingerprint"] != last.details["upstream_fingerprint"]
    assert all(result.details["upstream_parent_count"] == 3 for result in results)
    assert all(result.details["event_type"] == "upstream_bundle" for result in results)

    attached = attach_consumption(inputs, PRODUCT_IDS[5], results)
    assert len(attached) == 6000
    maximum_bytes = max(len(result.to_json().encode()) for result in attached)
    assert maximum_bytes < 64 * 1024 < MAXIMUM_RESULT_JSON_BYTES
    assert max(len(result.envelope.evidence_refs) for result in attached) == 3
    assert max(len(result.envelope.quality_gates) for result in attached) <= 14
    expected = {result.envelope.result_id: result for result in attached}
    with duckdb.connect(
        str(tmp_path / "event-scale.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        assert save_results(connection, parents, 128)["result"] == 18000
        assert save_results(connection, attached, 128)["result"] == 6000
        assert connection.execute("SELECT count(*) FROM research.result").fetchone() == (24000,)
        assert connection.execute(
            "SELECT count(*) FROM research.lineage WHERE (payload->>'parent_result_id') IS NOT NULL"
        ).fetchone() == (18000,)
        assert connection.execute(
            "SELECT count(*) FROM research.lineage edge LEFT JOIN research.result parent "
            "ON (edge.payload->>'parent_result_id')=parent.record_id "
            "WHERE (edge.payload->>'parent_result_id') IS NOT NULL AND parent.record_id IS NULL"
        ).fetchone() == (0,)
        cursor = connection.execute(
            "SELECT record_id,payload FROM research.result "
            "WHERE (payload->'envelope'->>'product_id')=?",
            [PRODUCT_IDS[5]],
        )
        rebuilt_count = 0
        while batch := cursor.fetchmany(128):
            for result_id, payload in batch:
                rebuilt = ResearchResult.from_json(payload)
                validate_result_consumption(rebuilt)
                assert len(rebuilt.details["consumption"]) == 5
                assert all(record["executed"] for record in rebuilt.details["consumption"])
                assert rebuilt.to_json() == expected[result_id].to_json()
                rebuilt_count += 1
        assert rebuilt_count == 6000
    print(
        "P13高基数验收 "
        + encode(
            {
                "entities": 6000,
                "unique_parents": 18000,
                "parent_input_visits": source.visits,
                "entity_lookups_per_build": 6000,
                "review_parent_candidates_per_build": 18000,
                "unique_parent_to_json_calls": counts["parent_to_json"],
                "entity_identity_reads_across_two_builds": counts["entity_identity_reads"],
                "bundle_count": len(results),
                "maximum_result_bytes": maximum_bytes,
                "maximum_refs": 3,
                "maximum_gates": max(len(result.envelope.quality_gates) for result in attached),
                "stored_results": 24000,
                "rebuilt_bundles": rebuilt_count,
            }
        )
    )
