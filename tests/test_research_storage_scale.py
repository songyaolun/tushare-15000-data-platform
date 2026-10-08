"""保存/完整重建的规模回归和主键检查的正式数量级SQL实验。"""

from collections import Counter
from dataclasses import replace
from time import perf_counter

import duckdb
import pytest
from test_research_model_reuse import rich_research_results

from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    count_consumption,
    summarize_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import (
    CONSUMPTION_VERSION,
    registry_fingerprint,
)
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    PRODUCT_IDS,
    RULE_VERSION,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.persistence_runtime import PersistenceProgress
from tushare_entitlement_hub.equity_research.projection import accumulate_records, project_result
from tushare_entitlement_hub.equity_research.storage import (
    RESEARCH_TABLES,
    _verify_keys,
    fingerprints,
    initialize_tables,
    record_rows,
    run_identity,
    save_results,
    stored_rows,
    verify_run,
)
from tushare_entitlement_hub.product_foundation import model


def test_two_hundred_rich_results_save_rebuild_once_and_keep_all_projections(tmp_path, monkeypatch):
    """200只证券含真实消费和共享证据，在512MB下完整保存重建；根只读/序列化一次且无投影丢失。"""
    inputs, base = rich_research_results(200)
    requested, products = (PRODUCT_IDS[4],), list(PRODUCT_IDS[1:5])
    run_id = run_identity("synthetic-reference", inputs.analysis_as_of, inputs.config, requested)
    inputs = replace(inputs, run_id=run_id)
    base = tuple(
        replace(result, envelope=replace(result.envelope, run_id=run_id)) for result in base
    )
    results = tuple(
        item
        for product in products
        for item in attach_consumption(
            inputs,
            product,
            base if product == PRODUCT_IDS[4] else (make_result(inputs, product, MARKET_ENTITY),),
        )
    )
    states, consumed = {}, {}
    for product in products:
        states[product] = dict(
            Counter(
                result.envelope.state.value
                for result in results
                if result.envelope.product_id == product
            )
        )
    for result in results:
        count_consumption(consumed, result)
    summary = {
        "schema_version": RULE_VERSION,
        "run_id": run_id,
        "analysis_as_of": inputs.analysis_as_of,
        "mode": "observed_research",
        "historical_pit_certified": False,
        "status": "incomplete",
        "requested_products": list(requested),
        "products": products,
        "product_states": states,
        "processing": {
            "scanned_rows": 200,
            "retained_rows": 200,
            "excluded_rows": 0,
            "discarded_rows": 0,
        },
        "consumption": summarize_consumption(consumed, products),
        "result_count": len(results),
    }
    calls = Counter()
    original_json, original_parse = model.Model.to_json, ResearchResult.from_json.__func__

    def serialize(value):
        """计数仍调用完整公开序列化；例：save只为每条结果调用根to_json一次。"""
        calls[type(value).__name__] += 1
        return original_json(value)

    def parse(cls, payload):
        """计数完整模型反序列化次数；例：203条结果重建恰好解析203次。"""
        calls["from_json"] += 1
        return original_parse(cls, payload)

    class ReadProbe:
        """计数根结果读取和每次预取行数，其余SQL仍在同一真实事务中执行。"""

        def __init__(self, connection):
            """保存连接及零计数；例：一次重建只需一次result载荷查询。"""
            self.connection = connection
            self.queries = self.maximum_fetch = self.rows = 0

        def execute(self, query, parameters):
            """拦截两种根载荷查询形态，覆盖旧版二次读取和新版单次读取。"""
            cursor = self.connection.execute(query, parameters)
            if query.startswith(
                (
                    "SELECT payload FROM research.result",
                    'SELECT record_id,run_id,result_id,payload FROM research."result"',
                )
            ):
                self.queries += 1
                return self
            return cursor

        def fetchmany(self, size):
            """记录实际预取大小并透传；例：512配置下每次仍只读取一条宽结果。"""
            self.maximum_fetch = max(self.maximum_fetch, size)
            batch = self.connection.fetchmany(size)
            self.rows += len(batch)
            return batch

    with duckdb.connect(
        str(tmp_path / "scale.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        with monkeypatch.context() as patch:
            patch.setattr(model.Model, "to_json", serialize)
            start = perf_counter()
            summary["counts"] = save_results(connection, iter(results), 512)
            saved_seconds = perf_counter() - start
            assert calls == {"ResearchResult": len(results)}
        metadata = {
            "run_id": run_id,
            "rule_version": RULE_VERSION,
            "consumption_version": CONSUMPTION_VERSION,
            "consumption_registry_fingerprint": registry_fingerprint(),
            "reference_sha256": "synthetic-reference",
            "analysis_as_of": inputs.analysis_as_of,
            "config": inputs.config.to_dict(),
            "selection": list(requested),
            "products": products,
            "summary": summary,
        }
        with monkeypatch.context() as patch:
            patch.setattr(ResearchResult, "from_json", classmethod(parse))
            start = perf_counter()
            probe = ReadProbe(connection)
            assert verify_run(probe, metadata) == summary
            verified_seconds = perf_counter() - start
        assert calls["from_json"] == len(results)
        assert probe.queries == 1 and probe.maximum_fetch == 1 and probe.rows == len(results)
        actual = fingerprints(stored_rows(connection, run_id, 512))
        expected = fingerprints(row for result in results for row in record_rows(result))
        assert (
            actual
            == expected
            == accumulate_records(
                row for result in reversed(results) for row in project_result(result)
            )
        )
        connection.execute("COMMIT")
        assert verify_run(connection, metadata) == summary
    print(
        f"完整规模回归：{len(results)}结果，{sum(summary['counts'].values())}物理行，保存{saved_seconds:.3f}s，重建{verified_seconds:.3f}s"
    )


def test_thirty_thousand_parent_keys_and_million_child_rows_keep_constraints(tmp_path):
    """3万父键、140万子表行在512MB事务内完成全引用检查，已有PK不再全表去重，孤儿仍拒绝。"""

    class RecordingConnection:
        """只记录SQL形态并使用真实数据库，不模拟约束或返回值。"""

        def __init__(self, connection):
            """绑定真实连接；例：初始queries为空，随后记录完整性查询。"""
            self.connection, self.queries = connection, []

        def execute(self, query, parameters):
            """记录并直接执行查询；例：主键视图查询使用真实约束信息。"""
            self.queries.append(query)
            return self.connection.execute(query, parameters)

    with duckdb.connect(
        str(tmp_path / "keys.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "INSERT INTO research.result SELECT sha256(i::VARCHAR),'run',sha256(i::VARCHAR),'{}' "
            "FROM range(30000) t(i)"
        )
        for table in ("feature", "gate", "evidence", "lineage"):
            connection.execute(
                f"INSERT INTO research.{table} SELECT sha256(? || i::VARCHAR),"
                "'run',sha256((i%30000)::VARCHAR),'{}' FROM range(350000) t(i)",
                [table],
            )
        recorded = RecordingConnection(connection)
        _verify_keys(recorded, "run", PersistenceProgress("verify", None, None))
        assert not any("count(DISTINCT record_id)" in query for query in recorded.queries)
        assert sum(
            "LEFT JOIN research.result parent ON child" in query for query in recorded.queries
        ) == len(RESEARCH_TABLES)
        connection.execute(
            "UPDATE research.feature SET result_id='absent' WHERE record_id="
            "(SELECT record_id FROM research.feature LIMIT 1)"
        )
        with pytest.raises(ValueError, match="research_result_reference_missing"):
            _verify_keys(connection, "run", PersistenceProgress("verify", None, None))
        connection.execute("ROLLBACK")


@pytest.mark.parametrize("duplicate", [False, True])
def test_other_attached_database_cannot_supply_missing_local_primary_key(tmp_path, duplicate):
    """其他库同名表有主键时，当前无主键表仍完整检查：唯一行可通过，重复行必须失败。"""
    with duckdb.connect(
        str(tmp_path / "local.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        initialize_tables(connection)
        connection.execute("INSERT INTO research.result VALUES ('parent','run','parent','{}')")
        connection.execute("DROP TABLE research.feature")
        connection.execute(
            "CREATE TABLE research.feature(record_id VARCHAR,run_id VARCHAR,"
            "result_id VARCHAR,payload JSON)"
        )
        connection.execute(
            "INSERT INTO research.feature VALUES ('dup','run','parent','{}'),"
            "(?,'run','parent','{}')",
            ["dup" if duplicate else "unique"],
        )
        connection.execute("ATTACH ':memory:' AS other")
        connection.execute("CREATE SCHEMA other.research")
        connection.execute("CREATE TABLE other.research.feature(record_id VARCHAR PRIMARY KEY)")
        if duplicate:
            with pytest.raises(ValueError, match="research_primary_key_invalid"):
                _verify_keys(connection, "run", PersistenceProgress("verify", None, None))
        else:
            _verify_keys(connection, "run", PersistenceProgress("verify", None, None))


def test_historical_runs_are_rebuilt_but_only_current_sources_are_rechecked(tmp_path, monkeypatch):
    """两个不同截止的运行都重建，来源回查只针对当前参考指纹；历史结果篡改仍不可通过。"""
    from research_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research import storage
    from tushare_entitlement_hub.equity_research.pipeline import run_research

    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "history.duckdb"
    first = run_research(reference, output, AS_OF, config, ("P01",))
    second = run_research(reference, output, "2026-10-04T12:00:00Z", config, ("P01",))
    calls = []
    original = storage.verify_source_evidence

    def checked(reference_path, products, run_id, config, **kwargs):
        """记录回查的运行键并调用真实来源校验；例：当前运行出现一次，历史运行不读新来源。"""
        calls.append(run_id)
        return original(reference_path, products, run_id, config, **kwargs)

    monkeypatch.setattr(storage, "verify_source_evidence", checked)
    verified = storage.validate_research(reference, output)
    assert verified["valid"] and len(verified["runs"]) == 2
    assert calls == [second["run_id"]]
    with duckdb.connect(str(output), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        connection.execute(
            "UPDATE research.result SET payload='{}' WHERE run_id=?", [first["run_id"]]
        )
    assert not storage.validate_research(reference, output)["valid"]
