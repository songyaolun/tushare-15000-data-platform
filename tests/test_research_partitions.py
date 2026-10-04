"""有界投影分段的透明读取、全局唯一键和视图/目录/键完整性。"""

import json
from hashlib import sha256

import duckdb
import pytest

from tushare_entitlement_hub.equity_research import partitions, storage
from tushare_entitlement_hub.equity_research.persistence_runtime import PersistenceProgress


@pytest.fixture
def segmented(tmp_path, monkeypatch):
    """构造小预算分段；例：12条原值相同宽度的记录产生多段，保留相同四列逻辑表。"""
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_BYTES", 1024)
    with duckdb.connect(
        str(tmp_path / "parts.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        storage.initialize_tables(connection)
        writer = partitions.PartitionWriter(connection)
        expected = []
        connection.execute("BEGIN")
        for number in range(12):
            key = sha256(str(number).encode()).hexdigest()
            payload = json.dumps({"synthetic": "x" * 300, "number": number}, separators=(",", ":"))
            writer.insert("result", [[key], ["run"], [key], [payload]], len(payload))
            expected.append((key, "run", key, payload))
        connection.execute("COMMIT")
        yield connection, expected


def test_partition_rows_keep_values_keys_and_deterministic_reading(segmented):
    """多段读取保留所有四列原值，重建键与指纹一致；新写者封存旧段，跨段重复键仍立即失败。"""
    connection, expected = segmented
    plan = partitions.validate_partition_layout(connection)
    assert len(plan["result"]) > 1
    actual = connection.execute("SELECT * FROM research.result").fetchall()
    assert sorted(actual) == sorted(expected)
    stored = list(storage.stored_rows(connection, "run", 512))
    assert len(stored) == len(expected)
    assert sorted(row[1]["record_id"] for row in stored) == sorted(row[0] for row in expected)
    storage._verify_keys(connection, "run", PersistenceProgress("verify", None, None))
    writer = partitions.PartitionWriter(connection)
    connection.execute("BEGIN")
    with pytest.raises(duckdb.ConstraintException):
        writer.insert("result", [[value] for value in expected[0]], len(expected[0][3]))
    connection.execute("ROLLBACK")
    assert partitions.validate_partition_layout(connection) == plan
    assert sorted(connection.execute("SELECT * FROM research.result").fetchall()) == sorted(
        expected
    )


@pytest.mark.parametrize(
    "change",
    [
        "bad_name",
        "unknown_table",
        "missing_keys",
        "extra_keys",
        "no_view",
        "wrong_view",
        "gap",
        "view_part",
        "no_key_constraint",
    ],
)
def test_partition_layout_damage_is_rejected(segmented, change):
    """目录名字、视图、段类型及全局键约束被改动时必须拒绝，不能把隐藏的物理记录当成完整结果。"""
    connection, _ = segmented
    names = partitions.partition_catalog(connection)["result"]
    if change in {"bad_name", "unknown_table"}:
        name = "_part_result_bad" if change == "bad_name" else "_part_unknown_000000"
        connection.execute(f'CREATE TABLE research."{name}" (i INT)')
    elif change == "missing_keys":
        connection.execute("DROP TABLE research._keys_result")
    elif change == "extra_keys":
        connection.execute("CREATE TABLE research._keys_unknown(i INT)")
    elif change == "no_view":
        connection.execute("DROP VIEW research.result")
    elif change == "wrong_view":
        connection.execute(
            f'CREATE OR REPLACE VIEW research.result AS SELECT * FROM research."{names[0]}"'
        )
    elif change == "gap":
        connection.execute(f'ALTER TABLE research."{names[-1]}" RENAME TO _part_result_000099')
    elif change == "view_part":
        connection.execute("DROP VIEW research.result")
        connection.execute(f'DROP TABLE research."{names[0]}"')
        connection.execute(
            f'CREATE VIEW research."{names[0]}" AS SELECT * FROM research."{names[1]}"'
        )
        connection.execute(f"CREATE VIEW research.result AS {partitions.view_select(names)}")
    else:
        connection.execute("DROP TABLE research._keys_result")
        connection.execute(
            "CREATE TABLE research._keys_result AS "
            "SELECT record_id,run_id,result_id FROM research.result"
        )
    with pytest.raises(ValueError, match="research_partition_layout_invalid"):
        partitions.validate_partition_layout(connection)


@pytest.mark.parametrize("change", ["missing", "wrong_metadata", "extra"])
def test_partition_key_registry_matches_each_physical_row(segmented, change):
    """键少一行、多一行或元数据被改写，即使JSON载荷未变也必须拒绝。"""
    connection, expected = segmented
    if change == "missing":
        connection.execute("DELETE FROM research._keys_result WHERE record_id=?", [expected[0][0]])
    elif change == "extra":
        connection.execute("INSERT INTO research._keys_result VALUES ('extra','run','extra')")
    else:
        connection.execute(
            "UPDATE research._keys_result SET result_id='wrong' WHERE record_id=?", [expected[0][0]]
        )
    with pytest.raises(ValueError, match="research_partition_keys_invalid"):
        partitions.verify_partition_keys(connection, "result", "run")


def test_partition_limits_and_missing_counts_fail_closed(segmented, monkeypatch):
    """未知表、非法序号、过大批及无法取得计数均有固定失败，不执行伪造SQL或忽略尾部。"""
    connection, _ = segmented
    for table, number in (
        ("unknown", 0),
        ("result", -1),
        ("result", partitions.MAXIMUM_PARTITIONS),
    ):
        with pytest.raises(ValueError, match="research_partition_capacity_exceeded"):
            partitions.partition_name(table, number)
    writer = partitions.PartitionWriter(connection)
    with pytest.raises(ValueError, match="unknown_research_table"):
        writer.insert("unknown", [[], [], [], []], 1)
    with pytest.raises(ValueError, match="research_partition_payload_capacity_exceeded"):
        writer.insert("result", [[], [], [], []], 1025)
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITIONS", len(writer.parts["result"]))
    with pytest.raises(ValueError, match="research_partition_capacity_exceeded"):
        writer.insert("result", [["new"], ["run"], ["new"], ["{}"]], 2)

    class MissingCount:
        """仅替换计数返回，其余目录读取仍来自真实库。"""

        def execute(self, query, parameters=None):
            """计数返回None；例：创建新表段前不可猜测已有行数。"""
            if query.startswith("SELECT count(*) FROM research.") or query.startswith(
                "SELECT (SELECT count(*)"
            ):
                return self
            return connection.execute(query, parameters)

        def fetchone(self):
            """模拟无计数结果；例：完整性检查必须明确失败。"""
            return None

    probe = MissingCount()
    with pytest.raises(ValueError, match="research_partition_layout_invalid"):
        partitions.PartitionWriter(probe).insert("feature", [["id"], ["run"], ["id"], ["{}"]], 2)
    with pytest.raises(ValueError, match="research_partition_keys_invalid"):
        partitions.verify_partition_keys(probe, "result", "run")


@pytest.mark.parametrize(
    "sql",
    [
        None,
        "SELECT 1",
        "CREATE VIEW v AS ",
        "CREATE VIEW v AS X",
        "CREATE VIEW v AS (X",
        "CREATE VIEW v AS SELECT record_id,run_id,result_id,payload() "
        "FROM research._part_result_000000",
        "CREATE VIEW v AS SELECT record_id,run_id,result_id,payload "
        "FROM research._part_result_000000 WHERE false",
        "CREATE VIEW v AS (SELECT record_id,run_id,result_id,payload "
        "FROM research._part_result_000000",
    ],
)
def test_view_grammar_rejects_non_projection_expressions(sql):
    """视图只允许固定投影联合语法；函数调用、过滤、残缺括号和空SQL均拒绝。"""
    with pytest.raises(ValueError, match="research_partition_layout_invalid"):
        partitions._view_parts(sql)


def test_reopened_union_tree_keeps_the_same_partition_contract(tmp_path, monkeypatch):
    """DuckDB重新打开时会平衡UNION树，括号变化不应误报；四列与段序列必须保持相同。"""
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_BYTES", 1024)
    path = tmp_path / "reopened-parts.duckdb"
    with duckdb.connect(str(path), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        storage.initialize_tables(connection)
        writer = partitions.PartitionWriter(connection)
        for index in range(30):
            text = json.dumps({"synthetic": "x" * 600})
            writer.insert("result", [[str(index)], ["run"], [str(index)], [text]], len(text))
        before = partitions.validate_partition_layout(connection)
    with duckdb.connect(
        str(path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        assert partitions.validate_partition_layout(connection) == before


def test_balanced_view_builder_retains_odd_tail_and_empty_contract():
    """联合树按平衡括号构建，奇数尾段不能丢失；空目录不生成SQL，单段不引入联合。"""
    names = tuple(partitions.partition_name("result", index) for index in range(5))
    assert partitions.view_select(()) == ""
    assert "UNION" not in partitions.view_select(names[:1])
    assert (
        partitions._view_parts("CREATE VIEW v AS " + partitions.view_select(names).replace('"', ""))
        == names
    )


def test_key_join_missing_count_is_rejected(segmented):
    """键数量相等但后续一一对应查询没有返回计数时仍失败，不能把缺失统计当作零。"""
    connection, _ = segmented

    class MissingJoin:
        """仅丢弃FULL JOIN的返回值，其余查询使用真实库。"""

        def execute(self, query, parameters=None):
            """FULL JOIN模拟无结果；例：前置数量核对通过后仍必须检查关系计数。"""
            if " FULL JOIN " in query:
                return self
            return connection.execute(query, parameters)

        def fetchone(self):
            """返回缺失统计；例：预期固定partition_keys_invalid。"""
            return None

    with pytest.raises(ValueError, match="research_partition_keys_invalid"):
        partitions.verify_partition_keys(MissingJoin(), "result", "run")


def test_missing_view_definition_fails_closed(segmented):
    """目录存在但无法读取视图定义时拒绝验证，不把未知SQL当作可信投影。"""
    connection, _ = segmented

    class MissingDefinition:
        """只模拟视图定义查询缺行，其余查询仍使用真实目录与约束。"""

        def execute(self, query, parameters=None):
            """返回缺失定义；例：SELECT sql得到空结果而非猜测UNION内容。"""
            if query.startswith("SELECT sql FROM duckdb_views()"):
                return self
            return connection.execute(query, parameters)

        def fetchall(self):
            """模拟目录竞态/损坏导致缺行；例：验证应报告layout_invalid。"""
            return []

    with pytest.raises(ValueError, match="research_partition_layout_invalid"):
        partitions.validate_partition_layout(MissingDefinition())


def test_partitioned_pipeline_failure_cleanup_and_auxiliary_orphans(tmp_path, monkeypatch):
    """私有分段已提交后验证失败仍保留旧库；合法发布后额外孤立键不能逃过全库验证。"""
    from research_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research import pipeline
    from tushare_entitlement_hub.equity_research.common import make_result
    from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS
    from tushare_entitlement_hub.reference_storage import file_checksum

    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "published.duckdb"
    pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    original = file_checksum(output)
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_BYTES", storage.MAXIMUM_INSERT_JSON_BYTES)
    monkeypatch.setattr(storage, "MAXIMUM_STAGE_TRANSACTION_ROWS", 1)

    def build(inputs):
        """产生24个小结果，真实消费附加后超过一个物理段，预期不得丢任何实体。"""
        return tuple(
            make_result(inputs, PRODUCT_IDS[2], f"cn:stock:test:{index}") for index in range(24)
        )

    def reject(*args, **kwargs):
        """已分段保存后模拟验证失败，预期只有私有副本被清理。"""
        raise ValueError("research_projection_mismatch")

    monkeypatch.setattr(pipeline, "build_governance", build)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "_verify_run", reject)
        with pytest.raises(ValueError, match="research_projection_mismatch"):
            pipeline.run_research(reference, output, AS_OF, config, ("P06",))
    assert file_checksum(output) == original
    assert not list(tmp_path.glob(".*.stage-*"))
    result = pipeline.run_research(reference, output, AS_OF, config, ("P06",))
    assert result["result_count"] == 24
    assert storage.validate_research(reference, output)["valid"]
    with duckdb.connect(str(output), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        assert "result" in partitions.partition_catalog(connection)
        connection.execute(
            "INSERT INTO research._keys_result VALUES ('orphan','absent_run','orphan')"
        )
    assert not storage.validate_research(reference, output)["valid"]


def test_eight_mib_rows_roll_at_exact_partition_boundary_without_lowering_capacity(tmp_path):
    """三个恰好8MiB的合法JSON跨越16MiB段边界均完整保存；多一字节在SQL前拒绝且不损坏已写行。"""
    from tushare_entitlement_hub.equity_research.projection import EncodedRecord

    payload = '{"capacity":"' + "x" * (storage.MAXIMUM_RESULT_JSON_BYTES - 15) + '"}'
    assert len(payload.encode()) == storage.MAXIMUM_RESULT_JSON_BYTES
    with duckdb.connect(
        str(tmp_path / "exact-boundary.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        storage.initialize_tables(connection)
        connection.execute("BEGIN")
        trace = PersistenceProgress("save", None, None)
        transactions = storage._StageTransactions(connection, trace)
        writer = partitions.PartitionWriter(connection)
        buffer = storage._InsertBuffer(
            connection, "result", 128, trace, transactions.advance, writer
        )
        for number in range(3):
            key = sha256(str(number).encode()).hexdigest()
            buffer.add(EncodedRecord("result", key, "run", key, payload))
        with pytest.raises(ValueError, match="research_result_payload_capacity_exceeded"):
            buffer.add(EncodedRecord("result", "overflow", "run", "overflow", payload + " "))
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        plan = partitions.validate_partition_layout(connection)
        assert len(plan["result"]) == 2
        assert connection.execute(
            'SELECT count(*) FROM research."' + plan["result"][0] + '"'
        ).fetchone() == (2,)
        stored = list(storage._stored_records(connection, "run", ("result",)))
        assert len(stored) == 3
        assert all(row.payload == payload for row in stored)
        storage._verify_keys(connection, "run", PersistenceProgress("verify", None, None))


@pytest.mark.parametrize("dangling", [False, True])
def test_partition_references_join_verified_keys_instead_of_union_views(segmented, dangling):
    """引用查询只连接已对账的窄键表，避免分段视图自连接；同步篡改物理行与窄键仍检出孤儿引用。"""
    connection, expected = segmented
    queries = []
    if dangling:
        first = partitions.partition_catalog(connection)["result"][0]
        for table in (first, "_keys_result"):
            connection.execute(
                f'UPDATE research."{table}" SET result_id=? WHERE record_id=?',
                ["missing_parent", expected[0][0]],
            )

    class Recording:
        """记录真实SQL，仅对操作的表范围做复杂度断言，不替换数据库返回。"""

        def execute(self, query, parameters=None):
            """转发所有语句；例：引用检查应JOIN _keys_result而不是展开两侧UNION。"""
            queries.append(query)
            return connection.execute(query, parameters)

    if dangling:
        with pytest.raises(ValueError, match="research_result_reference_missing"):
            storage._verify_keys(Recording(), "run", PersistenceProgress("verify", None, None))
    else:
        storage._verify_keys(Recording(), "run", PersistenceProgress("verify", None, None))
    assert any("LEFT JOIN research._keys_result parent" in query for query in queries)
    assert not any("LEFT JOIN research.result parent" in query for query in queries)


@pytest.mark.parametrize("dangling", [False, True])
def test_parent_lineage_checks_every_partition_including_the_tail(segmented, dangling):
    """父血缘逐段核查仍包含最后一段；尾段不存在的parent不能被窄键优化漏掉。"""
    connection, expected = segmented
    writer = partitions.PartitionWriter(connection)
    for number in range(3):
        payload = json.dumps({
            "relationship": "parent",
            "parent_result_id": "missing" if dangling and number == 2 else expected[0][0],
            "padding": "x" * 600,
        })
        writer.insert(
            "lineage", [[str(number)], ["run"], [expected[0][0]], [payload]], len(payload)
        )
    assert len(partitions.partition_catalog(connection)["lineage"]) == 3
    if dangling:
        with pytest.raises(ValueError, match="research_parent_reference_missing"):
            storage._verify_keys(connection, "run", PersistenceProgress("verify", None, None))
    else:
        storage._verify_keys(connection, "run", PersistenceProgress("verify", None, None))
