"""旧全局ART键表迁移、多run隔离、篡改检测及行数容量的回归。"""

from pathlib import Path

import duckdb
import pytest
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference

from tushare_entitlement_hub.equity_research import key_index, partitions, pipeline, storage
from tushare_entitlement_hub.reference_storage import file_checksum


def restore_legacy_keys(connection):
    """把小型合成库恢复为be67aa1全局主键布局，保持全部行值；例：两个旧run仍可完整验证。"""
    for table in partitions.partition_catalog(connection):
        name = "_keys_" + table
        connection.execute(
            'CREATE TABLE research.legacy_keys (record_id VARCHAR PRIMARY KEY,'
            'run_id VARCHAR NOT NULL,result_id VARCHAR NOT NULL)'
        )
        connection.execute(f'INSERT INTO research.legacy_keys SELECT * FROM research."{name}"')
        connection.execute(f'DROP TABLE research."{name}"')
        connection.execute(f'ALTER TABLE research.legacy_keys RENAME TO "{name}"')
    connection.execute("DELETE FROM research.metadata WHERE key='partition_keys'")


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_runs_migrate_privately_without_changing_existing_results(
    tmp_path, monkeypatch, fail
):
    """两个真实旧run迁移后保留字节并新增第三run；迁移后写入失败不得改变正式库，私有索引必须清理。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_ROWS", 1)
    monkeypatch.setattr(storage, "MAXIMUM_INSERT_ROWS", 1)
    first = pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    second = pipeline.run_research(reference, output, AS_OF, config, ("P06",))
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        restore_legacy_keys(connection)
        previous = connection.execute("SELECT * FROM research.result ORDER BY record_id").fetchall()
    before = file_checksum(output)
    assert storage.validate_research(reference, output)["valid"]
    directories = []

    class RecordedIndex(key_index.DiskKeyIndex):
        """只记录真实磁盘索引目录，用来断言成功、失败及验证开始前的清理。"""

        def __init__(self):
            """创建真实受限缓存索引；例：迁移完毕后目录不能随产品库发布。"""
            super().__init__()
            directories.append(Path(self.temporary.name))

    def progress(event):
        """验证开始时旧正式库仍保持原字节，SQLite登记器已关闭；例：无未验证窗口。"""
        if event["stage"] == "verify-start":
            assert file_checksum(output) == before
            assert directories and all(not path.exists() for path in directories)

    def reject(self, row):
        """在写入器已完成旧键迁移后注入IO错误；例：整个私有副本应被删除。"""
        raise duckdb.IOException("PRIVATE_WRITE")

    monkeypatch.setattr(partitions, "DiskKeyIndex", RecordedIndex)
    if fail:
        monkeypatch.setattr(storage._InsertBuffer, "add", reject)
        with pytest.raises(ValueError, match="research_save_io_failed"):
            pipeline.run_research(reference, output, AS_OF, config, ("P07",), progress=progress)
        assert file_checksum(output) == before
    else:
        third = pipeline.run_research(
            reference, output, AS_OF, config, ("P07",), progress=progress
        )
        assert third["run_id"] not in {first["run_id"], second["run_id"]}
        with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
            assert partitions._disk_key_layout(connection)
            assert connection.execute(
                "SELECT * FROM research.result WHERE run_id IN (?,?) ORDER BY record_id",
                [first["run_id"], second["run_id"]],
            ).fetchall() == previous
            assert connection.execute(
                "SELECT count(*) FROM duckdb_constraints() WHERE schema_name='research' "
                "AND starts_with(table_name,'_keys_') AND constraint_type='PRIMARY KEY'"
            ).fetchone() == (0,)
            connection.execute("BEGIN")
            with partitions.PartitionWriter(connection) as writer:
                duplicate = previous[0]
                with pytest.raises(
                    duckdb.ConstraintException, match="research_primary_key_invalid"
                ):
                    writer.insert("result", [[value] for value in duplicate], len(duplicate[3]))
            connection.execute("ROLLBACK")
    assert directories and all(not path.exists() for path in directories)
    assert not list(tmp_path.glob(".*.stage-*"))
    assert storage.validate_research(reference, output)["valid"]


def test_small_payloads_are_partitioned_by_row_count_and_keep_global_uniqueness(
    tmp_path, monkeypatch
):
    """极小JSON仍按行数滚动，防止原表ART无界增长；跨段/跨run同键立即失败，过大批有固定码。"""
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_ROWS", 2)
    with duckdb.connect(str(tmp_path / "rows.duckdb"), config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        with partitions.PartitionWriter(connection) as writer:
            for number in range(5):
                identity = str(number)
                writer.insert("result", [[identity], ["old_run"], [identity], ["{}"]], 2)
            with pytest.raises(duckdb.ConstraintException):
                writer.insert("result", [["0"], ["new_run"], ["0"], ["{}"]], 2)
            with pytest.raises(ValueError, match="research_partition_row_capacity_exceeded"):
                writer.insert("result", [["a", "b", "c"], [], [], []], 6)
        plan = partitions.partition_catalog(connection)
        assert len(plan["result"]) == 3
        assert all(
            connection.execute(f'SELECT count(*) FROM research."{name}"').fetchone()[0] <= 2
            for name in plan["result"]
        )
        partitions.verify_partition_keys(connection, "result", None)
        connection.execute("INSERT INTO research._keys_result VALUES ('0','different_run','0')")
        with pytest.raises(ValueError, match="research_primary_key_invalid"):
            partitions.verify_partition_keys(connection, "result", "old_run")
        with pytest.raises(ValueError, match="research_primary_key_invalid"):
            partitions.PartitionWriter(connection)


@pytest.mark.parametrize("change", ["version", "art", "unique", "nullability", "view"])
def test_disk_key_layout_rejects_tampered_format_or_reintroduced_art(tmp_path, monkeypatch, change):
    """未知版本、额外ART、UNIQUE约束和取消NOT NULL均拒绝，不能悄悄恢复无界索引或降低键合同。"""
    monkeypatch.setattr(partitions, "MAXIMUM_PARTITION_ROWS", 1)
    with duckdb.connect(str(tmp_path / "damaged.duckdb"), config=DATABASE_CONFIG) as connection:
        storage.initialize_tables(connection)
        with partitions.PartitionWriter(connection) as writer:
            for number in range(2):
                value = str(number)
                writer.insert("result", [[value], ["run"], [value], ["{}"]], 2)
        if change == "version":
            connection.execute(
                "UPDATE research.metadata SET payload='42' WHERE key='partition_keys'"
            )
        elif change == "art":
            connection.execute("CREATE INDEX bad_art ON research._keys_result(record_id)")
        elif change == "view":
            connection.execute("DROP TABLE research._keys_result")
            connection.execute(
                "CREATE VIEW research._keys_result AS "
                "SELECT record_id,run_id,result_id FROM research.result"
            )
        else:
            connection.execute("DROP TABLE research._keys_result")
            constraint = "UNIQUE NOT NULL" if change == "unique" else ""
            connection.execute(
                f"CREATE TABLE research._keys_result(record_id VARCHAR {constraint},"
                "run_id VARCHAR NOT NULL,result_id VARCHAR NOT NULL)"
            )
        with pytest.raises(ValueError, match="research_partition_layout_invalid"):
            partitions.validate_partition_layout(connection)
