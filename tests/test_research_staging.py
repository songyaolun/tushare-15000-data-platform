"""研究快照的原子发布、原生锁、并发、故障清理及已提交验证边界。"""

import json
import os
from pathlib import Path

import duckdb
import pytest
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference

from tushare_entitlement_hub.equity_research import pipeline, staging, storage
from tushare_entitlement_hub.equity_research.persistence_runtime import (
    PersistenceError,
    PersistenceProgress,
)
from tushare_entitlement_hub.reference_storage import file_checksum


def test_staging_preserves_old_runs_and_publishes_only_after_closed_write(tmp_path, monkeypatch):
    """两个旧run和其他schema完整保留；verify使用新只读连接且无活动写事务，发布前正式库不变。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    first = pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE TABLE unrelated(value INTEGER)")
        connection.execute("INSERT INTO unrelated VALUES (7)")
    before = file_checksum(output)
    original = pipeline.verify_run
    verified = []

    def verify(connection, metadata, **kwargs):
        """完整验证前确认旧库字节不变，且新连接可开始只读事务；旧事务实现会在BEGIN失败。"""
        assert file_checksum(output) == before
        connection.execute("BEGIN TRANSACTION")
        connection.execute("ROLLBACK")
        with pytest.raises(duckdb.InvalidInputException, match="read-only"):
            connection.execute("CREATE TABLE must_not_write(i INT)")
        verified.append(metadata["run_id"])
        return original(connection, metadata, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "verify_run", verify)
        second = pipeline.run_research(reference, output, AS_OF, config, ("P06",))
    assert verified == [second["run_id"]]
    assert first != second
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT * FROM unrelated").fetchall() == [(7,)]
        assert connection.execute("SELECT count(*) FROM research.run").fetchone() == (2,)
    # 复用旧run也经过私有验证后更新current_run，不丢后来的run。
    assert pipeline.run_research(reference, output, AS_OF, config, ("P01",)) == first
    assert storage.validate_research(reference, output)["valid"]
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "stage", ["open", "checkpoint", "verify_open", "verify_close", "publish", "sync", "changed"]
)
def test_staging_failures_keep_atomic_state_and_cleanup(tmp_path, monkeypatch, existing, stage):
    """打开/检查点/验证连接/发布/目录同步/参考变化失败均分类；未替换时旧库原样，替换后仍完整可验证。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    if existing:
        pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    before = file_checksum(output) if existing else None
    connect, replace, fsync, verify = (
        duckdb.connect,
        staging.os.replace,
        staging.os.fsync,
        pipeline.verify_run,
    )
    events = []

    class CheckpointFailure:
        """仅在真实写事务提交后的CHECKPOINT失败，其余SQL透传。"""

        def __init__(self, connection, read_only=False):
            """保存待测写连接；例：退出代理仍关闭数据库。"""
            self.connection = connection
            self.read_only = read_only

        def __enter__(self):
            """返回代理；例：initialize和save执行真实SQL。"""
            return self

        def __exit__(self, *args):
            """总是关闭真实连接；例：检查点报错也释放WAL和内存。"""
            self.connection.close()
            if stage == "verify_close" and self.read_only:
                raise OSError("PRIVATE_CLOSE")

        def execute(self, query, parameters=None):
            """只拒绝CHECKPOINT；例：COMMIT必须已真实完成。"""
            if stage == "checkpoint" and query == "CHECKPOINT":
                raise OSError("PRIVATE_CHECKPOINT")
            return self.connection.execute(query, parameters)

    def open_database(database, **kwargs):
        """只对私有副本注入故障，不影响合成reference或事后验证。"""
        if ".stage-" not in str(database):
            return connect(database, **kwargs)
        if stage == "open" or (stage == "verify_open" and kwargs.get("read_only")):
            raise OSError("PRIVATE_OPEN")
        result = connect(database, **kwargs)
        return (
            CheckpointFailure(result, kwargs.get("read_only", False))
            if stage in {"checkpoint", "verify_close"}
            else result
        )

    def replacing(*args):
        """原子替换前失败必须保留原路径；输入路径不会出现在安全进度中。"""
        if stage == "publish":
            raise OSError("PRIVATE_REPLACE")
        return replace(*args)

    def syncing(descriptor):
        """只在替换后目录同步失败；此时正式库应是完整已验证的新快照。"""
        if stage == "sync" and os.fstat(descriptor).st_mode & 0o170000 == 0o040000:
            raise OSError("PRIVATE_SYNC")
        return fsync(descriptor)

    def verifying(*args, **kwargs):
        """完成真实重建后修改reference；发布必须再次检查输入指纹。"""
        result = verify(*args, **kwargs)
        if stage == "changed":
            with reference.open("ab") as stream:
                stream.write(b"synthetic_change")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(duckdb, "connect", open_database)
        patch.setattr(staging.os, "replace", replacing)
        patch.setattr(staging.os, "fsync", syncing)
        patch.setattr(pipeline, "verify_run", verifying)
        with pytest.raises(PersistenceError) as caught:
            pipeline.run_research(
                reference, output, AS_OF, config, ("P06",), progress=events.append
            )
    assert caught.value.phase == (
        "stage"
        if stage in {"open", "checkpoint"}
        else "verify"
        if stage in {"verify_open", "verify_close"}
        else "publish"
    )
    if stage == "sync":
        assert storage.validate_research(reference, output)["valid"]
    else:
        assert (file_checksum(output) if output.exists() else None) == before
    assert "PRIVATE_" not in json.dumps(events)
    assert events[-1]["stage"] == "persistence-failed"
    assert events[-1]["committed"] is (stage == "sync")
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("path_type", ["symlink", "hardlink", "directory", "lock_symlink"])
def test_unsafe_output_aliases_are_rejected_without_modifying_data(tmp_path, path_type):
    """符号链接、硬链接、目录与锁链接均拒绝，不能通过别名绕过发布锁或覆盖目标。"""
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    target = tmp_path / "target"
    target.write_bytes(b"target")
    output = tmp_path / "products"
    if path_type == "symlink":
        output.symlink_to(target)
    elif path_type == "hardlink":
        output.hardlink_to(target)
    elif path_type == "directory":
        output.mkdir()
    else:
        (tmp_path / ".products.research.lock").symlink_to(target)
    with pytest.raises(PersistenceError, match="research_output_path_unsafe"):
        with staging.research_snapshot(reference, output, run_id="a" * 64):
            pytest.fail("unsafe path accepted")
    assert target.read_bytes() == b"target"


@pytest.mark.parametrize("destination", ["output", "lock", "wal", "hardlink"])
def test_reference_collision_includes_control_paths(tmp_path, destination):
    """reference与输出、路径锁、WAL或其硬链接碰撞均在创建/覆盖前拒绝。"""
    output = tmp_path / "products"
    paths = {
        "output": output,
        "lock": tmp_path / ".products.research.lock",
        "wal": tmp_path / "products.wal",
    }
    reference = paths.get(destination, tmp_path / "reference")
    reference.write_bytes(b"reference")
    if destination == "hardlink":
        output.hardlink_to(reference)
    with pytest.raises(PersistenceError, match="input_output_path_collision"):
        with staging.research_snapshot(reference, output, run_id="a" * 64):
            pytest.fail("collision accepted")
    assert reference.read_bytes() == b"reference"


def test_concurrent_publishers_and_native_writers_fail_closed(tmp_path):
    """路径锁排斥第二发布者；原生锁排斥同进程已有/新建DuckDB写连接，允许旧快照只读。"""
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    output = tmp_path / "products.duckdb"
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as writer:
        writer.execute("CREATE TABLE sample(i INT)")
        with pytest.raises(PersistenceError, match="research_output_busy"):
            with staging.research_snapshot(reference, output, run_id="a" * 64):
                pytest.fail("native writer accepted")
    with staging.research_snapshot(reference, output, run_id="a" * 64):
        with pytest.raises(PersistenceError, match="research_output_busy"):
            with staging.research_snapshot(reference, output, run_id="b" * 64):
                pytest.fail("concurrent publisher accepted")
        with pytest.raises(duckdb.IOException, match="lock"):
            duckdb.connect(str(output), config=DATABASE_CONFIG)
        with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as reader:
            assert reader.execute("SELECT count(*) FROM sample").fetchone() == (0,)
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as writer:
        writer.execute("INSERT INTO sample VALUES (1)")


@pytest.mark.parametrize(
    "mutation", ["changed", "new_file", "wal_before", "wal_after", "stage_wal", "interrupt"]
)
def test_snapshot_path_changes_wal_and_interrupt_never_publish(tmp_path, mutation):
    """发布前路径被替换、新文件出现、WAL存在或中断均拒绝；临时文件清理且锁可再次获取。"""
    reference = tmp_path / "reference"
    reference.write_bytes(b"reference")
    output = tmp_path / "products"
    if mutation != "new_file":
        output.write_bytes(b"old")
    if mutation == "wal_before":
        Path(str(output) + ".wal").write_bytes(b"unrecovered")
    error_type = KeyboardInterrupt if mutation == "interrupt" else PersistenceError
    with pytest.raises(error_type):
        with staging.research_snapshot(reference, output, run_id="a" * 64) as snapshot:
            snapshot.path.write_bytes(b"new")
            (snapshot.path.parent / "spill.tmp").write_bytes(b"temporary")
            if mutation == "interrupt":
                raise KeyboardInterrupt
            if mutation in {"changed", "new_file"}:
                output.write_bytes(b"changed")
            elif mutation == "wal_after":
                Path(str(output) + ".wal").write_bytes(b"late")
            elif mutation == "stage_wal":
                Path(str(snapshot.path) + ".wal").write_bytes(b"uncheckpointed")
            snapshot.publish(PersistenceProgress("publish", None, None))
    assert output.read_bytes() != b"new"
    assert not list(tmp_path.glob(".*.stage-*"))
    Path(str(output) + ".wal").unlink(missing_ok=True)
    with staging.research_snapshot(reference, output, run_id="a" * 64):
        pass


def test_lost_native_lock_prevents_publication(tmp_path):
    """原生锁进程意外退出后不得发布，避免在锁失效期间覆盖另一个写者；私有状态仍清理。"""
    reference, output = tmp_path / "reference", tmp_path / "products"
    reference.write_bytes(b"reference")
    output.write_bytes(b"old")
    with pytest.raises(PersistenceError, match="research_output_lock_lost"):
        with staging.research_snapshot(reference, output, run_id="a" * 64) as snapshot:
            snapshot.guard.terminate()
            snapshot.guard.wait()
            snapshot.path.write_bytes(b"new")
            snapshot.publish(PersistenceProgress("publish", None, None))
    assert output.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.stage-*"))


@pytest.mark.parametrize("budget", ["bytes", "rows"])
def test_private_batches_commit_with_complete_projections_and_failure_cleanup(
    tmp_path, monkeypatch, budget
):
    """小字节/行数预算分别触发真实分段提交，完整投影通过；后来verify失败仍不发布部分批。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "batches.duckdb"
    monkeypatch.setattr(
        storage,
        "MAXIMUM_STAGE_TRANSACTION_BYTES"
        if budget == "bytes"
        else "MAXIMUM_STAGE_TRANSACTION_ROWS",
        1000 if budget == "bytes" else 1,
    )
    events = []
    first = pipeline.run_research(
        reference, output, AS_OF, config, ("P06",), progress=events.append
    )
    assert any(event.get("operation") == "commit_batch" for event in events)
    assert storage.validate_research(reference, output)["valid"]
    before = file_checksum(output)

    def reject(*args, **kwargs):
        """在副本多次真实提交后拒绝重建；正式库仍只含之前已验证的run。"""
        raise ValueError("research_projection_mismatch")

    monkeypatch.setattr(storage, "_verify_run", reject)
    with pytest.raises(PersistenceError, match="research_projection_mismatch"):
        pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    assert file_checksum(output) == before
    assert not list(tmp_path.glob(".*.stage-*"))
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT run_id FROM research.run").fetchall() == [
            (first["run_id"],)
        ]
