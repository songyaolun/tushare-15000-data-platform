"""分段提交的固定子步骤诊断，包括DuckDB Fatal包裹的内存失败。"""

import duckdb
import pytest

from tushare_entitlement_hub.equity_research import storage
from tushare_entitlement_hub.equity_research.persistence_runtime import (
    PersistenceProgress,
    PersistenceStepError,
    database_error_code,
    error_category,
)


@pytest.mark.parametrize(
    "error,code,category",
    [
        (
            duckdb.FatalException(
                "FATAL Error: Failed to rollback transaction. Cannot continue operation.\n"
                "Original Error: Out of Memory Error: PRIVATE"
            ),
            "fatal_rollback_memory",
            "memory",
        ),
        (
            duckdb.FatalException(
                "FATAL Error: Failed to create checkpoint because of error: "
                "could not allocate block of size PRIVATE"
            ),
            "fatal_checkpoint_memory",
            "memory",
        ),
        (duckdb.FatalException("PRIVATE memory unrelated"), "fatal", "database"),
        (duckdb.InternalException("PRIVATE"), "internal", "database"),
        (duckdb.IOException("PRIVATE"), "duckdb_error", "io"),
        (ValueError("PRIVATE"), None, "validation"),
    ],
)
def test_database_types_and_known_fatal_wrappers_have_safe_codes(error, code, category):
    """只用类型与已复现的固定Fatal前缀分类，任意消息中的memory不能冒充内存根因。"""
    assert database_error_code(error) == code
    assert error_category(error) == category
    wrapped = PersistenceStepError("commit", error)
    assert str(wrapped) == f"research_save_batch_commit_{category}_failed"
    assert wrapped.database_error_code == code
    with pytest.raises(ValueError, match="invalid_persistence_substep"):
        PersistenceStepError("unknown", error)


@pytest.mark.parametrize(
    "substep,statement",
    [("commit", "COMMIT"), ("checkpoint", "CHECKPOINT"), ("reopen", "BEGIN TRANSACTION")],
)
def test_batch_failure_names_the_exact_statement_without_exposing_payload(substep, statement):
    """分别在三条原生语句失败，停止后续步骤，错误及进度只含固定子步骤和安全码。"""
    events, commands = [], []

    class FailingConnection:
        """只对指定语句报错，避免用墙钟或真实内存耗尽测试状态分支。"""

        def execute(self, query):
            """记录语句并在目标点失败；例：CHECKPOINT失败后不能执行BEGIN。"""
            commands.append(query)
            if query == statement:
                raise duckdb.InternalException("PRIVATE_PAYLOAD")

    trace = PersistenceProgress("save", events.append, "a" * 64)
    controller = storage._StageTransactions(FailingConnection(), trace)
    with pytest.raises(PersistenceStepError) as caught:
        controller.advance(0, storage.MAXIMUM_STAGE_TRANSACTION_BYTES)
    assert caught.value.substep == substep
    assert str(caught.value) == f"research_save_batch_{substep}_database_failed"
    assert commands[-1] == statement
    trace.failure(caught.value)
    assert events[-1]["substep"] == substep
    assert events[-1]["database_error_code"] == "internal"
    assert events[-1]["transaction_state"] == "failed"
    assert "PRIVATE" not in str(events)


def test_flat_wide_json_reproduces_fatal_commit_and_safe_root_cause(tmp_path):
    """旧单表布局累计宽JSON后，在512MB COMMIT重现Fatal包装内存错误；无需达到正式全库规模。"""
    from hashlib import sha256

    from tushare_entitlement_hub.equity_research.projection import EncodedRecord

    connection = duckdb.connect(
        str(tmp_path / "old-flat.duckdb"), config={"threads": 1, "memory_limit": "512MB"}
    )
    storage.initialize_tables(connection)
    connection.execute("BEGIN TRANSACTION")
    trace = PersistenceProgress("save", None, None)
    controller = storage._StageTransactions(connection, trace)
    # 故意使用原布局，不提供PartitionWriter，重现旧178310a的宽向量回扫。
    buffer = storage._InsertBuffer(connection, "result", 128, trace, controller.advance)
    payload = '{"synthetic":"' + "a" * 1048576 + '"}'
    try:
        with pytest.raises(PersistenceStepError) as caught:
            for index in range(640):
                key = sha256(str(index).encode()).hexdigest()
                buffer.add(EncodedRecord("result", key, "run", key, payload))
            buffer.flush()
        assert caught.value.substep == "commit"
        assert caught.value.database_error_code == "fatal_rollback_memory"
        assert caught.value.category == "memory"
        assert isinstance(caught.value.__cause__, duckdb.FatalException)
        # 失败事务交给关闭连接处理，既不重试提交，也不向已失败数据库再次ROLLBACK。
    finally:
        connection.close()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("substep", ["commit", "checkpoint", "reopen"])
def test_batch_failure_preserves_original_cause_and_private_cleanup(
    tmp_path, monkeypatch, existing, substep
):
    """真实流水线在分段子步骤失败时保留原始异常链，首跑不发布，已有库字节不变且副本清理。"""
    from research_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research import pipeline
    from tushare_entitlement_hub.reference_storage import file_checksum

    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    if existing:
        pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    before = file_checksum(output) if existing else None
    commands, events = [], []
    original = storage._StageTransactions
    statement = {"commit": "COMMIT", "checkpoint": "CHECKPOINT", "reopen": "BEGIN TRANSACTION"}[
        substep
    ]
    cause = duckdb.InternalException("PRIVATE_DATABASE_DETAIL")

    class FailingStep:
        """只拦截分段事务语句，所有载荷写入和上下文关闭仍由真实DuckDB连接执行。"""

        def __init__(self, connection):
            """保存真实私有连接；例：失败前的COMMIT/CHECKPOINT仍实际执行。"""
            self.connection = connection

        def execute(self, query):
            """在指定分段语句失败；例：BEGIN失败时前一事务已提交，仍不得发布私有库。"""
            commands.append(query)
            if query == statement:
                raise cause
            return self.connection.execute(query)

    def controller(connection, trace):
        """使用真实分段控制器与可定位故障连接；例：只改变一条SQL的返回，不替代保存逻辑。"""
        return original(FailingStep(connection), trace)

    monkeypatch.setattr(storage, "_StageTransactions", controller)
    monkeypatch.setattr(storage, "MAXIMUM_STAGE_TRANSACTION_ROWS", 1)
    with pytest.raises(PersistenceStepError) as caught:
        pipeline.run_research(reference, output, AS_OF, config, ("P06",), progress=events.append)
    assert caught.value.__cause__ is cause
    assert caught.value.substep == substep
    assert commands[-1] == statement
    failure = events[-1]
    assert failure["stage"] == "persistence-failed"
    assert failure["reason_code"] == f"research_save_batch_{substep}_database_failed"
    assert failure["database_error_code"] == "internal"
    assert failure["transaction_state"] == "failed"
    assert not any(event.get("phase") == "rollback" for event in events)
    assert "PRIVATE" not in str(events)
    assert (file_checksum(output) if output.exists() else None) == before
    assert not list(tmp_path.glob(".*.stage-*"))
