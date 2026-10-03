"""持久化阶段错误分类和安全、独立的进度事件。"""

import gc
import json
import weakref

import duckdb
import pytest
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.equity_research import cli, pipeline, storage
from tushare_entitlement_hub.equity_research.persistence_runtime import (
    PersistenceError,
    PersistenceProgress,
    ResultPayloadCapacityError,
    error_category,
)


@pytest.mark.parametrize(
    "error,category",
    [
        (ResultPayloadCapacityError({}, 9), "capacity"),
        (MemoryError("PRIVATE_PAYLOAD"), "memory"),
        (duckdb.OutOfMemoryException("PRIVATE_PAYLOAD"), "memory"),
        (OSError("PRIVATE_PAYLOAD"), "io"),
        (duckdb.IOException("PRIVATE_PAYLOAD"), "io"),
        (duckdb.ConstraintException("PRIVATE_PAYLOAD"), "constraint"),
        (duckdb.ConversionException("PRIVATE_PAYLOAD"), "conversion"),
        (duckdb.TransactionException("PRIVATE_PAYLOAD"), "transaction"),
        (duckdb.BinderException("PRIVATE_PAYLOAD"), "database"),
        (ValueError("PRIVATE_PAYLOAD"), "validation"),
        (TypeError("PRIVATE_PAYLOAD"), "validation"),
        (KeyError("PRIVATE_PAYLOAD"), "validation"),
        (RuntimeError("PRIVATE_PAYLOAD"), "unexpected"),
    ],
)
def test_error_types_have_fixed_safe_categories(error, category, monkeypatch, tmp_path):
    """同一异常类型得到固定分类，CLI不输出原异常或私有载荷；容量固定码保持原约定。"""
    assert error_category(error) == category
    wrapped = PersistenceError("verify", error)
    assert wrapped.phase == "verify" and wrapped.category == category

    def fail(*args, **kwargs):
        """模拟已分类阶段失败；例：任何run参数均返回同一安全包装异常。"""
        raise wrapped from error

    monkeypatch.setattr(cli, "run_research", fail)
    result = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            AS_OF,
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": [str(wrapped)],
        "phase": "verify",
        "category": category,
    }
    assert "PRIVATE_PAYLOAD" not in result.output


def test_progress_counters_and_callback_failure_do_not_expose_payload():
    """进度按行/结果阈值输出独立计数，未知运行标识不回显，失败回调不能覆盖原错误。"""
    events = []
    trace = PersistenceProgress("save", events.append, "PRIVATE_PAYLOAD")
    trace.emit("start-save")
    trace.step("write")
    trace.advance("result", 4095, 10)
    trace.advance("gate", 1, 20)
    for _ in range(16):
        trace.result_completed()
    assert len(events) == 4
    assert events[-1]["processed_rows"] == 4096 and events[-1]["payload_bytes"] == 30
    assert events[-1]["completed_results"] == 16
    events[-1]["processed_rows"] = -1
    assert trace.processed_rows == 4096
    assert "PRIVATE_PAYLOAD" not in json.dumps(events)
    known = PersistenceError("verify", ValueError("research_projection_mismatch"))
    assert str(known) == "research_projection_mismatch"
    assert trace.failure(known) is known

    def broken_callback(event):
        """模拟诊断通道不可用；任何事件都失败，原阶段失败仍须保留。"""
        raise OSError("PRIVATE_CALLBACK")

    assert PersistenceProgress("save", broken_callback, None).failure(known) is known
    silent = PersistenceProgress("save", None, "a" * 64)
    silent.emit("start-save")
    assert silent.failure(known) is known
    for constructor, argument in ((PersistenceError, ValueError()), (PersistenceProgress, None)):
        with pytest.raises(ValueError, match="invalid_persistence_phase"):
            if constructor is PersistenceProgress:
                constructor("invalid", argument, None)
            else:
                constructor("invalid", argument)


@pytest.mark.parametrize("phase", ["save", "verify"])
def test_stage_failure_rolls_back_all_tables_and_keeps_safe_progress(tmp_path, monkeypatch, phase):
    """写入部分结果后发生save内存或verify校验失败，全部表回滚；进度准确报告失败子阶段。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    events = []
    if phase == "save":
        original = storage._InsertBuffer.flush

        def fail(buffer):
            """先允许result写入，在非空gate批失败；事务必须撤销先前result。"""
            if buffer.table == "gate" and buffer.columns[0]:
                raise duckdb.OutOfMemoryException("PRIVATE_PAYLOAD")
            return original(buffer)

        monkeypatch.setattr(storage._InsertBuffer, "flush", fail)
    else:

        def reject(result):
            """模拟完整模型重建后的消费校验错误，不包含任何可安全输出的错误原文。"""
            raise ValueError("PRIVATE_PAYLOAD")

        monkeypatch.setattr(storage, "validate_result_consumption", reject)
    with pytest.raises(PersistenceError) as caught:
        pipeline.run_research(reference, output, AS_OF, config, ("P01",), progress=events.append)
    assert caught.value.phase == phase
    failures = [event for event in events if event["stage"] == "persistence-failed"]
    assert len(failures) == 1 and failures[0]["phase"] == phase
    assert failures[0]["operation"] == ("write" if phase == "save" else "rebuild")
    assert events[-1]["stage"] != "complete"
    assert "PRIVATE_PAYLOAD" not in json.dumps(events)
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        for table in (*storage.RESEARCH_TABLES, "run", "metadata"):
            assert connection.execute(f'SELECT count(*) FROM research."{table}"').fetchone()[0] == 0


def test_success_and_reuse_keep_summary_and_required_milestones(tmp_path):
    """成功保存和重跑分别报告完整阶段，摘要及run_id不改变，重跑仍进行完整verify。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    events = []
    first = pipeline.run_research(
        reference, output, AS_OF, config, ("P01",), progress=events.append
    )
    milestones = {"start-save", "save-complete", "verify-start", "verify-complete", "complete"}
    assert [event["stage"] for event in events if event["stage"] in milestones] == [
        "start-save",
        "save-complete",
        "verify-start",
        "verify-complete",
        "complete",
    ]
    assert all(
        event["committed"] is False
        for event in events
        if event["stage"] in milestones - {"complete"}
    )
    replay = []
    assert (
        pipeline.run_research(reference, output, AS_OF, config, ("P01",), progress=replay.append)
        == first
    )
    assert [event["stage"] for event in replay if event["stage"] != "verify-progress"] == [
        "start",
        "verify-start",
        "verify-complete",
        "reused",
    ]


def test_computed_inputs_and_results_are_released_before_rebuild(tmp_path, monkeypatch):
    """P13及其上游保存后释放所有计算结果和来源输入，完整verify不与原对象集合同时驻留。"""
    reference, config = write_reference(tmp_path, 0)
    references = []
    load, attach, verify = pipeline.load_inputs, pipeline.attach_consumption, pipeline.verify_run

    def loaded(*args, **kwargs):
        """仅保留输入弱引用；例：保存后不应再有强引用阻止ResearchInputs释放。"""
        inputs, processing = load(*args, **kwargs)
        references.append(weakref.ref(inputs))
        return inputs, processing

    def attached(*args, **kwargs):
        """记录最终消费结果的弱引用，不额外保留tuple或模型实例。"""
        results = attach(*args, **kwargs)
        references.extend(weakref.ref(result) for result in results)
        return results

    def verified(*args, **kwargs):
        """仍运行真实完整校验；例：全部弱引用先失效，再从数据库重建结果。"""
        gc.collect()
        assert references and all(reference() is None for reference in references)
        return verify(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_inputs", loaded)
    monkeypatch.setattr(pipeline, "attach_consumption", attached)
    monkeypatch.setattr(pipeline, "verify_run", verified)
    pipeline.run_research(reference, tmp_path / "products.duckdb", AS_OF, config, ("P13",))


@pytest.mark.parametrize(
    "scenario", ["begin", "metadata", "commit", "rollback", "reuse_commit", "reuse_verify"]
)
def test_transaction_and_reuse_failures_keep_phase_and_atomic_state(
    tmp_path, monkeypatch, scenario
):
    """BEGIN、元数据、COMMIT、ROLLBACK及复用失败均有固定阶段；新运行无残留，旧运行保持完整。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "products.duckdb"
    reused = scenario.startswith("reuse_")
    if reused:
        pipeline.run_research(reference, output, AS_OF, config, ("P01",))
    original_connect = duckdb.connect
    events = []

    class FaultyConnection:
        """使用真实连接，仅在指定事务语句失败；上下文退出仍关闭并回滚未提交事务。"""

        def __init__(self, connection):
            """保存实际连接；例：其余查询、主键约束和写入不被模拟。"""
            self.connection = connection

        def __enter__(self):
            """把代理提供给流水线；例：with内部的SQL仍转发真实连接。"""
            return self

        def __exit__(self, *args):
            """异常退出也关闭连接，DuckDB撤销未提交数据，不模拟成功提交。"""
            self.connection.close()

        def execute(self, query, parameters=None):
            """只拦截目标语句；例：COMMIT抛IO错误后，ROLLBACK仍执行真实SQL。"""
            failing = (
                (scenario == "begin" and query == "BEGIN TRANSACTION")
                or (
                    scenario in {"metadata", "reuse_commit"}
                    and "INSERT OR REPLACE INTO research.metadata" in query
                )
                or (scenario in {"commit", "rollback"} and query == "COMMIT")
            )
            if failing:
                raise duckdb.IOException("PRIVATE_DATABASE_PATH")
            if scenario == "rollback" and query == "ROLLBACK":
                raise duckdb.TransactionException("PRIVATE_ROLLBACK")
            return (
                self.connection.execute(query, parameters)
                if parameters is not None
                else self.connection.execute(query)
            )

    def connect(database, **kwargs):
        """只代理待写产品库，不影响只读合成reference连接。"""
        connection = original_connect(database, **kwargs)
        return FaultyConnection(connection) if str(database) == str(output) else connection

    def reject(*args, **kwargs):
        """复用时模拟重建IO失败，旧运行不得被当成成功复用。"""
        raise OSError("PRIVATE_VERIFY")

    with monkeypatch.context() as patch:
        patch.setattr(duckdb, "connect", connect)
        if scenario == "reuse_verify":
            patch.setattr(storage, "_verify_run", reject)
        with pytest.raises(PersistenceError) as caught:
            pipeline.run_research(
                reference, output, AS_OF, config, ("P01",), progress=events.append
            )
    expected = (
        "save"
        if scenario == "begin"
        else "rollback"
        if scenario == "rollback"
        else "verify"
        if scenario == "reuse_verify"
        else "commit"
    )
    assert caught.value.phase == expected
    assert any(
        event.get("phase") == expected for event in events if event["stage"] == "persistence-failed"
    )
    assert "PRIVATE_" not in json.dumps(events)
    with original_connect(str(output), config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT count(*) FROM research.run").fetchone()[0] == int(reused)
        assert connection.execute("SELECT count(*) FROM research.result").fetchone()[0] == int(
            reused
        )
