"""事务包装的内层错误分类，保留子步骤且不泄露引擎消息中的载荷。"""

import duckdb
import pytest

from tushare_entitlement_hub.equity_research.persistence_runtime import (
    PersistenceStepError,
    database_error_code,
)


def test_real_commit_constraint_is_classified_without_duplicate_value(tmp_path):
    """两个独立事务同时登记相同键，第二个COMMIT原生失败；分类必须识别约束且不输出重复值。"""
    path = str(tmp_path / "conflict.duckdb")
    options = {"threads": 1, "memory_limit": "512MB"}
    with duckdb.connect(path, config=options) as first:
        first.execute("CREATE TABLE sample(id VARCHAR PRIMARY KEY)")
        with duckdb.connect(path, config=options) as second:
            for connection in (first, second):
                connection.execute("BEGIN")
                connection.execute("INSERT INTO sample VALUES ('PRIVATE_SYNTHETIC_KEY')")
            first.execute("COMMIT")
            with pytest.raises(duckdb.TransactionException) as caught:
                second.execute("COMMIT")
            failure = PersistenceStepError("commit", caught.value)
            assert failure.database_error_code == "transaction_commit_constraint"
            assert failure.category == "constraint"
            assert str(failure) == "research_save_batch_commit_constraint_failed"
            assert "PRIVATE" not in str(failure)


@pytest.mark.parametrize(
    "message,code,category",
    [
        ("failed to pin block of size 256.0 KiB PRIVATE", "transaction_commit_memory", "memory"),
        ("could not allocate block of size 256 KiB PRIVATE", "transaction_commit_memory", "memory"),
        ("failed to allocate data of size 8.0 KiB PRIVATE", "transaction_commit_memory", "memory"),
        ("unexpected PRIVATE memory text", "transaction_commit_other", "transaction"),
    ],
)
def test_commit_raw_messages_use_fixed_engine_prefixes(message, code, category):
    """引擎提交包装剥去OutOfMemory类型前缀后仍识别固定内存模板；任意memory文本不能冒充根因。"""
    error = duckdb.TransactionException("TransactionContext Error: Failed to commit: " + message)
    failure = PersistenceStepError("commit", error)
    assert database_error_code(error) == code
    assert failure.category == category
    assert "PRIVATE" not in str(failure)


def test_other_transaction_failure_does_not_claim_commit_cause():
    """非提交事务错误保留transaction分类，不因消息片段包含内存或约束词而被误归因。"""
    error = duckdb.TransactionException("PRIVATE unrelated failed to pin block")
    assert database_error_code(error) == "transaction_other"
