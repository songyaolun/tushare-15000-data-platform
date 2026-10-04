"""持久化阶段的安全进度和固定异常分类；不保留或输出数据库载荷及异常原文。"""

from __future__ import annotations

import re
from typing import Any

import duckdb

from ..product_foundation.progress import ProgressCallback

RESULT_PAYLOAD_CAPACITY_ERROR = "research_result_payload_capacity_exceeded"
SAFE_VALIDATION_CODES = frozenset(
    "research_primary_key_invalid research_result_reference_missing "
    "research_parent_reference_missing "
    "research_projection_mismatch research_result_context_mismatch research_summary_mismatch "
    "research_run_identity_mismatch research_product_selection_invalid "
    "research_bond_aggregation_audit_invalid research_consumption_missing "
    "research_consumption_assignment_mismatch research_consumption_contract_mismatch "
    "research_consumption_reference_missing research_consumption_field_quality_invalid "
    "research_consumption_observation_without_evidence research_consumption_reason_codes_invalid "
    "research_consumption_scope_mismatch research_projection_payload_capacity_exceeded "
    "invalid_batch_size input_output_path_collision reference_changed_during_read "
    "reference_catalog_missing research_partition_capacity_exceeded "
    "research_partition_payload_capacity_exceeded research_partition_layout_invalid "
    "research_partition_keys_invalid "
    "research_output_path_unsafe research_output_busy research_output_changed "
    "research_output_recovery_required research_output_lock_lost".split()
)


class ResultPayloadCapacityError(ValueError):
    """结果超限的内部诊断：只保留产品、实体和字节数，异常文本仍为固定安全码。"""

    def __init__(self, payload: Any, payload_bytes: int) -> None:
        """提取最小定位字段，不保存载荷；例：P06市场结果超限时可读取product_id和payload_bytes。

        元数据缺失或不是字符串时记None；CLI只返回固定码，不输出这些属性。
        """
        super().__init__(RESULT_PAYLOAD_CAPACITY_ERROR)
        envelope = payload.get("envelope") if isinstance(payload, dict) else None
        metadata = envelope if isinstance(envelope, dict) else {}
        product_id, entity_id = metadata.get("product_id"), metadata.get("entity_id")
        self.product_id = product_id if isinstance(product_id, str) else None
        self.entity_id = entity_id if isinstance(entity_id, str) else None
        self.payload_bytes = payload_bytes


def database_error_code(error: Exception) -> str | None:
    """把DuckDB类型及已复现的Fatal包装映射为固定码，不输出异常消息、SQL或路径。

    例：COMMIT内存失败后自动回滚再次失败得到fatal_rollback_memory；任意其他Fatal得到fatal。
    只匹配引擎固定开头，不能因载荷内偶然出现memory字样改变分类。
    """
    if isinstance(error, duckdb.FatalException):
        text = str(error)
        if text.startswith(
            "FATAL Error: Failed to rollback transaction. Cannot continue operation.\n"
            "Original Error: Out of Memory Error:"
        ):
            return "fatal_rollback_memory"
        if text.startswith(
            "FATAL Error: Failed to create checkpoint because of error: "
            "could not allocate block of size "
        ):
            return "fatal_checkpoint_memory"
        return "fatal"
    if isinstance(error, duckdb.InternalException):
        return "internal"
    if isinstance(error, duckdb.Error):
        return "duckdb_error"
    return None


def error_category(error: Exception) -> str:
    """按异常类型及引擎固定错误前缀分类，不回显原文；例：DuckDB内存错误得到memory。"""
    if database_error_code(error) in {"fatal_rollback_memory", "fatal_checkpoint_memory"}:
        return "memory"
    for types, category in (
        ((ResultPayloadCapacityError,), "capacity"),
        ((MemoryError, duckdb.OutOfMemoryException), "memory"),
        ((OSError, duckdb.IOException), "io"),
        ((duckdb.ConstraintException,), "constraint"),
        ((duckdb.ConversionException,), "conversion"),
        ((duckdb.TransactionException,), "transaction"),
        ((duckdb.Error,), "database"),
        ((ValueError, TypeError, KeyError), "validation"),
    ):
        if isinstance(error, types):
            return category
    return "unexpected"


class PersistenceError(ValueError):
    """对外只带固定原因，阶段/类别可安全读取；原异常由调用方通过异常链保留供内部诊断。"""

    def __init__(self, phase: str, error: Exception) -> None:
        """为准备、保存、验证、提交、回滚和发布错误赋固定码；例：verify内存错误为research_verify_memory_failed。"""
        if phase not in {"stage", "save", "verify", "commit", "rollback", "publish"}:
            raise ValueError("invalid_persistence_phase")
        self.phase = phase
        self.category = error_category(error)
        self.database_error_code = database_error_code(error)
        self.substep: str | None = None
        known = str(error) if isinstance(error, ValueError) else ""
        self.reason_code = (
            known
            if known in SAFE_VALIDATION_CODES or known == RESULT_PAYLOAD_CAPACITY_ERROR
            else f"research_{phase}_{self.category}_failed"
        )
        super().__init__(self.reason_code)


class PersistenceStepError(PersistenceError):
    """分段事务子步骤错误：保留安全类别和精确步骤，不把Fatal根因压成笼统database。"""

    def __init__(self, substep: str, error: Exception) -> None:
        """封装commit/checkpoint/reopen失败；例：提交Fatal内存错误得到research_save_batch_commit_memory_failed。"""
        if substep not in {"commit", "checkpoint", "reopen"}:
            raise ValueError("invalid_persistence_substep")
        super().__init__("save", error)
        self.substep = substep
        self.reason_code = f"research_save_batch_{substep}_{self.category}_failed"
        self.args = (self.reason_code,)


class PersistenceProgress:
    """只报告阶段、白名单表名和计数，开始/结束事件不进入持久化摘要及运行键。"""

    def __init__(self, phase: str, callback: ProgressCallback | None, run_id: str | None) -> None:
        """创建单阶段计数器；例：save从0开始，非SHA256运行标识不输出到事件。"""
        if phase not in {"stage", "save", "verify", "commit", "rollback", "publish"}:
            raise ValueError("invalid_persistence_phase")
        self.phase, self.callback = phase, callback
        self.run_id = (
            run_id if isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{64}", run_id) else None
        )
        self.operation = "start"
        self.table: str | None = None
        self.processed_rows = self.payload_bytes = self.completed_results = 0
        self._last_report = 0
        self.published = False
        self.substep: str | None = None
        self.transaction_state = "untracked"

    def emit(self, stage: str) -> None:
        """发送不共享内部状态的安全事件；例：save-complete表示事务内写完，不代表已提交。"""
        if self.callback is not None:
            self.callback(
                {
                    "stage": stage,
                    "run_id": self.run_id,
                    "operation": self.operation,
                    "substep": self.substep,
                    "transaction_state": self.transaction_state,
                    "table": self.table,
                    "processed_rows": self.processed_rows,
                    "payload_bytes": self.payload_bytes,
                    "completed_results": self.completed_results,
                    "committed": self.published,
                }
            )

    def step(self, operation: str) -> None:
        """切换固定内部子阶段并立即报告；例：verify从primary_keys切换到rebuild时输出进度。"""
        self.operation, self.table = operation, None
        self.emit(f"{self.phase}-progress")

    def advance(self, table: str, rows: int, payload_bytes: int) -> None:
        """累计实际写入/校验的投影数量，每4096行输出一次进度；例：4096个gate触发一次事件。"""
        self.table = table
        self.processed_rows += rows
        self.payload_bytes += payload_bytes
        if self.processed_rows - self._last_report >= 4096:
            self._last_report = self.processed_rows
            self.emit(f"{self.phase}-progress")

    def result_completed(self) -> None:
        """每16个完整处理的结果补发进度；例：宽结果仍可按结果数观察，不依赖完成整个表。"""
        self.completed_results += 1
        if self.completed_results % 16 == 0:
            self.emit(f"{self.phase}-progress")

    def failure(self, error: Exception) -> PersistenceError:
        """分类并报告失败，失败回调自身报错不能覆盖原异常；例：IO错误只输出io类别和固定码。"""
        failure = (
            error if isinstance(error, PersistenceError) else PersistenceError(self.phase, error)
        )
        if self.callback is not None:
            try:
                self.callback(
                    {
                        "stage": "persistence-failed",
                        "phase": failure.phase,
                        "category": failure.category,
                        "reason_code": failure.reason_code,
                        "operation": self.operation,
                        "substep": failure.substep or self.substep,
                        "database_error_code": failure.database_error_code,
                        "transaction_state": self.transaction_state,
                        "table": self.table,
                        "run_id": self.run_id,
                        "processed_rows": self.processed_rows,
                        "completed_results": self.completed_results,
                        "committed": self.published,
                    }
                )
            except Exception:
                pass
        return failure
