"""全产品运行边界：私有分区提交、关闭重开验证后原子发布。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb

from ..equity_research.models import ResearchConfig
from ..equity_research.persistence_runtime import PersistenceProgress
from ..equity_research.projection import RESEARCH_TABLES, accumulate_records
from ..equity_research.staging import _version, research_snapshot
from ..equity_research.storage import _stored_records, initialize_tables, save_results
from ..product_foundation.progress import ProgressCallback
from ..reference_storage import file_checksum
from ..standard import normalize_timestamp
from ..store import encode
from .execution import ExecutionState, RunLimits, execute
from .registry import VERSION, fingerprint, select_products, validate_handlers
from .storage import execution_summary, identity, verify_saved

DEFAULT_LIMITS = RunLimits()


def run(
    reference: Path,
    output: Path,
    as_of: str,
    config: ResearchConfig,
    selection: tuple[str, ...] = (),
    *,
    holdings: tuple[dict[str, Any], ...] = (),
    limits: RunLimits = DEFAULT_LIMITS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """执行 206 消费依赖闭包并原子保存，失败保留旧库，相同输入校验后复用。

    例如 selection=('P05',) 仅发布预期产品；空选择发布全部十三产品。
    使用既有 512MB DuckDB、8MiB 结果门禁、16MiB 分段提交和磁盘唯一键登记。
    """
    from .bindings import verify_binding_audit

    if not reference.is_file():
        raise ValueError("reference_library_missing")
    if reference.resolve() == output.resolve() or (output.exists() and reference.samefile(output)):
        raise ValueError("input_output_path_collision")
    if len(holdings) > limits.maximum_parent_results:
        raise ValueError("portfolio_input_capacity_exceeded")
    cutoff = normalize_timestamp(as_of)
    products = select_products(selection)
    validate_handlers()
    reference_hash = file_checksum(reference)
    verify_binding_audit(config, reference_hash, cutoff)
    run_id = identity(reference_hash, cutoff, config, products, holdings, limits)
    saved_bytes = 0

    def bounded_progress(event: dict[str, Any]) -> None:
        """在私有写入阶段核算累计物理载荷；例如超过32GiB预算则拒绝发布。

        公共写入器在每4096物理行、16个结果及完成时报告，超限副本整体清理；
        单结果8MiB仍在其SQL调用前由公共门禁检查，不依赖本累计预算回调。
        """
        nonlocal saved_bytes
        saved_bytes = max(saved_bytes, int(event.get("payload_bytes", 0)))
        if saved_bytes > limits.maximum_output_bytes:
            raise ValueError("all_products_output_capacity_exceeded")
        if progress is not None:
            progress(event)

    with research_snapshot(reference, output, run_id=run_id, progress=progress) as work:
        with duckdb.connect(
            str(work.path), config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            initialize_tables(connection)
            existing = connection.execute(
                "SELECT payload FROM research.run WHERE run_id=?", [run_id]
            ).fetchone()
            current = connection.execute(
                "SELECT payload FROM research.metadata WHERE key='current_run'"
            ).fetchone()
            unchanged_current = existing is not None and current is not None and (
                json.loads(current[0]) == run_id
            )
            if existing is None:
                state = ExecutionState()
                connection.execute("BEGIN TRANSACTION")
                counts = save_results(
                    connection,
                    execute(
                        reference,
                        cutoff,
                        run_id,
                        config,
                        products,
                        holdings,
                        limits,
                        state,
                        progress,
                    ),
                    config.thresholds.fact_batch_size,
                    progress=bounded_progress,
                    run_id=run_id,
                    staged=True,
                )
                connection.execute("COMMIT")
                projections = accumulate_records(
                    _stored_records(connection, run_id, RESEARCH_TABLES)
                )
                summary = execution_summary(state) | {
                    "run_id": run_id,
                    "counts": counts,
                    "physical_json_bytes": saved_bytes,
                }
                metadata = {
                    "version": VERSION,
                    "registry_fingerprint": fingerprint(),
                    "run_id": run_id,
                    "reference_sha256": reference_hash,
                    "analysis_as_of": cutoff,
                    "config": config.to_dict(),
                    "products": list(products),
                    "holdings": list(holdings),
                    "limits": asdict(limits),
                    "summary": summary,
                    "projections": projections,
                }
                connection.execute(
                    "INSERT INTO research.run VALUES (?,?)", [run_id, encode(metadata)]
                )
            else:
                metadata = json.loads(existing[0])
            if not unchanged_current:
                connection.execute(
                    "INSERT OR REPLACE INTO research.metadata VALUES ('current_run',?)",
                    [encode(run_id)],
                )
            connection.execute("CHECKPOINT")
        with duckdb.connect(
            str(work.path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            summary = verify_saved(connection, metadata)
        if file_checksum(reference) != reference_hash:
            raise ValueError("reference_changed_during_read")
        if unchanged_current:
            if _version(work.output) != work.version or (
                work.guard is not None and work.guard.poll() is not None
            ):
                raise ValueError("research_output_changed")
        else:
            work.publish(PersistenceProgress("publish", progress, run_id))
    return summary
