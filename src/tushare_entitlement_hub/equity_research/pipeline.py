"""六产品依赖编排与原子运行边界，结果可按输入指纹确定性复用。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from ..product_foundation.contracts import Truth
from ..product_foundation.progress import ProgressCallback
from ..reference_storage import file_checksum
from ..standard import normalize_timestamp
from ..store import encode
from .adapter import load_inputs
from .binding_builder import binding_summary, verify_binding_audit
from .candidates import build_candidates
from .common import gate, make_result
from .consumption import attach_consumption, count_consumption, summarize_consumption
from .consumption_registry import (
    CONSUMPTION_VERSION,
    registry_fingerprint,
    validate_consumption_registry,
)
from .events import build_events
from .flow import build_flow
from .fundamental import build_fundamental
from .governance import build_governance
from .market import build_market
from .models import MARKET_ENTITY, PRODUCT_IDS, RULE_VERSION, ResearchConfig, ResearchResult
from .persistence_runtime import PersistenceError, PersistenceProgress
from .staging import research_snapshot
from .storage import initialize_tables, run_identity, save_results, verify_run

PRODUCT_ALIASES = dict(zip(("P01", "P04", "P06", "P07", "P03", "P13"), PRODUCT_IDS, strict=True))


def select_products(selection: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """展开产品别名和依赖并保持拓扑顺序；例：选择P03同时运行P04/P06/P07。

    空选择表示六项全部运行，未知产品直接拒绝，不默默跳过拼写错误。
    """
    requested = (
        {PRODUCT_ALIASES.get(name, name) for name in selection} if selection else set(PRODUCT_IDS)
    )
    if not requested <= set(PRODUCT_IDS):
        raise ValueError("unknown_research_product")
    expanded = set(requested)
    if requested & {"equity_research_candidates/v1", "event_thesis_review/v1"}:
        expanded.update(PRODUCT_IDS[1:4])
    return tuple(sorted(requested)), tuple(name for name in PRODUCT_IDS if name in expanded)


def run_research(
    reference_path: Path,
    product_path: Path,
    analysis_as_of: str,
    config: ResearchConfig | None = None,
    selection: tuple[str, ...] = (),
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """从只读参考库运行六产品或依赖闭包，原子保存并返回确定性运行摘要。

    例：相同reference/as-of/config再次运行直接校验复用，行数和摘要完全相同。
    缺绑定、缺字段和空输入仍输出明确降级结果；任何写入异常回滚本次全部记录。
    """
    if not reference_path.is_file():
        raise ValueError("reference_library_missing")
    if reference_path.resolve() == product_path.resolve() or (
        product_path.exists() and reference_path.samefile(product_path)
    ):
        raise ValueError("input_output_path_collision")
    cutoff = normalize_timestamp(analysis_as_of)
    configured = config or ResearchConfig()
    validate_consumption_registry()
    requested, products = select_products(selection)
    reference_hash = file_checksum(reference_path)
    verify_binding_audit(configured, reference_hash, cutoff)
    run_id = run_identity(reference_hash, cutoff, configured, requested)
    if progress is not None:
        progress({"stage": "start", "run_id": run_id, "products": list(products)})
    with research_snapshot(reference_path, product_path, run_id=run_id, progress=progress) as work:
        trace = PersistenceProgress("stage", progress, run_id)
        try:
            trace.operation = "open_snapshot"
            with duckdb.connect(
                str(work.path), config={"threads": 1, "memory_limit": "512MB"}
            ) as connection:
                trace.operation = "prepare_results"
                metadata, reused = _prepare_run(
                    connection,
                    reference_path,
                    reference_hash,
                    cutoff,
                    configured,
                    requested,
                    products,
                    run_id,
                    progress,
                )
                trace.operation = "checkpoint"
                connection.execute("CHECKPOINT")
            # 必须关闭写连接再重开：COMMIT释放事务本地存储，关闭连接同时释放索引/缓存预算。
            trace = PersistenceProgress("verify", progress, run_id)
            trace.operation = "open_snapshot"
            with duckdb.connect(
                str(work.path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
            ) as connection:
                trace.operation = "rebuild"
                try:
                    summary = verify_run(
                        connection,
                        metadata,
                        batch_size=configured.thresholds.fact_batch_size,
                        progress=progress,
                    )
                except Exception as error:
                    raise PersistenceError("verify", error) from error
                trace.operation = "close_snapshot"

            trace = PersistenceProgress("publish", progress, run_id)
            trace.operation = "reference_fingerprint"
            if file_checksum(reference_path) != reference_hash:
                raise ValueError("reference_changed_during_read")
            work.publish(trace)
        except PersistenceError:
            raise
        except Exception as error:
            # verify_run已报告校验内部错误；连接、checkpoint、发布失败由此补齐定位。
            raise trace.failure(error) from error
    if progress is not None:
        progress({"stage": "reused" if reused else "complete", "run_id": run_id})
    return summary


def _prepare_run(
    connection: duckdb.DuckDBPyConnection,
    reference_path: Path,
    reference_hash: str,
    cutoff: str,
    configured: ResearchConfig,
    requested: tuple[str, ...],
    products: tuple[str, ...],
    run_id: str,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], bool]:
    """只在私有快照内构建和提交研究数据，返回待完整验证的元数据及是否复用。

    例：已有run只更新私有current_run；新run按原事务写七表及摘要，任何异常撤销本次写入。
    提交不意味着对外可见，调用者必须关闭连接、重开验证后才能发布整个快照。
    """
    initialize_tables(connection)
    existing = connection.execute(
        "SELECT payload FROM research.run WHERE run_id=?", [run_id]
    ).fetchone()
    if existing is not None:
        trace = PersistenceProgress("commit", progress, run_id)
        trace.operation = "metadata"
        try:
            connection.execute(
                "INSERT OR REPLACE INTO research.metadata VALUES('current_run',?)", [encode(run_id)]
            )
        except Exception as error:
            raise trace.failure(error) from error
        return json.loads(existing[0]), True
    inputs, processing = load_inputs(reference_path, cutoff, run_id, configured, progress=progress)
    produced: dict[str, tuple[ResearchResult, ...]] = {}
    upstream: tuple[ResearchResult, ...] = ()
    for product in products:
        if product == PRODUCT_IDS[0]:
            results = build_market(inputs)
        elif product == PRODUCT_IDS[1]:
            results = build_fundamental(inputs)
        elif product == PRODUCT_IDS[2]:
            results = build_governance(inputs)
        elif product == PRODUCT_IDS[3]:
            results = build_flow(inputs)
        elif product == PRODUCT_IDS[4]:
            results = build_candidates(
                inputs,
                produced[PRODUCT_IDS[1]],
                produced[PRODUCT_IDS[2]],
                produced[PRODUCT_IDS[3]],
            )
        else:
            upstream = tuple(result for name in PRODUCT_IDS[1:4] for result in produced[name])
            results = build_events(inputs, upstream)
        if not results:
            results = (
                make_result(
                    inputs,
                    product,
                    MARKET_ENTITY,
                    gates=(gate("input_scope", Truth.UNKNOWN, "empty_input"),),
                    details={"mode": "observed_research", "reason": "empty_input"},
                ),
            )
        results = attach_consumption(inputs, product, results)
        produced[product] = results
        if progress is not None:
            progress({"stage": "product_complete", "product_id": product, "results": len(results)})
    if file_checksum(reference_path) != reference_hash:
        raise ValueError("reference_changed_during_read")
    all_results = tuple(result for name in products for result in produced[name])
    consumption_counts: dict[str, Counter[str]] = {}
    for result in all_results:
        count_consumption(consumption_counts, result)
    product_states = {
        name: dict(sorted(Counter(result.envelope.state.value for result in results).items()))
        for name, results in produced.items()
    }
    summary: dict[str, Any] = {
        "schema_version": RULE_VERSION,
        "run_id": run_id,
        "analysis_as_of": cutoff,
        "mode": "observed_research",
        "historical_pit_certified": False,
        "status": "incomplete"
        if any(
            result.envelope.state.value in {"blocked", "incomplete", "unsupported"}
            for result in all_results
        )
        else "complete",
        "requested_products": list(requested),
        "products": list(products),
        "product_states": product_states,
        "processing": processing,
        "consumption": summarize_consumption(consumption_counts, list(products)),
        "result_count": len(all_results),
    }
    if configured.binding_audit:
        summary["bindings"] = binding_summary(configured, summary["consumption"])
    active_phase = "save"
    begun = False
    transaction_trace = PersistenceProgress("commit", progress, run_id)
    try:
        connection.execute("BEGIN TRANSACTION")
        begun = True
        summary["counts"] = save_results(
            connection,
            all_results,
            configured.thresholds.fact_batch_size,
            progress=progress,
            run_id=run_id,
            staged=True,
        )
        # 保存完成后重建必须从数据库读取；释放上游、来源和结果对象，避免两套全量对象共存。
        produced.clear()
        del all_results, inputs, result, results, upstream
        active_phase = "commit"
        transaction_trace.operation = "metadata"
        metadata = {
            "run_id": run_id,
            "rule_version": RULE_VERSION,
            "consumption_version": CONSUMPTION_VERSION,
            "consumption_registry_fingerprint": registry_fingerprint(),
            "reference_sha256": reference_hash,
            "analysis_as_of": cutoff,
            "config": configured.to_dict(),
            "selection": list(requested),
            "products": list(products),
            "summary": summary,
        }
        connection.execute("INSERT INTO research.run VALUES (?,?)", [run_id, encode(metadata)])
        connection.execute(
            "INSERT OR REPLACE INTO research.metadata VALUES('current_run',?)", [encode(run_id)]
        )
        active_phase = "commit"
        transaction_trace.operation = "commit"
        connection.execute("COMMIT")
    except Exception as error:
        failure = PersistenceError(active_phase, error)
        if active_phase == "commit":
            transaction_trace.failure(error)
        elif not begun:
            trace = PersistenceProgress("save", progress, run_id)
            trace.operation = "begin"
            trace.failure(error)
        try:
            if begun:
                connection.execute("ROLLBACK")
        except Exception as rollback_error:
            rollback_trace = PersistenceProgress("rollback", progress, run_id)
            rollback_trace.failure(rollback_error)
        raise failure from error
    return metadata, False
