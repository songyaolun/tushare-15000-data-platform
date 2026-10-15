"""复用七表分区与磁盘键校验，全产品验证同时支持从 reference 重新计算。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb

from ..equity_research.consumption import validate_result_consumption
from ..equity_research.models import RULE_VERSION, ResearchConfig, ResearchResult
from ..equity_research.persistence_runtime import PersistenceProgress
from ..equity_research.projection import (
    RESEARCH_TABLES,
    EncodedRecord,
    accumulate_records,
    project_result,
)
from ..equity_research.storage import _stored_records, _verify_keys
from ..reference_storage import reference_fingerprint
from ..store import digest
from .execution import ExecutionState, RunLimits, execute
from .registry import NEW_PRODUCTS, PRODUCTS, VERSION, fingerprint, select_products


def identity(
    reference_hash: str,
    cutoff: str,
    config: ResearchConfig,
    products: tuple[str, ...],
    holdings: tuple[dict[str, Any], ...],
    limits: RunLimits,
) -> str:
    """把参考指纹、配置、持仓、产品依赖闭包和容量预算纳入确定性运行身份。

    相同输入得到相同 run_id；将持仓权重从 0.2 改成 0.3 会生成不同身份。
    """
    return digest(
        [
            VERSION,
            fingerprint(),
            reference_hash,
            cutoff,
            config.to_dict(),
            products,
            holdings,
            asdict(limits),
        ]
    )


def execution_summary(state: ExecutionState) -> dict[str, Any]:
    """从实际执行计数生成稳定摘要；例如有 unknown 结果时 status 为 incomplete。

    该摘要不含耗时和机器路径，重复执行逐字相同；历史 PIT 始终不自动认证。
    """
    return {
        "version": VERSION,
        "status": "incomplete"
        if state.consumption.get("unresolved", 0)
        or any(
            states["incomplete"] or states["blocked"] or states["unsupported"]
            for states in state.states.values()
        )
        else "complete",
        "historical_pit_certified": False,
        "result_count": state.result_count,
        "product_states": {
            name: dict(sorted(states.items())) for name, states in sorted(state.states.items())
        },
        "processing": state.processing,
        "consumption": state.consumption,
    }


def verify_saved(connection: duckdb.DuckDBPyConnection, metadata: dict[str, Any]) -> dict[str, Any]:
    """逐条重建模型及全部物理投影，验证跨段唯一键、父引用和运行上下文。

    输入被删除一条 feature 的库抛 projection_mismatch；完整库返回原运行摘要。
    不构造全局 Python 键集合，每次只读一条最多 8MiB 的存储记录。
    """
    config = ResearchConfig.from_dict(metadata["config"])
    products = tuple(metadata["products"])
    limits = RunLimits(**metadata["limits"])
    if products != select_products(products):
        raise ValueError("all_products_selection_invalid")
    run_id = identity(
        metadata["reference_sha256"],
        metadata["analysis_as_of"],
        config,
        products,
        tuple(metadata["holdings"]),
        limits,
    )
    if (
        metadata["version"] != VERSION
        or metadata["registry_fingerprint"] != fingerprint()
        or (metadata["run_id"] != run_id)
    ):
        raise ValueError("all_products_identity_mismatch")
    _verify_keys(connection, run_id, PersistenceProgress("verify", None, run_id))
    actual = {table: [0, 0] for table in RESEARCH_TABLES}
    states: dict[str, dict[str, int]] = {}
    physical_bytes = 0

    def rebuilt() -> Iterator[EncodedRecord]:
        """将每条已存结果独立重建后投影；例如同一 gate 被篡改会改变预期摘要。"""
        nonlocal physical_bytes
        for stored in _stored_records(connection, run_id, ("result",)):
            physical_bytes += len(stored.payload.encode("utf-8"))
            accumulate_records((stored,), actual)
            result = ResearchResult.from_json(stored.payload)
            product = next(
                (name for name, value in PRODUCTS.items() if value == result.envelope.product_id),
                None,
            )
            if (
                product is None
                or product not in products
                or result.envelope.run_id != run_id
                or result.envelope.model_version != RULE_VERSION
                or result.envelope.analysis_as_of != metadata["analysis_as_of"]
            ):
                raise ValueError("all_products_context_mismatch")
            if product not in NEW_PRODUCTS:
                validate_result_consumption(result)
            counts = states.setdefault(product, {})
            status = result.envelope.state.value
            counts[status] = counts.get(status, 0) + 1
            yield from project_result(result)

    expected = accumulate_records(rebuilt())
    for stored in _stored_records(connection, run_id, RESEARCH_TABLES[1:]):
        physical_bytes += len(stored.payload.encode("utf-8"))
        accumulate_records((stored,), actual)
    summary = metadata["summary"]
    if (
        expected != actual
        or actual != metadata["projections"]
        or summary["counts"] != {table: value[0] for table, value in actual.items()}
    ):
        raise ValueError("all_products_projection_mismatch")
    if (
        summary["result_count"] != actual["result"][0]
        or summary["product_states"] != states
        or (set(states) != set(products))
        or summary["physical_json_bytes"] != physical_bytes
        or physical_bytes > limits.maximum_output_bytes
        or actual["result"][0] > limits.maximum_results
    ):
        raise ValueError("all_products_summary_mismatch")
    return dict(summary)


def replay(reference: Path, metadata: dict[str, Any]) -> None:
    """从原参考重新执行所有消费者，并逐投影摘要与消费摘要核对，而非信任已存结果。

    例如修改一条期货压力指标并重写子表，物理一致性可能通过，本次重算仍拒绝。
    仅用于当前参考指纹对应批次，历史参考文件缺失时不能声称完成来源重算。
    """
    from .bindings import verify_binding_audit

    config = ResearchConfig.from_dict(metadata["config"])
    verify_binding_audit(config, metadata["reference_sha256"], metadata["analysis_as_of"])
    state = ExecutionState()

    def projections() -> Iterator[EncodedRecord]:
        """只保留当前结果的投影；例如十万结果重算不创建十万个模型的列表。"""
        for result in execute(
            reference,
            metadata["analysis_as_of"],
            metadata["run_id"],
            config,
            tuple(metadata["products"]),
            tuple(metadata["holdings"]),
            RunLimits(**metadata["limits"]),
            state,
        ):
            yield from project_result(result)

    rebuilt = accumulate_records(projections())
    summary = metadata["summary"]
    if rebuilt != metadata["projections"] or any(
        summary.get(key) != value for key, value in execution_summary(state).items()
    ):
        raise ValueError("all_products_replay_mismatch")


def validate(reference: Path, output: Path) -> dict[str, Any]:
    """只读核验全部批次，当前全产品批次必须从 reference 完整重算。

    正常输入返回 valid=true；未知、缺数据可以通过完整性验证，但仍保留 unresolved。
    错误只返回固定码，绝不输出提供方原始内容、凭据或数据库连接文本。
    """
    try:
        if reference.resolve() == output.resolve():
            raise ValueError("input_output_path_collision")
        reference_hash = reference_fingerprint(reference)
        summaries = []
        with duckdb.connect(
            str(output), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            current_row = connection.execute(
                "SELECT payload FROM research.metadata WHERE key='current_run'"
            ).fetchone()
            if current_row is None:
                raise ValueError("all_products_current_run_missing")
            current = json.loads(current_row[0])
            runs = connection.execute(
                "SELECT run_id,payload FROM research.run ORDER BY run_id"
            ).fetchall()
            if not runs or current not in {row[0] for row in runs}:
                raise ValueError("all_products_current_run_missing")
            for run_id, payload in runs:
                metadata = json.loads(payload)
                if metadata["run_id"] != run_id:
                    raise ValueError("all_products_identity_mismatch")
                if metadata.get("version") != VERSION:
                    from ..equity_research.storage import verify_run

                    summaries.append(verify_run(connection, metadata))
                    continue
                summaries.append(verify_saved(connection, metadata))
                if run_id == current:
                    if metadata["reference_sha256"] != reference_hash:
                        raise ValueError("all_products_reference_changed")
                    replay(reference, metadata)
            for table in RESEARCH_TABLES:
                orphan = connection.execute(
                    f'SELECT count(*) FROM research."{table}" child '
                    "LEFT JOIN research.run parent USING(run_id) WHERE parent.run_id IS NULL"
                ).fetchone()
                if orphan is None or orphan[0]:
                    raise ValueError("all_products_orphan_record")
        if reference_fingerprint(reference) != reference_hash:
            raise ValueError("all_products_reference_changed")
        return {
            "valid": True,
            "version": VERSION,
            "runs": summaries,
            "historical_pit_certified": False,
        }
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError):
        return {
            "valid": False,
            "reason_codes": ["all_products_validation_failed"],
            "historical_pit_certified": False,
        }
