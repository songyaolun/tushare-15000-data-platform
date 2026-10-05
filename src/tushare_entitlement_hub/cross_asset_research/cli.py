"""全产品离线命令，标准输出提供 JSON，任何失败都不回显来源载荷。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from ..equity_research.binding_builder import build_failure_code
from ..equity_research.binding_io import BuildLimits, check_paths, publish_files
from ..equity_research.cli import print_progress
from ..equity_research.models import ResearchConfig
from ..equity_research.persistence_runtime import SAFE_VALIDATION_CODES, PersistenceError
from ..equity_research.storage import _stored_records
from ..store import encode
from .execution import RunLimits
from .pipeline import run as run_products
from .registry import PRODUCTS
from .storage import validate as validate_products

all_products_app = typer.Typer(help="离线运行全部十三产品和206个真实接口消费者。")


@all_products_app.command("build-config")
def build_config(
    reference: Annotated[Path, typer.Option(help="只读 reference 数据库")],
    output: Annotated[Path, typer.Option(help="全产品配置 JSON")],
    as_of: Annotated[str, typer.Option("--as-of", help="带时区的研究截止时间")],
    audit: Annotated[Path | None, typer.Option(help="独立逐接口审计 JSON")] = None,
    maximum_source_rows: Annotated[int, typer.Option()] = 2_000_000,
    maximum_total_rows: Annotated[int, typer.Option()] = 5_000_000,
    maximum_entities: Annotated[
        int, typer.Option(help="跨市场主数据实体条数硬上限")
    ] = 500_000,
) -> None:
    """只读生成 206 接口绑定，先写审计再原子替换配置，配置作为提交标记。

    例如 research-all build-config --reference ref.duckdb --output config.json
    --as-of 2026-10-04T12:00:00Z；失败返回退出码 1，不产生部分配置。
    """
    from .bindings import build_config as build

    report = audit or output.with_suffix(".audit.json")
    try:
        check_paths(reference, output, report)
        result = build(
            reference,
            as_of,
            limits=BuildLimits(
                maximum_source_rows=maximum_source_rows,
                maximum_total_rows=maximum_total_rows,
                maximum_entities=maximum_entities,
            ),
        )
        publish_files(((report, encode(result.audit)), (output, result.config.to_json())))
        typer.echo(encode(result.audit))
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError) as error:
        typer.echo(
            encode(
                {
                    "status": "failed",
                    "reason_codes": ["all_products_build_config_failed"],
                    "detail_code": build_failure_code(error),
                }
            )
        )
        raise typer.Exit(1) from None


@all_products_app.command("run")
def run(
    reference: Annotated[Path, typer.Option(help="只读 reference 数据库")],
    output: Annotated[Path, typer.Option(help="独立全产品数据库")],
    as_of: Annotated[str, typer.Option("--as-of")],
    config: Annotated[Path, typer.Option(help="build-config 生成的配置")],
    product: Annotated[list[str] | None, typer.Option("--product")] = None,
    holdings: Annotated[Path | None, typer.Option(help="显式持仓 JSON 数组")] = None,
    maximum_results: Annotated[int, typer.Option()] = 100_000,
) -> None:
    """执行指定产品闭包，完整成功为 0、诚实降级为 2、执行失败为 1。

    例如 --product P05 只运行预期研究；不传 product 运行全部产品。
    holdings 文件由用户显式提供，不读取券商账户或自动发现本机持仓文件。
    """
    try:
        if holdings is not None and holdings.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("portfolio_input_capacity_exceeded")
        parsed: Any = json.loads(holdings.read_text(encoding="utf-8")) if holdings else []
        if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
            raise ValueError("holdings_array_required")
        summary = run_products(
            reference,
            output,
            as_of,
            ResearchConfig.from_json(config.read_text(encoding="utf-8")),
            tuple(product or ()),
            holdings=tuple(parsed),
            limits=RunLimits(maximum_results=maximum_results),
            progress=print_progress,
        )
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError) as error:
        safe_reason = (
            str(error)
            if isinstance(error, ValueError) and str(error) in SAFE_VALIDATION_CODES
            else "all_products_run_failed"
        )
        failure: dict[str, Any] = {"status": "failed", "reason_codes": [safe_reason]}
        if isinstance(error, PersistenceError):
            failure.update(
                reason_codes=[error.reason_code],
                phase=error.phase,
                category=error.category,
                database_error_code=error.database_error_code,
            )
        typer.echo(encode(failure))
        raise typer.Exit(1) from None
    typer.echo(encode(summary))
    if summary["status"] == "incomplete":
        raise typer.Exit(2)


@all_products_app.command("validate")
def validate(
    reference: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """重建投影并从当前 reference 重算，返回 valid 与仍然存在的业务缺口。

    例如修改一个已存特征后运行 validate，退出码为 1；unknown 本身不导致结构验证失败。
    """
    result = validate_products(reference, output)
    typer.echo(encode(result))
    if not result["valid"]:
        raise typer.Exit(1)


@all_products_app.command("results")
def results(
    output: Annotated[Path, typer.Option()],
    product: Annotated[str, typer.Option()],
    limit: Annotated[int, typer.Option(min=1, max=512)] = 20,
) -> None:
    """按产品流式输出当前批次 JSONL，默认最多 20 条，不把数据库全部读入内存。

    例如 --product P08 --limit 5 输出最多五条 ETF 研究结果，缺结果时输出零行。
    """
    try:
        product_id = PRODUCTS.get(product, product)
        if product_id not in PRODUCTS.values():
            raise ValueError("unknown_all_products_selection")
        with duckdb.connect(
            str(output), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            current = connection.execute(
                "SELECT payload FROM research.metadata WHERE key='current_run'"
            ).fetchone()
            if current is None:
                raise ValueError("all_products_current_run_missing")
            emitted = 0
            for row in _stored_records(connection, json.loads(current[0]), ("result",)):
                if json.loads(row.payload)["envelope"]["product_id"] == product_id:
                    typer.echo(row.payload)
                    emitted += 1
                    if emitted == limit:
                        break
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError):
        typer.echo(encode({"status": "failed", "reason_codes": ["all_products_read_failed"]}))
        raise typer.Exit(1) from None


@all_products_app.command("audit")
def audit(output: Annotated[Path, typer.Option()]) -> None:
    """读取当前批次逐接口消费审计，保持 observed 与 unresolved 两个独立维度。

    例如 audit --output products.duckdb 输出 206 项状态及各原因计数，不输出原始行情。
    """
    try:
        with duckdb.connect(
            str(output), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            row = connection.execute(
                "SELECT run.payload FROM research.run run JOIN research.metadata current "
                "ON run.run_id=(current.payload->>'$') WHERE current.key='current_run'"
            ).fetchone()
            if row is None:
                raise ValueError("all_products_current_run_missing")
            typer.echo(encode(json.loads(row[0])["summary"]["consumption"]))
    except (ValueError, OSError, duckdb.Error, KeyError, TypeError):
        typer.echo(encode({"status": "failed", "reason_codes": ["all_products_read_failed"]}))
        raise typer.Exit(1) from None
