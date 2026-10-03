"""离线研究CLI：参数、阶段进度与机器摘要，业务逻辑全部委托引擎。"""

from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from ..store import encode
from .binding_builder import build_failure_code, write_research_config
from .binding_io import BuildLimits
from .event_parents import PARENT_FAILURE_CODES
from .models import ResearchConfig
from .pipeline import run_research
from .storage import RESULT_PAYLOAD_CAPACITY_ERROR, validate_research

research_app = typer.Typer(help="离线运行P01/P03/P04/P06/P07/P13股票研究产品。")


@research_app.command("build-config")
def build_config(
    reference: Annotated[Path, typer.Option(help="只读reference DuckDB")],
    output: Annotated[Path, typer.Option(help="ResearchConfig JSON输出")],
    as_of: Annotated[str, typer.Option("--as-of", help="带时区的绑定生成截止时间")],
    audit: Annotated[Path | None, typer.Option(help="独立审计JSON，默认config.audit.json")] = None,
    batch_size: Annotated[int, typer.Option(help="单批最多512行")] = 128,
    maximum_source_rows: Annotated[int, typer.Option(help="单接口扫描硬上限")] = 2_000_000,
    maximum_total_rows: Annotated[int, typer.Option(help="全库扫描硬上限")] = 5_000_000,
    maximum_entities: Annotated[int, typer.Option(help="主数据条数硬上限")] = 50_000,
) -> None:
    """显式生成研究绑定，成功0、失败1；stdout提供不含来源载荷的机器审计摘要。

    例：research build-config --reference ref.duckdb --output config.json --as-of
    2026-10-03T12:00:00Z，随后将config.json传入research run的--config选项。
    """
    try:
        summary = write_research_config(
            reference,
            output,
            as_of,
            audit_path=audit,
            limits=BuildLimits(
                batch_size=batch_size,
                maximum_source_rows=maximum_source_rows,
                maximum_total_rows=maximum_total_rows,
                maximum_entities=maximum_entities,
            ),
        )
    except (ValueError, OSError, duckdb.Error, TypeError, KeyError) as error:
        typer.echo(
            encode(
                {
                    "status": "failed",
                    "reason_codes": ["research_build_config_failed"],
                    "detail_code": build_failure_code(error),
                }
            )
        )
        raise typer.Exit(1) from None
    typer.echo(encode(summary))


def print_progress(event: dict[str, Any]) -> None:
    """将单阶段JSON即时输出stderr；例：stage=complete不污染stdout最终摘要。"""
    typer.echo(encode(event), err=True)


@research_app.command("initialize")
@research_app.command("run")
def run(
    reference: Annotated[Path, typer.Option(help="只读reference DuckDB")],
    output: Annotated[Path, typer.Option(help="独立products DuckDB")],
    as_of: Annotated[str, typer.Option("--as-of", help="带时区研究截止时间")],
    config: Annotated[Path | None, typer.Option(help="显式研究绑定及规则JSON")] = None,
    product: Annotated[
        list[str] | None, typer.Option("--product", help="P01等，可重复选择")
    ] = None,
) -> None:
    """初始化并运行指定产品，stdout为JSON，完整0、诚实降级2、错误1。

    例：research run --as-of ... --product P03将同时运行其三个依赖产品。
    initialize与run同一原子路径，运行过程不会访问凭据或在线接口。
    """
    try:
        inputs = ResearchConfig.from_json(config.read_text(encoding="utf-8")) if config else None
        summary = run_research(
            reference, output, as_of, inputs, tuple(product or ()), progress=print_progress
        )
    except (ValueError, OSError, duckdb.Error, TypeError, KeyError) as error:
        reason = (
            str(error)
            if isinstance(error, ValueError)
            and str(error) in {RESULT_PAYLOAD_CAPACITY_ERROR, *PARENT_FAILURE_CODES}
            else "research_run_failed"
        )
        typer.echo(encode({"status": "failed", "reason_codes": [reason]}))
        raise typer.Exit(1) from None
    typer.echo(encode(summary))
    if summary["status"] == "incomplete":
        raise typer.Exit(2)


@research_app.command("validate")
def validate(
    reference: Annotated[Path, typer.Option(help="当前只读参考库")],
    output: Annotated[Path, typer.Option(help="待验证产品库")],
) -> None:
    """只读完整验证产品库，成功0、失败1；例：unknown被篡改为false会失败。

    valid=true仅证明存储与契约一致，不能视为历史回测或策略有效性认证。
    """
    summary = validate_research(reference, output)
    typer.echo(encode(summary))
    if not summary["valid"]:
        raise typer.Exit(1)
