"""离线研究CLI：参数、阶段进度与机器摘要，业务逻辑全部委托引擎。"""

from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from ..store import encode
from .models import ResearchConfig
from .pipeline import run_research
from .storage import validate_research

research_app = typer.Typer(help="离线运行P01/P03/P04/P06/P07/P13股票研究产品。")


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
    except (ValueError, OSError, duckdb.Error, TypeError, KeyError):
        typer.echo(encode({"status": "failed", "reason_codes": ["research_run_failed"]}))
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
