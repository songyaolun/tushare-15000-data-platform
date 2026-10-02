"""独立离线命令组，初始化和校验均不进入凭据或网络客户端。"""

from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from ..store import encode
from .bindings import FoundationBindings
from .config import FoundationPolicy, load_policy
from .reference import initialize_library
from .storage import validate_library


def _print_progress(event: dict[str, Any]) -> None:
    """将阶段计数写到 stderr，每行一个 JSON，stdout 保留最终摘要。

    输入 stage=start 事件，输出一行即时进度；不包含来源行内容或凭据。
    """
    typer.echo(encode(event), err=True)


foundation_app = typer.Typer(help="离线产品基础层：初始化共享库及校验，不执行具体研究引擎。")


@foundation_app.command("initialize")
def initialize(
    reference: Annotated[Path, typer.Option(help="只读 reference.duckdb 路径")],
    output: Annotated[Path, typer.Option(help="独立 products.duckdb 路径")],
    as_of: Annotated[str, typer.Option("--as-of", help="带时区研究截止时刻")],
    config: Annotated[Path | None, typer.Option(help="版本化阈值 JSON")] = None,
    bindings: Annotated[Path | None, typer.Option(help="显式实体、时间和单位认证 JSON")] = None,
) -> None:
    """初始化基础库，stderr 报阶段进度，stdout 输出单一 JSON；降级2、失败1。

    例如仅 daily 的参考库调用后输出 registered_apis=206、status=incomplete。
    """
    try:
        policy = (
            load_policy(config) if config else FoundationPolicy(version="foundation-default/v1")
        )
        inputs = FoundationBindings.from_json(bindings.read_text()) if bindings else None
        summary = initialize_library(
            reference, output, as_of, policy, inputs, progress=_print_progress
        )
    except (ValueError, OSError, duckdb.Error, TypeError):
        typer.echo(
            encode({"status": "failed", "reason_codes": ["foundation_initialization_failed"]})
        )
        raise typer.Exit(1) from None
    typer.echo(encode(summary))
    if summary["status"] == "incomplete":
        raise typer.Exit(2)


@foundation_app.command("validate")
def validate(
    reference: Annotated[Path, typer.Option(help="原只读 reference.duckdb")],
    output: Annotated[Path, typer.Option(help="待校验 products.duckdb")],
) -> None:
    """只读校验来源指纹及产品库，输出 JSON；valid=false 时退出 1。

    例如未变输入与产品库 → valid=true；该状态不等于研究数据完整或 PIT 已认证。
    """
    summary = validate_library(reference, output)
    typer.echo(encode(summary))
    if not summary["valid"]:
        raise typer.Exit(1)
