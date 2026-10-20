"""市场阶段与组合Alpha研究命令，采集必须显式联网，分析默认离线。"""

import json
from pathlib import Path

import typer

from ..client import HttpTransport
from ..credentials import configured_credential
from .collection import collect_partition, collection_jobs
from .panel import prepare_panel
from .runner import run_research

regime_app = typer.Typer(help="市场阶段、现金组合回放与Beta/Alpha对照。")


@regime_app.command("collect")
def collect_command(credential_file: Path, dataset: Path, output: Path, live: bool = False) -> None:
    """显式补采指数、市值、分红并可恢复；例如collect SECRET FEATURES RAW --live。

    未启用live时不读取凭据、不请求网络；所有响应先写公开原文缓存。
    """
    if not live:
        raise typer.BadParameter("regime_collection_requires_explicit_live")
    calendar = json.loads((dataset / "dataset.json").read_text())["calendar"]
    with HttpTransport(
        lambda: configured_credential(credential_file) or "", live=True
    ) as transport:
        for job in collection_jobs(calendar):
            collect_partition(transport, output, job)
    typer.echo(str(output))


@regime_app.command("prepare")
def prepare_command(dataset: Path, raw: Path, output: Path) -> None:
    """冻结因果研究面板；例如prepare FEATURES RAW PANEL打印清单路径。"""
    typer.echo(str(prepare_panel(dataset, raw, output)))


@regime_app.command("run")
def run_command(panel: Path, raw: Path, output: Path, limit: int | None = None) -> None:
    """离线执行组合对照及邻域搜索；例如run PANEL RAW OUT打印研究报告路径。

    limit仅用于烟测，有限运行不得标记为完整五年网格搜索。
    """
    typer.echo(str(run_research(panel, raw, output, limit=limit)))
