"""独立选股策略探索命令；仅使用已冻结的本地历史数据。"""

from pathlib import Path

import typer

from .dataset import prepare_features
from .runner import run_combination, run_discovery

discovery_app = typer.Typer(help="独立策略比较、有限邻域探索和资金组合归因。")


@discovery_app.command("prepare")
def prepare_command(source: Path, panel: Path, output: Path) -> None:
    """把既有逐证券技术特征转为研究面板；例如prepare FEATURES PANEL OUT打印清单。"""
    typer.echo(str(prepare_features(source, panel, output)))


@discovery_app.command("run")
def run_command(
    panel: Path,
    features: Path,
    raw: Path,
    output: Path,
    group: str = "all",
    selection: Path | None = None,
    limit: int | None = None,
) -> None:
    """运行独立策略组；例如run PANEL FEATURES RAW OUT --group trend打印报告路径。

    selection为具名配置选择JSON，neighbors=true追加有限邻域；limit仅用于烟测。
    """
    typer.echo(
        str(
            run_discovery(
                panel,
                features,
                raw,
                output,
                group=group,
                selection=selection,
                limit=limit,
            )
        )
    )


@discovery_app.command("combine")
def combine_command(panel: Path, features: Path, raw: Path, output: Path, selection: Path) -> None:
    """按真实子账户执行组合；例如combine PANEL FEATURES RAW OUT SELECTION打印指标路径。"""
    typer.echo(str(run_combination(panel, features, raw, output, selection)))
