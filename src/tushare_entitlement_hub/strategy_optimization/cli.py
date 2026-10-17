"""离线全样本策略优化入口，补采与搜索分开，默认不连接行情服务。"""

import json
from pathlib import Path

import typer

from ..client import HttpTransport
from ..credentials import configured_credential
from .collection import collect_concepts, collect_industries
from .concepts import attach_concepts
from .dataset import build_dataset
from .execution import ExitPolicy
from .search import run_search
from .strategies import StrategyDefinition
from .verification import verify_candidate

optimization_app = typer.Typer(help="全样本选股、买点和卖点优化，结果为历史重建研究。")


@optimization_app.command("prepare")
def prepare_command(history_database: Path, sector_root: Path, output: Path) -> None:
    """读取本地历史库和行业原始缓存，冻结五年特征；例如prepare DB RAW OUT打印清单路径。"""
    typer.echo(str(build_dataset(history_database, sector_root, output)))


@optimization_app.command("search")
def search_command(
    dataset: Path,
    output: Path,
    maximum_securities: int | None = None,
    accelerate: bool = False,
    concepts_only: bool = False,
) -> None:
    """离线运行全样本参数搜索；例如search FEATURES OUT打印全部尝试CSV路径。

    maximum_securities仅用于烟测，限制证券数的结果不能标记达到正式目标。
    """
    typer.echo(
        str(
            run_search(
                dataset,
                output,
                maximum_securities=maximum_securities,
                accelerate=accelerate,
                concepts_only=concepts_only,
            )
        )
    )


@optimization_app.command("verify")
def verify_command(
    dataset: Path,
    output: Path,
    family: str,
    context: str,
    take_profit: float,
    stop_loss: float,
    holding_sessions: int,
) -> None:
    """用独立状态机复算明确参数；例如verify目录及kdj_recovery参数打印最终清单路径。"""
    typer.echo(
        str(
            verify_candidate(
                dataset,
                output,
                StrategyDefinition(family, context),
                ExitPolicy(take_profit, stop_loss, holding_sessions),
            )
        )
    )


@optimization_app.command("attach-concepts")
def concepts_command(dataset: Path, concept_root: Path, output: Path) -> None:
    """将每日历史概念群动接入新数据集；例如attach-concepts BASE RAW OUT打印新清单。"""
    typer.echo(str(attach_concepts(dataset, concept_root, output)))


@optimization_app.command("collect-history")
def collect_history_command(
    credential_file: Path,
    calendar_file: Path,
    output: Path,
    live: bool = False,
) -> None:
    """显式联网补采两类主题历史；例：collect-history SECRET CAL OUT --live输出目录。

    未传live立即拒绝，不尝试读取凭据；凭据不写入响应、日志或数据指纹。
    """
    if not live:
        raise typer.BadParameter("history_collection_requires_explicit_live")
    calendar = json.loads(calendar_file.read_text(encoding="utf-8"))
    if isinstance(calendar, dict):
        calendar = calendar["trading_dates"]
    with HttpTransport(
        lambda: configured_credential(credential_file) or "", live=True
    ) as transport:
        collect_industries(transport, output / "sector-history")
        collect_concepts(transport, output / "concept-history", calendar)
    typer.echo(str(output))
