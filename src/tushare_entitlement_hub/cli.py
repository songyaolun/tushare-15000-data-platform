"""所有网络动作均显式开启的命令行入口，默认操作本地 fixture 和 DuckDB。"""

from __future__ import annotations

import getpass
import json
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import duckdb
import typer

from .catalog import fetch_documentation, sync_catalog
from .client import Client, FakeTransport, HttpTransport, ProviderError
from .coverage import coverage_report
from .credentials import configured_credential
from .cross_asset_research.cli import all_products_app
from .equity_research.cli import research_app
from .ingest import plan_tasks, probe, run_task
from .models import CatalogEntry
from .product_foundation.cli import foundation_app
from .products import build_products
from .quality import audit_rows
from .raw import RawStore, assert_public
from .reference import build_reference, rebuild_reference
from .signal_research.cli import signal_research_app
from .store import Store, encode
from .strategy_optimization.cli import optimization_app
from .survey import survey

app = typer.Typer(help="Tushare 权益总账、离线采集与信息产品。", no_args_is_help=True)
catalog_app = typer.Typer(help="发现接口并管理权益总账。")
capability_app = typer.Typer(help="按最小政策探测能力。")
ingest_app = typer.Typer(help="生成和执行可恢复采集任务。")
backfill_app = typer.Typer(help="制定历史回填计划。")
quality_app = typer.Typer(help="审计已存标准数据。")
coverage_app = typer.Typer(help="展示权益利用率和覆盖债务。")
products_app = typer.Typer(help="构建可审计的 Markdown/JSON 产品。")
reference_app = typer.Typer(help="正式采集并发布独立于产品的本地数据参考。")
for command_name, application in (
    ("catalog", catalog_app),
    ("capability", capability_app),
    ("ingest", ingest_app),
    ("backfill", backfill_app),
    ("quality", quality_app),
    ("coverage", coverage_app),
    ("products", products_app),
    ("reference", reference_app),
    ("product-foundation", foundation_app),
    ("research", research_app),
    ("research-all", all_products_app),
    ("signal-research", signal_research_app),
    ("strategy-optimize", optimization_app),
):
    app.add_typer(application, name=command_name)


def timestamp(value: str | None) -> str:
    """校验观察时间并返回 UTC，拒绝非法、缺少时区或晚于当前时刻的显式值。

    参数为带 T 分隔符和时区的 ISO 8601 时间或 None；None 返回当前 UTC 时间。
    返回值统一以 Z 结尾；无效输入抛出固定 ValueError，交由命令现有错误边界处理。
    示例：timestamp('2024-01-03T08:00:00+08:00') → '2024-01-03T00:00:00Z'。
    """
    current_time = datetime.now(UTC)
    if value is None:
        return current_time.isoformat().replace("+00:00", "Z")
    try:
        if "T" not in value:
            raise ValueError("invalid_observed_at")
        observed_time = datetime.fromisoformat(value)
        if observed_time.tzinfo is None or observed_time.utcoffset() is None:
            raise ValueError("invalid_observed_at")
        observed_time = observed_time.astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid_observed_at") from None
    if observed_time > current_time:
        raise ValueError("future_observed_at")
    return observed_time.isoformat().replace("+00:00", "Z")


@app.callback()
def configure(
    ctx: typer.Context,
    database: Annotated[Path, typer.Option("--db", help="本地 DuckDB 文件")] = Path(
        "data/hub.duckdb"
    ),
) -> None:
    """保存数据库路径，不在 --help 时创建文件；--db demo.duckdb 绑定本次调用。

    参数为 Typer 上下文和 Path；返回 None，只保存本次命令设置，不打开数据库。
    示例：configure(ctx, Path('demo.duckdb')) 后 ctx.obj['database'] 为该路径。
    """
    ctx.obj = {"database": database}


@contextmanager
def open_store(ctx: typer.Context) -> Iterator[Store]:
    """为一次命令打开并可靠关闭数据库；输入 Typer 上下文，产出 Store。

    参数是已配置的 Typer 上下文；上下文内产出 Store，退出时可靠关闭连接。
    示例：with open_store(ctx) as store 可调用 store.catalog()；失败输出有限错误码。
    """
    store = Store(ctx.obj["database"])
    try:
        yield store
    except (ValueError, OSError, ProviderError, duckdb.Error):
        typer.echo(encode({"error": "operation_failed", "hint": "请检查政策、fixture 和任务状态"}))
        raise typer.Exit(1) from None
    finally:
        store.close()


def find_entry(store: Store, api_name: str) -> CatalogEntry:
    """按准确名称查目录；daily 存在返回记录，不存在抛固定错误码。

    参数为 Store 和准确接口名，返回唯一 CatalogEntry；不接受模糊匹配。
    示例：目录仅有 daily 时 find_entry(store, 'daily').api_name → 'daily'。
    """
    entries = [entry for entry in store.catalog() if entry.api_name == api_name]
    if not entries:
        raise ValueError("unknown_api")
    return entries[0]


def read_json(path: Path) -> Any:
    """读取经过凭据字段检查的本地 JSON；输入 responses.json，输出其数据对象。

    参数是本地 UTF-8 JSON 路径，返回解码后的对象；凭据字段直接拒绝。
    示例：文件内容为 [] 时 read_json(path) → []。
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    assert_public(value)
    return value


def credential_prompt() -> str:
    """只在显式 live 传输最后发送前交互读取凭据；默认离线流程不调用此函数。

    示例：调用方显式执行 live 请求时，安全输入的值仅返回给当前 HTTPS 发送步骤。
    """
    return configured_credential() or getpass.getpass("Tushare 凭据（输入不回显）：")


@contextmanager
def open_client(
    entry: CatalogEntry,
    raw_directory: Path,
    fixture: Path | None,
    live: bool,
    account: str = "local",
    credential_file: Path | None = None,
) -> Iterator[Client]:
    """选择互斥的 fixture 或显式 live 传输；fixture daily.json 产生零网络 Client。

    参数为目录、原始目录、可选fixture、live标志与公开账户别名；产出 Client。
    示例：fixture指定且live=False时使用FakeTransport；两者同时指定抛ValueError。
    """
    if live == (fixture is not None):
        raise ValueError("choose_exactly_one_fixture_or_live")
    raw = RawStore(raw_directory)
    if fixture is not None:
        content = read_json(fixture)
        responses = content[entry.api_name] if isinstance(content, dict) else content
        yield Client(FakeTransport(responses), raw, account="fixture")
    else:
        def credential_provider() -> str:
            """在实际发送时读取显式凭据文件或既有环境；例如未指定文件则使用交互回调。"""
            if credential_file is not None:
                return configured_credential(credential_file) or ""
            return credential_prompt()

        with HttpTransport(credential_provider, live=True) as transport:
            yield Client(
                transport, raw, account=account, rate_per_minute=entry.frequency_per_minute or 60
            )


@catalog_app.command("sync")
def catalog_sync(
    ctx: typer.Context,
    fixture: Annotated[Path | None, typer.Option(help="HTML/Markdown 文件或目录")] = None,
    online: Annotated[bool, typer.Option(help="显式读取公开官方文档")] = False,
    destination: Path = Path("data/documentation"),
    observed_at: str | None = None,
) -> None:
    """同步公开文档总账；catalog sync --fixture fixtures/catalog/official 离线登记接口。

    参数指定上下文与互斥的离线目录/online开关；返回None并输出登记统计JSON。
    示例：精简演示目录输出 registered=11；未选择来源时安全退出且不联网。
    """
    with open_store(ctx) as store:
        if online == (fixture is not None):
            raise ValueError("choose_documentation_source")
        if online:
            paths = fetch_documentation(destination)
        elif fixture is not None and fixture.is_dir():
            paths = sorted(
                path
                for path in fixture.rglob("*")
                if path.suffix.lower() in {".html", ".md"} or path.name == "index.json"
            )
        else:
            paths = [fixture] if fixture is not None else []
        entries = sync_catalog(paths, timestamp(observed_at))
        store.save_catalog(entries)
        typer.echo(
            encode(
                {
                    "registered": len(entries),
                    "needs_review": sum(entry.needs_review for entry in entries),
                }
            )
        )


@catalog_app.command("list")
def catalog_list(ctx: typer.Context, lifecycle: str | None = None) -> None:
    """显示总账及可选生命周期过滤；catalog list --lifecycle unclassified 展示待归类接口。

    参数指定上下文及可选生命周期；返回None，标准输出为目录记录数组。
    示例：空库 catalog list 输出 []。
    """
    with open_store(ctx) as store:
        typer.echo(
            encode(
                [
                    asdict(entry)
                    for entry in store.catalog()
                    if lifecycle is None or entry.lifecycle == lifecycle
                ]
            )
        )


@capability_app.command("probe")
def capability_probe(
    ctx: typer.Context,
    api_name: str,
    fixture: Path | None = None,
    live: bool = False,
    observed_at: str | None = None,
    account: str = "local",
    raw_directory: Path = Path("data/raw"),
    credential_file: Path | None = None,
    parameters: Path | None = None,
) -> None:
    """按政策探活；capability probe daily --fixture responses.json 仅记录离线证据。

    参数指定接口、fixture或live、时间及原始目录；返回None并输出探活分类JSON。
    示例：daily成功单行fixture输出 status='available'、source='fixture'。
    """
    with open_store(ctx) as store:
        entry = find_entry(store, api_name)
        with open_client(entry, raw_directory, fixture, live, account, credential_file) as client:
            typer.echo(
                encode(
                    probe(
                        entry, client, store, timestamp(observed_at), "live" if live else "fixture",
                        read_json(parameters) if parameters else None,
                    )
                )
            )


@capability_app.command("survey")
def capability_survey(
    ctx: typer.Context,
    live: bool = False,
    credential_file: Path | None = None,
    output: Path = Path("data/capability-survey"),
    parameters: Path | None = None,
    account: str = "local",
    refresh: bool = False,
    apis: str | None = None,
) -> None:
    """实测公共只读目录并保存样本和结果；须显式 --live，写接口始终排除。

    参数含既有凭据路径、样本输出目录、按API参数映射及可选逗号分隔接口列表。
    示例：capability survey --live --apis trade_cal,namechange 输出逐项状态及 report.json。
    默认复用同账户同政策的成功证据；--refresh 强制重新请求。
    """
    with open_store(ctx) as store:
        if not live:
            raise ValueError("live_opt_in_required")
        entries = store.catalog()
        if not entries:
            raise ValueError("catalog_required")
        selected = apis.split(",") if apis else None
        if selected and set(selected) - {entry.api_name for entry in entries}:
            raise ValueError("unknown_api")
        with open_client(
            entries[0], output / "raw", None, True, account, credential_file
        ) as client:
            report = survey(
                store, client, output, lambda: timestamp(None),
                parameters=read_json(parameters) if parameters else None,
                refresh=refresh, api_names=selected,
            )
            typer.echo(encode({"counts": report["counts"], "report": str(output / "report.json")}))


@reference_app.command("build")
def reference_build(
    ctx: typer.Context, live: bool = False, credential_file: Path | None = None,
    output: Path = Path("data/reference"), parameters: Path | None = None,
    account: str = "local", refresh: bool = False, apis: str | None = None,
) -> None:
    """显式执行完整声明范围采集并导出通用本地数据；不依赖十类产品。

    输入凭据路径、输出位置、完整参数映射和可选接口列表，返回None并输出发布摘要。
    示例：reference build --live --apis daily --parameters ranges.json 生成Parquet和DuckDB。
    参数映射完整替换政策样例，清单始终保留实际请求范围，不声明未知历史覆盖。
    """
    if not live:
        raise typer.BadParameter("live_opt_in_required")
    with open_store(ctx) as store:
        entries = store.catalog()
        if not entries:
            raise ValueError("catalog_required")
        selected = apis.split(",") if apis else None
        if selected and set(selected) - {entry.api_name for entry in entries}:
            raise ValueError("unknown_api")
        with open_client(
            entries[0], output / "raw", None, True, account, credential_file
        ) as client:
            result = build_reference(
                store, client, output, lambda: timestamp(None),
                read_json(parameters) if parameters else None, selected, refresh,
            )
            typer.echo(encode({"counts": result["counts"], "database": result["database"]}))


@reference_app.command("rebuild")
def reference_rebuild(
    output: Path = Path("data/reference"), apis: str | None = None,
) -> None:
    """校验已有原始页并离线重建通用数据；例如reference rebuild --output data/reference。

    输入发布目录，返回None并输出计数。不会读取凭据、发起网络请求或更新观察时间。
    """
    result = rebuild_reference(output, api_names=apis.split(",") if apis else None)
    typer.echo(encode({"counts": result["counts"], "database": result["database"]}))


@ingest_app.command("plan")
def ingest_plan(
    ctx: typer.Context,
    api_name: str,
    start: str,
    end: str,
    output: Path = Path("data/plan.json"),
    mode: str = "initialize",
    observed_at: str | None = None,
    live: bool = False,
    parameters: Path | None = None,
    account: str = "local",
) -> None:
    """生成可复用任务计划；ingest plan daily 2024-01-02 2024-01-03 输出两个日期任务。

    参数指定接口、日期区间、输出文件、模式及可选分片；返回None并登记与保存计划。
    示例：daily的两日区间生成 tasks=2；mode='backfill'保留回填用途。
    """
    with open_store(ctx) as store:
        plans = plan_tasks(
            find_entry(store, api_name),
            start,
            end,
            mode,
            timestamp(observed_at),
            store,
            "live" if live else "fixture",
            read_json(parameters) if parameters else None,
            account,
        )
        store.save_plan(plans, timestamp(observed_at))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encode(plans) + "\n", encoding="utf-8")
        typer.echo(encode({"tasks": len(plans), "output": str(output)}))


@backfill_app.command("plan")
def backfill_plan(
    ctx: typer.Context,
    api_name: str,
    start: str,
    end: str,
    output: Path = Path("data/backfill.json"),
    observed_at: str | None = None,
) -> None:
    """生成历史回补任务；backfill plan daily 2024-01-02 2024-01-03 设置 mode=backfill。

    参数为上下文、接口、日期区间和目标文件；返回None并输出历史计划统计。
    示例：daily同一天区间生成一个 mode='backfill' 的任务。
    """
    ingest_plan(ctx, api_name, start, end, output, "backfill", observed_at)


@ingest_app.command("run")
def ingest_run(
    ctx: typer.Context,
    plan: Path,
    fixture: Path | None = None,
    live: bool = False,
    raw_directory: Path = Path("data/raw"),
    observed_at: str | None = None,
    references: Path | None = None,
    account: str = "local",
    credential_file: Path | None = None,
) -> None:
    """执行已保存任务；ingest run plan.json --fixture responses.json 可从失败任务恢复。

    参数包含计划路径、互斥传输来源、原始目录与可选参考；返回None并输出任务结果。
    示例：已提交计划再次执行输出 state='already_committed'，不再次请求数据。
    """
    with open_store(ctx) as store:
        tasks = read_json(plan)
        store.save_plan(tasks, timestamp(observed_at))
        reference = read_json(references) if references else {}
        results = []
        with ExitStack() as stack:
            clients: dict[str, Client] = {}
            for task in tasks:
                if task["source"] != ("live" if live else "fixture"):
                    raise ValueError("plan_source_mismatch")
                entry = find_entry(store, task["api_name"])
                if entry.api_name not in clients:
                    clients[entry.api_name] = stack.enter_context(
                        open_client(entry, raw_directory, fixture, live, account, credential_file)
                    )
                results.append(
                    run_task(
                        task,
                        entry,
                        clients[entry.api_name],
                        store,
                        timestamp(observed_at),
                        **reference.get(entry.api_name, {}),
                    )
                )
        typer.echo(encode(results))


@quality_app.command("audit")
def quality_audit(
    ctx: typer.Context, api_name: str, source: str = "fixture", references: Path | None = None
) -> None:
    """审计保留的标准数据；quality audit daily 返回问题与未检查项，有错误退出非零。

    参数选择接口、证据来源和质量参考；按观察批次审计历史数据并输出问题列表。
    字段基线优先使用前一观察批次，仅首批采用参考文件显式提供的 previous_fields。
    示例：空标准表输出 empty_response 并退出1；不同观察版本不混作同批重复主键。
    """
    with open_store(ctx) as store:
        entry = find_entry(store, api_name)
        if entry.policy is None:
            raise ValueError("missing_policy")
        reference = read_json(references) if references else {}
        reference_arguments = dict(reference.get(api_name, {}))
        reference_fields = reference_arguments.pop("previous_fields", None)
        batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in store.rows(api_name, source):
            batches[str(row["observed_at"])].append(row)
        findings = []
        previous: list[str] | None = None
        for batch in sorted(batches):
            rows = batches[batch]
            findings.extend(
                audit_rows(
                    rows,
                    entry.policy,
                    previous_fields=previous if previous is not None else reference_fields,
                    **reference_arguments,
                )
            )
            previous = list(rows[0]["source_fields"])
        if not batches:
            findings = audit_rows([], entry.policy)
        typer.echo(encode(findings))
        if any(issue["severity"] == "error" for issue in findings):
            raise typer.Exit(1)


@coverage_app.command("report")
def report_coverage(
    ctx: typer.Context,
    as_of: str = "2024-01-03",
    source: str = "fixture",
    gate: bool = False,
    output: Path | None = None,
) -> None:
    """输出覆盖分子分母及债务；coverage report --gate 遇未知接口或未消费数据退出非零。

    参数指定截止日期、来源、可选gate和目标文件；返回None并输出覆盖报告。
    示例：空库加 --gate 时 gate_passed=False 且退出1，零分母比例为null。
    """
    with open_store(ctx) as store:
        report = coverage_report(store, as_of, source)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encode(report) + "\n", encoding="utf-8")
        typer.echo(encode(report))
        if gate and not report["gate_passed"]:
            raise typer.Exit(1)


@products_app.command("build")
def products_build(
    ctx: typer.Context,
    output: Path = Path("data/products"),
    as_of: str = "2024-01-03",
    source: str = "fixture",
    parquet: Path | None = None,
) -> None:
    """构建十类产品并可导出 Parquet；products build --as-of 2024-01-03 不读取未来版本。

    参数为产品目录、截止时点、来源和可选Parquet路径；返回None并输出文件清单。
    示例：空库依然生成十类产品说明，业务产品 status='missing_inputs'。
    """
    with open_store(ctx) as store:
        paths = build_products(store, output, as_of, source)
        if parquet:
            store.export_parquet(parquet, source)
        typer.echo(encode({"files": [str(path) for path in paths]}))
