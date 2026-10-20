"""Two-stage public download plans and adapters to the existing historical panel builders."""

import json
from pathlib import Path
from typing import Any

from ..progress import progress_phase
from ..raw import assert_public
from ..regime_research.panel import prepare_panel
from ..research_history.collection import FIELDS, partition_quality, response_records
from ..signal_research.history import HISTORY_APIS, load_history_database
from ..strategy_discovery.replay_inputs import file_digest
from ..strategy_optimization.dataset import build_dataset
from .collection import read_collection
from .dataset import valid_date

EXTRA_FIELDS = {
    "moneyflow": "ts_code,trade_date,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount",
    "stock_st": "ts_code,name,trade_date,type,type_name",
    "cyq_perf": "ts_code,trade_date,cost_5pct,cost_50pct,cost_95pct,weight_avg",
    "forecast_vip": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,first_ann_date",
    "index_classify": "index_code,industry_name,level,industry_code,is_pub,parent_code,src",
    "index_member_all": (
        "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new"
    ),
    "sw_daily": "ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount",
    "index_daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "dividend": (
        "ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,"
        "record_date,ex_date,pay_date,div_listdate"
    ),
}


def request(api: str, parameters: dict[str, str]) -> dict[str, Any]:
    """Declare all requested fields and strict paging keys; example daily uses code/date.

    Event rows retain disclosure/report versions. No provider quota or availability is inferred.
    """
    fields = (FIELDS[api] if api in FIELDS else EXTRA_FIELDS[api]).split(",")
    keys = {
        "trade_cal": ["exchange", "cal_date"],
        "stock_basic": ["ts_code"],
        "index_classify": ["index_code"],
        "index_member_all": ["ts_code", "l1_code", "l2_code", "l3_code", "in_date"],
        "forecast_vip": ["ts_code", "end_date", "ann_date"],
        "fina_indicator_vip": ["ts_code", "end_date", "ann_date"],
        "income_vip": ["ts_code", "end_date", "ann_date", "report_type"],
        "dividend": ["ts_code", "end_date", "div_proc"],
    }.get(api, ["ts_code", "trade_date"] if "trade_date" in fields else fields)
    return {
        "api": api,
        "parameters": parameters,
        "fields": fields,
        "primary_key": keys,
        "page_size": 2000,
        "maximum_pages": 100,
        "paginate": True,
    }


def make_plan(output: Path, *, start: str, end: str, bootstrap: Path | None = None) -> Path:
    """Write bootstrap or full history plan from a verified calendar/security/industry catalog.

    Example: first plan collects SSE calendar, L/D/P securities and SW2021 L1 catalog;
    second collects all trading dates, historical memberships, reports and corporate actions.
    Historical source dates are requested explicitly, not synthesized from today's membership.
    """
    if not valid_date(start) or not valid_date(end) or start > end or output.exists():
        raise ValueError("invalid_plan_dates_or_existing_output")
    if bootstrap is None:
        requests = [
            request("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end}),
            request("index_classify", {"level": "L1", "src": "SW2021"}),
            *(request("stock_basic", {"list_status": status}) for status in ("L", "D", "P")),
        ]
    else:
        days = sorted(
            {
                row["cal_date"]
                for row in read_collection(bootstrap, "trade_cal")
                if row["is_open"] == 1 and start <= row["cal_date"] <= end
            }
        )
        codes = sorted(
            {
                row["ts_code"]
                for row in read_collection(bootstrap, "stock_basic")
                if row["ts_code"].endswith((".SH", ".SZ"))
            }
        )
        industries = sorted(
            {row["index_code"] for row in read_collection(bootstrap, "index_classify")}
        )
        if not days or not codes or not industries:
            raise ValueError("incomplete_bootstrap_catalog")
        requests = [
            request(api, {"trade_date": day})
            for day in days
            for api in (*HISTORY_APIS, "moneyflow", "stock_st", "cyq_perf")
        ]
        requests += [
            request("index_daily", {"ts_code": code, "start_date": start, "end_date": end})
            for code in ("000985.CSI", "H00985.CSI", "000300.SH")
        ]
        for code in industries:
            requests += [
                request("index_member_all", {"l1_code": code, "is_new": state})
                for state in ("Y", "N")
            ]
            requests.append(
                request("sw_daily", {"ts_code": code, "start_date": start, "end_date": end})
            )
        # Prior-year reports are needed for the first in-window TTM; keep all visible versions.
        periods = [
            f"{year}{quarter}"
            for year in range(int(start[:4]) - 2, int(end[:4]) + 1)
            for quarter in ("0331", "0630", "0930", "1231")
            if f"{year}{quarter}" <= end
        ]
        requests += [
            request(api, {"period": period})
            for period in periods
            for api in ("forecast_vip", "fina_indicator_vip", "income_vip")
        ]
        requests += [request("dividend", {"ts_code": code}) for code in codes]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "return80-collection-plan/v1",
                "start": start,
                "end": end,
                "requests": requests,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def materialize_sources(bootstrap: Path, collection: Path, output: Path) -> Path:
    """Verify each raw page, preserve it, then adapt explicit partitions for existing loaders.

    Example: daily_basic month-end also feeds capitalization; dividend pages retain
    record/ex/payment dates. No live calls or source repair; missing data fails validation.
    """
    if output.exists():
        raise ValueError("fresh_source_output_required")
    calendar = sorted(
        {row["cal_date"] for row in read_collection(bootstrap, "trade_cal") if row["is_open"] == 1}
    )
    months = set({day[:6]: day for day in calendar}.values())
    output.mkdir(parents=True)
    history, sectors, market = output / "history", output / "sectors", output / "market"
    for directory in (history, sectors, market):
        directory.mkdir()
    (history / "trading_dates.json").write_text(json.dumps(calendar))
    hashes: dict[str, Any] = {}
    aggregate: dict[Path, Any] = {}
    for root in (bootstrap, collection):
        manifest = json.loads((root / "collection.json").read_text())
        if manifest.get("schema") != "return80-collection/v1":
            raise ValueError("invalid_source_collection")
        for index, partition in enumerate(manifest["partitions"]):
            api, parameters = partition["request"]["api"], partition["request"]["parameters"]
            for page, item in enumerate(partition["files"]):
                source = root / item["path"]
                if (
                    not source.resolve().is_relative_to(root.resolve())
                    or file_digest(source) != item["sha256"]
                ):
                    raise ValueError("changed_source_raw_page")
                envelope = json.loads(source.read_text())
                assert_public(envelope)
                if envelope["api_name"] != api:
                    raise ValueError("source_api_mismatch")
                response = envelope["response"]
                records = response_records(response)
                # Client collection has already certified end-of-pagination; quality is rechecked.
                if api in HISTORY_APIS and not records:
                    continue
                adapted = {**envelope, "parameters": parameters}
                if api in HISTORY_APIS:
                    adapted["quality"] = partition_quality(api, parameters, response)
                    if records and adapted["quality"]["status"] != "passed":
                        raise ValueError("source_history_quality_failed")
                destination = history / "raw" / api / f"{index:06d}-{page:04d}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(adapted, ensure_ascii=False))
                hashes[destination.relative_to(output).as_posix()] = {
                    "adapted_sha256": file_digest(destination),
                    "original_sha256": item["sha256"],
                }
                target = None
                if api == "index_classify":
                    target = sectors / "catalog.json"
                elif api == "index_member_all":
                    target = (
                        sectors / f"members-{parameters['l1_code']}-{parameters['is_new']}.json"
                    )
                elif api == "sw_daily":
                    target = sectors / f"daily-{parameters['ts_code']}.json"
                elif api == "index_daily":
                    target = market / f"index-{parameters['ts_code']}.json"
                elif api == "daily_basic" and parameters["trade_date"] in months:
                    target = market / f"capitalization-{parameters['trade_date']}.json"
                elif api == "dividend":
                    target = market / f"dividend-{parameters['ts_code']}.json"
                if target:
                    current = aggregate.setdefault(
                        target,
                        {
                            "api_name": api,
                            "parameters": parameters,
                            "response": {
                                "code": 0,
                                "data": {"fields": response["data"]["fields"], "items": []},
                            },
                        },
                    )
                    if current["response"]["data"]["fields"] != response["data"]["fields"]:
                        raise ValueError("source_page_schema_changed")
                    current["response"]["data"]["items"].extend(response["data"]["items"])
            if (index + 1) % 500 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "materialize",
                            "completed": index + 1,
                            "total": len(manifest["partitions"]),
                        }
                    ),
                    flush=True,
                )
    for path, envelope in aggregate.items():
        path.write_text(json.dumps(envelope, ensure_ascii=False))
    for path in output.rglob("*.json"):
        relative = path.relative_to(output).as_posix()
        hashes.setdefault(relative, {"adapted_sha256": file_digest(path)})
    result = output / "sources.json"
    result.write_text(
        json.dumps(
            {"schema": "return80-sources/v1", "files": hashes, "historical_pit_certified": False},
            indent=2,
        )
        + "\n"
    )
    return result


def prepare_sources(sources: Path, output: Path) -> Path:
    """Reuse history quality, feature and market-panel engines, returning a rebuild specification.

    Example: prepared/rebuild.json references its own panel and the declared sources
    without any fixed user cache path. The saved calendar determines the sample end.
    """
    if output.exists():
        raise ValueError("fresh_prepared_output_required")
    manifest = json.loads((sources / "sources.json").read_text())
    for relative, item in manifest["files"].items():
        path = sources / relative
        if (
            not path.resolve().is_relative_to(sources.resolve())
            or file_digest(path) != item["adapted_sha256"]
        ):
            raise ValueError("changed_materialized_source")
    output.mkdir(parents=True)
    with progress_phase("history_database"):
        load_history_database(sources / "history", output / "history.duckdb")
    calendar = json.loads((sources / "history/trading_dates.json").read_text())
    with progress_phase("technical_dataset"):
        build_dataset(
            output / "history.duckdb",
            sources / "sectors",
            output / "features",
            end_date=calendar[-1],
        )
    with progress_phase("market_panel"):
        prepare_panel(output / "features", sources / "market", output / "panel")
    spec = {
        "schema": "return80-rebuild/v1",
        "panel": "panel",
        "source_dataset": "features",
        "raw_roots": [str((sources / "history/raw").resolve())],
        "benchmark": str((sources / "market/index-000300.SH.json").resolve()),
        "members": [
            str(path.resolve()) for path in sorted((sources / "sectors").glob("members-*.json"))
        ],
        "distributions": str((sources / "market").resolve()),
    }
    path = output / "rebuild.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return path
