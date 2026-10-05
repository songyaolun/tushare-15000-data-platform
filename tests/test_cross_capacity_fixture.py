"""容量验收专用的流式合成参考库，保留正式列类型、规范化规则及完整清单。"""

import csv
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import duckdb

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cross_asset_research.registry import registrations
from tushare_entitlement_hub.reference_contract import (
    load_reference_contracts,
    normalize_reference_row,
)
from tushare_entitlement_hub.reference_storage import reference_columns
from tushare_entitlement_hub.store import encode

AS_OF = "2026-10-03T12:00:00Z"
DAY = "20261002"
DATABASE_CONFIG = {"threads": 1, "memory_limit": "512MB"}
CAPACITY_APIS = (
    "opt_basic",
    "fund_manager",
    "dc_member",
    "tdx_member",
    "etf_sh_cons",
    "etf_sz_cons",
)


def capacity_payloads(
    api: str,
    *,
    rows_per_source: int,
    fund_members: int,
    theme_members: int,
    legacy_entities: int,
    factors_per_entity: int,
    stock_count: int,
) -> Iterator[dict[str, Any]]:
    """逐行构造明确业务键，不一次创建完整来源列表。

    例如 300 只股票各 450 个当前筹码价位得到 135000 行，每只股票仍低于 512 行门槛。
    """
    fund_count = max(1, (rows_per_source + fund_members - 1) // fund_members)
    theme_count = max(1, (rows_per_source + theme_members - 1) // theme_members)
    if api == "stock_basic":
        for index in range(stock_count):
            yield {
                "ts_code": f"{index + 1:06d}.SZ",
                "name": f"合成股票{index}",
                "symbol": f"{index + 1:06d}",
                "list_date": "20000101",
                "list_status": "L",
                "curr_type": "CNY",
                "exchange": "SZSE",
                "market": "主板",
            }
    elif api == "daily":
        for index in range(stock_count):
            yield {
                "ts_code": f"{index + 1:06d}.SZ",
                "trade_date": DAY,
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "pre_close": 9,
                "change": 1,
                "pct_chg": 10,
                "vol": 10,
                "amount": 100,
            }
    elif api == "cyq_chips":
        for security in range(legacy_entities):
            for factor in range(factors_per_entity):
                yield {
                    "ts_code": f"{security + 1:06d}.SZ",
                    "trade_date": DAY,
                    "price": (factor + 1) / 100,
                    "percent": 100 / factors_per_entity,
                }
    elif api in {"fund_basic", "etf_basic"}:
        for index in range(fund_count):
            values: dict[str, Any] = {
                "ts_code": f"F{index:04d}",
                "name": f"合成ETF{index}",
                "list_date": "20000101",
            }
            if api == "fund_basic":
                values.update(market="E", management="合成管理人", fund_type="股票型")
            else:
                values.update(mgr_name="合成管理人", etf_type="权益")
            yield values
    elif api == "fund_company":
        yield {
            "name": "合成管理人",
            "shortname": "合成",
            "org_code": "ORG",
            "credit_code": "CREDIT",
        }
    elif api in {"dc_index", "tdx_index"}:
        prefix = "DC" if api == "dc_index" else "TDX"
        for index in range(theme_count):
            yield {
                "ts_code": f"{prefix}{index:04d}",
                "trade_date": DAY,
                "name": f"合成主题{index}",
                "idx_type": "概念",
                "level": "1",
            }
    elif api == "opt_daily":
        for index in range(min(2, rows_per_source)):
            yield {
                "ts_code": f"OPT{index:06d}",
                "trade_date": DAY,
                "vol": 10 + index,
                "oi": 20 + index,
                "close": 2,
                "settle": 2,
            }
    elif api in CAPACITY_APIS:
        for index in range(rows_per_source):
            if api == "opt_basic":
                yield {
                    "ts_code": f"OPT{index:06d}",
                    "symbol": f"OPT{index:06d}",
                    "name": f"合成合约{index}",
                    "opt_code": "UNDERLYING",
                    "opt_type": "ETF",
                    "call_put": "C" if index % 2 == 0 else "P",
                    "exercise_type": "E",
                    "exercise_price": 2,
                    "opt_multiplier": 100,
                    "maturity_date": "20261231",
                    "list_date": "20000101",
                    "delist_date": "20270101",
                    "per_unit": "100",
                    "quote_unit": "元",
                }
            elif api == "fund_manager":
                yield {
                    "ts_code": f"F{index // fund_members:04d}",
                    "ann_date": "20261001",
                    "name": f"合成经理{index:06d}",
                    "begin_date": "20200101",
                    "end_date": None,
                }
            elif api in {"dc_member", "tdx_member"}:
                prefix = "DC" if api == "dc_member" else "TDX"
                yield {
                    "ts_code": f"{prefix}{index // theme_members:04d}",
                    "con_code": f"{index % theme_members + 1:06d}.SZ",
                    "trade_date": DAY,
                    "name" if api == "dc_member" else "con_name": f"合成成员{index}",
                }
            else:
                values = {
                    "ts_code": f"F{index // fund_members:04d}",
                    "con_code": f"{index % fund_members + 1:06d}.SZ",
                    "trade_date": DAY,
                    "qty": 10,
                    "sub_flag": "允许",
                    "cpr": 1,
                    "rdr": 1,
                    "exchange": "SZ",
                }
                values.update({"sca": 1} if api == "etf_sh_cons" else {"sub_cc": 1, "red_cc": 1})
                yield values


def write_capacity_reference(
    directory: Path,
    *,
    rows_per_source: int = 30_000,
    fund_members: int = 1_000,
    theme_members: int = 5_000,
    legacy_entities: int = 300,
    factors_per_entity: int = 450,
    stock_count: int = 5_000,
    excluded_examples: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """将生成器经正式规范化后分源写 CSV，再用 DuckDB COPY 有界载入真实参考库。

    默认生成 135000 条旧筹码价位和六个各 30000 行的新来源，另加未来、隔离反例。
    返回参考库与真实写入计数；不访问正式库、不补人工单位证书、不联网。
    """
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "capacity-reference.duckdb"
    contracts = load_reference_contracts(load_policies())
    source_names = (
        "stock_basic",
        "daily",
        "cyq_chips",
        "fund_basic",
        "etf_basic",
        "fund_company",
        "dc_index",
        "tdx_index",
        "opt_daily",
        *CAPACITY_APIS,
    )
    counts: dict[str, Any] = {}
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE TABLE catalog(api_name VARCHAR,manifest VARCHAR)")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        for api in source_names:
            payloads = capacity_payloads(
                api,
                rows_per_source=rows_per_source,
                fund_members=fund_members,
                theme_members=theme_members,
                legacy_entities=legacy_entities,
                factors_per_entity=factors_per_entity,
                stock_count=stock_count,
            )
            first = next(payloads, None)
            if first is None:
                continue
            contract = contracts[api]
            first_normalized = normalize_reference_row(
                first, contract, AS_OF, f"capacity-{api}", 0, "capacity-synthetic-request"
            )
            columns = reference_columns(contract, [first_normalized])
            schema = ",".join(f'"{name}" {kind}' for name, kind in columns.items())
            connection.execute(f'CREATE TABLE accepted."{api}" ({schema})')
            csv_path = directory / f"{api}.csv"
            written = 0
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for index, raw in enumerate(chain((first,), payloads)):
                    normalized = (
                        first_normalized
                        if index == 0
                        else normalize_reference_row(
                            raw,
                            contract,
                            AS_OF,
                            f"capacity-{api}",
                            index,
                            "capacity-synthetic-request",
                        )
                    )
                    writer.writerow(
                        [
                            "\\N"
                            if normalized.get(name) is None
                            else encode(normalized[name])
                            if isinstance(normalized[name], (dict, list))
                            else normalized[name]
                            for name in columns
                        ]
                    )
                    written += 1
                if excluded_examples and api in CAPACITY_APIS:
                    for reason in ("future", "quarantine"):
                        example = dict(first)
                        key = (
                            "ts_code"
                            if api == "opt_basic"
                            else ("name" if api == "fund_manager" else "con_code")
                        )
                        example[key] = f"excluded-{reason}"
                        normalized = normalize_reference_row(
                            example,
                            contract,
                            AS_OF,
                            f"capacity-{api}",
                            written,
                            "capacity-synthetic-request",
                        )
                        normalized["observed_at" if reason == "future" else "quality_tier"] = (
                            "2026-12-01T00:00:00Z" if reason == "future" else "quarantine"
                        )
                        writer.writerow(
                            [
                                "\\N"
                                if normalized.get(name) is None
                                else encode(normalized[name])
                                if isinstance(normalized[name], (dict, list))
                                else normalized[name]
                                for name in columns
                            ]
                        )
                        written += 1
            quoted_path = str(csv_path).replace("'", "''")
            connection.execute(
                f"COPY accepted.\"{api}\" FROM '{quoted_path}' (FORMAT CSV, NULL '\\N')"
            )
            csv_path.unlink()
            actual = connection.execute(f'SELECT count(*) FROM accepted."{api}"').fetchone()[0]
            assert actual == written
            counts[api] = {
                "accepted_rows": written,
                "expected_excluded_rows": 2 if excluded_examples and api in CAPACITY_APIS else 0,
            }
            if progress is not None:
                progress({"stage": "fixture_source_complete", "source_api": api, **counts[api]})
        for api in sorted({item.spec.api for item in registrations()}):
            count = counts.get(api, {}).get("accepted_rows", 0)
            manifest = {
                "contract_hash": contracts[api].fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": count,
                    "accepted_rows": count,
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?,?)", [api, encode(manifest)])
        assert connection.execute("SELECT count(*) FROM catalog").fetchone()[0] == 206
    return reference, counts
