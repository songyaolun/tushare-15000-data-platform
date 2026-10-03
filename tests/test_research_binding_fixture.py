"""研究绑定测试专用的确定性离线数据，不复用正式参考库。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.reference_contract import (
    load_reference_contracts,
    normalize_reference_row,
)
from tushare_entitlement_hub.reference_storage import reference_columns

AS_OF = "2026-10-03T12:00:00Z"
DATABASE_CONFIG = {"threads": 1, "memory_limit": "512MB"}


def default_rows() -> dict[str, list[dict[str, Any]]]:
    """返回两只合成股票的原始输入，每次调用返回独立容器。

    例如 daily.amount=100 千元经正式规范化为100000元，vol=10手变为1000股。
    """
    rows: dict[str, list[dict[str, Any]]] = {
        "stock_basic": [],
        "daily": [],
        "daily_basic": [],
        "stk_limit": [],
    }
    for index, code in enumerate(("000001.SZ", "000002.SZ")):
        rows["stock_basic"].append(
            {
                "ts_code": code,
                "symbol": code.split(".")[0],
                "name": f"合成股票{index}",
                "list_date": "20000101",
                "list_status": "L",
                "industry": "合成行业",
                "curr_type": "CNY",
                "exchange": "SZSE",
                "market": "主板",
            }
        )
        rows["daily"].append(
            {
                "ts_code": code,
                "trade_date": "20261002",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "pre_close": 10,
                "change": 0,
                "pct_chg": 0,
                "vol": 10,
                "amount": 100 + index,
            }
        )
        rows["daily_basic"].append(
            {
                "ts_code": code,
                "trade_date": "20261002",
                "close": 10,
                "turnover_rate": 5,
                "pe_ttm": 10 + index,
                "pb": 1 + index,
                "circ_mv": 100,
                "total_mv": 120,
                "total_share": 12,
                "float_share": 10,
            }
        )
        rows["stk_limit"].append(
            {
                "ts_code": code,
                "trade_date": "20261002",
                "pre_close": 10,
                "up_limit": 11,
                "down_limit": 9,
            }
        )
    return rows


def write_reference(
    directory: Path,
    rows_by_api: dict[str, list[dict[str, Any]]] | None = None,
) -> Path:
    """创建正式列类型、规范化单位和来源证据齐全的临时 accepted DuckDB。

    rows_by_api按API替换或添加默认数据；例如{'daily': []}生成空行情表，
    仍保留两只stock_basic。返回directory/reference.duckdb；不接触网络或正式缓存。
    """
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "reference.duckdb"
    contracts = load_reference_contracts(load_policies())
    sources = default_rows()
    sources.update(rows_by_api or {})
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE TABLE catalog(api_name VARCHAR, manifest VARCHAR)")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        for api, source_rows in sorted(sources.items()):
            contract = contracts[api]
            rows = [
                normalize_reference_row(
                    row,
                    contract,
                    AS_OF,
                    f"synthetic-{api}",
                    index,
                    "synthetic-request",
                )
                for index, row in enumerate(source_rows)
            ]
            # 空表仍保留审计字段，缺表/缺列由测试明确操作制造。
            metadata = {
                "source_api": api,
                "source_record_id": None,
                "business_group_key": None,
                "contract_hash": None,
                "observed_at": None,
                "quality_tier": None,
                "historical_pit_certified": False,
                "standard_units": {},
                "source_units": {},
            }
            columns = reference_columns(contract, rows or [metadata])
            schema = ", ".join(f'"{name}" {kind}' for name, kind in columns.items())
            connection.execute(f'CREATE TABLE accepted."{api}" ({schema})')
            if rows:
                values = [
                    [
                        json.dumps(row[name], ensure_ascii=False)
                        if isinstance(row.get(name), (dict, list))
                        else row.get(name)
                        for name in columns
                    ]
                    for row in rows
                ]
                placeholders = ",".join("?" for _ in columns)
                connection.executemany(
                    f'INSERT INTO accepted."{api}" VALUES ({placeholders})', values
                )
            manifest = {
                "contract_hash": contract.fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": len(rows),
                    "accepted_rows": len(rows),
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute(
                "INSERT INTO catalog VALUES (?, ?)", [api, json.dumps(manifest, sort_keys=True)]
            )
    return reference
