"""全产品离线验收夹具：正式标准化、完整206清单和确定性合成数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from test_research_binding_fixture import AS_OF, DATABASE_CONFIG
from test_research_binding_fixture import write_reference as write_equity_reference

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.reference_contract import load_reference_contracts
from tushare_entitlement_hub.store import encode


def product_rows() -> dict[str, list[dict[str, Any]]]:
    """返回六个新增研究域的非空数据及基金持仓报告，不包含真实账户持仓。

    例如forecast利润区间10—20万元，标准化中点150000元；基金份额2万份。
    """
    return {
        "ths_index": [
            {
                "ts_code": "885001.TI",
                "name": "合成主题",
                "list_date": "20000101",
                "exchange": "A",
                "type": "N",
            }
        ],
        "moneyflow_cnt_ths": [
            {"ts_code": "885001.TI", "trade_date": "20261002", "net_amount": 1, "currency": "CNY"}
        ],
        "etf_share_size": [
            {
                "ts_code": "510001.SH",
                "trade_date": day,
                "total_share": share,
                "total_size": share * 10,
                "nav": 10,
                "close": 10,
                "currency": "CNY",
            }
            for day, share in (("20261001", 1), ("20261002", 2))
        ],
        "fut_daily": [
            {
                "ts_code": "CU2612.SHF",
                "trade_date": "20261002",
                "close": 100,
                "settle": 110,
                "vol": 100,
                "oi": 200,
                "currency": "CNY",
            }
        ],
        "margin": [
            {
                "trade_date": "20261002",
                "exchange_id": "SSE",
                "rzye": 100,
                "rzmre": 30,
                "rzche": 20,
                "rzrqye": 125,
                "currency": "CNY",
            }
        ],
        "ths_daily": [
            {
                "ts_code": "885001.TI",
                "trade_date": "20261002",
                "pct_change": 2,
                "vol": 100,
                "turnover_rate": 3,
            }
        ],
        "dc_member": [{"ts_code": "BK0001.DC", "con_code": "000001.SZ", "trade_date": "20261002"}],
        "forecast": [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20261001",
                "first_ann_date": "20260901",
                "end_date": "20261231",
                "type": "预增",
                "net_profit_min": 10,
                "net_profit_max": 20,
            }
        ],
        "fund_basic": [
            {
                "ts_code": "510001.SH",
                "list_date": "20000101",
                "market": "E",
                "management": "合成基金管理人",
                "fund_type": "股票型",
                "benchmark": "合成指数",
            }
        ],
        "fund_share": [{"ts_code": "510001.SH", "trade_date": "20261002", "fd_share": 2}],
        "fund_daily": [
            {
                "ts_code": "510001.SH",
                "trade_date": "20261002",
                "close": 10,
                "amount": 100,
                "vol": 10,
                "currency": "CNY",
            }
        ],
        "fund_nav": [
            {
                "ts_code": "510001.SH",
                "ann_date": "20261002",
                "nav_date": "20261002",
                "unit_nav": 10,
                "adj_nav": 10,
                "net_asset": 1000,
                "currency": "CNY",
            }
        ],
        "fund_portfolio": [
            {
                "ts_code": "510001.SH",
                "ann_date": "20260901",
                "end_date": "20260630",
                "symbol": "000001.SZ",
                "mkv": 100,
                "amount": 10,
                "stk_mkv_ratio": 50,
            }
        ],
        "fut_index_daily": [
            {"ts_code": "CU.IDX", "trade_date": "20261002", "close": 100, "pct_chg": 2}
        ],
        "fut_trade_cal": [{"exchange": "SHFE", "cal_date": "20261002", "is_open": 1}],
        "us_tycr": [{"date": "20261002", "m3": 3, "y2": 4, "y5": 4.5, "y10": 5, "y30": 6}],
        "shibor": [{"date": "20261002", "on": 1, "1w": 1.5, "1m": 2, "3m": 2.5, "1y": 3}],
    }


def write_reference(
    directory: Path, datasets: dict[str, list[dict[str, Any]]] | None = None
) -> Path:
    """用正式契约生成合成库并补齐206项清单，无行的API明确保持缺表。

    输入directory得到reference.duckdb；datasets可覆盖单来源以制造缺口或边界。
    """
    rows = product_rows()
    rows.update(datasets or {})
    reference = write_equity_reference(directory, rows)
    contracts = load_reference_contracts(load_policies())
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        certify_synthetic_percentages(connection)
        present = {row[0] for row in connection.execute("SELECT api_name FROM catalog").fetchall()}
        for entry in load_registry():
            if entry.api in present:
                continue
            manifest = {
                "contract_hash": contracts[entry.api].fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": 0,
                    "accepted_rows": 0,
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?,?)", [entry.api, encode(manifest)])
    return reference


def certify_synthetic_percentages(connection: duckdb.DuckDBPyConnection) -> None:
    """为合成数据明确声明百分比来源及一次换算后的比例值，不覆盖文档明示未知字段。

    例如合成us_tycr.y10原始5表示5%，存储0.05、source_units=%、standard_units=比例；
    source_fields仍保留原始5。shibor.on与其他文档禁止字段不在声明列表。
    """
    declarations = {
        "ths_daily": ("pct_change",),
        "fut_index_daily": ("pct_chg",),
        "us_tycr": ("m3", "y2", "y5", "y10", "y30"),
        "shibor": ("1w", "1m", "3m", "1y"),
        "fund_portfolio": ("stk_mkv_ratio",),
    }
    contracts = load_reference_contracts(load_policies())
    for api, names in declarations.items():
        assert all(name not in contracts[api].policy.units for name in names)
        rows = connection.execute(
            f'SELECT source_record_id,source_units,standard_units FROM accepted."{api}"'
        ).fetchall()
        for identity, original_source, original_standard in rows:
            source_units = json.loads(original_source)
            standard_units = json.loads(original_standard)
            source_units.update(dict.fromkeys(names, "%"))
            standard_units.update(dict.fromkeys(names, "比例"))
            assignments = ",".join(f'"{name}"="{name}"/100' for name in names)
            connection.execute(
                f'UPDATE accepted."{api}" SET {assignments},source_units=?,standard_units=? '
                "WHERE source_record_id=?",
                [encode(source_units), encode(standard_units), identity],
            )
