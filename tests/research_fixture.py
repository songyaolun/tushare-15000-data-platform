"""可执行的确定性研究夹具，仅创建合成参考库和显式认证配置。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.product_foundation.bindings import (
    EntityMapping,
    FoundationBindings,
    TimeMapping,
)
from tushare_entitlement_hub.product_foundation.contracts import EvidenceRef
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityAlias, EntityIdentity
from tushare_entitlement_hub.product_foundation.numeric import UnitCertification
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.reference_contract import load_reference_contracts

AS_OF = "2026-10-03T12:00:00Z"
DAY = "2026-10-02"
REPORT_PERIOD = "2026-06-30"
EMPTY_SCOPES = (
    "share_float",
    "stk_seasoned",
    "cb_basic",
    "st",
    "stock_st",
    "stk_alert",
    "suspend_d",
    "stk_shock",
    "stk_high_shock",
)
DATABASE_CONFIG = {"threads": 1, "memory_limit": "512MB"}


def fixture_values(index: int) -> dict[str, dict[str, float]]:
    """返回各接口的合成值；例：第0只股票收入100、经营现金流20，现金利润率0.2。"""
    values = {
        "daily": {
            "close": 10,
            "amount": 100000 + index * 1000,
            "vol": 10000,
            "pct_chg": 1 if index % 2 else -1,
        },
        "daily_basic": {
            "pe_ttm": 10 + index,
            "pb": 1 + index / 10,
            "circ_mv": 1000000,
            "total_share": 1000000,
            "float_share": 800000,
            "turnover_rate": 0.05,
        },
        "stk_limit": {"up_limit": 11, "down_limit": 9},
        "income_vip": {"revenue": 100, "n_income": 15},
        "cashflow_vip": {"n_cashflow_act": 20 + index},
        "balancesheet_vip": {
            "total_assets": 1000,
            "total_liab": 400,
            "money_cap": 200,
            "total_cur_liab": 100,
            "total_hldr_eqy_inc_min_int": 600,
        },
        "moneyflow": {
            "buy_lg_amount": 1000,
            "buy_elg_amount": 2000,
            "sell_lg_amount": 500,
            "sell_elg_amount": 1000,
        },
        "moneyflow_dc": {"net_amount": 1200},
        "moneyflow_ths": {"net_amount": 1100},
        "margin_detail": {"rzye": 20000},
        "pledge_stat": {"pledge_ratio": 0.1},
        "fina_audit": {},
    }
    values.update({api: {} for api in EMPTY_SCOPES})
    values.update(
        {api: values[api + "_vip"].copy() for api in ("income", "cashflow", "balancesheet")}
    )
    return values


def unit_definition(api: str, field: str) -> tuple[str, str, str | None]:
    """定义合成数据已认证的单位；例：daily.close为人民币价格，pct_chg为比例。"""
    if field in {"close", "up_limit", "down_limit"}:
        return "元", "price", "CNY"
    if field in {"vol", "total_share", "float_share"}:
        return "股", "quantity", None
    if field == "pledge_ratio":
        return "fraction", "ratio", None
    if field in {"pct_chg", "pe_ttm", "pb", "turnover_rate"}:
        return "ratio", "ratio", None
    return "元", "money", "CNY"


def write_reference(directory: Path, count: int = 32) -> tuple[Path, ResearchConfig]:
    """按当前注册字段和合同生成只含合成行的DuckDB；例：count=32支持默认同行排名。"""
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "reference.duckdb"
    contracts = load_reference_contracts(load_policies())
    registry = {entry.api: entry for entry in load_registry()}
    sample = fixture_values(0)
    evidence = EvidenceRef(
        source_api="daily",
        source_record_id="synthetic/certification",
        business_group_key="synthetic/certification",
        contract_hash=contracts["daily"].fingerprint,
        observed_at=AS_OF,
        fields=("amount",),
        business_variant_key=None,
        locator="synthetic:unit-certification",
    )
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind="security", local_id=f"S{index:04}", lifecycle_id="fixture"
            ),
            name=f"合成股票{index}",
        )
        for index in range(count)
    )
    aliases = tuple(
        EntityAlias(
            namespace="synthetic",
            alias=item.identity.local_id,
            entity_id=item.identity.entity_id,
            effective_from="2000-01-01T00:00:00Z",
            effective_to=None,
            observed_at=AS_OF,
            source="synthetic",
            version="v1",
            confidence=100,
            evidence_refs=(evidence,),
        )
        for item in entities
    )
    certifications = tuple(
        UnitCertification(
            version="synthetic/v1",
            source_api=api,
            field=field,
            source_unit=unit,
            target_unit=unit,
            dimension=dimension,
            currency=currency,
            multiplier=1,
            evidence_refs=(evidence,),
        )
        for api, values in sample.items()
        for field in values
        for unit, dimension, currency in [unit_definition(api, field)]
    )
    financial_sources = {
        "income",
        "income_vip",
        "cashflow",
        "cashflow_vip",
        "balancesheet",
        "balancesheet_vip",
        "fina_audit",
    }
    mappings = tuple(
        TimeMapping(
            source_api=api,
            data_field="end_date" if api in financial_sources | {"pledge_stat"} else "trade_date",
            report_period_field="end_date" if api in financial_sources else None,
            public_fields=tuple(
                field
                for field in ("ann_date", "f_ann_date", "pub_date")
                if field in registry[api].selected_fields + registry[api].business_key
            ),
            market_timezone="Asia/Shanghai",
        )
        for api in sample
    )
    config = ResearchConfig(
        foundation=FoundationBindings(
            entities=entities,
            aliases=aliases,
            certifications=certifications,
            entity_mappings=tuple(
                EntityMapping(source_api=api, namespace="synthetic", key_field="ts_code")
                for api in sample
            ),
            time_mappings=mappings,
            expected_data_as_of={"daily": DAY, "stock_st": DAY},
        ),
        certified_report_types=("1",),
        complete_scopes=EMPTY_SCOPES,
        peer_paths={item.identity.entity_id: ("synthetic-industry",) for item in entities},
    )
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE TABLE catalog(api_name VARCHAR, manifest VARCHAR)")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        for api, values in sample.items():
            columns = sorted(
                set(registry[api].selected_fields + registry[api].business_key)
                | {
                    "source_api",
                    "source_record_id",
                    "business_group_key",
                    "contract_hash",
                    "observed_at",
                    "quality_tier",
                    "historical_pit_certified",
                    "standard_units",
                }
                | set(values)
            )
            schema = ",".join(
                '"'
                + name
                + '" '
                + (
                    "BOOLEAN"
                    if name == "historical_pit_certified"
                    else "DOUBLE"
                    if name in values
                    else "VARCHAR"
                )
                for name in columns
            )
            connection.execute(f'CREATE TABLE accepted."{api}" ({schema})')
            rows = []
            for index, entity in enumerate(entities if api not in EMPTY_SCOPES else ()):
                record = dict.fromkeys(columns)
                record.update(fixture_values(index)[api])
                record.update(
                    ts_code=entity.identity.local_id,
                    trade_date=DAY,
                    ann_date=DAY,
                    f_ann_date=DAY,
                    end_date=REPORT_PERIOD if api in financial_sources else DAY,
                    report_type="1",
                    comp_type="1",
                    source_api=api,
                    source_record_id=f"synthetic/{api}/{index}",
                    business_group_key=f"synthetic/{api}/{index}",
                    contract_hash=contracts[api].fingerprint,
                    observed_at=AS_OF,
                    quality_tier="verified",
                    historical_pit_certified=False,
                    standard_units=json.dumps(
                        {field: unit_definition(api, field)[0] for field in values}
                    ),
                )
                if api == "fina_audit":
                    record["audit_result"] = "标准无保留意见"
                rows.append([record[name] for name in columns])
            if rows:
                placeholders = ",".join("?" for _ in columns)
                connection.executemany(
                    f'INSERT INTO accepted."{api}" VALUES ({placeholders})', rows
                )
            manifest = {
                "contract_hash": contracts[api].fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": len(rows),
                    "accepted_rows": len(rows),
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?,?)", [api, json.dumps(manifest)])
    return reference, config


def main() -> None:
    """创建命令行验收夹具；例：--output-dir /tmp/research生成库与配置。"""
    parser = argparse.ArgumentParser(description="创建确定性离线研究夹具")
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    arguments = parser.parse_args()
    reference, config = write_reference(arguments.output, arguments.count)
    configuration = arguments.output / "research-config.json"
    configuration.write_text(config.to_json(), encoding="utf-8")
    print(json.dumps({"reference": str(reference), "config": str(configuration), "as_of": AS_OF}))


if __name__ == "__main__":
    main()
