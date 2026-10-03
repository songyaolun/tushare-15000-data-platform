"""99接口的确定性参考库，仅使用仓库文档中真实存在的字段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
from research_fixture import AS_OF, DATABASE_CONFIG, DAY, REPORT_PERIOD, fixture_values

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.equity_research.consumption_registry import (
    BOND_APIS,
    INDEX_APIS,
    PRODUCT_ALIASES,
)
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


def consumption_payloads() -> dict[str, dict[str, Any]]:
    """提供逐接口业务样本；例：LPR为3%，行情收盘10元，因子值只作为供应商旁证。

    字段随后与仓库官方快照取交集，缺失的契约字段不会通过夹具补造。
    """
    values: dict[str, dict[str, Any]] = dict(fixture_values(0))
    values.update(
        {
            "stock_basic": {
                "exchange": "SSE",
                "list_status": "L",
                "list_date": DAY,
                "industry": "制造",
            },
            "adj_factor": {"adj_factor": 1.0},
            "bak_basic": {"pe": 10.0, "pb": 1.0, "float_share": 800000.0, "total_assets": 1000.0},
            "bse_mapping": {"o_code": "S0000", "n_code": "S0000_NEW", "list_date": DAY},
            "factor_list": {
                "factor_name": "synthetic_alpha",
                "asset_type": "stock",
                "factor_type": "technical",
                "factor_desc": "合成供应商因子，非本地历史特征",
            },
            "factor_value": {"factor_name": "synthetic_alpha", "factor_value": 0.2},
            "stk_factor": {"adj_factor": 1.0, "close_qfq": 10.0, "macd": 0.1, "rsi_6": 50.0},
            "stk_factor_pro": {
                "adj_factor": 1.0,
                "close_qfq": 10.0,
                "pe_ttm": 10.0,
                "circ_mv": 1000000.0,
            },
            "stk_nineturn": {"freq": "daily", "nine_up_turn": 0, "nine_down_turn": 0},
            "stk_week_month_adj": {
                "freq": "week",
                "end_date": DAY,
                "close_qfq": 10.0,
                "close_hfq": 10.0,
            },
            "stk_weekly_monthly": {"freq": "week", "end_date": DAY, "close": 10.0, "vol": 10000.0},
            "daily_info": {
                "exchange": "SSE",
                "com_count": 1,
                "total_mv": 1000000.0,
                "amount": 100000.0,
            },
            "sz_daily_info": {"count": 1, "total_mv": 1000000.0, "amount": 100000.0},
            "index_basic": {
                "name": "合成指数",
                "market": "SSE",
                "publisher": "合成发布方",
                "index_type": "综合指数",
                "weight_rule": "市值",
                "list_date": DAY,
            },
            "index_dailybasic": {"pe_ttm": 10.0, "pb": 1.0, "turnover_rate": 0.05},
            "moneyflow_mkt_dc": {"net_amount": 100.0, "net_amount_rate": 0.01},
            "shibor_lpr": {"date": DAY, "1y": 0.03, "5y": 0.035},
            "stk_account": {
                "date": DAY,
                "weekly_new": 10.0,
                "total": 1000.0,
                "weekly_hold": 700.0,
                "weekly_trade": 500.0,
            },
            "trade_cal": {
                "exchange": "SSE",
                "cal_date": DAY,
                "is_open": 1,
                "pretrade_date": "2026-10-01",
            },
            "eco_cal": {
                "date": DAY,
                "time": "09:00",
                "country": "中国",
                "event": "合成经济日程",
                "value": "50",
                "pre_value": "49",
                "fore_value": "50",
            },
            "cn_cpi": {"month": "202609", "nt_yoy": 0.02, "nt_mom": 0.001},
            "cn_gdp": {
                "quarter": "2026Q2",
                "gdp": 1000.0,
                "gdp_yoy": 0.05,
                "pi_yoy": 0.02,
                "si_yoy": 0.04,
                "ti_yoy": 0.06,
            },
            "cn_m": {"month": "202609", "m1_yoy": 0.02, "m2_yoy": 0.05},
            "cn_pmi": {"month": "202609", "pmi010000": 50.1, "pmi010500": 51.0},
            "cn_ppi": {
                "month": "202609",
                "ppi_yoy": 0.01,
                "ppi_mom": 0.001,
                "ppi_mp_yoy": 0.01,
                "ppi_cg_yoy": 0.01,
            },
            "sf_month": {
                "month": "202609",
                "inc_month": 100.0,
                "inc_cumval": 1000.0,
                "stk_endval": 2000.0,
            },
            "fina_indicator": {"roe": 0.1, "debt_to_assets": 0.4, "ocf_to_profit": 2.0},
            "fina_indicator_vip": {"roe": 0.1, "debt_to_assets": 0.4, "ocf_to_profit": 2.0},
            "fina_audit": {"audit_result": "标准无保留意见", "audit_agency": "合成审计机构"},
            "new_share": {"ipo_date": DAY, "issue_date": DAY, "amount": 1000.0, "price": 10.0},
            "stk_holdernumber": {"holder_num": 1000.0},
            "stk_managers": {
                "name": "合成经理",
                "title": "董事",
                "begin_date": "2026-01-01",
                "end_date": None,
            },
            "stk_rewards": {
                "name": "合成经理",
                "title": "董事",
                "reward": 100000.0,
                "hold_vol": 1000.0,
            },
            "top10_holders": {"holder_name": "合成持有人", "hold_ratio": 0.1, "hold_change": 0.0},
            "top10_floatholders": {
                "holder_name": "合成持有人",
                "hold_amount": 1000.0,
                "hold_float_ratio": 0.1,
            },
            "cb_basic": {
                "stk_code": "S0000",
                "cb_type": "CB",
                "par": 100.0,
                "conv_price": 10.0,
                "remain_size": 100000.0,
                "conv_start_date": "2026-01-01",
                "conv_end_date": "2030-01-01",
            },
            "cb_daily": {
                "close": 105.0,
                "pre_close": 100.0,
                "cb_value": 100.0,
                "cb_over_rate": 0.05,
                "vol": 100.0,
                "amount": 10500.0,
            },
            "cb_factor_pro": {"close": 105.0, "vol": 100.0, "amount": 10500.0},
            "cb_issue": {"plan_issue_size": 100000.0, "issue_size": 100000.0, "res_ann_date": DAY},
            "cb_rate": {
                "rate_start_date": "2026-01-01",
                "rate_end_date": "2026-12-31",
                "coupon_rate": 0.01,
                "rate_freq": "1",
            },
            "cb_rating": {
                "rating_date": DAY,
                "rating_com_name": "合成评级机构",
                "rating_type": "债项",
                "rating": "AA",
                "rating_outlook": "稳定",
            },
            "cb_share": {
                "publish_date": DAY,
                "convert_price": 10.0,
                "convert_vol": 0.0,
                "acc_convert_vol": 0.0,
                "remain_size": 100000.0,
            },
            "cb_call": {
                "call_type": "不强赎",
                "is_call": "N",
                "call_date": DAY,
                "call_amount": 0.0,
            },
            "top10_cb_holders": {
                "holder_rank": 1,
                "holder_name": "合成债券持有人",
                "hold_ratio": 0.1,
            },
            "share_float": {
                "float_date": "2026-10-15",
                "float_share": 100.0,
                "float_ratio": 0.001,
                "holder_name": "合成股东",
                "share_type": "限售股",
            },
            "stk_seasoned": {
                "cur_stage": "已实施",
                "fo_type": "定增",
                "fo_vol_high": 1000.0,
                "fo_vol_act": 500.0,
                "fo_purpose": "合成再融资",
            },
            "pledge_detail": {
                "holder_name": "合成股东",
                "pledge_amount": 100.0,
                "is_release": "0",
                "start_date": DAY,
                "pledgor": "合成质权人",
                "release_date": None,
            },
            "repurchase": {
                "proc": "实施",
                "exp_date": "2026-12-31",
                "vol": 100.0,
                "amount": 1000.0,
            },
            "dividend": {
                "div_proc": "实施",
                "cash_div_tax": 0.1,
                "record_date": "2026-10-10",
                "ex_date": "2026-10-11",
                "pay_date": "2026-10-12",
            },
            "stk_holdertrade": {
                "holder_name": "合成股东",
                "in_de": "增持",
                "change_vol": 100.0,
                "change_ratio": 0.001,
                "begin_date": DAY,
                "close_date": DAY,
            },
            "st": {"pub_date": DAY, "imp_date": DAY, "st_type": "ST", "st_reason": "合成风险警示"},
            "stock_st": {"type": "ST"},
            "stk_alert": {"start_date": DAY, "end_date": "2026-10-15", "type": "重点监控"},
            "cyq_chips": {"price": 10.0, "percent": 1.0},
            "cyq_perf": {
                "cost_5pct": 10.0,
                "cost_50pct": 10.0,
                "cost_95pct": 10.0,
                "weight_avg": 10.0,
                "winner_rate": 0.5,
            },
            "hm_list": {"name": "合成标签", "desc": "仅供应商标签", "orgs": "合成营业部"},
            "hm_detail": {
                "hm_name": "合成标签",
                "hm_orgs": "合成营业部",
                "buy_amount": 100.0,
                "sell_amount": 50.0,
                "net_amount": 50.0,
            },
            "hsgt_top10": {"market_type": "1", "net_amount": 50.0, "amount": 100.0},
            "limit_step": {"nums": 2},
            "margin_secs": {"exchange": "SSE"},
            "slb_sec": {"lent_qnt": 100.0, "end_bal": 200.0},
            "slb_sec_detail": {"tenor": "7", "fee_rate": 0.01, "lent_qnt": 100.0},
            "stock_hsgt": {"type": "HK_SH", "type_name": "沪股通"},
            "top_list": {"reason": "合成异动", "l_buy": 100.0, "l_sell": 50.0, "net_amount": 50.0},
            "top_inst": {
                "exalter": "机构专用",
                "side": "买入",
                "buy": 50.0,
                "sell": 10.0,
                "net_buy": 40.0,
                "reason": "合成异动",
            },
            "limit_list_d": {
                "limit": "U",
                "open_times": 0,
                "fd_amount": 1000.0,
                "amount": 100000.0,
                "limit_times": 1,
            },
            "limit_list_ths": {"limit_type": "涨停", "open_num": 0, "lu_desc": "合成原因"},
            "stk_shock": {"period": "1日", "reason": "合成异动"},
            "stk_high_shock": {"period": "10日", "reason": "合成严重异动"},
            "suspend_d": {"suspend_type": "S", "suspend_timing": "全天"},
            "block_trade": {
                "price": 9.8,
                "vol": 100.0,
                "amount": 980.0,
                "buyer": "合成买方",
                "seller": "合成卖方",
            },
            "rt_min": {
                "time": "2026-10-02T14:59:00+08:00",
                "close": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
            },
            "major_news": {
                "title": "合成研究新闻",
                "content": "仅用于离线证据定位",
                "pub_time": "2026-10-02T01:00:00Z",
                "src": "合成来源",
            },
            "idx_anns": {
                "title": "合成指数公告",
                "url": "https://example.invalid/synthetic",
                "source": "合成指数机构",
                "type": "调整",
            },
            "cn_schedule": {
                "month": "202610",
                "publish_date": DAY,
                "title": "合成宏观日程",
                "issuing_org": "合成机构",
                "data_api": "cn_pmi",
            },
            "disclosure_date": {"pre_date": DAY, "actual_date": DAY, "modify_date": DAY},
            "namechange": {"name": "合成股票", "start_date": DAY, "end_date": None},
        }
    )
    for api in (
        "bak_daily",
        "weekly",
        "monthly",
        "pro_bar",
        "index_daily",
        "index_weekly",
        "index_monthly",
        "idx_factor_pro",
    ):
        values[api] = {
            "close": 10.0,
            "pre_close": 9.0,
            "pct_chg": 0.111111,
            "vol": 10000.0,
            "amount": 100000.0,
        }
    return values


def fixture_unit(field: str, api: str = "") -> tuple[str, str, str | None]:
    """返回仅对合成值生效的证书；例：amount为元，但new_share.amount为股，户数为户。"""
    if field in {"holder_num", "weekly_new", "weekly_hold", "weekly_trade", "total"}:
        return "户", "quantity", None
    if field in {"com_count", "count"}:
        return "家", "quantity", None
    if api == "new_share" and field == "amount":
        return "股", "quantity", None
    if field in {
        "close",
        "pre_close",
        "price",
        "par",
        "conv_price",
        "convert_price",
        "cb_value",
        "up_limit",
        "down_limit",
        "close_qfq",
        "close_hfq",
        "weight_avg",
    } or field.startswith("cost_"):
        return "元", "price", "CNY"
    if field in {
        "vol",
        "total_share",
        "float_share",
        "hold_vol",
        "hold_amount",
        "fo_vol_high",
        "fo_vol_act",
        "lent_qnt",
        "end_bal",
        "convert_vol",
        "acc_convert_vol",
        "change_vol",
        "pledge_amount",
    }:
        return "股", "quantity", None
    if (
        field
        in {
            "revenue",
            "n_income",
            "n_income_attr_p",
            "n_cashflow_act",
            "total_assets",
            "total_liab",
            "money_cap",
            "total_cur_liab",
            "total_hldr_eqy_inc_min_int",
            "total_mv",
            "circ_mv",
            "rzye",
            "remain_size",
            "issue_size",
            "plan_issue_size",
            "reward",
            "l_buy",
            "l_sell",
            "buy",
            "sell",
            "net_buy",
        }
        or "amount" in field
    ):
        return "元", "money", "CNY"
    return "ratio", "ratio", None


def write_consumption_reference(directory: Path) -> tuple[Path, ResearchConfig]:
    """生成99张accepted表及一条隔离毒值；例：运行后99来源各留一条当前合格观察。

    指数和转债独立实体，字典不强行绑定股票；字段只取官方快照与实际业务主键。
    夹具故意保留财务报告口径缺失和历史不足，以验证诚实降级。
    """
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "reference.duckdb"
    contracts = load_reference_contracts(load_policies())
    registry = tuple(item for item in load_registry() if item.product in PRODUCT_ALIASES)
    payloads = consumption_payloads()
    assert set(payloads) == {item.api for item in registry}
    evidence = EvidenceRef(
        source_api="daily",
        source_record_id="synthetic/certificate",
        business_group_key="synthetic/certificate",
        business_variant_key=None,
        contract_hash=contracts["daily"].fingerprint,
        observed_at=AS_OF,
        fields=("amount",),
        locator="synthetic:certified-units",
    )
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind=kind, local_id=code, lifecycle_id="fixture"
            ),
            name=name,
        )
        for code, kind, name in (
            ("S0000", "security", "合成股票"),
            ("B0000", "security", "合成转债"),
            ("I0000", "index", "合成指数"),
        )
    )
    aliases = tuple(
        EntityAlias(
            namespace="synthetic",
            alias=alias,
            entity_id=entity.identity.entity_id,
            effective_from="2000-01-01T00:00:00Z",
            effective_to=None,
            observed_at=AS_OF,
            source="synthetic",
            version="v1",
            confidence=100,
            evidence_refs=(evidence,),
        )
        for entity in entities
        for alias in (
            ("S0000", "S0000_NEW")
            if entity.identity.local_id == "S0000"
            else (entity.identity.local_id,)
        )
    )
    certifications, entity_mappings, time_mappings = [], [], []
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE TABLE catalog(api_name VARCHAR,manifest VARCHAR)")
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE SCHEMA quarantine")
        for spec in registry:
            api, contract = spec.api, contracts[spec.api]
            documented = {field["name"] for field in contract.documentation["output_fields"]}
            allowed = documented | set(spec.business_key)
            defaults = {
                "ts_code": "B0000"
                if api in BOND_APIS
                else "I0000"
                if api in INDEX_APIS
                else "S0000",
                "trade_date": DAY,
                "ann_date": DAY,
                "f_ann_date": DAY,
                "end_date": REPORT_PERIOD,
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "1",
            }
            record = {
                key: value for key, value in (defaults | payloads[api]).items() if key in allowed
            }
            # 上游财务契约本身缺合并范围时，保持缺失，不借夹具新增字段绕过门禁。
            numeric = {
                field: value for field, value in record.items() if type(value) in {int, float}
            }
            units = {field: fixture_unit(field, api)[0] for field in numeric}
            for field in numeric:
                unit, dimension, currency = fixture_unit(field, api)
                certifications.append(
                    UnitCertification(
                        version="synthetic-consumption/v1",
                        source_api=api,
                        field=field,
                        source_unit=unit,
                        target_unit=unit,
                        dimension=dimension,
                        currency=currency,
                        multiplier=1,
                        evidence_refs=(evidence,),
                    )
                )
            period = next(
                (
                    field
                    for field in (
                        "end_date",
                        "trade_date",
                        "cal_date",
                        "date",
                        "month",
                        "quarter",
                        "list_date",
                        "ipo_date",
                        "rate_start_date",
                        "ann_date",
                        "start_date",
                        "pub_date",
                    )
                    if record.get(field) is not None
                ),
                None,
            )
            public = tuple(
                field
                for field in (
                    "ann_date",
                    "f_ann_date",
                    "pub_date",
                    "publish_date",
                    "pub_time",
                    "res_ann_date",
                )
                if record.get(field) is not None
            )
            time_mappings.append(
                TimeMapping(
                    source_api=api,
                    data_field=period,
                    public_fields=public,
                    report_period_field="end_date"
                    if api.startswith(("fina_", "income", "cashflow", "balancesheet"))
                    else None,
                    market_timezone="Asia/Shanghai",
                    effective_from_field="time" if api == "rt_min" else None,
                )
            )
            if "ts_code" in record and (period is not None or api == "rt_min"):
                entity_mappings.append(
                    EntityMapping(source_api=api, namespace="synthetic", key_field="ts_code")
                )
            record.update(
                source_api=api,
                source_record_id=f"synthetic/{api}/0",
                business_group_key=f"synthetic/{api}/0",
                business_variant_key=None,
                contract_hash=contract.fingerprint,
                observed_at=AS_OF,
                quality_tier="verified",
                historical_pit_certified=False,
                is_quarantined=False,
                standard_units=json.dumps(units),
            )
            columns = sorted(set(record) | set(spec.business_key))
            types = {
                name: "BOOLEAN"
                if name in {"historical_pit_certified", "is_quarantined"}
                else "DOUBLE"
                if name in numeric
                else "VARCHAR"
                for name in columns
            }
            schema = ",".join(f'"{name}" {types[name]}' for name in columns)
            connection.execute(f'CREATE TABLE accepted."{api}" ({schema})')
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f'INSERT INTO accepted."{api}" VALUES ({placeholders})',
                [record.get(name) for name in columns],
            )
            quarantined = int(api == "idx_factor_pro")
            if quarantined:
                connection.execute(
                    f'CREATE TABLE quarantine."{api}" AS SELECT * FROM accepted."{api}"'
                )
                connection.execute(
                    f'UPDATE quarantine."{api}" SET close=999999, is_quarantined=true, '
                    "source_record_id='synthetic/quarantine/poison'"
                )
            manifest = {
                "contract_hash": contract.fingerprint,
                "observed_at": AS_OF,
                "collection_status": "complete_declared_request",
                "quality": {
                    "source_rows": 1 + quarantined,
                    "accepted_rows": 1,
                    "quarantined_rows": quarantined,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute("INSERT INTO catalog VALUES (?,?)", [api, json.dumps(manifest)])
    config = ResearchConfig(
        foundation=FoundationBindings(
            entities=entities,
            aliases=aliases,
            certifications=tuple(certifications),
            entity_mappings=tuple(entity_mappings),
            time_mappings=tuple(time_mappings),
            expected_data_as_of={"daily": DAY},
        ),
        certified_report_types=("1",),
        peer_paths={entities[0].identity.entity_id: ("synthetic-industry",)},
    )
    return reference, config


def main() -> None:
    """生成可复现CLI输入；例：--output-dir /tmp/consumption生成参考库与配置，不联网。"""
    parser = argparse.ArgumentParser(description="创建99接口离线消费夹具")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    reference, config = write_consumption_reference(arguments.output_dir)
    configuration = arguments.output_dir / "research-config.json"
    configuration.write_text(config.to_json(), encoding="utf-8")
    print(json.dumps({"reference": str(reference), "config": str(configuration), "as_of": AS_OF}))


if __name__ == "__main__":
    main()
