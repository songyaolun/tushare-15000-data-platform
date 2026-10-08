"""206 源严格参考链夹具：仅复用业务原值，认证由真实契约和绑定生成。"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import duckdb
from consumption_fixture import consumption_payloads
from test_research_binding_fixture import AS_OF, DATABASE_CONFIG
from test_research_binding_fixture import write_reference as write_formal_reference

import tushare_entitlement_hub as installed_package
from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.pipeline import run
from tushare_entitlement_hub.cross_asset_research.registry import NEW_PRODUCTS, PRODUCTS
from tushare_entitlement_hub.cross_asset_research.storage import validate
from tushare_entitlement_hub.equity_research.binding_rules import PUBLIC_FIELDS
from tushare_entitlement_hub.equity_research.consumption_registry import BOND_APIS, INDEX_APIS
from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.reference_contract import (
    load_reference_contracts,
    normalize_reference_row,
    reference_value,
)
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import digest, encode

CODE_REPO = Path(installed_package.__file__).resolve().parents[2]
CONTRACTS = load_reference_contracts(load_policies())
REGISTRY = {entry.api: entry for entry in load_registry()}
NEW_APIS = frozenset(api for api, entry in REGISTRY.items() if entry.product in NEW_PRODUCTS)
DAY = "20261002"
PERIOD = "20260930"
STOCK = "000001.SZ"


def raw_seeds():
    """复用99源payload、48源datasets和59源scenario的raw，不采纳其TypedFact或人工单位。"""
    import test_cross_market_api_evidence as market
    import test_cross_thematic_api_evidence as thematic

    data = {api: [dict(payload)] for api, payload in consumption_payloads().items()}
    data.update(thematic.datasets())
    for expectation in market.EXPECTATIONS:
        rows, _, _ = market.scenario(expectation.api)
        data[expectation.api] = [dict(row.raw) for row in rows]
    assert set(data) == set(REGISTRY)
    return data


def explicit_code(api):
    """声明合成证券或合约代码，不用运行环境或跨市场代码猜实体；例：港美同代码仍各有主数据。"""
    category = REGISTRY[api].category
    if api in BOND_APIS:
        return "123001.SZ"
    if api in INDEX_APIS:
        return "000300.SH"
    if api in {"fina_mainbz", "fina_mainbz_vip", "stock_company", "stk_ah_comparison"}:
        return STOCK
    if category == "futures":
        return (
            "CU.IDX"
            if api == "fut_index_daily"
            else "CU.SHF"
            if api == "fut_mapping"
            else "CU2612.SHF"
        )
    if category == "options":
        return "10009999.SH"
    if api.startswith("hk_") or api == "rt_hk_k":
        return "00700.HK"
    if api in {
        "us_basic",
        "us_daily",
        "us_income",
        "us_balancesheet",
        "us_cashflow",
        "us_fina_indicator",
    }:
        return "AAPL"
    if category == "fx":
        return "EURUSD.FXCM"
    if category == "spot":
        return "AU99.99"
    if api == "index_global":
        return "IXIC"
    if category == "bond":
        return "BOND1"
    return STOCK


def clean_datasets():
    """严格取正式输出字段，补具名业务键，非法占位数值转缺失而不造值或证书。

    例：HK财报模板中的'metadata'数值置None；未在契约声明的currency直接删除。
    """
    data = raw_seeds()
    actions = defaultdict(
        lambda: {
            "removed_noncontract": set(),
            "discarded_invalid_seed": set(),
            "filled_business_keys": set(),
        }
    )
    common = {
        "trade_date": DAY,
        "ann_date": "20261001",
        "f_ann_date": "20261001",
        "end_date": PERIOD,
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "1",
    }
    key_defaults = {
        "ann_date": "20261001",
        "f_ann_date": "20261001",
        "pub_date": "20261001",
        "publish_date": "20261001",
        "notice_date": "20261001",
        "actual_date": "20261001",
        "first_ann_date": "20260901",
        "report_date": "20261001",
        "start_date": "20260101",
        "end_date": PERIOD,
        "begin_date": "20260101",
        "in_date": "20200101",
        "rate_start_date": "20260101",
        "trade_date": DAY,
        "week_date": DAY,
        "cal_date": DAY,
        "date": DAY,
        "surv_date": DAY,
        "ipo_date": DAY,
        "nav_date": DAY,
        "float_date": DAY,
        "ex_date": DAY,
        "base_date": PERIOD,
        "rating_date": DAY,
        "month": "202609",
        "quarter": "2026Q3",
        "exchange": "SHFE",
        "exchange_id": "SSE",
        "fut_code": "CU",
        "fut_name": "铜",
        "symbol": "CU",
        "year": 2026,
        "area": "合成地区",
        "warehouse": "合成仓库",
        "grade": "合成品级",
        "brand": "合成品牌",
        "place": "合成产地",
        "is_ct": "0",
        "prd": "CU",
        "freq": "week",
        "qt_time": "10:00:00",
        "bank": "合成银行",
        "broker": "合成机构",
        "price": 100,
        "vol": 10,
        "amount": 100,
        "holder_rank": 1,
        "holder_name": "合成持有人",
        "buy_dp": "合成买方",
        "sell_dp": "合成卖方",
        "market_type": "1",
        "holder_type": "1",
        "type": "synthetic",
        "name": "合成名称",
        "title": "合成标题",
        "reason": "合成原因",
        "rating_type": "债项",
        "rating_com_name": "合成评级方",
        "call_type": "不强赎",
        "curr_type": "CNY",
        "ind_name": "合成指标",
        "ind_type": "consolidated",
        "rate_freq": "1",
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "1",
    }
    for public_field in PUBLIC_FIELDS:
        key_defaults.setdefault(public_field, "20261001")
    cleaned = {}
    for api, rows in sorted(data.items()):
        contract = CONTRACTS[api]
        types = {
            item["name"]: item["type"].lower() for item in contract.documentation["output_fields"]
        }
        fields = set(types)
        clean_rows = []
        for source in rows:
            candidate = {
                **{name: value for name, value in common.items() if name in fields},
                **source,
            }
            if api not in NEW_APIS or REGISTRY[api].product in {"P09", "P10", "P11"}:
                candidate["ts_code"] = explicit_code(api)
            if api in {
                "hk_basic",
                "us_basic",
                "fut_basic",
                "opt_basic",
                "index_basic",
                "cb_basic",
                "stock_basic",
            }:
                candidate["list_date"] = "20000101"
            if api == "stock_basic":
                candidate.update(
                    ts_code=STOCK,
                    symbol="000001",
                    name="合成股票一",
                    exchange="SZSE",
                    list_status="L",
                    market="主板",
                    industry="制造",
                    curr_type="CNY",
                )
            if api == "bse_mapping":
                candidate.update(
                    o_code="430001.BJ", n_code="920001.BJ", name="合成北交股", list_date="20000101"
                )
            if api == "cb_basic":
                candidate.update(stk_code=STOCK, cb_type="CB")
            if api == "fut_rcpt_mat":
                candidate["fut_code"] = "CU"
            if api == "etf_basic":
                candidate["list_date"] = "20000101"
            if api == "rt_hk_k":
                candidate.update(
                    pre_close=9, close=10, open=9.5, high=10.5, low=9, vol=100, amount=1000
                )
            if api == "stock_company":
                candidate.update(
                    com_id="SYNTHETIC_COMPANY_ID",
                    com_name="合成股票一企业",
                    main_business="合成制造业务",
                    business_scope="合成设备生产",
                    setup_date="20000101",
                    reg_capital=1000,
                )
            if api == "fut_mapping":
                candidate["mapping_ts_code"] = "CU2612.SHF"
            if api in {"fut_basic", "opt_basic"}:
                candidate["delist_date"] = "20261231"
            if api == "opt_basic":
                candidate.update(maturity_date="20261215", opt_code="510050.SH")
            if api == "stk_ah_comparison":
                candidate["hk_code"] = "00700.HK"
            if api == "us_fina_indicator":
                candidate["financial_date"] = "20261231"
            if api in {"fut_weekly_detail"}:
                candidate["week_date"] = DAY
            if api.startswith(("hk_", "us_")) and "end_date" in fields:
                candidate["end_date"] = PERIOD
            action = actions[api]
            action["removed_noncontract"].update(set(candidate) - fields)
            candidate = {name: value for name, value in candidate.items() if name in fields}
            for name, value in list(candidate.items()):
                if value is None:
                    continue
                try:
                    reference_value(value, name, types[name], contract)
                    invalid = False
                except (ValueError, TypeError, OverflowError):
                    invalid = True
                if invalid:
                    candidate[name] = None
                    action["discarded_invalid_seed"].add(name)
            for name in contract.primary_key:
                if candidate.get(name) is not None:
                    continue
                if name in contract.rules.get("nullable_keys", []):
                    candidate.setdefault(name, None)
                else:
                    if name not in key_defaults:
                        raise ValueError("unassigned_synthetic_business_key:" + api + ":" + name)
                    candidate[name] = key_defaults[name]
                    action["filled_business_keys"].add(name)
            # 正式normalizer预检，不沿用任一临时样本的quality/time/unit认证。
            try:
                normalize_reference_row(
                    candidate,
                    contract,
                    AS_OF,
                    digest(candidate),
                    len(clean_rows),
                    "reference-chain",
                )
            except ValueError as error:
                raise ValueError(api + ":" + str(error)) from None
            clean_rows.append(candidate)
        cleaned[api] = clean_rows
    stock = cleaned["stock_basic"][0]
    cleaned["stock_basic"].extend(
        [
            {**stock, "ts_code": "000002.SZ", "symbol": "000002", "name": "合成股票二"},
            {
                **stock,
                "ts_code": "430001.BJ",
                "symbol": "430001",
                "name": "合成北交股",
                "exchange": "BSE",
            },
        ]
    )
    return cleaned, {
        api: {key: sorted(value) for key, value in entry.items()} for api, entry in actions.items()
    }


def run_reference_chain(output_dir: Path) -> dict:
    """从 206 份合成原值建立参考库，执行产品并校验持久化来源证据。

    例如 run_reference_chain(tmp_path / "strict-reference-chain") 生成逐接口矩阵、
    配置、结果库和 reference_chain_summary.json；缺失业务事实仍允许保持 unknown。
    """
    data, actions = clean_datasets()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preparation.json").write_text(encode(actions), encoding="utf-8")
    print(
        encode(
            {
                "stage": "prepared",
                "apis": len(data),
                "new_apis": len(NEW_APIS),
                "rows": sum(map(len, data.values())),
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    raw_path = output_dir / "raw_payloads.json"
    raw_path.write_text(encode(data), encoding="utf-8")
    assert sum(map(len, data.values())) <= 4096
    reference = write_formal_reference(output_dir / "reference", data)
    reference_hash = file_checksum(reference)
    print(encode({"stage": "reference_created", "apis": len(data)}), flush=True)
    built = build_config(reference, AS_OF)
    (output_dir / "config.json").write_text(built.config.to_json(), encoding="utf-8")
    (output_dir / "binding_audit.json").write_text(encode(built.audit), encoding="utf-8")
    assert len(built.audit["sources"]) == 206
    print(
        encode(
            {
                "stage": "bindings_created",
                "readable": built.audit["summary"]["readable_apis"],
                "configured": built.audit["summary"]["configured_apis"],
            }
        ),
        flush=True,
    )
    output = output_dir / "products.duckdb"

    def progress(event):
        """仅显示产品阶段进展，不输出原始载荷；例如P09完成时显示结果条数。"""
        if event.get("stage") == "product_complete":
            print(
                encode(
                    {
                        "stage": "product_complete",
                        "product": event["product"],
                        "results": event["results"],
                    }
                ),
                flush=True,
            )

    summary = run(reference, output, AS_OF, built.config, progress=progress)
    (output_dir / "run_summary.json").write_text(encode(summary), encoding="utf-8")
    validation = validate(reference, output)
    (output_dir / "validation.json").write_text(encode(validation), encoding="utf-8")
    assert validation["valid"], validation
    assert file_checksum(reference) == reference_hash
    evidence = defaultdict(
        lambda: {
            "result_ids": set(),
            "referenced_records": set(),
            "fields": set(),
            "available_metrics": set(),
            "unknown_metric_reasons": set(),
            "unknown_gate_reasons": set(),
        }
    )
    with duckdb.connect(str(reference), read_only=True, config=DATABASE_CONFIG) as connection:
        source_rows = {
            api: {
                item[0]
                for item in connection.execute(
                    f'SELECT source_record_id FROM accepted."{api}"'
                ).fetchall()
            }
            for api in REGISTRY
        }
    product_by_id = {product_id: product for product, product_id in PRODUCTS.items()}
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        cursor = connection.execute(
            "SELECT payload FROM research.result WHERE run_id=?", [summary["run_id"]]
        )
        while rows := cursor.fetchmany(128):
            for (payload,) in rows:
                result = ResearchResult.from_json(payload)
                product = product_by_id[result.envelope.product_id]
                own_sources = set()
                for ref in result.envelope.evidence_refs:
                    if ref.source_api not in REGISTRY:
                        continue
                    assert ref.contract_hash == CONTRACTS[ref.source_api].fingerprint
                    assert ref.source_record_id in source_rows[ref.source_api]
                    if REGISTRY[ref.source_api].product != product:
                        continue
                    own_sources.add(ref.source_api)
                    record = evidence[ref.source_api]
                    record["result_ids"].add(result.envelope.result_id)
                    record["referenced_records"].add(ref.source_record_id)
                    record["fields"].update(ref.fields)
                for metric in result.metrics:
                    for ref in metric.evidence_refs:
                        if ref.source_api not in own_sources:
                            continue
                        if metric.status == "available":
                            evidence[ref.source_api]["available_metrics"].add(metric.name)
                        else:
                            evidence[ref.source_api]["unknown_metric_reasons"].update(
                                metric.reason_codes
                            )
                for api in own_sources:
                    for gate in result.envelope.quality_gates:
                        if gate.result.value != "true":
                            evidence[api]["unknown_gate_reasons"].update(gate.reason_codes)
    matrix = []
    for api in sorted(REGISTRY):
        record = summary["consumption"]["by_api"][api]
        assert record["registered"] and record["assigned"] and record["executed"], (api, record)
        source_evidence = evidence[api]
        exact_reasons = sorted(
            set(record["reason_counts"])
            | source_evidence["unknown_metric_reasons"]
            | source_evidence["unknown_gate_reasons"]
        )
        has_reference = bool(source_evidence["referenced_records"])
        # 每个新增API必须实际进入自己的已保存产品证据，或给出具体失败关闭原因。
        if api in NEW_APIS:
            assert has_reference or (
                set(exact_reasons)
                & {
                    "entity_business_time_unknown",
                    "source_local_key_missing",
                    "currency_unverified",
                    "unit_unknown",
                    "unit_documentation_unknown",
                }
            ), (api, "no_own_evidence_or_precise_reason")
        if record["observed"]:
            assert has_reference, (api, "observed_without_persisted_own_evidence")
        matrix.append(
            {
                "api": api,
                "product": REGISTRY[api].product,
                "is_new": api in NEW_APIS,
                **{
                    name: record[name]
                    for name in (
                        "registered",
                        "assigned",
                        "executed",
                        "readable",
                        "configured",
                        "observed",
                        "unresolved",
                        "retained_rows",
                        "observed_rows",
                    )
                },
                "source_rows": len(source_rows[api]),
                "own_product_reference": has_reference,
                "own_product_record_refs": len(source_evidence["referenced_records"]),
                "own_product_fields": sorted(source_evidence["fields"]),
                "available_metric_names": sorted(source_evidence["available_metrics"]),
                "precise_reason_codes": exact_reasons,
                "pipeline_reason_counts": record["reason_counts"],
                "binding_unit_gaps": built.audit["sources"][api].get("unit_gaps", {}),
                "binding_reason_counts": built.audit["sources"][api].get("reason_counts", {}),
                "unknown_metric_reasons": sorted(source_evidence["unknown_metric_reasons"]),
            }
        )
    new_matrix = [row for row in matrix if row["is_new"]]
    assert len(new_matrix) == 107
    (output_dir / "interface_matrix.json").write_text(encode(matrix), encoding="utf-8")
    csv_fields = [
        "api",
        "product",
        "is_new",
        "readable",
        "configured",
        "executed",
        "observed",
        "unresolved",
        "retained_rows",
        "observed_rows",
        "source_rows",
        "own_product_reference",
        "own_product_record_refs",
        "available_metric_names",
        "precise_reason_codes",
    ]
    with (output_dir / "interface_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in matrix:
            writer.writerow(
                {
                    name: encode(row[name]) if isinstance(row[name], (list, dict)) else row[name]
                    for name in csv_fields
                }
            )
    result = {
        "test_kind": "strict_reference_chain",
        "scope": "strict_contract_reference_chain",
        "valid": validation["valid"],
        "generated_reference_apis": len(matrix),
        "generated_source_rows": sum(row["source_rows"] for row in matrix),
        "new_api_states": {
            name: sum(row[name] for row in new_matrix)
            for name in (
                "readable",
                "configured",
                "executed",
                "observed",
                "unresolved",
                "own_product_reference",
            )
        },
        "all_api_states": {
            name: sum(row[name] for row in matrix)
            for name in ("readable", "configured", "executed", "observed", "unresolved")
        },
        "new_no_own_product_reference": [
            row["api"] for row in new_matrix if not row["own_product_reference"]
        ],
        "new_unobserved": [row["api"] for row in new_matrix if not row["observed"]],
        "strict_unit_policy": "no_injected_percent_units_or_synthetic_certificates",
        "source_currency_policy": "only_documented_raw_fields_or_real_stock_master_currency",
        "result_count": summary["result_count"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "runtime_repository": str(CODE_REPO),
        "source_fingerprints": {
            str(path.relative_to(CODE_REPO)): file_checksum(path)
            for path in sorted(
                (CODE_REPO / "src/tushare_entitlement_hub/cross_asset_research").glob("*.py")
            )
        },
    }
    (output_dir / "reference_chain_summary.json").write_text(encode(result), encoding="utf-8")
    print(encode(result), flush=True)

    return result
