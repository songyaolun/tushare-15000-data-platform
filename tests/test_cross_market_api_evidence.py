"""59个跨市场接口的逐项验收，生产单位门禁与合成数学验证分别统计。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import pytest
from test_cross_market_products import (
    DAY,
    NOW,
    A,
    B,
    commodity_sources,
    financing_sources,
    inputs,
    international_sources,
    relation,
    source,
    value,
)

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cross_asset_research import commodities, financing, international
from tushare_entitlement_hub.equity_research.adapter import prepare_numeric
from tushare_entitlement_hub.equity_research.binding_entities import EntityBuild
from tushare_entitlement_hub.equity_research.binding_units import UnitBuild
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.product_foundation.entities import EntityAlias
from tushare_entitlement_hub.reference_contract import load_reference_contracts
from tushare_entitlement_hub.standard import _convert_unit


@dataclass(frozen=True)
class Expectation:
    """单接口明确预期；例：margin使用真实单位门禁后净融资比例0.1及三项输入字段。"""

    api: str
    product: str
    metric: str
    expected: float | None
    fields: tuple[str, ...]
    classification: str
    gap_field: str | None = None
    gap_reason: str | None = None
    reason: str | None = None
    state_key: str | None = None
    state_value: str | None = None


# Classification describes production unit-rule reachability, never a real-data observed run.
EXPECTATIONS = (
    Expectation(
        "fut_basic",
        "P09",
        "specification_completeness",
        1,
        ("fut_code", "exchange", "d_month", "quote_unit", "trade_unit", "per_unit"),
        "metadata_state",
    ),
    Expectation(
        "fut_daily",
        "P09",
        "open_interest_volume_ratio",
        2,
        ("oi", "vol"),
        "documented_unit_formula",
    ),
    Expectation(
        "fut_daily_adj",
        "P09",
        "continuous_settlement_deviation",
        None,
        ("settle", "close"),
        "strict_unknown",
        "close",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_holding",
        "P09",
        "long_hld_sample_hhi",
        None,
        ("long_hld", "broker"),
        "strict_unknown",
        "long_hld",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_index_daily",
        "P09",
        "commodity_index_return",
        None,
        ("pct_chg",),
        "strict_unknown",
        "pct_chg",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_inv_weekly",
        "P09",
        "inventory_change_ratio",
        None,
        ("cur_total", "pre_total"),
        "strict_unknown",
        "pre_total",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_mapping",
        "P09",
        "specification_completeness",
        1,
        ("mapping_ts_code",),
        "metadata_state",
    ),
    Expectation(
        "fut_rcpt_mat",
        "P09",
        "current_next_maturity_ratio",
        None,
        ("cur_month", "next_month"),
        "strict_unknown",
        "cur_month",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_settle",
        "P09",
        "long_short_margin_spread",
        None,
        ("long_margin_rate", "short_margin_rate"),
        "strict_unknown",
        "long_margin_rate",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_trade_cal",
        "P09",
        "session_open",
        1,
        ("is_open",),
        "metadata_state",
        state_key="session_state",
        state_value="open",
    ),
    Expectation(
        "fut_trade_param",
        "P09",
        "price_limit_constraint",
        0.1,
        ("price_limit",),
        "documented_unit_constraint",
    ),
    Expectation(
        "fut_weekly_detail",
        "P09",
        "weekly_open_interest_volume_ratio",
        2,
        ("open_interest", "vol"),
        "documented_unit_formula",
    ),
    Expectation(
        "fut_weekly_monthly",
        "P09",
        "resampled_settlement_deviation",
        None,
        ("settle", "close"),
        "strict_unknown",
        "settle",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fut_wsr",
        "P09",
        "receipt_change_current_ratio",
        None,
        ("vol_chg", "vol"),
        "strict_unknown",
        "vol",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "ft_limit", "P09", "margin_requirement", 0.1, ("m_ratio",), "documented_unit_constraint"
    ),
    Expectation(
        "sge_basic",
        "P09",
        "specification_completeness",
        1,
        ("trade_type", "t_unit", "p_unit", "trade_time"),
        "metadata_state",
    ),
    Expectation(
        "sge_daily",
        "P09",
        "spot_average_close_deviation",
        None,
        ("price_avg", "close"),
        "strict_unknown",
        "close",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fina_mainbz",
        "P09",
        "segment_gross_margin",
        0.4,
        ("bz_sales", "bz_cost"),
        "documented_unit_formula",
    ),
    Expectation(
        "fina_mainbz_vip",
        "P09",
        "segment_gross_margin",
        0.4,
        ("bz_sales", "bz_cost"),
        "documented_unit_formula",
    ),
    Expectation(
        "stock_company",
        "P10",
        "metadata_completeness",
        1,
        ("com_id", "main_business", "business_scope"),
        "metadata_state",
        state_key="cross_market_equivalence",
        state_value="explicit_relation_required",
    ),
    Expectation(
        "fx_daily",
        "P10",
        "relative_bid_ask_spread",
        None,
        ("bid_close", "ask_close"),
        "strict_unknown",
        "bid_close",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "fx_obasic",
        "P10",
        "metadata_completeness",
        1,
        ("name", "classify", "exchange", "trading_hours"),
        "metadata_state",
    ),
    Expectation(
        "index_global",
        "P10",
        "observed_market_return",
        None,
        ("pct_chg",),
        "strict_unknown",
        "pct_chg",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hibor",
        "P10",
        "twelve_month_overnight_spread",
        None,
        ("12m", "on"),
        "strict_unknown",
        "on",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "libor",
        "P10",
        "historical_twelve_month_overnight_spread",
        None,
        ("12m", "on"),
        "strict_unknown",
        "on",
        "unit_documentation_unknown",
        "metric_unavailable",
        "benchmark_status",
        "historical_discontinued_reference",
    ),
    Expectation(
        "us_tbr",
        "P10",
        "discount_thirteen_four_week_spread",
        None,
        ("w13_bd", "w4_bd"),
        "strict_unknown",
        "w4_bd",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_tltr",
        "P10",
        "long_term_method_spread",
        None,
        ("ltc", "cmt"),
        "strict_unknown",
        "ltc",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_trltr",
        "P10",
        "long_term_real_rate_positive",
        None,
        ("ltr_avg",),
        "strict_unknown",
        "ltr_avg",
        "unit_documentation_unknown",
        "metric_unavailable",
        "maturity_scope",
        "long_term_average_not_ten_year_point",
    ),
    Expectation(
        "us_trycr",
        "P10",
        "real_thirty_five_year_spread",
        None,
        ("y30", "y5"),
        "strict_unknown",
        "y5",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_tycr",
        "P10",
        "nominal_ten_two_year_spread",
        None,
        ("y10", "y2"),
        "strict_unknown",
        "y2",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hk_basic",
        "P10",
        "metadata_completeness",
        1,
        ("isin", "curr_type", "list_date"),
        "metadata_state",
    ),
    Expectation(
        "hk_tradecal",
        "P10",
        "previous_session_calendar_gap",
        1,
        ("cal_date", "pretrade_date", "is_open"),
        "metadata_state",
        state_key="session_state",
        state_value="open",
    ),
    Expectation(
        "us_basic", "P10", "metadata_completeness", 1, ("classify", "list_date"), "metadata_state"
    ),
    Expectation(
        "us_tradecal",
        "P10",
        "previous_session_calendar_gap",
        1,
        ("cal_date", "pretrade_date", "is_open"),
        "metadata_state",
        state_key="session_state",
        state_value="open",
    ),
    Expectation(
        "rt_hk_k",
        "P10",
        "observed_market_return",
        None,
        ("close",),
        "strict_unknown",
        reason="quote_timestamp_unverified",
    ),
    Expectation(
        "us_daily",
        "P10",
        "observed_market_return",
        None,
        ("pct_change",),
        "strict_unknown",
        "pct_change",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hk_income",
        "P10",
        "net_income_margin",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hk_balancesheet",
        "P10",
        "liabilities_assets_ratio",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hk_cashflow",
        "P10",
        "operating_cashflow_absolute_share",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_income",
        "P10",
        "net_income_margin",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_balancesheet",
        "P10",
        "liabilities_assets_ratio",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "us_cashflow",
        "P10",
        "operating_cashflow_absolute_share",
        None,
        ("ind_name", "ind_value", "end_date"),
        "strict_unknown",
        "ind_value",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "hk_fina_indicator",
        "P10",
        "gross_margin_cash_conversion_gap",
        0.15,
        ("gross_profit_ratio", "ocf_sales"),
        "documented_unit_formula",
    ),
    Expectation(
        "us_fina_indicator",
        "P10",
        "metadata_completeness",
        1,
        ("currency", "accounting_standards", "report_type", "financial_date"),
        "metadata_state",
    ),
    Expectation(
        "stk_ah_comparison",
        "P10",
        "ah_premium",
        None,
        ("ts_code", "hk_code", "ah_premium"),
        "strict_unknown",
        "ah_premium",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "bc_bestotcqt",
        "P11",
        "best_buy_sell_yield_gap",
        0.005,
        ("best_buy_yield", "best_sell_yield"),
        "documented_unit_formula",
        state_key="observation_type",
        state_value="quote_not_execution",
    ),
    Expectation(
        "bc_otcqt",
        "P11",
        "buy_yield_bank_range",
        0.01,
        ("buy_yield", "bank"),
        "documented_unit_formula",
        state_key="observation_type",
        state_value="quote_not_execution",
    ),
    Expectation(
        "bond_blk",
        "P11",
        "largest_block_amount_share",
        None,
        ("amount",),
        "strict_unknown",
        "amount",
        "currency_unverified",
        "metric_unavailable",
        state_key="observation_type",
        state_value="execution",
    ),
    Expectation(
        "bond_blk_detail",
        "P11",
        "largest_block_amount_share",
        None,
        ("amount",),
        "strict_unknown",
        "amount",
        "currency_unverified",
        "metric_unavailable",
        state_key="aggregation_policy",
        state_value="summary_and_detail_never_added",
    ),
    Expectation(
        "repo_daily",
        "P11",
        "closing_weighted_funding_spread",
        0.005,
        ("close", "weight"),
        "documented_unit_formula",
        state_key="close_semantics",
        state_value="funding_rate_not_security_price",
    ),
    Expectation(
        "vix_index",
        "P11",
        "us_implied_volatility_change",
        None,
        ("pct_change",),
        "strict_unknown",
        "pct_change",
        "unit_unknown",
        "metric_unavailable",
        "volatility_market",
        "US_equity_implied_volatility_only",
    ),
    Expectation(
        "gz_index",
        "P11",
        "guangzhou_long_ten_day_spread",
        None,
        ("long_rate", "d10_rate"),
        "strict_unknown",
        "d10_rate",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "shibor",
        "P11",
        "one_year_overnight_spread",
        None,
        ("1y", "on"),
        "strict_unknown",
        "on",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "shibor_quote",
        "P11",
        "on_b_bank_range",
        None,
        ("on_b", "bank"),
        "strict_unknown",
        "on_b",
        "unit_documentation_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "wz_index",
        "P11",
        "wenzhou_long_month_spread",
        None,
        ("long_rate", "m1_rate"),
        "strict_unknown",
        "m1_rate",
        "unit_unknown",
        "metric_unavailable",
    ),
    Expectation(
        "margin",
        "P11",
        "net_financing_balance_ratio",
        None,
        ("rzmre", "rzche", "rzye"),
        "strict_unknown",
        "rzye",
        "currency_unverified",
        "metric_unavailable",
    ),
    Expectation(
        "slb_len",
        "P11",
        "refinancing_balance_identity_residual",
        None,
        ("ob", "auc_amount", "repo_amount", "repay_amount", "cb"),
        "strict_unknown",
        "ob",
        "currency_unverified",
        "metric_unavailable",
    ),
    Expectation(
        "opt_basic",
        "P11",
        "contract_specification_completeness",
        1,
        ("opt_type", "call_put", "exercise_type", "maturity_date", "opt_multiplier"),
        "metadata_state",
        state_key="option_side",
        state_value="call",
    ),
    Expectation(
        "opt_daily",
        "P11",
        "option_open_interest_volume_ratio",
        2,
        ("oi", "vol"),
        "documented_unit_formula",
    ),
)


def _replace_source(row, record, raw=None, values=None):
    """重新生成合成行的完整来源身份；例如第二家银行不能沿用第一家的record_id。"""
    original = {
        name: (fact.value, fact.unit, fact.dimension, fact.currency)
        for name, fact in row.numeric.items()
    }
    payload = {
        key: item
        for key, item in row.raw.items()
        if key not in {"source_api", "source_record_id", "business_group_key", "contract_hash"}
    }
    payload.update(raw or {})
    return source(
        row.source_api,
        record,
        entity=row.entity_id,
        raw=payload,
        values={**original, **(values or {})},
        day=row.availability.data_as_of,
    )


@lru_cache(maxsize=1)
def _base_rows():
    """一次准备既有确定性合成样本；例：每个API只得到自身原始输入，不复用输出作预期。"""
    grouped = {}
    for row in (*commodity_sources(), *international_sources(), *financing_sources()):
        grouped.setdefault(row.source_api, []).append(row)
    return grouped


def scenario(api):
    """给单接口准备明确原始量纲和锚点；例：A/H必须同时传入关系与tushare_hk别名。"""
    rows = list(_base_rows()[api])
    links, aliases = (), ()
    # Source values here are raw percentages. Production-unit preparation applies /100 exactly once.
    overrides = {
        "fut_trade_param": {"price_limit": value(10)},
        "ft_limit": {"m_ratio": value(10)},
        "hk_fina_indicator": {"gross_profit_ratio": value(40), "ocf_sales": value(25)},
        "bc_bestotcqt": {"best_buy_yield": value(3.5), "best_sell_yield": value(3)},
        "repo_daily": {"close": value(3), "weight": value(2.5)},
    }
    if api in overrides:
        rows = [_replace_source(rows[0], "one", values=overrides[api])]
    if api in {"bc_otcqt", "shibor_quote"}:
        field = "buy_yield" if api == "bc_otcqt" else "on_b"
        rows = [
            _replace_source(rows[0], "bank_a", raw={"bank": "bank_a"}, values={field: value(3)}),
            _replace_source(rows[0], "bank_b", raw={"bank": "bank_b"}, values={field: value(4)}),
        ]
    if api in {"bond_blk", "bond_blk_detail"}:
        rows = [
            _replace_source(rows[0], "trade_a", values={"amount": value(60, "money", "元", "CNY")}),
            _replace_source(rows[0], "trade_b", values={"amount": value(40, "money", "元", "CNY")}),
        ]
    if api == "stk_ah_comparison":
        rows = [
            _replace_source(
                rows[0], "ah_row", raw={"hk_code": "HK1"}, values={"ah_premium": value(0.1)}
            )
        ]
        links = (relation("ah_pair", A, B),)
        aliases = (
            EntityAlias(
                namespace="tushare_hk",
                alias="HK1",
                entity_id=B,
                effective_from="2000-01-01T00:00:00Z",
                effective_to=None,
                observed_at=NOW,
                source="synthetic",
                version="v1",
                confidence=100,
                evidence_refs=rows[0].evidence("hk_code"),
            ),
        )
    prepared = []
    for row in rows:
        currency = "HKD" if api.startswith("hk_") else "USD" if api.startswith("us_") else "CNY"
        monetary_currencies = {
            fact.currency for fact in row.numeric.values() if fact.currency is not None
        }
        if len(monetary_currencies) == 1:
            currency = next(iter(monetary_currencies))
        payload = {**row.raw, "quality_tier": "verified", "end_date": DAY}
        selected = {**commodities.API_FIELDS, **international.API_FIELDS, **financing.API_FIELDS}[
            api
        ]
        if "currency" in selected:
            payload["currency"] = currency
        else:
            payload.pop("currency", None)
        if "curr_type" in payload:
            if "curr_type" in selected:
                payload["curr_type"] = currency
            else:
                payload.pop("curr_type")
        if api in {"hk_basic", "us_basic"}:
            payload["list_date"] = "2000-01-01"
        if api == "us_fina_indicator":
            payload["accounting_standards"], payload["financial_date"] = "US-GAAP", "12-31"
        prepared.append(replace(row, raw=payload))
    return tuple(prepared), links, aliases


@lru_cache(maxsize=1)
def _contracts():
    """加载正式本地参考契约；例：fina_mainbz金额为元，未知字段不会借测试证书变成已知。"""
    return load_reference_contracts(load_policies())


def production_units(rows):
    """复用真实UnitBuild和prepare_numeric，仅对政策已知单位做正式转换与认证。

    例：fina_mainbz元金额且有curr_type可计算；margin没有币种则保留currency_unverified。
    缺policy单位时不向source_units凭空补值；这里不模拟真实证券生命周期或正式数据observed。
    """
    builders, normalized = {}, []
    for row in rows:
        contract = _contracts()[row.source_api]
        payload = {**row.raw, "source_units": {}, "standard_units": {}}
        numeric_fields = set(row.numeric)
        for name in numeric_fields:
            declared = contract.policy.units.get(name)
            if declared is None or "未说明" in declared:
                continue
            if declared == "手" and contract.policy.category in {"futures", "options"}:
                converted, target = payload[name], "手"
            else:
                converted, target = _convert_unit(payload[name], declared, contract.policy)
            payload[name] = converted
            payload["source_units"][name] = declared
            payload["standard_units"][name] = target
        clean = replace(row, raw=payload, numeric={})
        builder = builders.setdefault(
            row.source_api,
            UnitBuild(
                allowed_currencies=frozenset({"CNY", "HKD", "USD"}),
                inherit_stock_currency=False,
                preserve_contract_lots=True,
            ),
        )
        builder.add(clean.raw, clean.availability, contract, numeric_fields, EntityBuild(1))
        normalized.append(clean)
    adapted = tuple(
        prepare_numeric(row, ResearchConfig(), certificates=builders[row.source_api].accepted())
        for row in normalized
    )
    return adapted, builders


def build_case(expectation, use_production_units=True):
    """调用真实产品入口并精确选取本API指标结果；例：不会让普通主营的同名指标代替VIP。"""
    rows, links, aliases = scenario(expectation.api)
    rows, builders = production_units(rows) if use_production_units else (rows, {})
    data = inputs(*rows, relations=links, aliases=aliases)
    builder = {
        "P09": commodities.build_commodities,
        "P10": international.build_international,
        "P11": financing.build_financing,
    }[expectation.product]
    results = builder(data)
    candidates = [
        result
        for result in results
        if result.details.get("source_api") == expectation.api
        and any(metric.name == expectation.metric for metric in result.metrics)
    ]
    assert len(candidates) == 1
    result = candidates[0]
    metric = next(metric for metric in result.metrics if metric.name == expectation.metric)
    return rows, builders, result, metric


def assert_source_evidence(expectation, rows, result, metric):
    """逐行逐字段校验本来源三层血缘；例：银行A/B都须引用自己的收益率和银行字段。"""
    observed = {
        ref.source_record_id: set()
        for ref in metric.evidence_refs
        if ref.source_api == expectation.api
    }
    envelope = {ref.to_json() for ref in result.envelope.evidence_refs}
    for ref in metric.evidence_refs:
        if ref.source_api == expectation.api:
            assert ref.to_json() in envelope
            assert ref.contract_hash == "synthetic"
            assert ref.business_group_key == ref.source_record_id
            observed[ref.source_record_id].update(ref.fields)
    assert set(observed) == {row.availability.source_record_id for row in rows}
    for row in rows:
        assert set(expectation.fields) <= observed[row.availability.source_record_id]
        assert set(expectation.fields) <= row.raw.keys()


@pytest.mark.parametrize(
    "expectation", EXPECTATIONS, ids=lambda item: item.product + "-" + item.api
)
def test_each_api_production_unit_boundary_and_own_business_evidence(expectation):
    """场景：59接口分别走正式单位门禁；预期具体业务值或准确unknown，且逐API/记录/字段血缘匹配。"""
    rows, builders, result, metric = build_case(expectation)
    if expectation.expected is None:
        assert metric.value is None
        assert metric.status == "unknown"
        assert metric.reason_codes == (expectation.reason,)
        assert expectation.reason in result.envelope.gate_codes
        if expectation.gap_field is not None:
            gaps = builders[expectation.api].gaps[expectation.gap_field]
            assert expectation.gap_reason in gaps
            assert all(expectation.gap_field not in row.numeric for row in rows)
    else:
        assert metric.status == "available"
        assert metric.value == pytest.approx(expectation.expected)
        assert metric.reason_codes == ()
    if expectation.state_key is not None:
        assert result.details[expectation.state_key] == expectation.state_value
    if expectation.api in {"fut_basic", "fut_mapping", "sge_basic"}:
        assert result.details["specification_state"] == "complete_observed"
        assert result.details["missing_specifications"] == []
    if expectation.api in {
        "stock_company",
        "fx_obasic",
        "hk_basic",
        "us_basic",
        "us_fina_indicator",
    }:
        assert result.details["identity_or_reporting_state"] == "observed_complete"
        assert result.details["cross_market_equivalence"] == "explicit_relation_required"
    assert_source_evidence(expectation, rows, result, metric)


# Additional mathematics under externally supplied test certificates. These are explicitly
# NOT counted as production-binding-positive or actual_binding_observed results.
SYNTHETIC_EXPECTED = {
    "fut_daily_adj": 0.1,
    "fut_holding": 1,
    "fut_index_daily": 0.02,
    "fut_inv_weekly": 0.2,
    "fut_rcpt_mat": 0.5,
    "fut_settle": 0.02,
    "fut_weekly_monthly": 0.1,
    "fut_wsr": 0.05,
    "sge_daily": 0.01,
    "fx_daily": 0.2 / 7.1,
    "index_global": 0.03,
    "hibor": 0,
    "libor": 0,
    "us_tbr": 0,
    "us_tltr": 0,
    "us_trltr": 1,
    "us_trycr": 0,
    "us_tycr": 0,
    "us_daily": 0.03,
    "hk_income": 0.2,
    "hk_balancesheet": 0.2,
    "hk_cashflow": 0.5,
    "us_income": 0.2,
    "us_balancesheet": 0.2,
    "us_cashflow": 0.5,
    "stk_ah_comparison": 0.1,
    "vix_index": 0.03,
    "gz_index": 0,
    "shibor": 0,
    "shibor_quote": 1,
    "wz_index": 0,
    "bond_blk": 0.6,
    "bond_blk_detail": 0.6,
    "margin": 0.1,
    "slb_len": 0,
}


@pytest.mark.parametrize(
    "expectation",
    [item for item in EXPECTATIONS if item.api in SYNTHETIC_EXPECTED],
    ids=lambda item: item.api,
)
def test_synthetic_certified_formula_is_separate_from_production_binding(expectation):
    """场景：人工提供明确测试证书的算法输入；预期公式具体值和本来源血缘，绝不升级主验收分类。"""
    rows, _, result, metric = build_case(expectation, use_production_units=False)
    assert metric.status == "available"
    assert metric.value == pytest.approx(SYNTHETIC_EXPECTED[expectation.api])
    assert_source_evidence(expectation, rows, result, metric)


def test_acceptance_matrix_has_exactly_one_semantic_contract_for_each_api():
    """场景：逐API预期表完整性；预期59项恰好一一对应矩阵且未知项都有具体原因。"""
    expected = {*commodities.API_FIELDS, *international.API_FIELDS, *financing.API_FIELDS}
    assert len(EXPECTATIONS) == len({item.api for item in EXPECTATIONS}) == 59
    assert {item.api for item in EXPECTATIONS} == expected
    assert all(
        item.fields and (item.expected is not None or item.reason is not None)
        for item in EXPECTATIONS
    )
