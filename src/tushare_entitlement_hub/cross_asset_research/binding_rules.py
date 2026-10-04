"""新增107来源的显式时间、身份和保留规则，复用基础时间与单位校验。"""

from __future__ import annotations

from ..equity_research.binding_rules import PUBLIC_FIELDS
from ..product_foundation.bindings import EntityMapping, TimeMapping
from ..product_foundation.entities import EntityKind
from ..product_foundation.registry import ConsumerSpec, load_registry
from ..reference_contract import ReferenceContract
from ..store import digest

BINDING_VERSION = "all-products-bindings/v1"
ROW_UNIT_VERSION = "cross-row-units/v2"
SOURCE_SNAPSHOT_APIS = frozenset({"rt_hk_k", "stock_company"})
ALLOWED_CURRENCIES = frozenset({"CNY", "HKD", "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF"})
PRODUCTS = frozenset({"P02", "P05", "P08", "P09", "P10", "P11", "P12"})
# API、业务期间、公开字段、报告期间、生效起点、生效终点。
TIME_RULES = """
bc_bestotcqt|trade_date||||
bc_otcqt|trade_date||||
bond_blk|trade_date||||
bond_blk_detail|trade_date||||
broker_recommend|month||month||
ci_daily|trade_date||||
ci_index_member||||in_date|out_date
dc_daily|trade_date||||
dc_hot|trade_date||||
dc_index|trade_date||||
dc_member|trade_date||||
etf_basic||||list_date|
etf_index||pub_date||pub_date|
etf_limit|trade_date||||
etf_sh_cons|trade_date||||
etf_share_size|trade_date||||
etf_sz_cons|trade_date||||
express|end_date|ann_date|end_date||
express_vip|end_date|ann_date|end_date||
fina_mainbz|end_date||end_date||
fina_mainbz_vip|end_date||end_date||
forecast|end_date|ann_date,first_ann_date|end_date||
forecast_vip|end_date|ann_date,first_ann_date|end_date||
ft_limit|trade_date||||
fund_adj|trade_date||||
fund_basic||||list_date|delist_date
fund_company||||setup_date|end_date
fund_daily|trade_date||||
fund_div|base_date|ann_date,imp_anndate|base_date|ex_date|
fund_factor_pro|trade_date||||
fund_manager||ann_date||begin_date|end_date
fund_nav|nav_date|ann_date|||
fund_portfolio|end_date|ann_date|end_date||
fund_share|trade_date||||
fut_basic||||list_date|delist_date
fut_daily|trade_date||||
fut_daily_adj|trade_date||||
fut_holding|trade_date||||
fut_index_daily|trade_date||||
fut_inv_weekly|trade_date||||
fut_mapping|trade_date||||
fut_rcpt_mat|trade_date||||
fut_settle|trade_date||||
fut_trade_cal|cal_date||||
fut_trade_param|trade_date||||
fut_weekly_detail|week_date||||
fut_weekly_monthly|end_date||||
fut_wsr|trade_date||||
fx_daily|trade_date||||
fx_obasic|||||
gz_index|date||||
hibor|date||||
hk_balancesheet|end_date||end_date||
hk_basic||||list_date|delist_date
hk_cashflow|end_date||end_date||
hk_fina_indicator|end_date||end_date||
hk_income|end_date||end_date||
hk_tradecal|cal_date||||
index_classify|||||
index_global|trade_date||||
index_member_all||||in_date|out_date
index_weight|trade_date||||
kpl_concept_cons|trade_date||||
kpl_list|trade_date||||
libor|date||||
limit_cpt_list|trade_date||||
margin|trade_date||||
mkt_idx_bmk|||||
moneyflow_cnt_ths|trade_date||||
moneyflow_ind_dc|trade_date||||
moneyflow_ind_ths|trade_date||||
opt_basic||||list_date|delist_date
opt_daily|trade_date||||
repo_daily|trade_date||||
report_rc|report_date|report_date|quarter||
rt_hk_k|||||
sge_basic||||list_date|
sge_daily|trade_date||||
shibor|date||||
shibor_quote|date||||
slb_len|trade_date||||
slb_len_mm|trade_date||||
stk_ah_comparison|trade_date||||
stk_surv|surv_date||||
stock_company|||||
sw_daily|trade_date||||
tdx_daily|trade_date||||
tdx_index|trade_date||||
tdx_member|trade_date||||
ths_daily|trade_date||||
ths_hot|trade_date||||
ths_index||||list_date|
ths_member||||in_date|out_date
us_balancesheet|end_date||end_date||
us_basic||||list_date|delist_date
us_cashflow|end_date||end_date||
us_daily|trade_date||||
us_fina_indicator|end_date|notice_date|end_date||
us_income|end_date||end_date||
us_tbr|date||||
us_tltr|date||||
us_tradecal|cal_date||||
us_trltr|date||||
us_trycr|date||||
us_tycr|date||||
vix_index|trade_date||||
wz_index|date||||
"""
STOCK_APIS = frozenset(
    {
        "express",
        "express_vip",
        "forecast",
        "forecast_vip",
        "broker_recommend",
        "report_rc",
        "stk_surv",
        "fina_mainbz",
        "fina_mainbz_vip",
        "stock_company",
        "stk_ah_comparison",
        "dc_hot",
        "ths_hot",
        "kpl_list",
    }
)
SECURITY_APIS = STOCK_APIS | frozenset(
    {
        "hk_basic",
        "hk_balancesheet",
        "hk_cashflow",
        "hk_fina_indicator",
        "hk_income",
        "rt_hk_k",
        "us_basic",
        "us_balancesheet",
        "us_cashflow",
        "us_fina_indicator",
        "us_income",
        "us_daily",
    }
)
EVENT_APIS = frozenset({"forecast", "forecast_vip", "fund_div", "fund_manager"})
# 显式主数据能建立生命周期；没有这些证据时仅保留来源内序列身份。
MASTER_NAMESPACES = {
    "fund_basic": "tushare_fund",
    "hk_basic": "tushare_hk",
    "us_basic": "tushare_us",
    "fut_basic": "tushare_futures",
    "opt_basic": "tushare_option",
    "ths_index": "tushare_ths",
    "dc_index": "tushare_dc",
    "tdx_index": "tushare_tdx",
    "index_classify": "tushare_sw",
    "fx_obasic": "tushare_fx",
    "sge_basic": "tushare_sge",
    "ci_index_member": "tushare_ci",
}
OBSERVED_MASTERS = frozenset(
    {"dc_index", "tdx_index", "index_classify", "fx_obasic", "sge_basic", "ci_index_member"}
)
DIRECTORY_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {"ci_index_member": ("l3_code", "l3_name")}
MASTER_KINDS = {
    "fund_basic": EntityKind.ETF_SHARE,
    "hk_basic": EntityKind.SECURITY,
    "us_basic": EntityKind.SECURITY,
    "fut_basic": EntityKind.FUTURES_CONTRACT,
    "opt_basic": EntityKind.OPTION_CONTRACT,
    "ths_index": EntityKind.INDEX,
    **{api: EntityKind.SERIES for api in OBSERVED_MASTERS},
}
NAMESPACE_GROUPS = {
    "tushare_fund": "etf_basic etf_limit etf_sh_cons etf_share_size etf_sz_cons fund_adj "
    "fund_basic fund_daily fund_div fund_factor_pro fund_manager fund_nav fund_portfolio "
    "fund_share slb_len_mm",
    "tushare_hk": "hk_basic hk_balancesheet hk_cashflow hk_fina_indicator hk_income rt_hk_k",
    "tushare_us": "us_basic us_balancesheet us_cashflow us_fina_indicator us_income us_daily",
    "tushare_futures": "fut_basic fut_daily fut_daily_adj ft_limit fut_settle fut_weekly_monthly",
    "tushare_option": "opt_basic opt_daily",
    "tushare_ths": "ths_index ths_daily ths_member moneyflow_cnt_ths moneyflow_ind_ths",
    "tushare_dc": "dc_index dc_daily dc_member moneyflow_ind_dc",
    "tushare_tdx": "tdx_index tdx_daily tdx_member",
    "tushare_ci": "ci_daily ci_index_member",
    "tushare_sw": "sw_daily index_member_all index_classify",
    "tushare_index": "index_weight etf_index mkt_idx_bmk",
    "tushare_fx": "fx_daily fx_obasic",
    "tushare_sge": "sge_basic sge_daily",
}
KEY_FIELDS = {
    "ci_index_member": "l3_code",
    "index_member_all": "l3_code",
    "index_classify": "index_code",
    "index_weight": "index_code",
    "fund_company": "org_code",
    "fut_holding": "symbol",
    "fut_wsr": "symbol",
    "fut_rcpt_mat": "fut_code",
    "fut_trade_param": "fut_code",
    "fut_weekly_detail": "prd",
    "fut_trade_cal": "exchange",
    "margin": "exchange_id",
    "shibor_quote": "bank",
    "libor": "curr_type",
}


def cross_registry() -> tuple[ConsumerSpec, ...]:
    """返回矩阵中新增七产品的107项，绝不从前缀推断接口归属。

    例如fund_portfolio主归属P12，仍进入跨资产输入集合。
    """
    return tuple(entry for entry in load_registry() if entry.product in PRODUCTS)


def entity_mapping(api: str) -> EntityMapping:
    """把业务主实体绑定独立命名空间，成员行绑定父主题而非成员股票。

    例如index_weight使用index_code，ths_member使用ts_code；港美代码命名空间不同。
    """
    namespace = "tushare_stock" if api in STOCK_APIS else "source_" + api
    for group, names in NAMESPACE_GROUPS.items():
        if api in names.split():
            namespace = group
            break
    return EntityMapping(
        source_api=api, namespace=namespace, key_field=KEY_FIELDS.get(api, "ts_code")
    )


def time_registry(contracts: dict[str, ReferenceContract]) -> tuple[TimeMapping, ...]:
    """逐项核对107个显式字段声明，不以报告期或抓取时刻补公告时间。

    例如forecast.end_date为报告期，ann_date与first_ann_date分别参与公开时间门禁。
    """
    mappings = []
    for line in TIME_RULES.strip().splitlines():
        api, data, public, period, start, end = line.split("|")
        if api in SOURCE_SNAPSHOT_APIS and (
            contracts[api].policy.collection_mode != "snapshot" or contracts[api].policy.date_field
        ):
            raise ValueError("binding_snapshot_contract_mismatch")
        timezone: str | None = (
            "America/New_York"
            if api.startswith("us_")
            else (
                "Asia/Hong_Kong" if api.startswith("hk_") or api == "rt_hk_k" else "Asia/Shanghai"
            )
        )
        if api in {"fx_daily", "fx_obasic", "index_global"}:
            timezone = None
        mapping = TimeMapping(
            source_api=api,
            version=BINDING_VERSION,
            data_field=data or None,
            optional_public_fields=tuple(filter(None, public.split(","))),
            report_period_field=period or None,
            effective_from_field=start or None,
            effective_to_field=end or None,
            market_timezone=timezone,
        )
        declared = {field["name"] for field in contracts[api].documentation["output_fields"]}
        if (
            not ({data, period, start, end} | set(mapping.optional_public_fields)) - {""}
            <= declared
        ):
            raise ValueError("binding_time_contract_mismatch")
        mappings.append(mapping)
    if len(mappings) != 107 or {item.source_api for item in mappings} != {
        entry.api for entry in cross_registry()
    }:
        raise ValueError("binding_time_registry_mismatch")
    return tuple(sorted(mappings, key=lambda item: item.source_api))


def rules_fingerprint(contracts: dict[str, ReferenceContract]) -> str:
    """把全206项契约及新旧规则指纹绑定在一起，规则变化后不能复用旧配置。

    例如改变海外市场时区或港股身份命名空间，摘要随之改变。
    """
    from ..equity_research.binding_rules import rules_fingerprint as equity_fingerprint

    return digest(
        [
            BINDING_VERSION,
            ROW_UNIT_VERSION,
            sorted(ALLOWED_CURRENCIES),
            equity_fingerprint(contracts),
            [item.to_dict() for item in time_registry(contracts)],
            [entity_mapping(entry.api).to_dict() for entry in cross_registry()],
            {entry.api: contracts[entry.api].fingerprint for entry in cross_registry()},
            sorted(PUBLIC_FIELDS),
            sorted(EVENT_APIS),
            sorted(SECURITY_APIS),
        sorted(SOURCE_SNAPSHOT_APIS),
            sorted(OBSERVED_MASTERS),
            MASTER_NAMESPACES,
            DIRECTORY_IDENTITY_FIELDS,
            {api: kind.value for api, kind in MASTER_KINDS.items()},
        ]
    )
