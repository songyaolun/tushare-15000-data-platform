"""正式研究绑定的显式时间注册；修改注册必须更新版本和配置指纹。"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from ..pit import time_boundary
from ..product_foundation.bindings import TimeMapping
from ..product_foundation.entities import _business_time_bounds
from ..reference_contract import ReferenceContract
from ..store import digest
from .consumption_registry import load_consumption_registry

BINDING_VERSION = "research-bindings/v1"
PUBLIC_FIELDS = frozenset(
    {
        "ann_date",
        "f_ann_date",
        "pub_date",
        "publish_date",
        "pub_time",
        "res_ann_date",
        "imp_ann_date",
        "first_ann_date",
        "notice_date",
        "actual_date",
        "board_approval_dt",
        "sh_approval_dt",
        "sasac_approval_dt",
        "csrc_approval_dt",
        "plan_chg_ann_dt",
        "prospectus_pub_dt",
        "offer_intent_pub_dt",
    }
)
# API、业务期间、公开字段、报告期间、生效起点、生效终点。空项明确表示无法推定。
TIME_RULES = """
adj_factor|trade_date||||
bak_basic|trade_date||||
bak_daily|trade_date||||
balancesheet|end_date|ann_date,f_ann_date|end_date||
balancesheet_vip|end_date|ann_date,f_ann_date|end_date||
block_trade|trade_date||||
bse_mapping||||list_date|
cb_basic||||list_date|delist_date
cb_call|ann_date|ann_date||call_date|
cb_daily|trade_date||||
cb_factor_pro|trade_date||||
cb_issue|ann_date|ann_date,res_ann_date||onl_date|
cb_rate|rate_start_date|||rate_start_date|rate_end_date
cb_rating|rating_date|ann_date|||
cb_share|end_date|publish_date|end_date||
cashflow|end_date|ann_date,f_ann_date|end_date||
cashflow_vip|end_date|ann_date,f_ann_date|end_date||
cn_cpi|month||month||
cn_gdp|quarter||quarter||
cn_m|month||month||
cn_pmi|month||month||
cn_ppi|month||month||
cn_schedule|month|publish_date|month||
cyq_chips|trade_date||||
cyq_perf|trade_date||||
daily|trade_date||||
daily_basic|trade_date||||
daily_info|trade_date||||
disclosure_date|end_date|ann_date,actual_date|end_date||
dividend|end_date|ann_date,imp_ann_date|end_date|ex_date|
eco_cal|date||||
factor_list|||||
factor_value|trade_date||||
fina_audit|end_date|ann_date|end_date||
fina_indicator|end_date|ann_date|end_date||
fina_indicator_vip|end_date|ann_date|end_date||
hm_detail|trade_date||||
hm_list|||||
hsgt_top10|trade_date||||
idx_anns|ann_date|ann_date|||
idx_factor_pro|trade_date||||
income|end_date|ann_date,f_ann_date|end_date||
income_vip|end_date|ann_date,f_ann_date|end_date||
index_basic||list_date||list_date|exp_date
index_daily|trade_date||||
index_dailybasic|trade_date||||
index_monthly|trade_date||||
index_weekly|trade_date||||
limit_list_d|trade_date||||
limit_list_ths|trade_date||||
limit_step|trade_date||||
major_news||pub_time|||
margin_detail|trade_date||||
margin_secs|trade_date||||
moneyflow|trade_date||||
moneyflow_dc|trade_date||||
moneyflow_mkt_dc|trade_date||||
moneyflow_ths|trade_date||||
monthly|trade_date||||
namechange|start_date|ann_date||start_date|end_date
new_share|ipo_date|||issue_date|
pledge_detail|start_date|ann_date||start_date|end_date
pledge_stat|end_date||end_date||
pro_bar|trade_date||||
repurchase|ann_date|ann_date|||
rt_min|time||||
sf_month|month||month||
share_float|float_date|ann_date||float_date|
shibor_lpr|date||||
slb_sec|trade_date||||
slb_sec_detail|trade_date||||
st|pub_date|pub_date||imp_date|
stk_account|date||||
stk_alert|start_date|||start_date|end_date
stk_factor|trade_date||||
stk_factor_pro|trade_date||||
stk_high_shock|trade_date|trade_date|||
stk_holdernumber|end_date|ann_date|end_date||
stk_holdertrade|ann_date|ann_date||begin_date|close_date
stk_limit|trade_date||||
stk_managers|ann_date|ann_date||begin_date|end_date
stk_nineturn|trade_date||||
stk_rewards|end_date|ann_date|end_date||
stk_seasoned|ann_date|ann_date,first_ann_date,board_approval_dt,sh_approval_dt,sasac_approval_dt,csrc_approval_dt,plan_chg_ann_dt,prospectus_pub_dt,offer_intent_pub_dt||onl_issue_date|
stk_shock|trade_date|trade_date|||
stk_week_month_adj|end_date||||
stk_weekly_monthly|end_date||||
stock_basic||||list_date|delist_date
stock_hsgt|trade_date||||
stock_st|trade_date||||
suspend_d|trade_date||||
sz_daily_info|trade_date||||
top10_cb_holders|end_date||end_date||
top10_floatholders|end_date|ann_date|end_date||
top10_holders|end_date|ann_date|end_date||
top_inst|trade_date||||
top_list|trade_date||||
trade_cal|cal_date||||
weekly|trade_date||||
"""


def time_registry(contracts: dict[str, ReferenceContract]) -> tuple[TimeMapping, ...]:
    """加载并核对99项显式时间定义，契约缺字段或新增API时失败关闭。

    输入打包的schema/rules/policy契约，输出按API排序的TimeMapping。
    例如income把end_date映射报告期，ann_date和f_ann_date映射可空公开时间。
    """
    mappings = []
    for line in TIME_RULES.strip().splitlines():
        api, data, public, period, start, end = line.split("|")
        mapping = TimeMapping(
            source_api=api,
            version=BINDING_VERSION,
            data_field=data or None,
            optional_public_fields=tuple(filter(None, public.split(","))),
            report_period_field=period or None,
            effective_from_field=start or None,
            effective_to_field=end or None,
            market_timezone="Asia/Shanghai",
        )
        declared = {field["name"] for field in contracts[api].documentation["output_fields"]}
        used = {data, period, start, end} | set(mapping.optional_public_fields)
        if not (used - {""}) <= declared:
            raise ValueError("binding_time_contract_mismatch")
        mappings.append(mapping)
    if len(mappings) != 99 or {item.source_api for item in mappings} != {
        item.spec.api for item in load_consumption_registry()
    }:
        raise ValueError("binding_time_registry_mismatch")
    return tuple(sorted(mappings, key=lambda item: item.source_api))


def rules_fingerprint(contracts: dict[str, ReferenceContract]) -> str:
    """把时间注册及99个源契约一并绑定，政策或schema变化不能复用旧认证。

    输入当前契约，输出64位SHA256；例：修改daily单位后摘要随之变化。
    """
    return digest(
        [
            BINDING_VERSION,
            [item.to_dict() for item in time_registry(contracts)],
            {
                item.spec.api: contracts[item.spec.api].fingerprint
                for item in load_consumption_registry()
            },
        ]
    )


def period_visible(period: str, cutoff: str, timezone: str) -> bool:
    """按真实期间边界判断业务时间，日按市场日期、月季年须已结束。

    输入2026-Q2和2026-10-03截止返回True，2026-Q4返回False；
    UTC十月三日20时已经属于上海十月四日，不把当天业务误判成未来。
    """
    instant = time_boundary(cutoff)
    if len(period) == 10:
        return period <= instant.astimezone(ZoneInfo(timezone)).date().isoformat()
    return _business_time_bounds(period, timezone)[1] <= instant
