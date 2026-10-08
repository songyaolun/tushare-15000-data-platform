"""P10：隔离市场、币种、会计口径和交易时点的跨市场传导观察。"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..equity_research.models import Metric, ResearchInputs, ResearchResult
from ..product_foundation.entities import resolve_alias
from .market_support import (
    Frame,
    binary_flag,
    comparable,
    difference,
    finish,
    frames,
    number,
    run_sources,
    singleton,
    unit,
)

PRODUCT_ID = "international_transmission/v1"
API_FIELDS = {
    "stock_company": ("ts_code", "com_id", "main_business", "business_scope"),
    "fx_daily": ("ts_code", "trade_date", "bid_close", "ask_close", "exchange"),
    "fx_obasic": ("ts_code", "name", "classify", "exchange", "trading_hours"),
    "index_global": ("ts_code", "trade_date", "close", "pct_chg"),
    "hibor": ("date", "on", "1m", "3m", "12m"),
    "libor": ("date", "curr_type", "on", "1m", "3m", "12m"),
    "us_tbr": ("date", "w4_bd", "w4_ce", "w13_bd", "w13_ce"),
    "us_tltr": ("date", "ltc", "cmt", "e_factor"),
    "us_trltr": ("date", "ltr_avg"),
    "us_trycr": ("date", "y5", "y7", "y10", "y20", "y30"),
    "us_tycr": ("date", "m3", "y2", "y5", "y10", "y30"),
    "hk_basic": ("ts_code", "name", "isin", "curr_type", "list_date", "delist_date"),
    "hk_tradecal": ("cal_date", "is_open", "pretrade_date"),
    "us_basic": ("ts_code", "name", "classify", "list_date", "delist_date"),
    "us_tradecal": ("cal_date", "is_open", "pretrade_date"),
    "rt_hk_k": ("ts_code", "pre_close", "close", "vol", "amount"),
    "us_daily": ("ts_code", "trade_date", "close", "pct_change", "amount"),
    "hk_income": ("ts_code", "end_date", "ind_name", "ind_value"),
    "hk_balancesheet": ("ts_code", "end_date", "ind_name", "ind_value"),
    "hk_cashflow": ("ts_code", "end_date", "ind_name", "ind_value"),
    "us_income": ("ts_code", "end_date", "ind_name", "ind_type", "report_type", "ind_value"),
    "us_balancesheet": ("ts_code", "end_date", "ind_name", "ind_type", "report_type", "ind_value"),
    "us_cashflow": ("ts_code", "end_date", "ind_name", "ind_type", "report_type", "ind_value"),
    "hk_fina_indicator": (
        "ts_code",
        "end_date",
        "report_type",
        "currency",
        "gross_profit_ratio",
        "ocf_sales",
    ),
    "us_fina_indicator": (
        "ts_code",
        "end_date",
        "notice_date",
        "report_type",
        "currency",
        "accounting_standards",
        "financial_date",
    ),
    "stk_ah_comparison": ("ts_code", "hk_code", "trade_date", "close", "hk_close", "ah_premium"),
}
API_PURPOSES = {
    "stock_company": "公司主体和业务范围完整性；产业关系等待显式人工断言",
    "fx_daily": "同报价商双边中间价和相对报价差，不自动做投资组合换汇",
    "fx_obasic": "货币对及报价商、交易时段完整性",
    "index_global": "上一已知交易日的个体全球指数冲击",
    "hibor": "港币隔夜至十二月期限差",
    "libor": "历史 LIBOR 同币种期限差，固定标记停止更新的历史基准",
    "us_tbr": "贴现率和等价收益率各自期限差，不相互替代",
    "us_tltr": "长期曲线系列与固定期限系列差，保留来源方法差异",
    "us_trltr": "长期实际收益率样本背景，不与固定十年点相减",
    "us_trycr": "实际收益率五至三十年期限差及同期限名义实际利差",
    "us_tycr": "名义十减二年与十年减三月期限差",
    "hk_basic": "港股生命周期与币种证据完整性",
    "hk_tradecal": "港股最新可见开闭市和上次交易日间隔",
    "us_basic": "美股生命周期证据完整性，保留代码复用实体",
    "us_tradecal": "美股最新可见开闭市和上次交易日间隔",
    "rt_hk_k": "有行情时间戳才计算当前价格反应，不回填历史日线",
    "us_daily": "已采美股个体日冲击，不泛化为全美市场",
    "hk_income": "港股长表内同报告期净利率，认证指标语义后计算",
    "hk_balancesheet": "港股同报表资产负债比例",
    "hk_cashflow": "港股经营投资融资现金净流量合计及经营占比",
    "us_income": "美股同报告类型、指标类型和期间的净利率",
    "us_balancesheet": "美股同会计范围资产负债比例",
    "us_cashflow": "美股同口径三类现金净流量合计及经营占比",
    "hk_fina_indicator": "毛利率与经营现金销售比差，区分会计盈利和现金实现",
    "us_fina_indicator": "会计准则、币种和报告周期完整性；不把月日当日期",
    "stk_ah_comparison": "显式 A/H 关系下认证溢价旁证，不直接相减跨币价格",
}
DIMENSIONS = {
    "fx_daily": ("exchange",),
    "libor": ("curr_type",),
    "hk_fina_indicator": ("report_type", "currency"),
    "us_fina_indicator": ("report_type", "currency", "accounting_standards"),
    **{api: ("ind_type", "report_type") for api in ("us_income", "us_balancesheet", "us_cashflow")},
}
# Only these exact, reviewed semantic labels are used; no substring or fuzzy matching.
SEMANTICS = {
    "revenue": {"revenue", "营业收入", "营业额"},
    "net_income": {"net_income", "净利润", "除税后溢利"},
    "assets": {"total_assets", "资产总计", "总资产"},
    "liabilities": {"total_liabilities", "负债合计", "总负债"},
    "operating": {"operating_cashflow", "经营活动产生的现金流量净额"},
    "investing": {"investing_cashflow", "投资活动产生的现金流量净额"},
    "financing": {"financing_cashflow", "筹资活动产生的现金流量净额"},
}


def _financial(frame: Frame) -> list[Metric]:
    """将同一长表转为财务比率；例如认证净利润 20、收入 100 得净利率 0.2。

    只接受显式语义字段名和完整认证；重名指标、混币和无正分母一律不计算。
    """
    indexed = {}
    for row in frame.rows:
        for name, labels in SEMANTICS.items():
            if row.text("ind_name") in labels:
                if name in indexed:
                    frame.reasons.add("financial_indicator_conflict")
                indexed[name] = row
    names: tuple[str, ...]
    if frame.api.endswith("cashflow"):
        names, metric_name = (
            ("operating", "investing", "financing"),
            "operating_cashflow_absolute_share",
        )
    elif frame.api.endswith("income"):
        names, metric_name = ("net_income", "revenue"), "net_income_margin"
    else:
        names, metric_name = ("liabilities", "assets"), "liabilities_assets_ratio"
    rows = [indexed.get(name) for name in names]
    value = None
    if all(row is not None for row in rows):
        present = [row for row in rows if row is not None]
        amounts = [number(row, "ind_value", ("money",)) for row in present]
        if all(amount is not None for amount in amounts) and all(
            comparable(present[0], "ind_value", row, "ind_value") for row in present
        ):
            valid = [amount for amount in amounts if amount is not None]
            denominator = sum(abs(amount) for amount in valid) if len(valid) == 3 else valid[1]
            if denominator > 0:
                value = abs(valid[0]) / denominator if len(valid) == 3 else valid[0] / denominator
    return [frame.value(metric_name, value, "ratio", "ind_name", "ind_value", "end_date")]


def _process(frame: Frame, inputs: ResearchInputs) -> tuple[list[Metric], dict[str, Any]]:
    """加工跨市场来源；例如同单位 bid=7、ask=7.2 得中间价 7.1 和价差比例。

    每个市场保持自己的实体及报告范围，实时行情缺时间戳时不生成收益。
    """
    api = frame.api
    details: dict[str, Any] = {
        "purpose": API_PURPOSES[api],
        "market_aggregation": "source_and_entity_separate",
    }
    if api.endswith(("_income", "_balancesheet", "_cashflow")):
        return _financial(frame), details
    row = singleton(frame)
    if row is None:
        return [frame.value("snapshot_usable", None)], details
    metadata = {
        "stock_company": ("com_id", "main_business", "business_scope"),
        "fx_obasic": ("name", "classify", "exchange", "trading_hours"),
        "hk_basic": ("isin", "curr_type", "list_date"),
        "us_basic": ("classify", "list_date"),
        "us_fina_indicator": ("currency", "accounting_standards", "report_type", "financial_date"),
    }
    if api in metadata:
        names = metadata[api]
        details["identity_or_reporting_state"] = (
            "observed_complete" if all(row.text(name) for name in names) else "incomplete"
        )
        details["cross_market_equivalence"] = "explicit_relation_required"
        return [
            frame.value(
                "metadata_completeness",
                sum(row.text(name) is not None for name in names) / len(names),
                "fraction",
                *names,
            )
        ], details
    if api.endswith("tradecal"):
        lag = None
        try:
            lag = (
                date.fromisoformat(frame.latest_period)
                - date.fromisoformat(row.text("pretrade_date") or "")
            ).days
        except ValueError:
            pass
        details["session_state"] = {1.0: "open", 0.0: "closed", None: "unknown"}[
            binary_flag(row, "is_open")
        ]
        return [
            frame.value(
                "previous_session_calendar_gap",
                float(lag) if lag is not None and lag > 0 else None,
                "days",
                "cal_date",
                "pretrade_date",
                "is_open",
            )
        ], details
    if api == "fx_daily":
        bid, ask = number(row, "bid_close"), number(row, "ask_close")
        valid = (
            comparable(row, "bid_close", row, "ask_close")
            and bid is not None
            and ask is not None
            and 0 < bid <= ask
        )
        mid = (bid + ask) / 2 if valid and bid is not None and ask is not None else None
        return [
            frame.value("quoted_midpoint", mid, unit(row, "bid_close"), "bid_close", "ask_close"),
            frame.value(
                "relative_bid_ask_spread",
                (ask - bid) / mid
                if mid is not None and ask is not None and bid is not None
                else None,
                "ratio",
                "bid_close",
                "ask_close",
            ),
        ], details
    if api == "stk_ah_comparison":
        relations = [
            relation
            for relation in inputs.config.relations_by_source.get((frame.entity_id, "ah_pair"), ())
            if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
        ]
        details["confirmed_pair_count"] = len(relations)
        code = row.text("hk_code") or ""
        try:
            identity = resolve_alias(
                inputs.config.aliases_by_key.get(("tushare_hk", code), ()),
                "tushare_hk",
                code,
                inputs.analysis_as_of,
                effective_at=row.availability.data_as_of,
                market_timezone=row.availability.market_timezone,
            )
        except ValueError:
            identity = None
        if len(relations) != 1 or identity != relations[0].to_entity_id:
            frame.reasons.add("ah_pair_unverified")
        else:
            frame.cite(relations[0].evidence_refs)
            frame.cite(
                ref
                for alias in inputs.config.aliases_by_key[("tushare_hk", code)]
                for ref in alias.evidence_refs
            )
        details["pricing_method"] = "certified_provider_premium_no_implicit_fx"
        return [
            frame.value(
                "ah_premium",
                number(row, "ah_premium", ("ratio",)),
                unit(row, "ah_premium"),
                "ts_code",
                "hk_code",
                "ah_premium",
            )
        ], details
    if api in {"index_global", "us_daily", "rt_hk_k"}:
        if api == "rt_hk_k":
            frame.reasons.add("quote_timestamp_unverified")
            value, field_name = None, "close"
        else:
            field_name = "pct_chg" if api == "index_global" else "pct_change"
            value = number(row, field_name, ("ratio",))
        return [
            frame.value("observed_market_return", value, unit(row, field_name), field_name)
        ], details
    pairs = {
        "hibor": (("twelve_month_overnight_spread", "12m", "on"),),
        "libor": (("historical_twelve_month_overnight_spread", "12m", "on"),),
        "us_tbr": (
            ("discount_thirteen_four_week_spread", "w13_bd", "w4_bd"),
            ("equivalent_thirteen_four_week_spread", "w13_ce", "w4_ce"),
        ),
        "us_tltr": (("long_term_method_spread", "ltc", "cmt"),),
        "us_trycr": (("real_thirty_five_year_spread", "y30", "y5"),),
        "us_tycr": (
            ("nominal_ten_two_year_spread", "y10", "y2"),
            ("nominal_ten_year_three_month_spread", "y10", "m3"),
        ),
        "hk_fina_indicator": (
            ("gross_margin_cash_conversion_gap", "gross_profit_ratio", "ocf_sales"),
        ),
    }
    values = [
        frame.value(
            name, difference(row, left, right, ("ratio", "rate")), unit(row, left), left, right
        )
        for name, left, right in pairs.get(api, ())
    ]
    if api == "libor":
        details["benchmark_status"] = "historical_discontinued_reference"
    if api == "us_trltr":
        level = number(row, "ltr_avg", ("ratio", "rate"))
        values.append(
            frame.value(
                "long_term_real_rate_positive",
                None if level is None else float(level > 0),
                "fraction",
                "ltr_avg",
            )
        )
        details["maturity_scope"] = "long_term_average_not_ten_year_point"
    return values, details


def _breakeven(inputs: ResearchInputs) -> list[ResearchResult]:
    """在同日同期限认证名义和实际国债曲线间计算利差代理；例：4%-2%=2%。

    这只是期限匹配代理，保留税收、流动性及风险溢价差异，不称为真实通胀预测。
    """
    real = {
        frame.latest_period: frame
        for frame in frames(inputs, "us_trycr", API_FIELDS["us_trycr"])
        if len(frame.rows) == 1 and not frame.reasons
    }
    results = []
    for frame in frames(inputs, "us_tycr", API_FIELDS["us_tycr"]):
        if len(frame.rows) != 1 or frame.reasons or frame.latest_period not in real:
            continue
        other = real[frame.latest_period].rows[0]
        row = frame.rows[0]
        left, right = number(row, "y10", ("ratio", "rate")), number(other, "y10", ("ratio", "rate"))
        value = (
            left - right
            if comparable(row, "y10", other, "y10") and left is not None and right is not None
            else None
        )
        frame.scope = {"maturity": "10y", "method": "nominal_real_proxy"}
        frame.add(inputs, other)
        results.append(
            finish(
                inputs,
                PRODUCT_ID,
                frame,
                [frame.value("ten_year_breakeven_proxy", value, unit(row, "y10"), "y10", "date")],
                {
                    "interpretation": (
                        "inflation_compensation_proxy_including_risk_and_liquidity_premia"
                    )
                },
            )
        )
    return results


def build_international(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """加工 P10 的二十六项接口，输出可追溯的跨市场冲击和财务、利率比较。

    例如同日已认证 USD 名义和实际十年收益率生成利差；代码相同且实体不同保持分开。
    没有汇率和 A/H 显式关系时不给出跨币股票价差，不把实时观察填入历史日线。
    """

    def process(frame: Frame) -> tuple[list[Metric], dict[str, Any]]:
        """绑定当前截止和关系配置；例如 A/H 行可查询本次已确认的配对。"""
        return _process(frame, inputs)

    results = run_sources(inputs, PRODUCT_ID, API_FIELDS, DIMENSIONS, process)
    results.extend(_breakeven(inputs))
    return tuple(results)
