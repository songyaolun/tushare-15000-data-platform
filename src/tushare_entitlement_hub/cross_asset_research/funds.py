"""P08 ETF本体研究：明确基准穿透、申赎篮子、流动性及公告后净值核对。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ..equity_research.common import gate, latest, make_result, metric
from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..product_foundation.contracts import EvidenceRef, Truth
from .thematic_common import (
    DETAIL_SAMPLE,
    accepted_groups,
    comparable,
    date_text,
    finite,
    latest_snapshot,
    number,
    observation_summary,
    sampled_refs,
    scalar,
    text_values,
    unique_latest_by,
)

API_FIELDS = {
    "etf_basic": ("ts_code", "index_code", "etf_type", "mgr_name", "mgt_fee"),
    "etf_index": ("ts_code", "pub_party_name", "adj_circle"),
    "etf_limit": ("trade_date", "ts_code", "up_limit", "down_limit"),
    "etf_sh_cons": ("trade_date", "con_code", "qty", "sub_flag", "cpr", "rdr", "sca"),
    "etf_share_size": ("trade_date", "total_share", "total_size", "nav", "close"),
    "etf_sz_cons": ("trade_date", "con_code", "qty", "sub_flag", "cpr", "rdr", "sub_cc", "red_cc"),
    "fund_adj": ("trade_date", "adj_factor"),
    "fund_basic": ("ts_code", "management", "fund_type", "market", "m_fee", "c_fee", "benchmark"),
    "fund_company": ("name", "shortname", "org_code", "credit_code"),
    "fund_daily": ("trade_date", "close", "amount", "vol"),
    "fund_div": ("ann_date", "ex_date", "div_cash", "div_proc", "net_ex_date"),
    "fund_factor_pro": ("trade_date", "close", "vol", "amount"),
    "fund_manager": ("name", "ann_date", "begin_date", "end_date"),
    "fund_nav": ("ann_date", "nav_date", "unit_nav", "adj_nav", "net_asset"),
    "fund_share": ("trade_date", "fd_share"),
    "mkt_idx_bmk": ("ts_code", "bmk_level", "bmk_type", "bmk_src", "idx_type"),
    "index_weight": ("index_code", "con_code", "trade_date", "weight"),
    "slb_len_mm": ("ts_code", "trade_date", "ope_inv", "lent_qnt", "cls_inv", "end_bal"),
}
REFERENCE_APIS = {"etf_index", "mkt_idx_bmk", "index_weight", "fund_company"}


def _traded(row: ResearchRow | None) -> bool:
    """只有认证正价格、正量额及非参考报价才可用于基金价格收益和交易距离。

    输入close=10、vol=1、amount=10返回True；相同价格带no_trade状态返回False。
    """
    return (
        row is not None
        and all(
            (value := number(row, field, unit)) is not None and value > 0
            for field, unit in (("close", "元"), ("vol", "手"), ("amount", "元"))
        )
        and row.raw.get("market_activity_status") != "no_trade"
        and row.raw.get("price_observation_status")
        not in {"reference_close_without_trade", "no_price_observed"}
        and row.raw.get("price_type")
        not in {"reference", "reference_close", "reference_price", "settlement"}
    )


def _same_day(left: ResearchRow | None, right: ResearchRow | None) -> bool:
    """判断两个来源是否有同一明确数据日，不将未知日期相等当成对齐。

    输入两条2026-09-30记录返回True；两条data_as_of=None返回False。
    """
    return (
        left is not None
        and right is not None
        and left.availability.data_as_of is not None
        and left.availability.data_as_of == right.availability.data_as_of
    )


def _relative(
    name: str,
    left: ResearchRow | None,
    left_field: str,
    right: ResearchRow | None,
    right_field: str,
) -> Metric:
    """计算同日同量纲的相对差异，分母必须为正，缺单位或错日保留未知。

    输入ETF价格11元、同日净值10元得到0.1；价格11元、净值10港元返回unknown。
    """
    valid = (
        _same_day(left, right)
        and left is not None
        and right is not None
        and comparable(left, left_field, right, right_field)
    )
    a, b = number(left, left_field), number(right, right_field)
    value = finite(a / b - 1) if valid and a is not None and b is not None and b > 0 else None
    refs = (() if left is None else left.evidence(left_field)) + (
        () if right is None else right.evidence(right_field)
    )
    return metric(
        name,
        value,
        "ratio",
        refs,
        () if value is not None else ("aligned_comparable_values_unavailable",),
    )


def _fund_observations(
    sources: dict[str, tuple[ResearchRow, ...]],
) -> tuple[list[Metric], dict[str, Any]]:
    """加工基金行情、规模、净值和借券供给，金额只使用已认证元口径。

    输入同日成交额100元、规模1000元，输出成交规模比0.1；缺价格不影响规模观察。
    借券库存变化只描述做市供给，不解释成投资者做空或实际资金净流入。
    """
    quote = latest(sources.get("fund_daily", ()))
    size = latest(sources.get("etf_share_size", ()))
    nav = latest(sources.get("fund_nav", ()))
    nav = nav if nav is not None and nav.availability.public_available_at is not None else None
    share = latest(sources.get("fund_share", ()))
    factor = latest(sources.get("fund_factor_pro", ()))
    lending = latest(sources.get("slb_len_mm", ()))
    metrics = [
        scalar("fund_close", quote, "close", "元"),
        scalar("fund_turnover_amount", quote, "amount", "元"),
        scalar("fund_volume_lots", quote, "vol", "手"),
        scalar("etf_total_size", size, "total_size", "元"),
        scalar("etf_total_shares", size, "total_share", "份"),
        scalar("etf_nav", size, "nav", "元"),
        scalar("lending_inventory", lending, "cls_inv", "股"),
        scalar("lending_quantity", lending, "lent_qnt", "股"),
        scalar("lending_balance", lending, "end_bal", "元"),
        _relative("price_nav_premium", size, "close", size, "nav"),
        _relative("published_nav_difference", nav, "unit_nav", size, "nav"),
        _relative("adjusted_nav_uplift", nav, "adj_nav", nav, "unit_nav"),
        _relative("published_share_difference", share, "fd_share", size, "total_share"),
        _relative("provider_price_difference", factor, "close", quote, "close"),
        _relative("provider_amount_difference", factor, "amount", quote, "amount"),
        _relative("provider_volume_difference", factor, "vol", quote, "vol"),
        _relative("published_net_asset_difference", nav, "net_asset", size, "total_size"),
    ]
    amount, assets = number(quote, "amount", "元"), number(size, "total_size", "元")
    valid = (
        _same_day(quote, size)
        and amount is not None
        and amount >= 0
        and assets is not None
        and assets > 0
    )
    refs = (() if quote is None else quote.evidence("amount")) + (
        () if size is None else size.evidence("total_size")
    )
    metrics.append(
        metric(
            "turnover_to_assets",
            finite(cast(float, amount) / cast(float, assets)) if valid else None,
            "ratio",
            refs,
            () if valid else ("aligned_assets_denominator_unavailable",),
        )
    )
    shares, net_asset, unit_nav = (
        number(size, "total_share", "份"),
        number(size, "total_size", "元"),
        number(size, "nav", "元"),
    )
    valid_size = (
        shares is not None
        and shares >= 0
        and net_asset is not None
        and net_asset > 0
        and unit_nav is not None
        and unit_nav > 0
    )
    gap = (
        finite(cast(float, shares) * cast(float, unit_nav) / cast(float, net_asset) - 1)
        if valid_size
        else None
    )
    metrics.append(
        metric(
            "size_nav_identity_gap",
            gap,
            "ratio",
            () if size is None else size.evidence("total_share", "total_size", "nav"),
            () if gap is not None else ("share_nav_asset_units_unverified",),
        )
    )
    opening, closing = number(lending, "ope_inv", "股"), number(lending, "cls_inv", "股")
    delta = finite(closing - opening) if opening is not None and closing is not None else None
    metrics.append(
        metric(
            "lending_inventory_change",
            delta,
            "股",
            () if lending is None else lending.evidence("ope_inv", "cls_inv"),
            () if delta is not None else ("lending_inventory_units_unverified",),
        )
    )
    traded = _traded(quote)
    details = {
        "quote_date": None if quote is None else quote.availability.data_as_of,
        "size_date": None if size is None else size.availability.data_as_of,
        "price_status": "traded_observation" if traded else "price_activity_unverified",
        "lending_interpretation": "market_maker_supply_not_investor_short_position",
    }
    return metrics, details


def _shares(rows: Sequence[ResearchRow]) -> tuple[list[Metric], dict[str, Any]]:
    """用最近两个唯一份额快照计算份额变化及同日净值估值的申赎代理。

    输入份额100到110份、新净值2元，得到份额变化10份、估值代理20元；
    该代理不声称实际净流入，拆并份等公司行动未证实会在原因中单列。
    """
    unique, conflicts = unique_latest_by(rows, ("trade_date",))
    ordered = sorted(unique, key=lambda row: row.availability.data_as_of or "")
    before, after = (ordered[-2], ordered[-1]) if len(ordered) >= 2 else (None, None)
    a, b = number(before, "total_share", "份"), number(after, "total_share", "份")
    valid = (
        not conflicts
        and before is not None
        and after is not None
        and before.availability.data_as_of is not None
        and before.availability.data_as_of < (after.availability.data_as_of or "")
        and a is not None
        and a > 0
        and b is not None
        and b >= 0
    )
    change = finite(cast(float, b) - cast(float, a)) if valid else None
    refs = sampled_refs(
        tuple(row for row in (before, after) if row is not None), API_FIELDS["etf_share_size"]
    )
    nav = number(after, "nav", "元")
    proxy = finite(change * nav) if change is not None and nav is not None and nav > 0 else None
    return [
        metric(
            "share_change",
            change,
            "份",
            refs,
            () if change is not None else ("comparable_share_history_unavailable",),
        ),
        metric(
            "share_change_ratio",
            finite(change / a) if change is not None and a is not None else None,
            "ratio",
            refs,
            () if change is not None else ("comparable_share_history_unavailable",),
        ),
        metric(
            "share_change_nav_proxy",
            proxy,
            "元",
            refs,
            ("subscription_proxy_not_actual_cash_flow",)
            if proxy is not None
            else ("share_nav_proxy_unavailable",),
        ),
    ], {
        "start_date": None if before is None else before.availability.data_as_of,
        "end_date": None if after is None else after.availability.data_as_of,
        "snapshot_conflicts": conflicts,
        "corporate_action_adjustment": "unverified",
    }


def _price_limits(sources: dict[str, tuple[ResearchRow, ...]]) -> Metric:
    """计算有效成交价格到涨跌停的最小相对距离，未知边界不默认可交易。

    输入收盘10、下限9、上限11且存在成交，返回0.1；收盘超出边界返回unknown。
    """
    quote, limits = latest(sources.get("fund_daily", ())), latest(sources.get("etf_limit", ()))
    close, low, high = (
        number(quote, "close", "元"),
        number(limits, "down_limit", "元"),
        number(limits, "up_limit", "元"),
    )
    volume, amount = number(quote, "vol", "手"), number(quote, "amount", "元")
    valid = (
        _same_day(quote, limits)
        and close is not None
        and low is not None
        and high is not None
        and 0 < low <= close <= high
        and volume is not None
        and volume > 0
        and amount is not None
        and amount > 0
        and _traded(quote)
    )
    refs = (() if quote is None else quote.evidence("close", "vol", "amount")) + (
        () if limits is None else limits.evidence("down_limit", "up_limit")
    )
    return metric(
        "price_limit_distance",
        min(cast(float, close) - cast(float, low), cast(float, high) - cast(float, close))
        / cast(float, close)
        if valid
        else None,
        "ratio",
        refs,
        () if valid else ("fund_trade_or_limit_unverified",),
    )


def _adjusted_return(
    sources: dict[str, tuple[ResearchRow, ...]],
    benchmark: Sequence[ResearchRow],
    benchmark_history: tuple[dict[str | None, ResearchRow], int] | None = None,
) -> tuple[list[Metric], dict[str, Any]]:
    """固定首尾锚点核对基金复权收益，基准收益必须同日且显式收益口径一致。

    输入首尾收盘10、11及因子1、1，返回0.1；基准缺同端点或收益口径时跟踪差未知。
    不把累计复权因子与净值任意相乘，也不从两个观察值声称年化跟踪误差。
    """
    quotes, quote_conflicts = unique_latest_by(sources.get("fund_daily", ()), ("trade_date",))
    factors, factor_conflicts = unique_latest_by(sources.get("fund_adj", ()), ("trade_date",))
    factor_by_day = {row.availability.data_as_of: row for row in factors}
    ordered = sorted(quotes, key=lambda row: row.availability.data_as_of or "")
    points: list[tuple[ResearchRow, ResearchRow, float]] = []
    for quote in ordered:
        factor = factor_by_day.get(quote.availability.data_as_of)
        close, value = number(quote, "close", "元"), number(factor, "adj_factor", "ratio")
        if (
            quote.availability.data_as_of is not None
            and factor is not None
            and close is not None
            and close > 0
            and value is not None
            and value > 0
            and _traded(quote)
        ):
            adjusted = finite(close * value)
            if adjusted is not None:
                points.append((quote, factor, adjusted))
    valid = (
        not quote_conflicts
        and not factor_conflicts
        and len(points) >= 2
        and len(points) == len(quotes)
    )
    result = finite(points[-1][2] / points[0][2] - 1) if valid else None
    endpoint_rows = (
        tuple(row for point in (points[0], points[-1]) for row in point[:2]) if valid else ()
    )
    refs = sampled_refs(endpoint_rows, ("close", "adj_factor", "trade_date"))
    metrics = [
        metric(
            "adjusted_price_return",
            result,
            "ratio",
            refs,
            () if result is not None else ("fixed_anchor_adjustment_unavailable",),
        )
    ]
    if benchmark_history is None:
        index_rows, conflicts = unique_latest_by(benchmark, ("trade_date",))
        by_day = {row.availability.data_as_of: row for row in index_rows}
    else:
        by_day, conflicts = benchmark_history
    start = by_day.get(points[0][0].availability.data_as_of) if valid else None
    end = by_day.get(points[-1][0].availability.data_as_of) if valid else None
    basis = points[-1][0].text("return_basis") if valid else None
    comparable_basis = (
        start is not None
        and end is not None
        and basis is not None
        and start.text("return_basis") == basis
        and end.text("return_basis") == basis
    )
    first, last = number(start, "close"), number(end, "close")
    comparable_index = (
        start is not None and end is not None and comparable(start, "close", end, "close")
    )
    difference = (
        finite(result - (last / first - 1))
        if result is not None
        and not conflicts
        and comparable_basis
        and comparable_index
        and first is not None
        and first > 0
        and last is not None
        and last > 0
        else None
    )
    index_refs = sampled_refs(
        tuple(row for row in (start, end) if row is not None),
        ("close", "trade_date", "return_basis"),
    )
    metrics.append(
        metric(
            "benchmark_tracking_difference",
            difference,
            "ratio",
            refs + index_refs,
            () if difference is not None else ("aligned_benchmark_return_basis_unverified",),
        )
    )
    return metrics, {
        "anchor_start": points[0][0].availability.data_as_of if valid else None,
        "anchor_end": points[-1][0].availability.data_as_of if valid else None,
        "tracking_measure": "endpoint_return_difference_not_annualized_error",
    }


def _basket(
    api: str, rows: Sequence[ResearchRow]
) -> tuple[list[Metric], dict[str, Any], set[str], str | None]:
    """加工单交易所申赎篮子，股数不跨证券相加当经济权重，现金替代字段分源保留。

    输入两名成分且一名qty=0，输出正股数成分占比0.5；输出sub_flag分布及认证替代率。
    缺少一名成分股数时整体占比未知，不把已知一条当完整篮子。
    """
    snapshot = latest_snapshot(rows)
    members, conflicts = unique_latest_by(snapshot, ("con_code",))
    valid = bool(members) and not conflicts
    refs = sampled_refs(rows, API_FIELDS[api])
    quantities = [number(row, "qty", "股") for row in members]
    known = valid and all(value is not None and value >= 0 for value in quantities)
    metrics = [
        metric(
            api + "_member_count",
            len(members) if valid else None,
            "count",
            refs,
            () if valid else ("basket_membership_unverified",),
        ),
        metric(
            api + "_positive_quantity_fraction",
            sum(value > 0 for value in quantities if value is not None) / len(members)
            if known
            else None,
            "fraction",
            refs,
            () if known else ("basket_quantity_unverified",),
        ),
    ]
    for field in ("cpr", "rdr"):
        values = [number(row, field, "fraction") for row in members]
        complete = valid and all(value is not None and value >= 0 for value in values)
        average = (
            finite(sum(value for value in values if value is not None) / len(members))
            if complete
            else None
        )
        metrics.append(
            metric(
                api + "_mean_" + field,
                average,
                "ratio",
                refs,
                () if average is not None else ("cash_substitution_rate_unverified",),
            )
        )
    cash_fields = ("sca",) if api == "etf_sh_cons" else ("sub_cc", "red_cc")
    for field in cash_fields:
        values = [number(row, field, "元") for row in members]
        complete = valid and all(value is not None for value in values)
        total = finite(sum(value for value in values if value is not None)) if complete else None
        metrics.append(
            metric(
                api + "_" + field,
                total,
                "元",
                refs,
                () if total is not None else ("cash_substitution_amount_unverified",),
            )
        )
    details: dict[str, Any] = {
        "snapshot_date": snapshot[0].availability.data_as_of if snapshot else None,
        "substitution_flags": text_values(members, "sub_flag"),
        "member_conflicts": conflicts,
        "quantity_scope": "shares_per_security_not_portfolio_weights",
    }
    return (
        metrics,
        details,
        {code for row in members if (code := row.text("con_code")) is not None} if valid else set(),
        details["snapshot_date"],
    )


def _weights(
    rows: Sequence[ResearchRow],
) -> tuple[list[Metric], dict[str, Any], set[str], str | None]:
    """按完整可见指数权重快照计算观察覆盖与集中度，未知单位禁止把源值默认为百分比。

    输入认证权重0.6、0.4，输出覆盖1和HHI=.52；只有0.6的观察子集标明覆盖0.6。
    同一成分冲突或权重总和超过1时不产生集中度，也不靠归一化掩盖异常。
    """
    snapshot = latest_snapshot(rows)
    members, conflicts = unique_latest_by(snapshot, ("con_code",))
    values = [number(row, "weight", "fraction") for row in members]
    total = finite(sum(value for value in values if value is not None))
    valid = (
        bool(members)
        and not conflicts
        and all(value is not None and 0 <= value <= 1 for value in values)
        and total is not None
        and 0 < total <= 1.00000001
    )
    refs = sampled_refs(rows, API_FIELDS["index_weight"])
    metrics = [
        metric(
            "benchmark_observed_weight",
            min(cast(float, total), 1.0) if valid else None,
            "fraction",
            refs,
            () if valid else ("benchmark_weights_unverified",),
        ),
        metric(
            "benchmark_observed_hhi",
            sum((value / cast(float, total)) ** 2 for value in values if value is not None)
            if valid
            else None,
            "ratio",
            refs,
            () if valid else ("benchmark_weights_unverified",),
        ),
    ]
    details: dict[str, Any] = {
        "snapshot_date": snapshot[0].availability.data_as_of if snapshot else None,
        "constituent_count": len(members),
        "weight_conflicts": conflicts,
        "exposure_scope": "observed_benchmark_not_fund_actual_holdings",
        "largest_exposure_samples": [
            {"constituent": row.text("con_code"), "weight": value}
            for row, value in sorted(
                zip(members, values, strict=True),
                key=lambda pair: (-(pair[1] or 0), pair[0].text("con_code") or ""),
            )[:DETAIL_SAMPLE]
        ],
    }
    codes = {code for row in members if (code := row.text("con_code")) is not None}
    return metrics, details, codes if not conflicts else set(), details["snapshot_date"]


def _events(
    inputs: ResearchInputs, sources: dict[str, tuple[ResearchRow, ...]]
) -> tuple[list[Metric], dict[str, Any]]:
    """消费基金分红阶段、除息日和经理任职变动，不将计划分红计成已到账现金。

    输入一条已实施且到除息日的分红返回一件有效事件；未到除息日保持计划观察。
    经理按姓名和任职起点去重，任职结束日保留用于当前任职判断。
    """
    dividends = sources.get("fund_div", ())
    implemented = [
        row
        for row in dividends
        if row.text("div_proc") in {"实施", "已实施", "completed"}
        and (day := date_text(row.text("ex_date"))) is not None
        and day <= inputs.analysis_as_of[:10]
    ]
    dividend_keys = {
        (row.text("ann_date"), row.text("ex_date"), row.text("div_proc")) for row in implemented
    }
    managers, conflicts = unique_latest_by(sources.get("fund_manager", ()), ("name", "begin_date"))
    active = [
        row
        for row in managers
        if (begin := date_text(row.text("begin_date"))) is not None
        and begin <= inputs.analysis_as_of[:10]
        and (
            row.text("end_date") is None
            or (end := date_text(row.text("end_date"))) is not None
            and end > inputs.analysis_as_of[:10]
        )
    ]
    known = (
        bool(managers)
        and not conflicts
        and all(
            date_text(row.text("begin_date")) is not None
            and (row.text("end_date") is None or date_text(row.text("end_date")) is not None)
            for row in managers
        )
    )
    refs = sampled_refs(dividends, API_FIELDS["fund_div"])
    metrics = [
        metric(
            "implemented_dividend_events",
            len(dividend_keys) if dividends else None,
            "count",
            refs,
            () if dividends else ("dividend_event_coverage_unknown",),
        ),
        metric(
            "active_fund_managers",
            len(active) if known else None,
            "count",
            sampled_refs(managers, API_FIELDS["fund_manager"]),
            () if known else ("manager_tenure_unverified",),
        ),
    ]
    latest_dividend = latest(tuple(implemented))
    metrics.append(scalar("latest_dividend_cash", latest_dividend, "div_cash", "元"))
    return metrics, {
        "dividend_stages": text_values(dividends, "div_proc"),
        "net_value_ex_dates": text_values(dividends, "net_ex_date"),
        "manager_names": text_values(managers, "name"),
        "manager_change_events": len(managers),
        "manager_performance_attribution": "not_inferred",
    }


def _reference_index(inputs: ResearchInputs) -> dict[tuple[str, str], tuple[ResearchRow, ...]]:
    """一次索引已可见参考资料，按明确来源代码查找，不对每只ETF重扫全部指数成分。

    输入index_weight的I指数1000行，输出('index_weight','I')一个引用元组。
    fund_company按完整name匹配管理人，不用简称猜主体；歧义交给调用方检查。
    """
    result: dict[tuple[str, str], list[ResearchRow]] = {}
    groups = accepted_groups(inputs, REFERENCE_APIS | {"index_daily"})
    for sources in groups.values():
        for api, rows in sources.items():
            field = (
                "index_code"
                if api == "index_weight"
                else "name"
                if api == "fund_company"
                else "ts_code"
            )
            for row in rows:
                code = row.text(field)
                if code is not None:
                    result.setdefault((api, code), []).append(row)
    return {key: tuple(rows) for key, rows in result.items()}


def build_etf_exposure(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """把18个基金接口加工为ETF本体、明确基准暴露、篮子重合及时点一致的风险观察。

    输入ETF档案、日线和规模两日数据，即使没有用户组合，也输出流动性和份额变化；
    portfolio_overlap保持unknown。场外fund_basic.market='O'不进入ETF候选。
    所有成分、管理人和指数匹配均需明确来源代码或全名证据，不按名称相似度猜关系。
    """
    groups = accepted_groups(inputs, API_FIELDS.keys() - REFERENCE_APIS)
    reference = _reference_index(inputs)
    weight_cache = {}
    history_cache = {}
    summary_cache = {}
    evidence_cache = {}
    for key, rows in reference.items():
        api, code = key
        if api == "index_weight":
            weight_cache[code] = _weights(rows)
        if api == "index_daily":
            history, conflicts = unique_latest_by(rows, ("trade_date",))
            history_cache[code] = ({row.availability.data_as_of: row for row in history}, conflicts)
        if api in API_FIELDS:
            summary_cache[key] = observation_summary(rows, API_FIELDS[api])
            evidence_cache[key] = sampled_refs(rows, API_FIELDS[api])
    manager_counts: dict[str, int] = {}
    for sources in groups.values():
        basic = latest(sources.get("etf_basic", ()))
        if basic is not None and basic.text("mgr_name") is not None:
            name = basic.text("mgr_name") or ""
            manager_counts[name] = manager_counts.get(name, 0) + 1
    results = []
    for entity, sources in sorted(groups.items()):
        basic, fund = latest(sources.get("etf_basic", ())), latest(sources.get("fund_basic", ()))
        if basic is None and fund is not None and fund.text("market") == "O":
            continue
        index_code = None if basic is None else basic.text("index_code")
        manager = None if basic is None else basic.text("mgr_name")
        attached = dict(sources)
        for api in ("etf_index", "mkt_idx_bmk", "index_weight", "index_daily"):
            attached[api] = reference.get((api, index_code or ""), ())
        attached["fund_company"] = reference.get(("fund_company", manager or ""), ())
        metrics, observation = _fund_observations(sources)
        share_metrics, share_details = _shares(sources.get("etf_share_size", ()))
        metrics.extend(share_metrics)
        metrics.append(_price_limits(sources))
        return_metrics, return_details = _adjusted_return(
            sources, attached["index_daily"], history_cache.get(index_code or "")
        )
        metrics.extend(return_metrics)
        weight_metrics, weights, codes, weight_day = weight_cache.get(
            index_code or "", _weights(())
        )
        metrics.extend(weight_metrics)
        baskets = {}
        for api in ("etf_sh_cons", "etf_sz_cons"):
            basket_metrics, basket, basket_codes, day = _basket(api, sources.get(api, ()))
            metrics.extend(basket_metrics)
            aligned = bool(day and day == weight_day and codes and basket_codes)
            refs = sampled_refs(sources.get(api, ()), API_FIELDS[api]) + sampled_refs(
                attached["index_weight"], API_FIELDS["index_weight"]
            )
            metrics.append(
                metric(
                    api + "_benchmark_observed_overlap",
                    len(codes & basket_codes) / len(codes | basket_codes) if aligned else None,
                    "fraction",
                    refs,
                    () if aligned else ("aligned_basket_benchmark_unavailable",),
                )
            )
            baskets[api] = basket
        event_metrics, events = _events(inputs, sources)
        metrics.extend(event_metrics)
        company_rows = attached["fund_company"]
        company_ids = {(row.text("org_code"), row.text("credit_code")) for row in company_rows}
        manager_known = len(company_ids) == 1 and any(next(iter(company_ids)))
        metrics.append(
            metric(
                "same_manager_observed_etfs",
                manager_counts.get(manager or "", 0) if manager_known else None,
                "count",
                sampled_refs(company_rows, API_FIELDS["fund_company"])
                + (() if basic is None else basic.evidence("mgr_name")),
                () if manager_known else ("manager_legal_identity_unverified",),
            )
        )
        metrics.extend(
            [
                scalar("etf_management_fee", basic, "mgt_fee", "fraction"),
                scalar("fund_management_fee", fund, "m_fee", "fraction"),
                scalar("fund_custody_fee", fund, "c_fee", "fraction"),
                metric(
                    "portfolio_overlap", None, "fraction", reasons=("user_portfolio_not_provided",)
                ),
            ]
        )
        references: list[EvidenceRef] = []
        source_details = {}
        for api in API_FIELDS:
            rows = attached.get(api, ())
            if not rows and api not in sources:
                continue
            lookup = (api, (manager or "") if api == "fund_company" else (index_code or ""))
            if lookup in summary_cache:
                references.extend(evidence_cache[lookup])
                source_details[api] = dict(summary_cache[lookup])
            else:
                references.extend(sampled_refs(rows, API_FIELDS[api]))
                source_details[api] = observation_summary(rows, API_FIELDS[api])
            source_details[api]["reason_codes"] = list(inputs.reasons(api, entity))
        metadata = {
            api: {field: text_values(attached.get(api, ()), field) for field in fields}
            for api, fields in {
                "etf_basic": ("etf_type", "index_code", "mgr_name"),
                "fund_basic": ("management", "fund_type", "market", "benchmark"),
                "fund_company": ("name", "shortname", "org_code", "credit_code"),
                "etf_index": ("pub_party_name", "adj_circle"),
                "mkt_idx_bmk": ("bmk_level", "bmk_type", "bmk_src", "idx_type"),
            }.items()
        }
        details = {
            "sources": source_details,
            "metadata": metadata,
            "fund_observation": observation,
            "share_changes": share_details,
            "adjusted_return": return_details,
            "benchmark_exposure": weights,
            "subscription_baskets": baskets,
            "events": events,
            "benchmark_index_code": index_code,
            "benchmark_mapping_basis": "etf_basic_explicit_index_code"
            if index_code is not None
            else "unknown",
            "product_scope": "etf_standalone",
            "historical_pit_certified": False,
        }
        results.append(
            make_result(
                inputs,
                "etf_exposure/v1",
                entity,
                metrics=metrics,
                references=references,
                gates=(
                    gate(
                        "etf_identity",
                        Truth.TRUE if basic is not None else Truth.UNKNOWN,
                        "etf_identity_unverified",
                    ),
                    gate(
                        "etf_benchmark_mapping",
                        Truth.TRUE if index_code is not None else Truth.UNKNOWN,
                        "benchmark_mapping_unverified",
                    ),
                ),
                details=details,
                profile="etf_standalone_observation/v1",
            )
        )
    return tuple(results)
