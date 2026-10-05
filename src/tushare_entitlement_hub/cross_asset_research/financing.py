"""P11：融资流量、资金曲线、债券报价与期权尾部压力。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, time
from typing import Any

from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..store import digest
from .market_support import (
    Frame,
    comparable,
    difference,
    finish,
    frames,
    number,
    ratio,
    run_sources,
    singleton,
    spread,
    unit,
)

PRODUCT_ID = "financing_tail_risk/v1"
API_FIELDS = {
    "bc_bestotcqt": (
        "ts_code",
        "trade_date",
        "remain_maturity",
        "best_buy_yield",
        "best_sell_yield",
    ),
    "bc_otcqt": ("ts_code", "qt_time", "bank", "remain_maturity", "buy_yield", "sell_yield"),
    "bond_blk": ("ts_code", "trade_date", "price", "vol", "amount"),
    "bond_blk_detail": ("ts_code", "trade_date", "price", "amount", "buy_dp", "sell_dp"),
    "repo_daily": ("trade_date", "repo_maturity", "close", "weight", "amount"),
    "vix_index": ("trade_date", "close", "pct_change"),
    "gz_index": ("date", "d10_rate", "m1_rate", "m3_rate", "long_rate"),
    "shibor": ("date", "on", "1w", "1m", "3m", "1y"),
    "shibor_quote": ("date", "bank", "on_b", "on_a", "1m_b", "1m_a"),
    "wz_index": ("date", "comp_rate", "micro_rate", "m1_rate", "long_rate"),
    "margin": ("trade_date", "exchange_id", "rzye", "rzmre", "rzche", "rzrqye"),
    "slb_len": ("trade_date", "ob", "auc_amount", "repo_amount", "repay_amount", "cb"),
    "opt_basic": (
        "ts_code",
        "opt_code",
        "opt_type",
        "call_put",
        "exercise_type",
        "exercise_price",
        "maturity_date",
        "opt_multiplier",
    ),
    "opt_daily": ("ts_code", "trade_date", "vol", "oi", "close", "settle"),
}
API_PURPOSES = {
    "bc_bestotcqt": "最优买卖收益率差，与银行明细核对，不代表成交",
    "bc_otcqt": "同券同剩余期限最新银行报价极差",
    "bond_blk": "大宗成交价格分散及成交规模，作为汇总来源",
    "bond_blk_detail": "大宗明细成交价格分散及对手席位集中，不与汇总双计",
    "repo_daily": "收盘融资利率对加权利率偏离，close 禁作证券价格",
    "vix_index": "美国隐含波动变化背景，不代表中国市场波动率",
    "gz_index": "广州样本长短融资期限差，保持地方样本边界",
    "shibor": "人民币隔夜、一年和一月资金期限差",
    "shibor_quote": "银行隔夜和月度报价离散及双边差异",
    "wz_index": "温州长短融资及小微综合融资差，不与广州合并",
    "margin": "交易所融资净买入相对余额及融资余额占两融比例",
    "slb_len": "转融资偿还压力和期初期末资金恒等式核对",
    "opt_basic": "期权合约规格及到期、看涨看跌语义完整性",
    "opt_daily": "单合约持仓成交比及显式同标的到期组看跌看涨比",
}
DIMENSIONS: dict[str, tuple[str, ...]] = {
    "bc_otcqt": ("remain_maturity",),
    "bc_bestotcqt": ("remain_maturity",),
    "repo_daily": ("repo_maturity",),
    "margin": ("exchange_id",),
}


def _quotes(frame: Frame) -> tuple[list[Metric], dict[str, Any]]:
    """同银行只保留最新报价再计算横截面离散；例如两银行 2%、3% 极差为 1%。

    时标冲突、银行缺失和混合单位会阻断，不把逐笔报价数量当成交量。
    """
    latest: dict[str, ResearchRow] = {}
    latest_times: dict[str, str] = {}
    for row in frame.rows:
        bank, quote_time = row.text("bank"), row.text("qt_time")
        if bank is None or frame.api == "bc_otcqt" and quote_time is None:
            frame.reasons.add("bank_quote_identity_missing")
            continue
        try:
            normalized_time = time.fromisoformat(quote_time or "00:00:00").isoformat()
        except ValueError:
            frame.reasons.add("bank_quote_time_unverified")
            continue
        previous = latest.get(bank)
        if previous is None or normalized_time > latest_times[bank]:
            latest[bank] = row
            latest_times[bank] = normalized_time
        elif normalized_time == latest_times[bank]:
            frame.reasons.add("bank_quote_time_conflict")
    saved = frame.rows
    frame.rows = list(latest.values())
    fields = (
        ("buy_yield", "sell_yield") if frame.api == "bc_otcqt" else ("on_b", "on_a", "1m_b", "1m_a")
    )
    values = [
        frame.value(name + "_bank_range", *spread(frame, name), name, "bank") for name in fields
    ]
    frame.rows = saved
    return values, {"bank_count": len(latest), "observation_type": "quote_not_execution"}


def _block(frame: Frame) -> tuple[list[Metric], dict[str, Any]]:
    """计算大宗成交离散和最大单笔占比；例如 60、40 元两笔最大成交占比为 0.6。

    汇总和明细分别计算，二者从不相加；未知成交量不影响已认证金额的来源内占比。
    """
    price_range, price_unit = spread(frame, "price")
    amounts = [number(row, "amount", ("money",)) for row in frame.rows]
    valid = (
        bool(amounts)
        and all(value is not None and value >= 0 for value in amounts)
        and all(comparable(frame.rows[0], "amount", row, "amount") for row in frame.rows)
    )
    total = sum(value for value in amounts if value is not None)
    maximum_share = (
        max(value for value in amounts if value is not None) / total
        if valid and total > 0
        else None
    )
    return [
        frame.value("block_price_range", price_range, price_unit, "price"),
        frame.value("largest_block_amount_share", maximum_share, "fraction", "amount"),
    ], {"observation_type": "execution", "aggregation_policy": "summary_and_detail_never_added"}


def _process(frame: Frame) -> tuple[list[Metric], dict[str, Any]]:
    """加工逐接口融资指标；例如买入 30、偿还 20、余额 100 得净融资强度 0.1。

    回购 close 按利率证书读取，债券报价与成交采用独立指标和观察类型。
    """
    api = frame.api
    details: dict[str, Any] = {"purpose": API_PURPOSES[api]}
    if api in {"bc_otcqt", "shibor_quote"}:
        values, extra = _quotes(frame)
        return values, {**details, **extra}
    if api in {"bond_blk", "bond_blk_detail"}:
        values, extra = _block(frame)
        return values, {**details, **extra}
    row = singleton(frame)
    if row is None:
        return [frame.value("snapshot_usable", None)], details
    if api == "opt_basic":
        names = ("opt_type", "call_put", "exercise_type", "maturity_date", "opt_multiplier")
        details["underlying_mapping"] = "explicit_option_underlying_relation_required"
        details["option_side"] = {"C": "call", "P": "put", "认购": "call", "认沽": "put"}.get(
            row.text("call_put") or "", "unknown"
        )
        return [
            frame.value(
                "contract_specification_completeness",
                sum(row.text(name) is not None for name in names) / len(names),
                "fraction",
                *names,
            )
        ], details
    pairs = {
        "bc_bestotcqt": (("best_buy_sell_yield_gap", "best_buy_yield", "best_sell_yield"),),
        "repo_daily": (("closing_weighted_funding_spread", "close", "weight"),),
        "gz_index": (("guangzhou_long_ten_day_spread", "long_rate", "d10_rate"),),
        "wz_index": (
            ("wenzhou_long_month_spread", "long_rate", "m1_rate"),
            ("wenzhou_micro_composite_spread", "micro_rate", "comp_rate"),
        ),
        "shibor": (
            ("one_year_overnight_spread", "1y", "on"),
            ("one_month_overnight_spread", "1m", "on"),
        ),
    }
    values = [
        frame.value(
            name, difference(row, left, right, ("ratio", "rate")), unit(row, left), left, right
        )
        for name, left, right in pairs.get(api, ())
    ]
    if api == "bc_bestotcqt":
        details["observation_type"] = "quote_not_execution"
    if api == "repo_daily":
        details["close_semantics"] = "funding_rate_not_security_price"
        if number(row, "close", ("ratio", "rate")) is None:
            frame.reasons.add("repo_rate_dimension_unverified")
            values = [frame.value("closing_weighted_funding_spread", None)]
    if api == "margin":
        net = difference(row, "rzmre", "rzche")
        balance = number(row, "rzye", ("money",))
        valid = comparable(row, "rzmre", row, "rzye")
        values.extend(
            (
                frame.value(
                    "net_financing_balance_ratio",
                    net / balance
                    if net is not None and balance is not None and balance > 0 and valid
                    else None,
                    "ratio",
                    "rzmre",
                    "rzche",
                    "rzye",
                ),
                frame.value(
                    "financing_margin_balance_share",
                    ratio(row, "rzye", "rzrqye"),
                    "ratio",
                    "rzye",
                    "rzrqye",
                ),
            )
        )
    if api == "slb_len":
        names = ("ob", "auc_amount", "repo_amount", "repay_amount", "cb")
        amounts = [number(row, name, ("money",)) for name in names]
        compatible = all(comparable(row, "ob", row, name) for name in names)
        residual = None
        if all(amount is not None for amount in amounts) and compatible:
            opening, auction, repo, repay, closing = [
                amount for amount in amounts if amount is not None
            ]
            residual = closing - (opening + auction + repo - repay)
        values.extend(
            (
                frame.value(
                    "refinancing_balance_identity_residual", residual, unit(row, "ob"), *names
                ),
                frame.value(
                    "repayment_opening_balance_ratio",
                    ratio(row, "repay_amount", "ob"),
                    "ratio",
                    "repay_amount",
                    "ob",
                ),
            )
        )
    if api == "opt_daily":
        values.append(
            frame.value(
                "option_open_interest_volume_ratio", ratio(row, "oi", "vol"), "ratio", "oi", "vol"
            )
        )
    if api == "vix_index":
        values.append(
            frame.value(
                "us_implied_volatility_change",
                number(row, "pct_change", ("ratio",)),
                unit(row, "pct_change"),
                "pct_change",
            )
        )
        details["volatility_market"] = "US_equity_implied_volatility_only"
    return values, details


def option_member(
    inputs: ResearchInputs, spec: ResearchRow, frame: Frame,
) -> tuple[str, tuple[str, ...]] | None:
    """按原有明确关系和到期口径返回方向及完整 PCR 分组键。

    输入唯一确认标的、认购方向、有效到期与乘数时返回 ('call',分组键)；
    缺标的或到期早于行情日返回 None。紧凑日期先规范化，避免等价日期被错误分区。
    """
    relations = [
        relation
        for relation in inputs.config.relations_by_source.get(
            (frame.entity_id, "option_underlying"), ()
        )
        if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
    ]
    side = {"C": "call", "P": "put", "认购": "call", "认沽": "put"}.get(
        spec.text("call_put") or ""
    )
    keys = tuple(
        spec.text(name) or ""
        for name in ("maturity_date", "exercise_type", "opt_type", "opt_multiplier")
    )
    if len(relations) != 1 or side is None or not all(keys):
        return None
    try:
        maturity = date.fromisoformat(keys[0]).isoformat()
    except ValueError:
        return None
    if maturity < frame.latest_period:
        return None
    return side, (relations[0].to_entity_id, maturity, *keys[1:], frame.latest_period)


def _option_groups(inputs: ResearchInputs) -> list[ResearchResult]:
    """按已确认标的、到期、执行方式及合约乘数汇总 PCR；例：看跌 20/看涨 10 得 2。

    不凭 opt_code 或同名代码猜标的，成交与持仓分开，缺一个方向或零分母输出未知。
    """
    specs = {
        frame.entity_id: frame.rows[0]
        for frame in frames(inputs, "opt_basic", API_FIELDS["opt_basic"])
        if len(frame.rows) == 1 and not frame.reasons
    }
    groups: dict[tuple[str, ...], list[tuple[str, Frame, ResearchRow]]] = defaultdict(list)
    for frame in frames(inputs, "opt_daily", API_FIELDS["opt_daily"]):
        if frame.reasons or len(frame.rows) != 1 or frame.entity_id not in specs:
            continue
        spec = specs[frame.entity_id]
        membership = option_member(inputs, spec, frame)
        if membership is None:
            continue
        side, key = membership
        groups[key].append((side, frame, spec))
    results = []
    for key, members in groups.items():
        frame = Frame(
            "opt_daily",
            key[0],
            {
                "option_group": digest(key[1:])[:20],
                "maturity_date": key[1],
                "exercise_type": key[2],
            },
            API_FIELDS["opt_daily"],
        )
        for _, member, _ in members:
            frame.add(inputs, member.rows[0])
        for _, member, spec in members:
            frame.cite(spec.evidence(*API_FIELDS["opt_basic"]))
            frame.cite(
                ref
                for relation in inputs.config.relations_by_source[
                    (member.entity_id, "option_underlying")
                ]
                if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
                for ref in relation.evidence_refs
            )
        values = []
        for name in ("vol", "oi"):
            totals = {"put": 0.0, "call": 0.0}
            counts = {"put": 0, "call": 0}
            valid = True
            first = members[0][1].rows[0]
            for side, member, _ in members:
                row = member.rows[0]
                value = number(row, name, ("quantity",))
                if value is None or value < 0 or not comparable(first, name, row, name):
                    valid = False
                else:
                    totals[side] += value
                    counts[side] += 1
            pcr = (
                totals["put"] / totals["call"]
                if valid and counts["put"] and totals["call"] > 0
                else None
            )
            values.append(frame.value("put_call_" + name + "_ratio", pcr, "ratio", name))
        results.append(
            finish(
                inputs,
                PRODUCT_ID,
                frame,
                values,
                {
                    "contract_count": len(members),
                    "underlying_relation": "confirmed",
                    "interpretation": "tail_hedging_activity_not_directional_forecast",
                },
            )
        )
    return results


def build_financing(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """加工 P11 十四项来源的融资、流动性与尾部风险证据。

    例如回购收盘率 3%、加权率 2.8% 得利率差；债券明细与汇总从不双计。
    期权分到期和显式标的比较，不混入已隔离记录，不生成交易指令。
    """
    results = run_sources(inputs, PRODUCT_ID, API_FIELDS, DIMENSIONS, _process)
    results.extend(_option_groups(inputs))
    return tuple(results)
