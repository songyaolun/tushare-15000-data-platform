"""P09：合约曲线、仓单库存、会员持仓及有证据的产业传导。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..store import digest
from .market_support import (
    Frame,
    binary_flag,
    comparable,
    difference,
    finish,
    frames,
    number,
    ratio,
    run_sources,
    singleton,
    unit,
)

PRODUCT_ID = "commodity_transmission/v1"
API_FIELDS = {
    "fut_basic": (
        "ts_code",
        "fut_code",
        "exchange",
        "d_month",
        "quote_unit",
        "trade_unit",
        "per_unit",
    ),
    "fut_daily": ("ts_code", "trade_date", "settle", "close", "vol", "oi"),
    "fut_daily_adj": ("ts_code", "trade_date", "close", "settle", "vol"),
    "fut_holding": (
        "symbol",
        "exchange",
        "broker",
        "long_hld",
        "short_hld",
        "long_chg",
        "short_chg",
    ),
    "fut_index_daily": ("ts_code", "trade_date", "close", "pct_chg"),
    "fut_inv_weekly": (
        "exchange",
        "trade_date",
        "warehouse",
        "grade",
        "fut_name",
        "cur_total",
        "pre_total",
        "unit",
    ),
    "fut_mapping": ("ts_code", "trade_date", "mapping_ts_code"),
    "fut_rcpt_mat": ("trade_date", "exchange", "fut_name", "unit", "cur_month", "next_month"),
    "fut_settle": ("ts_code", "trade_date", "settle", "long_margin_rate", "short_margin_rate"),
    "fut_trade_cal": ("exchange", "cal_date", "is_open"),
    "fut_trade_param": (
        "trade_date",
        "fut_code",
        "exchange",
        "tick_size",
        "price_limit",
        "position_limit",
    ),
    "fut_weekly_detail": ("exchange", "prd", "week_date", "vol", "amount", "open_interest"),
    "fut_weekly_monthly": ("trade_date", "end_date", "freq", "close", "settle", "vol"),
    "fut_wsr": ("trade_date", "symbol", "warehouse", "grade", "is_ct", "unit", "vol", "vol_chg"),
    "ft_limit": ("trade_date", "ts_code", "up_limit", "down_limit", "m_ratio"),
    "sge_basic": ("ts_code", "trade_type", "t_unit", "p_unit", "trade_time"),
    "sge_daily": ("ts_code", "trade_date", "close", "price_avg", "vol", "oi"),
    "fina_mainbz": ("end_date", "bz_item", "bz_code", "bz_sales", "bz_cost", "curr_type"),
    "fina_mainbz_vip": ("end_date", "bz_item", "bz_code", "bz_sales", "bz_cost", "curr_type"),
}
API_PURPOSES = {
    "fut_basic": "认证交易所、品种、交割月和报价规格后构造合约曲线",
    "fut_daily": "结算与收盘偏离、持仓成交比和同品种期限结构",
    "fut_daily_adj": "连续序列结算偏离及成交活跃度；不拼接收益率",
    "fut_holding": "会员多空持仓集中度与申报样本不平衡度，不解释为净看多",
    "fut_index_daily": "商品指数的来源内冲击，保持指数和合约分离",
    "fut_inv_weekly": "仓库、品级分层库存相对变化，不混加合计与明细",
    "fut_mapping": "主力连续代码映射唯一性及实际合约核对入口",
    "fut_rcpt_mat": "同单位本月对次月到期仓单压力",
    "fut_settle": "多空保证金差及独立结算口径核对",
    "fut_trade_cal": "交易所最新可见开闭市状态",
    "fut_trade_param": "最小价位、限仓与涨跌幅约束的可用性",
    "fut_weekly_detail": "周口径持仓对成交比，不与日频重复计量",
    "fut_weekly_monthly": "分频率结算收盘偏离及重采样核对",
    "fut_wsr": "厂库、品级、仓库分层仓单变化压力",
    "ft_limit": "上下限价格带宽及保证金占比",
    "sge_basic": "贵金属现货计价和交易规格完整性",
    "sge_daily": "现货成交均价偏离及持仓成交压力",
    "fina_mainbz": "主营分部利润率，作为 VIP 同快照核对而非重复收入",
    "fina_mainbz_vip": "主营分部利润率及已确认商品关系的暴露证据",
}
DIMENSIONS = {
    "fut_holding": ("symbol", "exchange"),
    "fut_inv_weekly": ("exchange", "warehouse", "grade", "fut_name", "unit"),
    "fut_rcpt_mat": ("exchange", "fut_name", "unit"),
    "fut_trade_cal": ("exchange",),
    "fut_trade_param": ("exchange", "fut_code"),
    "fut_weekly_detail": ("exchange", "prd"),
    "fut_weekly_monthly": ("freq",),
    "fut_wsr": ("symbol", "warehouse", "grade", "is_ct", "unit"),
    "fina_mainbz": ("bz_item", "curr_type"),
    "fina_mainbz_vip": ("bz_item", "curr_type"),
}


def _holdings(frame: Frame) -> list[Metric]:
    """计算会员样本集中度；例如两个等额会员的多仓 HHI 为 0.5。

    会员缺失、合计混入、重复会员或量纲不一致使整组未知，卖空不等同方向预测。
    """
    brokers = [row.text("broker") for row in frame.rows]
    if (
        not brokers
        or None in brokers
        or len(set(brokers)) != len(brokers)
        or any(name in {"合计", "总计", "total"} for name in brokers)
    ):
        frame.reasons.add("member_scope_unverified")
    values = []
    for field_name in ("long_hld", "short_hld"):
        amounts = [number(row, field_name, ("quantity",)) for row in frame.rows]
        valid = bool(amounts) and all(value is not None and value >= 0 for value in amounts)
        compatible = bool(frame.rows) and all(
            comparable(frame.rows[0], field_name, row, field_name) for row in frame.rows
        )
        total = sum(value for value in amounts if value is not None)
        hhi = (
            sum((value / total) ** 2 for value in amounts if value is not None)
            if valid and compatible and total > 0
            else None
        )
        values.append(
            frame.value(field_name + "_sample_hhi", hhi, "fraction", field_name, "broker")
        )
    return values


def _process(frame: Frame) -> tuple[list[Metric], dict[str, Any]]:
    """把逐 API 快照转为明确业务指标；例如库存 120/100 输出变动 0.2。

    元数据生成规格/映射/交易状态，数值源必须通过认证，所有字段引用由 Frame 保留。
    """
    details: dict[str, Any] = {
        "purpose": API_PURPOSES[frame.api],
        "stock_attribution": "explicit_relation_required",
    }
    if frame.api == "fut_holding":
        return _holdings(frame), details
    row = singleton(frame)
    if row is None:
        return [frame.value("snapshot_usable", None)], details
    api = frame.api
    specifications = {
        "fut_basic": ("fut_code", "exchange", "d_month", "quote_unit", "trade_unit", "per_unit"),
        "sge_basic": ("trade_type", "t_unit", "p_unit", "trade_time"),
        "fut_mapping": ("mapping_ts_code",),
    }
    if api in specifications:
        names = specifications[api]
        complete = all(row.text(name) is not None for name in names)
        details["specification_state"] = "complete_observed" if complete else "incomplete"
        details["missing_specifications"] = [name for name in names if row.text(name) is None]
        return [
            frame.value(
                "specification_completeness",
                sum(row.text(name) is not None for name in names) / len(names),
                "fraction",
                *names,
            )
        ], details
    if api == "fut_trade_cal":
        status = binary_flag(row, "is_open")
        details["session_state"] = {1.0: "open", 0.0: "closed", None: "unknown"}[status]
        return [
            frame.value(
                "session_open",
                status,
                "fraction",
                "is_open",
            )
        ], details
    formulas: dict[str, tuple[tuple[str, str, str, bool], ...]] = {
        "fut_daily": (
            ("settlement_close_deviation", "settle", "close", True),
            ("open_interest_volume_ratio", "oi", "vol", False),
        ),
        "fut_daily_adj": (("continuous_settlement_deviation", "settle", "close", True),),
        "fut_inv_weekly": (("inventory_change_ratio", "cur_total", "pre_total", True),),
        "fut_rcpt_mat": (("current_next_maturity_ratio", "cur_month", "next_month", False),),
        "fut_weekly_detail": (
            ("weekly_open_interest_volume_ratio", "open_interest", "vol", False),
        ),
        "fut_weekly_monthly": (("resampled_settlement_deviation", "settle", "close", True),),
        "fut_wsr": (("receipt_change_current_ratio", "vol_chg", "vol", False),),
        "sge_daily": (
            ("spot_average_close_deviation", "price_avg", "close", True),
            ("spot_open_interest_volume_ratio", "oi", "vol", False),
        ),
        "fina_mainbz": (("segment_cost_ratio", "bz_cost", "bz_sales", False),),
        "fina_mainbz_vip": (("segment_cost_ratio", "bz_cost", "bz_sales", False),),
    }
    values = [
        frame.value(name, ratio(row, left, right, change=change), "ratio", left, right)
        for name, left, right, change in formulas.get(api, ())
    ]
    if api.startswith("fina_mainbz"):
        cost_ratio = ratio(row, "bz_cost", "bz_sales")
        values.append(
            frame.value(
                "segment_gross_margin",
                None if cost_ratio is None else 1 - cost_ratio,
                "ratio",
                "bz_sales",
                "bz_cost",
            )
        )
        details["revenue_aggregation"] = "separate_segments_no_cross_source_sum"
    if api == "fut_settle":
        values.append(
            frame.value(
                "long_short_margin_spread",
                difference(row, "long_margin_rate", "short_margin_rate", ("ratio", "rate")),
                unit(row, "long_margin_rate"),
                "long_margin_rate",
                "short_margin_rate",
            )
        )
    if api == "ft_limit":
        values.extend(
            (
                frame.value(
                    "limit_band_ratio",
                    ratio(row, "up_limit", "down_limit", change=True),
                    "ratio",
                    "up_limit",
                    "down_limit",
                ),
                frame.value(
                    "margin_requirement",
                    number(row, "m_ratio", ("ratio", "rate")),
                    unit(row, "m_ratio"),
                    "m_ratio",
                ),
            )
        )
    if api == "fut_trade_param":
        for name in ("tick_size", "position_limit", "price_limit"):
            values.append(
                frame.value(name + "_constraint", number(row, name), unit(row, name), name)
            )
    if api == "fut_index_daily":
        values.append(
            frame.value(
                "commodity_index_return",
                number(row, "pct_chg", ("ratio",)),
                unit(row, "pct_chg"),
                "pct_chg",
            )
        )
    return values, details


def curve_member(spec: ResearchRow, frame: Frame) -> tuple[str, tuple[str, ...]] | None:
    """给已选定的合约规格与唯一行情构造原有完整曲线分组键。

    输入交易所、品种、报价单位、交易单位、每手单位及 2026-12 交割月，
    返回 ('202612',规格与行情日期键)；缺规格或非法月份返回 None，不从代码猜月份。
    """
    names = ("exchange", "fut_code", "quote_unit", "trade_unit", "per_unit")
    key = tuple(spec.text(name) or "" for name in names)
    month = (spec.text("d_month") or "").replace("-", "")
    if not all(key) or len(month) != 6 or not month.isdigit() or not 1 <= int(month[4:]) <= 12:
        return None
    return month, (*key, frame.latest_period)


def _curves(inputs: ResearchInputs) -> list[ResearchResult]:
    """按显式品种、交易所和规格建立近远月曲线；例：远月 110/近月 100 得 0.1。

    不解析合约代码猜到期月；币种、单位、时点及品种规格必须一致，不年化未知交割日。
    """
    specifications = {
        frame.entity_id: frame.rows[0]
        for frame in frames(inputs, "fut_basic", API_FIELDS["fut_basic"])
        if len(frame.rows) == 1 and not frame.reasons
    }
    groups: dict[tuple[str, ...], list[tuple[str, Frame, ResearchRow]]] = defaultdict(list)
    for frame in frames(inputs, "fut_daily", API_FIELDS["fut_daily"]):
        if frame.reasons or len(frame.rows) != 1 or frame.entity_id not in specifications:
            continue
        spec = specifications[frame.entity_id]
        membership = curve_member(spec, frame)
        if membership is None:
            continue
        month, key = membership
        groups[key].append((month, frame, spec))
    results = []
    for key, items in groups.items():
        items.sort(key=lambda item: item[0])
        if len(items) < 2:
            continue
        frame = Frame(
            "fut_daily",
            "context:futures_product:" + digest(key[:-1])[:24] + ":observed",
            {"exchange": key[0], "fut_code": key[1], "curve_date": key[-1]},
            API_FIELDS["fut_daily"],
        )
        for _, point, _ in items:
            frame.add(inputs, point.rows[0])
        for _, _, specification in items:
            frame.cite(specification.evidence(*API_FIELDS["fut_basic"]))
        near, far = items[0][1].rows[0], items[-1][1].rows[0]
        near_value, far_value = number(near, "settle"), number(far, "settle")
        if len({month for month, _, _ in items}) != len(items):
            frame.reasons.add("delivery_month_conflict")
        value = (
            far_value / near_value - 1
            if comparable(near, "settle", far, "settle")
            and near_value is not None
            and near_value > 0
            and far_value is not None
            else None
        )
        if frame.reasons:
            value = None
        # Curve membership is backed by explicit contract specifications, not code parsing.
        frame.fields = (
            *frame.fields,
            "fut_code",
            "exchange",
            "d_month",
            "quote_unit",
            "trade_unit",
            "per_unit",
        )
        values = [frame.value("far_near_settlement_spread", value, "ratio", "settle", "trade_date")]
        results.append(
            finish(
                inputs,
                PRODUCT_ID,
                frame,
                values,
                {
                    "curve_state": "contango"
                    if value is not None and value > 0
                    else "backwardation"
                    if value is not None and value < 0
                    else "flat"
                    if value == 0
                    else "unknown",
                    "contract_count": len(items),
                    "near_month": items[0][0],
                    "far_month": items[-1][0],
                    "annualized": False,
                    "stock_attribution": "explicit_relation_required",
                },
            )
        )
    return results


def _cross_checks(inputs: ResearchInputs) -> list[ResearchResult]:
    """按显式实体、日期及细分范围核对不同来源，绝不把重复收入或成交相加。

    例如同企业、分部及报告期的普通/VIP收入100、100得到核对差零；错期不比较。
    连续期货仅通过 fut_mapping 的实际合约代码及唯一来源实体核对。
    """
    indexed = {
        api: {
            (frame.entity_id, frame.latest_period, tuple(frame.scope.items())): frame
            for frame in frames(inputs, api, API_FIELDS[api], DIMENSIONS.get(api, ()))
            if len(frame.rows) == 1 and not frame.reasons
        }
        for api in (
            "fut_daily",
            "fut_settle",
            "fina_mainbz",
            "fina_mainbz_vip",
            "sge_daily",
            "fut_daily_adj",
            "fut_mapping",
        )
    }
    results = []
    for left_api, right_api, fields_to_check in (
        ("fut_settle", "fut_daily", ("settle",)),
        ("fina_mainbz_vip", "fina_mainbz", ("bz_sales", "bz_cost")),
    ):
        for key, frame in indexed[left_api].items():
            other = indexed[right_api].get(key)
            if other is None:
                continue
            left, right = frame.rows[0], other.rows[0]
            frame.scope = {**frame.scope, "comparison_source": right_api}
            frame.add(inputs, right)
            values = []
            for name in fields_to_check:
                first, second = number(left, name), number(right, name)
                value = (
                    first - second
                    if comparable(left, name, right, name)
                    and first is not None
                    and second is not None
                    else None
                )
                values.append(
                    frame.value(name + "_source_difference", value, unit(left, name), name)
                )
            results.append(
                finish(
                    inputs,
                    PRODUCT_ID,
                    frame,
                    values,
                    {"aggregation_policy": "cross_check_not_addition"},
                )
            )
    # Within-source code index is used only after an explicit mapping declares the target code.
    contract_codes: dict[tuple[str, str], list[ResearchRow]] = defaultdict(list)
    for frame in indexed["fut_daily"].values():
        row = frame.rows[0]
        contract_codes[(row.text("ts_code") or "", frame.latest_period)].append(row)
    for key, frame in indexed["fut_daily_adj"].items():
        mapping = indexed["fut_mapping"].get(key)
        candidates = (
            contract_codes.get(
                (mapping.rows[0].text("mapping_ts_code") or "", frame.latest_period), []
            )
            if mapping
            else []
        )
        frame.scope = {"comparison_source": "mapped_actual_contract"}
        value = None
        if len(candidates) == 1 and mapping is not None:
            left, right = frame.rows[0], candidates[0]
            first, second = number(left, "close"), number(right, "close")
            if (
                comparable(left, "close", right, "close")
                and first is not None
                and second is not None
                and second > 0
            ):
                value = first / second - 1
            frame.cite(mapping.rows[0].evidence(*API_FIELDS["fut_mapping"]))
            frame.add(inputs, right)
        else:
            frame.reasons.add("continuous_mapping_unverified")
        results.append(
            finish(
                inputs,
                PRODUCT_ID,
                frame,
                [frame.value("continuous_actual_close_deviation", value, "ratio", "close")],
                {"return_stitching": "not_permitted"},
            )
        )
    futures_by_identity = {(key[0], key[1]): frame for key, frame in indexed["fut_daily"].items()}
    for frame in indexed["sge_daily"].values():
        relations = [
            relation
            for relation in inputs.config.relations_by_source.get(
                (frame.entity_id, "spot_futures_comparable"), ()
            )
            if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
        ]
        if len(relations) != 1:
            continue
        other = futures_by_identity.get((relations[0].to_entity_id, frame.latest_period))
        if other is None:
            continue
        left, right = frame.rows[0], other.rows[0]
        spot, future = number(left, "close"), number(right, "close")
        value = (
            future / spot - 1
            if comparable(left, "close", right, "close")
            and spot is not None
            and spot > 0
            and future is not None
            else None
        )
        frame.scope = {"comparison_source": "explicit_comparable_future"}
        frame.cite(relations[0].evidence_refs)
        frame.add(inputs, right)
        results.append(
            finish(
                inputs,
                PRODUCT_ID,
                frame,
                [frame.value("futures_spot_basis_ratio", value, "ratio", "close")],
                {"comparability": "confirmed_spot_futures_relation"},
            )
        )
    return results


def _industry_exposures(inputs: ResearchInputs) -> list[ResearchResult]:
    """把确认的企业商品关系附到分部利润率，禁止按文字推断受益股票。

    例如企业 A 已确认暴露铜品种时输出关系目标及分部利润率；无关系不输出归因。
    不把分部收入占比解释成价格敏感度，也不把多个分部重复合计。
    """
    results = []
    for frame in frames(
        inputs, "fina_mainbz_vip", API_FIELDS["fina_mainbz_vip"], DIMENSIONS["fina_mainbz_vip"]
    ):
        relations = [
            relation
            for relation in inputs.config.relations_by_source.get(
                (frame.entity_id, "commodity_exposure"), ()
            )
            if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
        ]
        if not relations:
            continue
        frame.scope = {**frame.scope, "attribution": "confirmed_commodity_relation"}
        frame.cite(ref for relation in relations for ref in relation.evidence_refs)
        values, details = _process(frame)
        details.update(
            stock_attribution="confirmed_relation_only",
            exposure_target_count=len(relations),
            exposure_targets=sorted({relation.to_entity_id for relation in relations})[:16],
            sensitivity="not_estimated",
        )
        results.append(finish(inputs, PRODUCT_ID, frame, values, details))
    return results


def build_commodities(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """加工 P09 的十九项来源，输出分层压力、曲线和明确产业关系。

    例如同日同规格两合约结算 100、110 生成远近月差 0.1；缺币种仅输出未知。
    主营业务文字不会自动建立股票商品关系；只使用已确认且截止前有效的断言。
    """
    results = run_sources(inputs, PRODUCT_ID, API_FIELDS, DIMENSIONS, _process)
    results.extend(_curves(inputs))
    results.extend(_cross_checks(inputs))
    results.extend(_industry_exposures(inputs))
    return tuple(results)
