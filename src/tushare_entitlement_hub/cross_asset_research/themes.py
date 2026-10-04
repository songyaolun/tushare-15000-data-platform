"""P02 行业与主题状态：来源内行情、成员广度及同日资金旁证。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ..equity_research.common import gate, latest, make_result, metric
from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.entities import resolve_alias
from .market_support import fact
from .thematic_common import (
    accepted_groups,
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
    "dc_daily": ("ts_code", "trade_date", "pct_change", "amount", "category"),
    "dc_hot": ("ts_code", "trade_date", "rank", "rank_time", "data_type"),
    "dc_index": ("ts_code", "trade_date", "name", "idx_type", "level"),
    "dc_member": ("ts_code", "con_code", "trade_date"),
    "kpl_concept_cons": ("ts_code", "con_code", "trade_date", "desc", "hot_num"),
    "kpl_list": ("ts_code", "trade_date", "tag", "theme", "lu_desc", "status"),
    "limit_cpt_list": ("ts_code", "trade_date", "cons_nums", "up_nums", "pct_chg", "rank"),
    "tdx_daily": (
        "ts_code",
        "trade_date",
        "pct_change",
        "up_num",
        "down_num",
        "limit_up_num",
        "limit_down_num",
    ),
    "tdx_index": ("ts_code", "trade_date", "name", "idx_type", "idx_count"),
    "tdx_member": ("ts_code", "con_code", "trade_date"),
    "ths_daily": ("ts_code", "trade_date", "pct_change", "vol", "turnover_rate"),
    "ths_hot": ("ts_code", "trade_date", "rank", "rank_time", "rank_reason", "hot"),
    "ths_index": ("ts_code", "name", "type", "exchange"),
    "ths_member": ("ts_code", "con_code", "weight", "in_date", "out_date"),
    "ci_daily": ("ts_code", "trade_date", "close", "pct_change", "amount"),
    "ci_index_member": ("l1_code", "l2_code", "l3_code", "ts_code", "in_date", "out_date"),
    "index_classify": ("index_code", "parent_code", "level", "industry_name", "src"),
    "index_member_all": ("l1_code", "l2_code", "l3_code", "ts_code", "in_date", "out_date"),
    "sw_daily": ("ts_code", "trade_date", "pct_change", "amount", "pe", "pb"),
    "moneyflow_cnt_ths": ("ts_code", "trade_date", "net_amount", "company_num"),
    "moneyflow_ind_dc": ("ts_code", "trade_date", "content_type", "net_amount", "net_amount_rate"),
    "moneyflow_ind_ths": ("ts_code", "trade_date", "industry", "net_amount", "company_num"),
}

CATALOG_FIELDS = {
    "dc_index": ("name", "idx_type", "level"),
    "tdx_index": ("name", "idx_type", "idx_count"),
    "ths_index": ("name", "type", "exchange"),
    "index_classify": ("index_code", "parent_code", "level", "industry_name", "src"),
}
MEMBER_FIELDS = {
    "dc_member": "con_code",
    "tdx_member": "con_code",
    "ths_member": "con_code",
    "kpl_concept_cons": "con_code",
    "ci_index_member": "ts_code",
    "index_member_all": "ts_code",
}
DAILY_FIELDS = {
    "dc_daily": {"pct_change": "fraction", "amount": "元"},
    "tdx_daily": {
        "pct_change": "fraction",
        "up_num": "count",
        "down_num": "count",
        "limit_up_num": "count",
        "limit_down_num": "count",
    },
    "ths_daily": {"pct_change": "fraction", "vol": "手", "turnover_rate": "fraction"},
    "ci_daily": {"close": "元", "pct_change": "fraction", "amount": "元"},
    "sw_daily": {"pct_change": "fraction", "amount": "元", "pe": "ratio", "pb": "ratio"},
}
FLOW_APIS = ("moneyflow_cnt_ths", "moneyflow_ind_dc", "moneyflow_ind_ths")


def _signed_return(row: ResearchRow | None, field: str) -> float | None:
    """读取完整认证的有符号收益比例，兼容ratio与fraction两种已规范化存储单位。

    输入pct_change=-0.02且量纲为ratio，返回-0.02；原始-2%、金额量纲或无证书返回None。
    不在此处猜百分数换算倍数，也不把负收益写成受0至1限制的Metric(fraction)。
    """
    value = None if row is None else fact(row, field)
    return (
        value.value
        if value is not None and value.dimension == "ratio" and value.unit in {"ratio", "fraction"}
        else None
    )


def _return_metric(name: str, row: ResearchRow | None, field: str) -> Metric:
    """生成带来源字段证据的有符号收益指标，缺少认证时保留原因。

    输入规范化收益-0.02输出unit='ratio'、value=-0.02；缺行输出unit_or_value_unverified。
    """
    value = _signed_return(row, field)
    return metric(
        name,
        value,
        "ratio",
        () if row is None else row.evidence(field),
        () if value is not None else ("unit_or_value_unverified",),
    )


def _breadth(name: str, row: ResearchRow | None, up: str, total: str) -> Metric:
    """以同条来源的认证整数计算板块广度，分母缺失或小于上涨数时关闭。

    输入up_nums=3、cons_nums=10输出0.3；输入3、2输出unknown，不裁剪成1。
    """
    numerator, denominator = number(row, up, "count"), number(row, total, "count")
    valid = (
        numerator is not None
        and denominator is not None
        and 0 <= numerator <= denominator
        and denominator > 0
    )
    return metric(
        name,
        cast(float, numerator) / cast(float, denominator) if valid else None,
        "fraction",
        () if row is None else row.evidence(up, total),
        () if valid else ("breadth_denominator_unverified",),
    )


def _member_quote(
    inputs: ResearchInputs,
    member: ResearchRow,
    field: str,
    prices: dict[str, tuple[ResearchRow, ...]],
) -> tuple[ResearchRow, tuple[EvidenceRef, ...]] | None:
    """用成员接口的明确股票代码与已认证股票别名解析同生命周期行情。

    输入con_code=S、tushare_stock/S→证券A有效别名及A日线，返回日线和别名证据；
    没有股票别名、业务日未知或同代码多个实体时返回None，禁止仅按字符串拼接。
    """
    code = member.text(field)
    candidates = inputs.config.aliases_by_key.get(("tushare_stock", code or ""), ())
    if code is None or member.availability.data_as_of is None:
        return None
    identity = resolve_alias(
        candidates,
        "tushare_stock",
        code,
        inputs.analysis_as_of,
        effective_at=member.availability.data_as_of,
        market_timezone=member.availability.market_timezone,
    )
    selected = latest(prices.get(identity or "", ()))
    if identity is None or selected is None:
        return None
    refs = tuple(
        ref for alias in candidates if alias.entity_id == identity for ref in alias.evidence_refs
    )
    return selected, refs


def _member_snapshot(
    rows: Sequence[ResearchRow],
) -> tuple[tuple[ResearchRow, ...], bool, str | None]:
    """区分有业务日的成员快照与只有来源观察时间的当前记录组。

    输入两条同observed_at且data_as_of=None的CI成员，返回两条记录、True及来源观察时间；
    有明确业务日时仍选最新业务日，第二项为False。空输入返回((),False,None)。
    观察时间只标记本次来源记录，不写入data_as_of，也不建立历史同日关系。
    """
    snapshot = latest_snapshot(rows)
    observation_only = bool(rows) and not snapshot
    if observation_only:
        latest_observation = max(
            (row.availability.observed_at for row in rows),
            key=lambda value: (time_boundary(value), value),
        )
        snapshot = tuple(
            row
            for row in rows
            if time_boundary(row.availability.observed_at) == time_boundary(latest_observation)
        )
    observed_at = max(
        (row.availability.observed_at for row in snapshot),
        key=lambda value: (time_boundary(value), value),
        default=None,
    )
    return snapshot, observation_only, observed_at


def _members(
    inputs: ResearchInputs,
    entity: str,
    api: str,
    rows: Sequence[ResearchRow],
    prices: dict[str, tuple[ResearchRow, ...]],
) -> tuple[list[Metric], dict[str, Any]]:
    """计算完整成员快照与经显式关系确认的同日成员收益贡献。

    输入两名成员且只有一名有价格，价格覆盖为0.5，整体广度和等权贡献保持unknown。
    原始in_date/out_date仅作关系线索；有效成员必须有适配层有效区间认证。
    没有业务日时可统计最新来源观察组的成员数，另标snapshot_observed_at；
    例如同次观察两条CI成员得到observed_members=2，同日价格覆盖仍为unknown。
    """
    field = MEMBER_FIELDS[api]
    snapshot, observation_only, snapshot_observed_at = _member_snapshot(rows)
    members, conflicts = unique_latest_by(snapshot, (field,))
    refs = sampled_refs(rows, API_FIELDS[api])
    count_known = bool(snapshot) and not conflicts
    metrics = [
        metric(
            api + "_observed_members",
            len(members) if count_known else None,
            "count",
            refs,
            () if count_known else ("membership_snapshot_ambiguous",),
        )
    ]
    certified = [row for row in members if row.availability.effective_from is not None]
    active = [
        row
        for row in certified
        if time_boundary(cast(str, row.availability.effective_from))
        <= time_boundary(inputs.analysis_as_of)
        and (
            row.availability.effective_to is None
            or time_boundary(inputs.analysis_as_of) < time_boundary(row.availability.effective_to)
        )
    ]
    interval_known = count_known and len(certified) == len(members)
    metrics.append(
        metric(
            api + "_effective_members",
            len(active) if interval_known else None,
            "count",
            refs,
            () if interval_known else ("membership_effective_interval_unverified",),
        )
    )
    relations = tuple(
        relation
        for relation in inputs.config.relations_by_source.get((entity, "theme_member"), ())
        if relation.confirmed and relation.usable_at(inputs.analysis_as_of)
    )
    by_code: dict[str, tuple[ResearchRow, tuple[EvidenceRef, ...]]] = {}
    ambiguous_codes: set[str] = set()
    for relation in relations:
        selected = latest(prices.get(relation.to_entity_id, ()))
        if selected is not None and (code := selected.text("ts_code")) is not None:
            if code in by_code and by_code[code][0].entity_id != selected.entity_id:
                ambiguous_codes.add(code)
            else:
                by_code[code] = (selected, relation.evidence_refs)
    for code in ambiguous_codes:
        by_code.pop(code, None)
    contributing_members = active if interval_known else members
    returns = []
    contribution_refs: list[EvidenceRef] = list(refs)
    for member in contributing_members:
        matched = by_code.get(member.text(field) or "")
        if matched is None and member.text(field) not in ambiguous_codes:
            matched = _member_quote(inputs, member, field, prices)
        if matched is None:
            continue
        price, relation_refs = matched
        value = _signed_return(price, "pct_chg")
        if (
            value is not None
            and member.availability.data_as_of is not None
            and price.availability.data_as_of == member.availability.data_as_of
        ):
            returns.append(value)
            if len(returns) <= 8:
                contribution_refs.extend(price.evidence("pct_chg", "trade_date"))
                contribution_refs.extend(relation_refs)
    coverage_known = count_known and bool(contributing_members) and not observation_only
    complete = coverage_known and len(returns) == len(contributing_members)
    contribution_reason = (
        "membership_business_date_unknown"
        if observation_only
        else "member_relation_or_aligned_prices_missing"
    )
    metrics.extend(
        [
            metric(
                api + "_member_price_coverage",
                len(returns) / len(contributing_members) if coverage_known else None,
                "fraction",
                refs,
                ()
                if coverage_known
                else (
                    "membership_business_date_unknown"
                    if observation_only
                    else "membership_denominator_unknown",
                ),
            ),
            metric(
                api + "_member_up_fraction",
                sum(value > 0 for value in returns) / len(contributing_members)
                if complete
                else None,
                "fraction",
                contribution_refs,
                () if complete else (contribution_reason,),
            ),
            metric(
                api + "_equal_weight_return",
                finite(sum(returns) / len(contributing_members)) if complete else None,
                "ratio",
                contribution_refs,
                () if complete else (contribution_reason,),
            ),
        ]
    )
    details = {
        "snapshot_date": snapshot[0].availability.data_as_of if snapshot else None,
        "snapshot_observed_at": snapshot_observed_at,
        "snapshot_scope": "latest_observed_at_records"
        if observation_only
        else "latest_business_date",
        "observed_snapshot_record_count": len(snapshot),
        "observed_member_key_coverage": (
            sum(row.text(field) is not None for row in snapshot) / len(snapshot)
            if snapshot
            else None
        ),
        "unresolved_business_keys": conflicts,
        "membership_basis": "certified_effective_interval"
        if interval_known and not observation_only
        else "observed_snapshot_only",
        "effective_interval_certified": interval_known,
        "contribution_basis": "equal_weight_observed_members",
        "contribution_member_count": len(contributing_members),
        "ambiguous_member_code_count": len(ambiguous_codes),
        "hierarchy": {
            name: text_values(rows, name)
            for name in ("l1_code", "l2_code", "l3_code")
            if name in API_FIELDS[api]
        },
    }
    if api == "kpl_concept_cons":
        details["member_explanations"] = text_values(rows, "desc")
        metrics.append(scalar(api + "_hot_count", latest(tuple(rows)), "hot_num", "count"))
    if api == "ths_member":
        weights = [number(row, "weight", "fraction") for row in members]
        known = (
            count_known
            and bool(members)
            and all(value is not None and 0 <= value <= 1 for value in weights)
        )
        metrics.append(
            metric(
                api + "_observed_weight_sum",
                finite(sum(value for value in weights if value is not None)) if known else None,
                "ratio",
                refs,
                () if known else ("member_weight_unit_unverified",),
            )
        )
    return metrics, details


def _daily(api: str, rows: Sequence[ResearchRow]) -> tuple[list[Metric], dict[str, Any]]:
    """输出来源内最新行情及同口径上涨广度，不合并不同供应商的行业价格序列。

    输入通达信上涨3家下跌1家，得到方向广度0.75；平盘数量未知，分母注明仅涨跌样本。
    """
    selected = latest(tuple(rows))
    metrics = [
        _return_metric(api + "_" + field, selected, field)
        if field == "pct_change"
        else scalar(api + "_" + field, selected, field, unit)
        for field, unit in DAILY_FIELDS[api].items()
    ]
    details: dict[str, Any] = {
        "quote_date": None if selected is None else selected.availability.data_as_of
    }
    if api == "tdx_daily":
        up, down = number(selected, "up_num", "count"), number(selected, "down_num", "count")
        valid = up is not None and down is not None and up >= 0 and down >= 0 and up + down > 0
        metrics.append(
            metric(
                api + "_directional_breadth",
                cast(float, up) / (cast(float, up) + cast(float, down)) if valid else None,
                "fraction",
                () if selected is None else selected.evidence("up_num", "down_num"),
                () if valid else ("breadth_denominator_unverified",),
            )
        )
        details["breadth_scope"] = "advancing_and_declining_only"
    return metrics, details


def _hot(api: str, rows: Sequence[ResearchRow]) -> tuple[list[Metric], dict[str, Any]]:
    """在来源及data_type内保留排名和热度，逆排名只描述关注拥挤程度。

    输入rank=2返回inverse_rank=.5；存在同时刻冲突或rank<=0时返回unknown。
    """
    by_type: dict[str, list[ResearchRow]] = {}
    for row in rows:
        by_type.setdefault(row.text("data_type") or "unspecified", []).append(row)
    metrics = []
    states = []
    for position, label in enumerate(sorted(by_type)):
        selected = latest(tuple(by_type[label]))
        rank = number(selected, "rank", "count")
        valid = rank is not None and rank > 0
        metrics.append(
            metric(
                api + f"_inverse_rank_{position}",
                1 / cast(float, rank) if valid else None,
                "ratio",
                () if selected is None else selected.evidence("rank", "rank_time", "data_type"),
                () if valid else ("source_rank_unverified",),
            )
        )
        states.append(
            {
                "group": label,
                "metric_suffix": str(position),
                "rank_time": None if selected is None else selected.text("rank_time"),
            }
        )
    details = {
        "rank_groups": states[:32],
        "rank_group_count": len(states),
        "crowding_basis": "source_rank_only",
    }
    if api == "ths_hot":
        details["rank_reasons"] = text_values(rows, "rank_reason")
        metrics.append(scalar(api + "_attention", latest(tuple(rows)), "hot", "count"))
    return metrics, details


def _flow(api: str, rows: Sequence[ResearchRow]) -> tuple[list[Metric], dict[str, Any]]:
    """保留来源内资金方向和公司分母，不相加不同接口的净流入。

    输入net_amount=-100元、company_num=10，输出每家公司-10元及净流出方向。
    东方财富content_type不同保留为独立口径，不能选任意类型掩盖冲突。
    """
    selected = latest(tuple(rows))
    amount = number(selected, "net_amount", "元")
    metrics = [scalar(api + "_net_amount", selected, "net_amount", "元")]
    refs = () if selected is None else selected.evidence("net_amount", "company_num")
    company_count = number(selected, "company_num", "count")
    if "company_num" in API_FIELDS[api]:
        valid = amount is not None and company_count is not None and company_count > 0
        metrics.append(
            metric(
                api + "_per_company_flow",
                finite(cast(float, amount) / cast(float, company_count)) if valid else None,
                "元",
                refs,
                () if valid else ("flow_company_denominator_unverified",),
            )
        )
    else:
        metrics.append(scalar(api + "_net_amount_rate", selected, "net_amount_rate", "fraction"))
    details: dict[str, Any] = {
        "flow_date": None if selected is None else selected.availability.data_as_of,
        "direction": "unknown"
        if amount is None
        else "inflow"
        if amount > 0
        else "outflow"
        if amount < 0
        else "flat",
    }
    for field in ("content_type", "industry"):
        if field in API_FIELDS[api]:
            details[field] = text_values(rows, field)
    return metrics, details


def _theme_state(sources: dict[str, tuple[ResearchRow, ...]]) -> tuple[Metric, str]:
    """只用同一显式主题实体、同日行情和资金判断确认状态。

    输入最新收益为正且同日净流入为正返回1及price_and_flow_positive；错日返回unknown。
    不把热搜、涨停理由或跨源同名字符串当作资金确认。
    """
    prices = [
        selected for api in DAILY_FIELDS if (selected := latest(sources.get(api, ()))) is not None
    ]
    flows = [
        selected for api in FLOW_APIS if (selected := latest(sources.get(api, ()))) is not None
    ]
    day = max((row.availability.data_as_of or "" for row in prices), default="")
    aligned_prices = [row for row in prices if row.availability.data_as_of == day]
    aligned_flows = [row for row in flows if row.availability.data_as_of == day]
    returns = [_signed_return(row, "pct_change") for row in aligned_prices]
    amounts = [number(row, "net_amount", "元") for row in aligned_flows]
    valid = bool(day and returns and amounts) and all(
        value is not None for value in returns + amounts
    )
    positive = valid and all(value > 0 for value in returns + amounts if value is not None)
    state = (
        "price_and_flow_positive" if positive else "not_jointly_positive" if valid else "unknown"
    )
    refs = sampled_refs(aligned_prices, ("pct_change", "trade_date")) + sampled_refs(
        aligned_flows, ("net_amount", "trade_date")
    )
    return metric(
        "price_flow_confirmation",
        float(positive) if valid else None,
        "fraction",
        refs,
        () if valid else ("aligned_theme_price_flow_unavailable",),
    ), state


def build_theme_opportunities(inputs: ResearchInputs) -> tuple[ResearchResult, ...]:
    """将P02的22个接口加工成时点一致的来源内主题观察产品。

    输入已绑定主题的成员、行情和资金证据，输出广度、动量、拥挤旁证及成员贡献；
    没有输入时返回()。未知成员有效期输出observed_snapshot_only，始终不认证历史PIT。
    指标计算用完整已适配组；只有证据和展示样本有上限，绝不截断业务分母。
    """
    groups = accepted_groups(inputs, API_FIELDS)
    price_groups = accepted_groups(inputs, ("daily",))
    prices = {entity: values.get("daily", ()) for entity, values in price_groups.items()}
    results = []
    for entity, sources in sorted(groups.items()):
        metrics: list[Metric] = []
        references: list[EvidenceRef] = []
        details: dict[str, Any] = {
            "sources": {},
            "namespace_policy": "explicit_entity_only",
            "historical_pit_certified": False,
        }
        for api, rows in sorted(sources.items()):
            fields = API_FIELDS[api]
            references.extend(sampled_refs(rows, fields))
            source_details = observation_summary(rows, fields)
            source_details["reason_codes"] = list(inputs.reasons(api, entity))
            if api in MEMBER_FIELDS:
                values, domain_details = _members(inputs, entity, api, rows, prices)
            elif api in DAILY_FIELDS:
                values, domain_details = _daily(api, rows)
            elif api in FLOW_APIS:
                values, domain_details = _flow(api, rows)
            elif api in {"dc_hot", "ths_hot"}:
                values, domain_details = _hot(api, rows)
            elif api in CATALOG_FIELDS:
                selected = latest(rows)
                values = [
                    metric(
                        api + "_catalog_consistent",
                        1.0 if selected is not None else None,
                        "fraction",
                        sampled_refs(rows, fields),
                        () if selected is not None else ("theme_catalog_ambiguous",),
                    )
                ]
                domain_details = {field: text_values(rows, field) for field in CATALOG_FIELDS[api]}
            elif api == "limit_cpt_list":
                selected = latest(rows)
                values = [
                    _breadth(api + "_limit_up_breadth", selected, "up_nums", "cons_nums"),
                    _return_metric(api + "_pct_chg", selected, "pct_chg"),
                    scalar(api + "_rank", selected, "rank", "count"),
                ]
                domain_details = {"breadth_scope": "provider_reported_constituents"}
            else:
                values = [
                    metric(
                        "kpl_list_observed_events",
                        float(len(rows)) if rows else None,
                        "count",
                        sampled_refs(rows, fields),
                        () if rows else ("theme_event_unavailable",),
                    )
                ]
                domain_details = {
                    field: text_values(rows, field)
                    for field in ("tag", "theme", "lu_desc", "status")
                }
                domain_details["interpretation"] = "provider_limit_reason_not_disclosure_fact"
            metrics.extend(values)
            source_details.update(domain_details)
            details["sources"][api] = source_details
        confirmation, state = _theme_state(sources)
        metrics.append(confirmation)
        details["theme_state"] = state
        results.append(
            make_result(
                inputs,
                "theme_opportunities/v1",
                entity,
                metrics=metrics,
                references=references,
                gates=(
                    gate(
                        "theme_source_identity",
                        Truth.TRUE if references else Truth.UNKNOWN,
                        "theme_evidence_unavailable",
                    ),
                ),
                details=details,
                profile="theme_observed_source_scope/v1",
            )
        )
    return tuple(results)
