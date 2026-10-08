"""显式持仓与基金披露组合诊断，不访问账户、不归一化缺失权重、不产生交易动作。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from itertools import islice
from math import fsum, isclose, isfinite
from typing import Any

from ..equity_research.common import gate, make_result, metric, unique_evidence
from ..equity_research.models import MARKET_ENTITY, Metric, ResearchInputs, ResearchResult
from ..pit import time_boundary
from ..product_foundation.contracts import EvidenceRef, Truth
from ..product_foundation.entities import resolve_alias
from ..product_foundation.temporal import availability_gate
from ..standard import normalize_timestamp
from ..store import digest
from .market_support import fact

PRODUCT_ID = "portfolio_diagnostics/v1"
API_FIELDS = {"fund_portfolio": ("ann_date", "end_date", "symbol", "mkv", "stk_mkv_ratio")}
INPUT_FIELDS = ("entity_type", "entity_id", "as_of", "weight")
MAXIMUM_HOLDINGS = 20_000
SAMPLE_LIMIT = 32
MAXIMUM_PARENT_METRICS = 128
EVIDENCE_POSITIONS = 8
METRIC_REFERENCE_LIMIT = 16


@dataclass(frozen=True)
class Position:
    """有明确身份、权重与来源的单项持仓，weight不经过自动归一化。"""

    entity_id: str
    entity_type: str
    weight: float
    evidence: tuple[EvidenceRef, ...]
    market_value: float | None = None
    currency: str | None = None


def holding_entity(inputs: ResearchInputs, symbol: str, report_date: str) -> str | None:
    """用显式别名与报告日生命周期解析持仓证券，跨市场同码有多个候选时拒绝。

    例如000001.SZ只有tushare_stock有效别名则返回该证券；同名港美代码不会合并。
    """
    candidates = set()
    for namespace, timezone in (
        ("tushare_stock", "Asia/Shanghai"),
        ("tushare_hk", "Asia/Hong_Kong"),
        ("tushare_us", "America/New_York"),
    ):
        found = resolve_alias(
            inputs.config.aliases_by_key.get((namespace, symbol), ()),
            namespace,
            symbol,
            inputs.analysis_as_of,
            effective_at=report_date,
            market_timezone=timezone,
        )
        if found is not None and found in inputs.entity_kinds:
            candidates.add(found)
    return next(iter(candidates)) if len(candidates) == 1 else None


def explicit_positions(
    inputs: ResearchInputs,
    holdings: tuple[dict[str, Any], ...],
) -> tuple[tuple[Position, ...], str, str]:
    """验证完整用户持仓，不猜实体或权重；返回持仓、组合名与统一声明时点。

    输入两个有效证券及0.6/0.4权重得到原权重；重复、未来或总和非1均抛固定原因。
    """
    if not holdings or len(holdings) > MAXIMUM_HOLDINGS:
        raise ValueError("portfolio_holdings_missing_or_capacity")
    kinds = inputs.entity_kinds
    positions = []
    identities: set[str] = set()
    names: set[str] = set()
    dates: set[str] = set()
    for item in holdings:
        if any(name not in item for name in INPUT_FIELDS):
            raise ValueError("portfolio_holding_fields_missing")
        entity, kind, weight = item["entity_id"], item["entity_type"], item["weight"]
        if type(entity) is not str or type(kind) is not str or kinds.get(entity) != kind:
            raise ValueError("portfolio_entity_unresolved")
        if entity in identities:
            raise ValueError("portfolio_duplicate_entity")
        if type(weight) not in {int, float} or not isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("portfolio_weight_invalid")
        when = normalize_timestamp(item["as_of"])
        if time_boundary(when) > time_boundary(inputs.analysis_as_of):
            raise ValueError("portfolio_future_holding")
        name = item.get("portfolio_id", "user_portfolio")
        if type(name) is not str or not name.strip() or len(name) > 128:
            raise ValueError("portfolio_id_invalid")
        identities.add(entity)
        names.add(name)
        dates.add(when)
        record_id = digest(item)
        reference = EvidenceRef(
            source_api="user_holdings",
            source_record_id=record_id,
            business_group_key=digest([name, entity]),
            business_variant_key=None,
            contract_hash=digest(["explicit-holdings/v1", INPUT_FIELDS]),
            fields=INPUT_FIELDS,
            observed_at=when,
            locator="input:holdings/" + record_id,
        )
        positions.append(Position(entity, kind, float(weight), (reference,)))
    if len(names) != 1 or len(dates) != 1:
        raise ValueError("portfolio_snapshot_ambiguous")
    total = fsum(item.weight for item in positions)
    if total > 1 or not isclose(total, 1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("portfolio_weight_total_invalid")
    return tuple(sorted(positions, key=lambda item: item.entity_id)), names.pop(), dates.pop()


def parent_metrics(
    inputs: ResearchInputs,
    positions: tuple[Position, ...],
    parents: tuple[ResearchResult, ...],
) -> tuple[list[Metric], tuple[str, ...], dict[str, Any]]:
    """按真实持仓权重汇总同名同单位上游比例特征，缺失或重复候选时不填零。

    例如两项持仓各50%、同名比例0.2/0.4得到0.3；一项缺特征则该汇总为unknown。
    最多展示128个不同特征，所有候选的指纹与数量仍保存，不截断一个特征的分母。
    """
    weights = {item.entity_id: item.weight for item in positions}
    indexed: dict[tuple[str, str, str], dict[str, list[Metric]]] = {}
    parent_ids: set[str] = set()
    covered: set[str] = set()
    parent_digest = 0
    for result in parents:
        envelope = result.envelope
        if envelope.entity_id not in weights:
            continue
        if envelope.analysis_as_of != inputs.analysis_as_of or envelope.run_id != inputs.run_id:
            raise ValueError("portfolio_parent_context_mismatch")
        references = envelope.evidence_refs + tuple(
            ref for value in result.metrics for ref in value.evidence_refs
        )
        if any(
            time_boundary(ref.observed_at) > time_boundary(inputs.analysis_as_of)
            or (
                ref.public_available_at is not None
                and time_boundary(ref.public_available_at) > time_boundary(inputs.analysis_as_of)
            )
            for ref in references
        ):
            raise ValueError("portfolio_parent_evidence_unavailable")
        parent_ids.add(envelope.result_id)
        parent_digest = (parent_digest + int(digest(result.to_dict()), 16)) % 2**256
        for value in result.metrics:
            if value.status == "available":
                covered.add(envelope.entity_id)
            if value.unit not in {"ratio", "fraction"}:
                continue
            key = (envelope.product_id, value.name, value.unit)
            indexed.setdefault(key, {}).setdefault(envelope.entity_id, []).append(value)
    refs = tuple(ref for item in positions[:EVIDENCE_POSITIONS] for ref in item.evidence)
    values = [
        metric(
            "upstream_research_covered_weight",
            fsum(weights[name] for name in covered),
            "fraction",
            refs,
        )
    ]
    for key in sorted(indexed)[:MAXIMUM_PARENT_METRICS]:
        available = indexed[key]
        complete = all(
            entity in available
            and len(available[entity]) == 1
            and available[entity][0].status == "available"
            for entity in weights
        )
        references = refs + tuple(
            islice(
                (
                    ref
                    for entity in sorted(available)
                    for value in available[entity]
                    for ref in value.evidence_refs
                ),
                METRIC_REFERENCE_LIMIT,
            )
        )
        total = (
            fsum(weights[entity] * float(available[entity][0].value or 0) for entity in weights)
            if complete
            else None
        )
        values.append(
            metric(
                "weighted/" + key[0] + "/" + key[1] + "/" + key[2],
                total,
                key[2],
                references,
                () if complete else ("portfolio_parent_metric_missing_or_ambiguous",),
            )
        )
    return (
        values,
        tuple(sorted(parent_ids)),
        {
            "parent_metric_count": len(indexed),
            "parent_metric_sample_count": min(len(indexed), MAXIMUM_PARENT_METRICS),
            "parent_metric_truncated": len(indexed) > MAXIMUM_PARENT_METRICS,
            "parent_multiset_sha256": f"{parent_digest:064x}",
        },
    )


def diagnose(
    inputs: ResearchInputs,
    positions: tuple[Position, ...],
    portfolio_id: str,
    as_of: str,
    parents: tuple[ResearchResult, ...],
    *,
    full_portfolio: bool,
    report_period: str | None = None,
) -> ResearchResult:
    """对已明确的权重计算集中度、资产类型暴露和假设情景，并汇总上游比例特征。

    输入60%/40%完整组合得到HHI=0.52、最大权重0.6；基金股票部分不会补齐或归一化。
    情景仅是假设所有证券和ETF价格下降20%，其他资产未知，不推导收益承诺或交易动作。
    """
    refs = unique_evidence(ref for item in positions[:EVIDENCE_POSITIONS] for ref in item.evidence)
    total = fsum(item.weight for item in positions)
    concentration = fsum(item.weight**2 for item in positions)
    exposure_members: dict[str, list[float]] = {}
    for item in positions:
        exposure_members.setdefault(item.entity_type, []).append(item.weight)
    exposures = {name: fsum(weights) for name, weights in exposure_members.items()}
    values = [
        metric("disclosed_weight", total, "fraction", refs),
        metric("holding_hhi", concentration, "fraction", refs),
        metric("largest_holding_weight", max(item.weight for item in positions), "fraction", refs),
        metric(
            "top_five_weight",
            fsum(sorted((item.weight for item in positions), reverse=True)[:5]),
            "fraction",
            refs,
        ),
        metric(
            "effective_holdings",
            1 / concentration if full_portfolio else None,
            "count",
            refs,
            () if full_portfolio else ("full_portfolio_weights_unavailable",),
        ),
        metric("unknown_stock_sleeve_weight", 1 - total, "fraction", refs),
    ]
    values.extend(
        metric("asset_type_weight/" + name, value, "fraction", refs)
        for name, value in sorted(exposures.items())
    )
    currencies = {item.currency for item in positions}
    value_known = (
        None not in currencies
        and len(currencies) == 1
        and all(item.market_value is not None for item in positions)
    )
    values.append(
        metric(
            "disclosed_market_value",
            fsum(item.market_value or 0 for item in positions) if value_known else None,
            str(positions[0].currency) + ":元" if value_known else "source_money",
            refs,
            () if value_known else ("holding_valuation_or_currency_unverified",),
        )
    )
    values.append(
        metric(
            "liquidation_days", None, "days", (), ("portfolio_notional_and_participation_required",)
        )
    )
    covered = all(item.entity_type in {"security", "etf_share"} for item in positions)
    values.append(
        metric(
            "scenario_equity_minus_20pct",
            -0.2 * total if covered else None,
            "ratio",
            refs,
            () if covered else ("scenario_asset_sensitivity_unknown",),
        )
    )
    upstream, parent_ids, parent_details = parent_metrics(inputs, positions, parents)
    values.extend(upstream)
    details: dict[str, Any] = {
        "status": "diagnosed",
        "portfolio_id": portfolio_id,
        "holdings_as_of": as_of,
        "report_period": report_period,
        "weight_basis": "explicit_portfolio" if full_portfolio else "disclosed_stock_sleeve",
        "full_fund_nonstock_weight": None if not full_portfolio else "not_applicable",
        "position_count": len(positions),
        "position_sample_count": min(len(positions), SAMPLE_LIMIT),
        "positions_truncated": len(positions) > SAMPLE_LIMIT,
        "evidence_position_count": min(len(positions), EVIDENCE_POSITIONS),
        "evidence_positions_truncated": len(positions) > EVIDENCE_POSITIONS,
        "positions_sha256": digest(
            [[p.entity_id, p.entity_type, p.weight, p.market_value, p.currency] for p in positions]
        ),
        "positions": [
            {
                "entity_id": p.entity_id,
                "entity_type": p.entity_type,
                "weight": p.weight,
                "market_value": p.market_value,
                "currency": p.currency,
            }
            for p in positions[:SAMPLE_LIMIT]
        ],
        "scenario_assumptions": {
            "security_price_change": -0.2,
            "etf_share_price_change": -0.2,
            "prediction": False,
        },
        **parent_details,
    }
    return make_result(
        inputs,
        PRODUCT_ID,
        "user:portfolio:" + digest(portfolio_id)[:16] + ":declared",
        metrics=values,
        parents=parent_ids,
        profile="portfolio/" + digest([as_of, report_period])[:16],
        gates=(
            gate("position_input", Truth.TRUE, "", references=refs),
            gate(
                "complete_portfolio",
                Truth.TRUE if full_portfolio else Truth.UNKNOWN,
                "fund_nonstock_exposure_unknown",
                references=refs,
            ),
        ),
        details=details,
    )


def input_required(
    inputs: ResearchInputs,
    reason: str,
    *,
    profile: str = "input_required",
    references: tuple[EvidenceRef, ...] = (),
) -> ResearchResult:
    """缺持仓或持仓不可信时输出结构化输入要求，不生成空排名和虚假权重。

    例如缺weight返回status=input_required与明确字段清单，metrics保持空。
    合法可见的基金披露若因权重未知降级，references保留其来源证据；
    无输入时仍为空，未知门禁不能据此认证observed或生成虚假权重。
    """
    return make_result(
        inputs,
        PRODUCT_ID,
        MARKET_ENTITY,
        profile=profile,
        gates=(gate("position_input", Truth.UNKNOWN, reason, references=references),),
        details={
            "status": "input_required",
            "reason_codes": [reason],
            "required_fields": list(INPUT_FIELDS),
        },
    )


def build_portfolio(
    inputs: ResearchInputs,
    parents: tuple[ResearchResult, ...] = (),
    holdings: tuple[dict[str, Any], ...] = (),
) -> tuple[ResearchResult, ...]:
    """优先使用显式用户持仓，否则逐基金披露版本诊断已披露股票部分。

    无持仓返回一条input_required；基金缺权重、实体或重复持仓时该版本整体降级。
    仅使用通过公告门禁的原持仓比例，绝不以市值比例猜补全基金净资产或未知剩余。
    降级版本保留按来源记录ID确定性抽样的最多8条披露证据，便于审计具体缺口。
    """
    if holdings:
        try:
            declared_positions, name, when = explicit_positions(inputs, holdings)
            return (diagnose(inputs, declared_positions, name, when, parents, full_portfolio=True),)
        except (ValueError, TypeError, KeyError):
            return (input_required(inputs, "portfolio_explicit_input_invalid"),)
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for row in inputs.rows("fund_portfolio"):
        if (
            row.entity_id is not None
            and availability_gate(row.availability, inputs.analysis_as_of).passed
            and (
                row.raw.get("quality_tier") in {"verified", "usable_with_annotation"}
                and row.raw.get("is_quarantined") is not True
            )
        ):
            key = (
                row.entity_id,
                row.availability.report_period or "",
                row.availability.public_available_at or "",
            )
            grouped[key].append(row)
    results = []
    for key, rows in sorted(grouped.items()):
        references = unique_evidence(
            reference
            for row in nsmallest(
                EVIDENCE_POSITIONS, rows, key=lambda item: str(item.raw["source_record_id"])
            )
            for reference in row.evidence(*API_FIELDS["fund_portfolio"])
        )
        positions = []
        seen: set[str] = set()
        reason = ""
        if not key[1] or not key[2] or len(rows) > MAXIMUM_HOLDINGS:
            reason = "fund_portfolio_disclosure_time_or_capacity_unknown"
        for row in rows:
            value = fact(row, "stk_mkv_ratio")
            symbol = row.text("symbol")
            entity = holding_entity(inputs, symbol, key[1]) if symbol and key[1] else None
            if (
                value is None
                or value.unit not in {"fraction", "ratio"}
                or value.value is None
                or not 0 <= value.value <= 1
                or entity is None
                or entity in seen
            ):
                reason = "fund_holding_weight_or_entity_unverified"
                continue
            seen.add(entity)
            valuation = fact(row, "mkv")
            value_known = (
                valuation is not None
                and valuation.unit == "元"
                and (
                    valuation.dimension == "money"
                    and valuation.currency is not None
                    and valuation.value is not None
                    and valuation.value >= 0
                )
            )
            positions.append(
                Position(
                    entity,
                    inputs.entity_kinds[entity],
                    value.value,
                    row.evidence(*API_FIELDS["fund_portfolio"]),
                    valuation.value if value_known and valuation is not None else None,
                    valuation.currency if value_known and valuation is not None else None,
                )
            )
        if fsum(item.weight for item in positions) > 1 or not positions:
            reason = reason or "fund_holding_weight_total_invalid"
        if reason:
            results.append(
                input_required(
                    inputs, reason, profile="fund/" + digest(key)[:16], references=references
                )
            )
        else:
            try:
                result = diagnose(
                    inputs,
                    tuple(sorted(positions, key=lambda p: p.entity_id)),
                    key[0],
                    key[2],
                    parents,
                    full_portfolio=False,
                    report_period=key[1],
                )
            except ValueError:
                result = input_required(
                    inputs,
                    "portfolio_parent_context_or_metric_invalid",
                    profile="fund/" + digest(key)[:16],
                    references=references,
                )
            results.append(result)
    return tuple(results) or (input_required(inputs, "portfolio_holdings_missing"),)
