"""P02 同花顺主题七因子加工、样本收缩和严格主榜排名。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

P02_FACTOR_WEIGHTS = {
    "member_up_fraction": 0.20,
    "equal_weight_return": 0.20,
    "flow_strength": 0.20,
    "heat_strength": 0.15,
    "limit_up_breadth": 0.10,
    "board_strength": 0.10,
    "reason_consistency": 0.05,
}


@dataclass(frozen=True)
class ThemeObservation:
    """一个主题在目标交易日的可复算来源观察。"""

    theme_code: str
    theme_name: str
    member_codes: tuple[str, ...]
    returns_by_member: dict[str, float]
    flow_strength: float | None
    heat_by_member: dict[str, float]
    limit_known_members: frozenset[str]
    limit_up_members: frozenset[str]
    board_counts: dict[str, int]
    reason_tokens_by_member: dict[str, tuple[str, ...]]
    evidence_ids: tuple[str, ...] = ()


def parse_board_count(status: str | None) -> int:
    """把开盘啦状态解析成板数，首板为 1，无法确认时返回 0。

    输入 ``3连板`` 返回 3，``5天3板`` 返回 3，``首板`` 返回 1；普通描述、负数和
    空值返回 0。板数只由确定性文本模式产生，不调用模型猜测。
    """
    if not status:
        return 0
    text = re.sub(r"\s+", "", status)
    if "首板" in text:
        return 1
    matches = [
        re.search(r"(\d+)\s*连板", text),
        re.search(r"\d+\s*天\s*(\d+)\s*板", text),
        re.search(r"(?:^|[^\d])(\d+)\s*板(?:$|[^\d])", text),
    ]
    for match in matches:
        if match is not None:
            return max(0, int(match.group(1)))
    return 0


def normalize_reason_tokens(theme: str | None, description: str | None) -> tuple[str, ...]:
    """确定性清洗涨停主题或原因，返回可比较的非空标签集合。

    优先使用结构化 ``theme``，按常见分隔符切分；缺少主题时保留清洗后的完整
    ``description``。输入 ``AI+传媒，国企`` 返回 ``('ai','传媒','国企')``。
    """
    source = theme.strip() if isinstance(theme, str) and theme.strip() else description or ""
    pieces = re.split(r"[+＋、,，;/；|｜]+", source)
    normalized = {re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", piece).casefold() for piece in pieces}
    return tuple(sorted(piece for piece in normalized if piece))


def average_tie_percentiles(values: dict[str, float]) -> dict[str, float]:
    """计算 0—1 平均秩分位，相同值共享名次，单个样本返回 0.5。

    输入 ``{'a': 1, 'b': 2, 'c': 2}`` 返回 a=0、b=c=0.75；非有限值会抛出
    ``non_finite_percentile_input``，避免 NaN 混入排名。
    """
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("non_finite_percentile_input")
    if len(values) == 1:
        return {next(iter(values)): 0.5}
    grouped: dict[float, list[str]] = {}
    for identity, value in values.items():
        grouped.setdefault(value, []).append(identity)
    ranked: dict[str, float] = {}
    before = 0
    denominator = len(values) - 1
    for value in sorted(grouped):
        identities = grouped[value]
        average_index = before + (len(identities) - 1) / 2
        for identity in identities:
            ranked[identity] = average_index / denominator
        before += len(identities)
    return ranked


def calculate_theme_features(observation: ThemeObservation) -> dict[str, Any]:
    """把成员行情、资金、热度和涨停事件加工成七项原始因子及硬门禁。

    行情覆盖率至少 90%、有效行情成员至少 5 才可进入价格扩散榜；主榜还要求资金
    和热度已知、涨跌停边界覆盖至少 90%。连板按二板起算、五板封顶，原因一致性
    至少两只有明确原因的涨停成员才计算，否则记 0。
    """
    members = tuple(dict.fromkeys(observation.member_codes))
    member_set = set(members)
    returns = {
        member: value
        for member, value in observation.returns_by_member.items()
        if member in member_set and math.isfinite(value)
    }
    member_count = len(members)
    effective_member_count = len(returns)
    price_coverage = effective_member_count / member_count if member_count else 0.0
    limit_known = member_set & set(observation.limit_known_members)
    limit_coverage = len(limit_known) / member_count if member_count else 0.0
    limit_up = set(observation.limit_up_members) & set(returns)
    priced_denominator = max(effective_member_count, 1)
    hot_scores = [
        score
        for member, score in observation.heat_by_member.items()
        if member in member_set and math.isfinite(score) and score >= 0
    ]
    board_scores = [
        min(max(observation.board_counts.get(member, 0) - 1, 0), 4) / 4 for member in limit_up
    ]
    reason_counts = Counter(
        token
        for member in limit_up
        for token in set(observation.reason_tokens_by_member.get(member, ()))
    )
    reason_members = sum(
        bool(observation.reason_tokens_by_member.get(member)) for member in limit_up
    )
    price_eligible = effective_member_count >= 5 and price_coverage >= 0.90
    factors = {
        "member_up_fraction": (
            sum(value > 0 for value in returns.values()) / effective_member_count
            if effective_member_count
            else None
        ),
        "equal_weight_return": (
            math.fsum(returns.values()) / effective_member_count if effective_member_count else None
        ),
        "flow_strength": observation.flow_strength,
        "heat_strength": math.fsum(hot_scores) / priced_denominator,
        "limit_up_breadth": len(limit_up) / priced_denominator,
        "board_strength": math.fsum(board_scores) / priced_denominator,
        "reason_consistency": (
            max(reason_counts.values()) / reason_members
            if reason_members >= 2 and reason_counts
            else 0.0
        ),
    }
    missing = [
        name
        for name, value in factors.items()
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value)
    ]
    reasons = []
    if effective_member_count < 5:
        reasons.append("effective_members_below_five")
    if price_coverage < 0.90:
        reasons.append("price_coverage_below_90_percent")
    if limit_coverage < 0.90:
        reasons.append("limit_coverage_below_90_percent")
    reasons.extend("factor_unknown:" + name for name in missing)
    return {
        "theme_code": observation.theme_code,
        "theme_name": observation.theme_name,
        "member_count": member_count,
        "effective_member_count": effective_member_count,
        "price_coverage": price_coverage,
        "limit_coverage": limit_coverage,
        "price_eligible": price_eligible,
        "main_eligible": price_eligible and limit_coverage >= 0.90 and not missing,
        "reason_codes": sorted(set(reasons)),
        "evidence_ids": list(dict.fromkeys(observation.evidence_ids)),
        **factors,
    }


def rank_theme_features(
    features: list[dict[str, Any]],
    *,
    release_ready: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """生成严格综合主榜和价格扩散榜，并执行 5—19 成员样本收缩。

    七因子都在本次同花顺主题集合内做平均秩分位。主榜要求全局 release_ready 和
    逐主题 main_eligible；价格榜只要求五名有效成员及 90% 行情覆盖。输入不会被修改，
    返回按分数降序、主题代码升序排列的两份独立字典。
    """
    calculated = [dict(item) for item in features]
    main_candidates = [item for item in calculated if item["main_eligible"]]
    percentiles: dict[str, dict[str, float]] = {}
    for factor in P02_FACTOR_WEIGHTS:
        percentiles[factor] = (
            average_tie_percentiles(
                {item["theme_code"]: float(item[factor]) for item in main_candidates}
            )
            if main_candidates
            else {}
        )
    main_rows = []
    if release_ready:
        for item in main_candidates:
            raw_score = 100 * math.fsum(
                P02_FACTOR_WEIGHTS[factor] * percentiles[factor][item["theme_code"]]
                for factor in P02_FACTOR_WEIGHTS
            )
            shrinkage = min(1.0, item["effective_member_count"] / 20)
            adjusted = 50 + (raw_score - 50) * shrinkage
            row = dict(item)
            row.update(
                raw_score=raw_score,
                score=adjusted,
                sample_shrinkage=shrinkage,
                factor_percentiles={
                    factor: percentiles[factor][item["theme_code"]] for factor in P02_FACTOR_WEIGHTS
                },
            )
            main_rows.append(row)
    main_rows.sort(key=lambda item: (-item["score"], item["theme_code"]))
    for rank, item in enumerate(main_rows, 1):
        item["rank"] = rank
    price_candidates = [item for item in calculated if item["price_eligible"]]
    breadth_percentiles = (
        average_tie_percentiles(
            {item["theme_code"]: float(item["member_up_fraction"]) for item in price_candidates}
        )
        if price_candidates
        else {}
    )
    return_percentiles = (
        average_tie_percentiles(
            {item["theme_code"]: float(item["equal_weight_return"]) for item in price_candidates}
        )
        if price_candidates
        else {}
    )
    price_rows = []
    for item in price_candidates:
        row = dict(item)
        row["price_diffusion_score"] = 50 * (
            breadth_percentiles[item["theme_code"]] + return_percentiles[item["theme_code"]]
        )
        price_rows.append(row)
    price_rows.sort(key=lambda item: (-item["price_diffusion_score"], item["theme_code"]))
    for rank, item in enumerate(price_rows, 1):
        item["price_diffusion_rank"] = rank
    return main_rows, price_rows
