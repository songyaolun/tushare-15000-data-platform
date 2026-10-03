"""同口径分位、加权缺失规则、分档排名及权重敏感性，不包含业务因子。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import fsum, isfinite
from typing import cast

from ..pit import time_boundary
from ..store import digest
from .config import FoundationPolicy
from .contracts import EvidenceRef, FeatureContribution
from .model import (
    Count,
    EntityId,
    Fraction,
    Identifier,
    Model,
    Positive,
    ReasonCode,
    Score,
    Timestamp,
)
from .numeric import TypedFact
from .temporal import availability_gate


@dataclass(frozen=True, kw_only=True)
class FeatureValue(Model):
    """特征携带窗口、分母、口径、版本及证据；评分输入需已归一至 0—1。"""

    entity_id: EntityId
    feature_id: Identifier
    analysis_as_of: Timestamp
    scope_id: Identifier
    window: Identifier
    feature_profile: Identifier
    version: Identifier
    value: float | None
    unit: str
    input_count: Count
    expected_count: Count
    evidence_refs: tuple[EvidenceRef, ...]
    reason_codes: tuple[ReasonCode, ...] = ()
    raw_value: float | None = None
    transform: str = "identity"
    excluded_evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        """验证分母、缺失解释与证据时点；例如 value=None 且无原因 → 拒绝。

        输入有来源的合格特征，输出能按口径和版本隔离的记录。
        """
        super().__post_init__()
        if self.input_count > self.expected_count:
            raise ValueError("feature_count_mismatch")
        if self.value is None and not self.reason_codes:
            raise ValueError("feature_missing_reason_required")
        if self.value is not None and (not self.input_count or not self.evidence_refs):
            raise ValueError("feature_evidence_required")
        cutoff = time_boundary(self.analysis_as_of)
        if any(
            time_boundary(item.observed_at) > cutoff
            or (
                item.public_available_at is not None
                and time_boundary(item.public_available_at) > cutoff
            )
            for item in self.evidence_refs
        ):
            raise ValueError("feature_before_evidence")


def feature_from_fact(
    fact: TypedFact,
    *,
    feature_id: str,
    analysis_as_of: str,
    scope_id: str,
    window: str,
    feature_profile: str,
    version: str,
) -> FeatureValue:
    """从已认证且可见的事实生成默认特征，拒绝隔离、未知、空值及未来数据。

    输入可用的 2000 元事实，输出 value=2000 的原值特征；不可用时 value=None，
    原证据进入 excluded_evidence，禁止将缺失补零或未来证据写成已使用来源。
    """
    gate = availability_gate(fact.availability, analysis_as_of)
    reasons = set(fact.reason_codes)
    if not gate.passed:
        reasons.update(gate.reason_codes)
    if fact.certification is None or fact.quality_tier != "verified":
        reasons.add("fact_not_feature_eligible")
    usable = fact.value is not None and not reasons
    return FeatureValue(
        entity_id=fact.entity_id,
        feature_id=feature_id,
        analysis_as_of=analysis_as_of,
        scope_id=scope_id,
        window=window,
        feature_profile=feature_profile,
        version=version,
        value=fact.value if usable else None,
        unit=fact.unit or "unknown",
        input_count=int(usable),
        expected_count=1,
        evidence_refs=fact.evidence_refs if usable else (),
        excluded_evidence=() if usable else fact.evidence_refs,
        reason_codes=tuple(sorted(reasons)),
        raw_value=fact.value,
    )


@dataclass(frozen=True, kw_only=True)
class PeerObservation(Model):
    """单一可比特征的同行样本；peer_path 按市场→上级行业→下级行业排列。"""

    entity_id: EntityId
    value: float | None
    analysis_as_of: Timestamp
    feature_profile: Identifier
    scope_id: Identifier
    peer_path: tuple[Identifier, ...]
    eligible: bool = True

    def __post_init__(self) -> None:
        """要求显式分组路径；例如 peer_path=() → 拒绝，不隐式混入全市场。"""
        super().__post_init__()
        if not self.peer_path:
            raise ValueError("peer_path_required")


@dataclass(frozen=True, kw_only=True)
class PercentileResult(Model):
    """平均秩百分位或明确拒绝结果。"""

    percentile: Fraction | None
    peer_count: Count
    peer_level: str | None
    reason_codes: tuple[ReasonCode, ...] = ()


def _average_ranks(values: dict[str, float], *, descending: bool = False) -> dict[str, float]:
    """对相同值分配平均秩，输出从 1 开始的确定性排名。

    例如 {'a':10,'b':10,'c':20} → {'a':1.5,'b':1.5,'c':3}。
    """
    groups: dict[float, list[str]] = defaultdict(list)
    for identity, value in values.items():
        groups[value].append(identity)
    result, start = {}, 1
    for value in sorted(groups, reverse=descending):
        members = groups[value]
        rank = start + (len(members) - 1) / 2
        result.update({identity: rank for identity in sorted(members)})
        start += len(members)
    return result


def percentile_ranks(
    rows: tuple[PeerObservation, ...],
    policy: FoundationPolicy,
    *,
    lower_is_better: bool = False,
) -> dict[str, PercentileResult]:
    """同截止、同范围、同档位逐级回退并计算 (平均秩-1)/(n-1)。

    例如三条 [10,10,20] 且最小样本为 3，输出 [0.25,0.25,1]；全同值为 0.5。
    空值和不合格行不计分母，最上层仍不足则给空分位及原因。
    每个实际选中的同行池只计算一次平均秩；例如5911行共享同池时仅排序一次。
    排名缓存只属于本次调用，键包含完整截止、范围、档位和路径，不跨输入复用。
    """
    if len({row.entity_id for row in rows}) != len(rows):
        raise ValueError("duplicate_peer_entity")
    pools: dict[tuple[str, str, str, tuple[str, ...]], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.eligible and row.value is not None:
            for length in range(1, len(row.peer_path) + 1):
                pools[
                    (row.analysis_as_of, row.scope_id, row.feature_profile, row.peer_path[:length])
                ][row.entity_id] = row.value
    ranks: dict[tuple[str, str, str, tuple[str, ...]], dict[str, float]] = {}
    result = {}
    for row in rows:
        choice = PercentileResult(
            percentile=None,
            peer_count=0,
            peer_level=None,
            reason_codes=("insufficient_peer_sample",),
        )
        if row.eligible and row.value is not None:
            for length in range(len(row.peer_path), 0, -1):
                key = (
                    row.analysis_as_of,
                    row.scope_id,
                    row.feature_profile,
                    row.peer_path[:length],
                )
                pool = pools[key]
                if len(pool) >= policy.minimum_peer_count:
                    if key not in ranks:
                        ranks[key] = _average_ranks(pool)
                    percentile = (ranks[key][row.entity_id] - 1) / (len(pool) - 1)
                    choice = PercentileResult(
                        percentile=1 - percentile if lower_is_better else percentile,
                        peer_count=len(pool),
                        peer_level=row.peer_path[length - 1],
                        reason_codes=("peer_level_fallback",)
                        if length < len(row.peer_path)
                        else (),
                    )
                    break
        else:
            choice = replace(choice, reason_codes=("peer_value_unavailable",))
        result[row.entity_id] = choice
    return result


@dataclass(frozen=True, kw_only=True)
class FeatureWeight(Model):
    """相对权重及核心依赖标记；实际贡献权重按可用分量归一。"""

    feature_id: Identifier
    weight: Positive
    required: bool = False


@dataclass(frozen=True, kw_only=True)
class ScorePolicy(Model):
    """具体评分调用方声明特征权重和公共阈值，基础包不内置 P01—P13 公式。"""

    version: Identifier
    weights: tuple[FeatureWeight, ...]
    thresholds: FoundationPolicy

    def __post_init__(self) -> None:
        """拒绝空、重复或溢出的权重配置；例如同 feature_id 两次 → 错误。"""
        super().__post_init__()
        if (
            not self.weights
            or len({item.feature_id for item in self.weights}) != len(self.weights)
            or not isfinite(sum(item.weight for item in self.weights))
        ):
            raise ValueError("invalid_score_weights")


@dataclass(frozen=True, kw_only=True)
class ScoreResult(Model):
    """评分、缺失组合和贡献整体保存，排名资格仍单独检查。"""

    entity_id: EntityId
    analysis_as_of: Timestamp
    scope_id: Identifier
    model_version: Identifier
    feature_profile: Identifier
    score: Score | None
    weight_coverage: Fraction
    components: tuple[FeatureContribution, ...]
    missing_features: tuple[Identifier, ...]
    reason_codes: tuple[ReasonCode, ...]

    @property
    def result_id(self) -> str:
        """生成稳定评分身份；同实体同口径重复计算 → 相同摘要。"""
        return digest(
            [
                self.entity_id,
                self.analysis_as_of,
                self.scope_id,
                self.model_version,
                self.feature_profile,
            ]
        )


def score_features(features: tuple[FeatureValue, ...], policy: ScorePolicy) -> ScoreResult:
    """核心特征齐全且可用权重达标才评分，每个非空分数须能分解为贡献和。

    例如权重 0.8/0.2、特征 0.5/1，输出 60 分；只有首项时输出 50 分及独立档位。
    缺失不是零，未知特征或混合实体、时点、范围、档位会直接拒绝。
    各特征自己的版本和窗口写入档位摘要，不要求不同特征使用相同窗口。
    不可用特征保留原值、单位、变换和全部原因，其来源只登记为排除证据；
    例如第二项因 unit_unverified 不可用时，原值仍可审计且贡献为空。
    """
    if not features:
        raise ValueError("score_context_required")
    first = features[0]
    contexts = {
        (item.entity_id, item.analysis_as_of, item.scope_id, item.feature_profile)
        for item in features
    }
    indexed = {item.feature_id: item for item in features}
    declared = {item.feature_id for item in policy.weights}
    if len(contexts) != 1 or len(indexed) != len(features) or not set(indexed) <= declared:
        raise ValueError("score_feature_context_mismatch")
    available = {
        name: item
        for name, item in indexed.items()
        if item.value is not None and not item.reason_codes
    }
    if any(not 0 <= cast(float, item.value) <= 1 for item in available.values()):
        raise ValueError("score_feature_not_normalized")
    total = fsum(item.weight for item in policy.weights)
    mass = fsum(item.weight for item in policy.weights if item.feature_id in available)
    missing = tuple(sorted(declared - set(available)))
    coverage = mass / total
    reasons = []
    if any(item.required and item.feature_id in missing for item in policy.weights):
        reasons.append("required_feature_missing")
    if coverage < policy.thresholds.minimum_weight_coverage or not mass:
        reasons.append("insufficient_weight_coverage")
    components = []
    for weight in sorted(policy.weights, key=lambda item: item.feature_id):
        item = indexed.get(weight.feature_id)
        available_item = available.get(weight.feature_id)
        effective_weight = weight.weight / mass if available_item is not None else 0.0
        component_reasons = item.reason_codes if item else ("feature_unavailable",)
        components.append(
            FeatureContribution(
                feature_id=weight.feature_id,
                raw_value=item.raw_value
                if item and item.raw_value is not None
                else item.value
                if item
                else None,
                unit=item.unit if item else "unknown",
                transformed_value=available_item.value if available_item else None,
                weight=effective_weight,
                contribution=100 * effective_weight * cast(float, available_item.value)
                if available_item
                else None,
                transform=item.transform if item else "unavailable",
                missing_reason=None if available_item else component_reasons[0],
                evidence_refs=available_item.evidence_refs if available_item else (),
                reason_codes=component_reasons,
                excluded_evidence=item.excluded_evidence
                + (item.evidence_refs if available_item is None else ())
                if item
                else (),
            )
        )
    score = (
        None
        if reasons
        else 100
        * (
            fsum(
                weight.weight * cast(float, available[weight.feature_id].value)
                for weight in policy.weights
                if weight.feature_id in available
            )
            / mass
        )
    )
    if (
        score is not None
        and abs(score - fsum(item.contribution or 0 for item in components))
        > policy.thresholds.contribution_tolerance
    ):
        raise ValueError("contribution_sum_mismatch")
    profile = digest(
        [
            first.feature_profile,
            {
                **policy.to_dict(),
                "weights": sorted(policy.to_dict()["weights"], key=lambda item: item["feature_id"]),
            },
            sorted(
                (name, item.version, item.window, item.unit, item.transform)
                for name, item in available.items()
            ),
        ]
    )
    return ScoreResult(
        entity_id=first.entity_id,
        analysis_as_of=first.analysis_as_of,
        scope_id=first.scope_id,
        model_version=policy.version,
        feature_profile=profile,
        score=score,
        weight_coverage=coverage,
        components=tuple(components),
        missing_features=missing,
        reason_codes=tuple(reasons),
    )


@dataclass(frozen=True, kw_only=True)
class RankedScore(Model):
    """同口径评分的平均排名，不足样本保持空名次。"""

    rank: float | None
    peer_count: Count
    reason_codes: tuple[ReasonCode, ...]


def rank_scores(
    results: tuple[ScoreResult, ...], policy: FoundationPolicy
) -> dict[str, RankedScore]:
    """按时点、范围、模型版本及真实缺失档位隔离排名，不混排完整与降级档。

    输入两个同档合法评分和一个缺失档评分，最小样本 2 时后者名次为空。
    重复result_id先拒绝，再按完整组键为达到样本门槛的组各计算一次降序平均秩。
    例：同组5911个评分只排序一次；缺分数或有原因的记录仍不排名，但保留同组有效分母。
    组索引仅属于当前调用，空组和不足样本组不排序，后续每个结果按键取名次。
    """
    if len({item.result_id for item in results}) != len(results):
        raise ValueError("duplicate_score_result")
    groups: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for item in results:
        if item.score is not None and not item.reason_codes:
            groups[(item.analysis_as_of, item.scope_id, item.model_version, item.feature_profile)][
                item.result_id
            ] = item.score
    group_ranks = {
        key: _average_ranks(group, descending=True)
        for key, group in groups.items()
        if len(group) >= policy.minimum_peer_count
    }
    ranked = {}
    for item in results:
        key = (item.analysis_as_of, item.scope_id, item.model_version, item.feature_profile)
        group = groups[key]
        eligible = item.result_id in group and len(group) >= policy.minimum_peer_count
        ranked[item.result_id] = RankedScore(
            rank=group_ranks[key][item.result_id] if eligible else None,
            peer_count=len(group),
            reason_codes=() if eligible else ("insufficient_rank_sample",),
        )
    return ranked


@dataclass(frozen=True, kw_only=True)
class StabilityScenario(Model):
    """单个权重单方向扰动下的集合相似度和可排名范围。"""

    feature_id: Identifier
    multiplier: Positive
    top_k_jaccard: Fraction | None
    ranks: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class StabilityResult(Model):
    """逐权重正负扰动结果；区间覆盖基线及每个有效扰动排名。"""

    scenarios: tuple[StabilityScenario, ...]
    rank_intervals: dict[str, tuple[float, ...]]
    sensitive: bool
    reason_codes: tuple[ReasonCode, ...]


def _scenario_ranks(
    populations: tuple[tuple[FeatureValue, ...], ...],
    policy: ScorePolicy,
) -> dict[str, float]:
    """计算同一口径集合的合格实体排名；输入两条合法样本 → 实体到名次映射。"""
    scores = tuple(score_features(items, policy) for items in populations)
    contexts = {
        (item.analysis_as_of, item.scope_id, item.model_version, item.feature_profile)
        for item in scores
    }
    if len(contexts) > 1:
        raise ValueError("stability_profile_mismatch")
    ranked = rank_scores(scores, policy.thresholds)
    return {
        item.entity_id: cast(float, ranked[item.result_id].rank)
        for item in scores
        if ranked[item.result_id].rank is not None
    }


def weight_stability(
    populations: tuple[tuple[FeatureValue, ...], ...],
    policy: ScorePolicy,
) -> StabilityResult:
    """逐权重正负扰动后重新归一、评分和排名，报告 Top-K Jaccard 与排名区间。

    例如输入 perturbation=0.1 及两个权重，输出四个场景；缺样本不报告虚假稳定性。
    边界同值一起进入 Top-K，集合按平均秩判断，输出定义完全可复算。
    """
    baseline = _scenario_ranks(populations, policy)
    selected = {key for key, rank in baseline.items() if rank <= policy.thresholds.top_k}
    intervals = {key: [rank] for key, rank in baseline.items()}
    scenarios = []
    for target in policy.weights:
        for direction in (-1, 1):
            multiplier = 1 + direction * policy.thresholds.weight_perturbation
            varied = replace(
                policy,
                weights=tuple(
                    replace(weight, weight=weight.weight * multiplier)
                    if weight.feature_id == target.feature_id
                    else weight
                    for weight in policy.weights
                ),
            )
            ranks = _scenario_ranks(populations, varied)
            top = {key for key, rank in ranks.items() if rank <= policy.thresholds.top_k}
            union = selected | top
            jaccard = len(selected & top) / len(union) if union else None
            for key, rank in ranks.items():
                intervals.setdefault(key, []).append(rank)
            scenarios.append(
                StabilityScenario(
                    feature_id=target.feature_id,
                    multiplier=multiplier,
                    top_k_jaccard=jaccard,
                    ranks=ranks,
                )
            )
    sensitive = any(
        item.top_k_jaccard is not None
        and item.top_k_jaccard < policy.thresholds.minimum_top_k_jaccard
        for item in scenarios
    )
    return StabilityResult(
        scenarios=tuple(scenarios),
        rank_intervals={key: (min(values), max(values)) for key, values in intervals.items()},
        sensitive=sensitive,
        reason_codes=("stability_sample_unavailable",)
        if not baseline
        else ("weight_sensitive",)
        if sensitive
        else (),
    )
