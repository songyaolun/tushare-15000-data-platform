"""按收盘候选排名缩减预算，保持原名额和账户容量限制。"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, PositionWeightProvider


@dataclass(frozen=True)
class RankWeightRule:
    """声明第一名和末名预算比例，候选之间线性插值，不把分数解释成胜率。

    例RankWeightRule(.05,.03)对三名候选给5%、4%、3%；同值.04为等预算对照。
    所有比例必须为正、不超过100%，实际调用另受账户单股上限约束。
    """

    first_weight: float = 0.05
    last_weight: float = 0.03

    def __post_init__(self) -> None:
        """检查有限真实数字和由高到低的预算；例(.03,.05)或零权重抛ValueError。"""
        if (
            any(
                isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
                for value in (self.first_weight, self.last_weight)
            )
            or not 0 < self.last_weight <= self.first_weight <= 1
        ):
            raise ValueError("invalid_rank_weight_rule")


def make_rank_weight_provider(rule: RankWeightRule) -> PositionWeightProvider:
    """返回只读取原候选排名的收盘预算回调；例三候选rank1返回首尾权重均值。

    单候选使用首名预算；同分沿用执行器稳定证券顺序。缺少名额时不重新归一化，
    持仓候选仍占排名位置，因此不能用未来成交结果给其他股票加预算。
    """

    def position_weight(
        panel: Panel,
        day: int,
        security: int,
        score: float,
        rank: int,
        candidate_count: int,
        policy: PortfolioPolicy,
    ) -> float:
        """输入信号日候选位置，输出正且不超过政策上限的权重；例(.05,.03),rank0得.05。"""
        if (
            type(candidate_count) is not int
            or type(rank) is not int
            or candidate_count < 1
            or not 0 <= rank < candidate_count
            or rule.first_weight > policy.position_weight
        ):
            raise ValueError("invalid_rank_weight_context")
        fraction = rank / (candidate_count - 1) if candidate_count > 1 else 0.0
        return rule.first_weight * (1 - fraction) + rule.last_weight * fraction

    return position_weight
