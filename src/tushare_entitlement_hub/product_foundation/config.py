"""所有研究阈值由可序列化、版本化配置传入计算组件。"""

from dataclasses import dataclass
from pathlib import Path

from .model import Count, Fraction, Identifier, Model, Positive

MAXIMUM_CONTRIBUTION_ERROR = 1e-8


@dataclass(frozen=True, kw_only=True)
class FoundationPolicy(Model):
    """公共初始规格不是实测校准结果；每次计算须显式传入此配置。"""

    version: Identifier
    minimum_peer_count: Count = 30
    minimum_weight_coverage: Fraction = 0.8
    contribution_tolerance: Positive = MAXIMUM_CONTRIBUTION_ERROR
    weight_perturbation: Fraction = 0.1
    minimum_top_k_jaccard: Fraction = 0.7
    top_k: Count = 10
    maximum_observation_age_days: Positive = 7

    def __post_init__(self) -> None:
        """拒绝没有统计意义的样本门槛和空 Top-K；例如 minimum_peer_count=1 → 错误。

        输入版本及阈值，输出校验后的不可变配置；默认 minimum_peer_count → 30。
        """
        super().__post_init__()
        if self.minimum_peer_count < 2 or self.top_k < 1 or self.weight_perturbation >= 1:
            raise ValueError("invalid_foundation_policy")


def load_policy(path: Path) -> FoundationPolicy:
    """从离线 JSON 读取完整版本配置，非法参数沿用模型错误码。

    例如文件包含 {"version":"v1"}，输出使用公开初始阈值的 FoundationPolicy。
    """
    return FoundationPolicy.from_json(path.read_text(encoding="utf-8"))
