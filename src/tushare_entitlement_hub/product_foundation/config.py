"""所有研究阈值由可序列化、版本化配置传入计算组件。"""

from dataclasses import dataclass
from pathlib import Path

from .model import Count, Fraction, Identifier, Model, Positive

MAXIMUM_CONTRIBUTION_ERROR = 1e-8
MAX_FACT_BATCH_SIZE = 512
DEFAULT_FACT_BATCH_SIZE = MAX_FACT_BATCH_SIZE
READ_BATCH_SIZE = 128


def validate_batch_size(size: int, *, error_code: str = "invalid_batch_size") -> None:
    """检查批大小为1到事实批上限之间的整数，避免配置驱动的无界Python对象分配。

    输入512返回None；输入513、百万、布尔值或非正整数抛ValueError。
    配置可传入自身错误码，直接读写入口默认使用invalid_batch_size。
    """
    if type(size) is not int or not 1 <= size <= MAX_FACT_BATCH_SIZE:
        raise ValueError(error_code)


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
    fact_batch_size: Count = DEFAULT_FACT_BATCH_SIZE

    def __post_init__(self) -> None:
        """拒绝没有统计意义的样本门槛和空 Top-K；例如 minimum_peer_count=1 → 错误。

        输入版本及阈值，输出校验后的不可变配置；默认 minimum_peer_count → 30。
        事实批大小仅允许1到512；例如 fact_batch_size=513 → invalid_foundation_policy。
        """
        super().__post_init__()
        validate_batch_size(self.fact_batch_size, error_code="invalid_foundation_policy")
        if self.minimum_peer_count < 2 or self.top_k < 1 or self.weight_perturbation >= 1:
            raise ValueError("invalid_foundation_policy")


def load_policy(path: Path) -> FoundationPolicy:
    """从离线 JSON 读取完整版本配置，非法参数沿用模型错误码。

    例如文件包含 {"version":"v1"}，输出使用公开初始阈值的 FoundationPolicy。
    """
    return FoundationPolicy.from_json(path.read_text(encoding="utf-8"))
