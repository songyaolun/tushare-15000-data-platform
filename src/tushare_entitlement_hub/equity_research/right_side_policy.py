"""统一 P03 的显式右侧结构阈值，初始规则不代表已验证交易策略。"""

from dataclasses import dataclass

from ..product_foundation.model import Count, Fraction, Identifier, Model, Positive


@dataclass(frozen=True, kw_only=True)
class RightSidePolicy(Model):
    """声明回调、确认和入场约束，全部进入研究配置与运行指纹。

    例如默认 maximum_entry_risk=0.08，表示观察价距结构失效价最多 8%；
    minimum_reward_risk=2.0 要求到历史阻力的空间至少为该风险的两倍。
    """

    version: Identifier = "right-side-policy/v1"
    history_observations: Count = 61
    consolidation_sessions: Count = 10
    confirmation_sessions: Count = 5
    minimum_pullback: Fraction = 0.05
    minimum_higher_low: Fraction = 0.005
    maximum_base_range: Fraction = 0.12
    maximum_breakout_extension: Fraction = 0.03
    maximum_entry_risk: Fraction = 0.08
    minimum_reward_risk: Positive = 2.0
    minimum_amount_ratio: Positive = 1.2
    retest_tolerance: Fraction = 0.01
    invalidation_buffer: Fraction = 0.005

    def __post_init__(self) -> None:
        """拒绝不可计算的窗口与阈值；例如 history_observations=20 抛配置错误。

        输入合法默认配置返回独立不可变规则；比率边界与窗口长度均失败关闭。
        """
        super().__post_init__()
        if (
            self.history_observations < 61
            or self.consolidation_sessions < 3
            or self.confirmation_sessions < 1
            or self.history_observations
            <= self.consolidation_sessions + self.confirmation_sessions + 20
            or self.history_observations
            <= 2 * self.consolidation_sessions + self.confirmation_sessions
            or not 0 < self.minimum_pullback < 1
            or not 0 < self.maximum_base_range < 1
            or not 0 < self.maximum_breakout_extension < 1
            or not 0 < self.maximum_entry_risk < 1
            or not 0 <= self.invalidation_buffer < 1
            or self.minimum_amount_ratio < 1
        ):
            raise ValueError("invalid_right_side_policy")
