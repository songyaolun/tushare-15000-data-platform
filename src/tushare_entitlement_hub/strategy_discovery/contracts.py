"""策略发现的统一契约；数组按交易日×证券排列。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray

DEFAULT_POLICY = PortfolioPolicy(switching=False, stage_allocation=False, force_rebalance=False)


@dataclass(frozen=True)
class StrategyDefinition:
    """绑定独立入场、排名和退出，默认总仓90%、单票5%、行业25%。

    例StrategyDefinition('volume', 'volume', mask, scores)使用默认仓位政策；
    maximum_entry_gap=.03拒绝相对信号日绝对复权收盘高开超过3%的订单，
    trailing_stop=.08表示任意市场阶段从持仓最高复权收盘回撤8%触发退出。
    intraday_exits=True启用T+1日线OHLC固定止盈止损，默认False保留收盘退出模型。
    mask和scores形状必须一致；调用方负责保持数组不变及信号计算的因果性。
    """

    name: str
    family: str
    entry_mask: NDArray[np.bool_]
    scores: FloatArray
    policy: PortfolioPolicy = DEFAULT_POLICY
    exit_mask: NDArray[np.bool_] | None = None
    maximum_entry_gap: float | None = None
    trailing_stop: float = 0.0
    intraday_exits: bool = False

    def __post_init__(self) -> None:
        """检查矩阵、策略开关和退出阈值，错误配置在执行前失败。

        例maximum_entry_gap=-.01或entry_mask为一维数组均抛出ValueError。
        分数允许NaN代表不可排名，执行时会排除这些证券。
        """
        if (
            not self.name
            or not self.family
            or self.entry_mask.ndim != 2
            or self.entry_mask.dtype != np.bool_
            or self.scores.shape != self.entry_mask.shape
            or not np.issubdtype(self.scores.dtype, np.floating)
            or (
                self.exit_mask is not None
                and (
                    self.exit_mask.shape != self.entry_mask.shape
                    or self.exit_mask.dtype != np.bool_
                )
            )
            or self.policy.switching
            or self.policy.stage_allocation
            or self.policy.force_rebalance
            or not 0 <= self.trailing_stop < 1
            or not isinstance(self.intraday_exits, bool)
            or (
                self.maximum_entry_gap is not None
                and (not np.isfinite(self.maximum_entry_gap) or self.maximum_entry_gap < 0)
            )
        ):
            raise ValueError("invalid_strategy_definition")
