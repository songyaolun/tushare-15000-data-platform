"""仅用信号收盘与次日开盘判断入场价格区间，保持绝对复权尺度一致。"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ..regime_research.panel import Panel


@dataclass(frozen=True)
class OpeningGapRule:
    """声明相对信号收盘的开盘涨跌幅区间，两端均包含，None表示不设该端。

    例OpeningGapRule(-.03, .01)只接受低开不超过3%、高开不超过1%的开盘。
    规则决定是否买入，不假设在区间边界挂限价单就一定成交。
    """

    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        """拒绝非法数字或倒置区间；例OpeningGapRule(.02, .01)抛ValueError。"""
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or not -1 < value < 1
            )
            for value in (self.minimum, self.maximum)
        ) or (
            self.minimum is not None and self.maximum is not None and self.minimum > self.maximum
        ):
            raise ValueError("invalid_opening_gap_rule")


def opening_gap_allowed(
    panel: Panel, confirmation: int, day: int, security: int, rule: OpeningGapRule
) -> bool:
    """以开盘已知价格比较区间，不读取当日高低收盘，价格或因子缺失时拒绝。

    仅允许信号下一市场日；例信号收盘10/因子2、次开盘5.05/因子4，
    OpeningGapRule(-.03,.01)返回True，避免除权被误认为巨大低开。
    无上下界仍需全部价格和因子为正且有限；次日开盘判断不能用于前一收盘选股。
    """
    if day != confirmation + 1:
        return False
    closing = float(panel.values["raw_close"][confirmation, security])
    reference_factor = float(panel.values["raw_adj_factor"][confirmation, security])
    opening = float(panel.values["raw_open"][day, security])
    opening_factor = float(panel.values["raw_adj_factor"][day, security])
    if not all(
        isfinite(value) and value > 0
        for value in (closing, reference_factor, opening, opening_factor)
    ):
        return False
    reference = closing * reference_factor
    adjusted_opening = opening * opening_factor
    return (rule.minimum is None or adjusted_opening >= reference * (1 + rule.minimum)) and (
        rule.maximum is None or adjusted_opening <= reference * (1 + rule.maximum)
    )
