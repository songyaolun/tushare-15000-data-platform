"""用截至收盘的持仓衰退事实触发退出或满仓替换，不比较新旧入场分数。"""

from dataclasses import dataclass
from math import isfinite

from ..regime_research.accounting import Holding
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy


@dataclass(frozen=True)
class ReplacementRule:
    """冻结持仓衰退和新入场阈值；例minimum_sessions=10表示第十个持有日开始判断。"""

    minimum_sessions: int = 10
    maximum_return: float = 0.01
    confirmation: str = "momentum"
    minimum_candidate_score: float = 0.0
    maximum_replacements: int = 1

    def __post_init__(self) -> None:
        """拒绝不合法阈值；例minimum_sessions=0或未知确认方式抛ValueError。"""
        if (
            type(self.minimum_sessions) is not int
            or not 1 <= self.minimum_sessions <= 30
            or type(self.maximum_replacements) is not int
            or not 1 <= self.maximum_replacements <= 10
            or self.confirmation not in {"age", "momentum", "trend", "sector"}
            or not isfinite(self.maximum_return)
            or not -0.08 < self.maximum_return < 0.045
            or not isfinite(self.minimum_candidate_score)
        ):
            raise ValueError("invalid_replacement_rule")


def fading_position(panel: Panel, holding: Holding, day: int, rule: ReplacementRule) -> bool:
    """用当前复权盈亏和五日动量识别衰退；例持有10日、收益0、五日跌2%返回True。

    age仅限制持有期与盈亏；trend另要求跌破MA20；sector另要求行业广度不足40%。
    所需价格或特征缺失、锁定送股、未达最短持有期均不支持主动换仓。
    """
    if (
        holding.sell_reason
        or holding.locked > 1e-8
        or day - holding.entry + 1 < rule.minimum_sessions
    ):
        return False
    security = holding.security
    closing = float(panel.values["raw_close"][day, security])
    factor = float(panel.values["raw_adj_factor"][day, security])
    if not all(
        isfinite(value) and value > 0 for value in (closing, factor, holding.entry_adjusted)
    ):
        return False
    if closing * factor / holding.entry_adjusted - 1 > rule.maximum_return:
        return False
    if rule.confirmation == "age":
        return True
    momentum = float(panel.values["return5"][day, security])
    if not isfinite(momentum) or momentum > 0:
        return False
    if rule.confirmation == "trend":
        current = float(panel.values["close"][day, security])
        average = float(panel.values["ma20"][day, security])
        return isfinite(current) and isfinite(average) and 0 < current < average
    if rule.confirmation == "sector":
        breadth = float(panel.values["sector_breadth"][day, security])
        return isfinite(breadth) and 0 <= breadth < 0.4
    return True


def make_fading_exit(rule: ReplacementRule):
    """创建衰退退出对照，持仓达到规则即退出；例未衰退返回空字符串。"""

    def close_exit(panel: Panel, holding: Holding, day: int, policy: PortfolioPolicy) -> str:
        """返回附加退出原因；例衰退持仓返回fading，原止盈止损优先。"""
        return "fading" if fading_position(panel, holding, day, rule) else ""

    return close_exit


def make_replacement_provider(rule: ReplacementRule):
    """创建满仓换仓回调；例最久未走强的持仓0与合格高分候选2配对为[(0,2)]。

    候选按已冻结排序输入。优先替换持有时间长者，同日入场按证券序；不读取未来价格。
    新候选分数仅设入场门槛，不解释为它与现有持仓的剩余收益差。
    """

    def replacements(panel, day, policy, holdings, candidates, pending):
        """从持仓副本和合格候选返回唯一配对；例无衰退持仓或无新候选返回[]。"""
        held_codes = {holding.security for holding in holdings}
        outgoing = sorted(
            (holding for holding in holdings if fading_position(panel, holding, day, rule)),
            key=lambda holding: (holding.entry, holding.security),
        )
        incoming = [
            security
            for security, _, score in candidates
            if security not in held_codes
            and security not in pending
            and isfinite(score)
            and score >= rule.minimum_candidate_score
        ]
        return [
            (holding.security, security)
            for holding, security in list(zip(outgoing, incoming, strict=False))[
                : rule.maximum_replacements
            ]
        ]

    return replacements
