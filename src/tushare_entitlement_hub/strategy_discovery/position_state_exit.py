"""只用信号日及当前已知收盘状态决定提前退出或延长止盈。"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..regime_research.accounting import Holding
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy


@dataclass(frozen=True)
class PositionStateExitRule:
    """冻结状态退出阈值，硬止损和持有期限仍由账户优先执行。

    例PositionStateExitRule(extend_profit=True)允许趋势与行业共同走强时将4%止盈
    延至8%；failure_after=5表示至少持有5个市场日才允许失败早退。
    行业广度及行业收益均取截至当前的5个市场日，不能使用日后值填补缺失。
    """

    extend_profit: bool = False
    extended_profit: float = 0.08
    minimum_breadth: float = 0.5
    failure_after: int = 0
    failure_return: float = -0.02
    breadth_drop: float = 0.10
    require_sector_failure: bool = True

    def __post_init__(self) -> None:
        """检查有限阈值和开关；例failure_after=-1或extended_profit=NaN抛ValueError。"""
        numbers = (
            self.extended_profit,
            self.minimum_breadth,
            self.failure_return,
            self.breadth_drop,
        )
        if (
            not all(type(value) in (int, float) and isfinite(value) for value in numbers)
            or type(self.extend_profit) is not bool
            or type(self.require_sector_failure) is not bool
            or type(self.failure_after) is not int
            or self.failure_after < 0
            or not 0.04 < self.extended_profit < 1
            or not 0 <= self.minimum_breadth <= 1
            or not -0.08 < self.failure_return <= 0
            or not 0 < self.breadth_drop <= 1
        ):
            raise ValueError("invalid_position_state_exit_rule")


@dataclass(frozen=True)
class PositionCloseState:
    """不含未来值的单次持仓快照；例return_multiple=1.05表示复权收盘盈利5%。

    sector_return是行业日均收益的五日算术和，只作方向指标，不视作可投资收益。
    peak_multiple包括已知收盘峰值，用于识别已经触及基础止盈的持仓。
    """

    held_sessions: int
    return_multiple: float
    peak_multiple: float
    above_average: bool | None
    stock_return: float
    sector_breadth: float
    signal_sector_breadth: float
    sector_return: float


def position_state_exit_reason(
    state: PositionCloseState,
    rule: PositionStateExitRule,
    *,
    take_profit: float = 0.04,
) -> str:
    """纯函数输出退出原因；例盈利5%但行业五日收益为负时输出state_profit。

    延长后只要共同强势消失就退出，另设固定盈利上限，不用跟踪回撤百分比。
    状态缺失不能支持延长或提前止损；基础止盈仍有效。失败早退只针对亏损且
    跌破20日均线、五日收益非正的持仓，可要求行业相对信号日恶化共同确认。
    """
    if not isfinite(state.return_multiple) or state.return_multiple <= 0:
        return ""
    known = state.above_average is not None and all(
        isfinite(value)
        for value in (
            state.stock_return,
            state.sector_breadth,
            state.signal_sector_breadth,
            state.sector_return,
        )
    )
    strong = (
        known
        and state.above_average
        and state.stock_return > 0
        and state.sector_breadth >= rule.minimum_breadth
        and state.sector_return > 0
        and state.sector_breadth >= state.signal_sector_breadth - rule.breadth_drop
    )
    activated = state.peak_multiple >= 1 + take_profit
    if state.return_multiple >= 1 + take_profit or (rule.extend_profit and activated):
        if (
            not rule.extend_profit
            or not strong
            or state.return_multiple >= 1 + rule.extended_profit
        ):
            return "state_profit"
    sector_failed = (
        state.sector_breadth <= state.signal_sector_breadth - rule.breadth_drop
        and state.sector_return <= 0
    )
    if (
        rule.failure_after
        and state.held_sessions >= rule.failure_after
        and known
        and state.return_multiple <= 1 + rule.failure_return
        and not state.above_average
        and state.stock_return <= 0
        and (not rule.require_sector_failure or sector_failed)
    ):
        return "state_failed"
    return ""


def known_sector_state(panel: Panel, day: int, security: int) -> tuple[float, float]:
    """返回当日及前4日行业广度均值、行业日均收益之和；例五日均为.6/.01得(.6,.05)。

    不足五日、任何缺失、无效行业或行业归属变化均返回NaN，不能跨行业拼接趋势。
    """
    if day < 4:
        return float("nan"), float("nan")
    window = slice(day - 4, day + 1)
    sectors = panel.values["sector_index"][window, security]
    breadth = panel.values["sector_breadth"][window, security]
    returns = panel.values["sector_mean_return"][window, security]
    if (
        not np.isfinite(sectors).all()
        or not (sectors >= 0).all()
        or not (sectors == sectors[-1]).all()
        or not np.isfinite(breadth).all()
        or not np.isfinite(returns).all()
    ):
        return float("nan"), float("nan")
    return float(breadth.mean()), float(returns.sum())


def make_position_state_exit(
    rule: PositionStateExitRule,
) -> Callable[[Panel, Holding, int, PortfolioPolicy], str]:
    """创建账户收盘回调；例make_position_state_exit(rule)(panel,holding,day,policy)。

    使用raw_close×raw_adj_factor与原账本同尺度比较成本，不修改持仓，不读当前收盘后信息。
    调用方须关闭固定止盈，但保留固定止损和期限；跨行业时缺失信号比较，不能延长。
    """

    def close_exit(panel: Panel, holding: Holding, day: int, policy: PortfolioPolicy) -> str:
        """将当前及信号日事实交给纯函数；例未触发任何状态返回空字符串。"""
        security = holding.security
        sector_breadth, sector_return = known_sector_state(panel, day, security)
        signal_breadth, _ = known_sector_state(panel, holding.confirmation, security)
        if (
            panel.values["sector_index"][day, security]
            != panel.values["sector_index"][holding.confirmation, security]
        ):
            signal_breadth = float("nan")
        raw_close = float(panel.values["raw_close"][day, security])
        adjustment_factor = float(panel.values["raw_adj_factor"][day, security])
        if not all(
            isfinite(value) and value > 0
            for value in (raw_close, adjustment_factor, holding.entry_adjusted)
        ):
            return ""
        adjusted = raw_close * adjustment_factor
        state = PositionCloseState(
            held_sessions=day - holding.entry + 1,
            return_multiple=float(adjusted / holding.entry_adjusted),
            peak_multiple=float(max(holding.peak_adjusted, adjusted) / holding.entry_adjusted),
            above_average=(
                bool(panel.values["close"][day, security] > panel.values["ma20"][day, security])
                if np.isfinite(panel.values["close"][day, security])
                and np.isfinite(panel.values["ma20"][day, security])
                else None
            ),
            stock_return=float(panel.values["return5"][day, security]),
            sector_breadth=sector_breadth,
            signal_sector_breadth=signal_breadth,
            sector_return=sector_return,
        )
        return position_state_exit_reason(state, rule, take_profit=policy.take_profit)

    return close_exit
