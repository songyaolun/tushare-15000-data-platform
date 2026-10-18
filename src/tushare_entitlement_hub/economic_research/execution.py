"""冻结短反信号的严格下一开盘适配，不重复查询选股资格。"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from .reversal import ReversalStrategy, candidate_orders


def next_open_allowed(panel: Panel, confirmation: int, day: int, security: int) -> bool:
    """只允许dispatch次日开盘且复权因子正有限，不读取成交日收盘资格。

    例如confirmation=5、day=6、raw_adj_factor=1返回True；第7日或NaN因子返回False。
    开盘价与涨停检查由账本先完成，本函数只负责执行时钟和有效因子。
    """
    factor = panel.values["raw_adj_factor"][day, security]
    return bool(day == confirmation + 1 and np.isfinite(factor) and factor > 0)


def cancel_expired_order(panel: Panel, confirmation: int, day: int, security: int) -> bool:
    """未成交买单在唯一允许日收盘撤销；例如confirmation=5、day=6返回True。

    即使缺开盘或涨停导致账本跳过entry_filter，本hook仍会撤单，禁止随后第三日补买。
    panel/security只用于适配账本接口，不读取新的选股条件。
    """
    return day >= confirmation + 1


def _validate_strategy(panel: Panel, strategy: ReversalStrategy) -> None:
    """核验冻结矩阵轴、source时钟及只持有五日政策；例shape错位或开启止损明确拒绝。"""
    shape = len(panel.calendar), len(panel.codes)
    if (
        strategy.entry_mask.shape != shape
        or strategy.scores.shape != shape
        or strategy.source_days.shape != (shape[0],)
    ):
        raise ValueError("reversal_strategy_shape_mismatch")
    active = np.any(strategy.entry_mask, axis=1)
    dispatch = np.arange(shape[0])[active]
    sources = strategy.source_days[active]
    lag = 1 if strategy.name == "delayed" else 0
    if np.any(sources < 0) or np.any(sources != dispatch - lag):
        raise ValueError("reversal_source_clock_mismatch")
    policy = strategy.policy
    if (
        policy.fixed_price_exits
        or policy.maximum_holding != 5
        or policy.trailing
        or policy.early_exit
        or policy.stage_allocation
        or policy.force_rebalance
        or strategy.maximum_pending_sessions != 1
    ):
        raise ValueError("reversal_requires_time_only_five_session_policy")


def simulate_reversal(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    strategy: ReversalStrategy,
    *,
    capital: float = 1_000_000,
    start: str = "20211018",
) -> dict[str, Any]:
    """用原资金账本重放冻结短反策略，保留dispatch与原信号两个时间轴。

    例如基础source=5在6日开盘买入，delayed的source=5/dispatch=6在7日买入。
    入场后第五市场日收盘提交时间退出、下个可成交开盘卖出；卖出可受跌停/停牌延后。
    固定止盈止损、跟踪、阶段减仓及额外退出全部关闭，candidate_orders只读取冻结矩阵。
    输出逐笔增加source_signal，任何晚入导致源时钟偏移均拒绝，不偷偷延長持有窗口。
    """
    _validate_strategy(panel, strategy)

    def candidate_provider(
        current_panel: Panel, day: int, stage: int, current_policy: PortfolioPolicy
    ) -> list[tuple[int, str, float]]:
        """仅读取冻结资格和分数；例如日5输出candidate_orders(strategy,5)，不重查成交日过滤。"""
        return candidate_orders(strategy, day)

    result = simulate_portfolio(
        panel,
        states,
        events,
        strategy.policy,
        capital=capital,
        start=start,
        candidate_provider=candidate_provider,
        entry_filter=next_open_allowed,
        cancel_pending_provider=cancel_expired_order,
    )
    delay = 1 if strategy.name == "delayed" else 0
    for trade in result["trades"]:
        dispatch = int(trade["confirmation"])
        source = int(strategy.source_days[dispatch])
        if int(trade["entry"]) != dispatch + 1 or int(trade["entry"]) != source + delay + 1:
            raise ValueError("reversal_fill_clock_violation")
        trade["source_signal"] = source
    result["execution_model"] = {
        "entry": "dispatch_plus_one_open_only",
        "source_delay_sessions": delay,
        "unfilled": "cancel_same_day_close",
        "holding_sessions": 5,
        "exit": "time_then_next_executable_open",
        "fixed_price_exits": False,
    }
    return result
