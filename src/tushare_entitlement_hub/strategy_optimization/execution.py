"""全市场参数搜索的向量成交引擎，保留逐笔可独立复算的时序规则。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import FloatArray


@dataclass(frozen=True)
class ExitPolicy:
    """固定比例止盈止损和市场日持有上限，全部收盘观察、随后开盘执行。

    例：ExitPolicy(.03,.06,20)表示盈利3%或亏损6%触发，最多观察20市场日。
    阈值不保证成交价格，跳空、限板和停牌可造成更大损益。
    """

    take_profit: float
    stop_loss: float
    holding_sessions: int

    def __post_init__(self) -> None:
        """拒绝无效交易参数；输入 stop_loss=0 或 holding_sessions=True 抛ValueError。"""
        if (
            not 0 < self.take_profit < 1
            or not 0 < self.stop_loss < 1
            or type(self.holding_sessions) is not int
            or self.holding_sessions < 1
        ):
            raise ValueError("invalid_exit_policy")

    @property
    def name(self) -> str:
        """返回可复算参数标识；输入(.03,.06,20)，输出tp0.03_sl0.06_h20。"""
        return f"tp{self.take_profit:g}_sl{self.stop_loss:g}_h{self.holding_sessions}"


def exit_grid() -> tuple[ExitPolicy, ...]:
    """返回预声明的45组退出参数；止盈1.5/2.5/4/6/8%，止损3/6/10%，期限20/40/60日。"""
    return tuple(
        ExitPolicy(target, stop, holding)
        for target in (0.015, 0.025, 0.04, 0.06, 0.08)
        for stop in (0.03, 0.06, 0.10)
        for holding in (20, 40, 60)
    )


def entry_indices(features: dict[str, FloatArray], expiry: int = 3) -> NDArray[np.int64]:
    """每个确认日寻找随后至第三市场日的首个合格开盘，不能当日买入。

    输入首日确认、次日涨停、第三日正常，输出首日对应entry_index=2。
    缺限价、零量或无合格OHLC拒绝，返回-1表示到期或样本尾端未成交。
    """
    if type(expiry) is not int or expiry < 1:
        raise ValueError("expiry_must_be_positive_integer")
    count = len(features["valid"])
    opening = features["raw_open"]
    upper = features["raw_up_limit"]
    eligible = (features["valid"] == 1) & np.isfinite(upper) & (upper > 0) & (opening < upper)
    entries = np.full(count, -1, dtype=np.int64)
    for offset in range(1, min(expiry, count - 1) + 1):
        candidates = np.flatnonzero((entries[:-offset] < 0) & eligible[offset:])
        entries[candidates] = candidates + offset
    return entries


def outcome_arrays(
    features: dict[str, FloatArray],
    calendar: list[str],
    policy: ExitPolicy,
    entries: NDArray[np.int64] | None = None,
) -> dict[str, np.ndarray]:
    """计算所有确认日的潜在交易，尚未去除同股票持仓重叠。

    例：确认后10元开盘买入，收盘到止盈线，次日11元开盘卖出，按实际开盘扣费。
    双边5bp滑点、3bp佣金，历史印花税分段；复权价格收益非现金分红账本。
    止盈止损同为收盘观察，T+1后才能卖；触发后等待首个非跌停有效开盘。
    exit_index=count表示未退出，entry_index=-1表示未买入，收益保持NaN。
    """
    count = len(calendar)
    if len(features["valid"]) != count or not count:
        raise ValueError("execution_calendar_length_invalid")
    entries = entry_indices(features) if entries is None else entries
    if len(entries) != count or np.any(entries >= count) or np.any(entries < -1):
        raise ValueError("entry_indices_invalid")
    # 所有向量均由实际成交日原始报价和当日复权因子构成。
    opening, factor = features["raw_open"], features["raw_adj_factor"]
    buy_prices = np.minimum(opening * 1.0005, features["raw_up_limit"])
    sell_prices = np.maximum(opening * 0.9995, features["raw_down_limit"])
    valid_entry = entries >= 0
    safe_entries = np.maximum(entries, 0)
    entry_adjusted = buy_prices[safe_entries] * factor[safe_entries]
    eligible_exit = (
        (features["valid"] == 1)
        & (features["raw_down_limit"] > 0)
        & (opening > features["raw_down_limit"])
    )
    next_exit = np.minimum.accumulate(np.where(eligible_exit, np.arange(count), count)[::-1])[::-1]
    next_exit = np.append(next_exit, count)
    trigger = np.full(count, count, dtype=np.int64)
    reason = np.zeros(count, dtype=np.int8)
    for held in range(policy.holding_sessions):
        positions = entries + held
        active = valid_entry & (positions < count) & (trigger == count)
        indices = np.flatnonzero(active)
        if not len(indices):
            continue
        day = positions[indices]
        closing_multiple = features["raw_close"][day] * factor[day] / entry_adjusted[indices]
        valid_close = features["valid"][day] == 1
        profit = valid_close & (closing_multiple >= 1 + policy.take_profit)
        loss = valid_close & (closing_multiple <= 1 - policy.stop_loss)
        time_cap = held + 1 >= policy.holding_sessions
        hit = profit | loss | time_cap
        trigger[indices[hit]] = day[hit]
        reason[indices[hit]] = np.where(profit[hit], 1, np.where(loss[hit], 2, 3))
    exits = next_exit[np.minimum(trigger + 1, count)]
    returns = np.full(count, np.nan)
    closed = valid_entry & (exits < count)
    days = exits[closed]
    multiples = sell_prices[days] * factor[days] / entry_adjusted[closed]
    stamp = np.where(np.asarray(calendar)[days] >= "20230828", 0.0005, 0.001)
    returns[closed] = multiples - 1 - 0.0003 - multiples * (0.0003 + stamp)
    return {
        "entry_index": entries,
        "exit_index": exits,
        "trigger_index": trigger,
        "return": returns,
        "exit_reason": reason,
        "buy_price": buy_prices,
        "sell_price": sell_prices,
    }


def independent_trades(
    signal_indices: NDArray[np.int64],
    outcomes: dict[str, np.ndarray],
    count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """按时间去除同证券重叠持仓和三日待买单，返回保留与跳过的确认日索引。

    例：0日确认、3日退出，1日信号跳过、3日收盘可再次确认；未退出占用到样本末。
    未成交订单占用至确认后第三日收盘，新信号在到期当日收盘可重新确认。
    """
    if np.any(np.diff(signal_indices) <= 0):
        raise ValueError("signals_must_be_unique_increasing")
    occupied_until = -1
    retained, skipped = [], []
    for index in signal_indices:
        if index < occupied_until:
            skipped.append(index)
            continue
        retained.append(index)
        occupied_until = (
            int(outcomes["exit_index"][index])
            if outcomes["entry_index"][index] >= 0
            else min(int(index) + 3, count)
        )
    return np.asarray(retained, dtype=np.int64), np.asarray(skipped, dtype=np.int64)
