"""只读当前及以前行情的市场状态和历史Beta。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray, lag, rolling

STAGES = ("transition", "advance", "repair", "decline", "risk")


@dataclass(frozen=True)
class RegimeParameters:
    """预声明状态阈值；例默认上涨广度60%、下跌广度40%、修复提高10个百分点。"""

    advance_breadth: float = 0.60
    decline_breadth: float = 0.40
    repair_improvement: float = 0.10
    confirmation: int = 2

    def __post_init__(self) -> None:
        """拒绝无序阈值和非正确认日数；例如confirmation=0抛ValueError。"""
        if not (
            0 < self.decline_breadth < self.advance_breadth < 1
            and 0 < self.repair_improvement < 1
            and type(self.confirmation) is int
            and self.confirmation > 0
        ):
            raise ValueError("invalid_regime_parameters")


DEFAULT_REGIME = RegimeParameters()


def rolling_beta(returns: FloatArray, benchmark: FloatArray, window: int = 120) -> FloatArray:
    """按成对完整历史窗口计算Beta并向1收缩25%，结果用于下一日交易。

    例收益为基准两倍，非零波动完整窗口输出1.75；不足窗口或零方差输出NaN。
    """
    average_market = rolling(benchmark, window, "mean")
    covariance = rolling(returns * benchmark, window, "mean") - (
        rolling(returns, window, "mean") * average_market
    )
    variance = rolling(benchmark * benchmark, window, "mean") - average_market**2
    result = np.full(len(returns), np.nan)
    np.divide(covariance, variance, out=result, where=variance > 1e-12)
    return 0.75 * np.clip(result, 0, 3) + 0.25


def classify_regimes(
    index_close: FloatArray,
    breadth: FloatArray,
    advancing: FloatArray,
    amount: FloatArray,
    limit_down: FloatArray,
    parameters: RegimeParameters = DEFAULT_REGIME,
) -> NDArray[np.int8]:
    """生成收盘可知的互斥状态：0过渡、1上涨、2修复、3下跌、4风险。

    风险优先且立即生效，其他变化连续确认；预热/缺值不交易(4)。
    例连续两日满足上涨条件，第二日收盘输出1，订单最早第三日开盘执行。
    所有分位阈值仅用此前252日，绝不用五年总体分位。
    """
    count = len(index_close)
    if any(len(array) != count for array in (breadth, advancing, amount, limit_down)):
        raise ValueError("market_array_length_mismatch")
    average20 = rolling(index_close, 20, "mean")
    average60 = rolling(index_close, 60, "mean")
    returns = index_close / lag(index_close) - 1
    volatility = rolling(returns, 20, "std")
    volume_reference = lag(rolling(amount, 20, "median"))
    change = breadth - lag(breadth, 5)
    falling = (index_close < average60) & (breadth < parameters.decline_breadth) & (change <= 0)
    recent_decline = rolling(lag(falling.astype(float)), 10, "max") == 1
    repair = (
        recent_decline
        & (change >= parameters.repair_improvement)
        & (advancing >= 0.6)
        & (amount > volume_reference)
    )
    advance = (
        (index_close > average60)
        & (average20 > average60)
        & (average20 > lag(average20, 5))
        & (breadth >= parameters.advance_breadth)
    )
    states = np.full(count, 4, dtype=np.int8)
    previous, candidate, streak = 4, -1, 0
    for day in range(count):
        known = all(
            np.isfinite(array[day])
            for array in (
                average60,
                breadth,
                advancing,
                volatility,
                limit_down,
                volume_reference,
            )
        )
        if day < 272 or not known:
            previous, candidate, streak = 4, -1, 0
            continue
        historical_volatility = volatility[day - 252 : day]
        historical_limits = limit_down[day - 252 : day]
        risk = limit_down[day] > max(0.03, float(np.quantile(historical_limits, 0.95)))
        risk |= volatility[day] > np.quantile(historical_volatility, 0.95) and returns[day] < -0.02
        current = (
            4 if risk else 2 if repair[day] else 1 if advance[day] else (3 if falling[day] else 0)
        )
        streak = streak + 1 if current == candidate else 1
        candidate = current
        if current == 4 or streak >= parameters.confirmation:
            previous = current
        states[day] = previous
    return states
