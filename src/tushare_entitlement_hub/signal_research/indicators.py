"""以同一观察日复权价格计算 TA-Lib 指标和可审计股性原始值。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import talib


@dataclass(frozen=True)
class PriceBar:
    """单日源行情；价格为未复权值，因子必须截至观察日已知。

    例：PriceBar('20260101', 10, 11, 9, 10, 2) 在锚点4下收盘为5。
    amount 和 market_cap 由调用方统一单位并保留来源，不转换、不用于金额排除。
    """

    trade_date: str
    open: float
    high: float
    low: float
    close: float
    adjustment_factor: float = 1.0
    turnover_rate: float | None = None
    amount: float | None = None
    market_cap: float | None = None
    vol: float | None = None


@dataclass(frozen=True)
class StockCharacteristics:
    """股性原始指标；窗口不足或必要输入缺失输出 None。

    例：60日单调上涨，direction_60>0、efficiency_60=1；无日振幅时影线比未知。
    方向和效率使用60个收益段（61收盘），影线使用60根K线。
    影线为上下影线合计/日振幅，跳空为与前日区间不重叠的日数/60。
    """

    direction_60: float | None
    efficiency_60: float | None
    wick_ratio_median_60: float | None
    wick_ratio_p90_60: float | None
    gap_ratio_60: float | None
    turnover_median_20: float | None
    amount_median_20: float | None
    market_cap: float | None
    zero_range_count_60: int | None = None


@dataclass(frozen=True)
class IndicatorPoint:
    """单日技术输入和原始股性；TA-Lib 预热期保留 None。

    例：histogram=0.1、close=10 时 normalized_histogram=0.01。
    所有价量关联指标的 OHLC 使用 factor/anchor_factor 同一复权口径。
    """

    trade_date: str
    close: float
    ma20: float | None
    ma60: float | None
    previous_ma20: float | None
    previous_ma60: float | None
    dif: float | None
    dea: float | None
    histogram: float | None
    normalized_histogram: float | None
    atr14: float | None
    characteristics: StockCharacteristics


def _optional_number(value: float) -> float | None:
    """将库的非有限预热值映射为未知；输入 nan 输出 None，输入1输出1.0。"""
    return float(value) if isfinite(value) else None


def _rolling_quantile(
    values: np.ndarray, window: int, quantile: float, require_complete: bool = False,
) -> np.ndarray:
    """批量计算只含过去窗口的线性插值分位数，非有限值排除或令完整窗口未知。

    输入[1,3,5]、window=2、quantile=.5输出[nan,2,4]；完整窗口含nan输出nan。
    """
    output = np.full(len(values), np.nan)
    if len(values) < window:
        return output
    finite_values = np.where(np.isfinite(values), values, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(finite_values, window)
    counts = np.sum(np.isfinite(windows), axis=1)
    sorted_values = np.sort(windows, axis=1)
    positions = np.maximum(counts - 1, 0) * quantile
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    indices = np.arange(len(windows))
    lower_values = sorted_values[indices, lower]
    upper_values = sorted_values[indices, upper]
    results = lower_values + (upper_values - lower_values) * (positions - lower)
    valid = counts == window if require_complete else counts > 0
    output[window - 1:] = np.where(valid, results, np.nan)
    return output


def _characteristic_series(
    bars: tuple[PriceBar, ...], adjusted_prices: np.ndarray,
) -> tuple[StockCharacteristics, ...]:
    """一次向量化所有过去60收益段和20日股性窗口，不裁剪研究前历史。

    输入61根单调上涨K线输出末日efficiency_60=1；前60根收益项均未知。
    影线排除零幅并记录数量，全部零幅或零收益路径保持未知。
    """
    count = len(bars)
    direction = np.full(count, np.nan)
    efficiency = np.full(count, np.nan)
    gap_ratio = np.full(count, np.nan)
    zero_counts = np.full(count, np.nan)
    closes = adjusted_prices[:, 3]
    ranges = adjusted_prices[:, 1] - adjusted_prices[:, 2]
    wicks = ranges - np.abs(closes - adjusted_prices[:, 0])
    ratios = np.divide(wicks, ranges, out=np.full(count, np.nan), where=ranges > 0)
    wick_median = _rolling_quantile(ratios, 60, .5)
    wick_p90 = _rolling_quantile(ratios, 60, .9)
    wick_median[:60] = np.nan
    wick_p90[:60] = np.nan
    if count > 60:
        direction[60:] = closes[60:] / closes[:-60] - 1
        paths = np.sum(np.lib.stride_tricks.sliding_window_view(
            np.abs(np.diff(closes)), 60), axis=1)
        efficiency[60:] = np.divide(np.abs(closes[60:] - closes[:-60]), paths,
                                    out=np.full(len(paths), np.nan), where=paths > 0)
        gaps = ((adjusted_prices[1:, 2] > adjusted_prices[:-1, 1])
                | (adjusted_prices[1:, 1] < adjusted_prices[:-1, 2]))
        gap_ratio[60:] = np.mean(np.lib.stride_tricks.sliding_window_view(gaps, 60), axis=1)
        zero_counts[60:] = np.sum(np.lib.stride_tricks.sliding_window_view(
            ranges[1:] == 0, 60), axis=1)
    turnover = _rolling_quantile(np.asarray([bar.turnover_rate for bar in bars], dtype=float),
                                20, .5, require_complete=True)
    amount = _rolling_quantile(np.asarray([bar.amount for bar in bars], dtype=float),
                              20, .5, require_complete=True)
    return tuple(StockCharacteristics(
        _optional_number(direction[index]), _optional_number(efficiency[index]),
        _optional_number(wick_median[index]), _optional_number(wick_p90[index]),
        _optional_number(gap_ratio[index]), _optional_number(turnover[index]),
        _optional_number(amount[index]),
        _optional_number(bar.market_cap) if bar.market_cap is not None else None,
        int(zero_counts[index]) if isfinite(zero_counts[index]) else None,
    ) for index, bar in enumerate(bars))


def compute_indicators(
    bars: tuple[PriceBar, ...] | list[PriceBar],
    observation_date: str,
    anchor_factor: float | None = None,
) -> tuple[IndicatorPoint, ...]:
    """截取观察日可见行情，以TA-Lib MACD(12,26,9)、SMA20/60和ATR14计算指标。

    输入日期为可排序的YYYYMMDD，行情按日期排序，重复日期和无效OHLC/因子拒绝。
    默认锚点为固定输入历史首根可见K线因子；全部OHLC乘每日因子/锚点。
    输入历史起始不得随观察日裁剪，首因子锚点保证全历史与前缀使用同一算术路径。
    H=2*(DIF-DEA)，normalized_histogram=H/复权收盘价。
    示例：仅一根10元、因子2的K线与锚点4，输出close=5且ma20=None。
    无可见行情输出空元组；未来行情不参与校验、预热、锚点或股性计算。
    """
    visible = tuple(sorted((bar for bar in bars if bar.trade_date <= observation_date),
                           key=lambda bar: bar.trade_date))
    if not visible:
        return ()
    if len({bar.trade_date for bar in visible}) != len(visible):
        raise ValueError("交易日期必须唯一")
    anchor = visible[0].adjustment_factor if anchor_factor is None else anchor_factor
    if not isfinite(anchor) or anchor <= 0:
        raise ValueError("观察日复权锚点必须为正有限数")
    for bar in visible:
        prices = (bar.open, bar.high, bar.low, bar.close, bar.adjustment_factor)
        if (any(not isfinite(value) or value <= 0 for value in prices)
                or bar.low > min(bar.open, bar.close)
                or bar.high < max(bar.open, bar.close)):
            raise ValueError("OHLC和复权因子必须有效且符合价格区间")
    adjusted = np.asarray([
        [price * bar.adjustment_factor / anchor
         for price in (bar.open, bar.high, bar.low, bar.close)] for bar in visible
    ], dtype=float)
    close_values = np.ascontiguousarray(adjusted[:, 3])
    differences, signal_lines, _ = talib.MACD(
        close_values, fastperiod=12, slowperiod=26, signalperiod=9
    )
    histograms = 2 * (differences - signal_lines)
    ma20_values = talib.SMA(close_values, timeperiod=20)
    ma60_values = talib.SMA(close_values, timeperiod=60)
    atr_values = talib.ATR(np.ascontiguousarray(adjusted[:, 1]),
                           np.ascontiguousarray(adjusted[:, 2]), close_values, timeperiod=14)
    characteristics = _characteristic_series(visible, adjusted)
    return tuple(IndicatorPoint(
        bar.trade_date, float(close_values[index]),
        _optional_number(ma20_values[index]), _optional_number(ma60_values[index]),
        _optional_number(ma20_values[index - 1]) if index else None,
        _optional_number(ma60_values[index - 1]) if index else None,
        _optional_number(differences[index]), _optional_number(signal_lines[index]),
        _optional_number(histograms[index]),
        _optional_number(histograms[index] / close_values[index]),
        _optional_number(atr_values[index]), characteristics[index],
    ) for index, bar in enumerate(visible))
