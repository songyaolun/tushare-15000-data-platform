"""只用观察日及以前行情构建可重复使用的量价和摆动指标。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import talib
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lag(values: FloatArray, periods: int = 1) -> FloatArray:
    """将序列向后平移，前端补未知；输入 [1,2,3] 与 1，输出 [nan,1,2]。

    periods 必须为正整数，空输入返回空数组，不循环引用尾端数据。
    """
    if type(periods) is not int or periods < 1:
        raise ValueError("lag_periods_must_be_positive_integer")
    result = np.full(len(values), np.nan)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def rolling(values: FloatArray, window: int, operation: str) -> FloatArray:
    """以包含当日的完整历史窗口计算统计量；[1,3,5] 两日均值为 [nan,2,4]。

    支持 mean、median、min、max、std，总体标准差 ddof=0；含缺值的窗口未知。
    """
    operations: dict[str, Any] = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": np.std,
    }
    if type(window) is not int or window < 1 or operation not in operations:
        raise ValueError("invalid_rolling_contract")
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        result[window - 1 :] = operations[operation](windows, axis=1)
    return result


def ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """只对正有限分母求比值；[2,2]/[1,0] 输出 [2,nan]。

    非有限分子或非正分母保持未知，避免零带宽和零成交额制造有效信号。
    """
    return cast(
        FloatArray,
        np.divide(
            numerator,
            denominator,
            out=np.full(len(numerator), np.nan),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0),
        ),
    )


def kdj(high: FloatArray, low: FloatArray, close: FloatArray) -> tuple[FloatArray, ...]:
    """计算中国常用 KDJ(9,3,3)，K/D初值50，J=3K-2D，不裁剪J。

    首个完整九日窗口 RSV=100 时 K=66.6667、D=55.5556、J=88.8889。
    零振幅或缺失窗口保持未知，并重置递推；不以TA-Lib SMA慢随机指标替代。
    """
    stochastic = 100 * ratio(
        close - rolling(low, 9, "min"), rolling(high, 9, "max") - rolling(low, 9, "min")
    )
    fast = np.full(len(close), np.nan)
    slow = np.full(len(close), np.nan)
    previous_fast = previous_slow = 50.0
    for index, value in enumerate(stochastic):
        if not np.isfinite(value):
            previous_fast = previous_slow = 50.0
            continue
        previous_fast = 2 / 3 * previous_fast + value / 3
        previous_slow = 2 / 3 * previous_slow + previous_fast / 3
        fast[index], slow[index] = previous_fast, previous_slow
    return fast, slow, 3 * fast - 2 * slow


def compute_features(
    rows: Sequence[Mapping[str, Any]],
    calendar: Sequence[str],
) -> dict[str, FloatArray]:
    """把唯一源日线映射到完整市场日历，计算只读历史的特征和原始成交输入。

    例如除权价10变5、因子1变2，复权收盘连续为10；原始open仍为5。
    价格以第一根合格行情因子固定锚定，完整序列与任意历史前缀同路径。
    停牌不参与指标递推，映射日的全部技术特征为空；时间距离按市场日计算。
    输入calendar为唯一递增YYYYMMDD，日历外未来行忽略；同日重复拒绝。
    """
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError("calendar_must_be_unique_sorted_nonempty")
    by_date = {
        str(row["trade_date"]): row for row in rows if str(row["trade_date"]) <= calendar[-1]
    }
    if len(by_date) != sum(str(row["trade_date"]) <= calendar[-1] for row in rows):
        raise ValueError("duplicate_quote_date")
    source_fields = (
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_factor",
        "up_limit",
        "down_limit",
        "pct_chg",
    )
    output = {"raw_" + name: np.full(len(calendar), np.nan) for name in source_fields}
    for index, day in enumerate(calendar):
        source = by_date.get(day, {})
        for name in source_fields:
            value = source.get(name)
            if value is not None and type(value) in (float, int) and np.isfinite(value):
                output["raw_" + name][index] = value
    valid = np.ones(len(calendar), dtype=bool)
    for name in source_fields[:7]:
        valid &= np.isfinite(output["raw_" + name]) & (output["raw_" + name] > 0)
    valid &= output["raw_low"] <= np.minimum(output["raw_open"], output["raw_close"])
    valid &= output["raw_high"] >= np.maximum(output["raw_open"], output["raw_close"])
    valid &= np.asarray(
        [
            by_date.get(day, {}).get("tradable", True) is True
            and not by_date.get(day, {}).get("suspended", False)
            for day in calendar
        ]
    )
    output["valid"] = valid.astype(float)
    positions = np.flatnonzero(valid)
    technical_names = (
        "open",
        "high",
        "low",
        "close",
        "ma20",
        "ma60",
        "atr",
        "dif",
        "histogram",
        "normalized_histogram",
        "boll_position",
        "boll_width",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "rsi6",
        "rsi14",
        "return1",
        "return5",
        "return20",
        "amount_ratio",
        "contraction5",
        "close_location",
        "extension_atr",
        "previous_high20",
        "previous_low10",
        "above_ma20_age",
        "boll_width_reference",
        "turnover",
        "limit_up",
    )
    output.update({name: np.full(len(calendar), np.nan) for name in technical_names})
    if not len(positions):
        return output
    factor = output["raw_adj_factor"][valid]
    adjusted = {
        name: output["raw_" + name][valid] * factor / factor[0]
        for name in ("open", "high", "low", "close")
    }
    close, high, low = adjusted["close"], adjusted["high"], adjusted["low"]
    average20, average60 = talib.SMA(close, 20), talib.SMA(close, 60)
    deviation = rolling(close, 20, "std")
    average_true_range = talib.ATR(high, low, close, 14)
    differences, signal, _ = talib.MACD(close, 12, 26, 9)
    fast, slow, oscillator = kdj(high, low, close)
    amount = output["raw_amount"][valid]
    prior_amount = lag(rolling(amount, 20, "median"))
    width = 4 * ratio(deviation, average20)
    above_age = np.full(len(close), np.nan)
    last_below_position: int | None = None
    for index, market_position in enumerate(positions):
        if np.isfinite(average20[index]):
            if close[index] <= average20[index]:
                last_below_position = int(market_position)
            if last_below_position is not None:
                above_age[index] = market_position - last_below_position
    technical = {
        **adjusted,
        "ma20": average20,
        "ma60": average60,
        "atr": average_true_range,
        "dif": differences,
        "histogram": 2 * (differences - signal),
        "normalized_histogram": 2 * ratio(differences - signal, close),
        "boll_position": ratio(close - average20 + 2 * deviation, 4 * deviation),
        "boll_width": width,
        "kdj_k": fast,
        "kdj_d": slow,
        "kdj_j": oscillator,
        "rsi6": talib.RSI(close, 6),
        "rsi14": talib.RSI(close, 14),
        "return1": ratio(close, lag(close)) - 1,
        "return5": ratio(close, lag(close, 5)) - 1,
        "return20": ratio(close, lag(close, 20)) - 1,
        "amount_ratio": ratio(amount, prior_amount),
        "contraction5": ratio(lag(rolling(amount, 5, "mean")), prior_amount),
        "close_location": ratio(close - low, high - low),
        "extension_atr": ratio(close - average20, average_true_range),
        "previous_high20": lag(rolling(high, 20, "max")),
        "previous_low10": lag(rolling(low, 10, "min")),
        "above_ma20_age": above_age,
        "boll_width_reference": lag(rolling(width, 60, "median")),
        "turnover": np.asarray(
            [by_date[calendar[i]].get("turnover_rate", np.nan) for i in positions], dtype=float
        ),
        "limit_up": np.where(
            np.isfinite(output["raw_up_limit"][valid]),
            (output["raw_close"][valid] >= output["raw_up_limit"][valid] - 1e-8).astype(float),
            np.nan,
        ),
    }
    for name, values in technical.items():
        output[name][valid] = values
    return output
