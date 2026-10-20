"""TA-Lib指标、观察时点复权和股性缺失语义的离线测试。"""

from dataclasses import replace

import numpy as np
import pytest
import talib

from tushare_entitlement_hub.signal_research.indicators import PriceBar, compute_indicators


def price_bars(count: int = 100) -> list[PriceBar]:
    """生成确定性变动K线；输入100输出100个唯一日期、正价格的离线行情。"""
    return [PriceBar(f"{index:08d}", 10 + index / 10, 11 + index / 10,
                     9 + index / 10, 10 + index / 10 + np.sin(index),
                     1 + index / 100, 2.0, 3.0, 4.0) for index in range(count)]


def test_indicators_match_talib_and_prefix_invariant() -> None:
    """复权OHLC与库输出逐值一致，追加未来因子不会改变历史归一柱和条件。"""
    bars = price_bars()
    points = compute_indicators(bars, "99999999")
    prices = np.asarray([bar.close * bar.adjustment_factor / bars[0].adjustment_factor
                         for bar in bars])
    differences, signal, _ = talib.MACD(prices, 12, 26, 9)
    assert points[-1].dif == pytest.approx(differences[-1])
    assert points[-1].histogram == pytest.approx(2 * (differences[-1] - signal[-1]))
    assert points[0].ma20 is None and points[0].atr14 is None
    assert points[18].ma20 is None and points[19].ma20 is not None
    assert points[59].characteristics.direction_60 is None
    prefix = compute_indicators(bars, bars[70].trade_date)
    assert prefix == compute_indicators(bars[:71], bars[70].trade_date)
    assert prefix[-1].normalized_histogram == pytest.approx(points[70].normalized_histogram)
    assert prefix[-1].characteristics.efficiency_60 == pytest.approx(
        points[70].characteristics.efficiency_60
    )
    assert points[-1].characteristics.market_cap == 4
    assert points[-1].characteristics.amount_median_20 == 3
    assert points[-1].characteristics.turnover_median_20 == 2
    assert points[-1].characteristics.zero_range_count_60 == 0
    assert points[-1].characteristics.wick_ratio_p90_60 is not None
    assert points[-1].characteristics.gap_ratio_60 == 0


def test_observation_adjustment_and_missing_source_features() -> None:
    """观察日只使用可见行情和显式锚点，源字段缺失时中位数保持未知。"""
    bar = PriceBar("20260101", 10, 11, 9, 10, 2)
    points = compute_indicators([bar, replace(bar, trade_date="20260102", close=-1)],
                                "20260101", anchor_factor=4)
    assert points[0].close == 5
    assert points[0].characteristics.market_cap is None
    assert compute_indicators([], "20260101") == ()
    missing = price_bars(61)
    missing[-1] = replace(missing[-1], turnover_rate=None, amount=float("nan"),
                          market_cap=float("nan"))
    result = compute_indicators(missing, "99999999")[-1].characteristics
    assert result.turnover_median_20 is None
    assert result.amount_median_20 is None
    assert result.market_cap is None


def test_flat_and_gap_characteristics() -> None:
    """60收益段平盘效率未知，零幅影线未知并计数，单调跳空效率与跳空比为1。"""
    flat = [PriceBar(f"{index:08d}", 10, 10, 10, 10) for index in range(61)]
    result = compute_indicators(flat, "99999999")[-1].characteristics
    assert result.direction_60 == 0
    assert result.efficiency_60 is None
    assert result.wick_ratio_median_60 is None
    assert result.wick_ratio_p90_60 is None
    assert result.zero_range_count_60 == 60
    gaps = [PriceBar(f"{index:08d}", 10 + 3 * index, 11 + 3 * index,
                    9 + 3 * index, 10 + 3 * index) for index in range(61)]
    result = compute_indicators(gaps, "99999999")[-1].characteristics
    assert result.direction_60 == 18
    assert result.efficiency_60 == 1
    assert result.gap_ratio_60 == 1
    assert result.wick_ratio_median_60 == 1


@pytest.mark.parametrize("invalid", [
    {"close": -1}, {"high": 8}, {"low": 12}, {"adjustment_factor": 0}, {"open": float("nan")},
])
def test_invalid_price_contract(invalid: dict[str, float]) -> None:
    """无效OHLC、区间或因子不得进入库计算，应抛出明确契约异常。"""
    with pytest.raises(ValueError):
        compute_indicators([replace(price_bars(1)[0], **invalid)], "99999999")


def test_duplicate_dates_and_invalid_anchor() -> None:
    """重复日期和非正锚点拒绝，防止错误预热和复权。"""
    bar = price_bars(1)[0]
    with pytest.raises(ValueError, match="唯一"):
        compute_indicators([bar, bar], "99999999")
    with pytest.raises(ValueError, match="锚点"):
        compute_indicators([bar], "99999999", 0)


def test_vectorized_characteristics_match_independent_window_oracle() -> None:
    """混合零幅、缺失源字段和1643根历史时，向量化股性逐日匹配独立窗口定义。"""
    bars = price_bars(1643)
    for index in range(0, len(bars), 17):
        bars[index] = replace(bars[index], open=bars[index].close,
                              high=bars[index].close, low=bars[index].close)
    for index in range(0, len(bars), 23):
        bars[index] = replace(bars[index], amount=None)
    points = compute_indicators(bars, "99999999")
    anchor = bars[0].adjustment_factor
    adjusted = np.asarray([
        [value * bar.adjustment_factor / anchor
         for value in (bar.open, bar.high, bar.low, bar.close)] for bar in bars
    ])
    for index in range(60, len(bars)):
        actual = points[index].characteristics
        window = adjusted[index - 60:index + 1]
        closes = window[:, 3]
        expected_direction = closes[-1] / closes[0] - 1
        expected_efficiency = abs(closes[-1] - closes[0]) / np.sum(abs(np.diff(closes)))
        ranges = window[1:, 1] - window[1:, 2]
        wicks = ranges - abs(window[1:, 3] - window[1:, 0])
        ratios = wicks[ranges > 0] / ranges[ranges > 0]
        assert actual.direction_60 == pytest.approx(expected_direction, abs=1e-12)
        assert actual.efficiency_60 == pytest.approx(expected_efficiency, abs=1e-12)
        assert actual.wick_ratio_median_60 == pytest.approx(np.median(ratios), abs=1e-12)
        assert actual.wick_ratio_p90_60 == pytest.approx(np.percentile(ratios, 90), abs=1e-12)
        assert actual.zero_range_count_60 == int(np.sum(ranges == 0))
        recent_amount = [bar.amount for bar in bars[index - 19:index + 1]]
        if None in recent_amount:
            assert actual.amount_median_20 is None
        else:
            assert actual.amount_median_20 == pytest.approx(np.median(recent_amount))
