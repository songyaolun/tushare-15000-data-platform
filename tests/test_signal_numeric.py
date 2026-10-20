"""固定历史锚点和舍入边界的失败优先回归。"""

from dataclasses import replace
from math import sin

from tushare_entitlement_hub.signal_research.filters import TruthValue, evaluate_filters
from tushare_entitlement_hub.signal_research.indicators import PriceBar, compute_indicators
from tushare_entitlement_hub.signal_research.numeric import ratio_greater, strictly_greater
from tushare_entitlement_hub.signal_research.signals import (
    ExperimentSwitches,
    confirm_signals,
    detect_macd_events,
)


def test_first_factor_anchor_and_flat_comparisons() -> None:
    """未来因子不得改变历史计算锚点，平盘微小库舍入不得令严格条件TRUE。"""
    bars = [PriceBar(f"{index:08d}", 10, 11, 9, 10, 1) for index in range(90)]
    bars[-1] = replace(bars[-1], adjustment_factor=2)
    full = compute_indicators(bars, "99999999")
    prefix = compute_indicators(bars[:-1], "99999999")
    assert full[:89] == prefix
    near_equal = replace(prefix[-1], close=10, ma20=10 - 1e-14, ma60=10 - 2e-14,
                         dif=1e-15, histogram=1e-15)
    results = evaluate_filters(near_equal)
    assert results["close_above_ma20"] is TruthValue.FALSE
    assert results["ma20_above_ma60"] is TruthValue.FALSE
    assert results["dif_positive"] is TruthValue.FALSE
    assert results["histogram_positive"] is TruthValue.FALSE


def test_rounding_tolerance_scale_invariance_and_ratio_boundary() -> None:
    """价格尺度改变不改变近等比较；H/C穿越和跌回使用同一绝对舍入边界。"""
    assert strictly_greater(11, 10)
    assert not strictly_greater(10, 10 - 1e-14)
    assert not strictly_greater(1e-15, 0, scale_ref=10)
    assert strictly_greater(.01, 0, scale_ref=10)
    assert ratio_greater(.003, .002)
    assert not ratio_greater(.002 + 1e-14, .002)
    bars = [PriceBar(f"{index:08d}", 10, 11, 9, 10) for index in range(61)]
    template = compute_indicators(bars, "99999999")[-1]
    points = tuple(replace(template, trade_date=f"{index:08d}", normalized_histogram=value)
                   for index, value in enumerate((.001, .002 + 1e-14, .003, .002 + 1e-14)))
    event = detect_macd_events(points)[0]
    assert event.start_index == 2
    assert event.active_dates == ("00000002",)


def test_all_atomic_conditions_and_signals_are_prefix_invariant() -> None:
    """混合平盘、微小变动、未来因子变化时，完整计算与前缀全部原子条件/确认一致。"""
    bars = []
    for index in range(180):
        close = 10 if index < 70 else 10 + .01 * index + .3 * sin(index / 4)
        factor = 1 if index < 140 else 2
        bars.append(PriceBar(f"{index:08d}", close, close + 1, close - 1, close, factor))
    full = compute_indicators(bars, "99999999")
    experiments = (ExperimentSwitches("events", ()), ExperimentSwitches("daily", (), False),
                   ExperimentSwitches("technical", ("ma20_rising", "dif_positive")))
    for cutoff in (65, 100, 139, 160):
        prefix = compute_indicators(bars, bars[cutoff].trade_date)
        assert prefix == full[:cutoff + 1]
        assert [evaluate_filters(point) for point in prefix] == [
            evaluate_filters(point) for point in full[:cutoff + 1]
        ]
        assert confirm_signals(prefix, experiments) == tuple(
            signal for signal in confirm_signals(full, experiments)
            if signal.confirmation_date <= bars[cutoff].trade_date
        )
    point = full[100]
    scaled = replace(point, close=point.close * 100, ma20=point.ma20 * 100,
                     ma60=point.ma60 * 100, previous_ma20=point.previous_ma20 * 100,
                     previous_ma60=point.previous_ma60 * 100, dif=point.dif * 100,
                     histogram=point.histogram * 100)
    assert evaluate_filters(scaled) == evaluate_filters(point)
