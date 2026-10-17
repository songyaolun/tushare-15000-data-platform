"""条件独立与事件确认消融的离线回归。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.signal_research.filters import (
    DEFAULT_FILTERS,
    TECHNICAL_FILTER_NAMES,
    FilterDefinition,
    TruthValue,
    evaluate_filter_series,
    evaluate_filters,
)
from tushare_entitlement_hub.signal_research.indicators import (
    IndicatorPoint,
    StockCharacteristics,
)
from tushare_entitlement_hub.signal_research.signals import (
    ExperimentSwitches,
    confirm_signals,
    detect_macd_events,
)


def point(index: int, histogram: float | None = .003, **changes: float) -> IndicatorPoint:
    """生成全部技术条件通过的点；输入index=1输出日期00000001，允许覆盖原始指标。"""
    return replace(IndicatorPoint(
        f"{index:08d}", 10, 9, 8, 8, 7, 1, .5, .03, histogram, 1,
        StockCharacteristics(None, None, None, None, None, None, None, None),
    ), **changes)


def test_independent_filter_truth_and_unknown() -> None:
    """MA失败不影响DIF成功，缺失板块财务保持未知，注入原始值可独立通过。"""
    outcomes = evaluate_filters(point(0, ma20=11, dif=-1))
    assert outcomes["close_above_ma20_above_ma60"] is TruthValue.FALSE
    assert outcomes["close_above_ma20"] is TruthValue.FALSE
    assert outcomes["ma20_above_ma60"] is TruthValue.TRUE
    assert "close_above_ma20_above_ma60" not in TECHNICAL_FILTER_NAMES
    assert len(TECHNICAL_FILTER_NAMES) == 6
    assert outcomes["ma20_rising"] is TruthValue.TRUE
    assert outcomes["dif_positive"] is TruthValue.FALSE
    assert outcomes["histogram_positive"] is TruthValue.TRUE
    assert outcomes["fundamentals"] is TruthValue.UNKNOWN
    assert outcomes["sector"] is TruthValue.UNKNOWN
    assert evaluate_filters(point(0), {"fundamentals": 1})["fundamentals"] is TruthValue.TRUE
    assert evaluate_filters(point(0), {"sector": 0})["sector"] is TruthValue.FALSE
    assert evaluate_filters(point(0), {"sector": True})["sector"] is TruthValue.UNKNOWN
    assert evaluate_filters(point(0), {"sector": float("nan")})["sector"] is TruthValue.UNKNOWN
    with pytest.raises(ValueError, match="覆盖"):
        evaluate_filters(point(0), {"dif": 10})
    with pytest.raises(ValueError, match="唯一"):
        evaluate_filters(point(0), definitions=(DEFAULT_FILTERS[0], DEFAULT_FILTERS[0]))
    with pytest.raises(ValueError, match="布尔"):
        evaluate_filters(point(0), {"dif_positive": 1})
    with pytest.raises(ValueError, match="其他规则"):
        evaluate_filters(point(0), definitions=(
            DEFAULT_FILTERS[0],
            FilterDefinition("dependent", (DEFAULT_FILTERS[0].name,), DEFAULT_FILTERS[0].predicate),
        ))


def test_custom_rules_receive_only_declared_features() -> None:
    """自定义条件只能收到声明的原始特征元组，未知输入不会调用判断函数。"""
    def greater_than_two(values: tuple[float, ...]) -> bool:
        """独立检查外部原始特征；输入(3,)输出True。"""
        assert values == (3.0,)
        return values[0] > 2

    rule = FilterDefinition("custom", ("external",), greater_than_two)
    assert evaluate_filters(point(0), {"external": 3}, (rule,))["custom"] is TruthValue.TRUE
    assert evaluate_filters(point(0), definitions=(rule,))["custom"] is TruthValue.UNKNOWN


def test_event_expiry_drop_and_unknown_crossing() -> None:
    """阈值穿越独立于DIF，T至T+3有效，跌回结束，未知前值不产生假事件。"""
    points = tuple(point(index, value, dif=-1) for index, value in enumerate(
        [.001, .003, .004, .005, .006, .007, .002, .004, .002]
    ))
    events = detect_macd_events(points)
    assert events[0].active_dates == tuple(f"{index:08d}" for index in range(1, 5))
    assert events[1].active_dates == ("00000007",)
    assert detect_macd_events((point(0, None), point(1))) == ()
    assert detect_macd_events((point(0, float("nan")), point(1))) == ()
    unknown_window = detect_macd_events((point(0, .001), point(1), point(2, None), point(3)))
    assert unknown_window[0].active_dates == ("00000001", "00000003")


def test_experiments_confirm_independently_and_full_calendar_without_events() -> None:
    """DIF在T确认不锁定另一个实验，MA在T+2自行确认，无事件实验覆盖全日历。"""
    points = (point(0, .001), point(1, ma20=11), point(2, ma20=11), point(3))
    experiments = (
        ExperimentSwitches("dif", ("dif_positive",)),
        ExperimentSwitches("ma", ("close_above_ma20_above_ma60",)),
        ExperimentSwitches("complete", ("fundamentals", "sector")),
        ExperimentSwitches("calendar", (), False),
    )
    signals = confirm_signals(points, experiments, ts_code="000001.SZ")
    assert [(signal.experiment_id, signal.confirmation_date) for signal in signals] == [
        ("dif", "00000001"), ("ma", "00000003"),
        ("calendar", "00000000"), ("calendar", "00000001"),
        ("calendar", "00000002"), ("calendar", "00000003"),
    ]
    assert signals[0].ts_code == "000001.SZ"
    assert signals[0].event_start_date == "00000001"
    assert signals[-1].event_id is None
    injected = {"00000002": {"fundamentals": 1, "sector": 1}}
    assert confirm_signals(points, (experiments[2],), injected)[0].confirmation_date == "00000002"


def test_signal_contract_validation() -> None:
    """乱序日期、错误窗口、错误阈值、未定义条件和重名实验拒绝。"""
    with pytest.raises(ValueError, match="递增"):
        detect_macd_events((point(1), point(0)))
    with pytest.raises(ValueError, match="阈值"):
        detect_macd_events((), float("nan"))
    with pytest.raises(ValueError, match="整数"):
        detect_macd_events((), confirmation_days=-1)
    with pytest.raises(ValueError, match="未定义"):
        confirm_signals((), (ExperimentSwitches("x", ("absent",)),))
    with pytest.raises(ValueError, match="唯一"):
        confirm_signals((), (ExperimentSwitches("x", ()), ExperimentSwitches("x", ())))


def test_shared_raw_condition_matrix_is_identical_and_validated() -> None:
    """共享逐日原始矩阵与独立求值输出一致，日历或三态不完整及双来源输入拒绝。"""
    points = (point(0, .001), point(1), point(2, dif=-1))
    experiments = (ExperimentSwitches("dif", ("dif_positive",)),
                   ExperimentSwitches("calendar", ("dif_positive",), False))
    matrix = evaluate_filter_series(points)
    assert matrix == {item.trade_date: evaluate_filters(item) for item in points}
    assert confirm_signals(points, experiments) == confirm_signals(
        points, experiments, precomputed_results_by_date=matrix,
    )
    with pytest.raises(ValueError, match="同时注入"):
        confirm_signals(points, experiments, {}, precomputed_results_by_date=matrix)
    with pytest.raises(ValueError, match="全部输入"):
        confirm_signals(points, experiments, precomputed_results_by_date={})
    with pytest.raises(ValueError, match="三态"):
        confirm_signals(points, experiments, precomputed_results_by_date={
            date: {} for date in matrix
        })
