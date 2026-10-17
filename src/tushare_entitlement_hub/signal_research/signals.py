"""MACD事件只依赖H/C，消融实验独立选择首次确认日期。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from .filters import DEFAULT_FILTERS, FilterDefinition, TruthValue, evaluate_filter_series
from .indicators import IndicatorPoint
from .numeric import ratio_greater


@dataclass(frozen=True)
class MacdEvent:
    """独立阈值穿越及有效交易日；例：T穿越且T+2跌回时仅T、T+1有效。

    event_id由起始日期生成，不包含实验过滤结果；最多允许T至T+3。
    """

    event_id: str
    start_date: str
    start_index: int
    active_dates: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSwitches:
    """实验独立条件开关；例：enabled_filters=()仅保留事件，event_filter=False遍历全日历。"""

    name: str
    enabled_filters: tuple[str, ...]
    event_filter: bool = True


@dataclass(frozen=True)
class SignalConfirmation:
    """一次实验确认及当日条件证据；例：事件T、确认T+1返回两日期和实验名。"""

    experiment_id: str
    ts_code: str
    confirmation_date: str
    event_id: str | None
    event_start_date: str | None
    filter_results: Mapping[str, TruthValue]


def _validate_points(points: tuple[IndicatorPoint, ...]) -> None:
    """确保输入按唯一交易日期递增；输入乱序日期抛ValueError，空序列允许。"""
    dates = [point.trade_date for point in points]
    if dates != sorted(set(dates)):
        raise ValueError("指标日期必须唯一且递增")


def detect_macd_events(
    points: tuple[IndicatorPoint, ...],
    threshold: float = 0.002,
    confirmation_days: int = 3,
) -> tuple[MacdEvent, ...]:
    """寻找H/C首次从<=threshold穿越到>threshold的独立事件。

    threshold=.002只是试验参数；确认窗口按输入交易日索引T至T+confirmation_days。
    柱跌回阈值或窗口过期结束事件，不读DIF/均线/财务条件。
    例：H/C=[.001,.003,.004,.001]输出事件start_index=1、active_dates为第二三日。
    缺失H/C不构造穿越；事件期间未知日不能确认，但不视为跌回，不延长窗口。
    输入需完整观察交易日序列，否则确认天数只能代表输入交易日而非市场日历。
    """
    _validate_points(points)
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("试验阈值必须为非负有限数")
    if type(confirmation_days) is not int or confirmation_days < 0:
        raise ValueError("确认交易日数量必须为非负整数")
    events = []
    for index in range(1, len(points)):
        previous = points[index - 1].normalized_histogram
        current = points[index].normalized_histogram
        if (previous is None or current is None
                or not isfinite(previous) or not isfinite(current)
                or ratio_greater(previous, threshold) or not ratio_greater(current, threshold)):
            continue
        active_dates = []
        for candidate in points[index : index + confirmation_days + 1]:
            value = candidate.normalized_histogram
            if value is None or not isfinite(value):
                continue
            if not ratio_greater(value, threshold):
                break
            active_dates.append(candidate.trade_date)
        events.append(MacdEvent(
            f"macd:{points[index].trade_date}", points[index].trade_date,
            index, tuple(active_dates),
        ))
    return tuple(events)


def confirm_signals(
    points: tuple[IndicatorPoint, ...],
    experiments: tuple[ExperimentSwitches, ...],
    features_by_date: Mapping[str, Mapping[str, float | None]] | None = None,
    *,
    ts_code: str = "",
    threshold: float = 0.002,
    confirmation_days: int = 3,
    definitions: tuple[FilterDefinition, ...] = DEFAULT_FILTERS,
    precomputed_results_by_date: Mapping[str, Mapping[str, TruthValue]] | None = None,
) -> tuple[SignalConfirmation, ...]:
    """每个实验从同一独立事件自行选择首次全部启用条件TRUE的日期。

    输入交易日指标、实验开关和截至各日已知的注入特征；UNKNOWN不会自动通过。
    输出可供模拟器消费的ts_code/confirmation_date/experiment_id及完整条件证据。
    例：A仅DIF、B仅MA，A在T确认不影响B在T+2确认。
    event_filter=False时逐个完整日历日评估并返回所有通过日期，event_id=None。
    precomputed_results_by_date只接受同一原始指标产生的完整日矩阵，不接受实验布尔。
    features_by_date仅表示调用方已完成PIT证明的当日特征，不读取未来收益或其他实验结果。
    """
    events = detect_macd_events(points, threshold, confirmation_days)
    names = {definition.name for definition in definitions}
    if len({experiment.name for experiment in experiments}) != len(experiments):
        raise ValueError("实验名称必须唯一")
    for experiment in experiments:
        if set(experiment.enabled_filters) - names:
            raise ValueError("实验包含未定义条件")
    if precomputed_results_by_date is None:
        results_by_date = evaluate_filter_series(points, features_by_date, definitions)
    else:
        if features_by_date is not None:
            raise ValueError("预计算矩阵不能同时注入原始特征")
        if set(precomputed_results_by_date) != {point.trade_date for point in points}:
            raise ValueError("预计算矩阵必须覆盖全部输入日历")
        if any(set(results) != names or any(not isinstance(value, TruthValue)
                                          for value in results.values())
               for results in precomputed_results_by_date.values()):
            raise ValueError("预计算矩阵条件名称或三态不完整")
        results_by_date = {date: dict(results)
                           for date, results in precomputed_results_by_date.items()}
    confirmations = []
    for experiment in experiments:
        candidates = [(event, event.active_dates) for event in events] if experiment.event_filter \
            else [(None, tuple(point.trade_date for point in points))]
        for event, dates in candidates:
            for date in dates:
                results = results_by_date[date]
                if all(results[name] is TruthValue.TRUE for name in experiment.enabled_filters):
                    confirmations.append(SignalConfirmation(
                        experiment.name, ts_code, date,
                        event.event_id if event else None, event.start_date if event else None,
                        results,
                    ))
                    if event is not None:
                        break
    return tuple(confirmations)
