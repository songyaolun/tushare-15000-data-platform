"""原放量反转的选股与收盘确认对照，退出参数和资金政策保持固定。"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray, rolling
from .contracts import StrategyDefinition
from .registry import control_candidates
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "close",
    "high",
    "return1",
    "return20",
    "turnover",
    "boll_position",
    "kdj_j",
    "sector_heat",
    "sector_breadth",
)
SOURCE_SIGNAL = "volume_reversal__sector_group"


def _previous(values: FloatArray) -> FloatArray:
    """沿市场日轴滞后一日，首日未知且不循环；例[[1],[2]]输出[[nan],[1]]。"""
    result = np.full(values.shape, np.nan)
    result[1:] = values[:-1]
    return result


def _historical_risk(features: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray]:
    """计算20市场日收益总体标准差与平均换手率，不把停牌前后收益当普通单日。

    例21日连续有效且收益交替±1%，索引20的波动为1%；20日换手均为2则均值2。
    波动要求21日连续有效正报价；换手要求20日完整非负值。缺值不填零。
    """
    close = features["close"]
    valid = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    volatility, turnover = np.full(close.shape, np.nan), np.full(close.shape, np.nan)
    for column in range(close.shape[1]):
        returns = features["return1"][:, column]
        known_returns = np.isfinite(returns) & (returns > -1)
        volatility[:, column] = rolling(np.where(known_returns, returns, np.nan), 20, "std")
        source_turnover = features["turnover"][:, column]
        known_turnover = valid[:, column] & np.isfinite(source_turnover) & (source_turnover >= 0)
        turnover[:, column] = rolling(np.where(known_turnover, source_turnover, np.nan), 20, "mean")
    volatility[~complete_history(valid, 20)] = np.nan
    return volatility, turnover


def _first_confirmation(
    source: NDArray[np.bool_],
    close: FloatArray,
    high: FloatArray,
    valid: NDArray[np.bool_],
    *,
    breakout: bool,
) -> NDArray[np.bool_]:
    """从原信号后第1至3市场日寻找第一次收盘确认，停牌/缺价取消待确认事件。

    例源日close=10/high=11，随后close=10.5/11.2，上涨版首日确认、突破版次日确认；
    breakout=False同时要求超过源收盘且高于昨收，True要求超过源最高价。
    待确认时忽略重叠源信号，确认当日不重新武装同证券；超时后新信号可重新开始。
    只输出收盘候选，下一合格开盘由共用账本成交，不引用确认日未来开盘。
    """
    result = np.zeros(source.shape, dtype=bool)
    active = np.zeros(source.shape[1], dtype=bool)
    source_day = np.full(source.shape[1], -1, dtype=int)
    reference = np.full(source.shape[1], np.nan)
    previous_close = _previous(close)
    for day in range(len(close)):
        active &= valid[day] & (day - source_day <= 3)
        confirmation = active & (close[day] > reference)
        if not breakout:
            confirmation &= np.isfinite(previous_close[day]) & (close[day] > previous_close[day])
        result[day] = confirmation
        active &= ~confirmation
        observation = high[day] if breakout else close[day]
        arm = (
            source[day]
            & valid[day]
            & ~active
            & ~confirmation
            & np.isfinite(observation)
            & (observation >= close[day])
        )
        reference[arm] = observation[arm]
        source_day[arm] = day
        active |= arm
    return result


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """返回原控制加八选股、八择时共17个预声明配置，固定TP4%/SL10%/持有20日。

    例volume_timing_gap_1pct保留原收盘信号/评分，仅限制下一开盘相对信号收盘追价1%。
    选股：20日低波动、20日低换手、行业热度靠近.6、原分/波动（下限.005）排序；
    四个单项过滤分别为热度<=.85、20日涨幅<=20%、20日跌幅不低于-20%、BOLL[0,.6]。
    择时：延后一市场日、三日内首次上涨/突破源高、追价0/1/2%、当日J上升、BOLL由
    <.2回收至>=.2。延后时使用确认日原评分，保留原行业群动源信号但不额外要求重复群动。
    所有迟确认均在收盘发出，执行器于之后合格开盘交易。选股和择时不更改止盈止损、
    仓位、收盘退出模型，也不保证入场频率。原控制保持原始mask及NaN回退评分，
    与既有control_volume一致；其可交易性仍由共同执行器过滤。缺字段/错维度拒绝。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("selection_timing_features_missing_or_misaligned")
    if SOURCE_SIGNAL not in panel.signals or panel.signals[SOURCE_SIGNAL].shape != shape:
        raise ValueError("selection_timing_source_signal_missing_or_misaligned")
    if any(
        name not in panel.values or panel.values[name].shape != shape
        for name in ("sector_heat", "sector_breadth", "kdj_j")
    ):
        raise ValueError("selection_timing_control_values_missing_or_misaligned")
    control = control_candidates(panel, features)[0]
    original_mask = control.entry_mask
    close = features["close"]
    valid = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    source = original_mask & valid
    heat, breadth, stochastic = (
        features[name] for name in ("sector_heat", "sector_breadth", "kdj_j")
    )
    original_scores = control.scores
    volatility, turnover = _historical_risk(features)
    moderate_scores = np.where(
        np.isfinite(heat) & np.isfinite(breadth) & np.isfinite(stochastic),
        -np.abs(heat - 0.6) + breadth - stochastic / 200,
        np.nan,
    )
    risk_scores = original_scores / np.maximum(volatility, 0.005)
    returns = features["return20"]
    known_return = np.isfinite(returns) & (returns > -1) & complete_history(valid, 20)
    boll = features["boll_position"]
    previous_valid = _previous(valid.astype(float)) == 1
    previous_boll, previous_stochastic = _previous(boll), _previous(stochastic)
    variants: list[tuple[str, NDArray[np.bool_], FloatArray, float | None]] = [
        ("selection_control", original_mask, original_scores, None),
        ("selection_low_vol20", source & np.isfinite(volatility), -volatility, None),
        ("selection_low_turnover20", source & np.isfinite(turnover), -turnover, None),
        ("selection_sector_moderate", source & np.isfinite(moderate_scores), moderate_scores, None),
        ("selection_risk_adjusted", source & np.isfinite(risk_scores), risk_scores, None),
        ("selection_heat_cap", source & np.isfinite(heat) & (heat <= 0.85), original_scores, None),
        ("selection_no_runup", source & known_return & (returns <= 0.2), original_scores, None),
        ("selection_no_collapse", source & known_return & (returns >= -0.2), original_scores, None),
        (
            "selection_boll_lower",
            source & np.isfinite(boll) & (boll >= 0) & (boll <= 0.6),
            original_scores,
            None,
        ),
        ("timing_delay1", (_previous(source.astype(float)) == 1) & valid, original_scores, None),
        (
            "timing_up_within3",
            _first_confirmation(source, close, features["high"], valid, breakout=False),
            original_scores,
            None,
        ),
        (
            "timing_high_within3",
            _first_confirmation(source, close, features["high"], valid, breakout=True),
            original_scores,
            None,
        ),
        ("timing_gap_0pct", source, original_scores, 0.0),
        ("timing_gap_1pct", source, original_scores, 0.01),
        ("timing_gap_2pct", source, original_scores, 0.02),
        (
            "timing_kdj_rising",
            source
            & previous_valid
            & np.isfinite(stochastic)
            & np.isfinite(previous_stochastic)
            & (stochastic > previous_stochastic),
            original_scores,
            None,
        ),
        (
            "timing_boll_reclaim",
            source
            & previous_valid
            & np.isfinite(boll)
            & np.isfinite(previous_boll)
            & (previous_boll < 0.2)
            & (boll >= 0.2),
            original_scores,
            None,
        ),
    ]
    definitions = []
    for suffix, entry_mask, scores, maximum_entry_gap in variants:
        name = "volume_" + suffix
        definitions.append(
            StrategyDefinition(
                name=name,
                family="volume_timing" if suffix.startswith("timing_") else "volume_selection",
                entry_mask=entry_mask,
                scores=scores,
                policy=replace(control.policy, name=name),
                maximum_entry_gap=maximum_entry_gap,
            )
        )
    return definitions
