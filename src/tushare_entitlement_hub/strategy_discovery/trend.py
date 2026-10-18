"""独立的波动收缩突破与趋势回调；小规模事先定义，不使用反转或市场阶段标签。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray, lag
from .contracts import StrategyDefinition

REQUIRED_FEATURES = (
    "valid",
    "close",
    "low",
    "ma20",
    "ma60",
    "return1",
    "return5",
    "return20",
    "extension_atr",
    "close_location",
    "boll_width",
    "boll_width_reference",
    "previous_high20",
    "previous_low10",
    "sector_peers",
    "sector_coverage",
    "sector_breadth",
)


def _lag_observations(values: FloatArray, valid: NDArray[np.bool_], periods: int) -> FloatArray:
    """逐证券滞后有效行情，停牌当日未知，技术缺值不跨越；只读过去观察。

    输入values列[1,nan,3,4]、valid列[真,假,真,真]和1，输出[nan,nan,1,3]。
    valid只标识行情有效性，不能由技术指标是否有限推导，以免跨越指标缺口。
    """
    result = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        positions = np.flatnonzero(valid[:, column])
        result[positions, column] = lag(values[positions, column], periods)
    return result


def _finite_mask(features: dict[str, FloatArray], names: tuple[str, ...]) -> NDArray[np.bool_]:
    """要求指定技术指标全部有限且当天行情有效；未知值不能成为入场依据。

    输入valid=[1,1]、close=[10,nan]与names=('close',)，输出[真,假]。
    返回日×证券矩阵，不改变传入特征。
    """
    result: NDArray[np.bool_] = features["valid"] == 1
    for name in names:
        result &= np.isfinite(features[name])
    return result


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成16个独立配置：每类两组入场阈值×两种退出×行业过滤开关。

    输入日×证券技术矩阵，输出8个compression_breakout及8个trend_pullback。
    突破要求前一有效日收缩、首次越过20日高点；回调要求上升均线、此前五日
    收益非正、当日触及并收回MA20。行业对照要求至少5名同业、80%覆盖和50%
    上涨家数，不借大盘状态。信号均为收盘已知；执行器次日合格开盘交易。
    止盈给出20%至40%空间，跟踪止盈为8%至12%，最大持有30至60市场日。
    例如缺失行业字段的数值只关闭行业过滤版本，未过滤版本仍可产生信号。
    """
    shape = (len(panel.calendar), len(panel.codes))
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("invalid_trend_feature_matrix")
    valid = features["valid"] == 1
    close, average20, average60 = features["close"], features["ma20"], features["ma60"]
    previous_close = _lag_observations(close, valid, 1)
    previous_high = _lag_observations(features["previous_high20"], valid, 1)
    previous_width = _lag_observations(features["boll_width"], valid, 1)
    previous_reference = _lag_observations(features["boll_width_reference"], valid, 1)
    previous_return5 = _lag_observations(features["return5"], valid, 1)
    historical_average20 = _lag_observations(average20, valid, 5)
    compression = np.divide(
        previous_width,
        previous_reference,
        out=np.full(shape, np.nan),
        where=np.isfinite(previous_width)
        & np.isfinite(previous_reference)
        & (previous_width >= 0)
        & (previous_reference > 0),
    )
    sector_mask = (
        _finite_mask(features, ("sector_peers", "sector_coverage", "sector_breadth"))
        & (features["sector_peers"] >= 5)
        & (features["sector_coverage"] >= 0.8)
        & (features["sector_breadth"] >= 0.5)
    )
    trend_base = (
        _finite_mask(
            features,
            (
                "close",
                "low",
                "ma20",
                "ma60",
                "return1",
                "return20",
                "extension_atr",
                "close_location",
            ),
        )
        & (average60 > 0)
        & (average20 > average60)
        & (close >= average20)
        & (average20 > historical_average20)
        & np.isfinite(previous_return5)
        & (previous_return5 <= 0)
        & (features["return1"] > 0)
        & (features["close_location"] >= 0.6)
    )
    breakout_base = (
        _finite_mask(
            features,
            (
                "close",
                "ma20",
                "ma60",
                "previous_high20",
                "return20",
                "close_location",
                "extension_atr",
            ),
        )
        & (average60 > 0)
        & (average20 >= average60)
        & (close > features["previous_high20"])
        & np.isfinite(previous_close)
        & np.isfinite(previous_high)
        & (previous_close <= previous_high)
        & np.isfinite(compression)
        & (features["extension_atr"] <= 3)
    )
    trend_scores = (
        np.clip(features["return20"], -0.5, 1)
        - 0.1 * np.abs(features["extension_atr"])
        + 0.1 * features["close_location"]
    )
    breakout_scores = (
        0.5 * np.clip(features["return20"], -0.5, 1)
        + 0.3 * features["close_location"]
        + 0.2 * (1 - np.clip(compression, 0, 2))
        - 0.1 * np.clip(features["extension_atr"], 0, 5)
    )
    fast_exit = (
        _finite_mask(features, ("close", "ma20"))
        & (close > 0)
        & (average20 > 0)
        & (close < average20 * 0.98)
    )
    structural_exit = (
        _finite_mask(features, ("close",))
        & (close > 0)
        & (
            (np.isfinite(average60) & (average60 > 0) & (close < average60))
            | (
                np.isfinite(features["previous_low10"])
                & (features["previous_low10"] > 0)
                & (close < features["previous_low10"])
            )
        )
    )
    definitions = []
    for family, prefix, scores in (
        ("compression_breakout", "breakout", breakout_scores),
        ("trend_pullback", "trend", trend_scores),
    ):
        for threshold_name, strict in (("selective", True), ("broad", False)):
            if family == "compression_breakout":
                entry = breakout_base & (compression <= (0.7 if strict else 0.9))
                entry &= features["close_location"] >= (0.75 if strict else 0.6)
            else:
                entry = trend_base & (features["low"] <= average20 * (1.01 if strict else 1.025))
                entry &= features["extension_atr"] <= (1.0 if strict else 1.5)
            for exit_name, stop_loss, take_profit, holding, trailing, exit_mask in (
                ("balanced", 0.07, 0.20, 30, 0.08, fast_exit),
                ("patient", 0.10, 0.40, 60, 0.12, structural_exit),
            ):
                for industry_name, industry_filter in (("all", False), ("sector", True)):
                    name = f"{prefix}_{threshold_name}_{exit_name}_{industry_name}"
                    definitions.append(
                        StrategyDefinition(
                            name=name,
                            family=family,
                            entry_mask=entry & sector_mask if industry_filter else entry,
                            scores=scores,
                            policy=PortfolioPolicy(
                                name=name,
                                switching=False,
                                stage_allocation=False,
                                force_rebalance=False,
                                position_weight=0.05,
                                sector_weight=0.25,
                                take_profit=take_profit,
                                stop_loss=stop_loss,
                                maximum_holding=holding,
                            ),
                            exit_mask=exit_mask,
                            maximum_entry_gap=0.03 if family == "compression_breakout" else 0.02,
                            trailing_stop=trailing,
                        )
                    )
    return definitions
