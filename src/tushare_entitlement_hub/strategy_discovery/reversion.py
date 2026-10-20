"""独立超跌回归候选；二十组预先声明的规则不依赖放量反转信号。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "close",
    "ma20",
    "rsi6",
    "rsi14",
    "boll_position",
    "return1",
    "return5",
    "sector_breadth",
    "sector_coverage",
    "sector_peers",
)


def _previous(values: FloatArray) -> FloatArray:
    """沿交易日轴滞后一日，首日未知且不跨证券读取。

    例输入[[1,2],[3,4]]，输出[[nan,nan],[1,2]]；空矩阵保持原形状。
    """
    previous = np.full(values.shape, np.nan)
    previous[1:] = values[:-1]
    return previous


def _cross_section_percentile(values: FloatArray, eligible: NDArray[np.bool_]) -> FloatArray:
    """计算每个当日有效截面的升序中点分位，同值同排名，不使用未来成员。

    至少五个有效观察才排名；例[1,2,3,4,5]得到[.1,.3,.5,.7,.9]，
    五个相同收益全部为.5；未知或不可交易成员不进入分母，输出NaN。
    """
    percentiles = np.full(values.shape, np.nan)
    for day in range(len(values)):
        columns = np.flatnonzero(eligible[day] & np.isfinite(values[day]))
        if len(columns) < 5:
            continue
        _, inverse, counts = np.unique(
            values[day, columns], return_inverse=True, return_counts=True
        )
        midpoint = (np.cumsum(counts) - counts / 2) / len(columns)
        percentiles[day, columns] = midpoint[inverse]
    return percentiles


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成五种入场×两档超跌门槛×有无行业支持的二十个独立策略。

    例五日收益位于当日最弱20%、跌幅至少3%、当日跌幅不超2%，
    可触发reversion_cross_section_balanced_none，无需放量或KDJ反转。
    balanced/deep使用6%/9%止盈与4%/5%止损，普通回归最多持有8/12日，
    横截面策略为5/8日；回归均线或振荡指标修复也能提前退出。
    行业对照要求至少五只同业、90%覆盖和50%上涨广度。横截面五日收益
    要求当日及此前五市场日连续有效；RSI/BOLL保持原有效行情观察窗口语义。
    信号只读取当日及历史信息；执行器另行过滤可成交性并在下一合格开盘执行。
    字段缺失或维度不符抛ValueError；未知值不触发依赖该字段的信号。
    """
    shape = (len(panel.calendar), len(panel.codes))
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("reversion_features_missing_or_misaligned")
    close = features["close"]
    valid = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    previous_valid = _previous(valid.astype(float)) == 1
    relative_strength = features["rsi6"]
    slow_strength = features["rsi14"]
    boll_position = features["boll_position"]
    previous_strength = _previous(relative_strength)
    previous_boll = _previous(boll_position)
    daily_return = features["return1"]
    weekly_return = features["return5"]
    comparable = complete_history(valid, 5)
    relative_percentile = _cross_section_percentile(weekly_return, comparable)
    sector_support = (
        np.isfinite(features["sector_breadth"])
        & np.isfinite(features["sector_coverage"])
        & np.isfinite(features["sector_peers"])
        & (features["sector_breadth"] >= 0.5)
        & (features["sector_coverage"] >= 0.9)
        & (features["sector_peers"] >= 5)
    )
    mean_reached = (
        np.isfinite(features["ma20"]) & (features["ma20"] > 0) & (close >= features["ma20"])
    )
    strategies = []
    for profile, deep in (("balanced", False), ("deep", True)):
        strength_threshold = 25 if deep else 30
        strength_exit = valid & (
            (np.isfinite(relative_strength) & (relative_strength >= (60 if deep else 55)))
            | mean_reached
        )
        boll_exit = valid & np.isfinite(boll_position) & (boll_position >= 0.5)
        cross_exit = valid & mean_reached
        templates = (
            (
                "rsi_low",
                np.isfinite(relative_strength)
                & np.isfinite(slow_strength)
                & np.isfinite(weekly_return)
                & (relative_strength < (20 if deep else 25))
                & (slow_strength < (35 if deep else 40))
                & (weekly_return < (-0.04 if deep else -0.02)),
                100 - relative_strength,
                strength_exit,
            ),
            (
                "rsi_recovery",
                previous_valid
                & np.isfinite(previous_strength)
                & np.isfinite(relative_strength)
                & np.isfinite(slow_strength)
                & np.isfinite(daily_return)
                & (previous_strength < strength_threshold)
                & (relative_strength >= strength_threshold)
                & (relative_strength <= 45)
                & (slow_strength < 50)
                & (daily_return > 0),
                100 + relative_strength - 2 * previous_strength,
                strength_exit,
            ),
            (
                "boll_deviation",
                np.isfinite(boll_position)
                & np.isfinite(weekly_return)
                & (boll_position < (-0.15 if deep else 0))
                & (weekly_return < (-0.06 if deep else -0.04)),
                -boll_position,
                boll_exit,
            ),
            (
                "boll_reentry",
                previous_valid
                & np.isfinite(previous_boll)
                & np.isfinite(boll_position)
                & np.isfinite(daily_return)
                & (previous_boll < (-0.1 if deep else 0))
                & (boll_position >= 0)
                & (boll_position <= (0.25 if deep else 0.4))
                & (daily_return > 0),
                boll_position - previous_boll,
                boll_exit,
            ),
            (
                "cross_section",
                np.isfinite(relative_percentile)
                & np.isfinite(daily_return)
                & (relative_percentile <= (0.1 if deep else 0.2))
                & (weekly_return < (-0.05 if deep else -0.03))
                & (daily_return >= (-0.01 if deep else -0.02)),
                np.where(comparable, -weekly_return, np.nan),
                cross_exit,
            ),
        )
        for family, entry, scores, exit_mask in templates:
            finite_scores = np.where(valid & np.isfinite(scores), scores, np.nan)
            for context in ("none", "sector"):
                name = f"reversion_{family}_{profile}_{context}"
                entry_mask = valid & entry & np.isfinite(finite_scores)
                if context == "sector":
                    entry_mask &= sector_support
                maximum_holding = (
                    (8 if deep else 5) if family == "cross_section" else (12 if deep else 8)
                )
                strategies.append(
                    StrategyDefinition(
                        name=name,
                        family=family,
                        entry_mask=entry_mask,
                        scores=finite_scores,
                        policy=PortfolioPolicy(
                            name=name,
                            switching=False,
                            stage_allocation=False,
                            force_rebalance=False,
                            allocations=(0.9,) * 5,
                            take_profit=0.09 if deep else 0.06,
                            stop_loss=0.05 if deep else 0.04,
                            maximum_holding=maximum_holding,
                        ),
                        exit_mask=exit_mask,
                        maximum_entry_gap=0.02,
                    )
                )
    return strategies
