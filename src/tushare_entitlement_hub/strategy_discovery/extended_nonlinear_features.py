"""六项连续量价时空增量特征；严格使用当前及此前完整市场日窗口。"""

from collections.abc import Mapping
from typing import cast

import numpy as np

from ..strategy_optimization.features import FloatArray
from .event_windows import moving_average, shift_matrix
from .windows import complete_history

EXTENDED_FEATURE_NAMES = (
    "relative_turnover20",
    "atr_expansion60",
    "momentum60",
    "price_support_spread",
    "volume_support_spread",
    "sector_breakout_diffusion5",
)
REQUIRED_FEATURE_NAMES = (
    "valid",
    "close",
    "raw_adj_factor",
    "turnover",
    "atr",
    "raw_vol",
    "previous_high20",
    "sector_index",
)


def _finite_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """只对有限分子和正有限分母求同形比值；例[2,2]/[1,0]得到[2,NaN]。"""
    return cast(
        FloatArray,
        np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0),
        ),
    )


def support_spread(values: FloatArray) -> FloatArray:
    """返回5/10/20市场日均线中最弱的相对间距，缺失窗口不前填。

    输入恒定80×4矩阵，首19日未知，之后均为0；均线5>10>20时输出正值。
    调用方预先把不合格观察置NaN；函数不把排列幅度解释为已完成三次金叉。
    """
    short = moving_average(values, 5)
    middle = moving_average(values, 10)
    long = moving_average(values, 20)
    return np.minimum(_finite_ratio(short, middle) - 1, _finite_ratio(middle, long) - 1)


def sector_breakout_diffusion(
    features: Mapping[str, FloatArray],
    quote_known: np.ndarray,
) -> FloatArray:
    """重算剔除自身的行业突破广度，再求当前与五市场日前的差。

    分子为收盘高于此前20日高点的合格同业数；分母仅计高点已知且有21个连续
    有效行情的同业。至少3个合格同业且覆盖所有已分配同业60%，否则NaN。
    例四股同业都合格、某股以外三股由零突破变一股突破，该股五日变化为1/3。
    六日广度均须已知，本股六日行业不变；新增未来行业不会改变过去结果。
    """
    membership = features["sector_index"]
    previous_high = features["previous_high20"]
    known_membership = (
        np.isfinite(membership) & (membership >= 0) & (membership == np.floor(membership))
    )
    eligible = complete_history(quote_known, 20) & np.isfinite(previous_high) & (previous_high > 0)
    breakout = eligible & (features["close"] > previous_high)
    breadth = np.full(membership.shape, np.nan)
    for day, membership_row in enumerate(membership):
        securities = np.flatnonzero(known_membership[day])
        if not len(securities):
            continue
        _, inverse, totals = np.unique(
            membership_row[securities],
            return_inverse=True,
            return_counts=True,
        )
        own_eligible = eligible[day, securities].astype(np.int64)
        own_breakout = breakout[day, securities].astype(np.int64)
        observed = np.bincount(inverse, weights=own_eligible)[inverse] - own_eligible
        winners = np.bincount(inverse, weights=own_breakout)[inverse] - own_breakout
        peers = totals[inverse] - 1
        enough = (observed >= 3) & (observed >= 0.6 * peers)
        breadth[day, securities] = np.divide(
            winners,
            observed,
            out=np.full(len(securities), np.nan),
            where=enough,
        )
    stable = complete_history(membership == shift_matrix(membership), 4)
    complete = complete_history(np.isfinite(breadth) & quote_known, 5)
    return np.where(stable & complete, breadth - shift_matrix(breadth, 5), np.nan)


def extended_nonlinear_features(features: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
    """构建六个同形日×证券矩阵，输入可包含额外旧字段，但不会修改输入。

    必须包含REQUIRED_FEATURE_NAMES中的实数二维同形数组。例80×4恒定正行情
    的相对换手和ATR比例在第60日后为1，60日动量和两种均线间距为0。
    不足窗口、停牌、未知价格或行业均保留NaN；量托另要求20日复权因子不变。
    ATR沿用来源的14根有效行情递推，本函数的60日比较严格采用市场日历。
    """
    if not set(REQUIRED_FEATURE_NAMES) <= features.keys():
        raise ValueError("missing_extended_feature_inputs")
    shape = features["close"].shape
    if (
        len(shape) != 2
        or not all(shape)
        or any(
            values.shape != shape
            or not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.complexfloating)
            for name in REQUIRED_FEATURE_NAMES
            for values in (features[name],)
        )
    ):
        raise ValueError("invalid_extended_feature_inputs")
    close, factor = features["close"], features["raw_adj_factor"]
    quote_known = (
        (features["valid"] == 1)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(factor)
        & (factor > 0)
    )
    clean_close = np.where(quote_known, close, np.nan)
    turnover = features["turnover"]
    turnover_known = quote_known & np.isfinite(turnover) & (turnover >= 0)
    clean_turnover = np.where(turnover_known, turnover, np.nan)
    relative_turnover = _finite_ratio(
        clean_turnover,
        shift_matrix(moving_average(clean_turnover, 20)),
    )
    relative_turnover[~complete_history(turnover_known, 20)] = np.nan
    atr = features["atr"]
    atr_known = quote_known & np.isfinite(atr) & (atr > 0)
    atr_fraction = np.where(atr_known, _finite_ratio(atr, clean_close), np.nan)
    atr_expansion = _finite_ratio(
        atr_fraction,
        shift_matrix(moving_average(atr_fraction, 60)),
    )
    atr_expansion[~complete_history(atr_known, 60)] = np.nan
    momentum = _finite_ratio(clean_close, shift_matrix(clean_close, 60)) - 1
    momentum[~complete_history(quote_known, 60)] = np.nan
    volume = features["raw_vol"]
    volume_known = quote_known & np.isfinite(volume) & (volume > 0)
    volume_spread = support_spread(np.where(volume_known, volume, np.nan))
    same_factor = complete_history(factor == shift_matrix(factor), 18)
    volume_spread[~same_factor] = np.nan
    return dict(
        zip(
            EXTENDED_FEATURE_NAMES,
            (
                relative_turnover,
                atr_expansion,
                momentum,
                support_spread(clean_close),
                volume_spread,
                sector_breakout_diffusion(features, quote_known),
            ),
            strict=True,
        )
    )
