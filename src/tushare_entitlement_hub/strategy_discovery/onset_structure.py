"""固定市场广度50%加仅行业动量退出基准的九组结构单因素对照。"""

from dataclasses import replace

import numpy as np

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray, lag
from .contracts import StrategyDefinition
from .onset_management import build_variants as build_management_variants
from .onset_quality import REQUIRED_FEATURES as REQUIRED_FEATURES
from .onset_quality import build_strategies as build_quality_strategies
from .sector import _sector_features
from .windows import complete_history

QUALITY_BASE_NAME = "onset_quality_market_breadth_50"
BASE_NAME = f"{QUALITY_BASE_NAME}__manage_momentum20_only"


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成控制加八项结构变体，保持原候选21日连续有效条件和全部持仓/退出政策。

    输入为日×证券特征与市场日历；例如市场广度从五日前.5升至当日.6且六日
    全部已知，则两个扩散过滤均通过，若中日缺失则均拒绝；控制仍沿用原入场。
    七项过滤为市场五日广度增加>0/≥.10、市场广度[.5,.75]、行业10日收益≥.01、
    股票20日收益≥/≤行业20日收益、当日收盘位置≥.6；第八项追价上限为2%。
    收盘位置使用已有同口径复权open/high/low/close，全部有限正值，最高高于最低，
    开收盘均在高低区间内，才计算(close-low)/(high-low)。可选OHLC缺失只关闭
    位置过滤，提供错维字段拒绝；市场广度必须一维，缺失/非法值不填零或跨日续接。
    基准由quality的market_breadth_50再叠加management的momentum20_only构造，
    控制只换name，过滤仅改entry_mask，追价仅改maximum_entry_gap。所有新信号
    都是原收盘候选的子集，无新发单时点，执行器继续负责下一合格开盘成交。
    """
    shape = len(panel.calendar), len(panel.codes)
    market_breadth = panel.market.get("breadth", np.full(shape[0], np.nan))
    if market_breadth.shape != (shape[0],):
        raise ValueError("onset_structure_market_breadth_misaligned")
    optional_prices = {}
    for name in ("open", "high", "low"):
        values = features.get(name, np.full(shape, np.nan))
        if values.shape != shape:
            raise ValueError("onset_structure_optional_price_misaligned")
        optional_prices[name] = values
    quality = next(
        item for item in build_quality_strategies(panel, features) if item.name == QUALITY_BASE_NAME
    )
    base = next(
        item for item in build_management_variants(quality, features) if item.name == BASE_NAME
    )
    industry = _sector_features(features)
    market_known = np.isfinite(market_breadth) & (market_breadth >= 0) & (market_breadth <= 1)
    market_window = complete_history(market_known[:, None], 5)[:, 0]
    previous_breadth = lag(market_breadth, 5)
    stock_return = features["return20"]
    sector_return = industry["momentum20"]
    comparable_returns = np.isfinite(stock_return) & np.isfinite(sector_return)
    opening, high, low = (optional_prices[name] for name in ("open", "high", "low"))
    close = features["close"]
    prices_known = (
        np.isfinite(opening)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (opening > 0)
        & (low > 0)
        & (high > low)
        & (low <= np.minimum(opening, close))
        & (high >= np.maximum(opening, close))
    )
    close_position = np.divide(
        close - low, high - low, out=np.full(shape, np.nan), where=prices_known
    )
    filters = {
        "market_diffusion5_positive": (market_window & (market_breadth > previous_breadth))[
            :, None
        ],
        "market_diffusion5_ge10pct": (market_window & (market_breadth >= previous_breadth + 0.10))[
            :, None
        ],
        "market_breadth_50_75": (market_known & (market_breadth >= 0.5) & (market_breadth <= 0.75))[
            :, None
        ],
        "sector_m10_ge1pct": np.isfinite(industry["momentum10"]) & (industry["momentum10"] >= 0.01),
        "stock_relative_strength_positive": comparable_returns & (stock_return >= sector_return),
        "stock_relative_strength_negative": comparable_returns & (stock_return <= sector_return),
        "close_position_ge60": prices_known & (close_position >= 0.6),
    }
    definitions = [replace(base, name="onset_structure_control")]
    definitions.extend(
        replace(base, name=f"onset_structure_{name}", entry_mask=base.entry_mask & mask)
        for name, mask in filters.items()
    )
    definitions.append(replace(base, name="onset_structure_gap_2pct", maximum_entry_gap=0.02))
    return definitions
