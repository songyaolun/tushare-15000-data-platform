"""行业初期胜者的十三组单因素质量/择时对照，不更改原退出、评分和资金政策。"""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .sector import _sector_features
from .sector_onset import REQUIRED_FEATURES as REQUIRED_FEATURES
from .sector_onset import build_strategies as build_sector_strategies
from .windows import complete_history

BASE_NAME = "sector_onset_selective_quiet_risk_tp8_sl8_h30"


def _lag_market(values: FloatArray, periods: int) -> FloatArray:
    """沿市场日轴向后移动正整数日，前端未知，不压缩停牌或从尾部循环。

    例如[[1],[2],[3]]与periods=2输出[[nan],[nan],[1]]；空矩阵保持原形状。
    """
    if values.ndim != 2 or type(periods) is not int or periods < 1:
        raise ValueError("invalid_onset_quality_lag")
    result = np.full(values.shape, np.nan)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def _delayed_confirmation(
    source: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    membership: FloatArray,
    breadth: FloatArray,
    periods: int,
) -> NDArray[np.bool_]:
    """在源事件后恰好第1/2市场日确认，期间任一失效都会取消，不等候未来恢复。

    例如第0日source=True且三日行业均1、广度均.7、valid均True，periods=2只在
    第2日输出True；第1日换行业、停牌、退出或广度<.6则第2日仍False。
    调用方source必须已扣除原exit_mask，valid包含当前技术有效及无退出信号。
    """
    if type(periods) is not int or periods not in (1, 2):
        raise ValueError("invalid_onset_quality_confirmation_periods")
    result = np.zeros(source.shape, dtype=bool)
    if len(source) <= periods:
        return result
    length = len(source) - periods
    original_sector = membership[:length]
    confirmed = source[:length].copy()
    confirmed &= np.isfinite(original_sector) & (original_sector >= 0)
    confirmed &= original_sector == np.floor(original_sector)
    for offset in range(periods + 1):
        current = slice(offset, offset + length)
        confirmed &= valid[current]
        confirmed &= membership[current] == original_sector
        confirmed &= np.isfinite(breadth[current]) & (breadth[current] >= 0.6)
    result[periods:] = confirmed
    return result


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """返回原胜者控制加十二个单因素变体，均保持TP8%/SL8%/30日及原行业退出。

    例如原事件在第30日且行业持续有效，一日确认只在31日、两日确认只在32日发单；
    收盘候选须由统一执行器在之后合格开盘成交，源日exit=True的事件不能复活。
    八项过滤分别为行业20日>=2%、10日>=1%、MA20较5日前上升、close>=MA20、
    J较昨升、BOLL[.2,.65]、当日市场breadth>=.35/≥.5；另有1/2日广度持续确认
    和1%/2%开盘追价限制。市场广度为当日已知一维收盘值，不是事后市场阶段。
    控制只改名称，其余字段逐一沿用原定义；所有变体共享原评分和policy/exit_mask。
    必需源特征缺失或错维拒绝；额外KDJ/BOLL/市场广度缺失仅关闭其对应过滤变体，
    不影响控制。提供了额外字段却shape错误会拒绝，数值未知不填零。
    """
    shape = len(panel.calendar), len(panel.codes)
    base = next(item for item in build_sector_strategies(panel, features) if item.name == BASE_NAME)
    industry = _sector_features(features)
    extra = {}
    for name in ("kdj_j", "boll_position"):
        values = features.get(name, np.full(shape, np.nan))
        if values.shape != shape:
            raise ValueError("onset_quality_optional_feature_misaligned")
        extra[name] = values
    market_breadth = panel.market.get("breadth", np.full(shape[0], np.nan))
    if market_breadth.shape != (shape[0],):
        raise ValueError("onset_quality_market_breadth_misaligned")
    market_known = np.isfinite(market_breadth) & (market_breadth >= 0) & (market_breadth <= 1)
    close, average = features["close"], features["ma20"]
    valid = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    average_known = valid & np.isfinite(average) & (average > 0)
    previous_average = _lag_market(average, 5)
    previous_oscillator = _lag_market(extra["kdj_j"], 1)
    filters = {
        "sector_m20_ge2pct": np.isfinite(industry["momentum20"]) & (industry["momentum20"] >= 0.02),
        "sector_m10_ge1pct": np.isfinite(industry["momentum10"]) & (industry["momentum10"] >= 0.01),
        "ma20_rising5": complete_history(average_known, 5) & (average > previous_average),
        "close_above_ma20": average_known & (close >= average),
        "kdj_j_rising1": np.isfinite(extra["kdj_j"])
        & np.isfinite(previous_oscillator)
        & (extra["kdj_j"] > previous_oscillator),
        "boll_20_65": np.isfinite(extra["boll_position"])
        & (extra["boll_position"] >= 0.2)
        & (extra["boll_position"] <= 0.65),
        "market_breadth_35": (market_known & (market_breadth >= 0.35))[:, None],
        "market_breadth_50": (market_known & (market_breadth >= 0.5))[:, None],
    }
    definitions = [replace(base, name="onset_quality_control")]
    definitions.extend(
        replace(base, name=f"onset_quality_{name}", entry_mask=base.entry_mask & mask)
        for name, mask in filters.items()
    )
    blocked = base.exit_mask if base.exit_mask is not None else np.zeros(shape, dtype=bool)
    effective_source = base.entry_mask & ~blocked
    continuing = valid & np.isfinite(base.scores) & ~blocked
    for periods in (1, 2):
        definitions.append(
            replace(
                base,
                name=f"onset_quality_confirm_breadth_{periods}d",
                entry_mask=_delayed_confirmation(
                    effective_source,
                    continuing,
                    features["sector_index"],
                    industry["breadth"],
                    periods,
                ),
            )
        )
    for limit in (0.01, 0.02):
        definitions.append(
            replace(
                base, name=f"onset_quality_gap_{round(limit * 100)}pct", maximum_entry_gap=limit
            )
        )
    return definitions
