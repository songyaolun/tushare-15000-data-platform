"""可枚举的策略族与背景过滤，全部条件仅读取信号日已计算特征。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .features import FloatArray, lag


@dataclass(frozen=True)
class StrategyDefinition:
    """声明策略族及市场过滤；例：StrategyDefinition('kdj_recovery','sector_group')。"""

    family: str
    context: str = "none"

    @property
    def name(self) -> str:
        """返回稳定可追踪标识；输入 family=a/context=b，输出 a__b。"""
        return self.family + "__" + self.context


FAMILIES = (
    "macd_event",
    "trend_pullback",
    "trend_boll_recovery",
    "kdj_recovery",
    "oversold_recovery",
    "boll_reentry",
    "range_bottom",
    "trend_breakout",
    "volume_reversal",
    "deep_pullback",
    "quiet_pullback",
    "kdj_low_cross",
)
CONTEXTS = ("none", "market_stable", "sector_hot", "sector_group", "sector_hot_group")


def strategy_grid(*, include_concepts: bool = False) -> tuple[StrategyDefinition, ...]:
    """列出十二策略族与五种背景的60组固定搜索候选；输入无，输出唯一标识序列。

    该网格可在全样本上选优，所有尝试均写搜索日志，不声称未见样本外效果。
    """
    contexts = CONTEXTS + (
        ("concept_hot", "concept_group", "sector_or_concept_group") if include_concepts else ()
    )
    return tuple(StrategyDefinition(family, context) for family in FAMILIES for context in contexts)


def strategy_masks(
    features: dict[str, FloatArray],
    definitions: tuple[StrategyDefinition, ...] | None = None,
) -> dict[str, NDArray[np.bool_]]:
    """将原始日指标变为每日候选，停牌与未知不通过；输入零有效报价输出全False。

    连续条件可以连续发出候选，由成交层处理持仓和挂单重叠；不按事后收益删信号。
    P02群动剔除本股、至少五只同业、覆盖90%，不是本股上涨的自证。
    """
    definitions = definitions if definitions is not None else strategy_grid()
    close, average20, average60 = (features[name] for name in ("close", "ma20", "ma60"))
    rising = close > lag(close)
    trend = (average20 > average60) & (average60 >= lag(average60))
    oscillator_rising = features["kdj_j"] > lag(features["kdj_j"])
    stochastic_cross = (features["kdj_k"] > features["kdj_d"]) & (
        lag(features["kdj_k"]) <= lag(features["kdj_d"])
    )
    bounded = features["extension_atr"] <= 1.5
    amount = features["amount_ratio"]
    normalized = features["normalized_histogram"]
    families = {
        "macd_event": (
            (normalized > 0.002)
            & (lag(normalized) <= 0.002)
            & trend
            & (close > average20)
            & (average20 > lag(average20))
            & (features["dif"] > 0)
        ),
        "trend_pullback": (
            trend
            & (close > average60)
            & bounded
            & rising
            & (features["rsi6"] < 50)
            & oscillator_rising
        ),
        "trend_boll_recovery": (
            trend & (lag(features["boll_position"]) < 0.3) & rising & oscillator_rising & bounded
        ),
        "kdj_recovery": (
            oscillator_rising & (lag(features["kdj_j"]) < 20) & (features["kdj_j"] < 60) & rising
        ),
        "oversold_recovery": (
            (features["rsi6"] < 35) & (features["return5"] < -0.04) & rising & oscillator_rising
        ),
        "boll_reentry": (
            (lag(features["boll_position"]) < 0) & (features["boll_position"] > 0) & rising
        ),
        "range_bottom": (
            (features["boll_position"] < 0.15) & (features["kdj_j"] < 20) & (features["rsi6"] < 30)
        ),
        "trend_breakout": (
            trend
            & (close > features["previous_high20"])
            & (amount >= 1.2)
            & (amount <= 3)
            & (features["close_location"] >= 2 / 3)
        ),
        "volume_reversal": (
            (features["return5"] < 0)
            & rising
            & (amount >= 1.2)
            & (amount <= 3)
            & (features["close_location"] >= 0.7)
            & (features["kdj_j"] < 80)
        ),
        "deep_pullback": (
            (features["return5"] < -0.08)
            & (features["rsi6"] < 25)
            & (features["boll_position"] < 0.2)
        ),
        "quiet_pullback": (
            trend
            & (close > average60)
            & (features["return5"] < -0.02)
            & (amount < 1)
            & (features["boll_position"] < 0.4)
        ),
        "kdj_low_cross": stochastic_cross & (features["kdj_k"] < 40) & rising,
    }
    group = (
        (features["sector_peers"] >= 5)
        & (features["sector_coverage"] >= 0.9)
        & (features["sector_breadth"] >= 0.6)
        & (features["sector_mean_return"] > 0)
    )
    hot = (
        (features["sector_heat"] >= 0.8)
        & (features["sector_amount_ratio"] >= 1)
        & (features["sector_trend"] == 1)
    )
    contexts = {
        "none": np.ones(len(close), dtype=bool),
        "market_stable": features["market_breadth"] >= 0.4,
        "sector_hot": hot,
        "sector_group": group,
        "sector_hot_group": hot & group,
    }
    concept_hot = features.get("concept_hot", np.zeros(len(close), dtype=bool))
    concept_group = features.get("concept_group", np.zeros(len(close), dtype=bool))
    contexts.update(
        {
            "concept_hot": concept_hot,
            "concept_group": concept_group,
            "sector_or_concept_group": (hot & group) | concept_group,
        }
    )
    result = {}
    for definition in definitions:
        if definition.family not in families or definition.context not in contexts:
            raise ValueError("unknown_strategy_definition")
        result[definition.name] = (
            families[definition.family] & contexts[definition.context] & (features["valid"] == 1)
        )
    return result


def strategy_manifest() -> list[dict[str, str]]:
    """导出完整搜索定义；输入无，输出包含family/context/name的60条记录。"""
    return [{**asdict(definition), "name": definition.name} for definition in strategy_grid()]
