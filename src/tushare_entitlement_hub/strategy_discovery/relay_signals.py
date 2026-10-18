"""冻结五日前行业参与身份的排序消融，四组共用候选，不删弱优先级股票。"""

from dataclasses import replace

import numpy as np

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .hierarchy_signals import BASE_NAME, _stable_membership_history
from .onset_structure import build_strategies as build_structure_strategies
from .sector_onset import REQUIRED_FEATURES as REQUIRED_FEATURES


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """返回原评分、旧参与优先、新参与优先及去掉先后身份的当日参与优先四组。

    输入为原行业初期面板；例如t-5收盘≤MA20且t收盘>MA20，则新参与组优先，
    但旧参与组仍保留该候选。共同entry是onset_structure_gap_2pct原entry再交
    t-5至t每日同一合法L1、行情和MA20有效的共同域；A→B→A不能伪造身份。
    旧参与身份是t-5严格在MA20上方，当日组仅看t严格在上方，新参与是两者之差；
    优先身份加2，原[0,1]低波/低换手分数作次级，等值保留等分，不以代码打破
    经济意义上的并列。未优先股票不删，四组entry完全相同。所有退出、撤单、
    资金政策、追价和成交时点沿用基准；没有等待未来窗口或回填过去领涨者。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("relay_features_missing_or_misaligned")
    base = next(
        item for item in build_structure_strategies(panel, features) if item.name == BASE_NAME
    )
    close, average = features["close"], features["ma20"]
    known = (
        (features["valid"] == 1)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(average)
        & (average > 0)
    )
    common = _stable_membership_history(known, features["sector_index"], 5)
    matched_entry = base.entry_mask & common
    above = close > average
    previous_above = np.zeros(shape, dtype=bool)
    if shape[0] > 5:
        previous_above[5:] = above[:-5]
    identities = (
        ("previous_leaders", previous_above),
        ("new_participants", ~previous_above & above),
        ("current_strength", above),
    )
    definitions = [replace(base, name="relay_matched_control", entry_mask=matched_entry)]
    for label, preferred in identities:
        scores = np.where(np.isfinite(base.scores), base.scores + 2 * preferred, np.nan)
        definitions.append(
            replace(base, name=f"relay_{label}", entry_mask=matched_entry, scores=scores)
        )
    return definitions
