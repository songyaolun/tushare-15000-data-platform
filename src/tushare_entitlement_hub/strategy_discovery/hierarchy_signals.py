"""历史L1/L2的共同可得排序对照及独立L2入场事件，持仓管理仍沿用L1。"""

from dataclasses import replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .onset_quality import _lag_market
from .onset_structure import build_strategies as build_structure_strategies
from .sector import _percentile, _sector_features
from .sector_onset import REQUIRED_FEATURES as SOURCE_FEATURES
from .sector_onset import _stock_statistics
from .windows import complete_history

BASE_NAME = "onset_structure_gap_2pct"
REQUIRED_FEATURES = (*SOURCE_FEATURES, "sector_level2_index")
BooleanArray = NDArray[np.bool_]


def _stable_membership_history(
    valid: BooleanArray, membership: FloatArray, lookback: int
) -> BooleanArray:
    """检查当日及此前lookback市场日全部有效且始终属于同一合法行业。

    例如valid全真、行业[1,2,1,1]、lookback=1输出[假,假,假,真]；
    A→B→A不会因两端相同被误认为连续。缺值、负数、非整数行业均中断。
    两矩阵必须同形二维，lookback为正整数，空日历保留输入形状。
    """
    if (
        valid.ndim != 2
        or valid.shape != membership.shape
        or type(lookback) is not int
        or lookback < 1
    ):
        raise ValueError("invalid_stable_membership_history")
    result = np.zeros(valid.shape, dtype=bool)
    consecutive = np.zeros(valid.shape[1], dtype=int)
    previous = np.full(valid.shape[1], np.nan)
    for day, identifiers in enumerate(membership):
        known = (
            np.isfinite(identifiers) & (identifiers >= 0) & (identifiers == np.floor(identifiers))
        )
        consecutive = np.where(
            valid[day] & known, np.where(identifiers == previous, consecutive + 1, 1), 0
        )
        result[day] = consecutive > lookback
        previous = identifiers
    return result


def _peer_statistics(
    values: FloatArray, membership: FloatArray, eligible: BooleanArray
) -> tuple[FloatArray, BooleanArray]:
    """返回本股值减剔除本股同业均值及至少5同业/60%覆盖的可比较矩阵。

    例如6个同业值[.1,0,0,0,0,0]全有效，首股差值.1，其余-.02；
    只剩5个有效成员时，每个有效成员仅有4同业，全部关闭。分母包含当日已知
    行业的其他成员，即使其价格缺失。所有合法行业ID只用于当日分组，不回填。
    按同业首个有效值中心化求和，避免全等值因组大小不同而产生虚假强弱。
    输入为同形二维矩阵，否则拒绝；未知或资格不足返回NaN/False。
    """
    if values.ndim != 2 or values.shape != membership.shape or eligible.shape != values.shape:
        raise ValueError("invalid_peer_statistics_shape")
    relative = np.full(values.shape, np.nan)
    comparable = np.zeros(values.shape, dtype=bool)
    for day, identifiers in enumerate(membership):
        known = (
            np.isfinite(identifiers) & (identifiers >= 0) & (identifiers == np.floor(identifiers))
        )
        positions = np.flatnonzero(known)
        if not len(positions):
            continue
        _, groups = np.unique(identifiers[positions], return_inverse=True)
        group_count = int(np.max(groups)) + 1
        observed = eligible[day, positions] & np.isfinite(values[day, positions])
        observed_positions = np.flatnonzero(observed)
        anchors = np.zeros(group_count)
        observed_groups, first = np.unique(groups[observed_positions], return_index=True)
        anchors[observed_groups] = values[day, positions[observed_positions[first]]]
        centered = np.where(observed, values[day, positions] - anchors[groups], 0)
        counts = np.bincount(groups, weights=observed, minlength=group_count)
        sums = np.bincount(groups, weights=centered, minlength=group_count)
        expected = np.bincount(groups, minlength=group_count)[groups] - 1
        peers = counts[groups] - observed
        usable = observed & (peers >= 5) & (peers >= 0.6 * expected)
        relative[day, positions] = np.where(
            usable,
            centered
            - np.divide(
                sums[groups] - centered, peers, out=np.full(len(positions), np.nan), where=peers > 0
            ),
            np.nan,
        )
        comparable[day, positions] = usable
    return relative, comparable


def _relative_returns(features: dict[str, FloatArray], membership: FloatArray) -> FloatArray:
    """比较21日连续有效且行业不变成员的20日累计收益，排除自身贡献。

    例如同业6股连续21日有效，首股20日收益.1、其余0，输出首股.1；
    任何一股中途停牌或换行业，则该股不能贡献同业收益，也不能参与排序。
    这里均值是当前可比同业各自20日累计收益均值，不冒充可交易行业指数。
    """
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    eligible = _stable_membership_history(tradable, membership, 20)
    relative, _ = _peer_statistics(features["return20"], membership, eligible)
    return relative


def _onset_event(features: dict[str, FloatArray], membership: FloatArray) -> BooleanArray:
    """只构造指定行业层级的原selective启动事件，额外要求足够可用同业。

    例如同组广度昨日.5、今日.7、五日前.5且10日动量.02时事件成立，
    前提当前/昨日/五日前本股以外至少5同业且覆盖60%。沿用原等权行业收益、
    广度和五日差分口径；群动指标仍包括本股，仅资格统计剔除本股。候选窗口
    内换行业直接关闭，不能把旧行业广度当新行业前值。此函数不包含个股选股、
    市场广度、持仓退出或入场否决；这些责任由调用方及原执行器承担。
    """
    industry = _sector_features(features | {"sector_index": membership})
    close, average = features["close"], features["ma20"]
    tradable = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    previous_tradable = np.zeros(tradable.shape, dtype=bool)
    previous_tradable[1:] = tradable[:-1]
    known = (
        tradable
        & previous_tradable
        & np.isfinite(average)
        & (average > 0)
        & np.isfinite(features["return1"])
        & (features["return1"] > -1)
    )
    _, peers_known = _peer_statistics(features["return1"], membership, known)
    previous_peers = _lag_market(peers_known.astype(float), 1) == 1
    earlier_peers = _lag_market(peers_known.astype(float), 5) == 1
    breadth = industry["breadth"]
    previous_breadth = _lag_market(breadth, 1)
    return cast(
        BooleanArray,
        (
            _stable_membership_history(tradable, membership, 5)
            & peers_known
            & previous_peers
            & earlier_peers
            & (previous_breadth < 0.65)
            & (breadth >= 0.65)
            & (industry["diffusion"] >= 0.15)
            & (industry["momentum10"] > 0)
            & (industry["momentum10"] <= 0.08)
        ),
    )


def _original_stock_mask(features: dict[str, FloatArray]) -> BooleanArray:
    """精确复建原L1 selective_quiet_risk个股条件，不含行业启动或市场条件。

    例如20日收益.05、日收益.002、close≥.98MA20且原L1可比截面的波动/换手
    分位分别≤.6/≤.7，输出True。更换L2入场事件时仍使用这组L1分位，防止
    L2可得性改变截面成员后暗中改变个股资格；21市场日条件与源实现一致。
    """
    industry = _sector_features(features)
    volatility, turnover_mean = _stock_statistics(features)
    close, average = features["close"], features["ma20"]
    tradable = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    comparable = (
        complete_history(tradable, 20)
        & np.isfinite(volatility)
        & np.isfinite(industry["breadth"])
        & np.isfinite(average)
        & (average > 0)
        & np.isfinite(features["return1"])
        & np.isfinite(features["return5"])
        & np.isfinite(features["return20"])
    )
    volatility_rank = _percentile(np.where(comparable, volatility, np.nan))
    turnover_rank = _percentile(np.where(comparable, turnover_mean, np.nan))
    return cast(
        BooleanArray,
        (
            comparable
            & (features["return20"] >= 0)
            & (features["return20"] <= 0.10)
            & (features["return1"] >= -0.01)
            & (features["return1"] <= 0.03)
            & (close >= average * 0.98)
            & (volatility_rank <= 0.6)
            & (turnover_rank <= 0.7)
        ),
    )


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """返回同研究域的L1/L2排序三组与仅替换行业入场事件的L2第四组。

    例如L1/L2各有至少5个完整同业时，matched_control与两排序组共享原entry；
    缺L2或矩阵错维抛ValueError，未知L2只关闭对应证券，不回填。共同域要求
    本股及有效同业21日连续行情/同层归属。两个相对收益在同候选截面转平均秩
    [0,1]，等值同分，非候选不排名。L2事件可新增入场日期，但仍保留原L1个股
    条件、评分、退出、entry_invalidation、风险及追价2%；不提前将否决塞入
    新entry_mask。原执行器继续负责入场禁止、挂单撤销与下一合格开盘成交。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("hierarchy_features_missing_or_misaligned")
    base = next(
        item for item in build_structure_strategies(panel, features) if item.name == BASE_NAME
    )
    l1_relative = _relative_returns(features, features["sector_index"])
    l2_relative = _relative_returns(features, features["sector_level2_index"])
    common = np.isfinite(l1_relative) & np.isfinite(l2_relative)
    matched_entry = base.entry_mask & common
    definitions = [replace(base, name="hierarchy_matched_control", entry_mask=matched_entry)]
    for label, relative in (("l1", l1_relative), ("l2", l2_relative)):
        scores = _percentile(np.where(matched_entry, relative, np.nan))
        definitions.append(
            replace(
                base,
                name=f"hierarchy_{label}_peer_relative20",
                entry_mask=matched_entry,
                scores=scores,
            )
        )
    market_breadth = panel.market.get("breadth", np.full(shape[0], np.nan))
    market_known = np.isfinite(market_breadth) & (market_breadth >= 0.5) & (market_breadth <= 1)
    l2_entry = (
        common
        & _original_stock_mask(features)
        & _onset_event(features, features["sector_level2_index"])
        & market_known[:, None]
    )
    definitions.append(replace(base, name="hierarchy_l2_onset", entry_mask=l2_entry))
    return definitions
