"""行业参与扩散与定期横截面轮动；P02历史行业代理不等同于历史概念板块。

行业指数逐日用当日已知成分形成等权收益，然后链接历史收益。它不是使用
观察日成分回算的指数。信号均在收盘后形成，由统一执行器处理下一开盘成交。
"""

import numpy as np

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray, rolling
from .contracts import StrategyDefinition
from .windows import complete_history

REQUIRED_FEATURES = ("valid", "close", "ma20", "return1", "return5", "return20", "sector_index")


def _percentile(values: FloatArray) -> FloatArray:
    """按日计算有限值的平均秩百分位，缺值保持未知，完全并列不制造领先者。

    例如[[1, 2, 2, nan]]输出[[0, .75, .75, nan]]；单一有效值为.5。
    """
    result = np.full(values.shape, np.nan)
    for day, row in enumerate(values):
        positions = np.flatnonzero(np.isfinite(row))
        if not len(positions):
            continue
        _, inverse, counts = np.unique(row[positions], return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        ranks = (starts + (counts - 1) / 2) / max(1, len(positions) - 1)
        result[day, positions] = ranks[inverse] if len(positions) > 1 else 0.5
    return result


def _sector_features(features: dict[str, FloatArray]) -> dict[str, FloatArray]:
    """用逐日行业归属聚合收益和广度，再把行业历史指标映射至当日证券。

    例如股票第15日从行业0调至行业1，其第14日收益仍只属于行业0。
    日收益仅接纳连续两个市场日可交易的证券，避免把复牌累计涨幅当日收益。
    每项至少3个有效成员且覆盖当日已知成员60%；不足时整日未知，不前填。
    10/20日动量要求完整行业收益窗口；扩散为当日广度减五市场日前广度。
    """
    membership = features["sector_index"]
    known = np.isfinite(membership) & (membership >= 0) & (membership == np.floor(membership))
    # 全样本唯一ID只分配列空间，不读取未来成员、价格或指标来计算历史数值。
    sector_identifiers = np.unique(membership[known])
    shape = membership.shape[0], len(sector_identifiers)
    sector_returns, sector_breadth = np.full(shape, np.nan), np.full(shape, np.nan)
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    previous_tradable = np.zeros(tradable.shape, dtype=bool)
    previous_tradable[1:] = tradable[:-1]
    return_known = (
        tradable & previous_tradable & np.isfinite(features["return1"]) & (features["return1"] > -1)
    )
    breadth_known = tradable & np.isfinite(features["ma20"]) & (features["ma20"] > 0)
    above_average = features["close"] > features["ma20"]
    for day in range(len(membership)):
        for column, sector_identifier in enumerate(sector_identifiers):
            members = known[day] & (membership[day] == sector_identifier)
            total = int(np.sum(members))
            for eligible, source, target in (
                (return_known, features["return1"], sector_returns),
                (breadth_known, above_average, sector_breadth),
            ):
                observed = members & eligible[day]
                count = int(np.sum(observed))
                if count >= 3 and count >= 0.6 * total:
                    target[day, column] = np.mean(source[day, observed])
    industry = {"breadth": sector_breadth, "diffusion": np.full(shape, np.nan)}
    industry["diffusion"][5:] = sector_breadth[5:] - sector_breadth[:-5]
    for lookback in (10, 20):
        momentum = np.full(shape, np.nan)
        for column in range(len(sector_identifiers)):
            momentum[:, column] = np.expm1(
                rolling(np.log1p(sector_returns[:, column]), lookback, "mean") * lookback
            )
        industry[f"momentum{lookback}"] = momentum
        industry[f"rank{lookback}"] = _percentile(momentum)
    result = {name: np.full(membership.shape, np.nan) for name in industry}
    for column, sector_identifier in enumerate(sector_identifiers):
        day_positions, security_positions = np.where(known & (membership == sector_identifier))
        for name, values in industry.items():
            result[name][day_positions, security_positions] = values[day_positions, column]
    return result


def _policy(name: str, holding_days: int) -> PortfolioPolicy:
    """构造不依赖市场阶段的90%总仓、5%单票、25%行业政策。

    例如(name='sector_leaders', holding_days=20)得到最多20日、止损7%、止盈12%。
    40日趋势配置放宽为9%止损、25%止盈；固定时钟轮动另由exit_mask缓冲退出。
    """
    return PortfolioPolicy(
        name=name,
        switching=False,
        stage_allocation=False,
        force_rebalance=False,
        allocations=(0.9,) * 5,
        maximum_holding=holding_days,
        stop_loss=0.07 if holding_days == 20 else 0.09,
        take_profit=0.12 if holding_days == 20 else 0.25,
    )


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """输出12个行业动量/扩散和8个固定时钟轮动配置，不读取panel.market。

    例如完整(60日,12证券)面板返回20个同形状策略；未知行业或停牌日不入场。
    行业逻辑包括领涨延续、参与扩散、广度持续，分别观察10/20日和持有20/40日。
    轮动使用行业/个股双强或行业内相对强度，按面板首日锚定的5/10市场日时钟
    选择排名前10%/20%，跌出前40%/50%才退出；缺失排名在调仓日保守退出。
    个股20日收益排名与行业相对强度要求当日及此前20市场日连续有效，避免
    复牌后观察窗口与行业市场日窗口不一致。Beta等资格由统一执行器筛选。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("sector_feature_shape_mismatch")
    industry = _sector_features(features)
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    valid = tradable & np.isfinite(industry["breadth"])
    comparable = valid & complete_history(tradable, 20)
    stock_strength = _percentile(np.where(comparable, features["return20"], np.nan))
    definitions = []
    for lookback in (10, 20):
        momentum, sector_rank = industry[f"momentum{lookback}"], industry[f"rank{lookback}"]
        base = comparable & (momentum > 0) & (features["close"] > features["ma20"])
        scores = 0.6 * sector_rank + 0.4 * stock_strength
        entries = {
            "leaders": base
            & (sector_rank >= 0.7)
            & (stock_strength >= 0.6)
            & (industry["breadth"] >= 0.55)
            & (features["return5"] > 0),
            "diffusion": base
            & (industry["diffusion"] >= 0.1)
            & (industry["breadth"] >= 0.55)
            & (stock_strength >= 0.5),
            "participation": base
            & (industry["breadth"] >= 0.7)
            & (industry["diffusion"] >= 0)
            & (features["return20"] > 0),
        }
        exit_mask = valid & ((momentum <= 0) | (industry["breadth"] < 0.4))
        for logic, entry_mask in entries.items():
            for holding_days in (20, 40):
                name = f"sector_{logic}_m{lookback}_h{holding_days}"
                definitions.append(
                    StrategyDefinition(
                        name,
                        "sector_momentum",
                        entry_mask,
                        scores,
                        _policy(name, holding_days),
                        exit_mask=exit_mask,
                        trailing_stop=0.07 if holding_days == 20 else 0.10,
                    )
                )
    relative_strength = _percentile(
        np.where(
            comparable,
            features["return20"] - industry["momentum20"],
            np.nan,
        )
    )
    rotation_scores = {
        "dual_strength": 0.6 * industry["rank20"] + 0.4 * stock_strength,
        "relative_strength": 0.2 * industry["rank20"]
        + 0.5 * stock_strength
        + 0.3 * relative_strength,
    }
    rotation_valid = comparable & np.isfinite(industry["momentum20"])
    for logic, scores in rotation_scores.items():
        ranks = _percentile(np.where(rotation_valid, scores, np.nan))
        for interval in (5, 10):
            clock = (np.arange(shape[0]) % interval == 0)[:, None]
            for top_fraction in (0.1, 0.2):
                name = f"rotation_{logic}_d{interval}_top{round(top_fraction * 100)}"
                entry_mask = clock & rotation_valid & (ranks >= 1 - top_fraction)
                exit_mask = clock & (
                    ~rotation_valid | ~np.isfinite(ranks) | (ranks < 1 - top_fraction - 0.3)
                )
                definitions.append(
                    StrategyDefinition(
                        name,
                        "cross_sectional_rotation",
                        entry_mask,
                        scores,
                        _policy(name, 40),
                        exit_mask=exit_mask,
                    )
                )
    return definitions
