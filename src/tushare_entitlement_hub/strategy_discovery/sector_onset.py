"""行业初次群动与强行业温和回调；P02历史行业代理不代表历史概念板块验证。

预声明2家族×2门槛×2确认评分×2退出，共16项，不读取旧反转信号或大盘阶段。
行业事件先按历史行业ID比较，再映射当日成员，不能把换行业误作广度跃升。
"""

import numpy as np

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray, rolling
from .contracts import StrategyDefinition
from .sector import _percentile, _sector_features
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "close",
    "ma20",
    "return1",
    "return5",
    "return20",
    "turnover",
    "sector_index",
)


def _previous_industry(values: FloatArray, membership: FloatArray) -> FloatArray:
    """把当前行业ID对应的昨日指标映射至当前股票，不滞后股票自己的旧行业指标。

    例如昨日股票A属行业0广度.2、行业1广度.8，今日A转行业1，则A前值为.8。
    昨日该行业不存在或指标未知时保持NaN；首日未知，不用未来行业成员倒填。
    """
    result = np.full(values.shape, np.nan)
    known = np.isfinite(membership) & (membership >= 0) & (membership == np.floor(membership))
    for day in range(1, len(values)):
        previous_positions = np.flatnonzero(known[day - 1] & np.isfinite(values[day - 1]))
        previous_identifiers, first_positions = np.unique(
            membership[day - 1, previous_positions],
            return_index=True,
        )
        for identifier, first in zip(previous_identifiers, first_positions, strict=True):
            members = known[day] & (membership[day] == identifier)
            result[day, members] = values[day - 1, previous_positions[first]]
    return result


def _stock_statistics(features: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray]:
    """以连续市场日计算20日收益总体标准差、20日平均换手，缺失不当零。

    例如21天有效收盘、最近20日收益交替±.01，索引20波动为.01；
    中间一天停牌会阻断21日完整性，复牌跨日收益不得混入可比较波动排名。
    收益<=-1、无穷或负换手转未知；换手缺失不阻止不依赖它的收复均线版本。
    """
    shape = features["valid"].shape
    volatility, turnover_mean = np.full(shape, np.nan), np.full(shape, np.nan)
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    comparable = complete_history(tradable, 20)
    for column in range(shape[1]):
        returns = features["return1"][:, column]
        returns_known = np.isfinite(returns) & (returns > -1) & tradable[:, column]
        volatility[:, column] = rolling(np.where(returns_known, returns, np.nan), 20, "std")
        turnover = features["turnover"][:, column]
        turnover_known = np.isfinite(turnover) & (turnover >= 0) & tradable[:, column]
        turnover_mean[:, column] = rolling(np.where(turnover_known, turnover, np.nan), 20, "mean")
    volatility[~comparable] = np.nan
    turnover_mean[~comparable] = np.nan
    return volatility, turnover_mean


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成16个独立策略；例如31日合成行业广度从.4到.8会触发当日onset而非次日。

    broad/selective的个股20日涨幅限制[-3%,15%]/[0,10%]。初期群动要求行业广度
    昨低于、今达到55%/65%，五日扩散至少10/15个百分点，10日动量在(0,10%/8%]。
    回调要求行业20日动量[2%,15%]/[3%,12%]、行业秩至少.5/.6、广度至少.55/.65；
    个股5日回报[-8%,-1%]/[-5%,-1%]，close/MA20在[.97,1.03]/[.98,1.02]。
    reclaim_mid要求昨收<昨日正且已知MA20、今收>=MA20且日涨>0，优先20日涨幅
    接近5%与低波动；quiet_risk
    要求日涨[-1%,3%]、close>=.98MA20、波动秩<=.6及换手秩<=.7，按低风险排名。
    所有股票比较要求21连续有效市场日；行业因果收益与覆盖门复用_sector_features。
    排名范围是行业/技术已知截面，Beta、市值、流动性和限价带由执行器另行筛选。
    两退出为4%TP/10%SL/20日及8%TP/8%SL/30日，均额外在行业广度<.4或20日动量
    <=0时退出。90%总仓/5%单票/25%行业、无跟踪、收盘确认后下一合格开盘成交。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("sector_onset_features_missing_or_misaligned")
    industry = _sector_features(features)
    previous_breadth = _previous_industry(industry["breadth"], features["sector_index"])
    volatility, turnover_mean = _stock_statistics(features)
    close, average = features["close"], features["ma20"]
    previous_close, previous_average = np.full(shape, np.nan), np.full(shape, np.nan)
    previous_close[1:], previous_average[1:] = close[:-1], average[:-1]
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
    moderate_rank = _percentile(np.where(comparable, -np.abs(features["return20"] - 0.05), np.nan))
    scores = {
        "reclaim_mid": 0.7 * moderate_rank + 0.3 * (1 - volatility_rank),
        "quiet_risk": 0.7 * (1 - volatility_rank) + 0.3 * (1 - turnover_rank),
    }
    confirmations = {
        "reclaim_mid": (features["return1"] > 0)
        & (close >= average)
        & np.isfinite(previous_average)
        & (previous_average > 0)
        & (previous_close < previous_average),
        "quiet_risk": (features["return1"] >= -0.01)
        & (features["return1"] <= 0.03)
        & (close >= average * 0.98)
        & (volatility_rank <= 0.6)
        & (turnover_rank <= 0.7),
    }
    exit_mask = tradable & ((industry["breadth"] < 0.4) | (industry["momentum20"] <= 0))
    definitions = []
    for profile, selective in (("broad", False), ("selective", True)):
        breadth_threshold = 0.65 if selective else 0.55
        stock_range = (
            comparable
            & (features["return20"] >= (0 if selective else -0.03))
            & (features["return20"] <= (0.10 if selective else 0.15))
        )
        onset = (
            stock_range
            & (previous_breadth < breadth_threshold)
            & (industry["breadth"] >= breadth_threshold)
            & (industry["diffusion"] >= (0.15 if selective else 0.10))
            & (industry["momentum10"] > 0)
            & (industry["momentum10"] <= (0.08 if selective else 0.10))
        )
        pullback = (
            stock_range
            & (industry["breadth"] >= breadth_threshold)
            & (industry["momentum20"] >= (0.03 if selective else 0.02))
            & (industry["momentum20"] <= (0.12 if selective else 0.15))
            & (industry["rank20"] >= (0.6 if selective else 0.5))
            & (features["return5"] >= (-0.05 if selective else -0.08))
            & (features["return5"] <= -0.01)
            & (close >= average * (0.98 if selective else 0.97))
            & (close <= average * (1.02 if selective else 1.03))
        )
        for prefix, family, event in (
            ("sector_onset", "sector_breadth_onset", onset),
            ("sector_pullback", "strong_sector_pullback", pullback),
        ):
            for confirmation, confirmed in confirmations.items():
                for exit_name, take_profit, stop_loss, holding in (
                    ("tp4_sl10_h20", 0.04, 0.10, 20),
                    ("tp8_sl8_h30", 0.08, 0.08, 30),
                ):
                    name = f"{prefix}_{profile}_{confirmation}_{exit_name}"
                    definitions.append(
                        StrategyDefinition(
                            name=name,
                            family=family,
                            entry_mask=event & confirmed,
                            scores=scores[confirmation],
                            exit_mask=exit_mask,
                            policy=PortfolioPolicy(
                                name=name,
                                switching=False,
                                stage_allocation=False,
                                force_rebalance=False,
                                allocations=(0.9,) * 5,
                                position_weight=0.05,
                                sector_weight=0.25,
                                take_profit=take_profit,
                                stop_loss=stop_loss,
                                maximum_holding=holding,
                            ),
                            maximum_entry_gap=None,
                            trailing_stop=0,
                            intraday_exits=False,
                        )
                    )
    return definitions
