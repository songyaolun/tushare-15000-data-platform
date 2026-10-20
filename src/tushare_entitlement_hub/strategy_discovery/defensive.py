"""六个预声明防御轮动配置；以历史低波动排序，不依赖原放量反转或大盘阶段。"""

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray, rolling
from .contracts import StrategyDefinition
from .sector import _percentile
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "close",
    "return1",
    "turnover",
    "beta",
    "capitalization",
    "sector_index",
    "raw_amount",
    "raw_up_limit",
    "raw_down_limit",
)


def _historical_risk(features: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray]:
    """返回60市场日总体收益标准差与20市场日平均换手率，不压缩停牌日。

    例如61日有效行情且收益交替±.01，首个可用波动率在索引60为.01；
    20日换手率全为2则均值为2。缺值、无穷、收益<=-1和负换手不补零。
    波动率还要求当日及此前60市场日连续有效正收盘，防止复牌跨度冒充单日收益。
    """
    shape = features["valid"].shape
    volatility, mean_turnover = np.full(shape, np.nan), np.full(shape, np.nan)
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    continuous = complete_history(tradable, 60)
    for column in range(shape[1]):
        returns = features["return1"][:, column]
        valid_returns = np.isfinite(returns) & (returns > -1)
        volatility[:, column] = rolling(np.where(valid_returns, returns, np.nan), 60, "std")
        turnover = features["turnover"][:, column]
        valid_turnover = np.isfinite(turnover) & (turnover >= 0) & tradable[:, column]
        mean_turnover[:, column] = rolling(np.where(valid_turnover, turnover, np.nan), 20, "mean")
    volatility[~continuous] = np.nan
    return volatility, mean_turnover


def _known_universe(features: dict[str, FloatArray]) -> NDArray[np.bool_]:
    """仅依据信号日数据取得可比较截面，原执行器仍复核共同成交资格。

    例如成交额19999千元或行业未知的股票返回False；20000千元、限价带10%
    且有效行情/Beta/正市值/非负整数历史行业已知时返回True。不读取未来开盘。
    """
    upper, lower = features["raw_up_limit"], features["raw_down_limit"]
    with np.errstate(divide="ignore", invalid="ignore"):
        band = (upper - lower) / (upper + lower)
    sector = features["sector_index"]
    eligible: NDArray[np.bool_] = (
        (features["valid"] == 1)
        & np.isfinite(features["close"])
        & (features["close"] > 0)
        & np.isfinite(features["beta"])
        & np.isfinite(features["capitalization"])
        & (features["capitalization"] > 0)
        & np.isfinite(sector)
        & (sector >= 0)
        & (sector == np.floor(sector))
        & np.isfinite(features["raw_amount"])
        & (features["raw_amount"] >= 20000)
        & np.isfinite(upper)
        & np.isfinite(lower)
        & (lower > 0)
        & (upper > lower)
        & (band > 0.07)
    )
    return eligible


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成低波动/低波动加换手过滤各5、10、20日时钟，共六个防御轮动配置。

    输入(160日,10证券)特征矩阵，输出六个同形状StrategyDefinition。
    主配置defensive_low_volatility_turnover_d10，其余五个作为预声明对照。
    在当日共同资格中排除60日波动最高20%；换手版另排除完整20日均换手最高20%，
    换手缺失只关闭换手过滤版本。过滤后按低波动平均秩排序，前20%入、跌出前50%
    或资格失效退出。并列分数不按证券编码伪造强弱；全部并列不制造前20%信号。
    时钟锚定面板首日，只在收盘发单；60日持有、12%止损、95%止盈，无跟踪、
    无盘中退出、无市场阶段切换，90%总仓/5%单票/25%行业，实际入场空窗不作保证。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("invalid_defensive_feature_matrix")
    volatility, mean_turnover = _historical_risk(features)
    universe = _known_universe(features) & np.isfinite(volatility)
    volatility_rank = _percentile(np.where(universe, volatility, np.nan))
    turnover_rank = _percentile(np.where(universe, mean_turnover, np.nan))
    low_volatility = universe & (volatility_rank <= 0.8)
    definitions = []
    for variant, use_turnover in (("low_volatility", False), ("low_volatility_turnover", True)):
        filtered = low_volatility & (turnover_rank <= 0.8) if use_turnover else low_volatility
        scores = 1 - _percentile(np.where(filtered, volatility, np.nan))
        for interval in (5, 10, 20):
            clock = (np.arange(shape[0]) % interval == 0)[:, None]
            name = f"defensive_{variant}_d{interval}"
            definitions.append(
                StrategyDefinition(
                    name=name,
                    family="defensive_rotation",
                    entry_mask=clock & filtered & (scores >= 0.8),
                    scores=scores,
                    exit_mask=clock & (~filtered | ~np.isfinite(scores) | (scores < 0.5)),
                    policy=PortfolioPolicy(
                        name=name,
                        switching=False,
                        stage_allocation=False,
                        force_rebalance=False,
                        allocations=(0.9,) * 5,
                        position_weight=0.05,
                        sector_weight=0.25,
                        maximum_holding=60,
                        stop_loss=0.12,
                        take_profit=0.95,
                        trailing=0,
                    ),
                    maximum_entry_gap=None,
                    trailing_stop=0,
                    intraday_exits=False,
                )
            )
    return definitions
