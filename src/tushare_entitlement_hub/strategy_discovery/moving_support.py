"""价托、量托与双托的独立事件研究，不继承原放量反转信号。"""

import numpy as np

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .event_windows import moving_average, recent_event, shift_matrix, support_formation
from .windows import complete_history

REQUIRED_FEATURES = ("valid", "close", "raw_vol", "raw_adj_factor", "return1", "close_location")


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成价托、量托、五日内双托各两种退出，共六个具名配置。

    输入日×证券复权收盘与原始成交量，使用5/10/20日均线，十日内依次三次金叉。
    例如今日价托成立、三日前量托且两组均线仍多头排列，今日产生双托信号。
    要求34连续有效市场日、该窗口复权因子不变，避免停牌或拆股制造均量交叉。
    当前close在MA20上方且追离<=10%、return1>0、收盘位置>=.6；分数偏好收盘强
    且离MA20较近的证券。量托仅代表成交量结构，不等价于资金流入。
    quick退出8%/5%/10日，patient退出20%/8%/30日并跟踪8%；均在跌破MA20时
    收盘发退出，次日合格开盘交易。缺行情只撤挂单，不伪造持仓的价格退出。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("invalid_moving_support_features")
    close, volume, factor = (features[name] for name in ("close", "raw_vol", "raw_adj_factor"))
    known = features["valid"] == 1
    for values in (close, volume, factor):
        known &= np.isfinite(values) & (values > 0)
    price_averages = [
        moving_average(np.where(known, close, np.nan), window) for window in (5, 10, 20)
    ]
    volume_averages = [
        moving_average(np.where(known, volume, np.nan), window) for window in (5, 10, 20)
    ]
    price_event = support_formation(*price_averages)
    volume_event = support_formation(*volume_averages)
    continuous = complete_history(known, 33)
    unchanged_factor = factor == shift_matrix(factor)
    continuous &= complete_history(unchanged_factor, 32)
    price_event &= continuous
    volume_event &= continuous
    price_short, price_middle, price_long = price_averages
    volume_short, volume_middle, volume_long = volume_averages
    aligned = (
        (price_short > price_middle)
        & (price_middle > price_long)
        & (volume_short > volume_middle)
        & (volume_middle > volume_long)
    )
    events = {
        "price_support": price_event,
        "volume_support": volume_event,
        "dual_support": recent_event(price_event, 5)
        & recent_event(volume_event, 5)
        & (price_event | volume_event)
        & aligned,
    }
    extension = np.divide(close, price_long, out=np.full(shape, np.nan), where=price_long > 0) - 1
    quality = (
        continuous
        & np.isfinite(features["return1"])
        & (features["return1"] > 0)
        & np.isfinite(features["close_location"])
        & (features["close_location"] >= 0.6)
        & (features["close_location"] <= 1)
        & (extension > 0)
        & (extension <= 0.10)
    )
    scores = np.where(quality, features["close_location"] - 5 * extension, np.nan)
    exit_mask = known & np.isfinite(price_long) & (close < price_long)
    invalidation = ~known | ~np.isfinite(price_long) | (close < price_long)
    definitions = []
    for family, event in events.items():
        for label, take_profit, stop_loss, holding, trailing in (
            ("quick", 0.08, 0.05, 10, 0.0),
            ("patient", 0.20, 0.08, 30, 0.08),
        ):
            name = f"{family}_{label}"
            definitions.append(
                StrategyDefinition(
                    name=name,
                    family=family,
                    entry_mask=event & quality,
                    scores=scores,
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
                    exit_mask=exit_mask,
                    entry_invalidation_mask=invalidation,
                    maximum_entry_gap=0.02,
                    trailing_stop=trailing,
                )
            )
    return definitions
