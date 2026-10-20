"""官方涨停价触发的次日接力研究，不模拟当日打板排队成交。"""

import numpy as np

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .event_windows import recent_event, shift_matrix
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "raw_open",
    "raw_close",
    "raw_high",
    "raw_low",
    "raw_up_limit",
    "raw_down_limit",
    "amount_ratio",
    "turnover",
    "close_location",
    "return1",
    "return20",
)


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """输出首板、二板、炸板修复各两组退出，共六个独立配置。

    例如连续十日未封板、今日收于官方涨停且盘中有换手，生成首板收盘信号，
    只允许次日非涨停开盘买入，不把今日涨停价当成可成交价。此前行情或限价缺失
    则不认定首板；二板要求连续两日封板、再前一日未封板。炸板要求最高触及
    官方涨停、收盘离涨停不超过3%但未封住。价位比较容差为半分，不用固定涨幅推断。
    所有配置共同要求成交额比>=1.2、换手1%至30%、收盘位置>=.6、当日收益为正。
    short版本止盈8%/止损5%/3日，patient版本15%/8%/10日加8%收盘跟踪退出。
    收盘退出均等下一合格开盘，阈值不是成交收益保证；无市场或行业群动前置条件。
    """
    shape = len(panel.calendar), len(panel.codes)
    if any(name not in features or features[name].shape != shape for name in REQUIRED_FEATURES):
        raise ValueError("invalid_limit_event_features")
    valid = features["valid"] == 1
    opening, close, high, low, upper, lower = (
        features[name]
        for name in (
            "raw_open",
            "raw_close",
            "raw_high",
            "raw_low",
            "raw_up_limit",
            "raw_down_limit",
        )
    )
    for values in (opening, close, high, low, upper, lower):
        valid &= np.isfinite(values) & (values > 0)
    band = np.divide(upper - lower, upper + lower, out=np.full(shape, np.nan), where=valid)
    valid &= (
        (band > 0.07)
        & (low <= np.minimum(opening, close))
        & (high >= np.maximum(opening, close))
        & (high <= upper + 0.005)
        & (low >= lower - 0.005)
    )
    sealed = valid & (np.abs(close - upper) < 0.005)
    previous_sealed = shift_matrix(sealed.astype(float)) == 1
    sealed_before_previous = shift_matrix(sealed.astype(float), 2) == 1
    previous_ten = shift_matrix(recent_event(sealed, 10).astype(float)) == 1
    quality = complete_history(valid, 10)
    for name in ("amount_ratio", "turnover", "close_location", "return1", "return20"):
        quality &= np.isfinite(features[name])
    quality &= (
        (features["amount_ratio"] >= 1.2)
        & (features["turnover"] >= 1)
        & (features["turnover"] <= 30)
        & (features["close_location"] >= 0.6)
        & (features["close_location"] <= 1)
        & (features["return1"] > 0)
    )
    events = {
        "first_limit": sealed & ~previous_ten & (low < upper - 0.005),
        "second_limit": sealed & previous_sealed & ~sealed_before_previous & (low < upper - 0.005),
        "failed_limit": (np.abs(high - upper) < 0.005) & ~sealed & (close >= upper * 0.97),
    }
    scores = (
        features["close_location"]
        - 0.01 * features["turnover"]
        - 0.5 * np.maximum(features["return20"], 0)
    )
    definitions = []
    for event_name, event in events.items():
        for label, take_profit, stop_loss, holding, trailing in (
            ("quick", 0.08, 0.05, 3, 0.0),
            ("patient", 0.15, 0.08, 10, 0.08),
        ):
            name = f"{event_name}_{label}"
            definitions.append(
                StrategyDefinition(
                    name=name,
                    family=event_name,
                    entry_mask=event & quality,
                    scores=np.where(quality, scores, np.nan),
                    policy=PortfolioPolicy(
                        name=name,
                        switching=False,
                        stage_allocation=False,
                        force_rebalance=False,
                        allocations=(0.9,) * 5,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        maximum_holding=holding,
                        position_weight=0.05,
                        sector_weight=0.25,
                    ),
                    maximum_entry_gap=0.03,
                    maximum_pending_sessions=1,
                    trailing_stop=trailing,
                )
            )
    return definitions
