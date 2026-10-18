"""把独立策略接入原资金账本，保留手续费、股数、分红与成交约束。"""

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import Distribution, Holding
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from .contracts import StrategyDefinition


def strategy_candidates(
    panel: Panel, day: int, definition: StrategyDefinition
) -> list[tuple[int, str, float]]:
    """按信号日共同资格与策略分数稳定排序，不使用旧策略的KDJ排名。

    输入当天分数[2,3,3]且全部合格，输出列号[1,2,0]对应的候选三元组。
    无效行情、未知行业/Beta/市值、成交额不足20000千元、限价带<=7%
    或非有限分数均排除；同分保持面板证券列顺序。当天退出信号优先于新入场。
    """
    values = panel.values
    upper = values["raw_up_limit"][day]
    lower = values["raw_down_limit"][day]
    with np.errstate(divide="ignore", invalid="ignore"):
        band = (upper - lower) / (upper + lower)
    eligible = (
        definition.entry_mask[day]
        & (values["valid"][day] == 1)
        & np.isfinite(values["beta"][day])
        & np.isfinite(values["capitalization"][day])
        & (values["sector_index"][day] >= 0)
        & (values["raw_amount"][day] >= 20000)
        & (band > 0.07)
        & np.isfinite(definition.scores[day])
    )
    if definition.exit_mask is not None:
        eligible &= ~definition.exit_mask[day]
    columns = np.flatnonzero(eligible)
    ranked = columns[np.lexsort((columns, -definition.scores[day, columns]))]
    return [
        (int(column), definition.family, float(definition.scores[day, column])) for column in ranked
    ]


def additional_exit_reason(
    panel: Panel, holding: Holding, day: int, definition: StrategyDefinition
) -> str:
    """追加任意阶段跟踪退出和策略收盘退出，原止损/止盈/期限优先。

    例峰值绝对复权价12、当日11、trailing_stop=.08输出'trailing'；
    exit_mask[day,security]为True时输出'signal'，均在下个合格开盘卖出。
    """
    adjusted_close = (
        panel.values["raw_close"][day, holding.security]
        * panel.values["raw_adj_factor"][day, holding.security]
    )
    if definition.trailing_stop and np.isfinite(adjusted_close):
        holding.peak_adjusted = max(holding.peak_adjusted, adjusted_close)
        if adjusted_close / holding.peak_adjusted <= 1 - definition.trailing_stop:
            return "trailing"
    if definition.exit_mask is not None and definition.exit_mask[day, holding.security]:
        return "signal"
    return ""


def entry_gap_allowed(
    panel: Panel, confirmation: int, day: int, security: int, maximum_entry_gap: float | None
) -> bool:
    """比较开盘绝对复权价与信号日绝对复权收盘，避免锚定复权尺度混用。

    例信号收盘10、因子2，开盘5.1、因子4，上限.03时返回True。
    不读取成交日收盘或valid；因子缺失/非正或追价超限返回False取消订单。
    None仅关闭追价上限，仍要求成交日因子可用以保障退出价格有效。
    """
    factor = panel.values["raw_adj_factor"][day, security]
    if not np.isfinite(factor) or factor <= 0:
        return False
    if maximum_entry_gap is None:
        return True
    reference = (
        panel.values["raw_close"][confirmation, security]
        * panel.values["raw_adj_factor"][confirmation, security]
    )
    adjusted_open = panel.values["raw_open"][day, security] * factor
    return bool(
        np.isfinite(reference)
        and reference > 0
        and np.isfinite(adjusted_open)
        and adjusted_open > 0
        and adjusted_open <= reference * (1 + maximum_entry_gap)
    )


def intraday_fixed_exit(
    panel: Panel, holding: Holding, day: int, policy: PortfolioPolicy
) -> tuple[float, str] | None:
    """用日线OHLC模拟T+1固定止损止盈，返回原始成交基准价和原因。

    例绝对复权成本20、当日因子2、止损10%，开盘8.8返回(8.8,'stop')；
    开盘10、最低8.9、最高11且双触线则返回(9,'stop')。先判开盘越界，
    再判盘中触线，双触保守按止损，不推测分钟路径。开盘跌停整日拒卖。
    开盘判断仅使用开盘价/因子/限价及已有持仓，沿用源日线开盘可执行的代理假设；
    未在开盘触发时才读取高低收盘价、valid和成交额，要求完整一致的OHLC、
    valid=1且成交额正有限，排除停牌占位日线。这不是分钟级成交认证。
    """
    if day <= holding.entry or holding.locked >= 1e-8:
        return None
    values = panel.values
    opening, factor, lower, upper = (
        float(values[name][day, holding.security])
        for name in (
            "raw_open",
            "raw_adj_factor",
            "raw_down_limit",
            "raw_up_limit",
        )
    )
    if (
        not np.all(np.isfinite([opening, factor, lower, upper]))
        or min(opening, factor, lower, upper) <= 0
        or lower >= upper
        or opening <= lower
        or opening > upper
        or not np.isfinite(holding.entry_adjusted)
        or holding.entry_adjusted <= 0
    ):
        return None
    stop_price = holding.entry_adjusted * (1 - policy.stop_loss) / factor
    profit_price = holding.entry_adjusted * (1 + policy.take_profit) / factor
    if opening <= stop_price:
        return opening, "stop"
    if opening >= profit_price:
        return opening, "profit"
    # 此后才使用完整日线判断盘中触线，不让全天质量标记回写已确定的开盘成交。
    high, low, closing, valid, amount = (
        float(values[name][day, holding.security])
        for name in ("raw_high", "raw_low", "raw_close", "valid", "raw_amount")
    )
    if (
        valid != 1
        or not np.all(np.isfinite([high, low, closing, amount]))
        or min(high, low, closing, amount) <= 0
        or low > min(opening, closing)
        or high < max(opening, closing)
        or low < lower
        or high > upper
    ):
        return None
    if low <= stop_price:
        return stop_price, "stop"
    if high >= profit_price:
        return profit_price, "profit"
    return None


def simulate_strategy(
    panel: Panel,
    states: NDArray[np.int8],
    events: list[Distribution],
    definition: StrategyDefinition,
    *,
    capital: float = 1_000_000,
    start: str = "20211018",
) -> dict[str, Any]:
    """通过可选hook执行独立策略，复用同一份现金、分红和交易账本。

    例simulate_strategy(panel,states,[],definition)输出daily/trades及资金核对结果；
    definition.trailing_stop跨全部阶段生效，旧policy.trailing由本适配层关闭。
    策略矩阵必须与面板一致，否则抛ValueError。信号日收盘下单、下一合格
    开盘执行，追价失败直接取消该次订单，保留原三日有效期和T+1规则。
    等待期间exit_mask命中会在收盘撤单；仅entry_mask变False不会撤单。
    intraday_exits=True时固定止损止盈采用日线OHLC模型，盘中收入不参与当天开盘买单；
    返回execution_model说明，日度开盘敞口基准不能精确匹配盘中卖出时点。
    """
    if definition.entry_mask.shape != (len(panel.calendar), len(panel.codes)):
        raise ValueError("strategy_panel_shape_mismatch")
    policy = replace(definition.policy, name=definition.name, trailing=0)

    def candidate_provider(
        current_panel: Panel, day: int, stage: int, current_policy: PortfolioPolicy
    ) -> list[tuple[int, str, float]]:
        """适配收盘候选接口；例如stage=4仍返回定义中的合格独立策略信号。"""
        return strategy_candidates(current_panel, day, definition)

    def exit_provider(
        current_panel: Panel, holding: Holding, day: int, current_policy: PortfolioPolicy
    ) -> str:
        """适配追加退出接口；例如退出矩阵命中时返回'signal'。"""
        return additional_exit_reason(current_panel, holding, day, definition)

    def entry_filter(current_panel: Panel, confirmation: int, day: int, security: int) -> bool:
        """适配开盘追价接口；例如超出definition.maximum_entry_gap时返回False。"""
        return entry_gap_allowed(
            current_panel, confirmation, day, security, definition.maximum_entry_gap
        )

    def cancel_pending_provider(
        current_panel: Panel, confirmation: int, day: int, security: int
    ) -> bool:
        """只按当前收盘的明确退出信号撤单；例exit_mask[day,security]=True返回True。"""
        return bool(definition.exit_mask is not None and definition.exit_mask[day, security])

    result = simulate_portfolio(
        panel,
        states,
        events,
        policy,
        capital=capital,
        start=start,
        candidate_provider=candidate_provider,
        exit_provider=exit_provider,
        entry_filter=entry_filter,
        cancel_pending_provider=cancel_pending_provider,
        intraday_exit_provider=intraday_fixed_exit if definition.intraday_exits else None,
    )
    if definition.intraday_exits:
        result["execution_model"] = {
            "fixed_exits": "daily_ohlc_t_plus_one",
            "same_bar_both_touched": "stop_first_unless_open_crossed_profit",
            "opening_quote_assumption": "source_daily_open_is_executable_as_in_legacy_model",
            "intraday_quality": "valid_equals_one_and_finite_positive_amount_required",
            "fill_certification": "daily_bar_proxy_not_minute_level_execution_certification",
            "market_benchmark": "daily_open_exposure_proxy_not_exact_intraday_timing",
        }
    return result
