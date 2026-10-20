"""组合净值、同步仓位基准、风格归因和按日期分块的不确定性估计。"""

from typing import Any

import numpy as np

from ..strategy_optimization.features import FloatArray
from ..strategy_optimization.search import maximum_signal_gap
from .accounting import transaction_fee
from .market import STAGES
from .panel import Panel


def performance(returns: FloatArray) -> dict[str, float]:
    """计算复利收益、250日年化、含初始资金峰值的回撤；例如[.1,-.1]累计-.01。

    输入必须是完整连续账户日收益；空数组输出零统计，不进行逐笔收益年化。
    """
    if not np.all(np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError("invalid_daily_returns")
    curve = np.r_[1.0, np.cumprod(1 + returns)]
    count = len(returns)
    deviation = float(np.std(returns, ddof=1)) if count > 1 else 0.0
    return {
        "total_return": float(curve[-1] - 1),
        "annualized_return": float(curve[-1] ** (250 / count) - 1) if count else 0.0,
        "maximum_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
        "annualized_volatility": deviation * np.sqrt(250),
        "sharpe_zero_cash_rate": float(np.mean(returns) / deviation * np.sqrt(250))
        if deviation
        else 0.0,
    }


def factor_series(panel: Panel) -> dict[str, FloatArray]:
    """以此前可知市值/20日动量构建描述性风格收益，同行业等权收益剔除市场。

    示例小市值组涨2%、大市值组涨1%，当日size因子为1%；停牌缺报价按零收益标记。
    这些是调整后的研究因子，非可直接交易且无成本的可实现组合。
    """
    values = panel.values
    count = len(panel.calendar)
    market = np.r_[0.0, panel.market["total_index"][1:] / panel.market["total_index"][:-1] - 1]
    size, momentum = np.zeros(count), np.zeros(count)
    sector_count = int(np.nanmax(values["sector_index"])) + 1
    sectors = np.full((count, sector_count), np.nan)
    for day in range(1, count):
        returns = np.nan_to_num(values["return1"][day], nan=0)
        for name, output in (("capitalization", size), ("return20", momentum)):
            characteristic = values[name][day - 1]
            valid = np.flatnonzero(np.isfinite(characteristic) & (values["valid"][day - 1] == 1))
            if len(valid) >= 10:
                ranked = valid[np.argsort(characteristic[valid], kind="stable")]
                fraction = max(1, int(len(ranked) * 0.3))
                spread = float(
                    np.mean(returns[ranked[-fraction:]]) - np.mean(returns[ranked[:fraction]])
                )
                output[day] = -spread if name == "capitalization" else spread
        for sector in range(sector_count):
            peers = (values["sector_index"][day - 1] == sector) & (values["valid"][day - 1] == 1)
            if np.sum(peers) >= 5:
                sectors[day, sector] = float(np.mean(returns[peers]))
    return {"market": market, "size": size, "momentum": momentum, "sectors": sectors}


def block_alpha(
    returns: FloatArray,
    factors: FloatArray,
    *,
    samples: int = 300,
    block: int = 20,
) -> dict[str, Any]:
    """OLS截距及连续日期块bootstrap区间，所有参数在全样本内解释，绝不声称样本外。

    例收益恰等于2倍因子且有变动，截距约0；样本不足20返回available=False。
    blocks保持同日所有交易共同扰动，避免把成千上万交易误认为独立样本。
    """
    if factors.ndim != 2 or len(factors) != len(returns) or samples < 1 or block < 1:
        raise ValueError("invalid_alpha_inputs")
    valid = np.isfinite(returns) & np.all(np.isfinite(factors), axis=1)
    response, exposures = returns[valid], factors[valid]
    if len(response) < max(20, factors.shape[1] + 2):
        return {"available": False, "observations": len(response)}
    design = np.column_stack((np.ones(len(response)), exposures))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    generator = np.random.default_rng(20261018)
    estimates = []
    width = min(block, len(response))
    for _ in range(samples):
        starts = generator.integers(
            0, len(response) - width + 1, size=int(np.ceil(len(response) / width))
        )
        indices = np.concatenate([np.arange(start, start + width) for start in starts])[
            : len(response)
        ]
        estimates.append(
            np.linalg.lstsq(design[indices], response[indices], rcond=None)[0][0] * 250
        )
    return {
        "available": True,
        "observations": len(response),
        "annualized_intercept": float(coefficients[0] * 250),
        "coefficients": coefficients[1:].tolist(),
        "block_interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "block_sessions": width,
        "bootstrap_samples": samples,
        "selection_adjusted": False,
        "sample_semantics": "full_sample_descriptive",
    }


def evaluate_portfolio(
    result: dict[str, Any], panel: Panel, factors: dict[str, FloatArray]
) -> dict[str, Any]:
    """将资金回放与同步仓位全指对照，输出年度收益、Beta和样本内因子剩余收益。

    例没有股票交易时账户收益为0、胜率None；不得用满仓指数收益替代账户收益。
    全收益指数无开盘，按价格指数开收盘比例分拆，股息调整计入隔夜；基准不可直接交易。
    """
    daily = result["daily"]
    indices = np.array([row["day"] for row in daily], dtype=int)
    assets = np.array([row["nav"] for row in daily])
    previous_assets = np.r_[result["capital"], assets[:-1]]
    returns = assets / previous_assets - 1
    market = factors["market"][indices]
    exposure = np.array([row["exposure"] for row in daily])
    beta = np.array([row["beta"] for row in daily])
    intraday = panel.market["index"][indices] / panel.market["index_open"][indices] - 1
    buy_hold = market.copy()
    if len(buy_hold):
        buy_hold[0] = 0.0
    if len(buy_hold) > 1:
        buy_hold[1] = intraday[1]
    overnight = (1 + market) / (1 + intraday) - 1
    previous_exposure = np.r_[0.0, exposure[:-1]]
    opening_exposure = np.array([row["open_exposure"] for row in daily])
    cost = np.array([row["cost"] for row in daily]) / previous_assets
    timed = (1 + previous_exposure * overnight - cost) * (1 + opening_exposure * intraday) - 1
    # 按此前收盘行业权重构造行业暴露因子；剩余部分仍可能包含日内择时、其他风格和误差。
    sector_excess = np.zeros(len(daily))
    for offset in range(1, len(daily)):
        day = indices[offset]
        for security, weight in daily[offset - 1]["weights"].items():
            sector = int(panel.values["sector_index"][day - 1, security])
            if sector >= 0 and np.isfinite(factors["sectors"][day, sector]):
                sector_excess[offset] += weight * (factors["sectors"][day, sector] - market[offset])
    design = np.column_stack(
        (
            np.r_[0.0, beta[:-1]] * market,
            sector_excess,
            factors["size"][indices],
            factors["momentum"][indices],
        )
    )
    trade_returns = np.array([trade["return"] for trade in result["trades"]])
    years = sorted({row["date"][:4] for row in daily})
    annual = {}
    for year in years:
        mask = np.array([row["date"].startswith(year) for row in daily])
        closed = [
            trade["return"]
            for trade in result["trades"]
            if panel.calendar[trade["exit"]].startswith(year)
        ]
        annual[year] = {
            **performance(returns[mask]),
            "market_return": performance(buy_hold[mask])["total_return"],
            "timed_market_return": performance(timed[mask])["total_return"],
            "trades_closed": len(closed),
            "win_rate": float(np.mean(np.array(closed) > 0)) if closed else None,
            "entries": sum(panel.calendar[day].startswith(year) for day in result["entry_days"]),
        }
    stages = {}
    for number, name in enumerate(STAGES):
        # 以昨收状态归组今日收益，避免将当日涨跌同时用于状态和收益解释。
        mask = np.r_[False, np.array([row["stage"] == number for row in daily[:-1]])]
        stages[name] = {
            "sessions": int(np.sum(mask)),
            "mean_daily_return": float(np.mean(returns[mask])) if np.any(mask) else None,
            "mean_exposure": float(np.mean(exposure[mask])) if np.any(mask) else None,
            "mean_beta": float(np.mean(beta[mask])) if np.any(mask) else None,
            "mean_market_return": float(np.mean(market[mask])) if np.any(mask) else None,
            "upside_capture": (
                float(np.mean(returns[mask & (market > 0)]) / np.mean(market[mask & (market > 0)]))
                if np.any(mask & (market > 0))
                else None
            ),
            "downside_capture": (
                float(np.mean(returns[mask & (market < 0)]) / np.mean(market[mask & (market < 0)]))
                if np.any(mask & (market < 0))
                else None
            ),
        }
    maximum_gap = (
        maximum_signal_gap(result["entry_days"], int(indices[0]), int(indices[-1]))
        if len(indices)
        else 0
    )
    metrics = {
        **performance(returns),
        "annual": annual,
        "stages": stages,
        "closed_trades": len(trade_returns),
        "win_rate": float(np.mean(trade_returns > 0)) if len(trade_returns) else None,
        "mean_trade_return": float(np.mean(trade_returns)) if len(trade_returns) else None,
        "maximum_entry_gap": maximum_gap,
        "both_targets": bool(
            len(trade_returns) and np.mean(trade_returns > 0) >= 0.65 and maximum_gap <= 10
        ),
        "mean_exposure": float(np.mean(exposure)) if len(exposure) else 0.0,
        "mean_beta": float(np.mean(beta)) if len(beta) else 0.0,
        "market": performance(buy_hold),
        "timed_market": performance(timed),
        "factor_alpha": block_alpha(returns, design),
        "matched_exposure_excess": block_alpha(returns - timed, np.empty((len(returns), 0))),
        "open_positions": result["open_positions"],
        "stale_value": result["stale_value"],
        "fees": result["fees"],
        "turnover": result["turnover"],
        "reconciliation_error": result["reconciliation_error"],
        "terminal_nav_if_stale_zero": float(assets[-1] - result["stale_value"]),
    }
    result["daily_returns"] = returns.tolist()
    result["timed_market_returns"] = timed.tolist()
    return metrics


def matched_trade_controls(result: dict[str, Any], panel: Panel, peers: int = 5) -> dict[str, Any]:
    """同一信号日选择同行业、相近市值/Beta的五只对照股，按原交易持有窗口观察。

    例目标市值/Beta与某同业完全相同，该同业距离为0；不使用之后收益择同业。
    任何入场限板或退出报价缺失都计为不可比较，不事后换股。收益使用复权总回报代理，
    扣与原交易相同成本率，不能冒称可交易组合或精确分红税后Alpha。
    """
    if peers < 1:
        raise ValueError("peers_must_be_positive")
    values = panel.values
    excesses, unavailable = [], 0
    for trade in result["trades"]:
        signal, entry, exit_day, security = (
            trade[name] for name in ("confirmation", "entry", "exit", "security")
        )
        sector = values["sector_index"][signal, security]
        candidates = np.flatnonzero(
            (values["sector_index"][signal] == sector)
            & np.isfinite(values["capitalization"][signal])
            & np.isfinite(values["beta"][signal])
            & (values["valid"][signal] == 1)
        )
        candidates = candidates[candidates != security]
        distances = np.abs(
            np.log(
                values["capitalization"][signal, candidates]
                / values["capitalization"][signal, security]
            )
        ) + np.abs(values["beta"][signal, candidates] - values["beta"][signal, security])
        chosen = candidates[np.argsort(distances, kind="stable")[:peers]]
        opening = values["raw_open"][entry, chosen]
        closing = values["raw_open"][exit_day, chosen]
        eligible = (
            (values["valid"][entry, chosen] == 1)
            & (opening < values["raw_up_limit"][entry, chosen])
            & (values["valid"][exit_day, chosen] == 1)
            & (closing > values["raw_down_limit"][exit_day, chosen])
        )
        if len(chosen) != peers or not np.all(eligible):
            unavailable += 1
            continue
        multiples = (
            closing
            * values["raw_adj_factor"][exit_day, chosen]
            / (opening * values["raw_adj_factor"][entry, chosen])
        )
        # 对照是观察窗口代理，明确固定交易费用假设而非伪造同业可成交股数。
        gross = float(np.mean(multiples * 0.9995 / 1.0005))
        budget = trade["cost"] / peers
        fees = (
            transaction_fee(budget, panel.calendar[entry], False)
            + transaction_fee(budget * gross, panel.calendar[exit_day], True)
        ) / budget
        benchmark_return = gross - 1 - fees
        excesses.append(
            {
                "confirmation": signal,
                "entry": entry,
                "exit": exit_day,
                "strategy_return": trade["return"],
                "peer_return": benchmark_return,
                "excess": trade["return"] - benchmark_return,
            }
        )
    return {
        "matched": len(excesses),
        "unavailable": unavailable,
        "mean_excess": float(np.mean([row["excess"] for row in excesses])) if excesses else None,
        "selection_uses_future_returns": False,
        "records": excesses,
    }
