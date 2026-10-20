"""最终参数交由独立逐笔状态机复算，输出完整交易和可核验汇总。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ..signal_research.simulation import SimulationConfiguration, simulate_signal_trades
from .concepts import load_features
from .execution import ExitPolicy, independent_trades, outcome_arrays
from .search import maximum_signal_gap
from .strategies import StrategyDefinition, strategy_masks


def source_bars(
    features: dict[str, np.ndarray],
    calendar: list[str],
    code: str,
) -> list[dict[str, Any]]:
    """从保存的原始字段重建独立模拟器输入；原始10元仍为10元，不用复权价成交。

    输入features含raw_open等字段，输出完整市场日历，缺源日保留不可成交记录。
    normalized_histogram不注入，因此最终止盈止损实验不会偷偷启用原MACD退出。
    """
    raw_fields = [name for name in features if name.startswith("raw_")]
    return [
        {
            "ts_code": code,
            "trade_date": day,
            **{name.removeprefix("raw_"): float(features[name][index]) for name in raw_fields},
        }
        for index, day in enumerate(calendar)
    ]


def open_position_mark(
    features: dict[str, np.ndarray],
    calendar: list[str],
    trade: dict[str, object],
) -> dict[str, object]:
    """按末次有效收盘估计未退出浮盈亏，单列报价距期末天数，不伪装可成交收益。

    例：10元买入、末次收盘9元且因子未变，gross_mark_return=-.1；不算入已退出胜率。
    """
    entry_day = str(trade["entry_date"]).replace("-", "")
    entry = calendar.index(entry_day)
    closes, factors = features["raw_close"], features["raw_adj_factor"]
    available = np.flatnonzero(
        (np.arange(len(calendar)) >= entry)
        & np.isfinite(closes)
        & (closes > 0)
        & np.isfinite(factors)
        & (factors > 0)
    )
    last = int(available[-1])
    value = float(
        closes[last] * factors[last] / (float(str(trade["entry_price"])) * factors[entry]) - 1
    )
    return {
        "ts_code": trade["ts_code"],
        "entry_date": trade["entry_date"],
        "last_quote_date": calendar[last],
        "stale_sessions": len(calendar) - 1 - last,
        "gross_mark_return": value,
        "holding_sessions": len(calendar) - entry,
        "exit_trigger_date": trade.get("exit_trigger_date"),
    }


def verify_candidate(
    dataset: Path,
    output: Path,
    definition: StrategyDefinition,
    policy: ExitPolicy,
) -> Path:
    """独立复算一组完整五年策略，逐笔核对向量引擎并保存全部状态，不只保存盈利交易。

    输入冻结特征、策略和退出参数，输出verified.json及trades.csv；任一日期或收益
    不一致立即抛异常，不发布已验证结论。期末未平仓另列，不算作已退出胜率。
    """
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    calendar = manifest["calendar"]
    calendar_index = {
        date.fromisoformat(day).isoformat(): index for index, day in enumerate(calendar)
    }
    active_window = np.asarray(["20211018" <= day <= "20261016" for day in calendar])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trades.csv").exists():
        raise ValueError("verified_output_exists")
    policy_configuration = SimulationConfiguration(
        maximum_entry_wait_sessions=3,
        maximum_holding_sessions=policy.holding_sessions,
        price_stop_loss_fraction=policy.stop_loss,
        price_take_profit_fraction=policy.take_profit,
    )
    columns = (
        "ts_code",
        "confirmation_date",
        "status",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "exit_trigger_date",
        "exit_reason",
        "holding_sessions",
        "fee_adjusted_research_return",
        "censor_reason",
        "skip_reason",
    )
    counts: dict[str, int] = {}
    returns = []
    holding = []
    entry_days = []
    annual: dict[str, list[float]] = {}
    worst = []
    open_positions = []
    with (output / "trades.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for processed, code in enumerate(manifest["securities"], 1):
            features = load_features(dataset, code)
            candidates = np.flatnonzero(
                strategy_masks(features, (definition,))[definition.name] & active_window
            )
            if not len(candidates):
                continue
            outcomes = outcome_arrays(features, calendar, policy)
            selected, skipped = independent_trades(candidates, outcomes, len(calendar))
            expected_skipped = set(int(index) for index in skipped)
            signals = [
                {
                    "ts_code": code,
                    "confirmation_date": calendar[index],
                    "experiment_id": definition.name,
                }
                for index in candidates
            ]
            trades = simulate_signal_trades(
                source_bars(features, calendar, code), signals, policy_configuration
            )
            if len(trades) != len(candidates):
                raise ValueError("independent_trade_count_mismatch")
            for index, trade in zip(candidates, trades, strict=True):
                expected_entry = int(outcomes["entry_index"][index])
                expected_exit = int(outcomes["exit_index"][index])
                if (trade["status"] == "skipped") != (int(index) in expected_skipped):
                    raise ValueError("independent_overlap_mismatch")
                if trade["status"] != "skipped":
                    actual_entry = (
                        calendar_index[str(trade["entry_date"])] if trade["entry_date"] else -1
                    )
                    actual_exit = (
                        calendar_index[str(trade["exit_date"])]
                        if trade["exit_date"]
                        else len(calendar)
                    )
                    if actual_entry != expected_entry or actual_exit != expected_exit:
                        raise ValueError("independent_execution_dates_mismatch")
                    actual_return = trade["fee_adjusted_research_return"]
                    expected_return = outcomes["return"][index]
                    if (actual_return is None) != (not np.isfinite(expected_return)) or (
                        actual_return is not None
                        and abs(float(str(actual_return)) - expected_return) > 1e-10
                    ):
                        raise ValueError("independent_execution_return_mismatch")
                status = str(trade["status"])
                counts[status] = counts.get(status, 0) + 1
                writer.writerow(trade)
                if trade["entry_date"] is not None:
                    entry_days.append(calendar_index[str(trade["entry_date"])])
                if status == "open":
                    open_positions.append(open_position_mark(features, calendar, trade))
                if status == "closed":
                    value = float(str(trade["fee_adjusted_research_return"]))
                    returns.append(value)
                    holding.append(float(str(trade["holding_sessions"])))
                    annual.setdefault(str(trade["confirmation_date"])[:4], []).append(value)
                    worst.append({name: trade.get(name) for name in columns})
                    worst = sorted(
                        worst, key=lambda row: float(str(row["fee_adjusted_research_return"]))
                    )[:20]
            if processed % 250 == 0:
                print(
                    f"[verify] securities={processed} closed={counts.get('closed', 0)}", flush=True
                )
    values = np.asarray(returns)
    positive = values[values > 0]
    negative = values[values < 0]
    win_rate = float(np.mean(values > 0)) if len(values) else None
    gap = maximum_signal_gap(entry_days, calendar.index("20211018"), calendar.index("20261016"))
    summary: dict[str, Any] = {
        "strategy": asdict(definition),
        "exit_policy": asdict(policy),
        "counts": counts,
        "closed_trades": len(values),
        "win_rate": win_rate,
        "mean_return": float(np.mean(values)) if len(values) else None,
        "median_return": float(np.median(values)) if len(values) else None,
        "average_win": float(np.mean(positive)) if len(positive) else None,
        "average_loss": float(np.mean(negative)) if len(negative) else None,
        "profit_factor": float(np.sum(positive) / -np.sum(negative)) if len(negative) else None,
        "mean_holding_sessions": float(np.mean(holding)) if holding else None,
        "entry_dates": len(set(entry_days)),
        "maximum_entry_gap": gap,
        "target_passed": bool(win_rate is not None and win_rate >= 0.65 and gap <= 10),
        "annual": {
            year: {
                "closed_trades": len(items),
                "win_rate": float(np.mean(np.array(items) > 0)),
                "mean_return": float(np.mean(items)),
            }
            for year, items in sorted(annual.items())
        },
        "worst_trades": worst,
        "open_positions": open_positions,
        "win_rate_if_all_open_positions_lose": (
            len(positive) / (len(values) + len(open_positions))
            if len(values) + len(open_positions)
            else None
        ),
        "historical_pit_certified": False,
        "all_data_used_for_optimization": True,
        "independent_execution_verified": True,
        "source_fingerprint": manifest["source_fingerprint"],
    }
    path = output / "verified.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
