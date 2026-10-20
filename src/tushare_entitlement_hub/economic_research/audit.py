"""独立复核短反逐笔成交成本和信号时钟，不调用执行器与手续费实现。"""

import csv
import json
import math
from pathlib import Path
from typing import Any

from ..regime_research.panel import Panel
from ..strategy_discovery.audit import audit_result


def fee_amount(notional: float, date: str, selling: bool, multiplier: float) -> float:
    """按声明费率独立计算费用；例1000元2024年买入、倍数1得到5.01元。

    股票佣金万三且最低5元，过户费与印花税按历史变更日期分段。
    倍数施加于全部手续费，不含滑点；例如倍数2返回10.02元。
    """
    transfer_rate = 0.00001 if date >= "20220429" else 0.00002
    stamp_rate = (0.0005 if date >= "20230828" else 0.001) if selling else 0.0
    return (max(5.0, 0.0003 * notional) + notional * (transfer_rate + stamp_rate)) * multiplier


def entry_shares(cost: float, price: float, date: str, multiplier: float) -> int:
    """从含费成本独立反解买入股数；例1005.01元、价格10、2024年、倍数1返回100。

    二分金额后要求股数为正整数，不用退出股数替代买入股数，避免送股误差。
    不可解为整数时抛ValueError，例如成本1006且价格10。
    """
    lower, upper = 0.0, cost
    for _ in range(70):
        midpoint = (lower + upper) / 2
        if midpoint + fee_amount(midpoint, date, False, multiplier) > cost:
            upper = midpoint
        else:
            lower = midpoint
    inferred = (lower + upper) / (2 * price)
    shares = round(inferred)
    if shares <= 0 or not math.isclose(inferred, shares, abs_tol=1e-6, rel_tol=0):
        raise ValueError("economic_audit_invalid_entry_shares")
    return shares


def audit_execution(path: Path, panel: Panel, *, delay: int = 0) -> dict[str, Any]:
    """复算CSV指标后按原始开盘和配置成本逐笔检查；例正常结果返回checked_trades笔数。

    基础组入场=原信号+1，延迟组delay=1时入场=原信号+2；每笔派发仅次日有效。
    至少持有5市场日后方可在开盘退出，所有退出原因为time；停牌/跌停可延后。
    期末仍持仓时只核平已平仓费用子集，明确不宣称未导出持仓费用已独立核平。
    """
    summary = audit_result(path)
    saved = json.loads((path / "metrics.json").read_text())
    metrics, policy = saved["metrics"], saved["configuration"]["policy"]
    slippage, multiplier = policy["slippage_rate"], policy["fee_multiplier"]
    with (path / "trades.csv").open(newline="") as stream:
        trades = list(csv.DictReader(stream))
    total_fees = 0.0
    for row in trades:
        security, confirmation, original_signal, entry, exit_day = (
            int(row[name])
            for name in ("security", "confirmation", "source_signal", "entry", "exit")
        )
        if not (
            0 <= original_signal <= confirmation < entry < exit_day < len(panel.calendar)
            and entry == confirmation + 1 == original_signal + delay + 1
            and exit_day >= entry + 5
            and row["reason"] == "time"
            and row["ts_code"] == panel.codes[security]
        ):
            raise ValueError("economic_audit_signal_or_exit_clock")
        opening, upper = (
            float(panel.values[name][entry, security]) for name in ("raw_open", "raw_up_limit")
        )
        exit_opening, lower = (
            float(panel.values[name][exit_day, security]) for name in ("raw_open", "raw_down_limit")
        )
        if not (0 < opening < upper and 0 < lower < exit_opening):
            raise ValueError("economic_audit_unexecutable_open")
        buy_price, sell_price = (
            min(opening * (1 + slippage), upper),
            max(exit_opening * (1 - slippage), lower),
        )
        cost = float(row["cost"])
        shares = entry_shares(cost, buy_price, panel.calendar[entry], multiplier)
        minimum, step = (200, 1) if row["ts_code"].startswith(("688", "689")) else (100, 100)
        if shares < minimum or shares % step:
            raise ValueError("economic_audit_board_lot")
        buy_fee = fee_amount(shares * buy_price, panel.calendar[entry], False, multiplier)
        exit_notional = float(row["shares"]) * sell_price
        sell_fee = fee_amount(exit_notional, panel.calendar[exit_day], True, multiplier)
        if not (
            math.isclose(cost, shares * buy_price + buy_fee, abs_tol=1e-6)
            and math.isclose(float(row["proceeds"]), exit_notional - sell_fee, abs_tol=1e-6)
        ):
            raise ValueError("economic_audit_execution_cost")
        total_fees += buy_fee + sell_fee
    fully_reconciled = metrics["open_positions"] == 0
    if total_fees > metrics["fees"] + 1e-6 or (
        fully_reconciled and not math.isclose(total_fees, metrics["fees"], abs_tol=1e-6)
    ):
        raise ValueError("economic_audit_fee_total")
    return summary | {
        "checked_trades": len(trades),
        "closed_trade_fees": total_fees,
        "fees_fully_reconciled": fully_reconciled,
        "open_positions": metrics["open_positions"],
        "fee_limitation": None if fully_reconciled else "Only closed-trade fees reconstructed.",
        "source_signal_delay": delay,
    }
