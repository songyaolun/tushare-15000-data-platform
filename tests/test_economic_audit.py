"""短反成交审计的离线场景，不使用真实行情或凭据。"""

import csv
import json

import pytest
from test_regime_research import panel as panel

from tushare_entitlement_hub.economic_research import audit


@pytest.mark.parametrize(
    "date,selling,multiplier,expected",
    [
        ("20240101", False, 1, 5.01),
        ("20200101", True, 2, 12.04),
        ("20240101", True, 1, 5.51),
        ("20240101", False, 0, 0),
    ],
)
def test_independent_fee(date, selling, multiplier, expected):
    """历史税费按时期与买卖方向变化，倍数施加于总费用。"""
    assert audit.fee_amount(1000, date, selling, multiplier) == pytest.approx(expected)


def test_entry_shares_rejects_nonintegral_cost():
    """合法含费成本还原100股；非整数股成本不能通过审计。"""
    assert audit.entry_shares(1005.01, 10, "20240101", 1) == 100
    with pytest.raises(ValueError, match="invalid_entry_shares"):
        audit.entry_shares(1006, 10, "20240101", 1)


@pytest.fixture
def execution_files(tmp_path, monkeypatch, panel):
    """单笔100股合成交易，输入临时目录，输出可修改的交易与指标。"""
    monkeypatch.setattr(audit, "audit_result", lambda path: {"csv_checked": True})
    entry, exit_day, security = 2, 7, 0
    buy_price = panel.values["raw_open"][entry, security] * 1.0005
    sell_price = panel.values["raw_open"][exit_day, security] * 0.9995
    buy_fee = audit.fee_amount(buy_price * 100, panel.calendar[entry], False, 1)
    sell_fee = audit.fee_amount(sell_price * 100, panel.calendar[exit_day], True, 1)
    row = dict(
        security=0,
        confirmation=1,
        source_signal=1,
        entry=2,
        exit=7,
        reason="time",
        ts_code=panel.codes[0],
        cost=buy_price * 100 + buy_fee,
        shares=100,
        proceeds=sell_price * 100 - sell_fee,
    )
    saved = {
        "metrics": {"fees": buy_fee + sell_fee, "open_positions": 0},
        "configuration": {"policy": {"slippage_rate": 0.0005, "fee_multiplier": 1}},
    }
    return tmp_path, panel, row, saved


def write_execution(path, row, saved):
    """保存单笔审计输入；例row包含entry=2则CSV保留该时钟。"""
    (path / "metrics.json").write_text(json.dumps(saved))
    with (path / "trades.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize("delay,open_positions", [(0, 0), (1, 1)])
def test_execution_audit_passes_with_explicit_fee_scope(execution_files, delay, open_positions):
    """真实成交日符合延迟约定；有未平仓时明确限制费用审计范围。"""
    path, panel, row, saved = execution_files
    row["source_signal"] -= delay
    saved["metrics"]["open_positions"] = open_positions
    write_execution(path, row, saved)
    result = audit.audit_execution(path, panel, delay=delay)
    assert result["checked_trades"] == 1
    assert result["fees_fully_reconciled"] == (open_positions == 0)
    assert (
        result["fee_limitation"] is not None if open_positions else result["fee_limitation"] is None
    )


@pytest.mark.parametrize("failure", ["clock", "limit", "lot", "cost", "fee", "short_fee"])
def test_execution_audit_rejects_false_execution(execution_files, failure):
    """损坏时钟、不可买开盘、整手、卖出金额与费用总额必须分别失败。"""
    path, panel, row, saved = execution_files
    expected = {
        "clock": "signal_or_exit_clock",
        "limit": "unexecutable_open",
        "lot": "board_lot",
        "cost": "execution_cost",
        "fee": "fee_total",
        "short_fee": "fee_total",
    }[failure]
    if failure == "clock":
        row["entry"] = 3
    elif failure == "limit":
        panel.values["raw_up_limit"][2, 0] = panel.values["raw_open"][2, 0]
    elif failure == "lot":
        buy_notional = panel.values["raw_open"][2, 0] * 1.0005 * 101
        row["cost"] = buy_notional + audit.fee_amount(buy_notional, panel.calendar[2], False, 1)
    elif failure == "cost":
        row["proceeds"] += 1
    elif failure == "fee":
        saved["metrics"]["fees"] += 1
    else:
        saved["metrics"]["fees"] -= 1
    write_execution(path, row, saved)
    with pytest.raises(ValueError, match=expected):
        audit.audit_execution(path, panel)
