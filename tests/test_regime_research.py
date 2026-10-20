"""市场阶段、真实现金账本与因果交易的确定性离线场景。"""

import json
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.accounting import (
    Distribution,
    DistributionBook,
    Holding,
    affordable_shares,
    load_distributions,
    transaction_fee,
)
from tushare_entitlement_hub.regime_research.attribution import (
    block_alpha,
    evaluate_portfolio,
    factor_series,
    matched_trade_controls,
    performance,
)
from tushare_entitlement_hub.regime_research.market import (
    RegimeParameters,
    classify_regimes,
    rolling_beta,
)
from tushare_entitlement_hub.regime_research.panel import FIELDS, Panel
from tushare_entitlement_hub.regime_research.portfolio import (
    PortfolioPolicy,
    entry_candidates,
    exit_reason,
    simulate_portfolio,
)


@pytest.fixture
def panel():
    """六十日二十证券的合成面板，所有价格上涨；不包含真实行情。"""
    count, securities = 60, 20
    calendar = [(date(2024, 1, 1) + timedelta(days=day)).strftime("%Y%m%d") for day in range(count)]
    values = {name: np.ones((count, securities)) for name in (*FIELDS, "beta", "capitalization")}
    price = np.broadcast_to((10 * 1.01 ** np.arange(count))[:, None], (count, securities)).copy()
    for name in ("raw_open", "raw_close", "close", "raw_low", "raw_high"):
        values[name] = price.copy()
    values["ma20"] = price * 0.98
    values["raw_up_limit"] = price * 1.1
    values["raw_down_limit"] = price * 0.9
    values["raw_amount"][:] = 50000
    values["sector_index"][:] = np.arange(securities) % 2
    values["capitalization"][:] = 100000 + np.arange(securities) * 1000
    values["return1"][:] = 0.01
    values["return20"][:] = np.arange(securities) * 0.01
    values["sector_heat"][:] = 0.8
    values["sector_breadth"][:] = 0.8
    values["kdj_j"][:] = 30
    signals = {
        name: np.ones((count, securities), dtype=bool)
        for name in (
            "volume_reversal__sector_group",
            "boll_reentry__sector_group",
            "trend_pullback__sector_group",
            "trend_breakout__sector_hot_group",
        )
    }
    index = 100 * 1.005 ** np.arange(count)
    market = {
        "index": index,
        "total_index": index,
        "index_open": index,
        "breadth": np.full(count, 0.7),
        "advancing": np.full(count, 0.7),
        "amount": np.full(count, 1e9),
        "limit_down": np.zeros(count),
    }
    return Panel(calendar, [f"{600000 + i}.SH" for i in range(securities)], values, signals, market)


def holding(**changes):
    """构造100股、成本1000元的持仓；可覆盖参数供单独场景使用。"""
    return replace(Holding(0, 0, 1, 100, 1000, 10, 10, 10, "volume", 2, 1), **changes)


def envelope(rows):
    """把确定性行转换为提供方成功表结构；例如[{x:1}]输出fields/items包络。"""
    fields = list(rows[0]) if rows else ["fixture"]
    return json.dumps(
        {
            "response": {
                "code": 0,
                "data": {
                    "fields": fields,
                    "items": [[row.get(name) for name in fields] for row in rows],
                },
            }
        }
    )


def test_fee_lots_and_invalid_notional():
    """小额佣金至少5元，历史印花税分段，科创板至少200股。"""
    assert transaction_fee(1000, "20240101", False) == pytest.approx(5.01)
    assert transaction_fee(1000, "20200101", True) == pytest.approx(6.02)
    assert transaction_fee(100000, "20240101", True) == pytest.approx(81)
    for value in (0, -1, np.nan):
        with pytest.raises(ValueError):
            transaction_fee(value, "20240101", True)
    assert affordable_shares(1005.1, 10, "600000.SH", "20240101") == 100
    assert affordable_shares(1000, 10, "600000.SH", "20240101") == 0
    assert affordable_shares(2005.1, 10, "688001.SH", "20240101") == 200
    assert affordable_shares(1005.1, 10, "688001.SH", "20240101") == 0
    assert affordable_shares(100, np.nan, "600000.SH", "20240101") == 0


def test_dividend_entitlement_cash_payment_and_share_unlock():
    """登记日持股才享分红，应收款不可提前花，送股上市前锁定。"""
    position = holding()
    event = Distribution(0, 2, 3, 5, 6, 1, 0.2)
    book = DistributionBook([event, Distribution(1, 2, 3, 5, 6, 1, 0)])
    book.record(2, {0: position})
    assert book.advance(2) == 0
    assert book.advance(3) == 0
    assert position.shares == 120 and position.locked == 20
    assert book.outstanding() == 80
    assert book.advance(4) == 0
    assert book.advance(5) == 80 and book.outstanding() == 0
    assert book.advance(6) == 0 and position.locked == 0
    assert position.dividend == 80


def test_distribution_loader_known_and_unknown_dates(tmp_path):
    """实施分红才计入，未知支付/上市日保留样本末以后；原文不能重复计数。"""
    row = dict(
        ts_code="600000.SH",
        div_proc="实施",
        record_date="20240101",
        ex_date="20240102",
        pay_date=None,
        div_listdate=None,
        cash_div_tax=1,
        stk_div=0.2,
        end_date="20231231",
    )
    path = tmp_path / "dividend-20240102.json"
    path.write_text(
        envelope(
            [
                row,
                row,
                dict(row, div_proc="预案"),
                dict(row, ts_code="600001.SH"),
                dict(row, ex_date="20250101"),
            ]
        )
    )
    events = load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])
    assert len(events) == 1 and events[0].pay == events[0].listing == 2
    (tmp_path / "dividend-probe.json").write_text("invalid probe ignored")
    assert len(load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])) == 1
    for change, message in [
        ({"record_date": None}, "record"),
        ({"cash_div_tax": -1}, "amount"),
        ({"record_date": "20240102"}, "timeline"),
    ]:
        path.write_text(envelope([dict(row, **change)]))
        with pytest.raises(ValueError, match=message):
            load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])
    path.write_text(envelope([row, dict(row, cash_div_tax=2)]))
    with pytest.raises(ValueError, match="conflicting"):
        load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])
    key = "600000.SH|20240102|20231231"
    correction = {key: {"cash_div_tax": 2, "source_url": "https://example.test/announcement"}}
    (tmp_path / "distribution-resolutions.json").write_text(json.dumps(correction))
    assert load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])[0].cash == 2
    correction[key]["source_url"] = "unverified"
    (tmp_path / "distribution-resolutions.json").write_text(json.dumps(correction))
    with pytest.raises(ValueError, match="resolution"):
        load_distributions(tmp_path, ["20240101", "20240102"], ["600000.SH"])


def test_market_states_confirmation_risk_and_causality():
    """上涨需连续确认，极端跌停立即风险；修改未来不得改变历史状态。"""
    count = 340
    index = 100 * 1.001 ** np.arange(count)
    breadth = np.full(count, 0.7)
    advancing = np.full(count, 0.7)
    amount = np.full(count, 1e9)
    limits = np.zeros(count)
    states = classify_regimes(index, breadth, advancing, amount, limits)
    assert states[271] == 4 and states[272] == 4 and states[273] == 1
    limits[300] = 0.1
    risk = classify_regimes(index, breadth, advancing, amount, limits)
    assert risk[300] == 4 and risk[301] == 4 and risk[302] == 1
    altered = index.copy()
    altered[310:] *= 0.5
    updated = classify_regimes(altered, breadth, advancing, amount, limits)
    np.testing.assert_array_equal(updated[:310], risk[:310])
    assert 0 in updated[312:]
    breadth[320:] = np.linspace(0.3, 0.1, 20)
    down = classify_regimes(altered, breadth, advancing, amount, limits)
    assert 3 in down[320:]
    breadth[330:] = 0.6
    amount[330:] = 2e9
    repair = classify_regimes(altered, breadth, advancing, amount, limits)
    assert 2 in repair[330:]
    with pytest.raises(ValueError):
        RegimeParameters(confirmation=0)
    with pytest.raises(ValueError):
        classify_regimes(index, breadth[:-1], advancing, amount, limits)


def test_rolling_beta_and_future_prefix():
    """Beta用历史完整窗口向1收缩，未来跳涨不改变历史估计。"""
    benchmark = np.sin(np.arange(150)) * 0.01
    beta = rolling_beta(2 * benchmark, benchmark, 20)
    assert np.isnan(beta[:19]).all()
    np.testing.assert_allclose(beta[19:], 1.75)
    np.testing.assert_allclose(rolling_beta(2 * benchmark[:100], benchmark[:100], 20), beta[:100])
    assert np.isnan(rolling_beta(np.ones(30), np.ones(30), 20)).all()


def test_entry_routing_and_risk_filter(panel):
    """上涨/修复/震荡路由独立；5%限价带不入选，风险状态不新买。"""
    policy = PortfolioPolicy()
    assert entry_candidates(panel, 1, 1, policy)[0][1] == "trend_pullback__sector_group"
    assert entry_candidates(panel, 1, 2, policy)[0][1] == "volume_reversal__sector_group"
    assert entry_candidates(panel, 1, 0, policy)[0][1] == "boll_reentry__sector_group"
    assert entry_candidates(panel, 1, 4, policy) == []
    assert entry_candidates(panel, 1, 4, replace(policy, mixture=True))
    assert entry_candidates(panel, 1, 4, replace(policy, allow_weak_entries=True))
    assert (
        entry_candidates(
            panel, 1, 1, replace(policy, advance_family="volume_reversal__sector_group")
        )[0][1]
        == "volume_reversal__sector_group"
    )
    for ranking in ("oversold", "liquid"):
        assert entry_candidates(panel, 1, 2, replace(policy, ranking=ranking))
    assert entry_candidates(panel, 1, 3, replace(policy, switching=False))
    panel.values["raw_up_limit"][:] = panel.values["raw_open"] * 1.05
    panel.values["raw_down_limit"][:] = panel.values["raw_open"] * 0.95
    assert entry_candidates(panel, 1, 1, policy) == []
    with pytest.raises(ValueError):
        PortfolioPolicy(position_weight=0)


def test_exit_policy_preserves_stop_and_trend_trailing():
    """止损、止盈、反弹失败、期限、移动止盈各自按收盘触发。"""
    policy = PortfolioPolicy()
    assert exit_reason(holding(), 2, 8.9, policy) == "stop"
    assert exit_reason(holding(), 2, 11, policy) == "profit"
    assert exit_reason(holding(), 10, 9.9, replace(policy, early_exit=5)) == "failed_rebound"
    assert exit_reason(holding(), 30, np.nan, policy) == "time"
    assert exit_reason(holding(), 2, 10.1, policy) == ""
    position = holding(stage=1)
    trailing = replace(policy, trailing=0.06)
    assert exit_reason(position, 2, 12, trailing) == ""
    assert exit_reason(position, 3, 11, trailing) == "trailing"


def test_cash_portfolio_next_open_limits_and_budget(panel):
    """账户不透支、不开同日仓；触发止损后遇跌停延迟卖出。"""
    states = np.full(len(panel.calendar), 2, dtype=np.int8)
    policy = PortfolioPolicy(switching=False, stage_allocation=False)
    result = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert result["entry_days"][0] == 1
    assert min(row["cash"] for row in result["daily"]) >= 0
    assert all(trade["exit"] > trade["entry"] for trade in result["trades"])
    assert result["trades"] and result["fees"] > 0
    assert abs(result["reconciliation_error"]) < 1e-6
    # 单股交易，次日涨停拒买，再下一日开盘买；之后跌停阻止卖出。
    for mask in panel.signals.values():
        mask[:] = False
        mask[0, 0] = True
    panel.values["raw_open"][1, 0] = panel.values["raw_up_limit"][1, 0]
    panel.values["raw_close"][2, 0] = 8
    panel.values["raw_open"][3, 0] = panel.values["raw_down_limit"][3, 0]
    panel.values["raw_open"][4, 0] = 7
    panel.values["raw_down_limit"][4, 0] = 6
    delayed = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert delayed["trades"][0]["entry"] == 2 and delayed["trades"][0]["exit"] == 4
    assert delayed["trades"][0]["return"] < -0.3
    assert delayed["trades"][0]["reason"] == "stop"
    with pytest.raises(ValueError):
        simulate_portfolio(panel, states, [], policy, capital=0)
    with pytest.raises(ValueError):
        simulate_portfolio(panel, states, [], policy, capital=np.nan)


def test_portfolio_prefix_and_current_close_do_not_change_open_fills(panel):
    """修改未来全部报价和当日收盘有效标记，不得回写历史成交或影响当日开盘买入。"""
    states = np.full(60, 2, dtype=np.int8)
    policy = PortfolioPolicy(switching=False, stage_allocation=False)
    baseline = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    for name in ("raw_open", "raw_close", "close"):
        panel.values[name][30:] *= 0.5
    modified = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert baseline["daily"][:30] == modified["daily"][:30]
    assert [t for t in baseline["trades"] if t["exit"] < 30] == [
        t for t in modified["trades"] if t["exit"] < 30
    ]
    for mask in panel.signals.values():
        mask[:] = False
        mask[0, 0] = True
    panel.values["valid"][1, 0] = 0
    changed = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert changed["entry_days"][0] == 1


def test_cash_dividend_reconciles_nav_without_reinvesting(panel):
    """1000股获得税后现金800元，除息价格下跌1元，账户仅承受200元税费损失。"""
    for mask in panel.signals.values():
        mask[:] = False
        mask[0, 0] = True
    for name in ("raw_open", "raw_close", "close"):
        panel.values[name][:] = 10
        panel.values[name][3:, 0] = 9
    panel.values["raw_adj_factor"][3:, 0] = 10 / 9
    states = np.full(60, 2, dtype=np.int8)
    policy = PortfolioPolicy(switching=False, stage_allocation=False, maximum_holding=100)
    event = Distribution(0, 2, 3, 5, 0, 1, 0)
    result = simulate_portfolio(panel, states, [event], policy, start=panel.calendar[0])
    before, ex, pay = (result["daily"][i] for i in (2, 3, 5))
    shares = 4900
    assert ex["receivable"] == pytest.approx(shares * 0.8)
    assert ex["nav"] - before["nav"] == pytest.approx(-shares * 0.2)
    assert pay["cash"] - ex["cash"] == pytest.approx(shares * 0.8)
    assert pay["receivable"] == 0
    assert abs(result["reconciliation_error"]) < 1e-6


def test_rebalance_order_expiry_staleness_and_tiny_budget(panel):
    """风险减仓次日执行，三日涨停买单到期，缺报价持仓单独报告陈旧价值。"""
    states = np.full(60, 2, dtype=np.int8)
    states[5:] = 4
    result = simulate_portfolio(
        panel, states, [], PortfolioPolicy(take_profit=0.2), start=panel.calendar[0]
    )
    assert any(trade["reason"] == "allocation" and trade["exit"] == 6 for trade in result["trades"])
    for mask in panel.signals.values():
        mask[:] = False
        mask[0, 0] = True
    panel.values["raw_open"][1:4, 0] = panel.values["raw_up_limit"][1:4, 0]
    policy = PortfolioPolicy(switching=False, stage_allocation=False, maximum_holding=100)
    expired = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert not expired["entry_days"]
    panel.values["raw_open"][1, 0] = 10
    tiny = simulate_portfolio(panel, states, [], policy, capital=10, start=panel.calendar[0])
    assert tiny["rejected_budget"] > 0
    panel.values["raw_open"][2:, 0] = np.nan
    panel.values["raw_close"][2:, 0] = np.nan
    stale = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    assert stale["stale_value"] > 0


def test_performance_alpha_and_block_intervals():
    """年化来自每日净值，因子完全解释收益时截距为零；不足样本不报告Alpha。"""
    metrics = performance(np.array([0.1, -0.1]))
    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.1)
    assert performance(np.array([]))["total_return"] == 0
    with pytest.raises(ValueError):
        performance(np.array([-1.0]))
    factor = np.sin(np.arange(100)) * 0.01
    result = block_alpha(factor * 2, factor[:, None], samples=30)
    assert result["annualized_intercept"] == pytest.approx(0, abs=1e-12)
    assert result["coefficients"][0] == pytest.approx(2)
    assert not block_alpha(factor[:2], factor[:2, None])["available"]
    with pytest.raises(ValueError):
        block_alpha(factor, factor)


def test_factor_and_matched_controls(panel):
    """同业匹配只看信号时市值/Beta；缺少可成交同业明确计为不可比较。"""
    states = np.full(60, 2, dtype=np.int8)
    policy = PortfolioPolicy(switching=False, stage_allocation=False)
    result = simulate_portfolio(panel, states, [], policy, start=panel.calendar[0])
    factors = factor_series(panel)
    metrics = evaluate_portfolio(result, panel, factors)
    assert metrics["closed_trades"] > 0 and metrics["factor_alpha"]["available"]
    assert metrics["annual"]["2024"]["entries"] == len(result["entry_days"])
    peers = matched_trade_controls(result, panel)
    assert peers["matched"] > 0
    panel.values["raw_open"][1] = np.nan
    missing = matched_trade_controls(result, panel)
    assert missing["unavailable"] > 0
    with pytest.raises(ValueError):
        matched_trade_controls(result, panel, peers=0)
    assert matched_trade_controls({"trades": []}, panel)["mean_excess"] is None
