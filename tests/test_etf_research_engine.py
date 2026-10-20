"""ETF整手现金、分红时点、月末决策及未来数据隔离的离线回归测试。"""

from dataclasses import replace

import pytest

from tushare_entitlement_hub.etf_research.engine import (
    Bar,
    Configuration,
    Dividend,
    commission,
    opening_trade,
    run_account,
    summarize,
    trend_weight,
)


def configuration(**changes):
    """生成小本金零成本场景；默认1000元和1交易日回看方便手算，输出Configuration。"""
    return Configuration(**{'start': '20210201', 'end': '20210302', 'capital': 1000,
                            'lookback': 1, 'commission_rate': 0, 'minimum_commission': 0,
                            'slippage_rate': 0, **changes})


def bar(price=1, **changes):
    """生成正常成交的原始价格Bar；例如bar(2)的开收盘均为2，限价1至4。"""
    return replace(Bar(price, price, price, price, 10000, 1, price * 2, price / 2), **changes)


def history():
    """返回跨三个月的合成交割日历；例如1月末决策只能在2月1日成交。"""
    calendar = ['20210128', '20210129', '20210201', '20210202', '20210226',
                '20210301', '20210302']
    bars = {day: bar() for day in calendar}
    bars['20210128'] = bar(.9)
    return calendar, bars


@pytest.mark.parametrize('changes', [
    {'capital': -1}, {'start': '20210303'}, {'lookback': 0}, {'lot': 0},
    {'capital': float('nan')}, {'commission_rate': -1}, {'commission_rate': 1},
    {'slippage_rate': 1}, {'start': 'bad'},
])
def test_invalid_configuration(changes):
    """场景：无效日期、资金或成本；预期立即拒绝，无资产回放。"""
    with pytest.raises(ValueError):
        configuration(**changes)


def test_commission_cash_rounding_and_no_stock_tax():
    """场景：1000元买1元ETF且最低佣金5元；预期900份、95元现金，无印花税。"""
    settings = configuration(minimum_commission=5)
    assert commission(1000, settings) == 5
    cash, shares, order = opening_trade('20210201', bar(), 1000, 0, 0, 1, settings)
    assert (cash, shares) == (95, 900)
    assert order['stamp_tax'] == order['transfer_fee'] == 0
    assert opening_trade('20210202', bar(), cash, shares, 0, 0, settings)[:2] == (990, 0)


@pytest.mark.parametrize('quote,reason', [
    (None, 'missing_or_suspended'), (bar(volume=0), 'missing_or_suspended'),
    (bar(upper=None), 'missing_limits'), (bar(upper=1), 'opening_at_price_limit'),
    (bar(lower=1), 'opening_at_price_limit'),
])
def test_unfillable_open_is_retained(quote, reason):
    """场景：停牌、缺限价或开盘封板；预期不假设成交，持仓与资金不变。"""
    cash, shares, order = opening_trade('20210201', quote, 1000, 0, 0, 1, configuration())
    assert (cash, shares) == (1000, 0)
    assert order['reason'] == reason


def test_slippage_limit_and_zero_lot():
    """场景：滑点触及限价或不足一手；预期前者阻塞，后者完成零交易并保留现金。"""
    blocked = opening_trade('20210201', bar(upper=1.001), 1000, 0, 0, 1,
                            configuration(slippage_rate=.01))[2]
    assert blocked['reason'] == 'slippage_exceeds_price_limit'
    assert opening_trade('20210201', bar(), 50, 0, 0, 1, configuration())[2]['status'] == 'no_trade'


def test_month_end_next_open_and_no_future_information():
    """场景：2月最后交易日才转负；预期3月1日卖出，未来价格变化不影响此前交易。"""
    calendar, bars = history()
    bars['20210226'] = bar(.8)
    result = run_account('monthly_trend_126', calendar, bars, [], configuration())
    assert [(row['date'], row['side']) for row in result['trades']] == [
        ('20210201', 'buy'), ('20210301', 'sell')]
    assert result['trades'][1]['decision_date'] == '20210226'
    changed = dict(bars, **{'20210302': bar(20)})
    replay = run_account('monthly_trend_126', calendar, changed, [], configuration())
    assert replay['trades'] == result['trades']
    assert result['summary']['roundtrip_win_rate'] is None


def test_cash_dividend_receivable_and_payment_without_reinvestment():
    """场景：登记1000份、除息.1、随后支付；预期支付日仍应收，次日现金100不重投。"""
    calendar, bars = history()
    for day in ['20210202', '20210226', '20210301', '20210302']:
        bars[day] = bar(.9, factor=1 / .9)
    event = Dividend('cash', '20210201', '20210202', '20210301', .1)
    result = run_account('buy_hold', calendar, bars, [event], configuration())
    assert result['daily'][1]['receivable'] == 100
    assert result['daily'][1]['cash'] == 0
    assert result['daily'][1]['assets'] == 1000
    assert result['daily'][-2]['receivable'] == 100
    assert result['daily'][-2]['cash'] == 0
    assert result['daily'][-1]['cash'] == 100
    assert result['daily'][-1]['shares'] == 1000
    assert result['summary']['paid_dividends'] == 100
    assert len(result['trades']) == 1


def test_monthly_half_rebalances_and_buy_gap_includes_end():
    """场景：ETF翻倍而现金不变；预期月末后卖整手恢复约半仓，买持间隔计至期末。"""
    calendar, bars = history()
    for day in ['20210226', '20210301', '20210302']:
        bars[day] = bar(2)
    result = run_account('monthly_half', calendar, bars, [], configuration())
    assert result['trades'][0]['shares'] == 500
    assert result['trades'][1]['date'] == '20210301'
    assert result['trades'][1]['side'] == 'sell'
    assert result['daily'][-1]['assets'] == 1500
    assert result['summary']['maximum_buy_gap_sessions'] == 4


def test_blocked_orders_retry_and_t_plus_one():
    """场景：首日开盘涨停，次日解封；预期仅次日买入，不出现当日回转或丢失订单。"""
    calendar, bars = history()
    bars['20210201'] = bar(upper=1)
    result = run_account('buy_hold', calendar, bars, [], configuration())
    assert result['orders'][0]['status'] == 'blocked'
    assert result['trades'][0]['date'] == '20210202'
    assert len(result['trades']) == 1


def test_missing_mark_carries_and_no_buy_owns_no_dividend():
    """场景：持仓后缺行情时保留市值，现金策略未持登记份额不领分红。"""
    calendar, bars = history()
    missing = dict(bars)
    del missing['20210202']
    result = run_account('buy_hold', calendar, missing, [], configuration())
    assert result['daily'][1]['stale_mark'] is True
    assert result['daily'][1]['assets'] == 1000
    bars['20210128'] = bar(2)
    event = Dividend('cash', '20210201', '20210202', '20210301', .1)
    result = run_account('monthly_trend_126', calendar, bars, [event], configuration())
    assert result['summary']['buy_count'] == 0
    assert result['summary']['paid_dividends'] == 0


def test_trend_fail_closed_and_annual_summary():
    """场景：信号预热不足或端点缺失必须拒绝；年度收益衔接且年化固定250交易日。"""
    calendar, bars = history()
    with pytest.raises(ValueError, match='warmup'):
        trend_weight(calendar[0], calendar, bars, 126)
    with pytest.raises(ValueError, match='prices'):
        trend_weight(calendar[1], calendar, {}, 1)
    result = summarize([{'date': '20211231', 'assets': 90},
                        {'date': '20220104', 'assets': 99}], 100)
    assert result['maximum_drawdown'] == pytest.approx(-.1)
    assert result['annual_returns']['2022']['return'] == pytest.approx(.1)
    assert result['annualized_return'] == pytest.approx(.99 ** 125 - 1)
    with pytest.raises(ValueError):
        summarize([], 100)


@pytest.mark.parametrize('strategy,calendar,settings,reason', [
    ('bad', ['20210201'], configuration(), 'unknown'),
    ('buy_hold', ['20210201', '20210201'], configuration(), 'calendar'),
    ('buy_hold', ['20210201'], configuration(), 'calendar'),
    ('buy_hold', ['20210201', '20210302'], configuration(), 'initial'),
])
def test_account_rejects_invalid_dates(strategy, calendar, settings, reason):
    """场景：未知策略、重复或不完整交易日历；预期不产出账户结果。"""
    with pytest.raises(ValueError, match=reason):
        run_account(strategy, calendar, {}, [], settings)


def test_impossible_fee_and_collapse_fails_cash_account():
    """场景：极端费用及暴跌卖出导致负资产；预期明确拒绝，不产出有隐性融资的净值。"""
    calendar, bars = history()
    bars['20210226'] = bar(.01)
    bars['20210301'] = bar(.01)
    with pytest.raises(ValueError, match='insolvent'):
        run_account('monthly_trend_126', calendar, bars, [],
                    configuration(capital=2000, minimum_commission=1000))
