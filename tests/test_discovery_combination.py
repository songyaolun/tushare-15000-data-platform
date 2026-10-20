"""固定资金子账户的真实成交、费用与合并资金账本测试。"""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.accounting import Distribution, transaction_fee
from tushare_entitlement_hub.regime_research.panel import FIELDS, Panel
from tushare_entitlement_hub.strategy_discovery.combination import _combine_daily, simulate_sleeves
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.execution import simulate_strategy


@pytest.fixture
def sleeve_inputs():
    """生成八日两证券及两个首日买同股的独立策略；输出panel/states/definitions。"""
    shape = (8, 2)
    values = {name: np.ones(shape) for name in (*FIELDS, "beta", "capitalization")}
    for name in ("raw_open", "raw_close", "close", "raw_low", "raw_high"):
        values[name][:] = 10
    values["raw_up_limit"][:] = 11
    values["raw_down_limit"][:] = 9
    values["raw_amount"][:] = 20000
    values["sector_index"][:] = [0, 1]
    values["capitalization"][:] = 100000
    panel = Panel(
        [f"202401{day + 1:02d}" for day in range(8)], ["600001.SH", "600002.SH"], values, {}, {}
    )
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0] = True
    definitions = [
        StrategyDefinition(name, "independent", mask.copy(), np.ones(shape))
        for name in ("alpha", "beta")
    ]
    definitions = [
        replace(definition, policy=replace(definition.policy, take_profit=0.9, stop_loss=0.9))
        for definition in definitions
    ]
    return panel, np.zeros(8, dtype=np.int8), definitions


def _run(inputs, weights=(0.5, 0.5), events=None, capital=100000, start="20240101"):
    """按小样本参数运行组合；例如各50%权重时分别启动两个50000元账户。"""
    panel, states, definitions = inputs
    return simulate_sleeves(
        panel, states, events or [], definitions, list(weights), capital=capital, start=start
    )


def test_real_accounts_recalculate_minimum_commission_and_same_security_weights(sleeve_inputs):
    """真实拆分账户各买200股并各付最低5元佣金，同股权重只汇总实际400股。"""
    panel, states, definitions = sleeve_inputs
    combined = _run(sleeve_inputs)
    standalone = simulate_strategy(
        panel, states, [], definitions[0], capital=100000, start=panel.calendar[0]
    )
    expected_fee = 2 * transaction_fee(200 * 10.005, "20240102", False)
    assert combined["fees"] == pytest.approx(expected_fee)
    assert combined["fees"] > standalone["fees"]
    assert combined["open_market_value"] == 4000
    assert combined["open_positions"] == 2
    assert combined["entry_days"] == [1, 1]
    for day in combined["daily"]:
        value = 0 if day["day"] == 0 else 4000
        assert day["nav"] == pytest.approx(day["cash"] + day["receivable"] + value)
        assert day["exposure"] == pytest.approx(value / day["nav"])
        assert sum(day["weights"].values()) == pytest.approx(day["exposure"])
    assert combined["daily"][-1]["weights"][0] == pytest.approx(4000 / combined["daily"][-1]["nav"])
    assert abs(combined["reconciliation_error"]) < 1e-8
    assert combined["sleeves"][0]["capital"] == 50000


def test_dynamic_nav_and_open_nav_weighting_with_independent_orders(sleeve_inputs):
    """子账户收益分化后按实际NAV加权；开盘仍用open_nav，保留各自持仓与现金。"""
    panel, states, definitions = sleeve_inputs
    definitions[1].entry_mask[:] = False
    definitions[1].entry_mask[0, 1] = True
    panel.values["raw_close"][2:, 0] = 15
    panel.values["raw_close"][2:, 1] = 6
    panel.values["beta"][:, 0] = 2
    panel.values["beta"][:, 1] = 0.5
    combined = _run(sleeve_inputs, weights=(0.6, 0.4))
    accounts = [
        simulate_strategy(panel, states, [], definition, capital=capital, start=panel.calendar[0])
        for definition, capital in zip(definitions, [60000, 40000], strict=True)
    ]
    for index, day in enumerate(combined["daily"]):
        rows = [account["daily"][index] for account in accounts]
        for field in ("nav", "open_nav", "cash", "receivable", "cost", "positions"):
            assert day[field] == pytest.approx(sum(row[field] for row in rows))
        assert day["beta"] == pytest.approx(
            sum(row["nav"] * row["beta"] for row in rows) / day["nav"]
        )
        assert day["open_exposure"] == pytest.approx(
            sum(row["open_nav"] * row["open_exposure"] for row in rows) / day["open_nav"]
        )
        assert day["nav"] == pytest.approx(
            day["cash"] + day["receivable"] + day["nav"] * sum(day["weights"].values())
        )
    rows = [account["daily"][2] for account in accounts]
    wrong_open_weight = (
        sum(row["nav"] * row["open_exposure"] for row in rows) / combined["daily"][2]["nav"]
    )
    assert abs(wrong_open_weight - combined["daily"][2]["open_exposure"]) > 1e-8
    assert combined["daily"][2]["open_nav"] != combined["daily"][2]["nav"]
    for field in (
        "fees",
        "turnover",
        "open_cost",
        "open_market_value",
        "stale_value",
        "reconciliation_error",
        "rejected_budget",
    ):
        assert combined[field] == pytest.approx(sum(account[field] for account in accounts))


def test_dividend_receivable_cash_identity_and_no_cross_sleeve_transfers(sleeve_inputs):
    """同股分红在各自股数上登记，除权应收与支付现金合并且从不移动账户资金。"""
    panel, states, definitions = sleeve_inputs
    panel.values["raw_close"][3:, 0] = 9
    panel.values["raw_open"][3:, 0] = 9
    panel.values["raw_adj_factor"][3:, 0] = 10 / 9
    events = [Distribution(0, 2, 3, 5, 0, 1, 0)]
    combined = _run(sleeve_inputs, events=events)
    assert combined["daily"][3]["receivable"] == pytest.approx(400 * 0.8)
    assert combined["daily"][5]["receivable"] == 0
    assert combined["daily"][5]["cash"] - combined["daily"][4]["cash"] == pytest.approx(320)
    for day in combined["daily"]:
        market_value = 0 if day["day"] == 0 else 400 * panel.values["raw_close"][day["day"], 0]
        assert day["nav"] == pytest.approx(day["cash"] + day["receivable"] + market_value)
    assert abs(combined["reconciliation_error"]) < 1e-8
    single = simulate_strategy(
        panel, states, events, definitions[0], capital=50000, start=panel.calendar[0]
    )
    assert combined["daily"][-1]["nav"] == pytest.approx(2 * single["daily"][-1]["nav"])


def test_trade_identification_sorting_and_independent_exit_days(sleeve_inputs):
    """先列晚入场策略也按实际entry排序交易，重复策略名仍有唯一子账户标识。"""
    panel, _, definitions = sleeve_inputs
    definitions[0].entry_mask[:] = False
    definitions[0].entry_mask[2, 0] = True
    for index, exit_day in enumerate((4, 2)):
        exits = np.zeros((8, 2), dtype=bool)
        exits[exit_day, 0] = True
        definitions[index] = replace(definitions[index], name="same", exit_mask=exits)
    combined = _run(sleeve_inputs)
    assert combined["entry_days"] == [1, 3]
    assert [(trade["entry"], trade["exit"], trade["sleeve"]) for trade in combined["trades"]] == [
        (1, 3, "1:same"),
        (3, 5, "0:same"),
    ]
    assert combined["open_positions"] == 0
    assert combined["open_market_value"] == combined["open_cost"] == 0
    assert combined["daily"][-1]["nav"] == pytest.approx(combined["daily"][-1]["cash"])


@pytest.mark.parametrize(
    "weights",
    [[], [1], [0, 1], [-0.1, 1.1], [np.nan, 0.5], [np.inf, 0.5], [0.4, 0.4], [[0.5], [0.5]]],
)
def test_invalid_weights_rejected_before_simulation(sleeve_inputs, weights):
    """数量、正有限性、总和与维度不符合要求时，不启动任何子账户。"""
    with patch("tushare_entitlement_hub.strategy_discovery.combination.simulate_strategy") as run:
        with pytest.raises(ValueError, match="sleeve"):
            _run(sleeve_inputs, weights=weights)
        run.assert_not_called()


@pytest.mark.parametrize("capital", [0, -1, np.nan, np.inf])
def test_invalid_capital_rejected(sleeve_inputs, capital):
    """非正或非有限初始资金在构建子账户前拒绝。"""
    with pytest.raises(ValueError, match="configuration"):
        _run(sleeve_inputs, capital=capital)


@pytest.mark.parametrize("definitions", [[], [None, None]])
def test_empty_or_invalid_definitions_rejected(sleeve_inputs, definitions):
    """空策略列表或非StrategyDefinition对象不产生伪组合。"""
    panel, states, _ = sleeve_inputs
    with pytest.raises(ValueError, match="configuration"):
        _run((panel, states, definitions))


def test_single_sleeve_and_empty_observation_window(sleeve_inputs):
    """单账户权重1保留原账本；开始日超出面板时返回空日账本与零费用。"""
    panel, states, definitions = sleeve_inputs
    single_inputs = panel, states, definitions[:1]
    combined = _run(single_inputs, weights=(1,))
    original = simulate_strategy(
        panel, states, [], definitions[0], capital=100000, start=panel.calendar[0]
    )
    assert combined["daily"] == original["daily"]
    empty = _run(single_inputs, weights=(1,), start="20990101")
    assert empty["daily"] == empty["trades"] == empty["entry_days"] == []
    assert empty["fees"] == empty["reconciliation_error"] == 0
    assert empty["capital"] == 100000


@pytest.mark.parametrize(
    "field,value",
    [
        ("day", 99),
        ("date", "20990101"),
        ("stage", 2),
        ("nav", 0),
        ("nav", np.nan),
        ("open_nav", np.inf),
    ],
)
def test_misaligned_or_invalid_daily_ledger_rejected(sleeve_inputs, field, value):
    """不同日期、阶段或无效资产值不得被合并掩盖。"""
    first = _run(sleeve_inputs)["daily"][1]
    second = deepcopy(first)
    second[field] = value
    with pytest.raises(ValueError, match="sleeve_daily"):
        _combine_daily([first, second])


def test_unequal_account_history_lengths_fail_closed(sleeve_inputs):
    """子账户历史长度不一致时拒绝截断到最短曲线。"""
    panel, states, definitions = sleeve_inputs
    account = simulate_strategy(
        panel, states, [], definitions[0], capital=50000, start=panel.calendar[0]
    )
    shortened = deepcopy(account)
    shortened["daily"].pop()
    with (
        patch(
            "tushare_entitlement_hub.strategy_discovery.combination.simulate_strategy",
            side_effect=[account, shortened],
        ),
        pytest.raises(ValueError, match="shorter"),
    ):
        _run(sleeve_inputs)
